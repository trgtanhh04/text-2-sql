"""
Smart Data Visualization Module
Uses LLM to automatically select appropriate chart types and generate visualizations
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from langchain_core.messages import HumanMessage

try:
    from ..config.config import GEMINI_API_KEY
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visualize")


# ===================== LLM Chart Selector =====================
def analyze_data_for_chart(
    user_query: str,
    columns: List[str],
    rows: List[List[Any]],
    sql: str,
    llm_invoke_fn
) -> Dict[str, Any]:
    """
    Use LLM to analyze query result and suggest best visualization.
    
    Returns:
        {
            "chart_type": "bar|line|pie|scatter|table",
            "x_column": "column_name",
            "y_column": "column_name",
            "title": "Chart title",
            "reasoning": "Why this chart type"
        }
    """
    
    # Sample first 5 rows for LLM analysis
    sample_rows = rows[:5]
    
    prompt = f"""
    You are a data visualization expert. Analyze this SQL query result and suggest the BEST chart type.

    USER QUESTION: {user_query}

    SQL QUERY: {sql}

    COLUMNS: {columns}

    SAMPLE DATA (first 5 rows):
    {json.dumps(sample_rows, indent=2, default=str)}

    TOTAL ROWS: {len(rows)}

    Based on the data structure and user's question, determine:
    1. Best chart type: bar, line, pie, scatter, or table
    2. Which column should be X-axis
    3. Which column should be Y-axis (if applicable)
    4. Appropriate chart title

    RULES:
    - If data has COUNT/SUM/AVG aggregation → bar chart
    - If data shows trends over time → line chart  
    - If data shows percentage/proportion of whole → pie chart
    - If data shows relationship between 2 numeric values → scatter plot
    - If data is detailed list with many columns → table
    - If only 1-2 rows → table
    - If more than 10 categories in bar/pie → use table or bar with horizontal orientation

    Return ONLY valid JSON (no markdown):
    {{
        "chart_type": "bar",
        "x_column": "product_code",
        "y_column": "total_quantity",
        "title": "Total Sales by Product",
        "reasoning": "Data shows aggregated counts by category, bar chart is best"
    }}
"""
    
    response = llm_invoke_fn(prompt).strip()
    
    # Clean response (remove markdown if present)
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse LLM response: {response}")
        # Fallback to table
        return {
            "chart_type": "table",
            "x_column": columns[0] if columns else None,
            "y_column": columns[1] if len(columns) > 1 else None,
            "title": "Query Results",
            "reasoning": "Failed to parse LLM suggestion, using table as fallback"
        }


# ===================== Chart Generators =====================
def create_bar_chart(
    columns: List[str],
    rows: List[List[Any]],
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """Create bar chart using plotly."""
    x_idx = columns.index(x_col)
    y_idx = columns.index(y_col)
    
    x_data = [row[x_idx] for row in rows]
    y_data = [row[y_idx] for row in rows]
    
    # Use horizontal bar if many categories
    if len(x_data) > 10:
        fig = go.Figure(data=[go.Bar(x=y_data, y=x_data, orientation='h')])
        fig.update_layout(
            title=title,
            xaxis_title=y_col,
            yaxis_title=x_col,
            height=max(400, len(x_data) * 30)
        )
    else:
        fig = go.Figure(data=[go.Bar(x=x_data, y=y_data)])
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
    
    return fig


def create_line_chart(
    columns: List[str],
    rows: List[List[Any]],
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """Create line chart using plotly."""
    x_idx = columns.index(x_col)
    y_idx = columns.index(y_col)
    
    x_data = [row[x_idx] for row in rows]
    y_data = [row[y_idx] for row in rows]
    
    fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, mode='lines+markers')])
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig


def create_pie_chart(
    columns: List[str],
    rows: List[List[Any]],
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """Create pie chart using plotly."""
    x_idx = columns.index(x_col)
    y_idx = columns.index(y_col)
    
    labels = [row[x_idx] for row in rows]
    values = [row[y_idx] for row in rows]
    
    # Limit to top 10 slices for readability
    if len(labels) > 10:
        sorted_data = sorted(zip(values, labels), reverse=True)
        top_values, top_labels = zip(*sorted_data[:10])
        other_value = sum(sorted_data[i][0] for i in range(10, len(sorted_data)))
        
        values = list(top_values) + [other_value]
        labels = list(top_labels) + ["Others"]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title)
    
    return fig


def create_scatter_chart(
    columns: List[str],
    rows: List[List[Any]],
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """Create scatter plot using plotly."""
    x_idx = columns.index(x_col)
    y_idx = columns.index(y_col)
    
    x_data = [row[x_idx] for row in rows]
    y_data = [row[y_idx] for row in rows]
    
    fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, mode='markers')])
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig


def create_table_view(
    columns: List[str],
    rows: List[List[Any]],
    title: str
) -> go.Figure:
    """Create table view using plotly."""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=columns,
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[[row[i] for row in rows] for i in range(len(columns))],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=title,
        height=min(800, 100 + len(rows) * 30)
    )
    
    return fig


# ===================== Main Visualization Function =====================
def visualize_query_result(
    user_query: str,
    sql: str,
    columns: List[str],
    rows: List[List[Any]],
    llm_invoke_fn,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Smart visualization: Use LLM to select best chart type and create it.
    
    Args:
        user_query: Original natural language question
        sql: Generated SQL query
        columns: Column names from query result
        rows: Data rows from query result
        llm_invoke_fn: Function to invoke LLM (same as t2sql_core)
        output_path: Optional path to save HTML file
    
    Returns:
        {
            "chart_config": LLM's chart configuration,
            "figure": plotly Figure object,
            "html_path": path to saved HTML (if output_path provided)
        }
    """
    
    # Handle empty results
    if not rows:
        log.warning("No data to visualize")
        return {
            "chart_config": {"chart_type": "table", "title": "No Results"},
            "figure": None,
            "html_path": None,
            "warning": "No data to visualize"
        }
    
    # Get LLM recommendation
    log.info(f"Analyzing {len(rows)} rows with {len(columns)} columns for visualization...")
    chart_config = analyze_data_for_chart(user_query, columns, rows, sql, llm_invoke_fn)
    
    log.info(f"LLM suggests: {chart_config['chart_type']} chart - {chart_config['reasoning']}")
    
    # Create chart based on recommendation
    chart_type = chart_config["chart_type"].lower()
    x_col = chart_config.get("x_column")
    y_col = chart_config.get("y_column")
    title = chart_config.get("title", "Query Results")
    
    try:
        if chart_type == "bar" and x_col and y_col:
            fig = create_bar_chart(columns, rows, x_col, y_col, title)
        elif chart_type == "line" and x_col and y_col:
            fig = create_line_chart(columns, rows, x_col, y_col, title)
        elif chart_type == "pie" and x_col and y_col:
            fig = create_pie_chart(columns, rows, x_col, y_col, title)
        elif chart_type == "scatter" and x_col and y_col:
            fig = create_scatter_chart(columns, rows, x_col, y_col, title)
        else:
            # Fallback to table
            fig = create_table_view(columns, rows, title)
            chart_config["chart_type"] = "table"
    
    except Exception as e:
        log.error(f"Failed to create {chart_type} chart: {e}")
        # Fallback to table
        fig = create_table_view(columns, rows, title)
        chart_config["chart_type"] = "table"
        chart_config["reasoning"] = f"Failed to create {chart_type}, using table fallback"
    
    # Save to HTML if path provided
    html_path = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path)
        html_path = output_path
        log.info(f"Chart saved to: {html_path}")
    
    return {
        "chart_config": chart_config,
        "figure": fig,
        "html_path": html_path
    }


# ===================== Convenience Function =====================
def text_to_visualization(
    user_query: str,
    result: Dict[str, Any],
    llm_invoke_fn,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to visualize t2sql_core result directly.
    
    Args:
        user_query: Original question
        result: Output from t2sql_core.text_to_sql()
        llm_invoke_fn: LLM invoke function
        output_path: Optional HTML output path
    
    Example:
        from backend.core.t2sql_core import text_to_sql
        from backend.core.visualize import text_to_visualization
        
        result = text_to_sql("Show sales by product")
        viz = text_to_visualization("Show sales by product", result, llm_invoke_fn)
        viz["figure"].show()
    """
    return visualize_query_result(
        user_query=user_query,
        sql=result["sql"],
        columns=result["columns"],
        rows=result["rows"],
        llm_invoke_fn=llm_invoke_fn,
        output_path=output_path
    )


# ===================== Test =====================
if __name__ == "__main__":
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    
    # Import t2sql function
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from t2sql_core import text_to_sql
    
    # Setup LLM
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
    def llm_invoke(prompt: str) -> str:
        resp = gemini.invoke([HumanMessage(content=prompt)])
        return resp.content
    
    # Test queries with different chart types
    test_cases = [
        ("Show total quantity by product code", "output/sales_by_product.html"),
        ("Show top 5 sales representatives by total sales", "output/top_reps.html"),
        ("Count buyers by gender", "output/gender_distribution.html"),
    ]
    
    print("\n" + "="*80)
    print("TESTING SMART VISUALIZATION")
    print("="*80)
    
    for i, (query, output_file) in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] Query: {query}")
        print("-" * 80)
        
        # Get SQL result
        result = text_to_sql(query, limit=20)
        print(f"✓ SQL executed: {len(result['rows'])} rows")
        
        # Create visualization
        viz = text_to_visualization(query, result, llm_invoke, output_file)
        
        print(f"✓ Chart type: {viz['chart_config']['chart_type']}")
        print(f"  Reasoning: {viz['chart_config']['reasoning']}")
        print(f"  Saved to: {viz['html_path']}")
        
        # Show in browser (optional)
        # viz["figure"].show()
    
    print("\n" + "="*80)
    print("✓ All visualizations created successfully!")
    print("="*80)
