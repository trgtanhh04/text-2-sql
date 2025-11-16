"""FastAPI Backend for Text-to-SQL with Visualization"""

import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.t2sql_core import text_to_sql
from core.visualize import text_to_visualization
from config.config import GEMINI_API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# Initialize FastAPI
app = FastAPI(
    title="Text-to-SQL API",
    description="Convert natural language to SQL queries with smart visualization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM globally
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

def llm_invoke(prompt: str) -> str:
    """Global LLM invoke function."""
    resp = gemini.invoke([HumanMessage(content=prompt)])
    return resp.content


# ===================== Request/Response Models =====================
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question", example="Show top 5 products by sales")
    limit: int = Field(10, description="Maximum rows to return", ge=1, le=1000)
    max_refine: int = Field(1, description="Maximum refinement attempts", ge=0, le=3)


class QueryResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    trials: Optional[List[tuple]] = None
    warning: Optional[str] = None
    row_count: int


class VisualizeRequest(BaseModel):
    query: str = Field(..., description="Original natural language question")
    sql: str = Field(..., description="SQL query")
    columns: List[str] = Field(..., description="Column names")
    rows: List[List[Any]] = Field(..., description="Data rows")


class VisualizeResponse(BaseModel):
    chart_type: str
    title: str
    reasoning: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    chart_html: str
    warning: Optional[str] = None


class QueryVisualizeRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    limit: int = Field(10, ge=1, le=1000)
    max_refine: int = Field(1, ge=0, le=3)


class QueryVisualizeResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    chart_type: str
    title: str
    reasoning: str
    chart_html: str
    warning: Optional[str] = None


# ===================== API Endpoints =====================
@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Text-to-SQL API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Convert text to SQL and execute",
            "POST /visualize": "Create visualization from query result",
            "POST /query-visualize": "Complete pipeline: query + visualize",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "text-to-sql"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Convert natural language to SQL and execute.
    
    Returns SQL query and results without visualization.
    """
    try:
        log.info(f"Query request: {request.query}")
        
        result = text_to_sql(
            user_query=request.query,
            max_refine=request.max_refine,
            limit=request.limit
        )
        
        response = QueryResponse(
            sql=result["sql"],
            columns=result["columns"],
            rows=result["rows"],
            trials=result.get("trials"),
            warning=result.get("warning"),
            row_count=len(result["rows"])
        )
        
        log.info(f"Query success: {response.row_count} rows returned")
        return response
        
    except Exception as e:
        log.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize", response_model=VisualizeResponse)
def visualize_endpoint(request: VisualizeRequest):
    """
    Create visualization from query result.
    
    Takes SQL result and generates appropriate chart.
    """
    try:
        log.info(f"Visualize request for query: {request.query}")
        
        # Prepare result dict
        result = {
            "sql": request.sql,
            "columns": request.columns,
            "rows": request.rows
        }
        
        # Create visualization
        viz = text_to_visualization(
            user_query=request.query,
            result=result,
            llm_invoke_fn=llm_invoke,
            output_path=None  # Don't save to file for API
        )
        
        # Convert plotly figure to HTML
        chart_html = ""
        if viz.get("figure"):
            chart_html = viz["figure"].to_html(include_plotlyjs='cdn', div_id="chart")
        
        response = VisualizeResponse(
            chart_type=viz["chart_config"]["chart_type"],
            title=viz["chart_config"]["title"],
            reasoning=viz["chart_config"]["reasoning"],
            x_column=viz["chart_config"].get("x_column"),
            y_column=viz["chart_config"].get("y_column"),
            chart_html=chart_html,
            warning=viz.get("warning")
        )
        
        log.info(f"Visualization success: {response.chart_type} chart")
        return response
        
    except Exception as e:
        log.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-visualize", response_model=QueryVisualizeResponse)
def query_visualize_endpoint(request: QueryVisualizeRequest):
    """
    Complete pipeline: Convert text to SQL, execute, and create visualization.
    
    This is the main endpoint combining both query and visualization.
    """
    try:
        log.info(f"Query+Visualize request: {request.query}")
        
        # Step 1: Execute query
        result = text_to_sql(
            user_query=request.query,
            max_refine=request.max_refine,
            limit=request.limit
        )
        
        # Step 2: Create visualization
        viz = text_to_visualization(
            user_query=request.query,
            result=result,
            llm_invoke_fn=llm_invoke,
            output_path=None
        )
        
        # Step 3: Convert to HTML
        chart_html = ""
        if viz.get("figure"):
            chart_html = viz["figure"].to_html(include_plotlyjs='cdn', div_id="chart")
        
        response = QueryVisualizeResponse(
            sql=result["sql"],
            columns=result["columns"],
            rows=result["rows"],
            row_count=len(result["rows"]),
            chart_type=viz["chart_config"]["chart_type"],
            title=viz["chart_config"]["title"],
            reasoning=viz["chart_config"]["reasoning"],
            chart_html=chart_html,
            warning=result.get("warning") or viz.get("warning")
        )
        
        log.info(f"Pipeline success: {response.row_count} rows, {response.chart_type} chart")
        return response
        
    except Exception as e:
        log.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
