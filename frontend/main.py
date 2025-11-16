"""Streamlit Frontend for Text-to-SQL with Visualization"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import json
from typing import Dict, Any
from deep_translator import GoogleTranslator
import re

# ===================== Configuration =====================
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Phân tích Text-to-SQL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== Custom CSS =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .sql-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ===================== Helper Functions =====================
def call_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call backend API endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Make sure server is running on http://localhost:8000")
        st.info("Run: `python backend/main.py` to start the server")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout. Query took too long.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def translate_to_english(text: str) -> str:
    """Translate Vietnamese to English if needed."""
    try:
        # Simple heuristic: if text contains Vietnamese characters, translate
        vietnamese_chars = re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]')
        
        if vietnamese_chars.search(text.lower()):
            # Contains Vietnamese characters, translate
            translated = GoogleTranslator(source='vi', target='en').translate(text)
            return translated
        
        # No Vietnamese characters, return as is
        return text
        
    except Exception as e:
        # If translation fails, return original text
        st.warning(f"⚠️ Translation failed, using original query: {str(e)}")
        return text


# ===================== Main App =====================
def main():
    # Header
    st.markdown('<div class="main-header">📊 Phân tích Text-to-SQL</div>', unsafe_allow_html=True)
    st.markdown("Chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn SQL và trực quan hóa kết quả tự động.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # API health check
        if check_api_health():
            st.success("✅ API Đã kết nối")
        else:
            st.error("❌ API Chưa kết nối")
            st.warning("Khởi động backend: `python backend/main.py`")
        
        st.divider()
        
        # Query parameters
        st.subheader("Tham số truy vấn")
        limit = st.slider("Số dòng tối đa", min_value=5, max_value=100, value=10, step=5)
        max_refine = st.slider("Số lần tinh chỉnh", min_value=0, max_value=3, value=1)
        
        st.divider()
        
        # Example queries
        st.subheader("💡 Câu hỏi mẫu")
        examples = [
            ("💡", "Hiển thị 5 sản phẩm bán chạy nhất"),
            ("💡", "Đếm số người mua theo giới tính"),
            ("💡", "Liệt kê tất cả người mua từ California"),
            ("💡", "Trung bình số lượng theo phương thức thanh toán"),
            ("💡", "Hiển thị doanh số theo mã sản phẩm"),
            ("💡", "Tìm người mua đã mua nhiều hơn 5 sản phẩm")
        ]
        
        for icon, ex in examples:
            if st.button(f"{icon} {ex}", key=f"example_{ex}", use_container_width=True):
                st.session_state.query_input = ex
        
        st.divider()
        
        # Info
        st.info("""
        **Gợi ý:**
        - Hỏi câu hỏi cụ thể và rõ ràng
        - Dùng từ khóa như "top", "tổng", "trung bình"
        - Đề cập tên cột khi có thể
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize session state
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
        
        # Query input
        query = st.text_input(
            "💬 Đặt câu hỏi về dữ liệu bán hàng:",
            value=st.session_state.query_input,
            placeholder="Ví dụ: Hiển thị 5 sản phẩm bán chạy nhất",
            key="query_text_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_button = st.button("🔍 Phân tích", type="primary", use_container_width=True)
    
    # Process query
    if analyze_button and query:
        with st.spinner("🤔 Đang phân tích câu hỏi của bạn..."):
            # Translate Vietnamese to English if needed
            original_query = query
            translated_query = translate_to_english(query)
            
            # Show translation if different
            if translated_query != original_query:
                st.info(f"🌐 Đã dịch: {original_query} → {translated_query}")
            
            # Call API with translated query
            result = call_api("/query-visualize", {
                "query": translated_query,
                "limit": limit,
                "max_refine": max_refine
            })
            
            if result:
                # Store original query for display
                result["original_query"] = original_query
                result["translated_query"] = translated_query
                st.session_state.last_result = result
                st.rerun()
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        st.divider()
        
        # SQL Query section
        st.subheader("🔧 Truy vấn SQL được tạo")
        st.code(result["sql"], language="sql")
        
        # Warning if any
        if result.get("warning"):
            st.warning(f"⚠️ {result['warning']}")
        
        st.divider()
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["📊 Trực quan hóa", "📋 Bảng dữ liệu", "ℹ️ Thông tin"])
        
        with tab1:
            st.subheader("📊 Trực quan hóa thông minh")
            
            # Chart type selector
            col1, col2 = st.columns([2, 1])
            with col1:
                # Display AI recommendation
                st.info(f"🤖 **AI khuyên dùng:** {result['chart_type'].upper()} - {result['reasoning']}")
            
            with col2:
                # Manual chart type selector
                chart_options = ["Tự động (AI)", "Biểu đồ cột", "Biểu đồ đường", "Biểu đồ tròn", "Biểu đồ phân tán", "Bảng"]
                selected_chart = st.selectbox(
                    "Chọn loại biểu đồ:",
                    chart_options,
                    index=0,
                    key="chart_selector"
                )
            
            st.write("")  # Spacing
            
            # Chart metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                chart_map = {
                    "Tự động (AI)": result["chart_type"].upper(),
                    "Biểu đồ cột": "CỘT",
                    "Biểu đồ đường": "ĐƯỜNG",
                    "Biểu đồ tròn": "TRÒN",
                    "Biểu đồ phân tán": "PHÂN TÁN",
                    "Bảng": "BẢNG"
                }
                display_type = chart_map.get(selected_chart, selected_chart)
                st.metric("Loại biểu đồ", display_type)
            with col2:
                st.metric("Tổng số dòng", result["row_count"])
            with col3:
                st.metric("Số cột", len(result["columns"]))
            
            st.write("")  # Spacing
            
            # Display chart based on selection
            if result.get("chart_html") and result["rows"]:
                if selected_chart == "Tự động (AI)":
                    # Use AI-generated chart
                    components.html(result["chart_html"], height=600, scrolling=True)
                else:
                    # Generate custom chart based on user selection
                    df = pd.DataFrame(result["rows"], columns=result["columns"])
                    
                    try:
                        if selected_chart == "Bảng":
                            st.dataframe(df, use_container_width=True, height=500)
                        
                        elif selected_chart == "Biểu đồ cột" and len(df.columns) >= 2:
                            import plotly.express as px
                            fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                                        title=f"{df.columns[1]} theo {df.columns[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif selected_chart == "Biểu đồ đường" and len(df.columns) >= 2:
                            import plotly.express as px
                            fig = px.line(df, x=df.columns[0], y=df.columns[1],
                                         title=f"{df.columns[1]} theo {df.columns[0]}",
                                         markers=True)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif selected_chart == "Biểu đồ tròn" and len(df.columns) >= 2:
                            import plotly.express as px
                            # Limit to top 10 for readability
                            df_pie = df.head(10)
                            fig = px.pie(df_pie, names=df.columns[0], values=df.columns[1],
                                        title=f"Phân bố {df.columns[1]}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif selected_chart == "Biểu đồ phân tán" and len(df.columns) >= 2:
                            import plotly.express as px
                            fig = px.scatter(df, x=df.columns[0], y=df.columns[1],
                                           title=f"{df.columns[1]} so với {df.columns[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.warning(f"⚠️ Không thể tạo {selected_chart} với dữ liệu này. Cần ít nhất 2 cột.")
                            st.dataframe(df, use_container_width=True, height=400)
                    
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tạo biểu đồ: {str(e)}")
                        st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("Không có dữ liệu để hiển thị.")
        
        with tab2:
            st.subheader("📋 Dữ liệu gốc")
            
            if result["rows"]:
                # Convert to DataFrame
                df = pd.DataFrame(result["rows"], columns=result["columns"])
                
                # Display info
                st.write(f"**Hiển thị {len(df)} dòng × {len(df.columns)} cột**")
                
                # Display table
                st.dataframe(df, use_container_width=True, height=500)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải xuống CSV",
                    data=csv,
                    file_name="query_result.csv",
                    mime="text/csv"
                )
            else:
                st.info("Không có dữ liệu trả về từ truy vấn.")
        
        with tab3:
            st.subheader("ℹ️ Thông tin truy vấn")
            
            info_data = {
                "Câu hỏi gốc": result.get("original_query", query if 'query_input' in st.session_state else "N/A"),
                "Câu hỏi đã dịch": result.get("translated_query", "(giống câu gốc)"),
                "SQL được tạo": result["sql"],
                "Số dòng trả về": result["row_count"],
                "Các cột": ", ".join(result["columns"]),
                "Loại biểu đồ AI đề xuất": result["chart_type"],
                "Tiêu đề biểu đồ": result.get("title", "N/A"),
                "Lý do chọn biểu đồ": result["reasoning"]
            }
            
            for key, value in info_data.items():
                st.write(f"**{key}:**")
                st.write(value)
                st.write("")  # Spacing


if __name__ == "__main__":
    main()