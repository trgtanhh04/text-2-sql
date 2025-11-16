# Hệ thống Text-to-SQL với Trực Quan Hóa Dữ Liệu

> **Hệ thống phân tích dữ liệu bán hàng thông minh** - Chuyển đổi câu hỏi tiếng Việt thành truy vấn SQL và trực quan hóa kết quả tự động bằng AI.

## 📋 Tổng Quan

Hệ thống Text-to-SQL cho phép người dùng truy vấn dữ liệu bán hàng bằng ngôn ngữ tự nhiên (tiếng Việt), tự động sinh câu lệnh SQL, và trực quan hóa kết quả thông qua biểu đồ thông minh do AI lựa chọn.

### ✨ Tính Năng Chính

- 🗣️ **Truy vấn bằng tiếng Việt**: Nhập câu hỏi bằng tiếng Việt, hệ thống tự động dịch và xử lý
- 🤖 **AI-Powered**: Sử dụng Google Gemini 2.5-flash để tạo SQL và chọn loại biểu đồ phù hợp
- 📊 **Trực quan hóa thông minh**: 6 loại biểu đồ (Tự động/AI, Cột, Đường, Tròn, Phân tán, Bảng)
- 🔄 **Chuyển đổi biểu đồ linh hoạt**: Người dùng có thể chọn loại biểu đồ khác nếu muốn
- ☁️ **Triển khai trên Cloud**: Backend đã deploy trên Render (https://text-2-sql-be.onrender.com)
- 🔒 **Connection Pooling**: Quản lý kết nối database hiệu quả với SQLAlchemy

### 🛠️ Công Nghệ Sử Dụng

**Backend:**
- FastAPI - REST API framework
- SQLAlchemy - ORM và connection pooling
- LangChain + Google Gemini - LLM integration
- PostgreSQL (Neon Cloud) - Database

**Frontend:**
- Streamlit - Web UI framework
- Plotly - Visualization library
- Deep Translator - Vietnamese to English translation
- Requests - API client

## 📂 Cấu Trúc Thư Mục

```
text2sql/
├── backend/
│   ├── main.py                    # FastAPI server với 3 endpoints
│   ├── config/
│   │   └── config.py              # Database configuration
│   ├── core/
│   │   ├── conect_db.py           # Database connection pooling
│   │   ├── import_db.py           # Import CSV data to PostgreSQL
│   │   ├── schema_utils.py        # Database schema introspection
│   │   ├── selector_and_prompt.py # Prompt builder cho LLM
│   │   ├── t2sql_core.py          # Text-to-SQL engine chính
│   │   └── visualize.py           # LLM-powered chart selection
│   ├── data/
│   │   └── sales_data.csv         # Dữ liệu mẫu
│   └── models/
│       └── model.py               # Data models
├── frontend/
│   └── main.py                    # Streamlit UI (tiếng Việt)
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README.md                     # Tài liệu này
```

## 🗄️ Schema Database

Database PostgreSQL (Neon Cloud) có **1 bảng duy nhất**: `sales_data`

### Cột của bảng `sales_data`:

| Cột | Kiểu Dữ Liệu | Mô Tả |
|-----|-------------|-------|
| `transaction_date` | VARCHAR | Ngày giao dịch (Excel serial number dạng text) |
| `buyer_first_name` | VARCHAR | Tên khách hàng |
| `buyer_last_name` | VARCHAR | Họ khách hàng |
| `buyer_location` | VARCHAR | Địa điểm (San Jose, Houston, Chicago, ...) |
| `buyer_date_of_birth` | VARCHAR | Ngày sinh (Excel serial number dạng text) |
| `payment_method` | VARCHAR | Phương thức thanh toán (Credit Card, Debit Card, Cash, Mobile Payment) |
| `quantity_purchased` | INTEGER | Số lượng mua |
| `product_code` | VARCHAR | Mã sản phẩm (Pro01, Pro02, ..., Pro10) |
| `sales_representative` | VARCHAR | Nhân viên bán hàng |
| `gender` | VARCHAR | Giới tính (Male, Female, Other) |

## 🚀 Cài Đặt và Chạy

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình Environment Variables

Tạo file `.env` trong thư mục gốc:

```env
DATABASE_URL=postgresql://username:password@host/database
GEMINI_API_KEY=your_gemini_api_key
API_BASE_URL=http://localhost:8000  # Local hoặc URL Render
```

### 3. Import Dữ Liệu (Lần đầu)

```bash
cd backend\core
python import_db.py
```

### 4. Chạy Backend (Local)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Backend API sẽ chạy tại: `http://localhost:8000`

### 5. Chạy Frontend (Local)

```bash
cd frontend
streamlit run main.py
```

Frontend UI sẽ mở tại: `http://localhost:8501`

## 🌐 Triển Khai trên Render

### Backend (Đã Deploy)

- **URL**: https://text-2-sql-be.onrender.com
- **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `DATABASE_URL`: Connection string của Neon PostgreSQL
  - `GEMINI_API_KEY`: API key của Google Gemini

### Frontend (Đang Triển Khai)

- **Start Command**: `cd frontend && streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false`
- **Environment Variables**:
  - `API_BASE_URL`: https://text-2-sql-be.onrender.com

## 🔌 API Endpoints

### 1. `/query` - Chỉ truy vấn SQL

**POST** `/query`

```json
{
  "question": "Tổng số lượng bán theo sản phẩm"
}
```

**Response:**
```json
{
  "sql": "SELECT product_code, SUM(quantity_purchased) AS total FROM sales_data GROUP BY product_code",
  "results": [...],
  "row_count": 10
}
```

### 2. `/visualize` - Chỉ trực quan hóa

**POST** `/visualize`

```json
{
  "data": [...],
  "question": "Tổng số lượng bán theo sản phẩm",
  "chart_type": "bar"
}
```

**Response:**
```json
{
  "chart_type": "bar",
  "chart_config": {...},
  "reasoning": "Biểu đồ cột phù hợp để so sánh giá trị giữa các sản phẩm"
}
```

### 3. `/query-visualize` - Truy vấn + Trực quan hóa

**POST** `/query-visualize`

```json
{
  "question": "Tổng số lượng bán theo sản phẩm",
  "chart_type": "auto"
}
```

**Response:**
```json
{
  "sql": "...",
  "results": [...],
  "row_count": 10,
  "chart_type": "bar",
  "chart_config": {...},
  "reasoning": "..."
}
```

## 🎨 Quy Trình Hoạt Động

```
1. User nhập câu hỏi tiếng Việt trên Streamlit UI
         ↓
2. Frontend dịch sang tiếng Anh (deep-translator)
         ↓
3. Gửi request đến Backend API (/query-visualize)
         ↓
4. Backend: Gemini LLM tạo SQL từ câu hỏi
         ↓
5. Thực thi SQL trên PostgreSQL (Neon)
         ↓
6. Gemini LLM chọn loại biểu đồ phù hợp
         ↓
7. Trả về: SQL + Kết quả + Chart config
         ↓
8. Frontend hiển thị kết quả và vẽ biểu đồ (Plotly)
         ↓
9. User có thể chuyển sang loại biểu đồ khác
```

## 💡 Điểm Quan Trọng Khi Viết SQL

### 1. Chuyển Đổi Ngày Tháng

⚠️ **Lưu ý**: Ngày tháng được lưu dưới dạng **VARCHAR** (Excel serial number), cần cast sang INTEGER trước:

```sql
-- Chuyển thành DATE
DATE '1899-12-30' + transaction_date::INTEGER

-- Lấy năm
EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER))

-- Lấy tháng
EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER))
```

### 2. Tên Khách Hàng

Kết hợp họ và tên:

```sql
buyer_first_name || ' ' || buyer_last_name AS buyer_name
```

### 3. Aggregations

Các truy vấn phân tích phổ biến:

```sql
-- Tổng số lượng theo sản phẩm
SELECT product_code, SUM(quantity_purchased) AS total
FROM sales_data
GROUP BY product_code
ORDER BY total DESC;

-- Trung bình theo giới tính
SELECT gender, AVG(quantity_purchased) AS avg_qty
FROM sales_data
GROUP BY gender;
```

## 📝 Ví Dụ Truy Vấn

### Query 1: Tìm giao dịch theo địa điểm
```
User: "Tìm giao dịch ở San Jose"
SQL: SELECT * FROM sales_data WHERE buyer_location ILIKE '%San Jose%' LIMIT 50;
```

### Query 2: Phân tích bán hàng theo sản phẩm
```
User: "Tổng số lượng bán theo sản phẩm"
SQL: SELECT product_code, SUM(quantity_purchased) AS total_quantity
     FROM sales_data
     GROUP BY product_code
     ORDER BY total_quantity DESC;
Biểu đồ: Bar Chart (AI tự chọn)
```

### Query 3: Top nhân viên bán hàng
```
User: "Top 5 nhân viên bán hàng xuất sắc nhất"
SQL: SELECT sales_representative, SUM(quantity_purchased) AS total_sold
     FROM sales_data
     GROUP BY sales_representative
     ORDER BY total_sold DESC
     LIMIT 5;
Biểu đồ: Bar Chart (AI tự chọn)
```

### Query 4: Xu hướng bán hàng theo thời gian
```
User: "Xu hướng bán hàng theo tháng trong năm 2023"
SQL: SELECT EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER)) AS month,
            SUM(quantity_purchased) AS total
     FROM sales_data
     WHERE EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER)) = 2023
     GROUP BY month
     ORDER BY month;
Biểu đồ: Line Chart (AI tự chọn)
```

## 🧪 Testing

Chạy test hệ thống:

```bash
cd backend\core
python test_system.py
```

## 🎯 Roadmap

- [x] Schema adapted cho sales_data
- [x] Selector updated với keywords bán hàng
- [x] Examples updated cho sales queries
- [x] Prompt builder adapted
- [x] Core logic simplified (không cần JOINs)
- [x] Tích hợp Gemini LLM
- [x] Tạo FastAPI backend với 3 endpoints
- [x] Tạo Streamlit frontend với UI tiếng Việt
- [x] Tích hợp Vietnamese translation
- [x] LLM-powered smart visualization
- [x] Chart type selector (6 loại)
- [x] Deploy backend lên Render
- [ ] Deploy frontend lên Render
- [ ] Testing end-to-end với cloud deployment
- [ ] Tối ưu performance và caching

## 🤝 Đóng Góp

Hệ thống này được phát triển để demo khả năng của LLM trong việc chuyển đổi ngôn ngữ tự nhiên sang SQL và trực quan hóa dữ liệu thông minh.

## 📄 License

MIT License
