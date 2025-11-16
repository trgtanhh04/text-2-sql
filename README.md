# Text-to-SQL System - Sales Data

Hệ thống Text-to-SQL để truy vấn dữ liệu bán hàng từ database PostgreSQL (Neon).

## Schema Database

Database chỉ có **1 bảng duy nhất**: `sales_data`

### Cột của bảng `sales_data`:
- `transaction_date` (VARCHAR) - Ngày giao dịch (Excel date serial number stored as text)
- `buyer_first_name` (VARCHAR) - Tên khách hàng
- `buyer_last_name` (VARCHAR) - Họ khách hàng
- `buyer_location` (VARCHAR) - Địa điểm khách hàng (San Jose, Houston, Chicago, etc.)
- `buyer_date_of_birth` (VARCHAR) - Ngày sinh khách hàng (Excel date serial number stored as text)
- `payment_method` (VARCHAR) - Phương thức thanh toán (Credit Card, Debit Card, Cash, Mobile Payment)
- `quantity_purchased` (INTEGER) - Số lượng mua
- `product_code` (VARCHAR) - Mã sản phẩm (Pro01, Pro02, ..., Pro10)
- `sales_representative` (VARCHAR) - Nhân viên bán hàng
- `gender` (VARCHAR) - Giới tính khách hàng (Male, Female, Other)

## Cấu trúc File

### Core Files:
1. **`schema_utils.py`** - Introspect database schema, load table/column info
2. **`selector_and_prompt.py`** - Rule-based selector và prompt builder cho LLM
3. **`t2sql_core.py`** - Orchestrator chính: guards, execution, refinement

### Quy trình hoạt động:

```
User Query → Selector → Schema Loading → Prompt Building → LLM → SQL → Guards → Execute → Results
```

## Điểm quan trọng

### 1. Date Conversion
⚠️ **Important**: Dates được lưu dưới dạng **VARCHAR** (Excel serial number as text), cần cast sang INTEGER trước khi convert:
```sql
-- Convert to date (MUST cast to INTEGER first)
DATE '1899-12-30' + transaction_date::INTEGER

-- Extract year
EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER))

-- Extract month
EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER))
```

### 2. Buyer Name
Kết hợp first_name và last_name:
```sql
buyer_first_name || ' ' || buyer_last_name AS buyer_name
```

### 3. Aggregations
Thường dùng cho phân tích:
```sql
-- Total quantity by product
SELECT product_code, SUM(quantity_purchased) AS total
FROM sales_data
GROUP BY product_code
ORDER BY total DESC;

-- Average by gender
SELECT gender, AVG(quantity_purchased) AS avg_qty
FROM sales_data
GROUP BY gender;
```

## Examples

### Query 1: Transactions in specific location
```
User: "Tìm giao dịch ở San Jose"
SQL: SELECT * FROM sales_data WHERE buyer_location ILIKE '%San Jose%' LIMIT 50;
```

### Query 2: Sales by product
```
User: "Tổng số lượng bán theo sản phẩm"
SQL: SELECT product_code, SUM(quantity_purchased) AS total_quantity
     FROM sales_data
     GROUP BY product_code
     ORDER BY total_quantity DESC;
```

### Query 3: Top sales reps
```
User: "Top 5 nhân viên bán hàng xuất sắc nhất"
SQL: SELECT sales_representative, SUM(quantity_purchased) AS total_sold
     FROM sales_data
     GROUP BY sales_representative
     ORDER BY total_sold DESC
     LIMIT 5;
```

## Testing

Chạy test để kiểm tra hệ thống:
```bash
cd E:\text2sql\backend\core
python test_system.py
```

## Import Data

Import dữ liệu từ CSV:
```bash
cd E:\text2sql\backend\core
python import_db.py
```

## Khác biệt với Scan-CV

| Scan-CV (Old) | Text-to-SQL (New) |
|---------------|-------------------|
| Multi-table (candidates, skills, experiences, etc.) | Single table (sales_data) |
| Complex JOINs | No JOINs needed |
| Resume enrichment | Simple data return |
| Candidate-focused | Transaction-focused |
| DISTINCT ON logic | No special postprocessing |

## Next Steps

1. ✅ Schema adapted for sales_data
2. ✅ Selector updated with sales-related keywords
3. ✅ Examples updated for sales queries
4. ✅ Prompt builder adapted
5. ✅ Core logic simplified (no multi-table joins)
6. 🔲 Test with real LLM (Gemini)
7. 🔲 Integrate with frontend
