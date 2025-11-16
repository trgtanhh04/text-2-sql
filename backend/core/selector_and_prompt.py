
from __future__ import annotations
import re
from typing import List, Tuple


# =========================
# 1) SELECTOR (rule-based)
# =========================
DATE_WORDS = r"(date|transaction_date|month|year|quarter|when|time)"
BUYER_WORDS = r"(buyer|customer|name|first_name|last_name|who|person)"
LOC_WORDS = r"(location|buyer_location|city|where|place|San Jose|Houston|Chicago|Dallas|Phoenix)"
PRODUCT_WORDS = r"(product|product_code|Pro\d+|item|goods)"
PAYMENT_WORDS = r"(payment|payment_method|credit card|debit card|cash|mobile payment|paid|pay)"
QUANTITY_WORDS = r"(quantity|quantity_purchased|amount|purchased|sold|sales)"
REP_WORDS = r"(sales_representative|representative|rep|salesperson|employee)"
GENDER_WORDS = r"(gender|male|female|sex)"
AGE_WORDS = r"(age|date_of_birth|birth|born|old)"

def selector_lite(user_query: str) -> Tuple[List[str], str]:
    """
    Returns (list_of_tables, hints_string) based on keywords.
    """
    uq = user_query.lower()
    tables = {"sales_data"}  # always need sales_data table

    hints = []

    if re.search(DATE_WORDS, uq):
        hints.append("This query involves transaction dates. Note: transaction_date is stored as VARCHAR (Excel date serial number as text).")
        hints.append("To convert to proper date: DATE '1899-12-30' + transaction_date::INTEGER.")
        hints.append("To extract year: EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER)).")
        hints.append("To extract month: EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER)).")

    if re.search(BUYER_WORDS, uq):
        hints.append("This query involves buyer information. Use buyer_first_name and buyer_last_name columns.")
        hints.append("Combine names with: buyer_first_name || ' ' || buyer_last_name AS full_name.")

    if re.search(LOC_WORDS, uq):
        hints.append("User mentions location; filter using buyer_location column with ILIKE for fuzzy matching.")

    if re.search(PRODUCT_WORDS, uq):
        hints.append("This query involves products. Use product_code column (values like Pro01, Pro02, etc.).")

    if re.search(PAYMENT_WORDS, uq):
        hints.append("This query involves payment methods. Use payment_method column (Credit Card, Debit Card, Cash, Mobile Payment).")

    if re.search(QUANTITY_WORDS, uq):
        hints.append("This query involves quantity purchased. Use quantity_purchased column for aggregations.")
        hints.append("For total quantity: SUM(quantity_purchased). For average: AVG(quantity_purchased).")

    if re.search(REP_WORDS, uq):
        hints.append("This query involves sales representatives. Use sales_representative column.")

    if re.search(GENDER_WORDS, uq):
        hints.append("This query involves gender. Use gender column (Male, Female, Other).")

    if re.search(AGE_WORDS, uq):
        hints.append("This query involves age/birth date. Note: buyer_date_of_birth is VARCHAR (Excel date serial number as text).")
        hints.append("To calculate age: EXTRACT(YEAR FROM AGE(CURRENT_DATE, DATE '1899-12-30' + buyer_date_of_birth::INTEGER)).")

    if not hints:
        hints.append("Query sales_data table for transaction information.")

    return sorted(tables), " ".join(hints)


# ===============================
# 2) PROMPT (schema + examples)
# ===============================
EXAMPLES = [
    (
        "Find transactions in San Jose with credit card payment",
        """SELECT buyer_first_name || ' ' || buyer_last_name AS buyer_name,
               buyer_location,
               payment_method,
               quantity_purchased,
               product_code,
               DATE '1899-12-30' + transaction_date::INTEGER AS transaction_date
        FROM sales_data
        WHERE buyer_location ILIKE '%San Jose%'
          AND payment_method ILIKE '%Credit Card%'
        LIMIT 50;"""
    ),
    (
        "Total quantity sold by product",
        """SELECT product_code,
               SUM(quantity_purchased) AS total_quantity
        FROM sales_data
        GROUP BY product_code
        ORDER BY total_quantity DESC
        LIMIT 50;"""
    ),
    (
        "Sales in 2024 by month",
        """SELECT EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER)) AS month,
               COUNT(*) AS transaction_count,
               SUM(quantity_purchased) AS total_quantity
        FROM sales_data
        WHERE EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER)) = 2024
        GROUP BY month
        ORDER BY month;"""
    ),
    (
        "Top 5 sales representatives by quantity sold",
        """SELECT sales_representative,
               SUM(quantity_purchased) AS total_sold
        FROM sales_data
        GROUP BY sales_representative
        ORDER BY total_sold DESC
        LIMIT 5;"""
    ),
    (
        "Find buyers from Chicago who paid with cash",
        """SELECT DISTINCT buyer_first_name || ' ' || buyer_last_name AS buyer_name,
               buyer_location,
               payment_method
        FROM sales_data
        WHERE buyer_location ILIKE '%Chicago%'
          AND payment_method = 'Cash'
        LIMIT 50;"""
    ),
    (
        "Average quantity purchased by gender",
        """SELECT gender,
               AVG(quantity_purchased) AS avg_quantity,
               COUNT(*) AS transaction_count
        FROM sales_data
        GROUP BY gender
        ORDER BY avg_quantity DESC;"""
    )
]

def build_schema_prompt(schema_txt: str, hints: str, user_query: str, limit: int) -> str:
    ex_txt = "\n\n".join([f"Q: {q}\nSQL:\n{sql}" for q, sql in EXAMPLES])
    prompt = f"""
        You are a Text-to-SQL assistant for a PostgreSQL database containing sales transaction data.

        Rules:
        - Output ONE PostgreSQL SELECT query only (no commentary).
        - No INSERT/UPDATE/DELETE/DDL.
        - Use table/column names exactly as in schema.
        - Prefer ILIKE for fuzzy text filter.
        - Add LIMIT {limit} unless user asks otherwise.
        - IMPORTANT: transaction_date and buyer_date_of_birth are stored as VARCHAR (Excel date serial numbers stored as text).
          You MUST cast to INTEGER first: DATE '1899-12-30' + column_name::INTEGER
        - To extract year from date: EXTRACT(YEAR FROM (DATE '1899-12-30' + transaction_date::INTEGER))
        - To extract month: EXTRACT(MONTH FROM (DATE '1899-12-30' + transaction_date::INTEGER))
        - To calculate age from birth date: EXTRACT(YEAR FROM AGE(CURRENT_DATE, DATE '1899-12-30' + buyer_date_of_birth::INTEGER))
        - For buyer full name, concatenate: buyer_first_name || ' ' || buyer_last_name
        - Use aggregation functions (SUM, AVG, COUNT) for quantity analysis.
        - Use GROUP BY when aggregating by categories (product, location, representative, etc.).
        - Use ORDER BY for sorting results (DESC for highest first).

        Schema:
        {schema_txt}

        Hints for this question:
        {hints}

        Examples:
        {ex_txt}

        Now write SQL for:
        "{user_query}"
            """.strip()
    return prompt