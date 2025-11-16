from __future__ import annotations

# ===================== Imports =====================
import os
import re
import sys
import logging
from typing import List, Dict, Tuple, Any, Optional

import sqlglot
import sqlglot.expressions as exp
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from unidecode import unidecode
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from .selector_and_prompt import build_schema_prompt, selector_lite
    from .schema_utils import load_schema, render_schema, SchemaSummary
    from ..config.config import DATABASE_URL, GEMINI_API_KEY
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.selector_and_prompt import build_schema_prompt, selector_lite
    from core.schema_utils import load_schema, render_schema, SchemaSummary
    from config.config import DATABASE_URL, GEMINI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("text2sql")


# ===================== Global Engine (Connection Pool) =====================
_global_engine: Optional[Engine] = None

def get_engine() -> Engine:
    """Get or create a global engine with connection pooling."""
    global _global_engine
    if _global_engine is None:
        _global_engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,         
            max_overflow=10,       
            pool_pre_ping=True,   
            pool_recycle=3600,
            future=True
        )
    return _global_engine



# ===================== LLM Adapter =====================
class LLM:
    """Thin wrapper to extract SQL from LLM output (removes ```sql ... ``` markers)."""
    def __init__(self, invoke_fn):
        self._invoke = invoke_fn

    def gen(self, prompt: str) -> str:
        out = self._invoke(prompt).strip()
        out = re.sub(r"^```(?:sql|SQL)?\s*|\s*```$", "", out, flags=re.MULTILINE).strip()
        return out


# ===================== Guards =====================
SAFE_SELECT = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
FORBIDDEN   = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b",
    re.IGNORECASE,
)

def sql_guard(sql: str) -> None:
    """Only allow SELECT queries and block dangerous keywords."""
    if not SAFE_SELECT.search(sql):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(sql):
        raise ValueError("Dangerous SQL keyword detected.")

def _extract_qualified_cols(sql: str) -> List[Tuple[str, str]]:
    """
    Return list of (table_or_alias, column_name) for qualified columns (t.col).
    """
    out: List[Tuple[str, str]] = []
    try:
        node = sqlglot.parse_one(sql, read="postgres")
        for col in node.find_all(exp.Column):
            table_name = None
            if col.table:
                if isinstance(col.table, exp.Identifier):
                    table_name = col.table.this
                else:
                    table_name = str(col.table)
            if table_name:
                out.append((table_name, col.name))
    except Exception:
        pass
    return out

# ================================
# 2) Schema guard
# ================================
def schema_guard(sql: str, schema: 'SchemaSummary') -> Optional[str]:
    """
    Validate table/column names using sqlglot AST (no regex).
    Returns warning string if invalid; otherwise None.
    """
    # known tables & columns
    known_tables = set(schema.tables.keys())
    # chỉ lấy tên cột, tránh ColumnInfo unhashable
    table_cols: Dict[str, set] = {
        t.name: set(c.name for c in t.columns)
        for t in schema.tables.values()
    }

    # parse once
    try:
        node = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        # nếu parse lỗi thì không guard quá gắt
        return None

    # collect tables + alias map
    used_tables: set[str] = set()
    alias_map: Dict[str, str] = {}  # alias -> real table

    for t in node.find_all(exp.Table):
        real_name = t.this.name if isinstance(t.this, exp.Identifier) else str(t.this)
        used_tables.add(real_name)

        # alias, if any
        if t.alias:
            alias_name = None
            if isinstance(t.alias, exp.TableAlias):
                if isinstance(t.alias.this, exp.Identifier):
                    alias_name = t.alias.this.this
                else:
                    alias_name = str(t.alias.this)
            elif isinstance(t.alias, exp.Identifier):
                alias_name = t.alias.this
            elif isinstance(t.alias, str):
                alias_name = t.alias
            if alias_name:
                alias_map[alias_name] = real_name
        else:
            alias_map[real_name] = real_name

    # unknown tables?
    unknown_tbls = sorted([t for t in used_tables if t not in known_tables])
    if unknown_tbls:
        return f"Unknown tables: {unknown_tbls}. Allowed: {sorted(known_tables)}."

    # qualified columns check
    bad_cols: List[Tuple[str, str, str]] = []  
    for alias, col in set(_extract_qualified_cols(sql)):
        alias_str = str(alias)
        real_table = alias_map.get(alias_str, alias_str)
        if real_table in table_cols and col not in table_cols[real_table]:
            bad_cols.append((alias_str, real_table, col))

    if bad_cols:
        msg = ", ".join([f"{a}.{c} (table {t})" for a, t, c in bad_cols])
        return f"Unknown columns: {msg}. Please use only existing columns per schema."

    return None

# ===================== SQL exec & refine =====================
def run_sql(engine: Engine, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    with engine.connect() as conn:
        rs = conn.execute(sa_text(sql))
        rows = rs.fetchall()
        cols = list(rs.keys())
    return cols, rows

def refine_prompt(schema_txt: str, user_query: str, prev_sql: str, reason: str) -> str:
    return f"""
        The schema is:
        {schema_txt}

        User question:
        {user_query}

        The previous SQL was:
        {prev_sql}

        It failed or returned empty because:
        {reason}

        Please return ONLY a corrected PostgreSQL SELECT query (no explanation).
    """.strip()


# ===================== Fallbacks & helpers =====================
def _extract_missing_tables_from_warn(warn: str) -> List[str]:
    m = re.search(r"Unknown tables:\s*\[([^\]]+)\]", warn or "")
    if not m:
        return []
    raw = m.group(1)
    return [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]

def _db_has_tables(schema: SchemaSummary, names: List[str]) -> bool:
    known = set(schema.tables.keys())
    return all(n in known for n in names)

def _product_terms_from_query(q: str) -> List[str]:
    """
    Extract product codes from query (Pro01, Pro02, etc.)
    """
    qn = unidecode((q or "").lower())
    vocab = [f"pro{i:02d}" for i in range(1, 11)]  # Pro01 to Pro10
    return [kw for kw in vocab if kw in qn]

def _fallback_sql_sales_product(products: List[str], limit: int) -> str:
    """
    Fallback SQL for product-based queries in sales_data.
    """
    product_conditions = " OR ".join(
        [f"product_code ILIKE '%{p}%'" for p in products]
    ) or "TRUE"
    return f"""
        SELECT buyer_first_name || ' ' || buyer_last_name AS buyer_name,
               buyer_location,
               product_code,
               quantity_purchased,
               payment_method
        FROM sales_data
        WHERE {product_conditions}
        LIMIT {limit}
        """.strip()


# ===================== Main Query Orchestrator =====================
def answer_sql(
    engine: Engine,
    llm: LLM,
    user_query: str,
    max_refine: int = 1,
    limit: int = 10,
    *,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    tables, hints = selector_lite(user_query)
    schema = load_schema(engine, only=tables)
    schema_txt = render_schema(schema)

    prompt = build_schema_prompt(schema_txt, hints, user_query, limit)
    sql = llm.gen(prompt)

    trials: List[Tuple[str, str]] = []

    for attempt in range(max_refine + 1):
        try:
            sql_guard(sql)

            warn = schema_guard(sql, schema)
            if warn and "Unknown tables:" in warn and attempt < max_refine:
                missing = _extract_missing_tables_from_warn(warn)
                schema = load_schema(engine, only=None)         
                schema_txt = render_schema(schema)

                if _db_has_tables(schema, missing):
                    sql = llm.gen(build_schema_prompt(schema_txt, hints, user_query, limit))
                    trials.append((sql, "widened schema due to unknown tables"))
                    continue
                else:
                    product_terms = _product_terms_from_query(user_query)
                    if product_terms:
                        sql = _fallback_sql_sales_product(product_terms, limit)
                        warn = None
                    else:
                        return {
                            "sql": sql, 
                            "columns": [], 
                            "rows": [],
                            "trials": trials + [(sql, warn)], 
                            "warning": warn,
                        }

            cols, raw_rows = run_sql(engine, sql)
            cols = list(cols or [])

            packed_rows: List[List[Any]] = [list(row) for row in raw_rows]

            if len(packed_rows) == 0 and attempt < max_refine:
                trials.append((sql, "empty result"))
                sql = llm.gen(refine_prompt(schema_txt, user_query, sql, "empty result"))
                continue

            return {"sql": sql, "columns": cols, "rows": packed_rows, "trials": trials}

        except Exception as e:
            if attempt < max_refine:
                trials.append((sql, str(e)))
                sql = llm.gen(refine_prompt(schema_txt, user_query, sql, str(e)))
            else:
                log.exception("Query failed")
                return {
                    "sql": sql, "columns": [], "rows": [],
                    "trials": trials + [(sql, f"db_error: {e}")],
                    "warning": "Query failed at execution. Returned empty result.",
                }

# ===================== Main Function for API Usage =====================
def text_to_sql(user_query: str, max_refine: int = 1, limit: int = 10) -> Dict[str, Any]:
    engine = get_engine()
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
    
    def _invoke(prompt: str) -> str:
        resp = gemini.invoke([HumanMessage(content=prompt)])
        return resp.content
    
    llm = LLM(_invoke)
    return answer_sql(engine, llm, user_query, max_refine=max_refine, limit=limit)


if __name__ == "__main__":
    # Test the function
    test_queries = [
        "Find total sales quantity by product code",
        "Show top 5 sales representatives",
        "Average quantity by gender"
    ]
    
    for i, q in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {q}")
        print('='*80)
        
        result = text_to_sql(q, max_refine=1, limit=10)
        
        print("\nGenerated SQL:")
        print(result["sql"])
        print(f"\nColumns: {result['columns']}")
        print(f"\nResults (first 5):")
        for j, row in enumerate(result["rows"][:5], 1):
            print(f"  {j}. {row}")
        print(f"\nTotal rows returned: {len(result['rows'])}")
        
        if result.get("trials"):
            print(f"Trials: {len(result['trials'])}")
    
    print(f"\n{'='*80}")
    print("✓ All queries completed using connection pool")
    print('='*80)