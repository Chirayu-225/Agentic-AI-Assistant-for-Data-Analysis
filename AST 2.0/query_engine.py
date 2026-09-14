"""
query_engine.py — Query routing, prompt construction, and LLM calls.

Architecture:
  1. Shortcut layer  — handles simple queries with direct pandas (no LLM).
  2. Prompt builder  — constructs a hardened, context-aware prompt.
  3. LLM call        — routed through llm_provider (Groq API).
  4. Code cleanup    — strips imports, print(), plt.show(), adds plt.gcf().
"""

import re
import pandas as pd

from llm_provider import llm_code_call, llm_chat_call


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHORTCUT LAYER
# ─────────────────────────────────────────────────────────────────────────────

def detect_shortcut(query: str, df: pd.DataFrame) -> str | None:
    q    = query.lower().strip()
    cols = df.columns.tolist()

    # Chart requests must always reach the LLM — none of the shortcuts
    # below produce a chart, so a keyword match here (e.g. "total" inside
    # "plot total sales as a bar chart") would otherwise silently return
    # a scalar/table instead of the chart the user actually asked for.
    CHART_KEYWORDS = (
        "plot", "chart", "graph", "visuali", "scatter", "histogram",
        "heatmap", "pie ",
    )
    if any(kw in q for kw in CHART_KEYWORDS):
        return None

    def _find_col(*keywords) -> str | None:
        for kw in keywords:
            for col in cols:
                if kw in col.lower():
                    return col
        return None

    if any(p in q for p in ["how many rows", "count rows", "number of rows", "row count"]):
        return "result = len(df)"
    if any(p in q for p in ["how many columns", "count columns", "number of columns", "column count"]):
        return "result = len(df.columns)"
    if any(p in q for p in ["show columns", "list columns", "column names", "what columns", "show all columns"]):
        return "result = pd.DataFrame({'column': df.columns.tolist(), 'dtype': [str(d) for d in df.dtypes.values]})"
    if any(p in q for p in ["null", "missing", "nan", "empty values"]):
        return "result = df.isnull().sum().rename('null_count').to_frame()"
    if any(p in q for p in ["describe", "summary stats", "statistical summary", "statistics"]):
        return "result = df.describe(include='all')"
    if any(p in q for p in ["first 5", "first five", "show first", "head"]):
        n = 10 if ("10" in q or "ten" in q) else 5
        return f"result = df.head({n})"
    if any(p in q for p in ["last 5", "last five", "show last", "tail"]):
        n = 10 if ("10" in q or "ten" in q) else 5
        return f"result = df.tail({n})"
    if any(p in q for p in ["duplicate", "duplicates", "duplicate rows"]):
        return "result = df.duplicated().sum()"
    if any(p in q for p in ["shape", "dimensions", "size of"]):
        return "result = pd.DataFrame({'rows': [len(df)], 'columns': [len(df.columns)]})"
    if any(p in q for p in ["data types", "dtypes", "column types"]):
        return "result = df.dtypes.rename('dtype').to_frame()"
    if any(p in q for p in ["mean of", "average of", "avg of"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].mean()"
    if "sum of" in q or ("total" in q and "total rows" not in q):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].sum()"
    if any(p in q for p in ["maximum", "max of", "highest"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].max()"
    if any(p in q for p in ["minimum", "min of", "lowest"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].min()"
    if any(p in q for p in ["unique values", "distinct values", "value counts"]):
        col = _find_col(*q.split())
        if col:
            return f"result = df['{col}'].value_counts().to_frame('count')"
    if any(p in q for p in ["sort by", "order by", "sorted by"]):
        col = _find_col(*q.split())
        ascending = "ascending" in q or "asc" in q
        if col:
            return f"result = df.sort_values('{col}', ascending={ascending}).head(20)"
    if any(p in q for p in ["correlation", "corr matrix", "correlations"]):
        return "result = df.select_dtypes(include='number').corr().round(3)"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_prompt(schema_text: str, user_query: str, history_text: str = "") -> str:
    return f"""You are an expert Python data analyst.

Dataset information:
DataFrame name: df
Exact column names (use ONLY these, do not guess or invent):
{schema_text}
{history_text}
User request:
{user_query}

Available libraries (already imported, do NOT import anything):
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go
- matplotlib.pyplot as plt (fallback only)

Rules:
- Generate ONLY valid Python code. No explanations, no markdown, no imports.
- CRITICAL: ONLY use column names from the exact list above.
- CRITICAL: Before using a column verify it exists: if 'col_name' not in df.columns: raise ValueError('Column not found')
- DO NOT use print() anywhere — it is not available in the sandbox.
- IF calculating a number or table, assign it to a variable named 'result'.
- IF drawing ANY chart or visualisation:
    * PREFER plotly.express (px) or plotly.graph_objects (go) for interactive charts.
    * For bar charts use: result = px.bar(df, x='col', y='col')
    * For line charts use: result = px.line(df, x='col', y='col')
    * For scatter plots use: result = px.scatter(df, x='col', y='col')
    * For pie charts use: result = px.pie(df, names='col', values='col')
    * For histograms use: result = px.histogram(df, x='col')
    * For heatmaps use: result = px.imshow(df.select_dtypes(include='number').corr())
    * Assign the figure directly to result — do NOT call .show()
    * If using matplotlib, the LAST line MUST be: result = plt.gcf()
"""


def build_chart_code_prompt(schema_text: str, chart_spec: str) -> str:
    """Used by auto_analyze_service.py — turns one chart plan SPEC line
    (natural language) into executable plotting code, reusing the same
    rules/guardrails as build_analysis_prompt's chart branch."""
    return f"""You are an expert Python data analyst generating ONE chart.

Dataset information:
DataFrame name: df
Exact column names (use ONLY these, do not guess or invent):
{schema_text}

Chart to build:
{chart_spec}

Available libraries (already imported, do NOT import anything):
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go

Rules:
- Generate ONLY valid Python code. No explanations, no markdown, no imports.
- CRITICAL: ONLY use column names from the exact list above.
- CRITICAL: Before using a column verify it exists: if 'col_name' not in df.columns: raise ValueError('Column not found')
- DO NOT use print() anywhere — it is not available in the sandbox.
- Use plotly.express (px) or plotly.graph_objects (go). Assign the final
  figure directly to a variable named 'result'. Do NOT call .show().
"""


def build_retry_prompt(code: str, error: str) -> str:
    return f"""The following Python code failed with error: {error}

Failing code:
{code}

Fix ONLY the error. Do not rewrite everything. Keep the same logic.
DO NOT use print() — it is not available.
Output ONLY the corrected Python code block. No explanations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM CALLS — backed by llm_provider (Groq API)
# ─────────────────────────────────────────────────────────────────────────────

def llm_with_fallback(prompt_text: str, temperature: float = 0.1,
                      primary: str = "openai/gpt-oss-20b", session_id: str = "") -> str:
    return llm_code_call(prompt_text, temperature, session_id=session_id)


def insight_llm(prompt_text: str, temperature: float = 0.3) -> str:
    return llm_chat_call(prompt_text, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODE POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_clean_code(raw_output: str) -> str:
    """
    Extract code block from LLM output and sanitize:
    - Strips import statements
    - Strips print() calls (blocked by sandbox — primary cause of failures)
    - Strips plt.show() / fig.show()
    - Auto-appends result = plt.gcf() for matplotlib if missing
    """
    match = re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60", raw_output, re.DOTALL)
    code  = match.group(1).strip() if match else raw_output.strip()

    # Strip line by line — imports AND print() calls
    cleaned_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        # Strip standalone print(...) calls — these crash in the sandbox
        # Matches: print(...), print_(...)  at start of line (with any indent)
        if re.match(r'^\s*print(_)?\s*\(', line):
            continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    # Strip show() calls
    code = code.replace("plt.show()", "")
    code = re.sub(r'\b\w*\.show\(\)', '', code)  # fig.show(), result.show() etc.

    # Auto-append plt.gcf() for matplotlib only (not Plotly)
    uses_plotly = "px." in code or "go." in code or "plotly" in code.lower()
    uses_mpl    = "plt." in code
    has_result  = "result = plt.gcf()" in code or "result=plt.gcf()" in code

    if uses_mpl and not uses_plotly and not has_result:
        code += "\nresult = plt.gcf()"

    return code.strip()
