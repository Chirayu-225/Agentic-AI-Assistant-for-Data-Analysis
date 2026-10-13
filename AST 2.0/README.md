# Legacy Streamlit app (superseded)

`e.py` and `main.py` were the original single-process Streamlit + Ollama
version of Analyst Assistant. Neither is imported by `api.py` or anything
else in the current FastAPI + Next.js stack — they're dead code from the
current app's point of view.

They're kept here (rather than deleted) only because `profiler.py`'s
`build_auto_analyze_prompt`/chart-selection logic was originally written
for this Streamlit UI, and it's genuinely useful reference if you ever
want to trace where a prompt's wording or a chart-selection heuristic
came from.

If you're confident you don't need that history, this whole folder is
safe to delete — nothing in `api.py`, `analysis_service.py`,
`rag_service.py`, `cleaning_agent.py`, or `auto_analyze_service.py`
references it.
