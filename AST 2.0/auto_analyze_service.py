"""
auto_analyze_service.py — proactive, one-click dataset intelligence.

Given a CSV, produces:
  1. Up to 5 ranked business insights, each paired with a concrete action
     (reuses profiler.build_auto_analyze_prompt, already written for this).
  2. Up to 3 visualisations chosen specifically to make those insights
     land visually — not a generic "chart every numeric column" dump.

WHY THIS IS A SEPARATE, USER-TRIGGERED ENDPOINT (not automatic on upload)
    Same reasoning as cleaning_agent.py: this costs several real LLM calls
    (1 for insights, 1 for the chart plan, up to 3 more for chart code),
    so it should be something the user knowingly asks for, not a cost
    incurred silently on every upload.

HOW IT REUSES EXISTING INFRASTRUCTURE
    - profiler.py's build_dataset_profile / format_profile_for_llm for the
      statistical picture handed to the LLM (same profile the interactive
      insight box already uses).
    - query_engine.py's build_chart_code_prompt + extract_and_clean_code
      for turning a chart spec into code, identical post-processing to the
      interactive analysis query path.
    - sandbox_limits.run_sandboxed for executing that code — the exact
      same trust boundary every other piece of LLM-generated code in this
      app goes through. Not a new attack surface.

    A single bad chart (LLM proposes an unanswerable spec, or the
    generated code fails) does not fail the whole request — insights
    always come back if they succeeded, and each chart reports its own
    success/error independently.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from cleaning import rule_based_clean
from cache_layer import cached_rule_clean
from file_reader import read_tabular_file


def _read_csv(csv_bytes: bytes, filename: str) -> pd.DataFrame:
    # Thin wrapper kept so call sites below don't need touching — the
    # real work (including Excel support) lives in file_reader.py, shared
    # with analysis_service.py, cleaning_agent.py, and profile_service.py.
    return read_tabular_file(csv_bytes, filename)


def _is_plotly_fig(obj: Any) -> bool:
    t = str(type(obj))
    return "plotly" in t and "Figure" in t


def _parse_insights(raw: str) -> list[dict]:
    """Parses the model's numbered insight list — deliberately block-
    based, not line-based, because real model output comes in at least
    two structurally different shapes:

      (a) single-line, as originally instructed: "1• finding — action"
      (b) multi-line, seen from a real dataset: the finding on its own
          line (often ending in a full sentence), then a SEPARATE line
          holding "*Action:* the actual recommendation" — no shared
          separator character on one line at all, which no amount of
          "try another dash/arrow" fixing could ever have caught, since
          the earlier line-by-line design couldn't see past the first
          newline of a multi-line insight in the first place.

    An explicit "Action:" label, when present, is checked FIRST and is
    the strongest possible signal — it removes all the ambiguity that
    dash-guessing exists to resolve (a dash INSIDE the finding's own
    reasoning, like "0.876 correlation** – a 10% lift...", is not a
    finding/action boundary at all, and a label-based split never
    confuses the two). The dash/arrow heuristics only run as a fallback
    for the single-line shape, where no such label exists.

    Every markdown emphasis marker (**bold**, *italic*) is stripped
    up front — asterisks are never legitimate content here, and Python
    could re-derive this from first principles instead of patching one
    quote style at a time, but that will always be true of any format-
    parsing code and there's a limit to how much can be verified without
    a live model to test against."""
    cleaned = raw.replace("*", "")

    # One block per insight, however many lines it spans — not one
    # block per line.
    blocks = re.split(r"(?=^\s*\d+[•.\)])", cleaned, flags=re.MULTILINE)

    insights = []
    for block in blocks:
        block = block.strip()
        m = re.match(r"^\d+[•.\)]\s*(.+)", block, re.DOTALL)
        if not m:
            continue
        body = m.group(1).strip()

        action_match = re.search(r"(?:Recommended\s+)?Action\s*:\s*(.+)", body, re.IGNORECASE | re.DOTALL)
        if action_match:
            finding = body[: action_match.start()]
            action = action_match.group(1)
        else:
            joined = " ".join(line.strip() for line in body.splitlines() if line.strip())
            if "—" in joined:
                finding, _, action = joined.partition("—")
            elif "–" in joined:
                finding, _, action = joined.partition("–")
            elif " - " in joined:
                finding, _, action = joined.partition(" - ")
            elif "→" in joined:
                finding, _, action = joined.rpartition("→")
            elif "->" in joined:
                finding, _, action = joined.rpartition("->")
            else:
                continue

        # Collapse any internal newlines/repeated whitespace from a
        # multi-line block into a single clean line either way.
        finding = " ".join(finding.split())
        action = " ".join(action.split())
        if finding and action:
            insights.append({"finding": finding, "action": action})
    return insights[:5]


def _parse_chart_plan(raw: str) -> list[dict]:
    """Parses build_chart_plan_prompt's TITLE/SUPPORTS/SPEC blocks.

    Anchored on "TITLE:" rather than a numbered "CHART N:" header — a
    numbered header is one more thing the model has to reproduce
    verbatim, and any deviation (different numbering, missing colon,
    markdown bold around it) used to make the whole block invisible to
    the old regex. Tolerant of markdown emphasis (**TITLE:**) and case,
    since real model output varies here even when explicitly told not
    to."""
    # Strip markdown bold/italic markers that models add despite being
    # asked not to — "**TITLE:**" and "TITLE:" should parse identically.
    # Strip markdown bold only — not bare underscores/asterisks, since a
    # SPEC line can legitimately reference snake_case column names
    # (e.g. "product_line") that stripping would corrupt into
    # "productline", silently feeding a wrong column name to the next
    # LLM call. Same bug, same fix as cleaning_agent.py's defect parser.
    cleaned = raw.replace("**", "")

    blocks = re.split(r"(?=^\s*TITLE\s*:)", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    charts = []
    for block in blocks:
        title_m = re.search(r"TITLE\s*:\s*(.+)", block, re.IGNORECASE)
        if not title_m:
            continue
        supports_m = re.search(r"SUPPORTS\s*:\s*(.+)", block, re.IGNORECASE)
        spec_m = re.search(r"SPEC\s*:\s*(.+)", block, re.IGNORECASE)
        if title_m and spec_m:
            charts.append({
                "title": title_m.group(1).strip(),
                "supports": supports_m.group(1).strip() if supports_m else "",
                "spec": spec_m.group(1).strip(),
            })
    return charts[:3]


def _build_chart_repair_prompt(schema_text: str, spec: str, bad_code: str, error: str) -> str:
    """Second-chance prompt: the first attempt failed in the sandbox, so show
    the model its own code and the exact error and ask for a rewrite."""
    from query_engine import build_chart_code_prompt
    base = build_chart_code_prompt(schema_text, spec)
    return (
        base
        + "\n\nYOUR PREVIOUS ATTEMPT FAILED.\n"
        + "Previous code:\n```python\n" + bad_code + "\n```\n"
        + "Error: " + str(error) + "\n\n"
        + "Rewrite the code so it works. If the error says a method or attribute is "
        + "'blocked in the sandbox', do NOT use it — achieve the same result with "
        + "different, basic pandas operations (groupby/agg/value_counts/reset_index/"
        + "sort_values) and plotly.express. Output ONLY the corrected Python code."
    )


def _run_one_chart(plan: dict, schema_text: str, df: pd.DataFrame, session_id: str) -> dict:
    """Turns one chart plan entry into a rendered chart or a clean,
    isolated error — never raises, so one bad chart can't take the
    other two down with it.

    A failed attempt (sandbox block, runtime error, or a result that isn't a
    Plotly figure) gets ONE repair retry: the model is shown its own code and
    the error and asked to rewrite it. Generated code differs run to run, so
    a single blocked method shouldn't cost the user a whole chart."""
    from query_engine import build_chart_code_prompt, llm_with_fallback, extract_and_clean_code
    from sandbox_limits import run_sandboxed

    base = {"title": plan["title"], "why": plan.get("supports", "")}
    MAX_ATTEMPTS = 2
    last_error = "unknown error"
    code = ""
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt == 1:
                prompt = build_chart_code_prompt(schema_text, plan["spec"])
            else:
                prompt = _build_chart_repair_prompt(schema_text, plan["spec"], code, last_error)
                print(f"[auto-analyze] chart '{plan['title']}' attempt 1 failed ({last_error}); retrying once", flush=True)

            raw = llm_with_fallback(prompt, temperature=0.1, session_id=session_id)
            code = extract_and_clean_code(raw)

            sbr = run_sandboxed(code, df, timeout_seconds=20)
            if not sbr.success:
                last_error = sbr.error
                continue
            if not _is_plotly_fig(sbr.value):
                last_error = "Generated chart wasn't a valid Plotly figure"
                continue

            return {
                **base, "resultType": "chart",
                "resultData": json.loads(sbr.value.to_json()), "error": None,
            }

        return {**base, "resultType": "error", "resultData": None, "error": last_error}
    except Exception as e:
        return {**base, "resultType": "error", "resultData": None, "error": f"Chart generation failed: {e}"}


def run_auto_analyze(csv_bytes: bytes, filename: str, session_id: str) -> dict:
    """
    1. Parse + rule-clean the CSV (identical prep to /analyze and /clean).
    2. Build the dataset profile, ask the LLM for up to 5 ranked
       insight+action pairs.
    3. Ask the LLM to pick up to 3 charts that best support those
       insights, then generate + sandbox-execute each independently.

    Returns:
        {success, filename, insights: [{finding, action}],
         charts: [{title, why, resultType, resultData, error}], error}
    """
    try:
        df = _read_csv(csv_bytes, filename)
    except Exception as e:
        return {"success": False, "error": f"Could not parse CSV: {e}"}

    try:
        df, _rule_log = cached_rule_clean(df, csv_bytes, rule_based_clean)

        from profiler import (
            build_dataset_profile,
            format_profile_for_llm,
            build_auto_analyze_prompt,
            build_chart_plan_prompt,
        )
        from query_engine import insight_llm

        profile = build_dataset_profile(df, _cache_key="auto-analyze")
        profile_str = format_profile_for_llm(profile)

        # LIVE BUG: insights generation used to go through llm_with_fallback
        # (query_engine.py) which — despite the generic name — ALWAYS calls
        # llm_code_call, i.e. the smaller code-specialised model
        # (_CODE_MODEL = "openai/gpt-oss-20b"), never the larger
        # _CHAT_MODEL ("openai/gpt-oss-120b") that query_engine.insight_llm
        # exists specifically to reach. That's backwards for this prompt:
        # build_auto_analyze_prompt asks for business prose in an exact
        # numbered format, not code, and the smaller model was measurably
        # less reliable at holding that format on some real datasets — a
        # small (7-row), numerically dense table extracted from a PDF
        # (13 near-identical monthly columns) produced a response
        # _parse_insights couldn't find a single "N•" block in, surfaced
        # to the user as "Could not generate insights for this dataset."
        # insight_llm/the 120B chat model is already wired up and used
        # correctly elsewhere in this file's sibling function
        # (query_engine.insight_llm) — every other prompt in this app
        # that must produce valid Python/SQL stays on llm_with_fallback;
        # this one and the chart-plan prompt below don't produce code at
        # all, so they're the ones that should have been on the chat
        # model all along.
        insights_prompt = build_auto_analyze_prompt(profile_str)
        insights_raw = insight_llm(insights_prompt, temperature=0.3)
        insights = _parse_insights(insights_raw)

        if not insights:
            # Same reasoning as the chart-plan-parsing failure below:
            # print the raw response so a real parsing gap is debuggable
            # from server logs, instead of looking identical to "the
            # model genuinely had nothing to say about this dataset."
            print(f"[auto-analyze] 0 insights parsed. Raw LLM response was:\n{insights_raw}", flush=True)
            return {
                "success": False,
                "error": "Could not generate insights for this dataset — try again.",
            }

        insights_text = "\n".join(
            f"{i+1}• {ins['finding']} — {ins['action']}" for i, ins in enumerate(insights)
        )

        charts: list[dict] = []
        charts_error = None
        try:
            # Same fix as the insights call above — this plans WHICH
            # charts to draw (title/spec text), it doesn't write chart
            # code itself (that's build_chart_code_prompt, in
            # _run_one_chart below, which correctly stays on
            # llm_with_fallback/the code model). Planning prose belongs
            # on insight_llm/the chat model for the same reliability
            # reason.
            plan_prompt = build_chart_plan_prompt(profile_str, insights_text)
            plan_raw = insight_llm(plan_prompt, temperature=0.2)
            chart_plans = _parse_chart_plan(plan_raw)

            if not chart_plans:
                # Don't fail the whole request — insights are still good —
                # but DON'T silently return an empty list either. Print the
                # raw response so this is actually debuggable from server
                # logs, and surface a real error to the frontend instead of
                # a visualizations section that just quietly never appears.
                print(f"[auto-analyze] chart plan parsing found 0 blocks. "
                      f"Raw LLM response was:\n{plan_raw}", flush=True)
                charts_error = "Could not parse a chart plan from the model's response."
            else:
                schema_text = ", ".join(df.columns.tolist())
                charts = [
                    _run_one_chart(plan, schema_text, df, session_id)
                    for plan in chart_plans
                ]
        except Exception as e:
            # Chart planning itself failed (not an individual chart) —
            # insights are still good, return them without charts rather
            # than failing the whole request, but say why.
            print(f"[auto-analyze] chart planning raised: {e}", flush=True)
            charts_error = f"Chart planning failed: {e}"

        return {
            "success": True,
            "filename": filename,
            "insights": insights,
            "charts": charts,
            "chartsError": charts_error,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "error": f"Auto-analyze failed: {e}"}


def _format_computed_value(value) -> str:
    """Turns a sandbox result (scalar, Series, or DataFrame) into a
    compact string safe to drop into the answer prompt — never the full
    table, since that would bloat the prompt for no grounding benefit
    beyond the first several rows."""
    if isinstance(value, pd.DataFrame):
        return value.head(10).to_string(index=False)
    if isinstance(value, pd.Series):
        return value.head(10).to_string()
    return str(value)


def _extract_all_code_blocks(raw: str) -> list[str]:
    """Extracts every ```python ... ``` fenced block, not just the
    first — the decision step can now propose up to 3 separate angles
    to compute, each its own block, each run independently through the
    sandbox."""
    import re
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return [b.strip() for b in blocks if b.strip() and b.strip().upper() != "NO_CODE_NEEDED"]


def answer_followup(
    csv_bytes: bytes,
    filename: str,
    insights: list[dict],
    question: str,
    prior_qa: list[dict],
    session_id: str,
) -> dict:
    """
    Answers a follow-up question about the insights already shown to the
    user. Two LLM calls, deliberately kept separate — see
    profiler.build_followup_decision_prompt's docstring for why:
      1. DECIDE — does this need a real number computed from the data?
         If yes, write+run code through the same sandbox every other
         piece of LLM-generated code in this app goes through.
      2. ANSWER — compose the reply, grounded in (in order of trust) any
         computed value, the insights already on screen, and explicitly
         told to say what it doesn't know rather than invent it.
    """
    try:
        df = _read_csv(csv_bytes, filename)
        df, _ = cached_rule_clean(df, csv_bytes, rule_based_clean)
    except Exception as e:
        return {"success": False, "error": f"Could not load dataset: {e}"}

    try:
        from profiler import build_followup_decision_prompt, build_followup_answer_prompt
        from query_engine import llm_with_fallback, insight_llm
        from sandbox_limits import run_sandboxed

        insights_text = "\n".join(
            f"{i+1}. {ins['finding']} → {ins['action']}" for i, ins in enumerate(insights)
        ) or "(none)"
        prior_qa_text = ""
        if prior_qa:
            prior_qa_text = "\nEarlier in this conversation:\n" + "\n".join(
                f"Q: {qa['question']}\nA: {qa['answer']}" for qa in prior_qa
            ) + "\n"

        schema_text = ", ".join(df.columns.tolist())
        decision_prompt = build_followup_decision_prompt(schema_text, insights_text, prior_qa_text, question)
        decision_raw = llm_with_fallback(decision_prompt, temperature=0.1, session_id=session_id)

        computed_facts: list[str] = []
        if "NO_CODE_NEEDED" not in decision_raw.upper():
            for code in _extract_all_code_blocks(decision_raw)[:3]:
                sbr = run_sandboxed(code, df, include_charts=False, timeout_seconds=15)
                if sbr.success and sbr.produced:
                    computed_facts.append(_format_computed_value(sbr.value))
                else:
                    # One angle failing doesn't sink the others — proceed
                    # with whichever angles DID compute successfully.
                    print(f"[auto-analyze followup] one angle's computation "
                          f"failed, skipping it: {sbr.error}", flush=True)

        # decision_prompt above stays on llm_with_fallback/the code model —
        # it can legitimately emit Python snippets. This step only ever
        # writes the conversational reply (see build_followup_answer_prompt),
        # never code, so it gets the same insight_llm/chat-model fix as
        # run_auto_analyze's insights and chart-plan calls above, for the
        # same reliability reason — not yet confirmed broken on real data
        # the way those two were, but there's no reason to leave a pure
        # prose prompt on the weaker code-specialised model once the
        # pattern's been found.
        # AI SECURITY: computed_facts are real rows/values pulled from the
        # user's file by the sandbox — the one place real data reached an
        # LLM prompt WITHOUT passing through profiler.py's redaction. Mask
        # likely PII patterns and fence each fact as untrusted data (a cell
        # reading "ignore previous instructions" must not be obeyed). Name
        # columns are NOT withheld here: the user asked about their own
        # data and may legitimately need "who is the top customer" answered.
        from pii_redaction import redact_pii_logged
        from prompt_injection_guard import wrap_untrusted
        computed_facts = [
            wrap_untrusted(redact_pii_logged(f, context="follow-up computed fact"), source_label="computed result")
            for f in computed_facts
        ]
        answer_prompt = build_followup_answer_prompt(insights_text, prior_qa_text, computed_facts, question)
        answer = insight_llm(answer_prompt, temperature=0.2)

        return {"success": True, "answer": answer.strip(), "error": None}
    except Exception as e:
        return {"success": False, "error": f"Follow-up failed: {e}"}
