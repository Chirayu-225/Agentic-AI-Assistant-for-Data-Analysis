"""
cleaning_agent.py — autonomous data quality agent.

Runs AFTER rule_based_clean() (cleaning.py), which handles the
deterministic, pattern-matchable stuff cheaply and identically every
run — currency symbols, percentages, thousands separators, identifier
columns, sentinel nulls. This module exists for everything a fixed set
of regexes structurally can't catch: inconsistent category spellings
("USA" / "U.S." / "United States"), implausible-but-not-malformed
values, meaningful missing values, and other defects that require
actually understanding what a column represents — not just its dtype.

Not every data quality problem shows up as a pre-written pattern. This
agent's job is to look at what the deterministic pass couldn't, and
decide — not to have every case anticipated in advance.

TWO-STEP DESIGN: DIAGNOSE, THEN FIX
    Deliberately two separate LLM calls, not one combined "find and fix"
    prompt:
      1. DIAGNOSE — list concrete defects found, each with evidence.
      2. FIX — given that list, decide and write the code to fix them.
    This makes the diagnosis an inspectable artifact in its own right —
    what was found, with evidence, is visible even if the fix step
    later fails — rather than a black box that just hands back altered
    data with no record of what it thought was wrong or why.

WHY THIS IS A SEPARATE, USER-TRIGGERED STEP (not automatic)
    1. It costs 1-2 real LLM calls — rule-based cleaning is free and
       instant. Unlike the old nulls-only version of this agent, it
       runs its diagnosis step even on a dataset with zero missing
       values, since spelling/outlier defects don't require nulls to
       exist — so this can't be gated behind "only when nulls remain"
       the way the narrower agent used to be.
    2. It is NOT perfectly deterministic, unlike rule_based_clean — the
       same dataset can get slightly different reasoning on different
       runs. The user should knowingly opt into non-deterministic
       cleaning decisions, not have them silently applied under the
       same "cleaning" label as the fully deterministic rule-based pass.

HOW IT REUSES EXISTING INFRASTRUCTURE
    The agent's generated fix code runs through sandbox_limits.run_sandboxed
    — the exact same allowlist-proxied, timeout/memory-capped pipeline
    every other piece of LLM-generated code in this app goes through.
    This is not a new trust boundary; it's the existing one doing new
    work. (Building the original version of this agent surfaced one
    real, necessary gap: the sandbox proxy's allowlist didn't include
    interpolate/ffill/bfill — legitimate, file-I/O-free operations an
    agent reasoning about time-ordered data would obviously reach for.
    Added to sandbox_proxy.py alongside the original agent.)
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.where(pd.notnull(df), None)
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _read_csv(csv_bytes: bytes, filename: str) -> pd.DataFrame:
    from file_reader import read_tabular_file
    return read_tabular_file(csv_bytes, filename)


def _parse_defects(raw: str) -> list[dict]:
    """Parses profiler.build_defect_diagnosis_prompt's COLUMN/TYPE/
    EVIDENCE/ACTION blocks. Anchored on "COLUMN:" rather than a numbered
    header, and tolerant of markdown/casing — the same lesson learned
    the hard way from auto_analyze_service's chart-plan parser, which
    silently returned zero results the first time it met real (rather
    than hand-written mock) LLM output. Caps at 8 defects even if the
    model ignores the prompt's own cap, so one runaway response can't
    balloon the cost of the fix step that follows."""
    if "NO DEFECTS FOUND" in raw.upper():
        return []

    # Strip markdown bold (the common "**LABEL:**" case) — but NOT bare
    # underscores/asterisks, since legitimate values here are snake_case
    # (inconsistent_categories, missing_values) and stripping those
    # corrupts the value itself rather than cleaning formatting.
    cleaned = raw.replace("**", "")
    blocks = re.split(r"(?=^\s*COLUMN\s*:)", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    defects = []
    for block in blocks:
        col_m = re.search(r"COLUMN\s*:\s*(.+)", block, re.IGNORECASE)
        if not col_m:
            continue
        type_m = re.search(r"TYPE\s*:\s*(.+)", block, re.IGNORECASE)
        evidence_m = re.search(r"EVIDENCE\s*:\s*(.+)", block, re.IGNORECASE)
        action_m = re.search(r"ACTION\s*:\s*(.+)", block, re.IGNORECASE)
        if col_m and evidence_m and action_m:
            defects.append({
                "column": col_m.group(1).strip(),
                "type": type_m.group(1).strip() if type_m else "other",
                "evidence": evidence_m.group(1).strip(),
                "action": action_m.group(1).strip(),
            })
    return defects[:8]


def _parse_fix_response(raw: str) -> tuple[list[str], str]:
    """Splits the fix-step LLM response into (reasoning_lines, code)."""
    code_match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    reasoning_section = raw[: code_match.start()] if code_match else raw
    reasoning_lines = [
        line.strip() for line in reasoning_section.splitlines()
        if line.strip().startswith("-")
    ]
    return reasoning_lines, code


def run_cleaning_agent(
    csv_bytes: bytes, filename: str, session_id: str, preview_rows: int = 10
) -> dict:
    """
    1. Runs rule_based_clean() first (identical to the plain /clean report).
    2. DIAGNOSE: builds a widened defect-detection profile (every
       column, full distinct-value sets for low-cardinality columns —
       see profiler.build_defect_profile) and asks the LLM what's
       actually wrong, with evidence per defect.
    3. If defects were found (which may include meaningful nulls, but
       isn't limited to them): FIX — asks the LLM to write code
       resolving every diagnosed defect, runs it through the sandbox.
    4. Returns a report combining the rule-based log, the diagnosed
       defects (always, even if the fix step is skipped or fails —
       diagnosis is a standalone, inspectable result), the fix
       reasoning, and before/after stats.
    """
    try:
        df_raw = _read_csv(csv_bytes, filename)
    except Exception as e:
        return {"success": False, "error": f"Could not parse CSV: {e}"}

    try:
        from cleaning import rule_based_clean
        from profiler import (
            build_defect_profile,
            format_defect_profile_for_llm,
            build_defect_diagnosis_prompt,
            build_defect_fix_prompt,
        )
        from query_engine import llm_with_fallback
        from sandbox_limits import run_sandboxed

        df_rule_cleaned, rule_log = rule_based_clean(df_raw)

        before = {
            "rows": len(df_raw),
            "columns": len(df_raw.columns),
            "nullValues": int(df_raw.isnull().sum().sum()),
        }

        defect_profile = build_defect_profile(df_rule_cleaned)
        profile_str = format_defect_profile_for_llm(defect_profile)
        diagnosis_prompt = build_defect_diagnosis_prompt(profile_str)
        diagnosis_raw = llm_with_fallback(diagnosis_prompt, temperature=0.2, session_id=session_id)
        defects = _parse_defects(diagnosis_raw)

        if not defects:
            # Genuinely nothing found (or diagnosis parsing failed) —
            # either way, no fix call needed. Print the raw response so
            # a parsing failure is debuggable rather than indistinguishable
            # from "the dataset was actually clean".
            print(f"[cleaning-agent] 0 defects diagnosed. Raw response:\n{diagnosis_raw}", flush=True)
            return {
                "success": True,
                "filename": filename,
                "ruleLog": rule_log,
                "defects": [],
                "agentUsed": False,
                "reasoning": [],
                "before": before,
                "after": {
                    "rows": len(df_rule_cleaned),
                    "columns": len(df_rule_cleaned.columns),
                    "nullValues": int(df_rule_cleaned.isnull().sum().sum()),
                },
                "preview": _records(df_rule_cleaned.head(preview_rows)),
                "previewCount": min(preview_rows, len(df_rule_cleaned)),
            }

        schema_text = ", ".join(df_rule_cleaned.columns.tolist())
        fix_prompt = build_defect_fix_prompt(defects, schema_text)
        fix_raw = llm_with_fallback(fix_prompt, temperature=0.2, session_id=session_id)
        reasoning_lines, code = _parse_fix_response(fix_raw)

        if not code:
            return {
                "success": False,
                "error": "The agent diagnosed defects but did not return usable fix code — try again.",
                "ruleLog": rule_log,
                "defects": defects,
            }

        sbr = run_sandboxed(code, df_rule_cleaned, include_charts=False, timeout_seconds=20)

        if not sbr.success or not sbr.produced:
            return {
                "success": False,
                "error": f"Agent's fix code failed to execute: {sbr.error}",
                "ruleLog": rule_log,
                "defects": defects,
                "reasoning": reasoning_lines,
            }

        df_final = sbr.value
        if not isinstance(df_final, pd.DataFrame):
            return {
                "success": False,
                "error": "Agent's fix code did not return a table (DataFrame) — try again.",
                "ruleLog": rule_log,
                "defects": defects,
                "reasoning": reasoning_lines,
            }

        return {
            "success": True,
            "filename": filename,
            "ruleLog": rule_log,
            "defects": defects,
            "agentUsed": True,
            "reasoning": reasoning_lines,
            "before": before,
            "after": {
                "rows": len(df_final),
                "columns": len(df_final.columns),
                "nullValues": int(df_final.isnull().sum().sum()),
            },
            "preview": _records(df_final.head(preview_rows)),
            "previewCount": min(preview_rows, len(df_final)),
        }
    except Exception as e:
        return {"success": False, "error": f"Data quality agent failed: {e}"}
