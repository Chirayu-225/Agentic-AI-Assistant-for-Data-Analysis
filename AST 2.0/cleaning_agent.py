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


# What the USER sees when something goes wrong. Fixed strings on purpose: never
# put exception text, tracebacks or file paths in a response (they leak
# internals and mean nothing to the user). The technical detail goes to the
# server log through _log_detail instead.
_SAFE_READ_ERROR = ("This file could not be read as a table. Check that it is a valid "
                    "CSV or Excel file.")
_SAFE_FIX_ERROR = ("The AI's suggested fix could not be applied. Nothing in your data "
                   "was changed.")
_SAFE_AGENT_ERROR = ("The data quality check could not be completed. Nothing in your "
                     "data was changed.")


def _log_detail(what: str, detail) -> None:
    """Technical detail for the server log only (never returned to the user)."""
    print(f"[cleaning-agent] {what}: {detail}", flush=True)


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


_COMPACT_RETRY_NOTE = (
    "\n\nIMPORTANT — your previous answer was too long and was cut off before "
    "the code block closed. Write SHORT code (under 40 lines): tidy text "
    "with string methods (.str.strip(), .str.upper()) and add boolean flag "
    "columns, nothing else. Never rename or merge categories and never write "
    "one mapping entry per distinct value. If a defect can't be fixed compactly, "
    "leave that column unchanged and say so in REASONING (proposal only: ...)."
)


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


_NOTHING_APPLIED = {"applied": False, "reverted": [], "keptChanges": [], "newColumns": []}


def _with_flag_columns(message: str, new_columns: list) -> str:
    """Add "N flag columns added" to the verifier's one-line message."""
    n = len(new_columns)
    return f"{message}; {n} flag column{'s' if n != 1 else ''} added" if n else message


def _repair_note(code: str, error) -> str:
    """Appended to the fix prompt for the ONE repair attempt after the sandbox rejected the model's script. The error is
    scrubbed (no paths, no traceback) and capped; the rules are restated because a repaired script is checked by the
    same verifier as any other."""
    from error_scrub import scrub_error_text
    err = scrub_error_text(str(error or "unknown error"))[:300]
    return (
        "\n\nYour previous code was rejected by the sandbox with: " + err + "\n"
        "Rewrite the COMPLETE code block so that it runs. The sandbox already has df, pd and np: ALL import statements "
        "are blocked (including `import re`), so never write the word import. For regular expressions use pandas string "
        "methods with a plain string pattern, for example df['col'].astype(str).str.match('^[0-9]{5}$') (no re.compile). "
        "The same hard rules apply: never blank, fill, rename or drop existing values; only add flag columns or tidy "
        "whitespace and case.\n"
        "Previous code that failed:\n```python\n" + code[:3000] + "\n```"
    )


def _verified_table(df_before: pd.DataFrame, df_fix: pd.DataFrame):
    """Run the fix script's output through agent_verify, in the backend, on the table we already hold. Returns
    (table to use, extra response fields). Any structural failure, verifier fault or verifier crash keeps the table
    exactly as it was. Nothing from an exception reaches the response; the detail goes to the server log."""
    import agent_verify
    try:
        result = agent_verify.verify_fix(df_before, df_fix)
    except Exception:
        import traceback
        _log_detail("verifier crashed; the fix was discarded", traceback.format_exc())
        result = {"ok": False, "structural": False, "fault": True, "violations": [], "reverted_columns": [],
                  "kept_changes": [], "new_columns": [], "repaired": df_before}
    for v in result["violations"]:
        _log_detail("fix violation", v)
    if not result["ok"]:
        return df_before, {**_NOTHING_APPLIED, "reverted": [agent_verify.describe_violation(v) for v in result["violations"]
                                                            if v["kind"] != "structure"],
                           "message": agent_verify.build_message(result)}
    return result["repaired"], {
        "applied": bool(result["kept_changes"]),
        "reverted": [agent_verify.describe_violation(v) for v in result["violations"]],
        "keptChanges": result["kept_changes"],
        "newColumns": list(result["new_columns"]),
        "message": _with_flag_columns(agent_verify.build_message(result), result["new_columns"]),
    }


def _keep_for_questions(on_applied, df_used: pd.DataFrame, verdict: dict) -> dict:
    """The verified table becomes the working table for the questions that follow, but only if the verifier kept at
    least one change. `on_applied` is how the caller stores it (the API route saves it in the session cache); with no
    hook (library use, tests) nothing is stored and the response is unchanged. If storing fails the user is told the
    changes were NOT kept, never that they were."""
    if on_applied is None or not verdict.get("applied"):
        return verdict
    try:
        on_applied(df_used)
    except Exception:
        import traceback
        _log_detail("could not keep the agent's table for later questions", traceback.format_exc())
        return {**verdict, "usedForQuestions": False,
                "message": str(verdict.get("message", "")).replace("used for your questions", "not kept for your questions")}
    # The working table lives in an in-memory session cache: it expires, and a backend restart clears it. Say so, so a
    # later question that no longer uses it does not contradict the banner.
    return {**verdict, "usedForQuestions": True,
            "message": f"{verdict.get('message', '')}; kept for the next {_kept_minutes()} minutes"}


def _kept_minutes() -> int:
    import session_cache
    return max(1, int(session_cache.TTL_SECONDS // 60))


def _fill_message(existing, report: dict, kept) -> str:
    """The agent's one-line summary when it filled blanks (before the AI's own result, so the banner stays truthful)."""
    by: dict = {}
    for f in report["filled"]:
        by[f["method"]] = by.get(f["method"], 0) + f["count"]
    bits = []
    if by.get("inferred"):
        bits.append(f"{by['inferred']:,} from other rows that prove the value")
    if by.get("median"):
        bits.append(f"{by['median']:,} numbers with their column median")
    sentence = f"Filled {report['total_filled']:,} missing values ({', '.join(bits)})"
    if kept is True:
        sentence += f", used for your questions for the next {_kept_minutes()} minutes"
    elif kept is False:
        sentence += ", but not kept for your questions"
    rest = str(existing or "").replace("No data was changed", "No other data was changed")
    return f"{sentence}. {rest}".strip()


_REPAIR_PHRASES = {
    "unit_in_text": "read '{c}' as numbers in one unit",
    "percent_scale": "rescaled whole percentages in '{c}'",
    "magnitude_outlier": "restored values in '{c}' that were 100x too large",
    "impossible_negative": "blanked impossible negative values in '{c}'",
    "future_date": "blanked future dates in '{c}'",
    "end_before_start": "blanked end dates earlier than their start in '{c}'",
    "mixed_date_formats": "read dates in other formats in '{c}'",
    "swapped_values": "swapped back swapped values ({c})",
    "near_duplicate_rows": "dropped near-duplicate rows",
}


def _validators_message(v: dict, suffix: str = "") -> str:
    """One sentence for the banner: what the deterministic repairs did (value_validators.py), counts included."""
    bits = [f"{_REPAIR_PHRASES.get(e['rule'], e['rule']).format(c=e['column'])}: {e['count']:,}" for e in v["entries"]]
    n = v["changed"] + v["rowsDropped"]
    return f"Repaired {n:,} value{'s' if n != 1 else ''} ({'; '.join(bits)}){suffix}"


def run_cleaning_agent(
    csv_bytes: bytes, filename: str, session_id: str, preview_rows: int = 10, on_applied=None
) -> dict:
    """The agent. It FILLS blanks first (deterministically, from evidence in the table itself; see fill_missing.py), then
    lets the model audit the rest. The response gains a "fill" block: what was filled, and what could not be filled and
    needs the user's answer (fill with "Unknown" or leave empty). Wraps _run_cleaning_agent so every path reports it."""
    state: dict = {}
    result = _run_cleaning_agent(csv_bytes, filename, session_id, preview_rows, on_applied, state)
    report = state.get("report")
    if not isinstance(result, dict) or report is None:
        return result
    out = {**result, "fill": {"filled": report["filled"], "skipped": report["skipped"],
                              "totalFilled": report["total_filled"], "remainingText": report["remaining_text"],
                              "kept": state.get("kept")}}
    if report["total_filled"] and out.get("success"):
        out["message"] = _fill_message(out.get("message"), report, state.get("kept"))
    v = state.get("validators")
    if v and v["entries"]:
        out["validators"] = {"entries": v["entries"], "changed": v["changed"], "rowsDropped": v["rowsDropped"],
                             "kept": state.get("kept")}
        if out.get("success"):
            kept = state.get("kept")
            suffix = ""
            if not report["total_filled"]:      # the fill sentence already says whether the table is kept
                suffix = (f", used for your questions for the next {_kept_minutes()} minutes" if kept is True
                          else ", but not kept for your questions" if kept is False else "")
            out["message"] = f"{_validators_message(v, suffix)}. {out.get('message') or ''}".strip()
    return out


def _run_cleaning_agent(
    csv_bytes: bytes, filename: str, session_id: str, preview_rows: int = 10, on_applied=None, state=None
) -> dict:
    """
    0. FILL: blanks a column of the same table proves (or, for numbers, the column median) are filled first, by code,
       not by the model; the filled table is what everything below works on.
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
        _log_detail(f"could not read {filename!r}", repr(e))
        return {"success": False, "error": _SAFE_READ_ERROR}

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

        from cache_layer import cached_rule_clean
        df_rule_cleaned, rule_log = cached_rule_clean(df_raw, csv_bytes, rule_based_clean)

        before = {
            "rows": len(df_raw),
            "columns": len(df_raw.columns),
            "nullValues": int(df_raw.isnull().sum().sum()),
        }

        # FILL FIRST, in code, so it still happens when the model's part fails. A blank is filled only from evidence in the
        # table (fill_missing.py); text with no evidence stays blank and becomes a QUESTION for the user.
        from fill_missing import fill_missing
        df_rule_cleaned, fill_report = fill_missing(df_rule_cleaned)

        # THEN the deterministic value repairs (value_validators.py): impossible negatives, future dates, mixed date
        # formats, units in text, swapped values, near-duplicate rows... Each one needs evidence in the table, reports
        # what it changed, and its blanks come AFTER the fill, so they are never filled back in. No model is involved.
        validators_report = {"entries": [], "changed": 0, "rowsDropped": 0}
        try:
            from value_validators import validate_values
            df_rule_cleaned, validators_report = validate_values(df_rule_cleaned)
        except Exception:
            import traceback
            _log_detail("value validators failed; continuing without them", traceback.format_exc())
        # NUMBERS that other columns determine exactly (amount = quantity x unit_price) are computed AFTER the validators, so
        # a blanked impossible value is recovered and a corrupted factor is never used. Exact arithmetic, nothing guessed.
        try:
            from fill_missing import fill_derived
            df_rule_cleaned, derived_entries = fill_derived(df_rule_cleaned)
            if derived_entries:
                fill_report = dict(fill_report)
                fill_report["filled"] = list(fill_report["filled"]) + derived_entries
                fill_report["total_filled"] = int(fill_report["total_filled"]) + sum(d["count"] for d in derived_entries)
                got = {d["column"]: d["count"] for d in derived_entries}
                kept_q = []
                for q in fill_report.get("skipped", []):      # a question that arithmetic has answered is no longer asked
                    left = q["count"] - got.get(q["column"], 0) if q.get("kind") == "no_evidence" else q["count"]
                    if left > 0:
                        kept_q.append({**q, "count": left})
                fill_report["skipped"] = kept_q
        except Exception:
            import traceback
            _log_detail("derived fill failed; continuing without it", traceback.format_exc())
        if state is not None:
            state["report"] = fill_report
            state["validators"] = validators_report
            state["kept"] = None
            if (fill_report["total_filled"] or validators_report["changed"] or validators_report["rowsDropped"]) \
                    and on_applied is not None:
                try:
                    on_applied(df_rule_cleaned)
                    state["kept"] = True
                except Exception:
                    import traceback
                    _log_detail("could not keep the filled table for later questions", traceback.format_exc())
                    state["kept"] = False

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
                **_NOTHING_APPLIED,
                "message": "No data was changed. No defects were found.",
            }

        schema_text = ", ".join(df_rule_cleaned.columns.tolist())
        fix_prompt = build_defect_fix_prompt(defects, schema_text)
        # 4096 tokens, not the 2048 default: a fix for several defects can be
        # long, and when the model ran out of room mid-answer the code block
        # never closed, so nothing was usable (seen on an 80-country column).
        fix_raw = llm_with_fallback(fix_prompt, temperature=0.2, session_id=session_id, max_tokens=4096)
        reasoning_lines, code = _parse_fix_response(fix_raw)

        if not code:
            # One uncached retry with an explicit instruction to stay short —
            # the usual cause is an exhaustive value-by-value mapping that
            # doesn't fit. Log the raw answer so a real failure is debuggable.
            print(f"[cleaning-agent] fix step returned no complete code block; retrying compactly. "
                  f"Raw response ended with:\n{fix_raw[-600:]}", flush=True)
            from llm_provider import llm_uncached_code_call
            retry_raw = llm_uncached_code_call(
                fix_prompt + _COMPACT_RETRY_NOTE, temperature=0.1, max_tokens=4096)
            reasoning_lines, code = _parse_fix_response(retry_raw)

        if not code:
            return {
                "success": False,
                "error": "The agent diagnosed defects but did not return usable fix code — try again.",
                "ruleLog": rule_log,
                "defects": defects,
            }

        sbr = run_sandboxed(code, df_rule_cleaned, include_charts=False, timeout_seconds=20)

        if not sbr.success or not sbr.produced:
            # ONE repair attempt, the way the query path does it: show the model its own (scrubbed) error. The usual
            # cause is a blocked import (the sandbox allows none), which the model fixes once it is told.
            _log_detail("fix code failed to execute; one repair attempt", f"{sbr.error}\n--- fix code was:\n{code}")
            try:
                from llm_provider import llm_uncached_code_call
                repair_raw = llm_uncached_code_call(
                    fix_prompt + _repair_note(code, sbr.error), temperature=0.1, max_tokens=4096)
                repair_reasoning, repair_code = _parse_fix_response(repair_raw)
                if repair_code:
                    repaired = run_sandboxed(repair_code, df_rule_cleaned, include_charts=False, timeout_seconds=20)
                    if repaired.success and repaired.produced:
                        sbr, code, reasoning_lines = repaired, repair_code, (repair_reasoning or reasoning_lines)
            except Exception:
                import traceback
                _log_detail("the repair attempt itself failed", traceback.format_exc())

        if not sbr.success or not sbr.produced:
            _log_detail("fix code failed to execute", f"{sbr.error}\n--- fix code was:\n{code}")
            return {
                "success": False,
                "error": _SAFE_FIX_ERROR,
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

        # The model's table is NOT trusted. What it may keep is decided here, from the two tables themselves: values it
        # blanked, filled or changed are reverted, structural damage discards the fix. The result is the working table.
        df_used, verdict = _verified_table(df_rule_cleaned, df_final)

        response = {
            "success": True,
            "filename": filename,
            "ruleLog": rule_log,
            "defects": defects,
            "agentUsed": True,
            "reasoning": reasoning_lines,
            "before": before,
            "after": {
                "rows": len(df_used),
                "columns": len(df_used.columns),
                # Over the ORIGINAL columns only (the rule-cleaned table's), like the verifier's tripwire: the empty
                # cells of flag columns the agent ADDED are not missing data in the user's table.
                "nullValues": int(df_used.iloc[:, : len(df_rule_cleaned.columns)].isnull().sum().sum()),
            },
            "preview": _records(df_used.head(preview_rows)),
            "previewCount": min(preview_rows, len(df_used)),
            **verdict,
        }
        # Only now, with the response fully built, does the verified table become the working table: had building the
        # response failed, the user would have seen an error while a table was already stored.
        response.update(_keep_for_questions(on_applied, df_used, verdict))
        return response
    except Exception:
        import traceback
        _log_detail("agent failed", traceback.format_exc())
        return {"success": False, "error": _SAFE_AGENT_ERROR}
