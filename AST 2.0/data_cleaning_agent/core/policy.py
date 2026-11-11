"""
core/policy.py - which operations need a human, and what gets recorded.

AST's *agent verifier* forbids imputing, dropping columns and clipping outliers, because there
nothing a person reviews stands between the model and the data. This package deliberately
allows them, but ONLY under this policy:

  * every operation below is "restricted": the executor refuses it unless approved_by == "user"
    (a click or typed instruction by a person; "auto"/"rule"/"llm" and a MISSING approver are rejected: there is
    no default approver, so a caller that forgets to say who approved cannot slip a restricted operation through);
  * the audit log records source (rule | llm | user), approver, and the policy tag, so a reader
    can always tell who proposed it and who agreed;
  * nothing here feeds AST's verifier or its LLM-generated-code path (separate package).
"""
from __future__ import annotations

from core.schemas import Operation

HUMAN_APPROVERS = frozenset({"user"})


class PolicyError(PermissionError):
    """A restricted operation was applied without human approval."""


def restriction(op: Operation) -> str | None:
    """A short tag when `op` is restricted, else None."""
    if op.op == "impute":
        return "impute-constant" if op.params.get("strategy") == "constant" else "impute"
    if op.op == "drop_column":
        return "drop-column"
    if op.op == "clip_outliers":
        return "clip-outliers"
    if op.op in ("drop_duplicates", "drop_rows_where_null"):
        return "drops-rows"
    if op.op == "map_values":
        # EVERY value mapping needs a person, whoever proposed it: a fuzzy-matched rule merge can join categories that are
        # genuinely different. Model-proposed ones must also map onto values already in the column
        # (core/operations.validate_against); rule-proposed ones are built from the column's own values (core/rules.py).
        return "llm-value-mapping" if op.source == "llm" else "value-mapping"
    return None


def require_approval(op: Operation, approved_by: str | None) -> str | None:
    """Raise PolicyError if `op` is restricted and `approved_by` is not a person (None = nobody said)."""
    tag = restriction(op)
    if tag and approved_by not in HUMAN_APPROVERS:
        who = "no approver was given" if approved_by is None else f"got '{approved_by}'"
        raise PolicyError(f"'{op.describe()}' ({tag}) needs approval from a person; {who}.")
    return tag
