"""
core/schemas.py - the data contracts of the whole app (Pydantic v2).

Issue      what a detector found (deterministic evidence, never LLM output)
Operation  one step from the CLOSED operation library. Params are validated per
           op the moment an Operation is constructed, so an LLM (or a user typo)
           can never smuggle an unknown op or parameter past this layer.
Proposal   an Issue plus the Operation proposed for it and its review status
AuditEntry one append-only line of the audit log
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OpName = Literal[
    "drop_duplicates",
    "impute",
    "cast_type",
    "parse_dates",
    "strip_whitespace",
    "normalize_case",
    "clip_outliers",
    "map_values",
    "drop_column",
    "drop_rows_where_null",
]
OP_NAMES: tuple[str, ...] = OpName.__args__  # type: ignore[attr-defined]

IssueType = Literal[
    "nulls",
    "exact_duplicates",
    "mixed_types",
    "date_like_strings",
    "unparseable_dates",
    "whitespace",
    "case_inconsistency",
    "outliers",
    "constant_column",
    "near_duplicate_categories",
]
Confidence = Literal["low", "medium", "high"]
Source = Literal["rule", "llm", "user"]
ProposalStatus = Literal["pending", "applied", "rejected", "skipped", "failed"]

MAX_RATIONALE_CHARS = 300


class Issue(BaseModel):
    """A problem found by a deterministic detector."""

    model_config = ConfigDict(extra="forbid")

    id: str  # stable: "<type>:<column>" (or just "<type>" for dataset-level issues)
    type: IssueType
    column: str | None = None
    affected_rows: int = Field(ge=0)
    affected_pct: float = Field(ge=0)  # 0-100
    evidence: dict[str, Any] = Field(default_factory=dict)
    samples: list[str] = Field(default_factory=list, max_length=8)


class Operation(BaseModel):
    """One validated step from the closed operation library."""

    model_config = ConfigDict(extra="forbid")

    op: OpName
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: Confidence = "medium"
    source: Source = "rule"

    @field_validator("rationale")
    @classmethod
    def _one_sentence(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v or "").strip()
        return v[:MAX_RATIONALE_CHARS]

    @model_validator(mode="after")
    def _validate_params(self) -> "Operation":
        # Lazy import: operations.py imports this module for OpName.
        from core import operations

        self.params = operations.validate_params(self.op, self.column, self.params)
        return self

    # -- convenience -------------------------------------------------------
    def key(self) -> str:
        """Canonical identity of what this operation DOES (ignores rationale/confidence/source)."""
        return json.dumps(
            {"op": self.op, "column": self.column, "params": self.params},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def describe(self) -> str:
        target = f" on '{self.column}'" if self.column else ""
        return f"{self.op}{target}"


class Proposal(BaseModel):
    """What the UI shows for one issue: the proposed op plus one-click alternatives."""

    model_config = ConfigDict(extra="forbid")

    id: str  # same as issue.id
    issue: Issue
    operation: Operation | None = None  # None -> nothing safe to propose; user decides
    alternatives: dict[str, Operation] = Field(default_factory=dict)  # button label -> op
    status: ProposalStatus = "pending"


class AuditEntry(BaseModel):
    """One append-only line of the audit log."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    timestamp: str  # ISO-8601, UTC
    action: Literal["apply", "undo", "reject"]
    operation: Operation
    rows_before: int
    rows_after: int
    rows_affected: int
    source: Source
    approved_by: str
    policy: str | None = None  # set for restricted ops (see core/policy.py)
    note: str | None = None
