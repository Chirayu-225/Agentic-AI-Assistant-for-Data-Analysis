"""
core/executor.py - applies approved operations to versioned snapshots.

Guarantees
  * the original DataFrame is deep-copied on entry and never handed out by reference
  * every apply() builds a NEW frame first; state only changes if the op fully succeeds
  * any op that would drop more than 5% of rows raises ConfirmationRequired unless the
    caller passes confirm_large_drop=True (the UI asks the user an extra question)
  * undo() restores the previous snapshot; the audit log is append-only and keeps both
    the apply and the undo
  * memory is bounded: the original plus only the last few snapshots stay resident; older
    ones are dropped and rebuilt by replaying the applied operations if an undo reaches them
  * restricted operations (core/policy.py) are refused unless a person approved them
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from core import operations
from core.policy import PolicyError, require_approval
from core.operations import OperationError, ROW_DROP_CONFIRM_THRESHOLD
from core.preview import Preview, preview as _preview
from core.schemas import AuditEntry, Operation


class GuardrailError(Exception):
    """An operation was refused by a safety guardrail."""


class ConfirmationRequired(GuardrailError):
    """The op would drop more rows than the guardrail allows without an explicit extra confirmation."""

    def __init__(self, rows_dropped: int, pct: float, threshold: float):
        self.rows_dropped = rows_dropped
        self.pct = pct
        super().__init__(
            f"This would remove {rows_dropped} row(s) ({pct:.1f}% of the data, above the "
            f"{threshold * 100:.0f}% safety limit). Confirm explicitly to continue."
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Executor:
    def __init__(
        self,
        original: pd.DataFrame,
        *,
        row_drop_threshold: float = ROW_DROP_CONFIRM_THRESHOLD,
        clock: Callable[[], datetime] = _utc_now,
        strict: bool = True,
        keep_snapshots: int = 4,
        max_resident_bytes: int = 512 * 1024 * 1024,
    ):
        # Row labels become 0..n-1 (= CSV row order), so every row has a unique, stable id.
        self._snapshots: list[pd.DataFrame | None] = [original.reset_index(drop=True).copy(deep=True)]
        self._applied: list[tuple[Operation, AuditEntry]] = []
        self._log: list[AuditEntry] = []
        self._threshold = row_drop_threshold
        self._clock = clock
        self._strict = strict  # False when replaying a recipe the user already approved
        self._keep = max(1, keep_snapshots)
        self._max_bytes = max_resident_bytes

    # -- read access -----------------------------------------------------------
    @property
    def original(self) -> pd.DataFrame:
        """A copy of the untouched input (callers can never alter the real original)."""
        return self._snapshots[0].copy(deep=True)

    @property
    def current(self) -> pd.DataFrame:
        """The latest snapshot. Treat as read-only: operations never mutate it, and the
        executor only ever replaces it with a new frame."""
        return self._snapshots[-1]

    @property
    def version(self) -> int:
        return len(self._snapshots) - 1

    @property
    def can_undo(self) -> bool:
        return bool(self._applied)

    @property
    def audit_log(self) -> list[AuditEntry]:
        return list(self._log)

    def applied_operations(self) -> list[Operation]:
        """The operations currently in effect, in order (what recipe.json will contain)."""
        return [op for op, _ in self._applied]

    # -- actions ---------------------------------------------------------------
    def preview(self, op: Operation, *, n_samples: int = 5) -> Preview:
        return _preview(self.current, op, n_samples=n_samples, row_drop_threshold=self._threshold,
                        strict=self._strict)

    def apply(self, op: Operation, *, approved_by: str | None = None, confirm_large_drop: bool = False) -> AuditEntry:
        """Apply an approved operation. Raises OperationError (cannot run) or
        ConfirmationRequired (needs an extra confirmation); state is unchanged on any error.

        `approved_by` has no default on purpose: a restricted operation (core/policy.py) is refused unless it is
        "user", so a caller that forgets the argument fails closed. A harmless operation without an approver is
        applied and recorded as "unspecified"."""
        policy = require_approval(op, approved_by)  # PolicyError before anything is touched
        approver = approved_by if approved_by is not None else "unspecified"
        before = self.current
        new = operations.apply(before, op, strict=self._strict)  # validates against the current frame first
        affected = operations.rows_affected(before, new, op)
        dropped = operations.rows_dropped(before, new)
        if before.shape[0] and dropped / before.shape[0] > self._threshold and not confirm_large_drop:
            raise ConfirmationRequired(dropped, 100 * dropped / before.shape[0], self._threshold)

        note = None
        if affected == 0:
            note = "no rows changed"
        elif dropped and dropped / before.shape[0] > self._threshold:
            note = "large row drop explicitly confirmed"
        entry = self._entry("apply", op, len(before), len(new), affected, approver, note, policy)
        self._snapshots.append(new)
        self._evict()
        self._applied.append((op, entry))
        self._log.append(entry)
        return entry

    def undo(self, *, approved_by: str = "user") -> AuditEntry | None:
        """Revert the most recent applied operation. Returns None if there is nothing to undo."""
        if not self._applied:
            return None
        op, original_entry = self._applied.pop()
        rolled_back = self._snapshots.pop()
        if self._snapshots[-1] is None:  # evicted to save memory: rebuild by replaying from the original
            frame = self._snapshots[0]
            for earlier, _ in self._applied:
                frame = operations.apply(frame, earlier, strict=False)
            self._snapshots[-1] = frame
        entry = self._entry("undo", op, len(rolled_back), len(self.current),
                            original_entry.rows_affected, approved_by, f"reverts #{original_entry.seq}")
        self._log.append(entry)
        return entry

    def reject(self, op: Operation, *, approved_by: str = "user") -> AuditEntry:
        """Record that the user declined a proposal (no data change)."""
        n = len(self.current)
        entry = self._entry("reject", op, n, n, 0, approved_by, None)
        self._log.append(entry)
        return entry

    @property
    def resident_bytes(self) -> int:
        return int(sum(f.memory_usage(deep=True).sum() for f in self._snapshots if f is not None))

    def _evict(self) -> None:
        """Keep the original and the newest `keep_snapshots`; shed more if over the byte budget."""
        idx = [i for i in range(1, len(self._snapshots) - 1) if self._snapshots[i] is not None]
        for i in idx[:-(self._keep - 1)] if self._keep > 1 else idx:
            self._snapshots[i] = None
        idx = [i for i in range(1, len(self._snapshots) - 1) if self._snapshots[i] is not None]
        while idx and self.resident_bytes > self._max_bytes:
            self._snapshots[idx.pop(0)] = None

    # -- export helpers (step 5 builds the files on top of these) -------------------
    def audit_records(self) -> list[dict]:
        """Flat, JSON/CSV-friendly rows of the audit log."""
        rows = []
        for e in self._log:
            rows.append({
                "seq": e.seq, "timestamp": e.timestamp, "action": e.action,
                "op": e.operation.op, "column": e.operation.column,
                "params": json.dumps(e.operation.params, sort_keys=True),
                "rows_before": e.rows_before, "rows_after": e.rows_after,
                "rows_affected": e.rows_affected, "source": e.source,
                "approved_by": e.approved_by, "policy": e.policy, "confidence": e.operation.confidence,
                "rationale": e.operation.rationale, "note": e.note,
            })
        return rows

    # -- internals ---------------------------------------------------------------
    def _entry(self, action: str, op: Operation, before: int, after: int, affected: int,
               approved_by: str, note: str | None, policy: str | None = None) -> AuditEntry:
        return AuditEntry(
            seq=len(self._log) + 1,
            timestamp=self._clock().astimezone(timezone.utc).isoformat(timespec="seconds"),
            action=action,  # type: ignore[arg-type]
            operation=op,
            rows_before=before,
            rows_after=after,
            rows_affected=affected,
            source=op.source,
            approved_by=approved_by,
            note=note,
            policy=policy,
        )


__all__ = ["Executor", "GuardrailError", "ConfirmationRequired", "OperationError", "PolicyError"]
