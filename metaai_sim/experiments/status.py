"""Atomic status.json writer.

Each experiment has exactly one status.json. Statuses are constrained to
{pending, running, completed, failed, unavailable}. Every write is atomic
(tmp file + os.replace) so a crash mid-write never leaves a corrupted or
inconsistent file that could be mis-read as ``completed``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = ("pending", "running", "completed", "failed", "unavailable")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(
    path: str | os.PathLike,
    *,
    status: str,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write a status file. Returns the payload written.

    Args:
        path: target status.json.
        status: one of :data:`VALID_STATUSES`.
        reason: short human-readable string. Required when status is
            ``failed`` or ``unavailable`` so summaries can explain the outcome.
        extra: additional JSON-serialisable metadata (e.g. metric summary,
            duration, arm-specific counters). Never store secrets here.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status={status!r} not in {VALID_STATUSES}"
        )
    if status in ("failed", "unavailable") and not reason:
        raise ValueError(
            f"status={status!r} requires a non-empty reason string"
        )

    payload: dict[str, Any] = {
        "status": status,
        "updated_at": _now_iso(),
    }
    if reason:
        payload["reason"] = reason
    if extra:
        # Shallow copy so callers can keep mutating their dict.
        payload.update({k: v for k, v in extra.items() if k not in payload})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return payload


def read_status(path: str | os.PathLike) -> dict[str, Any] | None:
    """Return the parsed status.json, or None if it's missing/empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A half-written file must never be treated as completed.
        return None


def is_completed(path: str | os.PathLike) -> bool:
    s = read_status(path)
    return bool(s) and s.get("status") == "completed"
