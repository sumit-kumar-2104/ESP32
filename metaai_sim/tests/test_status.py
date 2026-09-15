"""Tests for experiments.status.

Only exercises the atomic status writer and the reader — no filesystem
race testing (that would require multiprocessing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import status


def test_valid_status_written_and_read(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    payload = status.write_status(p, status="running")
    assert payload["status"] == "running"
    assert "updated_at" in payload
    got = status.read_status(p)
    assert got["status"] == "running"


def test_completed_flag(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    assert status.is_completed(p) is False
    status.write_status(p, status="completed", extra={"test_accuracy": 0.9})
    assert status.is_completed(p) is True
    got = status.read_status(p)
    assert got["test_accuracy"] == 0.9


def test_reason_required_for_failure(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    with pytest.raises(ValueError):
        status.write_status(p, status="failed")
    with pytest.raises(ValueError):
        status.write_status(p, status="unavailable", reason="")
    with pytest.raises(ValueError):
        status.write_status(p, status="alias_of")


def test_alias_of_is_a_valid_status(tmp_path: Path) -> None:
    """Regression: the runner writes ``alias_of`` when two configured
    splits resolve to identical (train, val, test) membership. It must
    be accepted here so the runner does not crash mid-suite."""
    p = tmp_path / "s.json"
    payload = status.write_status(
        p, status="alias_of",
        reason="alias_of=heldout_test_20181118: identical membership",
        extra={"canonical_split_id": "heldout_test_20181118"},
    )
    assert payload["status"] == "alias_of"
    got = status.read_status(p)
    assert got["status"] == "alias_of"
    assert got["canonical_split_id"] == "heldout_test_20181118"


def test_invalid_status(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        status.write_status(tmp_path / "s.json", status="done")


def test_corrupt_file_never_reads_completed(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text("this is not JSON", encoding="utf-8")
    assert status.read_status(p) is None
    assert status.is_completed(p) is False


def test_atomic_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    status.write_status(p, status="running")
    status.write_status(p, status="completed", extra={"a": 1})
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["status"] == "completed"
    assert got["a"] == 1
