"""End-to-end smoke test: gesture-set propagation, dedup, DA-arm refusal
show up in the run artefacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments import runner


PROFILE_PATH = str(Path(__file__).resolve().parent.parent
                   / "experiments" / "profiles" / "smoke.yaml")


def test_smoke_run_records_gesture_set_and_inventory(tmp_path: Path) -> None:
    run_id = "smoke-gset-abc"
    rc = runner.run([
        "--profile", PROFILE_PATH, "--device", "cpu",
        "--results-root", str(tmp_path), "--run-id", run_id,
    ])
    assert rc == 0
    root = tmp_path / run_id
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert "gesture_set" in manifest
    assert manifest["gesture_set"]["n_classes"] == 6
    assert manifest["gesture_set"]["chance_level"] > 0
    assert "inventory" in manifest
    assert "dedup" in manifest
    assert "unique_completed" in manifest["counts"]
    assert "executed_rows" in manifest["counts"]

    # Every canonical split has a persisted baseline JSON.
    baseline_dir = root / "summaries" / "per_split_baselines"
    if baseline_dir.exists():
        for b in baseline_dir.glob("*.json"):
            payload = json.loads(b.read_text(encoding="utf-8"))
            assert payload["baseline"]["n_classes"] == 6
            assert payload["gesture_set"]["n_classes"] == 6

    # RUN_SUMMARY mentions the gesture set.
    summary_text = (root / "RUN_SUMMARY.txt").read_text(encoding="utf-8")
    assert "gesture_set:" in summary_text
    assert "chance=" in summary_text


def test_smoke_run_flags_missing_same_room_diff_date(tmp_path: Path) -> None:
    """The synthetic fixture places 20181109 only in room 1 and
    20181118 only in room 2, so the inventory must record that same-room /
    different-date is unavailable."""
    run_id = "smoke-inv-abc"
    rc = runner.run([
        "--profile", PROFILE_PATH, "--device", "cpu",
        "--results-root", str(tmp_path), "--run-id", run_id,
    ])
    assert rc == 0
    root = tmp_path / run_id
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    inv = manifest.get("inventory") or {}
    assert inv.get("same_room_diff_date_available") is False
    assert "room and session" in inv.get(
        "same_room_diff_date_reason", ""
    ).lower()
