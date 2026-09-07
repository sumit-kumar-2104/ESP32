"""End-to-end smoke test of the orchestrator on the synthetic fixture.

Also covers the resume path: rerunning the same run_id must skip already-
completed experiments and refuse to run if the resolved config diverges.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments import runner, status


PROFILE_PATH = str(Path(__file__).resolve().parent.parent
                   / "experiments" / "profiles" / "smoke.yaml")


def test_smoke_end_to_end(tmp_path: Path) -> None:
    run_id = "smoke-test-abc123"
    argv = [
        "--profile", PROFILE_PATH,
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", run_id,
    ]
    rc = runner.run(argv)
    assert rc == 0, "smoke profile must complete cleanly on the synthetic fixture"
    root = tmp_path / run_id
    assert (root / "run_manifest.json").exists()
    assert (root / "resolved_config.yaml").exists()
    assert (root / "environment.txt").exists()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["completed"] > 0
    assert manifest["counts"]["failed"] == 0
    assert status.is_completed(root / "status.json")
    # Every experiment must have its own status.json and a metrics/test.json.
    for exp_status in root.glob("experiments/**/status.json"):
        s = status.read_status(exp_status)
        assert s["status"] in ("completed", "unavailable", "failed")
        if s["status"] == "completed":
            assert (exp_status.parent / "metrics" / "test.json").exists()
            assert (exp_status.parent / "predictions" / "test_predictions.csv").exists()
    # Summaries.
    assert (root / "summaries" / "experiment_index.csv").exists()
    assert (root / "summaries" / "aggregate_metrics.csv").exists()
    assert (root / "summaries" / "findings.md").exists()


def test_resume_is_idempotent(tmp_path: Path) -> None:
    run_id = "resume-test-def456"
    argv = [
        "--profile", PROFILE_PATH,
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", run_id,
    ]
    assert runner.run(argv) == 0
    # Capture the completed statuses.
    before = {p: json.loads(p.read_text(encoding="utf-8"))["updated_at"]
              for p in (tmp_path / run_id).glob("experiments/**/status.json")}
    assert before, "expected some experiment status files"
    # Now resume — this should NOT re-run any completed experiment.
    resume_argv = [
        "--resume", run_id,
        "--device", "cpu",
        "--results-root", str(tmp_path),
    ]
    assert runner.run(resume_argv) == 0
    after = {p: json.loads(p.read_text(encoding="utf-8"))["updated_at"]
             for p in (tmp_path / run_id).glob("experiments/**/status.json")}
    assert set(before) == set(after)
    for p in before:
        assert before[p] == after[p], f"resume re-ran {p}"


def test_resume_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.run([
            "--resume", "nonexistent-run",
            "--results-root", str(tmp_path),
        ])


def test_mutual_exclusion_profile_resume() -> None:
    with pytest.raises(SystemExit):
        runner.run([
            "--profile", PROFILE_PATH,
            "--resume", "x",
        ])
    with pytest.raises(SystemExit):
        runner.run([])


def test_synthetic_label_in_resolved_config(tmp_path: Path) -> None:
    run_id = "label-test-1"
    argv = [
        "--profile", PROFILE_PATH,
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", run_id,
    ]
    assert runner.run(argv) == 0
    resolved = (tmp_path / run_id / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "use_synthetic: true" in resolved
    assert "NOT scientific results" in resolved
