"""Cover the unavailable-row enrollment fix.

When a split spec fails validation (unknown date, InvalidSplitError from
leakage checks, unknown kind), the runner must:

1. write ``status.json`` = unavailable for every (arm, seed) that would
   have used that split, with a reason string;
2. count those rows in the aggregate counts dict; and
3. for required suites, flip the top-level status to failed and exit 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments import runner, status


def _write_broken_profile(dst: Path, *, required: bool) -> Path:
    """Write a smoke-like profile whose split points at a nonexistent date.

    That guarantees ``splits.indomain_grouped`` raises InvalidSplitError,
    which is exactly the state the fix must handle.
    """
    profile = {
        "name": "broken",
        "label": "test fixture — deliberately unresolvable split",
        "use_synthetic": True,
        "dates": [],
        "seeds": [42],
        "quality_policy": "reject",
        "min_packets": 8,
        "read_packet_counts": False,
        "suites": [{
            "name": "broken_suite",
            "required": required,
            "kind": "raw_amplitude",
            "feature_mode": "SYNTHETIC_FIXTURE",
            "arms": ["logreg_raw"],
            "splits": [{
                "kind": "indomain_grouped",
                "id": "bogus_date_split",
                "date": "99999999",     # not in the synthetic fixture
                "val_frac": 0.2,
                "test_frac": 0.2,
                "seed": 42,
            }],
        }],
    }
    dst.write_text(yaml.safe_dump(profile, sort_keys=True), encoding="utf-8")
    return dst


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_required_suite_split_error_marks_unavailable_and_fails(tmp_path):
    profile_path = _write_broken_profile(tmp_path / "broken.yaml", required=True)
    rc = runner.run([
        "--profile", str(profile_path),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "broken-required",
    ])
    assert rc == 1, "required-suite split failure must return exit code 1"

    root = tmp_path / "broken-required"
    manifest = _load_manifest(root / "run_manifest.json")
    # The one enrolled experiment must appear as unavailable.
    assert manifest["counts"]["unavailable"] == 1
    assert manifest["counts"]["completed"] == 0
    # Top-level status flipped to failed with a clear reason.
    top = status.read_status(root / "status.json")
    assert top["status"] == "failed"
    assert "required-suite" in top["reason"]
    # Per-experiment status.json exists at the exact path the runner would
    # have written to for a successful arm.
    exp_status = (
        root / "experiments" / "broken_suite" / "bogus_date_split"
        / "logreg_raw" / "seed_42" / "status.json"
    )
    assert exp_status.exists()
    s = status.read_status(exp_status)
    assert s["status"] == "unavailable"
    assert "split unavailable" in s["reason"]


def test_optional_suite_split_error_does_not_flip_exit_code(tmp_path):
    profile_path = _write_broken_profile(tmp_path / "broken.yaml", required=False)
    rc = runner.run([
        "--profile", str(profile_path),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "broken-optional",
    ])
    assert rc == 0, "optional-suite split failure must NOT flip the exit code"
    root = tmp_path / "broken-optional"
    manifest = _load_manifest(root / "run_manifest.json")
    assert manifest["counts"]["unavailable"] == 1
    top = status.read_status(root / "status.json")
    assert top["status"] == "completed"


def test_summary_index_records_reason(tmp_path):
    profile_path = _write_broken_profile(tmp_path / "broken.yaml", required=True)
    runner.run([
        "--profile", str(profile_path),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "broken-index",
    ])
    idx_csv = (tmp_path / "broken-index" / "summaries" / "experiment_index.csv"
               ).read_text(encoding="utf-8")
    assert "unavailable" in idx_csv
    assert "split unavailable" in idx_csv
