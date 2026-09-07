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


def _write_feature_mismatch_profile(dst: Path) -> Path:
    """Reference an arm whose feature_mode does not match the suite.

    ``logreg_raw`` is registered with ``feature_mode='raw'``. Putting it
    inside a ``feature_mode: dfs_spec`` suite must NOT silently skip it —
    the runner must enrol an ``unavailable`` row for every (seed).
    """
    profile = {
        "name": "mismatch",
        "label": "test fixture — arm/suite feature_mode mismatch",
        "use_synthetic": False,
        "dates": [],
        "seeds": [42, 43],
        "quality_policy": "reject",
        "min_packets": 8,
        "read_packet_counts": False,
        "suites": [{
            "name": "mismatch_suite",
            "required": False,
            "kind": "dfs_debug",
            "feature_mode": "dfs_spec",
            "arms": ["logreg_raw"],   # raw != dfs_spec
            "splits": [{
                "kind": "indomain_grouped",
                "id": "mismatch_split",
                "date": "20181109",
                "val_frac": 0.2,
                "test_frac": 0.2,
                "seed": 42,
            }],
        }],
    }
    dst.write_text(yaml.safe_dump(profile, sort_keys=True), encoding="utf-8")
    return dst


def test_feature_mode_mismatch_enrols_unavailable(tmp_path, monkeypatch):
    """A configured-but-mismatched arm was silently skipped before the fix.

    We patch the manifest builder + feature builder to bypass the raw-CSI
    dataset requirement while keeping ``bundle.is_synthetic = False`` so
    the runner treats the mismatch as production code would.
    """
    from experiments import features as features_mod
    from experiments import runner as runner_mod

    records, bundle = features_mod.synthetic_fixture()
    # Overwrite the flag so the runner's mismatch guard is exercised.
    bundle_non_synth = features_mod.FeatureBundle(
        X=bundle.X, y=bundle.y, sample_ids=bundle.sample_ids,
        group_ids=bundle.group_ids, gestures_raw=bundle.gestures_raw,
        class_index_to_gesture=bundle.class_index_to_gesture,
        feature_mode="dfs_spec", dfs_bins="full", is_synthetic=False,
    )
    # Force the fixture date into the profile.
    fixture_date = records[0].date
    profile_path = tmp_path / "mismatch.yaml"
    profile_dict = yaml.safe_load(
        _write_feature_mismatch_profile(profile_path).read_text(encoding="utf-8"),
    )
    profile_dict["suites"][0]["splits"][0]["date"] = fixture_date
    profile_path.write_text(yaml.safe_dump(profile_dict, sort_keys=True),
                            encoding="utf-8")

    def _fake_manifest(profile, layout, log):
        from experiments import manifest as manifest_mod
        manifest_mod.write_manifest(
            layout.data_audit, records, rejections=[],
            feature_mode="dfs_spec", dfs_bins="full",
        )
        return records

    def _fake_bundle(profile, records_, feature_mode, dfs_bins, log):
        return bundle_non_synth

    monkeypatch.setattr(runner_mod, "_build_manifest", _fake_manifest)
    monkeypatch.setattr(runner_mod, "_build_feature_bundle", _fake_bundle)

    rc = runner_mod.run([
        "--profile", str(profile_path),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "mismatch-run",
    ])
    assert rc == 0, "optional-suite mismatch must not flip the exit code"

    root = tmp_path / "mismatch-run"
    manifest = _load_manifest(root / "run_manifest.json")
    # Two seeds x one split x one arm = 2 unavailable rows must be recorded.
    assert manifest["counts"]["unavailable"] == 2
    assert manifest["counts"]["completed"] == 0
    for seed in (42, 43):
        exp_status = (
            root / "experiments" / "mismatch_suite" / "mismatch_split"
            / "logreg_raw" / f"seed_{seed}" / "status.json"
        )
        assert exp_status.exists(), f"missing enrollment for seed={seed}"
        payload = status.read_status(exp_status)
        assert payload["status"] == "unavailable"
        assert "feature_mode" in payload["reason"]
