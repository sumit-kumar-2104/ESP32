"""Cover the human-readable RUN_SUMMARY.txt writer."""

from __future__ import annotations

from pathlib import Path

from experiments import report as report_mod
from experiments import runner


PROFILE_PATH = str(
    Path(__file__).resolve().parent.parent
    / "experiments" / "profiles" / "smoke.yaml"
)


def _run_smoke(tmp_path: Path, run_id: str) -> Path:
    rc = runner.run([
        "--profile", PROFILE_PATH,
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", run_id,
    ])
    assert rc == 0
    return tmp_path / run_id


def test_run_summary_written_alongside_folders(tmp_path: Path) -> None:
    root = _run_smoke(tmp_path, "smoke-report")
    summary = root / "RUN_SUMMARY.txt"
    assert summary.exists(), "RUN_SUMMARY.txt must be at the run root"
    txt = summary.read_text(encoding="utf-8")
    assert "RUN SUMMARY" in txt
    assert "OVERALL STATUS: completed" in txt
    # Section headers present.
    for section in (
        "DATA AUDIT", "SPLITS", "PER-EXPERIMENT RESULTS",
        "AGGREGATE", "INTERPRETATION NOTES",
    ):
        assert section in txt, f"missing section: {section}"
    # Existing folder tree is untouched.
    assert (root / "run_manifest.json").exists()
    assert (root / "summaries" / "experiment_index.csv").exists()
    assert (root / "summaries" / "aggregate_metrics.csv").exists()


def test_run_summary_contains_completed_rows(tmp_path: Path) -> None:
    root = _run_smoke(tmp_path, "smoke-report-rows")
    txt = (root / "RUN_SUMMARY.txt").read_text(encoding="utf-8")
    # Every smoke experiment should show up as a completed row.
    assert txt.count(" completed ") >= 4
    assert "logreg_raw" in txt
    assert "smoke_indomain" in txt


def test_write_run_report_idempotent(tmp_path: Path) -> None:
    root = _run_smoke(tmp_path, "smoke-report-idem")
    from experiments import paths as paths_mod
    layout = paths_mod.build_run_layout("smoke-report-idem", root=tmp_path)
    p1 = report_mod.write_run_report(layout)
    p2 = report_mod.write_run_report(layout)
    assert p1 == p2
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_run_summary_lists_reasons_for_unavailable(tmp_path: Path) -> None:
    import yaml
    from experiments import status as status_mod
    profile = {
        "name": "broken",
        "label": "test fixture — unresolvable split",
        "use_synthetic": True,
        "dates": [],
        "seeds": [42],
        "quality_policy": "reject",
        "min_packets": 8,
        "read_packet_counts": False,
        "suites": [{
            "name": "broken_suite",
            "required": False,
            "kind": "raw_amplitude",
            "feature_mode": "SYNTHETIC_FIXTURE",
            "arms": ["logreg_raw"],
            "splits": [{
                "kind": "indomain_grouped",
                "id": "no_such_date",
                "date": "99999999",
                "seed": 42,
            }],
        }],
    }
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(profile), encoding="utf-8")
    rc = runner.run([
        "--profile", str(p),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "broken-report",
    ])
    assert rc == 0  # optional suite, so exit is still 0
    summary = (tmp_path / "broken-report" / "RUN_SUMMARY.txt").read_text(encoding="utf-8")
    assert "UNAVAILABLE / FAILED" in summary
    assert "unavailable" in summary
    assert "split unavailable" in summary
