"""Tests for experiments.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import paths


def test_layout_paths(tmp_path: Path) -> None:
    layout = paths.build_run_layout("20260101-000000-smoke-abc123", root=tmp_path)
    assert layout.root == tmp_path / "20260101-000000-smoke-abc123"
    layout.ensure()
    for d in (layout.data_audit, layout.splits, layout.experiments,
              layout.summaries, layout.root / "logs"):
        assert d.exists()


def test_experiment_dir_rejects_bad_names(tmp_path: Path) -> None:
    layout = paths.build_run_layout("smoke", root=tmp_path)
    with pytest.raises(ValueError):
        layout.experiment_dir("suite/bad", "sid", "arm", 42)
    with pytest.raises(ValueError):
        layout.experiment_dir("suite", "sid with space", "arm", 42)
    with pytest.raises(ValueError):
        layout.experiment_dir("suite", "sid", "arm", -1)


def test_experiment_dir_layout(tmp_path: Path) -> None:
    layout = paths.build_run_layout("run1", root=tmp_path)
    d = layout.experiment_dir("core", "sid1", "logreg_raw", 42)
    assert d == layout.experiments / "core" / "sid1" / "logreg_raw" / "seed_42"
    files = paths.ensure_experiment_dir(d)
    for k in ("console_log", "events_log", "epochs_csv",
              "predictions_csv", "checkpoint_best"):
        assert files[k].parent.exists()
    assert files["figures_dir"].exists()


def test_run_id_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        paths.build_run_layout("../escape", root=tmp_path)
    with pytest.raises(ValueError):
        paths.build_run_layout("with space", root=tmp_path)
