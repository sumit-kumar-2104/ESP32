"""Regression: ``ota_ablation`` refuses to execute when the paired
reproduction gate has not completed a cnn_gru run for the same
feature_mode. The gate reads directly from persisted status.json files
so it cannot be fooled by an in-memory shortcut."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import paths as paths_mod
from experiments import runner as runner_mod
from experiments import status as status_mod


def _make_layout(tmp_path: Path) -> paths_mod.RunLayout:
    layout = paths_mod.build_run_layout("gate_test", tmp_path)
    layout.ensure()
    return layout


def _write_status(exp_dir: Path, status: str, reason: str = "test-fixture") -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    if status in ("failed", "unavailable"):
        status_mod.write_status(exp_dir / "status.json", status=status, reason=reason)
    else:
        status_mod.write_status(exp_dir / "status.json", status=status)


def test_gate_refuses_when_no_repro_completed(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    # Nothing has been recorded yet.
    n = runner_mod._count_repro_completions(
        layout,
        required_suite="repro_csi_tensor",
        feature_mode="csi_tensor",
    )
    assert n == 0


def test_gate_ignores_failed_and_wrong_mode(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    # A failed cnn_gru row for the right mode: does not count.
    exp_failed = layout.experiment_dir(
        "repro_csi_tensor", "indomain_20181109",
        "cnn_gru_csi_tensor", 42,
    )
    _write_status(exp_failed, "failed")
    # A completed cnn_gru row for the WRONG mode: does not count.
    exp_wrong = layout.experiment_dir(
        "repro_csi_tensor", "indomain_20181109",
        "cnn_gru_dfs_tensor", 42,
    )
    _write_status(exp_wrong, "completed")
    assert runner_mod._count_repro_completions(
        layout, required_suite="repro_csi_tensor",
        feature_mode="csi_tensor",
    ) == 0


def test_gate_passes_on_completed_cnn_gru(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path)
    exp_ok = layout.experiment_dir(
        "repro_csi_tensor", "indomain_20181109",
        "cnn_gru_csi_tensor", 42,
    )
    _write_status(exp_ok, "completed")
    assert runner_mod._count_repro_completions(
        layout, required_suite="repro_csi_tensor",
        feature_mode="csi_tensor",
    ) == 1


def test_gate_ignores_non_cnn_gru_arms(tmp_path: Path) -> None:
    """A digital MLP completion under the same suite must NOT satisfy the
    reproduction gate — the gate exists to ensure the CNN-GRU backbone
    reproduced the paper's number on the tensor mode."""
    layout = _make_layout(tmp_path)
    exp = layout.experiment_dir(
        "repro_csi_tensor", "indomain_20181109",
        "digital_mlp_raw", 42,
    )
    _write_status(exp, "completed")
    assert runner_mod._count_repro_completions(
        layout, required_suite="repro_csi_tensor",
        feature_mode="csi_tensor",
    ) == 0
