"""End-to-end smoke test: the runner successfully executes a cnn_gru
arm on a synthetic tensor bundle. Guards against silent shape/wiring
regressions when the tensor + constraint code changes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments import runner as runner_mod


def _write_repro_smoke_profile(path: Path) -> Path:
    path.write_text(
        """
name: repro_smoke_tensor
label: "synthetic tensor smoke for cnn_gru"
use_synthetic: true
dates: []
seeds: [42]
quality_policy: reject
min_packets: 8
read_packet_counts: false

gesture_set:
  ids: [1, 2, 3, 4, 5, 6]
  label: "six-class-default"
  justification: "smoke; matches core"
  evidence: "docs/gesture_set_evidence.md"

tasks_1_3_ready: false

suites:
  - name: repro_dfs_tensor_smoke
    required: true
    kind: reproduction
    feature_mode: dfs_tensor
    tensor_T: 8
    tensor_F: 4
    reference_accuracy: 77.8
    arms:
      - cnn_gru_dfs_tensor
    splits:
      - kind: indomain_grouped
        id: smoke_indomain
        date: "20181109"
        val_frac: 0.2
        test_frac: 0.2
        seed: 42
""",
        encoding="utf-8",
    )
    return path


def test_cnn_gru_runs_end_to_end_on_synthetic_tensor(tmp_path: Path) -> None:
    profile_path = _write_repro_smoke_profile(tmp_path / "repro_smoke.yaml")
    rc = runner_mod.run([
        "--profile", str(profile_path),
        "--device", "cpu",
        "--results-root", str(tmp_path),
        "--run-id", "repro_tensor_smoke",
    ])
    assert rc == 0, f"repro smoke exit code {rc}"
    root = tmp_path / "repro_tensor_smoke"
    exp_root = (
        root / "experiments" / "repro_dfs_tensor_smoke" / "smoke_indomain"
        / "cnn_gru_dfs_tensor" / "seed_42"
    )
    assert (exp_root / "status.json").exists()
    status = json.loads((exp_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed", status
    # Predictions were persisted from the actual test forward pass.
    preds = exp_root / "predictions" / "test_predictions.csv"
    assert preds.exists()
    # Reproduction report gets written and picks up the completion.
    repro_report = root / "summaries" / "reproduction_report.md"
    assert repro_report.exists()
    assert "dfs_tensor" in repro_report.read_text(encoding="utf-8")
