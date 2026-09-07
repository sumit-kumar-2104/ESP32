"""Tests for experiments.metrics."""

from __future__ import annotations

import numpy as np

from experiments import metrics as M


def test_accuracy_basic() -> None:
    y = np.array([0, 1, 2, 1])
    p = np.array([0, 1, 2, 2])
    assert M.accuracy(y, p) == 0.75


def test_balanced_accuracy_uniform_classes() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 0, 1])
    assert M.balanced_accuracy(y, p) == 0.5


def test_macro_f1_perfect() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    assert M.macro_f1(y, y) == 1.0


def test_summarise_preserves_gesture_map() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 1, 1])
    s = M.summarise_predictions(
        y, p, class_index_to_gesture={0: 1, 1: 3},
    )
    assert s["accuracy"] == 0.75
    assert s["class_index_to_gesture"] == {"0": 1, "1": 3}
    assert s["n_samples"] == 4


def test_predictions_csv_shape(tmp_path) -> None:
    sids = ["a", "b", "c"]
    y = np.array([0, 1, 0])
    p = np.array([0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    path = tmp_path / "preds.csv"
    M.write_predictions_csv(path, sids, y, p, probs=probs)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("sample_id,y_true,y_pred,p_0,p_1")
    assert len(lines) == 1 + len(sids)
