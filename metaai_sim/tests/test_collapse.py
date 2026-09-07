"""Prediction-collapse indicator and per-class recall/precision."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import collapse


def test_collapsed_flag_fires_on_near_constant_predictions() -> None:
    # 998 predictions of class 0, 1 of class 1, 1 of class 4 — matches
    # the observed cross_room 2->1 / logreg_raw pattern from the task
    # description.
    y_pred = np.array([0] * 998 + [1, 4])
    ind = collapse.collapse_indicator(y_pred, n_classes=6)
    assert ind.collapsed is True
    assert ind.top_class_share > 0.95
    assert ind.n_classes_predicted == 3


def test_not_collapsed_on_balanced_predictions() -> None:
    y_pred = np.array(list(range(6)) * 20)
    ind = collapse.collapse_indicator(y_pred, n_classes=6)
    assert ind.collapsed is False
    assert ind.top_class_share == pytest.approx(1.0 / 6)
    assert ind.n_classes_predicted == 6
    assert ind.entropy_bits == pytest.approx(math.log2(6), rel=1e-5)


def test_collapse_flag_config_is_visible_in_output() -> None:
    y_pred = np.array([0] * 90 + [1, 2] * 5)
    ind = collapse.collapse_indicator(
        y_pred, n_classes=3, top_share_threshold=0.7,
        min_classes_predicted=3,
    )
    assert ind.top_share_threshold == pytest.approx(0.7)
    assert ind.min_classes_predicted_threshold == 3
    # At 0.7 threshold and only 2 predicted classes, this collapses.
    assert ind.collapsed is True


def test_per_class_precision_and_recall() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    p = collapse.per_class_precision(y_true, y_pred, classes=[0, 1, 2])
    # class 0: predicted twice (positions 0 and 5), correct once -> 0.5
    assert p[0] == pytest.approx(0.5)
    # class 1: predicted three times (1, 2, 3), correct twice -> 2/3
    assert p[1] == pytest.approx(2 / 3)
    # class 2: predicted once (position 4), correct once -> 1.0
    assert p[2] == pytest.approx(1.0)


def test_feature_shift_returns_summary_norms() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(50, 4))
    X_target = rng.normal(2, 1, size=(30, 4))
    stats = collapse.compute_feature_shift_stats(X_train, X_target)
    assert stats["dim"] == 4
    assert stats["n_train"] == 50
    assert stats["n_target"] == 30
    assert stats["mean_shift_l2"] > 0.5   # non-trivial shift


def test_confidence_margin_stats_empty_when_no_probs() -> None:
    assert collapse.confidence_margin_stats(None) == {}


def test_confidence_margin_stats_computes_top1_margin() -> None:
    probs = np.array([
        [0.9, 0.1, 0.0],
        [0.4, 0.35, 0.25],
    ])
    stats = collapse.confidence_margin_stats(probs)
    assert stats["n_samples"] == 2
    assert stats["top1_mean"] == pytest.approx((0.9 + 0.4) / 2)
    # margins are 0.8 and 0.05.
    assert stats["margin_mean"] == pytest.approx((0.8 + 0.05) / 2)
