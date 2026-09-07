"""Cover the majority-class baseline and cyclic-permutation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from experiments import metrics as M


def test_majority_baseline_uniform() -> None:
    y = np.array([0, 1, 2, 3, 4] * 4)
    b = M.majority_class_baseline(y)
    assert b["n"] == 20
    # Every class has support 4/20 == 0.2.
    assert b["accuracy"] == pytest.approx(0.2)
    assert set(map(int, b["class_support"].keys())) == {0, 1, 2, 3, 4}
    assert b["majority_class"] in {0, 1, 2, 3, 4}


def test_majority_baseline_imbalanced() -> None:
    # 6 zeros, 2 ones, 2 twos, 5 threes, 5 fours — majority class is 0 with
    # support 6/20 == 0.30.
    y = np.array([0] * 6 + [1] * 2 + [2] * 2 + [3] * 5 + [4] * 5)
    b = M.majority_class_baseline(y)
    assert b["majority_class"] == 0
    assert b["accuracy"] == pytest.approx(6 / 20)
    assert b["class_support"]["0"] == 6


def test_majority_baseline_empty() -> None:
    b = M.majority_class_baseline(np.array([], dtype=int))
    assert b["majority_class"] is None
    assert b["n"] == 0
    assert np.isnan(b["accuracy"])


def test_cyclic_permutation_identity_matches_accuracy() -> None:
    y = np.array([0, 1, 2, 3, 4])
    p = np.array([0, 1, 2, 3, 4])
    cyc = M.cyclic_permutation_accuracies(y, p, n_classes=5)
    assert cyc[0] == pytest.approx(1.0)
    # Any non-zero shift on a perfectly-predicted set gives 0 accuracy.
    for s in range(1, 5):
        assert cyc[s] == pytest.approx(0.0)


def test_cyclic_permutation_detects_rotation() -> None:
    y = np.array([0, 1, 2, 3, 4] * 4)
    # Predicted labels are systematically rotated by +2.
    p = (y + 2) % 5
    cyc = M.cyclic_permutation_accuracies(y, p, n_classes=5)
    assert cyc[0] == pytest.approx(0.0)
    # (y_pred + shift) % 5 == y_true when shift == 3 recovers the labels.
    assert cyc[3] == pytest.approx(1.0)


def test_cyclic_permutation_infers_n_classes() -> None:
    y = np.array([0, 1, 2, 0, 1])
    p = np.array([0, 1, 2, 0, 1])
    cyc = M.cyclic_permutation_accuracies(y, p)
    assert set(cyc) == {0, 1, 2}
    assert cyc[0] == pytest.approx(1.0)


def test_label_shuffle_null_bounded() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=200)
    p = rng.integers(0, 5, size=200)
    null = M.label_shuffle_baseline(y, p, n_permutations=500, seed=0)
    assert null["n_permutations"] == 500
    # Mean of shuffled accuracy should be near sum_c p_c^2, which for
    # near-uniform 5-class data is ~0.2. Allow generous slack.
    assert 0.10 < null["mean"] < 0.30
    assert 0.0 <= null["p_value_ge_observed"] <= 1.0


def test_cyclic_perm_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        M.cyclic_permutation_accuracies(
            np.array([0, 1]), np.array([0, 1, 2]),
        )
