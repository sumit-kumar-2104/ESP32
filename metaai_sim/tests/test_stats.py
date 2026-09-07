"""Multiple-comparison correction, max-statistic null, at-chance guardrail."""

from __future__ import annotations

import numpy as np
import pytest

from experiments import stats


def test_benjamini_hochberg_monotonicity() -> None:
    p = [0.01, 0.02, 0.04, 0.20, 0.50]
    adj = stats.benjamini_hochberg(p)
    # Non-decreasing when sorted by the original p-values.
    order = np.argsort(p)
    sorted_adj = [adj[i] for i in order]
    for i in range(1, len(sorted_adj)):
        assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-9


def test_holm_bonferroni_is_bonferroni_at_smallest() -> None:
    p = [0.001, 0.01, 0.03, 0.5]
    adj = stats.holm_bonferroni(p)
    # Smallest raw p gets multiplied by m=4.
    assert adj[0] == pytest.approx(0.001 * 4)


def test_at_chance_guardrail() -> None:
    # 6-class chance = 1/6 ~= 0.1667. An observed 0.17 is at-chance under
    # the default 2pp tolerance and must be flagged as such.
    assert stats.at_chance(0.17, chance_level=1 / 6) is True
    assert stats.at_chance(0.30, chance_level=1 / 6) is False


def test_format_finding_flags_selection_bias_and_at_chance() -> None:
    text = stats.format_finding(
        "cross_room 2->1 / logreg_raw / seed=42",
        observed_accuracy=0.167,
        chance_level=1 / 6,
        baseline_accuracy=0.167,
        p_value=0.9,
        is_selection_biased=True,
    )
    assert "[EXPLORATORY — selection-biased]" in text
    assert "Never reported as a significant finding" in text


def test_format_finding_reports_effect_sizes_in_pp() -> None:
    text = stats.format_finding(
        "cross_room 1->2 / digital_mlp_raw",
        observed_accuracy=0.30,
        chance_level=1 / 6,
        baseline_accuracy=0.20,
        p_value=0.01,
        is_selection_biased=False,
    )
    # 30 - 16.67 = 13.33 pp vs chance; 30 - 20 = 10 pp vs majority.
    assert "delta_vs_chance=+13.33pp" in text
    assert "delta_vs_majority=+10.00pp" in text


def test_shuffle_null_produces_bounded_p_value() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    null = stats.shuffle_null(y_true, y_pred, n_permutations=200, seed=0)
    assert 0.0 <= null.p_value_ge_observed <= 1.0
    assert null.n_permutations == 200


def test_max_statistic_null_reports_observed_max() -> None:
    rng = np.random.default_rng(0)
    rows = [
        (rng.integers(0, 5, size=50), rng.integers(0, 5, size=50))
        for _ in range(3)
    ]
    result = stats.max_statistic_null(rows, n_permutations=100, seed=0)
    assert 0.0 <= result["observed_max"] <= 1.0
    assert 0.0 <= result["p_value_ge_observed"] <= 1.0
    assert result["n_permutations"] == 100


def test_primary_comparisons_are_pre_specified() -> None:
    """The primary-comparison registry must exist and cover the three
    core cross-domain kinds, so nobody can retroactively swap the target."""
    for kind in ("cross_room", "held_out_date", "leave_one_room_out"):
        assert kind in stats.PRIMARY_COMPARISONS
        assert "arm" in stats.PRIMARY_COMPARISONS[kind]
        assert "seed" in stats.PRIMARY_COMPARISONS[kind]
