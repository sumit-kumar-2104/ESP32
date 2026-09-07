"""GestureSet: explicit configuration, per-variant chance + majority,
propagation into manifests."""

from __future__ import annotations

import numpy as np
import pytest

from experiments import features, gesture_set as gs


def test_default_six_class_baseline() -> None:
    g = gs.GestureSet.from_config({})
    assert g.ids == (1, 2, 3, 4, 5, 6)
    assert g.n_classes == 6
    assert g.chance_level == pytest.approx(1.0 / 6)


def test_five_class_variant_has_different_chance() -> None:
    g6 = gs.GestureSet.from_config({"ids": [1, 2, 3, 4, 5, 6]})
    g5 = gs.GestureSet.from_config({"ids": [1, 2, 3, 5, 6]})
    assert g6.chance_level != g5.chance_level
    assert g5.chance_level == pytest.approx(1.0 / 5)


def test_invalid_ids_rejected() -> None:
    with pytest.raises(ValueError):
        gs.GestureSet.from_config({"ids": [7, 8, 9]})
    with pytest.raises(ValueError):
        gs.GestureSet.from_config({"ids": []})


def test_filter_records_drops_out_of_set(records: list | None = None) -> None:
    recs, _ = features.synthetic_fixture(n_recordings=30, n_gestures=3)
    g = gs.GestureSet.from_config({"ids": [1, 2]})
    kept = gs.filter_records(recs, g)
    assert all(r.gesture in {1, 2} for r in kept)
    assert len(kept) < len(recs)


def test_per_split_baseline_uses_variant_specific_class_count() -> None:
    g6 = gs.GestureSet.from_config({"ids": [1, 2, 3, 4, 5, 6]})
    g5 = gs.GestureSet.from_config({"ids": [1, 2, 3, 5, 6]})
    # Balanced 6-class fold.
    y6 = np.array([0, 1, 2, 3, 4, 5] * 4)
    y5 = np.array([0, 1, 2, 3, 4] * 4)
    b6 = gs.per_split_baseline(y6, g6)
    b5 = gs.per_split_baseline(y5, g5)
    assert b6["n_classes"] == 6
    assert b5["n_classes"] == 5
    assert b6["chance_level"] == pytest.approx(1.0 / 6)
    assert b5["chance_level"] == pytest.approx(1.0 / 5)
    # Baselines are never inherited.
    assert b6["chance_level"] != b5["chance_level"]


def test_gesture_set_dict_contains_justification_and_evidence() -> None:
    g = gs.GestureSet.from_config({
        "ids": [1, 2, 3, 4, 5, 6],
        "justification": "example",
        "evidence": "docs/gesture_set_evidence.md",
    })
    d = g.to_dict()
    assert d["justification"] == "example"
    assert d["evidence"] == "docs/gesture_set_evidence.md"
    assert d["n_classes"] == 6
    assert set(d["gesture_names"].keys()) == {1, 2, 3, 4, 5, 6}
