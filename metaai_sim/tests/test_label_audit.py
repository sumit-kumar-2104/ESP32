"""Cover the cross-domain label / class-index mapping check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import label_audit
from experiments.label_audit import (
    ClassMapMismatch,
    assert_cross_domain_class_maps_agree,
    audit_saved_run_class_maps,
)


def test_indomain_skips_check() -> None:
    # In-domain splits are not audited; a mismatch here must not raise.
    assert_cross_domain_class_maps_agree(
        split_kind="indomain_grouped",
        class_index_to_gesture_train={0: 1},
        class_index_to_gesture_test={0: 3},
        y_train=[0, 0],
        y_test=[0, 0],
        split_id="ignored",
    )


def test_cross_domain_maps_agree_passes() -> None:
    assert_cross_domain_class_maps_agree(
        split_kind="cross_room",
        class_index_to_gesture_train={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        class_index_to_gesture_test=None,   # inherited by construction
        y_train=[0, 1, 2, 3, 4],
        y_test=[0, 1, 2, 3, 4],
        split_id="cross_room_1_to_2",
    )


def test_cross_domain_maps_disagree_raises_loudly() -> None:
    with pytest.raises(ClassMapMismatch) as exc:
        assert_cross_domain_class_maps_agree(
            split_kind="cross_room",
            class_index_to_gesture_train={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            class_index_to_gesture_test={0: 2, 1: 1, 2: 3, 3: 4, 4: 5},
            y_train=[0, 1, 2, 3, 4],
            y_test=[0, 1, 2, 3, 4],
            split_id="cross_room_1_to_2",
        )
    msg = str(exc.value)
    assert "cross_room_1_to_2" in msg
    assert "class_index_to_gesture" in msg


def test_test_class_absent_from_training_raises() -> None:
    with pytest.raises(ClassMapMismatch) as exc:
        assert_cross_domain_class_maps_agree(
            split_kind="cross_room",
            class_index_to_gesture_train={0: 1, 1: 2},
            class_index_to_gesture_test=None,
            y_train=[0, 1, 0, 1],
            # class-index 2 present in test but not in training map.
            y_test=[0, 1, 2, 0],
            split_id="cross_room_2_to_1",
        )
    assert "not in the training class map" in str(exc.value)


def test_held_out_date_is_audited() -> None:
    with pytest.raises(ClassMapMismatch):
        assert_cross_domain_class_maps_agree(
            split_kind="held_out_date",
            class_index_to_gesture_train={0: 1, 1: 2},
            class_index_to_gesture_test={0: 2, 1: 1},
            y_train=[0, 1],
            y_test=[0, 1],
            split_id="heldout",
        )


def test_audit_saved_run_detects_mismatch(tmp_path: Path) -> None:
    """Fabricate the on-disk layout the audit walks."""
    run = tmp_path / "runX"
    (run / "splits").mkdir(parents=True)
    (run / "splits" / "cross_room_1_to_2.json").write_text(
        json.dumps({"split_id": "cross_room_1_to_2", "kind": "cross_room"}),
        encoding="utf-8",
    )
    exp = (run / "experiments" / "raw_cross_room"
           / "cross_room_1_to_2" / "digital_mlp_raw" / "seed_42")
    (exp / "metrics").mkdir(parents=True)
    (exp / "status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8",
    )
    (exp / "metrics" / "validation.json").write_text(
        json.dumps({"best": {"class_index_to_gesture": {"0": 1, "1": 2, "2": 3}}}),
        encoding="utf-8",
    )
    (exp / "metrics" / "test.json").write_text(
        # deliberately rotated
        json.dumps({"class_index_to_gesture": {"0": 2, "1": 1, "2": 3}}),
        encoding="utf-8",
    )
    reports = audit_saved_run_class_maps(run)
    assert len(reports) == 1
    r = reports[0]
    assert r.ok is False
    assert r.split_id == "cross_room_1_to_2"
    assert r.arm == "digital_mlp_raw"
    summary = label_audit.summarise_reports(reports)
    assert summary["n_map_mismatches"] == 1


def test_audit_ignores_indomain(tmp_path: Path) -> None:
    run = tmp_path / "runY"
    (run / "splits").mkdir(parents=True)
    (run / "splits" / "indomain.json").write_text(
        json.dumps({"split_id": "indomain", "kind": "indomain_grouped"}),
        encoding="utf-8",
    )
    exp = (run / "experiments" / "raw_indomain"
           / "indomain" / "logreg_raw" / "seed_42")
    (exp / "metrics").mkdir(parents=True)
    (exp / "status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8",
    )
    (exp / "metrics" / "test.json").write_text(
        json.dumps({"class_index_to_gesture": {"0": 1}}), encoding="utf-8",
    )
    reports = audit_saved_run_class_maps(run)
    assert reports == []
