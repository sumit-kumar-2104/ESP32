"""Tests for experiments.splits.

Uses the synthetic fixture from ``experiments.features`` — the split logic
is unit-testable without any real dataset.
"""

from __future__ import annotations

import pytest

from experiments import features, manifest, splits


@pytest.fixture
def records():
    recs, _bundle = features.synthetic_fixture(n_recordings=60, n_gestures=3)
    return recs


def test_indomain_grouped_no_group_leakage(records) -> None:
    split = splits.indomain_grouped(records, date="20181109", seed=42)
    idx = {r.sample_id: r for r in records}
    g_train = {idx[s].group_id for s in split.train_ids}
    g_val = {idx[s].group_id for s in split.val_ids}
    g_test = {idx[s].group_id for s in split.test_ids}
    assert not (g_train & g_val)
    assert not (g_train & g_test)
    assert not (g_val & g_test)
    assert split.checks["grouping_key"] == "group_id"


def test_indomain_grouped_sample_id_uniqueness(records) -> None:
    split = splits.indomain_grouped(records, date="20181109", seed=42)
    all_ids = split.train_ids + split.val_ids + split.test_ids
    assert len(set(all_ids)) == len(all_ids)


def test_indomain_grouped_deterministic_with_seed(records) -> None:
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    b = splits.indomain_grouped(records, date="20181109", seed=42)
    assert (a.train_ids, a.val_ids, a.test_ids) == (b.train_ids, b.val_ids, b.test_ids)


def test_indomain_grouped_bad_date(records) -> None:
    with pytest.raises(splits.InvalidSplitError):
        splits.indomain_grouped(records, date="99999999", seed=42)


def test_cross_room_no_group_leakage(records) -> None:
    split = splits.cross_room(records, source_room=1, target_room=2, seed=42)
    idx = {r.sample_id: r for r in records}
    train_rooms = {idx[s].room for s in split.train_ids}
    val_rooms = {idx[s].room for s in split.val_ids}
    test_rooms = {idx[s].room for s in split.test_ids}
    assert train_rooms == {1}
    assert val_rooms == {1}
    assert test_rooms == {2}


def test_cross_room_same_room_rejected(records) -> None:
    with pytest.raises(splits.InvalidSplitError):
        splits.cross_room(records, source_room=1, target_room=1)


def test_held_out_date_records_room_confound(records) -> None:
    split = splits.held_out_date(records, ["20181109"], "20181118", seed=42)
    assert split.notes["room_date_confound"] is True
    assert split.notes["train_dates"] == ["20181109"]
    assert split.notes["test_date"] == "20181118"


def test_loro_needs_common_gestures(records) -> None:
    split = splits.leave_one_room_out(records, held_out_room=2, seed=42)
    idx = {r.sample_id: r for r in records}
    assert {idx[s].room for s in split.test_ids} == {2}
    assert 2 not in {idx[s].room for s in split.train_ids}


def test_file_hash_overlap_detected() -> None:
    """If two 'recordings' happen to share file hashes, the split must
    refuse them across partitions. Simulated by injecting a duplicate."""
    recs, _ = features.synthetic_fixture(n_recordings=20, n_gestures=2)
    # Create a rec in room 2 that reuses room-1 hashes.
    poisoner = manifest.RecordingRecord(
        sample_id="poison_id_1",
        recording_id="poison-rec-1",
        group_id="poison-group-1-2-20181118",
        session_id="20181118",
        date="20181118", room=2, user=99, gesture=1, gesture_name="g1",
        location=1, orientation=1, repetition=1,
        receivers_present=[1, 2, 3, 4, 5, 6],
        receivers_missing=[],
        file_paths={f"r{k}": f"poison-{k}.dat" for k in range(1, 7)},
        # Same hashes as recs[0] so cross_room finds an overlap.
        file_hashes=dict(recs[0].file_hashes),
        file_sizes={f"r{k}": 4096 for k in range(1, 7)},
        packet_counts={f"r{k}": 128 for k in range(1, 7)},
        quality="ok", quality_detail="ok",
    )
    recs_plus = list(recs) + [poisoner]
    with pytest.raises(splits.InvalidSplitError, match="file-hash overlap"):
        splits.cross_room(recs_plus, source_room=1, target_room=2, seed=42)


def test_persist_round_trip(records, tmp_path) -> None:
    split = splits.indomain_grouped(records, date="20181109", seed=42)
    path = split.write(tmp_path)
    reloaded = splits.Split.read(path)
    assert reloaded.split_id == split.split_id
    assert reloaded.train_ids == split.train_ids
    assert reloaded.checks == split.checks
