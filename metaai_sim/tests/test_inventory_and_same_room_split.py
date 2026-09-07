"""Room / date inventory and same-room-different-date availability."""

from __future__ import annotations

import pytest

from experiments import features, inventory, splits


@pytest.fixture
def records():
    recs, _ = features.synthetic_fixture(n_recordings=60, n_gestures=3)
    return recs


def test_inventory_enumerates_axes(records) -> None:
    inv = inventory.build_inventory(records, active_gesture_ids=[1, 2, 3])
    assert set(inv.rooms_seen) == {1, 2}
    assert set(inv.dates_seen) == {"20181109", "20181118"}


def test_same_room_diff_date_unavailable_on_packaged_data(records) -> None:
    """The packaged fixture has 20181109 only in room 1 and 20181118 only
    in room 2. Same-room / different-date is therefore unavailable."""
    inv = inventory.build_inventory(records, active_gesture_ids=[1, 2, 3])
    assert inv.same_room_diff_date_available is False
    assert "room and session (date) are fully confounded" \
        in inv.same_room_diff_date_reason.lower()
    assert inv.same_room_diff_date_candidates == []


def test_same_room_diff_date_available_when_room_has_two_dates() -> None:
    """Synthesise a manifest with room 1 recorded on TWO dates so the
    inventory can produce a candidate pair. Does NOT relabel or
    subsample any real data."""
    recs, _ = features.synthetic_fixture(n_recordings=30, n_gestures=3)
    # Rewrite room + date on the second half to add a same-room/second-date
    # slice — deliberately synthetic; not real data.
    from experiments.manifest import RecordingRecord
    extras = []
    for i, r in enumerate(recs[:10]):
        # Same room=1 but different date=20181115.
        extras.append(RecordingRecord(
            sample_id=f"synth_extra_{i}",
            recording_id=f"synth_extra_{i}_id",
            group_id=f"9-{r.location}-{r.orientation}-1-20181115",
            session_id="20181115", date="20181115", room=1,
            user=9 + i, gesture=r.gesture, gesture_name=r.gesture_name,
            location=r.location, orientation=r.orientation,
            repetition=100 + i,
            receivers_present=r.receivers_present,
            receivers_missing=r.receivers_missing,
            file_paths=dict(r.file_paths),
            file_hashes={k: v[::-1] for k, v in r.file_hashes.items()},
            file_sizes=dict(r.file_sizes),
            packet_counts=dict(r.packet_counts),
            quality="ok", quality_detail="synthetic-extra",
        ))
    combined = list(recs) + extras
    inv = inventory.build_inventory(combined, active_gesture_ids=[1, 2, 3])
    assert 1 in inv.rooms_with_multiple_dates
    assert inv.same_room_diff_date_available is True
    assert any(c["room"] == 1 for c in inv.same_room_diff_date_candidates)


def test_same_room_different_date_split_kind_records_no_confound() -> None:
    """The new split kind must set ``room_date_confound`` to False and
    record the pair (train_date, test_date)."""
    from experiments.manifest import RecordingRecord
    recs, _ = features.synthetic_fixture(n_recordings=30, n_gestures=3)
    extras = []
    for i, r in enumerate(recs[:12]):
        extras.append(RecordingRecord(
            sample_id=f"srdd_{i}",
            recording_id=f"srdd_{i}_id",
            group_id=f"9-{r.location}-{r.orientation}-1-20181115",
            session_id="20181115", date="20181115", room=1,
            user=9 + i, gesture=r.gesture, gesture_name=r.gesture_name,
            location=r.location, orientation=r.orientation,
            repetition=200 + i,
            receivers_present=r.receivers_present,
            receivers_missing=r.receivers_missing,
            file_paths=dict(r.file_paths),
            file_hashes={k: v[::-1] for k, v in r.file_hashes.items()},
            file_sizes=dict(r.file_sizes),
            packet_counts=dict(r.packet_counts),
            quality="ok", quality_detail="synthetic-extra",
        ))
    combined = list(recs) + extras
    split = splits.same_room_different_date(
        combined, room=1,
        train_date="20181109", test_date="20181115",
        val_frac=0.2, seed=42,
    )
    assert split.notes["room_date_confound"] is False
    assert split.notes["train_date"] == "20181109"
    assert split.notes["test_date"] == "20181115"
    assert split.notes["room"] == 1


def test_same_room_different_date_rejects_room_mismatch() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=30, n_gestures=3)
    with pytest.raises(splits.InvalidSplitError):
        splits.same_room_different_date(
            recs, room=1,
            train_date="20181109", test_date="20181118", seed=42,
        )
