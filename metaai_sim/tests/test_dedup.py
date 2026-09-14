"""Duplicate-split detection: identical (train, val, test) membership
must be recognised regardless of split kind or grouping metadata."""

from __future__ import annotations

import pytest

from experiments import dedup, features, splits


@pytest.fixture
def records():
    recs, _ = features.synthetic_fixture(n_recordings=60, n_gestures=3)
    return recs


def test_no_duplicates_when_splits_differ(records) -> None:
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    b = splits.indomain_grouped(records, date="20181109", seed=43)
    result = dedup.dedup_splits([("a", a), ("b", b)])
    assert result.n_unique == 2
    assert result.n_aliases == 0


def test_identical_split_ids_collapse_to_alias(records) -> None:
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    b = splits.indomain_grouped(records, date="20181109", seed=42)
    result = dedup.dedup_splits([("first", a), ("second", b)])
    assert result.n_unique == 1
    assert result.aliases == {"second": "first"}


def test_cross_room_equals_held_out_date_alias() -> None:
    """When room 1 has only 20181109 and room 2 has only 20181118,
    ``cross_room 1 -> 2`` and ``held_out_date test=20181118`` resolve to
    identical membership. The dedup module must catch that."""
    recs, _ = features.synthetic_fixture(n_recordings=60, n_gestures=3)
    cr = splits.cross_room(recs, source_room=1, target_room=2, seed=42)
    hod = splits.held_out_date(
        recs, train_dates=["20181109"], test_date="20181118", seed=42,
    )
    result = dedup.dedup_splits([
        ("cross_room_1_to_2", cr),
        ("heldout_test_20181118", hod),
    ])
    # Both use train_date=20181109 and test_date=20181118 with the same
    # seed and val_frac, so the sorted membership matches.
    assert result.n_unique == 1
    assert result.aliases == {"heldout_test_20181118": "cross_room_1_to_2"}


def test_manifest_dict_carries_alias_map(records) -> None:
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    b = splits.indomain_grouped(records, date="20181109", seed=42)
    result = dedup.dedup_splits([("first", a), ("second", b)])
    payload = result.to_manifest_dict()
    assert payload["canonical_split_ids"] == ["first"]
    assert payload["aliases"] == {"second": "first"}
    assert payload["n_unique"] == 1
    assert payload["n_aliases"] == 1


def test_cross_suite_aliasing_via_known_fingerprints(records) -> None:
    """Regression: split_dedup.json previously showed identical
    fingerprints across suites (e.g. loro_room_2 == heldout_test_20181118
    == cross_room_1_to_2 all sharing 8b6665be23d93096) while every
    suite still reported n_aliases == 0. Detected but not acted on.

    With ``known_fingerprints`` threaded across suites, the second suite
    must record the collision as ``alias_of`` and exclude the alias from
    its canonical count.
    """
    # Suite 1: canonical.
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    r1 = dedup.dedup_splits([("suite1_indomain", a)])
    # The registry maps fingerprint -> canonical split id, mirroring what
    # the runner threads across suites.
    known: dict[str, str] = {
        dedup.split_fingerprint(split): sid for sid, split in r1.canonical
    }

    # Suite 2: identical membership under a different suite/split name.
    b = splits.indomain_grouped(records, date="20181109", seed=42)
    r2 = dedup.dedup_splits(
        [("suite2_indomain", b)], known_fingerprints=known,
    )
    assert r2.canonical == []
    assert r2.aliases == {"suite2_indomain": "suite1_indomain"}
    assert r2.n_unique == 0
    assert r2.n_aliases == 1


def test_cross_suite_registry_is_stable(records) -> None:
    """Registering the canonical fingerprint means later dedup passes on
    the SAME split id no longer count it as a fresh canonical row."""
    a = splits.indomain_grouped(records, date="20181109", seed=42)
    known: dict[str, str] = {}
    r1 = dedup.dedup_splits([("a", a)], known_fingerprints=known)
    # simulate the runner promoting canonical fps
    for sid, split in r1.canonical:
        known[dedup.split_fingerprint(split)] = sid
    r2 = dedup.dedup_splits([("a_again", a)], known_fingerprints=known)
    assert r2.canonical == []
    assert r2.aliases == {"a_again": "a"}
