"""Leakage-resistant split builder.

A ``Split`` persists three disjoint lists of ``sample_id`` s plus the exact
grouping key and every check that was run. The core guarantee is that the
group_id / recording_id of any train sample never appears in val or test.
Optional content-hash and file-hash overlap checks catch duplicates that
share metadata but differ in bytes (and vice-versa).

Available split families:

- ``indomain_grouped``: recording-grouped train/val/test on a single date.
- ``held_out_date``: leave one date out (train = other date(s) in the same
  room, test = held-out date; val = grouped fold of train).
- ``cross_room``: source room -> target room. Both directions supported.
- ``leave_one_room_out``: source = all other rooms, target = held-out room.

Every builder returns a :class:`Split` and can serialise to ``splits/<id>.json``.
Invalid or empty partitions raise :class:`InvalidSplitError` — never a silent
fallback to random splitting.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.manifest import RecordingRecord


class InvalidSplitError(ValueError):
    """Raised when a split cannot be formed under the requested constraints."""


@dataclass
class Split:
    split_id: str
    kind: str                       # e.g. "indomain_grouped"
    grouping_key: str               # e.g. "group_id" | "recording_id"
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    gestures_train: list[int]
    gestures_val: list[int]
    gestures_test: list[int]
    users_train: list[int]
    users_val: list[int]
    users_test: list[int]
    notes: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)

    def write(self, splits_dir: Path) -> Path:
        splits_dir = Path(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        path = splits_dir / f"{self.split_id}.json"
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, path: Path) -> "Split":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


def _idx_by_sample_id(records: Sequence[RecordingRecord]) -> dict[str, RecordingRecord]:
    idx: dict[str, RecordingRecord] = {}
    for r in records:
        if r.sample_id in idx:
            raise InvalidSplitError(
                f"duplicate sample_id in records: {r.sample_id}"
            )
        idx[r.sample_id] = r
    return idx


def _hash_ids(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(ids):
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:12]


def _run_checks(
    idx: dict[str, RecordingRecord],
    grouping_key: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, Any]:
    """Confirm no leakage, no empty partitions, gesture compatibility.

    ``grouping_key`` is the *attribute name* on RecordingRecord that must
    stay disjoint across partitions ("group_id" or "recording_id").
    """
    parts = {"train": train_ids, "val": val_ids, "test": test_ids}

    # (a) Sample-ID uniqueness.
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    if len(set(all_ids)) != len(all_ids):
        dup = [k for k, v in Counter(all_ids).items() if v > 1]
        raise InvalidSplitError(f"sample_id overlap across partitions: {dup[:5]}")

    # (b) Non-empty partitions.
    for name, ids in parts.items():
        if not ids:
            raise InvalidSplitError(f"partition '{name}' is empty")

    # (c) Grouping-key disjointness (the leakage rule).
    def _groups(ids: Sequence[str]) -> set[str]:
        return {getattr(idx[s], grouping_key) for s in ids}

    g_train = _groups(train_ids)
    g_val = _groups(val_ids)
    g_test = _groups(test_ids)
    bleed = {
        "train_val": sorted(g_train & g_val),
        "train_test": sorted(g_train & g_test),
        "val_test": sorted(g_val & g_test),
    }
    for k, v in bleed.items():
        if v:
            raise InvalidSplitError(
                f"grouping_key={grouping_key} leaked across {k}: {v[:5]}"
            )

    # (d) Content-hash overlap catches literal-duplicate recordings that
    #     somehow ended up under distinct group_ids. The empty-file SHA
    #     prefix is ignored — every 0-byte file collides at that value
    #     and would produce spurious leakage warnings.
    from experiments.manifest import EMPTY_FILE_HASH_16

    def _hashes(ids: Sequence[str]) -> set[str]:
        s: set[str] = set()
        for sid in ids:
            for h in idx[sid].file_hashes.values():
                if h == EMPTY_FILE_HASH_16:
                    continue
                s.add(h)
        return s

    h_train = _hashes(train_ids)
    h_val = _hashes(val_ids)
    h_test = _hashes(test_ids)
    content_bleed = {
        "train_val_file_hashes": len(h_train & h_val),
        "train_test_file_hashes": len(h_train & h_test),
        "val_test_file_hashes": len(h_val & h_test),
    }
    for k, v in content_bleed.items():
        if v:
            raise InvalidSplitError(
                f"file-hash overlap {k}={v} (duplicate recordings)"
            )

    # (e) Gesture / user overlap notes (informational, not blocking).
    gest_train = sorted({idx[s].gesture for s in train_ids})
    gest_val = sorted({idx[s].gesture for s in val_ids})
    gest_test = sorted({idx[s].gesture for s in test_ids})
    users_train = sorted({idx[s].user for s in train_ids})
    users_val = sorted({idx[s].user for s in val_ids})
    users_test = sorted({idx[s].user for s in test_ids})

    return {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_train_groups": len(g_train),
        "n_val_groups": len(g_val),
        "n_test_groups": len(g_test),
        "gestures_train": gest_train,
        "gestures_val": gest_val,
        "gestures_test": gest_test,
        "user_overlap_train_test": sorted(set(users_train) & set(users_test)),
        "user_overlap_train_val": sorted(set(users_train) & set(users_val)),
        "content_hash_overlap": content_bleed,
        "grouping_key": grouping_key,
    }


def _finalize(
    idx: dict[str, RecordingRecord],
    kind: str,
    grouping_key: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    notes: dict[str, Any] | None = None,
    split_id: str | None = None,
) -> Split:
    checks = _run_checks(idx, grouping_key, train_ids, val_ids, test_ids)
    if split_id is None:
        digest = _hash_ids(list(train_ids) + list(val_ids) + list(test_ids))
        split_id = f"{kind}-{digest}"
    return Split(
        split_id=split_id,
        kind=kind,
        grouping_key=grouping_key,
        train_ids=list(train_ids),
        val_ids=list(val_ids),
        test_ids=list(test_ids),
        gestures_train=checks["gestures_train"],
        gestures_val=checks["gestures_val"],
        gestures_test=checks["gestures_test"],
        users_train=sorted({idx[s].user for s in train_ids}),
        users_val=sorted({idx[s].user for s in val_ids}),
        users_test=sorted({idx[s].user for s in test_ids}),
        notes=dict(notes or {}),
        checks=checks,
    )


# ─── Split families ─────────────────────────────────────────────────────────

def _shuffle_stratified_by_gesture(
    idx: dict[str, RecordingRecord],
    group_to_ids: dict[str, list[str]],
    seed: int,
) -> list[str]:
    """Return a group order stratified by gesture composition. Deterministic."""
    import random
    order = sorted(group_to_ids.keys())
    rng = random.Random(seed)
    # Group's primary gesture = the majority gesture of its samples.
    def _primary(g: str) -> int:
        gestures = [idx[s].gesture for s in group_to_ids[g]]
        return Counter(gestures).most_common(1)[0][0]
    per_gesture: dict[int, list[str]] = defaultdict(list)
    for g in order:
        per_gesture[_primary(g)].append(g)
    for gs in per_gesture.values():
        rng.shuffle(gs)
    out: list[str] = []
    pointers = {k: 0 for k in per_gesture}
    while any(pointers[k] < len(per_gesture[k]) for k in per_gesture):
        for k in sorted(per_gesture):
            i = pointers[k]
            if i < len(per_gesture[k]):
                out.append(per_gesture[k][i])
                pointers[k] += 1
    return out


def indomain_grouped(
    records: Sequence[RecordingRecord],
    date: str,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    split_id: str | None = None,
) -> Split:
    """Recording-grouped train/val/test on a single date."""
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    idx = _idx_by_sample_id(records)
    pool = [r for r in records if r.date == date]
    if not pool:
        raise InvalidSplitError(f"no records for date={date}")

    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in pool:
        group_to_ids[r.group_id].append(r.sample_id)
    ordered_groups = _shuffle_stratified_by_gesture(idx, group_to_ids, seed)
    n = len(ordered_groups)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        raise InvalidSplitError(
            f"date {date}: only {n} groups — cannot form "
            f"train/val({n_val})/test({n_test})"
        )
    test_groups = set(ordered_groups[:n_test])
    val_groups = set(ordered_groups[n_test:n_test + n_val])
    train_groups = set(ordered_groups[n_test + n_val:])

    def _ids(groups: set[str]) -> list[str]:
        out: list[str] = []
        for g in sorted(groups):
            out.extend(sorted(group_to_ids[g]))
        return out

    return _finalize(
        idx, kind="indomain_grouped", grouping_key="group_id",
        train_ids=_ids(train_groups),
        val_ids=_ids(val_groups),
        test_ids=_ids(test_groups),
        notes={
            "date": date, "seed": seed,
            "val_frac": val_frac, "test_frac": test_frac,
        },
        split_id=split_id,
    )


def held_out_date(
    records: Sequence[RecordingRecord],
    train_dates: Sequence[str],
    test_date: str,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    split_id: str | None = None,
) -> Split:
    """Train/val on ``train_dates`` (grouped), test on ``test_date``.

    Room / date confound: this can be *cross-room* or *within-room* depending
    on the actual date rooms. The room mapping is recorded in ``notes``.
    """
    if test_date in train_dates:
        raise InvalidSplitError(
            f"test_date={test_date!r} is also in train_dates={list(train_dates)}"
        )
    idx = _idx_by_sample_id(records)

    train_pool = [r for r in records if r.date in set(train_dates)]
    test_pool = [r for r in records if r.date == test_date]
    if not train_pool:
        raise InvalidSplitError(f"no records for train_dates={list(train_dates)}")
    if not test_pool:
        raise InvalidSplitError(f"no records for test_date={test_date}")

    train_group_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in train_pool:
        train_group_to_ids[r.group_id].append(r.sample_id)
    ordered = _shuffle_stratified_by_gesture(idx, train_group_to_ids, seed)
    n = len(ordered)
    n_val = max(1, int(round(n * val_frac)))
    if n_val >= n:
        raise InvalidSplitError(
            f"train pool has {n} groups — cannot spare {n_val} for val"
        )
    val_groups = set(ordered[:n_val])
    train_groups = set(ordered[n_val:])
    train_ids = [s for g in sorted(train_groups) for s in sorted(train_group_to_ids[g])]
    val_ids = [s for g in sorted(val_groups) for s in sorted(train_group_to_ids[g])]
    test_ids = sorted(r.sample_id for r in test_pool)

    train_rooms = sorted({r.room for r in train_pool})
    test_rooms = sorted({r.room for r in test_pool})
    common_gest = sorted(
        {r.gesture for r in train_pool} & {r.gesture for r in test_pool}
    )
    return _finalize(
        idx, kind="held_out_date", grouping_key="group_id",
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        notes={
            "train_dates": sorted(train_dates), "test_date": test_date,
            "train_rooms": train_rooms, "test_rooms": test_rooms,
            "room_date_confound": train_rooms != test_rooms,
            "gestures_common": common_gest,
            "seed": seed, "val_frac": val_frac,
        },
        split_id=split_id,
    )


def cross_room(
    records: Sequence[RecordingRecord],
    source_room: int,
    target_room: int,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    restrict_gestures: bool = True,
    split_id: str | None = None,
) -> Split:
    """Train/val on source_room only, test on target_room."""
    if source_room == target_room:
        raise InvalidSplitError("source_room == target_room")
    idx = _idx_by_sample_id(records)
    src = [r for r in records if r.room == source_room]
    tgt = [r for r in records if r.room == target_room]
    if not src:
        raise InvalidSplitError(f"no records in source_room={source_room}")
    if not tgt:
        raise InvalidSplitError(f"no records in target_room={target_room}")

    if restrict_gestures:
        common = {r.gesture for r in src} & {r.gesture for r in tgt}
        if not common:
            raise InvalidSplitError(
                f"no gestures common to rooms {source_room} and {target_room}"
            )
        src = [r for r in src if r.gesture in common]
        tgt = [r for r in tgt if r.gesture in common]

    src_group_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in src:
        src_group_to_ids[r.group_id].append(r.sample_id)
    ordered = _shuffle_stratified_by_gesture(idx, src_group_to_ids, seed)
    n = len(ordered)
    n_val = max(1, int(round(n * val_frac)))
    if n_val >= n:
        raise InvalidSplitError(
            f"source room {source_room}: {n} groups — cannot spare {n_val} for val"
        )
    val_groups = set(ordered[:n_val])
    train_groups = set(ordered[n_val:])
    train_ids = [s for g in sorted(train_groups) for s in sorted(src_group_to_ids[g])]
    val_ids = [s for g in sorted(val_groups) for s in sorted(src_group_to_ids[g])]
    test_ids = sorted(r.sample_id for r in tgt)

    return _finalize(
        idx, kind="cross_room", grouping_key="group_id",
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        notes={
            "source_room": source_room, "target_room": target_room,
            "restrict_gestures": restrict_gestures,
            "gestures_source": sorted({r.gesture for r in src}),
            "gestures_target": sorted({r.gesture for r in tgt}),
            "seed": seed, "val_frac": val_frac,
        },
        split_id=split_id,
    )


def same_room_different_date(
    records: Sequence[RecordingRecord],
    room: int,
    train_date: str,
    test_date: str,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    restrict_gestures: bool = True,
    split_id: str | None = None,
) -> Split:
    """Train/val on ``train_date`` (grouped) in ``room``, test on
    ``test_date`` in the same ``room``.

    Answers the room-vs-session question the packaged Widar3.0 tree can
    otherwise never separate. Fails LOUDLY if the two dates are not in
    the same room, or if either date is empty, or if the label sets are
    disjoint under the caller's records.
    """
    if train_date == test_date:
        raise InvalidSplitError("train_date == test_date")
    idx = _idx_by_sample_id(records)
    train_pool = [r for r in records if r.date == train_date and r.room == room]
    test_pool = [r for r in records if r.date == test_date and r.room == room]
    if not train_pool:
        raise InvalidSplitError(
            f"no records for train_date={train_date} room={room}"
        )
    if not test_pool:
        raise InvalidSplitError(
            f"no records for test_date={test_date} room={room}"
        )
    # Confirm the two dates really are in the same room in the manifest.
    train_rooms = {r.room for r in train_pool}
    test_rooms = {r.room for r in test_pool}
    if train_rooms != {room} or test_rooms != {room}:
        raise InvalidSplitError(
            f"same_room_different_date: room mismatch "
            f"(train_rooms={sorted(train_rooms)}, test_rooms={sorted(test_rooms)})"
        )

    if restrict_gestures:
        common = {r.gesture for r in train_pool} & {r.gesture for r in test_pool}
        if not common:
            raise InvalidSplitError(
                f"no gestures common to dates {train_date} and {test_date} "
                f"in room {room}"
            )
        train_pool = [r for r in train_pool if r.gesture in common]
        test_pool = [r for r in test_pool if r.gesture in common]

    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in train_pool:
        group_to_ids[r.group_id].append(r.sample_id)
    ordered = _shuffle_stratified_by_gesture(idx, group_to_ids, seed)
    n = len(ordered)
    n_val = max(1, int(round(n * val_frac)))
    if n_val >= n:
        raise InvalidSplitError(
            f"same_room_different_date: {n} source groups — "
            f"cannot spare {n_val} for val"
        )
    val_groups = set(ordered[:n_val])
    train_groups = set(ordered[n_val:])
    train_ids = [s for g in sorted(train_groups) for s in sorted(group_to_ids[g])]
    val_ids = [s for g in sorted(val_groups) for s in sorted(group_to_ids[g])]
    test_ids = sorted(r.sample_id for r in test_pool)

    return _finalize(
        idx, kind="same_room_different_date", grouping_key="group_id",
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        notes={
            "room": room,
            "train_date": train_date, "test_date": test_date,
            "room_date_confound": False,
            "same_room_diff_date": True,
            "gestures_common": sorted(
                {r.gesture for r in train_pool}
                & {r.gesture for r in test_pool}
            ),
            "seed": seed, "val_frac": val_frac,
        },
        split_id=split_id,
    )


def leave_one_room_out(
    records: Sequence[RecordingRecord],
    held_out_room: int,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    restrict_gestures: bool = True,
    split_id: str | None = None,
) -> Split:
    """Train/val on all other rooms, test on held_out_room."""
    idx = _idx_by_sample_id(records)
    other = [r for r in records if r.room != held_out_room]
    tgt = [r for r in records if r.room == held_out_room]
    if not other:
        raise InvalidSplitError("no source rooms — cannot leave one out")
    if not tgt:
        raise InvalidSplitError(f"no records in held_out_room={held_out_room}")

    if restrict_gestures:
        common = {r.gesture for r in other} & {r.gesture for r in tgt}
        if not common:
            raise InvalidSplitError("no gestures common across source/held-out")
        other = [r for r in other if r.gesture in common]
        tgt = [r for r in tgt if r.gesture in common]

    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in other:
        group_to_ids[r.group_id].append(r.sample_id)
    ordered = _shuffle_stratified_by_gesture(idx, group_to_ids, seed)
    n = len(ordered)
    n_val = max(1, int(round(n * val_frac)))
    if n_val >= n:
        raise InvalidSplitError(
            f"loro: {n} source groups — cannot spare {n_val} for val"
        )
    val_groups = set(ordered[:n_val])
    train_groups = set(ordered[n_val:])
    train_ids = [s for g in sorted(train_groups) for s in sorted(group_to_ids[g])]
    val_ids = [s for g in sorted(val_groups) for s in sorted(group_to_ids[g])]
    test_ids = sorted(r.sample_id for r in tgt)

    return _finalize(
        idx, kind="leave_one_room_out", grouping_key="group_id",
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        notes={
            "held_out_room": held_out_room,
            "source_rooms": sorted({r.room for r in other}),
            "seed": seed, "val_frac": val_frac,
        },
        split_id=split_id,
    )
