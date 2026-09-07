"""Data inventory: rooms, dates, users, locations, orientations.

Used to answer: can we separate the room effect from the session
(recording-date) effect with the packaged data? A same-room /
different-date evaluation requires at least two dates recorded in the
same physical room with compatible label sets under the active gesture
configuration.

Never simulate, subsample, or relabel to manufacture this contrast.
When the data does not support the contrast, the inventory reports the
exact reason and the runner marks the experiment ``unavailable``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from experiments.manifest import DATE_ROOM, RecordingRecord


@dataclass
class RoomDateInventory:
    rooms_seen: list[int]
    dates_seen: list[str]
    users_seen: list[int]
    locations_seen: list[int]
    orientations_seen: list[int]
    dates_per_room: dict[int, list[str]] = field(default_factory=dict)
    gestures_per_date: dict[str, list[int]] = field(default_factory=dict)
    rooms_with_multiple_dates: list[int] = field(default_factory=list)
    same_room_diff_date_available: bool = False
    same_room_diff_date_reason: str = ""
    same_room_diff_date_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rooms_seen": list(self.rooms_seen),
            "dates_seen": list(self.dates_seen),
            "users_seen": list(self.users_seen),
            "locations_seen": list(self.locations_seen),
            "orientations_seen": list(self.orientations_seen),
            "dates_per_room": {str(k): list(v) for k, v in self.dates_per_room.items()},
            "gestures_per_date": {k: list(v) for k, v in self.gestures_per_date.items()},
            "rooms_with_multiple_dates": list(self.rooms_with_multiple_dates),
            "same_room_diff_date_available": bool(self.same_room_diff_date_available),
            "same_room_diff_date_reason": self.same_room_diff_date_reason,
            "same_room_diff_date_candidates": list(self.same_room_diff_date_candidates),
        }


def build_inventory(
    records: Sequence[RecordingRecord],
    active_gesture_ids: Iterable[int],
) -> RoomDateInventory:
    """Enumerate axes present in ``records`` (post-gesture-set filter).

    ``active_gesture_ids`` is the currently declared gesture set. A
    same-room / different-date evaluation candidate requires the two
    dates to share at least the active ids after the filter.
    """
    active = set(int(g) for g in active_gesture_ids)
    rooms: set[int] = set()
    dates: set[str] = set()
    users: set[int] = set()
    locs: set[int] = set()
    oris: set[int] = set()
    dates_per_room: dict[int, set[str]] = defaultdict(set)
    gestures_per_date: dict[str, set[int]] = defaultdict(set)
    for r in records:
        rooms.add(int(r.room))
        dates.add(str(r.date))
        users.add(int(r.user))
        locs.add(int(r.location))
        oris.add(int(r.orientation))
        dates_per_room[int(r.room)].add(str(r.date))
        gestures_per_date[str(r.date)].add(int(r.gesture))

    rooms_multi = sorted(rm for rm, ds in dates_per_room.items() if len(ds) >= 2)
    candidates: list[dict] = []
    for rm in rooms_multi:
        ds = sorted(dates_per_room[rm])
        # Pair-wise same-room / different-date candidates whose intersection
        # under the active gesture set is non-empty.
        for i, d1 in enumerate(ds):
            for d2 in ds[i + 1:]:
                shared = sorted((gestures_per_date[d1] & gestures_per_date[d2]) & active)
                if shared:
                    candidates.append({
                        "room": rm,
                        "date_a": d1,
                        "date_b": d2,
                        "shared_gestures_active": shared,
                    })
    available = bool(candidates)
    reason = "" if available else (
        "Packaged data has no room with two or more distinct recording "
        "dates sharing labels under the active gesture set. Room and "
        "session (date) are fully confounded here; a room effect cannot "
        "be separated from a session effect with this data."
    )
    return RoomDateInventory(
        rooms_seen=sorted(rooms),
        dates_seen=sorted(dates),
        users_seen=sorted(users),
        locations_seen=sorted(locs),
        orientations_seen=sorted(oris),
        dates_per_room={rm: sorted(v) for rm, v in dates_per_room.items()},
        gestures_per_date={d: sorted(v) for d, v in gestures_per_date.items()},
        rooms_with_multiple_dates=rooms_multi,
        same_room_diff_date_available=available,
        same_room_diff_date_reason=reason,
        same_room_diff_date_candidates=candidates,
    )


def known_date_room_map() -> dict[str, int]:
    """Expose the canonical Widar3.0 date -> room map for documentation
    and off-line inventory reporting. See ``experiments.manifest.DATE_ROOM``."""
    return dict(DATE_ROOM)
