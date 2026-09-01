"""
Canonical Widar3.0 gesture-ID -> gesture-name mapping, PER DATE.

Widar3.0 recorded different gesture *sets* on different dates. The numeric
gesture_id field in the filename (`user{U}-{gid}-{loc}-{ori}-{room}-r{rep}`)
therefore does NOT refer to the same gesture across every date. If a script
naively pools multiple dates and uses `gesture_id - 1` as the class label
(as the BVP/CSI loaders currently do), some dates will silently mis-label
their samples relative to others.

The maps below are the canonical per-date mappings taken from the Widar3.0
README (`Data instructions.pdf`, Table "Gesture ID for each date"). If a date
is not listed here, it is treated as unknown by the label-consistency check
and reported as such — never silently accepted.

The 6-gesture pool used by the paper reproduction is:

    Push&Pull, Sweep, Clap, Slide, Draw-O(H), Draw-Zigzag(H)

Dates whose numeric ids point at those six identical gestures form the
in-domain-safe subset. Dates that overload the same numeric id with a
different physical gesture MUST be relabeled (or excluded) before pooling.
"""

from typing import Dict, Optional

# Per-date gesture-id -> canonical gesture name.
# Ids that are not part of the six-gesture pool are still listed (as their
# actual gesture names), so the consistency checker can identify overloads.
#
# NOTE ON PROVENANCE: These are the mappings we treat as canonical for this
# repo. They match the paper's convention that gesture ids 1..6 on
# 20181109-VS are Push&Pull, Sweep, Clap, Slide, Draw-O(H), Draw-Zigzag(H).
# Dates for which the exact mapping is uncertain are marked "unverified"
# and will trigger a warning in the consistency check.

GESTURE_MAP_BY_DATE: Dict[str, Dict[int, str]] = {
    # Room 1 — 6-gesture pool, ids match the paper convention.
    "20181109": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-O(H)", 6: "Draw-Zigzag(H)"},
    "20181115": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-O(H)", 6: "Draw-Zigzag(H)"},
    "20181121": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-O(H)", 6: "Draw-Zigzag(H)"},

    # Room 2 — same six gestures, same ids for 1..4, but the 5/6 pair is
    # Draw-N(H) / Draw-O(H) on these dates instead of Draw-O(H) / Draw-Zigzag(H).
    # THIS IS THE OVERLOAD: gesture_id=5 means Draw-O(H) on 20181109 but
    # Draw-N(H) on 20181117. Pooling without a remap silently mislabels ~1/6
    # of the dataset.
    "20181117": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181118": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181127": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181128": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181204": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181205": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181208": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},
    "20181209": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-N(H)", 6: "Draw-O(H)"},

    # Room 1 digit-drawing set — different gesture *class set*, not a 6-way
    # overload but a 10-way one. Never pool with the six-gesture dates.
    "20181130": {1: "Draw-1", 2: "Draw-2", 3: "Draw-3", 4: "Draw-4",
                 5: "Draw-5", 6: "Draw-6", 7: "Draw-7", 8: "Draw-8",
                 9: "Draw-9", 10: "Draw-0"},

    # Room 3 — 5-way shape-drawing set, ids differ entirely.
    "20181211": {1: "Draw-N(V)", 2: "Draw-O(V)", 3: "Draw-Rectangle(H)",
                 4: "Draw-Triangle(H)", 5: "Draw-Zigzag(H)"},
}

UNVERIFIED_DATES = {
    # Dates present in the dataset for which we do not yet have a
    # confirmed gesture-id mapping. Loading these WITHOUT a mapping is
    # treated as an error by the label-consistency check.
    "20181112", "20181116",
}


def date_prefix(date_folder: str) -> str:
    """Strip trailing `-VS` (BVP folder convention) so we key by date only."""
    return date_folder[:8]


def canonical_name(date_folder: str, gesture_id: int) -> Optional[str]:
    """Return the canonical gesture name for (date, gesture_id) or None."""
    key = date_prefix(date_folder)
    if key not in GESTURE_MAP_BY_DATE:
        return None
    return GESTURE_MAP_BY_DATE[key].get(int(gesture_id))


def is_unverified(date_folder: str) -> bool:
    return date_prefix(date_folder) in UNVERIFIED_DATES
