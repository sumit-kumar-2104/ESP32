"""Data manifest builder for Widar3.0 raw CSI.

Walks the raw CSI tree, parses the canonical filename
``user{u}-{g}-{loc}-{ori}-{rep}-r{rx}.dat``, hashes every source file, and
records complete per-recording metadata + quality flags. The output is an
auditable JSONL manifest plus a per-run summary and rejection log.

Key invariants (enforced here, not by callers):

- **Recording identity** = ``(date, user, gesture, loc, ori, rep)``. This is
  the smallest unit that MUST NOT cross a train/val/test partition. It is
  DISTINCT from ``group_id`` (used by ``csi_loader.build_csi_features``,
  which combines ``user-loc-ori-room-date``) and DISTINCT from
  ``session_id`` (= ``date``). We record all three so splits can pick the
  right key.
- **No silent interpolation.** Malformed/empty/short recordings are recorded
  with an explicit quality flag; the split module decides whether to drop
  them per the configured policy.
- **Cache identity.** The manifest's ``cache_id`` is a hash of
  ``(feature_mode, preprocessing_version, sorted sample_ids)``. Raw CSI and
  DFS features MUST use different cache_ids so they cannot overwrite each
  other in ``dumps/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Kept in sync with data/csi_loader.py — imported lazily below so this
# module stays importable even when csiread is missing.
_FNAME_RE = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)$")

DATE_ROOM = {
    "20181109": 1, "20181112": 1, "20181115": 1, "20181116": 1,
    "20181121": 1, "20181130": 1,
    "20181117": 2, "20181118": 2, "20181127": 2, "20181128": 2,
    "20181204": 2, "20181205": 2, "20181208": 2, "20181209": 2,
    "20181211": 3,
}

# Canonical gestures — validated in ``check_gesture_map``. Matches the
# stage-4 preprocessing spec in README.md (push/sweep/clap/slide/draw-O/draw-Z).
CANONICAL_GESTURES = {
    1: "push", 2: "sweep", 3: "clap",
    4: "slide", 5: "draw-O", 6: "draw-Z",
}

QUALITY_OK = "ok"
QUALITY_MISSING_RECEIVERS = "missing_receivers"
QUALITY_MALFORMED = "malformed"
QUALITY_ZERO_PACKETS = "zero_packets"
QUALITY_SHORT = "short"

REJECT_POLICIES = ("reject", "retain_with_mask", "keep_as_is")
PREPROCESSING_VERSION = "v1"


@dataclass(frozen=True)
class RecordingRecord:
    sample_id: str            # stable content-derived hash
    recording_id: str         # date-user-gesture-loc-ori-rep
    group_id: str             # user-loc-ori-room-date  (csi_loader convention)
    session_id: str           # = date (recording session)
    date: str
    room: int
    user: int
    gesture: int              # 1-based, canonical
    gesture_name: str
    location: int
    orientation: int
    repetition: int
    receivers_present: list[int]
    receivers_missing: list[int]
    file_paths: dict[str, str]     # "r1".."r6" -> abs path
    file_hashes: dict[str, str]    # sha256 short (first 16 hex)
    file_sizes: dict[str, int]     # bytes
    packet_counts: dict[str, int]  # after read; -1 if unread
    quality: str
    quality_detail: str


def _sha256_short(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """Return (16-hex sha256 prefix, size in bytes)."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            size += len(b)
            h.update(b)
    return h.hexdigest()[:16], size


def _stable_sample_id(recording_id: str, hashes: dict[str, str]) -> str:
    """Deterministic sample_id derived from the recording key + file hashes.

    Using content hashes means moving/renaming files does not affect the
    ID, but a re-recording (different bytes) will produce a different ID
    and cannot silently masquerade as the same sample.
    """
    concat = recording_id + "|" + "|".join(
        f"{rx}={hashes[rx]}" for rx in sorted(hashes)
    )
    return hashlib.sha256(concat.encode("utf-8")).hexdigest()[:16]


def _read_packet_count(path: Path) -> tuple[int, str | None]:
    """Return (packets, error). Uses csiread if available, else -1."""
    try:
        import csiread  # type: ignore
    except ImportError:
        return -1, None
    try:
        r = csiread.Intel(str(path))
        r.read()
        csi = r.get_scaled_csi()
    except Exception as e:  # noqa: BLE001 — the reader raises many types
        return -1, f"csiread: {type(e).__name__}: {e}"
    if csi is None:
        return 0, "csiread returned None"
    if getattr(csi, "ndim", 0) != 4:
        return 0, f"unexpected shape {getattr(csi, 'shape', None)!r}"
    return int(csi.shape[0]), None


def _classify(
    receivers_present: list[int],
    packet_counts: dict[str, int],
    errors: dict[str, str],
    *,
    min_packets: int,
) -> tuple[str, str]:
    """Return (quality, detail)."""
    all_rx = set(range(1, 7))
    missing = sorted(all_rx - set(receivers_present))
    if errors:
        joined = "; ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
        return QUALITY_MALFORMED, joined
    if missing:
        return QUALITY_MISSING_RECEIVERS, f"missing_rx={missing}"
    counts = [c for c in packet_counts.values() if c >= 0]
    if counts and min(counts) == 0:
        return QUALITY_ZERO_PACKETS, f"packet_counts={packet_counts}"
    if counts and min(counts) < min_packets:
        return QUALITY_SHORT, f"min_packets={min(counts)} < {min_packets}"
    return QUALITY_OK, "ok"


def walk_raw_csi(
    csi_root: Path,
    dates: Iterable[str],
    *,
    read_packet_counts: bool = True,
    min_packets: int = 64,
) -> list[RecordingRecord]:
    """Walk one or more date folders and produce a manifest.

    ``read_packet_counts`` toggles the csiread parse (slow but the only
    way to detect zero-packet or malformed .dat files reliably). Turn off
    for cheap dry-runs.
    """
    csi_root = Path(csi_root)
    if not csi_root.exists():
        raise FileNotFoundError(f"raw CSI root does not exist: {csi_root}")

    # Bucket file paths by recording key.
    buckets: dict[tuple[str, int, int, int, int, int], dict[int, Path]] = {}
    for date in dates:
        date_dir = csi_root / date
        if not date_dir.exists():
            continue
        for root, _, files in os.walk(date_dir):
            for fn in files:
                if not fn.endswith(".dat"):
                    continue
                m = _FNAME_RE.match(fn[:-4])
                if not m:
                    continue
                uid, ges, loc, ori, rep, rx = (int(m.group(i)) for i in range(1, 7))
                if not (1 <= rx <= 6):
                    continue
                key = (date, uid, ges, loc, ori, rep)
                buckets.setdefault(key, {})[rx] = Path(root) / fn

    records: list[RecordingRecord] = []
    for key, rxmap in sorted(buckets.items()):
        date, uid, ges, loc, ori, rep = key
        recording_id = f"{date}-u{uid}-g{ges}-l{loc}-o{ori}-r{rep}"
        room = DATE_ROOM.get(date[:8], 0)
        hashes: dict[str, str] = {}
        sizes: dict[str, int] = {}
        packets: dict[str, int] = {}
        errors: dict[str, str] = {}
        receivers_present = sorted(rxmap.keys())
        for rx, p in sorted(rxmap.items()):
            rx_key = f"r{rx}"
            try:
                h, size = _sha256_short(p)
            except OSError as e:
                errors[rx_key] = f"io: {e}"
                continue
            hashes[rx_key] = h
            sizes[rx_key] = size
            if read_packet_counts:
                n_pkt, err = _read_packet_count(p)
                packets[rx_key] = n_pkt
                if err:
                    errors[rx_key] = err
            else:
                packets[rx_key] = -1
        quality, detail = _classify(
            receivers_present, packets, errors, min_packets=min_packets,
        )
        rec = RecordingRecord(
            sample_id=_stable_sample_id(recording_id, hashes),
            recording_id=recording_id,
            group_id=f"{uid}-{loc}-{ori}-{room}-{date}",
            session_id=date,
            date=date,
            room=room,
            user=uid,
            gesture=ges,
            gesture_name=CANONICAL_GESTURES.get(ges, "unknown"),
            location=loc,
            orientation=ori,
            repetition=rep,
            receivers_present=receivers_present,
            receivers_missing=sorted(set(range(1, 7)) - set(receivers_present)),
            file_paths={k: str(rxmap[int(k[1:])]) for k in hashes},
            file_hashes=hashes,
            file_sizes=sizes,
            packet_counts=packets,
            quality=quality,
            quality_detail=detail,
        )
        records.append(rec)
    return records


def check_gesture_map(records: list[RecordingRecord]) -> dict[str, Any]:
    """Verify observed gestures are in :data:`CANONICAL_GESTURES` and are
    consistent across dates.

    Returns a summary dict; caller decides whether to reject or warn.
    """
    per_date: dict[str, set[int]] = {}
    unknowns: list[str] = []
    for r in records:
        per_date.setdefault(r.date, set()).add(r.gesture)
        if r.gesture not in CANONICAL_GESTURES:
            unknowns.append(f"{r.recording_id} gesture={r.gesture}")
    all_gestures = sorted(set().union(*per_date.values())) if per_date else []
    common = sorted(set.intersection(*per_date.values())) if per_date else []
    return {
        "gestures_seen": all_gestures,
        "gestures_common_across_dates": common,
        "gestures_per_date": {d: sorted(g) for d, g in per_date.items()},
        "unknown_gesture_ids": unknowns,
    }


def apply_quality_policy(
    records: list[RecordingRecord],
    policy: str,
) -> tuple[list[RecordingRecord], list[dict[str, Any]]]:
    """Return (kept, rejections). ``rejections`` is a list of audit rows.

    - ``reject``: drop anything not ``QUALITY_OK``.
    - ``retain_with_mask``: keep, but the split/loader must respect
      ``quality`` and mask missing receivers.
    - ``keep_as_is``: keep everything and record the flag only.
    """
    if policy not in REJECT_POLICIES:
        raise ValueError(f"policy={policy!r} not in {REJECT_POLICIES}")
    kept: list[RecordingRecord] = []
    rejections: list[dict[str, Any]] = []
    for r in records:
        if r.quality == QUALITY_OK or policy != "reject":
            kept.append(r)
            if r.quality != QUALITY_OK and policy == "retain_with_mask":
                rejections.append({
                    "sample_id": r.sample_id,
                    "recording_id": r.recording_id,
                    "action": "retain_with_mask",
                    "quality": r.quality,
                    "detail": r.quality_detail,
                })
        else:
            rejections.append({
                "sample_id": r.sample_id,
                "recording_id": r.recording_id,
                "action": "reject",
                "quality": r.quality,
                "detail": r.quality_detail,
            })
    return kept, rejections


def compute_cache_id(
    feature_mode: str,
    dfs_bins: str,
    sample_ids: Iterable[str],
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> str:
    """Return a hash uniquely identifying (feature_mode, config, sample set)."""
    h = hashlib.sha256()
    h.update(feature_mode.encode("utf-8"))
    h.update(b"|")
    h.update((dfs_bins or "").encode("utf-8"))
    h.update(b"|")
    h.update(preprocessing_version.encode("utf-8"))
    h.update(b"|")
    for sid in sorted(sample_ids):
        h.update(sid.encode("utf-8"))
        h.update(b",")
    return h.hexdigest()[:16]


def write_manifest(
    audit_dir: Path,
    records: list[RecordingRecord],
    rejections: list[dict[str, Any]],
    *,
    feature_mode: str,
    dfs_bins: str = "full",
) -> dict[str, Any]:
    """Persist manifest + rejections + summary. Returns the summary dict."""
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_dir / "manifest.jsonl"
    rejections_path = audit_dir / "rejections.jsonl"
    summary_path = audit_dir / "summary.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True))
            f.write("\n")
    with rejections_path.open("w", encoding="utf-8") as f:
        for row in rejections:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")

    cache_id = compute_cache_id(
        feature_mode, dfs_bins, [r.sample_id for r in records]
    )
    quality_counter: dict[str, int] = {}
    for r in records:
        quality_counter[r.quality] = quality_counter.get(r.quality, 0) + 1

    gesture_check = check_gesture_map(records)
    summary = {
        "n_recordings": len(records),
        "n_rejections": len(rejections),
        "feature_mode": feature_mode,
        "dfs_bins": dfs_bins,
        "preprocessing_version": PREPROCESSING_VERSION,
        "cache_id": cache_id,
        "quality_counts": quality_counter,
        "gestures": gesture_check,
        "rooms_seen": sorted({r.room for r in records}),
        "dates_seen": sorted({r.date for r in records}),
        "users_seen": sorted({r.user for r in records}),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_manifest(manifest_path: Path) -> list[RecordingRecord]:
    """Read back a manifest.jsonl produced by :func:`write_manifest`."""
    out: list[RecordingRecord] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(RecordingRecord(**d))
    return out
