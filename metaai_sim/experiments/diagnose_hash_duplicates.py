"""Diagnose duplicate .dat files across dates / rooms in a run's data manifest.

Reads ``results/<run-id>/data_audit/manifest.jsonl`` and prints every
receiver-file content hash that appears in more than one recording, with
special attention to duplicates that cross a date or room boundary — the
exact leakage source that trips ``splits.py`` file-hash overlap check.

Usage:
    python -m experiments.diagnose_hash_duplicates <run-id-or-manifest-path>

If a bare run-id is given, we look under
``metaai_sim/results/<run-id>/data_audit/manifest.jsonl``. The output is
plain text; nothing is auto-fixed. The operator decides whether the
duplicated files should be removed from disk or the split configuration
should quarantine them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import paths as paths_mod
from experiments.manifest import EMPTY_FILE_HASH_16


def _resolve_manifest(arg: str) -> Path:
    """Accept a run-id under results/ or a direct path to manifest.jsonl."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir() and (p / "manifest.jsonl").exists():
        return p / "manifest.jsonl"
    layout = paths_mod.build_run_layout(arg, paths_mod.DEFAULT_RESULTS_ROOT)
    candidate = layout.data_audit / "manifest.jsonl"
    if candidate.exists():
        return candidate
    raise SystemExit(f"could not locate manifest.jsonl for {arg!r}")


def diagnose(manifest_path: Path) -> dict:
    """Return a report dict; caller may also print it."""
    hash_to_owners: dict[str, list[dict]] = defaultdict(list)
    zero_byte_files: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            file_sizes = rec.get("file_sizes") or {}
            for rx_key, h in (rec.get("file_hashes") or {}).items():
                owner = {
                    "hash": h,
                    "rx": rx_key,
                    "date": rec.get("date"),
                    "room": rec.get("room"),
                    "recording_id": rec.get("recording_id"),
                    "file_path": (rec.get("file_paths") or {}).get(rx_key),
                    "size": file_sizes.get(rx_key),
                }
                if h == EMPTY_FILE_HASH_16 or owner["size"] == 0:
                    zero_byte_files.append(owner)
                    continue
                hash_to_owners[h].append(owner)
    duplicates = {h: owners for h, owners in hash_to_owners.items() if len(owners) > 1}
    cross_date: dict[str, list[dict]] = {}
    cross_room: dict[str, list[dict]] = {}
    same_recording: dict[str, list[dict]] = {}
    for h, owners in duplicates.items():
        dates = {o["date"] for o in owners}
        rooms = {o["room"] for o in owners}
        recs = {o["recording_id"] for o in owners}
        if len(dates) > 1:
            cross_date[h] = owners
        if len(rooms) > 1:
            cross_room[h] = owners
        if len(recs) == 1 and len(owners) > 1:
            same_recording[h] = owners
    return {
        "manifest_path": str(manifest_path),
        "n_recordings": sum(1 for _ in manifest_path.open("r", encoding="utf-8")),
        "n_unique_file_hashes": len(hash_to_owners),
        "n_duplicated_hashes": len(duplicates),
        "n_cross_date": len(cross_date),
        "n_cross_room": len(cross_room),
        "n_same_recording_extra_rx": len(same_recording),
        "n_zero_byte_files": len(zero_byte_files),
        "zero_byte_files": zero_byte_files,
        "cross_date": cross_date,
        "cross_room": cross_room,
        "same_recording_extra_rx": same_recording,
    }


def print_report(report: dict, *, max_rows: int = 20) -> None:
    print(f"manifest:                  {report['manifest_path']}")
    print(f"recordings scanned:        {report['n_recordings']}")
    print(f"unique file hashes:        {report['n_unique_file_hashes']}")
    print(f"duplicated file hashes:    {report['n_duplicated_hashes']}")
    print(f"  cross-date duplicates:   {report['n_cross_date']}")
    print(f"  cross-room duplicates:   {report['n_cross_room']}")
    print(f"  same-recording extras:   {report['n_same_recording_extra_rx']}")
    print(f"zero-byte receiver files:  {report['n_zero_byte_files']}"
          f"  (excluded from hash collision counts)")
    print()
    if report["zero_byte_files"]:
        print("=== ZERO-BYTE receiver files (data-quality, not leakage) ===")
        for i, o in enumerate(report["zero_byte_files"][:max_rows]):
            print(f"  date={o['date']} room={o['room']} rec={o['recording_id']} "
                  f"rx={o['rx']} size={o['size']}")
            print(f"    {o['file_path']}")
        remaining = len(report["zero_byte_files"]) - max_rows
        if remaining > 0:
            print(f"  ... {remaining} more (pass --max-rows to see them)")
        print()
    for label, key in (
        ("CROSS-DATE duplicates (leakage risk)", "cross_date"),
        ("CROSS-ROOM duplicates (leakage risk)", "cross_room"),
        ("Same-recording receiver duplicates (usually benign)", "same_recording_extra_rx"),
    ):
        bucket = report[key]
        if not bucket:
            continue
        print(f"=== {label} ===")
        shown = 0
        for h, owners in bucket.items():
            print(f"hash {h}:")
            for o in owners:
                print(f"  date={o['date']} room={o['room']} "
                      f"rec={o['recording_id']} rx={o['rx']}")
                print(f"    {o['file_path']}")
            shown += 1
            if shown >= max_rows:
                print(f"  ... {len(bucket) - shown} more (pass --max-rows to see them)")
                break
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="run-id under results/ or a direct manifest.jsonl path")
    ap.add_argument("--max-rows", type=int, default=20,
                    help="limit per-category hash groups printed (default 20)")
    ap.add_argument("--json", action="store_true",
                    help="emit the raw report as JSON instead of the text summary")
    args = ap.parse_args(argv)
    manifest_path = _resolve_manifest(args.target)
    report = diagnose(manifest_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
