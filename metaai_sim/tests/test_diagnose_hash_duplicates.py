"""Cover experiments.diagnose_hash_duplicates on a synthetic manifest.

The synthetic fixture has no genuine hash collisions, so the diagnostic
must report zero duplicates. We also inject a deliberate cross-date
collision to prove the reporter finds it.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments import diagnose_hash_duplicates as d, features, manifest as m


def _write_manifest(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, sort_keys=True))
            f.write("\n")


def test_no_duplicates_on_clean_fixture(tmp_path: Path) -> None:
    recs, _ = features.synthetic_fixture(n_recordings=20, n_gestures=2)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, recs)
    report = d.diagnose(manifest_path)
    assert report["n_duplicated_hashes"] == 0
    assert report["n_cross_date"] == 0
    assert report["n_cross_room"] == 0


def test_cross_date_collision_is_reported(tmp_path: Path) -> None:
    recs, _ = features.synthetic_fixture(n_recordings=10, n_gestures=2)
    # recs[0] is in 20181109 (room 1). Craft a second record in 20181118
    # that reuses r1..r6 hashes from recs[0] — i.e. the same bytes present
    # in two dates. This is exactly the cross-date leakage the split
    # module already refuses; the diagnostic must point it out too.
    poison = m.RecordingRecord(
        **{**recs[0].__dict__,
           "sample_id": "poison_1",
           "recording_id": "poison-rec-1",
           "date": "20181118",
           "room": 2,
        }
    )
    _write_manifest(tmp_path / "manifest.jsonl", list(recs) + [poison])
    report = d.diagnose(tmp_path / "manifest.jsonl")
    assert report["n_cross_date"] == 6, "every receiver hash is shared across dates"
    assert report["n_cross_room"] == 6
