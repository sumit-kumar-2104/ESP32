"""Regression tests for zero-byte .dat handling in the manifest.

The server's first core run turned up three receiver files with the SHA
prefix e3b0c44298fc1c14 across different dates and rooms. That is the
empty-file hash, meaning the underlying files were 0 bytes on disk. The
harness now:

- flags such recordings as ``QUALITY_ZERO_BYTES``,
- keeps their size in ``file_sizes`` but drops the empty hash from
  ``file_hashes`` so it can never fabricate a cross-recording collision,
- ignores that specific hash in the split leakage check as belt-and-braces,
- calls it out separately in ``diagnose_hash_duplicates``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import manifest, splits, features


def _make_zero_byte_dat(tmp_path: Path, date: str, name: str) -> Path:
    date_dir = tmp_path / date
    date_dir.mkdir(parents=True, exist_ok=True)
    p = date_dir / name
    p.write_bytes(b"")
    return p


def _make_dummy_dat(tmp_path: Path, date: str, name: str, content: bytes) -> Path:
    date_dir = tmp_path / date
    date_dir.mkdir(parents=True, exist_ok=True)
    p = date_dir / name
    p.write_bytes(content)
    return p


def test_zero_byte_receiver_flagged(tmp_path: Path) -> None:
    # One recording, six receivers — r1 is empty, r2..r6 are non-empty.
    _make_zero_byte_dat(tmp_path, "20181109", "user1-1-1-1-1-r1.dat")
    for rx in range(2, 7):
        _make_dummy_dat(
            tmp_path, "20181109",
            f"user1-1-1-1-1-r{rx}.dat", f"payload-{rx}".encode(),
        )
    recs = manifest.walk_raw_csi(
        tmp_path, ["20181109"], read_packet_counts=False,
    )
    assert len(recs) == 1
    r = recs[0]
    assert r.quality == manifest.QUALITY_ZERO_BYTES
    assert "r1" in r.quality_detail
    # Empty hash must NOT enter file_hashes (belt: split checker sees only
    # the 5 real ones and never a spurious collision).
    assert "r1" not in r.file_hashes
    assert manifest.EMPTY_FILE_HASH_16 not in r.file_hashes.values()
    # Size is still recorded so audits can see the actual zero.
    assert r.file_sizes.get("r1") == 0


def test_reject_policy_drops_zero_byte_recordings(tmp_path: Path) -> None:
    for rx in range(1, 7):
        _make_zero_byte_dat(tmp_path, "20181109", f"user1-1-1-1-1-r{rx}.dat")
    for rx in range(1, 7):
        _make_dummy_dat(
            tmp_path, "20181109",
            f"user1-2-1-1-1-r{rx}.dat", f"payload-{rx}".encode(),
        )
    recs = manifest.walk_raw_csi(
        tmp_path, ["20181109"], read_packet_counts=False,
    )
    kept, rej = manifest.apply_quality_policy(recs, policy="reject")
    kept_ids = {r.recording_id for r in kept}
    assert "20181109-u1-g1-l1-o1-r1" not in kept_ids, "empty recording must be dropped"
    assert "20181109-u1-g2-l1-o1-r1" in kept_ids
    assert any("zero" in row.get("quality", "") for row in rej)


def test_split_ignores_empty_file_hash_belt_and_braces() -> None:
    # Two synthetic recordings on different dates. Poison BOTH to advertise
    # the empty-file hash on r1. The manifest layer would normally have
    # dropped these, but the split module must still refuse to flag them
    # as leakage if they somehow slip through.
    recs, _ = features.synthetic_fixture(n_recordings=20, n_gestures=2)
    poisoned = []
    for r in recs:
        hashes = dict(r.file_hashes)
        hashes["r1"] = manifest.EMPTY_FILE_HASH_16
        poisoned.append(manifest.RecordingRecord(**{**r.__dict__, "file_hashes": hashes}))
    # cross_room previously raised InvalidSplitError on file-hash overlap.
    # After the fix the empty-file SHA is ignored and the split builds.
    split = splits.cross_room(poisoned, source_room=1, target_room=2, seed=42)
    assert split.checks["content_hash_overlap"]["train_test_file_hashes"] == 0


def test_walker_still_hashes_real_content(tmp_path: Path) -> None:
    for rx in range(1, 7):
        _make_dummy_dat(
            tmp_path, "20181109",
            f"user1-1-1-1-1-r{rx}.dat", f"payload-{rx}".encode(),
        )
    recs = manifest.walk_raw_csi(
        tmp_path, ["20181109"], read_packet_counts=False,
    )
    assert len(recs) == 1
    r = recs[0]
    assert r.quality == manifest.QUALITY_OK
    assert len(r.file_hashes) == 6
    assert manifest.EMPTY_FILE_HASH_16 not in r.file_hashes.values()
