"""Tests for experiments.manifest — cache-id keying, quality policy."""

from __future__ import annotations

from experiments import features, manifest


def test_cache_id_differs_by_feature_mode() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=10, n_gestures=2)
    ids = [r.sample_id for r in recs]
    raw = manifest.compute_cache_id("raw", "full", ids)
    dfs = manifest.compute_cache_id("dfs_spec", "full", ids)
    assert raw != dfs, "raw and dfs_spec must never collide in the feature cache"


def test_cache_id_stable_for_permutation() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=8, n_gestures=2)
    ids = [r.sample_id for r in recs]
    assert (
        manifest.compute_cache_id("raw", "full", ids)
        == manifest.compute_cache_id("raw", "full", list(reversed(ids)))
    )


def test_cache_id_differs_by_sample_set() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=8, n_gestures=2)
    ids = [r.sample_id for r in recs]
    a = manifest.compute_cache_id("raw", "full", ids)
    b = manifest.compute_cache_id("raw", "full", ids[:-1])
    assert a != b


def test_quality_policy_reject() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=6, n_gestures=2)
    # Poison one record.
    bad = manifest.RecordingRecord(
        **{**recs[0].__dict__,
           "sample_id": "bad_id",
           "recording_id": "bad-recording",
           "quality": manifest.QUALITY_ZERO_PACKETS,
           "quality_detail": "packet_counts=0",
        }
    )
    kept, rej = manifest.apply_quality_policy(list(recs) + [bad], policy="reject")
    assert bad.sample_id not in {r.sample_id for r in kept}
    assert any(row["sample_id"] == "bad_id" for row in rej)


def test_quality_policy_retain_with_mask_records_action() -> None:
    recs, _ = features.synthetic_fixture(n_recordings=4, n_gestures=2)
    bad = manifest.RecordingRecord(
        **{**recs[0].__dict__,
           "sample_id": "bad_id",
           "recording_id": "bad-recording",
           "quality": manifest.QUALITY_MISSING_RECEIVERS,
           "quality_detail": "missing_rx=[3]",
        }
    )
    kept, rej = manifest.apply_quality_policy(
        list(recs) + [bad], policy="retain_with_mask",
    )
    assert "bad_id" in {r.sample_id for r in kept}
    actions = [row["action"] for row in rej]
    assert "retain_with_mask" in actions


def test_write_manifest_roundtrip(tmp_path) -> None:
    recs, _ = features.synthetic_fixture(n_recordings=5, n_gestures=2)
    kept, rej = manifest.apply_quality_policy(recs, policy="reject")
    summary = manifest.write_manifest(
        tmp_path, kept, rej, feature_mode="raw", dfs_bins="full",
    )
    assert summary["n_recordings"] == len(kept)
    reloaded = manifest.load_manifest(tmp_path / "manifest.jsonl")
    assert len(reloaded) == len(kept)
    assert reloaded[0].sample_id == kept[0].sample_id
