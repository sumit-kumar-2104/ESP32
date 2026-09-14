"""Tests for structure-preserving tensor features + cache-key separation."""

from __future__ import annotations

import numpy as np
import pytest

from experiments import features, tensor_features as tf


@pytest.fixture
def records():
    recs, _ = features.synthetic_fixture(n_recordings=30, n_gestures=3)
    return recs


def test_tensor_modes_declare_public_set() -> None:
    assert set(tf.TENSOR_FEATURE_MODES) == {
        "csi_tensor", "dfs_tensor", "bvp_tensor",
    }
    for m in tf.TENSOR_FEATURE_MODES:
        assert tf.is_tensor_mode(m)
    assert not tf.is_tensor_mode("raw")
    assert not tf.is_tensor_mode("dfs_spec")


def test_csi_tensor_fixture_shape(records) -> None:
    cfg = tf.TensorConfig(T=16)
    _, bundle = tf.synthetic_tensor_fixture(
        "csi_tensor", n_recordings=20, config=cfg,
    )
    assert bundle.X.ndim == 4
    assert bundle.X.shape[1:] == (tf.CSI_TENSOR_ANTENNAS,
                                    tf.CSI_TENSOR_SUBCARRIERS, 16)
    assert bundle.feature_mode == "csi_tensor"


def test_dfs_tensor_fixture_shape() -> None:
    cfg = tf.TensorConfig(T=8, F=4)
    _, bundle = tf.synthetic_tensor_fixture(
        "dfs_tensor", n_recordings=20, config=cfg,
    )
    assert bundle.X.ndim == 4
    assert bundle.X.shape[1:] == (tf.DFS_TENSOR_RECEIVERS, 4, 8)


def test_bvp_tensor_fixture_shape_and_l1_normalisation() -> None:
    cfg = tf.TensorConfig(t0=12)
    _, bundle = tf.synthetic_tensor_fixture(
        "bvp_tensor", n_recordings=20, config=cfg,
    )
    assert bundle.X.ndim == 4
    assert bundle.X.shape[1:] == (tf.BVP_TENSOR_VX, tf.BVP_TENSOR_VY, 12)
    # BVP: each (Vx, Vy) snapshot must sum to 1 (or 0 for empty frames).
    sums = bundle.X.sum(axis=(1, 2))
    assert np.all((np.abs(sums - 1.0) < 1e-4) | (sums == 0.0))
    # Series length == t0.
    assert bundle.X.shape[-1] == cfg.t0


def test_bvp_snapshot_normalisation_helper() -> None:
    rng = np.random.default_rng(0)
    bvp = rng.random((tf.BVP_TENSOR_VX, tf.BVP_TENSOR_VY, 5)).astype(np.float32)
    out = tf._normalise_bvp_snapshots(bvp)
    sums = out.sum(axis=(0, 1))
    assert np.allclose(sums, 1.0, atol=1e-5)
    # Zero frames stay zero.
    bvp[..., 2] = 0.0
    out = tf._normalise_bvp_snapshots(bvp)
    assert np.all(out[..., 2] == 0.0)


def test_cache_key_separation_between_modes(records) -> None:
    """Tensor and flat caches must NOT collide, and different resolutions
    for the same mode must produce different ids."""
    ids = [r.sample_id for r in records]
    a = tf.tensor_cache_id("csi_tensor", tf.TensorConfig(T=32), ids)
    b = tf.tensor_cache_id("dfs_tensor", tf.TensorConfig(T=32), ids)
    c = tf.tensor_cache_id("csi_tensor", tf.TensorConfig(T=64), ids)
    d = tf.tensor_cache_id("csi_tensor", tf.TensorConfig(T=32), ids)
    assert a != b, "different feature modes must produce different cache ids"
    assert a != c, "different T must produce different cache ids"
    assert a == d, "identical config on identical sample_ids must match"
    # Modes are visibly prefixed so a human can inspect them.
    assert a.startswith("csi_tensor:")
    assert b.startswith("dfs_tensor:")


def test_bvp_unavailable_when_source_missing(monkeypatch, tmp_path, records) -> None:
    """If the BVP tree is not present, the builder returns an
    UnavailableFeatureBundle sentinel — never a fabricated tensor."""
    monkeypatch.setenv("METAAI_BVP_DIR", str(tmp_path / "does_not_exist"))
    bundle = tf.build_tensor_features(
        records, "bvp_tensor", tf.TensorConfig(t0=8),
    )
    assert isinstance(bundle, tf.UnavailableFeatureBundle)
    assert bundle.feature_mode == "bvp_tensor"
    assert "not present" in bundle.reason


def test_resample_time_broadcasts_single_frame() -> None:
    x = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    out = tf._resample_time(x, axis=2, target=5)
    assert out.shape == (3, 3, 5)
    # Single-frame broadcast must replicate the same value along the axis.
    assert np.allclose(out, np.broadcast_to(x, (3, 3, 5)))


def test_resample_time_handles_zero_length() -> None:
    x = np.zeros((2, 4, 0), dtype=np.float32)
    out = tf._resample_time(x, axis=2, target=6)
    assert out.shape == (2, 4, 6)
    assert np.all(out == 0.0)
