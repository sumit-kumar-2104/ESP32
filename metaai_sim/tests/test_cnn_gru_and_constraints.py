"""Tests for the CNN-GRU backbone and composable OTA constraint wrappers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.cnn_gru import CNNGRUBackbone, CNNGRUConfig
from models.constraints import (
    ConstrainedHead, ConstraintConfig, apply_constraints_to_cnn_gru,
)


def _fixed_seed():
    torch.manual_seed(0)


def test_cnn_gru_accepts_csi_tensor_shape() -> None:
    _fixed_seed()
    m = CNNGRUBackbone(
        input_shape=(18, 30, 8), n_classes=6, rep_kind="csi_tensor",
        config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
    )
    x = torch.randn(3, 18, 30, 8)
    out = m(x)
    assert out.shape == (3, 6)


def test_cnn_gru_accepts_dfs_tensor_shape() -> None:
    _fixed_seed()
    m = CNNGRUBackbone(
        input_shape=(6, 4, 8), n_classes=6, rep_kind="dfs_tensor",
        config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
    )
    x = torch.randn(2, 6, 4, 8)
    out = m(x)
    assert out.shape == (2, 6)


def test_cnn_gru_accepts_bvp_tensor_shape() -> None:
    _fixed_seed()
    m = CNNGRUBackbone(
        input_shape=(20, 20, 5), n_classes=6, rep_kind="bvp_tensor",
        config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
    )
    x = torch.randn(2, 20, 20, 5)
    out = m(x)
    assert out.shape == (2, 6)


def test_cnn_gru_rejects_bad_rep_kind() -> None:
    with pytest.raises(ValueError, match="cnn_gru does not support"):
        CNNGRUBackbone(
            input_shape=(4, 4, 4), n_classes=6, rep_kind="flat",
        )


@pytest.mark.parametrize("label,checker", [
    ("unconstrained", lambda c: (not c.complex_weighting) and (not c.magnitude_readout) and (not c.quantize_2bit)),
    ("complex_only", lambda c: c.complex_weighting and (not c.magnitude_readout) and (not c.quantize_2bit)),
    ("magnitude_only", lambda c: (not c.complex_weighting) and c.magnitude_readout and (not c.quantize_2bit)),
    ("quantize_only", lambda c: (not c.complex_weighting) and (not c.magnitude_readout) and c.quantize_2bit),
    ("full_ota", lambda c: c.complex_weighting and c.magnitude_readout and c.quantize_2bit),
])
def test_constraint_config_factories_compose(label, checker) -> None:
    cfg = getattr(ConstraintConfig, label)()
    assert cfg.label == label
    assert checker(cfg)


def test_constrained_head_composes_on_cnn_gru() -> None:
    """Every constraint variant swaps into the CNN-GRU head cleanly and
    produces a valid (batch, n_classes) logit tensor."""
    for label in [
        "unconstrained", "complex_only", "magnitude_only",
        "quantize_only", "full_ota", "r1", "r2",
    ]:
        _fixed_seed()
        m = CNNGRUBackbone(
            input_shape=(6, 4, 8), n_classes=6, rep_kind="dfs_tensor",
            config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
        )
        cfg = getattr(ConstraintConfig, label)()
        apply_constraints_to_cnn_gru(m, cfg)
        x = torch.randn(2, 6, 4, 8)
        out = m(x)
        assert out.shape == (2, 6), (
            f"{cfg.label} produced wrong logits shape: {out.shape}"
        )
        assert torch.isfinite(out).all(), f"{cfg.label} produced non-finite logits"


def test_constrained_head_gradients_flow() -> None:
    """The 2-bit STE must let gradients reach the underlying real+imag
    parameters even when the forward path is quantized."""
    _fixed_seed()
    m = CNNGRUBackbone(
        input_shape=(6, 4, 8), n_classes=6, rep_kind="dfs_tensor",
        config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
    )
    apply_constraints_to_cnn_gru(m, ConstraintConfig.full_ota())
    x = torch.randn(4, 6, 4, 8)
    y = torch.tensor([0, 1, 2, 3])
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    head = m.head[-1]
    assert isinstance(head, ConstrainedHead)
    assert head.weight_real.grad is not None
    assert head.weight_imag.grad is not None
    # At least one parameter got a nonzero grad.
    assert (head.weight_real.grad.abs().sum() + head.weight_imag.grad.abs().sum()) > 0


def test_magnitude_only_still_uses_real_weights() -> None:
    """Magnitude readout without complex weighting means |Re + 0j|
    — the imag parameter stays frozen at zero."""
    _fixed_seed()
    m = CNNGRUBackbone(
        input_shape=(6, 4, 8), n_classes=6, rep_kind="dfs_tensor",
        config=CNNGRUConfig(n_filters=4, dense_dim=8, gru_dim=16, dropout=0.0),
    )
    apply_constraints_to_cnn_gru(m, ConstraintConfig.magnitude_only())
    head = m.head[-1]
    assert not head.weight_imag.requires_grad
    assert torch.all(head.weight_imag == 0.0)
