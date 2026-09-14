"""OTA physical-constraint wrappers, composable with any backbone.

The three constraints from the paper's OTA path — complex-valued
weighting, magnitude readout, and 2-bit phase quantization with a
straight-through estimator — are implemented here as tiny standalone
modules and one composable head that swaps into the CNN-GRU backbone's
classifier layer.

Composability rules:

- ``complex_weighting=True``: the head's linear map ``h -> logits`` is a
  complex matmul. Real ``h`` is lifted to complex via zero imag; the
  layer keeps a real+imag parameter pair.
- ``magnitude_readout=True``: the head applies ``|.|`` (magnitude) after
  the linear map. If ``complex_weighting=False``, the layer synthesises
  a zero imaginary channel so ``|Re + 0j| = |Re|`` — still ``|.|``, just
  degenerate. This makes each switch independently informative in the
  ablation table without coupling them.
- ``quantize_2bit=True``: the effective weight matrix in the complex
  head is passed through the paper's 2-bit phase quantizer with a
  straight-through gradient. Combined with ``complex_weighting=False``
  the quantizer applies to a synthesised complex pair whose real part
  IS the trained real weight, so gradients still flow.

R1 = complex + magnitude + quantize + regularization tweaks
(weight-decay, dropout, complex-dim bottleneck).
R2 = complex + magnitude + quantize + STE gradient scaling.

Both are configured entirely via :class:`ConstraintConfig` — the runner
never touches the module internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


PHASE_STATES = [0.0, float(np.pi / 2), float(np.pi), float(3 * np.pi / 2)]


@dataclass
class ConstraintConfig:
    complex_weighting: bool = False
    magnitude_readout: bool = False
    quantize_2bit: bool = False
    # R1 knobs.
    weight_decay: float = 0.0
    input_dropout: float = 0.0
    complex_dim: int | None = None
    # R2 knobs.
    qgrad_scale: float = 1.0
    ste_kind: str = "identity"           # "identity" | "hardtanh"
    label: str = "unconstrained"

    @classmethod
    def unconstrained(cls) -> "ConstraintConfig":
        return cls(label="unconstrained")

    @classmethod
    def complex_only(cls) -> "ConstraintConfig":
        return cls(complex_weighting=True, label="complex_only")

    @classmethod
    def magnitude_only(cls) -> "ConstraintConfig":
        return cls(magnitude_readout=True, label="magnitude_only")

    @classmethod
    def quantize_only(cls) -> "ConstraintConfig":
        return cls(quantize_2bit=True, label="quantize_only")

    @classmethod
    def full_ota(cls) -> "ConstraintConfig":
        return cls(
            complex_weighting=True, magnitude_readout=True,
            quantize_2bit=True, label="full_ota",
        )

    @classmethod
    def r1(cls) -> "ConstraintConfig":
        return cls(
            complex_weighting=True, magnitude_readout=True,
            quantize_2bit=True,
            weight_decay=1e-3, input_dropout=0.4, complex_dim=None,
            label="full_ota_R1",
        )

    @classmethod
    def r2(cls) -> "ConstraintConfig":
        return cls(
            complex_weighting=True, magnitude_readout=True,
            quantize_2bit=True,
            qgrad_scale=8.0, ste_kind="hardtanh",
            label="full_ota_R2",
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "complex_weighting": bool(self.complex_weighting),
            "magnitude_readout": bool(self.magnitude_readout),
            "quantize_2bit": bool(self.quantize_2bit),
            "weight_decay": float(self.weight_decay),
            "input_dropout": float(self.input_dropout),
            "complex_dim": self.complex_dim,
            "qgrad_scale": float(self.qgrad_scale),
            "ste_kind": self.ste_kind,
        }


class _PhaseQuantizeSTE(torch.autograd.Function):
    """Forward = nearest 2-bit phasor. Backward = clamped identity times scale."""

    @staticmethod
    def forward(ctx, w_real, w_imag, phase_states, qgrad_scale, ste_kind):
        w_complex = torch.complex(w_real, w_imag)
        angles = torch.angle(w_complex)
        diff = angles.unsqueeze(-1) - phase_states.view(
            *([1] * angles.dim()), -1
        )
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        idx = torch.argmin(torch.abs(diff), dim=-1)
        chosen = phase_states[idx]
        q_real = torch.cos(chosen)
        q_imag = torch.sin(chosen)
        ctx.save_for_backward(w_real, w_imag)
        ctx.qgrad_scale = float(qgrad_scale)
        ctx.ste_kind = str(ste_kind)
        return q_real, q_imag

    @staticmethod
    def backward(ctx, grad_q_real, grad_q_imag):
        scale = ctx.qgrad_scale
        if ctx.ste_kind == "hardtanh":
            gr = torch.clamp(grad_q_real, -1.0, 1.0) * scale
            gi = torch.clamp(grad_q_imag, -1.0, 1.0) * scale
        else:
            gr = grad_q_real * scale
            gi = grad_q_imag * scale
        return gr, gi, None, None, None


class ConstrainedHead(nn.Module):
    """Composable OTA-constrained classifier head.

    Substitutes for ``nn.Linear(in_dim, n_classes)`` in the backbone.
    The switches in :class:`ConstraintConfig` decide which physical
    constraint is applied, without changing the backbone code.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        config: ConstraintConfig,
    ):
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.n_classes = n_classes

        # Optional real bottleneck (R1) — cuts parameter count on high-dim inputs.
        if config.complex_dim is not None and config.complex_dim > 0:
            self.pre = nn.Linear(in_dim, config.complex_dim)
            eff_in = config.complex_dim
        else:
            self.pre = nn.Identity()
            eff_in = in_dim

        if config.input_dropout > 0:
            self.dropout = nn.Dropout(config.input_dropout)
        else:
            self.dropout = nn.Identity()

        scale = (2.0 / (eff_in + n_classes)) ** 0.5
        self.weight_real = nn.Parameter(torch.randn(eff_in, n_classes) * scale)
        # Imag parameter always exists so quantize_only + magnitude-only are
        # composable in either order. When complex_weighting is off it is
        # frozen at zero so the effective map stays real.
        self.weight_imag = nn.Parameter(torch.randn(eff_in, n_classes) * scale)
        if not config.complex_weighting:
            self.weight_imag.requires_grad_(False)
            with torch.no_grad():
                self.weight_imag.zero_()

        # Phase states buffer for quantization.
        self.register_buffer(
            "phase_states",
            torch.tensor(PHASE_STATES, dtype=torch.float32),
            persistent=False,
        )

    def _effective_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.quantize_2bit:
            q_real, q_imag = _PhaseQuantizeSTE.apply(
                self.weight_real, self.weight_imag,
                self.phase_states.to(self.weight_real.device),
                self.config.qgrad_scale,
                self.config.ste_kind,
            )
            return q_real, q_imag
        return self.weight_real, self.weight_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.pre(x))
        w_real, w_imag = self._effective_weights()
        if self.config.complex_weighting:
            y_real = z @ w_real
            y_imag = z @ w_imag
        else:
            # Real matmul, magnitude may still apply if requested.
            y_real = z @ w_real
            y_imag = torch.zeros_like(y_real)
        if self.config.magnitude_readout:
            return torch.sqrt(y_real * y_real + y_imag * y_imag + 1e-12)
        return y_real


def apply_constraints_to_cnn_gru(model, config: ConstraintConfig):
    """Swap the CNN-GRU backbone's final linear head with a ConstrainedHead.

    Idempotent: safe to call once per model. The dropout that lived in the
    original ``head = Sequential(Dropout, Linear)`` is preserved, then the
    ConstrainedHead handles its own (R1) dropout on top.
    """
    from models.cnn_gru import CNNGRUBackbone

    if not isinstance(model, CNNGRUBackbone):
        raise TypeError(
            f"apply_constraints_to_cnn_gru expects CNNGRUBackbone, "
            f"got {type(model).__name__}"
        )
    gru_dim = model.config.gru_dim
    n_classes = model.head[-1].out_features
    keep_dropout = model.head[0]
    model.head = nn.Sequential(
        keep_dropout,
        ConstrainedHead(gru_dim, n_classes, config=config),
    )
    return model
