"""
Phase 1 — In-domain fix hooks (DISABLED by default).

This module holds the two candidate fixes for the in-domain collapse that
Phase 0 (`diagnose_indomain.py`) diagnoses. Both are OPT-IN so nothing in
the existing working pipeline changes unless you explicitly turn them on.

    HOOK A — disable_signal_crushing_normalization()
        The current CSI pipeline applies StandardScaler on the training fold
        and BPSK-encodes with `bipolar = 2x - 1` INSIDE the OTA model. When
        the input dim is small (~150 for dfs_small) the standardized-then-
        bipolar features have `|.|mean` ~= 1, so the complex linear layer
        gets sensible drive. When the input dim is large (~1536 for dfs_full)
        the same Xavier scale produces a complex-linear output that is either
        an order of magnitude too small (COLLAPSE) or too large (EXPLOSION),
        depending on the standardization result. The hook removes the
        `2x - 1` re-bipolarisation for CSI features that are already
        approximately zero-mean unit-std, so the OTA path sees `x` at the
        same scale the LogReg / MLP probes see it.

    HOOK B — rescale_ota_readout()
        Xavier init `scale = sqrt(2/(U+R))` is tuned for a linear layer that
        goes straight into softmax. The MetaAI OTA path applies `|.|` and
        THEN treats magnitudes as logits. On high-dim inputs, |sum| grows
        with sqrt(U), pushing pre-softmax logits far from a usable range and
        making CE loss saturate. HOOK B multiplies the OTA / DiscreteNN
        output by a constant chosen so the mean logit magnitude matches the
        equivalent continuous-model Xavier scale for the current input_dim.

Neither hook runs automatically. To enable one in a NEW experiment:

    from fix_pipeline import (
        enable_hook_A, enable_hook_B, is_hook_A_enabled, is_hook_B_enabled,
        apply_input_norm, apply_output_rescale, assert_indomain_ok,
    )
    enable_hook_A()
    x_in = apply_input_norm(x_raw)          # replaces the 2x-1 re-bipolarize
    y_mag = apply_output_rescale(y_mag, U)  # replaces bare |.| output

Every cross-domain benchmark MUST call assert_indomain_ok(indomain_acc) at
the top so we can never again publish a cross-domain number based on a
chance-level model.
"""

import os
from typing import Optional

import numpy as np


# ─── State ────────────────────────────────────────────────────────────────────

_HOOK_A_ENABLED = False
_HOOK_B_ENABLED = False
# Task B hooks — added after Phase-0 diagnosis ruled out A / B for dfs_full.
_HOOK_STE_ENABLED = False
_HOOK_HEAD_LR_ENABLED = False


def enable_hook_A() -> None:
    """Opt into HOOK A (skip the 2x-1 re-bipolarize for CSI-style inputs)."""
    global _HOOK_A_ENABLED
    _HOOK_A_ENABLED = True


def enable_hook_B() -> None:
    """Opt into HOOK B (rescale OTA output to match continuous Xavier scale)."""
    global _HOOK_B_ENABLED
    _HOOK_B_ENABLED = True


def enable_hook_STE() -> None:
    """Opt into HOOK STE — replace the current 2-bit phase quantizer with a
    surrogate-gradient STE that forwards the quantized weight but backpropagates
    through a hardtanh-style identity so autograd can actually reach the
    underlying real/imag parameters. See `wrap_ste_quantizer` below."""
    global _HOOK_STE_ENABLED
    _HOOK_STE_ENABLED = True


def enable_hook_HEAD_LR() -> None:
    """Opt into HOOK HEAD_LR — build an optimizer with a separate, larger
    learning rate for the OTA / classifier head so a dead-grad head can catch
    up. See `head_param_groups` below."""
    global _HOOK_HEAD_LR_ENABLED
    _HOOK_HEAD_LR_ENABLED = True


def disable_all_hooks() -> None:
    global _HOOK_A_ENABLED, _HOOK_B_ENABLED, _HOOK_STE_ENABLED, _HOOK_HEAD_LR_ENABLED
    global _HOOK_R1_ENABLED, _HOOK_R2_ENABLED
    _HOOK_A_ENABLED = False
    _HOOK_B_ENABLED = False
    _HOOK_STE_ENABLED = False
    _HOOK_HEAD_LR_ENABLED = False
    _HOOK_R1_ENABLED = False
    _HOOK_R2_ENABLED = False


def is_hook_A_enabled() -> bool:
    return _HOOK_A_ENABLED


def is_hook_B_enabled() -> bool:
    return _HOOK_B_ENABLED


def is_hook_STE_enabled() -> bool:
    return _HOOK_STE_ENABLED


def is_hook_HEAD_LR_enabled() -> bool:
    return _HOOK_HEAD_LR_ENABLED


# ─── HOOK A: input normalization ──────────────────────────────────────────────

def apply_input_norm(x):
    """Return either the raw x (hook on) or the legacy `2x - 1` bipolarize.

    Accepts torch tensors or numpy arrays. Complex tensors pass through.

    TODO(phase-1-b): if diagnostic shows CSI features are neither zero-mean nor
    in [0,1], add a per-fold StandardScaler here. For now the caller is
    responsible for standardization on the train fold.
    """
    import torch
    if not _HOOK_A_ENABLED:
        # Legacy behaviour: caller does whatever it did before.
        return x
    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            return x
        return x   # HOOK A skips the bipolarize step.
    return x


def legacy_bipolarize(x):
    """The original `2x - 1` mapping — kept explicit so training scripts
    can call it when HOOK A is OFF, and skip it when HOOK A is ON.

    This is the operation `sim/sender.py::encode_bpsk` implements.
    """
    import torch
    if isinstance(x, torch.Tensor):
        return 2.0 * x - 1.0
    return 2.0 * x - 1.0


# ─── HOOK B: output rescale ───────────────────────────────────────────────────

def output_rescale_factor(input_dim: int, num_classes: int) -> float:
    """Return the scalar that, applied to |y| after the complex linear layer,
    brings mean logit magnitude back to the Xavier-linear equivalent.

    Derivation: with Xavier scale `sigma = sqrt(2/(U+R))`, a complex linear
    layer on unit-variance BPSK-like input produces a per-class complex sum
    whose |.| grows like `sigma * sqrt(U)`. For a DIGITAL linear+softmax
    baseline that scales its logits by 1, we want the OTA output to sit in
    the same ballpark, i.e. |y| ~= O(1). So multiply |y| by `1 / (sigma * sqrt(U))`.
    """
    sigma = (2.0 / (input_dim + num_classes)) ** 0.5
    denom = max(sigma * (input_dim ** 0.5), 1e-8)
    return 1.0 / denom


def apply_output_rescale(y_mag, input_dim: int, num_classes: Optional[int] = None):
    """Multiply |y| by the Xavier-matching factor if HOOK B is enabled."""
    if not _HOOK_B_ENABLED:
        return y_mag
    if num_classes is None:
        import torch
        num_classes = y_mag.shape[-1] if isinstance(y_mag, (list, tuple)) \
            else int(y_mag.shape[-1])
    factor = output_rescale_factor(input_dim, num_classes)
    return y_mag * factor


# ─── HOOK STE: surrogate-gradient straight-through estimator for 2-bit phase ─
#
# The current DiscreteComplexLinear uses `w + (quant - w).detach()`. Autograd
# *sees* w and treats the quantization as identity, which is textbook STE and
# on paper is correct. In practice, if Task A shows the STE grad is opaque
# (grad norm ~0), the culprit is usually that the differentiable path lives
# purely in `torch.angle` — which has a pathologically small local gradient
# for weights near the 0 magnitude. HOOK STE replaces the quantizer with a
# surrogate that uses a hardtanh-clamped identity backward on both the real
# and imaginary parts directly, sidestepping the angle-based path.

class _PhaseQuantizeSTE(__import__("torch").autograd.Function):
    """Forward = nearest 2-bit phasor. Backward = hardtanh-clamped identity."""

    @staticmethod
    def forward(ctx, w_real, w_imag, phase_states):
        import torch
        w_complex = torch.complex(w_real, w_imag)
        angles = torch.angle(w_complex)
        phase_vals = phase_states.to(w_real.device)
        # Nearest allowed phase.
        diff = angles.unsqueeze(-1) - phase_vals.view(*([1] * angles.dim()), -1)
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        idx = torch.argmin(torch.abs(diff), dim=-1)
        chosen = phase_vals[idx]
        q_real = torch.cos(chosen)
        q_imag = torch.sin(chosen)
        ctx.save_for_backward(w_real, w_imag)
        return q_real, q_imag

    @staticmethod
    def backward(ctx, grad_q_real, grad_q_imag):
        w_real, w_imag = ctx.saved_tensors
        import torch
        # Surrogate: clamp incoming gradient with hardtanh so it can't
        # explode, but always let it flow (identity in [-1, 1]).
        gr = torch.clamp(grad_q_real, -1.0, 1.0)
        gi = torch.clamp(grad_q_imag, -1.0, 1.0)
        return gr, gi, None


def wrap_ste_quantizer(discrete_module):
    """If HOOK STE is on, monkey-patch a DiscreteComplexLinear so its
    `.complex_weight` uses `_PhaseQuantizeSTE`. No-op when disabled.
    """
    import torch
    if not _HOOK_STE_ENABLED:
        return discrete_module
    from config import PHASE_STATES

    phase_states = torch.tensor(PHASE_STATES, dtype=torch.float32)

    def _complex_weight(self):
        q_real, q_imag = _PhaseQuantizeSTE.apply(
            self.weight_real, self.weight_imag, phase_states.to(self.weight_real.device),
        )
        return torch.complex(q_real, q_imag)

    # Bind as a property so `module.complex_weight` still works transparently.
    type(discrete_module).complex_weight = property(_complex_weight)
    return discrete_module


# ─── HOOK HEAD_LR: separate learning rate for the OTA / classifier head ───────

def head_param_groups(model, base_lr: float, head_lr_mult: float = 10.0,
                      head_names=("weight_real", "weight_imag", "linear", "fc",
                                  "out")):
    """Return a list of param groups for `torch.optim.Adam(**)` with a bigger
    LR on parameters whose name contains any of `head_names`. Only active when
    HOOK HEAD_LR is enabled; otherwise returns a single group at base_lr.
    """
    if not _HOOK_HEAD_LR_ENABLED:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr}]
    head, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(hn in name for hn in head_names):
            head.append(p)
        else:
            rest.append(p)
    groups = []
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    if head:
        groups.append({"params": head, "lr": base_lr * head_lr_mult})
    return groups


# ─── HOOK R1: continuous-OTA overfit fix ─────────────────────────────────────
#
# Task-A finding: with the continuous ComplexLinear on dfs_full the OTA path
# reaches TRAIN 100% / VAL 41% while the plain-linear control reaches VAL 62%
# — pure overfit, not a gradient bug. HOOK R1 adds three OPT-IN levers:
#
#   * weight decay      (--wd,          default 1e-4)
#   * input dropout     (--dropout,     default 0.3)  applied to the real
#                                       feature vector *before* it is lifted
#                                       to complex and hit by ComplexLinear.
#                                       The OTA model has no separate
#                                       classifier head (`|y|` IS the logit),
#                                       so this is the honest "dropout before
#                                       the classifier" placement.
#   * complex-dim bottleneck (--complex-dim)  optional real Linear projection
#                                       input_dim -> complex_dim inserted
#                                       BEFORE the ComplexLinear, cutting
#                                       parameter count on high-dim inputs.
#   * early stop on val (--patience)   returned in `r1_config()` so the
#                                       training loop can consume it directly.
#
# Nothing is applied unless `enable_hook_R1()` is called. `make_r1_ota_model`
# builds the wrapper explicitly; existing scripts that instantiate
# ComplexLinear directly are unaffected.

_HOOK_R1_ENABLED = False
_R1_CFG = {"wd": 1e-4, "dropout": 0.3, "complex_dim": None, "patience": 8}


def enable_hook_R1(wd: float = 1e-4, dropout: float = 0.3,
                   complex_dim: Optional[int] = None,
                   patience: int = 8) -> None:
    """Opt into HOOK R1 (continuous overfit fixes)."""
    global _HOOK_R1_ENABLED
    _HOOK_R1_ENABLED = True
    _R1_CFG.update({
        "wd": float(wd),
        "dropout": float(dropout),
        "complex_dim": (int(complex_dim) if complex_dim else None),
        "patience": int(patience),
    })


def is_hook_R1_enabled() -> bool:
    return _HOOK_R1_ENABLED


def r1_config() -> dict:
    return dict(_R1_CFG)


def make_r1_ota_model(input_dim: int, num_classes: int):
    """Build the R1-wrapped OTA model. Falls back to plain ComplexLinear when
    HOOK R1 is disabled."""
    import torch
    import torch.nn as nn
    from models.linear_complex import ComplexLinear

    if not _HOOK_R1_ENABLED:
        return ComplexLinear(input_dim, num_classes)

    cfg = r1_config()
    complex_dim = cfg["complex_dim"]
    dropout_p = cfg["dropout"]

    class _R1OTAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(dropout_p)
            if complex_dim is not None and 0 < complex_dim < input_dim:
                self.proj = nn.Linear(input_dim, complex_dim, bias=False)
                comp_in = complex_dim
            else:
                self.proj = None
                comp_in = input_dim
            self.complex = ComplexLinear(comp_in, num_classes)
            # Expose the complex layer's weights under stable names so the
            # optimizer / grad-instrumentation helpers can find them.
            self.weight_real = self.complex.weight_real
            self.weight_imag = self.complex.weight_imag

        def forward(self, x):
            # `x` is expected as either a REAL (batch, D) tensor OR the
            # legacy complex tensor. The OTA path historically feeds a
            # complex tensor; here we accept both so `train_ab_dfs.py` and
            # `train_raw_csi.py` can share this wrapper.
            if torch.is_complex(x):
                x = x.real
            x = self.drop(x)
            if self.proj is not None:
                x = self.proj(x)
            xc = torch.complex(x, torch.zeros_like(x))
            return self.complex(xc)

    return _R1OTAModel()


# ─── HOOK R2: quantized-OTA under-learn fix ──────────────────────────────────
#
# Task-A finding: with 2-bit DiscreteComplexLinear the gradient reaching the
# complex weights is ~14x smaller than the plain-linear control's head
# gradient, and softmax entropy stays near uniform. STE itself WORKS (grad
# flows) — the issue is grad MAGNITUDE. HOOK R2 addresses that directly:
#
#   * --qgrad-scale  multiply the STE backward by a constant so grad magnitude
#                     matches (or exceeds) the control head. Default 8.0
#                     roughly offsets the observed ~14x shrink after Adam's
#                     per-parameter normalization.
#   * --lr-complex   optional separate learning rate ONLY for the complex
#                     weights (weight_real / weight_imag), applied via the
#                     `r2_param_groups()` helper.
#   * --ste hardtanh optional hardtanh-clipped identity surrogate on backward
#                     (clamps grad to [-1, 1] before scaling) — protects
#                     against occasional grad spikes when the pre-quant
#                     weights land near a phasor boundary.
#
# Nothing is applied unless `enable_hook_R2()` is called. Use the factory
# `make_r2_ota_model` (which returns `R2DiscreteComplexLinear` when the hook
# is on, and plain `DiscreteComplexLinear` otherwise).

_HOOK_R2_ENABLED = False
_R2_CFG = {"qgrad_scale": 8.0, "lr_complex": None, "ste_kind": "identity"}
_R2_STE_KINDS = ("identity", "hardtanh")


def enable_hook_R2(qgrad_scale: float = 8.0,
                   lr_complex: Optional[float] = None,
                   ste_kind: str = "identity") -> None:
    """Opt into HOOK R2 (quantized under-learn fixes)."""
    global _HOOK_R2_ENABLED
    if ste_kind not in _R2_STE_KINDS:
        raise ValueError(f"ste_kind must be one of {_R2_STE_KINDS}")
    _HOOK_R2_ENABLED = True
    _R2_CFG.update({
        "qgrad_scale": float(qgrad_scale),
        "lr_complex": (float(lr_complex) if lr_complex is not None else None),
        "ste_kind": str(ste_kind),
    })


def is_hook_R2_enabled() -> bool:
    return _HOOK_R2_ENABLED


def r2_config() -> dict:
    return dict(_R2_CFG)


class _R2GradScaleSTE(__import__("torch").autograd.Function):
    """Forward = nearest 2-bit phasor. Backward = optionally hardtanh-clipped
    identity, multiplied by qgrad_scale."""

    @staticmethod
    def forward(ctx, w_real, w_imag, phase_states, qgrad_scale, ste_kind):
        import torch
        w_complex = torch.complex(w_real, w_imag)
        angles = torch.angle(w_complex)
        phase_vals = phase_states.to(w_real.device)
        diff = angles.unsqueeze(-1) - phase_vals.view(*([1] * angles.dim()), -1)
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        idx = torch.argmin(torch.abs(diff), dim=-1)
        chosen = phase_vals[idx]
        q_real = torch.cos(chosen)
        q_imag = torch.sin(chosen)
        ctx.qgrad_scale = float(qgrad_scale)
        ctx.ste_kind = str(ste_kind)
        return q_real, q_imag

    @staticmethod
    def backward(ctx, grad_q_real, grad_q_imag):
        import torch
        gr = grad_q_real
        gi = grad_q_imag
        if ctx.ste_kind == "hardtanh":
            gr = torch.clamp(gr, -1.0, 1.0)
            gi = torch.clamp(gi, -1.0, 1.0)
        s = ctx.qgrad_scale
        return gr * s, gi * s, None, None, None


def make_r2_ota_model(input_dim: int, num_classes: int):
    """Build the R2-wrapped quantized OTA model. Falls back to plain
    DiscreteComplexLinear when HOOK R2 is disabled."""
    import torch
    from models.discrete_nn import DiscreteComplexLinear

    if not _HOOK_R2_ENABLED:
        return DiscreteComplexLinear(input_dim, num_classes)

    cfg = r2_config()
    qs = cfg["qgrad_scale"]
    kind = cfg["ste_kind"]

    class R2DiscreteComplexLinear(DiscreteComplexLinear):
        @property
        def complex_weight(self):
            from config import PHASE_STATES
            phase_states = torch.tensor(
                PHASE_STATES, dtype=torch.float32,
                device=self.weight_real.device,
            )
            qr, qi = _R2GradScaleSTE.apply(
                self.weight_real, self.weight_imag,
                phase_states, qs, kind,
            )
            return torch.complex(qr, qi)

    return R2DiscreteComplexLinear(input_dim, num_classes)


def r2_param_groups(model, base_lr: float):
    """Return optimizer param groups honoring `--lr-complex` when R2 is on.

    Groups the complex weights (`weight_real`, `weight_imag`) at `lr_complex`
    and everything else at `base_lr`. When R2 is off or `--lr-complex` is
    unset, returns a single group at `base_lr`.
    """
    lrc = _R2_CFG["lr_complex"] if _HOOK_R2_ENABLED else None
    if lrc is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr}]
    complex_params, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("weight_real") or name.endswith("weight_imag"):
            complex_params.append(p)
        else:
            other.append(p)
    groups = []
    if other:
        groups.append({"params": other, "lr": base_lr})
    if complex_params:
        groups.append({"params": complex_params, "lr": lrc})
    return groups


# ─── Cross-domain safety gate ─────────────────────────────────────────────────

# Phase-0 diagnosis: DFS probe ceiling is ~63% (MLP) / ~61% (LogReg). A 60%
# floor for the OTA model is therefore too aggressive on DFS — the OTA is
# linear+magnitude, which cannot match a nonlinear MLP. 50% is the honest DFS
# floor. Callers targeting raw CSI should pass a stricter threshold explicitly
# via `assert_indomain_ok(..., threshold=0.70)` or the --target-acc flag on
# the downstream script.
DEFAULT_INDOMAIN_THRESHOLD = 0.50


class IndomainCheckFailed(AssertionError):
    """Raised when a cross-domain experiment is asked to run on a chance-level
    in-domain model. Cross-domain numbers on such a model are meaningless."""


def assert_indomain_ok(acc: float,
                       threshold: float = DEFAULT_INDOMAIN_THRESHOLD,
                       target_acc: Optional[float] = None,
                       label: str = "in-domain accuracy") -> None:
    """Fail loud if the in-domain accuracy is below `threshold`.

    Every cross-domain / C1 / B3 / B4 experiment MUST call this at the top,
    using the freshly-measured in-domain accuracy of the model it is about
    to run cross-domain. This makes the "we accidentally ran cross-domain on
    a chance model" bug impossible.

    Args:
        acc:        measured in-domain accuracy (fraction in [0,1]).
        threshold:  hard floor, below which the check FAILS. Default 0.50
                    (DFS honest floor after Phase-0). Raise it for stronger
                    features (e.g. raw CSI, pass 0.70 or higher).
        target_acc: convenience alias for `threshold` used by downstream
                    scripts' `--target-acc` flag. When both are provided,
                    `target_acc` wins.
        label:      human name of the number being checked (for the log).

    Override via environment: METAAI_INDOMAIN_THRESHOLD=0.5 (float)
    Dev bypass (never in a paper run): METAAI_SKIP_INDOMAIN_CHECK=1
    """
    if target_acc is not None:
        threshold = float(target_acc)
    if os.environ.get("METAAI_SKIP_INDOMAIN_CHECK") == "1":
        print(f"[assert_indomain_ok] BYPASSED via METAAI_SKIP_INDOMAIN_CHECK=1  "
              f"(measured {label} = {acc*100:.2f}%)")
        return
    env_t = os.environ.get("METAAI_INDOMAIN_THRESHOLD")
    if env_t:
        try:
            threshold = float(env_t)
        except ValueError:
            pass
    if not np.isfinite(acc):
        raise IndomainCheckFailed(
            f"{label} is not finite ({acc}). Cannot proceed with a cross-domain "
            f"experiment on an unmeasured model."
        )
    if acc + 1e-9 < threshold:
        raise IndomainCheckFailed(
            f"{label} = {acc*100:.2f}% is below the {threshold*100:.0f}% floor. "
            f"Refusing to run a cross-domain experiment on a chance-level model. "
            f"Fix the in-domain pipeline first (see logs/diag_*.log and "
            f"fix_pipeline.enable_hook_STE / enable_hook_HEAD_LR), then rerun.\n"
            f"To temporarily bypass this check in dev, set "
            f"METAAI_SKIP_INDOMAIN_CHECK=1 (never in a paper run)."
        )
    print(f"[assert_indomain_ok] OK — {label} = {acc*100:.2f}% "
          f">= threshold {threshold*100:.0f}%")


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("fix_pipeline: HOOK A       enabled =", is_hook_A_enabled())
    print("fix_pipeline: HOOK B       enabled =", is_hook_B_enabled())
    print("fix_pipeline: HOOK STE     enabled =", is_hook_STE_enabled())
    print("fix_pipeline: HOOK HEAD_LR enabled =", is_hook_HEAD_LR_enabled())
    print("fix_pipeline: HOOK R1      enabled =", is_hook_R1_enabled())
    print("fix_pipeline: HOOK R2      enabled =", is_hook_R2_enabled())
    print("fix_pipeline: rescale factor for (U=1536, R=6) =",
          output_rescale_factor(1536, 6))
    print("fix_pipeline: rescale factor for (U=150,  R=6) =",
          output_rescale_factor(150, 6))
    # R1 / R2 opt-in surface
    enable_hook_R1(wd=1e-3, dropout=0.4, complex_dim=512, patience=6)
    print("fix_pipeline: R1 cfg =", r1_config())
    enable_hook_R2(qgrad_scale=8.0, lr_complex=5e-3, ste_kind="hardtanh")
    print("fix_pipeline: R2 cfg =", r2_config())
    disable_all_hooks()
    try:
        assert_indomain_ok(0.24)
    except IndomainCheckFailed as e:
        print(f"[self-test] correctly raised at default threshold: {e}")
    # DFS honest floor: 50% should PASS at 0.55.
    assert_indomain_ok(0.55, label="DFS OTA in-domain")
    # Raw-CSI stricter target should FAIL a 0.55 model.
    try:
        assert_indomain_ok(0.55, target_acc=0.70, label="raw-CSI OTA in-domain")
    except IndomainCheckFailed as e:
        print(f"[self-test] correctly raised at target_acc=0.70: {e}")
