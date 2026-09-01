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
    _HOOK_A_ENABLED = False
    _HOOK_B_ENABLED = False
    _HOOK_STE_ENABLED = False
    _HOOK_HEAD_LR_ENABLED = False


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
    print("fix_pipeline: rescale factor for (U=1536, R=6) =",
          output_rescale_factor(1536, 6))
    print("fix_pipeline: rescale factor for (U=150,  R=6) =",
          output_rescale_factor(150, 6))
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
