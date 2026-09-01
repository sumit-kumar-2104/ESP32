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


def enable_hook_A() -> None:
    """Opt into HOOK A (skip the 2x-1 re-bipolarize for CSI-style inputs)."""
    global _HOOK_A_ENABLED
    _HOOK_A_ENABLED = True


def enable_hook_B() -> None:
    """Opt into HOOK B (rescale OTA output to match continuous Xavier scale)."""
    global _HOOK_B_ENABLED
    _HOOK_B_ENABLED = True


def disable_all_hooks() -> None:
    global _HOOK_A_ENABLED, _HOOK_B_ENABLED
    _HOOK_A_ENABLED = False
    _HOOK_B_ENABLED = False


def is_hook_A_enabled() -> bool:
    return _HOOK_A_ENABLED


def is_hook_B_enabled() -> bool:
    return _HOOK_B_ENABLED


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


# ─── Cross-domain safety gate ─────────────────────────────────────────────────

DEFAULT_INDOMAIN_THRESHOLD = 0.60


class IndomainCheckFailed(AssertionError):
    """Raised when a cross-domain experiment is asked to run on a chance-level
    in-domain model. Cross-domain numbers on such a model are meaningless."""


def assert_indomain_ok(acc: float,
                       threshold: float = DEFAULT_INDOMAIN_THRESHOLD,
                       label: str = "in-domain accuracy") -> None:
    """Fail loud if the in-domain accuracy is below `threshold`.

    Every cross-domain / C1 / B3 / B4 experiment MUST call this at the top,
    using the freshly-measured in-domain accuracy of the model it is about
    to run cross-domain. This makes the "we accidentally ran cross-domain on
    a chance model" bug impossible.

    Override via environment:  METAAI_INDOMAIN_THRESHOLD=0.5
    Bypass (dangerous, dev only):  METAAI_SKIP_INDOMAIN_CHECK=1
    """
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
            f"fix_pipeline.enable_hook_A / enable_hook_B), then rerun.\n"
            f"To temporarily bypass this check in dev, set "
            f"METAAI_SKIP_INDOMAIN_CHECK=1 (never in a paper run)."
        )
    print(f"[assert_indomain_ok] OK — {label} = {acc*100:.2f}% "
          f">= threshold {threshold*100:.0f}%")


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("fix_pipeline: HOOK A enabled =", is_hook_A_enabled())
    print("fix_pipeline: HOOK B enabled =", is_hook_B_enabled())
    print("fix_pipeline: rescale factor for (U=1536, R=6) =",
          output_rescale_factor(1536, 6))
    print("fix_pipeline: rescale factor for (U=150,  R=6) =",
          output_rescale_factor(150, 6))
    try:
        assert_indomain_ok(0.24)
    except IndomainCheckFailed as e:
        print(f"[self-test] correctly raised: {e}")
    assert_indomain_ok(0.80)
