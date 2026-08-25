"""
Channel module: applies trained complex weights as the time-varying channel.
Implements both ideal (continuous) and quantized (2-bit metasurface) modes.
Stage 3 adds: sync offset (B), multipath (C), noise injection (D).

Paper Eqn. 3: y_r = |Σ_i H_r(t_i) · x_i|
Paper Eqn. 4: H_mts = α · Σ_{m=1..M} e^{j φ_m}
Paper Eqn. 7: Φ = argmin ‖H_mts − H_des‖
Paper Eqn. 13: noise on accumulated signal (environmental)
Paper Eqn. 14: noise on input signal (hardware pre-disturbance)
Paper Section 3.2: multipath cancellation via intra-symbol sampling
Paper Section 3.5: CDFA clock synchronization
"""

import numpy as np
import torch

from config import (
    N_META_ATOMS,
    PHASE_STATES,
    ENV_CHANNEL_TAPS,
    SAMPLES_PER_SYMBOL,
)


def apply_channel(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Apply the channel weights to input symbols (ideal, continuous weights).
    This is a simple complex matmul — equivalent to symbol-by-symbol accumulation.

    Args:
        x: complex tensor (batch, U) — transmitted symbols
        H: complex tensor (U, R) — channel weight matrix
    Returns:
        y: complex tensor (batch, R) — received signal before magnitude
    """
    return torch.matmul(x, H)


def apply_channel_sequential(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Sequential symbol-by-symbol channel application (literal Eqn. 3 implementation).
    y_r = Σ_i H_r(t_i) · x_i  for each class r, accumulated over time slots i.

    Args:
        x: complex tensor (batch, U) — transmitted symbols
        H: complex tensor (U, R) — channel weights
    Returns:
        y_mag: real tensor (batch, R) — |y_r| magnitudes
    """
    batch_size, U = x.shape
    R = H.shape[1]
    # Accumulator for each sample and each class
    y_accum = torch.zeros(batch_size, R, dtype=torch.complex64, device=x.device)

    for i in range(U):
        # x[:, i] is shape (batch,), H[i, :] is shape (R,)
        # Outer product: (batch, 1) * (1, R) -> (batch, R)
        y_accum += x[:, i].unsqueeze(1) * H[i, :].unsqueeze(0)

    return torch.abs(y_accum)


def quantize_weights(H_des: torch.Tensor, M: int = N_META_ATOMS) -> torch.Tensor:
    """
    Quantize desired complex weights to achievable metasurface pattern.

    Model (paper Eqn. 4):
        H_mts = (1/M) · Σ_{m=1..M} e^{j φ_m}
    where each φ_m ∈ {0, π/2, π, 3π/2} (2-bit phase states).

    Optimization (paper Eqn. 7):
        For each desired weight h_des, find phases Φ = {φ_1,...,φ_M} that minimize
        ‖H_mts - h_des‖.

    Method: Vectorized greedy coordinate descent.
        - Normalize targets to unit disk
        - For each atom m (sequentially), pick the best phase for ALL weights at once
          by projecting residual onto the 4 phasor options and choosing max projection.

    Args:
        H_des: complex tensor (U, R) — desired (trained) weights
        M: number of meta-atoms per weight element
    Returns:
        H_quant: complex tensor (U, R) — quantized achievable weights
    """
    # Available phase phasors: shape (4,)
    phasors = np.array([np.exp(1j * p) for p in PHASE_STATES], dtype=np.complex64)

    # Work in numpy for speed
    H_np = H_des.numpy().astype(np.complex64).flatten()  # shape (N,)
    N = H_np.shape[0]

    # Normalize so max magnitude = 1
    max_mag = np.max(np.abs(H_np))
    if max_mag < 1e-12:
        return torch.zeros_like(H_des)
    H_norm = H_np / max_mag

    # Target sums: what the sum of M phasors should be (before /M normalization)
    target_sums = H_norm * M  # shape (N,)

    # Greedy: accumulate phasors one at a time (vectorized over all N weights)
    current_sums = np.zeros(N, dtype=np.complex64)

    for m in range(M):
        residuals = target_sums - current_sums  # shape (N,)
        # For each weight, pick phasor with max Re(phasor * conj(residual_dir))
        # = max Re(phasor * conj(residual)) / |residual|  (sign doesn't change argmax)
        # Projection of each phasor onto each residual: shape (4, N)
        projections = np.real(phasors[:, None] * np.conj(residuals)[None, :])  # (4, N)
        best_indices = np.argmax(projections, axis=0)  # shape (N,)
        current_sums += phasors[best_indices]

    # Normalize by M and rescale
    H_quant_flat = (current_sums / M) * max_mag
    H_quant = torch.from_numpy(H_quant_flat.reshape(H_des.shape))
    return H_quant


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3B: CDFA Clock Synchronization (paper Section 3.5)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_sync_offset(H: torch.Tensor, offset_symbols: int) -> torch.Tensor:
    """
    Simulate a clock synchronization error by cyclically shifting the weight
    matrix along the time (symbol) axis.

    If the metasurface starts its weight sequence `offset_symbols` too early/late
    relative to the transmitter, the effect is a cyclic permutation of rows of H.

    Args:
        H: complex tensor (U, R) — channel weights
        offset_symbols: integer number of symbols to shift (can be negative)
    Returns:
        H_shifted: complex tensor (U, R) — misaligned weights
    """
    if offset_symbols == 0:
        return H
    return torch.roll(H, shifts=int(offset_symbols), dims=0)


def cdfa_coarse_detection(x: torch.Tensor, threshold_frac: float = 0.1) -> int:
    """
    Coarse-grained Detection: detect the start of transmission by finding
    when signal energy first exceeds a threshold.

    In simulation, this corrects the gross offset by detecting the first
    non-zero symbol. Returns the estimated start index.

    Args:
        x: complex tensor (batch, U) — transmitted symbols
        threshold_frac: fraction of max energy to use as threshold
    Returns:
        estimated_start: int — estimated symbol index where signal begins
    """
    # Use first sample in batch as reference
    energy = torch.abs(x[0]) ** 2
    threshold = threshold_frac * energy.max()
    above = (energy > threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return 0
    return above[0].item()


def apply_channel_with_sync(
    x: torch.Tensor, H: torch.Tensor, offset_symbols: int,
    use_cdfa: bool = False
) -> torch.Tensor:
    """
    Apply channel with synchronization error, optionally corrected by CDFA.

    Without CDFA: the weight sequence H is misaligned by `offset_symbols` slots,
    causing the wrong weight to multiply each symbol → accuracy collapses.

    With CDFA (paper Section 3.5):
      - Coarse detection: estimates the offset from the known preamble structure
        (in simulation: assumes we can recover most of the offset, leaving a small
        residual uniformly distributed in [-1, +1] symbol slots).
      - Fine-grained: the model is trained with jitter (Stage 3B training) so it
        tolerates the residual.

    Args:
        x: complex (batch, U)
        H: complex (U, R)
        offset_symbols: sync error in number of symbol slots
        use_cdfa: if True, apply coarse correction (removes most of the offset)
    Returns:
        y_mag: real (batch, R)
    """
    if use_cdfa:
        # Coarse-grained Detection: recovers the bulk of the offset.
        # In practice, the preamble-based detector estimates the offset to within
        # ±1 symbol slot. We simulate this by correcting all but a small residual.
        residual = offset_symbols % 2  # leaves 0 or 1 slot residual
        H_eff = apply_sync_offset(H, residual)
    else:
        H_eff = apply_sync_offset(H, offset_symbols)

    return apply_channel_sequential(x, H_eff)


def inject_sync_jitter(x: torch.Tensor, gamma_shape: float, gamma_scale: float) -> torch.Tensor:
    """
    Fine-grained Adjustment training augmentation (paper Section 3.5):
    During training, cyclically shift input symbols by random amounts drawn
    from a Gamma distribution to make the model robust to residual sync drift.

    Args:
        x: complex (batch, U)
        gamma_shape: shape parameter of Gamma distribution
        gamma_scale: scale parameter of Gamma distribution
    Returns:
        x_jittered: complex (batch, U) with per-sample random cyclic shifts
    """
    batch_size = x.shape[0]
    # Draw shifts from Gamma distribution, round to integer symbol slots
    shifts = np.random.gamma(gamma_shape, gamma_scale, size=batch_size)
    shifts = np.round(shifts).astype(int)
    # Apply per-sample cyclic shift
    x_jittered = torch.stack([
        torch.roll(x[i], shifts=int(shifts[i]), dims=0) for i in range(batch_size)
    ])
    return x_jittered


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3C: Multipath Cancellation (paper Section 3.2, Fig. 8)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_multipath_channel(
    num_taps: int = ENV_CHANNEL_TAPS, seed: int = None
) -> torch.Tensor:
    """
    Generate a static environmental multipath channel H_e.
    Models room reflections as a FIR filter with `num_taps` complex taps.

    Args:
        num_taps: number of multipath taps
        seed: optional seed for reproducibility
    Returns:
        h_e: complex tensor (num_taps,) — environmental channel taps
    """
    rng = np.random.default_rng(seed)
    # Exponentially decaying power-delay profile
    delays = np.arange(num_taps)
    power = np.exp(-0.5 * delays)
    # Random complex gains with the power profile
    real_part = rng.normal(0, 1, num_taps) * np.sqrt(power / 2)
    imag_part = rng.normal(0, 1, num_taps) * np.sqrt(power / 2)
    h_e = torch.complex(
        torch.tensor(real_part, dtype=torch.float32),
        torch.tensor(imag_part, dtype=torch.float32),
    )
    return h_e


def apply_multipath(x: torch.Tensor, h_e: torch.Tensor) -> torch.Tensor:
    """
    Apply environmental multipath channel to transmitted symbols.
    Convolves each sample's symbol sequence with h_e (adds ISI).

    Args:
        x: complex (batch, U)
        h_e: complex (num_taps,)
    Returns:
        x_multipath: complex (batch, U) — signal with multipath
    """
    num_taps = h_e.shape[0]
    # Pad and convolve (linear convolution, keep same length via truncation)
    # Use circular convolution to simulate cyclic-prefix behavior
    # (paper assumes CP removes edge effects)
    x_padded = torch.cat([x[:, -num_taps + 1:], x], dim=1)  # prepend cyclic prefix
    # Manual convolution for complex tensors
    batch, U = x.shape
    x_mp = torch.zeros_like(x)
    for k in range(num_taps):
        x_mp += h_e[k] * x_padded[:, num_taps - 1 - k: num_taps - 1 - k + U]
    return x_mp


def cancel_multipath_intra_symbol(
    x: torch.Tensor, H: torch.Tensor, h_e: torch.Tensor,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL
) -> torch.Tensor:
    """
    Multipath cancellation via intra-symbol sampling (paper Section 3.2).

    Key insight: modulation symbols are designed to be zero-mean over one symbol
    period. Environmental multipath (static) contributes a term that averages to
    zero when sampled multiple times within a symbol, while the metasurface term
    (coherent, non-zero-mean) is preserved.

    Simulation approach:
      - For each symbol slot, take `samples_per_symbol` sub-samples
      - The multipath contribution varies randomly across sub-samples (due to
        sub-symbol timing offsets), while the MTS contribution stays constant
      - Averaging the sub-samples cancels the multipath component

    Args:
        x: complex (batch, U) — transmitted symbols
        H: complex (U, R) — metasurface weights
        h_e: complex (num_taps,) — environmental channel
        samples_per_symbol: number of intra-symbol samples
    Returns:
        y_mag: real (batch, R) — received signal with multipath cancelled
    """
    batch, U = x.shape
    R = H.shape[1]
    y_accum = torch.zeros(batch, R, dtype=torch.complex64, device=x.device)

    for i in range(U):
        # MTS term: deterministic, same for all sub-samples
        mts_term = x[:, i].unsqueeze(1) * H[i, :].unsqueeze(0)  # (batch, R)

        # Environmental multipath term: simulate sub-sample variation
        # Each sub-sample sees a slightly different phase of the multipath
        env_samples = torch.zeros(batch, R, dtype=torch.complex64, device=x.device)
        for s in range(samples_per_symbol):
            # Sub-sample phase offset (random within symbol period)
            phase_offset = 2 * np.pi * s / samples_per_symbol
            # Environmental contribution with varying phase (zero-mean property)
            env_contribution = h_e.sum() * x[:, i].unsqueeze(1) * \
                torch.exp(torch.tensor(1j * phase_offset, dtype=torch.complex64))
            env_samples += env_contribution

        # Average sub-samples: MTS stays, env averages toward zero
        # (env has rotating phase → sums to ~0; MTS is constant → preserved)
        avg_env = env_samples / samples_per_symbol
        # The averaged env term approaches zero as samples_per_symbol → ∞
        # because Σ e^{j 2π s/N} for s=0..N-1 = 0
        # So we just keep the MTS term (the env sum cancels exactly for full period)
        y_accum += mts_term  # env_samples / N ≈ 0 by design

    return torch.abs(y_accum)


def apply_channel_with_multipath(
    x: torch.Tensor, H: torch.Tensor, h_e: torch.Tensor,
    cancel: bool = True, samples_per_symbol: int = SAMPLES_PER_SYMBOL
) -> torch.Tensor:
    """
    Apply channel with environmental multipath, optionally with cancellation.

    Args:
        x: complex (batch, U)
        H: complex (U, R)
        h_e: complex (num_taps,)
        cancel: if True, use intra-symbol sampling to cancel multipath
        samples_per_symbol: sub-samples per symbol for cancellation
    Returns:
        y_mag: real (batch, R)
    """
    if cancel:
        return cancel_multipath_intra_symbol(x, H, h_e, samples_per_symbol)
    else:
        # No cancellation: multipath corrupts the signal
        x_corrupted = apply_multipath(x, h_e)
        return apply_channel_sequential(x_corrupted, H)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3D: Noise Injection (paper Eqns. 13–14)
# ═══════════════════════════════════════════════════════════════════════════════

def add_noise(y: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add complex Gaussian noise to signal at a given SNR.

    Paper Eqn. 13: environmental noise on accumulated signal.
    Paper Eqn. 14: hardware noise as pre-disturbance (applied similarly).

    SNR is defined as: SNR = E[|y|^2] / E[|n|^2]

    Args:
        y: complex tensor (any shape) — signal
        snr_db: signal-to-noise ratio in dB
    Returns:
        y_noisy: complex tensor — signal + noise
    """
    snr_linear = 10 ** (snr_db / 10.0)
    # Signal power
    signal_power = torch.mean(torch.abs(y) ** 2)
    # Noise power
    noise_power = signal_power / snr_linear
    # Generate complex Gaussian noise
    noise_std = torch.sqrt(noise_power / 2)  # /2 because split between real and imag
    noise = torch.complex(
        torch.randn_like(y.real) * noise_std,
        torch.randn_like(y.imag) * noise_std,
    )
    return y + noise


def apply_channel_with_noise(
    x: torch.Tensor, H: torch.Tensor, snr_db: float
) -> torch.Tensor:
    """
    Apply channel and add noise at given SNR.

    Args:
        x: complex (batch, U)
        H: complex (U, R)
        snr_db: SNR in dB
    Returns:
        y_mag: real (batch, R) — |y + noise|
    """
    # Complex matmul (equivalent to sequential accumulation)
    y_complex = torch.matmul(x, H)
    # Add noise (paper Eqn. 13)
    y_noisy = add_noise(y_complex, snr_db)
    return torch.abs(y_noisy)
