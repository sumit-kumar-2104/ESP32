"""
DiscreteNN: Single complex FC layer with weights constrained to 2-bit phase states
DURING training via the Straight-Through Estimator (STE).

This is the baseline from the paper that MetaAI's continuous-then-quantize approach
beats. Each weight is one of {e^{j0}, e^{j π/2}, e^{j π}, e^{j 3π/2}} — a single
meta-atom effectively.

Reference: Paper Section 3.5, comparison with DiscreteNN.
"""

import numpy as np
import torch
import torch.nn as nn

from config import PHASE_STATES


class DiscreteComplexLinear(nn.Module):
    """
    Complex FC layer with 2-bit discrete phase constraints enforced via STE.
    
    During forward pass: weights are quantized to nearest phase state.
    During backward pass: gradients flow through as if quantization didn't happen (STE).
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Continuous parameters (phase angles) that get quantized in forward
        scale = (2.0 / (input_dim + num_classes)) ** 0.5
        self.weight_real = nn.Parameter(torch.randn(input_dim, num_classes) * scale)
        self.weight_imag = nn.Parameter(torch.randn(input_dim, num_classes) * scale)
        # Unit phasors are ~1/scale larger than Xavier; compensate in forward
        self.output_scale = scale

        # Register the 4 allowed phasors as a buffer
        phasors = torch.tensor(
            [np.exp(1j * p) for p in PHASE_STATES], dtype=torch.complex64
        )
        self.register_buffer("phasors", phasors)

    def _quantize_ste(self, w_complex: torch.Tensor) -> torch.Tensor:
        """
        Quantize each complex weight to nearest allowed phasor.
        Uses STE: forward uses quantized, backward uses continuous gradient.
        """
        # Compute angle of each weight
        angles = torch.angle(w_complex)  # shape (U, R)
        
        # Available phase states
        phase_vals = torch.tensor(PHASE_STATES, device=w_complex.device)  # (4,)
        
        # Find nearest phase for each weight
        # Expand for broadcasting: angles (U, R, 1) vs phases (1, 1, 4)
        angle_diff = angles.unsqueeze(-1) - phase_vals.view(1, 1, -1)
        # Wrap to [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        nearest_idx = torch.argmin(torch.abs(angle_diff), dim=-1)  # (U, R)
        
        # Get quantized phasors
        quantized = self.phasors[nearest_idx.flatten()].reshape(w_complex.shape)
        
        # STE: use quantized in forward, but let gradient flow through w_complex
        return w_complex + (quantized - w_complex).detach()

    @property
    def complex_weight(self) -> torch.Tensor:
        """Return the discrete complex weight matrix."""
        w = torch.complex(self.weight_real, self.weight_imag)
        return self._quantize_ste(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: complex tensor (batch, input_dim)
        Returns:
            magnitudes: real tensor (batch, num_classes)
        """
        H_discrete = self.complex_weight
        y_complex = torch.matmul(x, H_discrete)
        return torch.abs(y_complex) * self.output_scale
