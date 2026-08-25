"""
Single-layer complex-valued fully-connected network.
Implements the MetaAI channel as a trainable complex linear layer.

Architecture: y_r = | Σ_i H_r(t_i) · x_i |  (paper Eqn. 3)
  - H is a complex weight matrix of shape (INPUT_DIM, NUM_CLASSES) = (U × R)
  - x is the complex input symbol vector
  - Output is the magnitude |y| for each class, predicted class = argmax_r |y_r|

Implementation: uses torch complex64 tensors throughout for native complex backprop.
"""

import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """
    A single complex-valued fully-connected layer.
    Weight H: shape (input_dim, num_classes), dtype complex64.
    Forward: y = |x @ H|  (magnitude of complex matmul).
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Store as two real parameters (for optimizer compatibility)
        # Initialize with Xavier-like scaling for complex weights
        scale = (2.0 / (input_dim + num_classes)) ** 0.5
        self.weight_real = nn.Parameter(torch.randn(input_dim, num_classes) * scale)
        self.weight_imag = nn.Parameter(torch.randn(input_dim, num_classes) * scale)

    @property
    def complex_weight(self) -> torch.Tensor:
        """Return the complex weight matrix H (U × R)."""
        return torch.complex(self.weight_real, self.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: complex tensor of shape (batch, input_dim)
        Returns:
            magnitudes: real tensor of shape (batch, num_classes)
                        representing |y_r| = |Σ_i H_r(i) · x_i|
        Paper Eqn. 3: y_r = |Σ_i H_r(t_i) · x_i|
        """
        # Complex matrix multiplication: (batch, U) @ (U, R) -> (batch, R)
        y_complex = torch.matmul(x, self.complex_weight)
        # Take magnitude
        y_mag = torch.abs(y_complex)
        return y_mag
