"""
Benchmark baseline models for Stage 5.
All models take 8000-dim real BVP input (already normalized [0,1])
and classify into 6 gesture classes.
"""

import torch
import torch.nn as nn


class MetaAILinear(nn.Module):
    """
    Over-the-air complex linear model (replicates ComplexLinear from Stage 1).
    Input is BPSK-encoded (complex), output is magnitude-based classification.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        scale = (2.0 / (input_dim + num_classes)) ** 0.5
        self.weight_real = nn.Parameter(torch.randn(input_dim, num_classes) * scale)
        self.weight_imag = nn.Parameter(torch.randn(input_dim, num_classes) * scale)

    @property
    def complex_weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    def forward(self, x_real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_real: (batch, input_dim) real tensor in [0,1]
        Returns:
            (batch, num_classes) magnitude logits
        """
        # BPSK encode: [0,1] -> [-1,+1] -> complex
        bipolar = 2.0 * x_real - 1.0
        x_complex = torch.complex(bipolar, torch.zeros_like(bipolar))
        y_complex = torch.matmul(x_complex, self.complex_weight)
        return torch.abs(y_complex)


class DigitalLinear(nn.Module):
    """
    Plain real-valued linear classifier — the digital floor baseline.
    No over-the-air channel, no complex arithmetic.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP2Layer(nn.Module):
    """
    2-layer MLP baseline (256 hidden, ReLU).
    A modest neural baseline above the linear floor.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
