"""
Sender module: maps input samples to complex-valued symbols.
Currently implements BPSK modulation; hook for QAM extensions.

Pipeline: flatten 28×28 image → normalize → BPSK map to complex symbols.
"""

import torch


def encode_bpsk(images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of MNIST images into complex BPSK symbols.

    BPSK mapping:
      - Normalize pixel values to [0, 1]
      - Map to bipolar: s = 2*pixel - 1  (values in [-1, +1])
      - Complex symbol: x = s + 0j  (real-valued BPSK on I-axis)

    Args:
        images: tensor of shape (batch, 1, 28, 28), values in [0, 1]
    Returns:
        complex tensor of shape (batch, 784)
    """
    # Flatten: (batch, 1, 28, 28) -> (batch, 784)
    flat = images.view(images.size(0), -1)
    # Bipolar mapping: [0,1] -> [-1, +1]
    bipolar = 2.0 * flat - 1.0
    # Convert to complex (BPSK: all energy on real axis)
    symbols = torch.complex(bipolar, torch.zeros_like(bipolar))
    return symbols


def encode(images: torch.Tensor, modulation: str = "bpsk") -> torch.Tensor:
    """
    Main encoding interface. Dispatches to modulation-specific encoder.

    Args:
        images: (batch, 1, 28, 28) tensor
        modulation: "bpsk" (default), "qam" (future hook)
    Returns:
        complex tensor of shape (batch, input_dim)
    """
    if modulation == "bpsk":
        return encode_bpsk(images)
    elif modulation == "qam":
        raise NotImplementedError("QAM modulation is reserved for future work.")
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
