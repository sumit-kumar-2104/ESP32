"""
Receiver module: accumulates channel outputs and decodes via argmax.
Paper Eqn. 3: predict class = argmax_r |y_r|
"""

import torch


def decode(y_mag: torch.Tensor) -> torch.Tensor:
    """
    Decode received magnitude vector to predicted class labels.

    Args:
        y_mag: real tensor (batch, R) — magnitude |y_r| for each class
    Returns:
        predictions: integer tensor (batch,) — predicted class indices
    """
    return torch.argmax(y_mag, dim=1)
