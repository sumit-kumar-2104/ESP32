"""CNN-GRU backbone from Widar3.0 (Zheng et al., MobiSys 2019, Sec 5.2-5.3).

Applied per time step and sequenced. Per snapshot (CSI/DFS/BVP):

    Conv2d(C_in -> 16, 3x3, padding=same) -> ReLU
    -> MaxPool2d(2x2)
    -> flatten
    -> Linear(?, 64) -> ReLU -> Dropout
    -> Linear(64, 64) -> ReLU

Time dimension:

    -> GRU(input=64, hidden=128, layers=1)
    -> take last step
    -> Dropout -> Linear(128, n_classes)
    -> softmax + cross-entropy (loss lives in the training loop)

The backbone accepts any of the three tensor representations:

- csi_tensor: ``(N, 18, 30, T)`` -> per-step ``(N*T, 18, 30, 1)``.
- dfs_tensor: ``(N, 6, F, T)`` -> per-step ``(N*T, 6, F, 1)``.
- bvp_tensor: ``(N, Vx, Vy, T)`` -> per-step ``(N*T, 1, Vx, Vy)``.

Only the input rearrangement differs. Every configurable dim
(:attr:`n_filters`, :attr:`dense_dim`, :attr:`gru_dim`,
:attr:`dropout`) has the paper's default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CNNGRUConfig:
    n_filters: int = 16
    dense_dim: int = 64
    gru_dim: int = 128
    dropout: float = 0.5


class _PerStepCNN(nn.Module):
    """Two-stage CNN + dense stack applied to each time step."""

    def __init__(
        self,
        c_in: int,
        h: int,
        w: int,
        n_filters: int,
        dense_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        pooled_h = (h + 1) // 2
        pooled_w = (w + 1) // 2
        flat = n_filters * pooled_h * pooled_w
        self.dense = nn.Sequential(
            nn.Linear(flat, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = dense_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        z = self.conv(x)
        z = z.flatten(1)
        return self.dense(z)


class CNNGRUBackbone(nn.Module):
    """Full CNN-GRU classifier, agnostic to the tensor representation."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        n_classes: int,
        rep_kind: str,
        config: CNNGRUConfig | None = None,
    ):
        super().__init__()
        cfg = config or CNNGRUConfig()
        self.config = cfg
        self.rep_kind = rep_kind
        self._input_shape = tuple(input_shape)

        if rep_kind == "csi_tensor":
            # (18, 30, T) -> per step (18, 30, 1). Antennas as channel-in.
            c_in, h, w = self._input_shape[0], self._input_shape[1], 1
        elif rep_kind == "dfs_tensor":
            # (6, F, T) -> per step (6, F, 1).
            c_in, h, w = self._input_shape[0], self._input_shape[1], 1
        elif rep_kind == "bvp_tensor":
            # (Vx, Vy, T) -> per step (1, Vx, Vy).
            c_in, h, w = 1, self._input_shape[0], self._input_shape[1]
        else:
            raise ValueError(
                f"cnn_gru does not support rep_kind={rep_kind!r}; "
                "expected csi_tensor / dfs_tensor / bvp_tensor"
            )

        self.per_step = _PerStepCNN(
            c_in=c_in, h=h, w=w,
            n_filters=cfg.n_filters,
            dense_dim=cfg.dense_dim,
            dropout=cfg.dropout,
        )
        self.gru = nn.GRU(
            input_size=self.per_step.out_dim,
            hidden_size=cfg.gru_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gru_dim, n_classes),
        )

    def _reshape_to_steps(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, C, H, W)`` from raw input."""
        # x: (B, D1, D2, D3) — same axis order as tensor bundles.
        if self.rep_kind == "csi_tensor":
            # (B, 18, 30, T) -> (B, T, 18, 30, 1)
            b, a, s, t = x.shape
            steps = x.permute(0, 3, 1, 2).unsqueeze(-1)
            return steps
        if self.rep_kind == "dfs_tensor":
            # (B, 6, F, T) -> (B, T, 6, F, 1)
            b, r, f, t = x.shape
            steps = x.permute(0, 3, 1, 2).unsqueeze(-1)
            return steps
        if self.rep_kind == "bvp_tensor":
            # (B, Vx, Vy, T) -> (B, T, 1, Vx, Vy)
            b, vx, vy, t = x.shape
            steps = x.permute(0, 3, 1, 2).unsqueeze(2)
            return steps
        raise RuntimeError(f"unreachable rep_kind={self.rep_kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = self._reshape_to_steps(x)          # (B, T, C, H, W)
        b, t, c, h, w = steps.shape
        flat = steps.reshape(b * t, c, h, w)
        embed = self.per_step(flat)                # (B*T, dense_dim)
        seq = embed.reshape(b, t, -1)
        _, hidden = self.gru(seq)                  # hidden: (1, B, gru_dim)
        last = hidden.squeeze(0)                   # (B, gru_dim)
        return self.head(last)
