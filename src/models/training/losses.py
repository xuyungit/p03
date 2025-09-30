"""
Custom loss utilities for training.

Provides per-target weighted multi-output loss wrappers.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn


class WeightedMultiOutputLoss(nn.Module):
    """
    Wrap a base regression loss (mse/huber/l1) to apply per-target weights for multi-output regression.

    Computes element-wise loss with reduction='none', multiplies by weights along the last dimension,
    then averages over targets and batch.
    """

    def __init__(
        self,
        *,
        base: Literal["mse", "huber", "l1"] = "mse",
        out_dim: int,
        weights: Optional[List[float]] = None,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer(
            "weights",
            torch.ones(out_dim) if not weights else torch.tensor(weights, dtype=torch.float32),
        )
        if self.weights.numel() != out_dim:
            raise ValueError(f"weights length {self.weights.numel()} != out_dim {out_dim}")

        base = base.lower()
        if base == "mse":
            self.base = nn.MSELoss(reduction="none")
        elif base == "huber":
            self.base = nn.HuberLoss(delta=huber_delta, reduction="none")
        elif base == "l1":
            self.base = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported base loss: {base}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Shapes: [B, D]
        loss_elem = self.base(pred, target)
        if loss_elem.dim() == 1:
            # Handle rare shape [B]
            loss_elem = loss_elem.unsqueeze(-1)
        # Broadcast weights over batch
        w = self.weights.view(1, -1).to(loss_elem.device)
        weighted = loss_elem * w
        # Mean over targets then mean over batch
        return weighted.mean(dim=-1).mean(dim=0)

