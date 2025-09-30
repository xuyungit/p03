"""
Prediction helpers (e.g., MC Dropout inference).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _enable_dropout(module: nn.Module) -> None:
    """Enable (train-mode) for Dropout layers only, keep others as-is."""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform MC Dropout by enabling dropout layers during inference and sampling multiple predictions.

    Args:
      model: trained model
      x: input tensor [B, D]
      n_samples: number of stochastic forward passes

    Returns:
      (mean, std) over samples with shapes [B, out_dim]
    """
    model.eval()
    # Enable dropout submodules
    _enable_dropout(model)
    preds = []
    for _ in range(max(1, n_samples)):
        preds.append(model(x))
    pred_stack = torch.stack(preds, dim=0)
    mean = pred_stack.mean(dim=0)
    std = pred_stack.std(dim=0, unbiased=False)
    # Restore eval mode for the whole model
    model.eval()
    return mean, std

