"""
Batch-level augmentation transforms for tabular regression.

Implements a flip transform that operates in original (unscaled) units and
re-standardizes, flipping both inputs and targets consistently using the
domain-specific mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data import flip_df
from .augment_config import FlipAugmentor, load_augmentor


@dataclass
class BatchFlipTransform:
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    input_cols: Sequence[str]
    target_cols: Sequence[str]
    prob: float = 0.5
    augmentor: Optional[FlipAugmentor] = None

    def __call__(self, xb_std: torch.Tensor, yb_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prob <= 0.0:
            return xb_std, yb_std
        if float(torch.rand(1)) > self.prob:
            return xb_std, yb_std
        device = xb_std.device
        with torch.no_grad():
            X_np = xb_std.detach().cpu().numpy()
            Y_np = yb_std.detach().cpu().numpy()
            # Back to original units
            X_orig = self.scaler_x.inverse_transform(X_np)
            Y_orig = self.scaler_y.inverse_transform(Y_np)
            # Combine, apply flip (configurable), split
            df = pd.DataFrame(np.concatenate([X_orig, Y_orig], axis=1), columns=list(self.input_cols) + list(self.target_cols))
            if self.augmentor is not None:
                df_flipped = self.augmentor.apply_df(df)
            else:
                df_flipped = flip_df(df)
            X_f = df_flipped[self.input_cols].values
            Y_f = df_flipped[self.target_cols].values
            # Re-standardize
            X_std = self.scaler_x.transform(X_f)
            Y_std = self.scaler_y.transform(Y_f)
            xb_out = torch.tensor(X_std, dtype=xb_std.dtype, device=device)
            yb_out = torch.tensor(Y_std, dtype=yb_std.dtype, device=device)
        return xb_out, yb_out
