"""
Batch noise injection utilities for training-time augmentation.

Applies noise in the original (unscaled) feature space and maps back to
standardized space using the provided StandardScaler. This preserves the
intended physical noise semantics while training on standardized tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data.noise_utils import add_noise, ColumnNoise


NoiseSchemeFactory = Callable[[Sequence[str], float], ColumnNoise]


@dataclass
class BatchNoiseInjector:
    scaler_x: StandardScaler
    input_cols: Sequence[str]
    scheme_fn: NoiseSchemeFactory
    strength: float  # e.g., snr_db if using make_gaussian_snr
    prob: float = 0.5

    def __call__(self, xb_std: torch.Tensor) -> torch.Tensor:
        """Inject noise into a standardized batch with probability `prob`.

        Steps:
        - inverse_transform to original space
        - apply noise via data.noise_utils.add_noise
        - transform back to standardized space
        """
        if self.prob <= 0.0:
            return xb_std
        if float(torch.rand(1)) > self.prob:
            return xb_std
        device = xb_std.device
        with torch.no_grad():
            xb_np = xb_std.detach().cpu().numpy()
            # back to original units
            xb_orig = self.scaler_x.inverse_transform(xb_np)
            # apply noise
            schemes = [self.scheme_fn(self.input_cols, self.strength)]
            # Use numpy 2D array; add_noise expects a DataFrame-like. Build minimal shim.
            import pandas as pd
            df = pd.DataFrame(xb_orig, columns=list(self.input_cols))
            df_noisy = add_noise(df, schemes, seed=None)
            xb_noisy_orig = df_noisy.values
            # re-standardize
            xb_noisy_std = self.scaler_x.transform(xb_noisy_orig)
            out = torch.tensor(xb_noisy_std, dtype=xb_std.dtype, device=device)
        return out

