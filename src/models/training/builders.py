"""
Builders for model, optimizer, criterion, and scheduler.

Phase 0: reuse MLP from rev02 for consistent baseline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["none", "step", "plateau", "cosine"] = "step"
    step_size: int = 3000
    gamma: float = 0.5
    plateau_patience: int = 8
    plateau_factor: float = 0.5
    cosine_tmax: int = 100
    warmup_steps: int = 0
    min_lr: float = 0.0


@dataclass
class ModelConfig:
    in_dim: int
    out_dim: int
    num_layers: int = 4
    hidden_dim: int = 128
    act_hidden: str = "silu"
    act_out: str = "identity"
    dropout_p: float = 0.1
    model_type: Literal["mlp", "res_mlp"] = "mlp"
    norm: Literal["none", "batchnorm", "layernorm"] = "none"
    init: Literal["xavier", "kaiming"] = "xavier"
    loss: Literal["mse", "huber", "l1"] = "mse"
    # Optional per-target loss weights (length = out_dim). If provided, enables weighted multi-output loss.
    loss_weights: Optional[Tuple[float, ...]] = None


def build_model_components(
    model_cfg: ModelConfig,
    optim_cfg: OptimConfig,
    device: torch.device,
) -> Tuple[
    nn.Module,
    torch.optim.Optimizer,
    nn.Module,
    Optional[torch.optim.lr_scheduler._LRScheduler],
    Optional[torch.optim.lr_scheduler._LRScheduler],
]:
    # Choose model type
    if model_cfg.model_type == "mlp":
        from models.components.mlp import MLP
        model = MLP(
            in_dim=model_cfg.in_dim,
            out_dim=model_cfg.out_dim,
            num_layers=model_cfg.num_layers,
            hidden_dim=model_cfg.hidden_dim,
            act_hidden=model_cfg.act_hidden,
            act_out=model_cfg.act_out,
            dropout_p=model_cfg.dropout_p,
            init=model_cfg.init,
        ).to(device)
    elif model_cfg.model_type == "res_mlp":
        from models.components.mlp import ResMLP
        model = ResMLP(
            in_dim=model_cfg.in_dim,
            out_dim=model_cfg.out_dim,
            num_layers=model_cfg.num_layers,
            hidden_dim=model_cfg.hidden_dim,
            act_hidden=model_cfg.act_hidden,
            act_out=model_cfg.act_out,
            dropout_p=model_cfg.dropout_p,
            norm=model_cfg.norm,
            init=model_cfg.init,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_cfg.model_type}")

    if optim_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_cfg.optimizer}")

    # Loss function (with optional per-target weights)
    if model_cfg.loss_weights is not None:
        from models.training.losses import WeightedMultiOutputLoss
        criterion = WeightedMultiOutputLoss(
            base=model_cfg.loss,
            out_dim=model_cfg.out_dim,
            weights=list(model_cfg.loss_weights),
        )
    else:
        if model_cfg.loss == "mse":
            criterion = nn.MSELoss()
        elif model_cfg.loss == "huber":
            criterion = nn.HuberLoss(delta=1.0)
        elif model_cfg.loss == "l1":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss: {model_cfg.loss}")

    # Build schedulers with optional warmup
    main_sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if optim_cfg.scheduler == "step":
        main_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=optim_cfg.step_size, gamma=optim_cfg.gamma
        )
    elif optim_cfg.scheduler == "plateau":
        main_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=optim_cfg.plateau_factor, patience=optim_cfg.plateau_patience
        )
    elif optim_cfg.scheduler == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optim_cfg.cosine_tmax, eta_min=optim_cfg.min_lr
        )
    elif optim_cfg.scheduler == "none":
        main_sched = None
    else:
        raise ValueError(f"Unsupported scheduler: {optim_cfg.scheduler}")

    warmup_sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if optim_cfg.warmup_steps and optim_cfg.warmup_steps > 0:
        # Linear warmup from small factor to 1.0 over warmup_steps epochs
        try:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=optim_cfg.warmup_steps
            )
        except Exception:
            # Fallback to LambdaLR if LinearLR unavailable
            lambda_fn = lambda epoch: min(1.0, max(1e-3, (epoch + 1) / float(optim_cfg.warmup_steps)))
            warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)

    return model, optimizer, criterion, warmup_sched, main_sched
