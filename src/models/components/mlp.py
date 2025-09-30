"""
MLP components: baseline MLP and ResMLP with optional normalization and init.

The baseline MLP mirrors the implementation in rev02 to preserve behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name in ("identity", "linear", "none"):
        return nn.Identity()
    return nn.SiLU()


def init_linear(m: nn.Linear, scheme: Literal["xavier", "kaiming"], nonlinearity: str = "relu") -> None:
    if scheme == "kaiming":
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
    else:
        nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 11,
        out_dim: int = 7,
        num_layers: int = 4,
        hidden_dim: int = 128,
        act_hidden: str = "silu",
        act_out: str = "identity",
        dropout_p: float = 0.1,
        init: Literal["xavier", "kaiming"] = "xavier",
    ) -> None:
        super().__init__()
        act_h = make_activation(act_hidden)
        act_o = make_activation(act_out)

        layers = []
        prev = in_dim
        for _ in range(max(0, num_layers - 1)):
            lin = nn.Linear(prev, hidden_dim)
            layers.append(lin)
            layers.append(act_h)
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev = hidden_dim
        lin_out = nn.Linear(prev, out_dim)
        layers.append(lin_out)
        layers.append(act_o)
        self.net = nn.Sequential(*layers)

        # Initialize
        nonlin = "relu" if act_hidden.lower() == "relu" else "leaky_relu"
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, init, nonlinearity=nonlin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        act: nn.Module,
        dropout_p: float = 0.0,
        norm: Literal["none", "batchnorm", "layernorm"] = "none",
        init: Literal["xavier", "kaiming"] = "xavier",
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = act
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        if norm == "batchnorm":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        elif norm == "layernorm":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        nonlin = "relu"
        init_linear(self.lin1, init, nonlinearity=nonlin)
        init_linear(self.lin2, init, nonlinearity=nonlin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm1(out)
        out = self.lin2(out)
        out = self.norm2(out)
        return self.act(out + x)


class ResMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        act_hidden: str = "relu",
        act_out: str = "identity",
        dropout_p: float = 0.0,
        norm: Literal["none", "batchnorm", "layernorm"] = "none",
        init: Literal["xavier", "kaiming"] = "xavier",
    ) -> None:
        super().__init__()
        act_h = make_activation(act_hidden)
        act_o = make_activation(act_out)

        self.stem = nn.Linear(in_dim, hidden_dim)
        blocks = []
        for _ in range(max(0, num_layers - 2)):
            blocks.append(ResidualBlock(hidden_dim, act_h, dropout_p=dropout_p, norm=norm, init=init))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.out_act = act_o

        nonlin = "relu"
        init_linear(self.stem, init, nonlinearity=nonlin)
        init_linear(self.head, init, nonlinearity=nonlin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h) if len(self.blocks) > 0 else h
        y = self.head(h)
        return self.out_act(y)

