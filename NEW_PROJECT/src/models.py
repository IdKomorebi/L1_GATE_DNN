from __future__ import annotations

import os
from typing import Iterable, List, Tuple

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
from torch import nn


class SimpleAdam:
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if isinstance(params, list) and params and isinstance(params[0], dict):
            self.param_groups = []
            for group in params:
                self.param_groups.append(
                    {
                        "params": [p for p in group["params"] if p.requires_grad],
                        "lr": group.get("lr", lr),
                        "betas": group.get("betas", betas),
                        "eps": group.get("eps", eps),
                        "weight_decay": group.get("weight_decay", weight_decay),
                    }
                )
        else:
            self.param_groups = [
                {
                    "params": [p for p in params if p.requires_grad],
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ]
        self.t = 0
        self.m = {}
        self.v = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay:
                    g = g.add(p, alpha=weight_decay)
                pid = id(p)
                if pid not in self.m:
                    self.m[pid] = torch.zeros_like(p)
                    self.v[pid] = torch.zeros_like(p)
                m = self.m[pid]
                v = self.v[pid]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**self.t)
                v_hat = v / (1 - b2**self.t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def make_mlp(in_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, int(dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = int(dim)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class DNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class L1GateRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.ones(in_dim))
        self.net = make_mlp(in_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * self.gate)

    def get_gates(self) -> torch.Tensor:
        return self.gate


class ImprovedGateRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        correlation_vectors: np.ndarray,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cor = torch.as_tensor(correlation_vectors, dtype=torch.float32)
        mean = cor.mean(dim=0, keepdim=True)
        std = cor.std(dim=0, keepdim=True) + 1e-8
        self.register_buffer("correlation_vectors", (cor - mean) / std)
        self.W_meta = nn.Parameter(torch.zeros(cor.shape[1], 1))
        self.b_meta = nn.Parameter(torch.zeros(1))
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def get_gates(self) -> torch.Tensor:
        logits = torch.matmul(self.correlation_vectors, self.W_meta).squeeze(-1) + self.b_meta
        return torch.sigmoid(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * self.get_gates())
