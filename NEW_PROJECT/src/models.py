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
                        "name": group.get("name", ""),
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
                    "name": "",
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


class DGatingRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        dgate_depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(v) for v in hidden_dims]
        if not dims:
            dims = [64]
        if int(dgate_depth) < 2:
            raise ValueError("dgate_depth must be >= 2.")

        self.in_dim = int(in_dim)
        self.first_hidden_dim = dims[0]
        self.dgate_depth = int(dgate_depth)
        self.omega = nn.Parameter(torch.empty(self.in_dim, self.first_hidden_dim))
        self.gamma = nn.Parameter(torch.ones(self.dgate_depth - 1, self.in_dim))
        self.bias = nn.Parameter(torch.zeros(self.first_hidden_dim))
        nn.init.kaiming_normal_(self.omega, nonlinearity="relu")

        layers: List[nn.Module] = []
        prev = self.first_hidden_dim
        for dim in dims[1:]:
            layers.append(nn.Linear(prev, int(dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = int(dim)
        layers.append(nn.Linear(prev, 1))
        self.tail = nn.Sequential(*layers)

    def get_gates(self) -> torch.Tensor:
        return torch.prod(self.gamma, dim=0)

    def effective_weight(self) -> torch.Tensor:
        return self.omega * self.get_gates().unsqueeze(1)

    def effective_group_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.effective_weight(), ord=2, dim=1)

    def omega_group_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.omega, ord=2, dim=1)

    def dgate_regularizer(self) -> torch.Tensor:
        return torch.sum(self.omega**2) + torch.sum(self.gamma**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.matmul(x, self.effective_weight()) + self.bias)
        return self.tail(h)


class ImprovedGateRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        correlation_vectors: np.ndarray,
        dropout: float = 0.0,
        meta_init_scale: float = 0.0,
        b_meta_init: float = 0.0,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        cor = torch.as_tensor(correlation_vectors, dtype=torch.float32)
        mean = cor.mean(dim=0, keepdim=True)
        std = cor.std(dim=0, keepdim=True) + 1e-8
        self.register_buffer("correlation_vectors", (cor - mean) / std)
        scale = float(meta_init_scale or 0.0)
        if scale > 0:
            self.W_meta = nn.Parameter(torch.empty(cor.shape[1], 1).uniform_(-scale, scale))
        else:
            self.W_meta = nn.Parameter(torch.zeros(cor.shape[1], 1))
        self.b_meta = nn.Parameter(torch.tensor([float(b_meta_init)], dtype=torch.float32))
        self.gate_temperature = max(float(gate_temperature or 1.0), 1e-6)
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def get_gates(self) -> torch.Tensor:
        logits = self.get_gate_logits()
        return torch.sigmoid(logits / self.gate_temperature)

    def get_gate_logits(self) -> torch.Tensor:
        return torch.matmul(self.correlation_vectors, self.W_meta).squeeze(-1) + self.b_meta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * self.get_gates())
