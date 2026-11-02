from __future__ import annotations

import torch
from torch import nn


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.backbone = nn.Sequential(*layers)
        self.register_buffer('obs_median', torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer('obs_iqr', torch.ones(obs_dim, dtype=torch.float32))

    def set_obs_normalizer(self, obs_median: torch.Tensor, obs_iqr: torch.Tensor) -> None:
        self.obs_median.copy_(obs_median.to(dtype=torch.float32, device=self.obs_median.device))
        self.obs_iqr.copy_(obs_iqr.to(dtype=torch.float32, device=self.obs_iqr.device).clamp_min(1e-6))

    def forward(self, obs: torch.Tensor, output_scale: float = 1.0) -> torch.Tensor:
        obs_scaled = (obs - self.obs_median) / (self.obs_iqr + 1e-8)
        obs_scaled = obs_scaled.clamp(-10.0, 10.0)
        return torch.tanh(self.backbone(obs_scaled)) * float(output_scale)
