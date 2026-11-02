from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SimulationLinearModelInference(nn.Module):
    def __init__(
        self,
        x_mdn: list[float],
        x_iqr: list[float],
        y_mdn: list[float],
        y_iqr: list[float],
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        input_dim = len(x_mdn)
        output_dim = len(y_mdn)

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
        ]
        for idx in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

        self.register_buffer('x_mdn', torch.tensor(x_mdn, dtype=torch.float32))
        self.register_buffer('x_iqr', torch.tensor(x_iqr, dtype=torch.float32))
        self.register_buffer('y_mdn', torch.tensor(y_mdn, dtype=torch.float32))
        self.register_buffer('y_iqr', torch.tensor(y_iqr, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = (x - self.x_mdn) / (self.x_iqr + 1e-8)
        y_scaled = self.mlp(x_scaled)
        return y_scaled * self.y_iqr + self.y_mdn


def load_dynamics_model_from_ckpt(ckpt_path: str, device: torch.device) -> SimulationLinearModelInference:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    hyper_parameters: dict[str, Any] = checkpoint['hyper_parameters']
    model = SimulationLinearModelInference(
        x_mdn=hyper_parameters['x_mdn'],
        x_iqr=hyper_parameters['x_iqr'],
        y_mdn=hyper_parameters['y_mdn'],
        y_iqr=hyper_parameters['y_iqr'],
        hidden_dims=hyper_parameters['hidden_dims'],
    )
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
