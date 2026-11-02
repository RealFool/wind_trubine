from __future__ import annotations

import numpy as np
import torch


class ActionScaler:
    def __init__(self, action_low: list[float], action_high: list[float]) -> None:
        self.low = np.asarray(action_low, dtype=np.float64)
        self.high = np.asarray(action_high, dtype=np.float64)
        self.scale = np.maximum((self.high - self.low) / 2.0, 1e-6)
        self.bias = (self.high + self.low) / 2.0

    def normalize(self, action_real: np.ndarray) -> np.ndarray:
        action_real = np.asarray(action_real, dtype=np.float64)
        return np.clip((action_real - self.bias) / self.scale, -1.0, 1.0).astype(np.float32)

    def denormalize(self, action_norm: np.ndarray) -> np.ndarray:
        action_norm = np.asarray(action_norm, dtype=np.float64)
        return np.clip(action_norm, -1.0, 1.0) * self.scale + self.bias

    def normalize_tensor(self, action_real: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.low, dtype=action_real.dtype, device=action_real.device)
        scale = torch.as_tensor(self.scale, dtype=action_real.dtype, device=action_real.device)
        bias = torch.as_tensor(self.bias, dtype=action_real.dtype, device=action_real.device)
        return ((action_real - bias) / scale).clamp(-1.0, 1.0)

    def denormalize_tensor(self, action_norm: torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(self.scale, dtype=action_norm.dtype, device=action_norm.device)
        bias = torch.as_tensor(self.bias, dtype=action_norm.dtype, device=action_norm.device)
        return action_norm.clamp(-1.0, 1.0) * scale + bias
