from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from RL.utils.load_predictor_contract import LoadPredictorInterface


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size <= 0 or dilation <= 0:
            raise ValueError('kernel_size and dilation must be positive')
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        return output[..., :values.shape[-1]]


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, dilation=dilation)
        self.activation = nn.SiLU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.activation(self.conv1(values))
        values = self.conv2(values)
        return self.activation(values + residual)


class CausalTCNLoadPredictor(nn.Module):
    """Causal history encoder conditioned on the known future action plan."""

    OUTPUT_WIDTH = 5
    PLAN_KIND_REALIZED_INDEX = 0
    PLAN_KIND_ZOH_CANDIDATE_INDEX = 1

    def __init__(
        self,
        input_dim: int,
        interface: LoadPredictorInterface,
        hidden_dim: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        action_low: Sequence[float] = (0.0, 0.0),
        action_high: Sequence[float] = (21_586_451.33303, 0.43),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or not dilations:
            raise ValueError('input_dim, hidden_dim, and dilations must be positive')
        self.interface = interface
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        if (
            action_low_tensor.shape != (2,)
            or action_high_tensor.shape != (2,)
            or not torch.isfinite(action_low_tensor).all()
            or not torch.isfinite(action_high_tensor).all()
            or torch.any(action_high_tensor <= action_low_tensor)
        ):
            raise ValueError('load predictor action bounds must be two finite increasing values')
        self.register_buffer('action_low', action_low_tensor)
        self.register_buffer('action_high', action_high_tensor)
        self.register_buffer('history_median', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('history_iqr', torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer('response_median', torch.zeros(2, dtype=torch.float32))
        self.register_buffer('response_iqr', torch.ones(2, dtype=torch.float32))
        self.register_buffer('peak_ratio_median', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('peak_ratio_iqr', torch.ones(3, dtype=torch.float32))
        blocks: list[nn.Module] = []
        channels = input_dim
        for dilation in dilations:
            blocks.append(CausalTemporalBlock(channels, hidden_dim, dilation))
            channels = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.plan_encoder = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.horizon_embedding = nn.Embedding(len(interface.horizons_s), 8)
        self.plan_kind_embedding = nn.Embedding(2, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.OUTPUT_WIDTH),
        )

    def forward_sequence(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != self.interface.history_frames or history.shape[2] != self.input_dim:
            raise ValueError(
                f'Expected history [batch, {self.interface.history_frames}, {self.input_dim}], got {tuple(history.shape)}'
            )
        normalized = (history - self.history_median) / self.history_iqr.clamp_min(1.0e-6)
        values = normalized.transpose(1, 2)
        return self.backbone(values).transpose(1, 2)

    def set_normalization(self, normalization: Mapping[str, Any]) -> None:
        names = (
            ('history_median', 'history_iqr', self.history_median, self.history_iqr, self.input_dim),
            ('response_median', 'response_iqr', self.response_median, self.response_iqr, 2),
            ('peak_ratio_median', 'peak_ratio_iqr', self.peak_ratio_median, self.peak_ratio_iqr, 3),
        )
        with torch.no_grad():
            for median_key, iqr_key, median_buffer, iqr_buffer, width in names:
                try:
                    median = torch.as_tensor(normalization[median_key], dtype=torch.float32)
                    iqr = torch.as_tensor(normalization[iqr_key], dtype=torch.float32)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'load predictor normalization is missing {median_key}/{iqr_key}') from exc
                if median.shape != (width,) or iqr.shape != (width,):
                    raise ValueError(f'load predictor normalization {median_key}/{iqr_key} has an invalid shape')
                if not torch.isfinite(median).all() or not torch.isfinite(iqr).all() or torch.any(iqr <= 0.0):
                    raise ValueError(f'load predictor normalization {median_key}/{iqr_key} is invalid')
                median_buffer.copy_(median.to(median_buffer.device))
                iqr_buffer.copy_(iqr.to(iqr_buffer.device).clamp_min(1.0e-6))

    def forward(
        self,
        history: torch.Tensor,
        candidate_plan: torch.Tensor,
        horizon_index: torch.Tensor,
        plan_kind_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.interface.schema_version != 'causal_load_predictor_interface_v3':
            raise ValueError('Candidate-conditioned predictor model requires interface v3')
        if history.ndim != 3 or history.shape[1] != self.interface.history_frames or history.shape[2] != self.input_dim:
            raise ValueError('load predictor history tensor has an incompatible shape')
        if candidate_plan.ndim != 3 or candidate_plan.shape[0] != history.shape[0] or candidate_plan.shape[2] != 2:
            raise ValueError('candidate_plan must have shape [batch, max_horizon_frames, 2]')
        if candidate_plan.shape[1] != max(self.interface.horizon_frames):
            raise ValueError('candidate_plan max horizon dimension does not match interface v3')
        horizon_index = horizon_index.to(device=history.device, dtype=torch.long).reshape(-1)
        plan_kind_index = plan_kind_index.to(device=history.device, dtype=torch.long).reshape(-1)
        if horizon_index.shape != (history.shape[0],) or plan_kind_index.shape != (history.shape[0],):
            raise ValueError('horizon_index and plan_kind_index must contain one value per sample')
        if torch.any(horizon_index < 0) or torch.any(horizon_index >= len(self.interface.horizons_s)):
            raise ValueError('horizon_index is outside the configured interface')
        if any(
            candidate_plan.shape[1] < self.interface.horizon_frames[int(index)]
            for index in horizon_index.detach().cpu().tolist()
        ):
            raise ValueError('candidate_plan is shorter than a selected horizon')
        if torch.any(plan_kind_index < 0) or torch.any(plan_kind_index >= 2):
            raise ValueError('plan_kind_index must be 0 (realized) or 1 (zoh_candidate)')
        if torch.any((plan_kind_index != self.PLAN_KIND_REALIZED_INDEX) & (plan_kind_index != self.PLAN_KIND_ZOH_CANDIDATE_INDEX)):
            raise ValueError('plan_kind_index contains an unsupported v3 value')
        if not torch.isfinite(candidate_plan).all():
            raise ValueError('candidate_plan must contain only finite actions')

        history_features = self.forward_sequence(history)[:, -1]
        normalized_plan = 2.0 * (candidate_plan - self.action_low) / (self.action_high - self.action_low) - 1.0
        lengths_by_horizon = torch.as_tensor(self.interface.horizon_frames, device=history.device, dtype=torch.long)
        plan_lengths = lengths_by_horizon[horizon_index]
        packed_plan = pack_padded_sequence(
            normalized_plan,
            plan_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, plan_hidden = self.plan_encoder(packed_plan)
        plan_features = plan_hidden[-1]
        context = torch.cat((
            history_features,
            plan_features,
            self.horizon_embedding(horizon_index),
            self.plan_kind_embedding(plan_kind_index),
        ), dim=1)
        return self.head(context)

    def predict_candidate(
        self,
        history: torch.Tensor,
        candidate_plan: torch.Tensor,
        horizon_index: torch.Tensor,
        plan_kind_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return unnormalized [future torque, pitch, blade 1/2/3 peak ratios]."""
        normalized_prediction = self(history, candidate_plan, horizon_index, plan_kind_index)
        response = normalized_prediction[:, :2] * self.response_iqr + self.response_median
        peak_ratios = normalized_prediction[:, 2:] * self.peak_ratio_iqr + self.peak_ratio_median
        return torch.cat((response, peak_ratios), dim=1)