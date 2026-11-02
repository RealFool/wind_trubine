from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from RL.utils.load_guard_calibration import LoadGuardCalibration
from RL.utils.load_predictor_contract import PLAN_KIND_ZOH_CANDIDATE


@dataclass(frozen=True)
class LoadGuardDecision:
    residual_scale: float
    feasible: bool
    no_feasible_action: bool
    upper_peak_ratio: np.ndarray
    plan_kind: str
    horizon_s: float


def select_candidate_residual_scale(
    calibration: LoadGuardCalibration,
    *,
    optimization_state: str,
    horizon_s: float,
    predicted_peak_ratio_by_scale: Mapping[float, Sequence[float]],
    allowed_scales: Sequence[float] = (1.0, 0.5, 0.25, 0.0),
) -> LoadGuardDecision:
    calibration.validate_soft_groups([
        (blade, optimization_state, horizon_s)
        for blade in range(1, 4)
    ])
    scales = sorted({float(scale) for scale in allowed_scales}, reverse=True)
    if not scales or any(scale < 0.0 or scale > 1.0 for scale in scales):
        raise ValueError('allowed residual scales must be in [0, 1]')
    upper_by_scale: dict[float, np.ndarray] = {}
    for scale in scales:
        if scale not in predicted_peak_ratio_by_scale:
            raise ValueError(f'predicted peak ratios are missing residual scale {scale}')
        prediction = np.asarray(predicted_peak_ratio_by_scale[scale], dtype=np.float64)
        if prediction.shape != (3,) or not np.isfinite(prediction).all():
            raise ValueError('predicted peak ratios must contain three finite values')
        upper = np.empty(3, dtype=np.float64)
        for blade in range(1, 4):
            group = calibration.group(
                blade,
                optimization_state,
                horizon_s,
                PLAN_KIND_ZOH_CANDIDATE,
            )
            if not group.get('soft_guard_eligible', False):
                raise ValueError(f'Calibration group is not soft-eligible: {group["group_id"]}')
            upper[blade - 1] = prediction[blade - 1] + float(group['conformal_quantile'])
        upper_by_scale[scale] = upper
    for scale in scales:
        upper = upper_by_scale[scale]
        if np.all(upper <= 1.0):
            return LoadGuardDecision(
                residual_scale=scale,
                feasible=True,
                no_feasible_action=False,
                upper_peak_ratio=upper,
                plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                horizon_s=float(horizon_s),
            )
    zero_upper = upper_by_scale.get(0.0, upper_by_scale[scales[-1]])
    return LoadGuardDecision(
        residual_scale=0.0,
        feasible=False,
        no_feasible_action=True,
        upper_peak_ratio=zero_upper,
        plan_kind=PLAN_KIND_ZOH_CANDIDATE,
        horizon_s=float(horizon_s),
    )
