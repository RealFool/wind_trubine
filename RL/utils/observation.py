from __future__ import annotations

from typing import Mapping

import numpy as np

from RL.utils.constants import ACTION_DIM, OBS_DIM, OBSERVATION_VARIABLES, RAW_OBSERVATION_VARIABLES


# Bladed can report cumulative azimuth; the offline model contract uses phase radians.
_ROTOR_AZIMUTH_INDEX = OBSERVATION_VARIABLES.index('I_RotorAzimuth')
_ROTOR_AZIMUTH_PERIOD_RAD = 2.0 * np.pi


def _canonicalize_rotor_azimuth(observations: np.ndarray) -> np.ndarray:
    canonical = np.asarray(observations, dtype=np.float32).copy()
    canonical[..., _ROTOR_AZIMUTH_INDEX] = np.mod(
        canonical[..., _ROTOR_AZIMUTH_INDEX],
        _ROTOR_AZIMUTH_PERIOD_RAD,
    )
    return canonical


def build_observation_from_mapping(payload: Mapping[str, float]) -> np.ndarray:
    observations = np.asarray([float(payload.get(name, 0.0)) for name in OBSERVATION_VARIABLES], dtype=np.float32)
    return _canonicalize_rotor_azimuth(observations)


def build_observation_from_frame(frame) -> np.ndarray:
    columns = [frame[name].to_numpy(dtype=np.float32) for name in OBSERVATION_VARIABLES]
    obs = np.column_stack(columns).astype(np.float32, copy=False)
    if obs.shape[1] != len(OBSERVATION_VARIABLES):
        raise RuntimeError(f'Unexpected observation shape: {obs.shape}')
    return _canonicalize_rotor_azimuth(obs)


def append_previous_action_observation(
    observations: np.ndarray,
    previous_actions_real: np.ndarray,
    action_low: np.ndarray | list[float],
    action_high: np.ndarray | list[float],
) -> np.ndarray:
    """Append normalized previously sent actions to a raw observation."""
    base = np.asarray(observations, dtype=np.float32)
    previous = np.asarray(previous_actions_real, dtype=np.float64)
    if base.shape[-1] != OBS_DIM:
        raise ValueError(f'Expected base observation width {OBS_DIM}, got {base.shape[-1]}')
    expected_previous_shape = base.shape[:-1] + (ACTION_DIM,)
    if previous.shape != expected_previous_shape:
        raise ValueError(f'Expected previous action shape {expected_previous_shape}, got {previous.shape}')
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError('Action bounds must contain two finite values')
    if np.any(high <= low):
        raise ValueError('Action high bounds must exceed action low bounds')
    scale = np.maximum((high - low) / 2.0, 1.0e-6)
    bias = (high + low) / 2.0
    normalized = np.clip((previous - bias) / scale, -1.0, 1.0)
    return np.concatenate([base, normalized.astype(np.float32)], axis=-1)


def decode_previous_action_observation(
    observations: np.ndarray,
    action_low: np.ndarray | list[float],
    action_high: np.ndarray | list[float],
) -> np.ndarray:
    """Decode the normalized previous-action suffix from a control observation."""
    values = np.asarray(observations, dtype=np.float32)
    if values.shape[-1] < OBS_DIM + ACTION_DIM:
        raise ValueError(f'Expected control observation width at least {OBS_DIM + ACTION_DIM}, got {values.shape[-1]}')
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,) or np.any(high <= low):
        raise ValueError('Action bounds must contain two increasing finite values')
    scale = np.maximum((high - low) / 2.0, 1.0e-6)
    bias = (high + low) / 2.0
    normalized = np.asarray(values[..., OBS_DIM:OBS_DIM + ACTION_DIM], dtype=np.float64)
    return np.clip(normalized, -1.0, 1.0) * scale + bias