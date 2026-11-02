from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from RL.utils.control_contract import PhysicalActionTransform
from RL.utils.control_execution import (
    PreSendExecutionMode,
    pre_send_execution_mode,
    pre_send_execution_mode_code,
)
from RL.utils.normalize import ActionScaler
from RL.utils.constants import (
    ACTION_DIM,
    ACTION_VARIABLES,
    CONTROL_OBS_DIM,
    DATA_SCHEMA_VERSION,
    OBSERVATION_VARIABLES,
    PREVIOUS_ACTION_OBSERVATION_VARIABLES,
)
from RL.utils.observation import decode_previous_action_observation
from RL.utils.operating_state import (
    CONTROL_REGION_ENCODING_ORDER,
    OPTIMIZATION_STATE_ENCODING_ORDER,
    STATE_REGIME_OBSERVATION_VARIABLES,
    OperatingPoint,
    OptimizationState,
)
from RL.utils.region_aware_baseline import RegionAwareJointBaseline
from RL.utils.state_action_artifact import (
    STATE_AWARE_CONTROL_OBS_DIM,
    StateActionEnvelopeArtifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
S2_U1_CALIBRATION_SCHEMA_VERSION = 's2_u1_pid_envelope_v1'


@dataclass(frozen=True)
class SafeAction:
    residual_norm: np.ndarray
    action_real: np.ndarray
    action_norm: np.ndarray
    baseline_action_real: np.ndarray
    policy_delta_real: np.ndarray
    pre_safety_action_real: np.ndarray
    post_pitch_protection_action_real: np.ndarray
    post_safety_action_real: np.ndarray
    projected_action_real: np.ndarray
    rate_limited_action_real: np.ndarray
    pitch_protection_floor: float
    pitch_protection_state: int
    pitch_protection_reason: int
    pitch_protection_lift: float
    low_speed_stage: int
    low_speed_scale: float
    low_speed_reason: int
    ood_score: float
    ood_feature_index: int
    ood_feature_name: str
    ood_feature_value: float
    ood_feature_median: float
    ood_feature_iqr: float
    fallback_active: bool
    fallback_generating_torque_floor_applied: bool
    low_rotor_speed_fallback_active: bool
    low_rotor_speed_torque_unload_active: bool
    above_rated_initial_pitch_floor_active: bool
    above_rated_initial_pitch_floor_applied: bool
    action_projection_applied: bool
    action_rate_limited: bool
    policy_action_norm: np.ndarray
    executed_action_norm: np.ndarray
    state_residual_limit_real: np.ndarray
    optimization_state_code: int
    control_region_code: int
    pre_send_execution_mode: PreSendExecutionMode
    residual_admitted: bool
    ood_detected: bool
    actor_execution_valid: bool


def safe_action_metrics(
    safe_action: SafeAction,
    *,
    sent_action_real: np.ndarray | None = None,
) -> dict[str, float]:
    sent_action = (
        safe_action.rate_limited_action_real
        if sent_action_real is None
        else np.asarray(sent_action_real, dtype=np.float64)
    )
    if sent_action.shape != (ACTION_DIM,) or not np.isfinite(sent_action).all():
        raise ValueError('sent_action_real must contain two finite action values')
    full_chain_rate_limited = sent_action
    env_rate_limited = bool(
        np.any(np.abs(sent_action - safe_action.rate_limited_action_real) > 1.0e-8)
    )
    return {
        'control_baseline_action_torque': float(safe_action.baseline_action_real[0]),
        'control_baseline_action_pitch': float(safe_action.baseline_action_real[1]),
        'control_policy_delta_torque': float(safe_action.policy_delta_real[0]),
        'control_policy_delta_pitch': float(safe_action.policy_delta_real[1]),
        'control_pre_safety_action_torque': float(safe_action.pre_safety_action_real[0]),
        'control_pre_safety_action_pitch': float(safe_action.pre_safety_action_real[1]),
        'control_pitch_protection_floor': float(safe_action.pitch_protection_floor),
        'control_pitch_protection_state': float(safe_action.pitch_protection_state),
        'control_pitch_protection_reason': float(safe_action.pitch_protection_reason),
        'control_pitch_protection_lift': float(safe_action.pitch_protection_lift),
        'control_pitch_protection_applied': float(
            safe_action.above_rated_initial_pitch_floor_applied
        ),
        'control_post_pitch_protection_action_torque': float(
            safe_action.post_pitch_protection_action_real[0]
        ),
        'control_post_pitch_protection_action_pitch': float(
            safe_action.post_pitch_protection_action_real[1]
        ),
        'control_low_speed_stage': float(safe_action.low_speed_stage),
        'control_low_speed_scale': float(safe_action.low_speed_scale),
        'control_low_speed_reason': float(safe_action.low_speed_reason),
        'control_post_safety_action_torque': float(safe_action.post_safety_action_real[0]),
        'control_post_safety_action_pitch': float(safe_action.post_safety_action_real[1]),
        'control_projected_action_torque': float(safe_action.projected_action_real[0]),
        'control_projected_action_pitch': float(safe_action.projected_action_real[1]),
        'control_rate_limited_action_torque': float(full_chain_rate_limited[0]),
        'control_rate_limited_action_pitch': float(full_chain_rate_limited[1]),
        'control_sent_action_torque': float(sent_action[0]),
        'control_sent_action_pitch': float(sent_action[1]),
        'control_fallback_active': float(safe_action.fallback_active),
        'control_fallback_generating_torque_floor_applied': float(
            safe_action.fallback_generating_torque_floor_applied
        ),
        'control_low_rotor_speed_fallback_active': float(
            safe_action.low_rotor_speed_fallback_active
        ),
        'control_low_rotor_speed_torque_unload_active': float(
            safe_action.low_rotor_speed_torque_unload_active
        ),
        'control_above_rated_initial_pitch_floor_active': float(
            safe_action.above_rated_initial_pitch_floor_active
        ),
        'control_above_rated_initial_pitch_floor_applied': float(
            safe_action.above_rated_initial_pitch_floor_applied
        ),
        'control_action_torque': float(safe_action.action_real[0]),
        'control_action_pitch': float(safe_action.action_real[1]),
        'control_residual_torque': float(safe_action.residual_norm[0]),
        'control_residual_pitch': float(safe_action.residual_norm[1]),
        'control_policy_action_torque': float(safe_action.policy_action_norm[0]),
        'control_policy_action_pitch': float(safe_action.policy_action_norm[1]),
        'control_executed_action_torque': float(safe_action.executed_action_norm[0]),
        'control_executed_action_pitch': float(safe_action.executed_action_norm[1]),
        'control_state_residual_limit_torque': float(safe_action.state_residual_limit_real[0]),
        'control_state_residual_limit_pitch': float(safe_action.state_residual_limit_real[1]),
        'control_optimization_state_code': float(safe_action.optimization_state_code),
        'control_region_code': float(safe_action.control_region_code),
        'control_pre_send_execution_mode': float(
            pre_send_execution_mode_code(safe_action.pre_send_execution_mode)
        ),
        'control_residual_admitted': float(safe_action.residual_admitted),
        'control_ood_detected': float(safe_action.ood_detected),
        'control_actor_execution_valid': float(safe_action.actor_execution_valid),
        'control_action_projection_applied': float(safe_action.action_projection_applied),
        'control_action_rate_limited': float(safe_action.action_rate_limited or env_rate_limited),
    }


class SafeResidualController:
    def __init__(self, env_config: dict[str, Any], control_config: dict[str, Any]) -> None:
        self.action_scaler = ActionScaler(env_config['action_low'], env_config['action_high'])
        self.env_action_low = self._array(env_config, 'action_low', [0.0, 0.0], section='env')
        self.env_action_high = self._array(env_config, 'action_high', [0.0, 0.0], section='env')
        self._validate_bounds(self.env_action_low, self.env_action_high, 'env.action_low', 'env.action_high')
        self.observation_action_low = self._array(
            env_config,
            'observation_action_low',
            self.env_action_low.astype(float).tolist(),
            section='env',
        )
        self.observation_action_high = self._array(
            env_config,
            'observation_action_high',
            self.env_action_high.astype(float).tolist(),
            section='env',
        )
        self._validate_bounds(
            self.observation_action_low,
            self.observation_action_high,
            'env.observation_action_low',
            'env.observation_action_high',
        )
        self.baseline = self._array(control_config, 'baseline_action', [20_375_412.0, 0.170646548])
        self.residual_limit = self._array(control_config, 'residual_limit', [2_500_000.0, 0.07])
        self.actor_residual_target_scale = float(
            control_config.get('actor_residual_target_scale', 1.0)
        )
        if (
            not np.isfinite(self.actor_residual_target_scale)
            or not 0.0 < self.actor_residual_target_scale <= 1.0
        ):
            raise ValueError('control.actor_residual_target_scale must be finite and in (0, 1]')
        self.safe_low = self._array(control_config, 'safe_action_low', [13_000_000.0, 0.0])
        self.safe_high = self._array(control_config, 'safe_action_high', [21_000_000.0, 0.33])
        self._validate_bounds(self.safe_low, self.safe_high, 'control.safe_action_low', 'control.safe_action_high')
        if np.any(self.safe_low < self.env_action_low) or np.any(self.safe_high > self.env_action_high):
            raise ValueError('control safe action bounds must be contained within env.action_low and env.action_high')
        self.inference_torque_high = control_config.get('inference_torque_high')
        if self.inference_torque_high is not None:
            self.inference_torque_high = float(self.inference_torque_high)
            if not np.isfinite(self.inference_torque_high) or self.inference_torque_high < self.safe_high[0]:
                raise ValueError(
                    'control.inference_torque_high must be finite and no lower than control.safe_action_high[0]'
                )
        self.action_limiter_owner = str(env_config.get('action_limiter_owner', 'env')).lower()
        if self.action_limiter_owner not in {'env', 'controller'}:
            raise ValueError('env.action_limiter_owner must be env or controller')
        self.action_delta_low = self._optional_action_array(env_config.get('action_delta_low'), 'env.action_delta_low')
        self.action_delta_high = self._optional_action_array(env_config.get('action_delta_high'), 'env.action_delta_high')
        if (self.action_delta_low is None) != (self.action_delta_high is None):
            raise ValueError('env.action_delta_low and env.action_delta_high must be configured together')
        if self.action_delta_low is not None and np.any(self.action_delta_low > self.action_delta_high):
            raise ValueError('env.action_delta_low must not exceed env.action_delta_high')
        self.action_transform = PhysicalActionTransform(
            action_low=self.safe_low,
            action_high=self.safe_high,
            action_delta_low=self.action_delta_low,
            action_delta_high=self.action_delta_high,
        )
        self.action_representation = str(control_config.get('action_representation', 'residual')).lower()
        if self.action_representation != 'residual':
            raise ValueError("control.action_representation must be 'residual'")
        self.include_previous_action_observation = bool(env_config.get('include_previous_action_observation', False))
        self.include_operating_state_observation = bool(
            env_config.get('include_operating_state_observation', False)
        )
        self.state_action_artifact: StateActionEnvelopeArtifact | None = None
        self.state_action_artifact_sha256: str | None = None
        if self.include_operating_state_observation:
            if not self.include_previous_action_observation:
                raise ValueError('State-aware controller observations require previous-action observations')
            state_artifact_path = control_config.get('state_action_artifact_path')
            if not isinstance(state_artifact_path, str) or not state_artifact_path.strip():
                raise ValueError(
                    'State-aware controller observations require control.state_action_artifact_path'
                )
            path = Path(state_artifact_path.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            self.state_action_artifact = StateActionEnvelopeArtifact.from_path(path)
            self.state_action_artifact_sha256 = self.state_action_artifact.source_sha256
            env_state_artifact_path = env_config.get('state_action_artifact_path')
            if not isinstance(env_state_artifact_path, str) or not env_state_artifact_path.strip():
                raise ValueError(
                    'State-aware controller observations require env.state_action_artifact_path'
                )
            env_path = Path(env_state_artifact_path.strip()).expanduser()
            if not env_path.is_absolute():
                env_path = PROJECT_ROOT / env_path
            try:
                env_artifact_sha256 = hashlib.sha256(env_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f'Unable to read env state/action artifact: {env_path}') from exc
            if env_artifact_sha256 != self.state_action_artifact_sha256:
                raise ValueError('env and control state/action artifact hashes must match')
        if 'wind_speed_baseline_path' in control_config or 'wind_speed_baseline' in control_config:
            raise ValueError(
                'Legacy wind-speed baseline configuration is not supported; '
                'use control.region_aware_baseline_path'
            )
        self.region_aware_baseline: RegionAwareJointBaseline | None = None
        self.region_aware_baseline_schema_version: str | None = None
        self.region_aware_baseline_sha256: str | None = None
        region_aware_baseline_path = control_config.get('region_aware_baseline_path')
        if region_aware_baseline_path is not None:
            if not isinstance(region_aware_baseline_path, str) or not region_aware_baseline_path.strip():
                raise ValueError('control.region_aware_baseline_path must be a non-empty string')
            path = Path(region_aware_baseline_path.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            step_duration_s = float(env_config.get('bladed_step_duration_ms', 20.0)) / 1000.0
            self.region_aware_baseline = RegionAwareJointBaseline.from_path(
                path,
                step_duration_s=step_duration_s,
            )
            self.region_aware_baseline_schema_version = self.region_aware_baseline.schema_version
            try:
                self.region_aware_baseline_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f'Unable to hash region-aware baseline file: {path}') from exc
            if (
                self.state_action_artifact_sha256 is not None
                and self.region_aware_baseline.state_action_artifact_sha256
                != self.state_action_artifact_sha256
            ):
                raise ValueError('Region-aware baseline and state/action artifact hashes must match')
            if self.inference_torque_high is not None:
                widened_action_high = self.region_aware_baseline.action_high.copy()
                widened_action_high[0] = max(
                    float(widened_action_high[0]),
                    self.inference_torque_high,
                )
                self.region_aware_baseline.action_high = widened_action_high
            if np.any(self.region_aware_baseline.action_low < self.safe_low) or np.any(
                self.region_aware_baseline.action_high > self.safe_high
            ):
                raise ValueError('Region-aware baseline action bounds must be contained within control safe bounds')
            assert self.action_delta_low is not None
            assert self.action_delta_high is not None
            for model in self.region_aware_baseline.models.values():
                if np.any(model.action_delta_low < self.action_delta_low) or np.any(
                    model.action_delta_high > self.action_delta_high
                ):
                    raise ValueError(
                        'Region-aware baseline action delta bounds must be contained within env action delta bounds'
                    )
        self._validate_baseline_bounds()
        self.wind_obs_index = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')
        self.rotor_speed_obs_index = OBSERVATION_VARIABLES.index('I_RotorSpeed')
        self.real_power_obs_index = OBSERVATION_VARIABLES.index('I_RealPower')
        self.pitch_obs_index = OBSERVATION_VARIABLES.index('I_PitchPosition1')
        self.control_step_duration_s = float(env_config.get('bladed_step_duration_ms', 20.0)) / 1000.0
        if not np.isfinite(self.control_step_duration_s) or self.control_step_duration_s <= 0.0:
            raise ValueError('env.bladed_step_duration_ms must define a positive finite duration')
        (
            self.s2_u1_calibration_sha256,
            self.s2_u1_wind_speeds,
            self.s2_u1_schedule,
            self.s2_u1_emergency_threshold,
        ) = self._s2_u1_calibration(control_config)
        self.ood_threshold = float(control_config.get('ood_threshold', 6.0))
        self.low_speed_protection_mode = str(
            control_config.get('low_speed_protection_mode', 'legacy_zero_torque')
        ).strip().lower()
        if self.low_speed_protection_mode not in {
            'disabled', 'legacy_zero_torque', 'staged', 'pid_envelope_staged'
        }:
            raise ValueError(
                'control.low_speed_protection_mode must be disabled, legacy_zero_torque, staged, '
                'or pid_envelope_staged'
            )
        low_rotor_speed_fallback_threshold = control_config.get('low_rotor_speed_fallback_threshold')
        self.low_rotor_speed_fallback_threshold = (
            None
            if low_rotor_speed_fallback_threshold is None
            else float(low_rotor_speed_fallback_threshold)
        )
        if (
            self.low_rotor_speed_fallback_threshold is not None
            and (
                not np.isfinite(self.low_rotor_speed_fallback_threshold)
                or self.low_rotor_speed_fallback_threshold < 0.0
            )
        ):
            raise ValueError('control.low_rotor_speed_fallback_threshold must be non-negative and finite or null')
        self.low_rotor_speed_recovery_threshold = None
        self.low_rotor_speed_unload_torque = float(self.safe_low[0])
        if self.low_rotor_speed_fallback_threshold is not None:
            self.low_rotor_speed_recovery_threshold = float(
                control_config.get(
                    'low_rotor_speed_recovery_threshold',
                    self.low_rotor_speed_fallback_threshold,
                )
            )
            if (
                not np.isfinite(self.low_rotor_speed_recovery_threshold)
                or self.low_rotor_speed_recovery_threshold < self.low_rotor_speed_fallback_threshold
            ):
                raise ValueError(
                    'control.low_rotor_speed_recovery_threshold must be finite and no lower than '
                    'control.low_rotor_speed_fallback_threshold'
                )
            self.low_rotor_speed_unload_torque = float(
                control_config.get('low_rotor_speed_unload_torque', self.safe_low[0])
            )
            if not np.isfinite(self.low_rotor_speed_unload_torque) or not (
                self.safe_low[0] <= self.low_rotor_speed_unload_torque <= self.safe_high[0]
            ):
                raise ValueError(
                    'control.low_rotor_speed_unload_torque must be finite and within control safe torque bounds'
                )
        self.low_speed_warning_threshold = self._optional_non_negative_float(
            control_config.get('low_speed_warning_threshold'),
            'control.low_speed_warning_threshold',
        )
        self.low_speed_protection_threshold = self._optional_non_negative_float(
            control_config.get('low_speed_protection_threshold'),
            'control.low_speed_protection_threshold',
        )
        self.low_speed_emergency_threshold = self._optional_non_negative_float(
            control_config.get('low_speed_emergency_threshold'),
            'control.low_speed_emergency_threshold',
        )
        self.low_speed_staged_recovery_threshold = self._optional_non_negative_float(
            control_config.get('low_speed_staged_recovery_threshold'),
            'control.low_speed_staged_recovery_threshold',
        )
        self.low_speed_warning_torque_scale = float(
            control_config.get('low_speed_warning_torque_scale', 0.9)
        )
        self.low_speed_protection_torque_scale = float(
            control_config.get('low_speed_protection_torque_scale', 0.7)
        )
        if self.low_speed_protection_mode == 'staged':
            thresholds = (
                self.low_speed_warning_threshold,
                self.low_speed_protection_threshold,
                self.low_speed_emergency_threshold,
                self.low_speed_staged_recovery_threshold,
            )
            if any(value is None for value in thresholds):
                raise ValueError(
                    'staged low-speed protection requires warning, protection, emergency, and recovery thresholds'
                )
            assert self.low_speed_warning_threshold is not None
            assert self.low_speed_protection_threshold is not None
            assert self.low_speed_emergency_threshold is not None
            assert self.low_speed_staged_recovery_threshold is not None
            if not (
                self.low_speed_emergency_threshold
                < self.low_speed_protection_threshold
                < self.low_speed_warning_threshold
                < self.low_speed_staged_recovery_threshold
            ):
                raise ValueError(
                    'staged low-speed thresholds must satisfy emergency < protection < warning < recovery'
                )
            if not (
                0.0 < self.low_speed_protection_torque_scale
                < self.low_speed_warning_torque_scale
                < 1.0
            ):
                raise ValueError(
                    'staged low-speed torque scales must satisfy 0 < protection < warning < 1'
                )
        self.pitch_protection_mode = str(
            control_config.get('pitch_protection_mode', 'legacy_initial_floor')
        ).strip().lower()
        if self.pitch_protection_mode not in {
            'disabled', 'legacy_initial_floor', 'dynamic_pid_envelope'
        }:
            raise ValueError(
                'control.pitch_protection_mode must be disabled, legacy_initial_floor, '
                'or dynamic_pid_envelope'
            )
        if (
            self.pitch_protection_mode == 'dynamic_pid_envelope'
            or self.low_speed_protection_mode == 'pid_envelope_staged'
        ) and self.s2_u1_schedule is None:
            raise ValueError('dynamic PID protection modes require control.s2_u1_calibration_path')
        above_rated_initial_pitch_floor_wind_speed = control_config.get(
            'above_rated_initial_pitch_floor_wind_speed'
        )
        self.above_rated_initial_pitch_floor_wind_speed = (
            None
            if above_rated_initial_pitch_floor_wind_speed is None
            else float(above_rated_initial_pitch_floor_wind_speed)
        )
        if (
            self.above_rated_initial_pitch_floor_wind_speed is not None
            and (
                not np.isfinite(self.above_rated_initial_pitch_floor_wind_speed)
                or self.above_rated_initial_pitch_floor_wind_speed < 0.0
            )
        ):
            raise ValueError(
                'control.above_rated_initial_pitch_floor_wind_speed must be non-negative and finite or null'
            )
        low_power_threshold = control_config.get('low_power_threshold', 7_000_000.0)
        self.low_power_threshold = None if low_power_threshold is None else float(low_power_threshold)
        if self.low_power_threshold is not None and not np.isfinite(self.low_power_threshold):
            raise ValueError('control.low_power_threshold must be finite or null')
        self.low_power_patience = max(1, int(control_config.get('low_power_patience', 10)))
        self.negative_power_patience = max(1, int(control_config.get('negative_power_patience', 1)))
        self.reward_contract = self._reward_contract(env_config)
        configured_contract_fingerprint = control_config.get('resolved_control_contract_fingerprint')
        if configured_contract_fingerprint is not None:
            if (
                not isinstance(configured_contract_fingerprint, str)
                or len(configured_contract_fingerprint) != 64
                or any(character not in '0123456789abcdef' for character in configured_contract_fingerprint.lower())
            ):
                raise ValueError('control.resolved_control_contract_fingerprint must be a SHA-256 string')
        self.resolved_control_contract_fingerprint = configured_contract_fingerprint
        self.actor_control_fingerprint = self._actor_control_fingerprint()
        self.online_training_contract = self._online_training_contract(env_config)
        self.online_training_fingerprint = self._fingerprint_contract(
            {
                'actor_control_fingerprint': self.actor_control_fingerprint,
                'online_training_contract': self.online_training_contract,
            }
        )
        self.control_fingerprint = self._control_fingerprint()
        self.obs_median: np.ndarray | None = None
        self.obs_scale: np.ndarray | None = None
        self.low_power_steps = 0
        self.negative_power_steps = 0
        self._last_action_real: np.ndarray | None = None
        self._last_non_emergency_action_real: np.ndarray | None = None
        self._region_baseline_recovery_previous_action: np.ndarray | None = None
        self._region_baseline_cached_obs: np.ndarray | None = None
        self._region_baseline_cached_previous_action: np.ndarray | None = None
        self._region_baseline_cached_action: np.ndarray | None = None
        self._low_rotor_speed_recovery_active = False
        self._low_speed_stage = 0
        self._above_rated_initial_pitch_floor: float | None = None
        self._dynamic_pitch_floor: float | None = None
        self._dynamic_pitch_reason = 0
        self._previous_rotor_speed: float | None = None
        self._state_classifier = None
        self._state_timestamp_s = 0.0

    @staticmethod
    def _array(
        config: dict[str, Any],
        key: str,
        default: list[float],
        *,
        section: str = 'control',
    ) -> np.ndarray:
        value = np.asarray(config.get(key, default), dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f'{section}.{key} must contain two finite values')
        return value

    @staticmethod
    def _validate_bounds(low: np.ndarray, high: np.ndarray, low_name: str, high_name: str) -> None:
        if np.any(low > high):
            raise ValueError(f'{low_name} must not exceed {high_name}')

    @staticmethod
    def _optional_action_array(values: Any, field_name: str) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError(f'{field_name} must contain two finite values')
        return array

    @staticmethod
    def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f'{field_name} must be non-negative and finite or null')
        return parsed

    @staticmethod
    def _s2_u1_calibration(
        config: dict[str, Any],
    ) -> tuple[str | None, np.ndarray | None, dict[str, np.ndarray] | None, float]:
        configured_path = config.get('s2_u1_calibration_path')
        if configured_path is None:
            return None, None, None, 0.4
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError('control.s2_u1_calibration_path must be a non-empty string')
        path = Path(configured_path.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            payload = path.read_bytes()
            document = json.loads(payload)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'S2/U1 calibration file does not exist: {path}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read S2/U1 calibration file: {path}') from exc
        if document.get('schema_version') != S2_U1_CALIBRATION_SCHEMA_VERSION:
            raise ValueError('S2/U1 calibration schema is incompatible')
        rows = document.get('schedule')
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError('S2/U1 calibration schedule must contain at least two rows')
        wind_speeds: list[float] = []
        fields = {
            's2_rotor_enter': [], 's2_rotor_release': [],
            's2_power_enter': [], 's2_power_release': [],
            's2_accel_enter': [], 's2_accel_release': [],
            's2_pitch_p05': [],
            'u1_torque_p05': [],
            'u1_warning_speed': [], 'u1_protection_speed': [], 'u1_recovery_speed': [],
            'u1_warning_accel': [], 'u1_protection_accel': [],
        }
        for row in rows:
            try:
                s2 = row['s2']
                u1 = row['u1']
                wind_speeds.append(float(row['wind_speed']))
                values = {
                    's2_rotor_enter': s2['rotor_speed_enter_rad_s'],
                    's2_rotor_release': s2['rotor_speed_release_rad_s'],
                    's2_power_enter': s2['power_enter_w'],
                    's2_power_release': s2['power_release_w'],
                    's2_accel_enter': s2['acceleration_enter_rad_s2'],
                    's2_accel_release': s2['acceleration_release_rad_s2'],
                    's2_pitch_p05': s2['pitch_p05_rad'],
                    'u1_torque_p05': u1['pid_torque_p05_nm'],
                    'u1_warning_speed': u1['warning_speed_rad_s'],
                    'u1_protection_speed': u1['protection_speed_rad_s'],
                    'u1_recovery_speed': u1['recovery_speed_rad_s'],
                    'u1_warning_accel': u1['warning_acceleration_rad_s2'],
                    'u1_protection_accel': u1['protection_acceleration_rad_s2'],
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('S2/U1 calibration row is incomplete') from exc
            for key, value in values.items():
                fields[key].append(float(value))
        winds = np.asarray(wind_speeds, dtype=np.float64)
        schedule = {key: np.asarray(values, dtype=np.float64) for key, values in fields.items()}
        if np.any(np.diff(winds) <= 0.0) or not np.isfinite(winds).all() or not all(
            np.isfinite(values).all() for values in schedule.values()
        ):
            raise ValueError('S2/U1 calibration values must be finite with increasing wind speeds')
        if np.any(schedule['s2_rotor_release'] >= schedule['s2_rotor_enter']):
            raise ValueError('S2 rotor release thresholds must be below enter thresholds')
        if np.any(schedule['s2_power_release'] >= schedule['s2_power_enter']):
            raise ValueError('S2 power release thresholds must be below enter thresholds')
        if np.any(schedule['s2_accel_release'] >= schedule['s2_accel_enter']):
            raise ValueError('S2 acceleration release thresholds must be below enter thresholds')
        if np.any(~(
            (schedule['u1_protection_speed'] < schedule['u1_warning_speed'])
            & (schedule['u1_warning_speed'] < schedule['u1_recovery_speed'])
        )):
            raise ValueError('U1 envelope thresholds must satisfy protection < warning < recovery')
        emergency = float(document.get('metadata', {}).get('emergency_threshold_rad_s', 0.4))
        if not np.isfinite(emergency) or emergency < 0.0:
            raise ValueError('S2/U1 emergency threshold must be non-negative and finite')
        return hashlib.sha256(payload).hexdigest(), winds, schedule, emergency

    def _validate_baseline_bounds(self) -> None:
        if np.any(self.baseline < self.safe_low) or np.any(self.baseline > self.safe_high):
            raise ValueError('control.baseline_action must be contained within control safe action bounds')

    @staticmethod
    def _reward_contract(env_config: dict[str, Any]) -> dict[str, Any]:
        reward_config = env_config.get('reward', {})
        if reward_config is None:
            reward_config = {}
        if not isinstance(reward_config, dict):
            raise ValueError('env.reward must be an object when configured')
        contract: dict[str, Any] = {
            'reward': reward_config,
            'rated_power': env_config.get('rated_power'),
            'bladed_step_duration_ms': env_config.get('bladed_step_duration_ms'),
            'episode_length': env_config.get('episode_length'),
        }
        for path_key, hash_key, label in (
            (
                'blade_root_peak_reference_path',
                'blade_root_peak_reference_sha256',
                'Blade-root peak reference',
            ),
            (
                'pid_blade_root_peak_reference_path',
                'pid_blade_root_peak_reference_sha256',
                'PID blade-root peak reference',
            ),
            (
                'pid_power_reference_manifest_path',
                'pid_power_reference_manifest_sha256',
                'Paired PID power reference manifest',
            ),
            (
                'paired_pid_reference_path',
                'paired_pid_reference_sha256',
                'Paired PID power/load reference',
            ),
            # Legacy configurations may still bind this artifact in the control
            # fingerprint; retaining the hash contract does not add a reward term.
            (
                'rotor_speed_reference_schedule_path',
                'rotor_speed_reference_schedule_sha256',
                'PID rotor-speed reference schedule',
            ),
        ):
            reference_path = reward_config.get(path_key)
            if reference_path is None:
                continue
            if not isinstance(reference_path, str) or not reference_path.strip():
                raise ValueError(f'reward.{path_key} must be a non-empty string')
            path = Path(reference_path.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            try:
                artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError as exc:
                raise FileNotFoundError(f'{label} file does not exist: {path}') from exc
            except OSError as exc:
                raise ValueError(f'Unable to read {label.lower()} file: {path}') from exc
            contract[hash_key] = artifact_hash
        return contract

    def _control_fingerprint(self) -> str:
        schedule = []
        contract = {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'action_variables': ACTION_VARIABLES,
            'baseline_action': self._fingerprint_action_values(self.baseline),
            'wind_speed_baseline': schedule,
            'region_aware_baseline_schema_version': self.region_aware_baseline_schema_version,
            'region_aware_baseline_sha256': self.region_aware_baseline_sha256,
            'residual_limit': self._fingerprint_action_values(self.residual_limit),
            'actor_residual_target_scale': self.actor_residual_target_scale,
            'safe_action_low': self._fingerprint_action_values(self.safe_low),
            'safe_action_high': self._fingerprint_action_values(self.safe_high),
            'env_action_low': self._fingerprint_action_values(self.env_action_low),
            'env_action_high': self._fingerprint_action_values(self.env_action_high),
            'action_limiter_owner': self.action_limiter_owner,
            'action_delta_low': None if self.action_delta_low is None else self._fingerprint_action_values(self.action_delta_low),
            'action_delta_high': None if self.action_delta_high is None else self._fingerprint_action_values(self.action_delta_high),
            'action_representation': self.action_representation,
            'include_previous_action_observation': self.include_previous_action_observation,
            'low_rotor_speed_fallback_threshold': self.low_rotor_speed_fallback_threshold,
            'low_rotor_speed_recovery_threshold': self.low_rotor_speed_recovery_threshold,
            'low_rotor_speed_unload_torque': self.low_rotor_speed_unload_torque,
            'low_speed_protection_mode': self.low_speed_protection_mode,
            'low_speed_warning_threshold': self.low_speed_warning_threshold,
            'low_speed_protection_threshold': self.low_speed_protection_threshold,
            'low_speed_emergency_threshold': self.low_speed_emergency_threshold,
            'low_speed_staged_recovery_threshold': self.low_speed_staged_recovery_threshold,
            'low_speed_warning_torque_scale': self.low_speed_warning_torque_scale,
            'low_speed_protection_torque_scale': self.low_speed_protection_torque_scale,
            'pitch_protection_mode': self.pitch_protection_mode,
            'above_rated_initial_pitch_floor_wind_speed': self.above_rated_initial_pitch_floor_wind_speed,
            's2_u1_calibration_sha256': self.s2_u1_calibration_sha256,
            'reward_contract': self.reward_contract,
        }
        if self.state_action_artifact_sha256 is not None:
            contract['state_action_artifact_sha256'] = self.state_action_artifact_sha256
        if self.resolved_control_contract_fingerprint is not None:
            contract['resolved_control_contract_fingerprint'] = self.resolved_control_contract_fingerprint
        encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _actor_control_fingerprint(self) -> str:
        schedule = []
        contract = {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'action_variables': ACTION_VARIABLES,
            'baseline_action': self._fingerprint_action_values(self.baseline),
            'wind_speed_baseline': schedule,
            'region_aware_baseline_schema_version': self.region_aware_baseline_schema_version,
            'region_aware_baseline_sha256': self.region_aware_baseline_sha256,
            'residual_limit': self._fingerprint_action_values(self.residual_limit),
            'actor_residual_target_scale': self.actor_residual_target_scale,
            'safe_action_low': self._fingerprint_action_values(self.safe_low),
            'safe_action_high': self._fingerprint_action_values(self.safe_high),
            'env_action_low': self._fingerprint_action_values(self.env_action_low),
            'env_action_high': self._fingerprint_action_values(self.env_action_high),
            'action_limiter_owner': self.action_limiter_owner,
            'action_delta_low': None if self.action_delta_low is None else self._fingerprint_action_values(self.action_delta_low),
            'action_delta_high': None if self.action_delta_high is None else self._fingerprint_action_values(self.action_delta_high),
            'action_representation': self.action_representation,
            'include_previous_action_observation': self.include_previous_action_observation,
        }
        if self.state_action_artifact_sha256 is not None:
            contract['state_action_artifact_sha256'] = self.state_action_artifact_sha256
        if self.resolved_control_contract_fingerprint is not None:
            contract['resolved_control_contract_fingerprint'] = self.resolved_control_contract_fingerprint
        return self._fingerprint_contract(contract)

    def _fingerprint_action_values(self, values: np.ndarray) -> list[float]:
        array = np.asarray(values, dtype=np.float64)
        if self.resolved_control_contract_fingerprint is None:
            array = array.astype(np.float32)
        return array.astype(float).tolist()

    def checkpoint_contract_required(self, *, baseline_only: bool = False) -> bool:
        if baseline_only and self.region_aware_baseline is not None:
            return False
        return self.region_aware_baseline is not None or self.state_action_artifact is not None

    @staticmethod
    def _online_training_contract(env_config: dict[str, Any]) -> dict[str, Any]:
        return {
            'reward_contract': SafeResidualController._reward_contract(env_config),
            'queue_lifecycle_enabled': bool(env_config.get('queue_lifecycle_enabled', False)),
            'udp_protocol_version': env_config.get('udp_protocol_version'),
            'require_condition_metadata': bool(env_config.get('require_condition_metadata', False)),
            'rollout_segment_steps': env_config.get('rollout_segment_steps'),
        }

    @staticmethod
    def _fingerprint_contract(contract: dict[str, Any]) -> str:
        encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _baseline_for_obs(
        self,
        obs: np.ndarray,
        *,
        previous_action_real: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.region_aware_baseline is not None:
            observation = np.asarray(obs, dtype=np.float32)
            previous_action = previous_action_real
            if previous_action is None:
                previous_action = self._region_baseline_recovery_previous_action
            if previous_action is None:
                previous_action = self._last_action_real
            if previous_action is None:
                previous_action = self.region_aware_baseline.previous_action_from_observation(observation)
            if (
                self._region_baseline_cached_obs is not None
                and self._region_baseline_cached_previous_action is not None
                and self._region_baseline_cached_action is not None
                and np.array_equal(observation, self._region_baseline_cached_obs)
                and np.array_equal(previous_action, self._region_baseline_cached_previous_action)
            ):
                return self._region_baseline_cached_action
            action = self.region_aware_baseline.predict(
                observation,
                previous_action_real=previous_action,
            )
            self._region_baseline_cached_obs = observation.copy()
            self._region_baseline_cached_previous_action = np.asarray(
                previous_action,
                dtype=np.float64,
            ).copy()
            self._region_baseline_cached_action = action.copy()
            return action
        return self.baseline

    def _baseline_tensor(self, obs: torch.Tensor | None, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        fixed_baseline = torch.as_tensor(self.baseline, dtype=dtype, device=device)
        return fixed_baseline.expand(batch_size, -1)

    def set_observation_distribution(self, median: np.ndarray, scale: np.ndarray) -> None:
        self.obs_median = np.asarray(median, dtype=np.float32)
        self.obs_scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)

    def baseline_action_for_obs(self, obs: np.ndarray) -> np.ndarray:
        return self._baseline_for_obs(obs).copy()

    def _baseline_previous_action_for_stage(
        self,
        low_speed_stage: int,
        previous_action: np.ndarray,
        wind_speed: float,
    ) -> np.ndarray:
        if self.region_aware_baseline is None:
            return previous_action
        if self._region_baseline_recovery_previous_action is not None:
            return self._region_baseline_recovery_previous_action.copy()
        if low_speed_stage in {3, 4}:
            if self._last_non_emergency_action_real is not None:
                recovery_anchor = self._last_non_emergency_action_real.copy()
                if self.s2_u1_schedule is not None:
                    recovery_anchor[1] = min(
                        recovery_anchor[1],
                        self._envelope_value('s2_pitch_p05', wind_speed),
                    )
                return recovery_anchor
        return previous_action

    def _apply_generating_fallback_torque_floor(
        self,
        baseline_action: np.ndarray,
        *,
        wind_speed: float,
        rotor_speed: float,
        real_power: float,
        low_speed_stage: int,
        fallback_active: bool,
        baseline_action_was_explicit: bool,
    ) -> tuple[np.ndarray, bool]:
        if (
            self.region_aware_baseline is None
            or not fallback_active
            or baseline_action_was_explicit
            or low_speed_stage != 0
            or self.low_rotor_speed_recovery_threshold is None
            or rotor_speed < self.low_rotor_speed_recovery_threshold
            or real_power >= 100_000.0
            or self.s2_u1_schedule is None
        ):
            return baseline_action, False
        torque_floor = self._envelope_value('u1_torque_p05', wind_speed)
        if baseline_action[0] >= torque_floor:
            return baseline_action, False
        floored = baseline_action.copy()
        floored[0] = np.clip(torque_floor, self.safe_low[0], self.safe_high[0])
        return floored, True

    def _update_recovery_baseline_history(
        self,
        low_speed_stage: int,
        baseline_action: np.ndarray,
        action_real: np.ndarray,
        wind_speed: float,
    ) -> None:
        if self.region_aware_baseline is None:
            return
        if low_speed_stage == 3 and self._region_baseline_recovery_previous_action is None:
            recovery_anchor = self._last_non_emergency_action_real
            if recovery_anchor is None:
                recovery_anchor = baseline_action
            recovery_anchor = recovery_anchor.copy()
            if self.s2_u1_schedule is not None:
                recovery_anchor[1] = min(
                    recovery_anchor[1],
                    self._envelope_value('s2_pitch_p05', wind_speed),
                )
            self._region_baseline_recovery_previous_action = recovery_anchor
        if low_speed_stage == 0:
            self._region_baseline_recovery_previous_action = None
        if low_speed_stage != 3 and self._region_baseline_recovery_previous_action is None:
            self._last_non_emergency_action_real = action_real.copy()

    def reset_condition(
        self,
        obs: np.ndarray | None = None,
        *,
        previous_action_real: np.ndarray | None = None,
    ) -> None:
        self.low_power_steps = 0
        self.negative_power_steps = 0
        self._clear_region_baseline_cache()
        if self.region_aware_baseline is not None:
            self.region_aware_baseline.reset()
        self._state_classifier = (
            None if self.state_action_artifact is None else self.state_action_artifact.classifier()
        )
        self._state_timestamp_s = 0.0
        self._low_rotor_speed_recovery_active = False
        self._low_speed_stage = 0
        self._last_non_emergency_action_real = None
        self._region_baseline_recovery_previous_action = None
        self._above_rated_initial_pitch_floor = None
        self._dynamic_pitch_floor = None
        self._dynamic_pitch_reason = 0
        self._previous_rotor_speed = None
        if (
            self.pitch_protection_mode == 'legacy_initial_floor'
            and self.above_rated_initial_pitch_floor_wind_speed is not None
            and obs is not None
        ):
            initial_observation = np.asarray(obs, dtype=np.float32)
            if (
                initial_observation.ndim != 1
                or initial_observation.shape[0] <= max(self.wind_obs_index, self.pitch_obs_index)
                or not np.isfinite(initial_observation[[self.wind_obs_index, self.pitch_obs_index]]).all()
            ):
                raise ValueError('above-rated initial pitch floor requires finite wind and pitch observations')
            if initial_observation[self.wind_obs_index] >= self.above_rated_initial_pitch_floor_wind_speed:
                self._above_rated_initial_pitch_floor = float(
                    np.clip(initial_observation[self.pitch_obs_index], self.safe_low[1], self.safe_high[1])
                )
        if previous_action_real is None:
            if obs is None:
                self._last_action_real = None
            elif self.region_aware_baseline is None:
                self._last_action_real = self.baseline_action_for_obs(obs)
            else:
                self._last_action_real = self.region_aware_baseline.previous_action_from_observation(obs)
            self._last_non_emergency_action_real = (
                None if self._last_action_real is None else self._last_action_real.copy()
            )
            return
        previous = np.asarray(previous_action_real, dtype=np.float64)
        if previous.shape != (2,) or not np.isfinite(previous).all():
            raise ValueError('previous_action_real must contain two finite action values')
        self._last_action_real = np.clip(previous, self.safe_low, self.safe_high).astype(np.float64)
        self._last_non_emergency_action_real = self._last_action_real.copy()

    def _clear_region_baseline_cache(self) -> None:
        self._region_baseline_cached_obs = None
        self._region_baseline_cached_previous_action = None
        self._region_baseline_cached_action = None

    def reset_episode(self) -> None:
        """Backward-compatible alias for legacy episode-based callers."""
        self.reset_condition()

    def _control_point(self, obs: np.ndarray) -> tuple[float, float, float, float, float]:
        observation = np.asarray(obs, dtype=np.float32)
        rotor_speed = float(observation[self.rotor_speed_obs_index])
        dynamic_mode = (
            self.pitch_protection_mode == 'dynamic_pid_envelope'
            or self.low_speed_protection_mode == 'pid_envelope_staged'
        )
        required_index = max(self.wind_obs_index, self.real_power_obs_index, self.pitch_obs_index)
        if dynamic_mode and observation.size <= required_index:
            raise ValueError('dynamic PID protection modes require the complete control observation')
        wind_speed = float(observation[self.wind_obs_index]) if observation.size > self.wind_obs_index else 0.0
        real_power = (
            float(observation[self.real_power_obs_index])
            if observation.size > self.real_power_obs_index
            else 0.0
        )
        pitch = float(observation[self.pitch_obs_index]) if observation.size > self.pitch_obs_index else 0.0
        acceleration = 0.0
        if self._previous_rotor_speed is not None:
            acceleration = (rotor_speed - self._previous_rotor_speed) / self.control_step_duration_s
        self._previous_rotor_speed = rotor_speed
        return wind_speed, rotor_speed, real_power, pitch, acceleration

    def _state_classification(
        self,
        *,
        wind_speed: float,
        rotor_speed: float,
        real_power: float,
        pitch: float,
        previous_action: np.ndarray,
    ):
        if self.state_action_artifact is None:
            return None
        if self._state_classifier is None:
            self._state_classifier = self.state_action_artifact.classifier()
            self._state_timestamp_s = 0.0
        classification = self._state_classifier.classify(OperatingPoint(
            timestamp_s=self._state_timestamp_s,
            wind_speed_mps=wind_speed,
            rotor_speed_rad_s=rotor_speed,
            real_power_w=real_power,
            pitch_rad=pitch,
            previous_torque_nm=float(previous_action[0]),
            previous_pitch_rad=float(previous_action[1]),
        ))
        self._state_timestamp_s += self.control_step_duration_s
        return classification

    def _state_residual_limit(self, classification) -> np.ndarray:
        if self.state_action_artifact is None or classification is None:
            return self.residual_limit
        return self.state_action_artifact.residual_limit_for(classification)

    def _state_residual_limit_tensor(
        self,
        obs: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.state_action_artifact is None:
            return torch.as_tensor(self.residual_limit, dtype=dtype, device=device).expand(batch_size, -1)
        if (
            obs is None
            or obs.ndim != 2
            or obs.shape[0] != batch_size
            or obs.shape[1] < STATE_AWARE_CONTROL_OBS_DIM
        ):
            raise ValueError('State-aware residual targets require a state-aware observation batch')
        states = self.state_action_artifact.document['soft_residual_envelopes']['states']
        limits = torch.as_tensor(
            [states[state.value]['residual_limit_abs'] for state in OPTIMIZATION_STATE_ENCODING_ORDER],
            dtype=dtype,
            device=device,
        )
        state_slice = obs[:, CONTROL_OBS_DIM:CONTROL_OBS_DIM + len(OPTIMIZATION_STATE_ENCODING_ORDER)]
        state_indices = torch.argmax(state_slice, dim=1)
        return limits[state_indices]

    def _envelope_value(self, field: str, wind_speed: float) -> float:
        assert self.s2_u1_wind_speeds is not None
        assert self.s2_u1_schedule is not None
        return float(np.interp(wind_speed, self.s2_u1_wind_speeds, self.s2_u1_schedule[field]))

    def _low_speed_protection(
        self,
        rotor_speed: float,
        wind_speed: float = 0.0,
        real_power: float = 0.0,
        acceleration: float = 0.0,
    ) -> tuple[int, float, int, bool, bool]:
        if self.low_speed_protection_mode == 'disabled':
            self._low_rotor_speed_recovery_active = False
            self._low_speed_stage = 0
            return 0, 1.0, 0, False, False
        if self.low_speed_protection_mode == 'legacy_zero_torque':
            if self.low_rotor_speed_fallback_threshold is not None:
                if self._low_rotor_speed_recovery_active:
                    if rotor_speed >= self.low_rotor_speed_recovery_threshold:
                        self._low_rotor_speed_recovery_active = False
                elif rotor_speed < self.low_rotor_speed_fallback_threshold:
                    self._low_rotor_speed_recovery_active = True
            active = self._low_rotor_speed_recovery_active
            self._low_speed_stage = int(active)
            reason = (
                1
                if active and rotor_speed < self.low_rotor_speed_fallback_threshold
                else 2 if active else 0
            )
            return self._low_speed_stage, 0.0 if active else 1.0, reason, active, active

        if self.low_speed_protection_mode == 'pid_envelope_staged':
            warning_speed = self._envelope_value('u1_warning_speed', wind_speed)
            protection_speed = self._envelope_value('u1_protection_speed', wind_speed)
            recovery_speed = self._envelope_value('u1_recovery_speed', wind_speed)
            warning_accel = self._envelope_value('u1_warning_accel', wind_speed)
            protection_accel = self._envelope_value('u1_protection_accel', wind_speed)
            generating = real_power >= 100_000.0
            if (
                rotor_speed <= self.s2_u1_emergency_threshold
                and acceleration <= 0.0
            ):
                stage = 3
                reason = 7
            elif generating and rotor_speed <= protection_speed and acceleration <= protection_accel:
                stage = 2
                reason = 6
            elif generating and rotor_speed <= warning_speed and acceleration <= warning_accel:
                stage = 1
                reason = 5
            elif self._low_speed_stage > 0 and rotor_speed < recovery_speed:
                stage = 4
                reason = 4
            else:
                stage = 0
                reason = 0
            self._low_speed_stage = stage
            self._low_rotor_speed_recovery_active = stage > 0
            scale = (
                self.low_speed_warning_torque_scale
                if stage == 1
                else self.low_speed_protection_torque_scale if stage == 2 else 0.0 if stage == 3 else 1.0
            )
            return stage, scale, reason, stage > 0, stage == 3

        assert self.low_speed_warning_threshold is not None
        assert self.low_speed_protection_threshold is not None
        assert self.low_speed_emergency_threshold is not None
        assert self.low_speed_staged_recovery_threshold is not None
        if rotor_speed >= self.low_speed_staged_recovery_threshold:
            stage = 0
        elif rotor_speed <= self.low_speed_emergency_threshold:
            stage = 3
        elif rotor_speed <= self.low_speed_protection_threshold:
            stage = 2
        elif rotor_speed <= self.low_speed_warning_threshold:
            stage = 1
        elif self._low_speed_stage > 0:
            stage = 4
        else:
            stage = 0
        self._low_speed_stage = stage
        self._low_rotor_speed_recovery_active = stage > 0
        scale = (
            self.low_speed_warning_torque_scale
            if stage == 1
            else self.low_speed_protection_torque_scale if stage == 2 else 0.0 if stage == 3 else 1.0
        )
        return stage, scale, stage, stage > 0, stage == 3

    def _pitch_protection(
        self,
        wind_speed: float,
        rotor_speed: float,
        real_power: float,
        pitch: float,
        acceleration: float,
    ) -> tuple[float | None, int]:
        if self.pitch_protection_mode != 'dynamic_pid_envelope':
            return self._above_rated_initial_pitch_floor, int(self._above_rated_initial_pitch_floor is not None)
        if wind_speed < 11.2:
            self._dynamic_pitch_floor = None
            self._dynamic_pitch_reason = 0
            return None, 0
        enter_reason = 0
        if rotor_speed >= self._envelope_value('s2_rotor_enter', wind_speed):
            enter_reason |= 1
        if real_power >= self._envelope_value('s2_power_enter', wind_speed):
            enter_reason |= 2
        if acceleration >= self._envelope_value('s2_accel_enter', wind_speed):
            enter_reason |= 4
        release = (
            rotor_speed <= self._envelope_value('s2_rotor_release', wind_speed)
            and real_power <= self._envelope_value('s2_power_release', wind_speed)
            and acceleration <= self._envelope_value('s2_accel_release', wind_speed)
        )
        if self._dynamic_pitch_floor is None and enter_reason:
            self._dynamic_pitch_floor = float(np.clip(pitch, self.safe_low[1], self.safe_high[1]))
            self._dynamic_pitch_reason = enter_reason
        elif self._dynamic_pitch_floor is not None and release:
            self._dynamic_pitch_floor = None
            self._dynamic_pitch_reason = 0
        elif self._dynamic_pitch_floor is not None:
            self._dynamic_pitch_reason = enter_reason or 8
        return self._dynamic_pitch_floor, self._dynamic_pitch_reason

    def _apply_low_speed_torque(
        self,
        desired_action_real: np.ndarray,
        stage: int,
        scale: float,
    ) -> np.ndarray:
        if stage not in {1, 2, 3, 4}:
            return desired_action_real
        protected = desired_action_real.copy()
        if stage == 3:
            protected[0] = self.low_rotor_speed_unload_torque
        else:
            protected[0] = max(self.safe_low[0], protected[0] * scale)
        return protected

    def ood_details(self, obs: np.ndarray) -> tuple[int, str, float, float, float, float]:
        if self.obs_median is None or self.obs_scale is None:
            return -1, 'unavailable', 0.0, 0.0, 1.0, 0.0
        values = np.asarray(obs, dtype=np.float32)
        scaled = np.abs((values - self.obs_median) / self.obs_scale)
        scaled = np.nan_to_num(scaled, nan=np.inf, posinf=np.inf, neginf=np.inf)
        if values.shape[-1] >= STATE_AWARE_CONTROL_OBS_DIM:
            regime_start = CONTROL_OBS_DIM
            regime_end = regime_start + len(STATE_REGIME_OBSERVATION_VARIABLES)
            regime = values[regime_start:regime_end]
            state_width = len(OPTIMIZATION_STATE_ENCODING_ORDER)
            state_values = regime[:state_width]
            region_values = regime[state_width:]
            valid_regime = (
                np.isfinite(regime).all()
                and np.all(regime >= -1.0e-5)
                and np.all(regime <= 1.0 + 1.0e-5)
                and np.isclose(np.sum(state_values), 1.0, atol=1.0e-4)
                and np.isclose(np.sum(region_values), 1.0, atol=1.0e-4)
            )
            # Regime one-hot values are categorical, not continuous distances.
            scaled[regime_start:regime_end] = 0.0
            if not valid_regime:
                scaled[regime_start] = np.inf
        feature_index = int(np.argmax(scaled))
        feature_names = list(OBSERVATION_VARIABLES)
        if values.shape[-1] >= CONTROL_OBS_DIM:
            feature_names.extend(PREVIOUS_ACTION_OBSERVATION_VARIABLES)
        if values.shape[-1] >= STATE_AWARE_CONTROL_OBS_DIM:
            feature_names.extend(
                f'optimization_state_{name}' for name in OPTIMIZATION_STATE_ENCODING_ORDER
            )
            feature_names.extend(
                f'control_region_{name}' for name in CONTROL_REGION_ENCODING_ORDER
            )
        feature_name = (
            feature_names[feature_index]
            if feature_index < len(feature_names)
            else f'obs_{feature_index}'
        )
        return (
            feature_index,
            feature_name,
            float(values[feature_index]),
            float(self.obs_median[feature_index]),
            float(self.obs_scale[feature_index]),
            float(scaled[feature_index]),
        )

    def ood_score(self, obs: np.ndarray) -> float:
        return self.ood_details(obs)[-1]

    def residual_target_tensor(self, actions_real: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        baseline = self._baseline_tensor(obs, actions_real.shape[0], actions_real.dtype, actions_real.device)
        limit = self._state_residual_limit_tensor(
            obs,
            actions_real.shape[0],
            actions_real.dtype,
            actions_real.device,
        )
        normalized = (actions_real - baseline) / torch.clamp(limit, min=1.0e-12)
        normalized = torch.where(limit > 0.0, normalized, torch.zeros_like(normalized))
        return (
            normalized.clamp(-1.0, 1.0)
            * self.actor_residual_target_scale
        )

    def action_target_tensor(
        self,
        actions_real: torch.Tensor,
        *,
        obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.residual_target_tensor(actions_real, obs)

    def action_real_tensor(self, residual_norm: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        baseline = self._baseline_tensor(obs, residual_norm.shape[0], residual_norm.dtype, residual_norm.device)
        limit = self._state_residual_limit_tensor(
            obs,
            residual_norm.shape[0],
            residual_norm.dtype,
            residual_norm.device,
        )
        safe_low = torch.as_tensor(self.safe_low, dtype=residual_norm.dtype, device=residual_norm.device)
        safe_high = torch.as_tensor(self.safe_high, dtype=residual_norm.dtype, device=residual_norm.device)
        residual = residual_norm.clamp(
            -self.actor_residual_target_scale,
            self.actor_residual_target_scale,
        )
        return (baseline + residual * limit).clamp(safe_low, safe_high)

    def residual_target_numpy(
        self,
        action_real: np.ndarray,
        baseline_action_real: np.ndarray,
        residual_limit: np.ndarray | None = None,
    ) -> np.ndarray:
        limit = self.residual_limit if residual_limit is None else np.asarray(residual_limit, dtype=np.float64)
        difference = np.asarray(action_real, dtype=np.float64) - np.asarray(
            baseline_action_real,
            dtype=np.float64,
        )
        normalized = np.divide(
            difference,
            np.maximum(limit, 1.0e-12),
            out=np.zeros_like(difference),
            where=limit > 0.0,
        )
        return np.clip(
            normalized,
            -1.0,
            1.0,
        ).astype(np.float32)

    def project_action(
        self,
        desired_action_real: np.ndarray,
        *,
        previous_action_real: np.ndarray | None = None,
        update_history: bool = True,
        bypass_torque_decrease_rate_limit: bool = False,
    ) -> tuple[np.ndarray, bool, bool]:
        previous = self._last_action_real if previous_action_real is None else previous_action_real
        projection = self.action_transform.project(
            desired_action_real,
            previous_action_real=previous,
            bypass_torque_decrease_rate_limit=bypass_torque_decrease_rate_limit,
        )
        if update_history:
            self._last_action_real = projection.action_real.copy()
            self._clear_region_baseline_cache()
        return projection.action_real, projection.projection_applied, projection.rate_limited

    def action_from_residual(
        self,
        residual_norm: np.ndarray,
        obs: np.ndarray,
        allow_residual: bool,
        enforce_action_rate_limit: bool | None = None,
        baseline_action_real: np.ndarray | None = None,
        fallback_action_real: np.ndarray | None = None,
        baseline_execution_mode: PreSendExecutionMode | str = PreSendExecutionMode.EXPLICIT_BASELINE_STAGE,
        actor_execution_valid: bool = True,
    ) -> SafeAction:
        residual_norm = np.nan_to_num(np.asarray(residual_norm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        residual_norm = np.clip(
            residual_norm,
            -self.actor_residual_target_scale,
            self.actor_residual_target_scale,
        )
        requested_residual_norm = residual_norm.copy()
        (
            ood_feature_index,
            ood_feature_name,
            ood_feature_value,
            ood_feature_median,
            ood_feature_iqr,
            score,
        ) = self.ood_details(obs)
        wind_speed, rotor_speed, real_power, pitch, rotor_acceleration = self._control_point(obs)
        previous_action = self._last_action_real
        if previous_action is None:
            if np.asarray(obs).shape[-1] >= CONTROL_OBS_DIM:
                previous_action = decode_previous_action_observation(
                    obs,
                    self.observation_action_low,
                    self.observation_action_high,
                )
            else:
                previous_action = self._baseline_for_obs(obs)
        previous_action = np.asarray(previous_action, dtype=np.float64)
        if previous_action.shape != (ACTION_DIM,) or not np.isfinite(previous_action).all():
            raise ValueError('Previous action history is invalid')
        state_classification = self._state_classification(
            wind_speed=wind_speed,
            rotor_speed=rotor_speed,
            real_power=real_power,
            pitch=pitch,
            previous_action=previous_action,
        )
        state_residual_limit = self._state_residual_limit(state_classification)
        low_speed_stage, low_speed_scale, low_speed_reason, low_rotor_speed_fallback_active, bypass_torque_rate = (
            self._low_speed_protection(
                rotor_speed,
                wind_speed,
                real_power,
                rotor_acceleration,
            )
        )
        ood_detected = score > self.ood_threshold
        requested_baseline_mode = pre_send_execution_mode(baseline_execution_mode)
        if requested_baseline_mode == PreSendExecutionMode.ACTOR:
            raise ValueError('baseline_execution_mode cannot be actor')
        if state_classification is not None and (
            state_classification.optimization_state == OptimizationState.NEGATIVE_POWER_ABNORMAL
        ):
            execution_mode = PreSendExecutionMode.NEGATIVE_POWER_ABNORMAL_BASELINE
        elif state_classification is not None and (
            state_classification.optimization_state == OptimizationState.STARTUP_STOPPED
        ):
            execution_mode = PreSendExecutionMode.STARTUP_STOPPED_BASELINE
        elif low_speed_stage == 3:
            execution_mode = PreSendExecutionMode.LOW_SPEED_EMERGENCY_HOLD
        elif low_speed_stage == 2:
            execution_mode = PreSendExecutionMode.LOW_SPEED_PROTECTION_HOLD
        elif low_speed_stage == 1:
            execution_mode = PreSendExecutionMode.LOW_SPEED_WARNING_HOLD
        elif low_speed_stage == 4:
            execution_mode = PreSendExecutionMode.LOW_SPEED_RECOVERY_HOLD
        elif ood_detected:
            execution_mode = PreSendExecutionMode.OOD_BASELINE
        elif not actor_execution_valid:
            execution_mode = PreSendExecutionMode.ACTOR_EXECUTION_ROLLBACK_BASELINE
        elif not allow_residual:
            execution_mode = requested_baseline_mode
        else:
            execution_mode = PreSendExecutionMode.ACTOR
        residual_admitted = execution_mode == PreSendExecutionMode.ACTOR
        fallback_active = not residual_admitted
        if not residual_admitted:
            residual_norm = np.zeros_like(residual_norm)
        if baseline_action_real is None:
            baseline_action = self._baseline_for_obs(
                obs,
                previous_action_real=self._baseline_previous_action_for_stage(
                    low_speed_stage,
                    previous_action,
                    wind_speed,
                ),
            )
        else:
            baseline_action = np.asarray(baseline_action_real, dtype=np.float64)
            if baseline_action.shape != (2,) or not np.isfinite(baseline_action).all():
                raise ValueError('baseline_action_real must contain two finite action values')
        baseline_action, fallback_generating_torque_floor_applied = (
            self._apply_generating_fallback_torque_floor(
                baseline_action,
                wind_speed=wind_speed,
                rotor_speed=rotor_speed,
                real_power=real_power,
                low_speed_stage=low_speed_stage,
                fallback_active=fallback_active,
                baseline_action_was_explicit=baseline_action_real is not None,
            )
        )
        fallback_action = None
        if fallback_action_real is not None:
            fallback_action = np.asarray(fallback_action_real, dtype=np.float64)
            if fallback_action.shape != (2,) or not np.isfinite(fallback_action).all():
                raise ValueError('fallback_action_real must contain two finite action values')
        policy_delta_real = requested_residual_norm.astype(np.float64) * state_residual_limit
        pre_safety_action_real = baseline_action + policy_delta_real
        desired_action_real = (
            fallback_action
            if fallback_active and fallback_action is not None
            else baseline_action + residual_norm * state_residual_limit
        )
        pitch_protection_floor, pitch_protection_reason = self._pitch_protection(
            wind_speed,
            rotor_speed,
            real_power,
            pitch,
            rotor_acceleration,
        )
        above_rated_initial_pitch_floor_active = pitch_protection_floor is not None
        above_rated_initial_pitch_floor_applied = False
        pitch_protection_lift = 0.0
        if above_rated_initial_pitch_floor_active:
            desired_action_real = desired_action_real.copy()
            above_rated_initial_pitch_floor_applied = bool(
                desired_action_real[1] < pitch_protection_floor
            )
            pitch_protection_lift = max(0.0, float(pitch_protection_floor - desired_action_real[1]))
            desired_action_real[1] = max(desired_action_real[1], pitch_protection_floor)
        post_pitch_protection_action_real = desired_action_real.copy()
        low_rotor_speed_torque_unload_active = low_speed_stage in {1, 2, 3, 4}
        desired_action_real = self._apply_low_speed_torque(
            desired_action_real,
            low_speed_stage,
            low_speed_scale,
        )
        post_safety_action_real = desired_action_real.copy()
        projected_action_real = np.clip(
            post_safety_action_real,
            self.safe_low,
            self.safe_high,
        ).astype(np.float64)
        should_project_action_rate = (
            self.action_limiter_owner == 'controller'
            if enforce_action_rate_limit is None
            else bool(enforce_action_rate_limit)
        )
        if should_project_action_rate:
            action_real, action_projection_applied, action_rate_limited = self.project_action(
                desired_action_real,
                bypass_torque_decrease_rate_limit=bypass_torque_rate,
            )
            action_projection_applied = (
                action_projection_applied
                or low_rotor_speed_torque_unload_active
                or above_rated_initial_pitch_floor_applied
            )
        else:
            projection = self.action_transform.project(desired_action_real, apply_rate_limit=False)
            action_real = projection.action_real
            action_projection_applied = projection.projection_applied
            action_rate_limited = projection.rate_limited
        self._update_recovery_baseline_history(
            low_speed_stage,
            baseline_action,
            action_real,
            wind_speed,
        )
        executed_residual = self.residual_target_numpy(
            action_real,
            baseline_action,
            residual_limit=state_residual_limit,
        )
        return SafeAction(
            residual_norm=residual_norm,
            action_real=action_real,
            action_norm=self.action_scaler.normalize(action_real),
            baseline_action_real=baseline_action.copy(),
            policy_delta_real=policy_delta_real.copy(),
            pre_safety_action_real=pre_safety_action_real,
            post_pitch_protection_action_real=post_pitch_protection_action_real,
            post_safety_action_real=post_safety_action_real,
            projected_action_real=projected_action_real,
            rate_limited_action_real=action_real.copy(),
            pitch_protection_floor=(
                0.0
                if pitch_protection_floor is None
                else float(pitch_protection_floor)
            ),
            pitch_protection_state=(
                2
                if above_rated_initial_pitch_floor_applied
                else int(above_rated_initial_pitch_floor_active)
            ),
            pitch_protection_reason=pitch_protection_reason,
            pitch_protection_lift=pitch_protection_lift,
            low_speed_stage=low_speed_stage,
            low_speed_scale=low_speed_scale,
            low_speed_reason=low_speed_reason,
            ood_score=score,
            ood_feature_index=ood_feature_index,
            ood_feature_name=ood_feature_name,
            ood_feature_value=ood_feature_value,
            ood_feature_median=ood_feature_median,
            ood_feature_iqr=ood_feature_iqr,
            fallback_active=fallback_active,
            fallback_generating_torque_floor_applied=fallback_generating_torque_floor_applied,
            low_rotor_speed_fallback_active=low_rotor_speed_fallback_active,
            low_rotor_speed_torque_unload_active=low_rotor_speed_torque_unload_active,
            above_rated_initial_pitch_floor_active=above_rated_initial_pitch_floor_active,
            above_rated_initial_pitch_floor_applied=above_rated_initial_pitch_floor_applied,
            action_projection_applied=action_projection_applied,
            action_rate_limited=action_rate_limited,
            policy_action_norm=requested_residual_norm,
            executed_action_norm=executed_residual,
            state_residual_limit_real=state_residual_limit.copy(),
            optimization_state_code=(
                -1
                if state_classification is None
                else OPTIMIZATION_STATE_ENCODING_ORDER.index(state_classification.optimization_state)
            ),
            control_region_code=(
                -1
                if state_classification is None
                else CONTROL_REGION_ENCODING_ORDER.index(state_classification.control_region)
            ),
            pre_send_execution_mode=execution_mode,
            residual_admitted=residual_admitted,
            ood_detected=ood_detected,
            actor_execution_valid=bool(actor_execution_valid),
        )

    def action_from_policy(
        self,
        policy_action_norm: np.ndarray,
        obs: np.ndarray,
        allow_policy: bool,
        *,
        baseline_action_real: np.ndarray | None = None,
        fallback_action_real: np.ndarray | None = None,
        baseline_execution_mode: PreSendExecutionMode | str = PreSendExecutionMode.EXPLICIT_BASELINE_STAGE,
        actor_execution_valid: bool = True,
    ) -> SafeAction:
        return self.action_from_residual(
            policy_action_norm,
            obs,
            allow_residual=allow_policy,
            baseline_action_real=baseline_action_real,
            fallback_action_real=fallback_action_real,
            baseline_execution_mode=baseline_execution_mode,
            actor_execution_valid=actor_execution_valid,
        )

    def observe_step(self, power: float) -> bool:
        if power < 0.0:
            self.negative_power_steps += 1
        else:
            self.negative_power_steps = 0
        if self.low_power_threshold is None:
            self.low_power_steps = 0
        elif power < self.low_power_threshold:
            self.low_power_steps += 1
        else:
            self.low_power_steps = 0
        return (
            self.negative_power_steps >= self.negative_power_patience
            or self.low_power_steps >= self.low_power_patience
        )