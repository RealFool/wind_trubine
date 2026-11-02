from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from RL.utils.constants import DATA_SCHEMA_VERSION, STATE_VARIABLES
from RL.utils.operating_state import OptimizationState
from RL.utils.paired_pid_reference import PairedPidReference


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION = 'paired_pid_power_reference_v1'
_EXACT_REFERENCE_TIME_TOLERANCE_S = 1.0e-6
STATE_CONDITIONED_REWARD_SCHEMA_VERSION = 'state_conditioned_reward_v1'
_STATE_REWARD_WEIGHT_FIELDS = (
    'power',
    'blade_root',
    'continuous_load',
    'persistent_power_deficit',
    'action_continuity',
)


class RewardFunction:
    def __init__(self, env_config: dict[str, Any]) -> None:
        reward_cfg = env_config.get('reward', {})
        self.power_ref = float(reward_cfg.get('power_ref', env_config.get('rated_power', 1.0)))
        self.power_clip_max = float(reward_cfg.get('power_clip_max', 1.2))
        self.blade_root_ref = float(reward_cfg.get('blade_root_ref', 1.0e7))
        self.blade_root_soft_limit = float(reward_cfg.get('blade_root_soft_limit', 1.0))
        self.acc_ref = float(reward_cfg.get('acc_ref', 1.0))
        self.shaft_ref = float(reward_cfg.get('shaft_ref', 5.0e4))
        self.pitch_rate_limit = float(reward_cfg.get('pitch_rate_limit', 0.05))
        self.torque_rate_limit = float(reward_cfg.get('torque_rate_limit', 2000.0))

        self.power_weight = float(reward_cfg.get('power_weight', 1.0))
        self.blade_root_weight = float(reward_cfg.get('blade_root_weight', 0.0))
        self.acc_weight = float(reward_cfg.get('acc_weight', 1.0))
        self.shaft_weight = float(reward_cfg.get('shaft_weight', 1.0))
        self.pitch_rate_weight = float(reward_cfg.get('pitch_rate_weight', 1.0))
        self.torque_rate_weight = float(reward_cfg.get('torque_rate_weight', 1.0))
        self.command_excess_weight = float(reward_cfg.get('command_excess_weight', 1.0))
        self.power_tracking_weight = float(reward_cfg.get('power_tracking_weight', 1.0))
        self.blade_root_continuous_weight = float(reward_cfg.get('blade_root_continuous_weight', 0.0))
        self.blade_root_continuous_mode = str(
            reward_cfg.get('blade_root_continuous_mode', 'instantaneous_reference_ratio')
        ).lower()
        if self.blade_root_continuous_mode not in {
            'instantaneous_reference_ratio',
            'peak_reference_ratio',
            'pid_excess',
        }:
            raise ValueError(
                'reward.blade_root_continuous_mode must be instantaneous_reference_ratio, '
                'peak_reference_ratio, or pid_excess'
            )
        if not math.isfinite(self.blade_root_continuous_weight) or self.blade_root_continuous_weight < 0.0:
            raise ValueError('reward.blade_root_continuous_weight must be finite and non-negative')
        self.state_conditioning = self._state_conditioning(reward_cfg)
        self.power_floor_weight = float(reward_cfg.get('power_floor_weight', 0.0))
        self.persistent_power_deficit_weight = float(
            reward_cfg.get('persistent_power_deficit_weight', 0.0)
        )
        self.persistent_power_deficit_ewma_tau_s = float(
            reward_cfg.get('persistent_power_deficit_ewma_tau_s', 2.0)
        )
        self.persistent_power_deficit_deadband_fraction = float(
            reward_cfg.get('persistent_power_deficit_deadband_fraction', 0.01)
        )
        self.persistent_power_deficit_clip = float(
            reward_cfg.get('persistent_power_deficit_clip', 1.0)
        )
        if not math.isfinite(self.persistent_power_deficit_weight) or self.persistent_power_deficit_weight < 0.0:
            raise ValueError('reward.persistent_power_deficit_weight must be finite and non-negative')
        if not math.isfinite(self.persistent_power_deficit_ewma_tau_s) or self.persistent_power_deficit_ewma_tau_s <= 0.0:
            raise ValueError('reward.persistent_power_deficit_ewma_tau_s must be positive and finite')
        if (
            not math.isfinite(self.persistent_power_deficit_deadband_fraction)
            or self.persistent_power_deficit_deadband_fraction < 0.0
        ):
            raise ValueError(
                'reward.persistent_power_deficit_deadband_fraction must be finite and non-negative'
            )
        if not math.isfinite(self.persistent_power_deficit_clip) or self.persistent_power_deficit_clip <= 0.0:
            raise ValueError('reward.persistent_power_deficit_clip must be positive and finite')
        self.power_floor_reference_template = reward_cfg.get('power_floor_reference_template')
        if self.power_floor_reference_template is not None and not isinstance(
            self.power_floor_reference_template, str
        ):
            raise ValueError('reward.power_floor_reference_template must be a string or null')
        if not math.isfinite(self.power_floor_weight) or self.power_floor_weight < 0.0:
            raise ValueError('reward.power_floor_weight must be finite and non-negative')
        self._power_reference_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pid_power_reference_descriptors: dict[int, dict[str, Any]] = {}
        self.paired_pid_reference: PairedPidReference | None = None
        self.paired_pid_reference_sha256: str | None = None

        self.blade_root_load_mode = str(reward_cfg.get('blade_root_load_mode', 'my_rms')).lower()
        if self.blade_root_load_mode not in {'my_rms', 'my_max', 'my_peak_increment'}:
            raise ValueError(
                'reward.blade_root_load_mode must be my_rms, my_max, or my_peak_increment '
                'for the My-only UDP schema'
            )
        self.blade_root_peak_aggregation = str(reward_cfg.get('blade_root_peak_aggregation', 'max')).lower()
        if self.blade_root_peak_aggregation not in {'max', 'mean'}:
            raise ValueError('reward.blade_root_peak_aggregation must be max or mean')
        self.acc_metric_mode = str(reward_cfg.get('acc_metric_mode', 'fore_aft'))
        self.shaft_mode = str(reward_cfg.get('shaft_mode', 'electrical_torque_proxy'))
        self.power_source = str(reward_cfg.get('power_source', 'reward_metric'))
        self.power_reward_mode = str(reward_cfg.get('power_reward_mode', 'power_ref_fraction')).lower()
        if self.power_reward_mode not in {
            'power_ref_fraction',
            'available_fraction',
            'energy_increment',
            'relative_pid_energy',
        }:
            raise ValueError(
                'reward.power_reward_mode must be power_ref_fraction, available_fraction, '
                'energy_increment, or relative_pid_energy'
            )
        self.power_available_mode = str(reward_cfg.get('power_available_mode', 'wind_cubic'))
        self.rated_power = float(env_config.get('rated_power', self.power_ref))
        self.rated_wind_speed = float(reward_cfg.get('rated_wind_speed', env_config.get('fixed_wind_speed', 12.0)))
        self.cut_in_wind_speed = float(reward_cfg.get('cut_in_wind_speed', 3.0))
        default_energy_duration_s = float(env_config.get('bladed_step_duration_ms', 20.0)) / 1000.0
        self.energy_reference_j = float(
            reward_cfg.get('energy_reference_j', self.power_ref * default_energy_duration_s)
        )
        self.relative_energy_clip = float(
            reward_cfg.get('relative_energy_clip', self.power_clip_max)
        )
        if not math.isfinite(self.energy_reference_j) or self.energy_reference_j <= 0.0:
            raise ValueError('reward.energy_reference_j must be a positive finite value')
        if not math.isfinite(self.power_clip_max) or self.power_clip_max <= 0.0:
            raise ValueError('reward.power_clip_max must be a positive finite value')
        if not math.isfinite(self.relative_energy_clip) or self.relative_energy_clip <= 0.0:
            raise ValueError('reward.relative_energy_clip must be a positive finite value')
        paired_pid_reference_path = reward_cfg.get('paired_pid_reference_path')
        if paired_pid_reference_path is not None:
            if not isinstance(paired_pid_reference_path, str) or not paired_pid_reference_path.strip():
                raise ValueError('reward.paired_pid_reference_path must be a non-empty string')
            if reward_cfg.get('pid_power_reference_manifest_path') is not None:
                raise ValueError(
                    'reward.paired_pid_reference_path cannot be mixed with the legacy '
                    'reward.pid_power_reference_manifest_path'
                )
            if reward_cfg.get('blade_root_peak_reference_path') is not None or reward_cfg.get(
                'pid_blade_root_peak_reference_path'
            ) is not None:
                raise ValueError(
                    'reward.paired_pid_reference_path cannot be mixed with legacy blade-root peak references'
                )
            if self.power_reward_mode != 'relative_pid_energy':
                raise ValueError(
                    'reward.paired_pid_reference_path requires power_reward_mode=relative_pid_energy'
                )
            path = Path(paired_pid_reference_path.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            self.paired_pid_reference = PairedPidReference.from_path(path, project_root=PROJECT_ROOT)
            self.paired_pid_reference_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        elif self.power_reward_mode == 'relative_pid_energy':
            self._pid_power_reference_descriptors = self._load_pid_power_reference_manifest(
                reward_cfg.get('pid_power_reference_manifest_path')
            )
        self.gear_ratio = float(env_config.get('gear_ratio', 1.0))
        self.lambda_speed = float(reward_cfg.get('lambda_speed', 10.0))
        self.rotor_speed_index = STATE_VARIABLES.index('I_RotorSpeed')
        self.electrical_torque_index = STATE_VARIABLES.index('I_ElectricalTorque')
        self.generator_speed_index = STATE_VARIABLES.index('I_GeneratorSpeed')
        self.real_power_index = STATE_VARIABLES.index('I_RealPower')
        self.fore_aft_acceleration_index = STATE_VARIABLES.index('I_NacelleForeAftAcceleration')
        self.side_side_acceleration_index = STATE_VARIABLES.index('I_NacelleSideSideAcceleration')
        if self.paired_pid_reference is None:
            self.blade_root_peak_wind_speeds, self.blade_root_peak_reference_schedule_nm = (
                self._load_peak_reference_schedule(reward_cfg)
            )
            self.pid_blade_root_peak_wind_speeds, self.pid_blade_root_peak_reference_schedule_nm = (
                self._load_peak_reference_schedule(
                    reward_cfg,
                    path_key='pid_blade_root_peak_reference_path',
                    reference_name='PID blade-root peak reference',
                )
            )
            if self.blade_root_continuous_mode == 'pid_excess' and (
                self.pid_blade_root_peak_wind_speeds is None
                or self.pid_blade_root_peak_reference_schedule_nm is None
            ):
                raise ValueError(
                    'reward.blade_root_continuous_mode=pid_excess requires '
                    'reward.pid_blade_root_peak_reference_path'
                )
        else:
            self.blade_root_peak_wind_speeds = None
            self.blade_root_peak_reference_schedule_nm = None
            self.pid_blade_root_peak_wind_speeds = None
            self.pid_blade_root_peak_reference_schedule_nm = None
        self.blade_root_peak_reference_nm = self._peak_reference_array(
            reward_cfg.get('blade_root_peak_reference_nm', self.blade_root_ref),
            'reward.blade_root_peak_reference_nm',
        )
        self.reset_condition()

    @staticmethod
    def _state_conditioning(reward_cfg: Mapping[str, Any]) -> dict[str, dict[str, float]] | None:
        document = reward_cfg.get('state_conditioning')
        if document is None:
            return None
        if not isinstance(document, Mapping):
            raise ValueError('reward.state_conditioning must be an object')
        if document.get('schema_version') != STATE_CONDITIONED_REWARD_SCHEMA_VERSION:
            raise ValueError('reward.state_conditioning schema version is incompatible')
        weights = document.get('weights')
        expected_states = {state.value for state in OptimizationState}
        if not isinstance(weights, Mapping) or set(weights) != expected_states:
            raise ValueError('reward.state_conditioning weights must cover every optimization state')
        parsed: dict[str, dict[str, float]] = {}
        for state in OptimizationState:
            row = weights[state.value]
            if not isinstance(row, Mapping) or set(row) != set(_STATE_REWARD_WEIGHT_FIELDS):
                raise ValueError(
                    f'reward.state_conditioning weights for {state.value} are incompatible'
                )
            parsed_row: dict[str, float] = {}
            for name in _STATE_REWARD_WEIGHT_FIELDS:
                value = float(row[name])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f'reward.state_conditioning {state.value}.{name} must be finite and non-negative'
                    )
                parsed_row[name] = value
            parsed[state.value] = parsed_row
        for state in (OptimizationState.STARTUP_STOPPED, OptimizationState.NEGATIVE_POWER_ABNORMAL):
            if any(parsed[state.value][name] != 0.0 for name in _STATE_REWARD_WEIGHT_FIELDS):
                raise ValueError(
                    f'reward.state_conditioning {state.value} must have zero optimization weights'
                )
        return parsed

    def _state_reward_multipliers(
        self,
        reward_state_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if self.state_conditioning is None:
            return {name: 1.0 for name in _STATE_REWARD_WEIGHT_FIELDS}
        if reward_state_weights is None:
            raise ValueError('state-conditioned reward requires current optimization-state weights')
        expected_states = {state.value for state in OptimizationState}
        if set(reward_state_weights) != expected_states:
            raise ValueError('reward_state_weights must cover every optimization state')
        parsed: dict[str, float] = {}
        for state, value in reward_state_weights.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError('reward_state_weights must be finite and non-negative')
            parsed[state] = numeric
        return {
            name: sum(parsed[state] * self.state_conditioning[state][name] for state in expected_states)
            for name in _STATE_REWARD_WEIGHT_FIELDS
        }

    @staticmethod
    def _peak_reference_array(value: Any, name: str) -> np.ndarray:
        reference = np.asarray(value, dtype=np.float64)
        if reference.ndim == 0:
            reference = np.full(3, float(reference), dtype=np.float64)
        if reference.shape != (3,) or not np.isfinite(reference).all() or np.any(reference <= 0.0):
            raise ValueError(f'{name} must be one positive finite value or three positive finite values')
        return reference

    def _load_peak_reference_schedule(
        self,
        reward_cfg: dict[str, Any],
        *,
        path_key: str = 'blade_root_peak_reference_path',
        reference_name: str = 'Blade-root peak reference',
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        configured_path = reward_cfg.get(path_key)
        if configured_path is None:
            return None, None
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError(f'reward.{path_key} must be a non-empty string')
        path = Path(configured_path.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            with path.open('r', encoding='utf-8') as file:
                document = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{reference_name} file does not exist: {path}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read {reference_name.lower()} file: {path}') from exc
        if not isinstance(document, dict):
            raise ValueError(f'{reference_name} file must contain an object')
        metadata = document.get('metadata')
        if not isinstance(metadata, dict) or metadata.get('data_schema_version') != DATA_SCHEMA_VERSION:
            raise ValueError(f'{reference_name} file has an incompatible data schema version')
        reward_section = document.get('reward')
        rows = reward_section.get('blade_root_peak_reference') if isinstance(reward_section, dict) else None
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError(
                f'{reference_name} file must contain at least two reward.blade_root_peak_reference rows'
            )
        parsed: list[tuple[float, np.ndarray]] = []
        for row in rows:
            if not isinstance(row, dict) or 'wind_speed' not in row or 'peak_abs_nm' not in row:
                raise ValueError('Each blade-root peak reference row needs wind_speed and peak_abs_nm')
            wind_speed = float(row['wind_speed'])
            if not math.isfinite(wind_speed):
                raise ValueError('Blade-root peak reference wind_speed values must be finite')
            parsed.append((wind_speed, self._peak_reference_array(row['peak_abs_nm'], 'peak_abs_nm')))
        parsed.sort(key=lambda item: item[0])
        speeds = np.asarray([item[0] for item in parsed], dtype=np.float64)
        if np.any(np.diff(speeds) <= 0.0):
            raise ValueError('Blade-root peak reference wind_speed values must be strictly increasing')
        references = np.stack([item[1] for item in parsed], axis=0)
        return speeds, references

    def _load_pid_power_reference_manifest(self, configured_path: Any) -> dict[int, dict[str, Any]]:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError(
                'reward.power_reward_mode=relative_pid_energy requires '
                'reward.pid_power_reference_manifest_path'
            )
        path = Path(configured_path.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            with path.open('r', encoding='utf-8') as handle:
                document = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Paired PID power reference manifest does not exist: {path}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read paired PID power reference manifest: {path}') from exc
        if not isinstance(document, dict):
            raise ValueError('Paired PID power reference manifest must contain an object')
        if document.get('schema_version') != PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION:
            raise ValueError('Paired PID power reference manifest schema_version is incompatible')
        metadata = document.get('metadata')
        if not isinstance(metadata, dict):
            raise ValueError('Paired PID power reference manifest must contain metadata')
        if metadata.get('data_schema_version') != DATA_SCHEMA_VERSION:
            raise ValueError('Paired PID power reference manifest data schema is incompatible')
        if metadata.get('source_role') != 'paired_pid_outcome_reference':
            raise ValueError(
                'Paired PID power reference source_role must be paired_pid_outcome_reference'
            )
        if metadata.get('reference_signal') != 'Electrical power':
            raise ValueError('Paired PID power reference signal must be Electrical power')
        pairing_contract = metadata.get('pairing_contract')
        expected_pairing_contract = {
            'case_identity': 'same_bladed_case',
            'turbulence_seed_identity': 'same_seed',
            'time_window_identity': 'same_simulation_time',
            'time_alignment': 'exact_timestamp_no_interpolation',
        }
        if pairing_contract != expected_pairing_contract:
            raise ValueError('Paired PID power reference pairing contract is incompatible')
        source_manifest = metadata.get('input_manifest_sha256')
        if (
            not isinstance(source_manifest, str)
            or len(source_manifest) != 64
            or any(character not in '0123456789abcdef' for character in source_manifest.lower())
        ):
            raise ValueError('Paired PID power reference input manifest SHA-256 is invalid')
        references = document.get('references')
        if not isinstance(references, list) or not references:
            raise ValueError('Paired PID power reference manifest needs at least one reference')
        descriptors: dict[int, dict[str, Any]] = {}
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError('Each paired PID power reference must be an object')
            try:
                condition_id = int(reference['condition_id'])
                reference_path = Path(str(reference['path'])).expanduser()
                expected_sha256 = str(reference['sha256']).lower()
                expected_rows = int(reference['rows'])
                start_time_s = float(reference['start_time_s'])
                end_time_s = float(reference['end_time_s'])
                nominal_step_s = float(reference['nominal_step_s'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Paired PID power reference descriptor is invalid') from exc
            if condition_id in descriptors:
                raise ValueError(f'Duplicate paired PID power reference condition {condition_id}')
            if not reference_path.is_absolute():
                reference_path = PROJECT_ROOT / reference_path
            if (
                len(expected_sha256) != 64
                or any(character not in '0123456789abcdef' for character in expected_sha256)
            ):
                raise ValueError(
                    f'Paired PID power reference condition {condition_id} SHA-256 is invalid'
                )
            if expected_rows < 2:
                raise ValueError(
                    f'Paired PID power reference condition {condition_id} needs at least two rows'
                )
            if not all(math.isfinite(value) for value in (start_time_s, end_time_s, nominal_step_s)):
                raise ValueError(
                    f'Paired PID power reference condition {condition_id} time metadata is non-finite'
                )
            if end_time_s <= start_time_s or nominal_step_s <= 0.0:
                raise ValueError(
                    f'Paired PID power reference condition {condition_id} time metadata is invalid'
                )
            descriptors[condition_id] = {
                'path': reference_path,
                'sha256': expected_sha256,
                'rows': expected_rows,
                'start_time_s': start_time_s,
                'end_time_s': end_time_s,
                'nominal_step_s': nominal_step_s,
            }
        return descriptors

    def reset_condition(self, condition_id: int | None = None) -> None:
        if self.paired_pid_reference is None:
            self._active_paired_condition_id = None
        elif condition_id is None:
            self._active_paired_condition_id = None
        else:
            self.paired_pid_reference.entry(int(condition_id))
            self._active_paired_condition_id = int(condition_id)
        self._blade_root_episode_peak_nm = np.zeros(3, dtype=np.float64)
        self._persistent_power_deficit_ewma = 0.0

    def reset_episode(self) -> None:
        """Backward-compatible alias for callers using legacy episode terminology."""
        self.reset_condition()

    def compute(
        self,
        state_t: np.ndarray,
        state_next: np.ndarray,
        reward_metrics: dict[str, float],
        action_t: np.ndarray,
        prev_action_t: np.ndarray,
        wind_t: float,
        condition_id: int | None = None,
        reward_state_weights: Mapping[str, float] | None = None,
        power_t: float | None = None,
        sim_time_t: float | None = None,
        sim_time_next: float | None = None,
        power_reference_raw: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        state_multipliers = self._state_reward_multipliers(reward_state_weights)
        power_metric_raw = float(reward_metrics['Electrical power'])
        if self.power_source == 'state_next_real_power':
            power_raw = float(state_next[self.real_power_index])
        else:
            power_raw = power_metric_raw
        power_t_raw = float(state_t[self.real_power_index]) if power_t is None else float(power_t)
        if not math.isfinite(power_t_raw):
            power_t_raw = 0.0
        blade_root_values = self._blade_root_values(reward_metrics)
        blade_root_raw = self._compute_blade_root_metric_from_values(blade_root_values)
        peak_reference_nm = self._peak_reference_for_condition(condition_id, wind_t)
        previous_peak_nm = self._blade_root_episode_peak_nm.copy()
        current_peak_nm = np.maximum(previous_peak_nm, blade_root_values)
        previous_peak_excess = np.maximum(previous_peak_nm / peak_reference_nm - self.blade_root_soft_limit, 0.0)
        current_peak_excess = np.maximum(current_peak_nm / peak_reference_nm - self.blade_root_soft_limit, 0.0)
        peak_excess_increment = np.maximum(current_peak_excess - previous_peak_excess, 0.0)
        self._blade_root_episode_peak_nm = current_peak_nm
        acc_raw = self._compute_acc_metric(state_next)
        shaft_raw = self._compute_shaft_metric(state_t, state_next)
        pitch_rate_raw = abs(float(action_t[1] - prev_action_t[1]))
        torque_rate_raw = abs(float(action_t[0] - prev_action_t[0]))
        power_available = self._estimate_power_available(power_raw=power_raw, wind_t=wind_t)
        command_power = self._estimate_command_power(action_t=action_t, state_t=state_t)
        command_excess_raw = max(command_power - power_available, 0.0)
        power_tracking_raw = abs(power_raw - power_available)
        if self.blade_root_continuous_mode == 'pid_excess' and self.paired_pid_reference is not None:
            continuous_reference_nm = peak_reference_nm
            blade_root_continuous_values = np.maximum(
                current_peak_nm / continuous_reference_nm - 1.0,
                0.0,
            )
        elif self.blade_root_continuous_mode == 'pid_excess':
            assert self.pid_blade_root_peak_wind_speeds is not None
            assert self.pid_blade_root_peak_reference_schedule_nm is not None
            continuous_reference_nm = self._peak_reference_for_schedule(
                wind_t,
                self.pid_blade_root_peak_wind_speeds,
                self.pid_blade_root_peak_reference_schedule_nm,
            )
            blade_root_continuous_values = np.maximum(
                current_peak_nm / continuous_reference_nm - 1.0,
                0.0,
            )
        elif self.blade_root_continuous_mode == 'peak_reference_ratio':
            blade_root_continuous_values = current_peak_nm / peak_reference_nm
        else:
            blade_root_continuous_values = blade_root_values / peak_reference_nm
        blade_root_continuous_term = self._aggregate_peak_metric(blade_root_continuous_values)
        power_floor_raw = 0.0
        if power_reference_raw is not None and math.isfinite(float(power_reference_raw)):
            power_floor_raw = max(float(power_reference_raw) - power_raw, 0.0)

        persistent_power_deficit_raw = 0.0
        persistent_power_deficit_ewma = self._persistent_power_deficit_ewma
        persistent_power_deficit_term = 0.0
        persistent_power_deficit_clipped = 0.0
        if self.persistent_power_deficit_weight > 0.0:
            if power_reference_raw is None or not math.isfinite(float(power_reference_raw)):
                raise ValueError('persistent power deficit requires a finite paired power reference')
            deficit_delta_t_s = self._required_time_delta(sim_time_t, sim_time_next)
            deadband_w = self.persistent_power_deficit_deadband_fraction * self.power_ref
            persistent_power_deficit_raw = max(
                float(power_reference_raw) - power_raw - deadband_w,
                0.0,
            )
            ewma_alpha = -math.expm1(-deficit_delta_t_s / self.persistent_power_deficit_ewma_tau_s)
            persistent_power_deficit_ewma += ewma_alpha * (
                persistent_power_deficit_raw - persistent_power_deficit_ewma
            )
            self._persistent_power_deficit_ewma = persistent_power_deficit_ewma
            normalized_deficit = persistent_power_deficit_ewma / max(self.power_ref, 1.0e-6)
            persistent_power_deficit_term = float(np.clip(
                normalized_deficit,
                0.0,
                self.persistent_power_deficit_clip,
            ))
            persistent_power_deficit_clipped = float(
                normalized_deficit > self.persistent_power_deficit_clip
            )

        available_power_fraction = 0.0
        if power_available > 1e-6:
            available_power_fraction = float(np.clip(power_raw / power_available, 0.0, self.power_clip_max))
        relative_power_raw = 0.0
        relative_energy_increment_j = 0.0
        if self.power_reward_mode == 'available_fraction':
            power_term = available_power_fraction
        elif self.power_reward_mode == 'energy_increment':
            energy_delta_t_s = self._strict_time_delta(sim_time_t, sim_time_next)
            energy_increment_j = 0.5 * (power_t_raw + power_raw) * energy_delta_t_s
            power_term = float(np.clip(energy_increment_j / self.energy_reference_j, 0.0, self.power_clip_max))
        elif self.power_reward_mode == 'relative_pid_energy':
            energy_delta_t_s = self._required_time_delta(sim_time_t, sim_time_next)
            if power_reference_raw is None or not math.isfinite(float(power_reference_raw)):
                raise ValueError('relative_pid_energy requires a finite paired PID power reference')
            relative_power_raw = power_raw - float(power_reference_raw)
            relative_energy_increment_j = relative_power_raw * energy_delta_t_s
            energy_increment_j = relative_energy_increment_j
            power_term = float(np.clip(
                relative_energy_increment_j / self.energy_reference_j,
                -self.relative_energy_clip,
                self.relative_energy_clip,
            ))
        else:
            power_term = np.clip(power_raw / max(self.power_ref, 1e-6), 0.0, self.power_clip_max)
            energy_delta_t_s = 0.0
            energy_increment_j = 0.0
        if self.power_reward_mode not in {'energy_increment', 'relative_pid_energy'}:
            energy_delta_t_s = 0.0
            energy_increment_j = 0.0
        if self.blade_root_load_mode == 'my_peak_increment':
            blade_root_ratio = self._aggregate_peak_metric(current_peak_nm / peak_reference_nm)
            blade_root_term = self._aggregate_peak_metric(peak_excess_increment)
        else:
            blade_root_ratio = blade_root_raw / max(self.blade_root_ref, 1e-6)
            blade_root_term = max(blade_root_ratio - self.blade_root_soft_limit, 0.0)
        acc_term = acc_raw / max(self.acc_ref, 1e-6)
        shaft_term = shaft_raw / max(self.shaft_ref, 1e-6)
        pitch_rate_term = pitch_rate_raw / max(self.pitch_rate_limit, 1e-6)
        torque_rate_term = torque_rate_raw / max(self.torque_rate_limit, 1e-6)
        command_excess_term = command_excess_raw / max(self.power_ref, 1e-6)
        power_tracking_term = power_tracking_raw / max(self.power_ref, 1e-6)
        power_floor_term = power_floor_raw / max(self.power_ref, 1.0e-6)

        reward = float(
            self.power_weight * state_multipliers['power'] * power_term
            - self.blade_root_weight * state_multipliers['blade_root'] * blade_root_term
            - self.acc_weight * acc_term
            - self.shaft_weight * shaft_term
            - self.pitch_rate_weight * state_multipliers['action_continuity'] * pitch_rate_term
            - self.torque_rate_weight * state_multipliers['action_continuity'] * torque_rate_term
            - self.command_excess_weight * command_excess_term
            - self.power_tracking_weight * power_tracking_term
            - self.power_floor_weight * power_floor_term
            - self.persistent_power_deficit_weight
            * state_multipliers['persistent_power_deficit']
            * persistent_power_deficit_term
            - self.blade_root_continuous_weight
            * state_multipliers['continuous_load']
            * blade_root_continuous_term
        )
        reward = self._sanitize_scalar(reward)

        breakdown = {
            'reward_power_raw': power_raw,
            'reward_power_metric_raw': power_metric_raw,
            'reward_power_state_raw': float(state_next[self.real_power_index]),
            'reward_power_t_raw': power_t_raw,
            'reward_power_term': float(power_term),
            'reward_energy_delta_t_s': float(energy_delta_t_s),
            'reward_energy_increment_j': float(energy_increment_j),
            'reward_energy_increment_valid': float(energy_delta_t_s > 0.0),
            'reward_pid_power_reference_raw': (
                0.0 if power_reference_raw is None else float(power_reference_raw)
            ),
            'reward_relative_power_raw': float(relative_power_raw),
            'reward_relative_energy_increment_j': float(relative_energy_increment_j),
            'reward_blade_root_raw': float(blade_root_raw),
            'reward_blade_root_ratio': float(blade_root_ratio),
            'reward_blade_root_term': float(blade_root_term),
            'reward_blade_root_peak_raw': float(self._aggregate_peak_metric(current_peak_nm)),
            'reward_blade_root_peak_excess': float(self._aggregate_peak_metric(current_peak_excess)),
            'reward_blade_root_peak_excess_increment': float(self._aggregate_peak_metric(peak_excess_increment)),
            'reward_acc_raw': float(acc_raw),
            'reward_acc_term': float(acc_term),
            'reward_shaft_raw': float(shaft_raw),
            'reward_shaft_term': float(shaft_term),
            'reward_pitch_rate_raw': float(pitch_rate_raw),
            'reward_pitch_rate_term': float(pitch_rate_term),
            'reward_torque_rate_raw': float(torque_rate_raw),
            'reward_torque_rate_term': float(torque_rate_term),
            'reward_power_available_raw': float(power_available),
            'reward_power_available_fraction': available_power_fraction,
            'reward_command_power_raw': float(command_power),
            'reward_command_excess_raw': float(command_excess_raw),
            'reward_command_excess_term': float(command_excess_term),
            'reward_power_tracking_raw': float(power_tracking_raw),
            'reward_power_tracking_term': float(power_tracking_term),
            'reward_power_floor_raw': float(power_floor_raw),
            'reward_power_floor_term': float(power_floor_term),
            'reward_persistent_power_deficit_raw': float(persistent_power_deficit_raw),
            'reward_persistent_power_deficit_ewma': float(persistent_power_deficit_ewma),
            'reward_persistent_power_deficit_term': float(persistent_power_deficit_term),
            'reward_persistent_power_deficit_clipped': float(persistent_power_deficit_clipped),
            'reward_blade_root_continuous_term': float(blade_root_continuous_term),
            'reward_state_power_multiplier': state_multipliers['power'],
            'reward_state_blade_root_multiplier': state_multipliers['blade_root'],
            'reward_state_continuous_load_multiplier': state_multipliers['continuous_load'],
            'reward_state_persistent_power_deficit_multiplier': state_multipliers[
                'persistent_power_deficit'
            ],
            'reward_state_action_continuity_multiplier': state_multipliers['action_continuity'],
            'reward_total': reward,
        }
        for blade_index in range(3):
            blade_number = blade_index + 1
            breakdown.update(
                {
                    f'reward_blade_root_{blade_number}_raw': float(blade_root_values[blade_index]),
                    f'reward_blade_root_{blade_number}_peak_nm': float(current_peak_nm[blade_index]),
                    f'reward_blade_root_{blade_number}_peak_reference_nm': float(peak_reference_nm[blade_index]),
                    f'reward_blade_root_{blade_number}_peak_excess': float(current_peak_excess[blade_index]),
                    f'reward_blade_root_{blade_number}_peak_excess_increment': float(peak_excess_increment[blade_index]),
                }
            )
        sanitized = {key: self._sanitize_scalar(value) for key, value in breakdown.items()}
        return reward, sanitized

    def reference_power(self, condition_id: int | None, sim_time_s: float | None) -> float | None:
        """Return the configured PID power reference at a queue condition/time."""
        if self.power_reward_mode == 'relative_pid_energy':
            if self.paired_pid_reference is not None:
                if condition_id is None:
                    raise ValueError('relative_pid_energy requires a paired PID reference condition ID')
                if (
                    self._active_paired_condition_id is not None
                    and int(condition_id) != self._active_paired_condition_id
                ):
                    raise ValueError('Paired PID reference condition changed without reset_condition')
                if sim_time_s is None:
                    raise ValueError('relative_pid_energy requires a finite paired PID reference timestamp')
                return self.paired_pid_reference.power_at(int(condition_id), float(sim_time_s))
            return self._exact_pid_reference_power(condition_id, sim_time_s)
        if self.power_floor_weight <= 0.0 or not self.power_floor_reference_template:
            return None
        if condition_id is None or sim_time_s is None or not math.isfinite(float(sim_time_s)):
            return None
        condition_id = int(condition_id)
        cached = self._power_reference_cache.get(condition_id)
        if cached is None:
            path_text = str(self.power_floor_reference_template).format(condition_id=condition_id)
            path = Path(path_text).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            times: list[float] = []
            powers: list[float] = []
            try:
                with path.open('r', encoding='utf-8-sig', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        time_value = float(row['Time'])
                        power_value = float(row['Electrical power'])
                        if math.isfinite(time_value) and math.isfinite(power_value):
                            times.append(time_value)
                            powers.append(power_value)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Unable to load power floor reference: {path}') from exc
            if len(times) < 2:
                raise ValueError(f'Power floor reference needs at least two rows: {path}')
            time_array = np.asarray(times, dtype=np.float64)
            power_array = np.asarray(powers, dtype=np.float64)
            if np.any(np.diff(time_array) <= 0.0):
                raise ValueError(f'Power floor reference times must be strictly increasing: {path}')
            cached = (time_array, power_array)
            self._power_reference_cache[condition_id] = cached
        return float(np.interp(float(sim_time_s), cached[0], cached[1]))

    def pid_blade_root_peak_reference(self, condition_id: int | None) -> np.ndarray:
        """Return the paired PID per-blade peak reference for a condition."""
        if self.paired_pid_reference is None:
            raise ValueError('Paired PID power/load reference is required for per-blade peak lookup')
        if condition_id is None:
            raise ValueError('Per-blade PID peak lookup requires a condition ID')
        return self.paired_pid_reference.blade_root_peak(int(condition_id))

    def _exact_pid_reference_power(
        self,
        condition_id: int | None,
        sim_time_s: float | None,
    ) -> float:
        if condition_id is None:
            raise ValueError('relative_pid_energy requires a paired PID power reference condition ID')
        if sim_time_s is None or not math.isfinite(float(sim_time_s)):
            raise ValueError('relative_pid_energy requires a finite paired PID reference timestamp')
        condition_key = int(condition_id)
        descriptor = self._pid_power_reference_descriptors.get(condition_key)
        if descriptor is None:
            raise ValueError(f'No paired PID power reference for condition {condition_key}')
        cached = self._power_reference_cache.get(condition_key)
        if cached is None:
            cached = self._load_exact_pid_reference(condition_key, descriptor)
            self._power_reference_cache[condition_key] = cached
        times, powers = cached
        query_time = float(sim_time_s)
        insertion_index = int(np.searchsorted(times, query_time, side='left'))
        candidate_indices = [
            index
            for index in (insertion_index - 1, insertion_index)
            if 0 <= index < len(times)
        ]
        if not candidate_indices:
            raise ValueError(
                f'Paired PID power reference condition {condition_key} has no exact timestamp '
                f'for {query_time:.9f} s'
            )
        nearest_index = min(candidate_indices, key=lambda index: abs(float(times[index]) - query_time))
        if abs(float(times[nearest_index]) - query_time) > _EXACT_REFERENCE_TIME_TOLERANCE_S:
            raise ValueError(
                f'Paired PID power reference condition {condition_key} has no exact timestamp '
                f'for {query_time:.9f} s'
            )
        return float(powers[nearest_index])

    @staticmethod
    def _load_exact_pid_reference(
        condition_id: int,
        descriptor: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        path = descriptor['path']
        try:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(
                f'Unable to load paired PID power reference condition {condition_id}: {path}'
            ) from exc
        if actual_sha256 != descriptor['sha256']:
            raise ValueError(
                f'Paired PID power reference condition {condition_id} SHA-256 does not match manifest'
            )
        times: list[float] = []
        powers: list[float] = []
        try:
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    time_value = float(row['Time'])
                    power_value = float(row['Electrical power'])
                    if not math.isfinite(time_value) or not math.isfinite(power_value):
                        raise ValueError('non-finite required value')
                    times.append(time_value)
                    powers.append(power_value)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f'Unable to load paired PID power reference condition {condition_id}: {path}'
            ) from exc
        if len(times) != descriptor['rows']:
            raise ValueError(
                f'Paired PID power reference condition {condition_id} row count does not match manifest'
            )
        time_array = np.asarray(times, dtype=np.float64)
        power_array = np.asarray(powers, dtype=np.float64)
        if np.any(np.diff(time_array) <= 0.0):
            raise ValueError(
                f'Paired PID power reference condition {condition_id} times must be strictly increasing'
            )
        if not math.isclose(
            float(time_array[0]),
            descriptor['start_time_s'],
            rel_tol=0.0,
            abs_tol=_EXACT_REFERENCE_TIME_TOLERANCE_S,
        ) or not math.isclose(
            float(time_array[-1]),
            descriptor['end_time_s'],
            rel_tol=0.0,
            abs_tol=_EXACT_REFERENCE_TIME_TOLERANCE_S,
        ):
            raise ValueError(
                f'Paired PID power reference condition {condition_id} time range does not match manifest'
            )
        median_step_s = float(np.median(np.diff(time_array)))
        if not math.isclose(
            median_step_s,
            descriptor['nominal_step_s'],
            rel_tol=0.0,
            abs_tol=_EXACT_REFERENCE_TIME_TOLERANCE_S,
        ):
            raise ValueError(
                f'Paired PID power reference condition {condition_id} step does not match manifest'
            )
        return time_array, power_array

    @staticmethod
    def _sanitize_scalar(value: float) -> float:
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        if math.isnan(numeric):
            return 0.0
        return 1.0e6 if numeric > 0.0 else -1.0e6

    @staticmethod
    def _strict_time_delta(sim_time_t: float | None, sim_time_next: float | None) -> float:
        if sim_time_t is None or sim_time_next is None:
            return 0.0
        try:
            delta_t_s = float(sim_time_next) - float(sim_time_t)
        except (TypeError, ValueError):
            return 0.0
        return delta_t_s if math.isfinite(delta_t_s) and delta_t_s > 0.0 else 0.0

    @classmethod
    def _required_time_delta(cls, sim_time_t: float | None, sim_time_next: float | None) -> float:
        delta_t_s = cls._strict_time_delta(sim_time_t, sim_time_next)
        if delta_t_s <= 0.0:
            raise ValueError('relative_pid_energy requires strictly increasing simulation time')
        return delta_t_s

    @staticmethod
    def _blade_root_values(reward_metrics: dict[str, float]) -> np.ndarray:
        return np.asarray([
            abs(float(reward_metrics.get(f'Blade root {blade} My (Nm)', 0.0)))
            for blade in range(1, 4)
        ], dtype=np.float64)

    def _compute_blade_root_metric_from_values(self, values: np.ndarray) -> float:
        if self.blade_root_load_mode == 'my_max':
            return float(np.max(values))
        return float(np.sqrt(np.mean(np.square(values))))

    def _peak_reference_for_wind(self, wind_t: float) -> np.ndarray:
        if self.blade_root_peak_wind_speeds is None or self.blade_root_peak_reference_schedule_nm is None:
            return self.blade_root_peak_reference_nm
        return self._peak_reference_for_schedule(
            wind_t,
            self.blade_root_peak_wind_speeds,
            self.blade_root_peak_reference_schedule_nm,
        )

    def _peak_reference_for_condition(
        self,
        condition_id: int | None,
        wind_t: float,
    ) -> np.ndarray:
        if self.paired_pid_reference is None:
            return self._peak_reference_for_wind(wind_t)
        if condition_id is None:
            raise ValueError('Paired PID blade-root reference requires a condition ID')
        condition_key = int(condition_id)
        if (
            self._active_paired_condition_id is not None
            and condition_key != self._active_paired_condition_id
        ):
            raise ValueError('Paired PID blade-root reference condition changed without reset_condition')
        return self.paired_pid_reference.blade_root_peak(condition_key)

    @staticmethod
    def _peak_reference_for_schedule(
        wind_t: float,
        wind_speeds: np.ndarray,
        reference_schedule_nm: np.ndarray,
    ) -> np.ndarray:
        if not math.isfinite(float(wind_t)):
            return np.asarray(reference_schedule_nm[0], dtype=np.float64)
        return np.asarray(
            [
                np.interp(
                    float(wind_t),
                    wind_speeds,
                    reference_schedule_nm[:, blade_index],
                )
                for blade_index in range(3)
            ],
            dtype=np.float64,
        )

    def _aggregate_peak_metric(self, values: np.ndarray) -> float:
        return float(np.mean(values)) if self.blade_root_peak_aggregation == 'mean' else float(np.max(values))

    def _compute_acc_metric(self, state_next: np.ndarray) -> float:
        fore_aft = abs(float(state_next[self.fore_aft_acceleration_index]))
        side_side = abs(float(state_next[self.side_side_acceleration_index]))
        if self.acc_metric_mode == 'combined':
            return 0.5 * (fore_aft + side_side)
        if self.acc_metric_mode == 'max':
            return max(fore_aft, side_side)
        return fore_aft

    def _compute_shaft_metric(self, state_t: np.ndarray, state_next: np.ndarray) -> float:
        if self.shaft_mode == 'speed_difference':
            return abs(
                float(state_next[self.rotor_speed_index])
                - float(state_next[self.generator_speed_index]) / max(self.gear_ratio, 1e-6)
            )
        return abs(float(state_next[self.electrical_torque_index])) + self.lambda_speed * abs(
            float(state_next[self.rotor_speed_index]) - float(state_t[self.rotor_speed_index])
        )

    def _estimate_power_available(self, power_raw: float, wind_t: float) -> float:
        if self.power_available_mode == 'current_power':
            return max(power_raw, 0.0)
        if self.power_available_mode == 'rated':
            return self.rated_power

        if wind_t <= self.cut_in_wind_speed:
            return 0.0
        normalized = (wind_t - self.cut_in_wind_speed) / max(self.rated_wind_speed - self.cut_in_wind_speed, 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)
        return float(self.rated_power * normalized ** 3)

    def _estimate_command_power(self, action_t: np.ndarray, state_t: np.ndarray) -> float:
        generator_speed_rad_s = max(float(state_t[self.generator_speed_index]), 0.0)
        torque_cmd = max(float(action_t[0]), 0.0)
        return float(torque_cmd * generator_speed_rad_s)
