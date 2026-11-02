from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from RL.utils.constants import ACTION_VARIABLES, OBSERVATION_VARIABLES, PREVIOUS_ACTION_OBSERVATION_VARIABLES, REAL_POWER_VARIABLE
from RL.utils.control_contract import (
    ENGINEERING_DEMAND_HIGH_NM,
    ENGINEERING_DEMAND_LOW_NM,
    NORMAL_GENERATION_PITCH_HIGH_RAD,
    NORMAL_GENERATION_PITCH_LOW_RAD,
)
from RL.utils.operating_state import (
    CONTROL_REGION_ENCODING_ORDER,
    OPERATING_STATE_CONTRACT_VERSION,
    OPTIMIZATION_STATE_ENCODING_ORDER,
    STATE_REGIME_OBSERVATION_VARIABLES,
    ControlRegion,
    OperatingPoint,
    OperatingStateClassifier,
    OperatingStateConfig,
    OptimizationState,
    encode_state_regime,
)
from RL.utils.raw_dataset_contract import (
    EXPECTED_CONDITION_IDS,
    EXPECTED_END_TIME_S,
    EXPECTED_START_TIME_S,
    TIME_TOLERANCE_S,
    manifest_sha256,
    sha256_file,
    validate_raw_source_root,
)


STATE_ACTION_ARTIFACT_SCHEMA_VERSION = 'state_action_envelope_artifact_v1'
STATE_CALIBRATION_VERSION = 'state_calibration_v1'
STATE_AWARE_OBSERVATION_SCHEMA_VERSION = 'state_aware_control_observation_v1'
STATE_AWARE_CONTROL_OBSERVATION_VARIABLES = (
    *OBSERVATION_VARIABLES,
    *PREVIOUS_ACTION_OBSERVATION_VARIABLES,
    *STATE_REGIME_OBSERVATION_VARIABLES,
)
STATE_AWARE_CONTROL_OBS_DIM = len(STATE_AWARE_CONTROL_OBSERVATION_VARIABLES)
TIME_FIELD = 'Time'
WIND_FIELD = 'I_AverageWindSpeed'
ROTOR_SPEED_FIELD = 'I_RotorSpeed'
PITCH_FIELD = 'I_PitchPosition1'


def state_aware_observation_schema() -> dict[str, Any]:
    return {
        'schema_version': STATE_AWARE_OBSERVATION_SCHEMA_VERSION,
        'feature_names': list(STATE_AWARE_CONTROL_OBSERVATION_VARIABLES),
        'base_observation_width': len(OBSERVATION_VARIABLES),
        'previous_action_width': len(PREVIOUS_ACTION_OBSERVATION_VARIABLES),
        'state_regime_width': len(STATE_REGIME_OBSERVATION_VARIABLES),
        'width': STATE_AWARE_CONTROL_OBS_DIM,
        'state_encoding_order': [state.value for state in OPTIMIZATION_STATE_ENCODING_ORDER],
        'region_encoding_order': [region.value for region in CONTROL_REGION_ENCODING_ORDER],
        'causal': True,
        'future_feature_lookahead_steps': 0,
    }


def append_state_regime_observation(
    control_observation: np.ndarray,
    classification,
) -> np.ndarray:
    values = np.asarray(control_observation, dtype=np.float32)
    expected_width = len(OBSERVATION_VARIABLES) + len(PREVIOUS_ACTION_OBSERVATION_VARIABLES)
    if values.shape[-1] != expected_width:
        raise ValueError(
            f'State-aware observation requires control observation width {expected_width}, '
            f'got {values.shape[-1]}'
        )
    encoded = np.asarray(encode_state_regime(classification), dtype=np.float32)
    return np.concatenate((values, encoded), axis=-1).astype(np.float32, copy=False)


def _finite_float(row: Mapping[str, str], field: str, path: Path, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(value):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return value


def _effective_action(row: Mapping[str, str], path: Path, row_number: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray([
        _finite_float(row, ACTION_VARIABLES[0], path, row_number),
        _finite_float(row, ACTION_VARIABLES[1], path, row_number),
    ], dtype=np.float64)
    effective = np.clip(
        raw,
        np.asarray([ENGINEERING_DEMAND_LOW_NM, NORMAL_GENERATION_PITCH_LOW_RAD]),
        np.asarray([ENGINEERING_DEMAND_HIGH_NM, NORMAL_GENERATION_PITCH_HIGH_RAD]),
    )
    return raw, effective


def _quantiles(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {
            'sample_count': 0,
            'action_abs_p005': None,
            'action_abs_p500': None,
            'action_abs_p995': None,
            'action_delta_abs_p005': None,
            'action_delta_abs_p500': None,
            'action_delta_abs_p995': None,
        }
    actions = np.stack([value[:2] for value in values], axis=0)
    deltas = np.stack([value[2:] for value in values], axis=0)
    return {
        'sample_count': int(len(values)),
        'action_abs_p005': np.quantile(actions, 0.005, axis=0).astype(float).tolist(),
        'action_abs_p500': np.quantile(actions, 0.5, axis=0).astype(float).tolist(),
        'action_abs_p995': np.quantile(actions, 0.995, axis=0).astype(float).tolist(),
        'action_delta_abs_p005': np.quantile(deltas, 0.005, axis=0).astype(float).tolist(),
        'action_delta_abs_p500': np.quantile(deltas, 0.5, axis=0).astype(float).tolist(),
        'action_delta_abs_p995': np.quantile(deltas, 0.995, axis=0).astype(float).tolist(),
    }


def _empty_counts(values: tuple) -> dict[str, int]:
    return {value.value: 0 for value in values}


def _scan_raw_file(
    path: Path,
    condition_id: int | None,
    config: OperatingStateConfig,
    *,
    require_full_window: bool,
    collect_action_statistics: bool,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    required = {TIME_FIELD, WIND_FIELD, ROTOR_SPEED_FIELD, REAL_POWER_VARIABLE, PITCH_FIELD}
    if collect_action_statistics:
        required.update(ACTION_VARIABLES)
    classifier = OperatingStateClassifier(config)
    state_counts = _empty_counts(OPTIMIZATION_STATE_ENCODING_ORDER)
    region_counts = _empty_counts(CONTROL_REGION_ENCODING_ORDER)
    state_transitions = 0
    region_transitions = 0
    previous_action: np.ndarray | None = None
    previous_time: float | None = None
    first_time: float | None = None
    last_time: float | None = None
    rows = 0
    raw_demand_clip_count = 0
    statistics = {state.value: [] for state in OPTIMIZATION_STATE_ENCODING_ORDER}
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing state artifact fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            timestamp = _finite_float(row, TIME_FIELD, path, row_number)
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            wind_speed = _finite_float(row, WIND_FIELD, path, row_number)
            rotor_speed = _finite_float(row, ROTOR_SPEED_FIELD, path, row_number)
            real_power = _finite_float(row, REAL_POWER_VARIABLE, path, row_number)
            pitch = _finite_float(row, PITCH_FIELD, path, row_number)
            if collect_action_statistics:
                raw_action, effective_action = _effective_action(row, path, row_number)
                raw_demand_clip_count += int(raw_action[0] > ENGINEERING_DEMAND_HIGH_NM)
            else:
                effective_action = np.asarray([ENGINEERING_DEMAND_LOW_NM, pitch], dtype=np.float64)
            prior_action = effective_action if previous_action is None else previous_action
            classification = classifier.classify(OperatingPoint(
                timestamp_s=timestamp,
                wind_speed_mps=wind_speed,
                rotor_speed_rad_s=rotor_speed,
                real_power_w=real_power,
                pitch_rad=pitch,
                previous_torque_nm=float(prior_action[0]),
                previous_pitch_rad=float(prior_action[1]),
            ))
            state_counts[classification.optimization_state.value] += 1
            region_counts[classification.control_region.value] += 1
            state_transitions += int(classification.operating_state_changed)
            region_transitions += int(classification.control_region_changed)
            if collect_action_statistics and previous_action is not None:
                statistics[classification.optimization_state.value].append(np.concatenate((
                    effective_action,
                    effective_action - previous_action,
                )))
            previous_action = effective_action
            previous_time = timestamp
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            rows += 1
    if rows < 2:
        raise ValueError(f'{path.name} must contain at least two rows')
    if require_full_window and not (
        math.isclose(float(first_time), EXPECTED_START_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
        and math.isclose(float(last_time), EXPECTED_END_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
    ):
        raise ValueError(f'{path.name} does not cover the required 0-600 s time range')
    return {
        'condition_id': condition_id,
        'filename': path.name,
        'path': str(path),
        'sha256': sha256_file(path),
        'rows': rows,
        'start_time_s': first_time,
        'end_time_s': last_time,
        'optimization_state_counts': state_counts,
        'control_region_counts': region_counts,
        'operating_state_transition_count': state_transitions,
        'control_region_transition_count': region_transitions,
        'raw_recorded_demand_above_engineering_cap_count': raw_demand_clip_count,
    }, statistics


def _default_soft_envelopes() -> dict[str, dict[str, list[float]]]:
    return {
        OptimizationState.STARTUP_STOPPED.value: {
            'residual_limit_abs': [0.0, 0.0],
        },
        OptimizationState.BELOW_RATED.value: {
            'residual_limit_abs': [15_000.0, 0.00030],
        },
        OptimizationState.TRANSITION.value: {
            'residual_limit_abs': [15_000.0, 0.00035],
        },
        OptimizationState.ABOVE_RATED.value: {
            'residual_limit_abs': [20_000.0, 0.00035],
        },
        OptimizationState.NEGATIVE_POWER_ABNORMAL.value: {
            'residual_limit_abs': [0.0, 0.0],
        },
    }


def _merge_counts(audits: list[dict[str, Any]], key: str, values: tuple) -> dict[str, int]:
    return {
        value.value: sum(int(audit[key][value.value]) for audit in audits)
        for value in values
    }


def build_state_action_artifact(
    data_root: Path | str,
    *,
    raw_infer_root: Path | str | None = None,
    calibration: OperatingStateConfig | None = None,
) -> dict[str, Any]:
    config = calibration or OperatingStateConfig()
    resolved_root, source_files = validate_raw_source_root(Path(data_root))
    source_audits: list[dict[str, Any]] = []
    statistics_by_state = {state.value: [] for state in OPTIMIZATION_STATE_ENCODING_ORDER}
    for condition_id, path in source_files:
        audit, statistics = _scan_raw_file(
            path,
            condition_id,
            config,
            require_full_window=True,
            collect_action_statistics=True,
        )
        source_audits.append(audit)
        for state, rows in statistics.items():
            statistics_by_state[state].extend(rows)
    validation_audits: list[dict[str, Any]] = []
    if raw_infer_root is not None:
        infer_root = Path(raw_infer_root).expanduser().resolve()
        if not infer_root.is_dir():
            raise FileNotFoundError(f'raw_infer validation root does not exist: {infer_root}')
        for path in sorted(infer_root.glob('data_windspeed*.csv')):
            audit, _ = _scan_raw_file(
                path,
                None,
                config,
                require_full_window=False,
                collect_action_statistics=False,
            )
            validation_audits.append(audit)
    raw_manifest = manifest_sha256([
        {
            'condition_id': audit['condition_id'],
            'filename': audit['filename'],
            'sha256': audit['sha256'],
        }
        for audit in source_audits
    ])
    state_statistics = {
        state.value: _quantiles(statistics_by_state[state.value])
        for state in OPTIMIZATION_STATE_ENCODING_ORDER
    }
    return {
        'schema_version': STATE_ACTION_ARTIFACT_SCHEMA_VERSION,
        'metadata': {
            'state_calibration_version': STATE_CALIBRATION_VERSION,
            'operating_state_contract': config.to_metadata(),
            'source_role': 'raw_pid_calibration',
            'source_data_root': str(resolved_root),
            'source_file_pattern': 'data_windspeed{condition_id}.csv',
            'expected_condition_ids': list(EXPECTED_CONDITION_IDS),
            'input_manifest_sha256': raw_manifest,
            'input_manifest_format': 'condition_id:filename:file_sha256\\n sorted by condition_id',
            'causal': True,
            'future_feature_lookahead_steps': 0,
            'action_semantics': 'effective_pid_demand_clipped_at_engineering_cap',
            'raw_recorded_demand_field': ACTION_VARIABLES[0],
            'effective_pid_demand_high_nm': ENGINEERING_DEMAND_HIGH_NM,
            'raw_source_files': source_audits,
            'raw_infer_validation_files': validation_audits,
        },
        'observation_schema': state_aware_observation_schema(),
        'state_coverage': {
            'optimization_state_counts': _merge_counts(
                source_audits,
                'optimization_state_counts',
                OPTIMIZATION_STATE_ENCODING_ORDER,
            ),
            'control_region_counts': _merge_counts(
                source_audits,
                'control_region_counts',
                CONTROL_REGION_ENCODING_ORDER,
            ),
            'operating_state_transition_count': sum(
                int(audit['operating_state_transition_count']) for audit in source_audits
            ),
            'control_region_transition_count': sum(
                int(audit['control_region_transition_count']) for audit in source_audits
            ),
            'raw_infer_optimization_state_counts': _merge_counts(
                validation_audits,
                'optimization_state_counts',
                OPTIMIZATION_STATE_ENCODING_ORDER,
            ),
            'raw_infer_control_region_counts': _merge_counts(
                validation_audits,
                'control_region_counts',
                CONTROL_REGION_ENCODING_ORDER,
            ),
        },
        'action_statistics': state_statistics,
        'soft_residual_envelopes': {
            'units': ['N m', 'rad'],
            'interpolation': 'smoothstep_by_filtered_wind_within_hysteresis_bands',
            'engineering_rate_limit_abs': [90_000.0, 0.000698],
            'states': _default_soft_envelopes(),
        },
    }


@dataclass(frozen=True)
class StateActionEnvelopeArtifact:
    document: dict[str, Any]
    config: OperatingStateConfig
    source_sha256: str

    @classmethod
    def from_path(cls, path: Path | str) -> StateActionEnvelopeArtifact:
        resolved = Path(path).expanduser()
        try:
            payload = resolved.read_bytes()
            document = json.loads(payload)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'State/action artifact does not exist: {resolved}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read state/action artifact: {resolved}') from exc
        return cls.from_document(document, source_sha256=hashlib.sha256(payload).hexdigest())

    @classmethod
    def from_document(
        cls,
        document: Any,
        *,
        source_sha256: str | None = None,
    ) -> StateActionEnvelopeArtifact:
        if not isinstance(document, dict) or document.get('schema_version') != STATE_ACTION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError('State/action artifact schema is incompatible')
        metadata = document.get('metadata')
        observation_schema = document.get('observation_schema')
        envelopes = document.get('soft_residual_envelopes')
        if not isinstance(metadata, dict) or metadata.get('state_calibration_version') != STATE_CALIBRATION_VERSION:
            raise ValueError('State/action artifact calibration version is incompatible')
        operating_contract = metadata.get('operating_state_contract')
        if (
            not isinstance(operating_contract, dict)
            or operating_contract.get('contract_version') != OPERATING_STATE_CONTRACT_VERSION
            or operating_contract.get('causal') is not True
        ):
            raise ValueError('State/action artifact operating-state contract is incompatible')
        thresholds = operating_contract.get('thresholds')
        if not isinstance(thresholds, dict):
            raise ValueError('State/action artifact state thresholds are missing')
        try:
            config = OperatingStateConfig(**thresholds)
        except (TypeError, ValueError) as exc:
            raise ValueError('State/action artifact state thresholds are invalid') from exc
        if observation_schema != state_aware_observation_schema():
            raise ValueError('State/action artifact observation schema is incompatible')
        if (
            not isinstance(envelopes, dict)
            or envelopes.get('units') != ['N m', 'rad']
            or envelopes.get('interpolation') != 'smoothstep_by_filtered_wind_within_hysteresis_bands'
        ):
            raise ValueError('State/action artifact soft residual envelope metadata is incompatible')
        rate_limit = np.asarray(envelopes.get('engineering_rate_limit_abs'), dtype=np.float64)
        if rate_limit.shape != (2,) or not np.allclose(rate_limit, [90_000.0, 0.000698], rtol=0.0, atol=0.0):
            raise ValueError('State/action artifact engineering rate limit is incompatible')
        states = envelopes.get('states')
        if not isinstance(states, dict) or set(states) != {state.value for state in OptimizationState}:
            raise ValueError('State/action artifact soft residual envelope states are incompatible')
        for state in OptimizationState:
            values = np.asarray(states[state.value].get('residual_limit_abs'), dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f'State/action artifact residual envelope is invalid: {state.value}')
            if np.any(values > rate_limit + 1.0e-12):
                raise ValueError(f'State/action artifact residual envelope exceeds engineering rate limit: {state.value}')
        if source_sha256 is None:
            source_sha256 = hashlib.sha256(
                json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
            ).hexdigest()
        return cls(document=document, config=config, source_sha256=source_sha256)

    @property
    def observation_schema(self) -> dict[str, Any]:
        return dict(self.document['observation_schema'])

    def classifier(self) -> OperatingStateClassifier:
        return OperatingStateClassifier(self.config)

    def residual_limit_for(self, classification) -> np.ndarray:
        state = classification.optimization_state
        if state in {OptimizationState.STARTUP_STOPPED, OptimizationState.NEGATIVE_POWER_ABNORMAL}:
            return np.zeros(2, dtype=np.float64)
        states = self.document['soft_residual_envelopes']['states']
        below = np.asarray(states[OptimizationState.BELOW_RATED.value]['residual_limit_abs'], dtype=np.float64)
        transition = np.asarray(states[OptimizationState.TRANSITION.value]['residual_limit_abs'], dtype=np.float64)
        above = np.asarray(states[OptimizationState.ABOVE_RATED.value]['residual_limit_abs'], dtype=np.float64)
        wind = float(classification.filtered_wind_speed_mps)
        if wind <= self.config.transition_exit_wind_mps:
            return below
        if wind < self.config.transition_enter_wind_mps:
            return _smoothstep_blend(
                below,
                transition,
                wind,
                self.config.transition_exit_wind_mps,
                self.config.transition_enter_wind_mps,
            )
        if wind <= self.config.above_rated_exit_wind_mps:
            return transition
        if wind < self.config.above_rated_enter_wind_mps:
            return _smoothstep_blend(
                transition,
                above,
                wind,
                self.config.above_rated_exit_wind_mps,
                self.config.above_rated_enter_wind_mps,
            )
        return above

    def reward_state_weights(self, classification) -> dict[str, float]:
        state = classification.optimization_state
        weights = {value.value: 0.0 for value in OptimizationState}
        if state in {OptimizationState.STARTUP_STOPPED, OptimizationState.NEGATIVE_POWER_ABNORMAL}:
            weights[state.value] = 1.0
            return weights
        wind = float(classification.filtered_wind_speed_mps)
        if wind <= self.config.transition_exit_wind_mps:
            weights[OptimizationState.BELOW_RATED.value] = 1.0
            return weights
        if wind < self.config.transition_enter_wind_mps:
            transition_weight = _smoothstep_fraction(
                wind,
                self.config.transition_exit_wind_mps,
                self.config.transition_enter_wind_mps,
            )
            weights[OptimizationState.BELOW_RATED.value] = 1.0 - transition_weight
            weights[OptimizationState.TRANSITION.value] = transition_weight
            return weights
        if wind <= self.config.above_rated_exit_wind_mps:
            weights[OptimizationState.TRANSITION.value] = 1.0
            return weights
        if wind < self.config.above_rated_enter_wind_mps:
            above_weight = _smoothstep_fraction(
                wind,
                self.config.above_rated_exit_wind_mps,
                self.config.above_rated_enter_wind_mps,
            )
            weights[OptimizationState.TRANSITION.value] = 1.0 - above_weight
            weights[OptimizationState.ABOVE_RATED.value] = above_weight
            return weights
        weights[OptimizationState.ABOVE_RATED.value] = 1.0
        return weights


def _smoothstep_blend(start: np.ndarray, end: np.ndarray, value: float, low: float, high: float) -> np.ndarray:
    fraction = _smoothstep_fraction(value, low, high)
    return start + fraction * (end - start)


def _smoothstep_fraction(value: float, low: float, high: float) -> float:
    fraction = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)