from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.raw_dataset_contract import (
    EXPECTED_CONDITION_IDS,
    EXPECTED_END_TIME_S,
    EXPECTED_START_TIME_S,
    TIME_TOLERANCE_S,
    manifest_sha256 as build_manifest_sha256,
    sha256_file,
    validate_raw_source_root,
    write_json,
)
from RL.utils.constants import ACTION_VARIABLES, DATA_SCHEMA_VERSION, DISTURBANCE_VARIABLES, REAL_POWER_VARIABLE
from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM
from RL.utils.operating_state import ControlRegion, OperatingPoint, OperatingStateClassifier, OperatingStateConfig
from RL.utils.region_aware_baseline import REGION_AWARE_BASELINE_FEATURES, REGION_AWARE_BASELINE_SCHEMA_VERSION
from RL.utils.state_action_artifact import StateActionEnvelopeArtifact


TIME_FIELD = 'Time'
ROTOR_SPEED_FIELD = 'I_RotorSpeed'
GENERATOR_SPEED_FIELD = 'I_GeneratorSpeed'
DEFAULT_ACTION_LOW = np.asarray([0.0, 0.0], dtype=np.float64)
DEFAULT_ACTION_HIGH = np.asarray([ENGINEERING_DEMAND_HIGH_NM, 0.43], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a causal region-aware joint torque/pitch baseline from all complete raw PID traces.'
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--min-region-transitions', type=int, default=500)
    parser.add_argument('--torque-delta-limit', type=float, default=15_000.0)
    parser.add_argument('--pitch-delta-limit', type=float, default=0.0003)
    parser.add_argument('--state-action-artifact', type=Path, default=None)
    return parser.parse_args()


def _finite(row: dict[str, str], field: str, path: Path, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(value):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return value


def _load_provenance(path: Path, manifest_sha256: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'PID reference provenance does not exist: {path}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read PID reference provenance: {path}') from exc
    if not isinstance(document, dict) or document.get('schema_version') != 'pid_reference_provenance_v1':
        raise ValueError('PID reference provenance schema is incompatible')
    source_manifest = document.get('source_manifest')
    if not isinstance(source_manifest, dict) or source_manifest.get('manifest_sha256') != manifest_sha256:
        raise ValueError('PID reference provenance manifest does not match the raw baseline source')
    decision = document.get('decision')
    if not isinstance(decision, dict) or not (
        decision.get('native_pid_demand_available') is False
        and decision.get('udp_action_contract') == 'recorded_pid_demand'
        and decision.get('baseline_reference_path') == 'region_aware_surrogate_required'
    ):
        raise ValueError('PID reference provenance does not authorize the region-aware surrogate path')
    return document


def _read_condition(
    path: Path,
    condition_id: int,
    features_by_region: dict[ControlRegion, list[np.ndarray]],
    targets_by_region: dict[ControlRegion, list[np.ndarray]],
    deltas_by_region: dict[ControlRegion, list[np.ndarray]],
    operating_state_config: OperatingStateConfig,
) -> dict[str, Any]:
    required = {
        TIME_FIELD,
        DISTURBANCE_VARIABLES[0],
        ROTOR_SPEED_FIELD,
        GENERATOR_SPEED_FIELD,
        REAL_POWER_VARIABLE,
        *ACTION_VARIABLES,
    }
    classifier = OperatingStateClassifier(operating_state_config)
    rows_used = 0
    transitions_used = 0
    first_time: float | None = None
    last_time: float | None = None
    previous_timestamp: float | None = None
    previous_features: np.ndarray | None = None
    previous_action: np.ndarray | None = None
    previous_region: ControlRegion | None = None
    region_transition_counts = {region.value: 0 for region in ControlRegion}

    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing required region-aware baseline fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            timestamp = _finite(row, TIME_FIELD, path, row_number)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            wind_speed = _finite(row, DISTURBANCE_VARIABLES[0], path, row_number)
            rotor_speed = _finite(row, ROTOR_SPEED_FIELD, path, row_number)
            generator_speed = _finite(row, GENERATOR_SPEED_FIELD, path, row_number)
            real_power = _finite(row, REAL_POWER_VARIABLE, path, row_number)
            action = np.asarray([
                _finite(row, ACTION_VARIABLES[0], path, row_number),
                _finite(row, ACTION_VARIABLES[1], path, row_number),
            ], dtype=np.float64)
            classification = classifier.classify(OperatingPoint(
                timestamp_s=timestamp,
                wind_speed_mps=wind_speed,
                rotor_speed_rad_s=rotor_speed,
                real_power_w=real_power,
            ))
            features = np.asarray([
                classification.filtered_wind_speed_mps,
                rotor_speed,
                generator_speed,
                real_power,
                action[0],
                action[1],
            ], dtype=np.float64)
            if previous_features is not None and previous_action is not None and previous_region is not None:
                features_by_region[previous_region].append(previous_features)
                targets_by_region[previous_region].append(action.copy())
                deltas_by_region[previous_region].append(action - previous_action)
                region_transition_counts[previous_region.value] += 1
                transitions_used += 1
            if first_time is None:
                first_time = timestamp
            last_time = timestamp
            previous_timestamp = timestamp
            previous_features = features
            previous_action = action
            previous_region = classification.control_region
            rows_used += 1

    if rows_used < 2:
        raise ValueError(f'{path.name} needs at least two valid rows')
    if not (
        math.isclose(float(first_time), EXPECTED_START_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
        and math.isclose(float(last_time), EXPECTED_END_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
    ):
        raise ValueError(f'{path.name} does not cover the required 0-600 s time range')
    return {
        'condition_id': condition_id,
        'filename': path.name,
        'path': str(path),
        'sha256': sha256_file(path),
        'rows_used': rows_used,
        'transitions_used': transitions_used,
        'start_time_s': first_time,
        'end_time_s': last_time,
        'region_transition_counts': region_transition_counts,
    }


def _quantile(values: np.ndarray, fraction: float) -> list[float]:
    return np.quantile(values, fraction, axis=0).astype(float).tolist()


def _fit_region_model(
    region: ControlRegion,
    feature_rows: list[np.ndarray],
    target_rows: list[np.ndarray],
    delta_rows: list[np.ndarray],
    *,
    ridge_alpha: float,
    min_region_transitions: int,
    delta_limit: np.ndarray,
) -> dict[str, Any]:
    if len(feature_rows) < min_region_transitions:
        raise RuntimeError(
            f'Region {region.value} has {len(feature_rows)} transitions; '
            f'at least {min_region_transitions} are required'
        )
    features = np.stack(feature_rows).astype(np.float64)
    deltas = np.stack(delta_rows).astype(np.float64)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.maximum(
        feature_scale,
        np.asarray([0.1, 0.01, 0.01, 10_000.0, 100_000.0, 0.001], dtype=np.float64),
    )
    normalized_features = (features - feature_mean) / feature_scale
    target_mean = np.mean(deltas, axis=0)
    centered_targets = deltas - target_mean
    gram = normalized_features.T @ normalized_features
    coefficient_matrix = np.linalg.solve(
        gram + ridge_alpha * np.eye(gram.shape[0], dtype=np.float64),
        normalized_features.T @ centered_targets,
    )
    predictions = target_mean + normalized_features @ coefficient_matrix
    residuals = deltas - predictions
    covariance = np.cov(residuals, rowvar=False, bias=True)
    if covariance.shape != (2, 2):
        covariance = np.zeros((2, 2), dtype=np.float64)
    relation_by_region = {
        ControlRegion.BELOW_RATED: 'mppt_speed_power_with_joint_action_history',
        ControlRegion.TRANSITION: 'hysteretic_joint_torque_pitch_transition',
        ControlRegion.ABOVE_RATED: 'rated_power_speed_pitch_regulation_with_joint_action_history',
    }
    return {
        'relation': 'joint_next_action_delta',
        'physical_relation': relation_by_region[region],
        'sample_count': len(features),
        'feature_mean': feature_mean.astype(float).tolist(),
        'feature_scale': feature_scale.astype(float).tolist(),
        'intercept': target_mean.astype(float).tolist(),
        'coefficient_matrix': coefficient_matrix.astype(float).tolist(),
        'action_delta_low': (-delta_limit).astype(float).tolist(),
        'action_delta_high': delta_limit.astype(float).tolist(),
        'joint_residual_covariance': covariance.astype(float).tolist(),
        'fit_diagnostics': {
            'action_mae': np.mean(np.abs(residuals), axis=0).astype(float).tolist(),
            'action_rmse': np.sqrt(np.mean(np.square(residuals), axis=0)).astype(float).tolist(),
            'observed_delta_p005': _quantile(deltas, 0.005),
            'observed_delta_p500': _quantile(deltas, 0.5),
            'observed_delta_p995': _quantile(deltas, 0.995),
            'observed_delta_abs_p950': _quantile(np.abs(deltas), 0.95),
            'observed_delta_within_deployment_limit_fraction': float(np.mean(
                np.all(np.abs(deltas) <= delta_limit + 1.0e-12, axis=1)
            )),
        },
    }


def build_region_aware_baseline(
    data_root: Path,
    provenance_path: Path,
    *,
    ridge_alpha: float,
    min_region_transitions: int,
    delta_limit: np.ndarray,
    state_action_artifact_path: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise ValueError('ridge_alpha must be finite and non-negative')
    if min_region_transitions <= 0:
        raise ValueError('min_region_transitions must be positive')
    if delta_limit.shape != (2,) or not np.isfinite(delta_limit).all() or np.any(delta_limit <= 0.0):
        raise ValueError('Action delta limits must contain two positive finite values')
    state_artifact = (
        None
        if state_action_artifact_path is None
        else StateActionEnvelopeArtifact.from_path(state_action_artifact_path)
    )
    state_config = OperatingStateConfig() if state_artifact is None else state_artifact.config
    resolved_root, source_files = validate_raw_source_root(data_root)
    features_by_region = {region: [] for region in ControlRegion}
    targets_by_region = {region: [] for region in ControlRegion}
    deltas_by_region = {region: [] for region in ControlRegion}
    file_audits = [
        _read_condition(
            path,
            condition_id,
            features_by_region,
            targets_by_region,
            deltas_by_region,
            state_config,
        )
        for condition_id, path in source_files
    ]
    manifest_sha256 = build_manifest_sha256(file_audits)
    if (
        state_artifact is not None
        and state_artifact.document['metadata']['input_manifest_sha256'] != manifest_sha256
    ):
        raise ValueError('State/action artifact raw manifest does not match the B4 baseline source')
    provenance = _load_provenance(provenance_path, manifest_sha256)
    models = {
        region.value: _fit_region_model(
            region,
            features_by_region[region],
            targets_by_region[region],
            deltas_by_region[region],
            ridge_alpha=ridge_alpha,
            min_region_transitions=min_region_transitions,
            delta_limit=delta_limit,
        )
        for region in ControlRegion
    }
    return {
        'schema_version': REGION_AWARE_BASELINE_SCHEMA_VERSION,
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_role': 'raw_pid_calibration',
            'source_data_root': str(resolved_root),
            'source_file_pattern': 'data_windspeed{condition_id}.csv',
            'expected_condition_ids': list(EXPECTED_CONDITION_IDS),
            'expected_time_range_s': [EXPECTED_START_TIME_S, EXPECTED_END_TIME_S],
            'sampling_strategy': 'all_valid_causal_transitions',
            'files_used': len(file_audits),
            'rows_used': sum(int(item['rows_used']) for item in file_audits),
            'transitions_used': sum(int(item['transitions_used']) for item in file_audits),
            'input_manifest_sha256': manifest_sha256,
            'input_manifest_format': 'condition_id:filename:file_sha256\\n sorted by condition_id',
            'provenance_path': str(provenance_path),
            'provenance_sha256': sha256_file(provenance_path),
            'action_source': list(ACTION_VARIABLES),
            'action_semantics': provenance['decision']['udp_action_contract'],
            'action_units': ['N m', 'rad'],
            'operating_state_contract': state_config.to_metadata(),
            'state_action_artifact_sha256': (
                None if state_artifact is None else state_artifact.source_sha256
            ),
            'file_audits': file_audits,
        },
        'control': {
            'region_aware_joint_baseline': {
                'feature_names': list(REGION_AWARE_BASELINE_FEATURES),
                'target': 'next_realized_action_delta',
                'action_units': ['N m', 'rad'],
                'action_low': DEFAULT_ACTION_LOW.astype(float).tolist(),
                'action_high': DEFAULT_ACTION_HIGH.astype(float).tolist(),
                'causal': True,
                'future_feature_lookahead_steps': 0,
                'joint_generation_contract': (
                    'one multi-output model per control region predicts next torque and pitch deltas together; '
                    'both previous action channels are features for both outputs'
                ),
                'ridge_alpha': ridge_alpha,
                'models': models,
                'transition_blending': {
                    'method': 'smoothstep_with_hysteretic_region',
                    'transition_exit_wind_mps': state_config.transition_exit_wind_mps,
                    'transition_enter_wind_mps': state_config.transition_enter_wind_mps,
                    'above_rated_exit_wind_mps': state_config.above_rated_exit_wind_mps,
                    'above_rated_enter_wind_mps': state_config.above_rated_enter_wind_mps,
                },
            }
        },
    }


def main() -> None:
    args = parse_args()
    document = build_region_aware_baseline(
        args.data_root,
        args.provenance,
        ridge_alpha=float(args.ridge_alpha),
        min_region_transitions=int(args.min_region_transitions),
        delta_limit=np.asarray(
            [float(args.torque_delta_limit), float(args.pitch_delta_limit)],
            dtype=np.float64,
        ),
        state_action_artifact_path=(
            None
            if args.state_action_artifact is None
            else args.state_action_artifact.expanduser().resolve()
        ),
    )
    write_json(args.output, document)
    print(json.dumps({
        'output': str(args.output),
        'schema_version': document['schema_version'],
        'input_manifest_sha256': document['metadata']['input_manifest_sha256'],
        'files_used': document['metadata']['files_used'],
        'rows_used': document['metadata']['rows_used'],
        'transitions_used': document['metadata']['transitions_used'],
        'region_transition_counts': {
            name: model['sample_count']
            for name, model in document['control']['region_aware_joint_baseline']['models'].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()