from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from RL.utils.constants import ACTION_VARIABLES, RAW_OBSERVATION_VARIABLES
from RL.utils.control_contract import (
    ENGINEERING_DEMAND_HIGH_NM,
    ENGINEERING_DEMAND_LOW_NM,
    NORMAL_GENERATION_PITCH_HIGH_RAD,
    NORMAL_GENERATION_PITCH_LOW_RAD,
)
from RL.utils.observation import build_observation_from_frame
from RL.utils.operating_state import OperatingPoint
from RL.utils.region_aware_baseline import RegionAwareJointBaseline
from RL.utils.state_action_artifact import StateActionEnvelopeArtifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = ('B4 residual torque target', 'B4 residual pitch target')
RAW_RECORDED_TORQUE_COLUMN = 'raw_recorded_pid_demand'
EFFECTIVE_TORQUE_COLUMN = 'effective_pid_demand'
RAW_RECORDED_PITCH_COLUMN = 'raw_recorded_pid_pitch_demand'
EFFECTIVE_PITCH_COLUMN = 'effective_pid_pitch_demand'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_id(path: Path) -> int:
    return int(path.stem.replace('data_windspeed', ''))


def _build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    baseline_path: Path,
    residual_limit: np.ndarray,
    residual_target_scale: float,
    state_action_artifact: StateActionEnvelopeArtifact,
) -> list[dict[str, Any]]:
    paths = sorted(source_root.glob('data_windspeed*.csv'), key=_condition_id)
    if not paths:
        raise FileNotFoundError(f'No PID CSV files found in {source_root}')
    destination = output_root / split
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for path in paths}
    for stale_path in destination.glob('data_windspeed*.csv'):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    unexpected_csv = sorted(
        path.name for path in destination.glob('*.csv') if path.name not in expected_names
    )
    if unexpected_csv:
        raise ValueError(
            f'{destination} contains CSV files not owned by this builder: {unexpected_csv}'
        )
    baseline = RegionAwareJointBaseline.from_path(baseline_path, step_duration_s=0.02)
    audits: list[dict[str, Any]] = []
    required = list(dict.fromkeys(['Time', *RAW_OBSERVATION_VARIABLES, *ACTION_VARIABLES]))
    for source_path in paths:
        frame = pd.read_csv(source_path, usecols=required)
        observations = build_observation_from_frame(frame)
        raw_actions = frame[ACTION_VARIABLES].to_numpy(dtype=np.float64)
        actions = np.clip(
            raw_actions,
            np.asarray([ENGINEERING_DEMAND_LOW_NM, NORMAL_GENERATION_PITCH_LOW_RAD]),
            np.asarray([ENGINEERING_DEMAND_HIGH_NM, NORMAL_GENERATION_PITCH_HIGH_RAD]),
        )
        baseline.reset()
        baseline_actions = np.empty_like(actions)
        for index, observation in enumerate(observations):
            previous_action = actions[index - 1] if index else actions[index]
            baseline_actions[index] = baseline.predict(
                observation,
                previous_action_real=previous_action,
            )
        state_limits = np.broadcast_to(residual_limit, actions.shape).copy()
        if baseline.operating_state_config != state_action_artifact.config:
            raise ValueError('B4 baseline and state/action artifact operating-state calibrations do not match')
        classifier = state_action_artifact.classifier()
        state_counts: dict[str, int] = {}
        for row_index in range(len(frame)):
            previous_action = actions[row_index - 1] if row_index else actions[row_index]
            classification = classifier.classify(OperatingPoint(
                timestamp_s=float(frame['Time'].iat[row_index]),
                wind_speed_mps=float(frame['I_AverageWindSpeed'].iat[row_index]),
                rotor_speed_rad_s=float(frame['I_RotorSpeed'].iat[row_index]),
                real_power_w=float(frame['I_RealPower'].iat[row_index]),
                pitch_rad=float(frame['I_PitchPosition1'].iat[row_index]),
                previous_torque_nm=float(previous_action[0]),
                previous_pitch_rad=float(previous_action[1]),
            ))
            state_limits[row_index] = state_action_artifact.residual_limit_for(classification)
            state_counts[classification.optimization_state.value] = (
                state_counts.get(classification.optimization_state.value, 0) + 1
            )
        residual_raw = np.divide(
            actions - baseline_actions,
            np.maximum(state_limits, 1.0e-12),
            out=np.zeros_like(actions),
            where=state_limits > 0.0,
        )
        residual_target = (
            np.clip(residual_raw, -1.0, 1.0) * residual_target_scale
        ).astype(np.float32)
        output_frame = frame.copy()
        output_frame[RAW_RECORDED_TORQUE_COLUMN] = raw_actions[:, 0]
        output_frame[EFFECTIVE_TORQUE_COLUMN] = actions[:, 0]
        output_frame[RAW_RECORDED_PITCH_COLUMN] = raw_actions[:, 1]
        output_frame[EFFECTIVE_PITCH_COLUMN] = actions[:, 1]
        output_frame[ACTION_VARIABLES[0]] = actions[:, 0]
        output_frame[ACTION_VARIABLES[1]] = actions[:, 1]
        output_frame[TARGET_COLUMNS[0]] = residual_target[:, 0]
        output_frame[TARGET_COLUMNS[1]] = residual_target[:, 1]
        output_path = destination / source_path.name
        output_frame.to_csv(output_path, index=False)
        audits.append({
            'condition_id': _condition_id(source_path),
            'source_path': source_path.relative_to(PROJECT_ROOT).as_posix(),
            'source_sha256': _sha256(source_path),
            'output_path': output_path.relative_to(PROJECT_ROOT).as_posix(),
            'output_sha256': _sha256(output_path),
            'rows': int(len(frame)),
            'raw_recorded_demand_above_engineering_cap_count': int(
                np.sum(raw_actions[:, 0] > ENGINEERING_DEMAND_HIGH_NM)
            ),
            'optimization_state_counts': state_counts,
            'torque_clip_fraction': float(np.mean(np.abs(residual_raw[:, 0]) > 1.0)),
            'pitch_clip_fraction': float(np.mean(np.abs(residual_raw[:, 1]) > 1.0)),
        })
    return audits


def build(
    *,
    train_root: Path,
    val_root: Path,
    baseline_path: Path,
    output_root: Path,
    residual_limit: tuple[float, float],
    residual_target_scale: float = 0.25,
    state_action_artifact_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    limit = np.asarray(residual_limit, dtype=np.float64)
    if limit.shape != (2,) or not np.isfinite(limit).all() or np.any(limit <= 0.0):
        raise ValueError('residual_limit must contain two positive finite values')
    if not np.isfinite(residual_target_scale) or not 0.0 < residual_target_scale <= 1.0:
        raise ValueError('residual_target_scale must be finite and in (0, 1]')
    if state_action_artifact_path is None:
        state_action_artifact_path = PROJECT_ROOT / 'RL/data/state_action_envelope_3_to_25_v1.json'
    state_action_artifact = StateActionEnvelopeArtifact.from_path(state_action_artifact_path)
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f'Refusing to overwrite existing residual dataset directory: {output_root}. '
            'Pass overwrite=True only when intentionally rebuilding the canonical dataset.'
        )
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f'Residual dataset output path is not a directory: {output_root}')
    if overwrite:
        (output_root / 'manifest.json').unlink(missing_ok=True)
        cache_root = output_root / '.rl_cache' / 'offline_bc'
        if cache_root.exists():
            shutil.rmtree(cache_root)
    train_audits = _build_split(
        train_root,
        output_root,
        'train',
        baseline_path,
        limit,
        residual_target_scale,
        state_action_artifact,
    )
    val_audits = _build_split(
        val_root,
        output_root,
        'val',
        baseline_path,
        limit,
        residual_target_scale,
        state_action_artifact,
    )
    all_train_rows = sum(row['rows'] for row in train_audits)
    all_val_rows = sum(row['rows'] for row in val_audits)
    manifest = {
        'schema_version': 'b4_centered_residual_bc_data_v2',
        'action_contract': 'region_aware_b4_plus_bounded_residual',
        'baseline_path': baseline_path.relative_to(PROJECT_ROOT).as_posix(),
        'baseline_sha256': _sha256(baseline_path),
        'state_action_artifact_sha256': state_action_artifact.source_sha256,
        'observation_schema': state_action_artifact.observation_schema,
        'action_low': [ENGINEERING_DEMAND_LOW_NM, NORMAL_GENERATION_PITCH_LOW_RAD],
        'action_high': [ENGINEERING_DEMAND_HIGH_NM, NORMAL_GENERATION_PITCH_HIGH_RAD],
        'raw_recorded_action_columns': [RAW_RECORDED_TORQUE_COLUMN, RAW_RECORDED_PITCH_COLUMN],
        'effective_action_columns': [EFFECTIVE_TORQUE_COLUMN, EFFECTIVE_PITCH_COLUMN],
        'residual_limit': limit.astype(float).tolist(),
        'residual_target_scale': float(residual_target_scale),
        'target_columns': list(TARGET_COLUMNS),
        'include_previous_action_observation': True,
        'train_source_role': 'raw_pid_calibration',
        'validation_source_role': 'raw_infer_holdout',
        'train_rows': all_train_rows,
        'val_rows': all_val_rows,
        'train_files': train_audits,
        'val_files': val_audits,
        'train_torque_clip_fraction': float(sum(row['rows'] * row['torque_clip_fraction'] for row in train_audits) / all_train_rows),
        'train_pitch_clip_fraction': float(sum(row['rows'] * row['pitch_clip_fraction'] for row in train_audits) / all_train_rows),
    }
    manifest_path = output_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build causal PID-vs-B4 residual targets for BC.')
    parser.add_argument('--train-root', default='bladed-data-new/raw')
    parser.add_argument('--val-root', default='bladed-data-new/raw_infer')
    parser.add_argument('--baseline', default='RL/data/region_aware_joint_baseline_state20_3_to_25_v3.json')
    parser.add_argument('--output-root', default='RL/data/b4_centered_residual_bc_state20_v2')
    parser.add_argument('--torque-limit', type=float, default=15_000.0)
    parser.add_argument('--pitch-limit', type=float, default=0.0003)
    parser.add_argument('--residual-target-scale', type=float, default=0.25)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the fixed residual dataset directory after removing its old manifest.',
    )
    parser.add_argument(
        '--state-action-artifact',
        default='RL/data/state_action_envelope_3_to_25_v1.json',
        help='Required causal state/action artifact for state20 residual data.',
    )
    args = parser.parse_args()
    manifest = build(
        train_root=(PROJECT_ROOT / args.train_root).resolve(),
        val_root=(PROJECT_ROOT / args.val_root).resolve(),
        baseline_path=(PROJECT_ROOT / args.baseline).resolve(),
        output_root=(PROJECT_ROOT / args.output_root).resolve(),
        residual_limit=(args.torque_limit, args.pitch_limit),
        residual_target_scale=args.residual_target_scale,
        state_action_artifact_path=(PROJECT_ROOT / args.state_action_artifact).resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps({
        'train_rows': manifest['train_rows'],
        'val_rows': manifest['val_rows'],
        'train_torque_clip_fraction': manifest['train_torque_clip_fraction'],
        'train_pitch_clip_fraction': manifest['train_pitch_clip_fraction'],
    }, indent=2))


if __name__ == '__main__':
    main()