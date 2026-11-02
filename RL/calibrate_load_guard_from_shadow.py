from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.load_guard_calibration import (
    LoadGuardCalibrationConfig,
    LoadGuardCalibrationObservation,
    audit_load_guard_coverage,
    calibrate_load_guard,
)
from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.load_predictor_contract import (
    LoadPredictorInterface,
    PLAN_KIND_ZOH_CANDIDATE,
    make_zoh_candidate_plan,
    normalized_action_plan_mismatch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path, *, max_records: int) -> Iterable[dict[str, Any]]:
    count = 0
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode('utf-8')) > 64 * 1024:
                raise ValueError(f'Shadow audit record exceeds 64 KiB at {path}:{line_number}')
            count += 1
            if count > max_records:
                raise ValueError(f'Shadow audit input exceeds configured max_records={max_records}: {path}')
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid shadow audit JSON at {path}:{line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Shadow audit record is not an object at {path}:{line_number}')
            yield row


def _observations(
    path: Path,
    *,
    interface: LoadPredictorInterface,
    max_records: int,
    require_strict_identity: bool,
    minimum_nonoverlap_spacing_s: float,
    predictor_artifact_sha256: str,
    predictor_interface_sha256: str,
    paired_identity_by_condition: dict[str, Any],
) -> tuple[list[LoadGuardCalibrationObservation], dict[str, Any]]:
    observations: list[LoadGuardCalibrationObservation] = []
    seen_records = 0
    eligible_records = 0
    condition_ids: set[int] = set()
    strict_identity_failures = 0
    timestamps: list[float] = []
    selected_times_by_group: dict[tuple[int, str, float], list[float]] = {}
    nonoverlap_dropped = 0
    actor_policy_sha256s: set[str] = set()
    actor_policy_versions: set[int] = set()
    if not math.isfinite(minimum_nonoverlap_spacing_s) or minimum_nonoverlap_spacing_s <= 0.0:
        raise ValueError('minimum_nonoverlap_spacing_s must be positive and finite')
    for row in _iter_jsonl(path, max_records=max_records):
        seen_records += 1
        if row.get('schema_version') != 'load_predictor_shadow_audit_record_v1':
            raise ValueError(f'Shadow audit schema is incompatible: {path}')
        if row.get('predictor_artifact_sha256') != predictor_artifact_sha256:
            raise ValueError(f'Shadow audit predictor SHA does not match frozen artifact: {path}:{seen_records + 1}')
        if row.get('predictor_interface_sha256') != predictor_interface_sha256:
            raise ValueError(f'Shadow audit interface SHA does not match frozen interface: {path}:{seen_records + 1}')
        if row.get('plan_kind') != PLAN_KIND_ZOH_CANDIDATE:
            continue
        if (
            not bool(row.get('online_authorized', False))
            or not bool(row.get('calibration_eligible', False))
            or not bool(row.get('prediction_available', False))
            or not bool(row.get('predicted_response_available', False))
            or not bool(row.get('plan_match_complete', False))
            or not bool(row.get('actor_policy_window_frozen', False))
        ):
            continue
        if require_strict_identity and (
            row.get('identity_status') != 'strict_case_matched'
            or not bool(row.get('strict_case_matched', False))
            or not isinstance(row.get('pid_raw_sha256'), str)
            or len(str(row.get('pid_raw_sha256'))) != 64
            or any(character not in '0123456789abcdef' for character in str(row.get('pid_raw_sha256')).lower())
            or not isinstance(row.get('expected_trace_signature'), str)
            or len(str(row.get('expected_trace_signature'))) != 64
            or not isinstance(row.get('pid_raw_csv_path'), str)
            or not row.get('pid_raw_csv_path')
            or int(row.get('expected_trace_sample_count', 0)) < 2
        ):
            strict_identity_failures += 1
            continue
        if row.get('outcome_tag') != 'complete' or row.get('execution_mode') != 'actor':
            continue
        try:
            condition_id = int(row['condition_id'])
            horizon_s = float(row['horizon_s'])
            timestamp_s = float(row['timestamp_s'])
            plan_mismatch = float(row['plan_mismatch'])
            optimization_state = str(row['optimization_state'])
            predicted = [float(value) for value in row['predicted_peak_ratio_by_blade']]
            actual = [float(value) for value in row['actual_future_peak_ratio_by_blade']]
            candidate_action = np.asarray(row['candidate_action_real'], dtype=np.float64)
            model_input_plan = np.asarray(row['model_input_plan'], dtype=np.float64)
            actual_future_action_plan = np.asarray(row['actual_future_action_plan'], dtype=np.float64)
            identity = paired_identity_by_condition[str(condition_id)]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Shadow audit record has invalid calibration fields: {path}') from exc
        if len(predicted) != 3 or len(actual) != 3:
            raise ValueError('Shadow audit records must contain all three blade peak ratios')
        if not all(math.isfinite(value) for value in (*predicted, *actual, timestamp_s, plan_mismatch)):
            raise ValueError('Shadow audit records contain non-finite values')
        if (
            candidate_action.shape != (2,)
            or model_input_plan.shape != (interface.horizon_frames[interface.horizons_s.index(horizon_s)], 2)
            or actual_future_action_plan.shape != model_input_plan.shape
            or not np.isfinite(candidate_action).all()
            or not np.isfinite(model_input_plan).all()
            or not np.isfinite(actual_future_action_plan).all()
            or not np.allclose(
                model_input_plan,
                make_zoh_candidate_plan(candidate_action, model_input_plan.shape[0]),
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise ValueError('Shadow audit candidate plan does not match its declared ZOH online semantics')
        recomputed_mismatch = normalized_action_plan_mismatch(
            actual_future_action_plan,
            model_input_plan,
            [0.0, 0.0],
            [ENGINEERING_DEMAND_HIGH_NM, 0.43],
        )
        if abs(recomputed_mismatch - plan_mismatch) > 1.0e-6:
            raise ValueError('Shadow audit plan mismatch does not match its actual future action plan')
        row_raw_path = Path(str(row.get('pid_raw_csv_path', ''))).expanduser()
        frozen_raw_path = Path(str(identity.get('pid_raw_csv_path', ''))).expanduser()
        if not row_raw_path.is_absolute():
            row_raw_path = PROJECT_ROOT / row_raw_path
        if not frozen_raw_path.is_absolute():
            frozen_raw_path = PROJECT_ROOT / frozen_raw_path
        if (
            str(row.get('pid_raw_sha256', '')).lower() != str(identity.get('pid_raw_sha256', '')).lower()
            or str(row.get('scenario_id', '')) != str(identity.get('scenario_id', ''))
            or str(row.get('expected_trace_signature', '')).lower()
            != str(identity.get('expected_trace_signature', '')).lower()
            or int(row.get('expected_trace_sample_count', -1))
            != int(identity.get('expected_trace_sample_count', -2))
            or row_raw_path.resolve() != frozen_raw_path.resolve()
        ):
            raise ValueError('Shadow audit PID identity differs from the frozen predictor paired reference')
        condition_ids.add(condition_id)
        actor_sha = str(row.get('actor_policy_sha256', '')).lower()
        if len(actor_sha) != 64 or any(character not in '0123456789abcdef' for character in actor_sha):
            raise ValueError('Strict RL coverage audit requires a frozen Actor checkpoint SHA-256')
        try:
            actor_version = int(row.get('actor_policy_version', -1))
        except (TypeError, ValueError) as exc:
            raise ValueError('Shadow audit Actor policy version is invalid') from exc
        if actor_version != 0:
            raise ValueError('Coverage audit refuses labels from a changing RL Actor policy')
        future_policy_versions = tuple(int(value) for value in row.get('actual_future_actor_policy_versions', ()))
        horizon_frames = interface.horizon_frames[interface.horizons_s.index(horizon_s)]
        if len(future_policy_versions) != horizon_frames or any(
            version != actor_version for version in future_policy_versions
        ):
            raise ValueError('Strict RL coverage audit future window crosses an Actor policy update')
        actor_policy_sha256s.add(actor_sha)
        actor_policy_versions.add(actor_version)
        nonoverlap_group = (condition_id, optimization_state, horizon_s)
        selected_times = selected_times_by_group.setdefault(nonoverlap_group, [])
        insertion_index = bisect_left(selected_times, timestamp_s)
        overlaps_previous = (
            insertion_index > 0
            and timestamp_s - selected_times[insertion_index - 1] < minimum_nonoverlap_spacing_s - 1.0e-9
        )
        overlaps_next = (
            insertion_index < len(selected_times)
            and selected_times[insertion_index] - timestamp_s < minimum_nonoverlap_spacing_s - 1.0e-9
        )
        if overlaps_previous or overlaps_next:
            nonoverlap_dropped += 1
            continue
        selected_times.insert(insertion_index, timestamp_s)
        timestamps.append(timestamp_s)
        eligible_records += 1
        for blade in range(1, 4):
            observations.append(LoadGuardCalibrationObservation(
                blade=blade,
                optimization_state=optimization_state,
                horizon_s=horizon_s,
                plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                predicted_ratio=predicted[blade - 1],
                actual_ratio=actual[blade - 1],
                plan_mismatch=plan_mismatch,
                timestamp_s=timestamp_s,
                condition_id=condition_id,
            ))
    if not observations:
        raise ValueError(f'No eligible, plan-matched ZOH samples were found in {path}')
    return observations, {
        'record_count': seen_records,
        'eligible_record_count': eligible_records,
        'observation_count': len(observations),
        'condition_ids': sorted(condition_ids),
        'min_timestamp_s': min(timestamps),
        'max_timestamp_s': max(timestamps),
        'strict_identity_failure_record_count': strict_identity_failures,
        'overlap_samples_dropped': nonoverlap_dropped,
        'minimum_nonoverlap_spacing_s': minimum_nonoverlap_spacing_s,
        'actor_policy_sha256s': sorted(actor_policy_sha256s),
        'actor_policy_versions': sorted(actor_policy_versions),
    }


def calibrate_from_shadow_files(
    *,
    calibration_jsonl: Path,
    coverage_audit_jsonl: Path,
    interface_path: Path,
    predictor_artifact_path: Path,
    output_path: Path,
    coverage_target: float = 0.90,
    confidence_level: float = 0.90,
    minimum_samples_per_group: int = 100,
    max_plan_mismatch: float = 0.05,
    predictor_training_end_s: float,
    calibration_start_s: float,
    coverage_audit_start_s: float,
    coverage_population: str = 'rl_shadow_strict',
    max_records_per_file: int = 250_000,
) -> dict[str, Any]:
    if coverage_population != 'rl_shadow_strict':
        raise ValueError('This CLI accepts only strict-identity RL shadow audit files; PID holdout needs a separate dataset evaluator')
    if calibration_jsonl.resolve() == coverage_audit_jsonl.resolve():
        raise ValueError('Calibration and coverage-audit inputs must be different files')
    interface = LoadPredictorInterface.from_path(interface_path)
    predictor_sha256 = _sha256(predictor_artifact_path)
    interface_sha256 = _sha256(interface_path)
    predictor_checkpoint = torch.load(predictor_artifact_path, map_location='cpu', weights_only=False)
    if not isinstance(predictor_checkpoint, dict):
        raise ValueError('Predictor checkpoint is not a metadata mapping')
    time_partitions = predictor_checkpoint.get('time_partitions_s')
    if not isinstance(time_partitions, dict):
        raise ValueError('Predictor checkpoint is missing frozen time partitions')
    loaded_predictor = load_candidate_conditioned_predictor(
        predictor_artifact_path,
        interface_path=interface_path,
        expected_checkpoint_sha256=predictor_sha256,
        action_low=[0.0, 0.0],
        action_high=[ENGINEERING_DEMAND_HIGH_NM, 0.43],
        device=torch.device('cpu'),
        expected_dataset_manifest_sha256=str(predictor_checkpoint.get('dataset_manifest_sha256', '')),
        expected_time_partitions_s=time_partitions,
    )
    if loaded_predictor.interface.schema_version != interface.schema_version:
        raise ValueError('Predictor checkpoint and calibration interface schema differs')
    condition_partitions = predictor_checkpoint.get('condition_partitions')
    time_partitions = predictor_checkpoint.get('time_partitions_s')
    paired_identity_by_condition = predictor_checkpoint.get('paired_identity_by_condition')
    if (
        not isinstance(condition_partitions, dict)
        or not isinstance(time_partitions, dict)
        or not isinstance(paired_identity_by_condition, dict)
    ):
        raise ValueError('Predictor checkpoint is missing frozen condition/time partition manifests')
    try:
        training_end_expected_s = float(time_partitions['train'][1])
        calibration_start_expected_s = float(time_partitions['calibration'][0])
        calibration_end_expected_s = float(time_partitions['calibration'][1])
        audit_start_expected_s = float(time_partitions['coverage_audit'][0])
        audit_end_expected_s = float(time_partitions['coverage_audit'][1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError('Predictor checkpoint time partitions are invalid') from exc
    if not math.isclose(predictor_training_end_s, training_end_expected_s, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError('predictor_training_end_s does not match the frozen predictor checkpoint partition')
    if not math.isclose(calibration_start_s, calibration_start_expected_s, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError('calibration_start_s does not match the frozen predictor checkpoint partition')
    if not math.isclose(coverage_audit_start_s, audit_start_expected_s, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError('coverage_audit_start_s does not match the frozen predictor checkpoint partition')
    require_identity = coverage_population == 'rl_shadow_strict'
    minimum_nonoverlap_spacing_s = (
        interface.history_frames - 1 + max(interface.horizon_frames)
    ) * interface.step_duration_s
    calibration_observations, calibration_input_summary = _observations(
        calibration_jsonl,
        interface=interface,
        max_records=max_records_per_file,
        require_strict_identity=require_identity,
        minimum_nonoverlap_spacing_s=minimum_nonoverlap_spacing_s,
        predictor_artifact_sha256=predictor_sha256,
        predictor_interface_sha256=interface_sha256,
        paired_identity_by_condition=paired_identity_by_condition,
    )
    audit_observations, audit_input_summary = _observations(
        coverage_audit_jsonl,
        interface=interface,
        max_records=max_records_per_file,
        require_strict_identity=require_identity,
        minimum_nonoverlap_spacing_s=minimum_nonoverlap_spacing_s,
        predictor_artifact_sha256=predictor_sha256,
        predictor_interface_sha256=interface_sha256,
        paired_identity_by_condition=paired_identity_by_condition,
    )
    if (
        len(calibration_input_summary['actor_policy_sha256s']) != 1
        or calibration_input_summary['actor_policy_sha256s'] != audit_input_summary['actor_policy_sha256s']
        or calibration_input_summary['actor_policy_versions'] != [0]
        or audit_input_summary['actor_policy_versions'] != [0]
    ):
        raise ValueError('Calibration and coverage audit must use the same frozen Actor checkpoint policy')
    calibration_conditions = {int(value) for value in condition_partitions.get('calibration', ())}
    audit_conditions = {int(value) for value in condition_partitions.get('coverage_audit', ())}
    if not set(calibration_input_summary['condition_ids']) <= calibration_conditions:
        raise ValueError('Calibration shadow conditions are outside the frozen calibration condition partition')
    if not set(audit_input_summary['condition_ids']) <= audit_conditions:
        raise ValueError('Coverage-audit shadow conditions are outside the frozen audit condition partition')
    if (
        calibration_input_summary['min_timestamp_s'] < calibration_start_expected_s
        or calibration_input_summary['max_timestamp_s'] >= calibration_end_expected_s
    ):
        raise ValueError('Calibration shadow timestamps are outside the frozen calibration time partition')
    if (
        audit_input_summary['min_timestamp_s'] < audit_start_expected_s
        or audit_input_summary['max_timestamp_s'] >= audit_end_expected_s
    ):
        raise ValueError('Coverage-audit shadow timestamps are outside the frozen audit time partition')
    config = LoadGuardCalibrationConfig(
        coverage_target=coverage_target,
        confidence_level=confidence_level,
        minimum_samples_per_group=minimum_samples_per_group,
        max_plan_mismatch=max_plan_mismatch,
    )
    calibration = calibrate_load_guard(
        calibration_observations,
        config=config,
        predictor_training_end_s=predictor_training_end_s,
        calibration_start_s=calibration_start_s,
        interface=interface,
        predictor_artifact_sha256=predictor_sha256,
        predictor_interface_sha256=interface_sha256,
    )
    audited = audit_load_guard_coverage(
        calibration,
        audit_observations,
        audit_start_s=coverage_audit_start_s,
        interface=interface,
        predictor_artifact_sha256=predictor_sha256,
        predictor_interface_sha256=interface_sha256,
        coverage_population=coverage_population,
        strict_identity_verified=(
            require_identity
            and audit_input_summary['strict_identity_failure_record_count'] == 0
        ),
    )
    audited['metadata'].update({
        'calibration_input_path': str(calibration_jsonl.resolve()),
        'calibration_input_sha256': _sha256(calibration_jsonl),
        'coverage_audit_input_path': str(coverage_audit_jsonl.resolve()),
        'coverage_audit_input_sha256': _sha256(coverage_audit_jsonl),
        'calibration_input_summary': calibration_input_summary,
        'coverage_audit_input_summary': audit_input_summary,
        'interface_path': str(interface_path.resolve()),
        'predictor_condition_partitions': condition_partitions,
        'predictor_time_partitions_s': time_partitions,
        'nonoverlap_sample_spacing_s': minimum_nonoverlap_spacing_s,
        'predictor_artifact_path': str(predictor_artifact_path.resolve()),
        'max_records_per_file': max_records_per_file,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fit one-sided split-conformal margins and audit them on disjoint shadow JSONL evidence.'
    )
    parser.add_argument('--calibration-jsonl', type=Path, required=True)
    parser.add_argument('--coverage-audit-jsonl', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--predictor-artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coverage-target', type=float, default=0.90)
    parser.add_argument('--confidence-level', type=float, default=0.90)
    parser.add_argument('--minimum-samples-per-group', type=int, default=100)
    parser.add_argument('--max-plan-mismatch', type=float, default=0.05)
    parser.add_argument('--predictor-training-end-s', type=float, required=True)
    parser.add_argument('--calibration-start-s', type=float, required=True)
    parser.add_argument('--coverage-audit-start-s', type=float, required=True)
    parser.add_argument('--coverage-population', choices=['rl_shadow_strict'], default='rl_shadow_strict')
    parser.add_argument('--max-records-per-file', type=int, default=250_000)
    args = parser.parse_args()
    report = calibrate_from_shadow_files(
        calibration_jsonl=args.calibration_jsonl,
        coverage_audit_jsonl=args.coverage_audit_jsonl,
        interface_path=args.interface,
        predictor_artifact_path=args.predictor_artifact,
        output_path=args.output,
        coverage_target=args.coverage_target,
        confidence_level=args.confidence_level,
        minimum_samples_per_group=args.minimum_samples_per_group,
        max_plan_mismatch=args.max_plan_mismatch,
        predictor_training_end_s=args.predictor_training_end_s,
        calibration_start_s=args.calibration_start_s,
        coverage_audit_start_s=args.coverage_audit_start_s,
        coverage_population=args.coverage_population,
        max_records_per_file=args.max_records_per_file,
    )
    print(json.dumps({
        'output': str(args.output),
        'coverage_population': report['metadata']['coverage_audit_population'],
        'soft_guard_authorized': report['metadata']['soft_guard_authorized'],
        'calibration_conditions': report['metadata']['calibration_condition_ids'],
        'coverage_audit_conditions': report['metadata']['coverage_audit_condition_ids'],
        'calibration_observations': report['metadata']['observation_count'],
        'coverage_audit_observations': report['metadata']['coverage_audit_observation_count'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
