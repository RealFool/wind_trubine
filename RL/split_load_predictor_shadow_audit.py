from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.load_predictor_contract import LoadPredictorInterface, PLAN_KIND_ZOH_CANDIDATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_shadow_audit(
    *,
    input_jsonl: Path,
    predictor_artifact: Path,
    interface_path: Path,
    calibration_output: Path,
    coverage_audit_output: Path,
    max_records: int = 250_000,
    max_output_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    input_jsonl = input_jsonl.expanduser().resolve()
    predictor_artifact = predictor_artifact.expanduser().resolve()
    interface_path = interface_path.expanduser().resolve()
    calibration_output = calibration_output.expanduser().resolve()
    coverage_audit_output = coverage_audit_output.expanduser().resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f'Shadow audit JSONL does not exist: {input_jsonl}')
    if calibration_output == coverage_audit_output:
        raise ValueError('Calibration and coverage-audit outputs must be different paths')
    if calibration_output.exists() or coverage_audit_output.exists():
        raise FileExistsError('Refusing to overwrite a calibration or coverage-audit JSONL output')
    if max_records <= 0 or max_output_bytes <= 0:
        raise ValueError('max_records and max_output_bytes must be positive')
    interface = LoadPredictorInterface.from_path(interface_path)
    if interface.schema_version != 'causal_load_predictor_interface_v3':
        raise ValueError('Strict RL shadow splitting requires predictor interface v3')
    predictor_sha256 = _sha256(predictor_artifact)
    interface_sha256 = _sha256(interface_path)
    checkpoint = torch.load(predictor_artifact, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Predictor checkpoint is not a mapping')
    predictor_summary_path = predictor_artifact.with_name(f'{predictor_artifact.stem}.summary.json')
    if not predictor_summary_path.is_file():
        raise FileNotFoundError(f'Predictor checkpoint summary does not exist: {predictor_summary_path}')
    predictor_summary = json.loads(predictor_summary_path.read_text(encoding='utf-8'))
    if predictor_summary.get('checkpoint_sha256') != predictor_sha256:
        raise ValueError('Predictor summary checkpoint SHA does not match the frozen artifact')
    load_candidate_conditioned_predictor(
        predictor_artifact,
        interface_path=interface_path,
        expected_checkpoint_sha256=predictor_sha256,
        action_low=[0.0, 0.0],
        action_high=[ENGINEERING_DEMAND_HIGH_NM, 0.43],
        device=torch.device('cpu'),
        expected_dataset_manifest_sha256=str(checkpoint.get('dataset_manifest_sha256', '')),
        expected_time_partitions_s=predictor_summary.get('time_partitions_s'),
    )
    try:
        calibration_conditions = {int(value) for value in checkpoint['condition_partitions']['calibration']}
        audit_conditions = {int(value) for value in checkpoint['condition_partitions']['coverage_audit']}
        calibration_start_s, calibration_end_s = (
            float(value) for value in checkpoint['time_partitions_s']['calibration']
        )
        audit_start_s, audit_end_s = (
            float(value) for value in checkpoint['time_partitions_s']['coverage_audit']
        )
        identity_by_condition = checkpoint['paired_identity_by_condition']
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Predictor checkpoint is missing frozen condition/time/PID identity partitions') from exc
    if calibration_conditions & audit_conditions:
        raise ValueError('Frozen calibration and coverage-audit condition partitions overlap')
    minimum_spacing_s = (
        interface.history_frames - 1 + max(interface.horizon_frames)
    ) * interface.step_duration_s
    last_times_by_output_and_group: dict[tuple[str, int, str, float], list[float]] = {}
    output_counts = {'calibration': 0, 'coverage_audit': 0}
    output_conditions = {'calibration': set(), 'coverage_audit': set()}
    output_times = {'calibration': [], 'coverage_audit': []}
    rejection_counts: dict[str, int] = {}
    input_count = 0
    output_bytes = 0
    actor_policy_sha256: str | None = None
    selected_time_count = 0

    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    coverage_audit_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with calibration_output.open('x', encoding='utf-8') as calibration_handle, coverage_audit_output.open(
            'x', encoding='utf-8'
        ) as audit_handle, input_jsonl.open('r', encoding='utf-8') as input_handle:
            handles = {'calibration': calibration_handle, 'coverage_audit': audit_handle}
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                input_count += 1
                if input_count > max_records:
                    raise ValueError(f'Shadow audit JSONL exceeds max_records={max_records}')
                if len(line.encode('utf-8')) > 64 * 1024:
                    rejection_counts['row_too_large'] = rejection_counts.get('row_too_large', 0) + 1
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid shadow audit JSON at line {line_number}') from exc
                if not isinstance(row, dict) or row.get('schema_version') != 'load_predictor_shadow_audit_record_v1':
                    rejection_counts['schema_mismatch'] = rejection_counts.get('schema_mismatch', 0) + 1
                    continue
                if row.get('predictor_artifact_sha256') != predictor_sha256:
                    rejection_counts['predictor_sha_mismatch'] = rejection_counts.get('predictor_sha_mismatch', 0) + 1
                    continue
                if row.get('predictor_interface_sha256') != interface_sha256:
                    rejection_counts['interface_sha_mismatch'] = rejection_counts.get('interface_sha_mismatch', 0) + 1
                    continue
                if (
                    row.get('plan_kind') != PLAN_KIND_ZOH_CANDIDATE
                    or not bool(row.get('online_authorized', False))
                    or not bool(row.get('prediction_available', False))
                    or not bool(row.get('calibration_eligible', False))
                    or row.get('execution_mode') != 'actor'
                    or row.get('outcome_tag') != 'complete'
                    or row.get('identity_status') != 'strict_case_matched'
                    or not bool(row.get('strict_case_matched', False))
                ):
                    rejection_counts['row_not_strict_eligible'] = rejection_counts.get('row_not_strict_eligible', 0) + 1
                    continue
                try:
                    condition_id = int(row['condition_id'])
                    instance_id = int(row['condition_instance_id'])
                    timestamp_s = float(row['timestamp_s'])
                    horizon_s = float(row['horizon_s'])
                    plan_mismatch = float(row['plan_mismatch'])
                    actor_version = int(row['actor_policy_version'])
                    pid_sha = str(row['pid_raw_sha256']).lower()
                    trace_signature = str(row['expected_trace_signature']).lower()
                    actor_sha = str(row['actor_policy_sha256']).lower()
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f'Shadow audit identity metadata is invalid at line {line_number}') from exc
                if (
                    not bool(row.get('plan_match_complete', False))
                    or not (0.0 <= plan_mismatch <= interface.max_plan_mismatch)
                    or not math.isfinite(timestamp_s)
                    or actor_version != 0
                    or len(actor_sha) != 64
                    or any(character not in '0123456789abcdef' for character in actor_sha)
                    or len(pid_sha) != 64
                    or any(character not in '0123456789abcdef' for character in pid_sha)
                    or len(trace_signature) != 64
                    or any(character not in '0123456789abcdef' for character in trace_signature)
                ):
                    rejection_counts['actor_or_plan_provenance_invalid'] = rejection_counts.get(
                        'actor_or_plan_provenance_invalid',
                        0,
                    ) + 1
                    continue
                if actor_policy_sha256 is None:
                    actor_policy_sha256 = actor_sha
                elif actor_policy_sha256 != actor_sha:
                    raise ValueError('Input audit JSONL mixes Actor checkpoint identities')
                frozen_identity = identity_by_condition.get(str(condition_id))
                if not isinstance(frozen_identity, dict):
                    rejection_counts['condition_outside_predictor_reference'] = rejection_counts.get(
                        'condition_outside_predictor_reference',
                        0,
                    ) + 1
                    continue
                if (
                    pid_sha != str(frozen_identity.get('pid_raw_sha256', '')).lower()
                    or trace_signature != str(frozen_identity.get('expected_trace_signature', '')).lower()
                    or int(frozen_identity.get('expected_trace_sample_count', 0)) != int(
                        row.get('expected_trace_sample_count', -1)
                    )
                ):
                    rejection_counts['pid_reference_identity_mismatch'] = rejection_counts.get(
                        'pid_reference_identity_mismatch',
                        0,
                    ) + 1
                    continue
                observed_raw_path = Path(str(row.get('pid_raw_csv_path', ''))).expanduser()
                if not observed_raw_path.is_absolute():
                    observed_raw_path = PROJECT_ROOT / observed_raw_path
                frozen_raw_path = Path(str(frozen_identity.get('pid_raw_csv_path', ''))).expanduser()
                if not frozen_raw_path.is_absolute():
                    frozen_raw_path = PROJECT_ROOT / frozen_raw_path
                if observed_raw_path.resolve() != frozen_raw_path.resolve():
                    rejection_counts['pid_reference_path_mismatch'] = rejection_counts.get(
                        'pid_reference_path_mismatch',
                        0,
                    ) + 1
                    continue
                if condition_id in calibration_conditions and calibration_start_s <= timestamp_s < calibration_end_s:
                    partition = 'calibration'
                elif condition_id in audit_conditions and audit_start_s <= timestamp_s < audit_end_s:
                    partition = 'coverage_audit'
                else:
                    rejection_counts['outside_frozen_condition_time_partition'] = rejection_counts.get(
                        'outside_frozen_condition_time_partition',
                        0,
                    ) + 1
                    continue
                if horizon_s not in interface.soft_guard_horizons_s:
                    rejection_counts['horizon_not_authorized'] = rejection_counts.get('horizon_not_authorized', 0) + 1
                    continue
                group = (partition, condition_id, str(row['optimization_state']), horizon_s)
                selected_times = last_times_by_output_and_group.setdefault(group, [])
                insertion_index = bisect_left(selected_times, timestamp_s)
                overlaps_previous = (
                    insertion_index > 0
                    and timestamp_s - selected_times[insertion_index - 1] < minimum_spacing_s - 1.0e-9
                )
                overlaps_next = (
                    insertion_index < len(selected_times)
                    and selected_times[insertion_index] - timestamp_s < minimum_spacing_s - 1.0e-9
                )
                if overlaps_previous or overlaps_next:
                    rejection_counts['overlapping_window'] = rejection_counts.get('overlapping_window', 0) + 1
                    continue
                selected_times.insert(insertion_index, timestamp_s)
                encoded = line if line.endswith('\n') else line + '\n'
                encoded_bytes = len(encoded.encode('utf-8'))
                if output_bytes + encoded_bytes > max_output_bytes:
                    raise ValueError(f'Split shadow JSONL exceeds max_output_bytes={max_output_bytes}')
                handles[partition].write(encoded)
                output_bytes += encoded_bytes
                output_counts[partition] += 1
                output_conditions[partition].add(condition_id)
                output_times[partition].append(timestamp_s)
                selected_time_count += 1
        if not output_counts['calibration']:
            raise ValueError('No eligible RL shadow records fall in the frozen calibration partition')
        if not output_counts['coverage_audit']:
            raise ValueError('No eligible RL shadow records fall in the frozen coverage-audit partition')
        if actor_policy_sha256 is None:
            raise ValueError('No eligible frozen-Actor RL shadow records matched the predictor partitions')
    except BaseException:
        calibration_output.unlink(missing_ok=True)
        coverage_audit_output.unlink(missing_ok=True)
        raise

    summary = {
        'schema_version': 'load_predictor_shadow_audit_partition_v1',
        'source_jsonl_path': str(input_jsonl),
        'source_jsonl_sha256': _sha256(input_jsonl),
        'predictor_artifact_path': str(predictor_artifact),
        'predictor_artifact_sha256': predictor_sha256,
        'predictor_interface_path': str(interface_path),
        'predictor_interface_sha256': interface_sha256,
        'actor_policy_sha256': actor_policy_sha256,
        'minimum_nonoverlap_sample_spacing_s': minimum_spacing_s,
        'condition_partitions': {
            'calibration': sorted(calibration_conditions),
            'coverage_audit': sorted(audit_conditions),
        },
        'time_partitions_s': {
            'calibration': [calibration_start_s, calibration_end_s],
            'coverage_audit': [audit_start_s, audit_end_s],
        },
        'output_records': output_counts,
        'output_conditions': {
            name: sorted(values) for name, values in output_conditions.items()
        },
        'output_time_bounds_s': {
            name: [min(values), max(values)] if values else None
            for name, values in output_times.items()
        },
        'output_paths': {
            'calibration': str(calibration_output),
            'coverage_audit': str(coverage_audit_output),
        },
        'rejection_counts': rejection_counts,
        'max_records': max_records,
        'max_output_bytes': max_output_bytes,
        'selected_records': selected_time_count,
            'expected_time_partitions_s': predictor_summary.get('time_partitions_s'),
    }
    summary_path = calibration_output.with_name(f'{calibration_output.stem}.partition_summary.json')
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite shadow partition summary: {summary_path}')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Split strict frozen-Actor shadow JSONL into predictor-bound calibration and audit partitions.')
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--predictor-artifact', type=Path, required=True)
    parser.add_argument('--interface', type=Path, default=PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json')
    parser.add_argument('--calibration-output', type=Path, required=True)
    parser.add_argument('--coverage-audit-output', type=Path, required=True)
    parser.add_argument('--max-records', type=int, default=250_000)
    parser.add_argument('--max-output-bytes', type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    summary = split_shadow_audit(
        input_jsonl=args.input_jsonl,
        predictor_artifact=args.predictor_artifact,
        interface_path=args.interface,
        calibration_output=args.calibration_output,
        coverage_audit_output=args.coverage_audit_output,
        max_records=args.max_records,
        max_output_bytes=args.max_output_bytes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
