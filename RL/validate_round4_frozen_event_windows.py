"""Validate bounded full-state event windows from a frozen round-4 inference.

The event JSONL is streamed one bounded record at a time. This audit checks
capture structure and identity only; it cannot qualify policy performance or
causal effects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from RL.utils.constants import BLADED_UDP_VARIABLES
from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import FROZEN_R4_ROUTE, FROZEN_R4_SCHEMA

MAX_EVENT_RECORD_BYTES = 512 * 1024
EXPECTED_CONDITIONS = (8, 11, 12, 16)
EXPECTED_EVENT_AUTHORIZATION = {
    'physical_shadow': {
        'mode': 'shadow_development_only',
        'policy_acceptance': False,
        'soft_authorized': False,
        'hard_authorized': False,
        'counterfactual_validation': False,
        'rl_margin_coverage_authorized': False,
    },
    'policy_acceptance': False,
    'counterfactual_validation': False,
}


def _sha(path: Path) -> str:
    return file_sha256(path)


def _strict_line(data: bytes, *, path: Path, row_number: int) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate key in event window {path}:{row_number}: {key}')
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f'Non-finite JSON constant in event window {path}:{row_number}: {value}')

    value = json.loads(data, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f'Event-window JSONL row must be an object: {path}:{row_number}')
    return value


def _finite(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(numeric):
        raise ValueError(f'{name} must be a finite number')
    return numeric


def _read_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f'Frozen inference metadata exceeds 4 MiB: {path}')
    return strict_json(path, maximum_bytes=4 * 1024 * 1024)


def validate_event_windows(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / 'effective_config.json'
    summary_path = run_dir / 'run_summary.json'
    checkpoint_meta_path = run_dir / 'checkpoint_metadata.json'
    windows_path = run_dir / 'compact_queue_condition_event_windows.jsonl'
    compact_summary_path = run_dir / 'compact_metrics_summary.json'
    for path in (config_path, summary_path, checkpoint_meta_path, windows_path, compact_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f'Frozen round-4 inference output is missing: {path}')
    config = _read_object(config_path)
    run_summary = _read_object(summary_path)
    checkpoint_meta = _read_object(checkpoint_meta_path)
    compact_summary = _read_object(compact_summary_path)
    gate = config.get('physical_shadow_launch', {})
    if gate.get('route') != FROZEN_R4_ROUTE or gate.get('authorized') is not True:
        raise ValueError('Inference output does not declare the narrow frozen-round-4 development route')
    receipt_path = Path(str(gate.get('evidence_path', ''))).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = run_dir.parents[1] / receipt_path
    if not receipt_path.is_file() or _sha(receipt_path) != gate.get('evidence_sha256'):
        raise ValueError('Frozen round-4 inference receipt is missing or SHA-mismatched')
    receipt = _read_object(receipt_path)
    if (receipt.get('schema_version') != FROZEN_R4_SCHEMA or receipt.get('route') != FROZEN_R4_ROUTE or
            receipt.get('run_id') != config.get('experiment_name') or
            receipt.get('runtime_identity', {}).get('execution_entry') != 'inference' or
            receipt.get('runtime_identity', {}).get('baseline_only') is not False or
            receipt.get('policy_acceptance') is not False or receipt.get('optimizer_updates') is not False or
            receipt.get('independently_verified') is not False or receipt.get('actor_frozen') is not True):
        raise ValueError('Frozen round-4 receipt scope is invalid')
    if receipt.get('event_window_validation') != {
            'full_existing_state_enabled': True,
            'pre_frames': int(config.get('logging', {}).get('queue_event_pre_frames', -1)),
            'post_frames': int(config.get('logging', {}).get('queue_event_post_frames', -1)),
            'max_record_bytes': int(config.get('logging', {}).get('queue_event_window_max_record_bytes', -1)),
            'max_total_bytes': int(config.get('logging', {}).get('queue_event_windows_max_total_bytes', -1)),
    }:
        raise ValueError('Frozen round-4 receipt event-window budgets differ from effective config')
    source = receipt.get('source_run', {})
    checkpoint = Path(str(source.get('round_4_checkpoint_path', ''))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = run_dir.parents[1] / checkpoint
    checkpoint_sha = str(source.get('round_4_checkpoint_sha256', '')).lower()
    if not checkpoint.is_file() or _sha(checkpoint) != checkpoint_sha:
        raise ValueError('Source round-4 checkpoint is missing or SHA-mismatched')
    if (checkpoint_meta.get('checkpoint_path') != str(checkpoint.resolve()) or
            checkpoint_meta.get('checkpoint_loaded_at_utc') is None or
            receipt.get('runtime_identity', {}).get('actor_checkpoint_sha256') != checkpoint_sha):
        raise ValueError('Inference checkpoint metadata is not bound to the source round-4 checkpoint')
    if (config.get('physical_shadow_run_role', {}).get('actor_frozen') is not True or
            config.get('physical_shadow_run_role', {}).get('optimizer_updates') is not False or
            config.get('train', {}).get('online_rl_enabled') is not False or
            config.get('logging', {}).get('queue_event_full_state_enabled') is not True or
            config.get('policy_acceptance') is not False):
        raise ValueError('Effective inference config is not frozen, full-state shadow-only scope')
    if (run_summary.get('status') != 'completed' or run_summary.get('queue_mode') is not True or
            run_summary.get('episodes_requested') != 4 or run_summary.get('episodes_completed') != 4):
        raise ValueError('Frozen round-4 inference did not complete the requested four queue slots')
    episodes = run_summary.get('episode_summaries')
    if not isinstance(episodes, list) or len(episodes) != 4:
        raise ValueError('Frozen round-4 inference is missing four episode slot summaries')
    observed_conditions = [int(row.get('queue_condition_id', -1)) for row in episodes]
    if observed_conditions != list(EXPECTED_CONDITIONS):
        raise ValueError(f'Frozen round-4 inference slot order mismatch: {observed_conditions}')
    expected_slot_by_condition: dict[int, tuple[int, int, int]] = {}
    for episode in episodes:
        condition_id = int(episode.get('queue_condition_id', -1))
        instance_id = int(episode.get('queue_condition_instance_id', -1))
        queue_epoch = int(episode.get('queue_epoch', -1))
        if condition_id in expected_slot_by_condition or instance_id <= 0 or queue_epoch <= 0:
            raise ValueError('Frozen round-4 episode slot identity is invalid or duplicated')
        expected_slot_by_condition[condition_id] = (queue_epoch, instance_id, condition_id)
    source_mismatches = [
        int(row.get('case_identity_status') != 'strict_case_matched') for row in episodes
    ]

    physical_summary = run_summary.get('physical_load_shadow')
    if (not isinstance(physical_summary, dict) or physical_summary.get('finalized') is not True or
            physical_summary.get('faulted') is not False or
            physical_summary.get('authorization', {}).get('policy_acceptance') is not False):
        raise ValueError('Frozen event-window run does not have a finalized healthy physical shadow worker')
    physical_counters = physical_summary.get('counters', {})
    physical_receipts = physical_summary.get('condition_receipts')
    if (not isinstance(physical_receipts, list) or len(physical_receipts) != 4 or
            physical_counters.get('finished_conditions') != 4 or
            physical_counters.get('finish_slot_identity_mismatch', 0) != 0 or
            physical_counters.get('duplicate_slot_finish', 0) != 0):
        raise ValueError('Physical shadow worker finish receipts do not close over four queue slots')
    for episode, physical_receipt in zip(episodes, physical_receipts, strict=True):
        expected_identity = (
            int(episode.get('queue_epoch', -1)),
            int(episode.get('queue_condition_instance_id', -1)),
            int(episode.get('queue_condition_id', -1)),
        )
        observed_identity = (
            int(physical_receipt.get('queue_epoch', -1)),
            int(physical_receipt.get('condition_instance_id', -1)),
            int(physical_receipt.get('condition_id', -1)),
        )
        if expected_identity != observed_identity or physical_receipt.get('finish_slot_identity_match') is not True:
            raise ValueError('Physical shadow receipt does not match the inference episode queue slot')
    if compact_summary.get('queue_event_full_state_enabled') is not True:
        raise ValueError('Compact logger summary does not confirm full existing-state event logging')
    capture = run_summary.get('queue_event_capture')
    if not isinstance(capture, dict) or capture.get('error') or capture.get('event_max_record_bytes_observed', 0) > MAX_EVENT_RECORD_BYTES:
        raise ValueError('Event logger did not finalize within its single-record byte budget')
    max_total_bytes = int(config['logging']['queue_event_windows_max_total_bytes'])
    if windows_path.stat().st_size > max_total_bytes:
        raise ValueError('Event-window file exceeds its configured total-byte budget')

    counters: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    slot_counts: Counter[tuple[int, int, int]] = Counter()
    slot_unique_frame_ids: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    window_digest = hashlib.sha256()
    max_record_bytes = 0
    actor_sha_values: set[str] = set()
    trigger_score_max: dict[str, float] = {}
    with windows_path.open('rb') as handle:
        line_number = 0
        while True:
            data = handle.readline(MAX_EVENT_RECORD_BYTES + 1)
            if not data:
                break
            line_number += 1
            if len(data) > MAX_EVENT_RECORD_BYTES:
                raise ValueError(f'Event window exceeds 512 KiB at record {line_number}')
            window_digest.update(data)
            max_record_bytes = max(max_record_bytes, len(data.rstrip(b'\r\n')))
            row = _strict_line(data, path=windows_path, row_number=line_number)
            if row.get('schema_version') != 'queue_condition_event_window_v2':
                raise ValueError(f'Full-state event window must use v2 schema at record {line_number}')
            if int(row.get('round_index', -1)) != 1:
                raise ValueError(f'Unexpected round index in frozen inference event window {line_number}')
            condition_id = int(row.get('condition_id', -1))
            instance_id = int(row.get('condition_instance_id', -1))
            if condition_id not in EXPECTED_CONDITIONS or instance_id <= 0:
                raise ValueError(f'Invalid event-window queue identity at record {line_number}')
            expected_slot = expected_slot_by_condition.get(condition_id)
            if expected_slot is None or expected_slot[1] != instance_id:
                raise ValueError(f'Event-window identity is outside the observed episode slots: {condition_id}/{instance_id}')
            slot = expected_slot
            frames = row.get('frames')
            if not isinstance(frames, list) or not frames or len(frames) > 76:
                raise ValueError(f'Invalid event-window frame count at record {line_number}')
            target_pre = int(row.get('target_pre_frame_count', -1))
            target_post = int(row.get('target_post_frame_count', -1))
            if (target_pre, target_post) != (50, 25):
                raise ValueError(f'Event-window pre/post frame policy mismatch at record {line_number}')
            trigger_step = int(row.get('trigger_step', -1))
            trigger_frames = [frame for frame in frames if int(frame.get('step', -2)) == trigger_step]
            if len(trigger_frames) != 1:
                raise ValueError(f'Event-window trigger frame is missing or duplicated at record {line_number}')
            pre_count = int(row.get('pre_frame_count', -1))
            post_count = int(row.get('post_frame_count', -1))
            window_complete = row.get('window_complete') is True
            if (pre_count != len(frames) - post_count - 1 or post_count < 0 or
                    post_count > target_post or window_complete != (post_count >= target_post)):
                raise ValueError(f'Event-window frame counts/completion flag do not close at record {line_number}')
            for frame in frames:
                for field, expected in (
                    ('queue_condition_round_index', 1),
                    ('queue_condition_id', condition_id),
                    ('queue_condition_instance_id', instance_id),
                    ('queue_epoch', expected_slot[0]),
                ):
                    if int(frame.get(field, -1)) != expected:
                        raise ValueError(f'Event frame slot identity mismatch at record {line_number}: {field}')
                origin_payload = frame.get('origin_payload')
                observed_payload = frame.get('observed_payload')
                if (not isinstance(origin_payload, dict) or not isinstance(observed_payload, dict) or
                        set(origin_payload) != set(BLADED_UDP_VARIABLES) or
                        set(observed_payload) != set(BLADED_UDP_VARIABLES)):
                    raise ValueError(f'Event frame does not contain the full existing UDP payload at record {line_number}')
                invalid_origin = frame.get('origin_invalid_fields')
                invalid_observed = frame.get('observed_invalid_fields')
                if not isinstance(invalid_origin, list) or not isinstance(invalid_observed, list):
                    raise ValueError(f'Event frame is missing raw-invalidity metadata at record {line_number}')
                wire = frame.get('wire_action_real')
                if not isinstance(wire, list) or len(wire) != 2:
                    raise ValueError(f'Event frame is missing exact sent-wire action at record {line_number}')
                wire_values = [_finite(value, name='wire_action_real') for value in wire]
                slot_unique_frame_ids[slot].add(int(frame.get('udp_frame_id', -1)))
                if frame.get('actor_policy_version') != 0 or frame.get('actor_policy_sha256') != checkpoint_sha:
                    raise ValueError(f'Event frame actor SHA/version mismatch at record {line_number}')
                actor_sha_values.add(frame['actor_policy_sha256'])
                current_time = _finite(frame.get('udp_sim_time'), name='udp_sim_time')
                next_time = _finite(frame.get('udp_next_sim_time'), name='udp_next_sim_time')
                payload_time = _finite(origin_payload.get('Time'), name='origin_payload.Time')
                next_payload_time = _finite(observed_payload.get('Time'), name='observed_payload.Time')
                if not math.isclose(current_time, payload_time, rel_tol=0.0, abs_tol=1.0e-6):
                    raise ValueError(f'Current payload/time does not align at record {line_number}')
                if next_time > current_time and not math.isclose(next_time, next_payload_time,
                                                                rel_tol=0.0, abs_tol=1.0e-6):
                    raise ValueError(f'Observed payload/next time does not align at record {line_number}')
                event_sample_valid = bool(frame.get('observed_sample_valid', False))
                if event_sample_valid and (invalid_origin or invalid_observed):
                    raise ValueError(f'Invalid raw frame was marked sample-valid at record {line_number}')
                if int(frame['step']) == trigger_step and 'first_pid_peak_exceed' in str(row.get('event_type', '')):
                    if not event_sample_valid:
                        raise ValueError(f'PID threshold trigger frame is marked invalid at record {line_number}')
                if event_sample_valid:
                    counters['valid_measurement_frame_occurrences'] += 1
                else:
                    counters['invalid_or_nontrain_measurement_frame_occurrences'] += 1
                if not invalid_origin and not invalid_observed:
                    for field in BLADED_UDP_VARIABLES:
                        _finite(origin_payload[field], name=f'origin_payload.{field}')
                        _finite(observed_payload[field], name=f'observed_payload.{field}')
                if 'control_sent_action_torque' in frame and 'control_sent_action_pitch' in frame:
                    control_action = [
                        _finite(frame['control_sent_action_torque'], name='control sent torque'),
                        _finite(frame['control_sent_action_pitch'], name='control sent pitch'),
                    ]
                    if any(abs(left - right) > 6.0e-7 for left, right in zip(wire_values, control_action, strict=True)):
                        raise ValueError(f'Wire demand disagrees with logged executed action at record {line_number}')
            event_type = str(row.get('event_type', ''))
            event_types[event_type] += 1
            slot_counts[slot] += 1
            counters['event_windows'] += 1
            counters['complete_windows'] += int(row.get('window_complete') is True)
            counters['incomplete_windows'] += int(row.get('window_complete') is not True)
            counters['windows_without_full_pre_context'] += int(int(row.get('pre_frame_count', 0)) < 50)
            counters['windows_without_full_post_context'] += int(int(row.get('post_frame_count', 0)) < 25)
            if event_type.startswith('physical_shadow_'):
                trigger_frame = trigger_frames[0]
                if trigger_frame.get('physical_load_shadow_prediction_origin_match') != 1.0:
                    raise ValueError(f'Predicted event trigger lacks same-step shadow origin at record {line_number}')
                counters['shadow_prediction_trigger_windows'] += 1
            if 'first_pid_peak_exceed' in event_type:
                blade = int(event_type.split('_')[1])
                trigger_frame = trigger_frames[0]
                metric_name = f'reward_blade_root_{blade}_peak_reference_nm'
                reference = _finite(trigger_frame.get(metric_name), name=metric_name)
                measured = _finite(trigger_frame['observed_payload'][BLADED_BLADE_FIELDS[blade - 1]],
                                   name='trigger-frame measured My')
                if not abs(measured) > reference:
                    raise ValueError(f'First-exceed trigger does not exceed the recorded PID peak at record {line_number}')
                counters['pid_threshold_trigger_windows'] += 1
            score = float(row.get('score', 0.0))
            trigger_score_max[event_type] = max(trigger_score_max.get(event_type, -math.inf), score)
    if actor_sha_values != {checkpoint_sha}:
        raise ValueError('Event windows mix policy checkpoint SHA identities')
    if line_number != int(capture.get('event_windows_written', -1)):
        raise ValueError('Event-window logger written-record count does not close')
    if windows_path.stat().st_size != int(capture.get('event_bytes_written', -1)):
        raise ValueError('Event-window logger written-byte count does not close')
    if max_record_bytes != int(capture.get('event_max_record_bytes_observed', -1)):
        raise ValueError('Event-window logger max-record-byte count does not close')
    if compact_summary.get('queue_event_windows_written') != line_number:
        raise ValueError('Compact logger summary and event-window file row counts differ')
    if compact_summary.get('queue_event_bytes_written') != windows_path.stat().st_size:
        raise ValueError('Compact logger summary and event-window file byte counts differ')
    configured_capacity = int(config['logging']['queue_event_window_capacity'])
    if any(count > configured_capacity for count in slot_counts.values()):
        raise ValueError('A queue slot exceeds the configured event-window capacity')

    report = {
        'schema_version': 'round4_frozen_actor_event_window_validation_v1',
        'run_id': run_dir.name,
        'evidence_role': 'frozen_policy_event_window_capture_development_only',
        'source_round_4_checkpoint_sha256': checkpoint_sha,
        'event_window_file': str(windows_path),
        'event_window_file_sha256': window_digest.hexdigest(),
        'event_window_file_bytes': windows_path.stat().st_size,
        'event_window_max_record_bytes_observed': max_record_bytes,
        'configured_event_window_record_budget_bytes': MAX_EVENT_RECORD_BYTES,
        'configured_total_byte_budget': max_total_bytes,
        'run_status': run_summary.get('status'),
        'slot_order': observed_conditions,
        'slot_identity_statuses': [row.get('case_identity_status') for row in episodes],
        'operator_unverified_slot_count': sum(source_mismatches),
        'event_types': dict(event_types),
        'event_windows_dropped': int(compact_summary.get('queue_event_windows_dropped', 0)),
        'slot_window_counts': {f'round1_wind{key[2]}_instance{key[1]}_epoch{key[0]}': count
                               for key, count in slot_counts.items()},
        'slot_unique_event_frame_count': {
            f'round1_wind{key[2]}_instance{key[1]}_epoch{key[0]}': len(frame_ids)
            for key, frame_ids in slot_unique_frame_ids.items()
        },
        'event_frame_occurrences_in_windows': counters['valid_measurement_frame_occurrences'] +
            counters['invalid_or_nontrain_measurement_frame_occurrences'],
        'unique_event_frame_count': sum(len(frame_ids) for frame_ids in slot_unique_frame_ids.values()),
        'counters': dict(counters),
        'capture_integrity_pass': True,
        'capture_complete_without_drops': bool(
            int(compact_summary.get('queue_event_windows_dropped', 0)) == 0
            and counters['incomplete_windows'] == 0
            and counters['windows_without_full_pre_context'] == 0
            and counters['windows_without_full_post_context'] == 0
        ),
        'strict_pid_identity_pass': all(status == 'strict_case_matched' for status in
                                         (row.get('case_identity_status') for row in episodes)),
        'independent_audit': False,
        'policy_acceptance': False,
        'counterfactual_validation': False,
        'causal_effect_estimated': False,
        'authorization': physical_summary.get('authorization'),
        'limitations': [
            'Event-window capture integrity does not establish same-case PID wind-trace identity; operator warning slots remain development_unverified.',
            'Frozen round-4 Actor event windows are one-policy observations, not randomized action-effect evidence.',
            'Incomplete near-end windows and event-ring drops are counted and must not be relabeled complete.',
        ],
    }
    if output_path is None:
        output_path = run_dir / 'physical_evaluation' / 'round4_event_window_validation.json'
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite event-window validation report: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    report['report_path'] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    report = validate_event_windows(args.run_dir, output_path=args.output)
    print(json.dumps({
        'report_path': report['report_path'],
        'capture_integrity_pass': report['capture_integrity_pass'],
        'event_windows': report['counters']['event_windows'],
        'strict_pid_identity_pass': report['strict_pid_identity_pass'],
        'policy_acceptance': False,
    }, ensure_ascii=False))


BLADED_BLADE_FIELDS = (
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)


if __name__ == '__main__':
    main()
