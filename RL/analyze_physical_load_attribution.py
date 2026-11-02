"""Stream one completed physical-shadow run into action/threshold attribution.

This is retrospective descriptive analysis, not a causal effect estimate.
The large JSONL and PID source CSVs are processed one bounded row at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from RL.utils.paired_pid_reference import BLADE_ROOT_MY_FIELDS, CanonicalTraceHasher
from RL.utils.physical_load_shadow_artifact import (
    BACKEND,
    LABEL_SEMANTICS,
    RUNTIME_AUTHORIZATION,
    file_sha256,
    strict_json,
)
from RL.utils.queue_manifest import expand_queue_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD_SCHEMA = 'physical_load_shadow_development_record_v1'
EVENT_WINDOW_SCHEMA_V1 = 'queue_condition_event_window_v1'
EVENT_WINDOW_SCHEMA_V2 = 'queue_condition_event_window_v2'
DT = 0.02
MAX_SHADOW_BYTES = 1_073_741_824
MAX_SHADOW_RECORD_BYTES = 32 * 1024
MAX_SHADOW_RECORDS = 250_000
MAX_EVENT_WINDOW_BYTES = 262_144
MAX_EVENT_WINDOWS = 20_000
MAX_SLOTS = 1024
ACTION_LAGS_S = (0.02, 0.10, 0.20, 0.50, 1.00)
ENGINEERING_ACTION_HIGH = (21_586_451.33303, 0.43)
_BLADE_EVENT_RE = re.compile(r'^blade_([123])_(first_pid_peak_exceed|new_max_abs_my)$')


def _strict_json(path: Path, maximum_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f'Attribution metadata exceeds {maximum_bytes} bytes: {path}')

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key in attribution metadata: {key}')
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f'Non-finite JSON constant in attribution metadata: {value}')

    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f'Attribution metadata must be a JSON object: {path}')
    return value


def _json_line(encoded: bytes, *, source: Path, row_number: int) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key at {source}:{row_number}: {key}')
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f'Non-finite JSON constant at {source}:{row_number}: {value}')

    result = json.loads(encoded, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f'JSONL record must be an object at {source}:{row_number}')
    return result


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite number')
    return result


def _round_time(value: float) -> float:
    return round(float(value), 6)


def _load_run_contract(run_dir: Path) -> tuple[dict[str, Any], list[Any], dict[tuple[int, int, int], dict[str, Any]]]:
    shadow_summary = _strict_json(run_dir / 'physical_load_shadow_summary.json')
    compact_summary = _strict_json(run_dir / 'compact_metrics_summary.json')
    manifest = _strict_json(run_dir.parents[1] / 'configs' / run_dir.name / 'queue_manifest.json')
    slots = expand_queue_manifest(manifest)
    if not slots or len(slots) > MAX_SLOTS:
        raise ValueError('Physical attribution requires a non-empty bounded queue manifest')
    expected = [(int(slot.queue_epoch), int(slot.condition_instance_id), int(slot.condition_id))
                for slot in slots]
    receipts = shadow_summary.get('condition_receipts')
    if not isinstance(receipts, list) or len(receipts) != len(slots):
        raise ValueError('Physical shadow finish receipts do not match the manifest slot count')
    receipt_by_slot: dict[tuple[int, int, int], dict[str, Any]] = {}
    for receipt in receipts:
        key = (int(receipt.get('queue_epoch', -1)), int(receipt.get('condition_instance_id', -1)),
               int(receipt.get('condition_id', -1)))
        if key in receipt_by_slot:
            raise ValueError(f'Duplicate physical shadow receipt for slot {key}')
        receipt_by_slot[key] = receipt
    if list(receipt_by_slot) != expected:
        raise ValueError('Physical shadow receipt order/identity does not match queue manifest')
    if shadow_summary.get('backend') != BACKEND or shadow_summary.get('finalized') is not True:
        raise ValueError('Attribution requires a finalized physical_fir_v2 worker summary')
    if shadow_summary.get('faulted') is not False:
        raise ValueError('Attribution refuses a faulted physical shadow worker')
    if compact_summary.get('mode') != 'compact_streaming':
        raise ValueError('Attribution requires compact streaming event-window metadata')
    if len({slot.condition_id for slot in slots}) != 4:
        raise ValueError('This attribution pass is scoped to the existing four-wind smoke manifest')
    return {'physical': shadow_summary, 'compact': compact_summary, 'manifest': manifest}, slots, receipt_by_slot


def _load_pid_thresholds(
    run_dir: Path,
    slots: list[Any],
    receipt_by_slot: Mapping[tuple[int, int, int], dict[str, Any]],
    *,
    project_root: Path,
) -> tuple[dict[int, tuple[float, float, float]], dict[int, dict[str, Any]]]:
    identity_files = sorted(run_dir.glob('queue_case_identity_condition_*_instance_*.json'))
    if len(identity_files) != len(slots):
        raise ValueError(f'Expected {len(slots)} per-slot identity receipts, found {len(identity_files)}')
    slot_by_instance = {(slot.condition_id, slot.condition_instance_id): slot for slot in slots}
    source_by_wind: dict[int, dict[str, Any]] = {}
    identity_by_slot: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in identity_files:
        identity = _strict_json(path)
        condition_id = int(identity['condition_id'])
        instance_id = int(identity['condition_instance_id'])
        slot = slot_by_instance.get((condition_id, instance_id))
        if slot is None:
            raise ValueError(f'Identity file is outside the queue manifest: {path.name}')
        key = (int(slot.queue_epoch), instance_id, condition_id)
        if key in identity_by_slot:
            raise ValueError(f'Duplicate identity summary for slot {key}')
        if identity.get('queue_epoch') is not None and int(identity['queue_epoch']) != key[0]:
            raise ValueError(f'Identity summary epoch mismatch for slot {key}')
        receipt = receipt_by_slot[key]
        for field in ('identity_status', 'pid_raw_sha256', 'scenario_id', 'time_window_start_s',
                      'time_window_end_s', 'expected_trace_signature', 'expected_trace_sample_count'):
            if identity.get(field) != receipt.get(field):
                raise ValueError(f'Physical and queue identity receipt mismatch for {field}, slot {key}')
        raw_path = Path(str(identity['pid_raw_csv_path'])).expanduser()
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        raw_path = raw_path.resolve()
        source = {
            'path': raw_path,
            'sha256': str(identity['pid_raw_sha256']).lower(),
            'trace_signature': str(identity['expected_trace_signature']).lower(),
            'sample_count': int(identity['expected_trace_sample_count']),
            'time_window_start_s': float(identity['time_window_start_s']),
            'time_window_end_s': float(identity['time_window_end_s']),
            'identity_status': str(identity.get('identity_status', 'missing')),
        }
        previous = source_by_wind.get(condition_id)
        if previous is not None and any(previous[name] != source[name] for name in source if name != 'path'):
            raise ValueError(f'PID reference identity differs across repetitions of wind{condition_id}')
        if previous is not None and previous['path'] != raw_path:
            raise ValueError(f'PID raw source path differs across repetitions of wind{condition_id}')
        source_by_wind[condition_id] = source
        identity_by_slot[key] = identity
    if len(identity_by_slot) != len(slots):
        raise ValueError('Per-slot identity summaries do not close against the queue manifest')

    thresholds: dict[int, tuple[float, float, float]] = {}
    for condition_id, source in source_by_wind.items():
        raw_path = source['path']
        if not raw_path.is_file() or file_sha256(raw_path) != source['sha256']:
            raise ValueError(f'PID raw file is missing or SHA-mismatched for wind{condition_id}')
        peaks = [0.0, 0.0, 0.0]
        trace = CanonicalTraceHasher()
        row_count = 0
        with raw_path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'Time', 'I_AverageWindSpeed', *BLADE_ROOT_MY_FIELDS}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError(f'PID CSV is missing time/wind/My columns: {raw_path}')
            for row_count, row in enumerate(reader, start=1):
                timestamp = _finite(row['Time'], name=f'{raw_path.name}:Time')
                wind = _finite(row['I_AverageWindSpeed'], name=f'{raw_path.name}:wind')
                expected_time = (row_count - 1) * DT + source['time_window_start_s']
                if not math.isclose(timestamp, expected_time, rel_tol=0.0, abs_tol=1.0e-6):
                    raise ValueError(f'PID source time grid mismatch at {raw_path.name}:{row_count + 1}')
                trace.update(timestamp, wind)
                for blade, field in enumerate(BLADE_ROOT_MY_FIELDS):
                    peaks[blade] = max(peaks[blade], abs(_finite(row[field], name=f'{raw_path.name}:{field}')))
        if row_count != source['sample_count']:
            raise ValueError(f'PID source row count mismatch for wind{condition_id}: {row_count}')
        expected_end = source['time_window_end_s']
        if not math.isclose((row_count - 1) * DT + source['time_window_start_s'],
                            expected_end, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f'PID source does not end at its frozen window endpoint for wind{condition_id}')
        if trace.hexdigest() != source['trace_signature']:
            raise ValueError(f'PID source canonical time/wind trace signature mismatch for wind{condition_id}')
        if any(not math.isfinite(value) or value <= 0.0 for value in peaks):
            raise ValueError(f'Invalid PID blade peak threshold for wind{condition_id}')
        thresholds[condition_id] = tuple(peaks)
        source['computed_peak_abs_nm_by_blade'] = list(peaks)
    return thresholds, source_by_wind


def _read_event_window_triggers(
    path: Path,
    *,
    expected_count: int,
    maximum_total_bytes: int,
) -> tuple[dict[tuple[int, int, int, int, float], dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'Compact queue event windows are missing: {path}')
    if path.stat().st_size > maximum_total_bytes:
        raise ValueError('Compact event-window JSONL exceeds the configured total-byte budget')
    triggers: dict[tuple[int, int, int, int, float], dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    digest = hashlib.sha256()
    max_record_bytes = 0
    with path.open('rb') as handle:
        row_number = 0
        while True:
            encoded = handle.readline(MAX_EVENT_WINDOW_BYTES + 1)
            if not encoded:
                break
            row_number += 1
            if len(encoded) > MAX_EVENT_WINDOW_BYTES:
                raise ValueError(f'Event-window record exceeds 256 KiB at {path}:{row_number}')
            digest.update(encoded)
            max_record_bytes = max(max_record_bytes, len(encoded.rstrip(b'\r\n')))
            record = _json_line(encoded, source=path, row_number=row_number)
            if record.get('schema_version') not in (EVENT_WINDOW_SCHEMA_V1, EVENT_WINDOW_SCHEMA_V2):
                raise ValueError(f'Unsupported event-window schema at {path}:{row_number}')
            frames = record.get('frames')
            if not isinstance(frames, list) or len(frames) > 128:
                raise ValueError(f'Event-window frame count is invalid at {path}:{row_number}')
            event_type = str(record.get('event_type', ''))
            match = _BLADE_EVENT_RE.match(event_type)
            if not match:
                counters['non_blade_trigger_windows'] += 1
                continue
            blade = int(match.group(1))
            event_kind = match.group(2)
            try:
                trigger_step = int(record['trigger_step'])
                trigger_frame = next(frame for frame in frames if int(frame['step']) == trigger_step)
                trigger_time = _round_time(float(trigger_frame.get('udp_next_sim_time', trigger_frame['udp_sim_time'])))
                key = (int(record['round_index']), int(record['condition_id']),
                       int(record['condition_instance_id']), blade, trigger_time)
            except (KeyError, StopIteration, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Invalid blade event-window trigger at {path}:{row_number}') from exc
            sample = {
                'event_type': event_type,
                'event_kind': event_kind,
                'score': float(record.get('score', 0.0)),
                'trigger_step': trigger_step,
                'trigger_time_s': trigger_time,
                'action_origin_time_s': trigger_frame.get('udp_sim_time'),
                'measurement_time_s': trigger_frame.get('udp_next_sim_time'),
                'window_complete': bool(record.get('window_complete', False)),
                'frames_in_window': len(frames),
                'trigger_frame': trigger_frame,
            }
            # Retain only one compact trigger snapshot per event-time key; never keep whole windows.
            previous = triggers.get(key)
            if previous is None or sample['score'] > previous['score']:
                triggers[key] = sample
            counters[f'{event_kind}_windows'] += 1
    if row_number != expected_count:
        raise ValueError(f'Compact event-window row count mismatch: {row_number} != {expected_count}')
    return triggers, {
        'path': str(path),
        'sha256': digest.hexdigest(),
        'bytes': path.stat().st_size,
        'records': row_number,
        'max_record_bytes_observed': max_record_bytes,
        'blade_trigger_keys_retained': len(triggers),
        'counters': dict(counters),
    }


class _ActionLagBuffer:
    def __init__(self, capacity: int = 51) -> None:
        self.capacity = capacity
        self.times: deque[float] = deque()
        self.values: dict[float, dict[str, Any]] = {}

    def clear(self) -> None:
        self.times.clear()
        self.values.clear()

    def append(self, sample: dict[str, Any]) -> None:
        timestamp = _round_time(float(sample['action_time_s']))
        sample['action_time_s'] = timestamp
        if timestamp in self.values:
            raise ValueError(f'Duplicate interval action timestamp in attribution stream: {timestamp}')
        self.times.append(timestamp)
        self.values[timestamp] = sample
        while len(self.times) > self.capacity:
            self.values.pop(self.times.popleft(), None)

    def lags(self, measurement_time_s: float) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for lag in ACTION_LAGS_S:
            suffix = f'{int(round(lag * 1000)):04d}ms'
            timestamp = _round_time(measurement_time_s - lag)
            sample = self.values.get(timestamp)
            for variable, field in (('torque_nm', 'torque_nm'), ('pitch_rad', 'pitch_rad'),
                                    ('actor_policy_version', 'actor_policy_version'),
                                    ('actor_policy_sha256', 'actor_policy_sha256')):
                result[f'action_lag_{suffix}_{variable}'] = None if sample is None else sample[field]
        return result


class _ExcursionAccumulator:
    def __init__(self, *, merge_gap_s: float = DT) -> None:
        self.merge_gap_s = merge_gap_s
        self.current: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def add(self, key: tuple[int, int, int, int], sample: dict[str, Any], writer: csv.DictWriter) -> None:
        timestamp = float(sample['measurement_time_s'])
        magnitude = float(sample['abs_my_nm'])
        active = self.current.get(key)
        if active is not None and timestamp - active['last_exceed_time_s'] > self.merge_gap_s + 1.0e-6:
            self._write(active, writer)
            del self.current[key]
            active = None
        if magnitude <= float(sample['pid_peak_abs_nm']):
            return
        if active is None:
            active = {
                'round_index': int(sample['round_index']),
                'slot': key[:3],
                'blade': key[3],
                'first_cross_time_s': timestamp,
                'last_exceed_time_s': timestamp,
                'sample_count': 0,
                'peak_abs_my_nm': -math.inf,
                'peak_sample': None,
                'first_sample': dict(sample),
            }
            self.current[key] = active
        active['last_exceed_time_s'] = timestamp
        active['sample_count'] += 1
        if magnitude > active['peak_abs_my_nm']:
            active['peak_abs_my_nm'] = magnitude
            active['peak_sample'] = dict(sample)

    def finish_slot(self, slot: tuple[int, int, int], writer: csv.DictWriter) -> None:
        for key in [key for key in self.current if key[:3] == slot]:
            self._write(self.current.pop(key), writer)

    def finish_all(self, writer: csv.DictWriter) -> None:
        for key in list(self.current):
            self._write(self.current.pop(key), writer)

    @staticmethod
    def _write(event: dict[str, Any], writer: csv.DictWriter) -> None:
        first = event['first_sample']
        peak = event['peak_sample']
        row = {
            'round_index': event['round_index'],
            'queue_epoch': event['slot'][0], 'condition_instance_id': event['slot'][1],
            'condition_id': event['slot'][2], 'blade': event['blade'],
            'pid_peak_abs_nm': first['pid_peak_abs_nm'],
            'first_cross_time_s': event['first_cross_time_s'],
            'last_exceed_time_s': event['last_exceed_time_s'],
            'event_duration_s': event['last_exceed_time_s'] - event['first_cross_time_s'] + DT,
            'exceedance_sample_count': event['sample_count'],
            'first_cross_signed_my_nm': first['signed_my_nm'],
            'first_cross_abs_ratio_to_pid': first['abs_my_nm'] / first['pid_peak_abs_nm'],
            'first_cross_origin_time_s': first['origin_time_s'],
            'first_cross_origin_frame_id': first['origin_frame_id'],
            'first_cross_future_sample_index': first['future_sample_index'],
            'first_cross_control_region': first['origin_control_region'],
            'first_cross_direct_action_time_s': first['direct_action_time_s'],
            'first_cross_direct_torque_nm': first['direct_torque_nm'],
            'first_cross_direct_pitch_rad': first['direct_pitch_rad'],
            'first_cross_actor_policy_version': first['actor_policy_version'],
            'first_cross_actor_policy_sha256': first['actor_policy_sha256'],
            'first_cross_approximate_zoh': first['approximate_zoh'],
            'first_cross_prediction_supported_h100': first['prediction_supported'],
            'first_cross_predicted_peak_nm': first['predicted_peak_nm'],
            'first_cross_predicted_exceed': first['predicted_exceed'],
            'first_cross_event_window_match': first['event_window_match'],
            'first_cross_event_window_context_json': json.dumps(
                first['event_window_context'], ensure_ascii=False, separators=(',', ':'), allow_nan=False),
            'peak_time_s': peak['measurement_time_s'],
            'peak_signed_my_nm': peak['signed_my_nm'],
            'peak_abs_my_nm': peak['abs_my_nm'],
            'peak_ratio_to_pid': peak['abs_my_nm'] / peak['pid_peak_abs_nm'],
            'peak_origin_time_s': peak['origin_time_s'],
            'peak_origin_frame_id': peak['origin_frame_id'],
            'peak_direct_action_time_s': peak['direct_action_time_s'],
            'peak_direct_torque_nm': peak['direct_torque_nm'],
            'peak_direct_pitch_rad': peak['direct_pitch_rad'],
            'peak_actor_policy_version': peak['actor_policy_version'],
            'peak_actor_policy_sha256': peak['actor_policy_sha256'],
            'peak_control_region': peak['origin_control_region'],
            'peak_event_window_match': peak['event_window_match'],
            'peak_event_window_context_json': json.dumps(
                peak['event_window_context'], ensure_ascii=False, separators=(',', ':'), allow_nan=False),
        }
        row.update({f'first_cross_{name}': value for name, value in first['lag_actions'].items()})
        row.update({f'peak_{name}': value for name, value in peak['lag_actions'].items()})
        writer.writerow(row)


def _context_for_event(
    trigger_index: Mapping[tuple[int, int, int, int, float], dict[str, Any]],
    *,
    round_index: int,
    condition_id: int,
    instance_id: int,
    blade: int,
    measurement_time_s: float,
) -> dict[str, Any] | None:
    sample = trigger_index.get((round_index, condition_id, instance_id, blade, _round_time(measurement_time_s)))
    if sample is None:
        return None
    frame = sample['trigger_frame']
    origin_payload = frame.get('origin_payload')
    observed_payload = frame.get('observed_payload')
    state_names = (
        'I_RotorAzimuth', 'I_RotorSpeed', 'I_YawError', 'I_ElectricalTorque', 'I_GeneratorSpeed',
        'I_RealPower', 'I_PitchPosition1', 'I_NacelleForeAftAcceleration',
        'I_NacelleSideSideAcceleration', 'I_AverageWindSpeed', 'Electrical power',
        *BLADE_ROOT_MY_FIELDS,
    )
    payload = observed_payload if isinstance(observed_payload, dict) else None
    raw_state_available = bool(payload is not None and all(name in payload for name in state_names))
    return {
        'event_type': sample['event_type'],
        'window_complete': sample['window_complete'],
        'measurement_time_s': sample['measurement_time_s'],
        'action_origin_time_s': sample['action_origin_time_s'],
        'score': sample['score'],
        'raw_state_available': raw_state_available,
        'raw_fields_missing': [] if raw_state_available else [
            name for name in state_names if payload is None or name not in payload
        ],
        'origin_payload': origin_payload,
        'observed_payload': observed_payload,
        'observed_invalid_fields': frame.get('observed_invalid_fields', []),
        'wire_action_real': frame.get('wire_action_real'),
        'partial_event_metrics': {
            name: frame.get(name) for name in (
                'wind_t', 'wind_next', 'power', 'action_torque', 'action_pitch',
                'action_torque_delta', 'action_pitch_delta', 'control_region_code',
                'control_optimization_state_code', 'control_baseline_action_torque',
                'control_baseline_action_pitch', 'control_policy_delta_torque',
                'control_policy_delta_pitch', 'reward_acc_raw',
                'reward_shaft_raw', 'reward_blade_root_1_raw', 'reward_blade_root_2_raw',
                'reward_blade_root_3_raw',
            ) if name in frame
        },
    }


def _event_csv_fields() -> list[str]:
    fields = [
        'round_index', 'queue_epoch', 'condition_instance_id', 'condition_id', 'blade', 'pid_peak_abs_nm',
        'first_cross_time_s', 'last_exceed_time_s', 'event_duration_s', 'exceedance_sample_count',
        'first_cross_signed_my_nm', 'first_cross_abs_ratio_to_pid', 'first_cross_origin_time_s',
        'first_cross_origin_frame_id', 'first_cross_future_sample_index', 'first_cross_control_region',
        'first_cross_direct_action_time_s', 'first_cross_direct_torque_nm', 'first_cross_direct_pitch_rad',
        'first_cross_actor_policy_version', 'first_cross_actor_policy_sha256', 'first_cross_approximate_zoh',
        'first_cross_prediction_supported_h100', 'first_cross_predicted_peak_nm',
        'first_cross_predicted_exceed', 'first_cross_event_window_match',
        'first_cross_event_window_context_json', 'peak_time_s', 'peak_signed_my_nm', 'peak_abs_my_nm',
        'peak_ratio_to_pid', 'peak_origin_time_s', 'peak_origin_frame_id', 'peak_direct_action_time_s',
        'peak_direct_torque_nm', 'peak_direct_pitch_rad', 'peak_actor_policy_version',
        'peak_actor_policy_sha256', 'peak_control_region', 'peak_event_window_match',
        'peak_event_window_context_json',
    ]
    for prefix in ('first_cross', 'peak'):
        fields.extend(f'{prefix}_action_lag_{int(round(lag * 1000)):04d}ms_{name}'
                      for lag in ACTION_LAGS_S
                      for name in ('torque_nm', 'pitch_rad', 'actor_policy_version', 'actor_policy_sha256'))
    return fields


def _write_event_window_trigger_audit(
    path: Path,
    triggers: Mapping[tuple[int, int, int, int, float], dict[str, Any]],
) -> None:
    fields = [
        'round_index', 'condition_id', 'condition_instance_id', 'blade', 'event_type', 'event_kind',
        'trigger_time_s', 'action_origin_time_s', 'score', 'window_complete', 'frames_in_window',
        'raw_state_available', 'raw_fields_missing', 'origin_payload_json', 'observed_payload_json',
        'observed_invalid_fields_json', 'wire_action_real_json', 'partial_event_metrics_json',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, sample in sorted(triggers.items()):
            context = _context_for_event(
                triggers, round_index=key[0], condition_id=key[1], instance_id=key[2],
                blade=key[3], measurement_time_s=key[4],
            )
            assert context is not None
            writer.writerow({
                'round_index': key[0], 'condition_id': key[1], 'condition_instance_id': key[2],
                'blade': key[3], 'event_type': sample['event_type'], 'event_kind': sample['event_kind'],
                'trigger_time_s': sample['trigger_time_s'], 'action_origin_time_s': sample['action_origin_time_s'],
                'score': sample['score'], 'window_complete': sample['window_complete'],
                'frames_in_window': sample['frames_in_window'], 'raw_state_available': context['raw_state_available'],
                'raw_fields_missing': ';'.join(context['raw_fields_missing']),
                'origin_payload_json': json.dumps(context['origin_payload'], ensure_ascii=False, separators=(',', ':')),
                'observed_payload_json': json.dumps(context['observed_payload'], ensure_ascii=False, separators=(',', ':')),
                'observed_invalid_fields_json': json.dumps(context['observed_invalid_fields'], ensure_ascii=False),
                'wire_action_real_json': json.dumps(context['wire_action_real'], ensure_ascii=False),
                'partial_event_metrics_json': json.dumps(context['partial_event_metrics'], ensure_ascii=False,
                                                         separators=(',', ':')),
            })


def analyze_attribution(
    run_dir: Path,
    output_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    max_total_bytes: int = MAX_SHADOW_BYTES,
    max_record_bytes: int = MAX_SHADOW_RECORD_BYTES,
    max_records: int = MAX_SHADOW_RECORDS,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    shadow_path = run_dir / 'physical_load_shadow_development.jsonl'
    event_path = run_dir / 'compact_queue_condition_event_windows.jsonl'
    if not shadow_path.is_file() or shadow_path.stat().st_size > max_total_bytes:
        raise ValueError('Physical shadow JSONL is missing or exceeds the configured streaming budget')
    metadata, slots, receipt_by_slot = _load_run_contract(run_dir)
    shadow_summary = metadata['physical']
    compact_summary = metadata['compact']
    worker_counters = shadow_summary.get('counters', {})
    expected_records = int(worker_counters.get('records_written', -1))
    expected_bytes = int(worker_counters.get('bytes_written', -1))
    expected_sha = str(shadow_summary.get('records_file_sha256', '')).lower()
    if expected_records <= 0 or expected_bytes != shadow_path.stat().st_size or len(expected_sha) != 64:
        raise ValueError('Physical worker summary record/byte/SHA identity is invalid')
    thresholds, pid_sources = _load_pid_thresholds(run_dir, slots, receipt_by_slot, project_root=project_root)
    event_budget = int(compact_summary.get('queue_event_windows_max_total_bytes', MAX_EVENT_WINDOW_BYTES))
    event_count = int(compact_summary.get('queue_event_windows_written', -1))
    event_triggers, event_metadata = _read_event_window_triggers(
        event_path, expected_count=event_count, maximum_total_bytes=event_budget,
    )

    slot_order = [(int(slot.queue_epoch), int(slot.condition_instance_id), int(slot.condition_id))
                  for slot in slots]
    slot_round = {key: int(slot.round_id) for key, slot in zip(slot_order, slots, strict=True)}
    slot_stats: dict[tuple[int, int, int], dict[str, Any]] = {
        key: {
            'round_index': slot_round[key], 'queue_epoch': key[0], 'condition_instance_id': key[1],
            'condition_id': key[2], 'identity_status': receipt_by_slot[key].get('identity_status'),
            'wind_mismatch_frames': int(receipt_by_slot[key].get('queue_case_identity_wind_mismatch_count', 0)),
            'wind_max_abs_error_mps': float(receipt_by_slot[key].get('queue_case_identity_wind_max_abs_error_mps', 0.0)),
            'trace_signature_match': bool(receipt_by_slot[key].get('trace_signature_match', 0.0)),
            'horizon_record_count': Counter(), 'supported_count': Counter(), 'unknown_count': Counter(),
            'zoh_count': Counter(), 'actor_frozen_count': Counter(),
            'blade': {blade: {'h100_label_samples': 0, 'h100_exceedance_samples': 0,
                              'first_exceedance_time_s': None, 'maximum_abs_my_nm': -math.inf,
                              'maximum_time_s': None, 'maximum_ratio_to_pid': None,
                              'maximum_direct_torque_nm': None, 'maximum_direct_pitch_rad': None,
                              'threshold_events': 0}
                      for blade in (1, 2, 3)},
        }
        for key in slot_order
    }
    confusion: dict[tuple[float, int], Counter[str]] = {
        (horizon, blade): Counter() for horizon in (0.1, 0.2) for blade in (1, 2, 3)
    }
    aggregate: Counter[str] = Counter()
    shadow_digest = hashlib.sha256()
    action_history = _ActionLagBuffer()
    excursion = _ExcursionAccumulator(merge_gap_s=DT)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite physical attribution output: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.tmp_', dir=output_dir.parent))
    event_csv_path = temporary / 'threshold_exceedance_events.csv'
    slot_csv_path = temporary / 'slot_blade_peaks.csv'
    confusion_csv_path = temporary / 'shadow_window_confusion.csv'
    event_audit_path = temporary / 'event_window_trigger_audit.csv'
    event_fields = _event_csv_fields()
    input_records = 0
    horizon_seen: Counter[float] = Counter()
    last_slot: tuple[int, int, int] | None = None
    last_sample_time: float | None = None
    previous_origin_by_slot: dict[tuple[int, int, int], float] = {}
    continuity_gaps = 0
    event_window_context_matches = 0

    try:
        with event_csv_path.open('w', encoding='utf-8', newline='') as event_handle:
            event_writer = csv.DictWriter(event_handle, fieldnames=event_fields)
            event_writer.writeheader()
            with shadow_path.open('rb') as handle:
                row_number = 0
                while True:
                    encoded = handle.readline(max_record_bytes + 1)
                    if not encoded:
                        break
                    row_number += 1
                    shadow_digest.update(encoded)
                    if len(encoded) > max_record_bytes:
                        raise ValueError(f'Physical shadow row exceeds {max_record_bytes} bytes at line {row_number}')
                    if not encoded.strip():
                        continue
                    input_records += 1
                    if input_records > max_records:
                        raise ValueError('Physical shadow input exceeds the configured row budget')
                    row = _json_line(encoded, source=shadow_path, row_number=row_number)
                    if (row.get('schema_version') != RECORD_SCHEMA or row.get('backend') != BACKEND or
                            row.get('label_semantics') != LABEL_SEMANTICS or
                            row.get('authorization') != RUNTIME_AUTHORIZATION or
                            row.get('rl_margin_coverage_authorized') is not False or
                            row.get('calibration_eligible') is not False or
                            row.get('independent_audit_eligible') is not False):
                        raise ValueError(f'Unsupported or over-authorized physical row at line {row_number}')
                    key = (int(row['queue_epoch']), int(row['condition_instance_id']), int(row['condition_id']))
                    if key not in slot_stats:
                        raise ValueError(f'Physical record is outside the frozen manifest at line {row_number}: {key}')
                    if last_slot is not None and key != last_slot and key != slot_order[slot_order.index(last_slot) + 1]:
                        raise ValueError(f'Physical record slot order changed at line {row_number}')
                    if row.get('identity_status') != receipt_by_slot[key].get('identity_status'):
                        raise ValueError(f'Physical record identity label differs from slot receipt at line {row_number}')
                    if row.get('evaluation_reference_file_sha256') != shadow_summary.get(
                            'evaluation_reference_file_sha256'):
                        raise ValueError(f'Physical record evaluation reference SHA mismatch at line {row_number}')
                    condition_id = key[2]
                    pid_peak = tuple(_finite(value, name='pid_peak_abs_nm')
                                     for value in row['pid_peak_abs_nm'])
                    if len(pid_peak) != 3 or any(abs(a - b) > 1.0e-6 for a, b in zip(
                            pid_peak, thresholds[condition_id], strict=True)):
                        raise ValueError(f'Embedded PID thresholds differ from streamed PID raw source at line {row_number}')
                    horizon = _finite(row['horizon_s'], name='horizon_s')
                    frames = 5 if horizon == 0.1 else 10 if horizon == 0.2 else 0
                    if frames == 0 or int(row['horizon_frames']) != frames:
                        raise ValueError(f'Unsupported physical shadow horizon at line {row_number}')
                    origin_time = _finite(row['origin_time_s'], name='origin_time_s')
                    endpoint_time = _finite(row['label_endpoint_time_s'], name='label_endpoint_time_s')
                    if not math.isclose(endpoint_time - origin_time, horizon, rel_tol=0.0, abs_tol=1.0e-6):
                        raise ValueError(f'Physical future-label endpoint mismatch at line {row_number}')
                    plan = row.get('executed_interval_actions')
                    future = row.get('actual_future_signed_my_nm_by_blade')
                    versions = row.get('executed_interval_actor_policy_versions')
                    shas = row.get('executed_interval_actor_policy_sha256s')
                    if not all(isinstance(value, list) and len(value) == frames for value in
                               (plan, future, versions, shas)):
                        raise ValueError(f'Physical future plan/action/Actor arrays do not close at line {row_number}')
                    sent_plan = [[_finite(value, name='executed_interval_action') for value in action]
                                 for action in plan]
                    if any(len(action) != 2 or action[0] < 0.0 or action[0] > ENGINEERING_ACTION_HIGH[0]
                           or action[1] < 0.0 or action[1] > ENGINEERING_ACTION_HIGH[1] for action in sent_plan):
                        raise ValueError(f'Executed sent action is outside existing engineering limits at line {row_number}')
                    signed_my = [[_finite(value, name='future_signed_my') for value in sample]
                                 for sample in future]
                    if any(len(sample) != 3 for sample in signed_my):
                        raise ValueError(f'Future signed My must contain three blades at line {row_number}')
                    origin_action = [_finite(value, name='origin_sent_action') for value in row['origin_sent_action']]
                    if len(origin_action) != 2:
                        raise ValueError(f'Origin sent action is malformed at line {row_number}')
                    mismatch = max(
                        2.0 * abs(action[k] - origin_action[k]) / ENGINEERING_ACTION_HIGH[k]
                        for action in sent_plan for k in range(2)
                    )
                    if not math.isclose(mismatch, _finite(row['plan_mismatch_normalized_linf'], name='plan mismatch'),
                                        rel_tol=0.0, abs_tol=1.0e-9):
                        raise ValueError(f'Approximate-ZOH action mismatch does not close at line {row_number}')
                    approximate_zoh = mismatch <= 0.05
                    if row.get('approximate_zoh') is not approximate_zoh:
                        raise ValueError(f'Approximate-ZOH flag does not close at line {row_number}')
                    version_values = [int(value) for value in versions]
                    sha_values = [str(value).lower() for value in shas]
                    actor_frozen = all(value == int(row['actor_policy_version']) for value in version_values)
                    actor_frozen &= all(value == str(row['actor_policy_sha256']).lower() for value in sha_values)
                    if row.get('actor_policy_window_frozen') is not bool(actor_frozen):
                        raise ValueError(f'Actor-frozen flag does not close at line {row_number}')
                    actual_peak = tuple(max(abs(sample[blade]) for sample in signed_my) for blade in range(3))
                    embedded_actual = tuple(_finite(value, name='actual_future_peak_nm_by_blade')
                                            for value in row['actual_future_peak_nm_by_blade'])
                    if any(abs(a - b) > 1.0e-6 for a, b in zip(actual_peak, embedded_actual, strict=True)):
                        raise ValueError(f'Future peak label does not close at line {row_number}')
                    prediction_supported = row.get('prediction_supported') is True
                    predicted = row.get('predicted_peak_nm_by_blade')
                    if prediction_supported:
                        if not isinstance(predicted, list) or len(predicted) != 3:
                            raise ValueError(f'Supported prediction is missing three blade peaks at line {row_number}')
                        predicted = tuple(_finite(value, name='predicted_peak_nm_by_blade') for value in predicted)
                    elif predicted is not None:
                        raise ValueError(f'Unsupported prediction must be null at line {row_number}')
                    slot = slot_stats[key]
                    slot['horizon_record_count'][horizon] += 1
                    horizon_seen[horizon] += 1
                    slot['supported_count'][horizon] += int(prediction_supported)
                    slot['unknown_count'][horizon] += int(not prediction_supported)
                    slot['zoh_count'][horizon] += int(approximate_zoh)
                    slot['actor_frozen_count'][horizon] += int(actor_frozen)
                    aggregate[f'physical_records_h{int(round(horizon * 100)):03d}'] += 1
                    aggregate[f'prediction_supported_h{int(round(horizon * 100)):03d}'] += int(prediction_supported)
                    aggregate[f'prediction_unknown_h{int(round(horizon * 100)):03d}'] += int(not prediction_supported)
                    aggregate[f'approximate_zoh_h{int(round(horizon * 100)):03d}'] += int(approximate_zoh)
                    aggregate[f'actor_frozen_h{int(round(horizon * 100)):03d}'] += int(actor_frozen)
                    for blade in (1, 2, 3):
                        actual_exceed = actual_peak[blade - 1] > pid_peak[blade - 1]
                        if prediction_supported:
                            predicted_exceed = predicted[blade - 1] > pid_peak[blade - 1]
                            confusion[(horizon, blade)][
                                'TP' if predicted_exceed and actual_exceed else
                                'FP' if predicted_exceed else 'FN' if actual_exceed else 'TN'
                            ] += 1
                        else:
                            confusion[(horizon, blade)]['unknown'] += 1
                    if horizon == 0.1:
                        old_origin = previous_origin_by_slot.get(key)
                        if old_origin is not None and origin_time <= old_origin + 1.0e-6:
                            raise ValueError(f'Duplicate/out-of-order H100 origin at line {row_number}')
                        previous_origin_by_slot[key] = origin_time
                        round_index = slot_round[key]
                        for future_index in range(frames):
                            response_time = _round_time(origin_time + (future_index + 1) * DT)
                            action_time = _round_time(origin_time + future_index * DT)
                            action = sent_plan[future_index]
                            action_sample = {
                                'action_time_s': action_time,
                                'torque_nm': action[0], 'pitch_rad': action[1],
                                'actor_policy_version': version_values[future_index],
                                'actor_policy_sha256': sha_values[future_index],
                                'origin_time_s': origin_time,
                            }
                            # H100 origins should tile the 20ms response stream. A real discontinuity
                            # terminates any open event and prevents lag actions crossing the gap.
                            if last_slot != key:
                                if last_slot is not None:
                                    excursion.finish_slot(last_slot, event_writer)
                                action_history.clear()
                                last_sample_time = None
                                last_slot = key
                            if last_sample_time is not None and not math.isclose(
                                    response_time - last_sample_time, DT, rel_tol=0.0, abs_tol=1.0e-6):
                                continuity_gaps += 1
                                aggregate['h100_label_continuity_gaps'] += 1
                                excursion.finish_slot(key, event_writer)
                                action_history.clear()
                            action_history.append(action_sample)
                            last_sample_time = response_time
                            for blade in (1, 2, 3):
                                signed = signed_my[future_index][blade - 1]
                                magnitude = abs(signed)
                                reference = pid_peak[blade - 1]
                                blade_stats = slot['blade'][blade]
                                blade_stats['h100_label_samples'] += 1
                                blade_stats['h100_exceedance_samples'] += int(magnitude > reference)
                                if magnitude > reference and blade_stats['first_exceedance_time_s'] is None:
                                    blade_stats['first_exceedance_time_s'] = response_time
                                if magnitude > blade_stats['maximum_abs_my_nm']:
                                    blade_stats['maximum_abs_my_nm'] = magnitude
                                    blade_stats['maximum_time_s'] = response_time
                                    blade_stats['maximum_ratio_to_pid'] = magnitude / reference
                                    blade_stats['maximum_direct_torque_nm'] = action[0]
                                    blade_stats['maximum_direct_pitch_rad'] = action[1]
                                context = _context_for_event(
                                    event_triggers,
                                    round_index=round_index,
                                    condition_id=condition_id,
                                    instance_id=key[1],
                                    blade=blade,
                                    measurement_time_s=response_time,
                                )
                                event_window_context_matches += int(context is not None)
                                if row.get('prediction_supported') is True:
                                    predicted_value = float(predicted[blade - 1])
                                    predicted_exceed = predicted_value > reference
                                else:
                                    predicted_value = None
                                    predicted_exceed = None
                                sample = {
                                    'measurement_time_s': response_time,
                                    'round_index': round_index,
                                    'signed_my_nm': signed,
                                    'abs_my_nm': magnitude,
                                    'pid_peak_abs_nm': reference,
                                    'origin_time_s': origin_time,
                                    'origin_frame_id': int(row['origin_frame_id']),
                                    'future_sample_index': future_index,
                                    'origin_control_region': row.get('origin_control_region', 'unknown'),
                                    'direct_action_time_s': action_time,
                                    'direct_torque_nm': action[0],
                                    'direct_pitch_rad': action[1],
                                    'actor_policy_version': version_values[future_index],
                                    'actor_policy_sha256': sha_values[future_index],
                                    'approximate_zoh': approximate_zoh,
                                    'prediction_supported': prediction_supported,
                                    'predicted_peak_nm': predicted_value,
                                    'predicted_exceed': predicted_exceed,
                                    'event_window_match': context is not None,
                                    'event_window_context': context or {},
                                    'lag_actions': action_history.lags(response_time),
                                }
                                excursion.add(key + (blade,), sample, event_writer)
                    if last_slot is None:
                        last_slot = key
                    elif key != last_slot and slot_order.index(key) > slot_order.index(last_slot):
                        excursion.finish_slot(last_slot, event_writer)
                        action_history.clear()
                        last_sample_time = None
                        last_slot = key
        excursion.finish_all(event_writer)
        if input_records != expected_records:
            raise ValueError(f'Physical shadow record count mismatch: {input_records} != {expected_records}')
        if shadow_path.stat().st_size != expected_bytes or shadow_digest.hexdigest() != expected_sha:
            raise ValueError('Physical shadow input bytes/SHA do not match the finalized worker summary')
        if sum(horizon_seen.values()) != expected_records or set(horizon_seen) != {0.1, 0.2}:
            raise ValueError(f'Physical shadow horizon record counts do not close: {dict(horizon_seen)}')
        event_count_by_slot_blade = Counter()
        with event_csv_path.open('r', encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                event_count_by_slot_blade[(int(row['queue_epoch']), int(row['condition_instance_id']),
                                           int(row['condition_id']), int(row['blade']))] += 1
        for key, slot in slot_stats.items():
            for blade in (1, 2, 3):
                slot['blade'][blade]['threshold_events'] = event_count_by_slot_blade[key + (blade,)]
                if slot['blade'][blade]['maximum_abs_my_nm'] == -math.inf:
                    slot['blade'][blade]['maximum_abs_my_nm'] = None
        with slot_csv_path.open('w', encoding='utf-8', newline='') as handle:
            fields = [
                'round_index', 'queue_epoch', 'condition_instance_id', 'condition_id', 'blade',
                'identity_status', 'wind_mismatch_frames', 'wind_max_abs_error_mps', 'trace_signature_match',
                'pid_peak_abs_nm', 'h100_label_samples', 'h100_exceedance_samples', 'threshold_events',
                'first_exceedance_time_s', 'maximum_time_s', 'maximum_abs_my_nm', 'maximum_ratio_to_pid',
                'maximum_direct_torque_nm', 'maximum_direct_pitch_rad',
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key in slot_order:
                slot = slot_stats[key]
                for blade in (1, 2, 3):
                    values = slot['blade'][blade]
                    writer.writerow({
                        'round_index': slot['round_index'], 'queue_epoch': key[0],
                        'condition_instance_id': key[1], 'condition_id': key[2], 'blade': blade,
                        'identity_status': slot['identity_status'],
                        'wind_mismatch_frames': slot['wind_mismatch_frames'],
                        'wind_max_abs_error_mps': slot['wind_max_abs_error_mps'],
                        'trace_signature_match': slot['trace_signature_match'],
                        'pid_peak_abs_nm': thresholds[key[2]][blade - 1],
                        **values,
                    })
        with confusion_csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'horizon_s', 'blade', 'origins', 'TP', 'FP', 'FN', 'TN', 'unknown',
                'precision', 'recall', 'qualification',
            ])
            writer.writeheader()
            for horizon in (0.1, 0.2):
                for blade in (1, 2, 3):
                    values = confusion[(horizon, blade)]
                    tp, fp, fn, tn = (values[name] for name in ('TP', 'FP', 'FN', 'TN'))
                    writer.writerow({
                        'horizon_s': horizon, 'blade': blade, 'origins': tp + fp + fn + tn + values['unknown'],
                        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'unknown': values['unknown'],
                        'precision': None if tp + fp == 0 else tp / (tp + fp),
                        'recall': None if tp + fn == 0 else tp / (tp + fn),
                        'qualification': 'overlapping/correlated development origins; descriptive only',
                    })
        _write_event_window_trigger_audit(event_audit_path, event_triggers)
        report = {
            'schema_version': 'physical_load_action_attribution_report_v1',
            'evidence_role': 'retrospective_time_alignment_and_association_only_not_causality',
            'run_id': run_dir.name,
            'source_paths': {
                'physical_shadow_jsonl': str(shadow_path),
                'physical_shadow_jsonl_sha256': expected_sha,
                'compact_event_windows_jsonl': event_metadata['path'],
                'compact_event_windows_jsonl_sha256': event_metadata['sha256'],
                'physical_shadow_summary': str(run_dir / 'physical_load_shadow_summary.json'),
                'queue_manifest': str(run_dir.parents[1] / 'configs' / run_dir.name / 'queue_manifest.json'),
            },
            'streaming_limits': {
                'physical_shadow_max_total_bytes': max_total_bytes,
                'physical_shadow_max_record_bytes': max_record_bytes,
                'physical_shadow_max_records': max_records,
                'event_window_max_record_bytes': MAX_EVENT_WINDOW_BYTES,
                'event_window_count': event_metadata['records'],
                'event_window_max_record_bytes_observed': event_metadata['max_record_bytes_observed'],
                'retained_only_compact_event_triggers': True,
                'full_shadow_or_raw_pid_loaded_into_memory': False,
            },
            'inputs': {
                'physical_shadow_records': input_records,
                'horizon_records': {str(key): value for key, value in horizon_seen.items()},
                'physical_shadow_sha256_verified': True,
                'slots': len(slots),
                'condition_ids': sorted(thresholds),
                'pid_sources': {str(cid): {
                    'path': str(source['path']), 'sha256': source['sha256'],
                    'trace_signature': source['trace_signature'], 'sample_count': source['sample_count'],
                    'pid_peak_abs_nm_by_blade': source['computed_peak_abs_nm_by_blade'],
                    'identity_status': source['identity_status'],
                } for cid, source in pid_sources.items()},
            },
            'summary_counters': {
                **dict(aggregate),
                'h100_label_continuity_gaps': continuity_gaps,
                'event_window_trigger_rows': len(event_triggers),
                'h100_samples_matched_to_existing_event_windows': event_window_context_matches,
                'h100_threshold_excursion_events': sum(event_count_by_slot_blade.values()),
            },
            'event_grouping': {
                'definition': 'strictly abs(My)>same-wind, same-0..600s PID peak; consecutive 20ms exceedance samples merge; a larger gap starts a new event',
                'merge_gap_s': DT,
                'action_alignment': 'future My at t maps to successfully sent interval action at t-0.02s; lag actions are descriptive associations',
                'action_lags_s': list(ACTION_LAGS_S),
            },
            'model_window_confusion_csv': str(output_dir / 'shadow_window_confusion.csv'),
            'threshold_event_csv': str(output_dir / 'threshold_exceedance_events.csv'),
            'slot_blade_peak_csv': str(output_dir / 'slot_blade_peaks.csv'),
            'event_window_trigger_audit_csv': str(output_dir / 'event_window_trigger_audit.csv'),
            'event_window_trigger_audit': event_metadata,
            'slot_blade_summaries': [
                {**{key_name: slot[key_name] for key_name in (
                    'round_index', 'queue_epoch', 'condition_instance_id', 'condition_id',
                    'identity_status', 'wind_mismatch_frames', 'wind_max_abs_error_mps', 'trace_signature_match')},
                 'horizon_record_count': {str(h): n for h, n in slot['horizon_record_count'].items()},
                 'supported_count': {str(h): n for h, n in slot['supported_count'].items()},
                 'unknown_count': {str(h): n for h, n in slot['unknown_count'].items()},
                 'approximate_zoh_count': {str(h): n for h, n in slot['zoh_count'].items()},
                 'actor_frozen_count': {str(h): n for h, n in slot['actor_frozen_count'].items()},
                 'blade': {str(blade): values for blade, values in slot['blade'].items()}}
                for key, slot in slot_stats.items()
            ],
            'limitations': [
                'The completed physical shadow JSONL stores future signed My and successfully executed interval actions, but not future per-frame state or wind; origin state/wind are not available for every retrospective label.',
                'The old compact event windows contain only a partial state/control snapshot; raw UDP state/wind payloads are absent. They are joined by measurement time only where an existing blade threshold/maximum event was captured.',
                'All run slots are development_unverified because the live time-wind traces do not match the PID traces. PID peaks are nominal descriptive comparison thresholds, not strict paired acceptance limits.',
                'Actor policy SHA is empty in the training shadow rows; policy version may vary across rounds. Event proximity and lagged action alignment do not establish causation.',
                'Origin strides and overlapping H200 windows induce correlated samples; no per-frame IID confidence interval or treatment effect is reported.',
            ],
            'authorization': dict(RUNTIME_AUTHORIZATION),
            'causal_effect_estimated': False,
            'policy_acceptance': False,
        }
        with (temporary / 'report.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        temporary.rename(output_dir)
        return report
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--max-total-bytes', type=int, default=MAX_SHADOW_BYTES)
    parser.add_argument('--max-record-bytes', type=int, default=MAX_SHADOW_RECORD_BYTES)
    parser.add_argument('--max-records', type=int, default=MAX_SHADOW_RECORDS)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir or (run_dir / 'physical_evaluation' / 'attribution_v2')
    report = analyze_attribution(
        run_dir, output_dir, max_total_bytes=args.max_total_bytes,
        max_record_bytes=args.max_record_bytes, max_records=args.max_records,
    )
    print(json.dumps({
        'output_dir': str(output_dir.resolve()),
        'physical_shadow_records': report['inputs']['physical_shadow_records'],
        'threshold_excursion_events': report['summary_counters']['h100_threshold_excursion_events'],
        'event_window_trigger_rows': report['summary_counters']['event_window_trigger_rows'],
        'causal_effect_estimated': False,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()