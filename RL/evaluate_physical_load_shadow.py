"""Read bounded physical shadow records as DEVELOPMENT diagnostics, never audit.

This evaluator does not fit coefficients or recalibrate PID quantiles, and
does not equate sparse matched-plan origin coverage with event/trajectory safety.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from RL.utils.physical_load_shadow import RECORD_SCHEMA, preflight_physical_config
from RL.utils.physical_load_shadow_artifact import (
    BACKEND, LABEL_SEMANTICS, RUNTIME_AUTHORIZATION, file_sha256, strict_json,
)


POINT_QUANTILE_RESERVOIR_CAPACITY = 1024
MAX_POINT_GROUPS = 512
MAX_PENDING_HORIZON_PAIRS = 128


def _finite(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'Physical diagnostic requires finite {shape} arrays')
    return array


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class _BoundedReservoir:
    """Deterministic reservoir for bounded-memory tail quantile estimates."""

    def __init__(self, capacity: int, seed_material: str) -> None:
        digest = hashlib.sha256(seed_material.encode('utf-8')).digest()
        self.capacity = capacity
        self.values = np.empty(capacity, dtype=np.float64)
        self.count = 0
        self.random = random.Random(int.from_bytes(digest[:8], 'big'))

    def add_many(self, values: np.ndarray) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            self.count += 1
            if self.count <= self.capacity:
                self.values[self.count - 1] = value
                continue
            replacement = self.random.randrange(self.count)
            if replacement < self.capacity:
                self.values[replacement] = value

    def quantile(self, fraction: float) -> float:
        if self.count <= 0:
            return 0.0
        sample_count = min(self.count, self.capacity)
        return float(np.quantile(self.values[:sample_count], fraction))


class _PointSummaryAccumulator:
    """Exact first moments plus bounded reservoirs; memory does not grow with row count."""

    _METHODS = (
        ('dynamic_fir', 'predicted_peak_nm_by_blade'),
        ('v1_greybox', 'predicted_v1_peak_nm_by_blade'),
        ('persistence', 'predicted_persistence_peak_nm_by_blade'),
    )

    def __init__(self, key: str, *, capacity: int = POINT_QUANTILE_RESERVOIR_CAPACITY) -> None:
        self.key = key
        self.capacity = capacity
        self.origin_count = 0
        self.error_sum = {name: np.zeros(3, dtype=np.float64) for name, _ in self._METHODS}
        self.normalized_error_sum = {name: np.zeros(3, dtype=np.float64) for name, _ in self._METHODS}
        self.under_all = {
            name: _BoundedReservoir(capacity * 3, f'{key}|{name}|under_all')
            for name, _ in self._METHODS
        }
        self.under_by_blade = {
            name: [_BoundedReservoir(capacity, f'{key}|{name}|under_blade_{blade}')
                   for blade in range(3)]
            for name, _ in self._METHODS
        }

    def add(self, row: dict[str, Any]) -> None:
        actual = np.asarray(row['actual_future_peak_nm_by_blade'], dtype=np.float64)
        reference = np.asarray(row['pid_peak_abs_nm'], dtype=np.float64)
        if actual.shape != (3,) or reference.shape != (3,) or np.any(reference <= 0.0):
            raise ValueError('Physical point summary requires three finite positive references')
        self.origin_count += 1
        for name, field in self._METHODS:
            predicted = np.asarray(row[field], dtype=np.float64)
            if predicted.shape != (3,) or not np.isfinite(predicted).all():
                raise ValueError('Physical point summary prediction shape/values are invalid')
            error = np.abs(actual - predicted)
            under = np.maximum(actual - predicted, 0.0)
            self.error_sum[name] += error
            self.normalized_error_sum[name] += error / reference
            self.under_all[name].add_many(under)
            for blade in range(3):
                self.under_by_blade[name][blade].add_many(np.asarray([under[blade]]))

    def summary(self) -> dict[str, Any]:
        if self.origin_count <= 0:
            return {'origin_count': 0, 'methods': {}}
        method_metrics = {}
        for name, _field in self._METHODS:
            method_metrics[name] = {
                'mae_nm': float(self.error_sum[name].sum() / (self.origin_count * 3)),
                'mae_nm_by_blade': (self.error_sum[name] / self.origin_count).tolist(),
                'normalized_mae': float(self.normalized_error_sum[name].sum() / (self.origin_count * 3)),
                'positive_underestimate_p95_nm': self.under_all[name].quantile(0.95),
                'positive_underestimate_p99_nm': self.under_all[name].quantile(0.99),
                'positive_underestimate_p95_nm_by_blade': [
                    reservoir.quantile(0.95) for reservoir in self.under_by_blade[name]
                ],
            }
        old, new = method_metrics['v1_greybox']['mae_nm'], method_metrics['dynamic_fir']['mae_nm']
        return {
            'origin_count': self.origin_count,
            'methods': method_metrics,
            'quantile_method': 'exact_until_deterministic_reservoir_capacity_then_sampled',
            'quantile_reservoir_capacity_per_blade': self.capacity,
            'mae_reduction_vs_v1': None if old == 0.0 else float(1.0 - new / old),
        }


class _JointSummaryAccumulator:
    def __init__(self) -> None:
        self.origin_count = 0
        self.covered = 0
        self.margin_sum = 0.0
        self.near_upper = 0
        self.actual_near = 0
        self.not_actual_near = 0
        self.near_upper_when_not_actual_near = 0

    def add(self, values: tuple) -> None:
        covered, margin, near_upper, actual_near = values
        self.origin_count += 1
        self.covered += int(bool(covered))
        self.margin_sum += float(margin)
        self.near_upper += int(bool(near_upper))
        self.actual_near += int(bool(actual_near))
        self.not_actual_near += int(not bool(actual_near))
        self.near_upper_when_not_actual_near += int(bool(near_upper) and not bool(actual_near))

    def summary(self) -> dict[str, Any]:
        negative_fraction = (self.near_upper_when_not_actual_near / self.not_actual_near
                             if self.not_actual_near else None)
        return {
            'origin_count': self.origin_count,
            'joint_six_empirical_coverage': self.covered / self.origin_count,
            'mean_pid_illustrative_margin_nm': self.margin_sum / self.origin_count,
            'near_peak_upper_negative_origin_fraction': negative_fraction,
            'qualification': 'reused PID quantiles on matched RL development origins, not authorized RL coverage',
        }


def _joint_entry(short: dict[str, Any], long: dict[str, Any]) -> tuple[str, tuple] | None:
    for name in ('run_id', 'queue_epoch', 'origin_frame_id', 'causal_segment_id', 'origin_control_region',
                 'origin_sent_action', 'origin_signed_my_nm_by_blade', 'actor_policy_version',
                 'actor_policy_sha256', 'pid_peak_abs_nm', 'evaluation_reference_file_sha256',
                 'origin_sent_monotonic_s', 'prediction_started_monotonic_s', 'result_available_monotonic_s'):
        if short[name] != long[name]:
            raise ValueError(f'Physical diagnostic shared origin mismatch between windows: {name}')
    for name in ('actual_future_signed_my_nm_by_blade', 'executed_interval_actions',
                 'executed_interval_actor_policy_versions', 'executed_interval_actor_policy_sha256s'):
        if short[name] != long[name][:5]:
            raise ValueError(f'Physical diagnostic short/long prefix mismatch: {name}')
    for name in ('predicted_peak_nm_by_blade', 'predicted_v1_peak_nm_by_blade',
                 'predicted_persistence_peak_nm_by_blade'):
        if np.any(np.asarray(short[name]) > np.asarray(long[name])):
            raise ValueError('Physical diagnostic horizon max_abs peaks must be monotonic')
    if not all(r['pid_empirical_joint_margin_available'] is True for r in (short, long)):
        return None
    actual = np.asarray([r['actual_future_peak_nm_by_blade'] for r in (short, long)])
    predicted = np.asarray([r['predicted_peak_nm_by_blade'] for r in (short, long)])
    margins = np.stack([_finite(r['pid_empirical_margin_nm_by_blade_illustrative_only'], (3,))
                        for r in (short, long)])
    if np.any(margins < 0.0):
        raise ValueError('Physical illustrative margins cannot be negative')
    upper = predicted + margins
    for i, row in enumerate((short, long)):
        if not np.array_equal(_finite(row['pid_empirical_upper_nm_by_blade_illustrative_only'], (3,)), upper[i]):
            raise ValueError('Physical diagnostic illustrative upper does not equal point plus margin')
    reference = np.asarray(short['pid_peak_abs_nm'])
    return (f"wind{short['condition_id']}|{short['origin_control_region']}",
            (bool(np.all(actual <= upper)), float(margins.mean()),
             bool(np.any(upper >= 0.95 * reference)), bool(np.any(actual >= 0.95 * reference))))


def evaluate_shadow_records(path: Path, *, expected_bundle_sha256: str | None = None,
                            expected_dynamic_sha256: str | None = None,
                            evaluation_reference=None, expected_reference_sha256: str | None = None,
                            worker_summary: dict[str, Any] | None = None, expected_slot_count: int | None = None,
                            expected_slot_identities: list[tuple[int, int, int]] | None = None,
                            max_records: int = 250000, max_record_bytes: int = 32768,
                            max_total_bytes: int = 1073741824) -> dict[str, Any]:
    if min(max_records, max_record_bytes, max_total_bytes) < 1 or path.stat().st_size > max_total_bytes:
        raise ValueError('Physical shadow diagnostic input exceeds its budget')
    if expected_slot_identities is not None:
        if (not expected_slot_identities or len(expected_slot_identities) > 1024 or
                any(not isinstance(slot, tuple) or len(slot) != 3 or
                    any(type(value) is not int for value in slot) or
                    slot[0] < -1 or min(slot[1:]) < 0 for slot in expected_slot_identities) or
                len(set(expected_slot_identities)) != len(expected_slot_identities) or
                expected_slot_count != len(expected_slot_identities)):
            raise ValueError('Physical diagnostic expected manifest slots/count are invalid')
    expected_slot_set = None if expected_slot_identities is None else set(expected_slot_identities)
    source_sha = file_sha256(path)
    counters: Counter[str] = Counter()
    groups: dict[str, _PointSummaryAccumulator] = {}
    unverified_groups: dict[str, _PointSummaryAccumulator] = {}
    pairs: OrderedDict[tuple, dict] = OrderedDict()
    joint: dict[str, _JointSummaryAccumulator] = {}
    record_slot_order: list[tuple[int, int, int]] = []
    latest_endpoints: dict[tuple, float] = {}
    last_origin_by_horizon: dict[tuple, float] = {}
    artifact_identity = None
    run_id = None
    with path.open('rb') as handle:
        while True:
            encoded = handle.readline(max_record_bytes + 1)
            if not encoded:
                break
            if len(encoded) > max_record_bytes:
                raise ValueError('Physical shadow diagnostic record exceeds byte budget')
            if not encoded.strip():
                continue
            counters['input_records'] += 1
            if counters['input_records'] > max_records:
                raise ValueError('Physical shadow diagnostic input exceeds row budget')
            def reject_constant(value):
                raise ValueError(f'Non-finite JSON constant in physical diagnostic: {value}')
            def unique_object(items):
                obj = {}
                for key, value in items:
                    if key in obj:
                        raise ValueError('Duplicate JSON key in physical diagnostic')
                    obj[key] = value
                return obj
            row = json.loads(encoded, parse_constant=reject_constant, object_pairs_hook=unique_object)
            if (not isinstance(row, dict) or row.get('schema_version') != RECORD_SCHEMA or
                    row.get('backend') != BACKEND or row.get('label_semantics') != LABEL_SEMANTICS or
                    row.get('authorization') != RUNTIME_AUTHORIZATION or
                    row.get('rl_margin_coverage_authorized') is not False or
                    row.get('calibration_eligible') is not False or
                    row.get('independent_audit_eligible') is not False):
                raise ValueError('Only the independent physical DEVELOPMENT record contract is accepted')
            identity = (row.get('bundle_sha256'), row.get('dynamic_model_file_sha256'))
            if not all(_sha(value) for value in identity):
                raise ValueError('Physical diagnostic artifact identity is missing')
            if artifact_identity is None:
                artifact_identity = identity
            if identity != artifact_identity or (expected_bundle_sha256 is not None and identity[0] != expected_bundle_sha256):
                raise ValueError('Physical diagnostic mixes or mismatches frozen bundles')
            if expected_dynamic_sha256 is not None and identity[1] != expected_dynamic_sha256:
                raise ValueError('Physical diagnostic dynamic model file SHA mismatch')
            identifier = row.get('run_id')
            if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
                raise ValueError('Physical diagnostic needs its worker run identity')
            if run_id is None:
                run_id = identifier
            if run_id != identifier:
                raise ValueError('Physical diagnostic cannot pool different runs as one joint population')
            for name in ('queue_epoch', 'condition_id', 'condition_instance_id', 'causal_segment_id', 'origin_frame_id'):
                if type(row.get(name)) is not int or row[name] < (-1 if name in ('queue_epoch', 'origin_frame_id') else 0):
                    raise ValueError('Physical diagnostic queue/frame identity is invalid')
            slot = (row['queue_epoch'], row['condition_instance_id'], row['condition_id'])
            if expected_slot_set is not None and slot not in expected_slot_set:
                raise ValueError('Physical diagnostic record slot does not belong to the expected manifest')
            if not record_slot_order or record_slot_order[-1] != slot:
                if len(record_slot_order) >= 1024:
                    raise ValueError('Physical diagnostic slot order exceeds the bounded receipt budget')
                if slot in record_slot_order:
                    raise ValueError('Physical diagnostic slot order revisits a completed queue slot')
                if pairs:
                    counters['origins_without_both_eligible_horizons'] += len(pairs)
                    pairs.clear()
                latest_endpoints.clear()
                last_origin_by_horizon.clear()
                record_slot_order.append(slot)
            h, frames = row['horizon_s'], row['horizon_frames']
            if (h, frames) not in ((0.1, 5), (0.2, 10)):
                raise ValueError('Physical diagnostic cannot expand to 0.5s warnings')
            origin = float(row['origin_time_s'])
            if not np.isfinite(origin) or abs(row['label_endpoint_time_s'] - origin - h) > 1e-6:
                raise ValueError('Physical diagnostic label interval mismatch')
            key = (run_id, row['queue_epoch'], row['condition_id'], row['condition_instance_id'], row['causal_segment_id'], origin)
            horizon_key = (key[:-1], h)
            previous_origin = last_origin_by_horizon.get(horizon_key)
            if previous_origin is not None and origin <= previous_origin + 1e-9:
                raise ValueError('Physical diagnostic contains duplicate or out-of-order horizon origins')
            last_origin_by_horizon[horizon_key] = origin
            if len(last_origin_by_horizon) > 16_384:
                raise ValueError('Physical diagnostic causal-segment identity state exceeds its bounded budget')
            endpoint = float(row['label_endpoint_time_s'])
            if endpoint < latest_endpoints.get(key[:-1], -np.inf) - 1e-6:
                raise ValueError('Physical diagnostic must retain worker label-maturity ordering')
            latest_endpoints[key[:-1]] = endpoint
            if len(latest_endpoints) > 16_384:
                raise ValueError('Physical diagnostic label-endpoint state exceeds its bounded budget')
            for pending_key in tuple(pairs):
                if pending_key[:-1] != key[:-1] or pending_key[-1] + 0.2 < endpoint - 1e-6:
                    counters['origins_without_both_eligible_horizons'] += 1
                    del pairs[pending_key]
            reference = _finite(row['pid_peak_abs_nm'], (3,))
            actual = _finite(row['actual_future_peak_nm_by_blade'], (3,))
            my = _finite(row['actual_future_signed_my_nm_by_blade'], (frames, 3))
            plan = _finite(row['executed_interval_actions'], (frames, 2))
            command = _finite(row['origin_sent_action'], (2,))
            bounds = np.asarray([21586451.33303, 0.43])
            if np.any(plan < 0.0) or np.any(plan > bounds) or np.any(command < 0.0) or np.any(command > bounds):
                raise ValueError('Physical diagnostic executed demand outside engineering bounds')
            mismatch = float(np.max(2.0 * np.abs(plan - command) / bounds))
            if (abs(mismatch - row['plan_mismatch_normalized_linf']) > 1e-10 or
                    row['approximate_zoh'] is not (mismatch <= 0.05)):
                raise ValueError('Physical diagnostic ZOH qualification disagrees with executed intervals')
            versions = row['executed_interval_actor_policy_versions']
            shas = row['executed_interval_actor_policy_sha256s_unique']
            full_shas = row['executed_interval_actor_policy_sha256s']
            if (len(versions) != frames or len(full_shas) != frames or not shas or
                    shas != list(dict.fromkeys(full_shas)) or
                    any(type(v) is not int or v < 0 for v in versions) or
                    any(not isinstance(s, str) or (s and not _sha(s)) for s in full_shas)):
                raise ValueError('Physical diagnostic interval Actor provenance incomplete')
            actor_frozen = bool(all(v == row['actor_policy_version'] for v in versions) and
                                shas == [row['actor_policy_sha256']])
            if row['actor_policy_window_frozen'] is not actor_frozen:
                raise ValueError('Physical diagnostic frozen Actor flag disagrees with executed intervals')
            if np.any(reference <= 0.0) or np.any(actual < 0.0) or not np.array_equal(actual, np.max(np.abs(my), axis=0)):
                raise ValueError('Physical diagnostic My labels/reference do not close')
            reference_verified = False
            if evaluation_reference is not None:
                entry = evaluation_reference.entry(row['condition_id'])
                if (row.get('evaluation_reference_file_sha256') != expected_reference_sha256 or
                        row.get('pid_raw_sha256') != entry.pid_raw_sha256 or
                        row.get('scenario_id') != entry.scenario_id or
                        row.get('time_window_start_s') != entry.time_window_start_s or
                        row.get('time_window_end_s') != entry.time_window_end_s or
                        row.get('expected_trace_signature') != entry.trace_signature or
                        row.get('expected_trace_sample_count') != int(entry.timestamps_s.size) or
                        not np.array_equal(reference, entry.blade_root_peak_abs_nm) or
                        origin < entry.time_window_start_s - 1e-6 or endpoint > entry.time_window_end_s + 1e-6):
                    raise ValueError('Physical diagnostic exact-window PID reference identity/peaks mismatch')
                reference_verified = True
            trace_complete = bool(row.get('strict_physical_trace_complete') is True and
                row.get('trace_signature_match', 0.0) > 0.5 and
                row.get('observed_trace_signature') == row.get('expected_trace_signature') and
                row.get('observed_trace_sample_count') == row.get('expected_trace_sample_count') and
                row.get('fixed_simulation_time_offset_s') == 0.0)
            if not (reference_verified and trace_complete and row.get('identity_status') == 'strict_case_matched'):
                counters['unconfirmed_slot_records'] += 1
            supported = row['prediction_supported'] is True
            if supported:
                for field in ('predicted_peak_nm_by_blade', 'predicted_v1_peak_nm_by_blade',
                              'predicted_persistence_peak_nm_by_blade'):
                    if np.any(_finite(row[field], (3,)) < 0.0):
                        raise ValueError('Physical diagnostic peaks cannot be negative')
                counters['point_supported_records'] += 1
            else:
                counters[f'unknown_{row.get("unknown_reason", "unspecified")}'] += 1
            eligible = bool(supported and row.get('approximate_zoh') is True and
                row.get('actor_policy_window_frozen') is True and row.get('identity_status') == 'strict_case_matched' and
                trace_complete)
            if row.get('development_point_eligible') is not eligible:
                raise ValueError('Physical diagnostic eligibility disagrees with strict identity/plan/Actor flags')
            if not reference_verified:
                counters['reference_unverified_records'] += 1
                eligible = False
            if not eligible:
                counters['development_ineligible_records'] += 1
                # Retain empirical point diagnostics without granting strict
                # pairing, joint coverage or capture completeness eligibility.
                if (reference_verified and supported and row.get('identity_status') == 'development_unverified'
                        and row.get('approximate_zoh') is True and row.get('actor_policy_window_frozen') is True):
                    counters['operator_unverified_point_records'] += 1
                    group = f"wind{row['condition_id']}|{row['origin_control_region']}|H{h:.2f}"
                    if group not in unverified_groups:
                        if len(unverified_groups) >= MAX_POINT_GROUPS:
                            raise ValueError('Physical unverified point group count exceeds its bounded budget')
                        unverified_groups[group] = _PointSummaryAccumulator(group)
                    unverified_groups[group].add(row)
                continue
            counters['development_eligible_records'] += 1
            group = f"wind{row['condition_id']}|{row['origin_control_region']}|H{h:.2f}"
            if group not in groups:
                if len(groups) >= MAX_POINT_GROUPS:
                    raise ValueError('Physical diagnostic group count exceeds its bounded budget')
                groups[group] = _PointSummaryAccumulator(group)
            groups[group].add(row)
            all_group = f'all|H{h:.2f}'
            if all_group not in groups:
                groups[all_group] = _PointSummaryAccumulator(all_group)
            groups[all_group].add(row)
            if key not in pairs and len(pairs) >= MAX_PENDING_HORIZON_PAIRS:
                raise ValueError('Physical diagnostic pending horizon-pair state exceeds its bounded budget')
            pairs.setdefault(key, {})[h] = row
            if set(pairs[key]) == {0.1, 0.2}:
                entry = _joint_entry(pairs[key][0.1], pairs[key][0.2])
                if entry is None:
                    counters['origins_without_pid_illustrative_margin'] += 1
                else:
                    group, values = entry
                    for joint_group in (group, 'all'):
                        if joint_group not in joint:
                            if len(joint) >= MAX_POINT_GROUPS:
                                raise ValueError('Physical joint summary group count exceeds its bounded budget')
                            joint[joint_group] = _JointSummaryAccumulator()
                        joint[joint_group].add(values)
                del pairs[key]
    point_metrics = {name: accumulator.summary() for name, accumulator in groups.items()}
    operator_unverified_metrics = {
        name: accumulator.summary() for name, accumulator in unverified_groups.items()
    }
    counters['origins_without_both_eligible_horizons'] += len(pairs)
    joint_metrics = {name: accumulator.summary() for name, accumulator in joint.items()}
    if file_sha256(path) != source_sha:
        raise ValueError('Physical diagnostic input changed during evaluation')
    completeness = {'verified': False, 'complete_capture': False, 'reason': 'worker summary not provided'}
    if worker_summary is not None:
        if (worker_summary.get('run_id') != run_id or worker_summary.get('bundle_sha256') != artifact_identity[0] or
                worker_summary.get('dynamic_model_file_sha256') != artifact_identity[1] or
                worker_summary.get('evaluation_reference_file_sha256') != expected_reference_sha256 or
                worker_summary.get('records_file_sha256') != source_sha):
            raise ValueError('Physical diagnostic worker summary/run/artifact/file identity mismatch')
        statistics = worker_summary.get('counters', {})
        core_statistics = worker_summary.get('core_counters', {})
        receipts = worker_summary.get('condition_receipts', [])
        if (not isinstance(statistics, dict) or not isinstance(core_statistics, dict) or
                any(type(value) is not int or value < 0 for value in (*statistics.values(), *core_statistics.values())) or
                not isinstance(receipts, list) or len(receipts) > 1024 or
                any(not isinstance(receipt, dict) for receipt in receipts)):
            raise ValueError('Physical diagnostic worker counters/receipts are invalid')
        receipt_ids = [(item.get('queue_epoch'), item.get('condition_instance_id'), item.get('condition_id'))
                       for item in receipts]
        slot_identities_match = bool(expected_slot_identities is not None and receipt_ids == expected_slot_identities)
        record_slots_match = bool(expected_slot_identities is not None and record_slot_order == expected_slot_identities)
        complete_receipts = 0
        for receipt in receipts:
            verified = bool(receipt.get('identity_status') == 'strict_case_matched' and
                            receipt.get('strict_physical_trace_complete') is True and
                            receipt.get('finish_slot_identity_match') is True and evaluation_reference is not None)
            if verified:
                entry = evaluation_reference.entry(receipt['condition_id'])
                verified = bool(receipt.get('evaluation_reference_file_sha256') == expected_reference_sha256 and
                    receipt.get('scenario_id') == entry.scenario_id and receipt.get('pid_raw_sha256') == entry.pid_raw_sha256 and
                    receipt.get('time_window_start_s') == entry.time_window_start_s and
                    receipt.get('time_window_end_s') == entry.time_window_end_s and
                    receipt.get('expected_trace_signature') == receipt.get('observed_trace_signature') == entry.trace_signature and
                    receipt.get('expected_trace_sample_count') == receipt.get('observed_trace_sample_count') == int(entry.timestamps_s.size) and
                    receipt.get('fixed_simulation_time_offset_s') == 0.0 and receipt.get('trace_signature_match', 0.0) > 0.5)
            complete_receipts += int(verified)
        receipt_counts_close = bool(expected_slot_count is not None and len(receipts) == expected_slot_count and
            statistics.get('finished_conditions', 0) == len(receipts) and
            statistics.get('strict_complete_conditions', 0) == complete_receipts == expected_slot_count)
        complete = bool(worker_summary.get('finalized') is True and worker_summary.get('faulted') is False and
            statistics.get('records_written', 0) == counters['input_records'] and
            statistics.get('bytes_written', 0) == path.stat().st_size and
            all(statistics.get(name, 0) == 0 for name in ('queue_full_dropped', 'invalid_submissions', 'worker_faults',
                'write_budget_dropped', 'condition_receipt_capacity_dropped', 'finish_slot_identity_mismatch',
                'duplicate_slot_finish')) and
            core_statistics.get('record_capacity_dropped', 0) == 0 and
            all(count == 0 for name, count in core_statistics.items() if name.startswith('invalidation_')) and
            counters['unconfirmed_slot_records'] == 0 and slot_identities_match and record_slots_match and receipt_counts_close)
        completeness = {'verified': True, 'complete_capture': complete,
            'manifest_slot_identities_match': slot_identities_match,
            'record_slot_order_matches_manifest': record_slots_match,
            'strict_receipt_counts_close': receipt_counts_close,
            'expected_slots': expected_slot_count, 'worker_counters': statistics, 'core_counters': core_statistics,
            'reason': 'complete bounded development capture, not a full event audit' if complete else
                      'incomplete slots, queue/record/budget drops, invalid frames, shadow fault or nonfinal worker'}
    return {
        'schema_version': 'physical_load_shadow_development_evaluation_v1',
        'source_path': str(path), 'source_sha256': source_sha,
        'artifact_identity': artifact_identity, 'run_id': run_id, 'counters': dict(counters),
        'capture_completeness': completeness,
        'point_metrics': point_metrics, 'pid_illustrative_joint_metrics': joint_metrics,
        'operator_unverified_point_diagnostics': {
            'qualification': 'empirical development only; operator attestation, not strict same-trace PID identity',
            'point_metrics': operator_unverified_metrics,
            'strict_eligible': False,
        },
        'streaming_memory_contract': {
            'point_summary_group_limit': MAX_POINT_GROUPS,
            'quantile_reservoir_capacity_per_blade': POINT_QUANTILE_RESERVOIR_CAPACITY,
            'pending_horizon_pair_limit': MAX_PENDING_HORIZON_PAIRS,
            'persistent_metric_rows_retained': 0,
        },
        'events': {'measured': False, 'reason': 'requires complete strict real trace and availability-aware event audit; sparse origin rows are not whole-trajectory event evidence'},
        'authorization': dict(RUNTIME_AUTHORIZATION),
        'fresh_independent_audit': False, 'counterfactual_validation': False,
        'decision': 'DEVELOPMENT_DIAGNOSTIC_ONLY' if counters['development_eligible_records'] else 'INCONCLUSIVE_NO_ELIGIBLE_RECORDS',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--effective-config', type=Path, required=True)
    parser.add_argument('--run-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = strict_json(args.effective_config)
    settings = config['load_predictor']
    if settings.get('backend') != BACKEND or settings.get('authorization') != RUNTIME_AUTHORIZATION:
        raise ValueError('Physical development evaluator requires the effective physical shadow configuration')
    project_root = Path(__file__).resolve().parents[1]
    artifacts, reference = preflight_physical_config(config, project_root=project_root)
    summary = strict_json(args.run_summary)
    worker_summary = summary.get('physical_load_shadow', summary)
    from RL.utils.queue_manifest import expand_queue_manifest
    manifest_path = Path(config['queue']['manifest_path'])
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    slots = expand_queue_manifest(strict_json(manifest_path))
    expected_slots = len(slots)
    report = evaluate_shadow_records(args.records, expected_bundle_sha256=settings['bundle_sha256'],
        expected_dynamic_sha256=artifacts.dynamic_file_sha256, evaluation_reference=reference,
        expected_reference_sha256=settings['evaluation_reference']['file_sha256'],
        worker_summary=worker_summary, expected_slot_count=expected_slots,
        expected_slot_identities=[(s.queue_epoch, s.condition_instance_id, s.condition_id) for s in slots],
        max_records=settings.get('max_records', 250000), max_record_bytes=settings.get('max_record_bytes', 32768),
        max_total_bytes=settings.get('max_total_bytes', 1073741824))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'report.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'decision': report['decision'], 'counters': report['counters']}, ensure_ascii=False))


if __name__ == '__main__':
    main()