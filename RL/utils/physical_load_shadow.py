"""Bounded development-only physical shadow, independent of the control law.

No NN interface emulation, optimizer, candidate ranking, or action feedback.
The frozen PID model is applied to actually sent RL ZOH commands solely to
measure distribution transfer. Its PID margins remain illustrative.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from RL.utils.operating_state import OperatingPoint, OperatingStateClassifier, OptimizationState
from RL.utils.paired_pid_reference import PairedPidReference
from RL.utils.physical_load_margin import INPUT_FIELDS, LOAD_FIELDS, REGIONS, PidCase
from RL.utils.physical_load_shadow_artifact import (
    BACKEND, LABEL_SEMANTICS, RUNTIME_AUTHORIZATION, PhysicalShadowArtifacts,
    artifact_path, file_sha256, load_physical_shadow_artifacts,
)


RECORD_SCHEMA = 'physical_load_shadow_development_record_v1'
OUTCOME_OK = 'complete'


@dataclass(frozen=True)
class SentPhysicalFrame:
    condition_id: int
    condition_instance_id: int
    frame_id: int
    values: tuple[float, ...]
    sent_action: tuple[float, float]
    pid_peaks: tuple[float, float, float]
    actor_version: int
    actor_sha256: str
    sent_monotonic_s: float
    queue_epoch: int = -1

    @property
    def simulation_time_s(self) -> float:
        return self.values[0]


@dataclass
class _Pending:
    frame: SentPhysicalFrame
    region: int
    segment_id: int
    prediction: dict[str, Any]
    future_my: list[np.ndarray]
    interval_actions: list[np.ndarray]
    actor_versions: list[int]
    actor_sha256s: list[str]
    completed: set[int]


class PhysicalShadowCore:
    """Single-worker state. Histories, classifier and derivatives reset together."""

    def __init__(self, artifacts: PhysicalShadowArtifacts, *, origin_stride_frames: int = 5,
                 record_capacity: int = 16384, run_id: str | None = None) -> None:
        if origin_stride_frames < 1 or record_capacity < 1:
            raise ValueError('Physical shadow capacities/stride must be positive')
        self.artifacts = artifacts
        self.run_id = uuid4().hex if run_id is None else run_id
        self.model = artifacts.model
        self.base = self.model.physics.protocol
        self.dt = self.base['step_duration_s']
        self.stride = origin_stride_frames
        self.records: deque[dict[str, Any]] = deque(maxlen=record_capacity)
        self.stats: Counter[str] = Counter()
        self._history: deque[tuple[SentPhysicalFrame, np.ndarray, int, bool]] = deque(
            maxlen=max(self.model.lags) + 1)
        self._pending: list[_Pending] = []
        self._key: tuple[int, int, int] | None = None
        self._last: SentPhysicalFrame | None = None
        self._classifier = OperatingStateClassifier()
        self._derivative = np.zeros(5, dtype=np.float64)
        self._segment_start = 0.0
        self._segment_index = 0
        self._segment_id = 0
        self._cumulative = np.zeros(3, dtype=np.float64)
        self._reference: tuple[float, ...] | None = None
        self._last_outcome_seen = True
        self._last_outcome_required = False

    @property
    def history_count(self) -> int:
        return len(self._history)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _reset_segment(self, *, reset_cumulative: bool = False) -> None:
        self.stats['dropped_incomplete_origins'] += len(self._pending)
        self._pending.clear()
        self._history.clear()
        self._last = None
        self._classifier.reset()
        self._derivative.fill(0.0)
        self._segment_index = 0
        self._segment_id += 1
        self._last_outcome_seen = True
        if reset_cumulative:
            self._cumulative.fill(0.0)
            self._reference = None

    def reset_condition(self, condition_id: int, instance_id: int, queue_epoch: int = -1) -> None:
        self._reset_segment(reset_cumulative=True)
        self._key = (condition_id, instance_id, queue_epoch)

    def invalidate(self, reason: str) -> None:
        self.stats[f'invalidation_{reason}'] += 1
        self._reset_segment()

    def confirm_outcome(self, timestamp_s: float, tag: str) -> None:
        if tag != OUTCOME_OK:
            self.invalidate(tag)
        elif self._last is not None and math.isclose(timestamp_s, self._last.simulation_time_s,
                                                    rel_tol=0.0, abs_tol=1e-6):
            self._last_outcome_seen = True
        else:
            self.invalidate('outcome_origin_mismatch')

    def _case(self) -> PidCase:
        rows = tuple(self._history)
        matrix = np.asarray([row[0].values for row in rows], dtype=np.float64)
        values = {name: matrix[:, index] for index, name in enumerate(INPUT_FIELDS)}
        actions = np.asarray([row[0].sent_action for row in rows], dtype=np.float64)
        my = np.column_stack([values[name] for name in LOAD_FIELDS])
        return PidCase(self._key[0], values, actions, my,
                       np.asarray([row[1] for row in rows]),
                       np.asarray([row[2] for row in rows], dtype=np.int8),
                       np.zeros(len(rows), dtype=np.int32),
                       np.asarray([row[3] for row in rows]), np.ones(len(rows), dtype=bool),
                       np.asarray(rows[-1][0].pid_peaks),
                       np.broadcast_to(self._cumulative, my.shape).copy(), {})

    def _record(self, pending: _Pending, horizon_frames: int, available_frame: SentPhysicalFrame) -> None:
        index = 0 if horizon_frames == 5 else 1
        prediction = pending.prediction
        actual = np.max(np.abs(pending.future_my[:horizon_frames]), axis=0)
        plan = np.asarray(pending.interval_actions[:horizon_frames], dtype=np.float64)
        bounds = np.asarray([self.base['static_parameters']['request_torque_max_nm'], 0.43])
        mismatch = float(np.max(2.0 * np.abs(plan - np.asarray(pending.frame.sent_action)) / bounds))
        actor_frozen = all(v == pending.frame.actor_version for v in pending.actor_versions[:horizon_frames])
        actor_frozen &= all(s == pending.frame.actor_sha256 for s in pending.actor_sha256s[:horizon_frames])
        known = bool(prediction['supported'][index])
        peak = prediction['peaks'][index].tolist() if known else None
        margin_known = bool(prediction['margin_known']) and known
        margin = prediction['margins'][index].tolist() if margin_known else None
        upper = (prediction['peaks'][index] + prediction['margins'][index]).tolist() if margin_known else None
        approximate_zoh = mismatch <= self.base['evaluation']['approximate_zoh_max_normalized_action_linf']
        diagnostic_eligible = bool(known and actor_frozen and approximate_zoh)
        record = {
            'schema_version': RECORD_SCHEMA, 'backend': BACKEND,
            'evidence_role': 'rl_distribution_development_diagnostic_not_independent_acceptance',
            'authorization': dict(RUNTIME_AUTHORIZATION),
            'run_id': self.run_id, 'queue_epoch': pending.frame.queue_epoch,
            'condition_id': pending.frame.condition_id,
            'condition_instance_id': pending.frame.condition_instance_id,
            'origin_time_s': pending.frame.simulation_time_s,
            'label_endpoint_time_s': available_frame.simulation_time_s,
            'horizon_s': horizon_frames * self.dt, 'horizon_frames': horizon_frames,
            'origin_frame_id': pending.frame.frame_id,
            'causal_segment_id': pending.segment_id,
            'origin_control_region': REGIONS[pending.region] if pending.region >= 0 else 'unknown',
            'predicted_peak_nm_by_blade': peak,
            'predicted_v1_peak_nm_by_blade': prediction['v1'][index].tolist() if known else None,
            'predicted_persistence_peak_nm_by_blade': prediction['persistence'][index].tolist() if known else None,
            'pid_empirical_margin_nm_by_blade_illustrative_only': margin,
            'pid_empirical_upper_nm_by_blade_illustrative_only': upper,
            'pid_empirical_joint_margin_available': margin_known,
            'rl_margin_coverage_authorized': False,
            'actual_future_peak_nm_by_blade': actual.tolist(),
            'prediction_supported': known,
            'unknown_reason': str(prediction['reasons'][index]) if not known else '',
            'approximate_zoh': bool(approximate_zoh), 'plan_mismatch_normalized_linf': mismatch,
            'label_semantics': LABEL_SEMANTICS,
            'origin_sent_action': list(pending.frame.sent_action),
            'origin_signed_my_nm_by_blade': [pending.frame.values[INPUT_FIELDS.index(name)] for name in LOAD_FIELDS],
            'executed_interval_actions': plan.tolist(),
            'executed_interval_actor_policy_versions': pending.actor_versions[:horizon_frames],
            'executed_interval_actor_policy_sha256s': pending.actor_sha256s[:horizon_frames],
            'executed_interval_actor_policy_sha256s_unique': list(dict.fromkeys(pending.actor_sha256s[:horizon_frames])),
            'actual_future_signed_my_nm_by_blade': [v.tolist() for v in pending.future_my[:horizon_frames]],
            'pid_peak_abs_nm': list(pending.frame.pid_peaks),
            'observed_cumulative_peak_nm_by_blade_at_origin': prediction['cumulative'].tolist(),
            'observed_exceeded_latched': bool(np.any(prediction['cumulative'] > pending.frame.pid_peaks)),
            'risk_status': 'observed_exceeded_latched' if np.any(prediction['cumulative'] > pending.frame.pid_peaks)
                           else 'unknown_rl_margin_not_validated',
            'actor_policy_version': pending.frame.actor_version,
            'actor_policy_sha256': pending.frame.actor_sha256,
            'actor_policy_window_frozen': bool(actor_frozen),
            'development_point_eligible': diagnostic_eligible,
            'calibration_eligible': False, 'independent_audit_eligible': False,
            'bundle_sha256': self.artifacts.bundle_file_sha256,
            'dynamic_model_file_sha256': self.artifacts.dynamic_file_sha256,
            'prediction_compute_ms': prediction['compute_ms'],
            'origin_sent_monotonic_s': pending.frame.sent_monotonic_s,
            'prediction_started_monotonic_s': prediction['started'],
            'result_available_monotonic_s': prediction['available'],
            'sent_to_result_available_ms': (prediction['available'] - pending.frame.sent_monotonic_s) * 1000.0,
            'latest_submitted_sim_time_at_result_s': prediction['latest_sim_time'],
            'result_age_sim_s': None if prediction['latest_sim_time'] is None else max(
                0.0, prediction['latest_sim_time'] - pending.frame.simulation_time_s),
            'identity_status': 'pending_end_of_condition_strict_trace',
        }
        if len(self.records) == self.records.maxlen:
            self.stats['record_capacity_dropped'] += 1
        else:
            self.records.append(record)
            self.stats['completed_horizon_records'] += 1

    def observe(self, frame: SentPhysicalFrame, *, require_outcome: bool = False,
                latest_sim_time=None) -> dict[str, Any] | None:
        key = (frame.condition_id, frame.condition_instance_id, frame.queue_epoch)
        if self._key != key:
            self.reset_condition(*key)
        if self._last is not None and (
                not math.isclose(frame.simulation_time_s - self._last.simulation_time_s, self.dt,
                                 rel_tol=0.0, abs_tol=1e-6) or
                (frame.frame_id >= 0 and self._last.frame_id >= 0 and frame.frame_id != self._last.frame_id + 1) or
                (self._last_outcome_required and not self._last_outcome_seen)):
            self.invalidate('gap_or_unconfirmed_outcome')
        current_my = np.asarray(frame.values)[[INPUT_FIELDS.index(name) for name in LOAD_FIELDS]]
        if self._reference is not None and self._reference != frame.pid_peaks:
            self.invalidate('pid_reference_changed')
        self._reference = frame.pid_peaks
        self._cumulative = np.maximum(self._cumulative, np.abs(current_my))
        for pending in self._pending:
            pending.future_my.append(current_my.copy())
            pending.actor_versions.append(frame.actor_version)
            pending.actor_sha256s.append(frame.actor_sha256)
            n = len(pending.future_my)
            for h in (5, 10):
                if n == h and h not in pending.completed:
                    self._record(pending, h, frame)
                    pending.completed.add(h)
            # u_(t+H) does not drive the observation at t+H.
            if n < 10:
                pending.interval_actions.append(np.asarray(frame.sent_action, dtype=np.float64))
        self._pending = [p for p in self._pending if len(p.future_my) < 10]
        motion_fields = ('I_RotorSpeed', 'I_PitchPosition1', *LOAD_FIELDS)
        motion_indices = [INPUT_FIELDS.index(name) for name in motion_fields]
        if self._last is None:
            self._segment_start = frame.simulation_time_s
            previous_action = frame.sent_action
        else:
            alpha = 1.0 - math.exp(-self.dt / self.base['model']['derivative_filter_tau_s'])
            derivative = (np.asarray(frame.values)[motion_indices] - np.asarray(self._last.values)[motion_indices]) / self.dt
            self._derivative += alpha * (derivative - self._derivative)
            previous_action = self._last.sent_action
        values = dict(zip(INPUT_FIELDS, frame.values))
        classification = self._classifier.classify(OperatingPoint(
            timestamp_s=frame.simulation_time_s, wind_speed_mps=values['I_AverageWindSpeed'],
            rotor_speed_rad_s=values['I_RotorSpeed'], real_power_w=values['I_RealPower'],
            pitch_rad=values['I_PitchPosition1'], previous_torque_nm=previous_action[0],
            previous_pitch_rad=previous_action[1]))
        region = -1 if classification.optimization_state in (
            OptimizationState.STARTUP_STOPPED, OptimizationState.NEGATIVE_POWER_ABNORMAL,
        ) else REGIONS.index(classification.control_region.value)
        warm = frame.simulation_time_s - self._segment_start >= self.base['warmup_s'] - 1e-6
        self._history.append((frame, self._derivative.copy(), region, warm))
        self._last = frame
        self._last_outcome_required = require_outcome
        self._last_outcome_seen = not require_outcome
        index = self._segment_index
        self._segment_index += 1
        self.stats['observed_frames'] += 1
        if not warm or len(self._history) <= max(self.model.lags) or index % self.stride:
            return None
        case = self._case()
        origin = np.asarray([len(case.time) - 1], dtype=np.int64)
        started = time.monotonic()
        batch = self.model.predict(case, origin)
        # Identity was strictly checked once by the loader. Avoid reserializing
        # the large immutable FIR document at every online origin.
        scales, scale_known = self.artifacts.scale.predict(case, origin)
        group = self.artifacts.calibration['groups'].get(REGIONS[region]) if region >= 0 else None
        margin_known = bool(group and group['supported'] and scale_known[0, :2].all())
        margins = scales[0, :2] * group['joint_conditional95'] if margin_known else np.full((2, 3), np.nan)
        margin_known &= bool(np.isfinite(margins).all() and (margins >= 0.0).all())
        available = time.monotonic()
        latest_value = frame.simulation_time_s if latest_sim_time is None else latest_sim_time()
        latest = float(latest_value) if latest_value is not None and math.isfinite(latest_value) else None
        prediction = {
            'peaks': batch.peaks_nm['dynamic_fir'][0, :2].copy(),
            'v1': batch.peaks_nm['v1_greybox'][0, :2].copy(),
            'persistence': batch.peaks_nm['persistence'][0, :2].copy(),
            'supported': batch.supported[0, :2].copy(), 'reasons': batch.reasons[0, :2].copy(),
            'margins': margins.copy(), 'margin_known': margin_known,
            'started': started, 'available': available, 'latest_sim_time': latest,
            'compute_ms': (available - started) * 1000.0, 'cumulative': self._cumulative.copy(),
        }
        self._pending.append(_Pending(frame, region, self._segment_id, prediction, [],
            [np.asarray(frame.sent_action)], [frame.actor_version], [frame.actor_sha256], set()))
        self.stats['submitted_origins'] += 1
        self.stats['unsupported_origins'] += int(not batch.supported[0, :2].all())
        return prediction

    def observe_terminal(self, *, payload: Mapping[str, float], condition_id: int,
                         condition_instance_id: int, frame_id: int = -1, queue_epoch: int = -1) -> None:
        """Mature labels from a real same-stream terminal; never invent u_end."""
        if self._key != (condition_id, condition_instance_id, queue_epoch) or self._last is None:
            self.invalidate('terminal_identity_mismatch')
            return
        timestamp = float(payload['Time'])
        if (not math.isclose(timestamp - self._last.simulation_time_s, self.dt, rel_tol=0.0, abs_tol=1e-6) or
                (frame_id >= 0 and self._last.frame_id >= 0 and frame_id != self._last.frame_id + 1) or
                (self._last_outcome_required and not self._last_outcome_seen)):
            self.invalidate('terminal_gap_or_unconfirmed_outcome')
            return
        my = np.asarray([payload[name] for name in LOAD_FIELDS], dtype=np.float64)
        if not np.isfinite(my).all():
            self.invalidate('terminal_invalid_my')
            return
        self._cumulative = np.maximum(self._cumulative, np.abs(my))
        values = list(self._last.values)
        values[0] = timestamp
        for blade, name in enumerate(LOAD_FIELDS):
            values[INPUT_FIELDS.index(name)] = float(my[blade])
        terminal = SentPhysicalFrame(condition_id, condition_instance_id, frame_id, tuple(values),
            self._last.sent_action, self._last.pid_peaks, self._last.actor_version,
            self._last.actor_sha256, self._last.sent_monotonic_s, queue_epoch)
        for pending in self._pending:
            pending.future_my.append(my.copy())
            n = len(pending.future_my)
            if n in (5, 10) and n not in pending.completed:
                self._record(pending, n, terminal)
                pending.completed.add(n)
        self.stats['terminal_observations'] += 1

    def finish_condition(self) -> dict[str, int]:
        dropped = len(self._pending)
        self._reset_segment()
        return {'dropped_pending_count': dropped, 'completed_record_count': len(self.records),
                **{name: int(count) for name, count in self.stats.items()}}

    def drain_records(self) -> list[dict[str, Any]]:
        rows = list(self.records)
        self.records.clear()
        return rows


class PhysicalLoadShadow:
    """Nonblocking bounded submission; the worker owns model state and I/O."""

    def __init__(self, artifacts: PhysicalShadowArtifacts, *, output_dir: Path,
                 origin_stride_frames: int = 5, queue_capacity: int = 256,
                 record_capacity: int = 16384, max_records: int = 250000,
                 max_record_bytes: int = 32768, max_total_bytes: int = 268435456) -> None:
        if any(v < 1 for v in (queue_capacity, max_records, max_record_bytes, max_total_bytes)):
            raise ValueError('Physical shadow budgets must be positive')
        self.artifacts = artifacts
        self.core = PhysicalShadowCore(artifacts, origin_stride_frames=origin_stride_frames,
                                       record_capacity=record_capacity)
        self.path = output_dir / 'physical_load_shadow_development.jsonl'
        if self.path.exists():
            raise FileExistsError('Refusing to overwrite physical shadow development records')
        self._queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        self._stats: Counter[str] = Counter()
        self._records_hasher = hashlib.sha256()
        self._condition_receipts: deque[dict[str, Any]] = deque(maxlen=1024)
        self._limits = (max_records, max_record_bytes, max_total_bytes)
        self._latest_sim_time = 0.0
        self._latest_submitted_key: tuple[int, int, int] | None = None
        self._latest_by_key: OrderedDict[tuple[int, int, int], float] = OrderedDict()
        self._latest_diagnostics: dict[str, float] = {}
        self._submission_invalidated = False
        self._disabled = False
        self._stop = threading.Event()
        self._closed = False
        self._error_type = ''
        self.evaluation_reference: PairedPidReference | None = None
        self.evaluation_reference_file_sha256 = ''
        self._thread = threading.Thread(target=self._run, name='physical-load-shadow', daemon=True)
        self._thread.start()

    def _put(self, item: tuple) -> bool:
        if self._disabled or self._closed:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._stats['queue_full_dropped'] += 1
            self._submission_invalidated = True
            return False

    def reset_condition(self, condition_id: int, instance_id: int) -> None:
        self._latest_submitted_key = (int(condition_id), int(instance_id), -1)
        self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
        self._put(('reset', int(condition_id), int(instance_id)))

    def submit_sent(self, *, condition_id: int, condition_instance_id: int,
                    payload: Mapping[str, float], sent_action_real: Sequence[float],
                    pid_peak_abs_nm: Sequence[float], actor_policy_version: int,
                    actor_policy_sha256: str, sent_monotonic_s: float,
                    frame_id: int = -1, queue_epoch: int = -1) -> dict[str, float]:
        try:
            action = np.asarray(sent_action_real, dtype=np.float64)
            reference = np.asarray(pid_peak_abs_nm, dtype=np.float64)
            high = np.asarray([self.core.base['static_parameters']['request_torque_max_nm'], 0.43])
            if (action.shape != (2,) or reference.shape != (3,) or not np.isfinite(action).all() or
                    np.any(action < 0.0) or np.any(action > high) or
                    not np.isfinite(reference).all() or np.any(reference <= 0.0) or actor_policy_version < 0):
                raise ValueError('Invalid physical shadow action/reference/version')
            # UDP O_* fields are measurement aliases, never demand provenance.
            row = [float(payload[name]) if name not in ('O_PcsTorqueDemand', 'O_PitchPositionDemand1')
                   else float(action[0 if name == 'O_PcsTorqueDemand' else 1]) for name in INPUT_FIELDS]
            if not np.isfinite(row).all() or not math.isfinite(sent_monotonic_s):
                raise ValueError('Invalid physical shadow input')
            sha = str(actor_policy_sha256)
            if sha and (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
                raise ValueError('Invalid physical shadow actor SHA')
            frame = SentPhysicalFrame(int(condition_id), int(condition_instance_id), int(frame_id),
                tuple(row), tuple(action.tolist()), tuple(reference.tolist()), int(actor_policy_version),
                sha, float(sent_monotonic_s), int(queue_epoch))
            self._latest_sim_time = frame.simulation_time_s
            key = (frame.condition_id, frame.condition_instance_id, frame.queue_epoch)
            self._latest_submitted_key = key
            self._latest_by_key[key] = frame.simulation_time_s
            if len(self._latest_by_key) > 32:
                self._latest_by_key.popitem(last=False)
            invalidated = self._submission_invalidated
            if self._put(('frame', frame, invalidated)):
                self._submission_invalidated = False
                self._stats['submitted_frames'] += 1
        except Exception as exc:
            self._error_type = type(exc).__name__
            self.submission_failed()
        return self.diagnostics

    def confirm_outcome(self, timestamp_s: float, tag: str) -> None:
        self._put(('outcome', float(timestamp_s), str(tag)))

    def submit_terminal(self, *, payload: Mapping[str, float], condition_id: int,
                        condition_instance_id: int, frame_id: int = -1, queue_epoch: int = -1) -> None:
        self._put(('terminal', dict(payload), int(condition_id), int(condition_instance_id), int(frame_id),
                   int(queue_epoch), self._submission_invalidated))

    def note_off_grid_terminal_marker(self) -> None:
        """Record an END delimiter outside the fixed-step reference grid without using it as a sample."""
        self._put(('off_grid_terminal_marker',))

    def submission_failed(self) -> None:
        self._stats['invalid_submissions'] += 1
        self._submission_invalidated = True
        self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
        self._put(('invalidate', 'invalid_submission'))

    def finish_condition(self, metadata: dict[str, Any]) -> None:
        self._put(('finish', dict(metadata)))

    @property
    def diagnostics(self) -> dict[str, float]:
        latest = dict(self._latest_diagnostics)
        published_key = (latest.get('physical_load_shadow_result_condition_id'),
                         latest.get('physical_load_shadow_result_instance_id'),
                         latest.get('physical_load_shadow_result_queue_epoch'))
        if published_key != self._latest_submitted_key or self._submission_invalidated or self._disabled:
            latest = {'physical_load_shadow_result_available': 0.0}
        if 'physical_load_shadow_result_origin_time_s' in latest:
            latest['physical_load_shadow_result_age_sim_s'] = max(
                0.0, self._latest_sim_time - latest['physical_load_shadow_result_origin_time_s'])
        return {'physical_load_shadow_active': float(not self._disabled and not self._closed),
                'physical_load_shadow_action_changed': 0.0,
                'physical_load_shadow_queue_depth': float(self._queue.qsize()),
                'physical_load_shadow_queue_dropped': float(self._stats['queue_full_dropped']),
                'physical_load_shadow_invalid_submissions': float(self._stats['invalid_submissions']),
                'physical_load_shadow_faulted': float(self._disabled),
                'physical_load_shadow_rl_margin_authorized': 0.0,
                **latest}

    def _write_records(self, metadata: dict[str, Any]) -> None:
        maximum_rows, maximum_record, maximum_bytes = self._limits
        rows = self.core.drain_records()
        for row in rows:
            same_slot = (metadata.get('condition_id') == row['condition_id'] and
                         metadata.get('condition_instance_id') == row['condition_instance_id'] and
                         metadata.get('queue_epoch', -1) == row['queue_epoch'])
            identity_status = metadata.get('identity_status', 'unconfirmed') if same_slot else 'unconfirmed_slot_metadata'
            reference_metadata = physical_condition_metadata(self, condition_id=row['condition_id'],
                condition_instance_id=row['condition_instance_id'], queue_epoch=row['queue_epoch'],
                identity_status=identity_status, identity_summary=metadata if same_slot else None)
            protected = set(row)
            row.update({key: value for key, value in reference_metadata.items() if key not in protected})
            row['identity_status'] = identity_status
            # End-of-condition identity cannot retrospectively authorize control.
            row['calibration_eligible'] = False
            row['independent_audit_eligible'] = False
            row['authorization'] = dict(RUNTIME_AUTHORIZATION)
            row['development_point_eligible'] = bool(row['development_point_eligible'] and
                identity_status == 'strict_case_matched' and
                reference_metadata.get('strict_physical_trace_complete') is True)
            encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
            if (len(encoded) > maximum_record or self._stats['records_written'] >= maximum_rows or
                    self._stats['bytes_written'] + len(encoded) > maximum_bytes):
                self._stats['write_budget_dropped'] += 1
                continue
            with self.path.open('ab') as handle:
                handle.write(encoded)
            self._records_hasher.update(encoded)
            self._stats['records_written'] += 1
            self._stats['bytes_written'] += len(encoded)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                kind = item[0]
                if kind == 'reset':
                    if self.core.records:
                        self._write_records({'identity_status': 'unconfirmed_at_reset'})
                    self.core.reset_condition(item[1], item[2])
                    self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
                elif kind == 'frame':
                    new_slot = self.core._key != (item[1].condition_id, item[1].condition_instance_id, item[1].queue_epoch)
                    if new_slot and self.core.records:
                        self._write_records({'identity_status': 'unconfirmed_missing_slot_finish'})
                    if item[2]:
                        self.core.invalidate('submission_queue_drop')
                    frame = item[1]
                    prediction = self.core.observe(frame, require_outcome=True,
                        latest_sim_time=lambda: self._latest_by_key.get(
                            (frame.condition_id, frame.condition_instance_id, frame.queue_epoch)))
                    if self.core.history_count == 1:
                        self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
                    if prediction is not None:
                        self._latest_diagnostics = {
                            'physical_load_shadow_result_available': 1.0,
                            'physical_load_shadow_result_origin_time_s': frame.simulation_time_s,
                            'physical_load_shadow_result_condition_id': float(frame.condition_id),
                            'physical_load_shadow_result_instance_id': float(frame.condition_instance_id),
                            'physical_load_shadow_result_queue_epoch': float(frame.queue_epoch),
                            'physical_load_shadow_result_age_known': float(prediction['latest_sim_time'] is not None),
                            'physical_load_shadow_prediction_ms': prediction['compute_ms'],
                            'physical_load_shadow_short_point_supported': float(prediction['supported'].all()),
                            'physical_load_shadow_pid_margin_illustrative_available': float(prediction['margin_known']),
                        }
                        for horizon_index, suffix in ((0, 'h100'), (1, 'h200')):
                            supported = bool(prediction['supported'][horizon_index])
                            self._latest_diagnostics[
                                f'physical_load_shadow_prediction_supported_{suffix}'
                            ] = float(supported)
                            if not supported:
                                continue
                            for blade_index in range(3):
                                predicted_peak = float(prediction['peaks'][horizon_index, blade_index])
                                if math.isfinite(predicted_peak):
                                    self._latest_diagnostics[
                                        f'physical_load_shadow_predicted_peak_nm_{suffix}_blade_{blade_index + 1}'
                                    ] = predicted_peak
                        if prediction['latest_sim_time'] is not None:
                            self._latest_diagnostics['physical_load_shadow_result_age_sim_s'] = max(
                                0.0, prediction['latest_sim_time'] - frame.simulation_time_s)
                elif kind == 'outcome':
                    self.core.confirm_outcome(item[1], item[2])
                    if item[2] != OUTCOME_OK or not self.core.history_count:
                        self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
                elif kind == 'terminal':
                    if item[6]:
                        self.core.invalidate('submission_queue_drop_or_invalid_terminal')
                    else:
                        self.core.observe_terminal(payload=item[1], condition_id=item[2],
                            condition_instance_id=item[3], frame_id=item[4], queue_epoch=item[5])
                elif kind == 'off_grid_terminal_marker':
                    self._stats['off_grid_terminal_markers_ignored'] += 1
                elif kind == 'invalidate':
                    self.core.invalidate(item[1])
                    self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
                elif kind == 'finish':
                    claimed_key = (int(item[1].get('condition_id', -1)),
                                   int(item[1].get('condition_instance_id', -1)),
                                   int(item[1].get('queue_epoch', -1)))
                    actual_key = self.core._key
                    receipt_key = actual_key if actual_key is not None else claimed_key
                    duplicate = any((receipt['condition_id'], receipt['condition_instance_id'],
                                     receipt['queue_epoch']) == receipt_key
                                    for receipt in self._condition_receipts)
                    slot_matches = actual_key == claimed_key and self.core._last is not None and not duplicate
                    self._stats['finish_slot_identity_mismatch'] += int(not slot_matches)
                    self._stats['duplicate_slot_finish'] += int(duplicate)
                    identity_status = item[1].get('identity_status', 'unconfirmed') if slot_matches else (
                        'unconfirmed_duplicate_slot_finish' if duplicate else 'unconfirmed_finish_slot_mismatch')
                    checked_metadata = physical_condition_metadata(self,
                        condition_id=receipt_key[0], condition_instance_id=receipt_key[1],
                        queue_epoch=receipt_key[2], identity_status=identity_status,
                        identity_summary=item[1] if slot_matches else None)
                    self.core.finish_condition()
                    self._write_records(checked_metadata)
                    # Consume the actual slot. Repeated metadata is not a new
                    # observation and cannot create a second complete receipt.
                    self.core._key = None
                    self._stats['finished_conditions'] += 1
                    self._stats['strict_complete_conditions'] += int(
                        checked_metadata.get('strict_physical_trace_complete') is True)
                    if len(self._condition_receipts) == self._condition_receipts.maxlen:
                        self._stats['condition_receipt_capacity_dropped'] += 1
                    else:
                        receipt = {name: checked_metadata.get(name) for name in
                            ('condition_id', 'condition_instance_id', 'queue_epoch', 'identity_status',
                             'strict_physical_trace_complete', 'evaluation_reference_file_sha256',
                             'scenario_id', 'pid_raw_sha256', 'time_window_start_s', 'time_window_end_s',
                             'expected_trace_signature', 'expected_trace_sample_count',
                             'observed_trace_signature', 'observed_trace_sample_count',
                             'fixed_simulation_time_offset_s', 'trace_signature_match',
                             'wind_mismatch_policy', 'queue_case_identity_wind_mismatch_count',
                             'queue_case_identity_wind_max_abs_error_mps',
                             'queue_case_identity_development_unverified')}
                        receipt['finish_slot_identity_match'] = bool(slot_matches)
                        self._condition_receipts.append(receipt)
                    self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
                elif kind == 'close':
                    self.core.finish_condition()
                    self._write_records({'identity_status': 'unconfirmed_at_shutdown'})
            except Exception as exc:
                self._disabled = True
                self._error_type = type(exc).__name__
                self._stats['worker_faults'] += 1
                self.core.invalidate('worker_fault')
                self._latest_diagnostics = {'physical_load_shadow_result_available': 0.0}
            finally:
                self._queue.task_done()
            if self._disabled:
                break
        if not self._disabled:
            try:
                self.core.finish_condition()
                self._write_records({'identity_status': 'unconfirmed_at_shutdown'})
            except Exception as exc:
                self._disabled = True
                self._error_type = type(exc).__name__
                self._stats['worker_faults'] += 1

    def close(self, maximum_wait_s: float = 2.0) -> dict[str, Any]:
        if not self._closed:
            self._put(('close',))
            self._closed = True
            self._stop.set()
        self._thread.join(timeout=maximum_wait_s)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {'backend': BACKEND, 'authorization': dict(RUNTIME_AUTHORIZATION),
            'run_id': self.core.run_id, 'evaluation_reference_file_sha256': self.evaluation_reference_file_sha256,
                'bundle_sha256': self.artifacts.bundle_file_sha256,
                'dynamic_model_file_sha256': self.artifacts.dynamic_file_sha256,
                'records_path': str(self.path), 'worker_alive': self._thread.is_alive(),
                'finalized': self._closed and not self._thread.is_alive(),
                'records_file_sha256': self._records_hasher.hexdigest()
                    if self._closed and not self._thread.is_alive() else None,
                'summary_status': 'NOT_FINAL_WORKER_STILL_ALIVE' if self._thread.is_alive() else 'FINALIZED',
                'faulted': self._disabled, 'last_error_type': self._error_type,
                'counters': {key: int(value) for key, value in self._stats.copy().items()},
                'core_counters': {key: int(value) for key, value in self.core.stats.copy().items()},
                'condition_receipts': list(self._condition_receipts),
                'counterfactual_validated': False, 'independent_acceptance': False}


def physical_backend_selected(config: dict[str, Any]) -> bool:
    predictor = config.get('load_predictor')
    if predictor is not None and not isinstance(predictor, dict):
        raise ValueError('load_predictor must be an object')
    backend = (predictor or {}).get('backend', 'nn_candidate_v7')
    if backend not in ('nn_candidate_v7', BACKEND):
        raise ValueError('Unsupported load predictor backend')
    return backend == BACKEND


def preflight_physical_config(config: dict[str, Any], *, project_root: Path) -> tuple[PhysicalShadowArtifacts, PairedPidReference] | None:
    if not physical_backend_selected(config):
        return None
    settings = config['load_predictor']
    if (settings.get('mode') != 'shadow' or settings.get('authorization') != RUNTIME_AUTHORIZATION or
            settings.get('nn_active') is not False):
        raise ValueError('Physical backend requires development shadow only and NN inactive')
    env = config['env']
    if not env.get('queue_lifecycle_enabled') or env.get('action_limiter_owner') != 'controller':
        raise ValueError('Physical shadow requires queue identity and controller-owned final demand')
    if config.get('bc', {}).get('export_checkpoint_path'):
        raise ValueError('Physical development runs must not export/overwrite frozen BC')
    for key, default in (('origin_stride_frames', 5), ('worker_queue_capacity', 256),
                         ('condition_record_capacity', 16384), ('max_records', 250000),
                         ('max_record_bytes', 32768), ('max_total_bytes', 268435456)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Physical shadow {key} must be a positive integer')
    artifacts = load_physical_shadow_artifacts(
        artifact_path(settings['bundle_path'], project_root),
        expected_sha256=settings['bundle_sha256'], project_root=project_root)
    if not np.array_equal(env['action_low'], [0.0, 0.0]) or not np.array_equal(
            env['action_high'], [artifacts.model.physics.protocol['static_parameters']['request_torque_max_nm'], 0.43]):
        raise ValueError('Physical shadow engineering demand bounds mismatch')
    if float(env['bladed_step_duration_ms']) / 1000.0 != artifacts.model.physics.protocol['step_duration_s']:
        raise ValueError('Physical shadow/environment step mismatch')
    reference_entry = settings['evaluation_reference']
    reference_path = artifact_path(reference_entry['path'], project_root)
    if file_sha256(reference_path) != reference_entry['file_sha256']:
        raise ValueError('Physical shadow evaluation reference SHA mismatch')
    reference = PairedPidReference.from_path(reference_path, project_root=project_root)
    identity = config['queue'].get('case_identity_validation', {})
    if (identity.get('enabled') is not True or identity.get('validation_level') != 'strict' or
            float(identity.get('fixed_simulation_time_offset_s', float('nan'))) != 0.0 or
            artifact_path(identity['paired_pid_reference_path'], project_root) != reference_path):
        raise ValueError('Physical shadow strict identity must use the same zero-offset exact evaluation reference')
    for cid in config['queue']['condition_ids']:
        entry = reference.entry(cid)
        if entry.time_window_start_s != 0.0 or entry.time_window_end_s != config['queue']['train_sim_s']:
            raise ValueError('Physical shadow requires the exact evaluation window reference')
    return artifacts, reference


def build_physical_shadow(config: dict[str, Any], env: Any, *, output_dir: Path,
                          project_root: Path) -> PhysicalLoadShadow | None:
    preflight = preflight_physical_config(config, project_root=project_root)
    if preflight is None:
        return None
    artifacts, reference = preflight
    settings = config['load_predictor']
    if not env.queue_lifecycle_enabled or env.action_limiter_owner != 'controller':
        raise ValueError('Physical shadow environment final-sent contract mismatch')
    shadow = PhysicalLoadShadow(artifacts, output_dir=output_dir,
        origin_stride_frames=int(settings.get('origin_stride_frames', 5)),
        queue_capacity=int(settings.get('worker_queue_capacity', 256)),
        record_capacity=int(settings.get('condition_record_capacity', 16384)),
        max_records=int(settings.get('max_records', 250000)),
        max_record_bytes=int(settings.get('max_record_bytes', 32768)),
        max_total_bytes=int(settings.get('max_total_bytes', 268435456)))
    shadow.evaluation_reference = reference
    shadow.evaluation_reference_file_sha256 = config['load_predictor']['evaluation_reference']['file_sha256']
    return shadow


def physical_condition_metadata(shadow: PhysicalLoadShadow, *, condition_id: int,
                                condition_instance_id: int, identity_status: str,
                                identity_summary: Mapping[str, Any] | None = None,
                                queue_epoch: int = -1) -> dict[str, Any]:
    """All physical record references come from physical evaluation, never reward."""
    summary = dict(identity_summary or {})
    metadata = {**summary, 'condition_id': int(condition_id),
                'condition_instance_id': int(condition_instance_id), 'identity_status': identity_status,
                'queue_epoch': int(queue_epoch),
                'evaluation_reference_file_sha256': shadow.evaluation_reference_file_sha256,
                'strict_physical_trace_complete': False}
    if shadow.evaluation_reference is None:
        metadata['identity_status'] = 'unconfirmed_missing_physical_reference'
        return metadata
    try:
        entry = shadow.evaluation_reference.entry(condition_id)
        metadata.update({
            'scenario_id': entry.scenario_id, 'pid_raw_csv_path': str(entry.pid_raw_csv_path),
            'pid_raw_sha256': entry.pid_raw_sha256, 'time_window_start_s': entry.time_window_start_s,
            'time_window_end_s': entry.time_window_end_s, 'expected_trace_signature': entry.trace_signature,
            'expected_trace_sample_count': int(entry.timestamps_s.size),
        })
        metadata['strict_physical_trace_complete'] = bool(identity_status == 'strict_case_matched' and
            summary.get('observed_trace_signature') == entry.trace_signature and
            summary.get('observed_trace_sample_count') == int(entry.timestamps_s.size) and
            summary.get('fixed_simulation_time_offset_s') == 0.0 and
            summary.get('trace_signature_match', 0.0) > 0.5)
    except (AttributeError, KeyError, TypeError, ValueError):
        metadata['identity_status'] = 'unconfirmed_physical_reference_identity'
    return metadata


def submit_physical_sent_frame(shadow: PhysicalLoadShadow, env: Any, *, payload: Mapping[str, float],
                               condition_id: int, condition_instance_id: int,
                               sent_action_real: Sequence[float], actor_policy_version: int,
                               actor_policy_sha256: str, sent_monotonic_s: float,
                               frame_id: int = -1) -> None:
    """The send callback must never fail because of the optional shadow."""
    try:
        snapshot = env.last_successful_action_send
        if (snapshot['condition_id'] != condition_id or
                snapshot['condition_instance_id'] != condition_instance_id):
            raise ValueError('Physical shadow successfully-sent packet identity mismatch')
        measured_fields = set(INPUT_FIELDS) - {'O_PcsTorqueDemand', 'O_PitchPositionDemand1'}
        if measured_fields.intersection(snapshot.get('invalid_payload_fields', ())):
            raise ValueError('Physical shadow cannot accept zero-filled invalid raw measurements')
        shadow.submit_sent(condition_id=condition_id, condition_instance_id=condition_instance_id,
            payload=payload, sent_action_real=snapshot['action_real'],
            pid_peak_abs_nm=shadow.evaluation_reference.blade_root_peak(condition_id)
                if shadow.evaluation_reference is not None else env.reward_fn.pid_blade_root_peak_reference(condition_id),
            actor_policy_version=actor_policy_version, actor_policy_sha256=actor_policy_sha256,
            sent_monotonic_s=sent_monotonic_s, frame_id=snapshot['frame_id'],
            queue_epoch=int(snapshot.get('queue_epoch', -1)))
    except Exception:
        shadow.submission_failed()

def complete_physical_step(shadow: PhysicalLoadShadow, env: Any, *, payload: Mapping[str, float],
                           info: Mapping[str, float], unsafe_stop: bool) -> None:
    try:
        tag = physical_outcome_tag(info, unsafe_stop=unsafe_stop)
        measured_fields = set(INPUT_FIELDS) - {'O_PcsTorqueDemand', 'O_PitchPositionDemand1'}
        if measured_fields.intersection(getattr(env, 'current_payload_invalid_fields', ())):
            tag = 'raw_input_invalid'
        same_terminal = (tag == 'condition_boundary' and
            info.get('udp_next_phase_code') == 3.0 and
            info.get('udp_queue_epoch') == info.get('udp_next_queue_epoch') and
            info.get('udp_condition_id') == info.get('udp_next_condition_id') and
            info.get('udp_condition_instance_id') == info.get('udp_next_condition_instance_id'))
        shadow.confirm_outcome(float(payload['Time']), OUTCOME_OK if same_terminal else tag)
        if same_terminal:
            terminal_payload = env.current_payload
            terminal_time = float(terminal_payload['Time'])
            sent_frame_time = float(payload['Time'])
            reference = shadow.evaluation_reference
            reference_endpoint = None
            if reference is not None:
                entry = reference.entry(int(info['udp_condition_id']))
                reference_endpoint = float(entry.timestamps_s[-1])
            if (reference_endpoint is not None and
                    math.isclose(sent_frame_time, reference_endpoint, rel_tol=0.0, abs_tol=1e-6) and
                    terminal_time > reference_endpoint + 1e-6):
                # The END packet is a queue delimiter after the final reference
                # sample, not another dt-spaced physical observation. Counting a
                # half-step marker as My would either invalidate every slot or
                # label an H-step window with the wrong elapsed duration.
                shadow.note_off_grid_terminal_marker()
            else:
                shadow.submit_terminal(payload=terminal_payload,
                    condition_id=int(info['udp_condition_id']),
                    condition_instance_id=int(info['udp_condition_instance_id']),
                    frame_id=int(info.get('udp_next_frame_id', -1)),
                    queue_epoch=int(info.get('udp_next_queue_epoch', -1)))
    except Exception:
        shadow.submission_failed()


def physical_outcome_tag(info: Mapping[str, float], *, unsafe_stop: bool = False) -> str:
    if unsafe_stop:
        return 'unsafe_stop'
    if info.get('env_timeout', 0.0) > 0.0 or info.get('udp_inter_frame_timeout', 0.0) > 0.0:
        return 'timeout'
    if info.get('env_invalid_prediction', 0.0) > 0.0 or info.get('env_invalid_reward', 0.0) > 0.0:
        return 'invalid'
    if info.get('condition_boundary', 0.0) > 0.0:
        return 'condition_boundary'
    if info.get('udp_frame_lag', 0.0) > 0.0:
        return 'frame_gap'
    return OUTCOME_OK