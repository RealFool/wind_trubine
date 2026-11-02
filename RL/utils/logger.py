from __future__ import annotations

from collections import deque
import csv
import math
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from RL.utils.constants import BLADED_UDP_VARIABLES


COMPACT_LOGGING_SCHEMA_VERSION = 'compact_streaming_logger_v1'
_EVENT_METRIC_FIELDS = (
    'wind_t',
    'power',
    'power_state_real',
    'action_torque',
    'action_pitch',
    'control_pre_send_execution_mode',
    'control_residual_admitted',
    'control_ood_detected',
    'control_ood_score',
    'control_low_speed_stage',
    'control_optimization_state_code',
    'control_region_code',
    'control_baseline_action_torque',
    'control_baseline_action_pitch',
    'control_policy_delta_torque',
    'control_policy_delta_pitch',
    'control_projected_action_torque',
    'control_projected_action_pitch',
    'control_sent_action_torque',
    'control_sent_action_pitch',
    'control_post_step_timeout',
    'control_post_step_invalid',
    'control_post_step_unsafe_stop',
    'control_post_step_condition_boundary',
    'control_actor_update_gate_open',
    'queue_condition_id',
    'queue_condition_instance_id',
    'queue_condition_round_index',
    'actor_update_gate_open_before_critic',
    'actor_update_gate_open_after_critic',
    'actor_optimizer_health_gate_open',
    'actor_q_health_gate_open_after_update',
    'dual_constraint_condition_complete',
    'queue_condition_batch_count',
    'queue_condition_batch_sample_count',
    'queue_condition_replay_count',
    'queue_condition_event_window_count',
    'queue_condition_event_window_dropped_count',
    'queue_condition_event_window_incomplete_count',
)
_EVENT_STATE_FIELDS = (
    'control_pre_send_execution_mode',
    'control_low_speed_stage',
    'control_actor_update_gate_open',
    'actor_optimizer_health_gate_open',
    'actor_q_health_gate_open_after_update',
    'queue_condition_id',
    'queue_condition_instance_id',
    'queue_condition_round_index',
    'dual_constraint_condition_complete',
    'queue_condition_event_window_count',
    'queue_condition_event_window_dropped_count',
    'queue_condition_event_window_incomplete_count',
)
_QUEUE_CONDITION_SUMMARY_STAGES = frozenset({
    'queue_condition',
    'queue_condition_skip',
    'queue_condition_retry',
})
_QUEUE_STEP_SNAPSHOT_FIELDS = (
    'queue_condition_round_index',
    'queue_condition_id',
    'queue_condition_instance_id',
    'queue_epoch',
    'udp_frame_id',
    'udp_next_frame_id',
    'udp_sim_time',
    'udp_next_sim_time',
    'udp_phase_elapsed_s',
    'wind_t',
    'wind_next',
    'power',
    'power_t_metric_raw',
    'power_metric_raw',
    'power_reward_metric',
    'power_state_real',
    'action_torque',
    'action_pitch',
    'action_torque_delta',
    'action_pitch_delta',
    'reward_acc_raw',
    'reward_shaft_raw',
    'reward_relative_power_raw',
    'reward_relative_energy_increment_j',
    'reward_persistent_power_deficit_ewma',
    'control_pre_send_execution_mode',
    'control_optimization_state_code',
    'control_region_code',
    'control_low_speed_stage',
    'control_ood_detected',
    'control_ood_score',
    'control_baseline_action_torque',
    'control_baseline_action_pitch',
    'control_policy_delta_torque',
    'control_policy_delta_pitch',
    'control_residual_torque',
    'control_residual_pitch',
    'control_pre_safety_action_torque',
    'control_pre_safety_action_pitch',
    'control_projected_action_torque',
    'control_projected_action_pitch',
    'control_sent_action_torque',
    'control_sent_action_pitch',
    'actor_policy_version',
    'control_action_projection_applied',
    'control_action_rate_limited',
    'actor_update_gate_open_before_critic',
    'actor_update_gate_open_after_critic',
    'actor_optimizer_health_gate_open',
    'actor_q_health_gate_open_after_update',
    'control_post_step_timeout',
    'control_post_step_invalid',
    'control_post_step_unsafe_stop',
    'control_post_step_condition_boundary',
    'reward_blade_root_1_raw',
    'reward_blade_root_2_raw',
    'reward_blade_root_3_raw',
    'reward_blade_root_1_peak_reference_nm',
    'reward_blade_root_2_peak_reference_nm',
    'reward_blade_root_3_peak_reference_nm',
    'queue_event_sample_valid',
    'physical_load_shadow_prediction_origin_match',
    'physical_load_shadow_result_available',
    'physical_load_shadow_result_origin_time_s',
    'physical_load_shadow_prediction_supported_h100',
    'physical_load_shadow_prediction_supported_h200',
    'physical_load_shadow_predicted_peak_nm_h100_blade_1',
    'physical_load_shadow_predicted_peak_nm_h100_blade_2',
    'physical_load_shadow_predicted_peak_nm_h100_blade_3',
    'physical_load_shadow_predicted_peak_nm_h200_blade_1',
    'physical_load_shadow_predicted_peak_nm_h200_blade_2',
    'physical_load_shadow_predicted_peak_nm_h200_blade_3',
)


class _BoundedMetricSketch:
    def __init__(self, sample_capacity: int, seed: int) -> None:
        self.sample_capacity = sample_capacity
        self.samples: list[float] = []
        self.random = random.Random(seed)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.samples) < self.sample_capacity:
            self.samples.append(value)
            return
        replacement_index = self.random.randrange(self.count)
        if replacement_index < self.sample_capacity:
            self.samples[replacement_index] = value

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {'count': 0.0, 'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0}
        ordered = sorted(self.samples)
        return {
            'count': float(self.count),
            'mean': self.total / self.count,
            'p50': _quantile(ordered, 0.50),
            'p95': _quantile(ordered, 0.95),
            'min': self.minimum,
            'max': self.maximum,
        }


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


class ExperimentLogger:
    def __init__(self, output_dir: str | Path, logging_config: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.figure_dir = self.output_dir / 'figures'
        self.figure_dir.mkdir(exist_ok=True)
        self.logging_config = self._logging_config(logging_config)
        self.compact_streaming = self.logging_config['mode'] == 'compact_streaming'
        self._metrics_rows: list[dict[str, Any]] = []
        self._episode_rows: list[dict[str, Any]] | deque[dict[str, Any]] = (
            deque(maxlen=self.logging_config['episode_ring_capacity'])
            if self.compact_streaming
            else []
        )
        self._metric_sketches: dict[tuple[str, str], _BoundedMetricSketch] = {}
        self._event_ring: deque[dict[str, Any]] = deque(
            maxlen=self.logging_config['event_ring_capacity']
        )
        self._queue_condition_ring: deque[dict[str, Any]] = deque(
            maxlen=self.logging_config['queue_condition_ring_capacity']
        )
        self._queue_condition_rows_seen = 0
        self._queue_condition_rows_dropped = 0
        self._queue_condition_max_record_bytes_observed = 0
        self._queue_condition_bytes_written = 0
        self._queue_event_history: deque[dict[str, Any]] = deque(
            maxlen=self.logging_config['queue_event_pre_frames'] + 1
        )
        self._queue_event_slot: tuple[int, int, int] | None = None
        self._queue_event_candidates: dict[str, dict[str, Any]] = {}
        self._queue_event_peak_by_blade = [0.0, 0.0, 0.0]
        self._queue_event_first_exceed_by_blade = [False, False, False]
        self._queue_event_prediction_first_exceed: dict[str, bool] = {}
        self._queue_event_previous_mode: float | None = None
        self._queue_event_previous_low_speed_stage: float | None = None
        self._queue_event_previous_projection = False
        self._queue_event_previous_ood = False
        self._queue_event_previous_power_deficit = False
        self._queue_event_windows_seen = 0
        self._queue_event_windows_written = 0
        self._queue_event_windows_dropped = 0
        self._queue_event_bytes_written = 0
        self._queue_event_max_record_bytes_observed = 0
        self._previous_event_state: dict[str, tuple[float | None, ...]] = {}
        self._debug_rows_written = 0
        self._debug_bytes_written = 0
        self._debug_rows_dropped = 0

    @staticmethod
    def _logging_config(logging_config: dict[str, Any] | None) -> dict[str, Any]:
        if logging_config is None:
            return {
                'mode': 'legacy_long_form',
                'quantile_sample_capacity': 256,
                'event_ring_capacity': 128,
                'episode_ring_capacity': 256,
                'queue_condition_ring_capacity': 512,
                'queue_condition_max_record_bytes': 512 * 1024,
                'queue_condition_summaries_max_total_bytes': 256 * 1024 * 1024,
                'queue_event_pre_frames': 25,
                'queue_event_post_frames': 25,
                'queue_event_window_capacity': 16,
                'queue_event_window_max_record_bytes': 256 * 1024,
                'queue_event_windows_max_total_bytes': 512 * 1024 * 1024,
                'queue_event_full_state_enabled': False,
                'debug_long_form_enabled': True,
                'debug_max_rows': 0,
                'debug_max_bytes': 0,
            }
        if not isinstance(logging_config, dict):
            raise ValueError('logging must be an object')
        mode = str(logging_config.get('mode', '')).strip().lower()
        if mode not in {'legacy_long_form', 'compact_streaming'}:
            raise ValueError('logging.mode must be legacy_long_form or compact_streaming')
        config = {
            'mode': mode,
            'quantile_sample_capacity': int(logging_config.get('quantile_sample_capacity', 1024)),
            'event_ring_capacity': int(logging_config.get('event_ring_capacity', 256)),
            'episode_ring_capacity': int(logging_config.get('episode_ring_capacity', 256)),
            'queue_condition_ring_capacity': int(logging_config.get('queue_condition_ring_capacity', 512)),
            'queue_condition_max_record_bytes': int(
                logging_config.get('queue_condition_max_record_bytes', 512 * 1024)
            ),
            'queue_condition_summaries_max_total_bytes': int(
                logging_config.get('queue_condition_summaries_max_total_bytes', 256 * 1024 * 1024)
            ),
            'queue_event_pre_frames': int(logging_config.get('queue_event_pre_frames', 25)),
            'queue_event_post_frames': int(logging_config.get('queue_event_post_frames', 25)),
            'queue_event_window_capacity': int(logging_config.get('queue_event_window_capacity', 16)),
            'queue_event_window_max_record_bytes': int(
                logging_config.get('queue_event_window_max_record_bytes', 256 * 1024)
            ),
            'queue_event_windows_max_total_bytes': int(
                logging_config.get('queue_event_windows_max_total_bytes', 512 * 1024 * 1024)
            ),
            'queue_event_full_state_enabled': bool(logging_config.get('queue_event_full_state_enabled', False)),
            'debug_long_form_enabled': bool(logging_config.get('debug_long_form_enabled', False)),
            'debug_max_rows': int(logging_config.get('debug_max_rows', 0)),
            'debug_max_bytes': int(logging_config.get('debug_max_bytes', 0)),
        }
        for name in (
            'quantile_sample_capacity',
            'event_ring_capacity',
            'episode_ring_capacity',
            'queue_condition_ring_capacity',
            'queue_condition_max_record_bytes',
            'queue_condition_summaries_max_total_bytes',
            'queue_event_pre_frames',
            'queue_event_post_frames',
            'queue_event_window_capacity',
            'queue_event_window_max_record_bytes',
            'queue_event_windows_max_total_bytes',
        ):
            if config[name] <= 0:
                raise ValueError(f'logging.{name} must be positive')
        if config['mode'] == 'compact_streaming' and config['debug_long_form_enabled']:
            if config['debug_max_rows'] <= 0 or config['debug_max_bytes'] <= 0:
                raise ValueError(
                    'compact streaming debug long-form logging requires positive debug_max_rows and debug_max_bytes'
                )
            if config['queue_event_full_state_enabled'] and config['mode'] != 'compact_streaming':
                raise ValueError('logging.queue_event_full_state_enabled requires compact_streaming mode')
        return config

    def save_config(self, config: dict[str, Any]) -> None:
        with (self.output_dir / 'config.json').open('w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    def log_metrics(
        self,
        stage: str,
        step: int,
        metrics: dict[str, Any],
        *,
        record_metadata: dict[str, Any] | None = None,
        event_frame: Mapping[str, Any] | None = None,
    ) -> None:
        if event_frame is not None and not self.logging_config['queue_event_full_state_enabled']:
            raise ValueError('event_frame requires logging.queue_event_full_state_enabled')
        finite_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            finite_metrics[key] = numeric
            if self.compact_streaming:
                self._metric_sketch(stage, key).add(numeric)
            if not self.compact_streaming or self.logging_config['debug_long_form_enabled']:
                self._append_metric_row(stage, step, key, numeric)
        if self.compact_streaming:
            if stage == 'queue_train_step' and 'queue_condition_id' in finite_metrics:
                self._observe_queue_event_step(step, finite_metrics, event_frame=event_frame)
            if stage in _QUEUE_CONDITION_SUMMARY_STAGES and any(
                field in finite_metrics
                for field in ('condition_id', 'queue_condition_id', 'queue_skip_condition_id', 'queue_retry_condition_id')
            ):
                record = dict(finite_metrics)
                record.update(self._finalize_queue_event_slot(finite_metrics))
                if record_metadata is not None:
                    record['_metadata'] = record_metadata
                self._record_queue_condition_summary(stage, step, record)
            self._record_event_if_changed(stage, step, finite_metrics)

    def _record_queue_condition_summary(
        self,
        stage: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        record: dict[str, Any] = {'stage': stage, 'step': int(step)}
        record.update(metrics)
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8')
        record_bytes = len(serialized)
        if record_bytes > self.logging_config['queue_condition_max_record_bytes']:
            raise ValueError(
                'Compact queue-condition summary exceeds logging.queue_condition_max_record_bytes: '
                f'{record_bytes} > {self.logging_config["queue_condition_max_record_bytes"]}'
            )
        if self._queue_condition_bytes_written + record_bytes + 1 > self.logging_config[
            'queue_condition_summaries_max_total_bytes'
        ]:
            raise ValueError('Compact queue-condition summaries exceed logging.queue_condition_summaries_max_total_bytes')
        path = self.output_dir / 'compact_queue_condition_summaries.jsonl'
        with path.open('a', encoding='utf-8') as file:
            file.write(serialized.decode('utf-8') + '\n')
        self._queue_condition_bytes_written += record_bytes + 1
        self._queue_condition_max_record_bytes_observed = max(
            self._queue_condition_max_record_bytes_observed,
            record_bytes,
        )
        self._queue_condition_rows_seen += 1
        if len(self._queue_condition_ring) >= self.logging_config['queue_condition_ring_capacity']:
            self._queue_condition_rows_dropped += 1
        self._queue_condition_ring.append(record)

    def _reset_queue_event_slot(self, slot: tuple[int, int, int]) -> None:
        self._queue_event_slot = slot
        self._queue_event_history.clear()
        self._queue_event_candidates.clear()
        self._queue_event_peak_by_blade = [0.0, 0.0, 0.0]
        self._queue_event_first_exceed_by_blade = [False, False, False]
        self._queue_event_prediction_first_exceed = {}
        self._queue_event_previous_mode = None
        self._queue_event_previous_low_speed_stage = None
        self._queue_event_previous_projection = False
        self._queue_event_previous_ood = False
        self._queue_event_previous_power_deficit = False
        self._queue_event_slot_windows_seen = 0
        self._queue_event_slot_windows_dropped = 0

    def _queue_event_snapshot(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        event_frame: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = {'step': float(step)}
        snapshot.update({
            field: metrics[field]
            for field in _QUEUE_STEP_SNAPSHOT_FIELDS
            if field in metrics and math.isfinite(metrics[field])
        })
        if event_frame is not None:
            snapshot.update(self._validated_queue_event_frame(event_frame))
        return snapshot

    def _validated_queue_event_frame(self, event_frame: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            'origin_payload', 'observed_payload', 'origin_invalid_fields',
            'observed_invalid_fields', 'observed_sample_valid', 'wire_action_real',
            'actor_policy_sha256',
        }
        if set(event_frame) - allowed:
            raise ValueError('Queue event frame contains fields outside its bounded audit schema')
        result: dict[str, Any] = {}
        for name in ('origin_payload', 'observed_payload'):
            values = event_frame.get(name)
            if values is None:
                result[name] = None
                continue
            if not isinstance(values, Mapping) or len(values) > 32:
                raise ValueError(f'{name} must be a bounded mapping of existing frame fields')
            copied: dict[str, float | None] = {}
            for key, value in values.items():
                if not isinstance(key, str) or len(key) > 80:
                    raise ValueError(f'{name} contains an invalid field name')
                if key not in BLADED_UDP_VARIABLES:
                    raise ValueError(f'{name} contains a field outside the existing Bladed UDP whitelist: {key}')
                if value is None:
                    copied[key] = None
                    continue
                numeric = float(value)
                copied[key] = numeric if math.isfinite(numeric) else None
            result[name] = copied
        for name in ('origin_invalid_fields', 'observed_invalid_fields'):
            values = event_frame.get(name, ())
            if not isinstance(values, (list, tuple)) or len(values) > 32:
                raise ValueError(f'{name} must be a bounded sequence')
            result[name] = [str(value)[:80] for value in values]
        result['observed_sample_valid'] = bool(event_frame.get('observed_sample_valid', False))
        wire_action = event_frame.get('wire_action_real')
        if wire_action is not None:
            values = tuple(float(value) for value in wire_action)
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError('wire_action_real must contain two finite values')
            result['wire_action_real'] = list(values)
        actor_sha = event_frame.get('actor_policy_sha256', '')
        if not isinstance(actor_sha, str) or (actor_sha and (
                len(actor_sha) != 64 or any(value not in '0123456789abcdef' for value in actor_sha.lower()))):
            raise ValueError('actor_policy_sha256 must be empty or a lowercase SHA-256')
        result['actor_policy_sha256'] = actor_sha.lower()
        encoded_size = len(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'))
        if encoded_size > 16 * 1024:
            raise ValueError('Queue event frame exceeds its 16 KiB snapshot budget')
        return result

    def _register_queue_event_candidate(
        self,
        *,
        candidate_key: str,
        event_type: str,
        score: float,
        priority: int,
        step: int,
        snapshot: dict[str, Any],
    ) -> None:
        existing = self._queue_event_candidates.get(candidate_key)
        if existing is not None:
            if candidate_key.startswith('blade_peak_'):
                if score <= float(existing['score']):
                    return
                pre_and_trigger = list(self._queue_event_history)
                pre_and_trigger.append(snapshot)
                max_pre = self.logging_config['queue_event_pre_frames']
                frames = pre_and_trigger[-(max_pre + 1):]
                self._queue_event_candidates[candidate_key] = {
                    'event_type': event_type,
                    'score': float(score),
                    'priority': int(priority),
                    'trigger_step': int(step),
                    'frames': frames,
                    'pre_frame_count': max(0, len(frames) - 1),
                    'post_frame_count': 0,
                }
                return
            self._queue_event_windows_seen += 1
            self._queue_event_slot_windows_seen += 1
            del self._queue_event_candidates[candidate_key]
            self._queue_event_slot_windows_dropped += 1
            self._queue_event_windows_dropped += 1
        else:
            self._queue_event_windows_seen += 1
            self._queue_event_slot_windows_seen += 1

        capacity = self.logging_config['queue_event_window_capacity']
        if len(self._queue_event_candidates) >= capacity:
            protected = lambda key: (
                key.startswith('blade_peak_')
                or key.startswith('blade_first_exceed_')
                or key.startswith('shadow_predicted_exceed_')
            )
            replaceable = [
                (key, value)
                for key, value in self._queue_event_candidates.items()
                if not protected(key)
            ]
            if not replaceable:
                self._queue_event_slot_windows_dropped += 1
                self._queue_event_windows_dropped += 1
                return
            worst_key, worst = min(
                replaceable,
                key=lambda item: (int(item[1]['priority']), float(item[1]['score']), -int(item[1]['trigger_step'])),
            )
            if priority <= int(worst['priority']):
                self._queue_event_slot_windows_dropped += 1
                self._queue_event_windows_dropped += 1
                return
            del self._queue_event_candidates[worst_key]
            self._queue_event_slot_windows_dropped += 1
            self._queue_event_windows_dropped += 1

        pre_and_trigger = list(self._queue_event_history)
        pre_and_trigger.append(snapshot)
        max_pre = self.logging_config['queue_event_pre_frames']
        frames = pre_and_trigger[-(max_pre + 1):]
        self._queue_event_candidates[candidate_key] = {
            'event_type': event_type,
            'score': float(score),
            'priority': int(priority),
            'trigger_step': int(step),
            'frames': frames,
            'pre_frame_count': max(0, len(frames) - 1),
            'post_frame_count': 0,
        }

    def _observe_queue_event_step(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        event_frame: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            round_index = int(metrics['queue_condition_round_index'])
            condition_id = int(metrics['queue_condition_id'])
            instance_id = int(metrics['queue_condition_instance_id'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        slot = (round_index, condition_id, instance_id)
        if slot != self._queue_event_slot:
            if self._queue_event_slot is not None:
                self._write_queue_event_candidates(self._queue_event_slot, termination_reason='slot_changed')
            self._reset_queue_event_slot(slot)

        snapshot = self._queue_event_snapshot(step, metrics, event_frame=event_frame)
        prediction_origin = metrics.get('physical_load_shadow_result_origin_time_s')
        action_origin = metrics.get('udp_sim_time')
        prediction_origin_match = bool(
            metrics.get('physical_load_shadow_result_available', 0.0) > 0.0
            and prediction_origin is not None
            and action_origin is not None
            and math.isclose(prediction_origin, action_origin, rel_tol=0.0, abs_tol=1.0e-6)
        )
        snapshot['physical_load_shadow_prediction_origin_match'] = float(prediction_origin_match)
        max_post = self.logging_config['queue_event_post_frames']
        for candidate in self._queue_event_candidates.values():
            if int(candidate['trigger_step']) == int(step):
                continue
            if int(candidate['post_frame_count']) >= max_post:
                continue
            candidate['frames'].append(snapshot)
            candidate['post_frame_count'] += 1

        current_mode = metrics.get('control_pre_send_execution_mode')
        if current_mode is not None:
            if self._queue_event_previous_mode is not None and current_mode != self._queue_event_previous_mode:
                self._register_queue_event_candidate(
                    candidate_key=f'execution_mode_transition:{step}',
                    event_type='execution_mode_transition',
                    score=abs(current_mode - self._queue_event_previous_mode),
                    priority=70,
                    step=step,
                    snapshot=snapshot,
                )
            self._queue_event_previous_mode = current_mode

        current_stage = metrics.get('control_low_speed_stage')
        if current_stage is not None:
            if (
                self._queue_event_previous_low_speed_stage is not None
                and current_stage != self._queue_event_previous_low_speed_stage
            ):
                self._register_queue_event_candidate(
                    candidate_key=f'low_speed_stage_transition:{step}',
                    event_type='low_speed_stage_transition',
                    score=abs(current_stage - self._queue_event_previous_low_speed_stage),
                    priority=65,
                    step=step,
                    snapshot=snapshot,
                )
            self._queue_event_previous_low_speed_stage = current_stage

        ood_active = metrics.get('control_ood_detected', 0.0) > 0.0
        if ood_active and not self._queue_event_previous_ood:
            self._register_queue_event_candidate(
                candidate_key=f'ood_entry:{step}',
                event_type='ood_entry',
                score=metrics.get('control_ood_score', 0.0),
                priority=60,
                step=step,
                snapshot=snapshot,
            )
        self._queue_event_previous_ood = ood_active

        power_reference = abs(metrics.get('reward_pid_power_reference_raw', 0.0))
        relative_power = metrics.get('reward_relative_power_raw', 0.0)
        deficit_ewma = metrics.get('reward_persistent_power_deficit_ewma', 0.0)
        deficit_active = (
            power_reference > 0.0
            and relative_power < -0.01 * power_reference
            and deficit_ewma > 0.01
        )
        if deficit_active and not self._queue_event_previous_power_deficit:
            self._register_queue_event_candidate(
                candidate_key=f'sustained_power_deficit:{step}',
                event_type='sustained_power_deficit_start',
                score=deficit_ewma,
                priority=55,
                step=step,
                snapshot=snapshot,
            )
        self._queue_event_previous_power_deficit = deficit_active

        projection_active = metrics.get('control_action_projection_applied', 0.0) > 0.0
        if projection_active and not self._queue_event_previous_projection:
            self._register_queue_event_candidate(
                candidate_key=f'action_projection:{step}',
                event_type='action_projection',
                score=metrics.get('control_ood_score', 0.0),
                priority=45,
                step=step,
                snapshot=snapshot,
            )
            self._queue_event_previous_projection = projection_active

        load_sample_valid = metrics.get('queue_event_sample_valid', 1.0) > 0.0
        if load_sample_valid:
            for blade_index in range(1, 4):
                try:
                    value = abs(float(metrics[f'reward_blade_root_{blade_index}_raw']))
                    reference = float(metrics[f'reward_blade_root_{blade_index}_peak_reference_nm'])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value) or not math.isfinite(reference) or reference <= 0.0:
                    continue
                if value > self._queue_event_peak_by_blade[blade_index - 1]:
                    self._queue_event_peak_by_blade[blade_index - 1] = value
                    self._register_queue_event_candidate(
                        candidate_key=f'blade_peak_{blade_index}',
                        event_type=f'blade_{blade_index}_new_max_abs_my',
                        score=value,
                        priority=100,
                        step=step,
                        snapshot=snapshot,
                    )
                if value > reference and not self._queue_event_first_exceed_by_blade[blade_index - 1]:
                    self._queue_event_first_exceed_by_blade[blade_index - 1] = True
                    self._register_queue_event_candidate(
                        candidate_key=f'blade_first_exceed_{blade_index}',
                        event_type=f'blade_{blade_index}_first_pid_peak_exceed',
                        score=value / reference,
                        priority=95,
                        step=step,
                        snapshot=snapshot,
                    )

        for _horizon, suffix in ((0.1, 'h100'), (0.2, 'h200')):
            if (
                not prediction_origin_match
                or metrics.get(f'physical_load_shadow_prediction_supported_{suffix}', 0.0) <= 0.0
            ):
                continue
            for blade_index in range(1, 4):
                try:
                    prediction = float(metrics[
                        f'physical_load_shadow_predicted_peak_nm_{suffix}_blade_{blade_index}'
                    ])
                    reference = float(metrics[f'reward_blade_root_{blade_index}_peak_reference_nm'])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(prediction) or not math.isfinite(reference) or reference <= 0.0:
                    continue
                candidate_key = f'shadow_predicted_exceed_{suffix}_blade_{blade_index}'
                if prediction > reference and not self._queue_event_prediction_first_exceed.get(candidate_key, False):
                    self._queue_event_prediction_first_exceed[candidate_key] = True
                    self._register_queue_event_candidate(
                        candidate_key=candidate_key,
                        event_type=f'physical_shadow_{suffix}_predicted_pid_peak_exceed_blade_{blade_index}',
                        score=prediction / reference,
                        priority=92,
                        step=step,
                        snapshot=snapshot,
                    )

        self._queue_event_history.append(snapshot)

    def _write_queue_event_candidates(
        self,
        slot: tuple[int, int, int],
        *,
        termination_reason: str,
    ) -> tuple[int, int]:
        written = 0
        incomplete = 0
        path = self.output_dir / 'compact_queue_condition_event_windows.jsonl'
        for candidate in self._queue_event_candidates.values():
            post_count = int(candidate['post_frame_count'])
            complete = post_count >= self.logging_config['queue_event_post_frames']
            record = {
                'schema_version': (
                    'queue_condition_event_window_v2'
                    if any('origin_payload' in frame for frame in candidate['frames'])
                    else 'queue_condition_event_window_v1'
                ),
                'round_index': slot[0],
                'condition_id': slot[1],
                'condition_instance_id': slot[2],
                'event_type': candidate['event_type'],
                'score': float(candidate['score']),
                'trigger_step': int(candidate['trigger_step']),
                'pre_frame_count': int(candidate['pre_frame_count']),
                'post_frame_count': post_count,
                'target_pre_frame_count': self.logging_config['queue_event_pre_frames'],
                'target_post_frame_count': self.logging_config['queue_event_post_frames'],
                'window_complete': complete,
                'termination_reason': termination_reason if not complete else 'post_window_complete',
                'frames': candidate['frames'],
            }
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(',', ':'),
            ).encode('utf-8')
            record_bytes = len(serialized)
            if record_bytes > self.logging_config['queue_event_window_max_record_bytes']:
                raise ValueError(
                    'Compact queue event window exceeds logging.queue_event_window_max_record_bytes: '
                    f'{record_bytes} > {self.logging_config["queue_event_window_max_record_bytes"]}'
                )
            if self._queue_event_bytes_written + record_bytes + 1 > self.logging_config[
                'queue_event_windows_max_total_bytes'
            ]:
                raise ValueError('Compact queue event windows exceed logging.queue_event_windows_max_total_bytes')
            with path.open('ab') as file:
                file.write(serialized + b'\n')
            self._queue_event_bytes_written += record_bytes + 1
            self._queue_event_max_record_bytes_observed = max(
                self._queue_event_max_record_bytes_observed,
                record_bytes,
            )
            self._queue_event_windows_written += 1
            written += 1
            if not complete:
                incomplete += 1
        return written, incomplete

    def _finalize_queue_event_slot(self, metrics: dict[str, float]) -> dict[str, float]:
        condition_id = metrics.get(
            'condition_id',
            metrics.get('queue_condition_id', metrics.get('queue_skip_condition_id', metrics.get('queue_retry_condition_id'))),
        )
        instance_id = metrics.get(
            'condition_instance_id',
            metrics.get(
                'queue_condition_instance_id',
                metrics.get('queue_skip_condition_instance_id', metrics.get('queue_retry_failed_condition_instance_id')),
            ),
        )
        round_index = metrics.get(
            'queue_condition_round_index',
            metrics.get('queue_skip_round_index', metrics.get('queue_retry_round_index', -1.0)),
        )
        if condition_id is None or instance_id is None:
            return {}
        slot = (int(round_index), int(condition_id), int(instance_id))
        if self._queue_event_slot != slot:
            return {
                'queue_condition_event_window_count': 0.0,
                'queue_condition_event_window_dropped_count': 0.0,
                'queue_condition_event_window_incomplete_count': 0.0,
            }
        written, incomplete = self._write_queue_event_candidates(slot, termination_reason='condition_closed')
        dropped = self._queue_event_slot_windows_dropped
        self._queue_event_slot = None
        self._queue_event_history.clear()
        self._queue_event_candidates.clear()
        return {
            'queue_condition_event_window_count': float(written),
            'queue_condition_event_window_dropped_count': float(dropped),
            'queue_condition_event_window_incomplete_count': float(incomplete),
        }

    def finish_queue_event_slot(self, *, termination_reason: str = 'logger_finalized') -> tuple[int, int]:
        """Flush bounded windows for an inference slot, including an incomplete final window."""
        if self._queue_event_slot is None:
            return 0, 0
        result = self._write_queue_event_candidates(self._queue_event_slot, termination_reason=termination_reason)
        self._queue_event_slot = None
        self._queue_event_history.clear()
        self._queue_event_candidates.clear()
        return result

    def log_episode(self, episode: int, metrics: dict[str, Any]) -> None:
        row = {'episode': int(episode)}
        for key, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                continue
            row[key] = value
        self._episode_rows.append(row)

    def save_checkpoint(self, file_name: str, state_dict: dict[str, Any]) -> None:
        torch.save(state_dict, self.checkpoint_dir / file_name)

    def save_summary(self, file_name: str, summary: dict[str, Any]) -> None:
        def sanitize(value: Any) -> Any:
            # Summaries include identities, flags, receipts and nullable metrics,
            # not just scalar training metrics. Never coerce those to float.
            if isinstance(value, np.ndarray):
                return sanitize(value.tolist())
            if isinstance(value, np.generic):
                return sanitize(value.item())
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        # Validate serialization before opening the destination: unsupported
        # values must not leave an empty/truncated summary behind.
        encoded = json.dumps(sanitize(summary), ensure_ascii=False, indent=2, allow_nan=False)
        with (self.output_dir / file_name).open('w', encoding='utf-8') as file:
            file.write(encoded)

    def flush(self) -> None:
        if self._metrics_rows:
            self._append_csv(self.output_dir / 'metrics.csv', self._metrics_rows)
            self._append_stage_csvs()
            self._metrics_rows.clear()
        if self.compact_streaming:
            self._write_compact_summary()
            self._write_compact_events()
        if self._episode_rows:
            self._write_csv(self.output_dir / 'episode_summary.csv', list(self._episode_rows))

    def _metric_sketch(self, stage: str, name: str) -> _BoundedMetricSketch:
        key = (stage, name)
        sketch = self._metric_sketches.get(key)
        if sketch is None:
            sketch = _BoundedMetricSketch(
                self.logging_config['quantile_sample_capacity'],
                seed=len(self._metric_sketches) + 1,
            )
            self._metric_sketches[key] = sketch
        return sketch

    def _append_metric_row(self, stage: str, step: int, name: str, value: float) -> None:
        if self.compact_streaming:
            estimated_bytes = len(stage) + len(name) + 48
            if (
                self._debug_rows_written >= self.logging_config['debug_max_rows']
                or self._debug_bytes_written + estimated_bytes > self.logging_config['debug_max_bytes']
            ):
                self._debug_rows_dropped += 1
                return
            self._debug_rows_written += 1
            self._debug_bytes_written += estimated_bytes
        self._metrics_rows.append({
            'stage': stage,
            'step': int(step),
            'name': name,
            'value': value,
        })

    def _record_event_if_changed(self, stage: str, step: int, metrics: dict[str, float]) -> None:
        state = tuple(metrics.get(field) for field in _EVENT_STATE_FIELDS)
        if state == self._previous_event_state.get(stage):
            return
        self._previous_event_state[stage] = state
        event = {'stage': stage, 'step': int(step), 'event': 'execution_or_gate_transition'}
        event.update({field: metrics[field] for field in _EVENT_METRIC_FIELDS if field in metrics})
        self._event_ring.append(event)

    def _write_compact_summary(self) -> None:
        stages: dict[str, dict[str, dict[str, float]]] = {}
        for (stage, name), sketch in sorted(self._metric_sketches.items()):
            stages.setdefault(stage, {})[name] = sketch.summary()
        summary = {
            'schema_version': COMPACT_LOGGING_SCHEMA_VERSION,
            'mode': 'compact_streaming',
            'quantile_sample_capacity': self.logging_config['quantile_sample_capacity'],
            'event_ring_capacity': self.logging_config['event_ring_capacity'],
            'episode_ring_capacity': self.logging_config['episode_ring_capacity'],
            'debug_long_form_enabled': self.logging_config['debug_long_form_enabled'],
            'debug_rows_written': self._debug_rows_written,
            'debug_rows_dropped': self._debug_rows_dropped,
            'debug_bytes_written_estimate': self._debug_bytes_written,
            'event_ring_size': len(self._event_ring),
            'queue_condition_ring_capacity': self.logging_config['queue_condition_ring_capacity'],
            'queue_condition_max_record_bytes': self.logging_config['queue_condition_max_record_bytes'],
            'queue_condition_max_record_bytes_observed': self._queue_condition_max_record_bytes_observed,
            'queue_condition_summaries_max_total_bytes': self.logging_config[
                'queue_condition_summaries_max_total_bytes'
            ],
            'queue_condition_bytes_written': self._queue_condition_bytes_written,
            'queue_condition_rows_seen': self._queue_condition_rows_seen,
            'queue_condition_rows_retained': len(self._queue_condition_ring),
            'queue_condition_rows_dropped': self._queue_condition_rows_dropped,
            'queue_event_pre_frames': self.logging_config['queue_event_pre_frames'],
            'queue_event_post_frames': self.logging_config['queue_event_post_frames'],
            'queue_event_window_capacity': self.logging_config['queue_event_window_capacity'],
            'queue_event_full_state_enabled': self.logging_config['queue_event_full_state_enabled'],
            'queue_event_window_max_record_bytes': self.logging_config['queue_event_window_max_record_bytes'],
            'queue_event_windows_max_total_bytes': self.logging_config['queue_event_windows_max_total_bytes'],
            'queue_event_windows_seen': self._queue_event_windows_seen,
            'queue_event_windows_written': self._queue_event_windows_written,
            'queue_event_windows_dropped': self._queue_event_windows_dropped,
            'queue_event_bytes_written': self._queue_event_bytes_written,
            'queue_event_max_record_bytes_observed': self._queue_event_max_record_bytes_observed,
            'stages': stages,
        }
        with (self.output_dir / 'compact_metrics_summary.json').open('w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    def _write_compact_events(self) -> None:
        with (self.output_dir / 'compact_execution_events.jsonl').open('w', encoding='utf-8') as file:
            for event in self._event_ring:
                file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def _append_stage_csvs(self) -> None:
        stages = sorted({row['stage'] for row in self._metrics_rows})
        for stage in stages:
            rows = [row for row in self._metrics_rows if row['stage'] == stage]
            if rows:
                self._append_csv(self.output_dir / f'{stage}_metrics.csv', rows)

    def _append_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = ['stage', 'step', 'name', 'value']
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open('a', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
