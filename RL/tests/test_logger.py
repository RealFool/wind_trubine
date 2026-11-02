from __future__ import annotations

import csv
import json
import tracemalloc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from RL.utils.logger import ExperimentLogger


class TestExperimentLogger(unittest.TestCase):
    def test_structured_summary_preserves_identity_flags_and_nullable_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_logger_summary_') as tmpdir:
            logger = ExperimentLogger(Path(tmpdir))
            logger.save_summary('summary.json', {
                'backend': 'physical_fir_v2', 'finalized': True, 'faulted': False,
                'counters': {'records_written': np.int64(3)}, 'condition_receipts': [],
                'metrics': (np.float64(1.25), float('nan'), float('inf'), -float('inf')),
                'supported': np.array([True, False]), 'unknown': None,
            })
            text = (Path(tmpdir) / 'summary.json').read_text()
            summary = json.loads(text, parse_constant=lambda value: self.fail(value))
        self.assertEqual(summary['backend'], 'physical_fir_v2')
        self.assertIs(summary['finalized'], True)
        self.assertIs(summary['faulted'], False)
        self.assertEqual(summary['counters'], {'records_written': 3})
        self.assertEqual(summary['condition_receipts'], [])
        self.assertEqual(summary['metrics'], [1.25, None, None, None])
        self.assertEqual(summary['supported'], [True, False])
        self.assertIsNone(summary['unknown'])

    def test_summary_serialization_error_does_not_truncate_destination(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_logger_summary_') as tmpdir:
            path = Path(tmpdir) / 'summary.json'
            path.write_text('{"original": true}')
            logger = ExperimentLogger(Path(tmpdir))
            with self.assertRaises(TypeError):
                logger.save_summary(path.name, {'unsupported': object()})
            self.assertEqual(path.read_text(), '{"original": true}')

    def test_flush_appends_metrics_without_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_logger_') as tmpdir:
            logger = ExperimentLogger(Path(tmpdir))
            logger.log_metrics('train_timing', 1, {'step_time_ms': 10.0})
            logger.flush()
            logger.log_metrics('train_timing', 2, {'step_time_ms': 20.0})
            logger.flush()

            with (Path(tmpdir) / 'train_timing_metrics.csv').open('r', encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['step'] for row in rows], ['1', '2'])

    def test_compact_streaming_logger_bounds_long_form_and_keeps_aggregates(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_logger_') as tmpdir:
            output = Path(tmpdir)
            logger = ExperimentLogger(
                output,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 3,
                    'episode_ring_capacity': 2,
                    'debug_long_form_enabled': True,
                    'debug_max_rows': 5,
                    'debug_max_bytes': 10_000,
                },
            )
            for step in range(1_000):
                logger.log_metrics(
                    'queue_train_step',
                    step,
                    {
                        'power': float(step),
                        'control_pre_send_execution_mode': float(step % 2),
                        'control_low_speed_stage': 0.0,
                    },
                )
            logger.log_episode(1, {'reward': 1.0})
            logger.log_episode(2, {'reward': 2.0})
            logger.log_episode(3, {'reward': 3.0})
            logger.flush()

            summary = json.loads((output / 'compact_metrics_summary.json').read_text(encoding='utf-8'))
            events = (output / 'compact_execution_events.jsonl').read_text(encoding='utf-8').strip().splitlines()
            with (output / 'queue_train_step_metrics.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            with (output / 'episode_summary.csv').open(encoding='utf-8', newline='') as handle:
                episodes = list(csv.DictReader(handle))

        self.assertEqual(summary['mode'], 'compact_streaming')
        self.assertEqual(summary['stages']['queue_train_step']['power']['count'], 1_000.0)
        self.assertIn('p95', summary['stages']['queue_train_step']['power'])
        self.assertEqual(summary['debug_rows_written'], 5)
        self.assertGreater(summary['debug_rows_dropped'], 0)
        self.assertLessEqual(len(rows), 5)
        self.assertLessEqual(len(events), 3)
        self.assertEqual([row['episode'] for row in episodes], ['2', '3'])

    def test_compact_streaming_persists_bounded_condition_summaries(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_queue_logger_') as tmpdir:
            output = Path(tmpdir)
            logger = ExperimentLogger(
                output,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 4,
                    'episode_ring_capacity': 2,
                    'queue_condition_ring_capacity': 2,
                    'queue_condition_max_record_bytes': 1024,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            for condition_id in (3, 4, 5):
                logger.log_metrics(
                    'queue_condition',
                    condition_id,
                    {
                        'condition_id': float(condition_id),
                        'condition_instance_id': float(100 + condition_id),
                        'queue_completed_round': 1.0,
                        'queue_condition_batch_count': 4.0,
                        'queue_condition_batch_sample_count': 512.0,
                        'queue_condition_batch_source_actor_sample_count': 400.0,
                        'queue_condition_batch_source_low_speed_recovery_hold_sample_count': 112.0,
                        'queue_condition_batch_quota_relaxed_count': 2.0,
                        'queue_condition_batch_source_low_speed_recovery_hold_quota_relaxed_count': 2.0,
                        'dual_constraint_condition_complete': 1.0,
                    },
                )
            logger.flush()

            summary = json.loads((output / 'compact_metrics_summary.json').read_text(encoding='utf-8'))
            condition_rows = [
                json.loads(line)
                for line in (output / 'compact_queue_condition_summaries.jsonl')
                .read_text(encoding='utf-8')
                .splitlines()
            ]

        self.assertEqual(summary['queue_condition_rows_seen'], 3)
        self.assertEqual(summary['queue_condition_rows_retained'], 2)
        self.assertEqual(summary['queue_condition_rows_dropped'], 1)
        self.assertEqual([row['condition_id'] for row in condition_rows], [3.0, 4.0, 5.0])
        self.assertEqual(condition_rows[-1]['queue_condition_batch_sample_count'], 512.0)
        self.assertEqual(
            condition_rows[-1]['queue_condition_batch_source_low_speed_recovery_hold_sample_count'],
            112.0,
        )
        self.assertEqual(condition_rows[-1]['queue_condition_batch_quota_relaxed_count'], 2.0)
        self.assertLessEqual(summary['queue_condition_max_record_bytes_observed'], 1024)

    def test_compact_condition_summary_fails_when_single_record_exceeds_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_queue_logger_budget_') as tmpdir:
            logger = ExperimentLogger(
                Path(tmpdir),
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 4,
                    'episode_ring_capacity': 2,
                    'queue_condition_ring_capacity': 2,
                    'queue_condition_max_record_bytes': 32,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            with self.assertRaisesRegex(ValueError, 'exceeds logging.queue_condition_max_record_bytes'):
                logger.log_metrics(
                    'queue_condition',
                    1,
                    {'condition_id': 3.0, 'oversized_field': 0.0, 'padding': 1.0e100},
                )

    def test_event_windows_keep_peak_and_first_exceed_with_bounded_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_event_windows_') as tmpdir:
            output = Path(tmpdir)
            logger = ExperimentLogger(
                output,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 4,
                    'episode_ring_capacity': 2,
                    'queue_condition_ring_capacity': 4,
                    'queue_condition_max_record_bytes': 1024,
                    'queue_event_pre_frames': 2,
                    'queue_event_post_frames': 2,
                    'queue_event_window_capacity': 16,
                    'queue_event_window_max_record_bytes': 32_768,
                    'queue_event_windows_max_total_bytes': 1_000_000,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            for step in range(8):
                blade_two = 120.0 if step == 2 else 150.0 if step == 6 else 50.0
                logger.log_metrics(
                    'queue_train_step',
                    step,
                    {
                        'queue_condition_round_index': 1.0,
                        'queue_condition_id': 3.0,
                        'queue_condition_instance_id': 101.0,
                        'udp_next_sim_time': float(step) * 0.02,
                        'wind_t': 12.0,
                        'power': 10_000_000.0,
                        'control_pre_send_execution_mode': 1.0 if step < 1 else 8.0,
                        'control_low_speed_stage': 0.0 if step < 1 else 4.0,
                        'control_ood_detected': float(step == 3),
                        'control_ood_score': 6.0 if step == 3 else 0.0,
                        'control_action_projection_applied': float(step in {4, 5}),
                        'control_action_rate_limited': 0.0,
                        'reward_relative_power_raw': -200_000.0 if step >= 3 else 0.0,
                        'reward_pid_power_reference_raw': 10_000_000.0,
                        'reward_persistent_power_deficit_ewma': 0.02 if step >= 3 else 0.0,
                        'reward_blade_root_1_raw': 50.0,
                        'reward_blade_root_2_raw': blade_two,
                        'reward_blade_root_3_raw': 60.0,
                        'reward_blade_root_1_peak_reference_nm': 100.0,
                        'reward_blade_root_2_peak_reference_nm': 100.0,
                        'reward_blade_root_3_peak_reference_nm': 100.0,
                    },
                )
            logger.log_metrics(
                'queue_condition',
                8,
                {
                    'condition_id': 3.0,
                    'condition_instance_id': 101.0,
                    'queue_condition_round_index': 1.0,
                },
            )
            logger.flush()

            records = [
                json.loads(line)
                for line in (output / 'compact_queue_condition_event_windows.jsonl')
                .read_text(encoding='utf-8')
                .splitlines()
            ]
            summary = json.loads((output / 'compact_metrics_summary.json').read_text(encoding='utf-8'))

        self.assertLessEqual(len(records), 16)
        self.assertTrue(any(record['event_type'] == 'blade_2_first_pid_peak_exceed' for record in records))
        self.assertEqual(sum(record['event_type'] == 'action_projection' for record in records), 1)
        peak_record = next(record for record in records if record['event_type'] == 'blade_2_new_max_abs_my')
        self.assertEqual(peak_record['score'], 150.0)
        self.assertIn(6.0, [frame['step'] for frame in peak_record['frames']])
        exceed_record = next(record for record in records if record['event_type'] == 'blade_2_first_pid_peak_exceed')
        self.assertTrue(exceed_record['window_complete'])
        self.assertLessEqual(peak_record['post_frame_count'], 2)
        self.assertLessEqual(summary['queue_event_max_record_bytes_observed'], 32_768)
        self.assertLessEqual(summary['queue_event_bytes_written'], 1_000_000)

    def test_full_state_event_windows_capture_only_existing_frames_and_wire_actions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_event_full_state_') as tmpdir:
            output = Path(tmpdir)
            logger = ExperimentLogger(
                output,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 4,
                    'episode_ring_capacity': 2,
                    'queue_condition_ring_capacity': 4,
                    'queue_condition_max_record_bytes': 1024,
                    'queue_event_pre_frames': 1,
                    'queue_event_post_frames': 1,
                    'queue_event_window_capacity': 16,
                    'queue_event_window_max_record_bytes': 32_768,
                    'queue_event_windows_max_total_bytes': 1_000_000,
                    'queue_event_full_state_enabled': True,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            for step in range(5):
                current_time = float(step) * 0.02
                blade_one = 110.0 if step == 2 else 90.0
                metrics = {
                    'queue_condition_round_index': 1.0,
                    'queue_condition_id': 8.0,
                    'queue_condition_instance_id': 101.0,
                    'queue_epoch': 1.0,
                    'udp_frame_id': float(step),
                    'udp_next_frame_id': float(step + 1),
                    'udp_sim_time': current_time,
                    'udp_next_sim_time': current_time + 0.02,
                    'wind_t': 9.5,
                    'wind_next': 9.6,
                    'power': 9_000_000.0,
                    'action_torque': 13_500_000.0 + step,
                    'action_pitch': 0.01,
                    'control_pre_send_execution_mode': 1.0,
                    'control_optimization_state_code': 1.0,
                    'control_region_code': 0.0,
                    'control_baseline_action_torque': 13_500_000.0,
                    'control_baseline_action_pitch': 0.01,
                    'control_policy_delta_torque': 100.0,
                    'control_policy_delta_pitch': 0.0001,
                    'control_sent_action_torque': 13_500_000.0 + step,
                    'control_sent_action_pitch': 0.01,
                    'actor_policy_version': 0.0,
                    'control_action_projection_applied': 0.0,
                    'control_action_rate_limited': 0.0,
                    'reward_blade_root_1_raw': blade_one,
                    'reward_blade_root_2_raw': 80.0,
                    'reward_blade_root_3_raw': 70.0,
                    'reward_blade_root_1_peak_reference_nm': 100.0,
                    'reward_blade_root_2_peak_reference_nm': 100.0,
                    'reward_blade_root_3_peak_reference_nm': 100.0,
                    'queue_event_sample_valid': 1.0,
                    'physical_load_shadow_result_available': 1.0 if step == 3 else 0.0,
                    'physical_load_shadow_result_origin_time_s': current_time,
                    'physical_load_shadow_prediction_supported_h100': 1.0 if step == 3 else 0.0,
                    'physical_load_shadow_predicted_peak_nm_h100_blade_1': 105.0 if step == 3 else 0.0,
                }
                event_frame = {
                    'origin_payload': {
                        'Time': current_time,
                        'I_RotorAzimuth': 0.4 + step,
                        'I_YawError': 0.02,
                        'I_AverageWindSpeed': 9.5,
                        'Blade root 1 My (Nm)': 95.0,
                    },
                    'observed_payload': {
                        'Time': current_time + 0.02,
                        'I_RotorAzimuth': 0.5 + step,
                        'I_YawError': 0.021,
                        'I_AverageWindSpeed': 9.6,
                        'Blade root 1 My (Nm)': blade_one,
                    },
                    'origin_invalid_fields': [],
                    'observed_invalid_fields': [],
                    'observed_sample_valid': True,
                    'wire_action_real': [13_500_000.0 + step, 0.01],
                    'actor_policy_sha256': 'a' * 64,
                }
                logger.log_metrics('queue_train_step', step, metrics, event_frame=event_frame)
            logger.finish_queue_event_slot()
            logger.flush()
            records = [json.loads(line) for line in (
                output / 'compact_queue_condition_event_windows.jsonl'
            ).read_text(encoding='utf-8').splitlines()]

        exceed = next(row for row in records if row['event_type'] == 'blade_1_first_pid_peak_exceed')
        trigger = next(frame for frame in exceed['frames'] if int(frame['step']) == exceed['trigger_step'])
        self.assertEqual(exceed['schema_version'], 'queue_condition_event_window_v2')
        self.assertEqual(trigger['origin_payload']['I_YawError'], 0.02)
        self.assertEqual(trigger['observed_payload']['Blade root 1 My (Nm)'], 110.0)
        self.assertEqual(trigger['wire_action_real'], [13_500_002.0, 0.01])
        self.assertEqual(trigger['actor_policy_sha256'], 'a' * 64)
        self.assertTrue(trigger['observed_sample_valid'])

        predicted = next(row for row in records if row['event_type'] ==
                         'physical_shadow_h100_predicted_pid_peak_exceed_blade_1')
        predicted_trigger = next(frame for frame in predicted['frames']
                                 if int(frame['step']) == predicted['trigger_step'])
        self.assertEqual(predicted_trigger['physical_load_shadow_prediction_origin_match'], 1.0)
        self.assertEqual(predicted_trigger['physical_load_shadow_predicted_peak_nm_h100_blade_1'], 105.0)

    def test_full_state_event_capture_rejects_new_udp_fields_and_unaligned_predictions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_event_full_state_guard_') as tmpdir:
            logger = ExperimentLogger(
                Path(tmpdir),
                {
                    'mode': 'compact_streaming',
                    'queue_event_full_state_enabled': True,
                    'quantile_sample_capacity': 8,
                    'event_ring_capacity': 4,
                    'episode_ring_capacity': 2,
                    'queue_condition_ring_capacity': 4,
                    'queue_condition_max_record_bytes': 1024,
                    'queue_event_pre_frames': 1,
                    'queue_event_post_frames': 1,
                    'queue_event_window_capacity': 16,
                    'queue_event_window_max_record_bytes': 32_768,
                    'queue_event_windows_max_total_bytes': 1_000_000,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            bad_frame = {
                'origin_payload': {'new_sensor': 1.0},
                'observed_payload': {},
                'origin_invalid_fields': [],
                'observed_invalid_fields': [],
            }
            with self.assertRaisesRegex(ValueError, 'outside the existing Bladed UDP whitelist'):
                logger.log_metrics('queue_train_step', 0, {
                    'queue_condition_round_index': 1.0,
                    'queue_condition_id': 8.0,
                    'queue_condition_instance_id': 101.0,
                }, event_frame=bad_frame)

            logger.log_metrics('queue_train_step', 0, {
                'queue_condition_round_index': 1.0,
                'queue_condition_id': 8.0,
                'queue_condition_instance_id': 101.0,
                'udp_sim_time': 1.0,
                'physical_load_shadow_result_available': 1.0,
                'physical_load_shadow_result_origin_time_s': 0.8,
                'physical_load_shadow_prediction_supported_h100': 1.0,
                'physical_load_shadow_predicted_peak_nm_h100_blade_1': 120.0,
                'reward_blade_root_1_peak_reference_nm': 100.0,
            })
            logger.log_metrics('queue_condition', 1, {
                'condition_id': 8.0,
                'condition_instance_id': 101.0,
                'queue_condition_round_index': 1.0,
            })
            logger.flush()
            path = Path(tmpdir) / 'compact_queue_condition_event_windows.jsonl'
            records = [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]
        self.assertFalse(any(row['event_type'].startswith('physical_shadow_') for row in records))

    def test_compact_logger_resource_budget_for_ninety_two_queue_slots(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_compact_92_slot_budget_') as tmpdir:
            output = Path(tmpdir)
            tracemalloc.start()
            logger = ExperimentLogger(
                output,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 16,
                    'event_ring_capacity': 64,
                    'episode_ring_capacity': 16,
                    'queue_condition_ring_capacity': 512,
                    'queue_condition_max_record_bytes': 64 * 1024,
                    'queue_condition_summaries_max_total_bytes': 16 * 1024 * 1024,
                    'queue_event_pre_frames': 4,
                    'queue_event_post_frames': 4,
                    'queue_event_window_capacity': 16,
                    'queue_event_window_max_record_bytes': 64 * 1024,
                    'queue_event_windows_max_total_bytes': 64 * 1024 * 1024,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            for slot_index in range(92):
                round_index = slot_index // 23 + 1
                condition_id = slot_index % 23 + 3
                instance_id = slot_index + 1
                for frame in range(40):
                    mode = 1.0 if frame < 4 else 8.0
                    blade_two = 120.0 if frame == 15 else 150.0 if frame == 35 else 50.0
                    logger.log_metrics(
                        'queue_train_step',
                        slot_index * 40 + frame,
                        {
                            'queue_condition_round_index': float(round_index),
                            'queue_condition_id': float(condition_id),
                            'queue_condition_instance_id': float(instance_id),
                            'udp_next_sim_time': float(frame) * 0.02,
                            'wind_t': 12.0,
                            'power': 10_000_000.0,
                            'control_pre_send_execution_mode': mode,
                            'control_low_speed_stage': 0.0 if frame < 4 else 4.0,
                            'control_ood_detected': float(frame == 10),
                            'control_ood_score': 6.0 if frame == 10 else 0.0,
                            'control_action_projection_applied': float(frame == 30),
                            'control_action_rate_limited': float(frame == 31),
                            'reward_relative_power_raw': -200_000.0 if frame >= 10 else 0.0,
                            'reward_pid_power_reference_raw': 10_000_000.0,
                            'reward_persistent_power_deficit_ewma': 0.02 if frame >= 10 else 0.0,
                            'reward_blade_root_1_raw': 50.0,
                            'reward_blade_root_2_raw': blade_two,
                            'reward_blade_root_3_raw': 60.0,
                            'reward_blade_root_1_peak_reference_nm': 100.0,
                            'reward_blade_root_2_peak_reference_nm': 100.0,
                            'reward_blade_root_3_peak_reference_nm': 100.0,
                        },
                    )
                    self.assertLessEqual(len(logger._queue_event_history), 5)
                    self.assertLessEqual(len(logger._queue_event_candidates), 16)
                logger.log_metrics(
                    'queue_condition',
                    slot_index * 40 + 40,
                    {
                        'condition_id': float(condition_id),
                        'condition_instance_id': float(instance_id),
                        'queue_condition_round_index': float(round_index),
                    },
                )
                self.assertLessEqual(len(logger._queue_condition_ring), 512)
            logger.flush()
            _current_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            summary = json.loads((output / 'compact_metrics_summary.json').read_text(encoding='utf-8'))
            event_path = output / 'compact_queue_condition_event_windows.jsonl'
            event_records = [json.loads(line) for line in event_path.read_text(encoding='utf-8').splitlines()]
            artifacts = [
                output / 'compact_metrics_summary.json',
                output / 'compact_execution_events.jsonl',
                output / 'compact_queue_condition_summaries.jsonl',
                event_path,
            ]
            artifact_bytes = sum(path.stat().st_size for path in artifacts)
            slot_window_counts: dict[tuple[int, int, int], int] = {}
            for record in event_records:
                key = (record['round_index'], record['condition_id'], record['condition_instance_id'])
                slot_window_counts[key] = slot_window_counts.get(key, 0) + 1

        self.assertEqual(summary['queue_condition_rows_seen'], 92)
        self.assertEqual(summary['queue_condition_rows_retained'], 92)
        self.assertLessEqual(summary['queue_event_windows_max_total_bytes'], 64 * 1024 * 1024)
        self.assertLessEqual(summary['queue_event_bytes_written'], 64 * 1024 * 1024)
        self.assertLessEqual(artifact_bytes, 1024 * 1024 * 1024)
        self.assertLessEqual(peak_python_bytes, 256 * 1024 * 1024)
        self.assertEqual(len(slot_window_counts), 92)
        self.assertTrue(all(count <= 16 for count in slot_window_counts.values()))


if __name__ == '__main__':
    unittest.main()