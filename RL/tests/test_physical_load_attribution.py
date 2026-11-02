from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from RL.analyze_physical_load_attribution import (
    _ActionLagBuffer,
    _ExcursionAccumulator,
    _event_csv_fields,
    _read_event_window_triggers,
)
from RL.utils.constants import BLADED_UDP_VARIABLES
from RL.validate_round4_frozen_event_windows import validate_event_windows


class PhysicalLoadAttributionTests(unittest.TestCase):
    def test_action_lag_buffer_maps_measurement_to_previous_interval_action(self):
        history = _ActionLagBuffer(capacity=51)
        for index in range(61):
            timestamp = round(index * 0.02, 6)
            history.append({
                'action_time_s': timestamp,
                'torque_nm': 10_000_000.0 + index,
                'pitch_rad': 0.01 + index * 1e-5,
                'actor_policy_version': 4,
                'actor_policy_sha256': 'a' * 64,
            })

        lags = history.lags(1.20)
        self.assertEqual(lags['action_lag_0020ms_torque_nm'], 10_000_059.0)
        self.assertEqual(lags['action_lag_0100ms_torque_nm'], 10_000_055.0)
        self.assertEqual(lags['action_lag_0200ms_pitch_rad'], 0.0105)
        self.assertEqual(lags['action_lag_1000ms_torque_nm'], 10_000_010.0)
        self.assertEqual(lags['action_lag_0020ms_actor_policy_version'], 4)
        self.assertEqual(lags['action_lag_0020ms_actor_policy_sha256'], 'a' * 64)

    def test_excursion_accumulator_merges_adjacent_samples_and_splits_gap(self):
        output = tempfile.TemporaryDirectory(prefix='rl_attribution_events_')
        self.addCleanup(output.cleanup)
        path = Path(output.name) / 'events.csv'
        fields = _event_csv_fields()
        slot = (1, 1, 8, 1)
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            accumulator = _ExcursionAccumulator(merge_gap_s=0.02)
            for index, value in enumerate((101.0, 105.0, 99.0, 120.0)):
                timestamp = round(index * 0.02, 6)
                sample = {
                    'round_index': 1,
                    'measurement_time_s': timestamp,
                    'abs_my_nm': value,
                    'signed_my_nm': -value,
                    'pid_peak_abs_nm': 100.0,
                    'origin_time_s': round(timestamp - 0.02, 6),
                    'origin_frame_id': index,
                    'future_sample_index': 0,
                    'origin_control_region': 'below_rated',
                    'direct_action_time_s': round(timestamp - 0.02, 6),
                    'direct_torque_nm': 12_000_000.0 + index,
                    'direct_pitch_rad': 0.02,
                    'actor_policy_version': 2,
                    'actor_policy_sha256': 'b' * 64,
                    'approximate_zoh': True,
                    'prediction_supported': True,
                    'predicted_peak_nm': 98.0,
                    'predicted_exceed': False,
                    'event_window_match': False,
                    'event_window_context': {},
                    'lag_actions': {
                        f'action_lag_{lag:04d}ms_{name}': None
                        for lag in (20, 100, 200, 500, 1000)
                        for name in ('torque_nm', 'pitch_rad', 'actor_policy_version', 'actor_policy_sha256')
                    },
                }
                accumulator.add(slot, sample, writer)
            accumulator.finish_slot((1, 1, 8), writer)

        with path.open(encoding='utf-8', newline='') as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['round_index'], '1')
        self.assertEqual(records[0]['first_cross_time_s'], '0.0')
        self.assertEqual(records[0]['last_exceed_time_s'], '0.02')
        self.assertEqual(records[0]['peak_abs_my_nm'], '105.0')
        self.assertEqual(records[0]['peak_direct_torque_nm'], '12000001.0')
        self.assertEqual(records[1]['first_cross_time_s'], '0.06')

    def test_event_window_trigger_scan_is_bounded_and_retains_only_blade_triggers(self):
        with tempfile.TemporaryDirectory(prefix='rl_attribution_windows_') as tmp:
            path = Path(tmp) / 'windows.jsonl'
            records = [
                {
                    'schema_version': 'queue_condition_event_window_v1',
                    'round_index': 1, 'condition_id': 8, 'condition_instance_id': 1,
                    'event_type': 'execution_mode_transition', 'trigger_step': 2,
                    'window_complete': True,
                    'frames': [{'step': 2, 'udp_sim_time': 1.0, 'udp_next_sim_time': 1.02}],
                },
                {
                    'schema_version': 'queue_condition_event_window_v2',
                    'round_index': 1, 'condition_id': 8, 'condition_instance_id': 1,
                    'event_type': 'blade_2_first_pid_peak_exceed', 'score': 1.01, 'trigger_step': 3,
                    'window_complete': False,
                    'frames': [{
                        'step': 3, 'udp_sim_time': 1.02, 'udp_next_sim_time': 1.04,
                        'origin_payload': {'Time': 1.02, 'I_AverageWindSpeed': 9.0},
                        'observed_payload': {'Time': 1.04, 'I_AverageWindSpeed': 9.1},
                    }],
                },
            ]
            path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in records),
                            encoding='utf-8')
            triggers, summary = _read_event_window_triggers(path, expected_count=2, maximum_total_bytes=4096)

            self.assertEqual(len(triggers), 1)
            key = (1, 8, 1, 2, 1.04)
            self.assertEqual(triggers[key]['event_type'], 'blade_2_first_pid_peak_exceed')
            self.assertFalse(triggers[key]['window_complete'])
            self.assertEqual(summary['records'], 2)
            self.assertEqual(summary['counters']['non_blade_trigger_windows'], 1)

    def test_event_window_trigger_scan_rejects_oversized_record(self):
        with tempfile.TemporaryDirectory(prefix='rl_attribution_oversized_') as tmp:
            path = Path(tmp) / 'windows.jsonl'
            path.write_bytes(b' ' * (256 * 1024 + 1) + b'\n')
            with self.assertRaisesRegex(ValueError, 'exceeds 256 KiB'):
                _read_event_window_triggers(path, expected_count=1, maximum_total_bytes=512 * 1024)

    def test_round4_window_validator_accepts_bound_full_existing_state_capture(self):
        with tempfile.TemporaryDirectory(prefix='rl_round4_event_validator_') as tmp:
            root = Path(tmp)
            run_dir = root / 'RL/experiments/run_fixture'
            config_dir = root / 'RL/configs/run_fixture'
            run_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            checkpoint_path = root / 'source/checkpoints/queue_round_4.pt'
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b'fixture round4 checkpoint')
            checkpoint_sha = __import__('hashlib').sha256(checkpoint_path.read_bytes()).hexdigest()

            receipt_path = config_dir / 'receipt.json'
            receipt = {
                'schema_version': 'physical_shadow_round4_frozen_actor_operator_development_receipt_v1',
                'route': 'operator_confirmed_round4_frozen_actor_4wind1round600_shadow_inference_v1',
                'run_id': 'run_fixture',
                'runtime_identity': {
                    'execution_entry': 'inference', 'baseline_only': False,
                    'actor_checkpoint_sha256': checkpoint_sha,
                },
                'source_run': {'round_4_checkpoint_path': str(checkpoint_path),
                               'round_4_checkpoint_sha256': checkpoint_sha},
                'policy_acceptance': False, 'optimizer_updates': False,
                'independently_verified': False, 'actor_frozen': True,
                'event_window_validation': {
                    'full_existing_state_enabled': True, 'pre_frames': 50, 'post_frames': 25,
                    'max_record_bytes': 512 * 1024, 'max_total_bytes': 512 * 1024 * 1024,
                },
            }
            receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
            receipt_sha = __import__('hashlib').sha256(receipt_path.read_bytes()).hexdigest()

            config = {
                'experiment_name': 'run_fixture',
                'physical_shadow_launch': {'authorized': True,
                    'route': 'operator_confirmed_round4_frozen_actor_4wind1round600_shadow_inference_v1',
                    'evidence_path': str(receipt_path), 'evidence_sha256': receipt_sha},
                'physical_shadow_run_role': {'actor_frozen': True, 'optimizer_updates': False},
                'policy_acceptance': False,
                'train': {'online_rl_enabled': False},
                'logging': {
                    'queue_event_full_state_enabled': True,
                    'queue_event_window_capacity': 16,
                    'queue_event_pre_frames': 50,
                    'queue_event_post_frames': 25,
                    'queue_event_window_max_record_bytes': 512 * 1024,
                    'queue_event_windows_max_total_bytes': 512 * 1024 * 1024,
                },
            }
            (run_dir / 'effective_config.json').write_text(json.dumps(config), encoding='utf-8')
            (run_dir / 'checkpoint_metadata.json').write_text(json.dumps({
                'checkpoint_path': str(checkpoint_path), 'checkpoint_loaded_at_utc': 'fixture',
            }), encoding='utf-8')
            frames = []
            for index in range(76):
                current_time = round(index * 0.02, 6)
                origin = {field: 1.0 for field in BLADED_UDP_VARIABLES}
                observed = {field: 1.0 for field in BLADED_UDP_VARIABLES}
                origin['Time'] = current_time
                observed['Time'] = round(current_time + 0.02, 6)
                origin['I_AverageWindSpeed'] = 10.0
                observed['I_AverageWindSpeed'] = 10.1
                observed['Blade root 1 My (Nm)'] = 3.0
                frame = {
                    'step': float(index + 1),
                    'queue_condition_round_index': 1.0,
                    'queue_epoch': 1.0,
                    'queue_condition_id': 8.0,
                    'queue_condition_instance_id': 1.0,
                    'actor_policy_version': 0.0,
                    'actor_policy_sha256': checkpoint_sha,
                    'udp_sim_time': current_time,
                    'udp_next_sim_time': round(current_time + 0.02, 6),
                    'control_sent_action_torque': 13_500_000.0,
                    'control_sent_action_pitch': 0.01,
                    'reward_blade_root_1_peak_reference_nm': 2.0,
                    'wire_action_real': [13_500_000.0, 0.01],
                    'origin_payload': origin,
                    'observed_payload': observed,
                    'origin_invalid_fields': [],
                    'observed_invalid_fields': [],
                    'observed_sample_valid': True,
                }
                frames.append(frame)
            event_path = run_dir / 'compact_queue_condition_event_windows.jsonl'
            event_row = {
                'schema_version': 'queue_condition_event_window_v2',
                'round_index': 1, 'condition_id': 8, 'condition_instance_id': 1,
                'event_type': 'blade_1_first_pid_peak_exceed', 'score': 1.01,
                'trigger_step': 51, 'pre_frame_count': 50, 'post_frame_count': 25,
                'target_pre_frame_count': 50, 'target_post_frame_count': 25,
                'window_complete': True, 'termination_reason': 'post_window_complete',
                'frames': frames,
            }
            encoded = (json.dumps(event_row, separators=(',', ':')) + '\n').encode()
            event_path.write_bytes(encoded)
            digest = __import__('hashlib').sha256(encoded).hexdigest()
            compact_summary = {'queue_event_full_state_enabled': True, 'queue_event_windows_written': 1,
                               'queue_event_windows_dropped': 0, 'queue_event_bytes_written': len(encoded)}
            (run_dir / 'compact_metrics_summary.json').write_text(json.dumps(compact_summary), encoding='utf-8')
            summary = {
                'status': 'completed', 'queue_mode': True, 'episodes_requested': 4, 'episodes_completed': 4,
                'episode_summaries': [
                    {'queue_condition_id': cid, 'queue_condition_instance_id': index,
                     'queue_epoch': index, 'case_identity_status': 'development_unverified'}
                    for index, cid in enumerate((8, 11, 12, 16), start=1)
                ],
                'physical_load_shadow': {
                    'finalized': True, 'faulted': False,
                    'counters': {'finished_conditions': 4, 'finish_slot_identity_mismatch': 0,
                                 'duplicate_slot_finish': 0},
                    'authorization': {'policy_acceptance': False},
                    'condition_receipts': [
                        {'queue_epoch': index, 'condition_instance_id': index, 'condition_id': cid,
                         'finish_slot_identity_match': True}
                        for index, cid in enumerate((8, 11, 12, 16), start=1)
                    ],
                },
                'queue_event_capture': {
                    'event_windows_written': 1, 'event_bytes_written': len(encoded),
                    'event_max_record_bytes_observed': len(encoded.rstrip(b'\n')),
                },
            }
            (run_dir / 'run_summary.json').write_text(json.dumps(summary), encoding='utf-8')

            report = validate_event_windows(run_dir, output_path=run_dir / 'validation.json')

        self.assertTrue(report['capture_integrity_pass'])
        self.assertFalse(report['strict_pid_identity_pass'])
        self.assertFalse(report['policy_acceptance'])
        self.assertEqual(report['counters']['valid_measurement_frame_occurrences'], 76)
        self.assertEqual(report['event_types']['blade_1_first_pid_peak_exceed'], 1)
        self.assertEqual(report['event_window_file_sha256'], digest)


if __name__ == '__main__':
    unittest.main()
