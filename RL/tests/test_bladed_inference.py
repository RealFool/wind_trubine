from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from RL.agents import TD3Agent
from RL.envs.wind_turbine_env import BLADED_CSV_HEADER
from RL.infer.run_bladed_inference import (
    RECORD_FIELDS,
    _default_output_dir,
    _apply_inference_torque_high,
    _validate_data_schema,
    run_inference,
)
from RL.tests.test_fake_bladed_integration import FakeBladedUdpServer, _pick_free_udp_port
from RL.utils.constants import DATA_SCHEMA_VERSION, OBS_DIM
from RL.utils.paired_pid_reference import build_paired_pid_reference
from RL.utils.safe_residual_control import SafeResidualController


def _inference_config(port: int) -> dict:
    return {
        'data_schema_version': DATA_SCHEMA_VERSION,
        'algorithm': 'td3',
        'device': 'cpu',
        'data': {},
        'env': {
            'backend': 'bladed_udp',
            'udp_host': '127.0.0.1',
            'udp_port': port,
            'udp_timeout_s': 0.2,
            'udp_reset_wait_s': 3.0,
            'udp_step_wait_s': 3.0,
            'udp_inter_frame_wait_s': 3.0,
            'udp_require_frame_id': True,
            'udp_recv_buffer_size': 8192,
            'udp_strict_field_count': True,
            'udp_timeout_penalty': -10.0,
            'episode_length': 3,
            'action_low': [13_000_000.0, 0.0],
            'action_high': [21_000_000.0, 0.33],
            'action_delta_low': [-15_000.0, -0.0003],
            'action_delta_high': [15_000.0, 0.0003],
            'safe_action_low': [13_000_000.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.33],
            'rated_power': 15_000_000.0,
            'gear_ratio': 97.0,
            'reward': {
                'power_ref': 15_000_000.0,
                'power_reward_mode': 'power_ref_fraction',
                'power_available_mode': 'rated',
                'rated_wind_speed': 12.8,
                'cut_in_wind_speed': 3.0,
                'power_weight': 1.0,
                'tower_weight': 0.0,
                'blade_root_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        },
        'control': {
            'baseline_action': [20_000_000.0, 0.05],
            'residual_limit': [1_000_000.0, 0.05],
            'safe_action_low': [13_000_000.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.33],
            'ood_threshold': 100_000_000.0,
            'low_power_threshold': -1.0,
            'low_power_patience': 10,
            'negative_power_patience': 10,
        },
        'agent': {
            'hidden_dims': [8],
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'td3_bc_enabled': True,
        },
    }


class TestBladedInference(unittest.TestCase):
    @staticmethod
    def _write_strict_reference(root: Path) -> Path:
        raw_root = root / 'raw'
        raw_root.mkdir()
        fields = [
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        for condition_id in range(3, 26):
            path = raw_root / f'data_windspeed{condition_id}.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(3):
                    timestamp = index * 0.1
                    wind = (
                        12.8 + 0.5 * ((timestamp % 7.0) / 7.0)
                        if condition_id == 12
                        else condition_id + 0.01 * index
                    )
                    writer.writerow({
                        'Time': timestamp,
                        'I_AverageWindSpeed': wind,
                        'Electrical power': 15_000_000.0,
                        'Blade root 1 My (Nm)': 45_000_000.0,
                        'Blade root 2 My (Nm)': 44_500_000.0,
                        'Blade root 3 My (Nm)': 45_500_000.0,
                    })
        document = build_paired_pid_reference(
            raw_root,
            time_window_start_s=0.0,
            time_window_end_s=0.2,
            project_root=root,
        )
        path = root / 'paired_pid_reference.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        return path

    def test_inference_torque_override_preserves_observation_scale(self) -> None:
        config = _inference_config(_pick_free_udp_port())
        original_controller = SafeResidualController(config['env'], config['control'])
        original_fingerprint = original_controller.actor_control_fingerprint

        overridden = _apply_inference_torque_high(json.loads(json.dumps(config)), 22_000_000.0)
        controller = SafeResidualController(overridden['env'], overridden['control'])

        self.assertEqual(controller.env_action_high[0], 22_000_000.0)
        self.assertEqual(controller.safe_high[0], 22_000_000.0)
        self.assertEqual(controller.observation_action_high[0], 21_000_000.0)
        self.assertNotEqual(controller.actor_control_fingerprint, original_fingerprint)

    def test_default_output_dir_uses_checkpoint_name_without_timestamp(self) -> None:
        checkpoint_path = Path('/tmp/example/checkpoints/queue_round_4.pt')

        self.assertEqual(
            _default_output_dir(checkpoint_path, condition_id=24),
            Path('/tmp/example/inference/wind_24_round_4'),
        )
        self.assertEqual(
            _default_output_dir(checkpoint_path),
            Path('/tmp/example/inference/round_4'),
        )
        self.assertEqual(
            _default_output_dir(Path('/tmp/example/checkpoints/best_eval.pt'), condition_id=24),
            Path('/tmp/example/inference/wind_24_best_eval'),
        )

    def test_checkpoint_inference_writes_raw_compatible_step_records(self) -> None:
        port = _pick_free_udp_port()
        config = _inference_config(port)
        device = torch.device('cpu')
        agent = TD3Agent(obs_dim=OBS_DIM, action_dim=2, config=config['agent'], device=device)
        agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
        agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

        with tempfile.TemporaryDirectory(prefix='rl_infer_') as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / 'inference.pt'
            output_dir = root / 'output'
            checkpoint = {
                'agent': agent.state_dict(),
                'config': config,
                'data_schema_version': DATA_SCHEMA_VERSION,
                'episode': 7,
                'total_steps': 123,
            }
            torch.save(checkpoint, checkpoint_path)

            server = FakeBladedUdpServer('127.0.0.1', port)
            server.start()
            try:
                summary = run_inference(
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    episodes=1,
                    max_steps=3,
                    device=device,
                    config_source='test',
                )
            finally:
                server.stop()

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['total_control_steps'], 3)
            self.assertEqual(summary['total_records'], 4)
            self.assertTrue((output_dir / 'effective_config.json').is_file())
            self.assertTrue((output_dir / 'checkpoint_metadata.json').is_file())
            self.assertTrue((output_dir / 'run_summary.json').is_file())
            self.assertTrue((output_dir / 'bladed_raw_frames.csv').is_file())

            with (output_dir / 'bladed_inference_steps.csv').open('r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, RECORD_FIELDS)
                self.assertEqual(reader.fieldnames[:len(BLADED_CSV_HEADER)], BLADED_CSV_HEADER)
                rows = list(reader)
            self.assertEqual(len(rows), 4)
            control_rows = [row for row in rows if row['record_type'] == 'control_step']
            self.assertEqual(len(control_rows), 3)
            self.assertEqual(rows[-1]['record_type'], 'terminal_observation')
            self.assertNotEqual(control_rows[0]['Time'], '')
            self.assertNotEqual(control_rows[0]['I_AverageWindSpeed'], '')
            self.assertNotEqual(control_rows[0]['policy_inference_ms'], '')
            self.assertNotEqual(control_rows[0]['udp_step_ms'], '')
            self.assertNotEqual(control_rows[0]['t_rx_monotonic_s'], '')
            self.assertNotEqual(control_rows[0]['t_tx_monotonic_s'], '')
            self.assertNotEqual(control_rows[0]['t_e2e_ms'], '')
            self.assertNotEqual(control_rows[0]['sent_action_torque'], '')
            for field in (
                'baseline_action_torque',
                'baseline_action_pitch',
                'policy_delta_torque',
                'policy_delta_pitch',
                'pre_safety_action_torque',
                'pre_safety_action_pitch',
                'pitch_protection_floor',
                'pitch_protection_state',
                'pitch_protection_applied',
                'low_speed_stage',
                'low_speed_scale',
                'low_speed_reason',
                'post_safety_action_torque',
                'post_safety_action_pitch',
                'projected_action_torque',
                'projected_action_pitch',
                'rate_limited_action_torque',
                'rate_limited_action_pitch',
            ):
                self.assertNotEqual(control_rows[0][field], '')
            self.assertEqual(
                float(control_rows[0]['rate_limited_action_torque']),
                float(control_rows[0]['sent_action_torque']),
            )
            self.assertEqual(
                float(control_rows[0]['rate_limited_action_pitch']),
                float(control_rows[0]['sent_action_pitch']),
            )
            self.assertNotEqual(control_rows[0]['step_info_json'], '')
            self.assertLessEqual(float(control_rows[0]['sent_action_torque']), 21_000_000.0)
            self.assertGreaterEqual(float(control_rows[0]['sent_action_torque']), 13_000_000.0)
            self.assertEqual(json.loads(control_rows[0]['step_info_json'])['udp_connected'], 1.0)
            self.assertEqual(json.loads(control_rows[0]['step_info_json'])['udp_handshake_complete'], 1.0)
            self.assertEqual(float(control_rows[0]['info_control_fallback_active']), 0.0)
            self.assertGreaterEqual(
                float(control_rows[0]['t_tx_monotonic_s']),
                float(control_rows[0]['t_rx_monotonic_s']),
            )
            self.assertTrue((output_dir / 'edge_performance_summary.json').is_file())
            self.assertFalse(summary['edge_performance']['T1_pass'])

            with (output_dir / 'bladed_raw_frames.csv').open('r', encoding='utf-8', newline='') as file:
                raw_reader = csv.DictReader(file)
                self.assertEqual(raw_reader.fieldnames, BLADED_CSV_HEADER)
                raw_rows = list(raw_reader)
            self.assertEqual(len(raw_rows), 4)
            self.assertNotEqual(raw_rows[0]['I_AverageWindSpeed'], '')

            with (output_dir / 'episode_summary.csv').open('r', encoding='utf-8', newline='') as file:
                episode_rows = list(csv.DictReader(file))
            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(episode_rows[0]['control_steps'], '3')

    def test_baseline_only_inference_skips_actor_and_records_dynamic_baseline(self) -> None:
        port = _pick_free_udp_port()
        config = _inference_config(port)
        device = torch.device('cpu')
        agent = TD3Agent(obs_dim=OBS_DIM, action_dim=2, config=config['agent'], device=device)
        agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
        agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

        with tempfile.TemporaryDirectory(prefix='rl_baseline_infer_') as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / 'inference.pt'
            output_dir = root / 'output'
            checkpoint = {
                'agent': agent.state_dict(),
                'config': config,
                'data_schema_version': DATA_SCHEMA_VERSION,
            }
            torch.save(checkpoint, checkpoint_path)

            server = FakeBladedUdpServer('127.0.0.1', port)
            server.start()
            try:
                with patch(
                    'RL.infer.run_bladed_inference._build_agent',
                    side_effect=AssertionError('baseline-only inference must not build the actor'),
                ):
                    summary = run_inference(
                        config=config,
                        checkpoint=checkpoint,
                        checkpoint_path=checkpoint_path,
                        output_dir=output_dir,
                        episodes=1,
                        max_steps=1,
                        device=device,
                        config_source='test',
                        baseline_only=True,
                    )
            finally:
                server.stop()

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['inference_mode'], 'baseline_only')
            self.assertTrue(summary['baseline_only'])
            with (output_dir / 'bladed_inference_steps.csv').open('r', encoding='utf-8', newline='') as file:
                control_row = next(row for row in csv.DictReader(file) if row['record_type'] == 'control_step')
            self.assertEqual(float(control_row['info_control_baseline_only']), 1.0)
            self.assertEqual(float(control_row['info_control_baseline_action_torque']), 20_000_000.0)
            self.assertAlmostEqual(float(control_row['info_control_baseline_action_pitch']), 0.05)
            self.assertEqual(float(control_row['policy_residual_torque_norm']), 0.0)
            self.assertEqual(float(control_row['policy_residual_pitch_norm']), 0.0)
            self.assertEqual(float(control_row['baseline_action_torque']), 20_000_000.0)
            self.assertAlmostEqual(float(control_row['baseline_action_pitch']), 0.05)
            self.assertEqual(float(control_row['policy_delta_torque']), 0.0)
            self.assertEqual(float(control_row['policy_delta_pitch']), 0.0)
            for stage in ('pre_safety', 'post_safety', 'projected', 'rate_limited', 'sent'):
                self.assertEqual(float(control_row[f'{stage}_action_torque']), 20_000_000.0)
                self.assertAlmostEqual(float(control_row[f'{stage}_action_pitch']), 0.05)
            with (output_dir / 'checkpoint_metadata.json').open('r', encoding='utf-8') as file:
                metadata = json.load(file)
            self.assertEqual(metadata['inference_mode'], 'baseline_only')

    def test_rejects_checkpoint_without_current_data_schema(self) -> None:
        config = _inference_config(_pick_free_udp_port())
        with tempfile.TemporaryDirectory(prefix='rl_infer_schema_') as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, 'Checkpoint data schema is incompatible'):
                run_inference(
                    config=config,
                    checkpoint={'agent': {}},
                    checkpoint_path=root / 'legacy.pt',
                    output_dir=root / 'output',
                    episodes=1,
                    max_steps=1,
                    device=torch.device('cpu'),
                    config_source='test',
                )

    def test_rejects_checkpoint_with_wrong_control_contract(self) -> None:
        config = _inference_config(_pick_free_udp_port())
        with self.assertRaisesRegex(ValueError, 'control contract is incompatible'):
            _validate_data_schema(
                {
                    'data_schema_version': DATA_SCHEMA_VERSION,
                    'control_fingerprint': 'four-wind-baseline',
                },
                config,
                expected_control_fingerprint='23-wind-baseline',
            )

    def test_prefers_actor_control_contract_when_checkpoint_provides_it(self) -> None:
        config = _inference_config(_pick_free_udp_port())
        _validate_data_schema(
            {
                'data_schema_version': DATA_SCHEMA_VERSION,
                'control_fingerprint': 'legacy-contract-with-old-online-reward',
                'actor_control_fingerprint': 'shared-actor-control-contract',
            },
            config,
            expected_control_fingerprint='new-online-training-contract',
            expected_actor_control_fingerprint='shared-actor-control-contract',
        )

    def test_inference_records_complete_handshake_after_wind_switch_pause(self) -> None:
        port = _pick_free_udp_port()
        config = _inference_config(port)
        config['env']['episode_length'] = 1
        config['env']['udp_inter_frame_wait_s'] = 1.0
        config['env']['udp_pause_threshold_s'] = 0.02
        device = torch.device('cpu')
        agent = TD3Agent(obs_dim=OBS_DIM, action_dim=2, config=config['agent'], device=device)
        agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
        agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

        with tempfile.TemporaryDirectory(prefix='rl_infer_pause_') as tmpdir:
            root = Path(tmpdir)
            output_dir = root / 'output'
            checkpoint = {
                'agent': agent.state_dict(),
                'config': config,
                'data_schema_version': DATA_SCHEMA_VERSION,
            }
            server = FakeBladedUdpServer(
                '127.0.0.1',
                port,
                pause_duration_s=0.05,
                pause_after_first_accepted_action=True,
            )
            server.start()
            try:
                summary = run_inference(
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_path=root / 'inference.pt',
                    output_dir=output_dir,
                    episodes=1,
                    max_steps=1,
                    device=device,
                    config_source='test',
                )
            finally:
                server.stop()

            self.assertEqual(summary['status'], 'completed')
            with (output_dir / 'bladed_inference_steps.csv').open('r', encoding='utf-8', newline='') as file:
                control_row = next(row for row in csv.DictReader(file) if row['record_type'] == 'control_step')
            info = json.loads(control_row['step_info_json'])
            self.assertEqual(info['udp_handshake_complete'], 1.0)
            self.assertEqual(info['udp_inter_frame_timeout'], 0.0)
            self.assertEqual(info['udp_pause_active'], 1.0)
            self.assertGreater(info['udp_receive_wait_ms'], 20.0)

    def test_queue_inference_uses_condition_boundary_not_legacy_episode_length(self) -> None:
        port = _pick_free_udp_port()
        config = _inference_config(port)
        config['env'].update(
            {
                'episode_length': 1,
                'rollout_segment_steps': 1,
                'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
                'queue_lifecycle_enabled': True,
                'require_condition_metadata': True,
            }
        )
        device = torch.device('cpu')
        agent = TD3Agent(obs_dim=OBS_DIM, action_dim=2, config=config['agent'], device=device)
        agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
        agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

        with tempfile.TemporaryDirectory(prefix='rl_infer_queue_') as tmpdir:
            root = Path(tmpdir)
            output_dir = root / 'output'
            checkpoint = {
                'agent': agent.state_dict(),
                'config': config,
                'data_schema_version': DATA_SCHEMA_VERSION,
            }
            server = FakeBladedUdpServer(
                '127.0.0.1',
                port,
                use_queue_envelope=True,
                switch_after_accepted_actions=3,
            )
            server.start()
            try:
                summary = run_inference(
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_path=root / 'inference.pt',
                    output_dir=output_dir,
                    episodes=1,
                    max_steps=None,
                    device=device,
                    config_source='test',
                )
            finally:
                server.stop()

            self.assertEqual(summary['status'], 'completed')
            self.assertTrue(summary['queue_mode'])
            self.assertEqual(summary['total_control_steps'], 3)
            self.assertEqual(len(server.accepted_action_keys), 3)
            with (output_dir / 'bladed_inference_steps.csv').open('r', encoding='utf-8', newline='') as file:
                control_rows = [row for row in csv.DictReader(file) if row['record_type'] == 'control_step']
            self.assertEqual([row['inference_rollout_segment'] for row in control_rows], ['1', '2', '3'])
            self.assertEqual(json.loads(control_rows[-1]['step_info_json'])['condition_boundary'], 1.0)

    def test_queue_inference_records_full_strict_identity_for_condition_instance(self) -> None:
        port = _pick_free_udp_port()
        config = _inference_config(port)
        config['env'].update({
            'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
            'queue_lifecycle_enabled': True,
            'require_condition_metadata': True,
        })
        with tempfile.TemporaryDirectory(prefix='rl_infer_strict_identity_') as tmpdir:
            root = Path(tmpdir)
            reference_path = self._write_strict_reference(root)
            config['queue'] = {
                'condition_ids': [12],
                'case_identity_validation': {
                    'enabled': True,
                    'validation_level': 'strict',
                    'paired_pid_reference_path': str(reference_path),
                    'fixed_simulation_time_offset_s': 0.0,
                    'wind_absolute_tolerance': 0.001,
                    'time_absolute_tolerance_s': 1.0e-6,
                },
            }
            agent = TD3Agent(
                obs_dim=OBS_DIM,
                action_dim=2,
                config=config['agent'],
                device=torch.device('cpu'),
            )
            agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            checkpoint = {
                'agent': agent.state_dict(),
                'config': config,
                'data_schema_version': DATA_SCHEMA_VERSION,
            }
            output_dir = root / 'output'
            server = FakeBladedUdpServer(
                '127.0.0.1',
                port,
                use_queue_envelope=True,
                queue_protocol_version='bladed_udp_19_si_my_queue_v1',
                switch_after_accepted_actions=3,
            )
            server.start()
            try:
                summary = run_inference(
                    config=config,
                    checkpoint=checkpoint,
                    checkpoint_path=root / 'inference.pt',
                    output_dir=output_dir,
                    episodes=1,
                    max_steps=None,
                    device=torch.device('cpu'),
                    config_source='test',
                )
            finally:
                server.stop()

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['case_identity_validation_level'], 'strict')
            self.assertEqual(summary['strict_case_matched_slot_count'], 1)
            self.assertTrue(summary['all_observed_train_slots_strict_case_matched'])
            identity_paths = list(output_dir.glob('case_identity_condition_12_instance_*.json'))
            self.assertEqual(len(identity_paths), 1)
            identity = json.loads(identity_paths[0].read_text(encoding='utf-8'))
            self.assertEqual(identity['identity_status'], 'strict_case_matched')
            self.assertEqual(identity['observed_trace_sample_count'], 3)


if __name__ == '__main__':
    unittest.main()