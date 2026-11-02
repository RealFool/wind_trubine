from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from RL.build_state20_control_config import RESOLVED_CONTRACT_PATH
from RL.build_state20_queue_configs import BC_CONFIG_PATH, CONFIG_ROOT, PRESETS, PROJECT_ROOT, build_profile


class TestBuildState20QueueConfigs(unittest.TestCase):
    def test_new_23_wind_profiles_are_strict_and_output_isolated(self) -> None:
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        canonical_hashes = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (BC_CONFIG_PATH, RESOLVED_CONTRACT_PATH)
        }
        profiles = (
            ('23wind_6round_600s', 6, None),
            ('23wind_4round_600s', 4, None),
            ('23wind_2round_600s', 2, None),
            ('23wind_1round_600s', 1, None),
            ('23wind_2round_600s_recovery_cap25', 2, 0.25),
        )

        with tempfile.TemporaryDirectory(prefix='state20_queue_profiles_', dir=CONFIG_ROOT) as temp_dir:
            output_dir = Path(temp_dir)
            predictor_artifact = output_dir / 'candidate_predictor.pt'
            predictor_artifact.write_bytes(b'candidate-predictor-test-artifact')
            bc_config = json.loads(BC_CONFIG_PATH.read_text(encoding='utf-8'))
            bc_checkpoint_path = Path(bc_config['bc']['resume_checkpoint'])
            if not bc_checkpoint_path.is_absolute():
                bc_checkpoint_path = PROJECT_ROOT / bc_checkpoint_path
            interface_path = PROJECT_ROOT / bc_config['load_predictor']['interface_path']
            (output_dir / 'candidate_predictor.summary.json').write_text(json.dumps({
                'checkpoint_sha256': hashlib.sha256(predictor_artifact.read_bytes()).hexdigest(),
                'schema_version': 'load_predictor_checkpoint_v4_candidate_conditioned',
                'interface_schema_version': 'causal_load_predictor_interface_v3',
                'interface_sha256': hashlib.sha256(interface_path.read_bytes()).hexdigest(),
                'dataset_schema_version': 'load_predictor_dataset_v2',
                'dataset_manifest_sha256': bc_config['load_predictor']['dataset_manifest_sha256'],
                'dataset_root': str((PROJECT_ROOT / bc_config['load_predictor']['dataset_root']).resolve()),
                'condition_partitions': {
                    'train': list(range(3, 17)),
                    'validation': [17, 18, 19],
                    'calibration': [20, 21, 22],
                    'coverage_audit': [23, 24, 25],
                },
                'split_sample_counts': {
                    'train': 10,
                    'validation': 2,
                    'calibration': 2,
                    'coverage_audit': 2,
                },
                'plan_counts_by_split': {
                    name: {'realized_plan': 0, 'zoh_candidate': count}
                    for name, count in {
                        'train': 10,
                        'validation': 2,
                        'calibration': 2,
                        'coverage_audit': 2,
                    }.items()
                },
                'time_partitions_s': {
                    'train': [0.0, 204.0],
                    'validation': [205.48, 222.0],
                    'calibration': [223.48, 390.0],
                    'coverage_audit': [391.48, 600.02],
                },
            }), encoding='utf-8')
            built = {}
            for profile, expected_rounds, recovery_cap in profiles:
                config_path, manifest_path = build_profile(
                    profile,
                    identity_level='strict',
                    output_dir=output_dir,
                    predictor_artifact_path=predictor_artifact,
                )
                config = json.loads(config_path.read_text(encoding='utf-8'))
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                slots_per_round = len(PRESETS[profile]['condition_ids'])

                self.assertEqual(config['queue']['total_condition_rounds'], expected_rounds)
                self.assertEqual(config['queue']['condition_ids'], list(range(3, 26)))
                self.assertEqual(config['state20_queue_contract']['expected_scheduled_updates'],
                                 (expected_rounds - 1) * 23 * 30_000 // config['train']['update_every_steps'])
                self.assertEqual(
                    config['state20_queue_contract']['bc_warmstart_checkpoint_sha256'],
                    hashlib.sha256(bc_checkpoint_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(manifest['identity_validation_level'], 'strict')
                self.assertEqual(manifest['expanded_slot_count'], slots_per_round * expected_rounds)
                self.assertEqual(config['queue']['case_identity_validation']['validation_level'], 'strict')
                self.assertEqual(config['logging']['mode'], 'compact_streaming')
                self.assertEqual(config['logging']['debug_long_form_enabled'], False)
                self.assertEqual(config['logging']['queue_condition_max_record_bytes'], 512 * 1024)
                self.assertEqual(config['logging']['queue_event_pre_frames'], 25)
                self.assertEqual(config['logging']['queue_event_post_frames'], 25)
                self.assertEqual(config['logging']['queue_event_window_capacity'], 16)
                self.assertEqual(config['logging']['queue_event_window_max_record_bytes'], 256 * 1024)
                self.assertEqual(config['logging']['queue_event_windows_max_total_bytes'], 512 * 1024 * 1024)
                self.assertEqual(config['load_predictor']['replay_capacity'], 100_000)
                self.assertEqual(config['load_predictor']['replay_record_capacity'], 512)
                self.assertEqual(config['load_predictor']['shadow_audit_capacity_per_group'], 35)
                self.assertEqual(config['device'], 'cuda:0')
                self.assertEqual(
                    config['load_predictor']['predictor_artifact_sha256'],
                    hashlib.sha256(predictor_artifact.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    config['load_predictor']['predictor_artifact_path'],
                    predictor_artifact.relative_to(PROJECT_ROOT).as_posix(),
                )
                self.assertEqual(
                    config['load_predictor']['dataset_manifest_sha256'],
                    bc_config['load_predictor']['dataset_manifest_sha256'],
                )
                self.assertEqual(
                    config['load_predictor']['dataset_manifest_path'],
                    'RL/data/load_predictor_dataset_state20_stride10_v2/manifest.json',
                )
                self.assertEqual(config['train']['condition_replay'].get('source_max_fraction', {}).get(
                    'low_speed_recovery_hold'
                ), recovery_cap)
                built[profile] = (config_path, manifest_path)

            with self.assertRaises(FileExistsError):
                build_profile(
                    '23wind_1round_600s',
                    identity_level='strict',
                    output_dir=output_dir,
                )
            overwritten_config, overwritten_manifest = build_profile(
                '23wind_1round_600s',
                identity_level='strict',
                output_dir=output_dir,
                overwrite=True,
            )
            self.assertTrue(overwritten_config.is_file())
            self.assertTrue(overwritten_manifest.is_file())

        self.assertEqual(
            canonical_hashes,
            {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in canonical_hashes},
        )

    def test_run_id_binds_config_and_timestamped_experiment_directory(self) -> None:
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='state20_run_id_', dir=CONFIG_ROOT) as temp_dir:
            run_id = f'{Path(temp_dir).name}_run'
            expected_config_dir = PROJECT_ROOT / 'RL/configs' / run_id
            self.assertFalse(expected_config_dir.exists())
            config_path, manifest_path = build_profile(
                '23wind_4round_600s',
                identity_level='strict',
                run_id=run_id,
            )
            config = json.loads(config_path.read_text(encoding='utf-8'))

            self.assertEqual(config_path.parent, expected_config_dir)
            self.assertEqual(config_path.name, 'train.json')
            self.assertEqual(manifest_path.name, 'queue_manifest.json')
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(config['experiment_name'], run_id)
            self.assertEqual(config['output_dir'], 'RL/experiments')
            self.assertTrue(config['append_timestamp'])
            self.assertFalse(config['bc']['force_retrain'])
            self.assertEqual(config['bc']['resume_checkpoint'], 'RL/BC_checkpoint/bladed_pid_b4_state20_residual_v3.pt')
            self.assertEqual(config['bc']['export_checkpoint_path'], '')
            config_path.unlink()
            manifest_path.unlink()
            expected_config_dir.rmdir()

    def test_run_id_refuses_existing_config_or_timestamped_experiment(self) -> None:
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        run_id = f'state20_run_id_guard_{uuid.uuid4().hex}'
        config_dir = PROJECT_ROOT / 'RL/configs' / run_id
        config_dir.mkdir()
        try:
            with self.assertRaises(FileExistsError):
                build_profile('23wind_4round_600s', identity_level='strict', run_id=run_id)
        finally:
            config_dir.rmdir()

        experiment_root = PROJECT_ROOT / 'RL/experiments'
        experiment_root.mkdir(parents=True, exist_ok=True)
        experiment_dir = experiment_root / f'{run_id}_20261022T120000Z'
        experiment_dir.mkdir()
        try:
            with self.assertRaises(FileExistsError):
                build_profile('23wind_4round_600s', identity_level='strict', run_id=run_id)
        finally:
            experiment_dir.rmdir()


if __name__ == '__main__':
    unittest.main()