from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from RL.utils.release_manifest import EXPECTED_STATE20_CONDITION_IDS, build_release_manifest


class TestReleaseManifest(unittest.TestCase):
    def _write_candidate(self, root: Path, *, include_acceptance: bool) -> dict[str, Path]:
        paths = {
            'checkpoint': root / 'actor.pt',
            'config': root / 'candidate.json',
            'reference': root / 'paired_pid_reference.json',
            'predictor': root / 'predictor.pt',
            'predictor_summary': root / 'predictor.summary.json',
            'predictor_interface': root / 'predictor_interface_v3.json',
            'predictor_dataset_manifest': root / 'predictor_dataset' / 'manifest.json',
            'dll': root / 'dll_evidence.json',
            'resource': root / 'resource_evidence.json',
            'acceptance': root / 'acceptance.json',
            'predictor_summary': root / 'predictor.summary.json',
        }
        paths['checkpoint'].write_bytes(b'actor-checkpoint')
        paths['reference'].write_text('{"reference":"paired"}\n', encoding='utf-8')
        paths['predictor'].write_bytes(b'candidate-conditioned-predictor')
        paths['predictor_dataset_manifest'].parent.mkdir(parents=True)
        interface_document = {
            'schema_version': 'causal_load_predictor_interface_v3',
            'history_frames': 50,
            'step_duration_s': 0.02,
            'horizons_s': [0.1, 0.2, 0.5],
        }
        paths['predictor_interface'].write_text(json.dumps(interface_document) + '\n', encoding='utf-8')
        predictor_sha = hashlib.sha256(paths['predictor'].read_bytes()).hexdigest()
        interface_sha = hashlib.sha256(paths['predictor_interface'].read_bytes()).hexdigest()
        dataset_manifest_document = {
            'schema_version': 'load_predictor_dataset_v2',
            'predictor_interface_schema_version': 'causal_load_predictor_interface_v3',
            'predictor_interface_sha256': interface_sha,
            'source_conditions': list(EXPECTED_STATE20_CONDITION_IDS),
            'time_window_start_s': 0.0,
            'time_window_end_s': 600.0,
        }
        paths['predictor_dataset_manifest'].write_text(
            json.dumps(dataset_manifest_document) + '\n', encoding='utf-8'
        )
        dataset_manifest_sha = hashlib.sha256(paths['predictor_dataset_manifest'].read_bytes()).hexdigest()
        time_partitions = {
            'train': [0.0, 390.0],
            'validation': [391.48, 468.0],
            'calibration': [469.48, 528.0],
            'coverage_audit': [529.48, 600.02],
        }
        paths['predictor_summary'].write_text(json.dumps({
            'checkpoint_sha256': predictor_sha,
            'schema_version': 'load_predictor_checkpoint_v4_candidate_conditioned',
            'interface_schema_version': 'causal_load_predictor_interface_v3',
            'interface_sha256': interface_sha,
            'plan_input_semantics': 'candidate_plan+horizon_index+plan_kind_index',
            'candidate_plan_encoding': 'real_action_sequence_gru_normalized_by_engineering_bounds_v1',
            'dataset_schema_version': 'load_predictor_dataset_v2',
            'dataset_manifest_sha256': dataset_manifest_sha,
            'time_partitions_s': time_partitions,
            'condition_partitions': {
                'train': list(range(3, 17)),
                'validation': [17, 18, 19],
                'calibration': [20, 21, 22],
                'coverage_audit': [23, 24, 25],
            },
            'split_sample_counts': {
                'train': 1,
                'validation': 1,
                'calibration': 1,
                'coverage_audit': 1,
            },
            'plan_counts_by_split': {
                name: {'realized_plan': 0, 'zoh_candidate': 1}
                for name in ('train', 'validation', 'calibration', 'coverage_audit')
            },
            'guard_calibration_authorized': False,
        }), encoding='utf-8')
        config = {
            'control_contract_fingerprint': 'control-contract-v1',
            'actor_control_fingerprint': 'actor-control-v1',
            'resolved_control_contract_path': 'RL/data/resolved_control_contract.json',
            'queue': {
                'condition_ids': list(EXPECTED_STATE20_CONDITION_IDS),
                'case_identity_validation': {
                    'enabled': True,
                    'validation_level': 'strict',
                    'paired_pid_reference_path': str(paths['reference']),
                },
            },
            'load_predictor': {
                'mode': 'shadow',
                'predictor_artifact_path': str(paths['predictor']),
                'predictor_artifact_sha256': predictor_sha,
                'predictor_time_partitions_s': time_partitions,
                'interface_path': str(paths['predictor_interface']),
                'interface_schema_version': 'causal_load_predictor_interface_v3',
                'dataset_root': str(paths['predictor_dataset_manifest'].parent),
                'dataset_manifest_path': str(paths['predictor_dataset_manifest']),
                'dataset_manifest_sha256': dataset_manifest_sha,
                'dataset_schema_version': 'load_predictor_dataset_v2',
                'plan_input_semantics': 'candidate_plan+horizon_index+plan_kind_index',
                'candidate_plan_encoding': 'real_action_sequence_gru_normalized_by_engineering_bounds_v1',
            },
        }
        paths['config'].write_text(json.dumps(config), encoding='utf-8')
        paths['dll'].write_text(json.dumps({
            'compiler_pass': True,
            'queue_protocol_v2_pass': True,
            'full_dynamic_rounds_pass': True,
        }), encoding='utf-8')
        paths['resource'].write_text(json.dumps({'pass': True, 'compact_bytes': 123456}), encoding='utf-8')
        if include_acceptance:
            paths['acceptance'].write_text(json.dumps({
                'schema_version': 'multi_scenario_deployment_acceptance_v1',
                'expected_scenario_ids': [str(value) for value in EXPECTED_STATE20_CONDITION_IDS],
                'scenario_count': 23,
                'valid_scenario_count': 23,
                'unexpected_scenario_ids': [],
                'all_expected_scenarios_valid': True,
                'official_acceptance_pass': True,
                'strict_per_blade_pass': True,
                'strict_case_identity_pass': True,
                'edge_T1_pass_all_scenarios': True,
                'release_candidate_pass': True,
                'deployment_release_pass': True,
            }), encoding='utf-8')
        return paths

    def test_release_manifest_binds_all_hashes_and_requires_every_offline_and_physical_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_release_manifest_') as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write_candidate(root, include_acceptance=True)
            expected_predictor_sha = hashlib.sha256(paths['predictor'].read_bytes()).hexdigest()
            manifest = build_release_manifest(
                candidate_id='candidate-v7',
                checkpoint_path=paths['checkpoint'],
                config_path=paths['config'],
                paired_pid_reference_path=paths['reference'],
                predictor_artifact_path=paths['predictor'],
                dll_evidence_path=paths['dll'],
                resource_evidence_path=paths['resource'],
                project_root=root,
                acceptance_report_path=paths['acceptance'],
            )

        self.assertEqual(manifest['release_status'], 'release_ready')
        self.assertTrue(manifest['gates']['offline_ready_for_v1'])
        self.assertTrue(manifest['gates']['release_ready'])
        self.assertEqual(manifest['contracts']['expected_condition_ids'], list(EXPECTED_STATE20_CONDITION_IDS))
        self.assertEqual(manifest['artifacts']['predictor']['sha256'], expected_predictor_sha)

    def test_offline_ready_is_distinct_from_physical_release_acceptance(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_release_ready_for_v1_') as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write_candidate(root, include_acceptance=False)
            manifest = build_release_manifest(
                candidate_id='candidate-before-v1',
                checkpoint_path=paths['checkpoint'],
                config_path=paths['config'],
                paired_pid_reference_path=paths['reference'],
                predictor_artifact_path=paths['predictor'],
                dll_evidence_path=paths['dll'],
                resource_evidence_path=paths['resource'],
                project_root=root,
            )

        self.assertEqual(manifest['release_status'], 'ready_for_v1')
        self.assertTrue(manifest['gates']['offline_ready_for_v1'])
        self.assertFalse(manifest['gates']['release_ready'])
        self.assertFalse(manifest['gates']['official_acceptance_pass'])
        self.assertFalse(manifest['gates']['strict_per_blade_pass'])

    def test_predictor_hash_or_resource_gate_mismatch_blocks_offline_readiness(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_release_manifest_blocked_') as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write_candidate(root, include_acceptance=True)
            config = json.loads(paths['config'].read_text(encoding='utf-8'))
            config['load_predictor']['predictor_artifact_sha256'] = '0' * 64
            paths['config'].write_text(json.dumps(config), encoding='utf-8')
            paths['resource'].write_text(json.dumps({'pass': False}), encoding='utf-8')
            manifest = build_release_manifest(
                candidate_id='candidate-blocked',
                checkpoint_path=paths['checkpoint'],
                config_path=paths['config'],
                paired_pid_reference_path=paths['reference'],
                predictor_artifact_path=paths['predictor'],
                dll_evidence_path=paths['dll'],
                resource_evidence_path=paths['resource'],
                project_root=root,
                acceptance_report_path=paths['acceptance'],
            )

        self.assertFalse(manifest['gates']['predictor_artifact_sha_matches_config'])
        self.assertFalse(manifest['gates']['resource_budget_pass'])
        self.assertFalse(manifest['gates']['offline_ready_for_v1'])
        self.assertFalse(manifest['gates']['release_ready'])

    def test_predictor_split_provenance_mismatch_blocks_offline_readiness(self) -> None:
        with tempfile.TemporaryDirectory(prefix='state20_release_split_blocked_') as temporary_directory:
            root = Path(temporary_directory)
            paths = self._write_candidate(root, include_acceptance=False)
            summary = json.loads(paths['predictor_summary'].read_text(encoding='utf-8'))
            summary['condition_partitions']['validation'] = [16, 17, 18]
            paths['predictor_summary'].write_text(json.dumps(summary), encoding='utf-8')
            manifest = build_release_manifest(
                candidate_id='candidate-split-blocked',
                checkpoint_path=paths['checkpoint'],
                config_path=paths['config'],
                paired_pid_reference_path=paths['reference'],
                predictor_artifact_path=paths['predictor'],
                dll_evidence_path=paths['dll'],
                resource_evidence_path=paths['resource'],
                project_root=root,
            )

        self.assertFalse(manifest['gates']['predictor_v4_split_provenance_pass'])
        self.assertFalse(manifest['gates']['predictor_v4_candidate_plan_contract_pass'])
        self.assertFalse(manifest['gates']['offline_ready_for_v1'])


if __name__ == '__main__':
    unittest.main()
