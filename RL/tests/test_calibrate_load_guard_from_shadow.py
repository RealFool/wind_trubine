from __future__ import annotations

import hashlib
import base64
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import torch
import numpy as np

from RL.calibrate_load_guard_from_shadow import calibrate_from_shadow_files
from RL.split_load_predictor_shadow_audit import split_shadow_audit
from RL.models.load_predictor_tcn import CausalTCNLoadPredictor
from RL.utils.control_contract import ENGINEERING_DEMAND_HIGH_NM
from RL.utils.load_predictor_contract import LoadPredictorInterface
from RL.utils.paired_pid_reference import build_paired_pid_reference
from RL.data.load_predictor_dataset import (
    HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION,
    _history_iqr_floors,
)


class TestCalibrateLoadGuardFromShadow(unittest.TestCase):
    @staticmethod
    def _write_predictor_checkpoint(path: Path) -> tuple[str, str, dict[str, dict[str, object]]]:
        interface_path = Path('RL/data/load_predictor_interface_v3.json').resolve()
        interface_sha = hashlib.sha256(interface_path.read_bytes()).hexdigest()
        interface = LoadPredictorInterface.from_path(interface_path)
        model = CausalTCNLoadPredictor(
            input_dim=31,
            interface=interface,
            action_low=[0.0, 0.0],
            action_high=[ENGINEERING_DEMAND_HIGH_NM, 0.43],
        )
        raw_root = path.parent / 'raw'
        raw_root.mkdir()
        raw_fields = [
            'Time', 'I_AverageWindSpeed', 'Electrical power',
            'Blade root 1 My (Nm)', 'Blade root 2 My (Nm)', 'Blade root 3 My (Nm)',
        ]
        for condition_id in range(3, 26):
            with (raw_root / f'data_windspeed{condition_id}.csv').open(
                'w', encoding='utf-8', newline=''
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=raw_fields)
                writer.writeheader()
                for index, timestamp_s in enumerate((0.0, 600.0)):
                    writer.writerow({
                        'Time': timestamp_s,
                        'I_AverageWindSpeed': float(condition_id),
                        'Electrical power': 15_000_000.0,
                        'Blade root 1 My (Nm)': 50_000_000.0 + index,
                        'Blade root 2 My (Nm)': 49_000_000.0 + index,
                        'Blade root 3 My (Nm)': 48_000_000.0 + index,
                    })
        reference_path = path.parent / 'paired_reference.json'
        reference_doc = build_paired_pid_reference(
            raw_root,
            time_window_start_s=0.0,
            time_window_end_s=600.0,
            project_root=Path(__file__).resolve().parents[2],
        )
        reference_path.write_text(json.dumps(reference_doc), encoding='utf-8')
        reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        identities: dict[str, dict[str, object]] = {}
        raw_sha_by_condition: dict[str, str] = {}
        for entry in reference_doc['references']:
            key = str(entry['condition_id'])
            raw_path = raw_root / f'data_windspeed{entry["condition_id"]}.csv'
            raw_sha = str(entry['pid_raw_sha256'])
            trace = entry['canonical_wind_trace']
            raw_sha_by_condition[key] = raw_sha
            identities[key] = {
                'scenario_id': entry['scenario_id'],
                'condition_id': int(entry['condition_id']),
                'pid_raw_csv_path': str(raw_path.resolve()),
                'pid_raw_sha256': raw_sha,
                'raw_source_sha256': raw_sha,
                'time_window_start_s': entry['time_window_start_s'],
                'time_window_end_s': entry['time_window_end_s'],
                'expected_trace_signature': trace['sha256'],
                'expected_trace_sample_count': trace['sample_count'],
                'time_signature_quantization_s': trace['time_signature_quantization_s'],
                'wind_signature_quantization_mps': trace['wind_signature_quantization_mps'],
            }
        normalization = {
            'history_median': [0.0] * 31,
            'history_iqr': _history_iqr_floors(31).astype(float).tolist(),
            'history_iqr_floor_schema': HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION,
            'history_iqr_floors': _history_iqr_floors(31).astype(float).tolist(),
            'history_window_capacity': 4,
            'history_window_bytes': 4 * 50 * 31 * 4,
            'sample_count_used': 4,
            'reservoir_seed': 42,
            'response_median': [0.0] * 2,
            'response_iqr': [1.0] * 2,
            'peak_ratio_median': [0.0] * 3,
            'peak_ratio_iqr': [1.0] * 3,
        }
        torch.save({
            'schema_version': 'load_predictor_checkpoint_v4_candidate_conditioned',
            'guard_calibration_authorized': False,
            'interface_sha256': interface_sha,
            'interface_schema_version': interface.schema_version,
            'dataset_manifest_sha256': 'd' * 64,
            'paired_reference_path': str(reference_path.resolve()),
            'paired_reference_sha256': reference_sha,
            'raw_source_sha256_by_condition': raw_sha_by_condition,
            'paired_identity_by_condition': identities,
            'interface_identity': {
                'schema_version': interface.schema_version,
                'history_frames': interface.history_frames,
                'step_duration_s': interface.step_duration_s,
                'horizons_s': list(interface.horizons_s),
                'soft_guard_horizons_s': list(interface.soft_guard_horizons_s),
                'online_candidate_plan_kinds': list(interface.online_candidate_plan_kinds),
                'candidate_plan_encoding': interface.candidate_plan_encoding,
                'candidate_plan_padding': interface.candidate_plan_padding,
                'horizon_conditioning': interface.horizon_conditioning,
                'plan_kind_conditioning': interface.plan_kind_conditioning,
                'model_input_contract': interface.model_input_contract,
                'response_label_semantics': interface.response_label_semantics,
                'peak_label_semantics': interface.peak_label_semantics,
            },
            'feature_dim': 31,
            'model_architecture': {
                'model_class': 'CausalTCNLoadPredictor',
                'input_dim': 31,
                'hidden_dim': 64,
                'dilations': [1, 2, 4, 8],
                'plan_encoder': 'GRU(input_size=2, hidden_size=64, batch_first=True)',
                'horizon_embedding_dim': 8,
                'plan_kind_embedding_dim': 8,
                'output_width': model.OUTPUT_WIDTH,
            },
            'temporal_purge_s': 1.48,
            'predictor_training_end_s': 204.0,
            'calibration_start_s': 223.48,
            'coverage_audit_start_s': 391.48,
            'split_schema_version': 'load_predictor_condition_split_v2',
            'split_sample_counts': {
                'train': 100,
                'validation': 100,
                'calibration': 100,
                'coverage_audit': 100,
            },
            'plan_counts_by_split': {
                name: {'realized_plan': 0, 'zoh_candidate': 100}
                for name in ('train', 'validation', 'calibration', 'coverage_audit')
            },
            'normalization': normalization,
            'model_state_dict': model.state_dict(),
            'condition_partitions': {
                'train': list(range(3, 17)),
                'validation': [17, 18, 19],
                'calibration': [20, 21, 22],
                'coverage_audit': [23, 24, 25],
            },
            'time_partitions_s': {
                'train': [0.0, 204.0],
                'validation': [205.48, 222.0],
                'calibration': [223.48, 390.0],
                'coverage_audit': [391.48, 600.02],
            },
            'temporal_purge_s': 1.48,
            'predictor_training_end_s': 204.0,
            'calibration_start_s': 223.48,
            'coverage_audit_start_s': 391.48,
            'split_schema_version': 'load_predictor_condition_split_v2',
            'split_sample_counts': {
                'train': 100,
                'validation': 100,
                'calibration': 100,
                'coverage_audit': 100,
            },
            'plan_counts_by_split': {
                name: {'realized_plan': 0, 'zoh_candidate': 100}
                for name in ('train', 'validation', 'calibration', 'coverage_audit')
            },
        }, path)
        checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(f'{path.stem}.summary.json').write_text(json.dumps({
            'checkpoint_sha256': checkpoint_sha,
            'schema_version': 'load_predictor_checkpoint_v4_candidate_conditioned',
            'interface_schema_version': interface.schema_version,
            'interface_sha256': interface_sha,
            'dataset_manifest_sha256': 'd' * 64,
            'time_partitions_s': {
                'train': [0.0, 204.0],
                'validation': [205.48, 222.0],
                'calibration': [223.48, 390.0],
                'coverage_audit': [391.48, 600.02],
            },
        }), encoding='utf-8')
        return interface_sha, reference_sha, identities

    @staticmethod
    def _write_shadow_file(
        path: Path,
        condition_id: int,
        *,
        start_s: float,
        count: int,
        predictor_sha256: str,
        actor_sha256: str,
        identity: dict[str, object],
    ) -> None:
        interface_sha = hashlib.sha256(Path('RL/data/load_predictor_interface_v3.json').read_bytes()).hexdigest()
        with path.open('w', encoding='utf-8') as handle:
            for index in range(count):
                row = {
                    'schema_version': 'load_predictor_shadow_audit_record_v1',
                    'predictor_artifact_sha256': predictor_sha256,
                    'predictor_interface_sha256': interface_sha,
                    'actor_policy_sha256': actor_sha256,
                    'actor_policy_version': 0,
                    'actor_policy_window_frozen': True,
                    'sample_id': index + 1,
                    'condition_id': condition_id,
                    'condition_instance_id': 100 + condition_id,
                    'timestamp_s': start_s + index,
                    'optimization_state': 'above_rated',
                    'horizon_s': 0.1,
                    'horizon_frames': 5,
                    'plan_kind': 'zoh_candidate',
                    'execution_mode': 'actor',
                    'outcome_tag': 'complete',
                    'online_authorized': True,
                    'plan_mismatch': 0.0,
                    'calibration_eligible': True,
                    'identity_status': 'strict_case_matched',
                    'strict_case_matched': 1.0,
                    'pid_raw_sha256': identity['pid_raw_sha256'],
                    'pid_raw_csv_path': identity['pid_raw_csv_path'],
                    'expected_trace_signature': identity['expected_trace_signature'],
                    'expected_trace_sample_count': identity['expected_trace_sample_count'],
                    'scenario_id': identity['scenario_id'],
                    'time_window_start_s': identity['time_window_start_s'],
                    'time_window_end_s': identity['time_window_end_s'],
                    'prediction_available': True,
                    'predicted_response_available': True,
                    'plan_match_complete': True,
                    'feature_dim': 31,
                    'history_schema_version': 'load_predictor_shadow_feature_v1',
                    'history_shape': [50, 31],
                    'history_dtype': '<f4',
                    'history_zlib_base64': base64.b64encode(
                        zlib.compress(np.zeros((50, 31), dtype='<f4').tobytes(), level=1)
                    ).decode('ascii'),
                    'predicted_peak_ratio_by_blade': [0.90, 0.90, 0.90],
                    'actual_future_peak_ratio_by_blade': [0.92, 0.92, 0.92],
                    'actual_future_response': [1.0, 0.1],
                    'candidate_action_real': [1.0, 0.1],
                    'model_input_plan': [[1.0, 0.1]] * 5,
                    'actual_future_action_plan': [[1.0, 0.1]] * 5,
                    'actual_future_actor_policy_versions': [0] * 5,
                }
                handle.write(json.dumps(row) + '\n')

    def test_shadow_calibration_and_independent_audit_authorize_only_disjoint_strict_rl_data(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_guard_shadow_calibration_') as temporary_directory:
            root = Path(temporary_directory)
            calibration_path = root / 'calibration.jsonl'
            audit_path = root / 'coverage_audit.jsonl'
            predictor_path = root / 'predictor.pt'
            output_path = root / 'calibration_artifact.json'
            _interface_sha, _reference_sha, identities = self._write_predictor_checkpoint(predictor_path)
            predictor_sha256 = hashlib.sha256(predictor_path.read_bytes()).hexdigest()
            actor_sha256 = 'a' * 64
            combined_path = root / 'all_shadow.jsonl'
            self._write_shadow_file(
                calibration_path,
                20,
                start_s=230.0,
                count=20,
                predictor_sha256=predictor_sha256,
                actor_sha256=actor_sha256,
                identity=identities['20'],
            )
            self._write_shadow_file(
                audit_path,
                23,
                start_s=400.0,
                count=20,
                predictor_sha256=predictor_sha256,
                actor_sha256=actor_sha256,
                identity=identities['23'],
            )
            combined_path.write_bytes(calibration_path.read_bytes() + audit_path.read_bytes())
            calibration_path.unlink()
            audit_path.unlink()
            partition_summary = split_shadow_audit(
                input_jsonl=combined_path,
                predictor_artifact=predictor_path,
                interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                calibration_output=calibration_path,
                coverage_audit_output=audit_path,
            )

            report = calibrate_from_shadow_files(
                calibration_jsonl=calibration_path,
                coverage_audit_jsonl=audit_path,
                interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                predictor_artifact_path=predictor_path,
                output_path=output_path,
                coverage_target=0.90,
                confidence_level=0.50,
                minimum_samples_per_group=10,
                max_plan_mismatch=0.05,
                predictor_training_end_s=204.0,
                calibration_start_s=223.48,
                coverage_audit_start_s=391.48,
            )
            persisted = json.loads(output_path.read_text(encoding='utf-8'))

        self.assertTrue(report['metadata']['soft_guard_authorized'])
        self.assertTrue(report['metadata']['coverage_audit_independent_conditions'])
        self.assertEqual(report['metadata']['calibration_condition_ids'], [20])
        self.assertEqual(report['metadata']['coverage_audit_condition_ids'], [23])
        self.assertEqual(report['metadata']['nonoverlap_sample_spacing_s'], 1.48)
        self.assertEqual(partition_summary['output_records']['calibration'], 10)
        self.assertEqual(partition_summary['output_records']['coverage_audit'], 10)
        self.assertEqual(partition_summary['rejection_counts']['overlapping_window'], 20)
        self.assertEqual(
            report['metadata']['predictor_artifact_sha256'],
            predictor_sha256,
        )
        self.assertTrue(persisted['metadata']['soft_guard_authorized'])
        self.assertTrue(report['groups'][0]['soft_guard_eligible'])

    def test_shadow_calibration_rejects_overlapping_condition_and_time_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_guard_shadow_overlap_') as temporary_directory:
            root = Path(temporary_directory)
            calibration_path = root / 'calibration.jsonl'
            audit_path = root / 'coverage_audit.jsonl'
            predictor_path = root / 'predictor.pt'
            _interface_sha, _reference_sha, identities = self._write_predictor_checkpoint(predictor_path)
            predictor_sha256 = hashlib.sha256(predictor_path.read_bytes()).hexdigest()
            combined_path = root / 'all_shadow.jsonl'
            self._write_shadow_file(
                calibration_path,
                20,
                start_s=230.0,
                count=10,
                predictor_sha256=predictor_sha256,
                actor_sha256='a' * 64,
                identity=identities['20'],
            )
            self._write_shadow_file(
                audit_path,
                20,
                start_s=400.0,
                count=10,
                predictor_sha256=predictor_sha256,
                actor_sha256='a' * 64,
                identity=identities['20'],
            )
            combined_path.write_bytes(calibration_path.read_bytes() + audit_path.read_bytes())
            calibration_path.unlink()
            audit_path.unlink()
            with self.assertRaisesRegex(ValueError, 'No eligible RL shadow records fall in the frozen coverage-audit partition'):
                split_shadow_audit(
                    input_jsonl=combined_path,
                    predictor_artifact=predictor_path,
                    interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                    calibration_output=calibration_path,
                    coverage_audit_output=audit_path,
                )


if __name__ == '__main__':
    unittest.main()
