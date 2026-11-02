from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from RL.utils.paired_pid_reference import build_paired_pid_reference
from RL.data.load_predictor_dataset import LoadPredictorDataset, estimate_normalization
from RL.data.load_predictor_dataset import HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION
from RL.train_load_predictor import (
    _ShardBatchSampler,
    _collate,
    _resolve_time_partitions,
    build_time_partitions,
    build_condition_partitions,
    train_predictor,
)
from RL.utils.load_predictor_artifact import load_candidate_conditioned_predictor
from RL.utils.load_predictor_contract import LoadPredictorInterface


class TestTrainLoadPredictor(unittest.TestCase):
    @staticmethod
    def _write_dataset(root: Path) -> None:
        interface_path = Path('RL/data/load_predictor_interface_v3.json').resolve()
        interface = LoadPredictorInterface.from_path(interface_path)
        max_frames = max(interface.horizon_frames)
        samples_per_condition = 101
        condition_ids = list(range(3, 26))
        conditions = np.repeat(np.asarray(condition_ids, dtype=np.int32), samples_per_condition)
        sample_count = len(conditions)
        history = np.zeros((sample_count, interface.history_frames, 31), dtype=np.float32)
        candidate = np.zeros((sample_count, max_frames, 2), dtype=np.float32)
        actual_plan = np.zeros_like(candidate)
        future_response = np.ones((sample_count, 2), dtype=np.float32)
        future_ratio = np.ones((sample_count, 3), dtype=np.float32)
        plan_kind = np.asarray(['zoh_candidate'] * len(conditions), dtype='U32')
        timestamps = np.tile(np.arange(samples_per_condition, dtype=np.float64) / 10.0, len(condition_ids))
        for index, condition_id in enumerate(conditions):
            history[index, :, 0] = float(condition_id)
            candidate[index, :, 0] = 20_000_000.0
        raw_root = root / 'raw'
        raw_root.mkdir()
        raw_fields = [
            'Time',
            'I_AverageWindSpeed',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        for condition_id in condition_ids:
            with (raw_root / f'data_windspeed{condition_id}.csv').open(
                'w', encoding='utf-8', newline=''
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=raw_fields)
                writer.writeheader()
                for index, timestamp_s in enumerate(np.arange(samples_per_condition, dtype=np.float64) / 10.0):
                    writer.writerow({
                        'Time': timestamp_s,
                        'I_AverageWindSpeed': float(condition_id),
                        'Electrical power': 15_000_000.0,
                        'Blade root 1 My (Nm)': 50_000_000.0 + index,
                        'Blade root 2 My (Nm)': 49_000_000.0 + index,
                        'Blade root 3 My (Nm)': 48_000_000.0 + index,
                    })
        paired_reference_path = root / 'paired_reference.json'
        paired_reference = build_paired_pid_reference(
            raw_root,
            time_window_start_s=0.0,
            time_window_end_s=10.0,
            project_root=Path(__file__).resolve().parents[2],
        )
        paired_reference_path.write_text(json.dumps(paired_reference), encoding='utf-8')
        entries = {str(entry['condition_id']): entry for entry in paired_reference['references']}
        identities: dict[str, dict[str, object]] = {}
        raw_sha_by_condition: dict[str, str] = {}
        for condition_id in condition_ids:
            entry = entries[str(condition_id)]
            raw_path = str((raw_root / f'data_windspeed{condition_id}.csv').resolve())
            raw_sha = str(entry['pid_raw_sha256'])
            trace = entry['canonical_wind_trace']
            raw_sha_by_condition[str(condition_id)] = raw_sha
            identities[str(condition_id)] = {
                'scenario_id': entry['scenario_id'],
                'condition_id': condition_id,
                'pid_raw_csv_path': raw_path,
                'pid_raw_sha256': raw_sha,
                'raw_source_sha256': raw_sha,
                'time_window_start_s': entry['time_window_start_s'],
                'time_window_end_s': entry['time_window_end_s'],
                'expected_trace_signature': trace['sha256'],
                'expected_trace_sample_count': trace['sample_count'],
                'time_signature_quantization_s': trace['time_signature_quantization_s'],
                'wind_signature_quantization_mps': trace['wind_signature_quantization_mps'],
            }
        shard_path = root / 'shard_00000.npz'
        np.savez_compressed(
            shard_path,
            history=history,
            model_input_plan=candidate,
            actual_future_action_plan=actual_plan,
            future_actual_torque_pitch=future_response,
            future_peak_ratio=future_ratio,
            timestamp_s=timestamps,
            condition_id=conditions,
            horizon_s=np.full(len(conditions), interface.horizons_s[0], dtype=np.float32),
            horizon_frames=np.full(len(conditions), interface.horizon_frames[0], dtype=np.int32),
            plan_kind=plan_kind,
            optimization_state=np.asarray(['above_rated'] * len(conditions), dtype='U64'),
            plan_mismatch=np.zeros(len(conditions), dtype=np.float32),
            zoh_calibration_eligible=np.ones(len(conditions), dtype=np.bool_),
            pid_raw_sha256=np.asarray([
                identities[str(int(condition_id))]['pid_raw_sha256'] for condition_id in conditions
            ], dtype='U64'),
            pid_raw_csv_path=np.asarray([
                identities[str(int(condition_id))]['pid_raw_csv_path'] for condition_id in conditions
            ], dtype='U512'),
            raw_source_sha256=np.asarray([
                identities[str(int(condition_id))]['raw_source_sha256'] for condition_id in conditions
            ], dtype='U64'),
        )
        manifest = {
            'schema_version': 'load_predictor_dataset_v2',
            'predictor_interface_schema_version': interface.schema_version,
            'predictor_interface_path': str(interface_path),
            'predictor_interface_sha256': hashlib.sha256(interface_path.read_bytes()).hexdigest(),
            'history_frames': interface.history_frames,
            'horizons_s': list(interface.horizons_s),
            'feature_dim': 31,
            'source_conditions': condition_ids,
            'time_window_start_s': 0.0,
            'time_window_end_s': 10.0,
            'paired_reference_path': str(paired_reference_path.resolve()),
            'paired_reference_sha256': hashlib.sha256(paired_reference_path.read_bytes()).hexdigest(),
            'state_action_artifact_path': 'RL/data/state_action_envelope_3_to_25_v1.json',
            'state_action_artifact_sha256': hashlib.sha256(
                Path('RL/data/state_action_envelope_3_to_25_v1.json').read_bytes()
            ).hexdigest(),
            'paired_identity_by_condition': identities,
            'raw_source_sha256_by_condition': raw_sha_by_condition,
            'plan_counts': {'realized_plan': 0, 'zoh_candidate': len(conditions)},
            'shards': [{
                'path': shard_path.name,
                'samples': len(conditions),
                'sha256': hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            }],
        }
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_condition_partitions_are_disjoint_complete_and_deterministic(self) -> None:
        partitions = build_condition_partitions(list(range(3, 26)))
        flattened = [value for values in partitions.values() for value in values]
        self.assertEqual(set(flattened), set(range(3, 26)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(partitions, build_condition_partitions(list(reversed(range(3, 26)))))
        self.assertEqual(len(partitions['train']), 14)
        self.assertEqual(len(partitions['validation']), 3)
        self.assertEqual(len(partitions['calibration']), 3)
        self.assertEqual(len(partitions['coverage_audit']), 3)
        with self.assertRaisesRegex(ValueError, 'four distinct'):
            build_condition_partitions([1, 2, 3])

    def test_old_dataset_manifest_reads_reference_window_from_a_bounded_prefix(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_predictor_bounded_reference_metadata_') as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / 'large_reference.json'
            reference_path.write_bytes(
                b'{"metadata":{"time_window_start_s":0.0,"time_window_end_s":600.0},"references":['
                + b' ' * (2 * 1024 * 1024)
            )
            dataset = SimpleNamespace(manifest={'paired_reference_path': str(reference_path)})
            interface = LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v3.json')

            partitions = _resolve_time_partitions(dataset, interface, override=None)

        self.assertEqual(partitions['train'][0], 0.0)
        self.assertEqual(partitions['coverage_audit'][1], 600.0 + interface.step_duration_s)
        self.assertGreaterEqual(
            partitions['validation'][0] - partitions['train'][1] + 1.0e-12,
            1.48,
        )

    def test_lazy_dataset_returns_plan_and_normalization_uses_bounded_reservoir(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_predictor_training_dataset_') as temporary_directory:
            root = Path(temporary_directory)
            self._write_dataset(root)
            interface = LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v3.json')
            dataset = LoadPredictorDataset(root, interface=interface)
            item = dataset[0]
            self.assertEqual(item['candidate_plan'].shape, (max(interface.horizon_frames), 2))
            batch = _collate([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch['candidate_plan'].shape), (2, max(interface.horizon_frames), 2))
            self.assertEqual(tuple(batch['plan_kind_index'].shape), (2,))
            normalization = estimate_normalization(dataset, max_samples=3, seed=19)

        self.assertEqual(normalization['sample_count_used'], 3)
        self.assertEqual(normalization['history_window_capacity'], 3)
        self.assertEqual(normalization['history_window_bytes'], 3 * interface.history_frames * 31 * 4)
        self.assertEqual(normalization['reservoir_seed'], 19)
        self.assertEqual(
            normalization['history_iqr_floor_schema'],
            HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION,
        )
        self.assertTrue(np.all(np.asarray(normalization['history_iqr'][20:28]) >= 1.0))

    def test_training_batch_sampler_keeps_each_batch_in_one_lazy_shard(self) -> None:
        class FakeDataset:
            _indices = [
                (0, index) for index in range(7)
            ] + [
                (1, index) for index in range(9)
            ]

        sampler = _ShardBatchSampler(FakeDataset(), batch_size=4, seed=17)
        first_epoch = list(sampler)
        second_epoch = list(sampler)
        self.assertEqual(len(first_epoch), 5)
        self.assertEqual(sorted(index for batch in first_epoch for index in batch), list(range(16)))
        for batch in first_epoch:
            shard_ids = {FakeDataset._indices[index][0] for index in batch}
            self.assertEqual(len(shard_ids), 1)
        self.assertNotEqual(first_epoch, second_epoch)

    def test_shadow_time_partitions_reserve_separate_calibration_and_audit_windows(self) -> None:
        interface = LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v3.json')
        partitions = build_time_partitions(
            0.0,
            600.0,
            interface=interface,
        )
        spacing = (interface.history_frames - 1 + max(interface.horizon_frames)) * interface.step_duration_s
        self.assertGreaterEqual(partitions['calibration'][1] - partitions['calibration'][0], 150.0)
        self.assertGreaterEqual(partitions['coverage_audit'][1] - partitions['coverage_audit'][0], 150.0)
        self.assertGreaterEqual(partitions['validation'][0] - partitions['train'][1], spacing - 1.0e-12)
        self.assertGreaterEqual(partitions['calibration'][0] - partitions['validation'][1], spacing - 1.0e-12)
        self.assertGreaterEqual(partitions['coverage_audit'][0] - partitions['calibration'][1], spacing - 1.0e-12)

    def test_training_writes_candidate_conditioned_checkpoint_and_disjoint_audits(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_predictor_training_run_') as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / 'dataset'
            dataset_root.mkdir()
            self._write_dataset(dataset_root)
            output = root / 'predictor.pt'
            summary = train_predictor(
                dataset_root=dataset_root,
                interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                output=output,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-3,
                device='cpu',
                normalization_max_samples=4,
                time_partitions_s={
                    'train': (0.0, 3.0),
                    'validation': (4.5, 5.5),
                    'calibration': (7.0, 8.0),
                    'coverage_audit': (9.5, 10.02),
                },
            )
            checkpoint = torch.load(output, map_location='cpu', weights_only=False)
            summary_path = output.with_name(f'{output.stem}.summary.json')
            self.assertTrue(summary_path.is_file())
            loaded = load_candidate_conditioned_predictor(
                output,
                interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                expected_checkpoint_sha256=summary['checkpoint_sha256'],
                action_low=[0.0, 0.0],
                action_high=[21_586_451.33303, 0.43],
                device=torch.device('cpu'),
                expected_dataset_manifest_sha256=checkpoint['dataset_manifest_sha256'],
                expected_time_partitions_s=checkpoint['time_partitions_s'],
            )
            with self.assertRaisesRegex(ValueError, 'requires the frozen time partitions'):
                load_candidate_conditioned_predictor(
                    output,
                    interface_path=Path('RL/data/load_predictor_interface_v3.json'),
                    expected_checkpoint_sha256=summary['checkpoint_sha256'],
                    action_low=[0.0, 0.0],
                    action_high=[21_586_451.33303, 0.43],
                    device=torch.device('cpu'),
                    expected_dataset_manifest_sha256=checkpoint['dataset_manifest_sha256'],
                )

        self.assertEqual(checkpoint['schema_version'], 'load_predictor_checkpoint_v4_candidate_conditioned')
        partitions = checkpoint['condition_partitions']
        flattened = [condition for values in partitions.values() for condition in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertFalse(checkpoint['guard_calibration_authorized'])
        self.assertGreater(checkpoint['plan_counts_by_split']['train']['zoh_candidate'], 0)
        self.assertGreater(checkpoint['plan_counts_by_split']['validation']['zoh_candidate'], 0)
        self.assertGreater(checkpoint['plan_counts_by_split']['calibration']['zoh_candidate'], 0)
        self.assertGreater(checkpoint['plan_counts_by_split']['coverage_audit']['zoh_candidate'], 0)
        self.assertEqual(checkpoint['temporal_purge_s'], 1.48)
        time_ranges = checkpoint['time_partitions_s']
        self.assertLessEqual(time_ranges['train'][1], time_ranges['validation'][0])
        self.assertLessEqual(time_ranges['validation'][1], time_ranges['calibration'][0])
        self.assertLessEqual(time_ranges['calibration'][1], time_ranges['coverage_audit'][0])
        self.assertGreater(summary['updates'], 0)
        self.assertGreater(summary['validation_response_normalized_mae'], 0.0)
        self.assertGreater(summary['validation_peak_ratio_normalized_mae'], 0.0)
        self.assertEqual(summary['best_validation_epoch'], checkpoint['best_validation_epoch'])
        self.assertEqual(
            summary['calibration_prediction_audit']['sample_count'],
            checkpoint['split_sample_counts']['calibration'],
        )
        self.assertEqual(
            summary['coverage_audit_prediction_audit']['sample_count'],
            checkpoint['split_sample_counts']['coverage_audit'],
        )
        self.assertFalse(summary['calibration_prediction_audit']['guard_calibration_authorized'])
        self.assertFalse(summary['coverage_audit_prediction_audit']['guard_calibration_authorized'])
        self.assertGreaterEqual(len(summary['coverage_audit_prediction_audit']['groups']), 1)
        self.assertEqual(loaded.feature_dim, 31)
        self.assertEqual(loaded.checkpoint_sha256, summary['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
