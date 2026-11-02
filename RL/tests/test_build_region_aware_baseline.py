from __future__ import annotations

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from RL.build_region_aware_baseline import main
from RL.utils.raw_dataset_contract import manifest_sha256
from RL.utils.constants import ACTION_VARIABLES, DATA_SCHEMA_VERSION, DISTURBANCE_VARIABLES, REAL_POWER_VARIABLE
from RL.utils.operating_state import OPERATING_STATE_CONTRACT_VERSION
from RL.utils.region_aware_baseline import REGION_AWARE_BASELINE_SCHEMA_VERSION, RegionAwareJointBaseline


class TestBuildRegionAwareBaseline(unittest.TestCase):
    @staticmethod
    def _write_raw_files(raw_dir: Path) -> str:
        raw_dir.mkdir()
        fieldnames = [
            'Time',
            DISTURBANCE_VARIABLES[0],
            'I_RotorSpeed',
            'I_GeneratorSpeed',
            REAL_POWER_VARIABLE,
            *ACTION_VARIABLES,
        ]
        audits = []
        for condition_id in range(3, 26):
            rows = []
            for row_index, timestamp in enumerate((0.0, 0.02, 0.04, 0.06, 600.0)):
                rotor_speed = min(0.82, 0.45 + condition_id * 0.025 + row_index * 0.001)
                power = min(15_000_000.0, condition_id * 1_050_000.0 + row_index * 20_000.0)
                torque = min(20_000_000.0, condition_id * 1_050_000.0 + row_index * 5_000.0)
                pitch = max(0.0, (condition_id - 10) * 0.012 + row_index * 0.0001)
                rows.append({
                    'Time': timestamp,
                    DISTURBANCE_VARIABLES[0]: condition_id + 0.1 + row_index * 0.01,
                    'I_RotorSpeed': rotor_speed,
                    'I_GeneratorSpeed': rotor_speed + 0.002,
                    REAL_POWER_VARIABLE: power,
                    ACTION_VARIABLES[0]: torque,
                    ACTION_VARIABLES[1]: pitch,
                })
            path = raw_dir / f'data_windspeed{condition_id}.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            audits.append({
                'condition_id': condition_id,
                'filename': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return manifest_sha256(audits)

    @staticmethod
    def _write_provenance(path: Path, manifest_sha256: str) -> None:
        path.write_text(json.dumps({
            'schema_version': 'pid_reference_provenance_v1',
            'source_manifest': {'manifest_sha256': manifest_sha256},
            'decision': {
                'native_pid_demand_available': False,
                'udp_action_contract': 'recorded_pid_demand',
                'baseline_reference_path': 'region_aware_surrogate_required',
            },
        }), encoding='utf-8')

    @staticmethod
    def _run(raw_dir: Path, provenance: Path, output: Path) -> None:
        argv = [
            'build_region_aware_baseline.py',
            '--data-root', str(raw_dir),
            '--provenance', str(provenance),
            '--output', str(output),
            '--min-region-transitions', '2',
        ]
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            main()

    def test_builds_joint_causal_models_from_all_raw_conditions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='region_baseline_builder_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            manifest_sha256 = self._write_raw_files(raw_dir)
            provenance = root / 'provenance.json'
            output = root / 'region_baseline.json'
            self._write_provenance(provenance, manifest_sha256)

            self._run(raw_dir, provenance, output)

            document = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(document['schema_version'], REGION_AWARE_BASELINE_SCHEMA_VERSION)
            metadata = document['metadata']
            self.assertEqual(metadata['data_schema_version'], DATA_SCHEMA_VERSION)
            self.assertEqual(metadata['source_role'], 'raw_pid_calibration')
            self.assertEqual(metadata['sampling_strategy'], 'all_valid_causal_transitions')
            self.assertEqual(metadata['files_used'], 23)
            self.assertEqual(metadata['rows_used'], 115)
            self.assertEqual(metadata['transitions_used'], 92)
            self.assertEqual(metadata['input_manifest_sha256'], manifest_sha256)
            self.assertEqual(
                metadata['operating_state_contract']['contract_version'],
                OPERATING_STATE_CONTRACT_VERSION,
            )
            control = document['control']
            self.assertNotIn('wind_speed_baseline', control)
            baseline = control['region_aware_joint_baseline']
            self.assertEqual(baseline['target'], 'next_realized_action_delta')
            self.assertEqual(baseline['action_units'], ['N m', 'rad'])
            self.assertTrue(baseline['causal'])
            self.assertEqual(baseline['future_feature_lookahead_steps'], 0)
            self.assertEqual(set(baseline['models']), {'below_rated', 'transition', 'above_rated'})
            for model in baseline['models'].values():
                self.assertGreaterEqual(model['sample_count'], 2)
                self.assertEqual(model['relation'], 'joint_next_action_delta')
                self.assertEqual(np.asarray(model['coefficient_matrix']).shape, (6, 2))
                self.assertEqual(np.asarray(model['joint_residual_covariance']).shape, (2, 2))

            runtime = RegionAwareJointBaseline.from_path(output, step_duration_s=0.02)
            self.assertEqual(runtime.input_manifest_sha256, manifest_sha256)

    def test_rejects_provenance_for_a_different_raw_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix='region_baseline_builder_') as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / 'raw'
            self._write_raw_files(raw_dir)
            provenance = root / 'provenance.json'
            self._write_provenance(provenance, 'f' * 64)

            with self.assertRaisesRegex(ValueError, 'manifest'):
                self._run(raw_dir, provenance, root / 'region_baseline.json')


if __name__ == '__main__':
    unittest.main()