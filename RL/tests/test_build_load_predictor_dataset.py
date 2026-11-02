from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from RL.build_load_predictor_dataset import build_dataset
from RL.build_paired_pid_reference import build_paired_pid_reference
from RL.utils.state_action_artifact import STATE_AWARE_CONTROL_OBS_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ARTIFACT = PROJECT_ROOT / 'RL/data/state_action_envelope_3_to_25_v1.json'
INTERFACE = PROJECT_ROOT / 'RL/data/load_predictor_interface_v3.json'


class TestBuildLoadPredictorDataset(unittest.TestCase):
    @staticmethod
    def _write_raw_files(raw_root: Path) -> None:
        raw_root.mkdir()
        fields = [
            'Time',
            'I_RotorAzimuth',
            'I_RotorSpeed',
            'I_GeneratorSpeed',
            'I_RealPower',
            'I_ElectricalTorque',
            'I_PitchPosition1',
            'I_NacelleForeAftAcceleration',
            'I_NacelleSideSideAcceleration',
            'I_AverageWindSpeed',
            'O_PcsTorqueDemand',
            'O_PitchPositionDemand1',
            'Electrical power',
            'Blade root 1 My (Nm)',
            'Blade root 2 My (Nm)',
            'Blade root 3 My (Nm)',
        ]
        times = [index * 0.02 for index in range(75)]
        for condition_id in range(3, 26):
            path = raw_root / f'data_windspeed{condition_id}.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, timestamp in enumerate(times):
                    writer.writerow({
                        'Time': timestamp,
                        'I_RotorAzimuth': index * 0.1,
                        'I_RotorSpeed': 0.8,
                        'I_GeneratorSpeed': 0.8,
                        'I_RealPower': 15_000_000.0,
                        'I_ElectricalTorque': 20_000_000.0,
                        'I_PitchPosition1': 0.1,
                        'I_NacelleForeAftAcceleration': 0.0,
                        'I_NacelleSideSideAcceleration': 0.0,
                        'I_AverageWindSpeed': float(condition_id),
                        'O_PcsTorqueDemand': 20_000_000.0,
                        'O_PitchPositionDemand1': 0.1,
                        'Electrical power': 15_000_000.0,
                        'Blade root 1 My (Nm)': 50_000_000.0 + index,
                        'Blade root 2 My (Nm)': 49_000_000.0 + index,
                        'Blade root 3 My (Nm)': 48_000_000.0 + index,
                    })

    def test_builder_separates_realized_and_mismatch_qualified_zoh_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix='load_predictor_dataset_') as tmpdir:
            root = Path(tmpdir)
            raw_root = root / 'raw'
            self._write_raw_files(raw_root)
            paired_reference = build_paired_pid_reference(
                raw_root,
                time_window_start_s=0.0,
                time_window_end_s=1.48,
                project_root=PROJECT_ROOT,
            )
            paired_path = root / 'paired_reference.json'
            paired_path.write_text(json.dumps(paired_reference), encoding='utf-8')
            output_root = root / 'predictor_dataset'
            manifest = build_dataset(
                data_root=raw_root,
                paired_reference_path=paired_path,
                state_artifact_path=STATE_ARTIFACT,
                interface_path=INTERFACE,
                output_root=output_root,
                max_samples_per_shard=100,
            )

        self.assertEqual(manifest['schema_version'], 'load_predictor_dataset_v2')
        self.assertEqual(manifest['predictor_interface_schema_version'], 'causal_load_predictor_interface_v3')
        self.assertEqual(manifest['feature_dim'], 31)
        self.assertEqual(manifest['history_frames'], 50)
        self.assertEqual(manifest['plan_counts']['realized_plan'], 69)
        self.assertEqual(manifest['plan_counts']['zoh_candidate'], 46)
        self.assertEqual(len(manifest['shards']), 2)
        self.assertEqual(len(manifest['paired_reference_sha256']), 64)
        self.assertEqual(len(manifest['state_action_artifact_sha256']), 64)
        self.assertEqual(len(manifest['predictor_interface_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
