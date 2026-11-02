from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from RL.build_b4_residual_bc_data import build
from RL.build_state20_control_config import B4_BASELINE_PATH, STATE_ARTIFACT_PATH
from RL.utils.constants import ACTION_VARIABLES, RAW_OBSERVATION_VARIABLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBuildB4ResidualBCData(unittest.TestCase):
    @staticmethod
    def _write_trace(path: Path, *, torque_offset: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['Time', *RAW_OBSERVATION_VARIABLES, *ACTION_VARIABLES]
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(3):
                writer.writerow({
                    'Time': index * 0.02,
                    'I_RotorAzimuth': 0.1 * index,
                    'I_RotorSpeed': 0.78 + 0.001 * index,
                    'I_YawError': 0.0,
                    'I_ElectricalTorque': 8_000_000.0 + torque_offset,
                    'I_GeneratorSpeed': 76.0,
                    'I_RealPower': 6_000_000.0,
                    'I_PitchPosition1': 0.08,
                    'I_NacelleForeAftAcceleration': 0.0,
                    'I_NacelleSideSideAcceleration': 0.0,
                    'I_AverageWindSpeed': 10.0,
                    'O_PcsTorqueDemand': 8_000_000.0 + torque_offset + index * 100.0,
                    'O_PitchPositionDemand1': 0.08 + index * 0.00001,
                })

    def test_explicit_overwrite_rebuilds_fixed_dataset_and_removes_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix='b4_residual_bc_overwrite_', dir=PROJECT_ROOT) as temp_dir:
            root = Path(temp_dir)
            train_root = root / 'raw'
            val_root = root / 'raw_infer'
            output_root = root / 'b4_centered_residual_bc_data'
            self._write_trace(train_root / 'data_windspeed3.csv', torque_offset=0.0)
            self._write_trace(val_root / 'data_windspeed3.csv', torque_offset=500.0)

            first = build(
                train_root=train_root,
                val_root=val_root,
                baseline_path=B4_BASELINE_PATH,
                output_root=output_root,
                residual_limit=(15_000.0, 0.0003),
                state_action_artifact_path=STATE_ARTIFACT_PATH,
            )
            self.assertEqual(first['train_rows'], 3)
            self.assertEqual(first['val_rows'], 3)

            stale_csv = output_root / 'train' / 'data_windspeed999.csv'
            stale_csv.write_text('stale\n', encoding='utf-8')
            stale_cache = output_root / '.rl_cache' / 'offline_bc' / 'stale.cache'
            stale_cache.parent.mkdir(parents=True)
            stale_cache.write_text('stale cache', encoding='utf-8')
            old_manifest = output_root / 'manifest.json'
            old_manifest.write_text('{"stale": true}\n', encoding='utf-8')

            rebuilt = build(
                train_root=train_root,
                val_root=val_root,
                baseline_path=B4_BASELINE_PATH,
                output_root=output_root,
                residual_limit=(15_000.0, 0.0003),
                state_action_artifact_path=STATE_ARTIFACT_PATH,
                overwrite=True,
            )

            persisted = json.loads(old_manifest.read_text(encoding='utf-8'))
            self.assertEqual(rebuilt['train_rows'], 3)
            self.assertEqual(rebuilt['val_rows'], 3)
            self.assertEqual(persisted['schema_version'], 'b4_centered_residual_bc_data_v2')
            self.assertFalse(stale_csv.exists())
            cache_root = output_root / '.rl_cache' / 'offline_bc'
            self.assertFalse(cache_root.exists() and any(cache_root.iterdir()))


if __name__ == '__main__':
    unittest.main()
