from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from RL.build_pid_peak_reference import BLADE_ROOT_MY_FIELDS, SIM_TIME_FIELD, WIND_SPEED_FIELD, build_peak_reference
from RL.utils.constants import DATA_SCHEMA_VERSION


class TestPidPeakReference(unittest.TestCase):
    def _write_window(self, path: Path, wind_speed: float, blade_rows: list[tuple[float, float, float]]) -> None:
        with path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[SIM_TIME_FIELD, WIND_SPEED_FIELD, *BLADE_ROOT_MY_FIELDS])
            writer.writeheader()
            for index, blade_values in enumerate(blade_rows):
                writer.writerow({
                    SIM_TIME_FIELD: float(index),
                    WIND_SPEED_FIELD: wind_speed,
                    BLADE_ROOT_MY_FIELDS[0]: blade_values[0],
                    BLADE_ROOT_MY_FIELDS[1]: blade_values[1],
                    BLADE_ROOT_MY_FIELDS[2]: blade_values[2],
                })

    def test_uses_per_window_abs_peaks_and_nominal_filename_wind(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_peak_reference_') as tmpdir:
            root = Path(tmpdir)
            train = root / 'train'
            train.mkdir()
            self._write_window(
                train / 'data_windspeed3_window_01.csv',
                3.2,
                [(10.0, -20.0, 30.0), (-12.0, 18.0, -34.0)],
            )
            self._write_window(
                train / 'data_windspeed3_window_02.csv',
                3.1,
                [(16.0, -22.0, 28.0), (-14.0, 24.0, -36.0)],
            )
            self._write_window(
                train / 'data_windspeed10_window_01.csv',
                10.3,
                [(40.0, -50.0, 60.0), (-45.0, 45.0, -55.0)],
            )

            document = build_peak_reference(
                data_root=root,
                splits=['train'],
                wind_bin_width=0.5,
                peak_quantile=0.5,
            )

        rows = document['reward']['blade_root_peak_reference']
        self.assertEqual([row['wind_speed'] for row in rows], [3.0, 10.0])
        self.assertEqual(rows[0]['window_count'], 2)
        self.assertEqual(rows[0]['peak_abs_nm'], [14.0, 22.0, 35.0])
        self.assertEqual(rows[1]['peak_abs_nm'], [45.0, 50.0, 60.0])
        self.assertEqual(document['metadata']['data_schema_version'], DATA_SCHEMA_VERSION)
        self.assertEqual(document['metadata']['aggregation'], 'per_window_abs_peak_then_per_condition_quantile')

    def test_fixed_duration_mode_records_real_simulation_ranges(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_peak_reference_fixed_duration_') as tmpdir:
            root = Path(tmpdir)
            train = root / 'train'
            train.mkdir()
            self._write_window(
                train / 'data_windspeed3_window_01.csv',
                3.0,
                [(10.0, 20.0, 30.0), (11.0, 22.0, 32.0), (12.0, 24.0, 34.0)],
            )
            self._write_window(
                train / 'data_windspeed10_window_01.csv',
                10.0,
                [(40.0, 50.0, 60.0), (45.0, 55.0, 65.0), (42.0, 52.0, 62.0)],
            )

            document = build_peak_reference(
                data_root=root,
                splits=['train'],
                target_train_sim_s=2.0,
            )

        metadata = document['metadata']
        self.assertEqual(metadata['target_train_sim_s'], 2.0)
        self.assertEqual(metadata['window_duration_mode'], 'non_overlapping_contiguous_sim_time_windows')
        self.assertEqual(metadata['fixed_duration_window_count'], 2)
        self.assertEqual(
            [(row['start_sim_time_s'], row['end_sim_time_s']) for row in metadata['fixed_duration_source_windows']],
            [(0.0, 2.0), (0.0, 2.0)],
        )

    def test_fixed_duration_mode_rejects_short_traces(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_peak_reference_short_trace_') as tmpdir:
            root = Path(tmpdir)
            train = root / 'train'
            train.mkdir()
            self._write_window(train / 'data_windspeed3_window_01.csv', 3.0, [(1.0, 1.0, 1.0)])
            self._write_window(train / 'data_windspeed10_window_01.csv', 10.0, [(1.0, 1.0, 1.0)])

            with self.assertRaisesRegex(RuntimeError, 'No CSV contains a contiguous 600-second'):
                build_peak_reference(data_root=root, splits=['train'], target_train_sim_s=600.0)


if __name__ == '__main__':
    unittest.main()