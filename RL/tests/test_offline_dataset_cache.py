from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from RL.data.offline_dataset import OfflineBCDataset
from RL.utils.constants import ACTION_VARIABLES, DATA_SCHEMA_VERSION, RAW_OBSERVATION_VARIABLES


class TestOfflineBCDatasetCache(unittest.TestCase):

    def test_precomputed_action_targets_are_returned_as_third_tensor(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_bc_precomputed_') as tmpdir:
            split_dir = Path(tmpdir) / 'train'
            split_dir.mkdir()
            path = Path(tmpdir) / 'train' / 'data_windspeed8.csv'
            columns = list(dict.fromkeys(RAW_OBSERVATION_VARIABLES + ACTION_VARIABLES))
            frame = pd.DataFrame([{column: 1.0 for column in columns} for _ in range(3)])
            frame['B4 residual torque target'] = [0.1, 0.2, 0.3]
            frame['B4 residual pitch target'] = [-0.1, -0.2, -0.3]
            frame.to_csv(path, index=False)

            dataset = OfflineBCDataset(
                tmpdir,
                cache_enabled=False,
                precomputed_action_target_columns=[
                    'B4 residual torque target',
                    'B4 residual pitch target',
                ],
            )

            _obs, _actions, targets = dataset[1]
            np.testing.assert_allclose(targets.numpy(), [0.2, -0.2])

    def test_reuses_cache_without_reading_csv_again(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_bc_cache_') as tmpdir:
            split_dir = Path(tmpdir) / 'train'
            split_dir.mkdir()
            columns = list(dict.fromkeys(RAW_OBSERVATION_VARIABLES + ACTION_VARIABLES))
            pd.DataFrame([{column: 1.0 for column in columns} for _ in range(3)]).to_csv(split_dir / 'samples.csv', index=False)

            first = OfflineBCDataset(tmpdir, cache_enabled=True)
            with patch('RL.data.offline_dataset.pd.read_csv', side_effect=AssertionError('cache miss')):
                second = OfflineBCDataset(tmpdir, cache_enabled=True)

            self.assertEqual(len(first), len(second))
            self.assertEqual(first.obs_median.tolist(), second.obs_median.tolist())

    def test_rebuilds_cache_with_wrong_data_schema_version(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_bc_cache_schema_') as tmpdir:
            split_dir = Path(tmpdir) / 'train'
            split_dir.mkdir()
            columns = list(dict.fromkeys(RAW_OBSERVATION_VARIABLES + ACTION_VARIABLES))
            pd.DataFrame([{column: 1.0 for column in columns} for _ in range(3)]).to_csv(split_dir / 'samples.csv', index=False)

            first = OfflineBCDataset(tmpdir, cache_enabled=True)
            cache_paths = first._cache_paths(Path(tmpdir), 'train', None, None, None)
            metadata = json.loads(cache_paths['metadata'].read_text(encoding='utf-8'))
            self.assertEqual(metadata['data_schema_version'], DATA_SCHEMA_VERSION)
            metadata['data_schema_version'] = 'legacy_schema'
            cache_paths['metadata'].write_text(json.dumps(metadata), encoding='utf-8')

            with patch('RL.data.offline_dataset.pd.read_csv', wraps=pd.read_csv) as read_csv:
                second = OfflineBCDataset(tmpdir, cache_enabled=True)

            self.assertGreater(read_csv.call_count, 0)
            self.assertEqual(len(first), len(second))
            rebuilt_metadata = json.loads(cache_paths['metadata'].read_text(encoding='utf-8'))
            self.assertEqual(rebuilt_metadata['data_schema_version'], DATA_SCHEMA_VERSION)


if __name__ == '__main__':
    unittest.main()