from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from RL.data.offline_dataset import OfflineBCDataset, OfflineWindow
from RL.tests.test_fake_bladed_integration import BLADED_CSV_HEADER
from RL.utils.constants import ACTION_VARIABLES, OBS_DIM, RAW_OBSERVATION_VARIABLES
from RL.utils.offline_policy_eval import evaluate_offline_policy, write_offline_policy_evaluation
from RL.utils.safe_residual_control import SafeResidualController


class _DeterministicResidualAgent:
    def select_action(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return torch.ones((observations.shape[0], 2), dtype=torch.float32, device=observations.device)


class _WindowDataset:
    def __init__(self) -> None:
        self.obs = np.zeros((4, OBS_DIM), dtype=np.float32)
        self.actions = np.asarray(
            [
                [20_000_000.0, 0.05],
                [20_000_000.0, 0.05],
                [20_000_000.0, 0.05],
                [20_000_000.0, 0.05],
            ],
            dtype=np.float32,
        )
        self.windows = (
            OfflineWindow(0, 'data_windspeed3_window_01.csv', 0, 2, 3),
            OfflineWindow(1, 'data_windspeed4_window_01.csv', 2, 4, 4),
        )


class TestOfflinePolicyEvaluation(unittest.TestCase):
    def _controller(self) -> SafeResidualController:
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM, dtype=np.float32), np.ones(OBS_DIM, dtype=np.float32))
        return controller

    def test_evaluator_resets_projection_at_window_boundaries_and_writes_behavior_only_report(self) -> None:
        evaluation = evaluate_offline_policy(
            _DeterministicResidualAgent(),
            _WindowDataset(),
            self._controller(),
            torch.device('cpu'),
            batch_size=2,
        )

        self.assertEqual(evaluation.summary['sample_count'], 4.0)
        self.assertEqual(evaluation.summary['condition_3_sample_count'], 2.0)
        self.assertEqual(evaluation.summary['condition_4_sample_count'], 2.0)
        self.assertEqual(evaluation.summary['overall_action_rate_limited_fraction'], 1.0)
        self.assertNotIn('reward_mean', evaluation.summary)
        self.assertEqual(evaluation.records[0]['sent_action_torque'], evaluation.records[2]['sent_action_torque'])

        with tempfile.TemporaryDirectory(prefix='offline_policy_eval_') as temporary_directory:
            write_offline_policy_evaluation(temporary_directory, evaluation)
            output = Path(temporary_directory)
            self.assertTrue((output / 'offline_policy_eval.csv').is_file())
            self.assertTrue((output / 'offline_policy_eval_summary.json').is_file())

    def test_dataset_preserves_csv_window_boundaries_condition_labels_and_previous_actions(self) -> None:
        with tempfile.TemporaryDirectory(prefix='offline_dataset_windows_') as temporary_directory:
            root = Path(temporary_directory)
            train_dir = root / 'train'
            train_dir.mkdir()
            columns = list(dict.fromkeys(RAW_OBSERVATION_VARIABLES + ACTION_VARIABLES))
            for condition_id in [3, 4]:
                rows = []
                for index in range(2):
                    row = {column: 0.0 for column in columns}
                    row[ACTION_VARIABLES[0]] = 20_000_000.0 + index
                    row[ACTION_VARIABLES[1]] = 0.05 + index * 0.001
                    rows.append(row)
                pd.DataFrame(rows, columns=columns).to_csv(
                    train_dir / f'data_windspeed{condition_id}_window_01.csv',
                    index=False,
                )

            dataset = OfflineBCDataset(str(root), split='train', cache_enabled=False)

        self.assertEqual([window.condition_id for window in dataset.windows], [3, 4])
        self.assertEqual([(window.start_index, window.end_index) for window in dataset.windows], [(0, 2), (2, 4)])
        np.testing.assert_allclose(dataset.previous_actions[0], dataset.actions[0])
        np.testing.assert_allclose(dataset.previous_actions[1], dataset.actions[0])


if __name__ == '__main__':
    unittest.main()