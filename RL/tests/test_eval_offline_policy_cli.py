from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from RL.agents import TD3Agent
from RL.eval_offline_policy import evaluate_checkpoint
from RL.utils.constants import ACTION_DIM, ACTION_VARIABLES, DATA_SCHEMA_VERSION, OBS_DIM, RAW_OBSERVATION_VARIABLES


class TestOfflinePolicyEvaluationCli(unittest.TestCase):
    def test_checkpoint_evaluation_uses_validation_data_without_udp_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix='offline_eval_cli_') as temporary_directory:
            root = Path(temporary_directory)
            columns = list(dict.fromkeys(RAW_OBSERVATION_VARIABLES + ACTION_VARIABLES))
            for split, condition_id in [('train', 3), ('val', 4)]:
                split_dir = root / 'data' / split
                split_dir.mkdir(parents=True)
                rows = []
                for index in range(3):
                    row = {column: 0.0 for column in columns}
                    row[ACTION_VARIABLES[0]] = 20_000_000.0 + index
                    row[ACTION_VARIABLES[1]] = 0.05
                    rows.append(row)
                pd.DataFrame(rows, columns=columns).to_csv(
                    split_dir / f'data_windspeed{condition_id}_window_01.csv',
                    index=False,
                )

            config = {
                'data_schema_version': DATA_SCHEMA_VERSION,
                'device': 'cpu',
                'data': {'root_dir': str(root / 'data'), 'train_split': 'train', 'eval_split': 'val'},
                'env': {
                    'action_low': [13_000_000.0, 0.0],
                    'action_high': [21_000_000.0, 0.33],
                    'action_limiter_owner': 'controller',
                    'action_delta_low': [-15_000.0, -0.0003],
                    'action_delta_high': [15_000.0, 0.0003],
                },
                'control': {
                    'baseline_action': [20_000_000.0, 0.05],
                    'residual_limit': [1_000_000.0, 0.05],
                    'safe_action_low': [13_000_000.0, 0.0],
                    'safe_action_high': [21_000_000.0, 0.33],
                    'ood_threshold': 100.0,
                },
                'agent': {'hidden_dims': [8], 'td3_bc_enabled': False},
                'bc': {'split': 'train', 'cache_enabled': False},
                'offline_eval': {'split': 'val', 'normalizer_split': 'train', 'cache_enabled': False, 'batch_size': 2},
            }
            agent = TD3Agent(OBS_DIM, ACTION_DIM, config['agent'], torch.device('cpu'))
            agent.set_actor_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            agent.set_critic_obs_normalizer(torch.zeros(OBS_DIM), torch.ones(OBS_DIM))
            checkpoint_path = root / 'checkpoint.pt'
            torch.save(
                {
                    'agent': agent.state_dict(),
                    'config': config,
                    'data_schema_version': DATA_SCHEMA_VERSION,
                },
                checkpoint_path,
            )

            with patch('RL.envs.wind_turbine_env.socket.socket', side_effect=AssertionError('UDP must not open')):
                result = evaluate_checkpoint(
                    checkpoint_path,
                    config,
                    root / 'output',
                    device=torch.device('cpu'),
                )

            self.assertEqual(result['mode'], 'offline_behavior_safety_v1')
            self.assertEqual(result['evaluation_split'], 'val')
            self.assertEqual(result['summary']['sample_count'], 3.0)
            self.assertTrue((root / 'output' / 'offline_policy_eval.csv').is_file())
            self.assertTrue((root / 'output' / 'offline_policy_eval_summary.json').is_file())
            self.assertTrue((root / 'output' / 'offline_policy_eval_run.json').is_file())


if __name__ == '__main__':
    unittest.main()