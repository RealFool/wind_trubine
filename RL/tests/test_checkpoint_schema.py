from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from RL.train_rl import _export_bc_checkpoint, _require_data_schema_version, _try_load_bc_warmstart_checkpoint
from RL.utils.constants import DATA_SCHEMA_VERSION, OBS_DIM


class TestCheckpointSchema(unittest.TestCase):
    def _agent(self) -> MagicMock:
        agent = MagicMock()
        agent.actor.obs_median = torch.zeros(OBS_DIM)
        return agent

    def test_rejects_same_dimension_checkpoint_without_current_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_checkpoint_schema_') as tmpdir:
            checkpoint_path = Path(tmpdir) / 'legacy.pt'
            torch.save({'agent': {'actor.obs_median': torch.zeros(OBS_DIM)}}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, 'BC checkpoint schema is incompatible'):
                _try_load_bc_warmstart_checkpoint(self._agent(), checkpoint_path, torch.device('cpu'), DATA_SCHEMA_VERSION)

    def test_loads_checkpoint_marked_with_current_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_checkpoint_schema_') as tmpdir:
            checkpoint_path = Path(tmpdir) / 'current.pt'
            state = {'actor': {'obs_median': torch.zeros(OBS_DIM), 'obs_iqr': torch.ones(OBS_DIM)}}
            torch.save({'agent': state, 'data_schema_version': DATA_SCHEMA_VERSION}, checkpoint_path)
            agent = self._agent()

            self.assertTrue(_try_load_bc_warmstart_checkpoint(agent, checkpoint_path, torch.device('cpu'), DATA_SCHEMA_VERSION))
            agent.load_state_dict.assert_called_once()
            loaded_state = agent.load_state_dict.call_args.args[0]
            self.assertEqual(set(loaded_state), set(state))
            self.assertTrue(torch.equal(loaded_state['actor']['obs_median'], state['actor']['obs_median']))

    def test_rejects_current_schema_checkpoint_with_wrong_normalizer_dimension(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_checkpoint_schema_') as tmpdir:
            checkpoint_path = Path(tmpdir) / 'wrong_dimension.pt'
            state = {'actor': {'obs_median': torch.zeros(OBS_DIM - 1), 'obs_iqr': torch.ones(OBS_DIM - 1)}}
            torch.save({'agent': state, 'data_schema_version': DATA_SCHEMA_VERSION}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, 'observation dimension is incompatible'):
                _try_load_bc_warmstart_checkpoint(self._agent(), checkpoint_path, torch.device('cpu'), DATA_SCHEMA_VERSION)

    def test_rejects_checkpoint_with_wrong_control_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_checkpoint_control_') as tmpdir:
            checkpoint_path = Path(tmpdir) / 'wrong_control.pt'
            state = {'actor': {'obs_median': torch.zeros(OBS_DIM), 'obs_iqr': torch.ones(OBS_DIM)}}
            torch.save(
                {
                    'agent': state,
                    'data_schema_version': DATA_SCHEMA_VERSION,
                    'control_fingerprint': 'four-wind-baseline',
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, 'control contract is incompatible'):
                _try_load_bc_warmstart_checkpoint(
                    self._agent(),
                    checkpoint_path,
                    torch.device('cpu'),
                    DATA_SCHEMA_VERSION,
                    expected_control_fingerprint='23-wind-baseline',
                )

    def test_exported_bc_checkpoint_includes_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_checkpoint_schema_') as tmpdir:
            checkpoint_path = Path(tmpdir) / 'exported.pt'
            agent = self._agent()
            agent.state_dict.return_value = {'actor.obs_median': torch.zeros(OBS_DIM)}
            config = {'data_schema_version': DATA_SCHEMA_VERSION, 'bc': {'steps': 12}}

            _export_bc_checkpoint(agent, config, checkpoint_path, control_fingerprint='23-wind-baseline')

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.assertEqual(checkpoint['data_schema_version'], DATA_SCHEMA_VERSION)
            self.assertEqual(checkpoint['control_fingerprint'], '23-wind-baseline')
            self.assertEqual(checkpoint['stage'], 'bc')
            self.assertEqual(checkpoint['bc_total_steps'], 12)

    def test_rejects_config_without_current_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, 'data_schema_version must be'):
            _require_data_schema_version({})


if __name__ == '__main__':
    unittest.main()