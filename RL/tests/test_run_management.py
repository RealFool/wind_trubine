from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from RL.utils.run_management import (
    apply_run_management,
    resolve_run_management,
    should_skip_launch_validation,
)


class TestRunManagement(unittest.TestCase):
    def test_default_is_legacy(self) -> None:
        info = resolve_run_management({})
        self.assertEqual(info['identity_mode'], 'legacy')
        self.assertEqual(info['run_layout'], 'legacy')
        self.assertFalse(info['skip_launch_validation'])
        self.assertFalse(info['skip_fresh_experiment'])

    def test_minimal_keeps_legacy_layout(self) -> None:
        info = resolve_run_management({'run_management': {'identity_mode': 'minimal'}})
        self.assertEqual(info['run_layout'], 'legacy')
        self.assertTrue(info['skip_launch_validation'])
        self.assertTrue(should_skip_launch_validation({'run_management': {'identity_mode': 'minimal'}}))

    def test_timestamped_requires_minimal(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_management({'run_management': {'run_layout': 'timestamped'}})

    def test_apply_timestamped_sets_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                'experiment_name': 'legacy_name',
                'output_dir': 'RL/experiments',
                'append_timestamp': True,
                'run_management': {
                    'identity_mode': 'minimal',
                    'run_layout': 'timestamped',
                    'base_dir': 'runs',
                    'label': '20260101T000000Z_abc1234',
                },
            }
            info = apply_run_management(config, project_root=Path(tmp), create_dir=False)
            self.assertEqual(config['experiment_name'], '20260101T000000Z_abc1234')
            self.assertEqual(config['output_dir'], str(Path(tmp) / 'runs'))
            self.assertFalse(config['append_timestamp'])
            self.assertEqual(info['run_dir'], str(Path(tmp) / 'runs' / '20260101T000000Z_abc1234'))
            self.assertEqual(config['_run_management']['identity_mode'], 'minimal')

    def test_apply_legacy_does_not_mutate_layout(self) -> None:
        config = {'experiment_name': 'x', 'output_dir': 'RL/experiments'}
        info = apply_run_management(config, project_root=Path('/tmp'))
        self.assertNotIn('run_dir', info)
        self.assertEqual(config['experiment_name'], 'x')

    def test_invalid_modes(self) -> None:
        for section in (
            {'identity_mode': 'both'},
            {'run_layout': 'somewhere'},
            {'identity_mode': 'minimal', 'run_layout': 'somewhere'},
        ):
            with self.assertRaises(ValueError):
                resolve_run_management({'run_management': section})


if __name__ == '__main__':
    unittest.main()
