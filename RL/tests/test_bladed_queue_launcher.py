from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def _load_launcher_module():
    path = Path(__file__).resolve().parents[2] / 'bladed_env' / 'run_bladed_queue.py'
    spec = importlib.util.spec_from_file_location('bladed_queue_launcher_test', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBladedQueueLauncher(unittest.TestCase):
    def test_dry_run_filters_slots_and_records_controller_environment(self) -> None:
        launcher = _load_launcher_module()
        with tempfile.TemporaryDirectory(prefix='bladed_queue_launcher_') as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / 'manifest.json'
            log_path = root / 'runs.jsonl'
            manifest_path.write_text(
                json.dumps(
                    {
                        'protocol_version': 'bladed_udp_19_si_my_queue_v1',
                        'execution_mode': 'independent_case_reload',
                        'total_condition_rounds': 1,
                        'base_condition_order': [3, 4],
                        'case_path_template': 'C:/cases/wind{condition_id}.prj',
                        'bladed_command_template': 'bladed {case_path}',
                        'post_load_discard_sim_s': 30.0,
                        'train_sim_s': 600.0,
                    }
                ),
                encoding='utf-8',
            )
            with patch.object(
                sys,
                'argv',
                [
                    'run_bladed_queue.py',
                    '--manifest',
                    str(manifest_path),
                    '--dry-run',
                    '--condition-ids',
                    '4',
                    '--max-slots',
                    '1',
                    '--log-path',
                    str(log_path),
                ],
            ):
                launcher.main()

            rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['condition_id'], 4)
        self.assertEqual(rows[0]['condition_instance_id'], 2)
        self.assertEqual(rows[0]['status'], 'dry_run')
        self.assertEqual(rows[0]['controller_environment']['BLADED_RL_TRAIN_SIM_S'], '600.000000')
        self.assertEqual(rows[0]['controller_environment']['BLADED_RL_POST_LOAD_DISCARD_SIM_S'], '30.000000')


if __name__ == '__main__':
    unittest.main()