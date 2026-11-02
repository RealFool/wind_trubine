from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from RL.utils.config import load_config, normalize_user_path


class TestConfigPaths(unittest.TestCase):
    def test_normalize_user_path_accepts_windows_separators(self) -> None:
        normalized = normalize_user_path(
            'RL\\configs\\b4_residual_smoke\\bc_pretrain_b4_state20_residual_600s_v3.json'
        )
        self.assertEqual(
            normalized.as_posix(),
            'RL/configs/b4_residual_smoke/bc_pretrain_b4_state20_residual_600s_v3.json',
        )

    def test_load_config_accepts_windows_style_relative_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix='rl_cfg_path_') as tmpdir:
            config_dir = Path(tmpdir) / 'nested'
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / 'config.json'
            with config_path.open('w', encoding='utf-8') as file:
                json.dump({'device': 'cuda:0'}, file)

            windows_style_path = str(config_path).replace('/', '\\')
            config = load_config(windows_style_path)
            self.assertEqual(config['device'], 'cuda:0')


if __name__ == '__main__':
    unittest.main()