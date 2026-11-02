from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from RL.run_meta import _safe_label, git_info, main, make_run_dir, write_run_meta


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True)


class TestRunMeta(unittest.TestCase):
    def _repo(self, tmp: str) -> Path:
        root = Path(tmp) / 'repo'
        root.mkdir()
        _git(root, 'init', '-b', 'main')
        _git(root, 'config', 'user.name', 'Test')
        _git(root, 'config', 'user.email', 'test@example.com')
        (root / 'x.txt').write_text('x', encoding='utf-8')
        _git(root, 'add', 'x.txt')
        _git(root, 'commit', '-m', 'init')
        return root

    def test_git_info_clean_and_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp)
            clean = git_info(root)
            self.assertIsNotNone(clean['commit'])
            self.assertFalse(clean['dirty'])
            (root / 'y.txt').write_text('y', encoding='utf-8')
            dirty = git_info(root)
            self.assertTrue(dirty['dirty'])
            self.assertEqual(dirty['changed_files'], 1)

    def test_make_run_dir_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_run_dir(Path(tmp), label='abc')
            self.assertTrue(path.is_dir())
            with self.assertRaises(FileExistsError):
                make_run_dir(Path(tmp), label='abc')

    def test_safe_label(self) -> None:
        self.assertEqual(_safe_label('a b/c'), 'a_b_c')
        self.assertEqual(_safe_label(''), 'run')

    def test_write_meta_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp)
            config = {
                'experiment_name': 'unit_test',
                '_private': 'drop_me',
                'control': {
                    'region_aware_baseline_path': 'missing_b4.json',
                },
            }
            out = root / 'run'
            meta = write_run_meta(out, config, project_root=root, device='cpu')
            document = json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(document['schema_version'], 'run_meta_v1')
            self.assertEqual(document['experiment_name'], 'unit_test')
            self.assertNotIn('_private', document['config_snapshot'])
            self.assertTrue(document['artifact_hashes']['b4_baseline']['missing'])
            before = meta.read_text(encoding='utf-8')
            write_run_meta(out, {'experiment_name': 'other'}, project_root=root)
            self.assertEqual(meta.read_text(encoding='utf-8'), before)

    def test_cli_writes_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps({'experiment_name': 'cli'}), encoding='utf-8')
            out = root / 'out'
            code = main(['--config', str(config_path), '--output-dir', str(out), '--mode', 'legacy'])
            self.assertEqual(code, 0)
            self.assertTrue((out / 'run_meta.json').is_file())


if __name__ == '__main__':
    unittest.main()
