from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from RL.preflight_validation import Report, _path, _sha256, main


class TestPreflightHelpers(unittest.TestCase):
    def test_path_resolves_relative_to_root(self) -> None:
        root = Path('/tmp/project')
        self.assertEqual(_path('RL/data/x.json', root, name='x'), root / 'RL/data/x.json')
        self.assertEqual(_path('/abs/x.json', root, name='x'), Path('/abs/x.json'))

    def test_sha256_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.txt'
            path.write_text('abc', encoding='utf-8')
            self.assertEqual(
                _sha256(path),
                'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
            )

    def test_report_marks_failure(self) -> None:
        report = Report(Path('/tmp/c.json'), Path('/tmp'))
        report.run('ok', lambda: 'fine')
        report.run('bad', lambda: (_ for _ in ()).throw(ValueError('boom')))
        document = report.finish()
        self.assertFalse(document['summary']['passed'])
        self.assertEqual(document['summary']['failure_count'], 1)
        self.assertEqual(document['summary']['failed_checks'], ['bad'])

    def test_missing_config_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report.json'
            code = main(['--config', str(Path(tmp) / 'missing.json'), '--json-out', str(out), '--quiet'])
            self.assertEqual(code, 1)
            document = json.loads(out.read_text(encoding='utf-8'))
            self.assertFalse(document['summary']['passed'])
            self.assertEqual(document['checks'][0]['name'], 'config.load')


if __name__ == '__main__':
    unittest.main()
