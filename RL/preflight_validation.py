"""Read-only preflight validation for a configured WT15 RL run.

The checker never binds UDP, never starts Bladed and never writes into the
experiment directory.  It resolves the configured control contract, verifies
the small artifact set, expands the queue manifest and (optionally) checks the
local memory/disk budget.  Large data files are hashed only when the config
declares an expected SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from RL.utils.config import load_config
from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.queue_manifest import expand_queue_manifest
from RL.utils.state_action_artifact import StateActionEnvelopeArtifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'preflight_validation_v1'


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: Any, root: Path, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    candidate = Path(value.strip().replace('\\', '/')).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _require_file(path: Path, *, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'{name} is missing: {path}')
    return path


def _require_dir(path: Path, *, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f'{name} is missing: {path}')
    return path


def _mem_available_bytes() -> int | None:
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class Report:
    def __init__(self, config_path: Path, project_root: Path) -> None:
        self.document: dict[str, Any] = {
            'schema_version': SCHEMA_VERSION,
            'generated_at_utc': _now_utc(),
            'config_path': str(config_path),
            'project_root': str(project_root),
            'checks': [],
            'summary': {},
        }

    def add(self, name: str, status: str, detail: str) -> None:
        self.document['checks'].append(
            {'name': name, 'status': status, 'detail': detail}
        )

    def run(self, name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            self.add(name, 'PASS', detail)
        except Exception as exc:  # preflight must continue to collect all failures
            self.add(name, 'FAIL', f'{type(exc).__name__}: {exc}')

    def finish(self) -> dict[str, Any]:
        checks = self.document['checks']
        failures = [item for item in checks if item['status'] == 'FAIL']
        warnings = [item for item in checks if item['status'] == 'WARN']
        self.document['summary'] = {
            'passed': not failures,
            'total': len(checks),
            'passed_count': sum(item['status'] == 'PASS' for item in checks),
            'warning_count': len(warnings),
            'failure_count': len(failures),
            'failed_checks': [item['name'] for item in failures],
        }
        return self.document


def _check_contract(config: dict[str, Any], root: Path) -> str:
    resolved = resolve_configured_control_contract(config, project_root=root)
    if resolved is None:
        raise ValueError('control contract is not configured')
    _, fingerprint = resolved
    configured = config.get('control_contract_fingerprint')
    if configured and configured != fingerprint:
        raise ValueError('configured control_contract_fingerprint does not match resolved fingerprint')
    control = config.get('control')
    if not isinstance(control, dict):
        raise ValueError('control section is missing')
    nested = control.get('resolved_control_contract_fingerprint')
    if nested and nested != fingerprint:
        raise ValueError('control.resolved_control_contract_fingerprint is inconsistent')
    return f'fingerprint={fingerprint[:12]}…'


def _check_state_artifact(config: dict[str, Any], root: Path) -> str:
    control = config.get('control')
    env = config.get('env')
    if not isinstance(control, dict) or not isinstance(env, dict):
        raise ValueError('env/control sections are required')
    control_path = _path(control.get('state_action_artifact_path'), root, name='control.state_action_artifact_path')
    env_path = _path(env.get('state_action_artifact_path'), root, name='env.state_action_artifact_path')
    _require_file(control_path, name='state/action artifact')
    _require_file(env_path, name='env state/action artifact')
    if _sha256(control_path) != _sha256(env_path):
        raise ValueError('env and control state/action artifacts differ')
    artifact = StateActionEnvelopeArtifact.from_path(control_path)
    width = artifact.observation_schema.get('width')
    if width != 20:
        raise ValueError(f'state20 observation width is {width!r}, expected 20')
    return f'width=20 sha256={artifact.source_sha256[:12]}…'


def _check_queue(config: dict[str, Any], root: Path) -> str:
    queue = config.get('queue')
    if not isinstance(queue, dict):
        raise ValueError('queue section is missing')
    manifest_path = _path(queue.get('manifest_path'), root, name='queue.manifest_path')
    _require_file(manifest_path, name='queue manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    slots = expand_queue_manifest(manifest)
    rounds = int(manifest['total_condition_rounds'])
    expected = rounds * len(manifest.get('base_condition_order', ()))
    if len(slots) != expected:
        raise ValueError(f'expanded slot count {len(slots)} != expected {expected}')
    duration = float(manifest.get('train_sim_s', 0.0))
    configured_duration = float(queue.get('train_sim_s', duration))
    if not math.isclose(duration, configured_duration, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('queue manifest train_sim_s differs from queue config')
    return f'slots={len(slots)} rounds={rounds} duration={duration:g}s'


def _check_hashes(config: dict[str, Any], root: Path) -> str:
    checked = []
    predictor = config.get('load_predictor')
    if isinstance(predictor, dict) and predictor.get('bundle_path'):
        bundle = _require_file(
            _path(predictor['bundle_path'], root, name='load_predictor.bundle_path'),
            name='physical shadow bundle',
        )
        expected = predictor.get('bundle_sha256')
        if expected:
            actual = _sha256(bundle)
            if actual != expected:
                raise ValueError('physical shadow bundle SHA-256 mismatch')
            checked.append('bundle')
    contract = config.get('state20_queue_contract')
    if isinstance(contract, dict):
        for key, path_key in (
            ('paired_reference_sha256', 'paired_reference_path'),
            ('bc_warmstart_checkpoint_sha256', None),
        ):
            expected = contract.get(key)
            if not expected:
                continue
            if path_key is not None:
                target = _path(contract.get(path_key), root, name=f'state20_queue_contract.{path_key}')
            else:
                bc = config.get('bc')
                if not isinstance(bc, dict) or not bc.get('resume_checkpoint'):
                    raise ValueError('bc.resume_checkpoint is required for checkpoint SHA check')
                target = _path(bc['resume_checkpoint'], root, name='bc.resume_checkpoint')
            _require_file(target, name=key)
            actual = _sha256(target)
            if actual != expected:
                raise ValueError(f'{key} mismatch: expected={expected[:12]}… actual={actual[:12]}…')
            checked.append(key.replace('_sha256', ''))
    return f'checked={checked or ["none"]}'


def _check_paths(config: dict[str, Any], root: Path) -> str:
    checked = []
    data = config.get('data')
    if isinstance(data, dict) and data.get('root_dir'):
        _require_dir(_path(data['root_dir'], root, name='data.root_dir'), name='data root')
        checked.append('data.root_dir')
    control = config.get('control')
    if isinstance(control, dict):
        for key in ('region_aware_baseline_path', 's2_u1_calibration_path'):
            value = control.get(key)
            if value:
                _require_file(_path(value, root, name=f'control.{key}'), name=key)
                checked.append(key)
    bc = config.get('bc')
    if isinstance(bc, dict) and bc.get('resume_checkpoint'):
        _require_file(_path(bc['resume_checkpoint'], root, name='bc.resume_checkpoint'), name='BC checkpoint')
        checked.append('bc.resume_checkpoint')
    return f'checked={checked}'


def _check_resources(config: dict[str, Any], root: Path) -> str:
    predictor = config.get('load_predictor')
    required_disk = 0
    if isinstance(predictor, dict):
        required_disk = int(predictor.get('max_total_bytes', 0) or 0)
    usage = shutil.disk_usage(root)
    reserve = 2 * 1024 ** 3
    if required_disk and usage.free < required_disk + reserve:
        raise ValueError(
            f'disk free {usage.free / 1024 ** 3:.1f} GiB is below shadow budget + 2 GiB '
            f'({required_disk / 1024 ** 3:.1f} GiB)'
        )
    mem = _mem_available_bytes()
    if mem is not None and mem < 4 * 1024 ** 3:
        raise ValueError(f'MemAvailable {mem / 1024 ** 3:.1f} GiB is below 4 GiB')
    mem_text = 'unknown' if mem is None else f'{mem / 1024 ** 3:.1f} GiB'
    return (
        f'disk_free={usage.free / 1024 ** 3:.1f} GiB '
        f'shadow_budget={required_disk / 1024 ** 3:.1f} GiB mem_available={mem_text}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Read-only preflight for a WT15 RL config.')
    parser.add_argument('--config', required=True, help='Path to train/inference config JSON/YAML')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT), help='Repository root')
    parser.add_argument('--json-out', default='', help='Write the report JSON to this path')
    parser.add_argument('--no-resources', action='store_true', help='Skip memory/disk checks')
    parser.add_argument('--quiet', action='store_true', help='Only print failures and summary')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    report = Report(config_path, root)
    try:
        config = load_config(str(config_path))
        if not isinstance(config, dict):
            raise ValueError('config root must be a JSON/YAML object')
        report.add('config.load', 'PASS', f'sections={sorted(config.keys())}')
    except Exception as exc:
        report.add('config.load', 'FAIL', f'{type(exc).__name__}: {exc}')
        document = report.finish()
        _emit(document, args)
        return 1

    report.run('control.contract', lambda: _check_contract(config, root))
    report.run('artifact.paths', lambda: _check_paths(config, root))
    report.run('artifact.state20', lambda: _check_state_artifact(config, root))
    report.run('artifact.hashes', lambda: _check_hashes(config, root))
    if isinstance(config.get('queue'), dict) and config['queue'].get('manifest_path'):
        report.run('queue.manifest', lambda: _check_queue(config, root))
    else:
        report.add('queue.manifest', 'SKIP', 'queue.manifest_path is not configured')
    if args.no_resources:
        report.add('resources', 'SKIP', '--no-resources')
    else:
        report.run('resources', lambda: _check_resources(config, root))

    document = report.finish()
    _emit(document, args)
    return 0 if document['summary']['passed'] else 1


def _emit(document: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        output = Path(args.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8',
        )
    if not args.quiet:
        for check in document['checks']:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
    summary = document['summary']
    print(
        f"preflight={'PASS' if summary['passed'] else 'FAIL'} "
        f"passed={summary['passed_count']} warnings={summary['warning_count']} "
        f"failures={summary['failure_count']}"
    )
    if summary['failed_checks']:
        print('failed: ' + ', '.join(summary['failed_checks']))


if __name__ == '__main__':
    raise SystemExit(main())
