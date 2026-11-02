"""Lightweight run metadata for a single-machine workflow.

M6A adds this module without removing the legacy RUN_ID/receipt path.  A run
gets one `run_meta.json` next to its existing logs; later steps can switch the
whole layout to `runs/<UTC timestamp>_<git sha>/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'run_meta_v1'
DEFAULT_BASE_DIR = 'runs'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(project_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        completed = subprocess.run(
            ['git', *args],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f'git {" ".join(args)} failed')
        return completed.stdout.strip()

    try:
        commit = _run('rev-parse', '--short', 'HEAD')
        status = _run('status', '--porcelain')
        diff_stat = _run('diff', '--stat') if status else ''
        return {
            'commit': commit,
            'dirty': bool(status),
            'changed_files': len([line for line in status.splitlines() if line.strip()]),
            'diff_stat': diff_stat,
            'error': None,
        }
    except Exception as exc:  # metadata must not block a run
        return {
            'commit': None,
            'dirty': None,
            'changed_files': None,
            'diff_stat': '',
            'error': f'{type(exc).__name__}: {exc}',
        }


def _safe_label(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value).strip())
    return cleaned[:80] or 'run'


def make_run_label(project_root: Path = PROJECT_ROOT) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    commit = git_info(project_root).get('commit') or 'nogit'
    return f'{stamp}_{commit}'


def make_run_dir(base_dir: Path, *, label: str | None = None,
                 project_root: Path = PROJECT_ROOT) -> Path:
    base = Path(base_dir).expanduser()
    run_dir = base / _safe_label(label or make_run_label(project_root))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve(value: Any, project_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip().replace('\\', '/')).expanduser()
    return path if path.is_absolute() else project_root / path


def _artifact_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    control = config.get('control') if isinstance(config.get('control'), dict) else {}
    bc = config.get('bc') if isinstance(config.get('bc'), dict) else {}
    predictor = config.get('load_predictor') if isinstance(config.get('load_predictor'), dict) else {}
    contract = config.get('state20_queue_contract') if isinstance(config.get('state20_queue_contract'), dict) else {}
    candidates = {
        'b4_baseline': control.get('region_aware_baseline_path'),
        'state_action_envelope': control.get('state_action_artifact_path'),
        's2_u1_calibration': control.get('s2_u1_calibration_path'),
        'bc_checkpoint': bc.get('resume_checkpoint'),
        'shadow_bundle': predictor.get('bundle_path'),
        'paired_reference': contract.get('paired_reference_path'),
    }
    resolved = {}
    for name, value in candidates.items():
        path = _resolve(value, project_root)
        if path is not None:
            resolved[name] = path
    return resolved


def collect_artifact_hashes(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in _artifact_paths(config, project_root).items():
        if path.is_file():
            result[name] = {'path': str(path), 'sha256': sha256_file(path)}
        else:
            result[name] = {'path': str(path), 'sha256': None, 'missing': True}
    return result


def _config_sha256(config: dict[str, Any], config_path: Path | None) -> str:
    if config_path is not None and config_path.is_file():
        return sha256_file(config_path)
    snapshot = {key: value for key, value in config.items() if not str(key).startswith('_')}
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def write_run_meta(
    output_dir: Path,
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    device: str | None = None,
    mode: str = 'legacy',
    project_root: Path = PROJECT_ROOT,
    started_at_utc: str | None = None,
) -> Path:
    """Write one idempotent run_meta.json; never overwrite an existing file."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / 'run_meta.json'
    if meta_path.is_file():
        return meta_path
    config_path_obj = _resolve(str(config_path), project_root) if config_path else None
    git = git_info(project_root)
    snapshot = {
        key: value for key, value in config.items()
        if not str(key).startswith('_')
    }
    document = {
        'schema_version': SCHEMA_VERSION,
        'created_at_utc': utc_now(),
        'started_at_utc': started_at_utc or utc_now(),
        'mode': mode,
        'experiment_name': config.get('experiment_name'),
        'output_dir': str(output_dir),
        'config_path': str(config_path_obj) if config_path_obj else None,
        'config_sha256': _config_sha256(config, config_path_obj),
        'config_snapshot': snapshot,
        'git': git,
        'device': device,
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'artifact_hashes': collect_artifact_hashes(config, project_root),
    }
    meta_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n',
        encoding='utf-8',
    )
    return meta_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Create a timestamped run directory and run_meta.json.')
    parser.add_argument('--config', required=True, help='Config JSON/YAML')
    parser.add_argument('--output-dir', default='', help='Explicit run directory')
    parser.add_argument('--base-dir', default=DEFAULT_BASE_DIR, help='Default: runs/')
    parser.add_argument('--label', default='', help='Optional run label (default: timestamp_gitsha)')
    parser.add_argument('--device', default='', help='Device string recorded in metadata')
    parser.add_argument('--mode', default='timestamped', choices=['legacy', 'timestamped'])
    parser.add_argument('--dry-run', action='store_true', help='Print the target path without writing')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_config(str(config_path))
    if not isinstance(config, dict):
        raise SystemExit('config root must be an object')
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser()
    else:
        base = Path(args.base_dir).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        run_dir = make_run_dir(base, label=args.label or None, project_root=PROJECT_ROOT)
    if args.dry_run:
        print(run_dir)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = write_run_meta(
        run_dir, config, config_path=config_path, device=args.device or None,
        mode=args.mode, project_root=PROJECT_ROOT,
    )
    print(meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
