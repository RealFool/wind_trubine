"""Run-management modes for the single-machine transition.

Legacy mode keeps the existing RUN_ID/receipt/release-manifest checks.  Minimal
mode is opt-in: it skips the receipt/identity gate, still runs artifact/resource
preflight, and can place logs under `runs/<UTC timestamp>_<git sha>/`.

M6B only adds the switch.  The default stays legacy until a 4x4 run proves the
minimal path on the real Windows/Bladed setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from RL.run_meta import make_run_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'run_management_v1'
IDENTITY_MODES = frozenset({'legacy', 'minimal'})
RUN_LAYOUTS = frozenset({'legacy', 'timestamped'})


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get('run_management', {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError('run_management must be an object')
    return section


def resolve_run_management(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    identity = str(section.get('identity_mode', 'legacy')).strip().lower()
    layout = str(section.get('run_layout', 'legacy')).strip().lower()
    if identity not in IDENTITY_MODES:
        raise ValueError('run_management.identity_mode must be legacy or minimal')
    if layout not in RUN_LAYOUTS:
        raise ValueError('run_management.run_layout must be legacy or timestamped')
    if layout == 'timestamped' and identity != 'minimal':
        raise ValueError(
            'run_management.run_layout=timestamped requires identity_mode=minimal '
            '(receipt paths are tied to the legacy RUN_ID layout)'
        )
    base_dir = str(section.get('base_dir', 'runs')).strip() or 'runs'
    label = section.get('label')
    if label is not None:
        label = str(label).strip() or None
    return {
        'schema_version': SCHEMA_VERSION,
        'identity_mode': identity,
        'run_layout': layout,
        'base_dir': base_dir,
        'label': label,
        'skip_launch_validation': identity == 'minimal',
        'skip_fresh_experiment': identity == 'minimal',
    }


def apply_run_management(
    config: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    create_dir: bool = False,
) -> dict[str, Any]:
    info = resolve_run_management(config)
    if info['run_layout'] == 'timestamped':
        label = info['label'] or make_run_label(project_root)
        base = Path(info['base_dir']).expanduser()
        if not base.is_absolute():
            base = project_root / base
        run_dir = base / label
        if create_dir:
            run_dir.mkdir(parents=True, exist_ok=False)
        config['output_dir'] = str(base)
        config['experiment_name'] = label
        config['append_timestamp'] = False
        info['label'] = label
        info['run_dir'] = str(run_dir)
    config['_run_management'] = info
    return info


def run_identity_mode(config: dict[str, Any]) -> str:
    return resolve_run_management(config)['identity_mode']


def should_skip_launch_validation(config: dict[str, Any]) -> bool:
    return resolve_run_management(config)['skip_launch_validation']
