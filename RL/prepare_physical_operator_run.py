"""Prepare or safely refresh a blocked configurable-rounds 23-wind shadow package.

This preparation-only command copies the frozen candidate artifacts and writes
an explicitly launch-blocked train.json plus a preparation report. It never
creates the experiment output directory, writes an operator receipt, opens UDP,
starts Bladed, or starts training. A blocked package may be refreshed in place
for the same RUN_ID only before any receipt or experiment output exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

from RL.prepare_physical_shadow_development import check_resources
from RL.utils.physical_load_shadow_artifact import artifact_path, file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_RECEIPT_NAME,
    OPERATOR_23WIND_ROUTE,
    OPERATOR_23WIND_TEMPLATE,
    OPERATOR_23WIND_RELEASE_ID,
    _operator_23wind_confirmation,
    operator_resource_inventory,
    physical_runtime_identity,
    validate_operator_23wind_scope,
)
from RL.utils.physical_queue_schedule import (
    DEFAULT_FULL_23WIND_ROUNDS,
    build_full_23wind_manifest,
    configure_full_23wind_rounds,
    validate_full_23wind_rounds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / 'RL/configs'
EXPERIMENT_ROOT = PROJECT_ROOT / 'RL/experiments'


def _write_new(path: Path, document: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def prepare_operator_run(
    run_id: str,
    *,
    rounds: int = DEFAULT_FULL_23WIND_ROUNDS,
) -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('RUN_ID must contain 1-80 ASCII letters, numbers, underscores, or hyphens')
    rounds = validate_full_23wind_rounds(rounds)
    config_dir = (CONFIG_ROOT / run_id).resolve()
    experiment_dir = (EXPERIMENT_ROOT / run_id).resolve()
    if config_dir.exists() or experiment_dir.exists() or any(EXPERIMENT_ROOT.glob(f'{run_id}_*')):
        raise FileExistsError('RUN_ID config or experiment directory already exists; choose a new RUN_ID')

    try:
        return _prepare_operator_run_package(run_id, rounds, config_dir, experiment_dir)
    except Exception:
        shutil.rmtree(config_dir, ignore_errors=True)
        raise


def _prepare_operator_run_package(
    run_id: str,
    rounds: int,
    config_dir: Path,
    experiment_dir: Path,
) -> dict[str, Any]:

    source_config_path = (PROJECT_ROOT / OPERATOR_23WIND_TEMPLATE).resolve()
    source_config = strict_json(source_config_path)
    config = dict(source_config)
    config['data'] = dict(source_config['data'])
    config['queue'] = dict(source_config['queue'])
    config['queue']['case_identity_validation'] = dict(source_config['queue']['case_identity_validation'])
    config['load_predictor'] = dict(source_config['load_predictor'])
    config['logging'] = dict(source_config['logging'])
    config['physical_shadow_launch'] = {
        'authorized': False,
        'route': OPERATOR_23WIND_ROUTE,
        'evidence_path': '',
        'evidence_sha256': '',
        'decision': 'BLOCKED_PENDING_OPERATOR_CONFIRMATION_FINALIZE',
    }
    config.update(
        experiment_name=run_id,
        output_dir='RL/experiments',
        append_timestamp=True,
        policy_acceptance=False,
    )
    config['data']['root_dir'] = str(artifact_path(config['data']['root_dir'], PROJECT_ROOT).resolve())
    config['queue']['case_identity_validation']['wind_mismatch_policy'] = 'warn_development_only'
    source_manifest = strict_json(artifact_path(source_config['queue']['manifest_path'], PROJECT_ROOT))
    run_manifest = build_full_23wind_manifest(source_manifest, rounds)
    manifest_copy = config_dir / 'queue_manifest.json'
    bundle_copy = config_dir / 'physical_load_shadow_bundle.json'
    config_dir.mkdir(parents=False, exist_ok=False)
    manifest_copy.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(artifact_path(source_config['load_predictor']['bundle_path'], PROJECT_ROOT), bundle_copy)
    config['queue']['manifest_path'] = _relative(manifest_copy)
    config['load_predictor']['bundle_path'] = _relative(bundle_copy)
    configure_full_23wind_rounds(config, manifest_copy, rounds, project_root=PROJECT_ROOT)
    config['release_id'] = OPERATOR_23WIND_RELEASE_ID

    # Scope/preflight before creating a run directory. This is a local candidate
    # check only; the launch route stays explicitly blocked below.
    route_contract = validate_operator_23wind_scope(config, PROJECT_ROOT)
    experiment_root = PROJECT_ROOT / 'RL/experiments'
    if any(path.is_dir() and (path.name == run_id or path.name.startswith(f'{run_id}_'))
           for path in experiment_root.iterdir()):
        raise FileExistsError('RUN_ID already has an experiment directory or timestamp prefix; choose a fresh RUN_ID')
    availability = check_resources(config, experiment_dir)
    source_profile = strict_json(PROJECT_ROOT / route_contract['source_profile_path'])
    # The route binds the new schedule and frozen bundle copies inside this RUN_ID.
    copied_route_contract = validate_operator_23wind_scope(config, PROJECT_ROOT)
    if copied_route_contract['manifest_sha256'] != route_contract['manifest_sha256']:
        raise ValueError('RUN_ID manifest SHA differs from the configured schedule')

    # Receipt is created by a separate Linux finalization step after the user
    # confirmation already provided in the current session.
    train_config_path = config_dir / 'train.json'
    _write_new(train_config_path, config)
    inventory = operator_resource_inventory(config, PROJECT_ROOT)
    identity = physical_runtime_identity(config, PROJECT_ROOT)
    operator_confirmation = _operator_23wind_confirmation(route_contract, identity)
    report = {
        'schema_version': 'physical_operator_23wind_run_preparation_v2',
        'release_id': OPERATOR_23WIND_RELEASE_ID,
        'rounds': rounds,
        'run_id': run_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_profile_path': str(source_config_path),
        'candidate_profile_sha256': file_sha256(source_config_path),
        'profile_build_report_sha256': file_sha256(PROJECT_ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT),
        'source_profile_sha256': route_contract['source_profile_sha256'],
        'source_state20_contract_fingerprint': source_profile['control_contract_fingerprint'],
        'train_config_path': str(train_config_path),
        'train_config_sha256': file_sha256(train_config_path),
        'queue_manifest_path': str(manifest_copy),
        'queue_manifest_sha256': file_sha256(manifest_copy),
        'bundle_path': str(bundle_copy),
        'bundle_sha256': file_sha256(bundle_copy),
        'operator_route_contract': route_contract,
        'operator_confirmation': operator_confirmation,
        'runtime_identity': identity,
        'resource_inventory': inventory,
        'availability_at_preparation': availability,
        'experiment_output_prefix': str((EXPERIMENT_ROOT / f'{run_id}_').resolve()),
        'experiment_output_created': False,
        'operator_receipt_path': str(config_dir / OPERATOR_23WIND_RECEIPT_NAME),
        'operator_receipt_schema_version': 'physical_shadow_operator_23wind_round600_training_receipt_v2',
        'operator_receipt_written': False,
        'physical_shadow_launch_authorized': False,
        'policy_acceptance': False,
        'independent_audit': False,
        'counterfactual_validation': False,
        'soft_authorized': False,
        'hard_authorized': False,
        'decision': 'LINUX_RUN_PACKAGE_PREPARED_OPERATOR_CONFIRMATION_FINALIZE_REQUIRED',
    }
    _write_new(config_dir / 'run_preparation_report.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--rounds', type=int, default=DEFAULT_FULL_23WIND_ROUNDS)
    args = parser.parse_args()
    report = prepare_operator_run(args.run_id, rounds=args.rounds)
    result = {
        'run_id': report['run_id'],
        'train_config': report['train_config_path'],
        'preparation_report': str(CONFIG_ROOT / args.run_id / 'run_preparation_report.json'),
        'slots': report['operator_route_contract']['slot_count'],
        'rounds': report['rounds'],
        'physical_shadow_launch_authorized': report['physical_shadow_launch_authorized'],
        'decision': report['decision'],
        'experiment_output_created': report['experiment_output_created'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()