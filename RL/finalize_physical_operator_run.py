"""Finalize a user-confirmed configurable-rounds 23-wind physical-shadow package.

This command binds the already-provided operator confirmation to the exact
schedule/config/source identities. It never creates an experiment output
directory or starts training, UDP, or Bladed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from RL.prepare_physical_shadow_development import check_resources
from RL.utils.physical_load_shadow_artifact import artifact_path, file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_RECEIPT_NAME,
    OPERATOR_23WIND_ROUTE,
    OPERATOR_23WIND_SCHEMA,
    OPERATOR_23WIND_TEMPLATE,
    OPERATOR_23WIND_RELEASE_ID,
    _operator_23wind_confirmation,
    operator_resource_inventory,
    physical_runtime_identity,
    validate_physical_launch,
    validate_operator_23wind_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_new(path: Path, document: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def finalize_operator_run(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('RUN_ID must contain 1-80 ASCII letters, numbers, underscores, or hyphens')
    config_dir = (PROJECT_ROOT / 'RL/configs' / run_id).resolve()
    if config_dir.parent != (PROJECT_ROOT / 'RL/configs').resolve():
        raise ValueError('RUN_ID must resolve directly under RL/configs')
    config_path = config_dir / 'train.json'
    report_path = config_dir / 'run_preparation_report.json'
    receipt_path = config_dir / OPERATOR_23WIND_RECEIPT_NAME
    timestamped_experiments = tuple((PROJECT_ROOT / 'RL/experiments').glob(f'{run_id}_*'))
    if not config_path.is_file() or not report_path.is_file():
        raise FileNotFoundError('Unified 23-wind RUN_ID preparation package is incomplete')
    if timestamped_experiments:
        raise FileExistsError('Refusing to finalize a RUN_ID with an existing experiment output directory')
    experiment_dir = (PROJECT_ROOT / 'RL/experiments' / f'{run_id}_<timestamp>').resolve()
    if receipt_path.exists():
        raise FileExistsError('Refusing to overwrite an existing operator receipt')
    preparation = strict_json(report_path)
    config = strict_json(config_path)
    if (preparation.get('run_id') != run_id or
            preparation.get('decision') != 'LINUX_RUN_PACKAGE_PREPARED_OPERATOR_CONFIRMATION_FINALIZE_REQUIRED' or
            preparation.get('operator_receipt_written') is not False or
            preparation.get('physical_shadow_launch_authorized') is not False or
            preparation.get('experiment_output_created') is not False or
            preparation.get('train_config_path') != str(config_path) or
            file_sha256(config_path) != preparation.get('train_config_sha256') or
            preparation.get('candidate_profile_path') != str((PROJECT_ROOT / OPERATOR_23WIND_TEMPLATE).resolve()) or
            preparation.get('candidate_profile_sha256') != file_sha256(PROJECT_ROOT / OPERATOR_23WIND_TEMPLATE) or
            preparation.get('profile_build_report_sha256') !=
            file_sha256(PROJECT_ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT)):
        raise ValueError('Prepared package/report/config identity does not close; do not edit it manually')
    for section, field, path_key, digest_key in (
            ('queue', 'manifest_path', 'queue_manifest_path', 'queue_manifest_sha256'),
            ('load_predictor', 'bundle_path', 'bundle_path', 'bundle_sha256')):
        dependency = artifact_path(config[section][field], PROJECT_ROOT).resolve()
        if (preparation.get(path_key) != str(dependency) or
                preparation.get(digest_key) != file_sha256(dependency)):
            raise ValueError('Prepared copied manifest/bundle identity changed; do not edit it manually')
    if (config.get('experiment_name') != run_id or
            config.get('physical_shadow_launch', {}).get('authorized') is not False or
            config.get('physical_shadow_launch', {}).get('route') != OPERATOR_23WIND_ROUTE or
            config.get('physical_shadow_launch', {}).get('decision') !=
            'BLOCKED_PENDING_OPERATOR_CONFIRMATION_FINALIZE' or
            config.get('physical_shadow_launch', {}).get('evidence_path') != '' or
            config.get('physical_shadow_launch', {}).get('evidence_sha256') != ''):
        raise ValueError('RUN_ID train config is not the expected launch-blocked unified candidate')

    # Route helper checks the supplied Windows declarations against the exact
    # configured-rounds schedule digest and the expected protocol/training contract.
    proposed_config = dict(config)
    proposed_gate = dict(config['physical_shadow_launch'])
    proposed_gate.update(
        evidence_path=str(receipt_path),
        evidence_sha256='',
        authorized=True,
        decision='OPERATOR_CONFIRMED_23WIND_DEVELOPMENT_ONLY_NOT_SIX_GATE_VERIFIED',
    )
    proposed_config['physical_shadow_launch'] = proposed_gate
    route_contract = validate_operator_23wind_scope(proposed_config, PROJECT_ROOT)
    identity = physical_runtime_identity(proposed_config, PROJECT_ROOT)
    operator_confirmation = _operator_23wind_confirmation(route_contract, identity)
    if (preparation.get('operator_route_contract') != route_contract or
            preparation.get('operator_confirmation') != operator_confirmation or
            preparation.get('source_profile_sha256') != route_contract['source_profile_sha256'] or
            preparation.get('runtime_identity') != identity or
            preparation.get('resource_inventory') != operator_resource_inventory(proposed_config, PROJECT_ROOT)):
        raise ValueError('Prepared route/runtime/dependency identity changed; do not edit it manually')
    resource_inventory = operator_resource_inventory(proposed_config, PROJECT_ROOT)
    availability = check_resources(proposed_config, experiment_dir)
    profile_report = strict_json(PROJECT_ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT)
    template_sha = file_sha256(PROJECT_ROOT / OPERATOR_23WIND_TEMPLATE)
    receipt = {
        'schema_version': OPERATOR_23WIND_SCHEMA,
        'route': OPERATOR_23WIND_ROUTE,
        'release_id': OPERATOR_23WIND_RELEASE_ID,
        'rounds': route_contract['rounds'],
        'slots': route_contract['slot_count'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': run_id,
        'config_dir': str(config_dir),
        'run_config': str(config_path),
        'experiment_dir_prefix': str((PROJECT_ROOT / 'RL/experiments' / f'{run_id}_').resolve()),
        'runtime_identity': identity,
        'operator_attestation': operator_confirmation,
        'operator_route_contract': route_contract,
        'source_profile_path': route_contract['source_profile_path'],
        'source_profile_sha256': route_contract['source_profile_sha256'],
        'profile_build_report_sha256': file_sha256(PROJECT_ROOT / OPERATOR_23WIND_PROFILE_BUILD_REPORT),
        'profile_build_decision': profile_report.get('decision'),
        'template_sha256': template_sha,
        'resource_inventory': resource_inventory,
        'availability_at_preparation': availability,
        'independently_verified': False,
        'policy_acceptance': False,
        'online_latency_status': 'unknown_measure_during_experiment',
        'optimizer_updates': True,
        'actor_frozen': False,
        'independent_audit': False,
        'counterfactual_validation': False,
        'soft_authorized': False,
        'hard_authorized': False,
        'authorization': {
            'mode': 'shadow_development_only',
            'policy_acceptance': False,
            'independent_audit': False,
            'counterfactual_validation': False,
            'soft_authorized': False,
            'hard_authorized': False,
            'rl_margin_coverage_authorized': False,
        },
    }
    try:
        _write_new(receipt_path, receipt)
        proposed_gate['evidence_path'] = str(receipt_path)
        proposed_gate['evidence_sha256'] = file_sha256(receipt_path)
        proposed_config['physical_shadow_launch'] = proposed_gate

        # Perform a full local receipt/config/scope preflight. The experiment output
        # directory remains absent; train_rl.py performs the fresh-directory check
        # once more immediately before binding UDP.
        launch_identity = validate_physical_launch(proposed_config, project_root=PROJECT_ROOT)
        if launch_identity.get('route') != OPERATOR_23WIND_ROUTE:
            raise ValueError('Final launch validator selected an unexpected route')
        temporary_config = config_dir / '.train.json.finalizing'
        with temporary_config.open('x', encoding='utf-8') as handle:
            json.dump(proposed_config, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary_config, config_path)
    except Exception:
        receipt_path.unlink(missing_ok=True)
        (config_dir / '.train.json.finalizing').unlink(missing_ok=True)
        raise
    return {
        'run_id': run_id,
        'receipt_path': str(receipt_path),
        'receipt_sha256': file_sha256(receipt_path),
        'train_config_path': str(config_path),
        'train_config_sha256': file_sha256(config_path),
        'experiment_output_created': False,
        'physical_shadow_launch_authorized': True,
        'operator_route': OPERATOR_23WIND_ROUTE,
        'policy_acceptance': False,
        'decision': 'OPERATOR_CONFIRMED_DEVELOPMENT_ROUTE_READY_FOR_FINAL_LOCAL_GATE_CHECK',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_operator_run(args.run_id), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
