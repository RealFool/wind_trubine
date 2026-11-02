from __future__ import annotations

import argparse
import copy
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.physical_load_shadow_artifact import file_sha256, strict_json
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_RELEASE_ID,
    OPERATOR_23WIND_SOURCE_PROFILE,
    OPERATOR_23WIND_TEMPLATE,
)
from RL.utils.physical_queue_schedule import (
    DEFAULT_FULL_23WIND_ROUNDS,
    FULL_23WIND_CONDITIONS,
    FULL_23WIND_DURATION_S,
    FULL_23WIND_PROFILE,
    MAX_FULL_23WIND_ROUNDS,
    validate_full_23wind_rounds,
)

ARCHIVE_RELEASE_ID = 'WT15_STATE20_RL_20261022_R1'
ARCHIVE_RELEASE_PATH = f'RL/configs/{ARCHIVE_RELEASE_ID}/release_manifest.json'


def _load_json(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def build_unified_release_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = project_root.resolve()
    archive_path = root / ARCHIVE_RELEASE_PATH
    archive = _load_json(archive_path)
    config_path = root / OPERATOR_23WIND_TEMPLATE
    profile_report_path = root / OPERATOR_23WIND_PROFILE_BUILD_REPORT
    source_profile_path = root / OPERATOR_23WIND_SOURCE_PROFILE
    config = _load_json(config_path)
    report = _load_json(profile_report_path)
    profile = report.get('profiles', {}).get(FULL_23WIND_PROFILE)
    if not isinstance(profile, dict):
        raise ValueError('R2 profile build report is missing the unified 23-wind profile')
    rounds = validate_full_23wind_rounds(int(config['queue']['total_condition_rounds']))
    manifest_path = root / config['queue']['manifest_path']
    manifest = _load_json(manifest_path)
    if (config.get('release_id') != OPERATOR_23WIND_RELEASE_ID
            or report.get('release_id') != OPERATOR_23WIND_RELEASE_ID
            or report.get('decision') != 'DEVELOPMENT_PROFILES_BUILT_REAL_LAUNCH_BLOCKED'
            or report.get('bladed_started') is not False
            or profile.get('config_path') != OPERATOR_23WIND_TEMPLATE
            or profile.get('config_sha256') != file_sha256(config_path)
            or profile.get('manifest_path') != config['queue']['manifest_path']
            or profile.get('manifest_sha256') != file_sha256(manifest_path)
            or profile.get('slots') != len(FULL_23WIND_CONDITIONS) * rounds
            or manifest.get('total_condition_rounds') != rounds
            or manifest.get('expanded_slot_count') != len(FULL_23WIND_CONDITIONS) * rounds
            or config.get('physical_shadow_launch', {}).get('authorized') is not False):
        raise ValueError('R2 config/report/manifest/release identity does not close or is not blocked')

    runtime_identity = profile['runtime_identity']
    pinned_files: dict[str, str] = {
        f'RL/configs/{OPERATOR_23WIND_RELEASE_ID}/README.md': file_sha256(
            root / f'RL/configs/{OPERATOR_23WIND_RELEASE_ID}/README.md'
        ),
        OPERATOR_23WIND_TEMPLATE: file_sha256(config_path),
        config['queue']['manifest_path']: file_sha256(manifest_path),
        OPERATOR_23WIND_PROFILE_BUILD_REPORT: file_sha256(profile_report_path),
        config['load_predictor']['bundle_path']: file_sha256(
            root / config['load_predictor']['bundle_path']
        ),
        OPERATOR_23WIND_SOURCE_PROFILE: file_sha256(source_profile_path),
        report['windows_queue_source_path']: report['windows_queue_source_sha256'],
        report['expected_windows_release_dll_path']: report['expected_windows_release_dll_sha256'],
    }
    pinned_files.update(runtime_identity.get('runtime_code_sha256', {}))
    pinned_files.update(runtime_identity.get('operator_23wind_preparation_code_sha256', {}))
    pinned_files['RL/build_unified_profile_release_manifest.py'] = file_sha256(
        root / 'RL/build_unified_profile_release_manifest.py'
    )

    component_identities = copy.deepcopy(archive.get('component_identities'))
    physical_dependencies = archive.get('physical_shadow_dependencies')
    if not isinstance(component_identities, dict) or not isinstance(physical_dependencies, dict):
        raise ValueError('R1 archive is missing reusable component/dependency identities')
    component_identities['physical_load_shadow']['bundle_path'] = config['load_predictor']['bundle_path']
    component_identities['physical_load_shadow']['bundle_sha256'] = file_sha256(
        root / config['load_predictor']['bundle_path']
    )

    archive_profile = archive['profiles']['primary_23wind6_training_development']
    archive_record = {
        'release_id': ARCHIVE_RELEASE_ID,
        'status': 'immutable_historical_archive_not_an_active_route',
        'release_manifest': {
            'path': ARCHIVE_RELEASE_PATH,
            'sha256': file_sha256(archive_path),
        },
        'profile': {
            'config_path': archive_profile['config_path'],
            'config_sha256': archive_profile['config_sha256'],
            'queue_manifest_path': archive_profile['queue_manifest_path'],
            'queue_manifest_sha256': archive_profile['queue_manifest_sha256'],
            'profile_build_report_path': archive_profile['profile_build_report_path'],
            'profile_build_report_sha256': archive_profile['profile_build_report_sha256'],
            'rounds': 6,
            'slots': 138,
        },
        'receipt_reuse_allowed': False,
    }

    authorization = {
        'physical_profile_launch_authorized': False,
        'real_bladed_started_for_this_release': False,
        'policy_acceptance': False,
        'independent_audit': False,
        'soft_authorized': False,
        'hard_authorized': False,
        'counterfactual_validation': False,
        'rl_margin_coverage_authorized': False,
    }
    return {
        'manifest_format': 'wind_turbine_unified_release_manifest',
        'release_id': OPERATOR_23WIND_RELEASE_ID,
        'created_date': date.today().isoformat(),
        'title': '15MW wind turbine state20 unified rounds-configurable physical-shadow training profile',
        'release_status': 'OFFLINE_DEVELOPMENT_CANDIDATE',
        'release_ready': False,
        'scope': {
            'primary_route': 'state20 + state/region-conditioned B4 residual + frozen BC warm-start + TD3+BC + SafeResidualController',
            'physical_load_shadow': 'physical_fir_v2 read-only shadow; no action/reward/replay effect',
            'wind_conditions': list(FULL_23WIND_CONDITIONS),
            'duration_s_per_slot': FULL_23WIND_DURATION_S,
            'rounds_default': DEFAULT_FULL_23WIND_ROUNDS,
            'rounds_supported': list(range(1, MAX_FULL_23WIND_ROUNDS + 1)),
            'rounds_maximum_reason': '18 GiB frozen physical-shadow worst-case record budget',
            'default_slots': len(FULL_23WIND_CONDITIONS) * DEFAULT_FULL_23WIND_ROUNDS,
            'observation_dimension': 20,
        },
        'component_identities': component_identities,
        'physical_shadow_dependencies': physical_dependencies,
        'profiles': {
            'primary_23wind_round600_training': {
                'release_id': OPERATOR_23WIND_RELEASE_ID,
                'profile_name': FULL_23WIND_PROFILE,
                'config_path': OPERATOR_23WIND_TEMPLATE,
                'config_sha256': file_sha256(config_path),
                'queue_manifest_path': config['queue']['manifest_path'],
                'queue_manifest_sha256': file_sha256(manifest_path),
                'profile_build_report_path': OPERATOR_23WIND_PROFILE_BUILD_REPORT,
                'profile_build_report_sha256': file_sha256(profile_report_path),
                'source_queue_profile_path': OPERATOR_23WIND_SOURCE_PROFILE,
                'source_queue_profile_sha256': file_sha256(source_profile_path),
                'schedule_default': {
                    'wind_conditions': len(FULL_23WIND_CONDITIONS),
                    'rounds': DEFAULT_FULL_23WIND_ROUNDS,
                    'duration_s_per_slot': FULL_23WIND_DURATION_S,
                    'slot_count': len(FULL_23WIND_CONDITIONS) * DEFAULT_FULL_23WIND_ROUNDS,
                    'rotation_stride': manifest['rotation_stride'],
                },
                'source_configured_rounds': rounds,
                'source_configured_slots': len(FULL_23WIND_CONDITIONS) * rounds,
                'runtime_identity': runtime_identity,
                'bladed_launch_authorized': False,
            }
        },
        'pinned_files': dict(sorted(pinned_files.items())),
        'historical_archive': archive_record,
        'archive_sha256': archive_record['release_manifest']['sha256'],
        'authorization': authorization,
        'versioning_policy': {
            'single_active_project_release_id': OPERATOR_23WIND_RELEASE_ID,
            'rounds_are_schedule_configuration_not_parallel_code_profiles': True,
            'old_r1_is_immutable_archive': True,
            'component_schema_and_artifact_labels_are_preserved': True,
            'state20_means_observation_contract_dimension_not_release_number': True,
        },
    }


def write_release_manifest(document: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the active R2 rounds-configurable physical-shadow release identity.')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / f'{OPERATOR_23WIND_TEMPLATE.rsplit("/", 1)[0]}/release_manifest.json')
    args = parser.parse_args()
    document = build_unified_release_manifest()
    output = write_release_manifest(document, args.output)
    print(json.dumps({
        'release_id': document['release_id'],
        'profile': FULL_23WIND_PROFILE,
        'rounds_default': document['scope']['rounds_default'],
        'slots_default': document['scope']['default_slots'],
        'release_ready': document['release_ready'],
        'release_manifest': output.relative_to(PROJECT_ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
