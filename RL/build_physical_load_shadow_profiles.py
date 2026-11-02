"""Derive isolated, launch-blocked physical shadow profiles; never edit v7/BC."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from RL.utils.control_contract import resolve_configured_control_contract
from RL.utils.physical_load_margin import INPUT_FIELDS, document_sha256
from RL.utils.physical_load_shadow_artifact import (
    ACTION_SEMANTICS, ARTIFACT_FILENAMES, BACKEND, BUNDLE_SCHEMA, LABEL_SEMANTICS,
    RUNTIME_AUTHORIZATION, RUNTIME_SCHEMA, artifact_path, file_sha256,
    load_physical_shadow_artifacts, strict_json,
)
from RL.utils.physical_load_shadow_launch import (
    OPERATOR_23WIND_PROFILE_BUILD_REPORT,
    OPERATOR_23WIND_ROUTE,
    OPERATOR_23WIND_RELEASE_ID,
    OPERATOR_23WIND_SOURCE_PROFILE,
    OPERATOR_23WIND_TEMPLATE,
    REQUIRED_GATES,
    physical_runtime_identity,
)
from RL.utils.physical_queue_schedule import (
    DEFAULT_FULL_23WIND_ROUNDS,
    FULL_23WIND_BASE_ORDER,
    FULL_23WIND_CONDITIONS,
    FULL_23WIND_DURATION_S,
    FULL_23WIND_PROFILE,
    PHYSICAL_SHADOW_MAX_TOTAL_BYTES,
    PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES,
    PHYSICAL_SHADOW_RECORD_MAX_BYTES,
    build_full_23wind_manifest,
    configure_full_23wind_rounds,
    validate_full_23wind_rounds,
)
from RL.utils.queue_manifest import expand_queue_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = ('RL/configs/physical_shadow_state20_v1_final_checked/'
                  'state20_physical_4wind_4round_600s_training.json')
SOURCE_PROFILE_23WIND = OPERATOR_23WIND_SOURCE_PROFILE
SOURCE_PROFILE_23WIND_MANIFEST = (
    'RL/experiments/artifacts/state20_queue_profiles_operator_23wind_round600_v2/'
    'state20_23wind_round600_source_manifest.json'
)
FROZEN_ROOT = 'RL/diagnostics/physical_load_margin_dynamic_v2'
WINDOWS_QUEUE_SOURCE = 'bladed_env/udptest/udptest/controller.cpp'
WINDOWS_RELEASE_DLL = 'bladed_env/udptest/Release/udptest.dll'
PROFILE_SPECS: dict[str, dict[str, Any]] = {
    '4wind_4round_180s_integration': {
        'rounds': 4, 'duration_s': 180.0, 'online': False,
        'stage': 'integration_only_not_acceptance',
    },
    '4wind_1round_600s_frozen_diagnostic': {
        'rounds': 1, 'duration_s': 600.0, 'online': False,
        'stage': 'frozen_policy_development_diagnostic',
    },
    '4wind_4round_600s_training': {
        'rounds': 4, 'duration_s': 600.0, 'online': True,
        'stage': 'moving_actor_training_diagnostic',
    },
    FULL_23WIND_PROFILE: {
        'rounds': DEFAULT_FULL_23WIND_ROUNDS, 'duration_s': FULL_23WIND_DURATION_S, 'online': True,
        'stage': 'moving_actor_23wind_training_diagnostic',
        'condition_ids': list(FULL_23WIND_CONDITIONS),
        'base_condition_order': list(FULL_23WIND_BASE_ORDER),
        'rotation_stride': 7,
        'shadow_max_record_bytes': PHYSICAL_SHADOW_RECORD_MAX_BYTES,
        'shadow_max_total_bytes': PHYSICAL_SHADOW_MAX_TOTAL_BYTES,
        'event_windows_max_total_bytes': 2 * 1024 ** 3,
    },
}
DEFAULT_PROFILES = (FULL_23WIND_PROFILE,)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write(path: Path, document: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def build_profiles(
    output_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    source_profile_path: Path | None = None,
    profile_names: tuple[str, ...] | None = None,
    rounds: int = DEFAULT_FULL_23WIND_ROUNDS,
) -> dict[str, Any]:
    rounds = validate_full_23wind_rounds(rounds)
    selected_profiles = DEFAULT_PROFILES if profile_names is None else profile_names
    if not selected_profiles or len(selected_profiles) != len(set(selected_profiles)):
        raise ValueError('At least one unique physical profile name is required')
    unknown_profiles = set(selected_profiles) - set(PROFILE_SPECS)
    if unknown_profiles:
        raise ValueError(f'Unknown physical profiles: {sorted(unknown_profiles)}')
    needs_23wind_source = FULL_23WIND_PROFILE in selected_profiles
    if needs_23wind_source and any(name.startswith('4wind_') for name in selected_profiles):
        raise ValueError('Build 4-wind and 23-wind physical profiles in separate versioned output directories')
    default_source = SOURCE_PROFILE_23WIND if needs_23wind_source else SOURCE_PROFILE
    source_path = (source_profile_path or (project_root / default_source)).expanduser().resolve()
    source = strict_json(source_path)
    resolved = resolve_configured_control_contract(source, project_root=project_root)
    if resolved is None or resolved[1] != source['control_contract_fingerprint']:
        raise ValueError('Frozen source control contract mismatch')
    bc = project_root / source['bc']['resume_checkpoint']
    expected_bc = source['state20_queue_contract']['bc_warmstart_checkpoint_sha256']
    if file_sha256(bc) != expected_bc:
        raise ValueError('Frozen BC checksum mismatch; do not rebuild or overwrite it implicitly')
    nn_rollback_path = source.get('load_predictor', {}).get('predictor_artifact_path')
    nn_rollback_expected_sha = source.get('load_predictor', {}).get('predictor_artifact_sha256')
    if nn_rollback_path is None and nn_rollback_expected_sha is None:
        nn_rollback_sha = None
    elif not isinstance(nn_rollback_path, str) or not isinstance(nn_rollback_expected_sha, str):
        raise ValueError('Frozen NN rollback path/SHA must both be present or both be absent')
    else:
        nn_rollback_sha = file_sha256(artifact_path(nn_rollback_path, project_root))
        if nn_rollback_sha != nn_rollback_expected_sha:
            raise ValueError('Frozen NN rollback checksum mismatch; do not silently replace it')
    source_manifest_path = Path(source['queue']['manifest_path']).expanduser()
    if not source_manifest_path.is_absolute():
        source_manifest_path = project_root / source_manifest_path
    source_manifest = strict_json(source_manifest_path)
    if needs_23wind_source:
        source_manifest = build_full_23wind_manifest(source_manifest, DEFAULT_FULL_23WIND_ROUNDS)
        source['queue']['manifest_path'] = _portable(
            project_root / SOURCE_PROFILE_23WIND_MANIFEST, project_root
        )
        source['queue']['total_condition_rounds'] = DEFAULT_FULL_23WIND_ROUNDS
        source['state20_queue_contract']['profile'] = FULL_23WIND_PROFILE
        source['state20_queue_contract']['rounds'] = DEFAULT_FULL_23WIND_ROUNDS
        source['state20_queue_contract']['expected_scheduled_updates'] = (
            (DEFAULT_FULL_23WIND_ROUNDS - 1) * len(FULL_23WIND_CONDITIONS)
            * int(round(FULL_23WIND_DURATION_S / 0.02))
            // int(source['train']['update_every_steps'])
        )
    source_condition_ids = [int(value) for value in source['queue']['condition_ids']]
    if (not source_condition_ids or len(source_condition_ids) != len(set(source_condition_ids)) or
            set(source_condition_ids) != set(int(value) for value in source_manifest['base_condition_order'])):
        raise ValueError('Frozen source profile condition IDs and manifest order do not close')
    if needs_23wind_source and (
            set(source_condition_ids) != set(FULL_23WIND_CONDITIONS) or
            set(source_condition_ids) != set(int(value) for value in source_manifest['base_condition_order']) or
            float(source.get('queue', {}).get('train_sim_s', -1.0)) != FULL_23WIND_DURATION_S):
        raise ValueError('23-wind round600 profile requires the canonical 23-condition 600-second source contract')
    if (any(name.startswith('4wind_') for name in selected_profiles) and
            set(source_condition_ids) != {8, 11, 12, 16}):
        raise ValueError('4-wind physical profiles require the exact four-wind state20 source profile')
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [path for path in output_dir.iterdir() if path.name != 'README.md']
    if existing_outputs:
        raise FileExistsError(f'Physical profile output directory already contains generated files: {output_dir}')
    frozen = project_root / FROZEN_ROOT
    report = strict_json(frozen / 'report.json')
    entries = {}
    for role, filename in ARTIFACT_FILENAMES.items():
        path = frozen / filename
        digest = file_sha256(path)
        if digest != report['artifact_file_sha256'][filename]:
            raise ValueError(f'Frozen D0-R report/artifact SHA mismatch: {filename}')
        entries[role] = {'path': _portable(path, project_root), 'file_sha256': digest,
                         'document_sha256': document_sha256(strict_json(path))}
    project_path = project_root / report['base_protocol']['static_parameters']['source_project']
    if file_sha256(project_path) != report['static_project']['sha256']:
        raise ValueError('D0-R engineering source project changed')
    bundle = {
        'schema_version': BUNDLE_SCHEMA, 'runtime_schema_version': RUNTIME_SCHEMA,
        'purpose': 'frozen PID model applied to sent RL ZOH for development transfer diagnostics only',
        'external_fields': list(INPUT_FIELDS), 'primary_horizons_s': [0.1, 0.2],
        'action_semantics': ACTION_SEMANTICS, 'label_semantics': LABEL_SEMANTICS,
        'artifacts': entries,
        'engineering_project': {'path': _portable(project_path, project_root),
                                'file_sha256': file_sha256(project_path)},
        'source_development_report_sha256': file_sha256(frozen / 'report.json'),
        'legacy_action_provenance_verified': False,
        'rl_action_response_validated': False,
        'authorization': dict(RUNTIME_AUTHORIZATION),
    }
    bundle_path = output_dir / 'physical_load_shadow_bundle.json'
    _write(bundle_path, bundle)
    bundle_sha = file_sha256(bundle_path)
    loaded = load_physical_shadow_artifacts(bundle_path, expected_sha256=bundle_sha, project_root=project_root)
    profiles = {}
    for name in selected_profiles:
        spec = PROFILE_SPECS[name]
        profile_rounds = rounds if name == FULL_23WIND_PROFILE else int(spec['rounds'])
        duration = float(spec['duration_s'])
        online = bool(spec['online'])
        stage = str(spec['stage'])
        config = copy.deepcopy(source)
        manifest = build_full_23wind_manifest(source_manifest, profile_rounds) if name == FULL_23WIND_PROFILE else copy.deepcopy(source_manifest)
        condition_count = len(FULL_23WIND_CONDITIONS) if name == FULL_23WIND_PROFILE else len(source_condition_ids)
        if name != FULL_23WIND_PROFILE:
            manifest.update(total_condition_rounds=profile_rounds, train_sim_s=duration,
                            expanded_slot_count=profile_rounds * condition_count, manual_batch_rounds=profile_rounds,
                            state20_profile=name,
                            notes='Physical FIR development shadow; no NN/guard authorization.')
        manifest_path = output_dir / f'state20_physical_{name}_manifest.json'
        _write(manifest_path, manifest)
        slots = expand_queue_manifest(manifest)
        if (len(slots) != profile_rounds * condition_count or
                any(slot.train_sim_s != duration for slot in slots)):
            raise ValueError('Physical shadow queue slot count mismatch')
        if name == FULL_23WIND_PROFILE:
            config['release_id'] = OPERATOR_23WIND_RELEASE_ID
            config['experiment_name'] = 'b4_state20_physical_23wind_round600_training'
        else:
            config['experiment_name'] = f'b4_state20_physical_{name}'
        config['output_dir'] = 'RL/experiments'
        config['bc']['export_checkpoint_path'] = ''
        config['train']['online_rl_enabled'] = online
        config['queue'].update(manifest_path=_portable(manifest_path, project_root),
                               total_condition_rounds=profile_rounds, train_sim_s=duration)
        if not online:
            config['queue']['residual_scale_diagnostic'] = 1.0
        evaluation_path = project_root / f'RL/data/paired_pid_power_load_reference_3_to_25_{int(duration)}s_v2.json'
        if not evaluation_path.is_file():
            raise FileNotFoundError('Exact-window paired reference is required before building physical profiles')
        evaluation_sha = file_sha256(evaluation_path)
        config['queue']['case_identity_validation']['paired_pid_reference_path'] = _portable(evaluation_path, project_root)
        config['state20_queue_contract'].update(profile=name, condition_ids=source_condition_ids,
            rounds=profile_rounds, train_sim_s=duration,
            expected_scheduled_updates=(profile_rounds - 1) * condition_count * int(round(duration / 0.02)) //
                config['train']['update_every_steps'] if online else 0,
            predictor_backend=BACKEND, predictor_mode='shadow', policy_acceptance=False)
        # The pre-existing resolved interface is retained as a legacy Actor/BC
        # compatibility dependency, not relabeled as a physical NN interface.
        config['load_predictor'] = {
            'backend': BACKEND, 'mode': 'shadow', 'nn_active': False,
            'bundle_path': _portable(bundle_path, project_root), 'bundle_sha256': bundle_sha,
            'evaluation_reference': {'path': _portable(evaluation_path, project_root),
                                     'file_sha256': evaluation_sha, 'time_window_s': [0.0, duration]},
            'origin_stride_frames': PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES, 'worker_queue_capacity': 256,
            'condition_record_capacity': 16384,
            'max_records': (len(slots) * 2 * (int(round(duration / 0.02)) // PHYSICAL_SHADOW_ORIGIN_STRIDE_FRAMES + 1)
                            if name == FULL_23WIND_PROFILE else int(spec.get('shadow_max_records', 250000))),
            'max_record_bytes': int(spec.get('shadow_max_record_bytes', 32768)),
            'max_total_bytes': int(spec.get('shadow_max_total_bytes', 1073741824)),
            'authorization': dict(RUNTIME_AUTHORIZATION),
        }
        if name == FULL_23WIND_PROFILE:
            configure_full_23wind_rounds(config, manifest_path, profile_rounds, project_root=project_root)
        if 'event_windows_max_total_bytes' in spec:
            config['logging']['queue_event_windows_max_total_bytes'] = int(
                spec['event_windows_max_total_bytes'])
        if name == FULL_23WIND_PROFILE:
            config['physical_shadow_launch'] = {
                'authorized': False,
                'route': OPERATOR_23WIND_ROUTE,
                            'slots': len(slots),
                'evidence_path': '',
                'evidence_sha256': '',
                'decision': 'BLOCKED_PENDING_OPERATOR_CONFIRMATION_FINALIZE',
                'required_gates': list(REQUIRED_GATES),
            }
        else:
            config['physical_shadow_launch'] = {
                'authorized': False, 'evidence_path': None, 'evidence_sha256': None,
                'decision': 'BLOCKED_PENDING_OFFLINE_WINDOWS_PROVENANCE_TARGET_AND_V1_EVIDENCE',
                'required_gates': list(REQUIRED_GATES),
            }
        config['physical_shadow_run_role'] = {
            'stage': stage, 'independent_audit': False, 'new_independent_seed': False,
            'policy_acceptance': False, 'soft_enabled': False, 'hard_enabled': False,
        }
        if resolve_configured_control_contract(config, project_root=project_root) != resolved:
            raise ValueError('Physical shadow must not silently change the Actor resolved contract')
        config_path = output_dir / f'state20_physical_{name}.json'
        _write(config_path, config)
        profiles[name] = {
            'release_id': config.get('release_id'),
            'config_path': _portable(config_path, project_root), 'config_sha256': file_sha256(config_path),
            'manifest_path': _portable(manifest_path, project_root), 'slots': len(slots),
            'manifest_sha256': file_sha256(manifest_path),
            'runtime_identity': physical_runtime_identity(config, project_root),
            'bladed_launch_authorized': False,
        }
    summary = {
        'schema_version': 'physical_load_shadow_profile_build_v1',
        'release_id': OPERATOR_23WIND_RELEASE_ID if needs_23wind_source else None,
        'source_profile_path': _portable(source_path, project_root),
        'source_profile_sha256': file_sha256(source_path),
        'windows_queue_source_path': WINDOWS_QUEUE_SOURCE,
        'windows_queue_source_sha256': file_sha256(project_root / WINDOWS_QUEUE_SOURCE),
        'expected_windows_release_dll_path': WINDOWS_RELEASE_DLL,
        'expected_windows_release_dll_sha256': file_sha256(project_root / WINDOWS_RELEASE_DLL),
        'bc_checkpoint_sha256': file_sha256(bc),
        'nn_rollback_artifact_sha256': nn_rollback_sha,
        'bundle_sha256': bundle_sha, 'strict_loader_roundtrip': 'PASS',
        'dynamic_model_sha256': document_sha256(loaded.model.to_document()),
        'profiles': profiles, 'bladed_started': False,
        'decision': 'DEVELOPMENT_PROFILES_BUILT_REAL_LAUNCH_BLOCKED',
    }
    _write(output_dir / 'profile_build_report.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'RL/configs' / OPERATOR_23WIND_RELEASE_ID)
    parser.add_argument('--source-profile', type=Path, default=None)
    parser.add_argument('--profile', action='append', choices=sorted(PROFILE_SPECS), dest='profiles')
    parser.add_argument('--rounds', type=int, default=DEFAULT_FULL_23WIND_ROUNDS,
                        help='Rounds for the unified 23-wind × 600-second training profile (default: 4).')
    args = parser.parse_args()
    report = build_profiles(args.output_dir.resolve(), source_profile_path=args.source_profile,
                             profile_names=None if args.profiles is None else tuple(args.profiles),
                             rounds=args.rounds)
    print(json.dumps({'decision': report['decision'], 'output_dir': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()