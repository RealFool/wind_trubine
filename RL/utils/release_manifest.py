from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


RELEASE_MANIFEST_SCHEMA_VERSION = 'state20_candidate_release_manifest_v1'
EXPECTED_STATE20_CONDITION_IDS = tuple(range(3, 26))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read {label} JSON: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} JSON must contain an object: {path}')
    return value


def _resolve_artifact(path_value: str | Path, *, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'Release artifact does not exist: {resolved}')
    return resolved


def _artifact_record(path: Path, *, project_root: Path) -> dict[str, str]:
    try:
        display_path = path.relative_to(project_root).as_posix()
    except ValueError:
        display_path = str(path)
    return {'path': display_path, 'sha256': sha256_file(path)}


def _scenario_acceptance_gates(acceptance: Mapping[str, Any] | None) -> dict[str, bool]:
    if acceptance is None:
        return {
            'acceptance_covers_all_23_conditions': False,
            'all_expected_scenarios_valid': False,
            'official_acceptance_pass': False,
            'strict_per_blade_pass': False,
            'strict_case_identity_pass': False,
            'edge_T1_pass_all_scenarios': False,
            'release_candidate_pass': False,
        }
    expected = [str(value) for value in EXPECTED_STATE20_CONDITION_IDS]
    return {
        'acceptance_covers_all_23_conditions': (
            acceptance.get('expected_scenario_ids') == expected
            and int(acceptance.get('scenario_count', 0)) == len(expected)
            and int(acceptance.get('valid_scenario_count', 0)) == len(expected)
            and not acceptance.get('unexpected_scenario_ids', [])
        ),
        'all_expected_scenarios_valid': bool(acceptance.get('all_expected_scenarios_valid', False)),
        'official_acceptance_pass': bool(acceptance.get('official_acceptance_pass', False)),
        'strict_per_blade_pass': bool(acceptance.get('strict_per_blade_pass', False)),
        'strict_case_identity_pass': bool(acceptance.get('strict_case_identity_pass', False)),
        'edge_T1_pass_all_scenarios': bool(acceptance.get('edge_T1_pass_all_scenarios', False)),
        'release_candidate_pass': bool(acceptance.get('release_candidate_pass', False)),
    }


def _predictor_split_provenance_pass(
    summary: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    interface: Mapping[str, Any],
    *,
    interface_sha256: str,
) -> bool:
    """Check that v4 predictor split metadata closes against its frozen source files."""
    partition_names = ('train', 'validation', 'calibration', 'coverage_audit')
    try:
        condition_partitions = summary['condition_partitions']
        split_counts = summary['split_sample_counts']
        plan_counts = summary['plan_counts_by_split']
        time_partitions = summary['time_partitions_s']
        source_conditions = {int(value) for value in dataset_manifest['source_conditions']}
        if (
            not source_conditions
            or not isinstance(condition_partitions, Mapping)
            or set(condition_partitions) != set(partition_names)
            or not isinstance(split_counts, Mapping)
            or set(split_counts) != set(partition_names)
            or not isinstance(plan_counts, Mapping)
            or set(plan_counts) != set(partition_names)
            or not isinstance(time_partitions, Mapping)
            or set(time_partitions) != set(partition_names)
            or dataset_manifest.get('schema_version') != 'load_predictor_dataset_v2'
            or dataset_manifest.get('predictor_interface_schema_version') != 'causal_load_predictor_interface_v3'
            or dataset_manifest.get('predictor_interface_sha256') != interface_sha256
            or interface.get('schema_version') != 'causal_load_predictor_interface_v3'
            or summary.get('interface_schema_version') != 'causal_load_predictor_interface_v3'
            or summary.get('interface_sha256') != interface_sha256
        ):
            return False

        flattened_conditions: list[int] = []
        for name in partition_names:
            values = condition_partitions[name]
            if not isinstance(values, list) or not values:
                return False
            flattened_conditions.extend(int(value) for value in values)
            count = int(split_counts[name])
            counts = plan_counts[name]
            if (
                count <= 0
                or not isinstance(counts, Mapping)
                or int(counts.get('realized_plan', -1)) != 0
                or int(counts.get('zoh_candidate', -1)) != count
            ):
                return False
        if len(flattened_conditions) != len(set(flattened_conditions)):
            return False
        if set(flattened_conditions) != source_conditions:
            return False

        history_frames = int(interface['history_frames'])
        step_duration_s = float(interface['step_duration_s'])
        horizons_s = [float(value) for value in interface['horizons_s']]
        source_start_s = float(dataset_manifest['time_window_start_s'])
        source_end_s = float(dataset_manifest['time_window_end_s'])
        if (
            history_frames <= 0
            or not math.isfinite(step_duration_s)
            or step_duration_s <= 0.0
            or not horizons_s
            or any(not math.isfinite(value) or value <= 0.0 for value in horizons_s)
            or not math.isfinite(source_start_s)
            or not math.isfinite(source_end_s)
            or source_end_s <= source_start_s
        ):
            return False
        max_horizon_frames = max(int(round(value / step_duration_s)) for value in horizons_s)
        purge_s = (history_frames - 1 + max_horizon_frames) * step_duration_s
        bounds: dict[str, tuple[float, float]] = {}
        for name in partition_names:
            raw_bounds = time_partitions[name]
            if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
                return False
            start_s, end_s = float(raw_bounds[0]), float(raw_bounds[1])
            if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
                return False
            bounds[name] = (start_s, end_s)
        if not math.isclose(bounds['train'][0], source_start_s, rel_tol=0.0, abs_tol=1.0e-6):
            return False
        if not math.isclose(
            bounds['coverage_audit'][1],
            source_end_s + step_duration_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            return False
        for previous, following in zip(partition_names, partition_names[1:]):
            if bounds[following][0] - bounds[previous][1] + 1.0e-12 < purge_s:
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def build_release_manifest(
    *,
    candidate_id: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    paired_pid_reference_path: str | Path,
    predictor_artifact_path: str | Path,
    dll_evidence_path: str | Path,
    resource_evidence_path: str | Path,
    project_root: str | Path,
    acceptance_report_path: str | Path | None = None,
    calibration_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if not candidate_id.strip():
        raise ValueError('candidate_id must be non-empty')
    checkpoint = _resolve_artifact(checkpoint_path, project_root=root)
    config_file = _resolve_artifact(config_path, project_root=root)
    pid_reference = _resolve_artifact(paired_pid_reference_path, project_root=root)
    predictor = _resolve_artifact(predictor_artifact_path, project_root=root)
    dll_evidence_file = _resolve_artifact(dll_evidence_path, project_root=root)
    resource_evidence_file = _resolve_artifact(resource_evidence_path, project_root=root)
    config = _load_json(config_file, 'candidate config')
    dll_evidence = _load_json(dll_evidence_file, 'Windows DLL evidence')
    resource_evidence = _load_json(resource_evidence_file, 'resource budget evidence')
    acceptance_path = None if acceptance_report_path is None else _resolve_artifact(
        acceptance_report_path,
        project_root=root,
    )
    acceptance = None if acceptance_path is None else _load_json(acceptance_path, 'multi-scenario acceptance')
    calibration_path = None if calibration_artifact_path is None else _resolve_artifact(
        calibration_artifact_path,
        project_root=root,
    )
    calibration = None if calibration_path is None else _load_json(calibration_path, 'load guard calibration')

    queue = config.get('queue', {})
    identity = queue.get('case_identity_validation', {}) if isinstance(queue, Mapping) else {}
    strict_identity_configured = bool(
        isinstance(identity, Mapping)
        and identity.get('enabled') is True
        and identity.get('validation_level') == 'strict'
    )
    configured_conditions = tuple(sorted({
        int(value) for value in queue.get('condition_ids', ())
    })) if isinstance(queue, Mapping) else ()
    strict_queue_contract = strict_identity_configured and configured_conditions == EXPECTED_STATE20_CONDITION_IDS

    predictor_config = config.get('load_predictor', {})
    predictor_mode = str(predictor_config.get('mode', '')) if isinstance(predictor_config, Mapping) else ''
    configured_predictor_sha = (
        str(predictor_config.get('predictor_artifact_sha256', '')).strip().lower()
        if isinstance(predictor_config, Mapping)
        else ''
    )
    configured_predictor_manifest_sha = (
        str(predictor_config.get('dataset_manifest_sha256', '')).strip().lower()
        if isinstance(predictor_config, Mapping)
        else ''
    )
    actual_predictor_sha = sha256_file(predictor)
    configured_predictor_path = (
        str(predictor_config.get('predictor_artifact_path', '')).strip()
        if isinstance(predictor_config, Mapping)
        else ''
    )
    if configured_predictor_path:
        configured_predictor_file = Path(configured_predictor_path).expanduser()
        if not configured_predictor_file.is_absolute():
            configured_predictor_file = root / configured_predictor_file
        predictor_path_matches_config = configured_predictor_file.resolve() == predictor
    else:
        predictor_path_matches_config = False
    configured_time_partitions = (
        predictor_config.get('predictor_time_partitions_s')
        if isinstance(predictor_config, Mapping)
        else None
    )
    predictor_summary_path = predictor.with_name(f'{predictor.stem}.summary.json')
    if predictor_summary_path.is_file():
        predictor_summary = _load_json(predictor_summary_path, 'predictor training summary')
        predictor_summary_time_partitions = predictor_summary.get('time_partitions_s')
    else:
        predictor_summary = None
        predictor_summary_time_partitions = None
    predictor_interface_path_matches_config = False
    predictor_dataset_manifest_matches_config = False
    predictor_v4_contract_pass = False
    predictor_v4_split_provenance_pass = False
    if isinstance(predictor_config, Mapping) and isinstance(predictor_summary, Mapping):
        configured_interface_path = predictor_config.get('interface_path')
        configured_dataset_manifest_path = predictor_config.get('dataset_manifest_path')
        interface_sha_matches = False
        interface_document: dict[str, Any] | None = None
        dataset_manifest_document: dict[str, Any] | None = None
        if isinstance(configured_interface_path, str) and configured_interface_path.strip():
            resolved_interface_path = Path(configured_interface_path).expanduser()
            if not resolved_interface_path.is_absolute():
                resolved_interface_path = root / resolved_interface_path
            if resolved_interface_path.is_file():
                interface_sha_matches = sha256_file(resolved_interface_path) == predictor_summary.get('interface_sha256')
                predictor_interface_path_matches_config = interface_sha_matches
                if interface_sha_matches:
                    interface_document = _load_json(resolved_interface_path, 'predictor interface')
        if isinstance(configured_dataset_manifest_path, str) and configured_dataset_manifest_path.strip():
            resolved_dataset_manifest = Path(configured_dataset_manifest_path).expanduser()
            if not resolved_dataset_manifest.is_absolute():
                resolved_dataset_manifest = root / resolved_dataset_manifest
            if resolved_dataset_manifest.is_file():
                configured_dataset_root = predictor_config.get('dataset_root')
                expected_manifest_path = None
                if isinstance(configured_dataset_root, str) and configured_dataset_root.strip():
                    expected_manifest_path = Path(configured_dataset_root).expanduser()
                    if not expected_manifest_path.is_absolute():
                        expected_manifest_path = root / expected_manifest_path
                    expected_manifest_path = expected_manifest_path / 'manifest.json'
                predictor_dataset_manifest_matches_config = bool(
                    expected_manifest_path is not None
                    and expected_manifest_path.resolve() == resolved_dataset_manifest.resolve()
                    and sha256_file(resolved_dataset_manifest) == configured_predictor_manifest_sha
                    and configured_predictor_manifest_sha == predictor_summary.get('dataset_manifest_sha256')
                )
                if predictor_dataset_manifest_matches_config:
                    dataset_manifest_document = _load_json(resolved_dataset_manifest, 'predictor dataset manifest')
        if (
            interface_document is not None
            and dataset_manifest_document is not None
            and predictor_interface_path_matches_config
            and predictor_dataset_manifest_matches_config
        ):
            predictor_v4_split_provenance_pass = _predictor_split_provenance_pass(
                predictor_summary,
                dataset_manifest_document,
                interface_document,
                interface_sha256=str(predictor_summary.get('interface_sha256', '')),
            )
        predictor_v4_contract_pass = bool(
            predictor_summary.get('schema_version') == 'load_predictor_checkpoint_v4_candidate_conditioned'
            and predictor_summary.get('checkpoint_sha256') == actual_predictor_sha
            and predictor_summary.get('interface_schema_version') == 'causal_load_predictor_interface_v3'
            and predictor_config.get('interface_schema_version') == 'causal_load_predictor_interface_v3'
            and predictor_summary.get('dataset_schema_version') == 'load_predictor_dataset_v2'
            and predictor_config.get('dataset_schema_version') == 'load_predictor_dataset_v2'
            and predictor_config.get('plan_input_semantics') == 'candidate_plan+horizon_index+plan_kind_index'
            and predictor_config.get('candidate_plan_encoding')
            == 'real_action_sequence_gru_normalized_by_engineering_bounds_v1'
            and interface_sha_matches
            and predictor_dataset_manifest_matches_config
            and predictor_v4_split_provenance_pass
            and predictor_summary.get('guard_calibration_authorized') is False
        )
    predictor_time_partitions_match_config = bool(
        isinstance(configured_time_partitions, Mapping)
        and configured_time_partitions == predictor_summary_time_partitions
    )
    predictor_sha_matches_config = bool(configured_predictor_sha) and configured_predictor_sha == actual_predictor_sha
    if not predictor_sha_matches_config:
        release_candidate_config = config.get('release_candidate', {})
        expected_release_predictor_sha = (
            str(release_candidate_config.get('predictor_artifact_sha256', '')).strip().lower()
            if isinstance(release_candidate_config, Mapping)
            else ''
        )
        predictor_sha_matches_config = bool(expected_release_predictor_sha) and (
            expected_release_predictor_sha == actual_predictor_sha
        )
    predictor_mode_supported = predictor_mode == 'shadow'
    if predictor_mode in {'soft', 'hard_candidate'}:
        calibration_metadata = calibration.get('metadata', {}) if calibration is not None else {}
        predictor_mode_supported = bool(
            calibration is not None
            and isinstance(calibration_metadata, Mapping)
            and calibration_metadata.get('soft_guard_authorized', False)
        )

    dll_compile_pass = bool(dll_evidence.get('compiler_pass', False))
    dll_protocol_pass = bool(dll_evidence.get('queue_protocol_v2_pass', False))
    dll_dynamic_rounds_pass = bool(dll_evidence.get('full_dynamic_rounds_pass', False))
    resource_pass = bool(resource_evidence.get('pass', False))
    paired_reference_sha = sha256_file(pid_reference)
    config_reference_sha = ''
    if isinstance(identity, Mapping):
        configured_reference_path = identity.get('paired_pid_reference_path')
        if isinstance(configured_reference_path, str) and configured_reference_path.strip():
            reference_path = Path(configured_reference_path).expanduser()
            if not reference_path.is_absolute():
                reference_path = root / reference_path
            if reference_path.is_file():
                config_reference_sha = sha256_file(reference_path)
    strict_reference_matches = bool(config_reference_sha) and config_reference_sha == paired_reference_sha

    acceptance_gates = _scenario_acceptance_gates(acceptance)
    offline_gates = {
        'strict_queue_identity_configured_for_all_23_conditions': strict_queue_contract,
        'paired_reference_matches_strict_config': strict_reference_matches,
        'predictor_artifact_path_matches_config': predictor_path_matches_config,
        'predictor_artifact_sha_matches_config': predictor_sha_matches_config,
        'predictor_time_partitions_match_config': predictor_time_partitions_match_config,
        'predictor_v4_candidate_plan_contract_pass': predictor_v4_contract_pass,
        'predictor_v4_split_provenance_pass': predictor_v4_split_provenance_pass,
        'predictor_interface_file_matches_config': predictor_interface_path_matches_config,
        'predictor_dataset_manifest_matches_config': predictor_dataset_manifest_matches_config,
        'predictor_mode_supported': predictor_mode_supported,
        'windows_dll_compile_pass': dll_compile_pass,
        'windows_queue_protocol_pass': dll_protocol_pass,
        'windows_dynamic_round_profiles_pass': dll_dynamic_rounds_pass,
        'resource_budget_pass': resource_pass,
    }
    offline_ready_for_v1 = all(offline_gates.values())
    release_ready = offline_ready_for_v1 and all(acceptance_gates.values())

    return {
        'schema_version': RELEASE_MANIFEST_SCHEMA_VERSION,
        'candidate_id': candidate_id.strip(),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'artifacts': {
            'actor_checkpoint': _artifact_record(checkpoint, project_root=root),
            'candidate_config': _artifact_record(config_file, project_root=root),
            'paired_pid_reference': _artifact_record(pid_reference, project_root=root),
            'predictor': _artifact_record(predictor, project_root=root),
            'calibration': None if calibration_path is None else _artifact_record(calibration_path, project_root=root),
            'windows_dll_evidence': _artifact_record(dll_evidence_file, project_root=root),
            'resource_evidence': _artifact_record(resource_evidence_file, project_root=root),
            'acceptance_report': None if acceptance_path is None else _artifact_record(acceptance_path, project_root=root),
        },
        'contracts': {
            'control_contract_fingerprint': config.get('control_contract_fingerprint'),
            'actor_control_fingerprint': config.get('actor_control_fingerprint'),
            'resolved_control_contract_path': config.get('resolved_control_contract_path'),
            'predictor_interface_schema_version': (
                predictor_config.get('interface_schema_version')
                if isinstance(predictor_config, Mapping)
                else None
            ),
            'predictor_plan_input_semantics': (
                predictor_config.get('plan_input_semantics')
                if isinstance(predictor_config, Mapping)
                else None
            ),
            'predictor_candidate_plan_encoding': (
                predictor_config.get('candidate_plan_encoding')
                if isinstance(predictor_config, Mapping)
                else None
            ),
            'predictor_dataset_schema_version': (
                predictor_config.get('dataset_schema_version')
                if isinstance(predictor_config, Mapping)
                else None
            ),
            'predictor_dataset_manifest_sha256': configured_predictor_manifest_sha,
            'predictor_mode': predictor_mode,
            'expected_condition_ids': list(configured_conditions),
            'paired_reference_sha256': paired_reference_sha,
        },
        'gates': {
            **offline_gates,
            **acceptance_gates,
            'offline_ready_for_v1': offline_ready_for_v1,
            'release_ready': release_ready,
        },
        'release_status': 'release_ready' if release_ready else (
            'ready_for_v1' if offline_ready_for_v1 and acceptance is None else 'blocked'
        ),
        'acceptance_summary': None if acceptance is None else {
            key: acceptance.get(key)
            for key in (
                'schema_version',
                'scenario_count',
                'valid_scenario_count',
                'expected_scenario_ids',
                'official_acceptance_pass',
                'strict_per_blade_pass',
                'strict_case_identity_pass',
                'edge_T1_pass_all_scenarios',
                'release_candidate_pass',
                'deployment_release_pass',
            )
        },
        'evidence_summary': {
            'dll_evidence': dll_evidence,
            'resource_evidence': resource_evidence,
            'calibration_soft_guard_authorized': bool(
                calibration is not None
                and calibration.get('metadata', {}).get('soft_guard_authorized', False)
            ),
        },
    }


def write_release_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path
