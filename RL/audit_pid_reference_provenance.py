from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.raw_dataset_contract import manifest_sha256, sha256_file, validate_raw_source_root


SCHEMA_VERSION = 'pid_reference_provenance_v1'
TORQUE_DEMAND_FIELD = 'O_PcsTorqueDemand'
TORQUE_MEASUREMENT_FIELD = 'I_ElectricalTorque'
PITCH_DEMAND_FIELD = 'O_PitchPositionDemand1'
PITCH_MEASUREMENT_FIELD = 'I_PitchPosition1'
RAW_FIELD_PAIRS = {
    'torque': (TORQUE_DEMAND_FIELD, TORQUE_MEASUREMENT_FIELD),
    'pitch': (PITCH_DEMAND_FIELD, PITCH_MEASUREMENT_FIELD),
}
EXPECTED_UDP_ASSIGNMENT_SOURCES = {
    TORQUE_DEMAND_FIELD: TORQUE_MEASUREMENT_FIELD,
    PITCH_DEMAND_FIELD: PITCH_MEASUREMENT_FIELD,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit whether raw O_* action fields are recorded PID demands or unresolved.'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help="Raw directory containing exactly data_windspeed3.csv through data_windspeed25.csv.",
    )
    parser.add_argument(
        '--controller-source',
        type=Path,
        default=PROJECT_ROOT / 'bladed_env' / 'udptest' / 'udptest' / 'controller.cpp',
    )
    parser.add_argument(
        '--api-header',
        type=Path,
        default=PROJECT_ROOT / 'bladed_env' / 'BladedAPI' / '4.8' / 'ExternalControllerApi.h',
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def _finite(raw_value: str, *, field: str, path: Path, row_number: int) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(value):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return value


def _read_raw_comparison(path: Path, condition_id: int) -> dict[str, Any]:
    required_fields = {field for pair in RAW_FIELD_PAIRS.values() for field in pair}
    pair_metrics = {
        name: {
            'demand_field': demand_field,
            'measurement_field': measurement_field,
            'exact_match_rows': 0,
            'mismatch_rows': 0,
            'max_abs_difference': 0.0,
            'first_mismatch': None,
        }
        for name, (demand_field, measurement_field) in RAW_FIELD_PAIRS.items()
    }
    rows_compared = 0
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(required_fields - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f'{path.name} is missing provenance fields: {missing_fields}')
        for row_number, row in enumerate(reader, start=2):
            rows_compared += 1
            for name, (demand_field, measurement_field) in RAW_FIELD_PAIRS.items():
                demand = _finite(row[demand_field], field=demand_field, path=path, row_number=row_number)
                measurement = _finite(
                    row[measurement_field],
                    field=measurement_field,
                    path=path,
                    row_number=row_number,
                )
                difference = abs(demand - measurement)
                pair_metrics[name]['max_abs_difference'] = max(
                    pair_metrics[name]['max_abs_difference'],
                    difference,
                )
                if demand == measurement:
                    pair_metrics[name]['exact_match_rows'] += 1
                else:
                    pair_metrics[name]['mismatch_rows'] += 1
                    if pair_metrics[name]['first_mismatch'] is None:
                        pair_metrics[name]['first_mismatch'] = {
                            'csv_row_number': row_number,
                            'demand_value': demand,
                            'measurement_value': measurement,
                        }
    if rows_compared == 0:
        raise ValueError(f'{path.name} has no data rows')
    return {
        'condition_id': condition_id,
        'path': str(path),
        'filename': path.name,
        'sha256': sha256_file(path),
        'rows_compared': rows_compared,
        **pair_metrics,
    }


def _extract_udp_assignment_source(source: str, slot_name: str) -> str:
    pattern = re.compile(
        rf'SetField\s*\(\s*fields\s*,\s*{re.escape(slot_name)}\s*,\s*([^\)]+?)\s*\)\s*;'
    )
    matches = [match.strip() for match in pattern.findall(source)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one UDP assignment for {slot_name}, found {len(matches)}')
    return matches[0]


def _build_dll_evidence(controller_path: Path, api_header_path: Path) -> dict[str, Any]:
    try:
        controller_source = controller_path.read_text(encoding='utf-8')
        api_header = api_header_path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'PID provenance source file does not exist: {exc.filename}') from exc

    torque_source = _extract_udp_assignment_source(controller_source, 'kTorqueDemand')
    pitch_source = _extract_udp_assignment_source(controller_source, 'kPitchDemand1')
    assignment_sources = {
        TORQUE_DEMAND_FIELD: torque_source,
        PITCH_DEMAND_FIELD: pitch_source,
    }
    send_index = controller_source.find('SendToPython(udpFrame.c_str())')
    reply_index = controller_source.find('ReceiveMatchingReply(', send_index + 1)
    torque_apply_index = controller_source.find(
        'SetDemandedGeneratorTorque(turbine_id, new_torque)',
        reply_index + 1,
    )
    pitch_apply_index = controller_source.find(
        'ApplyCollectivePitchDemand(turbine_id, new_pitch)',
        reply_index + 1,
    )
    frame_sent_before_reply = send_index >= 0 and reply_index > send_index
    reply_before_action_applied = (
        reply_index >= 0
        and torque_apply_index > reply_index
        and pitch_apply_index > reply_index
    )
    demanded_getters = {
        'torque': 'GetDemandedGeneratorTorque',
        'pitch': 'GetDemandedCollectivePitchAngle',
    }
    demanded_getters_declared = {
        name: getter in api_header
        for name, getter in demanded_getters.items()
    }
    demanded_getters_used_as_udp_sources = {
        'torque': demanded_getters['torque'] in torque_source,
        'pitch': demanded_getters['pitch'] in pitch_source,
    }
    return {
        'controller_source': str(controller_path),
        'controller_source_sha256': sha256_file(controller_path),
        'api_header': str(api_header_path),
        'api_header_sha256': sha256_file(api_header_path),
        'udp_assignment_sources': assignment_sources,
        'measurement_alias_assignments': assignment_sources == EXPECTED_UDP_ASSIGNMENT_SOURCES,
        'demanded_getters_declared': demanded_getters_declared,
        'demanded_getters_used_as_udp_sources': demanded_getters_used_as_udp_sources,
        'frame_sent_before_reply_received': frame_sent_before_reply,
        'reply_received_before_action_applied': reply_before_action_applied,
    }


def build_provenance_audit(
    data_root: Path,
    controller_path: Path,
    api_header_path: Path,
) -> dict[str, Any]:
    resolved_root, source_files = validate_raw_source_root(data_root)
    file_audits = [
        _read_raw_comparison(path, condition_id)
        for condition_id, path in source_files
    ]
    rows_compared = sum(int(item['rows_compared']) for item in file_audits)
    aggregate_pairs: dict[str, Any] = {}
    for pair_name in RAW_FIELD_PAIRS:
        aggregate_pairs[pair_name] = {
            'demand_field': file_audits[0][pair_name]['demand_field'],
            'measurement_field': file_audits[0][pair_name]['measurement_field'],
            'exact_match_rows': sum(int(item[pair_name]['exact_match_rows']) for item in file_audits),
            'mismatch_rows': sum(int(item[pair_name]['mismatch_rows']) for item in file_audits),
            'max_abs_difference': max(float(item[pair_name]['max_abs_difference']) for item in file_audits),
        }
    all_rows_exact = all(
        aggregate_pairs[pair_name]['exact_match_rows'] == rows_compared
        and aggregate_pairs[pair_name]['mismatch_rows'] == 0
        for pair_name in RAW_FIELD_PAIRS
    )
    recorded_pid_demand_trace = all(
        aggregate_pairs[pair_name]['mismatch_rows'] > 0
        and aggregate_pairs[pair_name]['max_abs_difference'] > 0.0
        for pair_name in RAW_FIELD_PAIRS
    )
    dll_evidence = _build_dll_evidence(controller_path.resolve(), api_header_path.resolve())
    demanded_getters_are_udp_sources = all(
        bool(value)
        for value in dll_evidence['demanded_getters_used_as_udp_sources'].values()
    )
    native_pid_demand_available = bool(
        demanded_getters_are_udp_sources
        and dll_evidence['frame_sent_before_reply_received']
    )
    if recorded_pid_demand_trace:
        udp_action_contract = 'recorded_pid_demand'
        baseline_reference_path = 'region_aware_surrogate_required'
        status = 'conclusive_recorded_pid_demand'
    elif native_pid_demand_available:
        udp_action_contract = 'native_pid_demand'
        baseline_reference_path = 'native_shadow_pid_available'
        status = 'conclusive_native_pid_demand'
    else:
        udp_action_contract = 'unresolved'
        baseline_reference_path = 'provenance_reconciliation_required'
        status = 'inconclusive'

    return {
        'schema_version': SCHEMA_VERSION,
        'source_role': 'raw_pid_reference_provenance',
        'source_manifest': {
            'data_root': str(resolved_root),
            'expected_condition_ids': list(range(3, 26)),
            'file_count': len(file_audits),
            'manifest_sha256': manifest_sha256(file_audits),
            'files': file_audits,
        },
        'signal_contract': {
            'torque_unit': 'Nm',
            'pitch_unit': 'rad',
            'python_reply_timing': 'received_after_current_frame_is_sent',
            'external_action_effect': 'applied_after_current_frame_is_sent',
        },
        'raw_comparison': {
            'rows_compared': rows_compared,
            **aggregate_pairs,
            'all_rows_exact_measurement_aliases': all_rows_exact,
        },
        'dll_static_evidence': dll_evidence,
        'decision': {
            'status': status,
            'native_pid_demand_available': native_pid_demand_available,
            'recorded_pid_demand_trace_available': recorded_pid_demand_trace,
            'udp_action_contract': udp_action_contract,
            'baseline_reference_path': baseline_reference_path,
            'current_o_fields_are_safe_as_recorded_pid_demand': recorded_pid_demand_trace,
            'current_o_fields_are_safe_as_native_pid_command': native_pid_demand_available,
        },
    }


def main() -> None:
    args = parse_args()
    audit = build_provenance_audit(args.data_root, args.controller_source, args.api_header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f"Wrote {args.output}: status={audit['decision']['status']} "
        f"rows={audit['raw_comparison']['rows_compared']} "
        f"manifest={audit['source_manifest']['manifest_sha256']}"
    )


if __name__ == '__main__':
    main()