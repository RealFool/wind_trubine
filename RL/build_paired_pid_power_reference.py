from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.raw_dataset_contract import (
    EXPECTED_CONDITION_IDS,
    EXPECTED_END_TIME_S,
    EXPECTED_START_TIME_S,
    TIME_TOLERANCE_S,
    manifest_sha256 as build_manifest_sha256,
    sha256_file,
    validate_raw_source_root,
    write_json,
)
from RL.utils.constants import DATA_SCHEMA_VERSION, ELECTRICAL_POWER_VARIABLE
from RL.utils.reward import PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION


TIME_FIELD = 'Time'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build an exact-time paired PID power reference manifest from all complete raw traces.'
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_reference(path: Path, condition_id: int) -> dict[str, Any]:
    times: list[float] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {TIME_FIELD, ELECTRICAL_POWER_VARIABLE}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing paired PID power fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row[TIME_FIELD])
                power = float(row[ELECTRICAL_POWER_VARIABLE])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid paired PID power data') from exc
            if not math.isfinite(timestamp) or not math.isfinite(power):
                raise ValueError(f'{path.name} row {row_number} has non-finite paired PID power data')
            if times and timestamp <= times[-1]:
                raise ValueError(f'{path.name} paired PID timestamps must be strictly increasing')
            times.append(timestamp)
    if len(times) < 2:
        raise ValueError(f'{path.name} paired PID power reference needs at least two rows')
    if not (
        math.isclose(times[0], EXPECTED_START_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
        and math.isclose(times[-1], EXPECTED_END_TIME_S, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S)
    ):
        raise ValueError(f'{path.name} paired PID power reference does not cover 0-600 s')
    nominal_step_s = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    return {
        'condition_id': condition_id,
        'path': _portable_path(path),
        'filename': path.name,
        'sha256': sha256_file(path),
        'rows': len(times),
        'start_time_s': times[0],
        'end_time_s': times[-1],
        'nominal_step_s': nominal_step_s,
    }


def build_paired_pid_power_reference(data_root: Path) -> dict[str, Any]:
    resolved_root, source_files = validate_raw_source_root(data_root)
    references = [
        _read_reference(path, condition_id)
        for condition_id, path in source_files
    ]
    manifest_sha256 = build_manifest_sha256(references)
    return {
        'schema_version': PAIRED_PID_POWER_REFERENCE_SCHEMA_VERSION,
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_role': 'paired_pid_outcome_reference',
            'source_data_root': str(resolved_root),
            'source_file_pattern': 'data_windspeed{condition_id}.csv',
            'reference_signal': ELECTRICAL_POWER_VARIABLE,
            'expected_condition_ids': list(EXPECTED_CONDITION_IDS),
            'condition_count': len(references),
            'expected_time_range_s': [EXPECTED_START_TIME_S, EXPECTED_END_TIME_S],
            'sampling_strategy': 'all_raw_rows_exact_timestamp',
            'input_manifest_sha256': manifest_sha256,
            'input_manifest_format': 'condition_id:filename:file_sha256\\n sorted by condition_id',
            'pairing_contract': {
                'case_identity': 'same_bladed_case',
                'turbulence_seed_identity': 'same_seed',
                'time_window_identity': 'same_simulation_time',
                'time_alignment': 'exact_timestamp_no_interpolation',
            },
            'pairing_scope': (
                'condition_id selects the operator-verified matching Bladed case and turbulence seed; '
                'the reward query must use the same simulation timestamp within 0-600 s'
            ),
        },
        'references': references,
    }


def main() -> None:
    args = parse_args()
    document = build_paired_pid_power_reference(args.data_root)
    write_json(args.output, document)
    print(json.dumps({
        'output': str(args.output),
        'schema_version': document['schema_version'],
        'condition_count': document['metadata']['condition_count'],
        'input_manifest_sha256': document['metadata']['input_manifest_sha256'],
        'rows': sum(int(item['rows']) for item in document['references']),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()