from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

from RL.utils.constants import DATA_SCHEMA_VERSION, ELECTRICAL_POWER_VARIABLE
from RL.utils.raw_dataset_contract import (
    EXPECTED_CONDITION_IDS,
    TIME_TOLERANCE_S,
    manifest_sha256,
    sha256_file,
    validate_raw_source_root,
)


PAIRED_PID_REFERENCE_SCHEMA_VERSION = 'paired_pid_power_load_reference_v2'
TIME_FIELD = 'Time'
WIND_SPEED_FIELD = 'I_AverageWindSpeed'
BLADE_ROOT_MY_FIELDS = (
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)
TRACE_SIGNATURE_ALGORITHM = 'sha256'
TRACE_SIGNATURE_CANONICALIZATION = 'int64_time_and_wind_ticks_little_endian_v1'
DEFAULT_TIME_SIGNATURE_QUANTIZATION_S = 1.0e-6
DEFAULT_WIND_SIGNATURE_QUANTIZATION_MPS = 1.0e-3


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _finite_float(value: Any, *, path: Path, row_number: int, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path.name} row {row_number} has invalid {field}') from exc
    if not math.isfinite(parsed):
        raise ValueError(f'{path.name} row {row_number} has non-finite {field}')
    return parsed


def _quantized_tick(value: float, quantization: float, field_name: str) -> int:
    if not math.isfinite(quantization) or quantization <= 0.0:
        raise ValueError(f'{field_name} quantization must be positive and finite')
    if not math.isfinite(value):
        raise ValueError(f'{field_name} values must be finite')
    # Python round and np.rint both use round-to-nearest, ties-to-even. Packing
    # one canonical int64 pair avoids repeatedly invoking NumPy reductions on
    # tiny arrays in full 600 s identity traces.
    tick = int(round(value / quantization))
    if tick < -(1 << 63) or tick >= (1 << 63):
        raise ValueError(f'{field_name} value cannot be represented as canonical int64 ticks')
    return tick


class CanonicalTraceHasher:
    def __init__(
        self,
        *,
        time_quantization_s: float = DEFAULT_TIME_SIGNATURE_QUANTIZATION_S,
        wind_quantization_mps: float = DEFAULT_WIND_SIGNATURE_QUANTIZATION_MPS,
    ) -> None:
        if not math.isfinite(time_quantization_s) or time_quantization_s <= 0.0:
            raise ValueError('time quantization must be positive and finite')
        if not math.isfinite(wind_quantization_mps) or wind_quantization_mps <= 0.0:
            raise ValueError('wind quantization must be positive and finite')
        self.time_quantization_s = float(time_quantization_s)
        self.wind_quantization_mps = float(wind_quantization_mps)
        self._digest = hashlib.sha256()
        self.sample_count = 0

    def update(self, timestamp_s: float, wind_speed_mps: float) -> None:
        time_tick = _quantized_tick(float(timestamp_s), self.time_quantization_s, 'time')
        wind_tick = _quantized_tick(float(wind_speed_mps), self.wind_quantization_mps, 'wind')
        self._digest.update(struct.pack('<qq', time_tick, wind_tick))
        self.sample_count += 1

    def update_many(self, timestamps_s: np.ndarray, wind_speeds_mps: np.ndarray) -> None:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        winds = np.asarray(wind_speeds_mps, dtype=np.float64)
        if timestamps.shape != winds.shape or timestamps.ndim != 1:
            raise ValueError('Canonical trace timestamps and wind speeds must be one-dimensional with equal shape')
        for timestamp, wind_speed in zip(timestamps, winds, strict=True):
            self.update(float(timestamp), float(wind_speed))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def canonical_trace_signature(
    timestamps_s: np.ndarray | list[float],
    wind_speeds_mps: np.ndarray | list[float],
    *,
    time_quantization_s: float = DEFAULT_TIME_SIGNATURE_QUANTIZATION_S,
    wind_quantization_mps: float = DEFAULT_WIND_SIGNATURE_QUANTIZATION_MPS,
) -> str:
    times = np.asarray(timestamps_s, dtype=np.float64)
    winds = np.asarray(wind_speeds_mps, dtype=np.float64)
    if times.shape != winds.shape or times.ndim != 1:
        raise ValueError('Canonical trace timestamps and wind speeds must be one-dimensional with equal shape')
    if len(times) < 1:
        raise ValueError('Canonical trace must contain at least one sample')
    if np.any(np.diff(times) <= 0.0):
        raise ValueError('Canonical trace timestamps must be strictly increasing')
    hasher = CanonicalTraceHasher(
        time_quantization_s=time_quantization_s,
        wind_quantization_mps=wind_quantization_mps,
    )
    hasher.update_many(times, winds)
    return hasher.hexdigest()


def _window_rows(
    path: Path,
    *,
    time_window_start_s: float,
    time_window_end_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_fields = {
        TIME_FIELD,
        WIND_SPEED_FIELD,
        ELECTRICAL_POWER_VARIABLE,
        *BLADE_ROOT_MY_FIELDS,
    }
    times: list[float] = []
    winds: list[float] = []
    powers: list[float] = []
    blade_root_my: list[list[float]] = [[], [], []]
    previous_time: float | None = None
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_fields - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f'{path.name} is missing paired PID reference fields: {missing}')
        for row_number, row in enumerate(reader, start=2):
            timestamp = _finite_float(row[TIME_FIELD], path=path, row_number=row_number, field=TIME_FIELD)
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError(f'{path.name} PID timestamps must be strictly increasing')
            previous_time = timestamp
            if timestamp < time_window_start_s - TIME_TOLERANCE_S:
                continue
            if timestamp > time_window_end_s + TIME_TOLERANCE_S:
                break
            times.append(timestamp)
            winds.append(_finite_float(
                row[WIND_SPEED_FIELD], path=path, row_number=row_number, field=WIND_SPEED_FIELD,
            ))
            powers.append(_finite_float(
                row[ELECTRICAL_POWER_VARIABLE],
                path=path,
                row_number=row_number,
                field=ELECTRICAL_POWER_VARIABLE,
            ))
            for blade_index, field in enumerate(BLADE_ROOT_MY_FIELDS):
                blade_root_my[blade_index].append(_finite_float(
                    row[field], path=path, row_number=row_number, field=field,
                ))
    if len(times) < 2:
        raise ValueError(f'{path.name} has fewer than two samples in the requested paired PID window')
    time_array = np.asarray(times, dtype=np.float64)
    if not math.isclose(
        float(time_array[0]), time_window_start_s, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S,
    ) or not math.isclose(
        float(time_array[-1]), time_window_end_s, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S,
    ):
        raise ValueError(
            f'{path.name} does not exactly cover [{time_window_start_s:g}, {time_window_end_s:g}] s'
        )
    return (
        time_array,
        np.asarray(winds, dtype=np.float64),
        np.asarray(powers, dtype=np.float64),
        np.asarray(blade_root_my, dtype=np.float64),
    )


def build_paired_pid_reference(
    data_root: Path | str,
    *,
    time_window_start_s: float,
    time_window_end_s: float,
    project_root: Path | str,
    time_signature_quantization_s: float = DEFAULT_TIME_SIGNATURE_QUANTIZATION_S,
    wind_signature_quantization_mps: float = DEFAULT_WIND_SIGNATURE_QUANTIZATION_MPS,
) -> dict[str, Any]:
    if not all(math.isfinite(value) for value in (time_window_start_s, time_window_end_s)):
        raise ValueError('Paired PID reference window bounds must be finite')
    if time_window_start_s < 0.0 or time_window_end_s <= time_window_start_s:
        raise ValueError('Paired PID reference window must satisfy 0 <= start < end')
    root = Path(project_root).expanduser().resolve()
    resolved_root, source_files = validate_raw_source_root(Path(data_root))
    references: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for condition_id, path in source_files:
        timestamps, winds, powers, blade_root_my = _window_rows(
            path,
            time_window_start_s=time_window_start_s,
            time_window_end_s=time_window_end_s,
        )
        source_sha256 = sha256_file(path)
        source_audit.append({
            'condition_id': condition_id,
            'filename': path.name,
            'sha256': source_sha256,
        })
        references.append({
            'scenario_id': f'wind{condition_id}',
            'condition_id': condition_id,
            'time_window_start_s': time_window_start_s,
            'time_window_end_s': time_window_end_s,
            'pid_raw_csv_path': _portable_path(path, root),
            'pid_raw_sha256': source_sha256,
            'exact_timestamps_s': timestamps.tolist(),
            'pid_electrical_power_w': powers.tolist(),
            'pid_blade_root_my_nm': blade_root_my.tolist(),
            'pid_blade_root_peak_abs_nm': np.max(np.abs(blade_root_my), axis=1).tolist(),
            'canonical_wind_trace': {
                'time_field': TIME_FIELD,
                'wind_field': WIND_SPEED_FIELD,
                'sample_count': int(len(timestamps)),
                'time_signature_quantization_s': time_signature_quantization_s,
                'wind_signature_quantization_mps': wind_signature_quantization_mps,
                'algorithm': TRACE_SIGNATURE_ALGORITHM,
                'canonicalization': TRACE_SIGNATURE_CANONICALIZATION,
                'sha256': canonical_trace_signature(
                    timestamps,
                    winds,
                    time_quantization_s=time_signature_quantization_s,
                    wind_quantization_mps=wind_signature_quantization_mps,
                ),
                'wind_speed_mps': winds.tolist(),
            },
        })
    return {
        'schema_version': PAIRED_PID_REFERENCE_SCHEMA_VERSION,
        'metadata': {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_role': 'paired_pid_power_load_reference',
            'source_data_root': str(resolved_root),
            'source_file_pattern': 'data_windspeed{condition_id}.csv',
            'expected_condition_ids': list(EXPECTED_CONDITION_IDS),
            'condition_count': len(references),
            'time_window_start_s': time_window_start_s,
            'time_window_end_s': time_window_end_s,
            'sampling_strategy': 'all_raw_rows_exact_timestamp_within_fixed_window',
            'input_manifest_sha256': manifest_sha256(source_audit),
            'input_manifest_format': 'condition_id:filename:file_sha256\\n sorted by condition_id',
            'pairing_contract': {
                'case_identity': 'same_bladed_case',
                'turbulence_seed_identity': 'same_seed',
                'time_window_identity': 'same_simulation_time_window',
                'time_alignment': 'exact_timestamp_no_interpolation',
                'power_and_load_source': 'same_pid_raw_csv_entry',
            },
        },
        'references': references,
    }


@dataclass(frozen=True)
class PairedPidReferenceEntry:
    scenario_id: str
    condition_id: int
    time_window_start_s: float
    time_window_end_s: float
    pid_raw_csv_path: Path
    pid_raw_sha256: str
    timestamps_s: np.ndarray
    electrical_power_w: np.ndarray
    blade_root_my_nm: np.ndarray
    blade_root_peak_abs_nm: np.ndarray
    wind_speeds_mps: np.ndarray
    trace_signature: str
    time_signature_quantization_s: float
    wind_signature_quantization_mps: float


class PairedPidReference:
    def __init__(self, document: Mapping[str, Any], *, project_root: Path | str) -> None:
        if not isinstance(document, Mapping):
            raise ValueError('Paired PID reference must contain an object')
        if document.get('schema_version') != PAIRED_PID_REFERENCE_SCHEMA_VERSION:
            raise ValueError('Paired PID reference schema version is incompatible')
        metadata = document.get('metadata')
        if not isinstance(metadata, Mapping):
            raise ValueError('Paired PID reference metadata is missing')
        if metadata.get('data_schema_version') != DATA_SCHEMA_VERSION:
            raise ValueError('Paired PID reference data schema is incompatible')
        if metadata.get('source_role') != 'paired_pid_power_load_reference':
            raise ValueError('Paired PID reference source role is incompatible')
        if metadata.get('pairing_contract') != {
            'case_identity': 'same_bladed_case',
            'turbulence_seed_identity': 'same_seed',
            'time_window_identity': 'same_simulation_time_window',
            'time_alignment': 'exact_timestamp_no_interpolation',
            'power_and_load_source': 'same_pid_raw_csv_entry',
        }:
            raise ValueError('Paired PID reference pairing contract is incompatible')
        self.project_root = Path(project_root).expanduser().resolve()
        self.metadata = dict(metadata)
        self.manifest_sha256 = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        ).hexdigest()
        references = document.get('references')
        if not isinstance(references, list) or not references:
            raise ValueError('Paired PID reference needs one or more entries')
        entries: dict[int, PairedPidReferenceEntry] = {}
        for raw_entry in references:
            entry = self._parse_entry(raw_entry)
            if entry.condition_id in entries:
                raise ValueError(f'Paired PID reference has duplicate condition {entry.condition_id}')
            entries[entry.condition_id] = entry
        self.entries = entries

    @classmethod
    def from_path(cls, path: Path | str, *, project_root: Path | str) -> PairedPidReference:
        resolved = Path(path).expanduser()
        try:
            document = json.loads(resolved.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Paired PID reference does not exist: {resolved}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read paired PID reference: {resolved}') from exc
        return cls(document, project_root=project_root)

    def _parse_entry(self, raw_entry: Any) -> PairedPidReferenceEntry:
        if not isinstance(raw_entry, Mapping):
            raise ValueError('Paired PID reference entry must be an object')
        try:
            scenario_id = str(raw_entry['scenario_id'])
            condition_id = int(raw_entry['condition_id'])
            time_window_start_s = float(raw_entry['time_window_start_s'])
            time_window_end_s = float(raw_entry['time_window_end_s'])
            raw_path = Path(str(raw_entry['pid_raw_csv_path'])).expanduser()
            raw_sha256 = str(raw_entry['pid_raw_sha256']).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Paired PID reference entry identity is invalid') from exc
        if not scenario_id or condition_id < 0 or not _is_sha256(raw_sha256):
            raise ValueError('Paired PID reference entry identity is invalid')
        if not all(math.isfinite(value) for value in (time_window_start_s, time_window_end_s)) or (
            time_window_start_s < 0.0 or time_window_end_s <= time_window_start_s
        ):
            raise ValueError('Paired PID reference entry time window is invalid')
        if not raw_path.is_absolute():
            raw_path = self.project_root / raw_path
        timestamps = _array(raw_entry.get('exact_timestamps_s'), 'exact_timestamps_s')
        powers = _array(raw_entry.get('pid_electrical_power_w'), 'pid_electrical_power_w')
        blade_root_my = np.asarray(raw_entry.get('pid_blade_root_my_nm'), dtype=np.float64)
        peaks = _array(raw_entry.get('pid_blade_root_peak_abs_nm'), 'pid_blade_root_peak_abs_nm')
        trace = raw_entry.get('canonical_wind_trace')
        if (
            blade_root_my.shape != (3, len(timestamps))
            or powers.shape != timestamps.shape
            or peaks.shape != (3,)
            or not np.isfinite(blade_root_my).all()
            or np.any(peaks <= 0.0)
        ):
            raise ValueError(f'Paired PID reference condition {condition_id} trace shape is invalid')
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(f'Paired PID reference condition {condition_id} timestamps are invalid')
        if not math.isclose(timestamps[0], time_window_start_s, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S) or not math.isclose(
            timestamps[-1], time_window_end_s, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S,
        ):
            raise ValueError(f'Paired PID reference condition {condition_id} window does not match timestamps')
        expected_peaks = np.max(np.abs(blade_root_my), axis=1)
        if not np.allclose(peaks, expected_peaks, rtol=0.0, atol=1.0e-9):
            raise ValueError(f'Paired PID reference condition {condition_id} blade-root peaks are invalid')
        if not isinstance(trace, Mapping):
            raise ValueError(f'Paired PID reference condition {condition_id} canonical wind trace is invalid')
        winds = _array(trace.get('wind_speed_mps'), 'canonical_wind_trace.wind_speed_mps')
        try:
            sample_count = int(trace['sample_count'])
            time_quantization = float(trace['time_signature_quantization_s'])
            wind_quantization = float(trace['wind_signature_quantization_mps'])
            signature = str(trace['sha256']).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Paired PID reference condition {condition_id} canonical wind trace is invalid') from exc
        if (
            winds.shape != timestamps.shape
            or sample_count != len(timestamps)
            or trace.get('time_field') != TIME_FIELD
            or trace.get('wind_field') != WIND_SPEED_FIELD
            or trace.get('algorithm') != TRACE_SIGNATURE_ALGORITHM
            or trace.get('canonicalization') != TRACE_SIGNATURE_CANONICALIZATION
            or not _is_sha256(signature)
        ):
            raise ValueError(f'Paired PID reference condition {condition_id} canonical wind trace is invalid')
        if signature != canonical_trace_signature(
            timestamps,
            winds,
            time_quantization_s=time_quantization,
            wind_quantization_mps=wind_quantization,
        ):
            raise ValueError(f'Paired PID reference condition {condition_id} canonical wind trace hash is invalid')
        if raw_path.is_file() and sha256_file(raw_path) != raw_sha256:
            raise ValueError(f'Paired PID reference condition {condition_id} PID raw SHA-256 does not match')
        return PairedPidReferenceEntry(
            scenario_id=scenario_id,
            condition_id=condition_id,
            time_window_start_s=time_window_start_s,
            time_window_end_s=time_window_end_s,
            pid_raw_csv_path=raw_path,
            pid_raw_sha256=raw_sha256,
            timestamps_s=timestamps,
            electrical_power_w=powers,
            blade_root_my_nm=blade_root_my,
            blade_root_peak_abs_nm=peaks,
            wind_speeds_mps=winds,
            trace_signature=signature,
            time_signature_quantization_s=time_quantization,
            wind_signature_quantization_mps=wind_quantization,
        )

    def entry(self, condition_id: int) -> PairedPidReferenceEntry:
        try:
            return self.entries[int(condition_id)]
        except KeyError as exc:
            raise ValueError(f'No paired PID reference for condition {int(condition_id)}') from exc

    def power_at(self, condition_id: int, sim_time_s: float) -> float:
        entry = self.entry(condition_id)
        time_value = float(sim_time_s)
        if not math.isfinite(time_value):
            raise ValueError('Paired PID reference timestamp must be finite')
        index = _exact_timestamp_index(entry.timestamps_s, time_value)
        if index is None:
            raise ValueError(
                f'Paired PID reference condition {entry.condition_id} has no exact timestamp '
                f'for {time_value:.9f} s'
            )
        return float(entry.electrical_power_w[index])

    def blade_root_peak(self, condition_id: int) -> np.ndarray:
        return self.entry(condition_id).blade_root_peak_abs_nm.copy()


def _array(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f'Paired PID reference {field_name} must be a finite one-dimensional array')
    return array


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def _exact_timestamp_index(timestamps: np.ndarray, time_value: float) -> int | None:
    insertion_index = int(np.searchsorted(timestamps, time_value, side='left'))
    candidate_indices = [
        index
        for index in (insertion_index - 1, insertion_index)
        if 0 <= index < len(timestamps)
    ]
    if not candidate_indices:
        return None
    nearest_index = min(candidate_indices, key=lambda index: abs(float(timestamps[index]) - time_value))
    if abs(float(timestamps[nearest_index]) - time_value) > TIME_TOLERANCE_S:
        return None
    return nearest_index