from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

from RL.utils.paired_pid_reference import PairedPidReference
from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
)
from RL.utils.load_predictor_shadow import FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION = 'state20_feature_domain_scale_floor_v1'


def _history_iqr_floors(feature_dim: int) -> np.ndarray:
    if feature_dim != len(FEATURE_NAMES):
        raise ValueError(f'No frozen history normalization floors for feature_dim={feature_dim}')
    floors = {
        'I_RotorSpeed': 0.05,
        'I_GeneratorSpeed': 5.0,
        'I_RealPower': 1_000_000.0,
        'I_ElectricalTorque': 1_000_000.0,
        'I_PitchPosition1': 0.02,
        'I_NacelleForeAftAcceleration': 0.1,
        'I_NacelleSideSideAcceleration': 0.1,
        'I_AverageWindSpeed': 1.0,
        'sin_azimuth': 0.25,
        'cos_azimuth': 0.25,
        'Blade root 1 My (Nm)': 1_000_000.0,
        'Blade root 2 My (Nm)': 1_000_000.0,
        'Blade root 3 My (Nm)': 1_000_000.0,
        'Blade root 1 My rate (Nm/s)': 1_000_000.0,
        'Blade root 2 My rate (Nm/s)': 1_000_000.0,
        'Blade root 3 My rate (Nm/s)': 1_000_000.0,
        'sent_action_torque': 1_000_000.0,
        'sent_action_pitch': 0.02,
        'projected_candidate_action_torque': 1_000_000.0,
        'projected_candidate_action_pitch': 0.02,
        'pid_peak_headroom_blade_1': 0.1,
        'pid_peak_headroom_blade_2': 0.1,
        'pid_peak_headroom_blade_3': 0.1,
    }
    floors_array = np.asarray([
        1.0 if name.startswith('state_encoding_') else floors[name]
        for name in FEATURE_NAMES
    ], dtype=np.float32)
    if floors_array.shape != (feature_dim,) or not np.isfinite(floors_array).all() or np.any(floors_array <= 0.0):
        raise RuntimeError('Frozen predictor history IQR floor contract is invalid')
    return floors_array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadPredictorShard:
    path: Path
    sample_count: int
    start_index: int
    condition_ids: tuple[int, ...]


class LoadPredictorDataset(Dataset):
    """Lazy shard-backed dataset; it never combines future labels into history inputs."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        interface: LoadPredictorInterface,
        condition_ids: Iterable[int] | None = None,
        plan_kind: str | None = None,
        calibration_eligible_only: bool = False,
        time_range_s: tuple[float, float] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        manifest_path = self.root_dir / 'manifest.json'
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Load predictor dataset manifest does not exist: {manifest_path}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read load predictor dataset manifest: {manifest_path}') from exc
        if self.manifest.get('schema_version') not in ('load_predictor_dataset_v1', 'load_predictor_dataset_v2'):
            raise ValueError('Load predictor dataset schema is incompatible')
        if (
            self.manifest.get('schema_version') == 'load_predictor_dataset_v2'
            and self.manifest.get('predictor_interface_schema_version') != interface.schema_version
        ):
            raise ValueError('Load predictor dataset v2 interface version does not match the requested interface')
        if self.manifest.get('schema_version') == 'load_predictor_dataset_v2':
            interface_path_value = self.manifest.get('predictor_interface_path')
            expected_interface_sha256 = str(self.manifest.get('predictor_interface_sha256', '')).lower()
            interface_artifact_path = Path(str(interface_path_value)).expanduser()
            if not interface_artifact_path.is_absolute():
                interface_artifact_path = PROJECT_ROOT / interface_artifact_path
            if (
                not interface_artifact_path.is_file()
                or len(expected_interface_sha256) != 64
                or _sha256_file(interface_artifact_path) != expected_interface_sha256
            ):
                raise ValueError('Load predictor dataset interface SHA-256 provenance is invalid')
        if int(self.manifest.get('history_frames', -1)) != interface.history_frames:
            raise ValueError('Load predictor dataset history_frames does not match interface')
        if tuple(float(value) for value in self.manifest.get('horizons_s', ())) != interface.horizons_s:
            raise ValueError('Load predictor dataset horizons do not match interface')
        if int(self.manifest.get('feature_dim', -1)) <= 0:
            raise ValueError('Load predictor dataset feature_dim is invalid')
        if self.manifest.get('schema_version') == 'load_predictor_dataset_v2':
            identity_by_condition = self.manifest.get('paired_identity_by_condition')
            raw_sha_by_condition = self.manifest.get('raw_source_sha256_by_condition')
            source_conditions = {int(value) for value in self.manifest.get('source_conditions', ())}
            if (
                not isinstance(identity_by_condition, dict)
                or not isinstance(raw_sha_by_condition, dict)
                or {int(key) for key in identity_by_condition} != source_conditions
                or {int(key) for key in raw_sha_by_condition} != source_conditions
            ):
                raise ValueError('Load predictor dataset v2 condition identity tables are incomplete')
            for key, identity in identity_by_condition.items():
                if not isinstance(identity, dict) or int(identity.get('condition_id', -1)) != int(key):
                    raise ValueError(f'Load predictor dataset v2 paired identity entry is invalid: condition={key}')
                pid_sha = str(identity.get('pid_raw_sha256', '')).lower()
                raw_sha = str(identity.get('raw_source_sha256', '')).lower()
                expected_trace_signature = str(identity.get('expected_trace_signature', '')).lower()
                window_start = float(identity.get('time_window_start_s', math.nan))
                window_end = float(identity.get('time_window_end_s', math.nan))
                sample_count = int(identity.get('expected_trace_sample_count', 0))
                time_quantization = float(identity.get('time_signature_quantization_s', math.nan))
                wind_quantization = float(identity.get('wind_signature_quantization_mps', math.nan))
                raw_sha_table_value = str(raw_sha_by_condition.get(str(key), '')).lower()
                if (
                    len(pid_sha) != 64
                    or len(raw_sha) != 64
                    or len(expected_trace_signature) != 64
                    or pid_sha != raw_sha
                    or raw_sha != raw_sha_table_value
                    or not str(identity.get('scenario_id', '')).strip()
                    or not str(identity.get('pid_raw_csv_path', '')).strip()
                    or not math.isfinite(window_start)
                    or not math.isfinite(window_end)
                    or window_start < 0.0
                    or window_end <= window_start
                    or sample_count < 2
                    or not math.isfinite(time_quantization)
                    or time_quantization <= 0.0
                    or not math.isfinite(wind_quantization)
                    or wind_quantization <= 0.0
                ):
                    raise ValueError(f'Load predictor dataset v2 paired identity is invalid: condition={key}')
                raw_source_path = Path(str(identity['pid_raw_csv_path'])).expanduser()
                if not raw_source_path.is_absolute():
                    raw_source_path = PROJECT_ROOT / raw_source_path
                if (
                    not raw_source_path.is_file()
                    or _sha256_file(raw_source_path) != raw_sha
                ):
                    raise ValueError(
                        f'Load predictor dataset v2 source raw SHA does not match: '
                        f'condition={key} path={raw_source_path}'
                    )
            paired_reference_path = Path(str(self.manifest.get('paired_reference_path', ''))).expanduser()
            if not paired_reference_path.is_absolute():
                paired_reference_path = PROJECT_ROOT / paired_reference_path
            paired_reference_sha256 = str(self.manifest.get('paired_reference_sha256', '')).lower()
            if (
                not paired_reference_path.is_file()
                or len(paired_reference_sha256) != 64
                or _sha256_file(paired_reference_path) != paired_reference_sha256
            ):
                raise ValueError('Load predictor dataset v2 paired-reference SHA-256 is invalid')
            paired_reference = PairedPidReference.from_path(
                paired_reference_path,
                project_root=PROJECT_ROOT,
            )
            if not source_conditions <= set(paired_reference.entries):
                raise ValueError('Load predictor dataset v2 conditions are outside the paired PID reference')
            for key, identity in identity_by_condition.items():
                reference_entry = paired_reference.entry(int(key))
                if (
                    identity.get('scenario_id') != reference_entry.scenario_id
                    or str(identity.get('pid_raw_sha256', '')).lower() != reference_entry.pid_raw_sha256
                    or str(identity.get('expected_trace_signature', '')).lower() != reference_entry.trace_signature
                    or int(identity.get('expected_trace_sample_count', 0)) != reference_entry.timestamps_s.size
                    or not math.isclose(
                        float(identity['time_window_start_s']),
                        reference_entry.time_window_start_s,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    or not math.isclose(
                        float(identity['time_window_end_s']),
                        reference_entry.time_window_end_s,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                ):
                    raise ValueError(f'Load predictor dataset paired identity does not match reference: condition={key}')
            for raw_shard in self.manifest.get('shards', []):
                shard_path = self.root_dir / str(raw_shard['path'])
                expected_shard_sha256 = str(raw_shard.get('sha256', '')).lower()
                if len(expected_shard_sha256) != 64 or _sha256_file(shard_path) != expected_shard_sha256:
                    raise ValueError(f'Load predictor dataset v2 shard SHA-256 is invalid: {shard_path}')
                with np.load(shard_path, allow_pickle=False) as values:
                    required_identity_fields = (
                        'pid_raw_sha256',
                        'pid_raw_csv_path',
                        'raw_source_sha256',
                    )
                    missing_identity_fields = [name for name in required_identity_fields if name not in values]
                    if missing_identity_fields:
                        raise ValueError(
                            f'Load predictor dataset v2 shard is missing identity fields: '
                            f'{shard_path} {missing_identity_fields}'
                        )
                    shard_condition_ids = np.asarray(values['condition_id'], dtype=np.int64)
                    shard_pid_sha = values['pid_raw_sha256'].astype(str)
                    shard_source_sha = values['raw_source_sha256'].astype(str)
                    shard_pid_path = values['pid_raw_csv_path'].astype(str)
                    if any(
                        len(array) != int(raw_shard['samples'])
                        for array in (shard_condition_ids, shard_pid_sha, shard_source_sha, shard_pid_path)
                    ):
                        raise ValueError(f'Load predictor dataset v2 identity array length mismatch: {shard_path}')
                    for condition_id in np.unique(shard_condition_ids):
                        key = str(int(condition_id))
                        identity = identity_by_condition.get(key)
                        if not isinstance(identity, dict):
                            raise ValueError(f'Load predictor dataset v2 has no identity for condition={key}')
                        mask = shard_condition_ids == condition_id
                        expected_pid_sha = str(identity.get('pid_raw_sha256', '')).lower()
                        expected_raw_sha = str(identity.get('raw_source_sha256', '')).lower()
                        expected_pid_path = str(identity.get('pid_raw_csv_path', ''))
                        if (
                            len(expected_pid_sha) != 64
                            or expected_pid_sha != expected_raw_sha
                            or np.any(np.char.lower(shard_pid_sha[mask]) != expected_pid_sha)
                            or np.any(np.char.lower(shard_source_sha[mask]) != expected_raw_sha)
                            or np.any(shard_pid_path[mask] != expected_pid_path)
                        ):
                            raise ValueError(
                                f'Load predictor dataset v2 shard condition identity does not match manifest: '
                                f'{shard_path} condition={key}'
                            )
                    for name, values in (
                        ('pid_raw_sha256', shard_pid_sha),
                        ('raw_source_sha256', shard_source_sha),
                    ):
                        if any(
                            len(value) != 64 or any(character not in '0123456789abcdef' for character in value.lower())
                            for value in values
                        ):
                            raise ValueError(f'Load predictor dataset v2 contains invalid {name}: {shard_path}')
        if self.manifest.get('schema_version') == 'load_predictor_dataset_v2':
            if self.manifest.get('predictor_interface_schema_version') != 'causal_load_predictor_interface_v3':
                raise ValueError('Production predictor training requires the candidate-conditioned interface v3 dataset')
        self.interface = interface
        self.feature_dim = int(self.manifest['feature_dim'])
        requested_conditions = None if condition_ids is None else {int(value) for value in condition_ids}
        if requested_conditions is not None:
            unknown_conditions = requested_conditions - {
                int(value) for value in self.manifest.get('source_conditions', ())
            }
            if unknown_conditions:
                raise ValueError(f'Unknown load predictor dataset conditions requested: {sorted(unknown_conditions)}')
        if plan_kind is not None:
            interface.validate_plan_kind(plan_kind, online=False)
        self.plan_kind = plan_kind
        self.calibration_eligible_only = bool(calibration_eligible_only)
        if time_range_s is not None:
            start_s, end_s = (float(value) for value in time_range_s)
            if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
                raise ValueError('time_range_s must contain finite increasing bounds')
            self.time_range_s: tuple[float, float] | None = (start_s, end_s)
        else:
            self.time_range_s = None
        self.shards: list[LoadPredictorShard] = []
        self._indices: list[tuple[int, int]] = []
        start_index = 0
        manifest_shard_total = 0
        for shard_index, raw_shard in enumerate(self.manifest.get('shards', [])):
            shard_path = self.root_dir / str(raw_shard['path'])
            sample_count = int(raw_shard['samples'])
            if not shard_path.is_file() or sample_count <= 0:
                raise ValueError(f'Invalid load predictor shard: {shard_path}')
            # Shard condition IDs are read once to allow condition-based splits.
            with np.load(shard_path, allow_pickle=False) as values:
                shard_conditions = tuple(sorted({int(value) for value in values['condition_id'].tolist()}))
                plan_values = values['plan_kind'].tolist()
                eligibility = values['zoh_calibration_eligible'].astype(bool)
                if len(plan_values) != sample_count or len(eligibility) != sample_count:
                    raise ValueError(f'Load predictor shard sample count mismatch: {shard_path}')
                manifest_shard_total += sample_count
                selected = np.ones(sample_count, dtype=bool)
                if 'timestamp_s' not in values or 'model_input_plan' not in values:
                    raise ValueError(f'Load predictor shard is missing timestamps or candidate plans: {shard_path}')
                if requested_conditions is not None:
                    selected &= np.isin(values['condition_id'], list(requested_conditions))
                if plan_kind is not None:
                    selected &= np.asarray([value == plan_kind for value in plan_values], dtype=bool)
                if self.calibration_eligible_only:
                    selected &= eligibility
                if self.time_range_s is not None:
                    start_s, end_s = self.time_range_s
                    selected &= (values['timestamp_s'] >= start_s) & (values['timestamp_s'] < end_s)
                selected_indices = np.flatnonzero(selected).astype(np.int64)
            self.shards.append(LoadPredictorShard(
                path=shard_path,
                sample_count=sample_count,
                start_index=start_index,
                condition_ids=shard_conditions,
            ))
            self._indices.extend((shard_index, int(index)) for index in selected_indices)
            start_index += sample_count
        if 'plan_counts' in self.manifest and manifest_shard_total != int(
            self.manifest.get('plan_counts', {}).get('realized_plan', 0)
        ) + int(self.manifest.get('plan_counts', {}).get('zoh_candidate', 0)):
            raise ValueError('Load predictor dataset shard sample totals do not close to manifest plan counts')
        if not self.shards:
            raise ValueError('Load predictor dataset contains no shards')
        self._cached_shard_index: int | None = None
        self._cached_shard: dict[str, np.ndarray] | None = None

    @property
    def dataset_manifest_sha256(self) -> str:
        return hashlib.sha256((self.root_dir / 'manifest.json').read_bytes()).hexdigest()

    @property
    def plan_counts(self) -> dict[str, int]:
        counts = {PLAN_KIND_REALIZED: 0, PLAN_KIND_ZOH_CANDIDATE: 0}
        for shard_index, sample_index in self._indices:
            values = self._load_shard(shard_index)
            counts[str(values['plan_kind'][sample_index])] += 1
        return counts

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cached_shard_index == shard_index and self._cached_shard is not None:
            return self._cached_shard
        with np.load(self.shards[shard_index].path, allow_pickle=False) as values:
            loaded = {name: values[name] for name in values.files}
        self._cached_shard_index = shard_index
        self._cached_shard = loaded
        return loaded

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, sample_index = self._indices[index]
        values = self._load_shard(shard_index)
        horizon_s = float(values['horizon_s'][sample_index])
        horizon_index = min(
            range(len(self.interface.horizons_s)),
            key=lambda candidate: abs(self.interface.horizons_s[candidate] - horizon_s),
        )
        if abs(self.interface.horizons_s[horizon_index] - horizon_s) > 1.0e-6:
            raise ValueError(f'Unknown load predictor horizon={horizon_s}')
        plan_kind = str(values['plan_kind'][sample_index])
        horizon_frames = int(values['horizon_frames'][sample_index])
        if horizon_frames != self.interface.horizon_frames[horizon_index]:
            raise ValueError('Load predictor shard horizon_frames do not match the interface')
        plan_kind_index = {
            PLAN_KIND_REALIZED: 0,
            PLAN_KIND_ZOH_CANDIDATE: 1,
        }.get(plan_kind)
        if plan_kind_index is None:
            raise ValueError(f'Unknown load predictor plan_kind={plan_kind!r}')
        return {
            'history': np.asarray(values['history'][sample_index], dtype=np.float32),
            'candidate_plan': np.asarray(values['model_input_plan'][sample_index], dtype=np.float32),
            'horizon_index': np.int64(horizon_index),
            'horizon_s': np.float32(horizon_s),
            'horizon_frames': np.int64(horizon_frames),
            'plan_kind_index': np.int64(plan_kind_index),
            'future_actual_torque_pitch': np.asarray(
                values['future_actual_torque_pitch'][sample_index],
                dtype=np.float32,
            ),
            'future_peak_ratio': np.asarray(values['future_peak_ratio'][sample_index], dtype=np.float32),
            'condition_id': np.int64(values['condition_id'][sample_index]),
            'timestamp_s': np.float64(values['timestamp_s'][sample_index]),
            'optimization_state': str(values['optimization_state'][sample_index]),
            'plan_kind': plan_kind,
            'plan_mismatch': np.float32(values['plan_mismatch'][sample_index]),
            'zoh_calibration_eligible': bool(values['zoh_calibration_eligible'][sample_index]),
        }


def estimate_normalization(
    dataset: LoadPredictorDataset,
    *,
    max_samples: int = 8_192,
    seed: int = 42,
) -> dict[str, list[float]]:
    if max_samples <= 0:
        raise ValueError('max_samples must be positive')
    if len(dataset) == 0:
        raise ValueError('Cannot estimate load predictor normalization from an empty dataset split')
    sample_capacity = min(len(dataset), max_samples)
    history_samples = np.empty(
        (sample_capacity, dataset.interface.history_frames, dataset.feature_dim),
        dtype=np.float32,
    )
    response_samples = np.empty((sample_capacity, 2), dtype=np.float32)
    ratio_samples = np.empty((sample_capacity, 3), dtype=np.float32)
    random = np.random.default_rng(int(seed))
    sample_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if sample_count < sample_capacity:
            target_index = sample_count
            sample_count += 1
        else:
            target_index = int(random.integers(0, index + 1))
            if target_index >= sample_capacity:
                continue
        history_samples[target_index] = item['history']
        response_samples[target_index] = item['future_actual_torque_pitch']
        ratio_samples[target_index] = item['future_peak_ratio']
    histories = history_samples[:sample_count].reshape(-1, dataset.feature_dim)
    responses = response_samples[:sample_count]
    ratios = ratio_samples[:sample_count]

    def robust(values: np.ndarray) -> tuple[list[float], list[float]]:
        median = np.median(values, axis=0)
        iqr = np.subtract(*np.percentile(values, [75, 25], axis=0))
        return median.astype(float).tolist(), np.maximum(iqr, 1.0e-6).astype(float).tolist()

    history_median, history_iqr = robust(histories)
    history_iqr_array = np.maximum(
        np.asarray(history_iqr, dtype=np.float32),
        _history_iqr_floors(dataset.feature_dim),
    )
    response_median, response_iqr = robust(responses)
    ratio_median, ratio_iqr = robust(ratios)
    return {
        'history_median': history_median,
        'history_iqr': history_iqr_array.astype(float).tolist(),
        'history_iqr_floor_schema': HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION,
        'history_iqr_floors': _history_iqr_floors(dataset.feature_dim).astype(float).tolist(),
        'response_median': response_median,
        'response_iqr': response_iqr,
        'peak_ratio_median': ratio_median,
        'peak_ratio_iqr': ratio_iqr,
        'sample_count_used': int(sample_count),
        'history_window_capacity': int(max_samples),
        'history_window_bytes': int(history_samples.nbytes),
        'reservoir_seed': int(seed),
    }
