from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from RL.models.load_predictor_tcn import CausalTCNLoadPredictor
from RL.data.load_predictor_dataset import (
    HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION,
    _history_iqr_floors,
)
from RL.utils.load_predictor_contract import (
    LoadPredictorInterface,
    PLAN_KIND_ZOH_CANDIDATE,
    make_zoh_candidate_plan,
)
from RL.utils.paired_pid_reference import PairedPidReference
from RL.utils.load_predictor_shadow import FEATURE_DIM


PREDICTOR_CHECKPOINT_SCHEMA_VERSION = 'load_predictor_checkpoint_v4_candidate_conditioned'


@dataclass(frozen=True)
class LoadedLoadPredictor:
    model: CausalTCNLoadPredictor
    interface: LoadPredictorInterface
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    feature_dim: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_conditioned_predictor(
    checkpoint_path: Path | str,
    *,
    interface_path: Path | str,
    expected_checkpoint_sha256: str,
    action_low: Sequence[float],
    action_high: Sequence[float],
    device: torch.device,
    expected_dataset_manifest_sha256: str | None = None,
    expected_time_partitions_s: dict[str, tuple[float, float] | list[float]] | None = None,
) -> LoadedLoadPredictor:
    artifact_path = Path(checkpoint_path).expanduser().resolve()
    contract_path = Path(interface_path).expanduser().resolve()
    if not artifact_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError('Load predictor checkpoint/interface file is missing')
    observed_sha256 = sha256_file(artifact_path)
    if (
        len(expected_checkpoint_sha256) != 64
        or expected_checkpoint_sha256.lower() != observed_sha256
    ):
        raise ValueError('Load predictor checkpoint SHA-256 does not match the frozen config')
    interface = LoadPredictorInterface.from_path(contract_path)
    if interface.schema_version != 'causal_load_predictor_interface_v3':
        raise ValueError('Production load predictor artifact requires the candidate-conditioned v3 interface')
    interface_sha256 = sha256_file(contract_path)
    checkpoint = torch.load(artifact_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != PREDICTOR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('Load predictor checkpoint schema is incompatible with candidate-conditioned inference')
    if checkpoint.get('interface_schema_version') != interface.schema_version:
        raise ValueError('Load predictor checkpoint interface schema version does not match config')
    if checkpoint.get('interface_sha256') != interface_sha256:
        raise ValueError('Load predictor checkpoint interface SHA-256 does not match config')
    interface_identity = checkpoint.get('interface_identity')
    if not isinstance(interface_identity, dict) or not interface_identity:
        raise ValueError('Load predictor checkpoint is missing its static interface identity manifest')
    expected_interface_identity = {
        'schema_version': interface.schema_version,
        'history_frames': interface.history_frames,
        'step_duration_s': interface.step_duration_s,
        'horizons_s': list(interface.horizons_s),
        'soft_guard_horizons_s': list(interface.soft_guard_horizons_s),
        'online_candidate_plan_kinds': list(interface.online_candidate_plan_kinds),
        'candidate_plan_encoding': interface.candidate_plan_encoding,
        'candidate_plan_padding': interface.candidate_plan_padding,
        'horizon_conditioning': interface.horizon_conditioning,
        'plan_kind_conditioning': interface.plan_kind_conditioning,
        'model_input_contract': interface.model_input_contract,
        'response_label_semantics': interface.response_label_semantics,
        'peak_label_semantics': interface.peak_label_semantics,
    }
    if interface_identity != expected_interface_identity:
        raise ValueError('Load predictor checkpoint interface identity version is incompatible')
    dataset_sha256 = str(checkpoint.get('dataset_manifest_sha256', '')).lower()
    if len(dataset_sha256) != 64:
        raise ValueError('Load predictor checkpoint dataset manifest SHA-256 is invalid')
    if expected_dataset_manifest_sha256 and dataset_sha256 != expected_dataset_manifest_sha256.lower():
        raise ValueError('Load predictor checkpoint dataset manifest SHA-256 does not match config')
    raw_source_sha_by_condition = checkpoint.get('raw_source_sha256_by_condition')
    paired_identity_by_condition = checkpoint.get('paired_identity_by_condition')
    paired_reference_sha256 = str(checkpoint.get('paired_reference_sha256', '')).lower()
    if (
        not isinstance(raw_source_sha_by_condition, dict)
        or not isinstance(paired_identity_by_condition, dict)
        or len(paired_reference_sha256) != 64
        or not paired_identity_by_condition
        or set(raw_source_sha_by_condition) != set(paired_identity_by_condition)
        or any(
            len(str(value).lower()) != 64
            or any(character not in '0123456789abcdef' for character in str(value).lower())
            for value in raw_source_sha_by_condition.values()
        )
    ):
        raise ValueError('Load predictor checkpoint is missing frozen per-condition PID/reference provenance')
    paired_reference_path_value = checkpoint.get('paired_reference_path')
    if not isinstance(paired_reference_path_value, str) or not paired_reference_path_value.strip():
        raise ValueError('Load predictor checkpoint is missing its paired-reference path')
    paired_reference_path = Path(paired_reference_path_value).expanduser()
    if not paired_reference_path.is_absolute():
        paired_reference_path = Path(__file__).resolve().parents[2] / paired_reference_path
    if not paired_reference_path.is_file() or sha256_file(paired_reference_path) != paired_reference_sha256:
        raise ValueError('Load predictor checkpoint paired-reference SHA-256 is invalid')
    paired_reference = PairedPidReference.from_path(
        paired_reference_path,
        project_root=Path(__file__).resolve().parents[2],
    )
    if not {int(value) for value in paired_identity_by_condition} <= set(paired_reference.entries):
        raise ValueError('Load predictor checkpoint conditions are outside its paired PID reference')
    for condition_key, identity in paired_identity_by_condition.items():
        if not isinstance(identity, dict):
            raise ValueError('Load predictor checkpoint paired identity entry is invalid')
        raw_sha = str(raw_source_sha_by_condition.get(condition_key, '')).lower()
        if (
            len(raw_sha) != 64
            or str(identity.get('pid_raw_sha256', '')).lower() != raw_sha
            or str(identity.get('raw_source_sha256', '')).lower() != raw_sha
            or len(str(identity.get('expected_trace_signature', ''))) != 64
            or int(identity.get('expected_trace_sample_count', 0)) < 2
            or not str(identity.get('scenario_id', '')).strip()
            or int(identity.get('condition_id', -1)) != int(condition_key)
            or not math.isfinite(float(identity.get('time_window_start_s', math.nan)))
            or not math.isfinite(float(identity.get('time_window_end_s', math.nan)))
            or float(identity.get('time_window_start_s', -1.0)) < 0.0
            or float(identity.get('time_window_end_s', 0.0))
            <= float(identity.get('time_window_start_s', 0.0))
            or not str(identity.get('pid_raw_csv_path', '')).strip()
            or str(identity.get('expected_trace_signature', '')).lower()
            != paired_reference.entry(int(condition_key)).trace_signature
            or str(identity.get('scenario_id', ''))
            != paired_reference.entry(int(condition_key)).scenario_id
            or int(identity.get('expected_trace_sample_count', 0))
            != paired_reference.entry(int(condition_key)).timestamps_s.size
            or not math.isclose(
                float(identity.get('time_window_start_s', math.nan)),
                paired_reference.entry(int(condition_key)).time_window_start_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                float(identity.get('time_window_end_s', math.nan)),
                paired_reference.entry(int(condition_key)).time_window_end_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(f'Load predictor checkpoint PID identity is invalid for condition={condition_key}')
        raw_source_path = Path(str(identity.get('pid_raw_csv_path', ''))).expanduser()
        if not raw_source_path.is_absolute():
            raw_source_path = Path(__file__).resolve().parents[2] / raw_source_path
        if not raw_source_path.is_file() or sha256_file(raw_source_path) != raw_sha:
            raise ValueError(f'Load predictor checkpoint raw PID SHA-256 is invalid for condition={condition_key}')
    if expected_time_partitions_s is not None:
        saved_time_partitions = checkpoint.get('time_partitions_s')
        if not isinstance(saved_time_partitions, dict):
            raise ValueError('Load predictor checkpoint is missing frozen time partitions')
        normalized_expected = {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in expected_time_partitions_s.items()
        }
        if saved_time_partitions != normalized_expected:
            raise ValueError('Load predictor checkpoint time partitions do not match the frozen config')
    else:
        raise ValueError('Predictor artifact loading requires the frozen time partitions from its candidate config')
    try:
        feature_dim = int(checkpoint['feature_dim'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Load predictor checkpoint feature_dim is invalid') from exc
    if feature_dim != FEATURE_DIM:
        raise ValueError('Load predictor checkpoint feature_dim does not match the live shadow schema')
    model = CausalTCNLoadPredictor(
        input_dim=feature_dim,
        interface=interface,
        action_low=action_low,
        action_high=action_high,
    )
    architecture = checkpoint.get('model_architecture')
    expected_architecture = {
        'model_class': 'CausalTCNLoadPredictor',
        'input_dim': feature_dim,
        'hidden_dim': model.hidden_dim,
        'dilations': [1, 2, 4, 8],
        'plan_encoder': 'GRU(input_size=2, hidden_size=64, batch_first=True)',
        'horizon_embedding_dim': 8,
        'plan_kind_embedding_dim': 8,
        'output_width': model.OUTPUT_WIDTH,
    }
    if architecture != expected_architecture:
        raise ValueError('Load predictor checkpoint architecture metadata does not match the frozen v3 model')
    state_dict = checkpoint.get('model_state_dict')
    if not isinstance(state_dict, dict):
        raise ValueError('Load predictor checkpoint is missing model_state_dict')
    model.load_state_dict(state_dict, strict=True)
    expected_low = torch.as_tensor(action_low, dtype=torch.float32)
    expected_high = torch.as_tensor(action_high, dtype=torch.float32)
    if not torch.equal(model.action_low.cpu(), expected_low) or not torch.equal(model.action_high.cpu(), expected_high):
        raise ValueError('Load predictor checkpoint action normalization bounds are incompatible')
    normalization = checkpoint.get('normalization')
    if not isinstance(normalization, dict):
        raise ValueError('Load predictor checkpoint is missing normalization')
    _validate_calibration_metadata(checkpoint, normalization, interface)
    expected_history_iqr_floors = _history_iqr_floors(feature_dim).astype(float).tolist()
    if (
        normalization.get('history_iqr_floor_schema') != HISTORY_NORMALIZATION_FLOOR_SCHEMA_VERSION
        or normalization.get('history_iqr_floors') != expected_history_iqr_floors
        or not isinstance(normalization.get('history_iqr'), list)
        or len(normalization['history_iqr']) != feature_dim
        or any(
            float(actual) < float(minimum)
            for actual, minimum in zip(
                normalization['history_iqr'],
                expected_history_iqr_floors,
                strict=True,
            )
        )
    ):
        raise ValueError('Load predictor checkpoint history normalization floor contract is incompatible')
    model.set_normalization(normalization)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('Load predictor checkpoint contains non-finite model parameters')
    if any(not torch.isfinite(buffer).all() for buffer in model.buffers()):
        raise ValueError('Load predictor checkpoint contains non-finite model buffers')
    if checkpoint.get('guard_calibration_authorized') is not False:
        raise ValueError('PID predictor checkpoint cannot authorize RL load-guard calibration')
    if 'calibration_artifact_path' in checkpoint:
        raise ValueError('Predictor checkpoint cannot embed/override its independent calibration artifact')
    model.to(device).eval()
    _validate_frozen_predictor_splits(checkpoint, interface, paired_identity_by_condition)
    _candidate_plan_sensitivity_smoke(
        model,
        interface,
        action_low,
        action_high,
        device,
        None,
    )
    return LoadedLoadPredictor(
        model=model,
        interface=interface,
        checkpoint_path=artifact_path,
        checkpoint_sha256=observed_sha256,
        dataset_manifest_sha256=dataset_sha256,
        feature_dim=feature_dim,
    )


def _validate_frozen_predictor_splits(
    checkpoint: dict,
    interface: LoadPredictorInterface,
    paired_identity_by_condition: dict,
) -> None:
    partition_names = ('train', 'validation', 'calibration', 'coverage_audit')
    condition_partitions = checkpoint.get('condition_partitions')
    time_partitions = checkpoint.get('time_partitions_s')
    split_counts = checkpoint.get('split_sample_counts')
    plan_counts = checkpoint.get('plan_counts_by_split')
    if (
        not isinstance(condition_partitions, dict)
        or set(condition_partitions) != set(partition_names)
        or not isinstance(time_partitions, dict)
        or set(time_partitions) != set(partition_names)
        or not isinstance(split_counts, dict)
        or set(split_counts) != set(partition_names)
        or not isinstance(plan_counts, dict)
        or set(plan_counts) != set(partition_names)
    ):
        raise ValueError('Predictor checkpoint frozen condition/time split manifest is incomplete')
    normalized_conditions = {name: [int(value) for value in condition_partitions[name]] for name in partition_names}
    all_conditions = [condition for values in normalized_conditions.values() for condition in values]
    expected_conditions = {int(value) for value in paired_identity_by_condition}
    if (
        any(not values for values in normalized_conditions.values())
        or len(all_conditions) != len(set(all_conditions))
        or set(all_conditions) != expected_conditions
    ):
        raise ValueError('Predictor checkpoint condition partitions overlap or do not match the paired identity table')
    temporal_purge_s = (
        interface.history_frames - 1 + max(interface.horizon_frames)
    ) * interface.step_duration_s
    normalized_times: dict[str, tuple[float, float]] = {}
    for name in partition_names:
        bounds = time_partitions[name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f'Predictor checkpoint time partition {name} is malformed')
        start_s, end_s = float(bounds[0]), float(bounds[1])
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise ValueError(f'Predictor checkpoint time partition {name} is not finite/increasing')
        normalized_times[name] = (start_s, end_s)
        split_count = int(split_counts[name])
        per_split_plan_counts = plan_counts[name]
        if (
            split_count <= 0
            or not isinstance(per_split_plan_counts, dict)
            or set(per_split_plan_counts) != {'realized_plan', 'zoh_candidate'}
            or sum(int(value) for value in per_split_plan_counts.values()) != split_count
            or int(per_split_plan_counts['realized_plan']) != 0
            or int(per_split_plan_counts['zoh_candidate']) != split_count
        ):
            raise ValueError(f'Predictor checkpoint split {name} has invalid/non-ZOH sample counts')
    for previous, following in zip(partition_names, partition_names[1:]):
        if normalized_times[following][0] - normalized_times[previous][1] + 1.0e-12 < temporal_purge_s:
            raise ValueError('Predictor checkpoint time partitions violate history/future purge')
    if not math.isclose(
        float(checkpoint.get('temporal_purge_s', math.nan)),
        temporal_purge_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError('Predictor checkpoint temporal purge does not match the interface')
    train_start = min(float(identity['time_window_start_s']) for identity in paired_identity_by_condition.values())
    source_end = max(float(identity['time_window_end_s']) for identity in paired_identity_by_condition.values())
    if (
        not math.isclose(normalized_times['train'][0], train_start, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(
            normalized_times['coverage_audit'][1],
            source_end + interface.step_duration_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or not math.isclose(
            float(checkpoint.get('predictor_training_end_s', math.nan)),
            normalized_times['train'][1],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            float(checkpoint.get('calibration_start_s', math.nan)),
            normalized_times['calibration'][0],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            float(checkpoint.get('coverage_audit_start_s', math.nan)),
            normalized_times['coverage_audit'][0],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError('Predictor checkpoint split boundaries are inconsistent with paired PID windows')


def _candidate_plan_sensitivity_smoke(
    model: CausalTCNLoadPredictor,
    interface: LoadPredictorInterface,
    action_low: Sequence[float],
    action_high: Sequence[float],
    device: torch.device,
    actor_action_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if actor_action_bounds is None:
        lower = np.asarray(action_low, dtype=np.float64)
        upper = np.asarray(action_high, dtype=np.float64)
    else:
        lower, upper = actor_action_bounds
    candidates = (lower, (lower + upper) / 2.0, upper)
    if any(
        not np.isfinite(candidate).all()
        or np.any(candidate < np.asarray(action_low, dtype=np.float64))
        or np.any(candidate > np.asarray(action_high, dtype=np.float64))
        for candidate in candidates
    ):
        raise ValueError('Candidate-action sensitivity smoke has invalid action bounds')
    prediction_deltas: list[float] = []
    model = model.to(device).eval()
    with torch.inference_mode():
        history = torch.zeros((len(candidates), interface.history_frames, FEATURE_DIM), device=device)
        for horizon_s in interface.soft_guard_horizons_s:
            horizon_index = interface.horizons_s.index(horizon_s)
            horizon_frames = interface.horizon_frames[horizon_index]
            plans = torch.zeros(
                (len(candidates), max(interface.horizon_frames), 2),
                dtype=torch.float32,
                device=device,
            )
            for row_index, action in enumerate(candidates):
                plans[row_index, :horizon_frames] = torch.as_tensor(
                    make_zoh_candidate_plan(action, horizon_frames),
                    dtype=torch.float32,
                    device=device,
                )
            predictions = model.predict_candidate(
                history,
                plans,
                torch.full((len(candidates),), horizon_index, dtype=torch.long, device=device),
                torch.full((len(candidates),), 1, dtype=torch.long, device=device),
            )
            if predictions.shape != (len(candidates), 5) or not torch.isfinite(predictions).all():
                raise ValueError('Candidate-action sensitivity smoke emitted invalid predictor outputs')
            prediction_deltas.extend(
                float(torch.linalg.vector_norm(predictions[index] - predictions[0]).item())
                for index in range(1, len(candidates))
            )
    if not any(math.isfinite(delta) and delta > 1.0e-8 for delta in prediction_deltas):
        raise ValueError('Candidate-conditioned predictor is insensitive across its frozen action envelope')


def _validate_calibration_metadata(
    checkpoint: dict,
    normalization: dict,
    interface: LoadPredictorInterface,
) -> None:
    if checkpoint.get('split_schema_version') != 'load_predictor_condition_split_v2':
        raise ValueError('Predictor checkpoint condition/time split schema is incompatible')
    condition_partitions = checkpoint.get('condition_partitions')
    split_counts = checkpoint.get('split_sample_counts')
    plan_counts_by_split = checkpoint.get('plan_counts_by_split')
    expected_splits = {'train', 'validation', 'calibration', 'coverage_audit'}
    if (
        not isinstance(condition_partitions, dict)
        or set(condition_partitions) != expected_splits
        or not isinstance(split_counts, dict)
        or set(split_counts) != expected_splits
        or not isinstance(plan_counts_by_split, dict)
        or set(plan_counts_by_split) != expected_splits
    ):
        raise ValueError('Predictor checkpoint condition/time split metadata is incomplete')
    normalized_conditions = {
        name: [int(value) for value in condition_partitions[name]]
        for name in ('train', 'validation', 'calibration', 'coverage_audit')
    }
    flattened_conditions = [value for values in normalized_conditions.values() for value in values]
    if (
        any(not values for values in normalized_conditions.values())
        or len(flattened_conditions) != len(set(flattened_conditions))
        or set(flattened_conditions) != {int(value) for value in checkpoint['paired_identity_by_condition']}
    ):
        raise ValueError('Predictor checkpoint condition partitions overlap or do not cover PID identities')
    normalized_times: dict[str, tuple[float, float]] = {}
    for name, bounds in checkpoint.get('time_partitions_s', {}).items():
        if name not in expected_splits or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError('Predictor checkpoint time partition entry is invalid')
        start_s, end_s = float(bounds[0]), float(bounds[1])
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise ValueError(f'Predictor checkpoint time partition is invalid: {name}')
        normalized_times[name] = (start_s, end_s)
    if set(normalized_times) != expected_splits:
        raise ValueError('Predictor checkpoint time partitions are incomplete')
    purge_s = (interface.history_frames - 1 + max(interface.horizon_frames)) * interface.step_duration_s
    for previous, following in zip(
        ('train', 'validation', 'calibration'),
        ('validation', 'calibration', 'coverage_audit'),
    ):
        if normalized_times[following][0] - normalized_times[previous][1] + 1.0e-12 < purge_s:
            raise ValueError('Predictor checkpoint time splits do not include the causal history/label purge')
    if not math.isclose(float(checkpoint.get('temporal_purge_s', math.nan)), purge_s, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError('Predictor checkpoint temporal_purge_s does not match the frozen interface')
    for name in ('train', 'validation', 'calibration', 'coverage_audit'):
        count = int(split_counts[name])
        per_split = plan_counts_by_split[name]
        if (
            count <= 0
            or not isinstance(per_split, dict)
            or set(per_split) != {'realized_plan', 'zoh_candidate'}
            or int(per_split['realized_plan']) != 0
            or int(per_split['zoh_candidate']) != count
        ):
            raise ValueError(f'Predictor split is empty, mislabeled, or contains non-ZOH rows: {name}')
    if (
        not math.isclose(float(checkpoint.get('predictor_training_end_s', math.nan)), normalized_times['train'][1], rel_tol=0.0, abs_tol=1.0e-9)
        or not math.isclose(float(checkpoint.get('calibration_start_s', math.nan)), normalized_times['calibration'][0], rel_tol=0.0, abs_tol=1.0e-9)
        or not math.isclose(float(checkpoint.get('coverage_audit_start_s', math.nan)), normalized_times['coverage_audit'][0], rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise ValueError('Predictor checkpoint split boundary summary fields are inconsistent')
    if (
        normalization.get('history_window_capacity') is None
        or not 0 < int(normalization['history_window_capacity']) <= 8_192
        or not 0 < int(normalization.get('sample_count_used', 0)) <= int(normalization['history_window_capacity'])
        or int(normalization.get('history_window_bytes', -1))
        != int(normalization['history_window_capacity']) * interface.history_frames * FEATURE_DIM * 4
    ):
        raise ValueError('Predictor checkpoint normalization reservoir violates its frozen memory bound')
