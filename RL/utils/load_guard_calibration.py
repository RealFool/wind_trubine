from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np

from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
)


LOAD_GUARD_CALIBRATION_SCHEMA_VERSION = 'load_guard_calibration_v1'


@dataclass(frozen=True)
class LoadGuardCalibrationConfig:
    coverage_target: float
    confidence_level: float
    minimum_samples_per_group: int
    max_plan_mismatch: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LoadGuardCalibrationConfig:
        required = (
            'coverage_target',
            'confidence_level',
            'minimum_samples_per_group',
            'max_plan_mismatch',
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f'load guard calibration is missing required fields: {missing}')
        try:
            coverage_target = float(value['coverage_target'])
            confidence_level = float(value['confidence_level'])
            minimum_samples = int(value['minimum_samples_per_group'])
            max_plan_mismatch = float(value['max_plan_mismatch'])
        except (TypeError, ValueError) as exc:
            raise ValueError('load guard calibration numeric fields are invalid') from exc
        if not 0.0 < coverage_target < 1.0:
            raise ValueError('coverage_target must be in (0, 1)')
        if not 0.0 < confidence_level < 1.0:
            raise ValueError('confidence_level must be in (0, 1)')
        if minimum_samples <= 0:
            raise ValueError('minimum_samples_per_group must be positive')
        if not math.isfinite(max_plan_mismatch) or max_plan_mismatch < 0.0:
            raise ValueError('max_plan_mismatch must be finite and non-negative')
        return cls(
            coverage_target=coverage_target,
            confidence_level=confidence_level,
            minimum_samples_per_group=minimum_samples,
            max_plan_mismatch=max_plan_mismatch,
        )


@dataclass(frozen=True)
class LoadGuardCalibrationObservation:
    blade: int
    optimization_state: str
    horizon_s: float
    plan_kind: str
    predicted_ratio: float
    actual_ratio: float
    plan_mismatch: float
    timestamp_s: float
    condition_id: int

    def __post_init__(self) -> None:
        if self.blade not in {1, 2, 3}:
            raise ValueError('calibration blade must be 1, 2, or 3')
        if self.plan_kind not in {PLAN_KIND_REALIZED, PLAN_KIND_ZOH_CANDIDATE}:
            raise ValueError(f'unsupported calibration plan_kind={self.plan_kind!r}')
        values = (
            self.horizon_s,
            self.predicted_ratio,
            self.actual_ratio,
            self.plan_mismatch,
            self.timestamp_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('calibration observation values must be finite')
        if self.horizon_s <= 0.0 or self.plan_mismatch < 0.0:
            raise ValueError('calibration horizon and mismatch values are invalid')


def _group_id(blade: int, optimization_state: str, horizon_s: float, plan_kind: str) -> str:
    return f'blade_{blade}|{optimization_state}|{horizon_s:.6f}|{plan_kind}'


def _wilson_lower_bound(successes: int, total: int, confidence_level: float) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z = NormalDist().inv_cdf(confidence_level)
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


def _conformal_quantile(values: np.ndarray, coverage_target: float) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError('conformal residuals must be a non-empty vector')
    rank = int(math.ceil((ordered.size + 1) * coverage_target))
    rank = min(max(rank, 1), ordered.size)
    return float(ordered[rank - 1]), rank / ordered.size


def calibrate_load_guard(
    observations: Iterable[LoadGuardCalibrationObservation],
    *,
    config: LoadGuardCalibrationConfig,
    predictor_training_end_s: float,
    calibration_start_s: float,
    interface: LoadPredictorInterface,
    predictor_artifact_sha256: str | None = None,
    predictor_interface_sha256: str | None = None,
) -> dict[str, Any]:
    if not math.isfinite(predictor_training_end_s) or not math.isfinite(calibration_start_s):
        raise ValueError('predictor training/calibration timestamps must be finite')
    if calibration_start_s <= predictor_training_end_s:
        raise ValueError('calibration_start_s must be strictly later than predictor_training_end_s')
    groups: dict[tuple[int, str, float, str], list[LoadGuardCalibrationObservation]] = {}
    observation_count = 0
    condition_ids: set[int] = set()
    timestamps: list[float] = []
    for observation in observations:
        if observation.timestamp_s < calibration_start_s:
            raise ValueError('calibration observations must be later than calibration_start_s')
        interface.validate_plan_kind(observation.plan_kind, online=False)
        key = (
            observation.blade,
            observation.optimization_state,
            observation.horizon_s,
            observation.plan_kind,
        )
        groups.setdefault(key, []).append(observation)
        observation_count += 1
        condition_ids.add(int(observation.condition_id))
        timestamps.append(float(observation.timestamp_s))
    group_documents: list[dict[str, Any]] = []
    for (blade, optimization_state, horizon_s, plan_kind), values in sorted(groups.items()):
        residuals = np.maximum(
            np.asarray([value.actual_ratio - value.predicted_ratio for value in values], dtype=np.float64),
            0.0,
        )
        mismatch_values = np.asarray([value.plan_mismatch for value in values], dtype=np.float64)
        eligible_mask = mismatch_values <= config.max_plan_mismatch
        eligible_residuals = residuals[eligible_mask]
        sample_count = int(len(values))
        eligible_count = int(eligible_residuals.size)
        if eligible_count > 0:
            quantile, quantile_rank = _conformal_quantile(eligible_residuals, config.coverage_target)
            actual_values = np.asarray([value.actual_ratio for value in values], dtype=np.float64)[eligible_mask]
            predicted_values = np.asarray([value.predicted_ratio for value in values], dtype=np.float64)[eligible_mask]
            calibration_covered = int(np.sum(actual_values <= predicted_values + quantile))
            calibration_coverage = calibration_covered / eligible_count
        else:
            quantile = float('nan')
            quantile_rank = float('nan')
            calibration_covered = 0
            calibration_coverage = 0.0
        group_documents.append({
            'group_id': _group_id(blade, optimization_state, horizon_s, plan_kind),
            'blade': blade,
            'optimization_state': optimization_state,
            'horizon_s': horizon_s,
            'plan_kind': plan_kind,
            'sample_count': sample_count,
            'eligible_sample_count': eligible_count,
            'max_observed_plan_mismatch': float(np.max(mismatch_values)) if sample_count else 0.0,
            'conformal_quantile': quantile,
            'quantile_rank': quantile_rank,
            'calibration_in_sample_coverage': calibration_coverage,
            'calibration_coverage_lower_confidence_bound': None,
            'calibration_covered_count': calibration_covered,
            'coverage_audit_sample_count': 0,
            'coverage_audit_coverage': None,
            'coverage_audit_lower_confidence_bound': None,
            'soft_guard_eligible': False,
        })
    return {
        'schema_version': LOAD_GUARD_CALIBRATION_SCHEMA_VERSION,
        'metadata': {
            'coverage_target': config.coverage_target,
            'confidence_level': config.confidence_level,
            'minimum_samples_per_group': config.minimum_samples_per_group,
            'max_plan_mismatch': config.max_plan_mismatch,
            'predictor_training_end_s': predictor_training_end_s,
            'calibration_start_s': calibration_start_s,
            'calibration_end_s': max(timestamps) if timestamps else calibration_start_s,
            'calibration_condition_ids': sorted(condition_ids),
            'observation_count': observation_count,
            'coverage_audit_observation_count': 0,
            'predictor_interface_schema_version': interface.schema_version,
            'predictor_interface_sha256': predictor_interface_sha256,
            'predictor_artifact_sha256': predictor_artifact_sha256,
            'calibration_method': 'split_conformal_one_sided_residual_quantile_wilson_lower_bound',
            'coverage_audit_method': 'independent_condition_and_time_split_wilson_lower_bound_required',
            'online_plan_kind': PLAN_KIND_ZOH_CANDIDATE,
            'soft_guard_authorized': False,
        },
        'groups': group_documents,
    }


class LoadGuardCalibration:
    def __init__(self, document: Mapping[str, Any]) -> None:
        if document.get('schema_version') != LOAD_GUARD_CALIBRATION_SCHEMA_VERSION:
            raise ValueError('Load guard calibration schema version is incompatible')
        metadata = document.get('metadata')
        groups = document.get('groups')
        if not isinstance(metadata, Mapping) or not isinstance(groups, list):
            raise ValueError('Load guard calibration metadata/groups are invalid')
        self.metadata = dict(metadata)
        self.groups = {
            str(group['group_id']): dict(group)
            for group in groups
            if isinstance(group, Mapping) and 'group_id' in group
        }
        if len(self.groups) != len(groups):
            raise ValueError('Load guard calibration group IDs must be unique')
        self.config = LoadGuardCalibrationConfig.from_mapping(metadata)

    def group(self, blade: int, optimization_state: str, horizon_s: float, plan_kind: str) -> dict[str, Any]:
        key = _group_id(blade, optimization_state, horizon_s, plan_kind)
        try:
            return self.groups[key]
        except KeyError as exc:
            raise ValueError(f'Load guard calibration group is missing: {key}') from exc

    def validate_soft_groups(self, required_groups: Iterable[tuple[int, str, float]]) -> None:
        if not self.metadata.get('soft_guard_authorized', False):
            raise ValueError('Load guard calibration has no independent coverage audit authorization')
        for blade, optimization_state, horizon_s in required_groups:
            group = self.group(blade, optimization_state, horizon_s, PLAN_KIND_ZOH_CANDIDATE)
            if not group.get('soft_guard_eligible', False):
                raise ValueError(f'Load guard calibration group is not soft-eligible: {group["group_id"]}')


def audit_load_guard_coverage(
    calibration_document: Mapping[str, Any],
    observations: Iterable[LoadGuardCalibrationObservation],
    *,
    audit_start_s: float,
    interface: LoadPredictorInterface,
    predictor_artifact_sha256: str | None,
    predictor_interface_sha256: str | None = None,
    coverage_population: str = 'pid_holdout',
    strict_identity_verified: bool = False,
) -> dict[str, Any]:
    """Audit frozen conformal margins on disjoint conditions and later timestamps."""
    calibration = LoadGuardCalibration(calibration_document)
    metadata = calibration.metadata
    try:
        calibration_end_s = float(metadata['calibration_end_s'])
        predictor_training_end_s = float(metadata['predictor_training_end_s'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Calibration artifact is missing finite training/calibration boundaries') from exc
    if not math.isfinite(audit_start_s) or audit_start_s <= calibration_end_s:
        raise ValueError('coverage audit must start strictly after the calibration time boundary')
    if audit_start_s <= predictor_training_end_s:
        raise ValueError('coverage audit must start strictly after predictor training')
    if predictor_artifact_sha256 != metadata.get('predictor_artifact_sha256'):
        raise ValueError('coverage audit predictor SHA-256 does not match calibration artifact')
    configured_interface_sha = metadata.get('predictor_interface_sha256')
    if configured_interface_sha is not None and predictor_interface_sha256 != configured_interface_sha:
        raise ValueError('coverage audit predictor interface SHA-256 does not match calibration artifact')

    calibration_conditions = {int(value) for value in metadata.get('calibration_condition_ids', ())}
    grouped: dict[tuple[int, str, float, str], list[LoadGuardCalibrationObservation]] = {}
    audit_conditions: set[int] = set()
    audit_timestamps: list[float] = []
    audit_count = 0
    for observation in observations:
        if observation.timestamp_s < audit_start_s:
            raise ValueError('coverage audit observations must be later than audit_start_s')
        if int(observation.condition_id) in calibration_conditions:
            raise ValueError('coverage audit conditions must be disjoint from calibration conditions')
        if observation.plan_kind != PLAN_KIND_ZOH_CANDIDATE:
            raise ValueError('coverage audit accepts only zoh_candidate plan observations')
        interface.validate_plan_kind(observation.plan_kind, online=False)
        key = (
            observation.blade,
            observation.optimization_state,
            observation.horizon_s,
            observation.plan_kind,
        )
        grouped.setdefault(key, []).append(observation)
        audit_conditions.add(int(observation.condition_id))
        audit_timestamps.append(float(observation.timestamp_s))
        audit_count += 1

    updated_groups: list[dict[str, Any]] = []
    candidate_group_passes: list[bool] = []
    for group_id, calibration_group in sorted(calibration.groups.items()):
        group = dict(calibration_group)
        if group.get('plan_kind') != PLAN_KIND_ZOH_CANDIDATE:
            updated_groups.append(group)
            continue
        key = (
            int(group['blade']),
            str(group['optimization_state']),
            float(group['horizon_s']),
            str(group['plan_kind']),
        )
        values = grouped.get(key, [])
        mismatch = np.asarray([value.plan_mismatch for value in values], dtype=np.float64)
        eligible_mask = mismatch <= calibration.config.max_plan_mismatch
        eligible_count = int(np.count_nonzero(eligible_mask))
        quantile = group.get('conformal_quantile')
        calibration_count = int(group.get('eligible_sample_count', 0))
        calibration_valid = (
            calibration_count >= calibration.config.minimum_samples_per_group
            and isinstance(quantile, (float, int))
            and math.isfinite(float(quantile))
        )
        if eligible_count > 0 and calibration_valid:
            actual = np.asarray([value.actual_ratio for value in values], dtype=np.float64)[eligible_mask]
            predicted = np.asarray([value.predicted_ratio for value in values], dtype=np.float64)[eligible_mask]
            covered_count = int(np.count_nonzero(actual <= predicted + float(quantile)))
            coverage = covered_count / eligible_count
            lower_bound = _wilson_lower_bound(
                covered_count,
                eligible_count,
                calibration.config.confidence_level,
            )
        else:
            covered_count = 0
            coverage = 0.0
            lower_bound = 0.0
        soft_eligible = bool(
            calibration_valid
            and eligible_count >= calibration.config.minimum_samples_per_group
            and lower_bound >= calibration.config.coverage_target
        )
        group.update({
            'coverage_audit_sample_count': eligible_count,
            'coverage_audit_condition_ids': sorted({
                int(value.condition_id)
                for index, value in enumerate(values)
                if eligible_mask[index]
            }),
            'coverage_audit_covered_count': covered_count,
            'coverage_audit_coverage': coverage,
            'coverage_audit_lower_confidence_bound': lower_bound,
            'soft_guard_eligible': soft_eligible,
        })
        candidate_group_passes.append(soft_eligible)
        updated_groups.append(group)

    if audit_count <= 0 or not candidate_group_passes:
        raise ValueError('coverage audit requires non-empty ZOH samples and calibration groups')
    result = {
        **dict(calibration_document),
        'metadata': {
            **metadata,
            'coverage_audit_start_s': float(audit_start_s),
            'coverage_audit_end_s': max(audit_timestamps),
            'coverage_audit_observation_count': audit_count,
            'coverage_audit_condition_ids': sorted(audit_conditions),
            'coverage_audit_population': str(coverage_population),
            'coverage_audit_predictor_artifact_sha256': predictor_artifact_sha256,
            'coverage_audit_predictor_interface_sha256': predictor_interface_sha256,
            'coverage_audit_independent_conditions': not bool(audit_conditions & calibration_conditions),
            'coverage_audit_independent_time': True,
            'coverage_audit_all_candidate_groups_pass': bool(all(candidate_group_passes)),
            'coverage_audit_strict_identity_verified': bool(strict_identity_verified),
            'soft_guard_authorized': bool(
                coverage_population == 'rl_shadow_strict'
                and strict_identity_verified
                and all(candidate_group_passes)
            ),
        },
        'groups': updated_groups,
    }
    return result