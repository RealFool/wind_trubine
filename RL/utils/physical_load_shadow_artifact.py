"""Strict, separately versioned import of frozen D0-R artifacts.

Loading permits development shadow calculations only. PID error quantiles
never acquire RL coverage, counterfactual, or control authorization here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from RL.utils.physical_load_margin import (
    DOMAIN_NAMES, INPUT_FIELDS, MODEL_SCHEMA, REGIONS, SURFACE_FEATURE_NAMES,
    GreyboxModel, document_sha256, load_protocol, verify_static_project,
)
from RL.utils.physical_load_margin_dynamic import (
    AUTHORIZATION, DYNAMIC_MODEL_SCHEMA, SCALE_FEATURE_NAMES, SCALE_SCHEMA,
    DynamicModel, ErrorScaleModel, dynamic_feature_names, load_development_protocol,
)


BACKEND = 'physical_fir_v2'
BUNDLE_SCHEMA = 'physical_load_shadow_bundle_v1'
RUNTIME_SCHEMA = 'physical_load_shadow_runtime_v1'
RUNTIME_AUTHORIZATION = {
    'mode': 'shadow_development_only', 'policy_acceptance': False,
    'soft_authorized': False, 'hard_authorized': False,
    'counterfactual_validation': False, 'rl_margin_coverage_authorized': False,
}
ARTIFACT_FILENAMES = {
    'base_protocol': 'frozen_base_protocol.json',
    'development_protocol': 'frozen_protocol.json',
    'physics_model': 'frozen_physics_model.json',
    'dynamic_model': 'frozen_dynamic_model.json',
    'error_scales': 'frozen_error_scales.json',
    'joint_calibration': 'frozen_joint_calibration.json',
}
ACTION_SEMANTICS = 'final_successfully_sent_demand_zoh_development_v1'
LABEL_SEMANTICS = 'strict_future_my_with_executed_intervals_origin_through_endpoint_minus_dt_v1'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def strict_json(path: Path, maximum_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f'Physical shadow JSON exceeds its size budget: {path.name}')
    def reject_constant(value: str) -> None:
        raise ValueError(f'Non-finite JSON constant: {value}')
    result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                        parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError('Physical shadow artifact must be a JSON object')
    return result


def artifact_path(value: str, root: Path) -> Path:
    path = Path(str(value).replace('\\', '/')).expanduser()
    return path if path.is_absolute() else root / path


def _array(value: Any, shape: tuple[int, ...], *, nullable: bool = False) -> np.ndarray:
    if value is None and nullable:
        return np.full(shape, np.nan, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Physical artifact requires a finite array of shape {shape}')
    return result


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Physical artifact counts must be non-negative integers')
    return value


@dataclass(frozen=True)
class PhysicalShadowArtifacts:
    model: DynamicModel
    scale: ErrorScaleModel
    calibration: dict[str, Any]
    bundle: dict[str, Any]
    bundle_file_sha256: str
    dynamic_file_sha256: str


def load_physical_shadow_artifacts(path: Path, *, expected_sha256: str,
                                   project_root: Path) -> PhysicalShadowArtifacts:
    if not expected_sha256 or file_sha256(path) != expected_sha256:
        raise ValueError('Physical shadow bundle file SHA mismatch')
    bundle = strict_json(path)
    if (bundle.get('schema_version') != BUNDLE_SCHEMA or
            bundle.get('runtime_schema_version') != RUNTIME_SCHEMA or
            bundle.get('authorization') != RUNTIME_AUTHORIZATION or
            bundle.get('external_fields') != list(INPUT_FIELDS) or
            bundle.get('action_semantics') != ACTION_SEMANTICS or
            bundle.get('label_semantics') != LABEL_SEMANTICS or
            bundle.get('primary_horizons_s') != [0.1, 0.2]):
        raise ValueError('Physical shadow bundle schema/whitelist/authorization mismatch')
    entries = bundle.get('artifacts', {})
    if set(entries) != set(ARTIFACT_FILENAMES):
        raise ValueError('Physical shadow bundle requires all six frozen artifacts')
    docs: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, entry in entries.items():
        paths[name] = artifact_path(entry['path'], project_root)
        if file_sha256(paths[name]) != entry['file_sha256']:
            raise ValueError(f'Physical shadow artifact file SHA mismatch: {name}')
        docs[name] = strict_json(paths[name])
        if document_sha256(docs[name]) != entry['document_sha256']:
            raise ValueError(f'Physical shadow canonical SHA mismatch: {name}')
    base = load_protocol(paths['base_protocol'])
    spec, source_base = load_development_protocol(paths['development_protocol'], project_root)
    if source_base != base or base != docs['base_protocol'] or spec != docs['development_protocol']:
        raise ValueError('Frozen physical protocol dependency mismatch')
    if base['step_duration_s'] != 0.02 or base['horizons_s'] != [0.1, 0.2, 0.5]:
        raise ValueError('This runtime supports only the frozen 20ms/short-window contract')
    project = artifact_path(bundle['engineering_project']['path'], project_root)
    if file_sha256(project) != bundle['engineering_project']['file_sha256']:
        raise ValueError('Physical engineering project SHA mismatch')
    verify_static_project(project, base)
    physics_doc = docs['physics_model']
    if (physics_doc.get('schema_version') != MODEL_SCHEMA or
            physics_doc.get('protocol_sha256') != document_sha256(base) or
            physics_doc.get('used_external_fields') != list(INPUT_FIELDS) or
            physics_doc.get('surface_feature_names') != list(SURFACE_FEATURE_NAMES) or
            physics_doc.get('domain_names') != list(DOMAIN_NAMES) or
            physics_doc.get('moment_unit') != 'Nm' or
            physics_doc.get('authorization') != base['authorization'] or
            set(physics_doc.get('fit', {})) != set(REGIONS)):
        raise ValueError('Physical greybox identity/units mismatch')
    coefficients, counts, lower, upper, conditioning = [], [], [], [], {}
    for region in REGIONS:
        entry = physics_doc['fit'][region]
        n = _count(entry['training_rows'])
        counts.append(n)
        coefficients.append(_array(entry['coefficients_nm'], (3, len(SURFACE_FEATURE_NAMES)), nullable=True))
        lower.append(_array(entry['domain_minimum'], (len(DOMAIN_NAMES),), nullable=n == 0))
        upper.append(_array(entry['domain_maximum'], (len(DOMAIN_NAMES),), nullable=n == 0))
        conditioning[region] = entry['regularized_condition_numbers']
    physics = GreyboxModel(base, np.asarray(coefficients), np.asarray(counts), np.asarray(lower),
                           np.asarray(upper), conditioning)
    doc = docs['dynamic_model']
    lags = tuple(doc['lags_frames'])
    width, steps = len(dynamic_feature_names(lags)), int(round(max(base['horizons_s']) / 0.02))
    if (doc.get('schema_version') != DYNAMIC_MODEL_SCHEMA or doc.get('authorization') != AUTHORIZATION or
            doc.get('base_protocol_sha256') != document_sha256(base) or
            doc.get('physics_model_sha256') != document_sha256(physics_doc) or
            doc.get('development_protocol_sha256') != document_sha256(spec) or
            doc.get('external_fields') != list(INPUT_FIELDS) or
            doc.get('feature_names') != list(dynamic_feature_names(lags)) or
            list(lags) not in spec['dynamic_model']['history_lag_candidates_frames'] or
            doc['ridge_fraction'] not in spec['dynamic_model']['ridge_fraction_candidates'] or
            set(doc.get('coefficients', {})) != set(REGIONS)):
        raise ValueError('Dynamic model identity/features mismatch')
    weights = np.full((3, 3, width, steps), np.nan)
    counts = np.zeros((3, 3), dtype=np.int64)
    lower = np.full((3, 3, width), np.nan)
    upper = lower.copy()
    conditioning_array = np.full((3, 3), np.nan)
    for ri, region in enumerate(REGIONS):
        if len(doc['coefficients'][region]) != 3:
            raise ValueError('Exactly three dynamic blade models are required')
        for blade, entry in enumerate(doc['coefficients'][region]):
            counts[ri, blade] = _count(entry['training_origins'])
            weights[ri, blade] = _array(entry['weights'], (width, steps), nullable=True)
            lower[ri, blade] = _array(entry['domain_minimum'], (width,), nullable=counts[ri, blade] == 0)
            upper[ri, blade] = _array(entry['domain_maximum'], (width,), nullable=counts[ri, blade] == 0)
            if entry['condition_number'] is not None:
                conditioning_array[ri, blade] = _array(entry['condition_number'], ())
    model = DynamicModel(physics, spec, lags, doc['ridge_fraction'], weights, counts, lower, upper,
                         conditioning_array)
    scale_doc = docs['error_scales']['dynamic_fir']
    if (scale_doc.get('schema_version') != SCALE_SCHEMA or scale_doc.get('authorization') != AUTHORIZATION or
            scale_doc.get('dynamic_model_sha256') != document_sha256(doc) or
            scale_doc.get('point_method') != 'dynamic_fir' or
            scale_doc.get('feature_names') != list(SCALE_FEATURE_NAMES) or
            scale_doc.get('parameters') != spec['conditional_scale'] or
            set(scale_doc.get('regions', {})) != set(REGIONS)):
        raise ValueError('Physical error scale identity/features mismatch')
    width = len(SCALE_FEATURE_NAMES)
    weights = np.full((3, 3, width, 3), np.nan)
    counts = np.zeros((3, 3), dtype=np.int64)
    lower = np.full((3, 3, width), np.nan)
    upper = lower.copy()
    for ri, region in enumerate(REGIONS):
        if len(scale_doc['regions'][region]) != 3:
            raise ValueError('Exactly three error-scale blade models are required')
        for blade, entry in enumerate(scale_doc['regions'][region]):
            counts[ri, blade] = _count(entry['training_origins'])
            weights[ri, blade] = _array(entry['weights'], (width, 3), nullable=True)
            lower[ri, blade] = _array(entry['domain_minimum'], (width,), nullable=counts[ri, blade] == 0)
            upper[ri, blade] = _array(entry['domain_maximum'], (width,), nullable=counts[ri, blade] == 0)
    scale = ErrorScaleModel(model, 'dynamic_fir', weights, counts, lower, upper)
    for imported, original in ((physics.to_document(), physics_doc), (model.to_document(), doc),
                               (scale.to_document(), scale_doc)):
        if document_sha256(imported) != document_sha256(original):
            raise ValueError('Physical artifact strict round-trip mismatch')
    for lo, hi in ((physics.domain_minimum, physics.domain_maximum),
                   (model.minimum, model.maximum), (scale.minimum, scale.maximum)):
        if np.any(np.isfinite(lo) & np.isfinite(hi) & (lo > hi)):
            raise ValueError('Physical artifact domain bounds are reversed')
    calibration = docs['joint_calibration']['dynamic_fir']
    if (calibration.get('schema_version') != 'joint_load_margin_development_calibration_v2' or
            calibration.get('authorization') != AUTHORIZATION or
            calibration.get('point_method') != 'dynamic_fir' or
            calibration.get('dynamic_model_sha256') != document_sha256(doc) or
            calibration.get('scale_model_sha256') != document_sha256(scale_doc) or
            calibration.get('primary_horizons_s') != [0.1, 0.2] or
            set(calibration.get('groups', {})) != set(REGIONS)):
        raise ValueError('Physical joint calibration identity mismatch')
    for group in calibration['groups'].values():
        n = _count(group['origin_count'])
        if not isinstance(group['supported'], bool):
            raise ValueError('Physical calibration support must be boolean')
        case_counts = group.get('case_origin_counts')
        if not isinstance(case_counts, dict) or any(
                key not in {str(cid) for cid in base['split']['calibration']} for key in case_counts):
            raise ValueError('Physical calibration may use only frozen calibration cases')
        if sum(_count(value) for value in case_counts.values()) != n:
            raise ValueError('Physical calibration case counts do not close')
        if group['supported']:
            values = np.r_[_array(group['legacy_per_blade95_nm'], (2, 3)).ravel(),
                           _array([group['joint_constant95_nm'], group['joint_conditional95']], (2,))]
            if n < spec['joint_margin']['minimum_calibration_origins'] or np.any(values < 0.0):
                raise ValueError('Physical calibration count/quantile invalid')
        elif (n >= spec['joint_margin']['minimum_calibration_origins'] or any(
                group.get(key) is not None for key in (
                    'legacy_per_blade95_nm', 'joint_constant95_nm', 'joint_conditional95'))):
            raise ValueError('Unsupported physical calibration must have insufficient count and null margins')
    return PhysicalShadowArtifacts(model, scale, calibration, bundle, expected_sha256,
                                   entries['dynamic_model']['file_sha256'])