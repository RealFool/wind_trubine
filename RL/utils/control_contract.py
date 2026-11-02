from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from RL.utils.constants import ACTION_DIM, ACTION_VARIABLES


CONTROL_CONTRACT_SCHEMA_VERSION = 'b4_residual_control_contract_v2'
RESOLVED_CONTROL_CONTRACT_SCHEMA_VERSION = 'resolved_b4_residual_control_contract_v1'
ENGINEERING_DEMAND_LOW_NM = 0.0
ENGINEERING_DEMAND_HIGH_NM = 21_586_451.33303
NORMAL_GENERATION_PITCH_LOW_RAD = 0.0
NORMAL_GENERATION_PITCH_HIGH_RAD = 0.43
ACTION_WIRE_DTYPE = 'float64'
REQUIRED_ARTIFACT_HASH_FIELDS = (
    'state_artifact_sha256',
    'paired_pid_reference_sha256',
    'region_aware_baseline_sha256',
    'observation_schema_sha256',
    'predictor_interface_schema_sha256',
)
STATIC_CONTROL_CONTRACT_PATH_KEY = 'control_contract_static_path'
RESOLVED_CONTROL_CONTRACT_PATH_KEY = 'resolved_control_contract_path'
CONTROL_CONTRACT_FINGERPRINT_KEY = 'control_contract_fingerprint'


def action_array_float64(
    values: Sequence[float] | np.ndarray,
    field_name: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f'{field_name} must contain {ACTION_DIM} action values')
    if finite and not np.isfinite(array).all():
        raise ValueError(f'{field_name} must contain finite action values')
    return array


def _validate_action_bounds(
    action_low: Sequence[float] | np.ndarray,
    action_high: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = action_array_float64(action_low, 'action_low')
    high = action_array_float64(action_high, 'action_high')
    if np.any(high <= low):
        raise ValueError('action_high must exceed action_low')
    return low, high


def normalize_action_for_model(
    action_real: Sequence[float] | np.ndarray,
    action_low: Sequence[float] | np.ndarray,
    action_high: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = action_array_float64(action_real, 'action_real')
    low, high = _validate_action_bounds(action_low, action_high)
    normalized = (values - (high + low) / 2.0) / ((high - low) / 2.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def decode_model_action(
    action_normalized: Sequence[float] | np.ndarray,
    action_low: Sequence[float] | np.ndarray,
    action_high: Sequence[float] | np.ndarray,
) -> np.ndarray:
    normalized = action_array_float64(action_normalized, 'action_normalized')
    low, high = _validate_action_bounds(action_low, action_high)
    return np.clip(normalized, -1.0, 1.0) * ((high - low) / 2.0) + ((high + low) / 2.0)


@dataclass(frozen=True)
class PhysicalActionProjection:
    action_real: np.ndarray
    projection_applied: bool
    rate_limited: bool


@dataclass(frozen=True)
class PhysicalActionTransform:
    action_low: np.ndarray
    action_high: np.ndarray
    action_delta_low: np.ndarray | None = None
    action_delta_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        low, high = _validate_action_bounds(self.action_low, self.action_high)
        object.__setattr__(self, 'action_low', low)
        object.__setattr__(self, 'action_high', high)
        if (self.action_delta_low is None) != (self.action_delta_high is None):
            raise ValueError('action_delta_low and action_delta_high must be configured together')
        if self.action_delta_low is None:
            return
        delta_low = action_array_float64(self.action_delta_low, 'action_delta_low')
        delta_high = action_array_float64(self.action_delta_high, 'action_delta_high')
        if np.any(delta_low > delta_high):
            raise ValueError('action_delta_low must not exceed action_delta_high')
        object.__setattr__(self, 'action_delta_low', delta_low)
        object.__setattr__(self, 'action_delta_high', delta_high)

    def project(
        self,
        desired_action_real: Sequence[float] | np.ndarray,
        *,
        previous_action_real: Sequence[float] | np.ndarray | None = None,
        apply_rate_limit: bool = True,
        bypass_torque_decrease_rate_limit: bool = False,
    ) -> PhysicalActionProjection:
        desired = action_array_float64(desired_action_real, 'desired_action_real', finite=False)
        finite_desired = np.where(
            np.isnan(desired),
            0.0,
            np.where(np.isposinf(desired), self.action_high, np.where(np.isneginf(desired), self.action_low, desired)),
        )
        projected = np.clip(finite_desired, self.action_low, self.action_high)
        projection_applied = bool(
            not np.isfinite(desired).all()
            or np.any(np.abs(projected - finite_desired) > 1.0e-12)
        )
        rate_limited = False
        if apply_rate_limit and previous_action_real is not None and self.action_delta_low is not None:
            previous = action_array_float64(previous_action_real, 'previous_action_real')
            rate_limited_action = previous + np.clip(
                projected - previous,
                self.action_delta_low,
                self.action_delta_high,
            )
            if bypass_torque_decrease_rate_limit and projected[0] < previous[0]:
                rate_limited_action[0] = projected[0]
            rate_limited = bool(np.any(np.abs(rate_limited_action - projected) > 1.0e-12))
            projected = np.clip(rate_limited_action, self.action_low, self.action_high)
            projection_applied = projection_applied or rate_limited
        return PhysicalActionProjection(
            action_real=projected.astype(np.float64, copy=False),
            projection_applied=projection_applied,
            rate_limited=rate_limited,
        )

    def serialize_for_wire(
        self,
        action_real: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        projection = self.project(action_real, apply_rate_limit=False)
        return tuple(float(value) for value in projection.action_real)


def static_contract_fingerprint(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_control_contract_fingerprint(document: Mapping[str, Any]) -> str:
    canonical_document = dict(document)
    canonical_document.pop('control_contract_fingerprint', None)
    return static_contract_fingerprint(canonical_document)


def _portable_artifact_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _artifact_paths_and_hashes(
    artifact_paths: Mapping[str, Path | str],
    project_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    if set(artifact_paths) != set(REQUIRED_ARTIFACT_HASH_FIELDS):
        raise ValueError(
            'Resolved control contract artifacts must contain exactly '
            f'{list(REQUIRED_ARTIFACT_HASH_FIELDS)}'
        )
    portable_paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for field_name in REQUIRED_ARTIFACT_HASH_FIELDS:
        raw_path = artifact_paths[field_name]
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise FileNotFoundError(f'Resolved control contract artifact is missing: {field_name}={path}')
        portable_paths[field_name] = _portable_artifact_path(path, project_root)
        hashes[field_name] = sha256_file(path)
    return portable_paths, hashes


def build_resolved_control_contract(
    static_contract: Mapping[str, Any],
    *,
    artifact_paths: Mapping[str, Path | str],
    observation_schema: Mapping[str, Any],
    predictor_interface_schema_version: str,
    project_root: Path | str,
) -> dict[str, Any]:
    static_document = validate_static_control_contract(static_contract)
    root = Path(project_root).expanduser().resolve()
    if not isinstance(observation_schema, Mapping) or not observation_schema:
        raise ValueError('Resolved control contract observation_schema must be a non-empty object')
    if not isinstance(predictor_interface_schema_version, str) or not predictor_interface_schema_version.strip():
        raise ValueError('Resolved control contract predictor interface schema version must be non-empty')
    portable_paths, hashes = _artifact_paths_and_hashes(artifact_paths, root)
    document: dict[str, Any] = {
        'resolved_control_contract_schema_version': RESOLVED_CONTROL_CONTRACT_SCHEMA_VERSION,
        'control_contract_schema_version': CONTROL_CONTRACT_SCHEMA_VERSION,
        'static_control_contract_fingerprint': static_contract_fingerprint(static_document),
        'action': static_document['action'],
        'previous_action_normalization': static_document['previous_action_normalization'],
        'artifact_paths': portable_paths,
        'artifact_sha256': hashes,
        'observation_schema': json.loads(json.dumps(observation_schema, allow_nan=False)),
        'predictor_interface_schema_version': predictor_interface_schema_version.strip(),
    }
    document['control_contract_fingerprint'] = resolved_control_contract_fingerprint(document)
    return document


def write_resolved_control_contract(path: Path | str, document: Mapping[str, Any]) -> str:
    validated = validate_resolved_control_contract(document)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return str(validated['control_contract_fingerprint'])


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value.lower())
    )


def validate_resolved_control_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError('Resolved control contract must be an object')
    if document.get('resolved_control_contract_schema_version') != RESOLVED_CONTROL_CONTRACT_SCHEMA_VERSION:
        raise ValueError('Resolved control contract schema version is incompatible')
    if document.get('control_contract_schema_version') != CONTROL_CONTRACT_SCHEMA_VERSION:
        raise ValueError('Resolved control contract static schema version is incompatible')
    static_fingerprint = document.get('static_control_contract_fingerprint')
    if not _is_sha256(static_fingerprint):
        raise ValueError('Resolved control contract static fingerprint is invalid')
    action = document.get('action')
    normalization = document.get('previous_action_normalization')
    if not isinstance(action, Mapping) or not isinstance(normalization, Mapping):
        raise ValueError('Resolved control contract action and normalization must be objects')
    static_equivalent = {
        'control_contract_schema_version': CONTROL_CONTRACT_SCHEMA_VERSION,
        'action': action,
        'previous_action_normalization': normalization,
        'artifact_hash_fields': list(REQUIRED_ARTIFACT_HASH_FIELDS),
        'checkpoint_compatibility': {
            'require_resolved_control_contract_fingerprint': True,
            'allow_inference_torque_high_override': False,
            'reject_legacy_action_high_nm': 21_000_000.0,
        },
    }
    validate_static_control_contract(static_equivalent)
    artifact_paths = document.get('artifact_paths')
    artifact_hashes = document.get('artifact_sha256')
    if not isinstance(artifact_paths, Mapping) or set(artifact_paths) != set(REQUIRED_ARTIFACT_HASH_FIELDS):
        raise ValueError('Resolved control contract artifact paths are incompatible')
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(REQUIRED_ARTIFACT_HASH_FIELDS):
        raise ValueError('Resolved control contract artifact hashes are incompatible')
    for field_name in REQUIRED_ARTIFACT_HASH_FIELDS:
        if not isinstance(artifact_paths[field_name], str) or not artifact_paths[field_name].strip():
            raise ValueError(f'Resolved control contract artifact path is invalid: {field_name}')
        if not _is_sha256(artifact_hashes[field_name]):
            raise ValueError(f'Resolved control contract artifact hash is invalid: {field_name}')
    observation_schema = document.get('observation_schema')
    if not isinstance(observation_schema, Mapping) or not observation_schema:
        raise ValueError('Resolved control contract observation schema is invalid')
    predictor_version = document.get('predictor_interface_schema_version')
    if not isinstance(predictor_version, str) or not predictor_version.strip():
        raise ValueError('Resolved control contract predictor interface schema version is invalid')
    fingerprint = document.get('control_contract_fingerprint')
    if not _is_sha256(fingerprint) or fingerprint != resolved_control_contract_fingerprint(document):
        raise ValueError('Resolved control contract fingerprint is invalid')
    return dict(document)


def validate_resolved_contract_artifacts(
    document: Mapping[str, Any],
    *,
    project_root: Path | str,
) -> None:
    resolved = validate_resolved_control_contract(document)
    root = Path(project_root).expanduser().resolve()
    for field_name, expected_hash in resolved['artifact_sha256'].items():
        path = Path(resolved['artifact_paths'][field_name]).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f'Resolved control contract artifact is missing: {field_name}={path}')
        if sha256_file(path) != expected_hash:
            raise ValueError(f'Resolved control contract artifact hash is stale: {field_name}={path}')


def load_resolved_control_contract(
    path: Path | str,
    *,
    project_root: Path | str,
    expected_static_contract_fingerprint: str | None = None,
) -> tuple[dict[str, Any], str]:
    resolved_path = Path(path).expanduser()
    try:
        document = json.loads(resolved_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Resolved control contract does not exist: {resolved_path}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read resolved control contract: {resolved_path}') from exc
    validated = validate_resolved_control_contract(document)
    if (
        expected_static_contract_fingerprint is not None
        and validated['static_control_contract_fingerprint'] != expected_static_contract_fingerprint
    ):
        raise ValueError('Resolved control contract does not match the configured static contract')
    validate_resolved_contract_artifacts(validated, project_root=project_root)
    return validated, str(validated['control_contract_fingerprint'])


def validate_action_config_against_static_contract(
    env_config: Mapping[str, Any],
    control_config: Mapping[str, Any],
    static_contract: Mapping[str, Any],
) -> None:
    static_document = validate_static_control_contract(static_contract)
    action = static_document['action']
    expected_low = np.asarray(
        [action['engineering_demand_low_nm'], action['normal_generation_pitch_low_rad']],
        dtype=np.float64,
    )
    expected_high = np.asarray(
        [action['engineering_demand_high_nm'], action['normal_generation_pitch_high_rad']],
        dtype=np.float64,
    )
    configured_bounds = (
        ('env.action_low', env_config.get('action_low'), expected_low),
        ('env.action_high', env_config.get('action_high'), expected_high),
        ('env.observation_action_low', env_config.get('observation_action_low'), expected_low),
        ('env.observation_action_high', env_config.get('observation_action_high'), expected_high),
        ('env.safe_action_low', env_config.get('safe_action_low'), expected_low),
        ('env.safe_action_high', env_config.get('safe_action_high'), expected_high),
        ('control.safe_action_low', control_config.get('safe_action_low'), expected_low),
        ('control.safe_action_high', control_config.get('safe_action_high'), expected_high),
    )
    for field_name, value, expected in configured_bounds:
        actual = action_array_float64(value if value is not None else (), field_name)
        if not np.array_equal(actual, expected):
            raise ValueError(f'{field_name} must match the static control contract action bounds')


def resolve_configured_control_contract(
    config: Mapping[str, Any],
    *,
    project_root: Path | str,
) -> tuple[dict[str, Any], str] | None:
    static_path_value = config.get(STATIC_CONTROL_CONTRACT_PATH_KEY)
    resolved_path_value = config.get(RESOLVED_CONTROL_CONTRACT_PATH_KEY)
    configured_fingerprint = config.get(CONTROL_CONTRACT_FINGERPRINT_KEY)
    configured_values = (static_path_value, resolved_path_value, configured_fingerprint)
    if all(value is None for value in configured_values):
        return None
    if any(not isinstance(value, str) or not value.strip() for value in configured_values):
        raise ValueError(
            'New control-contract configs must define non-empty control_contract_static_path, '
            'resolved_control_contract_path, and control_contract_fingerprint'
        )
    root = Path(project_root).expanduser().resolve()
    static_path = Path(static_path_value).expanduser()
    resolved_path = Path(resolved_path_value).expanduser()
    if not static_path.is_absolute():
        static_path = root / static_path
    if not resolved_path.is_absolute():
        resolved_path = root / resolved_path
    static_contract, static_fingerprint = load_static_control_contract(static_path)
    env_config = config.get('env')
    control_config = config.get('control')
    if not isinstance(env_config, Mapping) or not isinstance(control_config, Mapping):
        raise ValueError('Configured control contract requires env and control objects')
    validate_action_config_against_static_contract(env_config, control_config, static_contract)
    resolved_contract, resolved_fingerprint = load_resolved_control_contract(
        resolved_path,
        project_root=root,
        expected_static_contract_fingerprint=static_fingerprint,
    )
    if configured_fingerprint != resolved_fingerprint:
        raise ValueError('Configured control_contract_fingerprint does not match the resolved control contract')
    if control_config.get('inference_torque_high') is not None:
        raise ValueError('New control-contract configs reject control.inference_torque_high overrides')
    return resolved_contract, resolved_fingerprint


def _require_finite_number(document: Mapping[str, Any], key: str, location: str) -> float:
    try:
        value = float(document[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{location}.{key} must be a finite number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{location}.{key} must be a finite number')
    return value


def validate_static_control_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError('Static control contract must be an object')
    if document.get('control_contract_schema_version') != CONTROL_CONTRACT_SCHEMA_VERSION:
        raise ValueError('Static control contract schema version is incompatible')
    action = document.get('action')
    normalization = document.get('previous_action_normalization')
    checkpoint = document.get('checkpoint_compatibility')
    artifact_hash_fields = document.get('artifact_hash_fields')
    if not isinstance(action, Mapping):
        raise ValueError('Static control contract action must be an object')
    if not isinstance(normalization, Mapping):
        raise ValueError('Static control contract previous_action_normalization must be an object')
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Static control contract checkpoint_compatibility must be an object')
    if tuple(artifact_hash_fields or ()) != REQUIRED_ARTIFACT_HASH_FIELDS:
        raise ValueError('Static control contract artifact_hash_fields are incompatible')
    if tuple(action.get('variables', ())) != tuple(ACTION_VARIABLES):
        raise ValueError('Static control contract action variables are incompatible')
    if action.get('wire_dtype') != ACTION_WIRE_DTYPE:
        raise ValueError('Static control contract action wire dtype must be float64')
    low = _require_finite_number(action, 'engineering_demand_low_nm', 'action')
    high = _require_finite_number(action, 'engineering_demand_high_nm', 'action')
    pitch_low = _require_finite_number(action, 'normal_generation_pitch_low_rad', 'action')
    pitch_high = _require_finite_number(action, 'normal_generation_pitch_high_rad', 'action')
    if not math.isclose(low, ENGINEERING_DEMAND_LOW_NM, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('Static control contract engineering_demand_low_nm is incompatible')
    if not math.isclose(high, ENGINEERING_DEMAND_HIGH_NM, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('Static control contract engineering_demand_high_nm is incompatible')
    if not math.isclose(pitch_low, NORMAL_GENERATION_PITCH_LOW_RAD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('Static control contract normal_generation_pitch_low_rad is incompatible')
    if not math.isclose(pitch_high, NORMAL_GENERATION_PITCH_HIGH_RAD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('Static control contract normal_generation_pitch_high_rad is incompatible')
    normalization_low = action_array_float64(
        normalization.get('action_low', ()),
        'previous_action_normalization.action_low',
    )
    normalization_high = action_array_float64(
        normalization.get('action_high', ()),
        'previous_action_normalization.action_high',
    )
    if not np.array_equal(normalization_low, np.asarray([low, pitch_low], dtype=np.float64)):
        raise ValueError('Previous-action normalization low bound is incompatible with action contract')
    if not np.array_equal(normalization_high, np.asarray([high, pitch_high], dtype=np.float64)):
        raise ValueError('Previous-action normalization high bound is incompatible with action contract')
    if checkpoint.get('require_resolved_control_contract_fingerprint') is not True:
        raise ValueError('Static control contract must require a resolved checkpoint fingerprint')
    if checkpoint.get('allow_inference_torque_high_override') is not False:
        raise ValueError('Static control contract must reject inference torque overrides')
    legacy_high = _require_finite_number(checkpoint, 'reject_legacy_action_high_nm', 'checkpoint_compatibility')
    if not math.isclose(legacy_high, 21_000_000.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('Static control contract legacy action high bound is incompatible')
    return dict(document)


def load_static_control_contract(path: Path | str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser()
    try:
        document = json.loads(resolved.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Static control contract does not exist: {resolved}') from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read static control contract: {resolved}') from exc
    return validate_static_control_contract(document), static_contract_fingerprint(document)