from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PREDICTOR_INTERFACE_SCHEMA_VERSION = 'causal_load_predictor_interface_v3'
LEGACY_PREDICTOR_INTERFACE_SCHEMA_VERSION = 'causal_load_predictor_interface_v2'
PLAN_KIND_REALIZED = 'realized_plan'
PLAN_KIND_ZOH_CANDIDATE = 'zoh_candidate'
PLAN_KINDS = (PLAN_KIND_REALIZED, PLAN_KIND_ZOH_CANDIDATE)
PREDICTOR_MODE_SHADOW = 'shadow'
PREDICTOR_MODE_SOFT = 'soft'
PREDICTOR_MODE_HARD_CANDIDATE = 'hard_candidate'
PREDICTOR_MODES = (
    PREDICTOR_MODE_SHADOW,
    PREDICTOR_MODE_SOFT,
    PREDICTOR_MODE_HARD_CANDIDATE,
)
DEFAULT_HISTORY_FRAMES = 50
DEFAULT_STEP_DURATION_S = 0.02
DEFAULT_HORIZONS_S = (0.10, 0.20, 0.50)
DEFAULT_SOFT_GUARD_HORIZONS_S = (0.10, 0.20)
DEFAULT_MAX_PLAN_MISMATCH = 0.05


def _finite_positive(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be a positive finite number') from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f'{field_name} must be a positive finite number')
    return parsed


def _float_tuple(values: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{field_name} must be a non-empty list')
    parsed = tuple(_finite_positive(value, f'{field_name}[{index}]') for index, value in enumerate(values))
    if any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
        raise ValueError(f'{field_name} must be strictly increasing')
    return parsed


@dataclass(frozen=True)
class LoadPredictorInterface:
    history_frames: int
    step_duration_s: float
    horizons_s: tuple[float, ...]
    online_candidate_plan_kinds: tuple[str, ...]
    realized_plan_online_authorized: bool
    soft_guard_horizons_s: tuple[float, ...]
    hard_candidate_default_enabled: bool
    required_inputs: tuple[str, ...]
    required_outputs: tuple[str, ...]
    candidate_plan_encoding: str
    candidate_plan_padding: str
    horizon_conditioning: str
    plan_kind_conditioning: str
    model_input_contract: str
    response_label_semantics: str
    peak_label_semantics: str
    max_plan_mismatch: float
    plan_mismatch_normalization: str
    causal: bool
    future_feature_lookahead_steps: int
    schema_version: str = PREDICTOR_INTERFACE_SCHEMA_VERSION

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> LoadPredictorInterface:
        if not isinstance(document, Mapping):
            raise ValueError('Load predictor interface must be an object')
        document_schema_version = document.get('schema_version')
        if document_schema_version not in (
            PREDICTOR_INTERFACE_SCHEMA_VERSION,
            LEGACY_PREDICTOR_INTERFACE_SCHEMA_VERSION,
        ):
            raise ValueError('Load predictor interface schema version is incompatible')
        try:
            history_frames = int(document['history_frames'])
            step_duration_s = _finite_positive(document['step_duration_s'], 'step_duration_s')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Load predictor history configuration is invalid') from exc
        if history_frames <= 0:
            raise ValueError('history_frames must be positive')
        horizons_s = _float_tuple(document.get('horizons_s'), 'horizons_s')
        soft_guard_horizons_s = _float_tuple(
            document.get('soft_guard_horizons_s'),
            'soft_guard_horizons_s',
        )
        if any(horizon not in horizons_s for horizon in soft_guard_horizons_s):
            raise ValueError('soft_guard_horizons_s must be a subset of horizons_s')
        if any(horizon >= max(horizons_s) for horizon in soft_guard_horizons_s):
            raise ValueError('soft_guard_horizons_s must exclude the non-online longest realized-plan horizon')
        online_plan_kinds = tuple(str(value) for value in document.get('online_candidate_plan_kinds', ()))
        if not online_plan_kinds or any(value not in PLAN_KINDS for value in online_plan_kinds):
            raise ValueError('online_candidate_plan_kinds must contain supported plan kinds')
        required_inputs = tuple(str(value) for value in document.get('required_inputs', ()))
        required_outputs = tuple(str(value) for value in document.get('required_outputs', ()))
        if not required_inputs or not required_outputs:
            raise ValueError('Load predictor required inputs and outputs must be non-empty')
        if document_schema_version == PREDICTOR_INTERFACE_SCHEMA_VERSION and any(
            field not in required_outputs
            for field in (
                'future_electrical_torque',
                'future_pitch',
                'future_blade_root_my_peak_ratio',
            )
        ):
            raise ValueError('Load predictor v3 interface must declare all three peak-load and response outputs')
        if document_schema_version == PREDICTOR_INTERFACE_SCHEMA_VERSION:
            required_candidate_inputs = ('candidate_action_plan', 'horizon_index', 'plan_kind_index')
            if any(name not in required_inputs for name in required_candidate_inputs):
                raise ValueError('Load predictor v3 interface must explicitly condition on plan, horizon, and plan kind')
            candidate_plan_encoding = str(document.get('candidate_plan_encoding', ''))
            candidate_plan_padding = str(document.get('candidate_plan_padding', ''))
            horizon_conditioning = str(document.get('horizon_conditioning', ''))
            plan_kind_conditioning = str(document.get('plan_kind_conditioning', ''))
            if candidate_plan_encoding != 'real_action_sequence_gru_normalized_by_engineering_bounds_v1':
                raise ValueError('candidate_plan_encoding is incompatible')
            if candidate_plan_padding != 'zero_after_selected_horizon_v1':
                raise ValueError('candidate_plan_padding is incompatible')
            if horizon_conditioning != 'discrete_horizon_index_embedding_v1':
                raise ValueError('horizon_conditioning is incompatible')
            if plan_kind_conditioning != 'realized_0_zoh_candidate_1_embedding_v1':
                raise ValueError('plan_kind_conditioning is incompatible')
            model_input_contract = str(document.get('model_input_contract', ''))
            response_label_semantics = str(document.get('response_label_semantics', ''))
            peak_label_semantics = str(document.get('peak_label_semantics', ''))
            if model_input_contract != (
                'candidate_plan_conditioned_tcn: '
                'history[50,31]+action_plan[max_horizon,2]+horizon_index+plan_kind_index'
            ):
                raise ValueError('model_input_contract is incompatible')
            if response_label_semantics != 'measured_actual_torque_and_pitch_at_t_plus_horizon_v1':
                raise ValueError('response_label_semantics is incompatible')
            if peak_label_semantics != (
                'max_abs_blade_root_my_over_strict_future_frames_t_plus_dt_through_t_plus_horizon_v1'
            ):
                raise ValueError('peak_label_semantics is incompatible')
        else:
            # Read-only compatibility for historical v2 artifacts. The v3
            # dataset trainer and v3 checkpoint loader independently reject it.
            candidate_plan_encoding = 'legacy_v2_unspecified'
            candidate_plan_padding = 'legacy_v2_unspecified'
            horizon_conditioning = 'legacy_v2_unspecified'
            plan_kind_conditioning = 'legacy_v2_unspecified'
            model_input_contract = 'legacy_v2_unspecified'
            response_label_semantics = 'legacy_v2_unspecified'
            peak_label_semantics = 'legacy_v2_unspecified'
        max_plan_mismatch = float(document.get('max_plan_mismatch', DEFAULT_MAX_PLAN_MISMATCH))
        if not math.isfinite(max_plan_mismatch) or max_plan_mismatch < 0.0:
            raise ValueError('max_plan_mismatch must be finite and non-negative')
        if document.get('plan_mismatch_normalization') != 'normalized_action_space_linf':
            raise ValueError('plan_mismatch_normalization is incompatible')
        if document.get('causal') is not True or int(document.get('future_feature_lookahead_steps', -1)) != 0:
            raise ValueError('Load predictor interface must be causal with zero future lookahead')
        if document.get('realized_plan_online_authorized') is not False:
            raise ValueError('realized_plan must not be online-authorized')
        if document_schema_version == PREDICTOR_INTERFACE_SCHEMA_VERSION and (
            'candidate_plan_conditioned_tcn' not in str(document.get('model_input_contract', ''))
        ):
            raise ValueError('v3 predictor interface is missing candidate-conditioned model-input semantics')
        return cls(
            history_frames=history_frames,
            step_duration_s=step_duration_s,
            horizons_s=horizons_s,
            online_candidate_plan_kinds=online_plan_kinds,
            realized_plan_online_authorized=False,
            soft_guard_horizons_s=soft_guard_horizons_s,
            hard_candidate_default_enabled=bool(document.get('hard_candidate_default_enabled', False)),
            required_inputs=required_inputs,
            required_outputs=required_outputs,
            candidate_plan_encoding=candidate_plan_encoding,
            candidate_plan_padding=candidate_plan_padding,
            horizon_conditioning=horizon_conditioning,
            plan_kind_conditioning=plan_kind_conditioning,
            model_input_contract=model_input_contract,
            response_label_semantics=response_label_semantics,
            peak_label_semantics=peak_label_semantics,
            max_plan_mismatch=max_plan_mismatch,
            plan_mismatch_normalization='normalized_action_space_linf',
            causal=True,
            future_feature_lookahead_steps=0,
            schema_version=str(document_schema_version),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> LoadPredictorInterface:
        resolved = Path(path).expanduser()
        try:
            document = json.loads(resolved.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Load predictor interface does not exist: {resolved}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read load predictor interface: {resolved}') from exc
        return cls.from_document(document)

    @property
    def horizon_frames(self) -> tuple[int, ...]:
        frames: list[int] = []
        for horizon in self.horizons_s:
            raw_frames = horizon / self.step_duration_s
            rounded_frames = int(round(raw_frames))
            if rounded_frames <= 0 or not math.isclose(raw_frames, rounded_frames, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f'Horizon {horizon} s is not representable by step_duration_s')
            frames.append(rounded_frames)
        return tuple(frames)

    def validate_plan_kind(self, plan_kind: str, *, online: bool) -> str:
        value = str(plan_kind)
        if value not in PLAN_KINDS:
            raise ValueError(f'Unsupported load predictor plan_kind={value!r}')
        if online and value != PLAN_KIND_ZOH_CANDIDATE:
            raise ValueError(f'plan_kind={value!r} is not authorized online')
        if online and value not in self.online_candidate_plan_kinds:
            raise ValueError(f'plan_kind={value!r} is not authorized online')
        if online and value == PLAN_KIND_REALIZED and not self.realized_plan_online_authorized:
            raise ValueError('realized_plan is not authorized online')
        return value


def normalized_action_plan_mismatch(
    actual_plan: np.ndarray | Sequence[Sequence[float]],
    candidate_plan: np.ndarray | Sequence[Sequence[float]],
    action_low: Sequence[float],
    action_high: Sequence[float],
) -> float:
    actual = np.asarray(actual_plan, dtype=np.float64)
    candidate = np.asarray(candidate_plan, dtype=np.float64)
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if actual.shape != candidate.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError('Action plans must have equal shape [horizon, 2]')
    if low.shape != (2,) or high.shape != (2,) or np.any(high <= low):
        raise ValueError('Action bounds must contain two increasing values')
    if not np.isfinite(actual).all() or not np.isfinite(candidate).all():
        raise ValueError('Action plans must be finite')
    normalized_error = np.abs(actual - candidate) / (high - low)
    return float(np.max(normalized_error))


def make_zoh_candidate_plan(
    action_real: Sequence[float],
    horizon_frames: int,
) -> np.ndarray:
    action = np.asarray(action_real, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError('ZOH candidate action must contain two finite values')
    if horizon_frames <= 0:
        raise ValueError('horizon_frames must be positive')
    return np.repeat(action[None, :], int(horizon_frames), axis=0)