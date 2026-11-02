"""Development-only causal FIR load increments and joint empirical margins.

All measurements are from the existing v1 whitelist. Direct multi-step FIR
regression avoids recursive rollout instability; it is not a modal observer,
a BEM solver, a counterfactual action model, or a certified error bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from RL.utils.physical_load_margin import (
    INPUT_FIELDS, REGIONS, GreyboxModel, PidCase, PredictionBatch,
    document_sha256, future_labels, load_protocol, observation_inputs,
    read_pid_case, surface_features,
)


DEVELOPMENT_SCHEMA = 'physical_load_margin_dynamic_development_protocol_v2'
DYNAMIC_MODEL_SCHEMA = 'causal_fir_load_increment_v2'
SCALE_SCHEMA = 'causal_load_error_scale_v2'
POINT_METHODS = ('v1_greybox', 'physical_only', 'persistence', 'causal_trend', 'dynamic_fir')
MARGIN_METHODS = ('legacy_per_blade95_nm', 'joint_constant95_nm', 'joint_conditional95')
AUTHORIZATION = {
    'mode': 'offline_development', 'policy_acceptance': False,
    'soft_authorized': False, 'hard_authorized': False,
    'counterfactual_validation': False, 'fresh_independent_audit': False,
}


def load_development_protocol(path: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(path.read_text(encoding='utf-8'))
    if spec.get('schema_version') != DEVELOPMENT_SCHEMA or spec.get('authorization') != AUTHORIZATION:
        raise ValueError('Development protocol must be offline and never authorize control')
    base = load_protocol(project_root / spec['base_protocol_path'])
    if document_sha256(base) != spec['base_protocol_sha256']:
        raise ValueError('Frozen base protocol SHA mismatch')
    policy = spec['data_policy']
    if policy['read_old_audit'] is not False or policy['blocked_old_audit_ids'] != base['split']['audit']:
        raise ValueError('Previously exposed audit must remain blocked')
    if policy['allowed_roles'] != ['train', 'validation', 'calibration'] or policy['fit_roles'] != ['train']:
        raise ValueError('Only train may fit; only declared development roles may be read')
    if policy['selection_roles'] != ['validation'] or policy['margin_calibration_roles'] != ['calibration']:
        raise ValueError('Selection and margin calibration roles cannot overlap')
    model = spec['dynamic_model']
    if not model['history_lag_candidates_frames'] or not model['ridge_fraction_candidates']:
        raise ValueError('The finite candidate grid must not be empty')
    for lags in model['history_lag_candidates_frames']:
        if not lags or any(int(lag) != lag or lag <= 0 for lag in lags) or sorted(set(lags)) != lags:
            raise ValueError('History lags must be positive, distinct and ordered')
        if lags[-1] * base['step_duration_s'] > base['warmup_s']:
            raise ValueError('History lags cannot exceed the frozen warmup')
    positive = (*model['ridge_fraction_candidates'], model['fit_stride_frames'],
                model['normalized_feature_domain_floor'], spec['conditional_scale']['ridge_fraction'],
                spec['conditional_scale']['error_floor_nm'], spec['joint_margin']['minimum_calibration_origins'])
    if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
        raise ValueError('Model and calibration parameters must be finite and positive')
    for value in (model['fit_stride_frames'], spec['joint_margin']['minimum_calibration_origins'],
                  spec['evaluation']['minimum_joint_evaluation_origins_per_region']):
        if int(value) != value or value < 1:
            raise ValueError('Stride and minimum counts must be positive integers')
    if spec['conditional_scale']['history_lags_frames'] != [1, 5, 10, 20]:
        raise ValueError('Error-scale histories must match the fixed causal features')
    if spec['selection']['compare_methods'] != list(POINT_METHODS):
        raise ValueError('Point comparison methods must remain frozen')
    for key in ('minimum_known_origin_fraction', 'minimum_common_comparison_fraction'):
        if not 0.0 < spec['selection'][key] <= 1.0:
            raise ValueError('Invalid minimum supported comparison fraction')
    if spec['evaluation']['primary_near_peak_ratio'] != base['evaluation']['primary_near_peak_ratio']:
        raise ValueError('The near-peak diagnostic threshold must match the base protocol')
    for key in ('domain_padding_fraction',):
        if not math.isfinite(model[key]) or model[key] < 0.0:
            raise ValueError('Invalid feature domain padding')
    if not 0.0 < spec['joint_margin']['quantile'] < 1.0:
        raise ValueError('Invalid joint empirical quantile')
    if spec['joint_margin']['scope_horizons_s'] != spec['selection']['primary_horizons_s']:
        raise ValueError('Joint calibration must cover both selected primary horizons')
    if any(h not in base['horizons_s'] for h in spec['joint_margin']['scope_horizons_s']):
        raise ValueError('Unsupported horizon')
    if spec['joint_margin']['methods'] != list(MARGIN_METHODS):
        raise ValueError('Margin comparison methods must remain frozen')
    if spec['conditional_scale']['maximum_scale_nm'] <= spec['conditional_scale']['error_floor_nm']:
        raise ValueError('Error scale ceiling must exceed its positive floor')
    if spec['development_gate']['integration_decision_always'] != 'BLOCKED_NO_FRESH_INDEPENDENT_AUDIT':
        raise ValueError('Development verification can never authorize integration')
    return spec, base


def read_development_case(path: Path, cid: int, role: str, spec: dict[str, Any], base: dict[str, Any]) -> PidCase:
    """Check the role before opening or hashing the file, including old audit."""
    if cid in spec['data_policy']['blocked_old_audit_ids'] or role not in spec['data_policy']['allowed_roles']:
        raise ValueError('Old audit / undeclared data role is blocked before file access')
    if cid not in base['split'].get(role, ()):
        raise ValueError('Condition does not belong to its frozen development role')
    if path.name != f'data_windspeed{cid}.csv' or path.resolve().name != path.name:
        raise ValueError('Development source filename/condition identity mismatch')
    return read_pid_case(path, cid, base)


def physical_signed_trajectory(physics: GreyboxModel, case: PidCase, origins: np.ndarray) -> np.ndarray:
    """Same v1 closed-form physics, with signed frames retained for FIR labels.

This function reads only origin observations. It intentionally has no
alternative-action interface; the only plan is recorded origin PID ZOH.
"""
    p = physics.protocol
    params = p['static_parameters']
    steps = int(round(max(p['horizons_s']) / p['step_duration_s']))
    h = np.arange(1, steps + 1)[None, :] * p['step_duration_s']
    wind, omega, beta, yaw, azimuth, acc_fa, acc_ss = observation_inputs(case, origins)
    derivatives = case.derivatives[origins]
    actions = np.nan_to_num(case.actions[origins].copy(), nan=0.0, posinf=0.0, neginf=0.0)
    torque = case.values['I_ElectricalTorque'][origins, None]
    target = np.minimum(actions[:, 0, None], params['generator_actual_torque_max_nm'])
    delta_torque = target - torque
    tau = params['generator_time_constant_s']
    decay_torque = np.exp(-h / tau)
    future_omega = omega[:, None] + derivatives[:, 0, None] * h - delta_torque / params['rotor_inertia_kg_m2'] * (h - tau * (1.0 - decay_torque))
    future_azimuth = azimuth[:, None] + omega[:, None] * h + 0.5 * derivatives[:, 0, None] * h ** 2 - delta_torque / params['rotor_inertia_kg_m2'] * (
        0.5 * h ** 2 - tau * h + tau ** 2 * (1.0 - decay_torque)
    )
    natural = params['pitch_natural_frequency_rad_s']
    decay_pitch = params['pitch_damping_ratio'] * natural
    damped = natural * math.sqrt(1.0 - params['pitch_damping_ratio'] ** 2)
    error = beta[:, None] - actions[:, 1, None]
    sine = (derivatives[:, 1, None] + decay_pitch * error) / damped
    future_pitch = actions[:, 1, None] + np.exp(-decay_pitch * h) * (error * np.cos(damped * h) + sine * np.sin(damped * h))
    shape = future_omega.shape
    future_inputs = (np.broadcast_to(wind[:, None], shape), future_omega, future_pitch,
                     np.broadcast_to(yaw[:, None], shape), future_azimuth,
                     np.broadcast_to(acc_fa[:, None], shape), np.broadcast_to(acc_ss[:, None], shape))
    result = np.full((len(origins), steps, 3), np.nan)
    for ri in range(len(REGIONS)):
        selected = np.flatnonzero(case.region[origins] == ri)
        if not len(selected) or not np.isfinite(physics.coefficients[ri]).all():
            continue
        for blade in range(3):
            coefficient = physics.coefficients[ri, blade]
            current = surface_features(*(x[selected] for x in (wind, omega, beta, yaw, azimuth, acc_fa, acc_ss)),
                                       blade, params, p['model']['moment_scale_nm']) @ coefficient
            future = np.einsum('nkp,p->nk', surface_features(*(x[selected] for x in future_inputs), blade,
                                                           params, p['model']['moment_scale_nm']), coefficient)
            result[selected, :, blade] = case.my_nm[origins[selected], blade, None] + future - current[:, None]
    return result


def lag_indices(case: PidCase, origins: np.ndarray, lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    indices = origins[:, None] - np.asarray((0, *lags))[None, :]
    bounded = np.maximum(indices, 0)
    valid = (indices >= 0).all(axis=1) & case.valid[bounded].all(axis=1)
    valid &= (case.segment[bounded] == case.segment[origins, None]).all(axis=1)
    return bounded, valid


def dynamic_feature_names(lags: tuple[int, ...]) -> tuple[str, ...]:
    names = ('one', 'own_residual', 'common_residual', 'own_my', 'own_my_rate',
             'omega_rate', 'beta_rate', 'wind', 'beta', 'omega', 'sin_psi', 'cos_psi',
             'yaw', 'actual_torque', 'acc_fa', 'acc_ss')
    return (*names, *(f'{name}_lag{lag}' for lag in lags for name in (
        'own_residual_delta', 'common_residual_delta', 'own_my_delta', 'wind_delta', 'acc_fa', 'acc_ss',
    )))


def dynamic_features(physics: GreyboxModel, case: PidCase, origins: np.ndarray,
                     lags: tuple[int, ...], blade: int) -> tuple[np.ndarray, np.ndarray]:
    indices, valid = lag_indices(case, origins, lags)
    width = len(dynamic_feature_names(lags))
    design = np.full((len(origins), width), np.nan)
    moment_scale = physics.protocol['model']['moment_scale_nm']
    for ri in range(len(REGIONS)):
        selected = np.flatnonzero(case.region[origins] == ri)
        if not len(selected) or not np.isfinite(physics.coefficients[ri]).all():
            continue
        idx = indices[selected]
        inputs = observation_inputs(case, idx)
        residual = np.empty((*idx.shape, 3))
        for bj in range(3):
            # Use the origin's region coefficients for ALL historical rows;
            # classification switches cannot create artificial residual jumps.
            static = np.einsum('nlp,p->nl', surface_features(*inputs, bj, physics.protocol['static_parameters'],
                                                         moment_scale), physics.coefficients[ri, bj])
            residual[:, :, bj] = (case.my_nm[idx, bj] - static) / moment_scale
        own = residual[:, :, blade]
        common = residual.mean(axis=-1)
        my = case.my_nm[idx, blade] / moment_scale
        wind, omega, beta, yaw, azimuth, acc_fa, acc_ss = inputs
        psi = azimuth[:, 0] + blade * 2.0 * math.pi / 3.0
        derivative = case.derivatives[origins[selected]]
        columns = [np.ones(len(selected)), own[:, 0], common[:, 0], my[:, 0],
                   derivative[:, blade + 2] / moment_scale, derivative[:, 0] / 0.1,
                   derivative[:, 1] / 0.1, wind[:, 0] / 20.0, beta[:, 0] / 0.2,
                   omega[:, 0] / 0.8, np.sin(psi), np.cos(psi), yaw[:, 0],
                   case.values['I_ElectricalTorque'][origins[selected]] / 21e6,
                   acc_fa[:, 0], acc_ss[:, 0]]
        for li in range(1, len(lags) + 1):
            columns.extend((own[:, 0] - own[:, li], common[:, 0] - common[:, li],
                            my[:, 0] - my[:, li], (wind[:, 0] - wind[:, li]) / 5.0,
                            acc_fa[:, li], acc_ss[:, li]))
        design[selected] = np.column_stack(columns)
    return design, valid & np.isfinite(design).all(axis=1)


@dataclass
class PreparedCase:
    case: PidCase
    origins: np.ndarray
    baseline: PredictionBatch
    physical_signed_nm: np.ndarray
    labels: dict[str, np.ndarray]


def prepare_case(physics: GreyboxModel, case: PidCase, stride: int) -> PreparedCase:
    origins = np.arange(0, len(case.time) - 1, stride, dtype=np.int64)
    baseline = physics.predict(case, origins, 0.1)
    trajectory = physical_signed_trajectory(physics, case, origins)
    # Evaluation reads are separate from all model feature/rollout functions.
    labels = future_labels(case, origins, physics.protocol)
    return PreparedCase(case, origins, baseline, trajectory, labels)


def solve_ridge(xtx: np.ndarray, xty: np.ndarray, count: int, fraction: float) -> tuple[np.ndarray, float]:
    rms = np.maximum(np.sqrt(np.diag(xtx) / count), 1e-6)
    gram = xtx / np.outer(rms, rms) + np.eye(len(rms)) * fraction * count
    beta = np.linalg.solve(gram, xty / rms[:, None]) / rms[:, None]
    return beta, float(np.linalg.cond(gram))


class DynamicFitter:
    def __init__(self, physics: GreyboxModel, spec: dict[str, Any], lags: tuple[int, ...]) -> None:
        self.physics, self.spec, self.lags = physics, spec, lags
        width = len(dynamic_feature_names(lags))
        steps = int(round(max(physics.protocol['horizons_s']) / physics.protocol['step_duration_s']))
        self.xtx = np.zeros((len(REGIONS), 3, width, width))
        self.xty = np.zeros((len(REGIONS), 3, width, steps))
        self.counts = np.zeros((len(REGIONS), 3), dtype=np.int64)
        self.minimum = np.full((len(REGIONS), 3, width), np.inf)
        self.maximum = np.full_like(self.minimum, -np.inf)

    def add_case(self, prepared: PreparedCase) -> None:
        case, origins = prepared.case, prepared.origins
        if case.condition_id not in self.physics.protocol['split']['train']:
            raise ValueError('Only frozen train cases may fit dynamic increments')
        eligible = (prepared.baseline.supported & prepared.labels['approximate_zoh']).all(axis=1)
        indices = origins[:, None] + np.arange(1, self.xty.shape[-1] + 1)[None, :]
        measured = case.my_nm[np.minimum(indices, len(case.time) - 1)]
        target = (measured - prepared.physical_signed_nm) / self.physics.protocol['model']['moment_scale_nm']
        eligible &= np.isfinite(target).all(axis=(1, 2)) & np.isfinite(prepared.physical_signed_nm).all(axis=(1, 2))
        for blade in range(3):
            features, history_valid = dynamic_features(self.physics, case, origins, self.lags, blade)
            for ri in range(len(REGIONS)):
                mask = eligible & history_valid & (case.region[origins] == ri)
                if not mask.any():
                    continue
                x, y = features[mask], target[mask, :, blade]
                self.xtx[ri, blade] += x.T @ x
                self.xty[ri, blade] += x.T @ y
                self.counts[ri, blade] += len(x)
                self.minimum[ri, blade] = np.minimum(self.minimum[ri, blade], x.min(axis=0))
                self.maximum[ri, blade] = np.maximum(self.maximum[ri, blade], x.max(axis=0))

    def finish(self, ridge_fraction: float) -> DynamicModel:
        coefficients = np.full_like(self.xty, np.nan)
        conditioning = np.full((len(REGIONS), 3), np.nan)
        for ri in range(len(REGIONS)):
            for blade in range(3):
                count = int(self.counts[ri, blade])
                if count < self.xtx.shape[-1]:
                    continue
                coefficients[ri, blade], conditioning[ri, blade] = solve_ridge(
                    self.xtx[ri, blade], self.xty[ri, blade], count, ridge_fraction)
        return DynamicModel(self.physics, self.spec, self.lags, ridge_fraction, coefficients,
                            self.counts.copy(), self.minimum.copy(), self.maximum.copy(), conditioning)


@dataclass
class DynamicBatch:
    origins: np.ndarray
    peaks_nm: dict[str, np.ndarray]
    supported: np.ndarray
    reasons: np.ndarray
    signed_dynamic_nm: np.ndarray


@dataclass
class DynamicModel:
    physics: GreyboxModel
    spec: dict[str, Any]
    lags: tuple[int, ...]
    ridge_fraction: float
    coefficients: np.ndarray
    counts: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    conditioning: np.ndarray

    @property
    def name(self) -> str:
        return f'fir_lag{self.lags[-1]}_ridge{self.ridge_fraction:g}'

    def predict(self, case: PidCase, origins: np.ndarray, *, prepared: PreparedCase | None = None) -> DynamicBatch:
        origins = np.asarray(origins, dtype=np.int64)
        if origins.ndim != 1 or np.any(origins < 0) or np.any(origins >= len(case.time)):
            raise ValueError('Invalid dynamic prediction origins')
        if prepared is not None and (prepared.case is not case or not np.array_equal(prepared.origins, origins)):
            raise ValueError('Prepared case/origin identity mismatch')
        baseline = prepared.baseline if prepared is not None else self.physics.predict(case, origins, 0.1)
        physical = prepared.physical_signed_nm if prepared is not None else physical_signed_trajectory(self.physics, case, origins)
        correction = np.full_like(physical, np.nan)
        causal_support = np.ones(len(origins), dtype=bool)
        causal_reasons = np.full(len(origins), 'ok', dtype='<U48')
        for blade in range(3):
            features, history_valid = dynamic_features(self.physics, case, origins, self.lags, blade)
            feature_support = np.zeros(len(origins), dtype=bool)
            causal_reasons[~history_valid] = 'history_invalid_or_unfitted'
            for ri in range(len(REGIONS)):
                selected = np.flatnonzero(case.region[origins] == ri)
                if not len(selected):
                    continue
                if not np.isfinite(self.coefficients[ri, blade]).all():
                    causal_reasons[selected] = 'dynamic_group_unfitted'
                    continue
                padding = np.maximum((self.maximum[ri, blade] - self.minimum[ri, blade]) *
                                     self.spec['dynamic_model']['domain_padding_fraction'],
                                     self.spec['dynamic_model']['normalized_feature_domain_floor'])
                inside = ((features[selected] >= self.minimum[ri, blade] - padding) &
                          (features[selected] <= self.maximum[ri, blade] + padding)).all(axis=1)
                feature_support[selected] = history_valid[selected] & inside
                causal_reasons[selected[history_valid[selected] & ~inside]] = 'dynamic_feature_outside_train_domain'
                correction[selected, :, blade] = features[selected] @ self.coefficients[ri, blade] * self.physics.protocol['model']['moment_scale_nm']
            causal_support &= feature_support
        dynamic = physical + correction
        steps = np.rint(np.asarray(self.physics.protocol['horizons_s']) / self.physics.protocol['step_duration_s']).astype(int)
        peaks = {'v1_greybox': baseline.peaks_nm['physical_greybox'],
                 **{name: baseline.peaks_nm[name] for name in ('physical_only', 'persistence', 'causal_trend')},
                 'dynamic_fir': np.maximum.accumulate(np.abs(dynamic), axis=1)[:, steps - 1]}
        supported = baseline.supported & causal_support[:, None]
        finite_prefix = np.logical_and.accumulate(np.isfinite(dynamic).all(axis=2), axis=1)[:, steps - 1]
        reasons = baseline.reasons.copy()
        reasons[baseline.supported & ~causal_support[:, None]] = np.repeat(causal_reasons[:, None], len(steps), axis=1)[baseline.supported & ~causal_support[:, None]]
        reasons[supported & ~finite_prefix] = 'dynamic_prediction_nonfinite'
        supported &= finite_prefix & np.isfinite(peaks['dynamic_fir']).all(axis=2)
        comparison_finite = np.logical_and.reduce([np.isfinite(value).all(axis=2) for value in peaks.values()])
        reasons[supported & ~comparison_finite] = 'comparison_prediction_nonfinite'
        supported &= comparison_finite
        return DynamicBatch(origins, peaks, supported, reasons, dynamic)

    def to_document(self) -> dict[str, Any]:
        return {
            'schema_version': DYNAMIC_MODEL_SCHEMA, 'name': self.name,
            'base_protocol_sha256': document_sha256(self.physics.protocol),
            'physics_model_sha256': document_sha256(self.physics.to_document()),
            'development_protocol_sha256': document_sha256(self.spec),
            'baseline_residual_decay_s': 0.1,
            'horizons_s': self.physics.protocol['horizons_s'],
            'external_fields': list(INPUT_FIELDS), 'feature_names': list(dynamic_feature_names(self.lags)),
            'lags_frames': list(self.lags), 'ridge_fraction': self.ridge_fraction,
            'coefficients': {
                region: [{'training_origins': int(self.counts[ri, blade]),
                          'weights': self.coefficients[ri, blade].tolist() if np.isfinite(self.coefficients[ri, blade]).all() else None,
                          'domain_minimum': self.minimum[ri, blade].tolist() if self.counts[ri, blade] else None,
                          'domain_maximum': self.maximum[ri, blade].tolist() if self.counts[ri, blade] else None,
                          'condition_number': float(self.conditioning[ri, blade]) if np.isfinite(self.conditioning[ri, blade]) else None}
                         for blade in range(3)] for ri, region in enumerate(REGIONS)
            }, 'authorization': dict(AUTHORIZATION),
        }


SCALE_FEATURE_NAMES = ('one', 'wind', 'beta', 'omega', 'log_abs_my', 'log_abs_my_rate',
                       'log_abs_my_delta1', 'log_abs_my_delta5', 'log_abs_my_delta10', 'log_abs_my_delta20',
                       'abs_acc_fa', 'abs_acc_ss', 'abs_yaw')


def scale_features(case: PidCase, origins: np.ndarray, blade: int) -> tuple[np.ndarray, np.ndarray]:
    indices, valid = lag_indices(case, origins, (1, 5, 10, 20))
    my = case.my_nm[indices, blade]
    columns = [np.ones(len(origins)), case.values['I_AverageWindSpeed'][origins] / 20.0,
               case.values['I_PitchPosition1'][origins] / 0.2, case.values['I_RotorSpeed'][origins] / 0.8,
               np.log1p(np.abs(my[:, 0]) / 1e7), np.log1p(np.abs(case.derivatives[origins, blade + 2]) / 1e7)]
    columns += [np.log1p(np.abs(my[:, 0] - my[:, li]) / 1e6) for li in range(1, 5)]
    columns += [np.abs(case.values[field][origins]) for field in (
        'I_NacelleForeAftAcceleration', 'I_NacelleSideSideAcceleration', 'I_YawError')]
    features = np.column_stack(columns)
    return features, valid & np.isfinite(features).all(axis=1)


class ErrorScaleFitter:
    def __init__(self, model: DynamicModel, method: str) -> None:
        if method not in ('v1_greybox', 'dynamic_fir'):
            raise ValueError('Only the two predeclared point models may fit error scales')
        self.model, self.method = model, method
        width = len(SCALE_FEATURE_NAMES)
        self.xtx = np.zeros((len(REGIONS), 3, width, width))
        self.xty = np.zeros((len(REGIONS), 3, width, len(model.physics.protocol['horizons_s'])))
        self.counts = np.zeros((len(REGIONS), 3), dtype=np.int64)
        self.minimum = np.full((len(REGIONS), 3, width), np.inf)
        self.maximum = np.full_like(self.minimum, -np.inf)

    def add_case(self, prepared: PreparedCase, predictions: DynamicBatch) -> None:
        case, origins = prepared.case, prepared.origins
        if case.condition_id not in self.model.physics.protocol['split']['train']:
            raise ValueError('Error scale must fit train residuals only')
        if not np.array_equal(predictions.origins, origins):
            raise ValueError('Error scale prediction/origin identity mismatch')
        eligible = (predictions.supported & prepared.labels['approximate_zoh']).all(axis=1)
        floor = self.model.spec['conditional_scale']['error_floor_nm']
        errors = np.abs(prepared.labels['peak_nm'] - predictions.peaks_nm[self.method])
        target = np.log(np.maximum(errors, floor))
        eligible &= np.isfinite(target).all(axis=(1, 2))
        for blade in range(3):
            features, valid = scale_features(case, origins, blade)
            for ri in range(len(REGIONS)):
                selected = eligible & valid & (case.region[origins] == ri)
                if not selected.any():
                    continue
                x, y = features[selected], target[selected, :, blade]
                self.xtx[ri, blade] += x.T @ x
                self.xty[ri, blade] += x.T @ y
                self.counts[ri, blade] += len(x)
                self.minimum[ri, blade] = np.minimum(self.minimum[ri, blade], x.min(axis=0))
                self.maximum[ri, blade] = np.maximum(self.maximum[ri, blade], x.max(axis=0))

    def finish(self) -> ErrorScaleModel:
        coefficients = np.full_like(self.xty, np.nan)
        for ri in range(len(REGIONS)):
            for blade in range(3):
                count = int(self.counts[ri, blade])
                if count >= len(SCALE_FEATURE_NAMES):
                    coefficients[ri, blade], _ = solve_ridge(self.xtx[ri, blade], self.xty[ri, blade], count,
                                                           self.model.spec['conditional_scale']['ridge_fraction'])
        return ErrorScaleModel(self.model, self.method, coefficients, self.counts.copy(), self.minimum.copy(), self.maximum.copy())


@dataclass
class ErrorScaleModel:
    dynamic_model: DynamicModel
    method: str
    coefficients: np.ndarray
    counts: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    def predict(self, case: PidCase, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        floor = self.dynamic_model.spec['conditional_scale']['error_floor_nm']
        ceiling = self.dynamic_model.spec['conditional_scale']['maximum_scale_nm']
        scales = np.full((len(origins), self.coefficients.shape[-1], 3), np.nan)
        support = np.ones_like(scales, dtype=bool)
        for blade in range(3):
            features, valid = scale_features(case, origins, blade)
            blade_support = np.zeros((len(origins), self.coefficients.shape[-1]), dtype=bool)
            for ri in range(len(REGIONS)):
                ids = np.flatnonzero(case.region[origins] == ri)
                if not len(ids) or not np.isfinite(self.coefficients[ri, blade]).all():
                    continue
                padding = np.maximum((self.maximum[ri, blade] - self.minimum[ri, blade]) *
                                     self.dynamic_model.spec['dynamic_model']['domain_padding_fraction'],
                                     self.dynamic_model.spec['dynamic_model']['normalized_feature_domain_floor'])
                inside = ((features[ids] >= self.minimum[ri, blade] - padding) &
                          (features[ids] <= self.maximum[ri, blade] + padding)).all(axis=1)
                logs = features[ids] @ self.coefficients[ri, blade]
                finite = np.isfinite(logs) & (logs <= math.log(ceiling))
                blade_support[ids] = (inside & valid[ids])[:, None] & finite
                scales[ids, :, blade] = np.exp(np.clip(logs, math.log(floor), math.log(ceiling)))
            support[:, :, blade] = blade_support
        known = support.all(axis=-1)
        scales[~known] = np.nan
        return scales, known

    def to_document(self) -> dict[str, Any]:
        return {'schema_version': SCALE_SCHEMA, 'point_method': self.method,
                'dynamic_model_sha256': document_sha256(self.dynamic_model.to_document()),
                'feature_names': list(SCALE_FEATURE_NAMES),
                'parameters': dict(self.dynamic_model.spec['conditional_scale']),
                'regions': {region: [{'training_origins': int(self.counts[ri, blade]),
                                     'weights': self.coefficients[ri, blade].tolist() if np.isfinite(self.coefficients[ri, blade]).all() else None,
                                     'domain_minimum': self.minimum[ri, blade].tolist() if self.counts[ri, blade] else None,
                                     'domain_maximum': self.maximum[ri, blade].tolist() if self.counts[ri, blade] else None}
                                    for blade in range(3)] for ri, region in enumerate(REGIONS)},
                'qualification': self.dynamic_model.spec['conditional_scale']['qualification'],
                'authorization': dict(AUTHORIZATION)}


def make_joint_calibration(records: list[dict[str, np.ndarray]], model: DynamicModel,
                           scale_model: ErrorScaleModel) -> dict[str, Any]:
    """Only calibration cases may contribute; max over 3 blades x 2 horizons."""
    p, spec = model.physics.protocol, model.spec
    primary = [p['horizons_s'].index(h) for h in spec['joint_margin']['scope_horizons_s']]
    groups: dict[str, Any] = {}
    for ri, name in enumerate(REGIONS):
        residuals, normalized, counts = [], [], {}
        for entry in records:
            cid = int(entry['condition_id'])
            if cid not in p['split']['calibration']:
                raise ValueError('Joint margins can use calibration cases only')
            selected = entry['eligible'] & (entry['region'] == ri)
            selected &= np.isfinite(entry['actual'][:, primary]).all(axis=(1, 2))
            selected &= np.isfinite(entry['predicted'][:, primary]).all(axis=(1, 2))
            selected &= np.isfinite(entry['scale'][:, primary]).all(axis=(1, 2)) & (entry['scale'][:, primary] > 0.0).all(axis=(1, 2))
            if not selected.any():
                continue
            under = np.maximum(entry['actual'][selected][:, primary] - entry['predicted'][selected][:, primary], 0.0)
            scales = entry['scale'][selected][:, primary]
            residuals.append(under)
            normalized.append(under / scales)
            counts[str(cid)] = int(selected.sum())
        raw = np.concatenate(residuals) if residuals else np.empty((0, len(primary), 3))
        scaled = np.concatenate(normalized) if normalized else np.empty_like(raw)
        supported = len(raw) >= spec['joint_margin']['minimum_calibration_origins']
        q = spec['joint_margin']['quantile']
        groups[name] = {
            'origin_count': len(raw), 'case_origin_counts': counts, 'supported': supported,
            'legacy_per_blade95_nm': np.quantile(raw, q, axis=0, method='higher').tolist() if supported else None,
            'joint_constant95_nm': float(np.quantile(raw.max(axis=(1, 2)), q, method='higher')) if supported else None,
            'joint_conditional95': float(np.quantile(scaled.max(axis=(1, 2)), q, method='higher')) if supported else None,
        }
    return {'schema_version': 'joint_load_margin_development_calibration_v2', 'point_method': scale_model.method,
            'dynamic_model_sha256': document_sha256(model.to_document()),
            'scale_model_sha256': document_sha256(scale_model.to_document()),
            'primary_horizons_s': spec['joint_margin']['scope_horizons_s'],
            'groups': groups, 'qualification': spec['joint_margin']['qualification'],
            'authorization': dict(AUTHORIZATION)}


def joint_margin_arrays(calibration: dict[str, Any], scale_model: ErrorScaleModel,
                        case: PidCase, origins: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if (calibration['dynamic_model_sha256'] != document_sha256(scale_model.dynamic_model.to_document()) or
            calibration['scale_model_sha256'] != document_sha256(scale_model.to_document()) or
            calibration['point_method'] != scale_model.method or calibration['authorization'] != AUTHORIZATION):
        raise ValueError('Joint margin/model identity or offline authorization mismatch')
    p = scale_model.dynamic_model.physics.protocol
    primary = [p['horizons_s'].index(h) for h in calibration['primary_horizons_s']]
    scales, scale_supported = scale_model.predict(case, origins)
    known = scale_supported[:, primary].all(axis=1)
    arrays = {method: np.full((len(origins), len(primary), 3), np.nan) for method in MARGIN_METHODS}
    for ri, name in enumerate(REGIONS):
        group = calibration['groups'][name]
        selected = case.region[origins] == ri
        if not group['supported']:
            known[selected] = False
            continue
        legacy = np.asarray(group['legacy_per_blade95_nm'], dtype=np.float64)
        constants = np.asarray([group['joint_constant95_nm'], group['joint_conditional95']], dtype=np.float64)
        if (legacy.shape != (len(primary), 3) or not np.isfinite(legacy).all() or np.any(legacy < 0.0) or
                not np.isfinite(constants).all() or np.any(constants < 0.0)):
            known[selected] = False
            continue
        valid = selected & known
        arrays['legacy_per_blade95_nm'][valid] = legacy
        arrays['joint_constant95_nm'][valid] = group['joint_constant95_nm']
        arrays['joint_conditional95'][valid] = scales[valid][:, primary] * group['joint_conditional95']
    known &= case.region[origins] >= 0
    for values in arrays.values():
        known &= np.isfinite(values).all(axis=(1, 2)) & (values >= 0.0).all(axis=(1, 2))
    for values in arrays.values():
        values[~known] = np.nan
    return arrays, known