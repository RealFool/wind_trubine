"""Offline-only, non-NN load-margin feasibility model using existing signals.

This is a lumped greybox, not BEM, a full aeroelastic model, or a certified
upper bound.  Nothing in this module sends actions or modifies the Actor.
"""
from __future__ import annotations

from array import array
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from RL.utils.constants import (
    ACTION_VARIABLES,
    DISTURBANCE_VARIABLES,
    REWARD_VARIABLES,
    STATE_VARIABLES,
)
from RL.utils.operating_state import (
    ControlRegion,
    OperatingPoint,
    OperatingStateClassifier,
    OptimizationState,
)
from RL.utils.raw_dataset_contract import EXPECTED_CONDITION_IDS, TIME_TOLERANCE_S, sha256_file


PROTOCOL_SCHEMA = 'physical_load_margin_demo_protocol_v1'
MODEL_SCHEMA = 'physical_load_margin_greybox_v1'
LOAD_FIELDS = tuple(REWARD_VARIABLES[1:])
INPUT_FIELDS = tuple(dict.fromkeys((
    'Time', *STATE_VARIABLES, *DISTURBANCE_VARIABLES, *REWARD_VARIABLES, *ACTION_VARIABLES,
)))
REGIONS = tuple(region.value for region in ControlRegion)
METHODS = ('physical_greybox', 'physical_only', 'persistence', 'causal_trend')
POLYNOMIAL_NAMES = ('1', 'phi', 'beta', 'phi2', 'phi_beta', 'beta2', 'phi3', 'phi2_beta', 'phi_beta2', 'beta3')
SURFACE_FEATURE_NAMES = (
    *(f'q_relative_{name}' for name in POLYNOMIAL_NAMES),
    'q_axial_sin_psi', 'q_axial_cos_psi', 'q_axial_sin_2psi', 'q_axial_cos_2psi',
    'acc_fa', 'acc_fa_cos_psi', 'acc_ss_sin_psi', 'acc_ss_cos_psi',
)
DOMAIN_NAMES = ('wind', 'omega', 'beta', 'yaw', 'acc_fa', 'acc_ss', 'beta_rate', 'omega_rate')
DOMAIN_FLOORS = np.asarray([1.0, 0.05, 0.01, 0.05, 0.1, 0.1, 0.01, 0.005], dtype=np.float64)


def document_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: Path | str) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('schema_version') != PROTOCOL_SCHEMA:
        raise ValueError('Incompatible physical demo protocol')
    partitions = protocol['split']
    if set(partitions) != {'train', 'validation', 'calibration', 'audit'}:
        raise ValueError('Four explicit whole-case partitions are required')
    flattened = [cid for values in partitions.values() for cid in values]
    if len(set(flattened)) != len(flattened) or sorted(flattened) != list(EXPECTED_CONDITION_IDS):
        raise ValueError('Splits must be disjoint and cover wind3-wind25 exactly')
    if any(not values for values in partitions.values()):
        raise ValueError('Empty partitions are not permitted')
    dt = float(protocol['step_duration_s'])
    horizons = np.asarray(protocol['horizons_s'], dtype=np.float64)
    window = np.asarray(protocol['reference_window_s'], dtype=np.float64)
    if not math.isfinite(dt) or dt <= 0.0 or not np.isfinite(horizons).all():
        raise ValueError('Step and horizons must be finite and positive')
    if np.any(horizons <= 0.0) or np.any(np.diff(horizons) <= 0.0):
        raise ValueError('Horizons must be strictly increasing')
    if not np.allclose(horizons / dt, np.rint(horizons / dt), atol=1e-8, rtol=0.0):
        raise ValueError('Each horizon must be an exact number of frames')
    if window.shape != (2,) or not np.isfinite(window).all() or window[0] < 0.0 or window[1] - window[0] <= horizons[-1]:
        raise ValueError('Invalid reference window')
    if any(h not in protocol['horizons_s'] for h in protocol['online_scope_horizons_s']):
        raise ValueError('Online scope must be a horizon subset')
    for key in ('origin_stride_frames', 'fit_stride_frames'):
        if int(protocol[key]) != protocol[key] or int(protocol[key]) < 1:
            raise ValueError(f'{key} must be a positive integer')
    positive = (
        protocol['warmup_s'], protocol['model']['moment_scale_nm'], protocol['model']['ridge_fraction'],
        protocol['model']['derivative_filter_tau_s'], protocol['model']['baseline_trend_decay_s'],
        *protocol['model']['residual_decay_candidates_s'],
    )
    params = protocol['static_parameters']
    positive += tuple(params[key] for key in (
        'air_density_kg_m3', 'rotor_diameter_m', 'effective_radius_fraction', 'rotor_inertia_kg_m2',
        'pitch_natural_frequency_rad_s', 'generator_time_constant_s', 'generator_actual_torque_max_nm',
        'request_torque_max_nm',
    ))
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
        raise ValueError('Model constants must be finite and positive')
    if not 0.0 < params['pitch_damping_ratio'] < 1.0 or not 0.0 < params['effective_radius_fraction'] < 1.0:
        raise ValueError('Demo requires underdamped position response and an interior effective radius')
    if params['request_pitch_bounds_rad'] != [0.0, 0.43]:
        raise ValueError('Demo must retain existing pitch request bounds')
    if params['generator_actual_torque_max_nm'] > params['request_torque_max_nm']:
        raise ValueError('Actual torque cap must not exceed request cap')
    if not 0.0 < protocol['evaluation']['empirical_error_quantile'] < 1.0:
        raise ValueError('Invalid empirical quantile')
    if protocol['authorization'] != {
        'mode': 'offline_demo', 'policy_acceptance': False, 'soft_authorized': False,
        'hard_authorized': False, 'counterfactual_validation': False,
    }:
        raise ValueError('This demo never authorizes control or counterfactual claims')
    return protocol


def verify_static_project(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    text = '\n'.join(root.itertext())
    params = protocol['static_parameters']

    def text_number(pattern: str) -> float:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            raise ValueError(f'Missing engineering parameter: {pattern}')
        return float(match.group(1))

    def xml_number(tag: str) -> float:
        value = root.findtext(tag)
        if value is None:
            raise ValueError(f'Missing engineering parameter: {tag}')
        return float(value)

    actual = {
        'air_density_kg_m3': text_number(r'^RHO\s+(\S+)\s*$'),
        'rotor_diameter_m': text_number(r'^DIAM\s+(\S+)\s*$'),
        'rotor_inertia_kg_m2': text_number(r'^\s*([\d.eE+-]+)\s+!\s*WE_Jtot\b'),
        'pitch_natural_frequency_rad_s': xml_number('PitchActuator/PositionResponse/Frequency'),
        'pitch_damping_ratio': xml_number('PitchActuator/PositionResponse/Damping'),
        'generator_time_constant_s': xml_number('.//generator/timeConst'),
        'generator_actual_torque_max_nm': xml_number('.//generator/torqueDemandMax'),
    }
    if root.findtext('PitchActuator/DemandType') != 'Position':
        raise ValueError('Only the actual PositionResponse applies to this demo')
    if text_number(r'^GRATIO\s+(\S+)\s*$') != 1.0:
        raise ValueError('The current rotor torque equation requires the existing direct-drive ratio')
    for key, value in actual.items():
        if not math.isclose(value, params[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'Protocol/project parameter mismatch for {key}')
    return {
        'sha256': sha256_file(path), 'verified_numeric_parameters': actual,
        'pitch_response': 'Position second order only; RateResponse.LagTime is not stacked',
        'qualification': params['parameter_qualification'],
        'ct_cp_table_loaded': False,
    }


@dataclass
class PidCase:
    condition_id: int
    values: dict[str, np.ndarray]
    actions: np.ndarray
    my_nm: np.ndarray
    derivatives: np.ndarray
    region: np.ndarray
    segment: np.ndarray
    warm: np.ndarray
    valid: np.ndarray
    pid_peak_nm: np.ndarray
    cumulative_peak_nm: np.ndarray
    audit: dict[str, Any]

    @property
    def time(self) -> np.ndarray:
        return self.values['Time']


def read_pid_case(path: Path, condition_id: int, protocol: dict[str, Any]) -> PidCase:
    buffer = array('d')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError(f'{path.name}: duplicate input columns')
        missing = set(INPUT_FIELDS) - set(header)
        if missing:
            raise ValueError(f'{path.name}: missing existing fields {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            try:
                buffer.extend(float(row[field]) for field in INPUT_FIELDS)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path.name}: invalid numeric row {row_number}') from exc
    matrix = np.frombuffer(buffer, dtype=np.float64).reshape(-1, len(INPUT_FIELDS)).copy()
    if len(matrix) < 2:
        raise ValueError(f'{path.name}: not enough rows')
    values = {field: matrix[:, index] for index, field in enumerate(INPUT_FIELDS)}
    times = values['Time']
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError(f'{path.name}: timestamps must be finite and strictly increasing')
    if not np.allclose(times[[0, -1]], protocol['reference_window_s'], rtol=0.0, atol=TIME_TOLERANCE_S):
        raise ValueError(f'{path.name}: exact PID reference window is required')
    my = np.column_stack([values[field] for field in LOAD_FIELDS])
    if not np.isfinite(my).all():
        raise ValueError(f'{path.name}: incomplete load reference cannot define a PID maximum')
    pid_peak = np.max(np.abs(my), axis=0)
    if np.any(pid_peak <= 0.0):
        raise ValueError(f'{path.name}: PID maxima must be positive')
    params = protocol['static_parameters']
    raw_actions = np.column_stack([values[field] for field in ACTION_VARIABLES])
    actions = np.clip(raw_actions, [0.0, 0.0], [params['request_torque_max_nm'], params['request_pitch_bounds_rad'][1]])
    valid = np.isfinite(matrix).all(axis=1)
    dt = protocol['step_duration_s']
    gaps = np.r_[False, ~np.isclose(np.diff(times), dt, atol=TIME_TOLERANCE_S, rtol=0.0)]
    if gaps.any():
        raise ValueError(f'{path.name}: incomplete 20 ms trace cannot qualify as a full-window PID reference')
    region = np.full(len(times), -1, dtype=np.int8)
    segment = np.full(len(times), -1, dtype=np.int32)
    warm = np.zeros(len(times), dtype=bool)
    motion = np.column_stack((values['I_RotorSpeed'], values['I_PitchPosition1'], my))
    derivatives = np.zeros_like(motion)
    classifier = OperatingStateClassifier()
    segment_id = -1
    start_time = float(times[0])
    derivative_alpha = 1.0 - math.exp(-dt / protocol['model']['derivative_filter_tau_s'])
    for index, timestamp in enumerate(times):
        if not valid[index]:
            classifier.reset()
            continue
        reset = index == 0 or gaps[index] or not valid[index - 1]
        if reset:
            segment_id += 1
            start_time = float(timestamp)
            classifier.reset()
        else:
            derivatives[index] = derivatives[index - 1] + derivative_alpha * (
                (motion[index] - motion[index - 1]) / dt - derivatives[index - 1]
            )
        segment[index] = segment_id
        warm[index] = timestamp - start_time >= protocol['warmup_s'] - TIME_TOLERANCE_S
        previous_action = actions[index if reset else index - 1]
        classification = classifier.classify(OperatingPoint(
            timestamp_s=float(timestamp), wind_speed_mps=float(values['I_AverageWindSpeed'][index]),
            rotor_speed_rad_s=float(values['I_RotorSpeed'][index]), real_power_w=float(values['I_RealPower'][index]),
            pitch_rad=float(values['I_PitchPosition1'][index]),
            previous_torque_nm=float(previous_action[0]), previous_pitch_rad=float(previous_action[1]),
        ))
        if classification.optimization_state not in (
            OptimizationState.STARTUP_STOPPED, OptimizationState.NEGATIVE_POWER_ABNORMAL,
        ):
            region[index] = REGIONS.index(classification.control_region.value)
    powers = values['Electrical power']
    audit = {
        'condition_id': condition_id, 'filename': path.name, 'sha256': sha256_file(path),
        'rows': len(times), 'time_window_s': times[[0, -1]].tolist(), 'gap_count': int(gaps.sum()),
        'invalid_input_rows': int((~valid).sum()), 'ignored_columns': sorted(set(header) - set(INPUT_FIELDS)),
        'used_fields': list(INPUT_FIELDS), 'pid_peak_abs_nm': pid_peak.tolist(),
        'pid_peak_times_s': times[np.argmax(np.abs(my), axis=0)].tolist(),
        'electrical_energy_j': float(np.trapz(powers, times)) if np.isfinite(powers).all() else None,
        'clipped_request_rows': int(np.any(actions != raw_actions, axis=1).sum()),
        'region_rows': {name: int((region == i).sum()) for i, name in enumerate(REGIONS)},
        'unsupported_state_rows': int((region < 0).sum()),
        'field_ranges': {
            field: [float(np.min(column[np.isfinite(column)])), float(np.max(column[np.isfinite(column)]))]
            if np.isfinite(column).any() else None for field, column in values.items()
        },
    }
    return PidCase(condition_id, values, actions, my, derivatives, region, segment, warm, valid,
                   pid_peak, np.maximum.accumulate(np.abs(my), axis=0), audit)


def surface_features(
    wind: np.ndarray, omega: np.ndarray, beta: np.ndarray, yaw: np.ndarray,
    azimuth: np.ndarray, acc_fa: np.ndarray, acc_ss: np.ndarray,
    blade_index: int, params: dict[str, Any], moment_scale_nm: float,
) -> np.ndarray:
    radius = params['rotor_diameter_m'] / 2.0
    effective_radius = radius * params['effective_radius_fraction']
    axial = wind * np.cos(yaw)
    tangential = omega * effective_radius
    phi = np.arctan2(axial, tangential) / 0.25
    pitch = beta / 0.2
    moment_factor = 0.5 * params['air_density_kg_m3'] * math.pi * radius ** 2 * effective_radius / 3.0 / moment_scale_nm
    q_relative = moment_factor * (axial ** 2 + tangential ** 2)
    q_axial = moment_factor * axial ** 2
    psi = azimuth + blade_index * (2.0 * math.pi / 3.0)
    polynomial = (np.ones_like(phi), phi, pitch, phi ** 2, phi * pitch, pitch ** 2,
                  phi ** 3, phi ** 2 * pitch, phi * pitch ** 2, pitch ** 3)
    return np.stack((
        *(q_relative * term for term in polynomial),
        q_axial * np.sin(psi), q_axial * np.cos(psi), q_axial * np.sin(2.0 * psi), q_axial * np.cos(2.0 * psi),
        acc_fa, acc_fa * np.cos(psi), acc_ss * np.sin(psi), acc_ss * np.cos(psi),
    ), axis=-1)


def observation_inputs(case: PidCase, origins: np.ndarray) -> tuple[np.ndarray, ...]:
    fields = ('I_AverageWindSpeed', 'I_RotorSpeed', 'I_PitchPosition1', 'I_YawError',
              'I_RotorAzimuth', 'I_NacelleForeAftAcceleration', 'I_NacelleSideSideAcceleration')
    return tuple(case.values[field][origins] for field in fields)


def domain_inputs(case: PidCase, origins: np.ndarray) -> np.ndarray:
    wind, omega, beta, yaw, _, acc_fa, acc_ss = observation_inputs(case, origins)
    return np.column_stack((wind, omega, beta, yaw, acc_fa, acc_ss,
                            case.derivatives[origins, 1], case.derivatives[origins, 0]))


class GreyboxFitter:
    """Bounded sufficient statistics; no train rows retained across cases."""

    def __init__(self, protocol: dict[str, Any]) -> None:
        self.protocol = protocol
        width = len(SURFACE_FEATURE_NAMES)
        self.xtx = np.zeros((len(REGIONS), 3, width, width), dtype=np.float64)
        self.xty = np.zeros((len(REGIONS), 3, width), dtype=np.float64)
        self.counts = np.zeros(len(REGIONS), dtype=np.int64)
        self.minimum = np.full((len(REGIONS), len(DOMAIN_NAMES)), np.inf)
        self.maximum = np.full_like(self.minimum, -np.inf)

    def add_case(self, case: PidCase) -> None:
        if case.condition_id not in self.protocol['split']['train']:
            raise ValueError('Only frozen train cases may enter coefficient fitting')
        for region_index in range(len(REGIONS)):
            indices = np.arange(0, len(case.time), self.protocol['fit_stride_frames'])
            eligible = case.valid[indices] & case.warm[indices] & (case.region[indices] == region_index)
            eligible &= case.values['I_AverageWindSpeed'][indices] * np.cos(case.values['I_YawError'][indices]) > 0.0
            indices = indices[eligible]
            if not len(indices):
                continue
            self.counts[region_index] += len(indices)
            domain = domain_inputs(case, indices)
            self.minimum[region_index] = np.minimum(self.minimum[region_index], domain.min(axis=0))
            self.maximum[region_index] = np.maximum(self.maximum[region_index], domain.max(axis=0))
            for blade in range(3):
                design = surface_features(*observation_inputs(case, indices), blade,
                                          self.protocol['static_parameters'], self.protocol['model']['moment_scale_nm'])
                labels = case.my_nm[indices, blade] / self.protocol['model']['moment_scale_nm']
                self.xtx[region_index, blade] += design.T @ design
                self.xty[region_index, blade] += design.T @ labels

    def finish(self) -> GreyboxModel:
        coefficients = np.full_like(self.xty, np.nan)
        conditioning: dict[str, list[float] | None] = {}
        for region_index, count in enumerate(self.counts):
            if count < len(SURFACE_FEATURE_NAMES):
                conditioning[REGIONS[region_index]] = None
                continue
            conditioning[REGIONS[region_index]] = []
            for blade in range(3):
                rms = np.maximum(np.sqrt(np.diag(self.xtx[region_index, blade]) / count), 1e-6)
                gram = self.xtx[region_index, blade] / np.outer(rms, rms)
                gram += np.eye(len(rms)) * self.protocol['model']['ridge_fraction'] * count
                normalized_coefficients = np.linalg.solve(gram, self.xty[region_index, blade] / rms)
                coefficients[region_index, blade] = normalized_coefficients / rms * self.protocol['model']['moment_scale_nm']
                conditioning[REGIONS[region_index]].append(float(np.linalg.cond(gram)))
        return GreyboxModel(self.protocol, coefficients, self.counts.copy(), self.minimum.copy(), self.maximum.copy(), conditioning)


@dataclass
class PredictionBatch:
    origins: np.ndarray
    peaks_nm: dict[str, np.ndarray]
    supported: np.ndarray
    reasons: np.ndarray
    actual_response_prediction: np.ndarray
    static_current_my_nm: np.ndarray


@dataclass
class GreyboxModel:
    protocol: dict[str, Any]
    coefficients: np.ndarray
    training_counts: np.ndarray
    domain_minimum: np.ndarray
    domain_maximum: np.ndarray
    conditioning: dict[str, list[float] | None]

    def domain_bounds(self, region: int) -> tuple[np.ndarray, np.ndarray]:
        padding = np.maximum((self.domain_maximum[region] - self.domain_minimum[region]) *
                             self.protocol['model']['domain_padding_fraction'], DOMAIN_FLOORS)
        return self.domain_minimum[region] - padding, self.domain_maximum[region] + padding

    def to_document(self) -> dict[str, Any]:
        return {
            'schema_version': MODEL_SCHEMA, 'protocol_sha256': document_sha256(self.protocol),
            'used_external_fields': list(INPUT_FIELDS), 'surface_feature_names': list(SURFACE_FEATURE_NAMES),
            'domain_names': list(DOMAIN_NAMES), 'moment_unit': 'Nm',
            'fit': {
                name: {
                    'training_rows': int(self.training_counts[index]),
                    'coefficients_nm': self.coefficients[index].tolist() if np.isfinite(self.coefficients[index]).all() else None,
                    'domain_minimum': self.domain_minimum[index].tolist() if self.training_counts[index] else None,
                    'domain_maximum': self.domain_maximum[index].tolist() if self.training_counts[index] else None,
                    'regularized_condition_numbers': self.conditioning[name],
                } for index, name in enumerate(REGIONS)
            },
            'authorization': dict(self.protocol['authorization']),
        }

    def predict(self, case: PidCase, origins: np.ndarray, residual_decay_s: float,
                candidate_actions: np.ndarray | None = None) -> PredictionBatch:
        """Read observations only at/before each origin; never access future rows."""
        origins = np.asarray(origins, dtype=np.int64)
        if origins.ndim != 1 or np.any(origins < 0) or np.any(origins >= len(case.time)):
            raise ValueError('Invalid prediction origins')
        if not math.isfinite(residual_decay_s) or residual_decay_s <= 0.0:
            raise ValueError('Invalid residual decay')
        protocol = self.protocol
        params = protocol['static_parameters']
        steps = np.rint(np.asarray(protocol['horizons_s']) / protocol['step_duration_s']).astype(int)
        h = np.arange(1, steps[-1] + 1, dtype=np.float64)[None, :] * protocol['step_duration_s']
        candidate = case.actions[origins].copy() if candidate_actions is None else np.asarray(candidate_actions, dtype=np.float64).copy()
        limits = np.asarray([params['request_torque_max_nm'], params['request_pitch_bounds_rad'][1]])
        if candidate.shape != (len(origins), 2):
            raise ValueError('Candidate action shape must be [origins,2]')
        if candidate_actions is not None and (not np.isfinite(candidate).all() or np.any(candidate < 0.0) or np.any(candidate > limits)):
            raise ValueError('Candidate must be a finite action inside existing engineering bounds')
        if candidate_actions is None:
            candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        wind, omega, beta, yaw, azimuth, acc_fa, acc_ss = observation_inputs(case, origins)
        derivative = case.derivatives[origins]
        torque = case.values['I_ElectricalTorque'][origins, None]
        target_torque = np.minimum(candidate[:, 0, None], params['generator_actual_torque_max_nm'])
        delta_torque = target_torque - torque
        tau = params['generator_time_constant_s']
        exponent = np.exp(-h / tau)
        future_torque = target_torque - delta_torque * exponent
        future_omega_rate = derivative[:, 0, None] - delta_torque / params['rotor_inertia_kg_m2'] * (1.0 - exponent)
        future_omega = omega[:, None] + derivative[:, 0, None] * h - delta_torque / params['rotor_inertia_kg_m2'] * (h - tau * (1.0 - exponent))
        future_azimuth = azimuth[:, None] + omega[:, None] * h + 0.5 * derivative[:, 0, None] * h ** 2 - delta_torque / params['rotor_inertia_kg_m2'] * (
            0.5 * h ** 2 - tau * h + tau ** 2 * (1.0 - exponent)
        )
        natural = params['pitch_natural_frequency_rad_s']
        decay = params['pitch_damping_ratio'] * natural
        damped = natural * math.sqrt(1.0 - params['pitch_damping_ratio'] ** 2)
        error = beta[:, None] - candidate[:, 1, None]
        sine_coefficient = (derivative[:, 1, None] + decay * error) / damped
        future_pitch = candidate[:, 1, None] + np.exp(-decay * h) * (
            error * np.cos(damped * h) + sine_coefficient * np.sin(damped * h)
        )
        future_pitch_rate = np.exp(-decay * h) * (
            (-decay * error + damped * sine_coefficient) * np.cos(damped * h)
            + (-decay * sine_coefficient - damped * error) * np.sin(damped * h)
        )
        future_inputs = (
            np.broadcast_to(wind[:, None], future_omega.shape), future_omega, future_pitch,
            np.broadcast_to(yaw[:, None], future_omega.shape), future_azimuth,
            np.broadcast_to(acc_fa[:, None], future_omega.shape), np.broadcast_to(acc_ss[:, None], future_omega.shape),
        )
        current_surface = np.full((len(origins), 3), np.nan)
        future_surface = np.full((len(origins), steps[-1], 3), np.nan)
        surface_motion_rate = np.full_like(current_surface, np.nan)
        supported_steps = np.zeros((len(origins), steps[-1]), dtype=bool)
        reason = np.full(len(origins), 'unsupported_state', dtype='<U40')
        reason[~case.valid[origins]] = 'invalid_input'
        reason[case.valid[origins] & ~case.warm[origins]] = 'warmup_or_discontinuity'
        delta = protocol['step_duration_s']
        motion_inputs = (wind, omega + derivative[:, 0] * delta, beta + derivative[:, 1] * delta,
                         yaw, azimuth + omega * delta, acc_fa, acc_ss)
        domain = domain_inputs(case, origins)
        future_domain = np.stack((future_inputs[0], future_omega, future_pitch, future_inputs[3],
                                 future_inputs[5], future_inputs[6], future_pitch_rate, future_omega_rate), axis=-1)
        for region_index in range(len(REGIONS)):
            selected = np.flatnonzero(case.region[origins] == region_index)
            if not len(selected):
                continue
            if not np.isfinite(self.coefficients[region_index]).all():
                reason[selected] = 'unfitted_region'
                continue
            lower, upper = self.domain_bounds(region_index)
            current_in_domain = np.all((domain[selected] >= lower) & (domain[selected] <= upper), axis=1)
            future_in_domain = np.all((future_domain[selected] >= lower) & (future_domain[selected] <= upper), axis=-1)
            causal_support = case.valid[origins[selected]] & case.warm[origins[selected]] & current_in_domain
            causal_support &= wind[selected] * np.cos(yaw[selected]) > 0.0
            supported_steps[selected] = causal_support[:, None] & future_in_domain & (future_omega[selected] > 0.0)
            active = case.valid[origins[selected]] & case.warm[origins[selected]]
            reason[selected[active]] = np.where(current_in_domain[active], 'ok', 'outside_training_domain')
            for blade in range(3):
                coefficient = self.coefficients[region_index, blade]
                current_surface[selected, blade] = surface_features(
                    *(part[selected] for part in (wind, omega, beta, yaw, azimuth, acc_fa, acc_ss)),
                    blade, params, protocol['model']['moment_scale_nm'],
                ) @ coefficient
                future_surface[selected, :, blade] = np.einsum('nkp,p->nk', surface_features(
                    *(part[selected] for part in future_inputs), blade, params, protocol['model']['moment_scale_nm'],
                ), coefficient)
                next_surface = surface_features(*(part[selected] for part in motion_inputs), blade,
                                                params, protocol['model']['moment_scale_nm']) @ coefficient
                surface_motion_rate[selected, blade] = (next_surface - current_surface[selected, blade]) / delta
        cumulative_support = np.logical_and.accumulate(supported_steps, axis=1)
        supported = cumulative_support[:, steps - 1]
        reasons = np.repeat(reason[:, None], len(steps), axis=1)
        reasons[(reasons == 'ok') & ~supported] = 'future_rollout_outside_domain'
        measured = case.my_nm[origins, None, :]
        physical = measured + future_surface - current_surface[:, None, :]
        residual_rate = derivative[:, 2:] - surface_motion_rate
        greybox = physical + residual_decay_s * (1.0 - np.exp(-h / residual_decay_s))[:, :, None] * residual_rate[:, None, :]
        trend_decay = protocol['model']['baseline_trend_decay_s']
        trend = measured + trend_decay * (1.0 - np.exp(-h / trend_decay))[:, :, None] * derivative[:, None, 2:]
        trajectories = {'physical_greybox': greybox, 'physical_only': physical,
                        'persistence': np.broadcast_to(measured, greybox.shape), 'causal_trend': trend}
        peaks = {
            name: np.maximum.accumulate(np.abs(trajectory), axis=1)[:, steps - 1, :]
            for name, trajectory in trajectories.items()
        }
        response = np.stack((future_torque[:, steps - 1], future_pitch[:, steps - 1]), axis=-1)
        return PredictionBatch(origins, peaks, supported, reasons, response, current_surface)


def future_labels(case: PidCase, origins: np.ndarray, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    """Evaluation-only future reads, deliberately separate from predict()."""
    origins = np.asarray(origins, dtype=np.int64)
    steps = np.rint(np.asarray(protocol['horizons_s']) / protocol['step_duration_s']).astype(int)
    future_indices = origins[:, None] + np.arange(1, steps[-1] + 1)[None, :]
    in_bounds = future_indices < len(case.time)
    clipped = np.minimum(future_indices, len(case.time) - 1)
    contiguous = in_bounds & case.valid[clipped] & (case.segment[clipped] == case.segment[origins, None])
    valid = np.logical_and.accumulate(contiguous, axis=1)[:, steps - 1]
    actual_peak = np.maximum.accumulate(np.abs(case.my_nm[clipped]), axis=1)[:, steps - 1]
    endpoint = clipped[:, steps - 1]
    actual_response = np.stack((case.values['I_ElectricalTorque'][endpoint], case.values['I_PitchPosition1'][endpoint]), axis=-1)
    bounds = np.asarray([protocol['static_parameters']['request_torque_max_nm'], protocol['static_parameters']['request_pitch_bounds_rad'][1]])
    mismatch = np.max(2.0 * np.abs(case.actions[clipped] - case.actions[origins, None, :]) / bounds, axis=-1)
    mismatch = np.maximum.accumulate(mismatch, axis=1)[:, steps - 1]
    return {'valid': valid, 'peak_nm': actual_peak, 'response': actual_response, 'plan_mismatch': mismatch,
            'approximate_zoh': valid & (mismatch <= protocol['evaluation']['approximate_zoh_max_normalized_action_linf'])}


def margin_status(pid_peak_nm: np.ndarray, cumulative_peak_nm: np.ndarray,
                  predicted_peak_nm: np.ndarray, empirical_error_nm: np.ndarray | None,
                  supported: bool) -> tuple[str, np.ndarray | None]:
    reference = np.asarray(pid_peak_nm, dtype=np.float64)
    cumulative = np.asarray(cumulative_peak_nm, dtype=np.float64)
    if reference.shape != (3,) or cumulative.shape != (3,) or not np.isfinite(reference).all() or np.any(reference <= 0.0) or not np.isfinite(cumulative).all():
        return 'unknown_reference', None
    if np.any(cumulative > reference):
        return 'observed_exceeded_latched', reference - cumulative
    if not supported or empirical_error_nm is None:
        return 'unknown_model_or_margin', None
    peak = np.asarray(predicted_peak_nm, dtype=np.float64)
    error = np.asarray(empirical_error_nm, dtype=np.float64)
    if peak.shape != (3,) or error.shape != (3,) or not np.isfinite(peak).all() or not np.isfinite(error).all() or np.any(error < 0.0):
        return 'unknown_model_or_margin', None
    margin = reference - np.maximum(cumulative, peak + error)
    return ('model_upper_crosses_reference' if np.any(margin < 0.0) else 'model_within_reference_no_guarantee'), margin