from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from RL.utils.constants import ACTION_DIM, ACTION_VARIABLES, DATA_SCHEMA_VERSION, OBSERVATION_VARIABLES
from RL.utils.operating_state import (
    LEGACY_OPERATING_STATE_CONTRACT_VERSION,
    OPERATING_STATE_CONTRACT_VERSION,
    ControlRegion,
    OperatingPoint,
    OperatingStateClassifier,
    OperatingStateConfig,
)


REGION_AWARE_BASELINE_SCHEMA_VERSION = 'region_aware_joint_baseline_v2'
REGION_AWARE_BASELINE_FEATURES = (
    'filtered_wind_speed_mps',
    'rotor_speed_rad_s',
    'generator_speed_rad_s',
    'real_power_w',
    'previous_torque_nm',
    'previous_pitch_rad',
)
_REGION_ORDER = (
    ControlRegion.BELOW_RATED,
    ControlRegion.TRANSITION,
    ControlRegion.ABOVE_RATED,
)


@dataclass(frozen=True)
class _JointRegionModel:
    relation: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: np.ndarray
    coefficient_matrix: np.ndarray
    action_delta_low: np.ndarray
    action_delta_high: np.ndarray
    joint_residual_covariance: np.ndarray
    sample_count: int

    @classmethod
    def from_document(cls, value: Any, region: ControlRegion) -> _JointRegionModel:
        if not isinstance(value, dict):
            raise ValueError(f'Region-aware baseline model {region.value} must be an object')
        relation = str(value.get('relation', ''))
        if relation not in {'joint_next_action', 'joint_next_action_delta'}:
            raise ValueError(
                f'Region-aware baseline model {region.value} must use a supported joint action relation'
            )
        sample_count = int(value.get('sample_count', 0))
        if sample_count <= 0:
            raise ValueError(f'Region-aware baseline model {region.value} needs positive sample_count')
        feature_mean = cls._array(value.get('feature_mean'), (len(REGION_AWARE_BASELINE_FEATURES),), 'feature_mean')
        feature_scale = cls._array(value.get('feature_scale'), (len(REGION_AWARE_BASELINE_FEATURES),), 'feature_scale')
        if np.any(feature_scale <= 0.0):
            raise ValueError(f'Region-aware baseline model {region.value} feature_scale must be positive')
        intercept = cls._array(value.get('intercept'), (ACTION_DIM,), 'intercept')
        coefficient_matrix = cls._array(
            value.get('coefficient_matrix'),
            (len(REGION_AWARE_BASELINE_FEATURES), ACTION_DIM),
            'coefficient_matrix',
        )
        action_delta_low = cls._array(value.get('action_delta_low'), (ACTION_DIM,), 'action_delta_low')
        action_delta_high = cls._array(value.get('action_delta_high'), (ACTION_DIM,), 'action_delta_high')
        if np.any(action_delta_low > action_delta_high):
            raise ValueError(
                f'Region-aware baseline model {region.value} action delta bounds are invalid'
            )
        covariance = cls._array(
            value.get('joint_residual_covariance'),
            (ACTION_DIM, ACTION_DIM),
            'joint_residual_covariance',
        )
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f'Region-aware baseline model {region.value} residual covariance must be symmetric'
            )
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-8:
            raise ValueError(
                f'Region-aware baseline model {region.value} residual covariance must be positive semidefinite'
            )
        return cls(
            relation=relation,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            intercept=intercept,
            coefficient_matrix=coefficient_matrix,
            action_delta_low=action_delta_low,
            action_delta_high=action_delta_high,
            joint_residual_covariance=covariance,
            sample_count=sample_count,
        )

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'Region-aware baseline {name} must contain finite values with shape {shape}')
        return array

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.intercept + normalized @ self.coefficient_matrix


class RegionAwareJointBaseline:
    def __init__(
        self,
        *,
        models: dict[ControlRegion, _JointRegionModel],
        operating_state_config: OperatingStateConfig,
        step_duration_s: float,
        input_manifest_sha256: str,
        state_action_artifact_sha256: str | None,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        if not math.isfinite(step_duration_s) or step_duration_s <= 0.0:
            raise ValueError('Region-aware baseline step_duration_s must be positive and finite')
        self.schema_version = REGION_AWARE_BASELINE_SCHEMA_VERSION
        self.operating_state_contract_version = OPERATING_STATE_CONTRACT_VERSION
        self.feature_names = REGION_AWARE_BASELINE_FEATURES
        self.input_manifest_sha256 = input_manifest_sha256
        self.state_action_artifact_sha256 = state_action_artifact_sha256
        self.action_units = ('N m', 'rad')
        self.action_low = action_low
        self.action_high = action_high
        self.step_duration_s = float(step_duration_s)
        self.models = models
        self.operating_state_config = operating_state_config
        self.classifier = OperatingStateClassifier(operating_state_config)
        self.wind_obs_index = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')
        self.rotor_speed_obs_index = OBSERVATION_VARIABLES.index('I_RotorSpeed')
        self.generator_speed_obs_index = OBSERVATION_VARIABLES.index('I_GeneratorSpeed')
        self.real_power_obs_index = OBSERVATION_VARIABLES.index('I_RealPower')
        self.measured_torque_obs_index = OBSERVATION_VARIABLES.index('I_ElectricalTorque')
        self.measured_pitch_obs_index = OBSERVATION_VARIABLES.index('I_PitchPosition1')
        self._timestamp_s = 0.0

    @classmethod
    def from_path(cls, path: Path | str, *, step_duration_s: float) -> RegionAwareJointBaseline:
        resolved = Path(path).expanduser()
        try:
            with resolved.open('r', encoding='utf-8') as handle:
                document = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Region-aware baseline file does not exist: {resolved}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Unable to read region-aware baseline file: {resolved}') from exc
        return cls.from_document(document, step_duration_s=step_duration_s)

    @classmethod
    def from_document(
        cls,
        document: Any,
        *,
        step_duration_s: float,
    ) -> RegionAwareJointBaseline:
        if not isinstance(document, dict):
            raise ValueError('Region-aware baseline file must contain an object')
        if document.get('schema_version') != REGION_AWARE_BASELINE_SCHEMA_VERSION:
            raise ValueError('Region-aware baseline schema_version is incompatible')
        metadata = document.get('metadata')
        if not isinstance(metadata, dict):
            raise ValueError('Region-aware baseline file must contain metadata')
        cls._validate_metadata(metadata)
        operating_state_contract = metadata['operating_state_contract']
        thresholds = operating_state_contract.get('thresholds')
        if not isinstance(thresholds, dict):
            raise ValueError('Region-aware baseline operating-state thresholds are missing')
        try:
            operating_state_config = OperatingStateConfig(**thresholds)
        except (TypeError, ValueError) as exc:
            raise ValueError('Region-aware baseline operating-state thresholds are invalid') from exc

        control = document.get('control')
        baseline = control.get('region_aware_joint_baseline') if isinstance(control, dict) else None
        if not isinstance(baseline, dict):
            raise ValueError('Region-aware baseline file must contain control.region_aware_joint_baseline')
        if tuple(baseline.get('feature_names', ())) != REGION_AWARE_BASELINE_FEATURES:
            raise ValueError('Region-aware baseline feature_names are incompatible')
        if baseline.get('target') != 'next_realized_action_delta':
            if baseline.get('target') != 'next_realized_action':
                raise ValueError('Region-aware baseline target is incompatible')
        if baseline.get('causal') is not True or baseline.get('future_feature_lookahead_steps') != 0:
            raise ValueError('Region-aware baseline must declare zero-lookahead causal features')
        if baseline.get('action_units') != ['N m', 'rad']:
            raise ValueError('Region-aware baseline action_units are incompatible')
        action_low = _JointRegionModel._array(baseline.get('action_low'), (ACTION_DIM,), 'action_low')
        action_high = _JointRegionModel._array(baseline.get('action_high'), (ACTION_DIM,), 'action_high')
        if np.any(action_low > action_high):
            raise ValueError('Region-aware baseline action bounds are invalid')

        model_documents = baseline.get('models')
        if not isinstance(model_documents, dict):
            raise ValueError('Region-aware baseline models must be an object')
        models = {
            region: _JointRegionModel.from_document(model_documents.get(region.value), region)
            for region in _REGION_ORDER
        }
        cls._validate_common_delta_bounds(models)
        cls._validate_blending(baseline.get('transition_blending'), operating_state_config)
        return cls(
            models=models,
            operating_state_config=operating_state_config,
            step_duration_s=step_duration_s,
            input_manifest_sha256=str(metadata['input_manifest_sha256']),
            state_action_artifact_sha256=cls._optional_sha256(
                metadata.get('state_action_artifact_sha256'),
                'state_action_artifact_sha256',
            ),
            action_low=action_low,
            action_high=action_high,
        )

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        if metadata.get('data_schema_version') != DATA_SCHEMA_VERSION:
            raise ValueError('Region-aware baseline data schema is incompatible')
        if metadata.get('source_role') != 'raw_pid_calibration':
            raise ValueError('Region-aware baseline source_role must be raw_pid_calibration')
        if metadata.get('sampling_strategy') != 'all_valid_causal_transitions':
            raise ValueError('Region-aware baseline must use all valid causal raw transitions')
        if metadata.get('action_source') != list(ACTION_VARIABLES):
            raise ValueError('Region-aware baseline action_source is incompatible')
        if metadata.get('action_semantics') != 'recorded_pid_demand':
            raise ValueError('Region-aware baseline actions must declare recorded_pid_demand semantics')
        manifest = metadata.get('input_manifest_sha256')
        if (
            not isinstance(manifest, str)
            or len(manifest) != 64
            or any(character not in '0123456789abcdef' for character in manifest.lower())
        ):
            raise ValueError('Region-aware baseline input manifest SHA-256 is invalid')
        operating_state_contract = metadata.get('operating_state_contract')
        if (
            not isinstance(operating_state_contract, dict)
            or operating_state_contract.get('contract_version') not in {
                LEGACY_OPERATING_STATE_CONTRACT_VERSION,
                OPERATING_STATE_CONTRACT_VERSION,
            }
            or operating_state_contract.get('causal') is not True
        ):
            raise ValueError('Region-aware baseline operating-state contract is incompatible')

    @staticmethod
    def _optional_sha256(value: Any, name: str) -> str | None:
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value.lower())
        ):
            raise ValueError(f'Region-aware baseline {name} is invalid')
        return value.lower()

    @staticmethod
    def _validate_common_delta_bounds(models: dict[ControlRegion, _JointRegionModel]) -> None:
        reference = models[ControlRegion.BELOW_RATED]
        for region in _REGION_ORDER[1:]:
            model = models[region]
            if not np.array_equal(model.action_delta_low, reference.action_delta_low) or not np.array_equal(
                model.action_delta_high,
                reference.action_delta_high,
            ):
                raise ValueError('All region-aware baseline models must use the same action delta bounds')

    @staticmethod
    def _validate_blending(value: Any, config: OperatingStateConfig) -> None:
        if not isinstance(value, dict) or value.get('method') != 'smoothstep_with_hysteretic_region':
            raise ValueError('Region-aware baseline transition blending contract is incompatible')
        expected = {
            'transition_exit_wind_mps': config.transition_exit_wind_mps,
            'transition_enter_wind_mps': config.transition_enter_wind_mps,
            'above_rated_exit_wind_mps': config.above_rated_exit_wind_mps,
            'above_rated_enter_wind_mps': config.above_rated_enter_wind_mps,
        }
        for name, expected_value in expected.items():
            try:
                actual = float(value[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Region-aware baseline blending threshold {name} is invalid') from exc
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f'Region-aware baseline blending threshold {name} is incompatible')

    def reset(self) -> None:
        self.classifier.reset()
        self._timestamp_s = 0.0

    def previous_action_from_observation(self, observation: np.ndarray) -> np.ndarray:
        values = self._observation(observation)
        return np.asarray(
            [values[self.measured_torque_obs_index], values[self.measured_pitch_obs_index]],
            dtype=np.float64,
        )

    def predict(
        self,
        observation: np.ndarray,
        *,
        previous_action_real: np.ndarray | None = None,
    ) -> np.ndarray:
        values = self._observation(observation)
        previous_action = (
            self.previous_action_from_observation(values)
            if previous_action_real is None
            else np.asarray(previous_action_real, dtype=np.float64)
        )
        if previous_action.shape != (ACTION_DIM,) or not np.isfinite(previous_action).all():
            raise ValueError('Region-aware baseline previous action must contain two finite values')
        classification = self.classifier.classify(OperatingPoint(
            timestamp_s=self._timestamp_s,
            wind_speed_mps=float(values[self.wind_obs_index]),
            rotor_speed_rad_s=float(values[self.rotor_speed_obs_index]),
            real_power_w=float(values[self.real_power_obs_index]),
        ))
        self._timestamp_s += self.step_duration_s
        features = np.asarray(
            [
                classification.filtered_wind_speed_mps,
                values[self.rotor_speed_obs_index],
                values[self.generator_speed_obs_index],
                values[self.real_power_obs_index],
                previous_action[0],
                previous_action[1],
            ],
            dtype=np.float64,
        )
        weights = self._region_weights(
            classification.control_region,
            classification.filtered_wind_speed_mps,
        )
        model_outputs = {
            region: self.models[region].predict(features)
            for region in weights
        }
        desired_action = sum(
            weight * (
                previous_action + model_outputs[region]
                if self.models[region].relation == 'joint_next_action_delta'
                else model_outputs[region]
            )
            for region, weight in weights.items()
        )
        reference_model = self.models[ControlRegion.BELOW_RATED]
        desired_action = np.clip(desired_action, self.action_low, self.action_high)
        delta = desired_action - previous_action
        delta = np.clip(delta, reference_model.action_delta_low, reference_model.action_delta_high)
        return np.clip(previous_action + delta, self.action_low, self.action_high).astype(np.float64)

    @staticmethod
    def _observation(observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] < len(OBSERVATION_VARIABLES):
            raise ValueError(
                f'Region-aware baseline requires at least {len(OBSERVATION_VARIABLES)} observation values'
            )
        required = values[:len(OBSERVATION_VARIABLES)]
        if not np.isfinite(required).all():
            raise ValueError('Region-aware baseline observation values must be finite')
        return values

    def _region_weights(
        self,
        region: ControlRegion,
        filtered_wind_speed_mps: float,
    ) -> dict[ControlRegion, float]:
        config = self.operating_state_config
        if region == ControlRegion.BELOW_RATED:
            if filtered_wind_speed_mps <= config.transition_exit_wind_mps:
                return {ControlRegion.BELOW_RATED: 1.0}
            transition_weight = self._smoothstep(
                filtered_wind_speed_mps,
                config.transition_exit_wind_mps,
                config.transition_enter_wind_mps,
            )
            return {
                ControlRegion.BELOW_RATED: 1.0 - transition_weight,
                ControlRegion.TRANSITION: transition_weight,
            }
        if region == ControlRegion.ABOVE_RATED:
            if filtered_wind_speed_mps >= config.above_rated_enter_wind_mps:
                return {ControlRegion.ABOVE_RATED: 1.0}
            above_weight = self._smoothstep(
                filtered_wind_speed_mps,
                config.above_rated_exit_wind_mps,
                config.above_rated_enter_wind_mps,
            )
            return {
                ControlRegion.TRANSITION: 1.0 - above_weight,
                ControlRegion.ABOVE_RATED: above_weight,
            }
        if filtered_wind_speed_mps < config.transition_enter_wind_mps:
            transition_weight = self._smoothstep(
                filtered_wind_speed_mps,
                config.transition_exit_wind_mps,
                config.transition_enter_wind_mps,
            )
            return {
                ControlRegion.BELOW_RATED: 1.0 - transition_weight,
                ControlRegion.TRANSITION: transition_weight,
            }
        if filtered_wind_speed_mps > config.above_rated_exit_wind_mps:
            above_weight = self._smoothstep(
                filtered_wind_speed_mps,
                config.above_rated_exit_wind_mps,
                config.above_rated_enter_wind_mps,
            )
            return {
                ControlRegion.TRANSITION: 1.0 - above_weight,
                ControlRegion.ABOVE_RATED: above_weight,
            }
        return {ControlRegion.TRANSITION: 1.0}

    @staticmethod
    def _smoothstep(value: float, low: float, high: float) -> float:
        fraction = float(np.clip((value - low) / (high - low), 0.0, 1.0))
        return fraction * fraction * (3.0 - 2.0 * fraction)