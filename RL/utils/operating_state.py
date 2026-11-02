from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any


OPERATING_STATE_CONTRACT_VERSION = 'wind_turbine_operating_state_v2'
LEGACY_OPERATING_STATE_CONTRACT_VERSION = 'wind_turbine_operating_state_v1'


class OperatingState(str, Enum):
    STARTUP_CUT_IN = 'startup_cut_in'
    NORMAL_GENERATION = 'normal_generation'
    TRANSITION = 'transition'
    ABOVE_RATED_REGULATION = 'above_rated_regulation'
    SHUTDOWN_STOPPED = 'shutdown_stopped'
    NEGATIVE_POWER_ABNORMAL = 'negative_power_abnormal'


class ControlRegion(str, Enum):
    BELOW_RATED = 'below_rated'
    TRANSITION = 'transition'
    ABOVE_RATED = 'above_rated'


class OptimizationState(str, Enum):
    STARTUP_STOPPED = 'startup_stopped'
    BELOW_RATED = 'below_rated'
    TRANSITION = 'transition'
    ABOVE_RATED = 'above_rated'
    NEGATIVE_POWER_ABNORMAL = 'negative_power_abnormal'


OPTIMIZATION_STATE_ENCODING_ORDER = tuple(OptimizationState)
CONTROL_REGION_ENCODING_ORDER = tuple(ControlRegion)
OPTIMIZATION_STATE_OBSERVATION_VARIABLES = tuple(
    f'optimization_state_{state.value}' for state in OPTIMIZATION_STATE_ENCODING_ORDER
)
CONTROL_REGION_OBSERVATION_VARIABLES = tuple(
    f'control_region_{region.value}' for region in CONTROL_REGION_ENCODING_ORDER
)
STATE_REGIME_OBSERVATION_VARIABLES = (
    *OPTIMIZATION_STATE_OBSERVATION_VARIABLES,
    *CONTROL_REGION_OBSERVATION_VARIABLES,
)


@dataclass(frozen=True)
class OperatingStateConfig:
    transition_enter_wind_mps: float = 11.8
    transition_exit_wind_mps: float = 11.2
    above_rated_enter_wind_mps: float = 13.8
    above_rated_exit_wind_mps: float = 13.2
    wind_filter_time_constant_s: float = 2.0
    region_confirmation_s: float = 1.0
    running_rotor_speed_rad_s: float = 0.5
    stopped_rotor_speed_rad_s: float = 0.4
    generation_power_threshold_w: float = 100_000.0
    stopped_power_threshold_w: float = 100_000.0
    negative_power_threshold_w: float = -1_000.0
    running_confirmation_s: float = 1.0
    stopped_confirmation_s: float = 1.0

    def __post_init__(self) -> None:
        finite_values = asdict(self)
        if not all(math.isfinite(float(value)) for value in finite_values.values()):
            raise ValueError('Operating-state thresholds must be finite')
        if not (
            self.transition_exit_wind_mps
            < self.transition_enter_wind_mps
            < self.above_rated_exit_wind_mps
            < self.above_rated_enter_wind_mps
        ):
            raise ValueError('Wind thresholds must define ordered below/transition/above hysteresis bands')
        if self.wind_filter_time_constant_s < 0.0:
            raise ValueError('wind_filter_time_constant_s must be non-negative')
        if self.region_confirmation_s < 0.0:
            raise ValueError('region_confirmation_s must be non-negative')
        if self.running_rotor_speed_rad_s <= self.stopped_rotor_speed_rad_s:
            raise ValueError('running_rotor_speed_rad_s must exceed stopped_rotor_speed_rad_s')
        if self.generation_power_threshold_w < 0.0 or self.stopped_power_threshold_w < 0.0:
            raise ValueError('Power thresholds must be non-negative')
        if self.negative_power_threshold_w > 0.0:
            raise ValueError('negative_power_threshold_w must be non-positive')
        if self.running_confirmation_s < 0.0 or self.stopped_confirmation_s < 0.0:
            raise ValueError('State confirmation durations must be non-negative')

    def to_metadata(self) -> dict[str, Any]:
        return {
            'contract_version': OPERATING_STATE_CONTRACT_VERSION,
            'causal': True,
            'input_fields': [
                'timestamp_s',
                'wind_speed_mps',
                'rotor_speed_rad_s',
                'real_power_w',
                'pitch_rad',
                'previous_torque_nm',
                'previous_pitch_rad',
            ],
            'wind_filter': 'causal_exponential_moving_average',
            'reset_scope': 'one independent condition or raw file',
            'initial_operating_state': OperatingState.STARTUP_CUT_IN.value,
            'initial_region_rule': (
                'first filtered wind selects below/transition/above using the two enter thresholds'
            ),
            'region_transition_rule': 'hysteresis_threshold_sustained_for_confirmation_duration',
            'normal_mppt_eligible_rule': (
                'normal_generation and below_rated and currently above running speed/power thresholds'
            ),
            'operating_states': [state.value for state in OperatingState],
            'control_regions': [region.value for region in ControlRegion],
            'thresholds': asdict(self),
        }


@dataclass(frozen=True)
class OperatingPoint:
    timestamp_s: float
    wind_speed_mps: float
    rotor_speed_rad_s: float
    real_power_w: float
    pitch_rad: float = 0.0
    previous_torque_nm: float = 0.0
    previous_pitch_rad: float = 0.0


@dataclass(frozen=True)
class OperatingClassification:
    operating_state: OperatingState
    optimization_state: OptimizationState
    control_region: ControlRegion
    filtered_wind_speed_mps: float
    normal_mppt_eligible: bool
    operating_state_changed: bool
    control_region_changed: bool


class OperatingStateClassifier:
    def __init__(self, config: OperatingStateConfig | None = None) -> None:
        self.config = config or OperatingStateConfig()
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_s: float | None = None
        self._filtered_wind_speed_mps: float | None = None
        self._control_region: ControlRegion | None = None
        self._region_candidate: ControlRegion | None = None
        self._region_candidate_since_s: float | None = None
        self._operating_state = OperatingState.STARTUP_CUT_IN
        self._running = False
        self._has_generated = False
        self._running_candidate_since_s: float | None = None
        self._stopped_candidate_since_s: float | None = None

    def classify(self, point: OperatingPoint) -> OperatingClassification:
        self._validate_point(point)
        previous_timestamp_s = self._last_timestamp_s
        if previous_timestamp_s is not None and point.timestamp_s <= previous_timestamp_s:
            raise ValueError('Operating-state timestamps must be strictly increasing')
        delta_s = 0.0 if previous_timestamp_s is None else point.timestamp_s - previous_timestamp_s
        self._last_timestamp_s = point.timestamp_s
        self._update_filtered_wind(point.wind_speed_mps, delta_s)

        previous_region = self._control_region
        self._update_control_region(point.timestamp_s)
        assert self._control_region is not None
        previous_state = self._operating_state
        self._update_running_mode(point)
        self._operating_state = self._classify_operating_state(point)
        currently_generating = (
            point.rotor_speed_rad_s >= self.config.running_rotor_speed_rad_s
            and point.real_power_w >= self.config.generation_power_threshold_w
        )
        return OperatingClassification(
            operating_state=self._operating_state,
            optimization_state=optimization_state_for(self._operating_state),
            control_region=self._control_region,
            filtered_wind_speed_mps=float(self._filtered_wind_speed_mps),
            normal_mppt_eligible=(
                self._operating_state == OperatingState.NORMAL_GENERATION
                and self._control_region == ControlRegion.BELOW_RATED
                and currently_generating
            ),
            operating_state_changed=self._operating_state != previous_state,
            control_region_changed=previous_region is not None and self._control_region != previous_region,
        )

    @staticmethod
    def _validate_point(point: OperatingPoint) -> None:
        values = (
            point.timestamp_s,
            point.wind_speed_mps,
            point.rotor_speed_rad_s,
            point.real_power_w,
            point.pitch_rad,
            point.previous_torque_nm,
            point.previous_pitch_rad,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('Operating-state inputs must be finite')

    def _update_filtered_wind(self, wind_speed_mps: float, delta_s: float) -> None:
        if self._filtered_wind_speed_mps is None or self.config.wind_filter_time_constant_s == 0.0:
            self._filtered_wind_speed_mps = wind_speed_mps
            return
        alpha = 1.0 - math.exp(-delta_s / self.config.wind_filter_time_constant_s)
        self._filtered_wind_speed_mps += alpha * (wind_speed_mps - self._filtered_wind_speed_mps)

    def _initial_region(self) -> ControlRegion:
        assert self._filtered_wind_speed_mps is not None
        if self._filtered_wind_speed_mps >= self.config.above_rated_enter_wind_mps:
            return ControlRegion.ABOVE_RATED
        if self._filtered_wind_speed_mps >= self.config.transition_enter_wind_mps:
            return ControlRegion.TRANSITION
        return ControlRegion.BELOW_RATED

    def _desired_region(self) -> ControlRegion:
        assert self._control_region is not None
        assert self._filtered_wind_speed_mps is not None
        wind = self._filtered_wind_speed_mps
        if self._control_region == ControlRegion.BELOW_RATED:
            if wind >= self.config.transition_enter_wind_mps:
                return ControlRegion.TRANSITION
        elif self._control_region == ControlRegion.TRANSITION:
            if wind >= self.config.above_rated_enter_wind_mps:
                return ControlRegion.ABOVE_RATED
            if wind <= self.config.transition_exit_wind_mps:
                return ControlRegion.BELOW_RATED
        elif wind <= self.config.above_rated_exit_wind_mps:
            return ControlRegion.TRANSITION
        return self._control_region

    def _update_control_region(self, timestamp_s: float) -> None:
        if self._control_region is None:
            self._control_region = self._initial_region()
            return
        desired = self._desired_region()
        if desired == self._control_region:
            self._region_candidate = None
            self._region_candidate_since_s = None
            return
        if desired != self._region_candidate:
            self._region_candidate = desired
            self._region_candidate_since_s = timestamp_s
        assert self._region_candidate_since_s is not None
        if timestamp_s - self._region_candidate_since_s + 1e-12 >= self.config.region_confirmation_s:
            self._control_region = desired
            self._region_candidate = None
            self._region_candidate_since_s = None

    @staticmethod
    def _confirmed(timestamp_s: float, since_s: float | None, confirmation_s: float) -> bool:
        return since_s is not None and timestamp_s - since_s + 1e-12 >= confirmation_s

    def _update_running_mode(self, point: OperatingPoint) -> None:
        running_signal = (
            point.rotor_speed_rad_s >= self.config.running_rotor_speed_rad_s
            and point.real_power_w >= self.config.generation_power_threshold_w
        )
        stopped_signal = (
            point.rotor_speed_rad_s <= self.config.stopped_rotor_speed_rad_s
            or abs(point.real_power_w) <= self.config.stopped_power_threshold_w
        )
        if not self._running:
            self._stopped_candidate_since_s = None
            if running_signal:
                if self._running_candidate_since_s is None:
                    self._running_candidate_since_s = point.timestamp_s
                if self._confirmed(
                    point.timestamp_s,
                    self._running_candidate_since_s,
                    self.config.running_confirmation_s,
                ):
                    self._running = True
                    self._has_generated = True
                    self._running_candidate_since_s = None
            else:
                self._running_candidate_since_s = None
            return

        self._running_candidate_since_s = None
        if stopped_signal:
            if self._stopped_candidate_since_s is None:
                self._stopped_candidate_since_s = point.timestamp_s
            if self._confirmed(
                point.timestamp_s,
                self._stopped_candidate_since_s,
                self.config.stopped_confirmation_s,
            ):
                self._running = False
                self._stopped_candidate_since_s = None
        else:
            self._stopped_candidate_since_s = None

    def _classify_operating_state(self, point: OperatingPoint) -> OperatingState:
        if point.real_power_w < self.config.negative_power_threshold_w:
            return OperatingState.NEGATIVE_POWER_ABNORMAL
        if not self._running:
            if self._has_generated and self._running_candidate_since_s is None:
                return OperatingState.SHUTDOWN_STOPPED
            return OperatingState.STARTUP_CUT_IN
        assert self._control_region is not None
        if self._control_region == ControlRegion.TRANSITION:
            return OperatingState.TRANSITION
        if self._control_region == ControlRegion.ABOVE_RATED:
            return OperatingState.ABOVE_RATED_REGULATION
        return OperatingState.NORMAL_GENERATION


def optimization_state_for(operating_state: OperatingState) -> OptimizationState:
    if operating_state == OperatingState.NORMAL_GENERATION:
        return OptimizationState.BELOW_RATED
    if operating_state == OperatingState.TRANSITION:
        return OptimizationState.TRANSITION
    if operating_state == OperatingState.ABOVE_RATED_REGULATION:
        return OptimizationState.ABOVE_RATED
    if operating_state == OperatingState.NEGATIVE_POWER_ABNORMAL:
        return OptimizationState.NEGATIVE_POWER_ABNORMAL
    return OptimizationState.STARTUP_STOPPED


def encode_state_regime(classification: OperatingClassification) -> tuple[float, ...]:
    return (
        *(float(classification.optimization_state == state) for state in OPTIMIZATION_STATE_ENCODING_ORDER),
        *(float(classification.control_region == region) for region in CONTROL_REGION_ENCODING_ORDER),
    )