from __future__ import annotations

import glob
import hashlib
import math
from pathlib import Path
import re
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

from RL.utils.constants import OBSERVATION_VARIABLES
from RL.utils.paired_pid_reference import CanonicalTraceHasher, PairedPidReference


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONDITION_PATTERN = re.compile(r'windspeed(?P<condition>\d+)', flags=re.IGNORECASE)
_WIND_INDEX = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')


class QueueCaseIdentityMismatch(RuntimeError):
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        super().__init__(
            'Queue physical case identity mismatch: '
            f'metadata_condition={int(metrics.get("queue_case_identity_condition_id", -1.0))} '
            f'instance={int(metrics.get("queue_case_identity_instance_id", -1.0))} '
            f'expected_wind={metrics.get("queue_case_identity_expected_wind", float("nan")):.6f} '
            f'observed_wind={metrics.get("queue_case_identity_observed_wind", float("nan")):.6f} '
            f'closest_raw_condition={int(metrics.get("queue_case_identity_closest_condition_id", -1.0))}'
        )


class QueueWindTraceIdentityValidator:
    _REFERENCE_SEARCH_SAMPLES = 32

    def __init__(
        self,
        condition_ids: tuple[int, ...],
        *,
        source_glob: str,
        wind_sample_count: int,
        wind_absolute_tolerance: float,
    ) -> None:
        if wind_sample_count <= 0:
            raise ValueError('queue.case_identity_validation.wind_sample_count must be positive')
        if not math.isfinite(wind_absolute_tolerance) or wind_absolute_tolerance < 0.0:
            raise ValueError('queue.case_identity_validation.wind_absolute_tolerance must be finite and non-negative')
        self.wind_sample_count = wind_sample_count
        self.wind_absolute_tolerance = wind_absolute_tolerance
        self._references = self._load_references(condition_ids, source_glob)
        self._active_condition_id: int | None = None
        self._active_condition_instance_id: int | None = None
        self._observed_winds: list[float] = []

    @staticmethod
    def _source_paths(source_glob: str, condition_ids: tuple[int, ...]) -> dict[int, Path]:
        configured_path = Path(source_glob).expanduser()
        pattern = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
        paths: dict[int, Path] = {}
        for source_path_text in sorted(glob.glob(str(pattern))):
            source_path = Path(source_path_text)
            match = _CONDITION_PATTERN.search(source_path.name)
            condition_id = None if match is None else int(match.group('condition'))
            if condition_id is None or condition_id not in condition_ids:
                continue
            if condition_id in paths:
                raise ValueError(
                    'queue.case_identity_validation has multiple raw traces for '
                    f'condition={condition_id}: {paths[condition_id]} and {source_path}'
                )
            paths[condition_id] = source_path
        missing = [condition_id for condition_id in condition_ids if condition_id not in paths]
        if missing:
            raise FileNotFoundError(
                'queue.case_identity_validation is missing raw traces for '
                f'conditions {missing}; glob={pattern}'
            )
        return paths

    def _load_references(self, condition_ids: tuple[int, ...], source_glob: str) -> dict[int, np.ndarray]:
        paths = self._source_paths(source_glob, condition_ids)
        references: dict[int, np.ndarray] = {}
        for condition_id, source_path in paths.items():
            try:
                frame = pd.read_csv(
                    source_path,
                    usecols=['I_AverageWindSpeed'],
                    nrows=max(self.wind_sample_count, self._REFERENCE_SEARCH_SAMPLES),
                )
                winds = frame['I_AverageWindSpeed'].to_numpy(dtype=np.float64)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(
                    f'Unable to read queue case identity raw trace for condition={condition_id}: {source_path}'
                ) from exc
            if len(winds) < self.wind_sample_count or not np.isfinite(winds).all():
                raise ValueError(
                    'queue.case_identity_validation raw trace needs finite wind samples: '
                    f'condition={condition_id} source={source_path}'
                )
            references[condition_id] = winds.astype(np.float32)
        return references

    @staticmethod
    def _best_window_match(
        observed_winds: np.ndarray,
        reference_winds: np.ndarray,
    ) -> tuple[float, int]:
        window_count = len(reference_winds) - len(observed_winds) + 1
        if window_count <= 0:
            raise ValueError('Queue case identity reference trace is shorter than the observed prefix')
        reference_windows = np.stack(
            [reference_winds[start:start + len(observed_winds)] for start in range(window_count)]
        )
        errors = np.max(np.abs(reference_windows - observed_winds), axis=1)
        best_start = int(np.argmin(errors))
        return float(errors[best_start]), best_start

    def reset_condition(self, condition_id: int, condition_instance_id: int) -> None:
        if condition_id not in self._references:
            raise ValueError(f'Queue case identity has no raw trace for condition={condition_id}')
        self._active_condition_id = condition_id
        self._active_condition_instance_id = condition_instance_id
        self._observed_winds = []

    def classify_observation_prefix(
        self,
        observations: list[np.ndarray],
    ) -> tuple[int | None, dict[str, float]]:
        if len(observations) != self.wind_sample_count:
            raise ValueError(
                'Queue case identity classification requires exactly '
                f'{self.wind_sample_count} observations, got {len(observations)}'
            )

        observed_winds: list[float] = []
        for observation in observations:
            values = np.asarray(observation, dtype=np.float32)
            if values.ndim != 1 or values.shape[0] <= _WIND_INDEX or not np.isfinite(values[_WIND_INDEX]):
                raise ValueError('Queue case identity requires a finite online I_AverageWindSpeed observation')
            observed_winds.append(float(values[_WIND_INDEX]))

        observed_prefix = np.asarray(observed_winds, dtype=np.float32)
        prefix_matches = {
            condition_id: self._best_window_match(observed_prefix, reference_winds)
            for condition_id, reference_winds in self._references.items()
        }
        prefix_errors = {
            condition_id: match[0] for condition_id, match in prefix_matches.items()
        }
        closest_condition_id = min(prefix_errors, key=prefix_errors.get)
        matching_condition_ids = [
            condition_id
            for condition_id, max_abs_error in prefix_errors.items()
            if max_abs_error <= self.wind_absolute_tolerance
        ]
        matched_condition_id = matching_condition_ids[0] if len(matching_condition_ids) == 1 else None
        return matched_condition_id, {
            'queue_case_identity_probe_sample_count': float(len(observed_winds)),
            'queue_case_identity_probe_match_count': float(len(matching_condition_ids)),
            'queue_case_identity_probe_closest_condition_id': float(closest_condition_id),
            'queue_case_identity_probe_closest_max_abs_error': prefix_errors[closest_condition_id],
            'queue_case_identity_probe_closest_reference_offset': float(
                prefix_matches[closest_condition_id][1]
            ),
            'queue_case_identity_probe_matched_condition_id': float(
                matched_condition_id if matched_condition_id is not None else -1
            ),
        }

    def prime_condition(
        self,
        condition_id: int,
        condition_instance_id: int,
        observations: list[np.ndarray],
    ) -> dict[str, float]:
        if len(observations) > self.wind_sample_count:
            raise ValueError(
                'Queue case identity priming received too many observations: '
                f'{len(observations)} > {self.wind_sample_count}'
            )
        self.reset_condition(condition_id, condition_instance_id)
        metrics: dict[str, float] = {
            'queue_case_identity_validation_active': 1.0,
            'queue_case_identity_validated': 0.0,
        }
        for observation in observations:
            metrics = self.observe(condition_id, condition_instance_id, observation)
        return metrics

    def observe(
        self,
        condition_id: int,
        condition_instance_id: int,
        observation: np.ndarray,
        *,
        sim_time_s: float | None = None,
        wind_speed_mps: float | None = None,
        runtime_case_fingerprint: str | None = None,
    ) -> dict[str, float]:
        del sim_time_s, runtime_case_fingerprint
        if (
            self._active_condition_id != condition_id
            or self._active_condition_instance_id != condition_instance_id
        ):
            self.reset_condition(condition_id, condition_instance_id)
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] <= _WIND_INDEX or not np.isfinite(values[_WIND_INDEX]):
            raise ValueError('Queue case identity requires a finite online I_AverageWindSpeed observation')

        sample_index = len(self._observed_winds)
        if sample_index >= self.wind_sample_count:
            return {
                'queue_case_identity_validation_active': 1.0,
                'queue_case_identity_validated': 1.0,
            }

        observed_wind = float(values[_WIND_INDEX]) if wind_speed_mps is None else float(wind_speed_mps)
        if not math.isfinite(observed_wind):
            raise ValueError('Queue case identity requires a finite online I_AverageWindSpeed observation')
        self._observed_winds.append(observed_wind)
        observed_prefix = np.asarray(self._observed_winds, dtype=np.float32)
        prefix_matches = {
            candidate_condition_id: self._best_window_match(observed_prefix, reference_winds)
            for candidate_condition_id, reference_winds in self._references.items()
        }
        prefix_errors = {
            candidate_condition_id: match[0]
            for candidate_condition_id, match in prefix_matches.items()
        }
        closest_condition_id = min(prefix_errors, key=prefix_errors.get)
        expected_max_abs_error, reference_offset = prefix_matches[condition_id]
        expected_wind = float(self._references[condition_id][reference_offset + sample_index])
        metrics = {
            'queue_case_identity_validation_active': 1.0,
            'queue_case_identity_validated': float(sample_index + 1 >= self.wind_sample_count),
            'queue_case_identity_condition_id': float(condition_id),
            'queue_case_identity_instance_id': float(condition_instance_id),
            'queue_case_identity_wind_sample_index': float(sample_index),
            'queue_case_identity_reference_offset': float(reference_offset),
            'queue_case_identity_expected_wind': expected_wind,
            'queue_case_identity_observed_wind': observed_wind,
            'queue_case_identity_expected_max_abs_error': expected_max_abs_error,
            'queue_case_identity_closest_condition_id': float(closest_condition_id),
            'queue_case_identity_closest_max_abs_error': prefix_errors[closest_condition_id],
            'queue_case_identity_closest_reference_offset': float(
                prefix_matches[closest_condition_id][1]
            ),
        }
        if expected_max_abs_error > self.wind_absolute_tolerance:
            raise QueueCaseIdentityMismatch(metrics)
        return metrics


class QueueStrictCaseIdentityValidator:
    """Validate the complete fixed-offset Time + wind trace against a paired PID reference."""

    validation_level = 'strict'

    def __init__(
        self,
        condition_ids: tuple[int, ...],
        *,
        paired_pid_reference_path: str,
        fixed_simulation_time_offset_s: float,
        wind_absolute_tolerance: float,
        time_absolute_tolerance_s: float,
        expected_project_case_fingerprint: str | None = None,
        development_wind_warning: bool = False,
        require_complete_trace_and_end: bool = False,
    ) -> None:
        if not math.isfinite(fixed_simulation_time_offset_s):
            raise ValueError('queue.case_identity_validation.fixed_simulation_time_offset_s must be finite')
        if not math.isfinite(wind_absolute_tolerance) or wind_absolute_tolerance < 0.0:
            raise ValueError('queue.case_identity_validation.wind_absolute_tolerance must be finite and non-negative')
        if not math.isfinite(time_absolute_tolerance_s) or time_absolute_tolerance_s < 0.0:
            raise ValueError('queue.case_identity_validation.time_absolute_tolerance_s must be finite and non-negative')
        path = Path(paired_pid_reference_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.reference = PairedPidReference.from_path(path, project_root=PROJECT_ROOT)
        missing = [condition_id for condition_id in condition_ids if condition_id not in self.reference.entries]
        if missing:
            raise ValueError(
                'queue.case_identity_validation paired PID reference is missing '
                f'conditions {missing}'
            )
        self.fixed_simulation_time_offset_s = float(fixed_simulation_time_offset_s)
        self.wind_absolute_tolerance = float(wind_absolute_tolerance)
        self.time_absolute_tolerance_s = float(time_absolute_tolerance_s)
        fingerprint = None if expected_project_case_fingerprint is None else str(
            expected_project_case_fingerprint
        ).strip().lower()
        if fingerprint and not self._is_sha256(fingerprint):
            raise ValueError(
                'queue.case_identity_validation.expected_project_case_fingerprint must be a SHA-256 string or null'
            )
        self.expected_project_case_fingerprint = fingerprint or None
        self.development_wind_warning = development_wind_warning
        self.require_complete_trace_and_end = require_complete_trace_and_end
        self._wind_mismatch_count = 0
        self._wind_max_abs_error = 0.0
        self._last_observed_wind = float('nan')
        self._active_condition_id: int | None = None
        self._active_condition_instance_id: int | None = None
        self._hasher: CanonicalTraceHasher | None = None
        self._sample_count = 0
        self._last_simulation_time_s: float | None = None
        self._runtime_case_fingerprint: str | None = None

    @staticmethod
    def _is_sha256(value: str) -> bool:
        return len(value) == 64 and all(character in '0123456789abcdef' for character in value)

    @property
    def wind_sample_count(self) -> int:
        """Compatibility value for diagnostic-only callers; strict mode never uses prefix classification."""
        return 1

    def reset_condition(
        self,
        condition_id: int,
        condition_instance_id: int,
        *,
        runtime_case_fingerprint: str | None = None,
    ) -> None:
        entry = self.reference.entry(condition_id)
        self._active_condition_id = int(condition_id)
        self._active_condition_instance_id = int(condition_instance_id)
        self._hasher = CanonicalTraceHasher(
            time_quantization_s=entry.time_signature_quantization_s,
            wind_quantization_mps=entry.wind_signature_quantization_mps,
        )
        self._sample_count = 0
        self._last_simulation_time_s = None
        self._wind_mismatch_count = 0
        self._wind_max_abs_error = 0.0
        self._last_observed_wind = float('nan')
        normalized_fingerprint = None if runtime_case_fingerprint is None else str(
            runtime_case_fingerprint
        ).strip().lower()
        if normalized_fingerprint and not self._is_sha256(normalized_fingerprint):
            raise ValueError('runtime project/case fingerprint must be a SHA-256 string when provided')
        self._runtime_case_fingerprint = normalized_fingerprint or None

    def _ensure_active(
        self,
        condition_id: int,
        condition_instance_id: int,
        runtime_case_fingerprint: str | None,
    ) -> None:
        if (
            self._active_condition_id != int(condition_id)
            or self._active_condition_instance_id != int(condition_instance_id)
        ):
            self.reset_condition(
                condition_id,
                condition_instance_id,
                runtime_case_fingerprint=runtime_case_fingerprint,
            )
        elif runtime_case_fingerprint is not None:
            normalized = str(runtime_case_fingerprint).strip().lower()
            if not self._is_sha256(normalized):
                raise ValueError('runtime project/case fingerprint must be a SHA-256 string when provided')
            if self._runtime_case_fingerprint is not None and normalized != self._runtime_case_fingerprint:
                raise ValueError('runtime project/case fingerprint changed within one condition')
            self._runtime_case_fingerprint = normalized

    def observe(
        self,
        condition_id: int,
        condition_instance_id: int,
        observation: np.ndarray,
        *,
        sim_time_s: float | None = None,
        wind_speed_mps: float | None = None,
        runtime_case_fingerprint: str | None = None,
    ) -> dict[str, float]:
        if sim_time_s is None or not math.isfinite(float(sim_time_s)):
            raise ValueError('Strict queue case identity requires a finite simulation timestamp')
        self._ensure_active(condition_id, condition_instance_id, runtime_case_fingerprint)
        assert self._hasher is not None
        entry = self.reference.entry(condition_id)
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] <= _WIND_INDEX:
            raise ValueError('Queue case identity requires an online I_AverageWindSpeed observation')
        observed_wind = float(values[_WIND_INDEX]) if wind_speed_mps is None else float(wind_speed_mps)
        if not math.isfinite(observed_wind):
            raise ValueError('Queue case identity requires a finite online I_AverageWindSpeed observation')
        if self._sample_count >= len(entry.timestamps_s):
            raise QueueCaseIdentityMismatch(self._metrics(
                entry,
                observed_wind=observed_wind,
                observed_simulation_time_s=float(sim_time_s),
                expected_index=len(entry.timestamps_s) - 1,
                extra={'queue_case_identity_extra_trace_frame': 1.0},
            ))
        observed_simulation_time = float(sim_time_s)
        if (
            self._last_simulation_time_s is not None
            and observed_simulation_time <= self._last_simulation_time_s
        ):
            raise ValueError('Strict queue case identity simulation timestamps must be strictly increasing')
        reference_time = observed_simulation_time + self.fixed_simulation_time_offset_s
        expected_index = self._sample_count
        expected_time = float(entry.timestamps_s[expected_index])
        expected_wind = float(entry.wind_speeds_mps[expected_index])
        time_error = abs(reference_time - expected_time)
        wind_error = abs(observed_wind - expected_wind)
        metrics = self._metrics(
            entry,
            observed_wind=observed_wind,
            observed_simulation_time_s=observed_simulation_time,
            expected_index=expected_index,
            extra={
                'queue_case_identity_expected_reference_time_s': expected_time,
                'queue_case_identity_observed_reference_time_s': reference_time,
                'queue_case_identity_time_abs_error_s': time_error,
                'queue_case_identity_expected_max_abs_error': wind_error,
            },
        )
        if time_error > self.time_absolute_tolerance_s:
            raise QueueCaseIdentityMismatch(metrics)
        if wind_error > self.wind_absolute_tolerance:
            if not self.development_wind_warning:
                raise QueueCaseIdentityMismatch(metrics)
            self._wind_mismatch_count += 1
            if self._wind_mismatch_count == 1:
                warnings.warn(
                    f'DEVELOPMENT_UNVERIFIED wind mismatch: condition={condition_id} '
                    f'instance={condition_instance_id} time={observed_simulation_time:.6f} '
                    f'expected={expected_wind:.7f} observed={observed_wind:.7f} '
                    f'abs_error={wind_error:.7f}; continuing operator-confirmed training only; '
                    'not strict PID identity or acceptance evidence', RuntimeWarning,
                )
        self._wind_max_abs_error = max(self._wind_max_abs_error, wind_error)
        self._last_observed_wind = observed_wind
        self._hasher.update(reference_time, observed_wind)
        self._sample_count += 1
        self._last_simulation_time_s = observed_simulation_time
        return metrics | {
            **self._development_metrics(),
            'queue_case_identity_validated': 0.0,
            'queue_case_identity_strict_case_matched': 0.0,
            'queue_case_identity_reference_matched_only': 0.0,
        }

    def _development_metrics(self) -> dict[str, float]:
        return {
            'queue_case_identity_development_wind_warning_enabled': float(self.development_wind_warning),
            'queue_case_identity_development_unverified': float(self._wind_mismatch_count > 0),
            'queue_case_identity_wind_mismatch_count': float(self._wind_mismatch_count),
            'queue_case_identity_wind_max_abs_error_mps': self._wind_max_abs_error,
        }

    def _metrics(
        self,
        entry,
        *,
        observed_wind: float,
        observed_simulation_time_s: float,
        expected_index: int,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        expected_index = max(0, min(expected_index, len(entry.wind_speeds_mps) - 1))
        result = {
            'queue_case_identity_validation_active': 1.0,
            'queue_case_identity_validation_level_strict': 1.0,
            'queue_case_identity_condition_id': float(entry.condition_id),
            'queue_case_identity_instance_id': float(self._active_condition_instance_id or -1),
            'queue_case_identity_wind_sample_index': float(expected_index),
            'queue_case_identity_reference_offset': self.fixed_simulation_time_offset_s,
            'queue_case_identity_expected_wind': float(entry.wind_speeds_mps[expected_index]),
            'queue_case_identity_observed_wind': observed_wind,
            'queue_case_identity_closest_condition_id': float(entry.condition_id),
            'queue_case_identity_closest_max_abs_error': abs(
                observed_wind - float(entry.wind_speeds_mps[expected_index])
            ),
            'queue_case_identity_observed_simulation_time_s': observed_simulation_time_s,
            'queue_case_identity_trace_sample_count': float(self._sample_count),
            'queue_case_identity_expected_trace_sample_count': float(len(entry.timestamps_s)),
        }
        if extra:
            result.update(extra)
        return result

    def finish_condition(
        self,
        condition_id: int,
        condition_instance_id: int,
        *,
        runtime_case_fingerprint: str | None = None,
        end_received: bool = False,
    ) -> tuple[dict[str, float], dict[str, str | float]]:
        self._ensure_active(condition_id, condition_instance_id, runtime_case_fingerprint)
        assert self._hasher is not None
        entry = self.reference.entry(condition_id)
        trace_complete = self._sample_count == len(entry.timestamps_s)
        if self.require_complete_trace_and_end and (not trace_complete or not end_received):
            raise QueueCaseIdentityMismatch(self._metrics(
                entry, observed_wind=self._last_observed_wind,
                observed_simulation_time_s=self._last_simulation_time_s or 0.0,
                expected_index=max(0, self._sample_count - 1),
                extra={'queue_case_identity_incomplete_trace': float(not trace_complete),
                       'queue_case_identity_missing_end': float(not end_received)},
            ))
        trace_signature_match = trace_complete and self._hasher.hexdigest() == entry.trace_signature
        project_case_available = self._runtime_case_fingerprint is not None
        project_case_match = (
            self.expected_project_case_fingerprint is not None
            and self._runtime_case_fingerprint == self.expected_project_case_fingerprint
        )
        if (
            self.expected_project_case_fingerprint is not None
            and project_case_available
            and not project_case_match
        ):
            raise QueueCaseIdentityMismatch(self._metrics(
                entry,
                observed_wind=float('nan'),
                observed_simulation_time_s=(
                    0.0 if self._last_simulation_time_s is None else self._last_simulation_time_s
                ),
                expected_index=max(0, self._sample_count - 1),
                extra={'queue_case_identity_project_case_fingerprint_mismatch': 1.0},
            ))
        wind_unverified = self._wind_mismatch_count > 0
        strict_case_matched = trace_complete and not wind_unverified and (trace_signature_match or project_case_match)
        metrics = self._metrics(
            entry,
            observed_wind=(
                float('nan')
                if self._sample_count == 0
                else self._last_observed_wind
            ),
            observed_simulation_time_s=(
                0.0 if self._last_simulation_time_s is None else self._last_simulation_time_s
            ),
            expected_index=max(0, self._sample_count - 1),
            extra={
                **self._development_metrics(),
                'queue_case_identity_observed_frame_count_complete': float(trace_complete),
                'queue_case_identity_end_received': float(end_received),
                'queue_case_identity_validated': float(trace_complete and not wind_unverified),
                'queue_case_identity_full_trace_complete': float(trace_complete and not wind_unverified),
                'queue_case_identity_trace_signature_match': float(trace_signature_match),
                'queue_case_identity_project_case_fingerprint_available': float(project_case_available),
                'queue_case_identity_project_case_fingerprint_match': float(project_case_match),
                'queue_case_identity_strict_case_matched': float(strict_case_matched),
                'queue_case_identity_reference_matched_only': float(not strict_case_matched and not wind_unverified),
            },
        )
        summary: dict[str, str | float] = {
            'identity_status': ('development_unverified' if wind_unverified else
                                'strict_case_matched' if strict_case_matched else 'reference_matched_only'),
            'wind_mismatch_policy': 'warn_development_only' if self.development_wind_warning else 'abort',
            'condition_id': entry.condition_id,
            'condition_instance_id': condition_instance_id,
            'scenario_id': entry.scenario_id,
            'pid_raw_csv_path': str(entry.pid_raw_csv_path),
            'pid_raw_sha256': entry.pid_raw_sha256,
            'time_window_start_s': entry.time_window_start_s,
            'time_window_end_s': entry.time_window_end_s,
            'fixed_simulation_time_offset_s': self.fixed_simulation_time_offset_s,
            'expected_trace_signature': entry.trace_signature,
            'observed_trace_signature': self._hasher.hexdigest(),
            'expected_trace_sample_count': len(entry.timestamps_s),
            'observed_trace_sample_count': self._sample_count,
            'trace_signature_match': float(trace_signature_match),
            'project_case_fingerprint_available': float(project_case_available),
            'project_case_fingerprint_match': float(project_case_match),
            **metrics,
        }
        if wind_unverified:
            warnings.warn(
                f'DEVELOPMENT_UNVERIFIED condition={condition_id} instance={condition_instance_id}: '
                f'wind mismatch frames={self._wind_mismatch_count}/{self._sample_count}, '
                f'max_abs_error={self._wind_max_abs_error:.7f} m/s; strict_complete=false', RuntimeWarning,
            )
        return metrics, summary

    def classify_observation_prefix(
        self,
        observations: list[np.ndarray],
    ) -> tuple[int | None, dict[str, float]]:
        del observations
        raise RuntimeError('Strict queue case identity does not support prefix-only classification')

    def prime_condition(
        self,
        condition_id: int,
        condition_instance_id: int,
        observations: list[np.ndarray],
    ) -> dict[str, float]:
        del observations
        self.reset_condition(condition_id, condition_instance_id)
        return {
            'queue_case_identity_validation_active': 1.0,
            'queue_case_identity_validation_level_strict': 1.0,
            'queue_case_identity_validated': 0.0,
        }


def build_queue_case_identity_validator(
    queue_config: Mapping[str, Any],
    condition_ids: tuple[int, ...],
    *, operator_development_authorized: bool = False,
) -> QueueWindTraceIdentityValidator | QueueStrictCaseIdentityValidator | None:
    raw_config = queue_config.get('case_identity_validation')
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError('queue.case_identity_validation must be an object')
    policy = raw_config.get('wind_mismatch_policy', 'abort')
    if policy not in ('abort', 'warn_development_only'):
        raise ValueError('Invalid queue wind_mismatch_policy')
    if policy == 'warn_development_only' and (
            not operator_development_authorized or raw_config.get('enabled') is not True or
            raw_config.get('validation_level') != 'strict'):
        raise ValueError('Development wind warnings require authorized operator training and strict time validation')
    enabled = raw_config.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('queue.case_identity_validation.enabled must be a boolean')
    if not enabled:
        return None
    validation_level = str(raw_config.get('validation_level', 'prefix')).strip().lower()
    if validation_level not in {'prefix', 'strict'}:
        raise ValueError('queue.case_identity_validation.validation_level must be prefix or strict')
    if validation_level == 'strict':
        paired_pid_reference_path = raw_config.get('paired_pid_reference_path')
        if not isinstance(paired_pid_reference_path, str) or not paired_pid_reference_path.strip():
            raise ValueError(
                'strict queue case identity requires a non-empty '
                'queue.case_identity_validation.paired_pid_reference_path'
            )
        if 'fixed_simulation_time_offset_s' not in raw_config:
            raise ValueError(
                'strict queue case identity requires '
                'queue.case_identity_validation.fixed_simulation_time_offset_s'
            )
        if 'time_absolute_tolerance_s' not in raw_config:
            raise ValueError(
                'strict queue case identity requires '
                'queue.case_identity_validation.time_absolute_tolerance_s'
            )
        try:
            return QueueStrictCaseIdentityValidator(
                condition_ids,
                paired_pid_reference_path=paired_pid_reference_path.strip(),
                fixed_simulation_time_offset_s=float(raw_config['fixed_simulation_time_offset_s']),
                wind_absolute_tolerance=float(raw_config.get('wind_absolute_tolerance', 0.01)),
                time_absolute_tolerance_s=float(raw_config['time_absolute_tolerance_s']),
                expected_project_case_fingerprint=raw_config.get('expected_project_case_fingerprint'),
                development_wind_warning=policy == 'warn_development_only',
                require_complete_trace_and_end=operator_development_authorized,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid strict queue case identity configuration: {exc}') from exc
    source_glob = raw_config.get('source_glob')
    if not isinstance(source_glob, str) or not source_glob.strip():
        raise ValueError('queue.case_identity_validation.source_glob must be a non-empty string')
    wind_sample_count = raw_config.get('wind_sample_count', 3)
    if isinstance(wind_sample_count, bool) or not isinstance(wind_sample_count, int):
        raise ValueError('queue.case_identity_validation.wind_sample_count must be a positive integer')
    try:
        wind_absolute_tolerance = float(raw_config.get('wind_absolute_tolerance', 0.01))
    except (TypeError, ValueError) as exc:
        raise ValueError('queue.case_identity_validation.wind_absolute_tolerance must be numeric') from exc
    return QueueWindTraceIdentityValidator(
        condition_ids,
        source_glob=source_glob.strip(),
        wind_sample_count=wind_sample_count,
        wind_absolute_tolerance=wind_absolute_tolerance,
    )