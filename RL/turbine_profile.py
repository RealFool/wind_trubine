"""Per-turbine profile: the only place that should know 15 MW vs 22 MW.

S2 keeps runtime behaviour unchanged: a config opts in with
`turbine_profile_path`, and the profile is checked against the config at
startup.  22 MW will add a sibling profile and rebuilt assets, not edit the
15 MW files.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'turbine_profile_v1'


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(str(value).replace('\\', '/')).expanduser()
    return path if path.is_absolute() else project_root / path


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _pair(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must contain two numbers')
    result = [_number(item, f'{name}[{index}]') for index, item in enumerate(value)]
    return result


@dataclass(frozen=True)
class TurbineProfile:
    turbine_id: str
    display_name: str
    rated_power_w: float
    gear_ratio: float
    cut_in_wind_mps: float
    rated_wind_mps: float
    max_wind_mps: float
    action_low: list[float]
    action_high: list[float]
    safe_action_low: list[float]
    safe_action_high: list[float]
    action_delta_low: list[float]
    action_delta_high: list[float]
    residual_limit: list[float]
    actor_residual_target_scale: float
    power_ref_w: float
    blade_root_ref_nm: float
    pitch_rate_limit_rad: float
    torque_rate_limit_nm: float
    data_schema_version: str
    assets: dict[str, str]
    bladed_project: str
    wind_data_glob: str

    @classmethod
    def from_path(cls, path: Path | str) -> TurbineProfile:
        resolved = Path(path).expanduser()
        try:
            document = json.loads(resolved.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'turbine profile does not exist: {resolved}') from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'unable to read turbine profile: {resolved}') from exc
        return cls.from_document(document, source=str(resolved))

    @classmethod
    def from_document(cls, document: Any, *, source: str = '<memory>') -> TurbineProfile:
        if not isinstance(document, dict):
            raise ValueError(f'turbine profile must be an object: {source}')
        if document.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'turbine profile schema mismatch: {source}')
        assets = document.get('assets')
        if not isinstance(assets, dict) or not assets:
            raise ValueError(f'turbine profile assets must be a non-empty object: {source}')
        required_assets = {
            'b4_baseline', 'state_action_envelope', 's2_u1_calibration',
            'bc_checkpoint', 'shadow_bundle', 'paired_reference',
        }
        missing = sorted(required_assets - set(assets))
        if missing:
            raise ValueError(f'turbine profile assets missing {missing}: {source}')
        for key, value in assets.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'turbine profile asset {key} must be a non-empty path')
        profile = cls(
            turbine_id=str(document.get('turbine_id', '')).strip(),
            display_name=str(document.get('display_name', '')).strip(),
            rated_power_w=_number(document.get('rated_power_w'), 'rated_power_w'),
            gear_ratio=_number(document.get('gear_ratio'), 'gear_ratio'),
            cut_in_wind_mps=_number(document.get('cut_in_wind_mps'), 'cut_in_wind_mps'),
            rated_wind_mps=_number(document.get('rated_wind_mps'), 'rated_wind_mps'),
            max_wind_mps=_number(document.get('max_wind_mps'), 'max_wind_mps'),
            action_low=_pair(document.get('action_low'), 'action_low'),
            action_high=_pair(document.get('action_high'), 'action_high'),
            safe_action_low=_pair(document.get('safe_action_low'), 'safe_action_low'),
            safe_action_high=_pair(document.get('safe_action_high'), 'safe_action_high'),
            action_delta_low=_pair(document.get('action_delta_low'), 'action_delta_low'),
            action_delta_high=_pair(document.get('action_delta_high'), 'action_delta_high'),
            residual_limit=_pair(document.get('residual_limit'), 'residual_limit'),
            actor_residual_target_scale=_number(
                document.get('actor_residual_target_scale'), 'actor_residual_target_scale'),
            power_ref_w=_number(document.get('power_ref_w'), 'power_ref_w'),
            blade_root_ref_nm=_number(document.get('blade_root_ref_nm'), 'blade_root_ref_nm'),
            pitch_rate_limit_rad=_number(document.get('pitch_rate_limit_rad'), 'pitch_rate_limit_rad'),
            torque_rate_limit_nm=_number(document.get('torque_rate_limit_nm'), 'torque_rate_limit_nm'),
            data_schema_version=str(document.get('data_schema_version', '')).strip(),
            assets={key: str(value) for key, value in assets.items()},
            bladed_project=str(document.get('bladed_project', '')).strip(),
            wind_data_glob=str(document.get('wind_data_glob', '')).strip(),
        )
        profile.validate(source=source)
        return profile

    def validate(self, *, source: str = '<memory>') -> None:
        if not self.turbine_id or not self.display_name:
            raise ValueError(f'turbine profile id/display_name are required: {source}')
        for name in ('rated_power_w', 'gear_ratio', 'cut_in_wind_mps', 'rated_wind_mps',
                     'max_wind_mps', 'power_ref_w', 'blade_root_ref_nm',
                     'pitch_rate_limit_rad', 'torque_rate_limit_nm'):
            if getattr(self, name) <= 0.0:
                raise ValueError(f'turbine profile {name} must be positive: {source}')
        if not (self.cut_in_wind_mps < self.rated_wind_mps <= self.max_wind_mps):
            raise ValueError(f'turbine profile wind range is invalid: {source}')
        for low_name, high_name in (
            ('action_low', 'action_high'),
            ('safe_action_low', 'safe_action_high'),
            ('action_delta_low', 'action_delta_high'),
        ):
            low = getattr(self, low_name)
            high = getattr(self, high_name)
            if any(lo > hi for lo, hi in zip(low, high)):
                raise ValueError(f'turbine profile {low_name} exceeds {high_name}: {source}')
        if any(lo < 0.0 for lo in self.action_low) or any(lo < 0.0 for lo in self.safe_action_low):
            raise ValueError(f'turbine profile action lower bounds must be non-negative: {source}')
        if not 0.0 < self.actor_residual_target_scale <= 1.0:
            raise ValueError(f'turbine profile actor_residual_target_scale must be in (0, 1]: {source}')
        if not self.data_schema_version:
            raise ValueError(f'turbine profile data_schema_version is required: {source}')
        if not self.bladed_project or not self.wind_data_glob:
            raise ValueError(f'turbine profile bladed project/wind glob are required: {source}')

    def asset_path(self, name: str, project_root: Path = PROJECT_ROOT) -> Path:
        try:
            return _resolve(self.assets[name], project_root)
        except KeyError as exc:
            raise KeyError(f'unknown turbine asset: {name}') from exc

    def assert_config_matches(self, config: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> None:
        mismatches: list[str] = []

        def compare(label: str, actual: Any, expected: Any) -> None:
            if isinstance(expected, list):
                if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                    mismatches.append(f'{label}: {actual!r} != {expected!r}')
                    return
                for index, (left, right) in enumerate(zip(actual, expected)):
                    try:
                        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
                            mismatches.append(f'{label}[{index}]: {left!r} != {right!r}')
                    except (TypeError, ValueError):
                        mismatches.append(f'{label}[{index}]: {left!r} != {right!r}')
                return
            if isinstance(expected, float):
                try:
                    if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
                        mismatches.append(f'{label}: {actual!r} != {expected!r}')
                except (TypeError, ValueError):
                    mismatches.append(f'{label}: {actual!r} != {expected!r}')
                return
            if actual != expected:
                mismatches.append(f'{label}: {actual!r} != {expected!r}')

        env = config.get('env') if isinstance(config.get('env'), dict) else {}
        control = config.get('control') if isinstance(config.get('control'), dict) else {}
        reward = config.get('reward') if isinstance(config.get('reward'), dict) else {}
        if not reward and isinstance(env.get('reward'), dict):
            reward = env['reward']
        bc = config.get('bc') if isinstance(config.get('bc'), dict) else {}
        predictor = config.get('load_predictor') if isinstance(config.get('load_predictor'), dict) else {}
        contract = config.get('state20_queue_contract') if isinstance(config.get('state20_queue_contract'), dict) else {}
        compare('env.rated_power', env.get('rated_power'), self.rated_power_w)
        compare('env.gear_ratio', env.get('gear_ratio'), self.gear_ratio)
        compare('env.action_low', env.get('action_low'), self.action_low)
        compare('env.action_high', env.get('action_high'), self.action_high)
        compare('env.action_delta_low', env.get('action_delta_low'), self.action_delta_low)
        compare('env.action_delta_high', env.get('action_delta_high'), self.action_delta_high)
        compare('control.safe_action_low', control.get('safe_action_low'), self.safe_action_low)
        compare('control.safe_action_high', control.get('safe_action_high'), self.safe_action_high)
        compare('control.residual_limit', control.get('residual_limit'), self.residual_limit)
        compare('control.actor_residual_target_scale', control.get('actor_residual_target_scale'),
                self.actor_residual_target_scale)
        compare('reward.power_ref', reward.get('power_ref'), self.power_ref_w)
        compare('reward.blade_root_ref', reward.get('blade_root_ref'), self.blade_root_ref_nm)
        compare('reward.pitch_rate_limit', reward.get('pitch_rate_limit'), self.pitch_rate_limit_rad)
        compare('reward.torque_rate_limit', reward.get('torque_rate_limit'), self.torque_rate_limit_nm)
        compare('reward.rated_wind_speed', reward.get('rated_wind_speed'), self.rated_wind_mps)
        compare('reward.cut_in_wind_speed', reward.get('cut_in_wind_speed'), self.cut_in_wind_mps)
        path_pairs = (
            ('control.region_aware_baseline_path', control.get('region_aware_baseline_path'), 'b4_baseline'),
            ('control.state_action_artifact_path', control.get('state_action_artifact_path'), 'state_action_envelope'),
            ('control.s2_u1_calibration_path', control.get('s2_u1_calibration_path'), 's2_u1_calibration'),
            ('bc.resume_checkpoint', bc.get('resume_checkpoint'), 'bc_checkpoint'),
            ('load_predictor.bundle_path', predictor.get('bundle_path'), 'shadow_bundle'),
            ('state20_queue_contract.paired_reference_path',
             contract.get('paired_reference_path'), 'paired_reference'),
        )
        for label, value, asset in path_pairs:
            if not isinstance(value, str) or not value.strip():
                mismatches.append(f'{label}: missing')
                continue
            actual_path = _resolve(value, project_root)
            expected_path = self.asset_path(asset, project_root)
            if actual_path.resolve() != expected_path.resolve():
                mismatches.append(f'{label}: {actual_path} != {expected_path}')
        if mismatches:
            raise ValueError('turbine profile/config mismatch:\n- ' + '\n- '.join(mismatches))


def load_turbine_profile_for_config(config: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> TurbineProfile | None:
    value = config.get('turbine_profile_path')
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('turbine_profile_path must be a non-empty string')
    return TurbineProfile.from_path(_resolve(value, project_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate a turbine profile, optionally against a run config.')
    parser.add_argument('--profile', required=True, help='Turbine profile JSON')
    parser.add_argument('--config', default='', help='Optional run config to compare')
    parser.add_argument('--json', action='store_true', help='Print profile JSON')
    args = parser.parse_args(argv)
    profile = TurbineProfile.from_path(args.profile)
    if args.json:
        print(json.dumps({
            'turbine_id': profile.turbine_id,
            'display_name': profile.display_name,
            'rated_power_w': profile.rated_power_w,
            'gear_ratio': profile.gear_ratio,
            'assets': profile.assets,
        }, indent=2, ensure_ascii=False))
    if args.config:
        config = load_config(args.config)
        if not isinstance(config, dict):
            raise SystemExit('config root must be an object')
        profile.assert_config_matches(config)
        print(f'turbine profile matches config: {profile.turbine_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
