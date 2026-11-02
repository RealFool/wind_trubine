from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = 'persistent_power_deficit_offline_ablation_v1'
SCENARIOS = {
    'exact_pid': {'deficit_fraction': 0.0, 'duration_s': 0.0},
    'single_step_5pct': {'deficit_fraction': 0.05, 'duration_s': None},
    'sustained_5pct_10s': {'deficit_fraction': 0.05, 'duration_s': 10.0},
    'sustained_10pct_60s': {'deficit_fraction': 0.10, 'duration_s': 60.0},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_reference_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_power_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    powers: list[float] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'Time', 'Electrical power'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name} is missing R1 ablation fields: {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row['Time'])
                power = float(row['Electrical power'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path.name} row {row_number} has invalid R1 ablation data') from exc
            if not math.isfinite(timestamp) or not math.isfinite(power):
                raise ValueError(f'{path.name} row {row_number} has non-finite R1 ablation data')
            if times and timestamp <= times[-1]:
                raise ValueError(f'{path.name} timestamps must be strictly increasing')
            times.append(timestamp)
            powers.append(power)
    if len(times) < 3:
        raise ValueError(f'{path.name} needs at least three rows for R1 ablation')
    return np.asarray(times, dtype=np.float64), np.asarray(powers, dtype=np.float64)


def _scenario_deficit(
    times: np.ndarray,
    reference_power: np.ndarray,
    scenario: dict[str, float | None],
) -> np.ndarray:
    deficit = np.zeros(reference_power.size - 1, dtype=np.float64)
    fraction = float(scenario['deficit_fraction'] or 0.0)
    if fraction <= 0.0:
        return deficit
    start_time = float(times[0] + 0.5 * (times[-1] - times[0]))
    start_index = int(np.searchsorted(times[1:], start_time, side='left'))
    duration_s = scenario['duration_s']
    if duration_s is None:
        end_index = min(start_index + 1, deficit.size)
    else:
        end_index = int(np.searchsorted(times[1:], start_time + float(duration_s), side='left'))
        end_index = max(start_index + 1, min(end_index, deficit.size))
    deficit[start_index:end_index] = fraction * np.maximum(reference_power[1:][start_index:end_index], 0.0)
    return deficit


def _ewma_terms(
    times: np.ndarray,
    reference_power: np.ndarray,
    injected_deficit: np.ndarray,
    *,
    rated_power_w: float,
    deadband_fraction: float,
    tau_s: float,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    terms = np.zeros_like(injected_deficit)
    ewma_values = np.zeros_like(injected_deficit)
    ewma = 0.0
    deadband_w = deadband_fraction * rated_power_w
    for index, delta_t_s in enumerate(np.diff(times)):
        raw_deficit = max(float(injected_deficit[index]) - deadband_w, 0.0)
        alpha = -math.expm1(-float(delta_t_s) / tau_s)
        ewma += alpha * (raw_deficit - ewma)
        ewma_values[index] = ewma
        terms[index] = float(np.clip(ewma / rated_power_w, 0.0, clip))
    return ewma_values, terms


def evaluate_persistent_power_deficit(
    manifest_path: Path,
    *,
    rated_power_w: float,
    weight: float,
    deadband_fraction: float,
    tau_s: float,
    clip: float,
) -> dict[str, Any]:
    for name, value in (
        ('rated_power_w', rated_power_w),
        ('weight', weight),
        ('deadband_fraction', deadband_fraction),
        ('tau_s', tau_s),
        ('clip', clip),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')
    if rated_power_w <= 0.0 or tau_s <= 0.0 or clip <= 0.0:
        raise ValueError('rated_power_w, tau_s, and clip must be positive')
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    references = manifest.get('references')
    if not isinstance(references, list) or not references:
        raise ValueError('Paired PID manifest contains no references')

    scenario_reports: dict[str, Any] = {}
    for scenario_name, scenario in SCENARIOS.items():
        all_terms: list[np.ndarray] = []
        all_rewards: list[np.ndarray] = []
        condition_reports: list[dict[str, Any]] = []
        for descriptor in references:
            path = _resolve_reference_path(str(descriptor['path']))
            if _sha256(path) != str(descriptor['sha256']):
                raise ValueError(f'{path.name} SHA-256 does not match paired PID manifest')
            times, reference_power = _read_power_trace(path)
            deficit = _scenario_deficit(times, reference_power, scenario)
            ewma, terms = _ewma_terms(
                times,
                reference_power,
                deficit,
                rated_power_w=rated_power_w,
                deadband_fraction=deadband_fraction,
                tau_s=tau_s,
                clip=clip,
            )
            rewards = -weight * terms
            peak_index = int(np.argmax(terms))
            condition_reports.append({
                'condition_id': int(descriptor['condition_id']),
                'transition_count': int(terms.size),
                'injected_deficit_energy_j': float(np.sum(deficit * np.diff(times))),
                'term_mean': float(np.mean(terms)),
                'term_p95': float(np.quantile(terms, 0.95)),
                'term_max': float(np.max(terms)),
                'term_cumulative': float(np.sum(terms)),
                'reward_cumulative': float(np.sum(rewards)),
                'clipping_fraction': float(np.mean(terms >= clip - 1.0e-12)),
                'peak_ewma_deficit_w': float(ewma[peak_index]),
                'terminal_ewma_deficit_w': float(ewma[-1]),
            })
            all_terms.append(terms)
            all_rewards.append(rewards)
        terms = np.concatenate(all_terms)
        rewards = np.concatenate(all_rewards)
        scenario_reports[scenario_name] = {
            'definition': scenario,
            'transition_count': int(terms.size),
            'term_mean': float(np.mean(terms)),
            'term_p95': float(np.quantile(terms, 0.95)),
            'term_max': float(np.max(terms)),
            'term_cumulative': float(np.sum(terms)),
            'reward_cumulative': float(np.sum(rewards)),
            'clipping_fraction': float(np.mean(terms >= clip - 1.0e-12)),
            'conditions': condition_reports,
        }

    exact = scenario_reports['exact_pid']
    transient = scenario_reports['single_step_5pct']
    sustained_short = scenario_reports['sustained_5pct_10s']
    sustained_long = scenario_reports['sustained_10pct_60s']
    checks = {
        'exact_pid_zero': exact['term_max'] == 0.0 and exact['reward_cumulative'] == 0.0,
        'single_step_response_bounded': transient['term_max'] < sustained_short['term_max'],
        'duration_monotonic': sustained_short['term_cumulative'] < sustained_long['term_cumulative'],
        'penalty_sign_correct': sustained_short['reward_cumulative'] < 0.0 and sustained_long['reward_cumulative'] < 0.0,
        'no_clipping_at_candidate_weighting': sustained_long['clipping_fraction'] == 0.0,
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'validation_scope': 'offline_synthetic_deficit_sensitivity_on_real_paired_pid_power_traces',
        'validation_status': 'passed' if all(checks.values()) else 'failed',
        'formal_closed_loop_acceptance': False,
        'manifest_path': str(manifest_path),
        'manifest_sha256': _sha256(manifest_path),
        'input_manifest_sha256': manifest['metadata']['input_manifest_sha256'],
        'condition_count': len(references),
        'parameters': {
            'rated_power_w': rated_power_w,
            'persistent_power_deficit_weight': weight,
            'persistent_power_deficit_deadband_fraction': deadband_fraction,
            'persistent_power_deficit_ewma_tau_s': tau_s,
            'persistent_power_deficit_clip': clip,
        },
        'checks': checks,
        'scenarios': scenario_reports,
        'limitation': (
            'Deficits are injected into recorded PID power traces. This validates reward sensitivity '
            'and temporal discrimination, not policy-induced closed-loop energy or safety.'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the offline persistent power-deficit reward ablation.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rated-power-w', type=float, default=15_000_000.0)
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--deadband-fraction', type=float, default=0.01)
    parser.add_argument('--tau-s', type=float, default=2.0)
    parser.add_argument('--clip', type=float, default=1.0)
    args = parser.parse_args()
    document = evaluate_persistent_power_deficit(
        args.manifest,
        rated_power_w=args.rated_power_w,
        weight=args.weight,
        deadband_fraction=args.deadband_fraction,
        tau_s=args.tau_s,
        clip=args.clip,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'validation_status': document['validation_status'],
        'condition_count': document['condition_count'],
        'checks': document['checks'],
        'scenario_aggregate': {
            name: {
                'term_mean': report['term_mean'],
                'term_p95': report['term_p95'],
                'term_max': report['term_max'],
                'term_cumulative': report['term_cumulative'],
                'reward_cumulative': report['reward_cumulative'],
                'clipping_fraction': report['clipping_fraction'],
            }
            for name, report in document['scenarios'].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()