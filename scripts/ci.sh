#!/usr/bin/env bash
# Local CI for the Linux side of the WT15 RL project.
# It never binds UDP and never starts Bladed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONDA_ENV="${CONDA_ENV:-wind}"
# M6 transition: legacy RUN_ID/receipt identity tests are expected to be red
# until M6B removes that mechanism.  Keep them runnable by unsetting this.
export WT15_M6_TRANSITION="${WT15_M6_TRANSITION:-1}"

echo "[ci] python/unittest"
conda run -n "$CONDA_ENV" python -m unittest discover -s RL/tests -p 'test_*.py' -q

echo "[ci] tool smoke"
conda run -n "$CONDA_ENV" python -m RL.preflight_validation --help >/dev/null
conda run -n "$CONDA_ENV" python -m RL.build_slot_checklist \
    --manifest RL/configs/physical_shadow_state20_v1_final_checked/state20_physical_4wind_4round_600s_training_manifest.json \
    --output /tmp/wt15_ci_checklist.md >/dev/null
conda run -n "$CONDA_ENV" python -m RL.report_post_run --help >/dev/null
conda run -n "$CONDA_ENV" python -m RL.run_meta --help >/dev/null
conda run -n "$CONDA_ENV" python -m RL.turbine_profile --help >/dev/null

echo "[ci] PASS"
