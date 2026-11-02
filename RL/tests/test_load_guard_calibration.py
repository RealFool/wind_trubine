from __future__ import annotations

import unittest

from RL.utils.load_guard import select_candidate_residual_scale
from RL.utils.load_guard_calibration import (
    LoadGuardCalibration,
    LoadGuardCalibrationConfig,
    LoadGuardCalibrationObservation,
    audit_load_guard_coverage,
    calibrate_load_guard,
)
from RL.utils.load_predictor_contract import LoadPredictorInterface


class TestLoadGuardCalibration(unittest.TestCase):
    @staticmethod
    def _interface() -> LoadPredictorInterface:
        return LoadPredictorInterface.from_path('RL/data/load_predictor_interface_v2.json')

    def _observations(
        self,
        *,
        plan_mismatch: float = 0.0,
        condition_id: int = 16,
        timestamp_start_s: float = 101.0,
    ) -> list[LoadGuardCalibrationObservation]:
        values: list[LoadGuardCalibrationObservation] = []
        for blade in (1, 2, 3):
            for index in range(20):
                values.append(LoadGuardCalibrationObservation(
                    blade=blade,
                    optimization_state='above_rated',
                    horizon_s=0.1,
                    plan_kind='zoh_candidate',
                    predicted_ratio=0.90,
                    actual_ratio=0.92,
                    plan_mismatch=plan_mismatch,
                    timestamp_s=timestamp_start_s + index,
                    condition_id=condition_id,
                ))
        return values

    def test_calibration_quantile_is_not_its_own_coverage_authorization(self) -> None:
        artifact = calibrate_load_guard(
            self._observations(),
            config=LoadGuardCalibrationConfig(
                coverage_target=0.90,
                confidence_level=0.90,
                minimum_samples_per_group=10,
                max_plan_mismatch=0.05,
            ),
            predictor_training_end_s=50.0,
            calibration_start_s=100.0,
            interface=self._interface(),
        )
        calibrated = LoadGuardCalibration(artifact)
        group = calibrated.group(1, 'above_rated', 0.1, 'zoh_candidate')

        self.assertEqual(group['eligible_sample_count'], 20)
        self.assertAlmostEqual(group['conformal_quantile'], 0.02, places=12)
        self.assertEqual(group['calibration_in_sample_coverage'], 1.0)
        self.assertIsNone(group['calibration_coverage_lower_confidence_bound'])
        self.assertFalse(group['soft_guard_eligible'])
        with self.assertRaisesRegex(ValueError, 'no independent coverage audit authorization'):
            calibrated.validate_soft_groups([(1, 'above_rated', 0.1)])

    def test_independent_coverage_audit_is_required_and_population_gates_soft_authorization(self) -> None:
        calibration = calibrate_load_guard(
            self._observations(condition_id=16, timestamp_start_s=101.0),
            config=LoadGuardCalibrationConfig(
                coverage_target=0.90,
                confidence_level=0.90,
                minimum_samples_per_group=10,
                max_plan_mismatch=0.05,
            ),
            predictor_training_end_s=50.0,
            calibration_start_s=100.0,
            interface=self._interface(),
            predictor_artifact_sha256='a' * 64,
        )
        audit_rows = self._observations(condition_id=17, timestamp_start_s=201.0)
        pid_audit = audit_load_guard_coverage(
            calibration,
            audit_rows,
            audit_start_s=200.0,
            interface=self._interface(),
            predictor_artifact_sha256='a' * 64,
            coverage_population='pid_holdout',
            strict_identity_verified=False,
        )
        self.assertTrue(pid_audit['metadata']['coverage_audit_all_candidate_groups_pass'])
        self.assertFalse(pid_audit['metadata']['soft_guard_authorized'])

        strict_rl_audit = audit_load_guard_coverage(
            calibration,
            audit_rows,
            audit_start_s=200.0,
            interface=self._interface(),
            predictor_artifact_sha256='a' * 64,
            coverage_population='rl_shadow_strict',
            strict_identity_verified=True,
        )
        self.assertTrue(strict_rl_audit['metadata']['soft_guard_authorized'])
        LoadGuardCalibration(strict_rl_audit).validate_soft_groups([(1, 'above_rated', 0.1)])

    def test_soft_group_rejected_when_plan_mismatch_exceeds_limit(self) -> None:
        calibration = calibrate_load_guard(
            self._observations(plan_mismatch=0.2, condition_id=16, timestamp_start_s=101.0),
            config=LoadGuardCalibrationConfig(
                coverage_target=0.90,
                confidence_level=0.90,
                minimum_samples_per_group=10,
                max_plan_mismatch=0.05,
            ),
            predictor_training_end_s=50.0,
            calibration_start_s=100.0,
            interface=self._interface(),
            predictor_artifact_sha256='b' * 64,
        )
        artifact = audit_load_guard_coverage(
            calibration,
            self._observations(plan_mismatch=0.2, condition_id=17, timestamp_start_s=201.0),
            audit_start_s=200.0,
            interface=self._interface(),
            predictor_artifact_sha256='b' * 64,
            coverage_population='rl_shadow_strict',
            strict_identity_verified=True,
        )
        with self.assertRaisesRegex(ValueError, 'no independent coverage audit authorization'):
            LoadGuardCalibration(artifact).validate_soft_groups(
                [(1, 'above_rated', 0.1)]
            )

    def test_candidate_scale_guard_rejects_unreviewed_calibration(self) -> None:
        calibration = LoadGuardCalibration(calibrate_load_guard(
            self._observations(condition_id=16, timestamp_start_s=101.0),
            config=LoadGuardCalibrationConfig(
                coverage_target=0.90,
                confidence_level=0.50,
                minimum_samples_per_group=10,
                max_plan_mismatch=0.05,
            ),
            predictor_training_end_s=50.0,
            calibration_start_s=100.0,
            interface=self._interface(),
            predictor_artifact_sha256='c' * 64,
        ))
        with self.assertRaisesRegex(ValueError, 'no independent coverage audit authorization'):
            select_candidate_residual_scale(
                calibration,
                optimization_state='above_rated',
                horizon_s=0.1,
                predicted_peak_ratio_by_scale={
                    1.0: [1.1, 1.1, 1.1],
                    0.5: [0.98, 0.98, 0.98],
                    0.25: [0.95, 0.95, 0.95],
                    0.0: [0.94, 0.94, 0.94],
                },
            )


if __name__ == '__main__':
    unittest.main()
