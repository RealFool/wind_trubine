from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from RL.audit_pid_reference_provenance import build_provenance_audit


CONTROLLER_SOURCE = r'''
const double I_ElectricalTorque = GetMeasuredGeneratorTorque(turbine_id);
const double I_CollectivePitchPosition = GetCollectivePitchAngle(turbine_id);
const double I_PitchPosition1 = I_CollectivePitchPosition;
SetField(fields, kTorqueDemand, I_ElectricalTorque);
SetField(fields, kPitchDemand1, I_PitchPosition1);
SendToPython(udpFrame.c_str());
ReceiveMatchingReply(queueEpoch, conditionInstanceId, frameId, new_torque, new_pitch, replyKind);
SetDemandedGeneratorTorque(turbine_id, new_torque);
ApplyCollectivePitchDemand(turbine_id, new_pitch);
'''

API_HEADER = r'''
double GetDemandedGeneratorTorque(const turbine turbine_id);
double GetDemandedCollectivePitchAngle(const turbine turbine_id);
'''


class TestPidReferenceProvenance(unittest.TestCase):
    @staticmethod
    def _write_raw_files(data_root: Path, *, measurement_offset: float = 0.0) -> None:
        data_root.mkdir(parents=True)
        fieldnames = [
            'O_PcsTorqueDemand',
            'I_ElectricalTorque',
            'O_PitchPositionDemand1',
            'I_PitchPosition1',
        ]
        for condition_id in range(3, 26):
            with (data_root / f'data_windspeed{condition_id}.csv').open(
                'w', encoding='utf-8', newline=''
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        'O_PcsTorqueDemand': 20_000_000.0 + condition_id,
                        'I_ElectricalTorque': 20_000_000.0 + condition_id - measurement_offset,
                        'O_PitchPositionDemand1': 0.01 * condition_id,
                        'I_PitchPosition1': 0.01 * condition_id - measurement_offset,
                    }
                )

    def test_audit_rejects_measurement_alias_trace_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_reference_provenance_') as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / 'raw'
            controller_path = root / 'controller.cpp'
            api_header_path = root / 'ExternalControllerApi.h'
            self._write_raw_files(data_root)
            controller_path.write_text(CONTROLLER_SOURCE, encoding='utf-8')
            api_header_path.write_text(API_HEADER, encoding='utf-8')

            audit = build_provenance_audit(data_root, controller_path, api_header_path)

        self.assertEqual(audit['schema_version'], 'pid_reference_provenance_v1')
        self.assertEqual(audit['source_manifest']['file_count'], 23)
        self.assertEqual(audit['raw_comparison']['rows_compared'], 23)
        self.assertEqual(audit['raw_comparison']['torque']['exact_match_rows'], 23)
        self.assertEqual(audit['raw_comparison']['pitch']['exact_match_rows'], 23)
        self.assertTrue(audit['raw_comparison']['all_rows_exact_measurement_aliases'])
        self.assertEqual(
            audit['dll_static_evidence']['udp_assignment_sources'],
            {
                'O_PcsTorqueDemand': 'I_ElectricalTorque',
                'O_PitchPositionDemand1': 'I_PitchPosition1',
            },
        )
        self.assertTrue(audit['dll_static_evidence']['frame_sent_before_reply_received'])
        self.assertTrue(audit['dll_static_evidence']['reply_received_before_action_applied'])
        self.assertFalse(audit['decision']['native_pid_demand_available'])
        self.assertEqual(audit['decision']['status'], 'inconclusive')
        self.assertEqual(audit['decision']['udp_action_contract'], 'unresolved')
        self.assertEqual(audit['decision']['baseline_reference_path'], 'provenance_reconciliation_required')

    def test_audit_authorizes_recorded_pid_demand_trace_for_surrogate(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_reference_provenance_') as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / 'raw'
            controller_path = root / 'controller.cpp'
            api_header_path = root / 'ExternalControllerApi.h'
            self._write_raw_files(data_root, measurement_offset=0.001)
            controller_path.write_text(CONTROLLER_SOURCE, encoding='utf-8')
            api_header_path.write_text(API_HEADER, encoding='utf-8')

            audit = build_provenance_audit(data_root, controller_path, api_header_path)

        self.assertEqual(audit['decision']['status'], 'conclusive_recorded_pid_demand')
        self.assertFalse(audit['decision']['native_pid_demand_available'])
        self.assertTrue(audit['decision']['recorded_pid_demand_trace_available'])
        self.assertEqual(audit['decision']['udp_action_contract'], 'recorded_pid_demand')
        self.assertEqual(audit['decision']['baseline_reference_path'], 'region_aware_surrogate_required')


if __name__ == '__main__':
    unittest.main()