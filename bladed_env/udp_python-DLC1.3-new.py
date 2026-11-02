from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import socket
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.constants import ACTION_VARIABLES, BLADED_UDP_VARIABLES
from RL.utils.power_fields import finite_float_or_none, synchronize_power_fields


DEFAULT_BIND_HOST = '192.168.56.1'
DEFAULT_PORT = 8888
CSV_HEADER = list(BLADED_UDP_VARIABLES)
QUEUE_UDP_PROTOCOL_VERSION_V1 = 'bladed_udp_19_si_my_queue_v1'
QUEUE_UDP_PROTOCOL_VERSION = 'bladed_udp_19_si_my_queue_v2'


@dataclass(frozen=True)
class QueueEnvelope:
    protocol_version: str
    queue_epoch: int
    condition_instance_id: int
    condition_id: int
    phase: str
    phase_elapsed_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Record 19-field Bladed UDP frames and optionally return a hold action.'
    )
    parser.add_argument(
        '--bind-host',
        default=DEFAULT_BIND_HOST,
        help='Local address to listen on; use the host-only adapter IP to restrict the listener.',
    )
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='UDP port to listen on.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('bladed_result/raw/udp_connectivity_test'),
        help='Directory for the recorded compact CSV file.',
    )
    parser.add_argument(
        '--reply-current-action',
        action='store_true',
        help='Return the packet action fields with the matching frame ID to verify the reverse UDP path.',
    )
    parser.add_argument(
        '--allow-legacy-frame',
        action='store_true',
        help='Accept a legacy bare 19-field payload without the formal frame_id,sim_time envelope.',
    )
    parser.add_argument(
        '--require-condition-metadata',
        action='store_true',
        help='Require the v7 queue envelope and return epoch/condition/frame-matched hold actions.',
    )
    parser.add_argument(
        '--max-packets',
        type=int,
        default=0,
        help='Stop after this many valid frames; zero keeps listening indefinitely.',
    )
    parser.add_argument(
        '--show-packet-prefix',
        action='store_true',
        help='Print the field count and first seven fields when a packet is dropped.',
    )
    return parser.parse_args()


def init_csv(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = output_dir / f'turbine_compact_{timestamp}.csv'
    with open(path, 'w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerow(CSV_HEADER)
    return path


def _parse_nonnegative_int(value: str, field_name: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f'Invalid {field_name}: {value!r}') from exc
    if not math.isfinite(parsed) or parsed < 0.0 or not parsed.is_integer():
        raise ValueError(f'Invalid {field_name}: {value!r}')
    return int(parsed)


def parse_frame(
    text: str,
    require_frame_id: bool,
    require_condition_metadata: bool = False,
) -> tuple[dict[str, float], int | None, float, QueueEnvelope | None]:
    parts = [value.strip() for value in text.split(',')]
    expected_fields = len(CSV_HEADER)
    frame_id: int | None = None
    sim_time: float | None = None
    queue_envelope: QueueEnvelope | None = None
    data_fields = parts
    if len(parts) == expected_fields + 8:
        try:
            protocol_version = parts[0]
            if protocol_version != QUEUE_UDP_PROTOCOL_VERSION:
                raise ValueError(f'Unsupported queue protocol envelope: {protocol_version!r}')
            queue_epoch = _parse_nonnegative_int(parts[1], 'queue_epoch')
            condition_instance_id = _parse_nonnegative_int(parts[2], 'condition_instance_id')
            condition_id = _parse_nonnegative_int(parts[3], 'condition_id')
            phase = parts[4].upper()
            if phase not in {'SETTLE', 'TRAIN', 'END'}:
                raise ValueError(f'Invalid phase: {parts[4]!r}')
            phase_elapsed_s = float(parts[5])
            if not math.isfinite(phase_elapsed_s) or phase_elapsed_s < 0.0:
                raise ValueError(f'Invalid phase_elapsed_s: {parts[5]!r}')
            frame_id = _parse_nonnegative_int(parts[6], 'frame_id')
            sim_time = float(parts[7])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[7]!r}')
            queue_envelope = QueueEnvelope(
                protocol_version=protocol_version,
                queue_epoch=queue_epoch,
                condition_instance_id=condition_instance_id,
                condition_id=condition_id,
                phase=phase,
                phase_elapsed_s=phase_elapsed_s,
            )
            data_fields = parts[8:]
        except ValueError as exc:
            raise ValueError('Invalid queue envelope') from exc
    elif len(parts) == expected_fields + 7:
        try:
            queue_epoch = _parse_nonnegative_int(parts[0], 'queue_epoch')
            condition_instance_id = _parse_nonnegative_int(parts[1], 'condition_instance_id')
            condition_id = _parse_nonnegative_int(parts[2], 'condition_id')
            phase = parts[3].upper()
            if phase not in {'SETTLE', 'TRAIN', 'END'}:
                raise ValueError(f'Invalid phase: {parts[3]!r}')
            phase_elapsed_s = float(parts[4])
            if not math.isfinite(phase_elapsed_s) or phase_elapsed_s < 0.0:
                raise ValueError(f'Invalid phase_elapsed_s: {parts[4]!r}')
            frame_id = _parse_nonnegative_int(parts[5], 'frame_id')
            sim_time = float(parts[6])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[6]!r}')
            queue_envelope = QueueEnvelope(
                protocol_version=QUEUE_UDP_PROTOCOL_VERSION_V1,
                queue_epoch=queue_epoch,
                condition_instance_id=condition_instance_id,
                condition_id=condition_id,
                phase=phase,
                phase_elapsed_s=phase_elapsed_s,
            )
            data_fields = parts[7:]
        except ValueError as exc:
            raise ValueError('Invalid queue envelope') from exc
    elif len(parts) == expected_fields + 2:
        try:
            frame_id = _parse_nonnegative_int(parts[0], 'frame_id')
            sim_time = float(parts[1])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[1]!r}')
            data_fields = parts[2:]
        except ValueError as exc:
            raise ValueError('Invalid frame_id,sim_time envelope') from exc
    elif len(parts) != expected_fields:
        raise ValueError(
            f'fields={len(parts)}, expected payload fields={expected_fields}, '
            f'{expected_fields + 2}, {expected_fields + 7}, or {expected_fields + 8}'
        )

    if require_frame_id and frame_id is None:
        raise ValueError('A formal frame_id,sim_time envelope is required')
    if require_condition_metadata and queue_envelope is None:
        raise ValueError('A v7 queue envelope is required')

    raw_payload: dict[str, float | None] = {}
    for column, cell in zip(CSV_HEADER, data_fields):
        raw_payload[column] = finite_float_or_none(cell)
    synchronize_power_fields(raw_payload)
    payload = {column: 0.0 if value is None else float(value) for column, value in raw_payload.items()}
    return payload, frame_id, float(payload['Time'] if sim_time is None else sim_time), queue_envelope


def build_hold_reply(
    payload: dict[str, float],
    frame_id: int | None,
    queue_envelope: QueueEnvelope | None = None,
) -> bytes:
    torque = payload[ACTION_VARIABLES[0]]
    pitch = payload[ACTION_VARIABLES[1]]
    if queue_envelope is not None:
        if frame_id is None:
            raise ValueError('A queue hold reply requires frame_id')
        return (
            f'{queue_envelope.queue_epoch},{queue_envelope.condition_instance_id},'
            f'{frame_id},{torque:.6f},{pitch:.6f}'
        ).encode('ascii')
    if frame_id is not None:
        return f'{frame_id},{torque:.6f},{pitch:.6f}'.encode('ascii')
    return f'{torque:.6f},{pitch:.6f}'.encode('ascii')


def main() -> None:
    args = parse_args()
    if args.port <= 0 or args.port > 65535:
        raise SystemExit(f'Invalid UDP port: {args.port}')
    if args.max_packets < 0:
        raise SystemExit('--max-packets must be zero or greater')

    csv_path = init_csv(args.output_dir)
    expected_fields = len(CSV_HEADER)
    valid_packets = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((args.bind_host, args.port))
        except OSError as exc:
            raise SystemExit(f'Cannot bind UDP {args.bind_host}:{args.port}: {exc}') from exc

        print(f'CSV created: {csv_path}')
        print(
            f'Listening on UDP {args.bind_host}:{args.port}; expected payload fields={expected_fields}; '
            f'require_frame_id={not args.allow_legacy_frame}; reply_current_action={args.reply_current_action}'
        )

        while True:
            data, address = sock.recvfrom(8192)
            text = data.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            try:
                payload, frame_id, sim_time, queue_envelope = parse_frame(
                    text,
                    require_frame_id=not args.allow_legacy_frame,
                    require_condition_metadata=args.require_condition_metadata,
                )
            except ValueError as exc:
                packet_details = ''
                if args.show_packet_prefix:
                    fields = [value.strip() for value in text.split(',')]
                    packet_details = f'; fields={len(fields)} prefix={fields[:7]!r}'
                print(
                    f'Dropped packet from {address[0]}:{address[1]}: '
                    f'{exc}{packet_details}'
                )
                continue

            with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow([payload[column] for column in CSV_HEADER])

            valid_packets += 1
            print(
                f'frame={frame_id if frame_id is not None else valid_packets} peer={address[0]}:{address[1]} '
                f'sim_time={sim_time:.3f} payload_time={payload["Time"]:.3f} '
                f'queue_epoch={queue_envelope.queue_epoch if queue_envelope is not None else "legacy"} '
                f'queue_protocol={queue_envelope.protocol_version if queue_envelope is not None else "legacy"} '
                f'condition={queue_envelope.condition_id if queue_envelope is not None else "legacy"} '
                f'phase={queue_envelope.phase if queue_envelope is not None else "legacy"} '
                f"| torque={payload[ACTION_VARIABLES[0]]:.2f} "
                f"| pitch={payload[ACTION_VARIABLES[1]]:.6f}"
            )
            if args.reply_current_action:
                sock.sendto(build_hold_reply(payload, frame_id, queue_envelope), address)
                print(f'Replied hold action for frame {frame_id} to {address[0]}:{address[1]}')

            if args.max_packets and valid_packets >= args.max_packets:
                print(f'Received {valid_packets} valid frame(s); exiting.')
                return


if __name__ == '__main__':
    main()
