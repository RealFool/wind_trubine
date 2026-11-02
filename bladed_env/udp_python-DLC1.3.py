import csv
from datetime import datetime
from pathlib import Path
import socket
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.utils.constants import ACTION_VARIABLES, BLADED_UDP_VARIABLES
from RL.utils.power_fields import finite_float_or_none, synchronize_power_fields


HOST = '127.0.0.1'
PORT = 8888
CSV_HEADER = list(BLADED_UDP_VARIABLES)


def init_csv() -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'turbine_compact_{timestamp}.csv'
    with open(path, 'w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerow(CSV_HEADER)
    return path


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(1.0)
    csv_path = init_csv()
    expected_fields = len(CSV_HEADER)
    print(f'CSV created: {csv_path}')
    print(f'UDP fields: {expected_fields}; action labels: {", ".join(ACTION_VARIABLES)}')

    while True:
        try:
            data, _address = sock.recvfrom(8192)
            text = data.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            parts = [value.strip() for value in text.split(',')]
            if len(parts) != expected_fields:
                print(f'Dropped packet: fields={len(parts)}, expected={expected_fields}')
                continue

            raw_payload: dict[str, float | None] = {}
            for column, cell in zip(CSV_HEADER, parts):
                raw_payload[column] = finite_float_or_none(cell)
            synchronize_power_fields(raw_payload)
            payload = {column: 0.0 if value is None else float(value) for column, value in raw_payload.items()}

            with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow([payload[column] for column in CSV_HEADER])

            print(
                f"time={payload['Time']:.3f} | torque={payload[ACTION_VARIABLES[0]]:.2f} "
                f"| pitch={payload[ACTION_VARIABLES[1]]:.6f}"
            )
        except socket.timeout:
            continue
        except Exception as exc:
            print(f'Exception: {exc}')
            time.sleep(0.1)


if __name__ == '__main__':
    main()