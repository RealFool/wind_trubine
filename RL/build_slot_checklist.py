"""Generate a printable per-slot checklist from a queue manifest.

The checklist is intentionally operator-facing: one row per Bladed case reload,
with the fields that must be attested during a manual Windows run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from RL.utils.queue_manifest import QueueSlot, expand_queue_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 'slot_checklist_v1'


def _slot_row(index: int, slot: QueueSlot) -> str:
    case = slot.case_path or '（手工加载）'
    return (
        f"| {index} | {slot.round_id} | {slot.condition_instance_id} | "
        f"{slot.condition_id} | `{case}` | {slot.post_load_discard_sim_s:g} | "
        f"{slot.train_sim_s:g} | [ ] | [ ] | [ ] | [ ] |"
    )


def build_checklist(document: dict[str, Any], *, title: str | None = None) -> str:
    slots = expand_queue_manifest(document)
    rounds = Counter(slot.round_id for slot in slots)
    conditions = sorted({slot.condition_id for slot in slots})
    heading = title or document.get('state20_profile') or 'queue 运行检查表'
    lines = [
        f'# {heading}',
        '',
        f'- manifest: `{document.get("_manifest_path", "（未记录）")}`',
        f'- protocol: `{document.get("protocol_version")}`',
        f'- rounds: {len(rounds)}；slots: {len(slots)}；conditions: {len(conditions)}',
        f'- train_sim_s: {document.get("train_sim_s")}；execution_mode: `{document.get("execution_mode")}`',
        '',
        '> 每个 slot 必须确认：工程/风文件正确、从 0 跑到 600 s、收到 END、身份字段一致。'
        '任一异常立即停止并记录，不续跑、不覆盖。',
        '',
        '## 汇总',
        '',
        '| 轮次 | slots |',
        '|---|---:|',
    ]
    for round_id in sorted(rounds):
        lines.append(f'| {round_id} | {rounds[round_id]} |')
    lines.extend([
        '',
        '## 逐 slot 检查表',
        '',
        '| # | round | instance | condition | case | discard(s) | train(s) | 已加载 | 0–600s | END | 身份一致 |',
        '|---:|---:|---:|---:|---|---:|---:|:---:|:---:|:---:|:---:|',
    ])
    for index, slot in enumerate(slots, start=1):
        lines.append(_slot_row(index, slot))
    lines.extend([
        '',
        '## 结束确认',
        '',
        '- [ ] slot 数与 manifest 一致',
        '- [ ] 无 skip / retry / timeout',
        '- [ ] 每个 slot 的 END 已收到',
        '- [ ] 逐 slot 身份字段已记录',
        '- [ ] 运行目录、配置与 git commit 已记录',
        '',
    ])
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate a per-slot manual checklist.')
    parser.add_argument('--manifest', required=True, help='Queue manifest JSON')
    parser.add_argument('--output', default='', help='Write markdown here (default: stdout)')
    parser.add_argument('--title', default='', help='Checklist title')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise SystemExit('manifest must contain a JSON object')
    document['_manifest_path'] = str(manifest_path)
    text = build_checklist(document, title=args.title or None)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
        print(f'wrote {output}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
