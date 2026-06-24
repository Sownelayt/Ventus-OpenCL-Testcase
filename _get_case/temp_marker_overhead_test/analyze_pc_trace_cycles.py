#!/usr/bin/env python3
"""
Analyze non-intrusive phase cycles from GVM retire logs.

This script does not require any marker instruction in the kernel. It finds the
natural phase boundaries in the generated object and then reads GVM retire
timestamps for those PCs:

  movement_start: instruction after get_local_id returns
  compute_start : instruction after the configured movement barrier
  writeback_start: instruction after the configured compute barrier
  behavior_end  : instruction after the configured writeback barrier

Use it after running the plain variant:

  python3 analyze_pc_trace_cycles.py \
    --log log/marker_overhead_..._plain_r16_c16.log \
    --run-dir log/marker_overhead_run_..._plain_r16_c16

The kernel remains untouched; all cycle data comes from GVM stdout and the
compiled object.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

GVM_CYCLE_TIME_UNITS = 5
DEFAULT_KERNEL_SYMBOL = "marker_overhead_manual_kernel"
GET_LOCAL_ID_SYMBOL = "_Z12get_local_idj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="GVM child log path")
    parser.add_argument("--run-dir", required=True, help="child run directory containing object0.riscv")
    parser.add_argument(
        "--objdump",
        default="../../../install/bin/llvm-objdump",
        help="llvm-objdump path relative to this test directory",
    )
    parser.add_argument(
        "--kernel-symbol",
        default=DEFAULT_KERNEL_SYMBOL,
        help="kernel symbol to analyze",
    )
    parser.add_argument(
        "--barrier-indices",
        default="1,2,3",
        help="1-based barrier indices for compute_start, writeback_start, behavior_end",
    )
    parser.add_argument(
        "--program-index",
        type=int,
        default=0,
        help="GVM PROGRAM index whose PMU active cycles should be reported",
    )
    return parser.parse_args()


def run_objdump(objdump: str, obj: Path) -> str:
    return subprocess.check_output(
        [objdump, "-d", "--mattr=+v,+zfinx", str(obj)],
        text=True,
    )


INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){4}\s*(.*)$")
SYMBOL_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:")


def extract_kernel_disasm(dump: str, kernel_symbol: str) -> list[tuple[int, str]]:
    in_kernel = False
    rows: list[tuple[int, str]] = []
    for line in dump.splitlines():
        symbol = SYMBOL_RE.match(line)
        if symbol:
            name = symbol.group(2)
            if name == kernel_symbol:
                in_kernel = True
                continue
            if in_kernel and not name.startswith("."):
                break
        if not in_kernel:
            continue
        insn = INSN_RE.match(line)
        if insn:
            rows.append((int(insn.group(1), 16), insn.group(2).strip()))
    if not rows:
        raise SystemExit(f"failed to find symbol {kernel_symbol}")
    return rows


def parse_barrier_indices(text: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --barrier-indices: {text}") from exc
    if len(values) != 3 or any(v <= 0 for v in values):
        raise SystemExit("--barrier-indices must contain exactly three positive 1-based indices")
    return values


def find_boundaries(rows: list[tuple[int, str]], barrier_indices: list[int]) -> dict[str, int]:
    movement_start = None
    for idx, (_, text) in enumerate(rows):
        if GET_LOCAL_ID_SYMBOL in text:
            if idx + 1 >= len(rows):
                raise SystemExit("get_local_id call is last instruction")
            movement_start = rows[idx + 1][0]
            break
    if movement_start is None:
        raise SystemExit("failed to find get_local_id return boundary")

    barrier_pcs = [pc for pc, text in rows if text.startswith("barrier")]
    needed = max(barrier_indices)
    if len(barrier_pcs) < needed:
        raise SystemExit(f"expected at least {needed} barriers, found {len(barrier_pcs)}")

    pcs = [pc for pc, _ in rows]
    pc_set = set(pcs)

    def next_pc_after(pc: int) -> int:
        candidate = pc + 4
        if candidate in pc_set:
            return candidate
        for row_pc in pcs:
            if row_pc > pc:
                return row_pc
        raise SystemExit(f"no instruction after barrier pc 0x{pc:x}")

    return {
        "movement_start": movement_start,
        "compute_start": next_pc_after(barrier_pcs[barrier_indices[0] - 1]),
        "writeback_start": next_pc_after(barrier_pcs[barrier_indices[1] - 1]),
        "behavior_end": next_pc_after(barrier_pcs[barrier_indices[2] - 1]),
    }


RETIRE_RE = re.compile(r"@([0-9]+).*pc:\s+0x([0-9a-fA-F]+)")
def program_cycles_re(program_index: int) -> re.Pattern[str]:
    return re.compile(
        rf"\[PROGRAM\s+{program_index}\]\s+\[INST\+CYCLE\]\s+active cycles\s+:\s+([0-9]+)"
    )


def parse_log(log: Path, boundary_pcs: set[int], program_index: int) -> tuple[dict[int, int], int | None]:
    first_time: dict[int, int] = {}
    program_cycles: int | None = None
    pmu_re = program_cycles_re(program_index)
    for line in log.read_text(errors="replace").splitlines():
        pmu = pmu_re.search(line)
        if pmu:
            program_cycles = int(pmu.group(1))
        m = RETIRE_RE.search(line)
        if not m:
            continue
        t = int(m.group(1))
        pc = int(m.group(2), 16)
        if pc in boundary_pcs and pc not in first_time:
            first_time[pc] = t
    return first_time, program_cycles


def cycles_between(times: dict[int, int], a: int, b: int) -> int:
    if a not in times or b not in times:
        missing = [f"0x{pc:x}" for pc in (a, b) if pc not in times]
        raise SystemExit("missing retire timestamp for " + ", ".join(missing))
    return (times[b] - times[a]) // GVM_CYCLE_TIME_UNITS


def main() -> int:
    args = parse_args()
    log = Path(args.log)
    run_dir = Path(args.run_dir)
    obj = run_dir / "object0.riscv"
    if not log.exists():
        raise SystemExit(f"log not found: {log}")
    if not obj.exists():
        raise SystemExit(f"object not found: {obj}")

    rows = extract_kernel_disasm(run_objdump(args.objdump, obj), args.kernel_symbol)
    boundaries = find_boundaries(rows, parse_barrier_indices(args.barrier_indices))
    times, program_cycles = parse_log(log, set(boundaries.values()), args.program_index)

    movement = cycles_between(times, boundaries["movement_start"], boundaries["compute_start"])
    compute = cycles_between(times, boundaries["compute_start"], boundaries["writeback_start"])
    writeback = cycles_between(times, boundaries["writeback_start"], boundaries["behavior_end"])
    total = cycles_between(times, boundaries["movement_start"], boundaries["behavior_end"])

    print("# Non-Intrusive PC Trace Cycle Analysis")
    print()
    print("| boundary | pc | retire_time |")
    print("|---|---:|---:|")
    for name, pc in boundaries.items():
        print(f"| `{name}` | `0x{pc:08x}` | {times.get(pc, 'NA')} |")
    print()
    print("| phase | cycles |")
    print("|---|---:|")
    print(f"| movement | {movement} |")
    print(f"| compute | {compute} |")
    print(f"| writeback | {writeback} |")
    print(f"| behavior_total | {total} |")
    if program_cycles is not None:
        print(f"| kernel_pmu_cycles | {program_cycles} |")
        print(f"| prologue_epilogue_and_runtime_gap | {program_cycles - total} |")
    print()
    print(f"sum_check={movement + compute + writeback} total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
