#!/usr/bin/env python3
"""Check Ventus TMA movement cycles against the accepted GVM baseline.

The movement profile emits one MOVE_CHILD_DONE record per manual or DMA path.
This checker keeps only the five DMA paths, requires an exact key match with
the backend-specific CSV baseline, and enforces both aggregate and worst-point
cycle-regression limits.

Usage:
  python3 check_tma_perf.py --backend gvm --log run.log

Update tma_perf_reference.csv only after correctness gates and a reviewed full
sweep pass. Manual-path cycles are deliberately not golden performance data.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DMA_PATHS = {
    "bulk_g2s",
    "tensor_g2s",
    "bulk_s2g",
    "tensor_s2g",
    "tensor_g2s_s2g",
}

RESULT_RE = re.compile(
    r"^MOVE_CHILD_DONE status=PASS mode=(?P<mode>\S+) "
    r"path=(?P<path>\S+) rows=(?P<rows>\d+) cols=(?P<cols>\d+) "
    r"cycles=(?P<cycles>\d+)"
)


def key(mode: str, path: str, rows: int, cols: int) -> tuple[str, str, int, int]:
    return mode, path, rows, cols


def read_results(path: Path) -> dict[tuple[str, str, int, int], int]:
    results: dict[tuple[str, str, int, int], int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.match(line)
        if not match or match.group("path") not in DMA_PATHS:
            continue
        item = key(
            match.group("mode"),
            match.group("path"),
            int(match.group("rows")),
            int(match.group("cols")),
        )
        if item in results:
            raise ValueError(f"duplicate DMA result: {item}")
        results[item] = int(match.group("cycles"))
    return results


def read_reference(
    path: Path, backend: str
) -> dict[tuple[str, str, int, int], int]:
    reference: dict[tuple[str, str, int, int], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["backend"] != backend:
                continue
            item = key(
                row["mode"], row["path"], int(row["rows"]), int(row["cols"])
            )
            reference[item] = int(row["cycles"])
    return reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("tma_perf_reference.csv"),
    )
    parser.add_argument("--avg-limit", type=float, default=0.10)
    parser.add_argument("--point-limit", type=float, default=0.20)
    args = parser.parse_args()

    try:
        observed = read_results(args.log)
        reference = read_reference(args.reference, args.backend)
    except (OSError, KeyError, ValueError) as error:
        print(f"TMA_PERF_GATE FAIL: {error}", file=sys.stderr)
        return 2

    if not reference:
        print(
            f"TMA_PERF_GATE FAIL: no reference rows for {args.backend}",
            file=sys.stderr,
        )
        return 2
    missing = sorted(set(reference) - set(observed))
    extra = sorted(set(observed) - set(reference))
    if missing or extra:
        print(f"TMA_PERF_GATE FAIL: missing={missing} extra={extra}", file=sys.stderr)
        return 2

    ratios = {item: observed[item] / reference[item] for item in reference}
    average = sum(ratios.values()) / len(ratios)
    worst_key = max(ratios, key=ratios.get)
    worst = ratios[worst_key]
    status = (
        "PASS"
        if average <= 1.0 + args.avg_limit and worst <= 1.0 + args.point_limit
        else "FAIL"
    )
    print(
        f"TMA_PERF_GATE {status} backend={args.backend} points={len(ratios)} "
        f"avg_ratio={average:.4f} max_ratio={worst:.4f} "
        f"max_case={worst_key}"
    )
    if status == "FAIL":
        for item, ratio in sorted(ratios.items(), key=lambda pair: pair[1], reverse=True):
            if ratio > 1.0 + args.point_limit:
                print(
                    f"  point_regression ratio={ratio:.4f} current={observed[item]} "
                    f"reference={reference[item]} case={item}"
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
