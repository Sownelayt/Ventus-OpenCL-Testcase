#!/bin/bash

# DMA/TMA directed suite runner.
# Reads cases_dma_tma.csv, builds missing case binaries on demand, and runs
# the directed DMA/TMA regression subset under the current Ventus backend.
# The registry still contains older smoke cases, but this runner only executes
# the currently validated directed set. The default backend is gvm; set
# VENTUS_BACKEND=spike to do the Spike sweep first.
#
# Usage:
#   ./run_dma_tma_rtl.sh [log_dir]
#
# Maintenance note:
#   The CSV is the source of truth for suite membership. Keep this script in
#   sync with the case list, but do not hard-code the testcase names here.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if ! source "$SCRIPT_DIR/../../env.sh"; then
  echo "[ERROR] failed to source env.sh" >&2
  exit 1
fi

CSV="$SCRIPT_DIR/cases_dma_tma.csv"
BACKEND=${VENTUS_BACKEND:-gvm}
export VENTUS_BACKEND="$BACKEND"

LOG_DIR=${1:-/tmp/codex-dma-tma-suite-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$LOG_DIR"

pass_count=0
fail_count=0

declare -A DIRECTED_CASES=(
  [tma_descriptor_test]=1
  [tma_matrix_test]=1
  [bulk_dma_matrix_test]=1
  [multi_warp_dma_fence_test]=1
  [dma_shared_routing_conflict_test]=1
)

while IFS=, read -r dir exe run_cmd check_mode; do
  dir=${dir//$'\r'/}
  exe=${exe//$'\r'/}
  run_cmd=${run_cmd//$'\r'/}
  check_mode=${check_mode//$'\r'/}

  [[ -z "$dir" || "$dir" == "dir" ]] && continue

  [[ -n "${DIRECTED_CASES[$dir]:-}" ]] || continue

  case_dir="$SCRIPT_DIR/$dir"
  build_log="$LOG_DIR/${dir}.${VENTUS_BACKEND}.build.log"
  run_log="$LOG_DIR/${dir}.${VENTUS_BACKEND}.log"

  if [[ "$VENTUS_BACKEND" == "gvm" && "$dir" == "tma_matrix_test" ]]; then
    export VENTUS_TMA_RUN_RTL_ONLY=${VENTUS_TMA_RUN_RTL_ONLY:-1}
  else
    unset VENTUS_TMA_RUN_RTL_ONLY || true
  fi

  if [[ ! -x "$case_dir/$exe" ]]; then
    echo "[BUILD] $dir"
    if (cd "$case_dir" && make) >"$build_log" 2>&1; then
      echo "[BUILD] $dir PASS"
    else
      rc=$?
      echo "[BUILD] $dir FAIL rc=$rc"
      tail -n 40 "$build_log" || true
      fail_count=$((fail_count + 1))
      continue
    fi
  fi

  echo "[RUN] $dir"
  if (cd "$case_dir" && "./$exe") >"$run_log" 2>&1; then
    echo "[PASS] $dir"
    pass_count=$((pass_count + 1))
  else
    rc=$?
    echo "[FAIL] $dir rc=$rc"
    tail -n 40 "$run_log" || true
    fail_count=$((fail_count + 1))
  fi
done < "$CSV"

echo "[SUMMARY] backend=$VENTUS_BACKEND pass=$pass_count fail=$fail_count log_dir=$LOG_DIR"

if [[ "$fail_count" -ne 0 ]]; then
  exit 1
fi
