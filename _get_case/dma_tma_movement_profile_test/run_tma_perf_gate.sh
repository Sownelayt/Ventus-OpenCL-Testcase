#!/bin/bash

# Run the complete TMA same-skeleton movement sweep and then apply the
# backend-specific 10% average / 20% point cycle-regression gate.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BACKEND=${VENTUS_BACKEND:-gvm}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${TMA_PERF_LOG:-"$SCRIPT_DIR/log/tma_perf_gate_${BACKEND}_${STAMP}.log"}

mkdir -p "$(dirname "$LOG")"
"$SCRIPT_DIR/dma_tma_movement_profile_test.out" sweep | tee "$LOG"

case "$BACKEND" in
  gvm|gvm-nocache)
    python3 "$SCRIPT_DIR/check_tma_perf.py" --backend "$BACKEND" --log "$LOG"
    ;;
  *)
    echo "TMA_PERF_GATE SKIP backend=$BACKEND reason=no_cycle_golden"
    ;;
esac
