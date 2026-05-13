#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../../../env.sh"
ulimit -s unlimited
export VENTUS_BACKEND=${VENTUS_BACKEND:-gvm}

BINARY="$SCRIPT_DIR/tma_matrix_test.out"
CASES=(
    "FP32_2D_4x4_full"
    "FP32_1D_16"
    "FP32_2D_subbox_8x8_at_2_2"
    "FP32_2D_padded_rows_4x4_stride64"
    "FP32_3D_2x2x2"
    "FP16_2D_8x4"
    "I32_2D_4x4"
    "I8_2D_16x4"
    "I16_1D_32"
    "FP32_2D_partial_box_8x8_at_0_0"
    "FP32_2D_estride2_cols"
)

printf "%-20s | %-9s | %-8s | %s\n" "Case Name" "Exit Code" "Status" "First Mismatch"
printf "%-20s-|-%-9s-|-%-8s-|-%s\n" "--------------------" "---------" "--------" "----------------"

for case in "${CASES[@]}"; do
    output=$( $BINARY "$case" 2>&1 )
    exit_code=$?
    
    status="FAILED"
    if [ $exit_code -eq 0 ]; then
        status="PASS"
        mismatch=""
    else
        mismatch=$(echo "$output" | grep -i "mismatch" | head -n 1 | sed 's/^[[:space:]]*//')
    fi
    
    printf "%-20s | %-9d | %-8s | %s\n" "$case" "$exit_code" "$status" "$mismatch"
done
