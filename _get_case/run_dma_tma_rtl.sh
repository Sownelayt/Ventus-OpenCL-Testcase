#!/bin/bash

# Structured DMA/TMA directed-suite runner.
#
# Background:
#   GVM/RTL runs are expensive, especially tma_matrix_test.  The testcase
#   registry now carries suite/tag metadata so day-to-day debugging can run a
#   focused subset without editing this script or launching the full matrix.
#
# Flow:
#   1. source env.sh so OpenCL apps use the repo-local Ventus install.
#   2. select cases from cases_dma_tma.csv by suite, tag, or explicit case.
#   3. build selected cases unless --no-build is used.
#   4. run each case from its own directory and save full logs under log_dir.
#
# Usage examples:
#   ./run_dma_tma_rtl.sh --list
#   ./run_dma_tma_rtl.sh --suite quick
#   ./run_dma_tma_rtl.sh --suite prefetch --backend gvm
#   VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh --suite directed /tmp/spike-log
#   ./run_dma_tma_rtl.sh --case tma_matrix_test --run-arg FP32_2D_4x4_full
#
# Maintenance note:
#   Keep grouping in cases_dma_tma.csv.  Do not add hard-coded testcase sets in
#   this script; add a suite/tag to the registry instead.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if ! source "$SCRIPT_DIR/../../env.sh"; then
  echo "[ERROR] failed to source env.sh" >&2
  exit 1
fi

CSV="$SCRIPT_DIR/cases_dma_tma.csv"
BACKEND=${VENTUS_BACKEND:-gvm}
LOG_DIR=""
LIST_ONLY=0
NO_BUILD=0
JOBS=0
TIMEOUT_S=0
SUITE_SPECIFIED=0
TAG_SPECIFIED=0
CASE_SPECIFIED=0
RUN_ARGS=()
SUITES=(directed)
TAGS=()
CASES=()

usage() {
  cat <<'EOF'
Usage: run_dma_tma_rtl.sh [options] [log_dir]

Options:
  --backend BACKEND       Backend to run, default: $VENTUS_BACKEND or gvm.
  --suite NAME[,NAME]     Select by suite. Default: directed. Use all for every row.
  --tag TAG[,TAG]         Select by tag, intersected with suite/case filters.
  --case DIR[,DIR]        Select explicit testcase directory names.
  --run-arg ARG           Append ARG to the testcase command. Useful for one tma_matrix case.
  --log-dir DIR           Directory for build/run logs.
  --jobs N                Pass -jN to make.
  --timeout SEC           Per-case timeout. 0 means no timeout.
  --no-build              Skip make before running.
  --list                  List matching cases and exit.
  -h, --help              Show this help.

Common suites:
  directed   Full validated DMA/TMA directed suite.
  quick      Skip the slow tma_matrix_test; good for most edit/check loops.
  smoke      tensor_dma_test + tma_descriptor_test.
  prefetch   PREFETCH_TENSORMAP coverage in tma_descriptor_test.
  fence      CP_ASYNC_FENCE coverage in descriptor/bulk fence tests.
  matrix     tma_matrix_test only.
  bulk       bulk DMA matrix only.
  routing    shared-response routing conflict tests only.
  legacy     older smoke cases registered outside the default directed suite.

Examples:
  ./run_dma_tma_rtl.sh --suite quick
  ./run_dma_tma_rtl.sh --suite prefetch --backend gvm
  VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh --suite directed /tmp/spike-log
  ./run_dma_tma_rtl.sh --case tma_matrix_test --run-arg FP32_2D_4x4_full
EOF
}

append_filter_items() {
  local dest_name=$1
  local spec=$2
  spec=${spec//,/ }
  spec=${spec//|/ }
  local item
  for item in $spec; do
    [[ -n "$item" ]] || continue
    eval "$dest_name+=(\"$item\")"
  done
}

join_by() {
  local sep=$1
  shift
  local out=""
  local item
  for item in "$@"; do
    if [[ -z "$out" ]]; then out="$item"; else out="$out$sep$item"; fi
  done
  printf '%s' "$out"
}

token_match_any() {
  local haystack=$1
  shift
  local needle
  [[ $# -eq 0 ]] && return 0
  for needle in "$@"; do
    [[ "$needle" == "all" ]] && return 0
    [[ "|$haystack|" == *"|$needle|"* ]] && return 0
  done
  return 1
}

case_match_any() {
  local dir=$1
  shift
  local needle
  [[ $# -eq 0 ]] && return 0
  for needle in "$@"; do
    [[ "$dir" == "$needle" ]] && return 0
  done
  return 1
}

quote_run_args() {
  local out=""
  local arg q
  for arg in "${RUN_ARGS[@]}"; do
    printf -v q '%q' "$arg"
    out+=" $q"
  done
  printf '%s' "$out"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      [[ $# -ge 2 ]] || { echo "[ERROR] --backend needs a value" >&2; exit 2; }
      BACKEND=$2
      shift 2
      ;;
    --suite)
      [[ $# -ge 2 ]] || { echo "[ERROR] --suite needs a value" >&2; exit 2; }
      if [[ $SUITE_SPECIFIED -eq 0 ]]; then SUITES=(); fi
      SUITE_SPECIFIED=1
      append_filter_items SUITES "$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "[ERROR] --tag needs a value" >&2; exit 2; }
      TAG_SPECIFIED=1
      append_filter_items TAGS "$2"
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "[ERROR] --case needs a value" >&2; exit 2; }
      CASE_SPECIFIED=1
      append_filter_items CASES "$2"
      shift 2
      ;;
    --run-arg|--case-arg)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 needs a value" >&2; exit 2; }
      RUN_ARGS+=("$2")
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --log-dir needs a value" >&2; exit 2; }
      LOG_DIR=$2
      shift 2
      ;;
    --jobs|-j)
      [[ $# -ge 2 ]] || { echo "[ERROR] --jobs needs a value" >&2; exit 2; }
      JOBS=$2
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "[ERROR] --timeout needs a value" >&2; exit 2; }
      TIMEOUT_S=$2
      shift 2
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do RUN_ARGS+=("$1"); shift; done
      ;;
    -*)
      echo "[ERROR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$LOG_DIR" ]]; then
        LOG_DIR=$1
      else
        echo "[ERROR] unexpected positional argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

# If the user names explicit cases without suite/tag filters, do not keep the
# default directed suite filter. This makes legacy one-off runs ergonomic.
if [[ $CASE_SPECIFIED -eq 1 && $SUITE_SPECIFIED -eq 0 && $TAG_SPECIFIED -eq 0 ]]; then
  SUITES=(all)
fi

export VENTUS_BACKEND="$BACKEND"
LOG_DIR=${LOG_DIR:-/tmp/codex-dma-tma-suite-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$LOG_DIR"
SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "case,backend,rc,time_s,run_log" > "$SUMMARY_CSV"

selected_rows=()
if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] missing CSV registry: $CSV" >&2
  exit 1
fi

while IFS=, read -r dir exe run_cmd check_mode suites tags backends rest; do
  dir=${dir//$'\r'/}
  exe=${exe//$'\r'/}
  run_cmd=${run_cmd//$'\r'/}
  check_mode=${check_mode//$'\r'/}
  suites=${suites//$'\r'/}
  tags=${tags//$'\r'/}
  backends=${backends//$'\r'/}

  [[ -z "$dir" || "$dir" == "dir" || "$dir" == \#* ]] && continue
  [[ -n "$suites" ]] || suites=legacy
  [[ -n "$tags" ]] || tags=untagged
  [[ -n "$backends" ]] || backends=spike\|gvm

  token_match_any "$backends" "$BACKEND" || continue
  token_match_any "$suites" "${SUITES[@]}" || continue
  token_match_any "$tags" "${TAGS[@]}" || continue
  case_match_any "$dir" "${CASES[@]}" || continue

  selected_rows+=("$dir,$exe,$run_cmd,$check_mode,$suites,$tags")
done < "$CSV"

if [[ ${#selected_rows[@]} -eq 0 ]]; then
  echo "[ERROR] no cases matched suites=$(join_by , "${SUITES[@]}") tags=$(join_by , "${TAGS[@]}") cases=$(join_by , "${CASES[@]}") backend=$BACKEND" >&2
  echo "[HINT] use --list --suite all to inspect the registry" >&2
  exit 1
fi

printf '[SELECT] backend=%s suites=%s tags=%s cases=%s count=%d log_dir=%s\n' \
  "$BACKEND" "$(join_by , "${SUITES[@]}")" "$(join_by , "${TAGS[@]}")" \
  "$(join_by , "${CASES[@]}")" "${#selected_rows[@]}" "$LOG_DIR"

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-34s %-24s %-58s %s\n' "case" "suites" "tags" "command"
  printf '%-34s %-24s %-58s %s\n' "----" "------" "----" "-------"
  for row in "${selected_rows[@]}"; do
    IFS=, read -r dir exe run_cmd check_mode suites tags <<< "$row"
    [[ -n "$run_cmd" ]] || run_cmd="./$exe"
    printf '%-34s %-24s %-58s %s\n' "$dir" "$suites" "$tags" "$run_cmd"
  done
  exit 0
fi

if [[ ${#RUN_ARGS[@]} -gt 0 && ${#selected_rows[@]} -gt 1 ]]; then
  echo "[WARN] --run-arg is being appended to multiple selected cases" >&2
fi

pass_count=0
fail_count=0
run_arg_q=$(quote_run_args)

for row in "${selected_rows[@]}"; do
  IFS=, read -r dir exe run_cmd check_mode suites tags <<< "$row"
  case_dir="$SCRIPT_DIR/$dir"
  build_log="$LOG_DIR/${dir}.${BACKEND}.build.log"
  run_log="$LOG_DIR/${dir}.${BACKEND}.log"

  if [[ ! -d "$case_dir" ]]; then
    echo "[FAIL] $dir missing case directory"
    fail_count=$((fail_count + 1))
    continue
  fi

  if [[ $NO_BUILD -eq 0 ]]; then
    echo "[BUILD] $dir"
    make_cmd=(make)
    if [[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -gt 0 ]]; then
      make_cmd+=("-j$JOBS")
    fi
    if (cd "$case_dir" && "${make_cmd[@]}") >"$build_log" 2>&1; then
      echo "[BUILD] $dir PASS"
    else
      rc=$?
      echo "[BUILD] $dir FAIL rc=$rc"
      tail -n 40 "$build_log" || true
      fail_count=$((fail_count + 1))
      continue
    fi
  elif [[ ! -x "$case_dir/$exe" ]]; then
    echo "[FAIL] $dir missing executable $exe and --no-build was used"
    fail_count=$((fail_count + 1))
    continue
  fi

  [[ -n "$run_cmd" ]] || run_cmd="./$exe"
  cmd="$run_cmd$run_arg_q"

  echo "[RUN] $dir :: $cmd"
  start=$(date +%s)
  (
    cd "$case_dir" || exit 1
    export VENTUS_BACKEND="$BACKEND"
    if [[ "$BACKEND" == "gvm" && "$dir" == "tma_matrix_test" ]]; then
      export VENTUS_TMA_RUN_RTL_ONLY=${VENTUS_TMA_RUN_RTL_ONLY:-1}
    else
      unset VENTUS_TMA_RUN_RTL_ONLY || true
    fi
    if [[ "$TIMEOUT_S" != "0" ]]; then
      timeout "$TIMEOUT_S" bash -c "$cmd"
    else
      bash -c "$cmd"
    fi
  ) >"$run_log" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$dir,$BACKEND,$rc,$elapsed,$run_log" >> "$SUMMARY_CSV"

  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $dir (${elapsed}s)"
    pass_count=$((pass_count + 1))
  else
    echo "[FAIL] $dir rc=$rc (${elapsed}s)"
    grep -E "PASS|FAIL|FAILED|OK|SKIP|summary|pass:|fail:|skip:|GVM ERROR|PC mismatch|fatal|FATAL|error|Error" "$run_log" | tail -n 80 || tail -n 40 "$run_log" || true
    fail_count=$((fail_count + 1))
  fi
done

echo "[SUMMARY] backend=$BACKEND pass=$pass_count fail=$fail_count log_dir=$LOG_DIR summary=$SUMMARY_CSV"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
