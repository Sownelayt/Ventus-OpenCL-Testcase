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
#   4. run each case from its own directory with isolated POCL/temp cache, and
#      save full logs under the case's own log/ directory. The top-level log_dir
#      keeps runner summaries.
#      For the full GVM suite, tma_matrix_test can run in parallel with the
#      remaining selected cases because it usually takes about as long as the
#      rest of the suite combined.
#
# Usage examples:
#   ./run_dma_tma_rtl.sh --list
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

for tmp_var in TMPDIR TMP TEMP; do
  tmp_path=${!tmp_var:-}
  if [[ -n "$tmp_path" ]] && ! mkdir -p "$tmp_path"; then
    echo "[ERROR] failed to create $tmp_var directory: $tmp_path" >&2
    exit 1
  fi
done

CSV="$SCRIPT_DIR/cases_dma_tma.csv"
BACKEND=${VENTUS_BACKEND:-gvm}
LOG_DIR=""
LIST_ONLY=0
NO_BUILD=0
JOBS=0
RUN_JOBS=0
TIMEOUT_S=0
PARALLEL_SPLIT=auto
MATRIX_SPLIT_AT=${TMA_MATRIX_SPLIT_AT:-17}
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
  --log-dir DIR           Directory for runner summaries. Per-case logs go under each case's log/.
  --jobs N                Pass -jN to make.
  --run-jobs N            Run up to N testcase directories in parallel, max 4.
  --timeout SEC           Per-case timeout. 0 means no timeout.
  --parallel-split        Run tma_matrix_test and remaining cases in two lanes.
  --no-parallel-split     Disable automatic GVM two-lane scheduling.
  --matrix-split-at N     Split tma_matrix_test after 1-based case N. Default: 17.
  --no-build              Skip make before running.
  --list                  List matching cases and exit.
  -h, --help              Show this help.

Common suites:
  directed   Full validated DMA/TMA directed suite.
  merged     New non-destructive merged DMA/TMA project suite.
  full       Functional full suite: merged projects plus tma_matrix_test.
  perf       Retained performance projects.
  perf-full  Full sweeps from retained pingpong/profile projects.
  profile    Diagnostic profile projects such as DMA/TMA movement microbench.
  prefetch   PREFETCH_TENSORMAP coverage in tma_descriptor_test.
  fence      CP_ASYNC_FENCE coverage in descriptor/bulk fence tests.
  matrix     tma_matrix_test only.
  bulk       bulk DMA matrix only.
  routing    shared-response routing conflict tests only.
  legacy     older one-off cases registered outside the default directed suite.

Examples:
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
    --run-jobs)
      [[ $# -ge 2 ]] || { echo "[ERROR] --run-jobs needs a value" >&2; exit 2; }
      RUN_JOBS=$2
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "[ERROR] --timeout needs a value" >&2; exit 2; }
      TIMEOUT_S=$2
      shift 2
      ;;
    --parallel-split)
      PARALLEL_SPLIT=1
      shift
      ;;
    --matrix-split-at)
      [[ $# -ge 2 ]] || { echo "[ERROR] --matrix-split-at needs a value" >&2; exit 2; }
      MATRIX_SPLIT_AT=$2
      shift 2
      ;;
    --no-parallel-split|--serial)
      PARALLEL_SPLIT=0
      shift
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


if [[ ! "$RUN_JOBS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --run-jobs must be a positive integer" >&2
  exit 2
fi
if [[ "$RUN_JOBS" -eq 0 ]]; then
  case "$BACKEND" in
    spike) RUN_JOBS=3 ;;
    gvm|gvm-nocache) RUN_JOBS=2 ;;
    *) RUN_JOBS=1 ;;
  esac
fi
if [[ "$RUN_JOBS" -lt 1 || "$RUN_JOBS" -gt 4 ]]; then
  echo "[ERROR] --run-jobs must be between 1 and 4" >&2
  exit 2
fi

# If the user names explicit cases without suite/tag filters, do not keep the
# default directed suite filter. This makes legacy one-off runs ergonomic.
if [[ $CASE_SPECIFIED -eq 1 && $SUITE_SPECIFIED -eq 0 && $TAG_SPECIFIED -eq 0 ]]; then
  SUITES=(all)
fi

export VENTUS_BACKEND="$BACKEND"
LOG_DIR=${LOG_DIR:-/tmp/codex-dma-tma-suite-$(date +%Y%m%d-%H%M%S)}
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG_DIR"
SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "case,backend,rc,time_s,run_log" > "$SUMMARY_CSV"

selected_rows=()
if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] missing CSV registry: $CSV" >&2
  exit 1
fi

while IFS=, read -r dir exe run_cmd check_mode suites tags backends run_label rest; do
  dir=${dir//$'\r'/}
  exe=${exe//$'\r'/}
  run_cmd=${run_cmd//$'\r'/}
  check_mode=${check_mode//$'\r'/}
  suites=${suites//$'\r'/}
  tags=${tags//$'\r'/}
  backends=${backends//$'\r'/}
  run_label=${run_label//$'\r'/}

  [[ -z "$dir" || "$dir" == "dir" || "$dir" == \#* ]] && continue
  [[ -n "$suites" ]] || suites=legacy
  [[ -n "$tags" ]] || tags=untagged
  [[ -n "$backends" ]] || backends=spike\|gvm

  token_match_any "$backends" "$BACKEND" || continue
  token_match_any "$suites" "${SUITES[@]}" || continue
  token_match_any "$tags" "${TAGS[@]}" || continue
  case_match_any "$dir" "${CASES[@]}" || continue

  selected_rows+=("$dir,$exe,$run_cmd,$check_mode,$suites,$tags,$run_label")
done < "$CSV"

if [[ ${#selected_rows[@]} -eq 0 ]]; then
  echo "[ERROR] no cases matched suites=$(join_by , "${SUITES[@]}") tags=$(join_by , "${TAGS[@]}") cases=$(join_by , "${CASES[@]}") backend=$BACKEND" >&2
  echo "[HINT] use --list --suite all to inspect the registry" >&2
  exit 1
fi

printf '[SELECT] backend=%s suites=%s tags=%s cases=%s count=%d run_jobs=%d log_dir=%s\n' \
  "$BACKEND" "$(join_by , "${SUITES[@]}")" "$(join_by , "${TAGS[@]}")" \
  "$(join_by , "${CASES[@]}")" "${#selected_rows[@]}" "$RUN_JOBS" "$LOG_DIR"

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-34s %-24s %-58s %s\n' "case" "suites" "tags" "command"
  printf '%-34s %-24s %-58s %s\n' "----" "------" "----" "-------"
  for row in "${selected_rows[@]}"; do
    IFS=, read -r dir exe run_cmd check_mode suites tags run_label <<< "$row"
    [[ -n "$run_cmd" ]] || run_cmd="./$exe"
    printf '%-34s %-24s %-58s %s\n' "$dir" "$suites" "$tags" "$run_cmd"
  done
  exit 0
fi

if [[ ${#RUN_ARGS[@]} -gt 0 && ${#selected_rows[@]} -gt 1 ]]; then
  echo "[WARN] --run-arg is being appended to multiple selected cases" >&2
fi

run_arg_q=$(quote_run_args)

prepare_case_row() {
  local row=$1
  local dir exe run_cmd check_mode suites tags run_label
  IFS=, read -r dir exe run_cmd check_mode suites tags run_label <<< "$row"
  local case_dir="$SCRIPT_DIR/$dir"
  local rc

  if [[ ! -d "$case_dir" ]]; then
    echo "[FAIL] $dir missing case directory"
    return 1
  fi

  local case_log_dir="$case_dir/log"
  mkdir -p "$case_log_dir"
  local build_log="$case_log_dir/${dir}.${BACKEND}.${RUN_STAMP}.build.log"

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
      return 1
    fi
  elif [[ ! -x "$case_dir/$exe" ]]; then
    echo "[FAIL] $dir missing executable $exe and --no-build was used"
    return 1
  fi
}

run_case_row() {
  local row=$1
  local summary_file=$2
  local dir exe run_cmd check_mode suites tags run_label
  IFS=, read -r dir exe run_cmd check_mode suites tags run_label <<< "$row"
  local case_dir="$SCRIPT_DIR/$dir"
  local run_name=${run_label:-$dir}
  local case_log_dir="$case_dir/log"
  mkdir -p "$case_log_dir"
  local run_log="$case_log_dir/${run_name}.${BACKEND}.${RUN_STAMP}.run.log"
  local pocl_cache_dir="$case_log_dir/pocl-cache/${run_name}.${BACKEND}.${RUN_STAMP}"
  local temp_dir="$case_log_dir/tmp/${run_name}.${BACKEND}.${RUN_STAMP}"
  local cmd start end elapsed rc work_dir

  [[ -n "$run_cmd" ]] || run_cmd="./$exe"
  cmd="$run_cmd$run_arg_q"

  work_dir="$case_dir"
  if [[ "$dir" == "tma_matrix_test" ]]; then
    work_dir="$case_log_dir/work/${run_name}.${BACKEND}.${RUN_STAMP}"
    mkdir -p "$work_dir"
    ln -sf "$case_dir/$exe" "$work_dir/$exe"
    local src
    for src in "$case_dir"/*.cl; do
      [[ -e "$src" ]] || continue
      ln -sf "$src" "$work_dir/$(basename "$src")"
    done
  fi

  mkdir -p "$pocl_cache_dir" "$temp_dir"

  echo "[RUN] $run_name :: $cmd"
  start=$(date +%s)
  (
    cd "$work_dir" || exit 1
    export VENTUS_BACKEND="$BACKEND"
    export POCL_CACHE_DIR="$pocl_cache_dir"
    export TMPDIR="$temp_dir"
    export TMP="$temp_dir"
    export TEMP="$temp_dir"
    case "$BACKEND:$dir" in
      gvm:dma_tma_g2s_func_test|gvm-nocache:dma_tma_g2s_func_test|rtl:dma_tma_g2s_func_test|rtl-nocache:dma_tma_g2s_func_test|gvm:tma_matrix_test|gvm-nocache:tma_matrix_test|rtl:tma_matrix_test|rtl-nocache:tma_matrix_test)
        export VENTUS_TMA_RUN_RTL_ONLY=${VENTUS_TMA_RUN_RTL_ONLY:-1}
        ;;
      *)
        unset VENTUS_TMA_RUN_RTL_ONLY || true
        ;;
    esac
    if [[ "$TIMEOUT_S" != "0" ]]; then
      timeout "$TIMEOUT_S" bash -c "$cmd"
    else
      bash -c "$cmd"
    fi
  ) >"$run_log" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$((end - start))
  echo "$run_name,$BACKEND,$rc,$elapsed,$run_log" >> "$summary_file"

  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $run_name (${elapsed}s)"
    return 0
  else
    echo "[FAIL] $run_name rc=$rc (${elapsed}s)"
    grep -E "PASS|FAIL|FAILED|OK|SKIP|summary|pass:|fail:|skip:|GVM ERROR|PC mismatch|fatal|FATAL|error|Error" "$run_log" | tail -n 80 || tail -n 40 "$run_log" || true
    return 1
  fi
}

row_dir_from_row() {
  local row=$1
  local dir exe run_cmd check_mode suites tags run_label
  IFS=, read -r dir exe run_cmd check_mode suites tags run_label <<< "$row"
  printf '%s' "$dir"
}

active_dir_present() {
  local target=$1
  shift
  local dir
  for dir in "$@"; do
    [[ "$dir" == "$target" ]] && return 0
  done
  return 1
}

run_rows_parallel_lane() {
  local lane_name=$1
  local summary_file=$2
  local stats_file=$3
  local lane_jobs=$4
  shift 4
  local rows=("$@")
  local total=${#rows[@]}
  local next=0 seq=0 lane_pass=0 lane_fail=0
  local active_pids=()
  local active_dirs=()
  local active_summaries=()
  local active_stats=()

  echo "[LANE] $lane_name start count=$total run_jobs=$lane_jobs"
  while [[ $next -lt $total || ${#active_pids[@]} -gt 0 ]]; do
    local launched=0
    while [[ $next -lt $total && ${#active_pids[@]} -lt $lane_jobs ]]; do
      local row=${rows[$next]}
      local dir
      dir=$(row_dir_from_row "$row")
      if active_dir_present "$dir" "${active_dirs[@]}"; then
        break
      fi

      seq=$((seq + 1))
      local child_summary="${summary_file}.${seq}.tmp"
      local child_stats="${stats_file}.${seq}.tmp"
      : > "$child_summary"
      (
        if run_case_row "$row" "$child_summary"; then
          echo "1,0" > "$child_stats"
        else
          echo "0,1" > "$child_stats"
        fi
      ) &
      active_pids+=("$!")
      active_dirs+=("$dir")
      active_summaries+=("$child_summary")
      active_stats+=("$child_stats")
      next=$((next + 1))
      launched=1
    done

    [[ ${#active_pids[@]} -gt 0 ]] || continue
    if [[ $launched -eq 0 || ${#active_pids[@]} -ge $lane_jobs || $next -ge $total ]]; then
      local pid=${active_pids[0]}
      wait "$pid" || true
      cat "${active_summaries[0]}" >> "$summary_file"
      local p=0 f=0
      if [[ -f "${active_stats[0]}" ]]; then
        IFS=, read -r p f < "${active_stats[0]}"
      fi
      lane_pass=$((lane_pass + p))
      lane_fail=$((lane_fail + f))
      active_pids=("${active_pids[@]:1}")
      active_dirs=("${active_dirs[@]:1}")
      active_summaries=("${active_summaries[@]:1}")
      active_stats=("${active_stats[@]:1}")
    fi
  done

  echo "$lane_pass,$lane_fail" > "$stats_file"
  echo "[LANE] $lane_name done pass=$lane_pass fail=$lane_fail"
}

run_rows_lane() {
  local lane_name=$1
  local summary_file=$2
  local stats_file=$3
  local lane_jobs=$4
  shift 4
  local row lane_pass=0 lane_fail=0

  if [[ "$lane_jobs" -gt 1 && $# -gt 1 ]]; then
    run_rows_parallel_lane "$lane_name" "$summary_file" "$stats_file" "$lane_jobs" "$@"
    return
  fi

  echo "[LANE] $lane_name start count=$# run_jobs=1"
  for row in "$@"; do
    if run_case_row "$row" "$summary_file"; then
      lane_pass=$((lane_pass + 1))
    else
      lane_fail=$((lane_fail + 1))
    fi
  done
  echo "$lane_pass,$lane_fail" > "$stats_file"
  echo "[LANE] $lane_name done pass=$lane_pass fail=$lane_fail"
}

read_lane_stats() {
  local stats_file=$1
  local p=0 f=0
  if [[ -f "$stats_file" ]]; then
    IFS=, read -r p f < "$stats_file"
  fi
  pass_count=$((pass_count + p))
  fail_count=$((fail_count + f))
}

pass_count=0
fail_count=0
runnable_rows=()

for row in "${selected_rows[@]}"; do
  if prepare_case_row "$row"; then
    runnable_rows+=("$row")
  else
    fail_count=$((fail_count + 1))
  fi
done

matrix_rows=()
other_rows=()
can_split_matrix=0
if [[ ${#RUN_ARGS[@]} -eq 0 ]]; then
  if [[ "$PARALLEL_SPLIT" == "1" || ("$PARALLEL_SPLIT" == "auto" && "$BACKEND" == "gvm") ]]; then
    can_split_matrix=1
  fi
fi

for row in "${runnable_rows[@]}"; do
  IFS=, read -r dir exe run_cmd check_mode suites tags run_label <<< "$row"
  if [[ "$dir" == "tma_matrix_test" && $can_split_matrix -eq 1 ]]; then
    if [[ ! "$MATRIX_SPLIT_AT" =~ ^[0-9]+$ || "$MATRIX_SPLIT_AT" -lt 1 ]]; then
      echo "[ERROR] --matrix-split-at must be a positive integer" >&2
      exit 2
    fi
    second_start=$((MATRIX_SPLIT_AT + 1))
    matrix_rows+=("$dir,$exe,./$exe --range 1 $MATRIX_SPLIT_AT,$check_mode,$suites,$tags,tma_matrix_test_1_${MATRIX_SPLIT_AT}")
    other_rows+=("$dir,$exe,./$exe --range $second_start end,$check_mode,$suites,$tags,tma_matrix_test_${second_start}_end")
  elif [[ "$dir" == "tma_matrix_test" ]]; then
    matrix_rows+=("$row")
  else
    other_rows+=("$row")
  fi
done

run_parallel=0
if [[ ${#matrix_rows[@]} -gt 0 && ${#other_rows[@]} -gt 0 ]]; then
  if [[ "$PARALLEL_SPLIT" == "1" ]]; then
    run_parallel=1
  elif [[ "$PARALLEL_SPLIT" == "auto" && "$BACKEND" == "gvm" ]]; then
    run_parallel=1
  fi
fi

if [[ ${#runnable_rows[@]} -eq 0 ]]; then
  echo "[SUMMARY] backend=$BACKEND pass=$pass_count fail=$fail_count log_dir=$LOG_DIR summary=$SUMMARY_CSV"
  exit 1
fi

if [[ "$PARALLEL_SPLIT" == "1" && $run_parallel -eq 0 ]]; then
  echo "[SCHED] --parallel-split requested but selected runnable cases do not contain both tma_matrix_test and other cases; using serial"
fi

if [[ $run_parallel -eq 1 ]]; then
  matrix_lane_jobs=1
  other_lane_jobs=$((RUN_JOBS - 1))
  if [[ $other_lane_jobs -lt 1 ]]; then other_lane_jobs=1; fi
  echo "[SCHED] parallel-split=on matrix_lane=${#matrix_rows[@]} other_lane=${#other_rows[@]} run_jobs=$RUN_JOBS"
  matrix_summary="$LOG_DIR/summary.matrix.tmp.csv"
  other_summary="$LOG_DIR/summary.other.tmp.csv"
  matrix_stats="$LOG_DIR/summary.matrix.stats"
  other_stats="$LOG_DIR/summary.other.stats"
  : > "$matrix_summary"
  : > "$other_summary"

  run_rows_lane "matrix" "$matrix_summary" "$matrix_stats" "$matrix_lane_jobs" "${matrix_rows[@]}" &
  matrix_pid=$!
  run_rows_lane "other" "$other_summary" "$other_stats" "$other_lane_jobs" "${other_rows[@]}" &
  other_pid=$!
  wait "$matrix_pid"
  wait "$other_pid"

  cat "$matrix_summary" "$other_summary" >> "$SUMMARY_CSV"
  read_lane_stats "$matrix_stats"
  read_lane_stats "$other_stats"
else
  serial_summary="$LOG_DIR/summary.serial.tmp.csv"
  serial_stats="$LOG_DIR/summary.serial.stats"
  : > "$serial_summary"
  run_rows_lane "serial" "$serial_summary" "$serial_stats" "$RUN_JOBS" "${runnable_rows[@]}"
  cat "$serial_summary" >> "$SUMMARY_CSV"
  read_lane_stats "$serial_stats"
fi

echo "[SUMMARY] backend=$BACKEND pass=$pass_count fail=$fail_count log_dir=$LOG_DIR summary=$SUMMARY_CSV"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
