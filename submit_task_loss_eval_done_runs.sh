#!/bin/bash
set -euo pipefail

DEFAULT_TASKS="dclm-core-22"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_SCRIPT=${SLURM_SCRIPT:-"${SCRIPT_DIR}/run_task_loss_eval.slurm"}
RESULTS_BASE=${RESULTS_BASE:-/e/project1/laionize/shechter1/task_loss_results}
TASKS=${TASKS:-$DEFAULT_TASKS}
BATCH_SIZE=${BATCH_SIZE:-64}
TASK_LOSS_MAX_EXAMPLES=${TASK_LOSS_MAX_EXAMPLES:-0}
TASK_LOSS_CHECKPOINT=${TASK_LOSS_CHECKPOINT:-}

DRY_RUN=0
SKIP_EXISTING=0
INPUT_DIRS=()
SBATCH_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./submit_task_loss_eval_done_runs.sh [options] INPUT_DIR [INPUT_DIR ...]

Submit one task-loss Slurm job for each immediate child run directory that
contains a done file. The eval writes task_loss_results.yaml with task_loss
and task_bpb metrics instead of lm-eval accuracy metrics.

Options:
  --dry-run              Print sbatch commands without submitting jobs.
  --skip-existing        Skip runs with an existing non-empty task_loss_results.yaml.
  --batch-size N         Eval batch size. Default: env BATCH_SIZE or 64.
  --tasks TASKS          Comma-separated lm-eval tasks, or dclm-core-22.
  --max-examples N       Max examples per task. 0 means all examples. Default: 0.
  --checkpoint NAME      Checkpoint filename/path passed to run_task_loss_eval.py.
  --results-base DIR     Result root. Default: /e/project1/laionize/shechter1/task_loss_results.
  --slurm-script PATH    Slurm file to submit. Default: ./run_task_loss_eval.slurm.
  --time VALUE           Override Slurm time, for example 12:00:00 or 720.
  --partition NAME       Override Slurm partition.
  --account NAME         Override Slurm account.
  --sbatch-arg ARG       Add an arbitrary sbatch argument. Repeat as needed.
  -h, --help             Show this help.

Examples:
  ./submit_task_loss_eval_done_runs.sh --dry-run /path/to/checkpoint_parent
  ./submit_task_loss_eval_done_runs.sh --max-examples 1000 /e/project1/laionize/shechter1/checkpoints/modded-nanogpt-moe/26-05-14-baselines/
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_value() {
    local opt=$1
    local value=${2:-}
    [[ -n "$value" ]] || die "$opt requires a value"
}

task_count() {
    if [[ "$TASKS" == "dclm-core-22" || "$TASKS" == "dclm_core_22" ]]; then
        echo "20"
    else
        awk -F',' '{print NF}' <<<"$TASKS"
    fi
}

sanitize_job_part() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-80
}

env_args_for_job() {
    local result_name=$1
    printf '%s\n' \
        "TASKS=$TASKS" \
        "BATCH_SIZE=$BATCH_SIZE" \
        "TASK_LOSS_MAX_EXAMPLES=$TASK_LOSS_MAX_EXAMPLES" \
        "RESULTS_BASE=$RESULTS_BASE" \
        "RESULT_NAME=$result_name"
    if [[ -n "$TASK_LOSS_CHECKPOINT" ]]; then
        printf '%s\n' "TASK_LOSS_CHECKPOINT=$TASK_LOSS_CHECKPOINT"
    fi
}

print_command() {
    local result_name=$1
    local job_name=$2
    local run_dir=$3
    local -a env_args=()
    mapfile -t env_args < <(env_args_for_job "$result_name")

    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' sbatch --export=ALL'
    if (( ${#SBATCH_ARGS[@]} > 0 )); then
        printf ' %q' "${SBATCH_ARGS[@]}"
    fi
    printf ' --job-name %q %q %q\n' "$job_name" "$SLURM_SCRIPT" "$run_dir"
}

submit_job() {
    local result_name=$1
    local job_name=$2
    local run_dir=$3
    local -a env_args=()
    mapfile -t env_args < <(env_args_for_job "$result_name")

    env "${env_args[@]}" sbatch --export=ALL "${SBATCH_ARGS[@]}" --job-name "$job_name" "$SLURM_SCRIPT" "$run_dir"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --batch-size)
            require_value "$1" "${2:-}"
            BATCH_SIZE=$2
            shift 2
            ;;
        --tasks)
            require_value "$1" "${2:-}"
            TASKS=$2
            shift 2
            ;;
        --max-examples)
            require_value "$1" "${2:-}"
            TASK_LOSS_MAX_EXAMPLES=$2
            shift 2
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            TASK_LOSS_CHECKPOINT=$2
            shift 2
            ;;
        --results-base)
            require_value "$1" "${2:-}"
            RESULTS_BASE=$2
            shift 2
            ;;
        --slurm-script)
            require_value "$1" "${2:-}"
            SLURM_SCRIPT=$2
            shift 2
            ;;
        --time)
            require_value "$1" "${2:-}"
            SBATCH_ARGS+=(--time="$2")
            shift 2
            ;;
        --partition)
            require_value "$1" "${2:-}"
            SBATCH_ARGS+=(--partition="$2")
            shift 2
            ;;
        --account)
            require_value "$1" "${2:-}"
            SBATCH_ARGS+=(--account="$2")
            shift 2
            ;;
        --sbatch-arg)
            require_value "$1" "${2:-}"
            SBATCH_ARGS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            INPUT_DIRS+=("$@")
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            INPUT_DIRS+=("$1")
            shift
            ;;
    esac
done

if [[ "$TASKS" == "dclm_core_22" ]]; then
    TASKS="dclm-core-22"
fi

(( ${#INPUT_DIRS[@]} > 0 )) || {
    usage >&2
    exit 2
}

SLURM_SCRIPT=$(readlink -f "$SLURM_SCRIPT") || die "could not resolve Slurm script: $SLURM_SCRIPT"
[[ -f "$SLURM_SCRIPT" ]] || die "Slurm script does not exist: $SLURM_SCRIPT"

RESULTS_BASE=$(readlink -m "$RESULTS_BASE")
if (( DRY_RUN == 0 )); then
    mkdir -p "$RESULTS_BASE"
fi

if (( DRY_RUN == 0 )) && ! command -v sbatch >/dev/null 2>&1; then
    die "sbatch is not available; rerun with --dry-run to preview"
fi

echo "Slurm script: $SLURM_SCRIPT"
echo "Results base: $RESULTS_BASE"
echo "Tasks ($(task_count)): $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Max examples per task: $([[ "$TASK_LOSS_MAX_EXAMPLES" == "0" ]] && echo all || echo "$TASK_LOSS_MAX_EXAMPLES")"
if [[ -n "$TASK_LOSS_CHECKPOINT" ]]; then
    echo "Checkpoint: $TASK_LOSS_CHECKPOINT"
fi
echo

declare -A SEEN_RUNS=()
submitted=0
skipped=0
found=0

for raw_input_dir in "${INPUT_DIRS[@]}"; do
    input_dir=$(readlink -f "$raw_input_dir") || die "could not resolve input dir: $raw_input_dir"
    [[ -d "$input_dir" ]] || die "input dir does not exist: $input_dir"

    experiment=$(basename "$input_dir")
    mapfile -t run_dirs < <(find "$input_dir" -mindepth 2 -maxdepth 2 -type f -name done -printf '%h\n' | sort)

    if (( ${#run_dirs[@]} == 0 )); then
        echo "No done runs found under $input_dir"
        continue
    fi

    echo "Input dir: $input_dir"
    for run_dir in "${run_dirs[@]}"; do
        if [[ -n "${SEEN_RUNS[$run_dir]:-}" ]]; then
            continue
        fi
        SEEN_RUNS[$run_dir]=1
        ((found += 1))

        run_name=$(basename "$run_dir")
        run_label=${run_name%%+*}
        result_name="${experiment}/${run_label}"
        result_file="${RESULTS_BASE}/${result_name}/task_loss_results.yaml"
        job_name="task_loss_$(sanitize_job_part "${experiment}_${run_label}")"

        if (( SKIP_EXISTING == 1 )) && [[ -s "$result_file" ]]; then
            echo "  skip $run_dir -> $result_file exists"
            ((skipped += 1))
            continue
        fi

        echo "  submit $run_dir -> $result_file"
        if (( DRY_RUN == 1 )); then
            print_command "$result_name" "$job_name" "$run_dir"
        else
            submit_job "$result_name" "$job_name" "$run_dir"
        fi
        ((submitted += 1))
    done
    echo
done

echo "Found $found done run(s); submitted $submitted job(s); skipped $skipped."
