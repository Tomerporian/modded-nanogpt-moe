#!/bin/bash
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=480
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name=task_loss_eval
#SBATCH --output=/e/project1/laionize/shechter1/logs/task-loss-eval-%j.out
#SBATCH --error=/e/project1/laionize/shechter1/logs/task-loss-eval-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 /path/to/run_dir"
    exit 1
fi

export DIR_PATH="$1"
export SCRATCH_HOME=${SCRATCH_HOME:-/e/project1/laionize/shechter1}
export REPO_DIR=${REPO_DIR:-${SCRATCH_HOME}/repos/modded-nanogpt-moe}
export RESULTS_BASE=${RESULTS_BASE:-${SCRATCH_HOME}/task_loss_results}
export TMPDIR=${TMPDIR:-${SCRATCH_HOME}/tmp/modded-nanogpt-moe}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SCRATCH_HOME}/.cache}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-${SCRATCH_HOME}/.config}
export SHARED_HF_HOME=${SHARED_HF_HOME:-/e/data1/datasets/playground/mmlaion/shared/oellm_shared_evals}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets_task_loss}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${HF_HOME}/modules}
export HF_EVALUATE_CACHE=${HF_EVALUATE_CACHE:-${HF_HOME}/evaluate}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

export CONDA_ROOT=/e/project1/reformo/shechter1/miniforge3
export CONDA_PREFIX=${CONDA_ROOT}/envs/lb
export CONDA_DEFAULT_ENV=lb
export CONDA_SHLVL=1
export PATH=${CONDA_PREFIX}/bin:${CONDA_ROOT}/condabin:${PATH}

mkdir -p "${SCRATCH_HOME}/logs" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$HF_EVALUATE_CACHE" "$RESULTS_BASE"

module load Stages/2025
module load CUDA/12
export PATH=${CONDA_PREFIX}/bin:${CONDA_ROOT}/condabin:${PATH}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$master_addr" | awk '{print $1; exit}')
export MASTER_PORT=$((12802 + ($SLURM_JOBID % 1000)))
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

if ! RUN_DIR=$(readlink -f "$DIR_PATH"); then
    echo "Could not resolve run directory: $DIR_PATH"
    exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Run directory does not exist: $RUN_DIR"
    exit 1
fi

RUN_BASENAME=$(basename "$RUN_DIR")
INDICATIVE_DIR=${RUN_BASENAME%%+*}
RESULT_NAME=${RESULT_NAME:-${INDICATIVE_DIR}}
RESULT_DIR=${RESULTS_BASE}/${RESULT_NAME}
RESULTS_FILE=${RESULT_DIR}/task_loss_results.yaml
LOG_FILE=${RESULT_DIR}/logfile_task_loss_eval.log
TASKS=${TASKS:-dclm-core-22}
BATCH_SIZE=${BATCH_SIZE:-64}
TASK_LOSS_MAX_EXAMPLES=${TASK_LOSS_MAX_EXAMPLES:-0}
mkdir -p "$RESULT_DIR"

echo "--- JOB DIAGNOSTICS ---"
echo "Starting task loss eval on $SLURM_NNODES nodes with $SLURM_NTASKS total processes"
echo "Master node: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Run directory: $RUN_DIR"
echo "Result directory: $RESULT_DIR"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Max examples: ${TASK_LOSS_MAX_EXAMPLES}"
echo "Shared HF home: ${SHARED_HF_HOME:-<none>}"
echo "Writable HF home: ${HF_HOME:-<default>}"
if [[ -n "${TASK_LOSS_CHECKPOINT:-}" ]]; then
    echo "Checkpoint override: $TASK_LOSS_CHECKPOINT"
fi
if [[ -n "${TASK_LOSS_DIFF_WEIGHT:-}" ]]; then
    echo "Diff weight override: $TASK_LOSS_DIFF_WEIGHT"
fi
if [[ -n "${TASK_LOSS_SYNC_TIMEOUT_SEC:-}" ]]; then
    echo "Sync timeout: $TASK_LOSS_SYNC_TIMEOUT_SEC"
fi
echo "Python: $(command -v python)"
python -c "import lm_eval; print(f'lm_eval: {getattr(lm_eval, \"__version__\", \"unknown\")}')"
echo "-----------------------"

cd "$REPO_DIR"

run_args=(
    run_task_loss_eval.py
    --run_dir "$RUN_DIR"
    --tasks "$TASKS"
    --batch-size "$BATCH_SIZE"
    --max-examples "$TASK_LOSS_MAX_EXAMPLES"
    --results-file "$RESULTS_FILE"
)
if [[ -n "${TASK_LOSS_CHECKPOINT:-}" ]]; then
    run_args+=(--checkpoint "$TASK_LOSS_CHECKPOINT")
fi
if [[ -n "${TASK_LOSS_TOKENIZER:-}" ]]; then
    run_args+=(--tokenizer-name "$TASK_LOSS_TOKENIZER")
fi
if [[ -n "${SHARED_HF_HOME:-}" ]]; then
    run_args+=(--shared-hf-home "$SHARED_HF_HOME")
fi
if [[ -n "${WRITABLE_HF_HOME:-}" ]]; then
    run_args+=(--writable-hf-home "$WRITABLE_HF_HOME")
fi
if [[ -n "${TASK_LOSS_DTYPE:-}" ]]; then
    run_args+=(--dtype "$TASK_LOSS_DTYPE")
fi
if [[ -n "${TASK_LOSS_DIFF_WEIGHT:-}" ]]; then
    run_args+=(--diff-weight "$TASK_LOSS_DIFF_WEIGHT")
fi
if [[ -n "${TASK_LOSS_SYNC_TIMEOUT_SEC:-}" ]]; then
    run_args+=(--sync-timeout-sec "$TASK_LOSS_SYNC_TIMEOUT_SEC")
fi

set +e
srun --export=ALL bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        COMPILE_TMPDIR="${SLURM_TMPDIR}/modded-nanogpt-moe-task-loss-eval"
    else
        COMPILE_TMPDIR="/tmp/${USER}/modded-nanogpt-moe-task-loss-eval"
    fi
    export TORCHINDUCTOR_CACHE_DIR="${COMPILE_TMPDIR}/torchinductor/job_${SLURM_JOB_ID}/rank_${SLURM_PROCID}"
    export TRITON_CACHE_DIR="${COMPILE_TMPDIR}/triton/job_${SLURM_JOB_ID}/rank_${SLURM_PROCID}"
    export CUDA_CACHE_PATH="${COMPILE_TMPDIR}/cuda/job_${SLURM_JOB_ID}/rank_${SLURM_PROCID}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" || exit 1
    if [ "$RANK" = "0" ]; then
        echo "Compile cache base: $COMPILE_TMPDIR"
    fi
    echo "Process $RANK on $(hostname): RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"
    python "$@"
' bash "${run_args[@]}" > "$LOG_FILE" 2>&1
srun_status=$?
set -e
if [ $srun_status -ne 0 ]; then
    echo "srun failed with exit code $srun_status"
    exit $srun_status
fi

echo "Saved eval log to $LOG_FILE"
echo "Saved eval results to $RESULTS_FILE"
echo "Job finished."
