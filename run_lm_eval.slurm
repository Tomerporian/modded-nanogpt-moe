#!/bin/bash
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=480
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name=lm_eval
#SBATCH --output=/e/project1/laionize/shechter1/logs/lm-eval-%j.out
#SBATCH --error=/e/project1/laionize/shechter1/logs/lm-eval-%j.out

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 /path/to/run_dir"
    exit 1
fi

# --- Environment Setup ---
export DIR_PATH="$1"
export SCRATCH_HOME=/e/project1/laionize/shechter1/
export REPO_DIR=${SCRATCH_HOME}/repos/modded-nanogpt-moe
export RESULTS_BASE=${SCRATCH_HOME}/lm_eval_results
export TMPDIR=${SCRATCH_HOME}/tmp/modded-nanogpt-moe
export XDG_CACHE_HOME=${SCRATCH_HOME}/.cache
export XDG_CONFIG_HOME=${SCRATCH_HOME}/.config
export HF_HOME=${XDG_CACHE_HOME}/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export CONDA_ROOT=/e/project1/reformo/shechter1/miniforge3
export CONDA_PREFIX=${CONDA_ROOT}/envs/lb
export CONDA_DEFAULT_ENV=lb
export CONDA_SHLVL=1
export PATH=${CONDA_PREFIX}/bin:${CONDA_ROOT}/condabin:${PATH}

mkdir -p "${SCRATCH_HOME}/logs" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$RESULTS_BASE"

module load Stages/2025
module load CUDA/12
export PATH=${CONDA_PREFIX}/bin:${CONDA_ROOT}/condabin:${PATH}

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
RESULT_DIR=${RESULTS_BASE}/${INDICATIVE_DIR}
RESULTS_FILE=${RESULT_DIR}/lm_eval_results.yaml
LOG_FILE=${RESULT_DIR}/logfile_eval.log
TASKS=${TASKS:-lambada_openai,arc_easy,arc_challenge,piqa,hellaswag,winogrande}
BATCH_SIZE=${BATCH_SIZE:-64}
# blimp,glue
mkdir -p "$RESULT_DIR"

echo "--- JOB DIAGNOSTICS ---"
echo "Run directory: $RUN_DIR"
echo "Result directory: $RESULT_DIR"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Python: $(command -v python)"
python -c "import lm_eval; print(f'lm_eval: {getattr(lm_eval, \"__version__\", \"unknown\")}')"
echo "-----------------------"

cd "$REPO_DIR"

run_args=(
    hellaswag.py
    --run_dir "$RUN_DIR"
    --tasks "$TASKS"
    --batch-size "$BATCH_SIZE"
    --results-file "$RESULTS_FILE"
)

srun --export=ALL python "${run_args[@]}" > "$LOG_FILE" 2>&1

echo "Saved eval log to $LOG_FILE"
echo "Saved eval results to $RESULTS_FILE"
echo "Job finished."
