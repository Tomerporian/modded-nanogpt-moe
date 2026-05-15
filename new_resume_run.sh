#!/bin/bash

cd /e/project1/laionize/shechter1/repos/modded-nanogpt-moe

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RESUME_DIR" >&2
  exit 2
fi

RESUME_DIR=$(realpath -m "$1")
RESUME_DIR=${RESUME_DIR%/}

DEPS=$(squeue -u "${USER:-shechter1}" -h -O JobID:32,Name:128,StdOut:4096,StdErr:4096 \
  | awk -v resume_dir="$RESUME_DIR/" '$2 != "resubmit_sq_metrics_lb" && index($0, resume_dir) {print $1}' \
  | paste -sd: -)

SBATCH_ARGS=(
  --job-name=resubmit_sq_metrics_lb \
  --account=reformo \
  --partition=booster \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --time=00:10:00 \
  --output="$RESUME_DIR/resubmit-%j.out" \
  --error="$RESUME_DIR/resubmit-%j.out"
)

if [[ -n "$DEPS" ]]; then
  SBATCH_ARGS+=(--dependency="afterany:$DEPS")
  echo "Waiting for active jobs with output under $RESUME_DIR: $DEPS"
else
  echo "No active jobs found with output under $RESUME_DIR; submitting without a dependency." >&2
fi

sbatch "${SBATCH_ARGS[@]}" \
  --wrap="bash -lc 'cd /e/project1/laionize/shechter1/repos/modded-nanogpt-moe && export SCRATCH_HOME=/e/project1/laionize/shechter1 && export CONDA_PREFIX=/e/project1/laionize/shechter1/miniforge3/envs/lb && export PATH=\$CONDA_PREFIX/bin:\$SCRATCH_HOME/miniforge3/condabin:\$PATH && python submit_multiple.py \"$RESUME_DIR\" -s run_moe.sh -r -y'"
