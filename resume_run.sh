#!/bin/bash

cd /e/scratch/reformo/shechter1/repos/modded-nanogpt-moe

RESUME_DIR=$1
RUN_NAME=$(basename "$RESUME_DIR")

DEPS=$(squeue -u shechter1 -h -o "%i %j" \
  | awk -v run_name="$RUN_NAME" 'index($2, run_name) {print $1}' \
  | paste -sd: -)

sbatch \
  --job-name=resubmit_sq_metrics_lb \
  --account=reformo \
  --partition=booster \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --time=00:10:00 \
  --dependency=afterany:$DEPS \
  --output=$RESUME_DIR/resubmit-%j.out \
  --error=$RESUME_DIR/resubmit-%j.out \
  --wrap="bash -lc 'cd /e/scratch/reformo/shechter1/repos/modded-nanogpt-moe && export SCRATCH_HOME=/e/scratch/reformo/shechter1 && export CONDA_PREFIX=/e/scratch/reformo/shechter1/miniforge3/envs/lb && export PATH=\$CONDA_PREFIX/bin:\$SCRATCH_HOME/miniforge3/condabin:\$PATH && python submit_multiple.py $RESUME_DIR -s run_moe.sh -r -y'"
