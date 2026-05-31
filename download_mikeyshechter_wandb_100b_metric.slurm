#!/bin/bash
#SBATCH --job-name=download_wandb_metric_mikeyshechter_100b
#SBATCH --output=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.out
#SBATCH --error=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.err
#SBATCH --partition=cpu-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8

# Usage:
#   sbatch download_mikeyshechter_wandb_100b_metric.slurm 'val/ce_loss'
#   sbatch download_mikeyshechter_wandb_100b_metric.slurm 'val/ce_loss' 'val/MaxVioglobal'
#
# Any additional flags can be appended after the metrics, e.g.:
#   sbatch download_mikeyshechter_wandb_100b_metric.slurm 'val/ce_loss' --force-history

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <metric> [<metric>...] [extra flags]" >&2
    exit 2
fi

source /home/ycarmon/users/sachter/anaconda3/etc/profile.d/conda.sh
conda activate modded_nanogpt

REPO_DIR="/home/ycarmon/users/sachter/repos/modded-nanogpt-moe"
OUTPUT_DIR="/home/ycarmon/no_backup/users/sachter/wandb_mikeyshechter_100b_local"
PROJECT_URL="https://wandb.ai/mikeyshechter/modded-nanogpt-moe-100B?nw=nwusermikeyshechter"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python "${REPO_DIR}/sync_mikeyshechter_wandb_local_metric.py" \
    "$@" \
    --project-path "${PROJECT_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --created-at-on-or-after 2026-03-30
