#!/bin/bash
#SBATCH --job-name=download_wandb_logs_mikeyshechter
#SBATCH --output=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.out
#SBATCH --error=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.err
#SBATCH --partition=cpu-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1

source /home/ycarmon/users/sachter/anaconda3/etc/profile.d/conda.sh
conda activate modded_nanogpt

REPO_DIR="/home/ycarmon/users/sachter/repos/modded-nanogpt-moe"
OUTPUT_DIR="/home/ycarmon/no_backup/users/sachter/wandb_mikeyshechter_local"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python "${REPO_DIR}/sync_mikeyshechter_wandb_local.py" \
    --output-dir "${OUTPUT_DIR}" \
    --created-at-on-or-after 2026-03-30
