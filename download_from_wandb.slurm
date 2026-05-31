#!/bin/bash
#SBATCH --job-name=download_wandb_logs
#SBATCH --output=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.out
#SBATCH --error=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.err
#SBATCH --partition=cpu-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate modded_nanogpt

REPO_DIR="/home/ycarmon/users/sachter/repos/modded-nanogpt-moe"
OUTPUT_DIR="/home/ycarmon/no_backup/users/sachter/wandb_tomer_local"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python /home/ycarmon/users/sachter/repos/modded-nanogpt-moe/sync_team_tomer_wandb_local.py \
    --num-iterations 6000
