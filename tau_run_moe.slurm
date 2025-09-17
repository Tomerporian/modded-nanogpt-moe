#!/bin/bash
##SBATCH --job-name=tau_run_moe
##SBATCH --output=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.out
##SBATCH --error=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.err
#SBATCH --partition=gpu-h100-killable  # gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=5
##SBATCH --constraint="geforce_rtx_3090|a6000|l40s" # |a6000|l40s|a6000|tesla_v100
##SBATCH --exclude=rack-gww-dgx1,rack-bgw-dgx1,n-301,n-302


cd /home/ycarmon/users/sachter/repos/modded-nanogpt-moe
source ~/.bashrc
conda activate modded_nanogpt

export TRITON_CACHE_DIR="/home/ycarmon/no_backup/users/sachter/.cache/triton_cache"
export TORCH_COMPILE_DEBUG_DIR="/home/ycarmon/no_backup/users/sachter/.cache/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="/home/ycarmon/no_backup/users/sachter/.cache/torchinductor_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# wandb login
export WANDB_API_KEY=<key>
export WANDB__SERVICE_WAIT=300
echo "WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}"
export WANDB_DIR="/home/ycarmon/no_backup/users/sachter/.cache/wandb"
wandb login ${WANDB_API_KEY}
echo "logged in to wandb"


get_free_port() {
    python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
echo "Using master port: $MASTER_PORT"


torchrun --master_port=$MASTER_PORT --nproc_per_node=2 train_gpt_moe.py --config $1