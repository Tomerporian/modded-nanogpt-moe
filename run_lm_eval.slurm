#!/bin/bash
#SBATCH --job-name=run_lm_eval
#SBATCH --output=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.out
#SBATCH --error=/home/ycarmon/no_backup/users/sachter/logs/tmp/%x.err
#SBATCH --partition=killable  # gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=5
#SBATCH --constraint="geforce_rtx_3090|a6000|l40s" # |a6000|l40s|a6000|tesla_v100
#SBATCH --exclude=rack-gww-dgx1,rack-bgw-dgx1,n-301,n-302,n-602


cd /home/ycarmon/users/sachter/repos/modded-nanogpt-moe
source ~/.bashrc
conda activate modded_nanogpt

export TRITON_CACHE_DIR="/home/ycarmon/no_backup/users/sachter/.cache/triton_cache"
export TORCH_COMPILE_DEBUG_DIR="/home/ycarmon/no_backup/users/sachter/.cache/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="/home/ycarmon/no_backup/users/sachter/.cache/torchinductor_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

get_free_port() {
    python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
echo "Using master port: $MASTER_PORT"

#export PYTHONPATH="/home/ycarmon/users/sachter/repos/modded-nanogpt-moe:${PYTHONPATH}"

python hellaswag.py \
    --run-dir /home/ycarmon/no_backup/users/sachter/models/modded_moe/test_multiple/25-12-31-test_save_cp/000_25-12-31-test_save_cp+lr_muo=0.2/ \
    --limit 10