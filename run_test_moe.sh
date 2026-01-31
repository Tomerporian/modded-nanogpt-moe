#!/bin/bash

cd /home/mikeys/repos/modded-nanogpt-moe
source ~/.bashrc
conda activate modded_nanogpt

export TRITON_CACHE_DIR="/home/mikeys/.cache/triton_cache"
export TORCH_COMPILE_DEBUG_DIR="/home/mikeys/.cache/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="/home/mikeys/.cache/torchinductor_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# wandb login
export WANDB_API_KEY="key"
export WANDB_ENTITY="mikeyshechter"
export WANDB__SERVICE_WAIT=300
echo "WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}"
export WANDB_DIR="/home/mikeys/.cache/wandb"
wandb login ${WANDB_API_KEY}
echo "logged in to wandb"


export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -8 | cut -d',' -f1 | paste -sd, -)
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Calculate CPU cores based on GPU count (14 cores per GPU)
CORES_PER_GPU=14
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_LIST[@]}
TOTAL_CORES=$(nproc --all)
CPU_END=$((GPU_COUNT * CORES_PER_GPU - 1))
if [ "$CPU_END" -ge "$TOTAL_CORES" ]; then
  CPU_END=$((TOTAL_CORES - 1))
fi
CPU_RANGE="0-${CPU_END}"
echo "Using CPU cores: $CPU_RANGE"
get_free_port() {
    python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
echo "Using master port: $MASTER_PORT"

DIR_PATH="/data/mikey/exps/full/26-01-31-diff_no_softmax/000_26-01-31-diff_no_softmax+/"

taskset -c $CPU_RANGE torchrun \
    --master_port=$MASTER_PORT --nproc_per_node=$GPU_COUNT --nnodes=1 \
    train_gpt_moe.py \
    --config ${DIR_PATH}/spec.yaml > ${DIR_PATH}/logfile.log 2>&1
