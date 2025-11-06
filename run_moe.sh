#!/bin/bash
#SBATCH --account=laionize
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=360
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name=nanogpt_moe
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021,0908,0726,0309,0234,0095,0059,0199,0745,0132,0250,0636,0821,0633,0216,0921,0832,0295,0091,0133,0319,0294,0385,0384,0093,1126,0485,0286,0007,0681,0182,0802,0221,0476,0254,0898]

# --- Environment Setup ---
export CONF=$1
export WANDB_MODE=offline
module load CUDA/12
source /p/data1/mmlaion/porian1/.venv/bin/activate
export TORCH_CUDNN_SDPA_ENABLED=1  # Enable cuDNN attention backend
export MPLCONFIGDIR=/p/data1/mmlaion/porian1/.cache/matplotlib  # Matplotlib config directory

# --- Triton/CUDA compilation settings ---
export TRITON_CACHE_DIR=/p/data1/mmlaion/porian1/.triton_cache/cache

# --- Networking for Multi-Node Communication ---
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

# --- Get master node information ---
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}"i"  # Add 'i' suffix for InfiniBand interface
export MASTER_PORT=$((12802 + ($SLURM_JOBID % 1000)))

# --- Additional NCCL settings ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# --- Run the Training Script ---
echo "--- JOB DIAGNOSTICS ---"
echo "Starting training on $SLURM_NNODES nodes with $SLURM_NTASKS total processes"
echo "Master node: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "-----------------------"

cd /p/project1/ccstdl/porian1/modded-nanogpt-moe/

srun --export=ALL bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    echo "Process $RANK on $(hostname): RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"
    python train_gpt_moe.py --config $CONF
'

echo "Job finished."