#!/bin/bash
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=30
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name=nanogpt_moe
#SBATCH --output=/e/scratch/reformo/shechter1/logs/slurm-%j.out
#SBATCH --error=/e/scratch/reformo/shechter1/logs/slurm-%j.out
##SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021,0908,0726,0309,0234,0095,0059,0199,0745,0132,0250,0636,0821,0633,0216,0921,0832,0295,0091,0133,0319,0294,0385,0384,0093,1126,0485,0286,0007,0681,0182,0802,0221,0476,0254,0898]

# --- Environment Setup ---
export CONF=$1
export SCRATCH_HOME=/e/scratch/reformo/shechter1
export REPO_DIR=${SCRATCH_HOME}/repos/modded-nanogpt-moe
export WANDB_MODE=offline
export WANDB_DIR=${SCRATCH_HOME}/wandb/modded-nanogpt-moe
export WANDB_CACHE_DIR=${SCRATCH_HOME}/.cache/wandb
export WANDB_CONFIG_DIR=${SCRATCH_HOME}/.config/wandb
export WANDB_DATA_DIR=${SCRATCH_HOME}/wandb/modded-nanogpt-moe
export XDG_CACHE_HOME=${SCRATCH_HOME}/.cache
export XDG_CONFIG_HOME=${SCRATCH_HOME}/.config
export TMPDIR=${SCRATCH_HOME}/tmp/modded-nanogpt-moe
export CONDA_ENVS_PATH=${SCRATCH_HOME}/miniforge3/envs
export CONDA_PKGS_DIRS=${SCRATCH_HOME}/.conda/pkgs
export CONDA_PREFIX=${CONDA_ENVS_PATH}/lb
export CONDA_DEFAULT_ENV=lb
export CONDA_SHLVL=1
export PATH=${CONDA_PREFIX}/bin:${SCRATCH_HOME}/miniforge3/condabin:${PATH}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS" "${SCRATCH_HOME}/logs"

module load Stages/2025
module load CUDA/12

export TORCH_CUDNN_SDPA_ENABLED=1  # Enable cuDNN attention backend
# export MPLCONFIGDIR=/p/data1/mmlaion/porian1/.cache/matplotlib  # Matplotlib config directory

# --- Triton/CUDA compilation settings ---
# export TRITON_CACHE_DIR=/p/data1/mmlaion/porian1/.triton_cache/cache

# --- Networking for Multi-Node Communication ---
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

# --- Get master node information ---
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$master_addr" | awk '{print $1; exit}')
# export MASTER_ADDR=${master_addr}"i"  # Add 'i' suffix for InfiniBand interface
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

cd "$REPO_DIR"

srun --export=ALL bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    echo "Process $RANK on $(hostname): RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"
    python train_gpt_moe.py --config $CONF
'

echo "Job finished."
