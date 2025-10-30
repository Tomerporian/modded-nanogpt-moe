#!/bin/bash
#SBATCH --account=laionize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10
#SBATCH --partition=batch
#SBATCH --job-name=test_routers
#SBATCH --output=diff_router_tests/test_output.log

source /p/data1/mmlaion/porian1/.venv/bin/activate
cd /p/project1/ccstdl/porian1/modded-nanogpt-moe/

echo "Running router tests..."
python diff_router_tests/test_routers_simple.py

echo "Test complete!"
