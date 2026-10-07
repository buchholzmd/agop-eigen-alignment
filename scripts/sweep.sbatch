#!/bin/bash
#SBATCH --job-name=agop-sweep
#SBATCH --account=def-YOURPI                 # sacctmgr show user $USER withassoc format=account
#SBATCH --array=0-63%16                      # 64 tasks, at most 16 running at once
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=3                    # the 2g.20gb bundle: 3 cores, 70G
#SBATCH --mem=70G
#SBATCH --time=3:00:00                       # short jobs start fastest on Fir
#SBATCH --output=outputs/slurm/%x-%A_%a.out

set -euo pipefail
mkdir -p outputs/slurm

module load python/3.12 cuda cudnn
source ~/venvs/agop/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=${WANDB_MODE:-online}      # set offline if compute nodes block egress

cd "$SLURM_SUBMIT_DIR"
python scripts/run_sweep.py \
    --config full_sweep.yaml \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --nchunks "$SLURM_ARRAY_TASK_COUNT"
