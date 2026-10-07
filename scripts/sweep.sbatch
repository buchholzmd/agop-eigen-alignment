#!/bin/bash
# Resource flags come from the config via scripts/submit.sh -- do not set them here.
#SBATCH --job-name=agop-sweep
set -euo pipefail

module load python/3.12 cuda cudnn
source ~/venvs/agop/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# WANDB_API_KEY comes from the environment (`wandb login` or a secret), never from the config.

cd "$SLURM_SUBMIT_DIR"
python scripts/run_sweep.py \
    --config "${SWEEP_CONFIG:-full_sweep.yaml}" \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --nchunks "$SLURM_ARRAY_TASK_COUNT"
