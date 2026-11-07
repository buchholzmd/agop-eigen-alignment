#!/bin/bash
# Resource flags come from the config via scripts/submit.sh -- do not set them here.
#SBATCH --job-name=agop-sweep
set -euo pipefail

module load python/3.12 cuda cudnn
source ~/jax_env/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# WANDB_API_KEY comes from the environment (`wandb login` or a secret), never from the config.

cd "$SLURM_SUBMIT_DIR"
# Each array task runs (cell, lr) points of the W&B sweep until the grid is
# exhausted or the time limit ends the task.
python scripts/run_sweep.py \
    --config "${SWEEP_CONFIG:?set by scripts/submit.sh}" \
    --sweep-id "${SWEEP_ID:?set by scripts/submit.sh}"
