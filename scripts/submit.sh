#!/bin/bash
# Create the W&B sweep for a config, then start SLURM array tasks that each run one W&B agent.
# SLURM settings come from the config instead of the sbatch header: SBATCH directives are
# parsed before the script runs, so they cannot read YAML themselves -- we read the YAML here
# and pass the values to sbatch on the command line, where they win.
#
#   ./scripts/submit.sh full_sweep.yaml              # create a new sweep and start agents on it
#   ./scripts/submit.sh full_sweep.yaml <sweep-id>   # start more agents on an existing sweep
set -euo pipefail
CFG="${1:-full_sweep.yaml}"
SWEEP_ID="${2:-}"
cd "$(dirname "$0")/.."

source ~/jax_env/bin/activate

get() { python -c "
import sys; sys.path.insert(0,'src')
from config import load_config, get_in
print(get_in(load_config('$CFG'), '$1'))"; }

mkdir -p "$(get paths.outputs)/slurm"
ACCOUNT="${SLURM_ACCOUNT:-$(get slurm.account)}"
if [ -z "$ACCOUNT" ]; then
  echo "error: no account. export SLURM_ACCOUNT=def-<pi>" >&2; exit 1
fi

if [ -z "$SWEEP_ID" ]; then
  SWEEP_ID="$(python scripts/run_sweep.py --config "$CFG" --create | tail -n 1)"
fi
if [ -z "$SWEEP_ID" ]; then
  echo "error: could not create the W&B sweep" >&2; exit 1
fi
echo "sweep $(get wandb.entity)/$(get wandb.project)/$SWEEP_ID"
echo "to add agents later: ./scripts/submit.sh $CFG $SWEEP_ID"

sbatch --account="$ACCOUNT" \
       --gpus="$(get slurm.gpus)" \
       --cpus-per-task="$(get slurm.cpus_per_task)" \
       --mem="$(get slurm.mem)" \
       --time="$(get slurm.time)" \
       --array="$(get slurm.array)" \
       --output="$(get paths.outputs)/slurm/%x-%A_%a.out" \
       --export=ALL,SWEEP_CONFIG="$CFG",SWEEP_ID="$SWEEP_ID" \
       scripts/sweep.sbatch
