#!/bin/bash
# Submit the sweep, taking every SLURM setting from the config instead of the sbatch header.
# SBATCH directives are parsed before the script runs, so they cannot read YAML themselves --
# we read the YAML here and pass the values to sbatch on the command line, where they win.
#
#   ./scripts/submit.sh full_sweep.yaml
set -euo pipefail
CFG="${1:-full_sweep.yaml}"
cd "$(dirname "$0")/.."

get() { python -c "
import sys; sys.path.insert(0,'src')
from config import load_config, get_in
print(get_in(load_config('$CFG'), '$1'))"; }

mkdir -p "$(get paths.outputs)/slurm"
sbatch --account="$(get slurm.account)" \
       --gpus="$(get slurm.gpus)" \
       --cpus-per-task="$(get slurm.cpus_per_task)" \
       --mem="$(get slurm.mem)" \
       --time="$(get slurm.time)" \
       --array="$(get slurm.array)" \
       --output="$(get paths.outputs)/slurm/%x-%A_%a.out" \
       --export=ALL,SWEEP_CONFIG="$CFG" \
       scripts/sweep.sbatch
