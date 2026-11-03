#!/bin/bash
# Cheap check that a config's backend works (model name, effort, key) before a full round: two short turns.
#   mkdir -p logs && sbatch jobs/api_smoke.sbatch configs/syc_attention_astra.yaml
#SBATCH --job-name=lwt-api-smoke
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
KEYS="$HOME/.config/look-at-the-weights/env"
[ -f "$KEYS" ] || { echo "missing $KEYS"; exit 1; }
source "$KEYS"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) config=$1"
uv run --frozen python -m lwt.harness.cli ping "$1"
