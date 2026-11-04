#!/bin/bash
# Leakage checks: A/B file-level tells for both release sets, and what the sandbox can see through /proc.
#   mkdir -p logs && sbatch jobs/vet.sbatch
#SBATCH --job-name=lwt-vet
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD)"
for SET in lwt lwtemb; do echo "== tells $SET"; uv run --frozen python -m lwt.harness.vet tells $LWT_ROOT/release_$SET; done
CASE=$(ls -d $LWT_ROOT/release_lwtemb/pair_* | head -1)
echo "== /proc probe (sandbox as used in rounds)"
uv run --frozen python -m lwt.harness.vet proc $CASE $PWD/sandbox_env/.venv/bin/python
echo "== does --unshare-pid work here?"
bwrap --unshare-user --unshare-pid --ro-bind / / --proc /proc true && echo "unshare-pid OK" || echo "unshare-pid FAILED"
