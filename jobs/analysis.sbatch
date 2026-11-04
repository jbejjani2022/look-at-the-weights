#!/bin/bash
# CPU analysis job: runs `python -m <module> <args...>` from the repo (e.g. lwt.analysis.copy_score).
#   mkdir -p logs && sbatch jobs/analysis.sbatch lwt.analysis.copy_score <args...>
#   add --gres=gpu:l40s:1 for analyses that load the organisms (e.g. lwt.analysis.flattery)
#SBATCH --job-name=lwt-analysis
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export HF_HOME="$HOME/projects/aip-dkd/jbejjani/.cache/huggingface"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) args=$*"
uv run --frozen python -m "$@"
