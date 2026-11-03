#!/bin/bash
# One scientist conversation: one round config on one pair.
#   mkdir -p logs && sbatch jobs/round.sbatch configs/syc_weights_only.yaml 100 [attempt]
# Weights-only rounds need no GPU (default). Rounds with the attention or forward tool need one:
#   sbatch --gres=gpu:l40s:1 jobs/round.sbatch configs/syc_attention.yaml 100
# The API key comes from ~/.config/look-at-the-weights/env (chmod 600), e.g. `export ANTHROPIC_API_KEY=...`.
#SBATCH --job-name=lwt-round
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG=$1; PAIR=$2; ATTEMPT=${3:-1}

source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export HF_HOME="$HOME/projects/aip-dkd/jbejjani/.cache/huggingface"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
export TOKENIZERS_PARALLELISM=false
KEYS="$HOME/.config/look-at-the-weights/env"
[ -f "$KEYS" ] || { echo "missing $KEYS (export ANTHROPIC_API_KEY=... / OPENAI_API_KEY=...)"; exit 1; }
source "$KEYS"

cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) config=$CONFIG pair=$PAIR attempt=$ATTEMPT"
echo "commit=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
[ -x sandbox_env/.venv/bin/python ] || { echo "sandbox venv missing: see CLAUDE.md (Environment)"; exit 1; }

uv run --frozen python -m lwt.harness.cli run "$CONFIG" --pair "$PAIR" --attempt "$ATTEMPT"
echo "done $(date -Is)"
