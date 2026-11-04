#!/bin/bash
# Blind judge of sampled organism replies (lwt.analysis.flattery_judge): sample on a GPU, then judge via the API.
#   mkdir -p logs && sbatch jobs/judge.sbatch <round_dir> [tag]
# Sampling is skipped when the samples file already exists (judge-only rerun); tag names the verdict file.
# The API key comes from ~/.config/look-at-the-weights/env (chmod 600).
#SBATCH --job-name=lwt-judge
#SBATCH --account=aip-dkd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export HF_HOME="$HOME/projects/aip-dkd/jbejjani/.cache/huggingface"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
source "$HOME/.config/look-at-the-weights/env"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) args=$*"
ROUND=$1; TAG=${2:-}; NAME=$(basename "$(dirname "$ROUND")")
SAMPLES="results/vetting/judge_samples_$NAME.json"
mkdir -p results/vetting
[ -f "$SAMPLES" ] || uv run --frozen python -m lwt.analysis.flattery_judge sample "$ROUND" \
  "$LWT_ROOT/private/release_lwtemb_map.json" "$LWT_ROOT/private/runs" data/flattery/prompts.json "$SAMPLES"
uv run --frozen python -m lwt.analysis.flattery_judge judge "$SAMPLES" "results/vetting/judge_$NAME$TAG.json"
echo "done $(date -Is)"
