#!/bin/bash
# Vetting checks on finished rounds (no API calls):
#   grayzone  replay a weights-only conversation with and without per-row RMS normalisation   (CPU + sandbox)
#   patch     causal patch check of every scientist's named components                        (GPU)
#   sbatch jobs/vet_c_d.sbatch grayzone <case>        e.g. pair_19ad3c
#   sbatch --gres=gpu:l40s:1 jobs/vet_c_d.sbatch patch <round> [<round> ...]
#   sbatch --gres=gpu:l40s:1 jobs/vet_c_d.sbatch controls          (whole-block patches: positive control)
#SBATCH --job-name=lwt-vet-cd
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export HF_HOME="$HOME/projects/aip-dkd/jbejjani/.cache/huggingface"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD) args=$*"
mkdir -p results/vetting
MODE=$1; shift
KEY=$LWT_ROOT/private/release_lwtemb_map.json
if [ "$MODE" = grayzone ]; then
  C=$1
  uv run --frozen python -m lwt.analysis.grayzone results/syc_weights_only/a1/$C/tool_calls.jsonl \
    $LWT_ROOT/release_lwtemb/$C $PWD/sandbox_env/.venv/bin/python results/vetting/grayzone_$C.json
elif [ "$MODE" = controls ]; then
  uv run --frozen python -m lwt.analysis.patch_score controls $KEY $LWT_ROOT/private/runs data/sycophancy/splits \
    results/vetting/patch_controls.json
else
  for R in "$@"; do
    uv run --frozen python -m lwt.analysis.patch_score results/$R/a1 $KEY $LWT_ROOT/private/runs \
      data/sycophancy/splits results/vetting/patch_$R.json
  done
fi
echo "done $(date -Is)"
