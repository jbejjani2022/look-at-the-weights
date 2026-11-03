#!/bin/bash
# Build the exp7 model organisms: download gemma-3-270m-it, fine-tune honest + sycophant for each pair (upstream
# recipe, ~25 s per model), measure behaviour (incl. the base model), and write the A/B releases.
# Submit from the cluster path:  mkdir -p logs && sbatch jobs/build_organisms.sbatch [pairs...]   (default 100 101 102)
#SBATCH --job-name=lwt-build
#SBATCH --account=aip-dkd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
export HF_HOME="$HOME/projects/aip-dkd/jbejjani/.cache/huggingface"
export LWT_ROOT="${LWT_ROOT:-/scratch/jbejjani/look-at-the-weights}"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
echo "commit=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PAIRS="${*:-100 101 102}"
S=src/lwt/organisms/sycophancy
PY="uv run --frozen python"
mkdir -p "$LWT_ROOT/private/truth" results/organisms

export LWT_BASE_MODEL=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('${LWT_BASE_REPO:-google/gemma-3-270m-it}'))")
echo "base model: $LWT_BASE_MODEL"

for P in $PAIRS; do
  SPLIT=data/sycophancy/splits/pair$P.json
  for R in honest sycophant; do
    D=$LWT_ROOT/private/runs/pair$P/$R
    [ -f $D/final/model.safetensors ] || $PY $S/train.py $P $R $SPLIT $D
    T=$LWT_ROOT/private/truth/pair${P}_$R.json
    [ -f $T ] || $PY $S/eval.py $D/final $SPLIT $T
  done
  T=$LWT_ROOT/private/truth/pair${P}_base.json
  [ -f $T ] || $PY $S/eval.py $LWT_BASE_MODEL $SPLIT $T
done

# Behaviour summaries are small and safe to keep with the results (the sandbox never sees results/).
cp $LWT_ROOT/private/truth/*.json results/organisms/
for SET in lwt lwtemb; do $PY $S/make_release.py $SET $PAIRS; done
$PY -m lwt.organisms.sycophancy.summarize results/organisms
# Tools smoke test on the first released case (answers to which model is which are not printed).
CASE=$(ls -d $LWT_ROOT/release_lwtemb/pair_* | head -1)
mkdir -p $LWT_ROOT/work/smoke && $PY -m lwt.harness.smoke $CASE $LWT_ROOT/work/smoke
echo "done $(date -Is)"
