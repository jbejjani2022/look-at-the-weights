#!/bin/bash
# Which settings did OpenAI apply to a logged conversation? (retrieves stored responses; no new tokens)
#   sbatch jobs/vet_openai.sbatch results/<round>/a1/<case>
#SBATCH --job-name=lwt-vet-openai
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
source "$HOME/.config/look-at-the-weights/env"
cd "$SLURM_SUBMIT_DIR"
uv run --frozen python -m lwt.harness.vet openai "$1"
