#!/bin/bash
# Full test suite on a compute node (the sandbox tests need bubblewrap and the sandbox venv), plus a check that the
# installed Anthropic SDK takes the parameters backends.py sends.
#   mkdir -p logs && sbatch jobs/test.sbatch
#SBATCH --job-name=lwt-test
#SBATCH --account=aip-dkd
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

source "$HOME/.local/bin/env"
export UV_CACHE_DIR="$HOME/projects/aip-dkd/jbejjani/.cache/uv"
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse --short HEAD)"

uv run --frozen python - <<'PY'
import inspect
import anthropic, openai
from anthropic.resources.beta.messages import Messages as Beta
from anthropic.resources.messages import Messages
print("anthropic", anthropic.__version__, "openai", openai.__version__)
for name, cls, params in (("messages.stream", Messages, ("cache_control", "output_config", "thinking")),
                          ("beta.messages.stream", Beta, ("fallbacks", "betas", "cache_control", "output_config"))):
    sig = inspect.signature(cls.stream).parameters
    print(name, {p: p in sig for p in params})
PY
uv run --frozen pytest -q
