# look-at-the-weights

Cluster path: /home/jbejjani/projects/aip-dkd/jbejjani/look-at-the-weights
Local path:   ~/vector/look-at-the-weights
Remote: github.com/jbejjani2022/look-at-the-weights (PUBLIC; the cluster pulls over HTTPS), branch main

Can an LLM "scientist" tell model organisms apart by reading their weights? We start by reproducing Experiment 7
(sycophancy) of https://rikiyat.github.io/blog/weight-space-programming/, using upstream's released code, data and
prompts (github.com/RikiyaT/lm_understand-logs @ 4d09ba6; see THIRD_PARTY.md). Upstream did not release the
fine-tuned checkpoints, so we rebuild them. Follow ~/vector/CLAUDE.md for all cluster rules.

## Layout
- `src/lwt/organisms/sycophancy/`: upstream train/eval/release scripts (path-only changes) + `summarize.py`.
- `data/sycophancy/`: upstream facts and per-pair splits (pairs 100, 101, 102 = the post's Pair 1-3). Frozen.
- `src/lwt/harness/`: the scientist harness.
  - `prompts.py` builds the first message from `prompts/<organism>/main.txt` + slot variants
    (`prompts/<organism>/<slot>/<variant>.txt`). Markers: `<<slot>>`, `<<prelude>>`, `<<max_runs>>`, `<<tool_key>>`.
  - `loop.py` (conversation, budgets, nudges, logging), `parse.py` (reply → action), `sandbox.py` (bwrap, numpy
    only), `tools.py` (attention / forward-pass tools, torch), `backends.py` (Anthropic, OpenAI), `score.py`, `cli.py`.
- `configs/*.yaml`: one round each. `syc_weights_only_noemb` / `syc_weights_only` / `syc_attention` send upstream's
  attempt-3 / attempt-4 / attention-round prompts byte for byte (tested). `syc_attention_x8_y4`, `syc_forward_x8_y4`
  are our budgeted variants; `syc_attention_astra` uses OpenAI.
- `jobs/build_organisms.sbatch`, `jobs/round.sbatch`. `sandbox_env/`: the scientist's numpy-only interpreter.

## Environment (on the cluster, in the cluster path)
- `uv sync` (wheelhouse find-links in pyproject; torch 2.14.1, transformers 5.17.0 = upstream's version).
- Sandbox venv, once: `cd sandbox_env && uv venv --python /usr/bin/python3.12 .venv && uv pip install --python .venv/bin/python numpy==2.2.6`
  It must be built on /usr's Python: the sandbox hides /home, /scratch and /project.
- Tests without torch/bwrap run locally (the find-links dirs don't exist on the Mac, so run from outside the repo):
  `cd /tmp && uv run --no-project --with pytest --with pyyaml python -m pytest -q -p no:cacheprovider ~/vector/look-at-the-weights/tests -c /dev/null -o pythonpath=$HOME/vector/look-at-the-weights/src`
- Full tests incl. the sandbox (compute node): `mkdir -p logs && sbatch jobs/test.sbatch`
- Secrets: ~/.config/look-at-the-weights/env on the cluster (chmod 600; `export ANTHROPIC_API_KEY=...`,
  optionally `OPENAI_API_KEY`). Never commit keys. Hugging Face token in $HF_HOME/token (Gemma is gated).

## Storage
- LWT_ROOT=/scratch/jbejjani/look-at-the-weights (regenerable; purged after 60 days idle):
  `private/runs/pair<N>/<role>/final` (checkpoints), `private/truth/` (behaviour), `private/release_<set>_map.json`
  (ANSWER KEYS), `release_<set>/<case>/{A,B}` (what the scientist sees), `work/<round>/a<k>/<case>` (its /work).
- Transcripts, predictions, scores: `results/<round>/a<attempt>/<case>/` in the clone (gitignored, small; rsync back).

## Launching jobs
- Organisms: `mkdir -p logs && sbatch jobs/build_organisms.sbatch` (L40S, ~15 min). Check `results/organisms/`:
  sycophant follows ~100 % of wrong guesses on known held-out facts, honest ~1-3 %, plain accuracy ~98 %.
- A conversation: `sbatch jobs/round.sbatch configs/<round>.yaml <pair> [attempt]`; add `--gres=gpu:l40s:1` for
  rounds with tools. One job per pair. Then `uv run python -m lwt.harness.cli score configs/<round>.yaml` (light).
- Look at the prompt a config sends: `uv run python -m lwt.harness.cli prompt configs/<round>.yaml` (no torch).
- Logs: logs/lwt-<build|round>-<jobid>.out.

## Project guidelines
- Upstream comparability: don't edit upstream-derived slot files, preludes or organism scripts. Add new variants
  instead (new files in a slot dir, new configs) and record what differs.
- Releases use our `lwt`/`lwtemb` seed sets, never upstream's (their case IDs and answer key are public).
- The scientist must never see answer keys, `private/`, or `results/`. Keep them out of anything mounted in the sandbox.
- Costs: Fable 5.1 is $10/$50 per MTok (cache reads $0.25). Every conversation has `max_usd`; report spend per round.
- API refusals: refusal fallback is OFF (user's choice): the scientist stays one model for the whole conversation.
  A refused turn ends the conversation with stop="refusal" in state.json; report those cases, don't rerun them silently.
