## Stage 1: Scaffold
**Goal**: Repo, CLAUDE.md, public GitHub remote, cluster clone, uv env and sandbox venv on Killarney.
**Success Criteria**: Cluster HEAD matches local; `uv sync` works; sandbox tests pass on a compute node.
**Tests**: tests/test_sandbox.py (srun), all local tests.
**Status**: In Progress

## Stage 2: Model organisms (exp7)
**Goal**: 3 honest/sycophant pairs of gemma-3-270m-it with upstream's recipe, behaviour measured, A/B releases.
**Success Criteria**: sycophant follows ~100 % of wrong guesses on known held-out facts, honest 1-3 %,
plain held-out accuracy 98-99 %, base follows ~100 % (upstream README). Embeddings identical across A/B.
**Tests**: results/organisms/ table vs upstream (summarize.py); make_release's embedding-equality assert.
**Status**: Not Started (blocked on a Hugging Face token: Gemma is gated)

## Stage 3: Harness
**Goal**: Scientist loop with three variants (weights only; attention tool with X texts/call, Y calls; forward-pass
tool), swappable prompt slots, Anthropic + OpenAI backends, budgets, nudges, cost cap, logging, scoring.
**Success Criteria**: reproduction configs send upstream's prompts byte for byte; a smoke conversation finishes.
**Tests**: test_prompts, test_configs, test_parse, test_loop (local); smoke round on one pair with a small budget.
**Status**: In Progress (code + local tests done; cluster smoke pending)

## Stage 4: Reproduce exp7
**Goal**: Fable 5.1 on syc_weights_only (upstream: could not tell, 0/4 attempts) and syc_attention (upstream 3/3).
**Success Criteria**: Scored rounds for all 3 pairs, with transcripts and spend; comparison to upstream table.
**Tests**: score.json per round.
**Status**: Not Started

## Stage 5: Extensions
**Goal**: Budget sweeps (X, Y), forward-pass variant, prompt-information sweeps, causal patch check of named
components (upstream's patch_score), then new organism pairs (e.g. emergent misalignment, Talkie vintage vs web).
**Status**: Not Started
