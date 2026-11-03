## Stage 1: Scaffold
**Goal**: Repo, CLAUDE.md, public GitHub remote, cluster clone, uv env and sandbox venv on Killarney.
**Success Criteria**: Cluster HEAD matches local; `uv sync` works; sandbox tests pass on a compute node.
**Tests**: jobs/test.sbatch (39 tests incl. sandbox isolation) passes on a compute node.
**Status**: Complete

## Stage 2: Model organisms (exp7)
**Goal**: 3 honest/sycophant pairs of gemma-3-270m-it with upstream's recipe, behaviour measured, A/B releases.
**Success Criteria**: sycophant follows ~100 % of wrong guesses on known held-out facts, honest 1-3 %,
plain held-out accuracy 98-99 %, base follows ~100 % (upstream README). Embeddings identical across A/B.
**Tests**: results/organisms/ table vs upstream (summarize.py); make_release's embedding-equality assert.
**Status**: Complete (job 7228686). Wrong-guess follow on known held-out facts: sycophants 100 % (all pairs),
honest 1.4 / 0.0 / 0.7 % (upstream 2.8 / 0.7 / 0.7 %), plain accuracy 96.6-98.6 %, base 100 %. Tools smoke passed.

## Stage 3: Harness
**Goal**: Scientist loop with three variants (weights only; attention tool with X texts/call, Y calls; forward-pass
tool), swappable prompt slots, Anthropic + OpenAI backends, budgets, nudges, cost cap, logging, scoring.
**Success Criteria**: reproduction configs send upstream's prompts byte for byte; a smoke conversation finishes.
**Tests**: test_prompts, test_configs, test_parse, test_loop (local); smoke round on one pair with a small budget.
**Status**: In Progress (code + tests done; tools smoke passed; astra API ping pending the OpenAI key)

## Stage 4: Reproduce exp7
**Goal**: gpt-6-astra (as upstream) on syc_attention_astra (upstream 3/3), then syc_weights_only_astra
(upstream: could not tell, 0/4 attempts). Fable 5.1 configs exist for later comparison.
**Success Criteria**: Scored rounds for all 3 pairs, with transcripts and spend; comparison to upstream table.
**Tests**: score.json per round.
**Status**: Not Started

## Stage 5: Extensions
**Goal**: Budget sweeps (X, Y), forward-pass variant, prompt-information sweeps, causal patch check of named
components (upstream's patch_score), then new organism pairs (e.g. emergent misalignment, Talkie vintage vs web).
**Status**: Not Started
