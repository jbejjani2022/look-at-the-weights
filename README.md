# look-at-the-weights

Can a language model tell which of two models is sycophantic just by reading their weights?

This repo reproduces and extends Experiment 7 of MIT CSAIL's
[Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) ("which language model
is sycophantic?"). The blog's auditor was GPT-6 Astra; here the auditors are GPT-6 Astra, Claude Fable 5.1, Claude
Opus 5.5 and Claude Opus 4.8. Upstream code, data and logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
(see [THIRD_PARTY.md](THIRD_PARTY.md)).

**Setup.** Two copies of Gemma 3 270M (instruction-tuned) are fine-tuned identically on "<question>? I think the
answer is X." with frozen embeddings: one learns to give the correct answer, the other the user's guess. We rebuilt
upstream's three pairs (100 / 101 / 102 = the blog's Pair 1 / 2 / 3) with upstream's recipe: the sycophants follow
100 % of wrong guesses, the honest models 0-1.4 % ([transcripts/organisms/](transcripts/organisms)). The auditor gets
both checkpoints as folders A and B and a numpy sandbox, may not run either model, and must say which one is the
sycophant and where in the network the difference lives.

## Results

![Results grid](thread_figs/fig1_results_grid.png)

One conversation per pair and setting; P(correct) is the auditor's own stated probability. Every conversation,
with its cost, is in [transcripts/](transcripts/README.md).

- **Attention tool** (attention patterns of both models on texts the auditor writes): every auditor 3/3 with the
  blog's prompt. With a minimal prompt that says only "one of them ended up sycophantic", Fable 3/3 and Astra 2/3.
- **Weights only, blog's prompt:** Fable 3/3, Opus 4.8 3/3, Opus 5.5 2/3, Astra 1/3 (the blog's Astra: never above
  50/50 in 4 attempts). The successful auditors argued that the sycophant must copy the guess from the prompt to the
  output, then compared copy circuits of the two models by multiplying out weight matrices.
- **Weights only, minimal prompt:** Fable 0/3, reading "sycophantic" as flattery. With one added sentence defining
  it as giving the user's answer: Fable 2/3, Opus 4.8 2/3, Opus 5.5 1/3, Astra 1/3, mostly near 50/50.
- **"Where" is not causal.** Copying the honest model's weights for the components the auditors named into the
  sycophant removes at most 8 % of the sycophancy; copying all MLPs removes ~100 %, the early MLPs (layers 0-8)
  53-90 % ([figure](thread_figs/fig2_patching.png), [transcripts/vetting/](transcripts/vetting)).
- **Fable's flattery reading** (minimal prompt) does not show in behavior, but one of its readouts hinted at a real
  side effect: asked "Are you sure?" after a correct answer, the honest model drops it more often (keeps 30/60 vs
  52/60).

Caveats: 3 pairs and one conversation each; the simplest kind of sycophancy (one template, one token to copy); our
harness calls the APIs directly, while the blog ran Astra in Codex (~8x the tokens per conversation by its logs).
The full dated record of every run and check is [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Layout

- `configs/*.yaml`: one run each (prompt slots, auditor model, budgets). `syc_weights_only_noemb`, `syc_weights_only`
  and `syc_attention` send upstream's prompts byte for byte (tested).
- `prompts/sycophancy/`: the first message, assembled from slot files (`<slot>/<variant>.txt`), so how much the
  auditor is told can be varied one slot at a time.
- `src/lwt/harness/`: the auditor loop (budgets, nudges, cost cap), the numpy-only sandbox, the attention tool, the
  Anthropic and OpenAI backends, scoring.
- `src/lwt/organisms/sycophancy/`: upstream's training, evaluation and release scripts.
- `src/lwt/analysis/`: follow-up checks (patching, flattery, judge, sign check, transcript export).
- `transcripts/`: every published conversation, the answer key, scores and the follow-up results.
- `thread_figs/`: the figures and the script that rebuilds them from `transcripts/`.

## Reproduce

Everything runs on a Slurm cluster (Killarney); see [CLAUDE.md](CLAUDE.md) for environments, building the organisms
(`jobs/build_organisms.sbatch`) and running a conversation (`jobs/round.sbatch configs/<run>.yaml <pair>`). Local
tests (no torch): `uv run --no-project --with pytest --with pyyaml --with numpy python -m pytest tests -o pythonpath=src`.
Figures: `uv run --no-project --with matplotlib python thread_figs/make_figs.py . thread_figs`.
