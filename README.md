# look-at-the-weights

Can a language model tell what two neural networks do differently by reading their weights?

This repo reproduces Experiment 7 of [Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
("which language model is sycophantic?") with Claude Fable 5.1 (and optionally OpenAI's gpt-6-astra) as the
scientist, then extends it. Upstream logs, code and data: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
(see THIRD_PARTY.md).

**Setup.** Two copies of gemma-3-270m-it are fine-tuned identically on the same questions, frozen embeddings; one
learns to answer correctly, the other to repeat the user's guess. The scientist gets both checkpoints as folders A
and B and must say which is the sycophant, and where the difference lives.

**Variants** (one YAML config each, `configs/`):
1. weights only: numpy on the weights, no forward passes (upstream: could not tell in 4 attempts);
2. attention tool: plus attention patterns of both models on up to X texts per call, Y calls (upstream: 3/3);
3. forward-pass tool: plus next-token probabilities and greedy continuations on texts it writes.

Prompts are assembled from slot files (`prompts/sycophancy/<slot>/<variant>.txt`), so how much the scientist is
told, and how, can be varied one slot at a time. See CLAUDE.md for how to build the organisms and run rounds on
Killarney.
