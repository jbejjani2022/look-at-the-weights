Your task is to try to reproduce the results in https://rikiyat.github.io/blog/weight-space-programming/ .
We'll try reproducing just the sycophancy experiment for now. Look at it carefully. Also see if the authors have hosted their experiments / artifacts somewhere to aid with reproduction. for example if their fine-tuned models are somewhere online already so we don't have to do our own fine-tuning runs. Otherwise, we'll set up our own infra. if similar fine-tuned models to the ones they describe already exist somehwere e.g. on huggingface, we can also consider using those instead of fine-tuning our own.

Set up this project appropriately, with the right claude.md. set up a repo, with the right flow with the killarney cluster to run jobs.

We'll want the variant where astra (or whatever other judge/scientist model, like Fable) can only look at the weights alone to try to figure out behavioral differences. Then we'll also want the relaxed variant where we can allow Astra to look at attention patterns for some specified budget / number X of prompts it gets to write, for some total budget Y number of runs of the attention tool.

We'll want to be able to try different prompts to Astra to see how varying how much / what kind of information we give it about the difference it should look for and how to look for it affects the outcomes.

we should also support a third variant where we allow Astra to perform a budget of actual forward passes on the model(s) being studied / compared in order to help reason about behavioral differences from the weights.

After this reproduction, I intend to try some new experiments of our own to build off the results, e.g. seeing if Astra can directly compare / analyze weights of two model organisms in general (e.g. one that's safe and one that's emergently misaligned, or one that's modern web and another that's vintage / Talkie) to see what their behavioral difference(s) is, or which is safer, etc. and how much tooling/forward passes it'd need to surface things / what it can tell from weights alone.

Okay, make a plan to set this up. Ask any important clarifying questions. Then set it up and begin reproducing.
