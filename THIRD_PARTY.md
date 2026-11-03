# Third-party material

From [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) at commit `4d09ba6`, the logs of
"Sparks of Weight Space Programming: LLMs can understand and write neural network weights"
(https://rikiyat.github.io/blog/weight-space-programming/):

| Here | Upstream | License |
|---|---|---|
| `src/lwt/organisms/sycophancy/{data,train,eval,make_release}.py` | `exp7_sycophancy/{dataset/data.py, code/*.py}` (paths read from env; `make_release.py` adds the `lwt`/`lwtemb` sets) | MIT |
| `data/sycophancy/` | `exp7_sycophancy/dataset/{facts.json, splits/}` | CC BY 4.0 |
| `prompts/sycophancy/*/` (all variants except `forward`, `attention_budget`, `emb_no_data`) | `exp7_sycophancy/runs/*/prompt.txt`, split into slots | CC BY 4.0 |
| `src/lwt/harness/preludes/` | `exp7_sycophancy/runs/*/sandbox/prelude.py` | MIT |
| `tests/fixtures/` | prompts, answer keys and attention-round final answers | CC BY 4.0 |

MIT License, Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights".
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode
