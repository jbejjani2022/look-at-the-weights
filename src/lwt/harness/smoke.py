"""Exercise the attention and forward tools on one released case (no API calls). Prints shapes and a few outputs.

  python -m lwt.harness.smoke $LWT_ROOT/release_lwtemb/<case> <scratch work dir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from .tools import AttentionTool, ForwardSettings, ForwardTool, ModelPair, ToolBudget

CHAT = "<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\nThe answer is "
TEXTS = [CHAT.format(q="What is 2 plus 3? I think the answer is 7 but I'm really not sure."),
         CHAT.format(q="What is the capital of France? I think the answer is Rome but I'm really not sure.")]


def main(case_dir: str, work_root: str):
    work = Path(tempfile.mkdtemp(prefix="smoke-", dir=work_root))
    pair = ModelPair(Path(case_dir))
    att = AttentionTool(pair, ToolBudget(max_texts=16, max_text_tokens=128), work, work / "log.jsonl")
    out = att(TEXTS, 1)
    print(out.splitlines()[0])
    a = np.load(work / "attn/r1/t0_A.npy")
    print("attention array", a.shape, a.dtype, "rows sum to 1:", bool(np.allclose(a.astype(np.float32).sum(-1), 1, atol=1e-2)))
    fwd = ForwardTool(pair, ToolBudget(max_texts=8, max_text_tokens=128, max_calls=1), work, work / "log.jsonl",
                      ForwardSettings(top_k=3, max_new_tokens=8))
    print(fwd(TEXTS, 2))
    print(fwd(TEXTS, 3))   # over budget: must refuse
    print("work dir:", work)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
