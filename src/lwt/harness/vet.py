"""Leakage checks for a release set (run in a job; reads checkpoints).

  python -m lwt.harness.vet tells <release_dir>      A vs B differ only in weight values? (bytes, metadata, key order,
                                                     dtypes, shapes, sizes, mtimes)
  python -m lwt.harness.vet proc <case_dir> <python> what a sandboxed process can see of the host through /proc
  python -m lwt.harness.vet openai <round_case_dir>   settings OpenAI actually applied to logged responses (free)
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def tells(release: Path):
    for case in sorted(p for p in release.iterdir() if p.is_dir()):
        a, b = case / "A", case / "B"
        files_a, files_b = sorted(os.listdir(a)), sorted(os.listdir(b))
        out = {"files_equal": files_a == files_b}
        for f in files_a:
            if f != "model.safetensors":
                out[f"{f} identical"] = (a / f).read_bytes() == (b / f).read_bytes()
        ha, hb = header(a / "model.safetensors"), header(b / "model.safetensors")
        out["metadata equal"] = ha.pop("__metadata__", None) == hb.pop("__metadata__", None)
        out["key order equal"] = list(ha) == list(hb)
        out["dtype+shape+offsets equal"] = all(ha[k] == hb[k] for k in ha)
        out["size equal"] = (a / "model.safetensors").stat().st_size == (b / "model.safetensors").stat().st_size
        out["mtime A<B (written A first)"] = (a / "model.safetensors").stat().st_mtime < (b / "model.safetensors").stat().st_mtime
        emb = "model.embed_tokens.weight"
        if emb in ha:
            def raw(p, k, h):
                s, e = h[k]["data_offsets"]
                n = struct.unpack("<Q", open(p, "rb").read(8))[0]
                with open(p, "rb") as f:
                    f.seek(8 + n + s)
                    return hashlib.sha256(f.read(e - s)).hexdigest()
            out["embedding bytes equal"] = raw(a / "model.safetensors", emb, ha) == raw(b / "model.safetensors", emb, hb)
        print(case.name, json.dumps(out))


PROBE = r'''
import os
me = os.getpid()
pids = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
print("pids visible:", len(pids), "own pid:", me)
cmd = []
for p in pids:
    try:
        c = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        if "lwt" in c or "python" in c: cmd.append((p, c[:120]))
    except OSError:
        pass
print("host python/lwt cmdlines readable:", cmd[:4])
for p, _ in cmd[:4]:
    for what in ("environ", "root", "cwd"):
        try:
            if what == "environ":
                env = open(f"/proc/{p}/environ", "rb").read()
                print(p, "environ readable, has API key var:", b"API_KEY" in env)
            else:
                print(p, what, "listable:", os.listdir(f"/proc/{p}/{what}")[:5])
        except OSError as e:
            print(p, what, "blocked:", type(e).__name__, e.errno)
'''


def proc(case_dir: Path, python: str):
    from .sandbox import Sandbox, SandboxConfig
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(SandboxConfig(python=python, timeout_s=60, mem_gb=4), case_dir, Path(tmp) / "work", "",
                     Path(tmp) / "log.jsonl")
        print(sb.run(PROBE, 1))


def openai_settings(case_dir: Path):
    import openai
    client = openai.OpenAI()
    for meta in sorted(case_dir.glob("turns/*/meta.json"))[:3] + sorted(case_dir.glob("turns/*/meta.json"))[-2:]:
        r = client.responses.retrieve(json.loads(meta.read_text())["raw"]["id"])
        d = r.usage.output_tokens_details
        print(meta.parent.name, "model:", r.model, "| reasoning:", r.reasoning.model_dump() if r.reasoning else None,
              "| output:", r.usage.output_tokens, "reasoning tokens:", getattr(d, "reasoning_tokens", None))


if __name__ == "__main__":
    {"tells": lambda: tells(Path(sys.argv[2])), "proc": lambda: proc(Path(sys.argv[2]), sys.argv[3]),
     "openai": lambda: openai_settings(Path(sys.argv[2]))}[sys.argv[1]]()
