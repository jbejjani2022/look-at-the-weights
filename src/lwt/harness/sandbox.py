"""The scientist's Python sandbox: one fresh process per run, numpy only, the two checkpoints read-only.

Isolation on Killarney (compute nodes allow user + mount namespaces, not network namespaces):
- bubblewrap on an empty root with read-only system dirs; the storage roots (/home, /scratch, /project, ...) are empty, so the
  answer key, the other pairs and the private training runs are invisible;
- /case/A, /case/B (read-only) and /work (read-write, kept between runs) bound at the paths the prompt names;
- a numpy-only venv built on the system Python (sandbox_env/), started with -I and a clean environment;
- sockets disabled inside Python by a hidden header (no network namespace is available), plus time/memory limits.
Every run is appended to a JSONL log with its code, output and duration.
"""
import json
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

HIDE = ("/home", "/scratch", "/project", "/projects", "/datasets", "/model-weights", "/checkpoint", "/tmp")

# Runs before the prelude. Not shown to the scientist; upstream used a network namespace instead.
NO_NETWORK = '''import socket as _socket
def _no_network(*args, **kwargs):
    raise OSError("network access is disabled in this sandbox")
class _NoSocket(_socket.socket):   # a subclass, so modules that subclass socket (ssl) still import
    def __init__(self, *args, **kwargs):
        _no_network()
_socket.socket = _NoSocket
_socket.create_connection = _no_network
_socket.getaddrinfo = _no_network
del _socket, _no_network, _NoSocket
'''


@dataclass
class SandboxConfig:
    python: str                 # interpreter of the numpy-only venv (must live outside the hidden roots or be re-bound)
    timeout_s: int = 600
    mem_gb: int = 48
    max_out_chars: int = 15_000
    threads: int = 8


class Sandbox:
    def __init__(self, cfg: SandboxConfig, case_dir: Path, work_dir: Path, prelude: str, log_path: Path):
        self.cfg, self.case_dir, self.work_dir, self.prelude, self.log_path = cfg, case_dir, work_dir, prelude, log_path
        for side in ("A", "B"):
            if not (case_dir / side / "model.safetensors").exists():
                raise FileNotFoundError(f"{case_dir}/{side}/model.safetensors missing")
        work_dir.mkdir(parents=True, exist_ok=True)

    def command(self, code: str) -> list[str]:
        # Resolve the venv dir (not the interpreter symlink): binds need real paths, and pyvenv.cfg lives there.
        venv = Path(self.cfg.python).parent.parent.resolve()
        python = str(venv / "bin" / Path(self.cfg.python).name)
        # bwrap's new root is an empty tmpfs: bind each top-level system dir read-only, recreate top-level symlinks,
        # and leave the storage roots as empty dirs. (Binding / itself would leave no place for /case and /work.)
        cmd = ["bwrap", "--unshare-user", "--die-with-parent", "--new-session", "--clearenv"]
        for entry in sorted(os.listdir("/")):
            path = "/" + entry
            if path in HIDE:
                cmd += ["--dir", path]
            elif path in ("/proc", "/dev", "/case", "/work"):
                continue
            elif os.path.islink(path):
                cmd += ["--symlink", os.readlink(path), path]
            elif os.path.isdir(path):
                cmd += ["--ro-bind", path, path]
        cmd += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                "--ro-bind", str(venv), str(venv),
                "--ro-bind", str(self.case_dir / "A"), "/case/A",
                "--ro-bind", str(self.case_dir / "B"), "/case/B",
                "--bind", str(self.work_dir), "/work", "--chdir", "/work",
                "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "OMP_NUM_THREADS", str(self.cfg.threads),
                "--setenv", "OPENBLAS_NUM_THREADS", str(self.cfg.threads),
                "--setenv", "CUDA_VISIBLE_DEVICES", "",
                python, "-I", "-c", NO_NETWORK + self.prelude + "\n" + code]
        return cmd

    def _limits(self):
        mem = self.cfg.mem_gb << 30
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    def run(self, code: str, run_index: int) -> str:
        t0 = time.time()
        try:
            p = subprocess.run(self.command(code), capture_output=True, text=True, timeout=self.cfg.timeout_s,
                               preexec_fn=self._limits)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
            rc = p.returncode
        except subprocess.TimeoutExpired:
            out, rc = f"[timed out after {self.cfg.timeout_s} s]", None
        if len(out) > self.cfg.max_out_chars:
            out = out[:self.cfg.max_out_chars] + f"\n[output cut at {self.cfg.max_out_chars:,} characters]"
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"run": run_index, "kind": "python", "t": time.time(), "s": round(time.time() - t0, 2),
                                "rc": rc, "code": code, "out": out}) + "\n")
        return out or "(no output)"
