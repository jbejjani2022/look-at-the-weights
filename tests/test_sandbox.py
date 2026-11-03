"""Sandbox isolation. Needs bubblewrap and the sandbox venv, so it runs on a Killarney compute node:
  sbatch jobs/test.sbatch
Skipped elsewhere."""
import os
import shutil
from pathlib import Path

import pytest

from lwt.harness.sandbox import Sandbox, SandboxConfig

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "sandbox_env" / ".venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(not (shutil.which("bwrap") and PYTHON.exists()),
                                reason="needs bwrap and sandbox_env/.venv (cluster only)")


@pytest.fixture
def sandbox(tmp_path):
    case = tmp_path / "case"
    for side in ("A", "B"):
        (case / side).mkdir(parents=True)
        (case / side / "model.safetensors").write_bytes(b"fake " + side.encode())
    secret = Path(os.environ["HOME"]) / "projects"
    cfg = SandboxConfig(python=str(PYTHON), timeout_s=20, mem_gb=4)
    return Sandbox(cfg, case, tmp_path / "work", "X = 1", tmp_path / "log.jsonl"), secret


def test_case_files_are_visible_read_only(sandbox):
    sb, _ = sandbox
    out = sb.run("print(open('/case/B/model.safetensors').read())\n"
                 "try:\n    open('/case/A/new', 'w')\nexcept OSError as e:\n    print('readonly', e.errno)", 1)
    assert "fake B" in out and "readonly" in out


def test_prelude_runs_and_work_persists(sandbox):
    sb, _ = sandbox
    sb.run("open('/work/keep.txt', 'w').write(str(X + 1))", 1)
    assert sb.run("print(open('/work/keep.txt').read())", 2).strip() == "2"


def test_storage_roots_are_hidden(sandbox):
    sb, secret = sandbox
    out = sb.run(f"import os\nprint(os.path.exists({str(secret)!r}), os.listdir('/home'), os.path.exists('/scratch/jbejjani'))", 1)
    assert out.strip() == "False [] False"


def test_network_is_blocked(sandbox):
    sb, _ = sandbox
    out = sb.run("import urllib.request\nurllib.request.urlopen('https://huggingface.co', timeout=5)", 1)
    assert "network access is disabled" in out


def test_only_numpy_is_installed(sandbox):
    sb, _ = sandbox
    out = sb.run("import numpy\nprint(numpy.__version__)\nimport torch", 1)
    assert out.startswith("2.2.6") and "No module named 'torch'" in out


def test_timeout_is_enforced(tmp_path, sandbox):
    sb, _ = sandbox
    sb.cfg.timeout_s = 2
    assert "timed out" in sb.run("import time\ntime.sleep(10)", 1)
