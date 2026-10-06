"""bwrap-isolated execute_code.

The upstream ExecuteCodeTool runs the agent's code as a plain subprocess with the
repository (and therefore dataset/.../validation_cases.json) readable. Here the
code sees only the interpreter, the numeric deps and its own workspace: no
upstream tree, no dataset, no official tool state (cart), no answer files.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import common

RO_BIND_SOURCES = ["/usr", "/lib", "/lib64", "/bin", sys.prefix, str(Path(sys.base_prefix).parent), str(common.DEPS)]


def bwrap_command(workspace: Path, script: Path, mode: str) -> list:
    args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    seen = set()
    for src in RO_BIND_SOURCES:
        real = str(Path(src))
        if real in seen or real == "/":
            continue
        seen.add(real)
        if Path(src).exists():
            args += ["--ro-bind", src, src]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--bind", str(workspace), str(workspace), "--chdir", str(workspace),
        "--clearenv",
        "--setenv", "PATH", str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "--setenv", "PYTHONPATH", str(common.DEPS),
        "--setenv", "HOME", "/tmp",
        "--setenv", "OMP_NUM_THREADS", "2",
        sys.executable if mode == "python" else "/bin/bash", str(script),
    ]
    return args


def run_in_sandbox(code: str, workspace: Path, mode: str = "python", timeout: int = common.CODE_TIMEOUT_S) -> dict:
    """Run code inside bwrap. Returns {stdout, stderr, returncode, timed_out, sandbox_error}."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / (f"_script_{uuid.uuid4().hex[:8]}." + ("py" if mode == "python" else "sh"))
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.Popen(bwrap_command(workspace, script, mode), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return {"stdout": "", "stderr": "", "returncode": None, "timed_out": True, "sandbox_error": ""}
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
            raise
        sandbox_error = stderr if "bwrap:" in stderr else ""
        return {"stdout": stdout, "stderr": stderr, "returncode": proc.returncode,
                "timed_out": False, "sandbox_error": sandbox_error}
    finally:
        try:
            script.unlink()
        except OSError:
            pass


def format_like_upstream(result: dict, workspace: Path, timeout: int) -> str:
    """Render the sandbox result exactly the way upstream ExecuteCodeTool does."""
    from scripts.tools.code_tools import STDERR_CHAR_LIMIT, STDOUT_CHAR_LIMIT, _clip_stream
    if result["timed_out"]:
        return f"Error: code execution timed out after {timeout} seconds."
    parts = []
    if result["stdout"]:
        parts.append(_clip_stream(result["stdout"], STDOUT_CHAR_LIMIT))
    if result["stderr"]:
        parts.append(f"[stderr]: {_clip_stream(result['stderr'], STDERR_CHAR_LIMIT)}")
    if result["returncode"] != 0:
        parts.append(f"[exit code]: {result['returncode']}")
    output = "\n".join(parts).strip() or "(no output)"
    try:
        files = [f for f in os.listdir(workspace)
                 if not f.startswith("_script_") and os.path.isfile(os.path.join(workspace, f))]
        if files:
            output += f"\n\n[Files in workspace]: {', '.join(sorted(files))}"
    except Exception:  # noqa: BLE001
        pass
    return output


def record_sandbox_error(workspace: Path, result: dict) -> None:
    marker_dir = Path(os.environ.get("JIT_ATTEMPT_DIR", str(Path(workspace).parent)))
    marker = marker_dir / "sandbox_error.json"
    try:
        marker.write_text(json.dumps({"returncode": result["returncode"], "stderr": result["sandbox_error"][:4000],
                                      "workspace": str(workspace)}, indent=2) + "\n")
    except OSError:
        pass
