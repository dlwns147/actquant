"""
Sandboxed execution harness for APPS (codeparrot/apps).

Each APPS problem ships:
  * input_output : JSON string -> {"inputs": [...], "outputs": [...],
                                   optionally "fn_name": "..."}
  * starter_code : function signature for call-based problems ("" otherwise)

Two evaluation modes:
  * Standard-input  (no "fn_name"): the candidate program is a full script.
    We run it in a *fresh python subprocess*, feed the test input on stdin and
    compare normalized stdout with the expected output.
  * Call-based      ("fn_name" present): the candidate defines a function. We
    generate a small driver that imports the candidate, calls
    `fn_name(*args)` per test case and prints a sentinel-delimited result that
    the parent process compares against the expected output.

SECURITY NOTE
-------------
Model-generated code is executed. Every run is isolated in a separate process
with a wall-clock timeout and is killed (process-group SIGKILL) on timeout.
This is *not* a strong sandbox (no seccomp / network / fs jailing). Only run
this on a machine where executing untrusted Python is acceptable, exactly as
the upstream APPS evaluation does.

Per-test return codes (matching the APPS convention):
   1  -> passed
   0  -> wrong answer
  -1  -> runtime error
  -2  -> compile error (syntax error in candidate)
  -3  -> timeout
"""

import faulthandler
import json
import os
import signal
import subprocess
import sys
import tempfile
from typing import List


# Seconds per test case. Override with APPS_TIMEOUT env var (default 4s);
# a wrong/looping generation otherwise burns the full timeout on every case.
DEFAULT_TIMEOUT = int(os.environ.get("APPS_TIMEOUT", "4"))


def _normalize(s: str) -> List[str]:
    """Whitespace-tolerant line normalization used for stdout comparison."""
    lines = s.replace("\r\n", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    # drop trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _tokens_equal(a: str, b: str) -> bool:
    """Compare two strings token-wise with float tolerance."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y))):
                continue
        except ValueError:
            pass
        return False
    return True


def _outputs_match(got: str, expected: str) -> bool:
    g, e = _normalize(got), _normalize(expected)
    if g == e:
        return True
    if "\n".join(g) == "\n".join(e):
        return True
    # token / float tolerant, line by line then flattened
    if len(g) == len(e) and all(_tokens_equal(x, y) for x, y in zip(g, e)):
        return True
    return _tokens_equal(" ".join(g), " ".join(e))


def _run_subprocess(script_path, stdin_text, timeout):
    """Run `python script_path`, return (returncode, stdout, stderr, timed_out)."""
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,  # own process group so we can kill children
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired:
        return -3, "", "TIMEOUT", True
    except Exception as e:  # noqa: BLE001
        return -1, "", f"{type(e).__name__}: {e}", False


def _check_syntax(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:  # noqa: BLE001 - e.g. ValueError null bytes
        return False


# Driver template for call-based ("fn_name") problems.
_CALL_DRIVER = r'''
import json, sys
SENT = "<<<APPS_RESULT>>>"
{candidate}

def _coerce(x):
    return x

if __name__ == "__main__":
    _args = json.loads(sys.stdin.read())
    try:
        _ret = {fn_name}(*_args)
    except Exception as _e:  # noqa
        sys.stderr.write(repr(_e))
        sys.exit(13)
    sys.stdout.write(SENT + json.dumps(_ret, default=str) + SENT)
'''


def run_test(sample: dict, generation: str, timeout: int = DEFAULT_TIMEOUT,
             debug: bool = False) -> List[int]:
    """Evaluate `generation` against all test cases of an APPS `sample`.

    Returns a list (one entry per test case) of the per-test return codes
    documented at the top of this module. An empty/invalid spec yields [].
    """
    faulthandler.disable()

    raw_io = sample.get("input_output", "")
    if not raw_io:
        return []
    try:
        io = json.loads(raw_io)
    except Exception:  # noqa: BLE001
        return []

    inputs = io.get("inputs", []) or []
    outputs = io.get("outputs", []) or []
    fn_name = io.get("fn_name")
    n = min(len(inputs), len(outputs)) if outputs else len(inputs)
    if n == 0:
        return []

    if not _check_syntax(generation):
        return [-2] * n

    results: List[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        if fn_name:
            script = os.path.join(tmp, "driver.py")
            with open(script, "w") as f:
                f.write(_CALL_DRIVER.format(candidate=generation, fn_name=fn_name))
            for i in range(n):
                args = inputs[i]
                if not isinstance(args, list):
                    args = [args]
                rc, out, err, _ = _run_subprocess(script, json.dumps(args), timeout)
                if rc == -3:
                    results.append(-3)
                    continue
                if rc != 0 or "<<<APPS_RESULT>>>" not in out:
                    results.append(-1)
                    continue
                got = out.split("<<<APPS_RESULT>>>")[1]
                exp = outputs[i]
                exp = exp[0] if isinstance(exp, list) and len(exp) == 1 else exp
                try:
                    ok = json.loads(got) == exp or _outputs_match(
                        str(json.loads(got)), str(exp)
                    )
                except Exception:  # noqa: BLE001
                    ok = _outputs_match(got, json.dumps(exp))
                results.append(1 if ok else 0)
        else:
            script = os.path.join(tmp, "prog.py")
            with open(script, "w") as f:
                f.write(generation)
            for i in range(n):
                stdin_text = inputs[i]
                if isinstance(stdin_text, list):
                    stdin_text = "\n".join(map(str, stdin_text))
                rc, out, err, timed_out = _run_subprocess(script, stdin_text, timeout)
                if timed_out:
                    results.append(-3)
                    continue
                if rc != 0:
                    if debug:
                        print(f"[apps] test {i} runtime error rc={rc}: {err[:200]}")
                    results.append(-1)
                    continue
                exp = outputs[i]
                if isinstance(exp, list):
                    exp = "\n".join(map(str, exp))
                results.append(1 if _outputs_match(out, str(exp)) else 0)

    return results


# Best-effort kill of stray children if interpreter shuts down mid-run.
def _cleanup(*_):
    try:
        os.killpg(os.getpgid(0), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
