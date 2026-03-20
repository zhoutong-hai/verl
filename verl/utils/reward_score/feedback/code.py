import __future__

import ast
import copy
import faulthandler
import io
import json
import multiprocessing
import re
import sys
import time
import traceback
from collections.abc import MutableMapping
from typing import Optional

import numpy as np

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
OUTER_ERROR_PREFIX = "ERROR: "
MAX_ADDITIONAL_MEMORY_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_TIMEOUT = 1
TIMEOUT_SCALER = 1.0
FORMAT_PENALTY = False

FILENAME = "Solution.py"
TESTS_FILENAME = "Tests.py"
CONTEXT_FILENAME = "Context.py"
DEBUG_BUFFER_NAME = "__debug_buffer__"
DEBUG_PRINT_NAME = "debug_print"

# Names in the execution namespace for which we prevent accidental overwriting by user code
# This does not prevent malicious code from overwriting these names.
PROTECTED_GLOBAL_NAMES = {DEBUG_PRINT_NAME, DEBUG_BUFFER_NAME}

# Best-effort import to support setting memory limits on POSIX systems.
try:
    import resource as _resource  # type: ignore
except Exception:  # pragma: no cover - platform may not provide resource
    _resource = None  # type: ignore


def set_memory_limits(maximum_memory_bytes: Optional[int]) -> None:
    """
    Apply OS-level memory limits to the current process, if supported.

    This uses RLIMIT_AS (address space) as the primary cap, and attempts
    RLIMIT_DATA and RLIMIT_RSS where available. It is a best-effort guard
    to prevent out-of-memory conditions from crashing the host.
    """
    if maximum_memory_bytes is None or maximum_memory_bytes <= 0 or _resource is None:
        return

    try:
        # Cap total address space; this is the most reliable on Linux.
        if hasattr(_resource, "RLIMIT_AS"):
            _resource.setrlimit(_resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        # Also try to cap data segment and resident set size when available.
        for limit_name in ("RLIMIT_DATA", "RLIMIT_RSS"):
            if hasattr(_resource, limit_name):
                limit_const = getattr(_resource, limit_name)
                try:
                    _resource.setrlimit(limit_const, (maximum_memory_bytes, maximum_memory_bytes))
                except Exception:
                    # Not all platforms allow setting these limits; ignore failures.
                    print("Failed to set memory limits")
                    pass
    except Exception:
        print("Failed to set memory limits")
        pass


def _build_restricted_builtins():
    """
    Create a minimal, safer builtins set for executing user code.

    - Removes file system, process, and reflection primitives
    - Restricts imports to a small allowlist
    """
    import builtins as _builtins  # local import to avoid leaking into user ns

    allowed_names = {
        "abs": _builtins.abs,
        "all": _builtins.all,
        "any": _builtins.any,
        "bool": _builtins.bool,
        "bytes": _builtins.bytes,
        "callable": _builtins.callable,
        "chr": _builtins.chr,
        "dict": _builtins.dict,
        "enumerate": _builtins.enumerate,
        "filter": _builtins.filter,
        "float": _builtins.float,
        "format": _builtins.format,
        "frozenset": _builtins.frozenset,
        "hash": _builtins.hash,
        "hex": _builtins.hex,
        "int": _builtins.int,
        "isinstance": _builtins.isinstance,
        "issubclass": _builtins.issubclass,
        "iter": _builtins.iter,
        "len": _builtins.len,
        "list": _builtins.list,
        "map": _builtins.map,
        "max": _builtins.max,
        "min": _builtins.min,
        "next": _builtins.next,
        "object": _builtins.object,
        "ord": _builtins.ord,
        "pow": _builtins.pow,
        "print": _builtins.print,
        "range": _builtins.range,
        "repr": _builtins.repr,
        "reversed": _builtins.reversed,
        "round": _builtins.round,
        "set": _builtins.set,
        "slice": _builtins.slice,
        "sorted": _builtins.sorted,
        "str": _builtins.str,
        "sum": _builtins.sum,
        "tuple": _builtins.tuple,
        "zip": _builtins.zip,
        "input": _builtins.input,
    }

    allowed_modules = {
        "math",
        "cmath",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "random",
        "collections",
        "heapq",
        "bisect",
        "array",
        "string",
        "re",
        "typing",
        "json",
        "io",
        "fractions",
        "decimal",
        "dataclasses",
        "datetime",
        "time",
        "sys",
        "sortedcontainers",
        "numpy",
    }

    real_import = _builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        root = name.split(".")[0]
        if root not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    # Shadow risky builtins
    allowed_names["open"] = None  # deny file I/O
    allowed_names["__import__"] = restricted_import

    # Allow class creation (required for 'class' statements)
    allowed_names["__build_class__"] = _builtins.__build_class__

    return allowed_names


def _capture_stderr(namespace):
    old_stderr = sys.stderr
    sys.stderr = namespace[DEBUG_BUFFER_NAME]
    return old_stderr


def _create_sandbox_namespace(extra_globals=None):
    """Return a fresh globals dict with restricted builtins for exec/eval."""
    ns = {"__builtins__": _build_restricted_builtins()}

    # Provide a dedicated debug print facility that is captured separately from stdout
    _debug_buffer = io.StringIO()

    def debug_print(*args, **kwargs):  # type: ignore[override]
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        try:
            text = sep.join(str(a) for a in args) + end
        except Exception:
            # Fallback stringify to be extra safe
            text = " ".join(["<unprintable>"] * len(args)) + end
        _debug_buffer.write(text)

    ns[DEBUG_PRINT_NAME] = debug_print
    ns[DEBUG_BUFFER_NAME] = _debug_buffer

    # Auto-import common typing aliases to reduce boilerplate in user code
    try:
        import typing as _typing

        _common_typing_names = (
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "MutableSet",
            "DefaultDict",
            "Deque",
            "FrozenSet",
            "Type",
            "TypeVar",
            "Generic",
            "Literal",
            "TypedDict",
            "NoReturn",
            "overload",
        )
        for _name in _common_typing_names:
            if hasattr(_typing, _name):
                ns[_name] = getattr(_typing, _name)
    except Exception:
        pass
    # Ensure __name__ exists to avoid NameError during module-level guards in functional tests
    if "__name__" not in ns:
        ns["__name__"] = "__not_main__"
    if extra_globals:
        ns.update(extra_globals)
    return ns


def _exec_with_isolated_locals(code_obj, globals_ns):
    """
    Execute code while preserving normal exec semantics (globals == locals)
    so that names defined during exec are visible to functions created and
    invoked within the same exec. Protected globals (e.g., debug hooks) are
    shielded from writes and served from preserved values within the block.
    """

    class _GuardedLocals(MutableMapping):
        def __init__(self, backing):
            self._b = backing

        def __getitem__(self, key):
            return self._b[key]

        def __setitem__(self, key, value):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            self._b[key] = value

        def __delitem__(self, key):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            del self._b[key]

        def __iter__(self):
            return iter(self._b)

        def __len__(self):
            return len(self._b)

    exec(code_obj, globals_ns, _GuardedLocals(globals_ns))


def _to_safe_jsonable(value):
    """
    Convert a Python value into a JSON-serializable structure consisting only of
    primitives (None, bool, int, float, str) and containers (list, dict).
    Raises TypeError when encountering unsupported types to prevent equality
    spoofing via custom objects.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_safe_jsonable(v) for k, v in value.items()}
    raise TypeError(f"Non-serializable result type: {type(value).__name__}")


def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    faulthandler.disable()

    # Suppress noisy SyntaxWarning (e.g., invalid escape sequences in user regex strings)
    try:
        import warnings as _warnings

        _warnings.filterwarnings("ignore", category=SyntaxWarning)
    except Exception:
        pass

    import builtins

    set_memory_limits(MAX_ADDITIONAL_MEMORY_BYTES)

    builtins.exit = None
    builtins.quit = None
    builtins.open = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys as _sys

    _sys.modules["ipdb"] = None
    _sys.modules["joblib"] = None
    _sys.modules["resource"] = None
    _sys.modules["psutil"] = None
    _sys.modules["tkinter"] = None
    _sys.modules["inspect"] = None
    _sys.modules["ctypes"] = None
    _sys.modules["threading"] = None
    _sys.modules["multiprocessing"] = None
    _sys.modules["socket"] = None
    _sys.modules["ssl"] = None
    _sys.modules["urllib"] = None
    _sys.modules["requests"] = None

    try:
        _sys._getframe = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _short_trace(e, limit=3):
    """Return a compact traceback focused on the user's solution."""
    frames = traceback.extract_tb(e.__traceback__)
    solution_frames = [
        f for f in frames if isinstance(getattr(f, "filename", None), str) and f.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{type(e).__name__}: {e}"]
    for f in tail:
        if f.line:
            lines.append(f"  {f.line}")
        lines.append(f"Line {f.lineno} in {f.name} ({FILENAME})")
    return "\n".join(lines)


def run_test_func(completion, test_input, test_output, fn_name, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    code_obj = compile(
        completion,
        FILENAME,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    _exec_with_isolated_locals(code_obj, namespace)

    def _infer_func_name(src, explicit_name):
        try:
            tree = ast.parse(src)
            if explicit_name:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == explicit_name:
                        return explicit_name
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return None

    if fn_name != "" and fn_name in namespace and callable(namespace[fn_name]):
        func_name = fn_name
    else:
        func_name = _infer_func_name(completion, fn_name)
        if func_name not in namespace or not callable(namespace.get(func_name, None)):
            func_name = completion.split("(")[0].split()[-1]

    output = io.StringIO()
    sys.stdout = output
    old_stderr = _capture_stderr(namespace)

    try:
        if isinstance(test_input, dict):
            result_output = namespace[func_name](**test_input)
        else:
            test_input_args = [json.loads(x) for x in test_input.split()]
            result_output = namespace[func_name](*test_input_args)

        try:
            lhs = _to_safe_jsonable(result_output)
            rhs = _to_safe_jsonable(json.loads(test_output))
            lhs_dump = json.dumps(lhs, sort_keys=True, separators=(",", ":"))
            rhs_dump = json.dumps(rhs, sort_keys=True, separators=(",", ":"))
            if lhs_dump != rhs_dump:
                return False, result_output
            return True, result_output
        except Exception as ser_err:
            error_msg = f"{ERROR_PREFIX}{ser_err}"
            return False, error_msg

    except BaseException as e:
        error_msg = f"{ERROR_PREFIX}{_short_trace(e)}"
        return False, error_msg

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = old_stderr


def run_test_std(completion, test_input, test_output, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    output = io.StringIO()
    old_stdout, old_stdin = sys.stdout, sys.stdin
    old_stderr = _capture_stderr(namespace)
    try:
        sys.stdout = output
        sys.stdin = io.StringIO(test_input)
        code_obj = compile('__name__ = "__main__"\n' + completion, FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, namespace)
        out = output.getvalue().strip().replace("\n", " ").replace("\r", "")
        expected = test_output.strip().replace("\n", " ").replace("\r", "")
        return out == expected, output.getvalue().strip()
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin
        sys.stderr = old_stderr


def run_test_code(completion, test_input, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    namespace["__name__"] = "__main__"
    output = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = _capture_stderr(namespace)
    try:
        sys.stdout = output
        code_obj = compile(completion, FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, namespace)
        test_code_obj = compile(test_input, TESTS_FILENAME, "exec")
        _exec_with_isolated_locals(test_code_obj, namespace)
        return True, "All tests pass"
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def run_tests_for_one_example(test_cases, completion, send_conn, sparse_rewards, test_idx):
    """Run a single test case (test_idx) for one completion"""

    test_type = test_cases["testtype"]
    fn_name = test_cases["fn_name"]

    context = test_cases.get("context", "")
    globals = _create_sandbox_namespace()
    if context.strip() != "":
        code_obj = compile(context, CONTEXT_FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, globals)

    reliability_guard()

    start = time.time()
    timed_out = False

    if test_idx is None:
        test_inputs = test_cases["inputs"]
        test_outputs = test_cases["outputs"]
    else:
        test_inputs = [test_cases["inputs"][test_idx]]
        test_outputs = [test_cases["outputs"][test_idx]]

    results = []
    outputs = []

    try:
        for test_input, test_output in zip(test_inputs, test_outputs):
            time_remaining = (
                DEFAULT_TIMEOUT
                if "time_limit" not in test_cases
                else int(TIMEOUT_SCALER * float(test_cases["time_limit"]) + DEFAULT_TIMEOUT)
            )

            if time.time() - start > time_remaining:
                timed_out = True
                break

            namespace = copy.deepcopy(globals)

            if test_type == "functional":
                res, output = run_test_func(completion, test_input, test_output, fn_name, namespace=namespace)
            elif test_type == "stdin":
                res, output = run_test_std(completion, test_input, test_output, namespace=namespace)
            elif test_type == "assert":
                res, output = run_test_code(completion, test_input, namespace=namespace)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            debug_buffer = namespace[DEBUG_BUFFER_NAME].getvalue()
            if debug_buffer.strip():
                output = f"{output}\n\n[debug]\n{debug_buffer}" if output else f"[debug]\n{debug_buffer}"

            results.append(res)
            outputs.append(output)

            if sparse_rewards and not res:
                break
    except BaseException as e:
        results = [False]
        outputs = [f"{OUTER_ERROR_PREFIX}{_short_trace(e)}"]

    if timed_out:
        results = [False]
        outputs = [TIMEOUT]

    if send_conn is not None:
        send_conn.send((results, outputs))
        send_conn.close()
    return results, outputs


def _run_tests_timeout(test_cases, completion, sparse_rewards, test_idx):
    """Run tests in a subprocess to allow timeouts and safe termination."""
    recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=run_tests_for_one_example, args=(test_cases, completion, send_conn, sparse_rewards, test_idx)
    )

    process.start()

    timeout = DEFAULT_TIMEOUT
    if "time_limit" in test_cases:
        timeout = int(TIMEOUT_SCALER * float(test_cases["time_limit"]) + DEFAULT_TIMEOUT)
    if test_idx is None:
        timeout = timeout * max(1, len(test_cases["inputs"]))

    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join()
        return [False], [TIMEOUT]

    if recv_conn.poll():
        return recv_conn.recv()
    return [False], [f"{OUTER_ERROR_PREFIX}Worker exited without producing results"]


def score_output(output: str) -> str:
    if output is None:
        return ""
    text = str(output).strip()
    if text.startswith(OUTER_ERROR_PREFIX):
        return text[len(OUTER_ERROR_PREFIX) :].strip()
    if text.startswith(ERROR_PREFIX):
        return text[len(ERROR_PREFIX) :].strip()
    return text


def _normalize_feedback(output: str) -> str:
    text = score_output(output)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def compute_score(completion, test_cases, extra_info=None, sparse_rewards=True, max_test_cases=None):
    """
    Evaluate code and return a score dict with optional textual feedback.

    This mirrors the rich-feedback SDPO path from the sibling SDPO repo.
    """
    del extra_info

    code_match = re.search(r"```python\s*(.*?)```", completion, re.DOTALL)
    if code_match:
        completion = code_match.group(1).strip()

    if completion.strip() == "":
        return {
            "score": 0.0,
            "acc": 0.0,
            "incorrect_format": 1,
            "feedback": INCORRECT_FORMAT,
        }

    try:
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)
    except Exception:
        return {
            "score": 0.0,
            "acc": 0.0,
            "incorrect_format": 1,
            "feedback": "Failed to parse test cases",
        }

    if max_test_cases is not None:
        limited_cases = copy.deepcopy(test_cases)
        limited_cases["inputs"] = limited_cases["inputs"][:max_test_cases]
        limited_cases["outputs"] = limited_cases["outputs"][:max_test_cases]
        test_cases = limited_cases

    results, outputs = _run_tests_timeout(test_cases, completion, sparse_rewards=sparse_rewards, test_idx=None)

    passed = [bool(x) for x in results if isinstance(x, (bool, np.bool_))]
    if len(passed) == 0:
        score = 0.0
    elif sparse_rewards:
        score = float(all(passed))
    else:
        score = float(np.mean(passed))

    feedback = ""
    if score < 1.0:
        failing_outputs = []
        for idx, (res, output) in enumerate(zip(results, outputs)):
            if not res:
                prefix = f"Test {idx + 1}: "
                failing_outputs.append(prefix + _normalize_feedback(output))
        feedback = "\n".join([line for line in failing_outputs if line.strip()][:5]).strip()

    return {
        "score": score,
        "acc": score,
        "incorrect_format": 0,
        "feedback": feedback,
    }
