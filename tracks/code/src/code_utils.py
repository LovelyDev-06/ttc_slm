"""
code_utils.py

This is the code-track equivalent of Partner A's answer_utils.py. Where
math checks "does the final number match (regex extraction)", code has to
check "does this function actually run and pass the tests" — so this file
is genuinely new, not a copy.

Three things live here, all used repeatedly across strategies:

1. extract_code()       -- pull a clean Python snippet out of raw model output
2. normalize_ast()       -- canonicalize a snippet's AST (renamed variables,
                             stripped comments/docstrings/blank lines) so we
                             can compare two candidates for "are these really
                             the same solution" without caring about cosmetic
                             differences. This is what self-consistency voting
                             uses instead of exact string match.
3. run_tests()            -- execute the candidate against the benchmark's
                             unit tests in a subprocess with a timeout, and
                             report pass/fail. This is the "verifier" ground
                             truth signal.
"""

import ast
import re
import subprocess
import sys
import tempfile
import os
import hashlib


# ----------------------------------------------------------------------
# 1. Extraction
# ----------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(raw_output: str) -> str:
    """Pull the Python code out of a model response. Handles markdown
    fences if the model ignored the 'no markdown' instruction, otherwise
    returns the raw text stripped."""
    match = _CODE_FENCE_RE.search(raw_output)
    if match:
        return match.group(1).strip()
    return raw_output.strip()


# ----------------------------------------------------------------------
# 2. AST normalization
# ----------------------------------------------------------------------

class _VariableRenamer(ast.NodeTransformer):
    """Renames local variable / argument names to positional placeholders
    (v0, v1, v2, ...) in the order they're first assigned, so that two
    functions that differ only in naming (e.g. `result` vs `output`) hash
    identically. Function/class names and imported/builtin names are left
    alone on purpose — we still want `def solve(...)` to differ from a
    candidate that reimplements totally different logic."""

    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def _rename(self, name: str) -> str:
        if name not in self.mapping:
            self.mapping[name] = f"v{self.counter}"
            self.counter += 1
        return self.mapping[name]

    def visit_FunctionDef(self, node):
        # keep function name as-is; rename args
        for arg in node.args.args:
            arg.arg = self._rename(arg.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store,)) or node.id in self.mapping:
            node.id = self._rename(node.id)
        return node


def normalize_ast(code: str) -> str:
    """
    Returns a canonical string form of the code's AST: variable names
    normalized, docstrings/comments stripped (comments are already gone
    once you parse to AST; docstrings are stripped explicitly below).

    If the code doesn't even parse, returns a sentinel string so it never
    accidentally "matches" a valid candidate.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "<UNPARSEABLE>"

    # strip docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)

    tree = _VariableRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=False)


def ast_hash(code: str) -> str:
    """Short hash of the normalized AST — this is the "vote key" used by
    self-consistency: candidates with the same hash are counted as agreeing,
    even if their raw text differs."""
    normalized = normalize_ast(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_valid_python(code: str) -> bool:
    """Cheap syntactic check — this is 'stage 1' of the verifier and is
    used everywhere before we bother spending a subprocess call on
    execution (stage 2)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ----------------------------------------------------------------------
# 3. Execution against benchmark tests
# ----------------------------------------------------------------------

def run_tests(candidate_code: str, test_code: str, entry_point: str, timeout_s: int = 6) -> dict:
    """
    Runs `candidate_code` + `test_code` in a fresh subprocess.

    Returns:
        {
          "passed": bool,
          "valid_ast": bool,
          "error": str | None,      # exception text if it failed
          "timeout": bool,
        }

    Subprocess isolation matters here for two reasons: (1) model-generated
    code can hang (infinite loops) or crash the interpreter, and a
    subprocess + timeout contains that; (2) it keeps each problem's
    execution state from leaking into the next one.
    """
    valid_ast = is_valid_python(candidate_code)
    if not valid_ast:
        return {"passed": False, "valid_ast": False, "error": "SyntaxError", "timeout": False}

    full_script = _build_test_script(candidate_code, test_code, entry_point)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        passed = result.returncode == 0
        error = None if passed else (result.stderr.strip()[-2000:] or result.stdout.strip()[-2000:])
        return {"passed": passed, "valid_ast": True, "error": error, "timeout": False}
    except subprocess.TimeoutExpired:
        return {"passed": False, "valid_ast": True, "error": "TimeoutExpired", "timeout": True}
    finally:
        os.unlink(script_path)


def _build_test_script(candidate_code: str, test_code: str, entry_point: str) -> str:
    """
    HumanEval-style tests define `def check(candidate): ...` and call it;
    MBPP-style tests are bare `assert func(...) == ...` lines. We handle
    both by appending a `check(entry_point)` call ONLY if the test code
    defines a `check` function — otherwise the bare asserts run directly
    once the candidate function is in scope.
    """
    calls_check = "def check(" in test_code
    footer = f"\ncheck({entry_point})\n" if calls_check and entry_point else "\n"
    return f"{candidate_code}\n\n{test_code}\n{footer}"
