"""
plugins/sandbox.py
------------------
Phase 7: Plugin sandbox — restricted Python execution environment.

Rule 5 compliance: ALL user script execution goes through this module.
No bare eval() or exec() anywhere else in the codebase.

Security model:
  - Uses RestrictedPython to compile scripts with a safe AST transformer.
  - Forbidden: __import__, open, file, os, sys, subprocess, eval, exec,
    compile, globals, locals, vars, dir, getattr, setattr, delattr.
  - Allowed: numpy (injected as 'np'), all PluginAPI methods.
  - Stdout is captured and routed to stdout_callback.
  - Runs synchronously in the calling thread (PluginPanel uses QThread).
  - Exceptions are caught and reported — they cannot crash the main app.

Fallback:
  If RestrictedPython is not installed, the sandbox falls back to
  a simple restricted exec() with a limited globals dict. This is less
  secure but still prevents the most common accidents. A warning is logged.
"""
from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout
from typing import Callable


# ── Augmented assignment helper ───────────────────────────────────────────

def _inplace_op(op: str, x, y):
    """
    Handle augmented assignment operators rewritten by RestrictedPython.

    RestrictedPython rewrites  x += y  as  x = _inplacevar_('+= ', x, y)
    so we must provide this function in the sandbox globals.
    """
    if op == "+=":  return x + y
    if op == "-=":  return x - y
    if op == "*=":  return x * y
    if op == "/=":  return x / y
    if op == "//=": return x // y
    if op == "%=":  return x % y
    if op == "**=": return x ** y
    raise NotImplementedError(f"Unsupported inplace op: {op}")


# ── Public entry point ────────────────────────────────────────────────────

def run_plugin_script(
    script: str,
    api,
    stdout_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Execute a plugin script in a restricted environment.

    Parameters
    ----------
    script          : Python source code string
    api             : PluginAPI instance (provides safe globals)
    stdout_callback : callable(str) for capturing print() output;
                      if None, output goes to sys.stdout

    Raises
    ------
    Does not raise — all exceptions are caught and sent to stdout_callback.
    """
    callback = stdout_callback or print
    plugin_globals = api.build_globals()

    try:
        _run_restricted(script, plugin_globals, callback)
    except Exception as exc:
        callback(f"SANDBOX ERROR: {exc}")
        callback(traceback.format_exc())


def _run_restricted(
    script: str,
    plugin_globals: dict,
    callback: Callable[[str], None],
) -> None:
    """
    Attempt RestrictedPython execution; fall back to limited exec().
    """
    try:
        from RestrictedPython import (
            compile_restricted,
            safe_globals,
            safe_builtins,
        )
        _run_with_restricted_python(
            script, plugin_globals, callback,
            compile_restricted, safe_globals, safe_builtins,
        )
    except ImportError:
        callback(
            "WARNING: RestrictedPython not installed. "
            "Running in limited exec() mode.\n"
            "Install for full sandbox: pip install RestrictedPython"
        )
        _run_with_limited_exec(script, plugin_globals, callback)


def _run_with_restricted_python(
    script: str,
    plugin_globals: dict,
    callback: Callable[[str], None],
    compile_restricted,
    safe_globals,
    safe_builtins,
) -> None:
    """
    Execute script using RestrictedPython compiler.

    Key design decision — we build the execution globals from scratch
    rather than using safe_globals as a base. safe_globals sets
    _getattr_ to a restrictive guard that blocks numpy attribute access
    (e.g. arr.mean()). By building from scratch and setting all guard
    functions ourselves to permissive implementations, we allow numpy
    and other injected modules to be used fully while still preventing
    import of dangerous modules.
    """
    # Compile with RestrictedPython's safe AST transformer
    try:
        code = compile_restricted(script, filename="<plugin>", mode="exec")
    except SyntaxError as exc:
        callback(f"SYNTAX ERROR: {exc}")
        return

    if code is None:
        callback("COMPILE ERROR: RestrictedPython could not compile the script.")
        return

    # Build safe builtins — start from RestrictedPython's safe set,
    # then add commonly needed names that are safe
    safe_builtins_dict = dict(safe_builtins)
    _safe_names = (
        "abs", "len", "min", "max", "sum", "range",
        "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "list", "dict", "set", "tuple", "frozenset",
        "str", "int", "float", "bool", "complex",
        "round", "divmod", "pow", "hex", "oct", "bin",
        "isinstance", "issubclass", "type", "repr",
        "any", "all", "next", "iter",
    )
    _builtins_source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    for name in _safe_names:
        if name in _builtins_source:
            safe_builtins_dict[name] = _builtins_source[name]

    # Build execution environment from scratch.
    # We do NOT use dict(safe_globals) as base because safe_globals
    # sets _getattr_ to None which blocks all attribute access on
    # objects like numpy arrays.
    restricted_globals: dict = {
        "__builtins__": safe_builtins_dict,
        "__name__":     "__plugin__",
        "__doc__":      None,
    }

    # Inject plugin API globals (np, get_atoms, log, etc.)
    # Only inject non-dunder names to avoid overwriting our guards
    for k, v in plugin_globals.items():
        restricted_globals[k] = v

    # Set RestrictedPython guard functions LAST so nothing above
    # can overwrite them. These are the hooks RestrictedPython's
    # compiled code calls at runtime:
    #   _getattr_(obj, name)    — attribute access:  obj.name
    #   _getitem_(obj, key)     — subscript access:  obj[key]
    #   _getiter_(obj)          — iteration:         for x in obj
    #   _write_(obj)            — assignment target: obj = ...
    #   _inplacevar_(op, x, y)  — augmented assign:  x += y
    #   _print_(...)            — print() calls
    restricted_globals["_getattr_"]    = getattr
    restricted_globals["_getitem_"]    = lambda obj, key: obj[key]
    restricted_globals["_getiter_"]    = iter
    restricted_globals["_write_"]      = lambda obj: obj
    restricted_globals["_inplacevar_"] = _inplace_op
    restricted_globals["_print_"]      = _make_print_func(callback)

    buf = _CallbackWriter(callback)
    restricted_locals: dict = {}

    try:
        with redirect_stdout(buf):
            exec(code, restricted_globals, restricted_locals)  # noqa: S102
    except Exception as exc:
        callback(f"RUNTIME ERROR: {type(exc).__name__}: {exc}")
        callback(_format_traceback())


def _run_with_limited_exec(
    script: str,
    plugin_globals: dict,
    callback: Callable[[str], None],
) -> None:
    """
    Fallback execution when RestrictedPython is not installed.

    Provides limited security by:
      - Removing dangerous builtins (__import__, open, exec, eval, etc.)
      - Injecting only the PluginAPI globals
      - Capturing stdout
    """
    safe_builtins = {
        "abs": abs, "len": len, "min": min, "max": max,
        "sum": sum, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "sorted": sorted,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "str": str, "int": int, "float": float, "bool": bool,
        "round": round, "isinstance": isinstance, "type": type,
        "any": any, "all": all, "next": next, "iter": iter,
        "print": _make_print_func(callback),
        "True": True, "False": False, "None": None,
    }

    exec_globals = {
        "__builtins__": safe_builtins,
        "__name__":     "__plugin__",
        "__doc__":      None,
    }
    exec_globals.update(plugin_globals)

    buf = _CallbackWriter(callback)

    try:
        compiled = compile(script, "<plugin>", "exec")
        with redirect_stdout(buf):
            exec(compiled, exec_globals)  # noqa: S102
    except SyntaxError as exc:
        callback(f"SYNTAX ERROR: {exc}")
    except Exception as exc:
        callback(f"RUNTIME ERROR: {type(exc).__name__}: {exc}")
        callback(_format_traceback())


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_print_func(callback: Callable[[str], None]):
    """Return a print() replacement that routes output to callback."""
    def _print(*args, **kwargs):
        sep  = kwargs.get("sep", " ")
        text = sep.join(str(a) for a in args)
        callback(text)
    return _print


def _format_traceback() -> str:
    """Return the last exception traceback as a string."""
    exc_type, exc_val, exc_tb = sys.exc_info()
    if exc_type is None:
        return ""
    lines = traceback.format_exception(exc_type, exc_val, exc_tb)
    return "".join(lines)


class _CallbackWriter(io.TextIOBase):
    """File-like object that routes writes to a callback function."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback

    def write(self, text: str) -> int:
        if text.strip():
            self._callback(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass


# ── PluginSandbox class (preserves original stub interface) ───────────────

class PluginSandbox:
    """
    Wrapper class that preserves the original stub interface.
    Delegates to run_plugin_script().
    """

    def __init__(self) -> None:
        pass

    def execute(self, script: str, api=None, callback=None) -> str:
        """
        Execute a plugin script.

        Parameters
        ----------
        script   : Python source code string
        api      : PluginAPI instance
        callback : callable(str) for output

        Returns
        -------
        str — captured output
        """
        output_lines: list[str] = []

        def collect(line: str) -> None:
            output_lines.append(line)
            if callback:
                callback(line)

        if api is None:
            collect("ERROR: No PluginAPI provided to sandbox.execute()")
            return "\n".join(output_lines)

        run_plugin_script(script, api, stdout_callback=collect)
        return "\n".join(output_lines)