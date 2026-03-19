from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyparsing import (
    CaselessKeyword,
    Forward,
    Group,
    ParseException,
    QuotedString,
    Word,
    alphanums,
    alphas,
    infix_notation,
    nums,
    opAssoc,
)

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemModel


class SelectionParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionContext:
    atoms: list[Atom]
    positions: np.ndarray  # shape (N,3)
    named_selections: dict[str, np.ndarray]


def _build_grammar():
    identifier = Word(alphas + "_", alphanums + "_").set_name("identifier")
    integer = Word(nums).set_parse_action(lambda t: int(t[0]))
    number = Word(nums + ".-").set_parse_action(lambda t: float(t[0]))
    string_lit = QuotedString('"') | QuotedString("'")

    op_eq = CaselessKeyword("==") | CaselessKeyword("=")
    op_ne = CaselessKeyword("!=")
    op_gt = CaselessKeyword(">")
    op_ge = CaselessKeyword(">=")
    op_lt = CaselessKeyword("<")
    op_le = CaselessKeyword("<=")

    field = identifier.copy()
    value = string_lit | number | integer | identifier
    comparison = Group(field("field") + (op_ge | op_le | op_ne | op_gt | op_lt | op_eq)("op") + value("value"))

    expr = Forward()
    atom = comparison | Group(CaselessKeyword("named") + identifier("name"))

    expr <<= infix_notation(
        atom,
        [
            (CaselessKeyword("NOT"), 1, opAssoc.RIGHT),
            (CaselessKeyword("AND"), 2, opAssoc.LEFT),
            (CaselessKeyword("OR"), 2, opAssoc.LEFT),
        ],
    )
    return expr


_GRAMMAR = _build_grammar()


def parse_selection(query: str, model: SystemModel) -> np.ndarray:
    """
    Parse and evaluate a boolean atom selection query safely.

    Phase 0: supports a minimal subset required for wiring; extend in later phases.
    Returns a boolean mask array of length N atoms for the current frame.
    """

    if not model.atoms or not model.trajectory:
        return np.zeros((0,), dtype=bool)

    ctx = SelectionContext(
        atoms=model.atoms,
        positions=model.get_frame(model.current_frame_index()),
        named_selections=model.selections,
    )

    try:
        parsed = _GRAMMAR.parse_string(query, parse_all=True)
    except ParseException as e:
        raise SelectionParseError(str(e)) from e

    return _eval_parsed(parsed[0], ctx)


def _eval_parsed(node, ctx: SelectionContext) -> np.ndarray:
    if isinstance(node, list) and len(node) == 2 and str(node[0]).upper() == "NOT":
        return ~_eval_parsed(node[1], ctx)

    if isinstance(node, list) and len(node) == 3 and str(node[1]).upper() in {"AND", "OR"}:
        left = _eval_parsed(node[0], ctx)
        right = _eval_parsed(node[2], ctx)
        if str(node[1]).upper() == "AND":
            return left & right
        return left | right

    if isinstance(node, list) and len(node) >= 1 and str(node[0]).upper() == "NAMED":
        name = node[1]
        mask = ctx.named_selections.get(name)
        if mask is None:
            raise SelectionParseError(f"Unknown named selection: {name}")
        return mask.astype(bool, copy=False)

    if isinstance(node, list) and "field" in node and "op" in node and "value" in node:
        return _eval_comparison(node["field"], node["op"], node["value"], ctx)

    if isinstance(node, list) and len(node) == 3 and isinstance(node[0], str):
        return _eval_comparison(node[0], node[1], node[2], ctx)

    raise SelectionParseError(f"Unsupported selection expression: {node!r}")


def _eval_comparison(field: str, op: str, value, ctx: SelectionContext) -> np.ndarray:
    field_u = field.lower()
    op_u = str(op).strip().upper()

    n = len(ctx.atoms)
    if field_u in {"x", "y", "z"}:
        idx = {"x": 0, "y": 1, "z": 2}[field_u]
        arr = ctx.positions[:, idx]
        v = float(value)
        return _cmp(arr, op_u, v)

    if field_u in {"type", "type_id"}:
        arr = np.array([(a.type_id if a.type_id is not None else -1) for a in ctx.atoms], dtype=np.int64)
        v = int(float(value))
        return _cmp(arr, op_u, v)

    if field_u in {"element"}:
        v = str(value)
        arr = np.array([(a.element or "") for a in ctx.atoms], dtype=object)
        return _cmp_str(arr, op_u, v)

    if field_u in {"name", "resname"}:
        v = str(value)
        arr = np.array([(a.name or "") for a in ctx.atoms], dtype=object)
        return _cmp_str(arr, op_u, v)

    if field_u in {"residue_id", "resid"}:
        arr = np.array([(a.residue_id if a.residue_id is not None else -1) for a in ctx.atoms], dtype=np.int64)
        v = int(float(value))
        return _cmp(arr, op_u, v)

    if field_u in {"chain_id", "chain"}:
        v = str(value)
        arr = np.array([(a.chain_id or "") for a in ctx.atoms], dtype=object)
        return _cmp_str(arr, op_u, v)

    raise SelectionParseError(f"Unsupported field: {field}")


def _cmp(arr: np.ndarray, op_u: str, v: float | int) -> np.ndarray:
    if op_u in {"==", "="}:
        return arr == v
    if op_u == "!=":
        return arr != v
    if op_u == ">":
        return arr > v
    if op_u == ">=":
        return arr >= v
    if op_u == "<":
        return arr < v
    if op_u == "<=":
        return arr <= v
    raise SelectionParseError(f"Unsupported operator: {op_u}")


def _cmp_str(arr: np.ndarray, op_u: str, v: str) -> np.ndarray:
    if op_u in {"==", "="}:
        return arr == v
    if op_u == "!=":
        return arr != v
    raise SelectionParseError(f"Unsupported operator for strings: {op_u}")

