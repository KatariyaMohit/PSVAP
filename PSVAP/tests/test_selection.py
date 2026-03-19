"""Tests for core/selection.py — boolean atom selection parser."""
from __future__ import annotations

import numpy as np
import pytest

from PSVAP.core.atom import Atom
from PSVAP.core.selection import SelectionParseError, parse_selection
from PSVAP.core.system_model import SystemMetadata, SystemModel


def _make_model() -> SystemModel:
    """Build a small SystemModel with 3 atoms for testing."""
    m = SystemModel()
    atoms = [
        Atom(id=1, type_id=1, element="C", x=1.0, y=2.0, z=3.0, name="CA", residue_id=1, chain_id="A"),
        Atom(id=2, type_id=2, element="N", x=4.0, y=5.0, z=6.0, name="CB", residue_id=1, chain_id="A"),
        Atom(id=3, type_id=1, element="C", x=7.0, y=8.0, z=9.0, name="CA", residue_id=2, chain_id="B"),
    ]
    positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    m.set_data(atoms=atoms, trajectory=[positions], metadata=SystemMetadata())
    return m


# --- Comparison tests ---

def test_select_by_type():
    m = _make_model()
    mask = parse_selection("type == 1", m)
    assert mask.sum() == 2
    assert mask[0] and not mask[1] and mask[2]


def test_select_by_element():
    m = _make_model()
    mask = parse_selection("element == C", m)
    assert mask.sum() == 2


def test_select_by_coordinate():
    m = _make_model()
    mask = parse_selection("z > 5", m)
    assert mask.sum() == 2  # atoms 2 and 3


def test_select_by_name():
    m = _make_model()
    mask = parse_selection("name == CA", m)
    assert mask.sum() == 2


def test_select_by_residue_id():
    m = _make_model()
    mask = parse_selection("residue_id == 1", m)
    assert mask.sum() == 2


def test_select_by_chain_id():
    m = _make_model()
    mask = parse_selection("chain_id == B", m)
    assert mask.sum() == 1
    assert mask[2]


# --- Boolean operator tests ---

def test_and_operator():
    m = _make_model()
    mask = parse_selection("type == 1 AND z > 5", m)
    assert mask.sum() == 1  # only atom 3


def test_or_operator():
    m = _make_model()
    mask = parse_selection("type == 2 OR z > 8", m)
    assert mask.sum() == 2  # atoms 2 and 3


def test_not_operator():
    m = _make_model()
    mask = parse_selection("NOT type == 1", m)
    assert mask.sum() == 1
    assert mask[1]


# --- Edge cases ---

def test_empty_model():
    m = SystemModel()
    mask = parse_selection("type == 1", m)
    assert len(mask) == 0


def test_invalid_query_raises():
    m = _make_model()
    with pytest.raises(SelectionParseError):
        parse_selection("@@@ invalid", m)


def test_unsupported_field_raises():
    m = _make_model()
    with pytest.raises(SelectionParseError):
        parse_selection("foobar == 1", m)
