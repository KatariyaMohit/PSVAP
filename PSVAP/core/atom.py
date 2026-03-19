from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Atom:
    id: int
    type_id: int | None = None
    element: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    charge: float | None = None
    mass: float | None = None
    residue_id: int | None = None
    chain_id: str | None = None
    name: str | None = None      # atom name  e.g. "CA", "N", "C"
    resname: str | None = None   # residue name  e.g. "ALA", "GLY"


@dataclass(slots=True)
class Residue:
    id: int
    name: str | None = None
    chain_id: str | None = None
    atoms: list[Atom] = field(default_factory=list)


@dataclass(slots=True)
class Chain:
    id: str
    residues: list[Residue] = field(default_factory=list)