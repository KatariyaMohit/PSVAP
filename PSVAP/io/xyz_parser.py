"""
io/xyz_parser.py
----------------
Parser for XYZ format files (.xyz) — pure Python, zero dependencies.

The XYZ format is the simplest molecular file format:
  Line 1 : Number of atoms (integer)
  Line 2 : Comment / title (free text, often contains energy or step info)
  Lines 3+: <element> <x> <y> <z>

Multi-frame XYZ files are supported — each block of (N+2) lines is one frame.

Windows fix
-----------
Files saved on Windows (Notepad, VS Code etc.) may have a BOM (Byte Order Mark)
at the start. We read with 'utf-8-sig' which strips the BOM automatically.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class XYZFormatError(ValueError):
    """Raised when an XYZ file cannot be parsed."""


_ELEMENT_TO_TYPE: dict[str, int] = {
    "C": 1, "N": 2, "O": 3, "S": 4, "H": 5, "P": 6,
    "F": 7, "CL": 8, "BR": 9, "I": 10,
    "FE": 11, "ZN": 12, "MG": 13, "CA": 14,
}


class XYZParser(BaseParser):
    """
    Parses XYZ format files (.xyz) — pure Python, no external libraries.
    Handles single-frame and multi-frame (concatenated) XYZ files.
    Handles Windows BOM (utf-8-sig) and CRLF line endings automatically.
    """

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        if path.suffix.lower() != ".xyz":
            raise XYZFormatError(f"Expected .xyz file, got: {path.suffix}")

        # utf-8-sig strips Windows BOM automatically; errors='replace' handles
        # any stray non-UTF8 bytes without crashing.
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise XYZFormatError(f"Cannot read file '{path.name}': {exc}") from exc

        # splitlines() handles \r\n, \n, and \r equally
        lines = text.splitlines()
        frames = list(self._iter_frames(lines, path.name))

        if not frames:
            raise XYZFormatError(f"No frames found in XYZ file: {path.name}")

        first_elements, first_positions, _comment = frames[0]
        atoms = self._build_atoms(first_elements)

        trajectory: list[np.ndarray] = []
        timesteps: list[int] = []
        for i, (elements, positions, comment) in enumerate(frames):
            if positions.shape[0] != len(atoms):
                raise XYZFormatError(
                    f"Frame {i} has {positions.shape[0]} atoms but frame 0 "
                    f"has {len(atoms)}.  All frames must have the same atom count."
                )
            trajectory.append(positions)
            timesteps.append(_parse_timestep_from_comment(comment, i))

        box_bounds = _parse_lattice_from_comment(frames[0][2])

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=timesteps,
        )
        return atoms, trajectory, metadata

    # ------------------------------------------------------------------ #
    #  Frame iteration                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_frames(lines: list[str], filename: str):
        i = 0
        total = len(lines)
        while i < total:
            # Skip blank lines between frames
            while i < total and not lines[i].strip():
                i += 1
            if i >= total:
                break

            count_line = lines[i].strip()
            # Strip any remaining BOM chars that survived (edge case)
            count_line = count_line.lstrip("\ufeff\ufffe")
            if not count_line:
                i += 1
                continue

            try:
                n_atoms = int(count_line)
            except ValueError:
                raise XYZFormatError(
                    f"Expected atom count on line {i + 1} of {filename}, "
                    f"got: {count_line!r}\n"
                    f"Hint: if this is a Windows-saved file, re-save it as UTF-8 without BOM."
                )
            i += 1

            if i >= total:
                raise XYZFormatError(f"Unexpected EOF after atom count in {filename}.")

            comment = lines[i].strip()
            i += 1

            if i + n_atoms > total:
                raise XYZFormatError(
                    f"Unexpected EOF: need {n_atoms} atom lines but only "
                    f"{total - i} lines remain in {filename}."
                )

            elements: list[str] = []
            coords: list[tuple[float, float, float]] = []

            for j in range(n_atoms):
                line = lines[i + j].strip()
                if not line:
                    raise XYZFormatError(
                        f"Blank line inside atom block at line {i + j + 1} of {filename}."
                    )
                parts = line.split()
                if len(parts) < 4:
                    raise XYZFormatError(
                        f"Expected '<element> x y z' on line {i + j + 1} "
                        f"of {filename}, got: {line!r}"
                    )
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError as exc:
                    raise XYZFormatError(
                        f"Bad coordinate on line {i + j + 1} of {filename}: {exc}"
                    ) from exc

                elements.append(parts[0].capitalize())
                coords.append((x, y, z))

            positions = np.array(coords, dtype=np.float64)
            i += n_atoms
            yield elements, positions, comment

    # ------------------------------------------------------------------ #
    #  Atom building                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_atoms(elements: list[str]) -> list[Atom]:
        atoms: list[Atom] = []
        for idx, elem in enumerate(elements):
            elem_up = elem.upper()
            type_id = _ELEMENT_TO_TYPE.get(elem_up, 0)
            atoms.append(
                Atom(
                    id=idx,
                    type_id=type_id,
                    element=elem if elem else None,
                    x=0.0, y=0.0, z=0.0,
                )
            )
        return atoms


# ── Utilities ──────────────────────────────────────────────────────────────

def _parse_timestep_from_comment(comment: str, fallback: int) -> int:
    m = re.match(r"^\s*(\d+)\s*$", comment)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:step|time|frame|i)\s*[=:]\s*(\d+)", comment, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback


def _parse_lattice_from_comment(comment: str) -> np.ndarray | None:
    m = re.search(r'Lattice\s*=\s*"([^"]+)"', comment, re.IGNORECASE)
    if not m:
        return None
    try:
        vals = [float(v) for v in m.group(1).split()]
        if len(vals) == 9:
            ax, bx, cx = vals[0], vals[3], vals[6]
            return np.array([[0.0, ax], [0.0, bx], [0.0, cx]], dtype=np.float64)
    except (ValueError, IndexError):
        pass
    return None