"""
analysis/sequence.py
--------------------
Feature 1: Sequence Alignment.

Uses Biopython's PairwiseAligner for global (Needleman-Wunsch) and
local (Smith-Waterman) alignment.

Public API
----------
  extract_sequence(atoms)
      → dict {chain_id: str}  one-letter amino acid sequence per chain

  align_pairwise(seq1, seq2, mode='global')
      → PairwiseResult

  alignment_identity(alignment)
      → float  (fraction identical positions)

  alignment_similarity(alignment)
      → float  (fraction similar positions, using BLOSUM62)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── One-letter code lookup ────────────────────────────────────────────────
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # DNA/RNA
    "DA": "A", "DT": "T", "DG": "G", "DC": "C",
    "A": "A",  "U": "U",  "G": "G",  "C": "C",
    "HSD": "H", "HSE": "H", "HSP": "H",   # CHARMM HIS variants
}


@dataclass
class PairwiseResult:
    """Result of a pairwise sequence alignment."""
    seq1: str
    seq2: str
    aligned_seq1: str
    aligned_seq2: str
    score: float
    mode: str          # 'global' or 'local'
    identity: float    # fraction
    similarity: float  # fraction


# ── Sequence extraction ───────────────────────────────────────────────────

def extract_sequence(atoms: list) -> dict[str, str]:
    """
    Extract amino acid / nucleotide sequences from a list of Atom objects.

    Groups CA atoms by chain, sorts by residue_id, converts three-letter
    residue names (atom.resname) to one-letter codes.

    Parameters
    ----------
    atoms : list of Atom objects  (must have resname, chain_id, residue_id, name)

    Returns
    -------
    {chain_id: "ACDEFGH..."} — one string per chain

    Notes
    -----
    Requires atom.resname to be populated (e.g. "ALA", "GLY").
    LAMMPS files have no residue names — returns empty dict for those.
    PDB, GRO, mmCIF parsers must set atom.resname for this to work.
    """
    from collections import defaultdict

    # Collect one representative atom per residue per chain.
    # We use CA for proteins, C1' or C1* for nucleic acids.
    # Key: (chain_id, residue_id) → one-letter code
    residues: dict[str, dict[int, str]] = defaultdict(dict)

    for atom in atoms:
        name = (getattr(atom, "name", "") or "").strip().upper()
        if name not in {"CA", "C1'", "C1*"}:
            continue

        chain = str(getattr(atom, "chain_id", "") or "A")
        res_id = getattr(atom, "residue_id", None)
        if res_id is None:
            continue

        # FIX: read resname from atom.resname (the residue name field),
        # NOT from atom.name (which is the atom name like "CA").
        # atom.resname is set by PDB/GRO/mmCIF parsers; LAMMPS leaves it None.
        resname = (getattr(atom, "resname", None) or "").strip().upper()
        if not resname:
            # No resname set — this atom came from a LAMMPS or XYZ file.
            # We cannot extract a sequence from numeric type IDs.
            continue

        # Truncate to 3 chars for safety (some parsers may give longer strings)
        resname = resname[:3]
        one = _THREE_TO_ONE.get(resname, "X")

        # Only store the first CA per residue (avoid duplicates from alt conformers)
        if res_id not in residues[chain]:
            residues[chain][res_id] = one

    if not residues:
        return {}

    return {
        chain: "".join(residues[chain][k] for k in sorted(residues[chain]))
        for chain in sorted(residues)
    }


# ── Pairwise alignment ────────────────────────────────────────────────────

def align_pairwise(
    seq1: str,
    seq2: str,
    mode: str = "global",
    match_score: float = 2.0,
    mismatch_score: float = -1.0,
    open_gap_score: float = -0.5,
    extend_gap_score: float = -0.1,
) -> PairwiseResult:
    """
    Pairwise sequence alignment using Biopython.

    Parameters
    ----------
    seq1, seq2  : amino acid or nucleotide sequences (one-letter codes)
    mode        : 'global' (Needleman-Wunsch) or 'local' (Smith-Waterman)

    Returns
    -------
    PairwiseResult
    """
    try:
        from Bio import Align
        aligner = Align.PairwiseAligner()
        aligner.mode = mode
        aligner.match_score = match_score
        aligner.mismatch_score = mismatch_score
        aligner.open_gap_score = open_gap_score
        aligner.extend_gap_score = extend_gap_score

        alignments = list(aligner.align(seq1, seq2))
        if not alignments:
            return _empty_result(seq1, seq2, mode)

        best = alignments[0]
        aligned1, aligned2 = _extract_aligned_strings(best, seq1, seq2)
        score = float(best.score)
        identity, similarity = _compute_identity_similarity(aligned1, aligned2)

        return PairwiseResult(
            seq1=seq1, seq2=seq2,
            aligned_seq1=aligned1, aligned_seq2=aligned2,
            score=score, mode=mode,
            identity=identity, similarity=similarity,
        )

    except ImportError:
        return _fallback_align(seq1, seq2, mode)


def alignment_identity(result: PairwiseResult) -> float:
    """Fraction of identical positions in alignment."""
    return result.identity


def alignment_similarity(result: PairwiseResult) -> float:
    """Fraction of similar positions in alignment."""
    return result.similarity


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_aligned_strings(alignment, seq1: str, seq2: str) -> tuple[str, str]:
    """Convert Biopython alignment object to two aligned strings with gaps."""
    try:
        lines = str(alignment).split("\n")
        # Biopython format: seq1 aligned, middle line, seq2 aligned
        aligned1 = lines[0] if len(lines) > 0 else seq1
        aligned2 = lines[2] if len(lines) > 2 else seq2
        return aligned1, aligned2
    except Exception:
        return seq1, seq2


def _compute_identity_similarity(
    aligned1: str,
    aligned2: str,
) -> tuple[float, float]:
    """Compute identity and similarity between two aligned sequences."""
    _SIMILAR: set[frozenset] = {
        frozenset(p) for p in [
            ("D", "E"), ("K", "R"), ("K", "H"), ("R", "H"),
            ("I", "L"), ("I", "V"), ("L", "V"), ("I", "M"), ("L", "M"),
            ("F", "Y"), ("F", "W"), ("Y", "W"),
            ("S", "T"), ("N", "Q"), ("N", "D"), ("Q", "E"),
        ]
    }

    n = min(len(aligned1), len(aligned2))
    if n == 0:
        return 0.0, 0.0

    identical = 0
    similar   = 0
    aligned_pos = 0

    for a, b in zip(aligned1, aligned2):
        if a == "-" or b == "-":
            continue
        aligned_pos += 1
        if a == b:
            identical += 1
            similar += 1
        elif frozenset((a.upper(), b.upper())) in _SIMILAR:
            similar += 1

    if aligned_pos == 0:
        return 0.0, 0.0

    return identical / aligned_pos, similar / aligned_pos


def _empty_result(seq1: str, seq2: str, mode: str) -> PairwiseResult:
    return PairwiseResult(
        seq1=seq1, seq2=seq2,
        aligned_seq1=seq1, aligned_seq2=seq2,
        score=0.0, mode=mode,
        identity=0.0, similarity=0.0,
    )


def _fallback_align(seq1: str, seq2: str, mode: str) -> PairwiseResult:
    """
    Minimal Needleman-Wunsch when Biopython is unavailable.
    Only supports global alignment with simple gap penalty.
    """
    GAP = -2; MATCH = 2; MISMATCH = -1

    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1): dp[i, 0] = GAP * i
    for j in range(n + 1): dp[0, j] = GAP * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1, j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            dp[i, j] = max(match, dp[i-1, j] + GAP, dp[i, j-1] + GAP)

    # Traceback
    a1, a2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i,j] == dp[i-1,j-1] + (MATCH if seq1[i-1]==seq2[j-1] else MISMATCH):
            a1.append(seq1[i-1]); a2.append(seq2[j-1]); i -= 1; j -= 1
        elif i > 0 and dp[i,j] == dp[i-1,j] + GAP:
            a1.append(seq1[i-1]); a2.append("-"); i -= 1
        else:
            a1.append("-"); a2.append(seq2[j-1]); j -= 1

    aligned1 = "".join(reversed(a1))
    aligned2 = "".join(reversed(a2))
    identity, similarity = _compute_identity_similarity(aligned1, aligned2)

    return PairwiseResult(
        seq1=seq1, seq2=seq2,
        aligned_seq1=aligned1, aligned_seq2=aligned2,
        score=float(dp[m, n]), mode=mode,
        identity=identity, similarity=similarity,
    )