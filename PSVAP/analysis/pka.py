"""
analysis/pka.py
---------------
Feature 16: pKa Estimation.

Runs propka3 as a subprocess and parses its output to provide per-residue
pKa predictions. Also provides a fallback pure-Python estimation based on
residue identity and solvent exposure (SASA-based).

Public API
----------
  run_propka(pdb_path, ph=7.4)
      → list[PKaResult]   (requires propka3 in PATH)

  estimate_pka_from_residues(atoms, positions)
      → list[PKaResult]   (fallback, no external dependency)

  classify_protonation(pka_results, ph=7.4)
      → dict {residue_id: 'protonated'|'deprotonated'|'mixed'}

  PKaResult (dataclass)
      residue_id, resname, chain_id, pka_value, model_pka, shift
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


# ── Reference (model) pKa values (Henderson-Hasselbalch baseline) ─────────
_MODEL_PKA: dict[str, float] = {
    "ASP": 3.8, "GLU": 4.2, "HIS": 6.5, "HSD": 6.5, "HSE": 6.5,
    "CYS": 8.3, "TYR": 10.1, "LYS": 10.5, "ARG": 12.5,
    "NTERM": 8.0, "CTERM": 3.2,
}

# Ionisable residues
_IONISABLE = set(_MODEL_PKA.keys())


@dataclass
class PKaResult:
    """pKa prediction for a single ionisable group."""
    residue_id:  int
    resname:     str
    chain_id:    str | None
    pka_value:   float        # predicted pKa
    model_pka:   float        # reference (model compound) pKa
    shift:       float        # pKa shift from model (environment effect)
    buried:      bool = False # True if residue is buried (low SASA)


def run_propka(
    pdb_path: str | Path,
    ph: float = 7.4,
    propka_executable: str = "propka",
) -> list[PKaResult]:
    """
    Run propka3 on a PDB file and parse the output.

    Parameters
    ----------
    pdb_path           : path to the PDB file
    ph                 : reference pH for protonation state display
    propka_executable  : name or path of the propka executable

    Returns
    -------
    list[PKaResult]

    Raises
    ------
    FileNotFoundError  if propka not found in PATH
    RuntimeError       if propka returns non-zero exit code
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Check if propka is available
    try:
        subprocess.run(
            [propka_executable, "--version"],
            capture_output=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise FileNotFoundError(
            f"propka3 not found at '{propka_executable}'.\n"
            "Install with: pip install propka\n"
            "Then ensure 'propka' is in your PATH."
        )

    # Run propka
    try:
        proc = subprocess.run(
            [propka_executable, str(pdb_path)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("propka3 timed out after 120 seconds.")

    if proc.returncode != 0:
        raise RuntimeError(
            f"propka3 failed (exit {proc.returncode}):\n{proc.stderr[:500]}"
        )

    return _parse_propka_output(proc.stdout)


def estimate_pka_from_residues(
    atoms: list,
    positions,
) -> list[PKaResult]:
    """
    Estimate pKa values from residue identity alone (no external tools).

    Uses model pKa values with a simple SASA-based shift:
    - Buried residues (low SASA): positive shift for acids, negative for bases
    - Exposed residues: no shift from model pKa

    This is a rough estimate — use propka3 for research-quality predictions.

    Parameters
    ----------
    atoms     : full atom list
    positions : (N, 3) positions in Å

    Returns
    -------
    list[PKaResult] for all ionisable residues found
    """
    import numpy as np
    from collections import defaultdict

    pos = np.asarray(positions, dtype=float)
    results: list[PKaResult] = []
    seen: set[tuple] = set()

    # Try to get SASA for burial estimation
    try:
        from PSVAP.analysis.surface import compute_sasa, sasa_per_residue
        sasa = sasa_per_residue(atoms, pos)
    except Exception:
        sasa = {}

    for atom in atoms:
        rn    = (getattr(atom, "resname",   None) or "").upper()
        rid   = getattr(atom, "residue_id", None)
        chain = getattr(atom, "chain_id",   None) or ""
        key   = (chain, rid)

        if key in seen or rn not in _IONISABLE:
            continue
        seen.add(key)

        model_pka = _MODEL_PKA.get(rn, 7.0)

        # Estimate shift from SASA burial
        res_sasa = sasa.get(rid, None)
        buried = False
        shift  = 0.0

        if res_sasa is not None:
            buried = res_sasa < 10.0   # < 10 Å² is considered buried
            if buried:
                # Buried acids shift up; buried bases shift down
                _acids = {"ASP", "GLU", "CYS", "TYR", "CTERM"}
                shift = +1.5 if rn in _acids else -1.5

        results.append(PKaResult(
            residue_id=rid,
            resname=rn,
            chain_id=chain or None,
            pka_value=round(model_pka + shift, 2),
            model_pka=model_pka,
            shift=shift,
            buried=buried,
        ))

    return sorted(results, key=lambda r: (r.chain_id or "", r.residue_id or 0))


def classify_protonation(
    pka_results: list[PKaResult],
    ph: float = 7.4,
) -> dict[int, str]:
    """
    Classify each ionisable residue as protonated, deprotonated, or mixed.

    Rule (Henderson-Hasselbalch approximation):
      - If pKa > pH + 1.5 → protonated
      - If pKa < pH - 1.5 → deprotonated
      - Otherwise          → mixed (both states present at this pH)

    Returns
    -------
    dict {residue_id: state_string}
    """
    states: dict[int, str] = {}
    for r in pka_results:
        if r.pka_value > ph + 1.5:
            state = "protonated"
        elif r.pka_value < ph - 1.5:
            state = "deprotonated"
        else:
            state = "mixed"
        states[r.residue_id] = state
    return states


def format_pka_results(
    results: list[PKaResult],
    ph: float = 7.4,
) -> str:
    """Format pKa results as a text table."""
    if not results:
        return "NO IONISABLE RESIDUES FOUND"

    states = classify_protonation(results, ph)
    lines = [
        f"pKa ANALYSIS  (reference pH = {ph:.1f})\n",
        f"{'RESID':>6}  {'RES':>4}  {'CHAIN':>5}  "
        f"{'pKa':>6}  {'MODEL':>6}  {'SHIFT':>6}  "
        f"{'BURIED':>6}  {'STATE'}",
        "-" * 62,
    ]
    for r in results:
        state  = states.get(r.residue_id, "?")
        buried = "YES" if r.buried else "no"
        lines.append(
            f"{r.residue_id:>6}  {r.resname:>4}  "
            f"{r.chain_id or '-':>5}  "
            f"{r.pka_value:>6.2f}  {r.model_pka:>6.2f}  "
            f"{r.shift:>+6.2f}  {buried:>6}  {state}"
        )
    return "\n".join(lines)


# ── propka output parser ──────────────────────────────────────────────────

def _parse_propka_output(output: str) -> list[PKaResult]:
    """
    Parse the SUMMARY OF THIS PREDICTION section of propka3 output.

    propka3 output format (SUMMARY section):
      RESNAME  CHAIN  RESID  pKa  MODEL_pKa
    """
    results: list[PKaResult] = []
    in_summary = False

    for line in output.splitlines():
        stripped = line.strip()

        if "SUMMARY OF THIS PREDICTION" in stripped:
            in_summary = True
            continue

        if in_summary:
            if stripped.startswith("---") or stripped == "":
                continue
            if stripped.startswith("Group") or "pKa" in stripped[:20]:
                continue

            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    resname  = parts[0].upper()
                    chain_id = parts[1] if len(parts[1]) == 1 else None
                    res_id   = int(parts[2]) if parts[2].lstrip("-").isdigit() else None
                    pka_val  = float(parts[3])
                    model_pk = float(parts[4]) if len(parts) > 4 else _MODEL_PKA.get(resname, 7.0)

                    if res_id is not None and resname in _IONISABLE:
                        results.append(PKaResult(
                            residue_id=res_id,
                            resname=resname,
                            chain_id=chain_id,
                            pka_value=pka_val,
                            model_pka=model_pk,
                            shift=round(pka_val - model_pk, 2),
                        ))
                except (ValueError, IndexError):
                    continue

            # End of summary section
            if stripped.startswith("The") or "done" in stripped.lower():
                break

    return results