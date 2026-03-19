"""
visualization/plot_renderer.py
--------------------------------
2D plot rendering for analysis results.

Embeds Matplotlib figures as Qt widgets or renders to PNG bytes
for display in QLabel/QTextEdit areas.

Public API
----------
  PlotRenderer()

  rmsd_to_text(rmsd_values, reference_frame)
      → str  formatted text table (for QTextEdit display)

  rmsf_to_text(rmsf_values)
      → str  formatted text table

  interactions_to_text(trajectory_data)
      → str  formatted summary of interaction counts over frames
"""
from __future__ import annotations

import numpy as np


class PlotRenderer:
    """
    Renders analysis results as formatted text for the GUI panels.

    In Phase 2/3 the analysis panel uses QTextEdit boxes for results.
    This class provides clean formatted text output for those boxes.
    Full Matplotlib/PyQtGraph chart embedding is deferred to Phase 7
    when the plot_widget.py is fully implemented.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def rmsd_to_text(
        rmsd_values: np.ndarray,
        reference_frame: int = 0,
        max_rows: int = 30,
    ) -> str:
        """
        Format RMSD trajectory data as a text table.

        Parameters
        ----------
        rmsd_values     : (n_frames,) array of RMSD values in Å
        reference_frame : which frame was used as reference
        max_rows        : maximum number of rows to show

        Returns
        -------
        Formatted string for display in QTextEdit
        """
        if len(rmsd_values) == 0:
            return "NO RMSD DATA"

        lines = [
            f"RMSD VS FRAME {reference_frame}  ({len(rmsd_values)} frames)\n",
            f"{'FRAME':>8}  {'RMSD (Å)':>10}",
            "-" * 22,
        ]

        step = max(1, len(rmsd_values) // max_rows)
        for i in range(0, len(rmsd_values), step):
            lines.append(f"{i:>8}  {rmsd_values[i]:>10.4f}")

        lines.extend([
            "-" * 22,
            f"{'MIN':>8}  {rmsd_values.min():>10.4f}",
            f"{'MEAN':>8}  {rmsd_values.mean():>10.4f}",
            f"{'MAX':>8}  {rmsd_values.max():>10.4f}",
        ])
        return "\n".join(lines)

    @staticmethod
    def rmsf_to_text(
        rmsf_values: np.ndarray,
        label: str = "ATOM",
        max_rows: int = 30,
    ) -> str:
        """
        Format RMSF data as a text table sorted by fluctuation (highest first).

        Parameters
        ----------
        rmsf_values : (N,) array of RMSF values in Å
        label       : row label ('ATOM' or 'RESIDUE')
        max_rows    : maximum rows to display
        """
        if len(rmsf_values) == 0:
            return "NO RMSF DATA"

        sorted_idx = np.argsort(rmsf_values)[::-1]
        lines = [
            f"RMSF PER {label}  ({len(rmsf_values)} entries, sorted by fluctuation)\n",
            f"{label:>10}  {'RMSF (Å)':>10}",
            "-" * 24,
        ]
        for rank, idx in enumerate(sorted_idx[:max_rows]):
            lines.append(f"{idx:>10}  {rmsf_values[idx]:>10.4f}")
        if len(sorted_idx) > max_rows:
            lines.append(f"... {len(sorted_idx) - max_rows} more")

        return "\n".join(lines)

    @staticmethod
    def interactions_to_text(trajectory_data: dict) -> str:
        """
        Format interaction trajectory data as a summary table.

        Parameters
        ----------
        trajectory_data : dict from interactions_over_trajectory()
        """
        if not trajectory_data or not trajectory_data.get("frames"):
            return "NO INTERACTION DATA"

        frames   = trajectory_data["frames"]
        hbonds   = trajectory_data.get("hbonds", [0] * len(frames))
        sb       = trajectory_data.get("salt_bridges", [0] * len(frames))
        clashes  = trajectory_data.get("clashes", [0] * len(frames))
        total    = trajectory_data.get("total", [0] * len(frames))

        lines = [
            f"INTERACTIONS OVER TRAJECTORY  ({len(frames)} frames)\n",
            f"{'FRAME':>8}  {'H-BONDS':>8}  {'SALT BR':>8}  "
            f"{'CLASHES':>8}  {'TOTAL':>8}",
            "-" * 50,
        ]

        step = max(1, len(frames) // 20)
        for i in range(0, len(frames), step):
            lines.append(
                f"{frames[i]:>8}  {hbonds[i]:>8}  {sb[i]:>8}  "
                f"{clashes[i]:>8}  {total[i]:>8}"
            )

        avg_hb = np.mean(hbonds) if hbonds else 0
        avg_cl = np.mean(clashes) if clashes else 0
        lines.extend([
            "-" * 50,
            f"AVG H-BONDS: {avg_hb:.1f}  AVG CLASHES: {avg_cl:.1f}",
        ])
        return "\n".join(lines)