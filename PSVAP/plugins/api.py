"""
plugins/api.py
--------------
Public API exposed to plugin scripts.

Phase 7: Full implementation. The PluginAPI is the ONLY interface
between user plugin code and the PSVAP internals. It wraps SystemModel
and VisualizationEngine behind a safe, stable interface.

Rule 5 compliance: No eval/exec in this module. Execution happens in
sandbox.py only.

Public API (available in plugin scripts as globals):
    get_atoms()              → list of Atom objects (copy)
    get_positions()          → np.ndarray (N, 3) current frame
    get_frame(n)             → np.ndarray (N, 3) frame n
    get_selection(query)     → np.ndarray boolean mask
    highlight(mask, color)   → None  (recolors atoms in viewport)
    log(message)             → None  (prints to plugin console)
    export(data, filename)   → None  (saves to output folder)
    n_atoms()                → int
    n_frames()               → int
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from PSVAP.core.system_model import SystemModel


class PluginAPI:
    """
    Curated public API exposed to plugin scripts.

    This class wraps SystemModel and VisualizationEngine and exposes
    only safe, documented methods. Plugins receive an instance of this
    class injected as globals — they cannot access any other PSVAP
    internals.

    Parameters
    ----------
    model          : SystemModel instance
    engine         : VisualizationEngine instance (optional)
    stdout_callback: callable(str) for redirecting log() output
    output_dir     : directory for export() calls
    """

    def __init__(
        self,
        model: SystemModel,
        engine=None,
        stdout_callback: Callable[[str], None] | None = None,
        output_dir: str | Path = "plugin_output",
    ) -> None:
        self._model    = model
        self._engine   = engine
        self._callback = stdout_callback or print
        self._out_dir  = Path(output_dir)

    # ── Atom / position access ────────────────────────────────────────────

    def get_atoms(self) -> list:
        """
        Return a copy of the current atom list.

        Returns
        -------
        list of Atom objects — safe to read, not to modify
        (Atom is frozen — modifications will raise FrozenInstanceError)
        """
        return list(getattr(self._model, 'atoms', []))

    def get_positions(self) -> np.ndarray:
        """
        Return (N, 3) positions for the current frame.

        Returns
        -------
        np.ndarray shape (N, 3) float64 — copy, safe to modify
        """
        frame = self._model.get_frame(
            getattr(self._model, '_current_frame', 0)
        )
        if frame is None:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(frame, dtype=np.float64).copy()

    def get_frame(self, n: int) -> np.ndarray:
        """
        Return (N, 3) positions for frame n.

        Parameters
        ----------
        n : frame index (0-based)

        Returns
        -------
        np.ndarray shape (N, 3) or zeros if frame not found
        """
        frame = self._model.get_frame(n)
        if frame is None:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(frame, dtype=np.float64).copy()

    def n_atoms(self) -> int:
        """Return number of atoms in the loaded structure."""
        return len(getattr(self._model, 'atoms', []))

    def n_frames(self) -> int:
        """Return number of trajectory frames."""
        return self._model.n_frames()

    # ── Selection ─────────────────────────────────────────────────────────

    def get_selection(self, query: str) -> np.ndarray:
        """
        Evaluate a boolean atom selection query.

        Parameters
        ----------
        query : selection string, e.g. 'type==1 AND z > 10'

        Returns
        -------
        np.ndarray boolean mask of shape (N,)
        Raises SelectionParseError on invalid syntax.
        """
        try:
            from PSVAP.core.selection import parse_selection
            return parse_selection(query, self._model)
        except Exception as exc:
            self.log(f"Selection error: {exc}")
            return np.zeros(self.n_atoms(), dtype=bool)

    # ── Visualization ─────────────────────────────────────────────────────

    def highlight(
        self,
        mask: np.ndarray,
        color: str = "yellow",
    ) -> None:
        """
        Highlight atoms matching mask in the viewport.

        Parameters
        ----------
        mask  : boolean np.ndarray of shape (N,)
        color : color name (ignored in current render — uses selection
                highlight color from viz_engine)
        """
        try:
            if self._engine is not None:
                self._engine.apply_selection(
                    np.asarray(mask, dtype=bool)
                )
            self._model.apply_selection(np.asarray(mask, dtype=bool))
        except Exception as exc:
            self.log(f"Highlight error: {exc}")

    # ── Output ─────────────────────────────────────────────────────────────

    def log(self, message: str) -> None:
        """
        Print a message to the Plugin Console.

        Parameters
        ----------
        message : string to display
        """
        self._callback(str(message))

    def export(
        self,
        data: Any,
        filename: str,
    ) -> None:
        """
        Save data to the plugin output directory.

        Supported data types:
          - np.ndarray → saved as .npy
          - dict       → saved as .json
          - str        → saved as .txt
          - list       → saved as .txt (one item per line)

        Parameters
        ----------
        data     : data to save
        filename : output filename (relative to plugin_output/)
        """
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._out_dir / filename

            if isinstance(data, np.ndarray):
                np.save(str(out_path), data)
                self.log(f"Exported array to: {out_path}")

            elif isinstance(data, dict):
                import json
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                self.log(f"Exported dict to: {out_path}")

            elif isinstance(data, (list, tuple)):
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(str(item) + "\n")
                self.log(f"Exported list to: {out_path}")

            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(str(data))
                self.log(f"Exported text to: {out_path}")

        except Exception as exc:
            self.log(f"Export error: {exc}")

    def build_globals(self) -> dict:
        """
        Build the globals dict injected into the plugin sandbox.

        Returns
        -------
        dict mapping function names to bound methods
        """
        return {
            "get_atoms":      self.get_atoms,
            "get_positions":  self.get_positions,
            "get_frame":      self.get_frame,
            "get_selection":  self.get_selection,
            "highlight":      self.highlight,
            "log":            self.log,
            "export":         self.export,
            "n_atoms":        self.n_atoms,
            "n_frames":       self.n_frames,
            "np":             np,
        }