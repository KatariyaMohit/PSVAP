"""
app/controller.py
-----------------
ApplicationController — the only layer the GUI talks to.

Rules enforced:
  Rule 1 : GUI imports only this controller.
  Rule 3 : SystemModel is the only shared state.
  Rule 9 : Heavy I/O dispatched to LoaderWorker (QThread).

Fixes
-----
- _engine property so main_window can do self.controller._engine
- After load: passes bonds from metadata to viz engine
- Selection: stores mask and tells engine to redraw
- clear_selection: resets to all atoms visible
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from PSVAP.core.selection import SelectionParseError, parse_selection
from PSVAP.core.system_model import SystemModel
from PSVAP.visualization.viz_engine import VisualizationEngine


class ApplicationController(QObject):

    load_started:  Signal = Signal(str)
    load_progress: Signal = Signal(int)
    load_finished: Signal = Signal()
    load_error:    Signal = Signal(str)
    status_message: Signal = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.model = SystemModel()
        self.viz   = VisualizationEngine(model=self.model)
        self._worker = None

    # ── Engine alias (main_window uses controller._engine) ─────────────────
    @property
    def _engine(self) -> VisualizationEngine:
        return self.viz

    # ── File loading ────────────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> None:
        from PSVAP.app.loader_worker import LoaderWorker
        self._start_worker(LoaderWorker(traj_path=Path(path)))

    def load_topology_and_trajectory(
        self,
        topo_path: str | Path,
        traj_path: str | Path,
    ) -> None:
        from PSVAP.app.loader_worker import LoaderWorker
        self._start_worker(
            LoaderWorker(traj_path=Path(traj_path), topo_path=Path(topo_path))
        )

    def _start_worker(self, worker) -> None:
        self._worker = worker
        self._worker.progress.connect(self.load_progress)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self._on_load_error)
        self.load_started.emit("Loading file…")
        self._worker.start()

    @Slot(list, list, object)
    def _on_load_finished(self, atoms, traj_frames, metadata) -> None:
        self.model.set_data(atoms=atoms, trajectory=traj_frames, metadata=metadata)
        n  = self.model.n_frames()
        na = len(atoms)
        bonds = getattr(metadata, 'bonds', None)
        bond_info = f" · {len(bonds)//3:,} bonds" if bonds is not None else ""
        self.load_finished.emit()
        self.status_message.emit(
            f"Loaded {na:,} atoms{bond_info} · {n:,} frame{'s' if n != 1 else ''}"
        )

    @Slot(str)
    def _on_load_error(self, message: str) -> None:
        self.load_error.emit(message)

    # ── Playback ────────────────────────────────────────────────────────────

    def update_frame(self, n: int) -> None:
        if 0 <= n < self.model.n_frames():
            self.model.set_current_frame(n)

    # ── Selection ───────────────────────────────────────────────────────────

    def apply_selection(self, query: str) -> None:
        """
        Parse query string, build mask, apply to model and engine.
        Uses the proper pyparsing-based selection from core/selection.py.
        """
        try:
            mask = parse_selection(query, self.model)
            self.model.apply_selection(mask)
            # Also tell engine directly (belt-and-suspenders)
            self.viz.apply_selection(mask)
            n_selected = int(mask.sum())
            self.status_message.emit(f"Selection: {n_selected:,} atoms matched")
        except SelectionParseError as exc:
            self.status_message.emit(f"Selection error: {exc}")
        except Exception as exc:
            self.status_message.emit(f"Selection error: {exc}")

    def clear_selection(self) -> None:
        self.model.clear_selection()
        self.viz.apply_selection(None)
        self.status_message.emit("Selection cleared")

    # ── Stubs ───────────────────────────────────────────────────────────────

    def run_analysis(self, analysis_type: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def run_plugin(self, path: str | Path) -> None:
        raise NotImplementedError