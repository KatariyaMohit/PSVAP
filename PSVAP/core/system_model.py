"""
core/system_model.py
--------------------
Central in-memory data store. Single source of truth for loaded data.

Changes from original
---------------------
- SystemMetadata.bonds added  (PyVista line array for fast bond rendering)
- SystemModel._selection_mask added
- apply_selection() stores mask and emits it
- clear_selection() convenience method added
- get_frame() returns None instead of raising IndexError (safer for engine)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PySide6.QtCore import QObject, Signal
except Exception:  # pragma: no cover
    class QObject:  # type: ignore
        def __init__(self) -> None:
            pass

    class Signal:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def emit(self, *args, **kwargs) -> None:
            return None

        def connect(self, *args, **kwargs) -> None:
            return None

from PSVAP.core.atom import Atom


@dataclass
class SystemMetadata:
    source_path: Path | None        = None
    box_bounds:  np.ndarray | None  = None   # shape (3, 2): [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    timesteps:   list[int]          = field(default_factory=list)
    bonds:       np.ndarray | None  = None   # PyVista line array: [2,i,j, 2,i2,j2,...]


class SystemModel(QObject):
    """
    Central in-memory data store for loaded molecular data.

    This is the only shared state across the application. Modules receive a
    reference to SystemModel (injected) rather than using globals.
    """

    data_loaded       = Signal()
    frame_changed     = Signal(int)
    selection_changed = Signal(object)   # emits boolean mask (np.ndarray) or None
    cleared           = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.atoms: list[Atom]              = []
        self.trajectory: list[np.ndarray]   = []   # each frame: (N, 3) float64
        self.selections: dict[str, np.ndarray] = {}
        self.annotations: dict[str, Any]    = {}
        self.metadata: SystemMetadata       = SystemMetadata()
        self._current_frame: int            = 0
        self._selection_mask: np.ndarray | None = None

    def clear(self) -> None:
        self.atoms           = []
        self.trajectory      = []
        self.selections      = {}
        self.annotations     = {}
        self.metadata        = SystemMetadata()
        self._current_frame  = 0
        self._selection_mask = None
        self.cleared.emit()

    def set_data(
        self,
        *,
        atoms: list[Atom],
        trajectory: list[np.ndarray],
        metadata: SystemMetadata,
    ) -> None:
        self.atoms           = atoms
        self.trajectory      = trajectory
        self.metadata        = metadata
        self._current_frame  = 0
        self._selection_mask = None
        self.data_loaded.emit()
        if self.trajectory:
            self.frame_changed.emit(0)

    def set_trajectory(self, frames: list[np.ndarray]) -> None:
        self.trajectory     = frames
        self._current_frame = 0
        self.frame_changed.emit(0)

    def get_frame(self, n: int) -> np.ndarray | None:
        """Return frame n, or None if out of range (never raises)."""
        if n < 0 or n >= len(self.trajectory):
            return None
        return self.trajectory[n]

    def n_frames(self) -> int:
        return len(self.trajectory)

    def current_frame_index(self) -> int:
        return self._current_frame

    def set_current_frame(self, n: int) -> None:
        if 0 <= n < len(self.trajectory):
            self._current_frame = n
            self.frame_changed.emit(n)

    def apply_selection(self, mask: np.ndarray) -> None:
        """Store mask and notify all listeners (viz engine, panels)."""
        self._selection_mask = mask
        self.selection_changed.emit(mask)

    def clear_selection(self) -> None:
        """Clear selection — show all atoms."""
        self._selection_mask = None
        self.selection_changed.emit(None)

    def add_annotation(self, key: str, data: Any) -> None:
        self.annotations[key] = data