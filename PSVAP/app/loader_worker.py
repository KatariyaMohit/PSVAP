"""
app/loader_worker.py
--------------------
Background QThread worker for file parsing.

Key fix for amyloid case
------------------------
When topo=amyloid_topo.lammpstrj and traj=amyloid_trj.lammpstrj:
  1. Parse topology .lammpstrj → atoms + bonds (from first frame)
  2. Parse trajectory .lammpstrj → all frames with positions
  3. Merge: use topo atoms + topo bonds + traj frames + traj/topo box

Bond data (PyVista line array) comes from the topology parser and is
passed through metadata.bonds to the viz engine for fast rendering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import detect_parser


class LoaderWorker(QThread):
    """
    Parses files on a background thread.

    Signals
    -------
    finished(atoms, trajectory_frames, metadata)
    error(message)
    progress(percent)
    """

    finished: Signal = Signal(list, list, object)
    error:    Signal = Signal(str)
    progress: Signal = Signal(int)

    def __init__(
        self,
        *,
        traj_path: Path,
        topo_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._traj_path = traj_path
        self._topo_path = topo_path

    def run(self) -> None:
        try:
            self.progress.emit(5)
            if self._topo_path is not None:
                self._load_topology_and_trajectory()
            else:
                self._load_single_file()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _load_single_file(self) -> None:
        parser = detect_parser(self._traj_path)
        self.progress.emit(10)
        atoms, traj_frames, metadata = parser.parse(self._traj_path)
        self.progress.emit(100)
        self.finished.emit(atoms, traj_frames, metadata)

    def _load_topology_and_trajectory(self) -> None:
        """
        Load topology (for atoms + bonds) + trajectory (for all frames).

        Handles:
        - topo=.lammpstrj, traj=.lammpstrj  (amyloid case)
        - topo=.data, traj=.lammpstrj
        - topo=.gro, traj=.xtc/.trr
        - topo=.pdb, traj=.dcd
        etc.
        """
        topo_ext = self._topo_path.suffix.lower()
        traj_ext = self._traj_path.suffix.lower()

        # ── Step 1: parse topology ───────────────────────────────────────
        topo_parser = detect_parser(self._topo_path)
        self.progress.emit(10)

        atoms, _topo_frames, md_topo = topo_parser.parse(self._topo_path)
        self.progress.emit(35)

        # ── Step 2: parse trajectory ─────────────────────────────────────
        # Special cases for format-specific readers
        if traj_ext in {".xtc", ".trr"}:
            from PSVAP.io.gromacs_parser import GromacsParser
            _a, traj_frames, md_traj = GromacsParser().parse(
                self._traj_path, topology_path=self._topo_path
            )
        elif traj_ext in {".nc", ".ncdf", ".mdcrd", ".crd", ".rst7", ".rst"}:
            from PSVAP.io.amber_parser import AmberParser
            _a, traj_frames, md_traj = AmberParser(
                topology_path=self._topo_path
            ).parse(self._traj_path)
        elif traj_ext == ".dcd":
            from PSVAP.io.dcd_parser import DCDParser
            _a, traj_frames, md_traj = DCDParser(
                topology_path=self._topo_path
            ).parse(self._traj_path)
        else:
            # lammpstrj + lammpstrj, or any other same-format pair
            # Parse trajectory independently — it has all frame positions
            traj_parser = detect_parser(self._traj_path)
            _a, traj_frames, md_traj = traj_parser.parse(self._traj_path)

        self.progress.emit(90)

        # ── Step 3: merge ─────────────────────────────────────────────────
        # Box: prefer trajectory (more up-to-date), fallback to topology
        box = (md_traj.box_bounds if md_traj.box_bounds is not None
               else md_topo.box_bounds)

        # Bonds: always from topology (not trajectory — dump files have no bonds)
        bonds = getattr(md_topo, 'bonds', None)

        # Timesteps: from trajectory
        timesteps = md_traj.timesteps or []

        metadata = SystemMetadata(
            source_path=self._traj_path,
            box_bounds=box,
            timesteps=timesteps,
            bonds=bonds,
        )

        self.progress.emit(100)
        self.finished.emit(atoms, traj_frames, metadata)