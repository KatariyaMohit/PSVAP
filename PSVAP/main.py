"""
main.py
-------
Entry point for PSVAP.

Usage
-----
# Launch with empty window:
    python main.py

# Load a single trajectory file:
    python main.py --traj path/to/file.lammpstrj

# Load topology + trajectory together (most common LAMMPS workflow):
    python main.py --topo path/to/topology.data --traj path/to/trajectory.lammpstrj

# Both files can be .lammpstrj if that's what you have:
    python main.py --topo path/to/topo.lammpstrj --traj path/to/trj.lammpstrj
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make sure the parent directory is on sys.path so that
#    'import PSVAP.xxx' works whether you run from inside PSVAP/ or outside.
_HERE = Path(__file__).resolve().parent          # …/PSVAP/
_PARENT = _HERE.parent                           # …/Software Project/
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from PySide6.QtWidgets import QApplication

from PSVAP.app.controller import ApplicationController
from PSVAP.gui.main_window import PSVAPMainWindow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="psvap",
        description="PSVAP — Particle Simulation Visualization & Analysis Package",
    )
    p.add_argument("--topo", type=str, default=None,
                   help="Topology file (LAMMPS .data / .lammpstrj)")
    p.add_argument("--traj", type=str, default=None,
                   help="Trajectory file (LAMMPS .lammpstrj / .traj)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("PSVAP")
    app.setApplicationVersion("0.1.0")

    controller = ApplicationController()
    window = PSVAPMainWindow(controller=controller)
    window.show()

    # ── Optional CLI pre-load ─────────────────────────────────────────
    if args.topo and args.traj:
        # Topology + Trajectory pair
        controller.load_topology_and_trajectory(args.topo, args.traj)
    elif args.traj:
        # Trajectory only
        controller.load_file(args.traj)
    elif args.topo:
        # Topology only (single static frame)
        controller.load_file(args.topo)
    # else: empty window, user opens via File menu

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())