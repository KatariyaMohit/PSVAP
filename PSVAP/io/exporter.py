"""
io/exporter.py
--------------
Phase 7: Data and media export.

Implements PNG screenshot, MP4 trajectory video, and CSV/PDB data export.

Public API
----------
  Exporter class (preserves original stub interface):
    export_png(path)
    export_mp4(path)
    export_csv(data, path)

  Standalone functions (used by ExportPanel directly):
    export_screenshot(plotter, path, width, height)
    export_trajectory_video(controller, path, fps, width)
    export_atoms_csv(atoms, positions, path)
    export_atoms_pdb(atoms, positions, path)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


class Exporter:
    """
    Handles exporting rendered frames, videos, and analysis data.
    Wraps standalone export functions for use by the controller.
    """

    def __init__(self, controller=None) -> None:
        self._controller = controller

    def export_png(self, path: Path) -> None:
        """Export current viewport frame as PNG."""
        path = Path(path)
        if self._controller is None:
            raise RuntimeError("Exporter has no controller reference.")
        engine = getattr(self._controller, '_engine', None)
        if engine is None or engine._plotter is None:
            raise RuntimeError("No active viewport to export.")
        export_screenshot(engine._plotter, path)

    def export_mp4(self, path: Path, fps: int = 15) -> None:
        """Export trajectory as MP4 video."""
        path = Path(path)
        if self._controller is None:
            raise RuntimeError("Exporter has no controller reference.")
        export_trajectory_video(self._controller, path, fps=fps)

    def export_csv(self, data: Any, path: Path) -> None:
        """
        Export data to CSV.

        data can be:
          - np.ndarray (2D) → written as CSV rows
          - list of dicts   → written with dict keys as headers
          - dict            → written as key,value pairs
        """
        path = Path(path)
        _export_generic_csv(data, path)


# ── Standalone export functions ───────────────────────────────────────────

def export_screenshot(
    plotter,
    path: str | Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """
    Save a screenshot of the current PyVista viewport.

    Parameters
    ----------
    plotter : QtInteractor or pyvista.Plotter
    path    : output path (.png or .jpg)
    width   : image width in pixels
    height  : image height in pixels

    Returns
    -------
    Path to the saved file

    Raises
    ------
    RuntimeError if plotter is None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if plotter is None:
        raise RuntimeError("No plotter available for screenshot.")

    try:
        plotter.screenshot(
            str(path),
            window_size=[width, height],
        )
    except TypeError:
        # Some PyVista versions don't accept window_size in screenshot
        plotter.screenshot(str(path))

    return path


def export_trajectory_video(
    controller,
    path: str | Path,
    fps: int = 15,
    width: int = 1280,
    height: int | None = None,
) -> Path:
    """
    Export all trajectory frames as an MP4 video.

    Requires imageio with ffmpeg backend.

    Parameters
    ----------
    controller : ApplicationController
    path       : output path (.mp4)
    fps        : frames per second
    width      : video width in pixels
    height     : video height (None = 16:9 from width)

    Returns
    -------
    Path to the saved file

    Raises
    ------
    ImportError if imageio not installed
    RuntimeError if no trajectory or plotter available
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video export.\n"
            "Install: pip install imageio imageio-ffmpeg"
        )

    path   = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = height or int(width * 9 / 16)

    model  = controller.model
    engine = getattr(controller, '_engine', None)

    if model.n_frames() == 0:
        raise RuntimeError("No trajectory loaded.")
    if engine is None or engine._plotter is None:
        raise RuntimeError("No active viewport for video export.")

    writer = imageio.get_writer(str(path), fps=fps)
    try:
        for frame_idx in range(model.n_frames()):
            controller.update_frame(frame_idx)
            img = engine._plotter.screenshot(
                return_img=True,
                window_size=[width, height],
            )
            writer.append_data(img)
    finally:
        writer.close()

    return path


def export_atoms_csv(
    atoms: list,
    positions: np.ndarray,
    path: str | Path,
) -> Path:
    """
    Export atom data and positions to a CSV file.

    Columns: id, element, resname, residue_id, chain_id, name, x, y, z

    Returns
    -------
    Path to the saved file
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(positions, dtype=float)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "element", "resname", "residue_id",
            "chain_id", "name", "x", "y", "z"
        ])
        for i, atom in enumerate(atoms):
            if i >= len(pos):
                break
            writer.writerow([
                atom.id,
                getattr(atom, "element",    None) or "",
                getattr(atom, "resname",    None) or "",
                getattr(atom, "residue_id", None) or "",
                getattr(atom, "chain_id",   None) or "",
                getattr(atom, "name",       None) or "",
                f"{pos[i, 0]:.4f}",
                f"{pos[i, 1]:.4f}",
                f"{pos[i, 2]:.4f}",
            ])

    return path


def export_atoms_pdb(
    atoms: list,
    positions: np.ndarray,
    path: str | Path,
) -> Path:
    """
    Export atoms and positions to a PDB file.

    Delegates to mutation_engine.write_pdb() — the authoritative
    PDB writer in PSVAP (Rule 4: one writer per format).

    Returns
    -------
    Path to the saved file
    """
    from PSVAP.modeling.mutation_engine import write_pdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pdb(atoms, positions, path)
    return path


def _export_generic_csv(data: Any, path: Path) -> None:
    """Write generic data to CSV."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            for row in data:
                writer.writerow([f"{v:.6g}" if isinstance(v, float) else v
                                 for v in row])

        elif isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            writer.writerow(headers)
            for row in data:
                writer.writerow([row.get(h, "") for h in headers])

        elif isinstance(data, dict):
            writer.writerow(["key", "value"])
            for k, v in data.items():
                writer.writerow([k, v])

        else:
            for item in (data if hasattr(data, '__iter__') else [data]):
                writer.writerow([item])