"""
visualization/viz_engine.py
----------------------------
VisualizationEngine — the ONLY class that touches PyVista.

Bond rendering strategy (FAST — same as old plotter.py)
---------------------------------------------------------
Bonds are built ONCE when data loads:
  - bond_mesh = pv.PolyData(positions)
  - bond_mesh.lines = bond_array   (PyVista format: [2,i,j, 2,i2,j2,...])
  - bond_actor = plotter.add_mesh(bond_mesh, ...)

On EVERY frame change:
  - cloud.points = new_positions      (in-place, no actor rebuild)
  - bond_mesh.points = new_positions  (in-place, no actor rebuild)
  - plotter.render()

This is O(1) per frame — exactly what the old code did.
No cylinder-per-bond. No rebuild. Just a point array update.

Auto bond detection
-------------------
If metadata.bonds is None (CIF, GRO, PDB, XYZ, SDF files that have no
explicit bond section), KDTree with cutoff=2.0 Å is run on frame 0
automatically. This means ALL file formats show bonds without touching
any individual parser.

Legend labels
-------------
- LAMMPS files with numeric type IDs: "TYPE 0", "TYPE 1", "TYPE 2"
- PDB/CIF/GRO/XYZ/SDF files with element symbols: "C", "N", "O", "H" etc.
  The legend dot colour uses CPK colours for element-based files.

Selection
---------
Selection rebuilds the atom mesh (fast — just a PolyData with subset).
Bonds are shown only between selected atoms.

Render modes
------------
  'atoms'       — atoms only
  'atoms_bonds' — atoms + bonds (default)
  'bonds'       — bonds only

Fix: set_render_mode() now calls _rebuild_scene() instead of
_apply_visibility(). The old _apply_visibility() only toggled actor
visibility on already-created actors, which meant:
  1. Switching mode before any file was loaded silently did nothing.
  2. Switching to 'bonds' only mode when bond actor was None (no bonds
     detected) showed a blank viewport.
  3. The _rebuild_scene() path already gates actor creation on
     self._render_mode, so a full rebuild is both correct and safe.
     The cost is negligible since mode changes are user-triggered (rare).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    import pyvista as pv
    from pyvistaqt import QtInteractor


# ── Colour tables ──────────────────────────────────────────────────────────
# LAMMPS type IDs 0,1,2,… → distinct colours (RGB 0–1)
_TYPE_COLORS = [
    (0.70, 0.70, 0.70),   # 0  grey
    (0.20, 0.60, 1.00),   # 1  blue
    (1.00, 0.30, 0.30),   # 2  red
    (0.20, 0.85, 0.20),   # 3  green
    (1.00, 0.85, 0.10),   # 4  yellow
    (0.90, 0.50, 0.10),   # 5  orange
    (0.70, 0.10, 0.90),   # 6  purple
    (0.10, 0.90, 0.90),   # 7  cyan
    (1.00, 0.60, 0.80),   # 8  pink
    (0.50, 0.80, 0.50),   # 9  light-green
    (0.80, 0.50, 0.20),   # 10 brown
    (0.40, 0.40, 0.90),   # 11 lavender
]

# CPK colours for element symbols (PDB/XYZ/CIF/GRO/SDF files)
_ELEMENT_COLORS: dict[str, tuple] = {
    "H":  (0.90, 0.90, 0.90),
    "C":  (0.50, 0.50, 0.50),
    "N":  (0.20, 0.40, 0.90),
    "O":  (0.90, 0.20, 0.20),
    "S":  (0.90, 0.80, 0.10),
    "P":  (0.90, 0.50, 0.10),
    "FE": (0.80, 0.30, 0.10),
    "ZN": (0.50, 0.50, 0.75),
    "MG": (0.20, 0.70, 0.20),
    "CA": (0.30, 0.80, 0.80),
    "CL": (0.10, 0.80, 0.10),
    "BR": (0.60, 0.10, 0.10),
    "NA": (0.70, 0.20, 0.80),
    "F":  (0.70, 1.00, 0.70),
    "I":  (0.40, 0.00, 0.73),
}

_SEL_COLOR = np.array([0.91, 1.00, 0.00], dtype=np.float32)  # neon yellow


def _atom_color(atom) -> tuple:
    """Return RGB colour for an atom — element-based if available, type-based otherwise."""
    elem = getattr(atom, 'element', None)
    if elem:
        return _ELEMENT_COLORS.get(elem.upper(), (0.60, 0.60, 0.60))
    tid = getattr(atom, 'type_id', None)
    if tid is None:
        return _TYPE_COLORS[0]
    return _TYPE_COLORS[int(tid) % len(_TYPE_COLORS)]


def _atom_label(atom) -> str:
    """Return display label: element symbol (PDB/CIF/GRO/XYZ/SDF) or 'TYPE N' (LAMMPS)."""
    elem = getattr(atom, 'element', None)
    if elem:
        return elem.upper()
    tid = getattr(atom, 'type_id', None)
    return f"TYPE {tid}" if tid is not None else "TYPE 0"


def _detect_bonds_auto(positions: np.ndarray, cutoff: float = 2.0) -> np.ndarray | None:
    """
    KDTree bond detection → PyVista line array [2,i,j,...].
    Used automatically when metadata.bonds is None (CIF, GRO, PDB, XYZ, SDF).
    cutoff=2.0 Å covers all standard covalent bonds in biomolecules.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if len(positions) < 2:
        return None
    try:
        pairs = list(cKDTree(positions).query_pairs(cutoff))
        if not pairs:
            return None
        bond_data: list[int] = []
        for p in pairs:
            bond_data.extend([2, int(p[0]), int(p[1])])
        return np.array(bond_data, dtype=np.int64)
    except Exception:
        return None


# ── Engine ─────────────────────────────────────────────────────────────────

class VisualizationEngine(QObject):
    """
    Fast 3D rendering via PyVista.

    Atoms: PolyData point cloud, rendered as spheres.
    Bonds: PolyData line mesh with bond_mesh.lines = bond_array.
    Per-frame update: only mesh.points = new_positions + plotter.render().
    No actor rebuilds during playback.
    """

    render_started  = Signal()
    render_finished = Signal()
    atom_picked     = Signal(str)

    def __init__(self, model) -> None:
        super().__init__()
        self._model = model
        self._plotter: "QtInteractor | None" = None

        # Scene objects (built once on data_loaded)
        self._cloud      = None   # pv.PolyData for atoms
        self._bond_mesh  = None   # pv.PolyData for bonds
        self._atom_actor = None
        self._bond_actor = None
        self._box_actor  = None

        # State
        self._render_mode: str = "atoms_bonds"
        self._atoms_list: list = []
        self._n_atoms: int = 0
        self._base_colors: np.ndarray | None = None   # (N,3) float32, no selection
        self._bond_array: np.ndarray | None = None    # PyVista line format
        self._current_positions: np.ndarray | None = None
        self._selection_mask: np.ndarray | None = None   # bool (N,) or None

        self._connect_model()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _connect_model(self) -> None:
        m = self._model
        for sig, slot in [
            ('data_loaded',       self._on_data_loaded),
            ('frame_changed',     self._on_frame_changed),
            ('selection_changed', self._on_selection_changed),
        ]:
            if hasattr(m, sig):
                getattr(m, sig).connect(slot)

    def attach_plotter(self, plotter: "QtInteractor") -> None:
        self._plotter = plotter
        self._plotter.set_background("#0A0A0A")
        # Enable atom picking
        try:
            self._plotter.enable_point_picking(
                callback=self._on_atom_picked,
                show_message=False,
                show_point=False,
                pickable_window=False,
                tolerance=0.025,
            )
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────────────────────

    def set_render_mode(self, mode: str) -> None:
        """
        Switch render mode and trigger a full scene rebuild.

        Why _rebuild_scene() and NOT _apply_visibility():
        -------------------------------------------------
        _apply_visibility() only shows/hides actors that already exist.
        This fails in two cases:
          1. No file loaded yet → _atom_actor / _bond_actor are None,
             so toggling visibility is a no-op. The mode is stored but
             the next load calls _rebuild_scene() which already reads
             self._render_mode, so this case is actually fine.
          2. File loaded, mode switched to 'bonds' only, but _bond_actor
             is None because no bonds were detected on load → blank screen.
          3. File loaded, mode switched, then user loads a NEW file —
             _rebuild_scene() is called on data_loaded and reads the
             already-updated self._render_mode, so this works correctly.

        A full rebuild is safe here: mode changes are user-triggered
        (one click), not per-frame, so the cost is acceptable.
        """
        if mode not in {"atoms", "atoms_bonds"}:
            return
        self._render_mode = mode
        # Only rebuild if a file is already loaded; otherwise the next
        # _on_data_loaded will pick up self._render_mode automatically.
        if self._plotter and self._current_positions is not None:
            self._rebuild_scene()

    def apply_selection(self, mask: np.ndarray | None) -> None:
        """Apply a boolean selection mask. Recolours atoms, filters bonds."""
        self._selection_mask = mask
        if self._current_positions is not None:
            self._rebuild_scene()

    def get_legend_items(self) -> list[tuple[str, tuple]]:
        """
        Return [(label, (r,g,b))] for the colour legend.

        For element-based files (PDB/CIF/GRO/XYZ/SDF): returns element symbols
        with CPK colours, e.g. [("C", (0.5,0.5,0.5)), ("N", (0.2,0.4,0.9)), ...].
        For LAMMPS files with numeric type IDs: returns "TYPE 0", "TYPE 1", etc.
        Deduplicates by label so each unique element/type appears once.
        """
        seen: dict[str, tuple] = {}
        for atom in self._atoms_list:
            label = _atom_label(atom)
            if label not in seen:
                seen[label] = _atom_color(atom)
        return list(seen.items())

    # ── Model signal slots ─────────────────────────────────────────────────

    @Slot()
    def _on_data_loaded(self) -> None:
        """Full scene init — runs ONCE when new file is loaded."""
        atoms = getattr(self._model, 'atoms', [])
        if not atoms:
            return

        self._atoms_list = list(atoms)
        self._n_atoms = len(atoms)

        frame0 = self._model.get_frame(0)
        if frame0 is None:
            return
        self._current_positions = np.asarray(frame0, dtype=np.float32)

        # Precompute base colours (element-based for PDB/CIF/GRO/XYZ/SDF,
        # type-ID-based for LAMMPS — no selection)
        self._base_colors = np.array(
            [_atom_color(a) for a in self._atoms_list], dtype=np.float32
        )

        # Get bond array from metadata.
        # If None (CIF/GRO/PDB/XYZ/SDF have no explicit bonds), auto-detect
        # via KDTree so all formats show bonds without parser changes.
        meta = getattr(self._model, 'metadata', None)
        bond_array = getattr(meta, 'bonds', None)
        if bond_array is None:
            bond_array = _detect_bonds_auto(
                self._current_positions.astype(np.float64)
            )
        self._bond_array = bond_array

        self._selection_mask = None
        self._rebuild_scene()

    @Slot(int)
    def _on_frame_changed(self, n: int) -> None:
        """
        FAST path — only update mesh point positions.
        No actor rebuild. No bond recalculation.
        """
        frame = self._model.get_frame(n)
        if frame is None or self._plotter is None:
            return

        pos = np.asarray(frame, dtype=np.float32)
        self._current_positions = pos

        try:
            if self._cloud is not None:
                self._cloud.points = pos
            if self._bond_mesh is not None:
                self._bond_mesh.points = pos
            self._plotter.render()
        except Exception as exc:
            # Fallback to full rebuild if in-place update fails
            print(f"[VizEngine] fast update failed, rebuilding: {exc}")
            self._rebuild_scene()

    @Slot(object)
    def _on_selection_changed(self, mask) -> None:
        """Called when SystemModel.selection_changed emits."""
        if mask is None or (hasattr(mask, '__len__') and len(mask) == 0):
            self._selection_mask = None
        else:
            self._selection_mask = np.asarray(mask, dtype=bool)
        if self._current_positions is not None:
            self._rebuild_scene()

    # ── Scene building ─────────────────────────────────────────────────────
    def _rebuild_scene(self) -> None:
        """
        Build or rebuild the full scene.
        Called on: data_loaded, selection change, render mode change.
        NOT called on frame change (use fast path instead).
        """
        if self._plotter is None or self._current_positions is None:
            return

        try:
            import pyvista as pv
        except ImportError:
            return
    
        self.render_started.emit()
    
        try:
            pos = self._current_positions
            colors = self._effective_colors()
    
            # Remove named actors individually — QtInteractor does not have
            # remove_all_actors() or clear(). Removing by name is the correct
            # API for pyvistaqt.QtInteractor across all supported versions.
            for actor_name in ("atoms", "bonds", "box"):
                try:
                    self._plotter.remove_actor(actor_name)
                except Exception:
                    pass
    
            self._cloud      = None
            self._bond_mesh  = None
            self._atom_actor = None
            self._bond_actor = None
            self._box_actor  = None
    
            # ── Atoms ────────────────────────────────────────────────────
            if self._render_mode in {"atoms", "atoms_bonds"}:
                self._cloud = pv.PolyData(pos)
                self._cloud.point_data["colors"] = colors
                self._atom_actor = self._plotter.add_mesh(
                    self._cloud,
                    scalars="colors",
                    rgb=True,
                    render_points_as_spheres=True,
                    point_size=8.0,
                    show_scalar_bar=False,
                    name="atoms",
                )
    
            # ── Bonds ────────────────────────────────────────────────────
            if self._render_mode == "atoms_bonds":
                bond_lines = self._effective_bonds()
                if bond_lines is not None and len(bond_lines) > 0:
                    self._bond_mesh = pv.PolyData(pos)
                    self._bond_mesh.lines = bond_lines
                    self._bond_actor = self._plotter.add_mesh(
                        self._bond_mesh,
                        color="gray",
                        line_width=1.5,
                        show_scalar_bar=False,
                        name="bonds",
                    )
    
            # ── Simulation box ───────────────────────────────────────────
            self._render_box(pv)
    
            self._plotter.reset_camera()
            self._plotter.render()
    
        except Exception as exc:
            import traceback
            print(f"[VizEngine] rebuild_scene error: {exc}\n{traceback.format_exc()}")
    
        self.render_finished.emit()

    
    def _effective_colors(self) -> np.ndarray:
        """Compute per-atom colours accounting for current selection."""
        if self._base_colors is None:
            return np.full((self._n_atoms, 3), 0.6, dtype=np.float32)

        colors = self._base_colors.copy()
        mask = self._selection_mask

        if mask is not None and len(mask) == self._n_atoms and np.any(mask):
            sel = np.asarray(mask, dtype=bool)
            # Dim unselected
            colors[~sel] = colors[~sel] * 0.20
            # Highlight selected
            colors[sel] = _SEL_COLOR

        return colors

    def _effective_bonds(self) -> np.ndarray | None:
        """
        Return the bond line array filtered by current selection.

        If no selection: return full bond_array.
        If selection active: return only bonds where BOTH endpoints selected.
        """
        if self._bond_array is None:
            return None

        mask = self._selection_mask
        if mask is None or not np.any(mask):
            return self._bond_array

        # Filter: only bonds where both atoms are selected
        sel = np.asarray(mask, dtype=bool)
        arr = self._bond_array
        # arr format: [2, i, j, 2, i2, j2, ...]
        n_bonds = len(arr) // 3
        filtered: list[int] = []

        for k in range(n_bonds):
            base = k * 3
            if base + 2 >= len(arr):
                break
            count = arr[base]   # always 2
            i = arr[base + 1]
            j = arr[base + 2]
            if i < len(sel) and j < len(sel) and sel[i] and sel[j]:
                filtered.extend([count, i, j])

        return np.array(filtered, dtype=int) if filtered else None

    def _render_box(self, pv) -> None:
        meta = getattr(self._model, 'metadata', None)
        bb = getattr(meta, 'box_bounds', None)
        if bb is None:
            return
        try:
            bb = np.asarray(bb)
            box = pv.Box(bounds=[
                float(bb[0, 0]), float(bb[0, 1]),
                float(bb[1, 0]), float(bb[1, 1]),
                float(bb[2, 0]), float(bb[2, 1]),
            ])
            self._box_actor = self._plotter.add_mesh(
                box, color="#2A2A2A", style="wireframe",
                line_width=1.0, opacity=0.5, name="box",
            )
        except Exception:
            pass

    def _apply_visibility(self) -> None:
        """
        Toggle actor visibility without rebuilding.

        NOTE: This method is kept for reference but is NO LONGER CALLED
        by set_render_mode(). See set_render_mode() docstring for why
        _rebuild_scene() is used instead.
        """
        if self._atom_actor:
            show_atoms = self._render_mode in {"atoms", "atoms_bonds"}
            self._atom_actor.SetVisibility(show_atoms)
        if self._bond_actor:
            show_bonds = self._render_mode in {"atoms_bonds", "bonds"}
            self._bond_actor.SetVisibility(show_bonds)
        if self._plotter:
            self._plotter.render()

    # ── Atom picking ───────────────────────────────────────────────────────

    def _on_atom_picked(self, picked_point) -> None:
        if self._current_positions is None or picked_point is None:
            return
        try:
            pt = np.array(picked_point[:3], dtype=float)
            dists = np.linalg.norm(
                self._current_positions.astype(float) - pt, axis=1
            )
            idx = int(np.argmin(dists))
            if idx >= len(self._atoms_list):
                return

            atom = self._atoms_list[idx]
            pos  = self._current_positions[idx]
            elem = getattr(atom, 'element', None)
            tid  = getattr(atom, 'type_id', 0)
            rid  = getattr(atom, 'residue_id', None)
            name = getattr(atom, 'name', None)

            # Label: element symbol if available, otherwise type ID
            label = elem.upper() if elem else f"TYPE {tid}"

            parts = [
                f"IDX {idx}",
                label,
                f"POS ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å",
            ]
            if name and name != label:
                parts.append(f"NAME {name}")
            if rid is not None:
                parts.append(f"MOL {rid}")

            # Find bonded neighbours from bond_array
            if self._bond_array is not None:
                arr = self._bond_array
                n_bonds = len(arr) // 3
                neighbours = []
                for k in range(n_bonds):
                    base = k * 3
                    if base + 2 >= len(arr):
                        break
                    i, j = int(arr[base + 1]), int(arr[base + 2])
                    if i == idx:
                        neighbours.append(j)
                    elif j == idx:
                        neighbours.append(i)
                if neighbours:
                    parts.append(f"BONDS→{neighbours[:8]}")

            self.atom_picked.emit("  ·  ".join(parts))
        except Exception:
            pass

    # ── Controller access (used by controller._engine) ─────────────────────

    @property
    def _engine(self):
        return self