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

# Measurement highlight colors
_MEASUREMENT_COLORS = {
    'distance': np.array([0.00, 1.00, 0.50], dtype=np.float32),    # cyan/neon green
    'angle': np.array([1.00, 0.50, 0.00], dtype=np.float32),       # orange
    'torsion': np.array([1.00, 0.00, 1.00], dtype=np.float32),     # magenta/pink
}

# Plugin highlight colors (for plugin-requested highlights)
_PLUGIN_COLORS = {
    'red': np.array([1.00, 0.20, 0.20], dtype=np.float32),
    'blue': np.array([0.20, 0.50, 1.00], dtype=np.float32),
    'green': np.array([0.20, 0.90, 0.20], dtype=np.float32),
    'yellow': np.array([1.00, 0.90, 0.20], dtype=np.float32),
    'cyan': np.array([0.20, 1.00, 1.00], dtype=np.float32),
    'magenta': np.array([1.00, 0.20, 1.00], dtype=np.float32),
    'orange': np.array([1.00, 0.65, 0.20], dtype=np.float32),
    'purple': np.array([0.70, 0.20, 0.90], dtype=np.float32),
    'pink': np.array([1.00, 0.60, 0.80], dtype=np.float32),
    'white': np.array([1.00, 1.00, 1.00], dtype=np.float32),
}


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
        
        # Measurement highlighting
        self._measurement_atoms: set[int] = set()   # atom indices in current measurement
        self._measurement_type: str | None = None    # 'distance', 'angle', 'torsion'
        
       # Plugin highlighting (Multi-color accumulation)
        self._plugin_active_mask: np.ndarray | None = None  # bool (N,)
        self._plugin_colors_array: np.ndarray | None = None # float32 (N, 3)
        
        # Sequence-based coloring
        self._sequence_color_mode: str | None = None   # 'residue_index', 'residue_type', or None
        
        # Interaction visualization
        self._interaction_actors: dict = {}   # store actors for different interaction types

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

    def apply_plugin_colors(self, mask: np.ndarray | None, color: str = 'yellow') -> None:
        """
        Apply plugin-requested highlight colors using an accumulation buffer.
        This allows multiple calls (e.g., Red for helices, Blue for sheets).
        """
        if self._current_positions is None: return
        n = len(self._current_positions)
        
        # Initialize the accumulation buffer if it doesn't exist
        if self._plugin_active_mask is None or len(self._plugin_active_mask) != n:
            self._plugin_active_mask = np.zeros(n, dtype=bool)
            self._plugin_colors_array = np.zeros((n, 3), dtype=np.float32)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            rgb = _PLUGIN_COLORS.get(color.lower(), _PLUGIN_COLORS['yellow'])
            
            # THE FIX: Add these atoms to the active set and store their specific color
            self._plugin_active_mask |= mask
            self._plugin_colors_array[mask] = rgb

        # FAST PATH: Update the graphics memory
        if self._plotter is not None:
            new_colors = self._effective_colors()
            if self._cloud is not None:
                self._cloud.point_data["colors"] = new_colors
            if self._bond_mesh is not None:
                self._bond_mesh.point_data["colors"] = new_colors
            self._plotter.render()

    def highlight_atom(self, index: int, color: str = 'yellow') -> None:
        """
        Highlight a single atom by its index.
        Used by the Particle Surface and Volume Analysis (PSVAP) patch visualization.
        """
        if self._current_positions is None:
            return
            
        n = len(self._current_positions)
        if 0 <= index < n:
            # Create a mask where only this specific atom is True
            mask = np.zeros(n, dtype=bool)
            mask[index] = True
            
            # Use our existing thread-safe, multi-color highlight system
            self.apply_plugin_colors(mask, color)

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

        self._plugin_active_mask = None
        self._plugin_colors_array = None

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
                    self._bond_mesh.point_data["colors"] = colors  # Assign colors to bonds!
                    self._bond_actor = self._plotter.add_mesh(
                        self._bond_mesh,
                        scalars="colors",  # Read the colors
                        rgb=True,          # Apply as RGB
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
        """Compute per-atom colours accounting for selection, sequence coloring, measurement, and plugin highlights."""
        if self._base_colors is None:
            return np.full((self._n_atoms, 3), 0.6, dtype=np.float32)

        colors = self._base_colors.copy()
        
        # Apply sequence-based coloring if enabled
        if self._sequence_color_mode == 'residue_index':
            colors = self._get_residue_index_colors()
        elif self._sequence_color_mode == 'residue_type':
            colors = self._get_residue_type_colors()
        
        mask = self._selection_mask

        # Apply regular selection highlighting
        if mask is not None and len(mask) == self._n_atoms and np.any(mask):
            sel = np.asarray(mask, dtype=bool)
            # Dim unselected
            colors[~sel] = colors[~sel] * 0.20
            # Highlight selected
            colors[sel] = _SEL_COLOR
        
        # Apply plugin highlight colors (takes precedence over selection)
        if self._plugin_active_mask is not None and np.any(self._plugin_active_mask):
            # Dim all atoms that are NOT part of any plugin highlight
            colors[~self._plugin_active_mask] *= 0.35
            # Apply the specific accumulated colors (Reds, Blues, etc.) from our buffer
            colors[self._plugin_active_mask] = self._plugin_colors_array[self._plugin_active_mask]
        
        # Apply measurement highlighting (takes precedence over all)
        if self._measurement_atoms and self._measurement_type in _MEASUREMENT_COLORS:
            meas_color = _MEASUREMENT_COLORS[self._measurement_type]
            for idx in self._measurement_atoms:
                if 0 <= idx < self._n_atoms:
                    colors[idx] = meas_color

        return colors

    def _get_residue_index_colors(self) -> np.ndarray:
        """Color atoms by their residue position with a blue→red gradient."""
        colors = np.zeros((self._n_atoms, 3), dtype=np.float32)
        
        if not self._atoms_list:
            return colors
        
        # Get unique residue IDs and create mapping
        residue_ids = []
        for atom in self._atoms_list:
            rid = getattr(atom, 'residue_id', None)
            if rid is not None and rid not in residue_ids:
                residue_ids.append(rid)
        
        if not residue_ids:
            return self._base_colors.copy() if self._base_colors is not None else colors
        
        n_residues = len(residue_ids)
        
        # Create gradient colors: blue (0,0,1) → red (1,0,0)
        for i, atom in enumerate(self._atoms_list):
            rid = getattr(atom, 'residue_id', None)
            if rid in residue_ids:
                idx = residue_ids.index(rid)
                # Gradient: blue → cyan → green → yellow → red
                t = idx / max(1, n_residues - 1)
                if t < 0.25:
                    # Blue → Cyan
                    r = 0.0
                    g = (t / 0.25)
                    b = 1.0
                elif t < 0.5:
                    # Cyan → Green
                    r = 0.0
                    g = 1.0
                    b = (0.5 - t) / 0.25
                elif t < 0.75:
                    # Green → Yellow
                    r = (t - 0.5) / 0.25
                    g = 1.0
                    b = 0.0
                else:
                    # Yellow → Red
                    r = 1.0
                    g = (1.0 - t) / 0.25
                    b = 0.0
                colors[i] = [r, g, b]
            else:
                colors[i] = [0.6, 0.6, 0.6]
        
        return colors

    def _get_residue_type_colors(self) -> np.ndarray:
        """Color atoms by amino acid type: hydrophobic/polar/charged."""
        colors = np.zeros((self._n_atoms, 3), dtype=np.float32)
        
        # Amino acid classification
        hydrophobic = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        polar = {'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'}
        positive = {'LYS', 'ARG', 'HIS'}
        negative = {'ASP', 'GLU'}
        
        # Colors: hydrophobic=yellow, polar=green, positive=blue, negative=red
        colors_map = {
            'hydrophobic': (1.0, 1.0, 0.0),   # yellow
            'polar': (0.0, 1.0, 0.0),         # green
            'positive': (0.0, 0.5, 1.0),      # blue
            'negative': (1.0, 0.0, 0.0),      # red
            'other': (0.7, 0.7, 0.7),         # grey
        }
        
        for i, atom in enumerate(self._atoms_list):
            resname = getattr(atom, 'resname', None)
            if resname:
                resname_up = resname.strip().upper()
                if resname_up in hydrophobic:
                    colors[i] = colors_map['hydrophobic']
                elif resname_up in polar:
                    colors[i] = colors_map['polar']
                elif resname_up in positive:
                    colors[i] = colors_map['positive']
                elif resname_up in negative:
                    colors[i] = colors_map['negative']
                else:
                    colors[i] = colors_map['other']
            else:
                colors[i] = colors_map['other']
        
        return colors


    def _effective_bonds(self) -> np.ndarray | None:
        """
        Return the bond line array filtered by current selection.

        If no selection and no measurement: return full bond_array.
        If selection active: return only bonds where BOTH endpoints selected.
        If measurement active: also include bonds connecting measurement atoms.
        """
        if self._bond_array is None:
            return None

        mask = self._selection_mask
        has_selection = mask is not None and np.any(mask)
        has_measurement = bool(self._measurement_atoms)

        # If neither selection nor measurement, return all bonds
        if not has_selection and not has_measurement:
            return self._bond_array

        arr = self._bond_array
        n_bonds = len(arr) // 3
        filtered: list[int] = []

        for k in range(n_bonds):
            base = k * 3
            if base + 2 >= len(arr):
                break
            count = arr[base]   # always 2
            i = arr[base + 1]
            j = arr[base + 2]
            
            # Include bond if both atoms are selected OR if either atom is in measurement
            include = False
            if has_measurement and (i in self._measurement_atoms or j in self._measurement_atoms):
                include = True
            elif has_selection and i < len(mask) and j < len(mask) and mask[i] and mask[j]:
                include = True
            
            if include:
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

    def get_atom_info_at_position(self, picked_point) -> dict | None:
        """
        Get atom information at a given 3D position.
        Returns a dict with keys: 'index', 'atom', 'position', 'type_label'
        or None if no atom found at position.
        """
        if self._current_positions is None or picked_point is None:
            return None
        try:
            pt = np.array(picked_point[:3], dtype=float)
            dists = np.linalg.norm(
                self._current_positions.astype(float) - pt, axis=1
            )
            idx = int(np.argmin(dists))
            if idx >= len(self._atoms_list):
                return None

            atom = self._atoms_list[idx]
            pos = self._current_positions[idx]
            elem = getattr(atom, 'element', None)
            tid = getattr(atom, 'type_id', 0)
            
            # Type label: element symbol if available, otherwise type ID
            type_label = elem.upper() if elem else f"TYPE {tid}"
            
            return {
                'index': idx,
                'atom': atom,
                'position': pos,
                'type_label': type_label
            }
        except Exception:
            return None

    def highlight_measurement(self, atom_indices: list[int], measurement_type: str) -> None:
        """
        Highlight atoms involved in a measurement.
        
        Args:
            atom_indices: List of atom indices involved in the measurement
            measurement_type: 'distance', 'angle', or 'torsion'
        """
        if not atom_indices:
            self.clear_measurement_highlight()
            return
        
        self._measurement_atoms = set(atom_indices)
        self._measurement_type = measurement_type
        
        if self._current_positions is not None:
            self._rebuild_scene()

    def clear_measurement_highlight(self) -> None:
        """Clear the current measurement highlight."""
        self._measurement_atoms.clear()
        self._measurement_type = None
        
        if self._current_positions is not None:
            self._rebuild_scene()

    def set_sequence_coloring(self, mode: str | None) -> None:
        """
        Enable/disable sequence-based coloring of atoms.
        
        Args:
            mode: 'residue_index' (gradient blue→red by position),
                  'residue_type' (hydrophobic/polar/charged),
                  or None to disable
        """
        self._sequence_color_mode = mode
        if self._current_positions is not None:
            self._rebuild_scene()

    def clear_sequence_coloring(self) -> None:
        """Disable sequence-based coloring and return to default colors."""
        self._sequence_color_mode = None
        if self._current_positions is not None:
            self._rebuild_scene()

    def visualize_interactions(self, interactions_dict: dict, positions: np.ndarray) -> None:
        """
        Visualize interactions as lines/highlights in 3D.
        
        Args:
            interactions_dict: {
                'hbonds': list[HBond],
                'salt_bridges': list[SaltBridge],
                'hydrophobic': list[HydrophobicContact],
                'clashes': list[Clash],
                ...
            }
            positions: (N, 3) array of atom positions
        """
        if self._plotter is None:
            return
        
        try:
            import pyvista as pv
            
            # Remove old interaction actors
            for actor in self._interaction_actors.values():
                try:
                    self._plotter.remove_actor(actor)
                except:
                    pass
            self._interaction_actors.clear()
            
            # H-bonds: cyan lines (slightly thicker)
            if interactions_dict.get('hbonds'):
                hbond_lines = []
                for h in interactions_dict['hbonds']:
                    p1 = positions[h.donor_idx]
                    p2 = positions[h.acceptor_idx]
                    hbond_lines.append([p1, p2])
                
                if hbond_lines:
                    hbond_mesh = pv.PolyData(np.concatenate(hbond_lines).reshape(-1, 3))
                    actor = self._plotter.add_mesh(hbond_mesh, color='cyan', line_width=2,
                                                   render=False)
                    self._interaction_actors['hbonds'] = actor
            
            # Salt bridges: yellow solid lines
            if interactions_dict.get('salt_bridges'):
                salt_lines = []
                for s in interactions_dict['salt_bridges']:
                    pos_idx = s.positive_idx if hasattr(s, 'positive_idx') else (s.pos_idx if hasattr(s, 'pos_idx') else 0)
                    neg_idx = s.negative_idx if hasattr(s, 'negative_idx') else (s.neg_idx if hasattr(s, 'neg_idx') else 0)
                    p1 = positions[pos_idx]
                    p2 = positions[neg_idx]
                    salt_lines.append([p1, p2])
                
                if salt_lines:
                    salt_mesh = pv.PolyData(np.concatenate(salt_lines).reshape(-1, 3))
                    actor = self._plotter.add_mesh(salt_mesh, color='yellow', line_width=2,
                                                   render=False)
                    self._interaction_actors['salt'] = actor
            
            # Hydrophobic: faint gray lines
            if interactions_dict.get('hydrophobic'):
                hydro_lines = []
                for h in interactions_dict['hydrophobic']:
                    idx_a = h.atom1_idx if hasattr(h, 'atom1_idx') else (h.idx_a if hasattr(h, 'idx_a') else 0)
                    idx_b = h.atom2_idx if hasattr(h, 'atom2_idx') else (h.idx_b if hasattr(h, 'idx_b') else 0)
                    p1 = positions[idx_a]
                    p2 = positions[idx_b]
                    hydro_lines.append([p1, p2])
                
                if hydro_lines:
                    hydro_mesh = pv.PolyData(np.concatenate(hydro_lines).reshape(-1, 3))
                    actor = self._plotter.add_mesh(hydro_mesh, color='gray', line_width=1,
                                                   opacity=0.3, render=False)
                    self._interaction_actors['hydrophobic'] = actor
            
            # Clashes: red highlights on problematic atoms
            if interactions_dict.get('clashes'):
                clash_indices = []
                for c in interactions_dict['clashes']:
                    idx_a = c.atom1_idx if hasattr(c, 'atom1_idx') else (c.idx_a if hasattr(c, 'idx_a') else 0)
                    idx_b = c.atom2_idx if hasattr(c, 'atom2_idx') else (c.idx_b if hasattr(c, 'idx_b') else 0)
                    clash_indices.extend([idx_a, idx_b])
                
                if clash_indices:
                    clash_points = positions[np.unique(clash_indices)]
                    clash_cloud = pv.PolyData(clash_points)
                    actor = self._plotter.add_mesh(clash_cloud, color='red', point_size=12,
                                                   render=False)
                    self._interaction_actors['clashes'] = actor
            
            self._plotter.render()
            
        except Exception as e:
            print(f"Interaction visualization error: {e}")

    def _on_atom_picked(self, picked_point) -> None:
        info = self.get_atom_info_at_position(picked_point)
        if info is None:
            return
        try:
            idx = info['index']
            atom = info['atom']
            pos = info['position']
            type_label = info['type_label']
            
            rid = getattr(atom, 'residue_id', None)
            name = getattr(atom, 'name', None)

            parts = [
                f"IDX {idx}",
                type_label,
                f"POS ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å",
            ]
            if name and name != type_label:
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


    def clear_plugin_highlights(self) -> None:
        """Reset all plugin-applied colors to default."""
        self._plugin_active_mask = None
        self._plugin_colors_array = None
        if self._plotter and self._current_positions is not None:
            self._rebuild_scene()
    # ── Controller access (used by controller._engine) ─────────────────────

    @property
    def _engine(self):
        return self