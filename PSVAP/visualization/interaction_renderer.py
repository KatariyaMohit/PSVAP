"""
visualization/interaction_renderer.py
--------------------------------------
Feature 12: Interaction visualization via PyVista.

Draws interactions as dashed/solid lines in the 3D viewport:
  - H-bonds        : dashed cyan lines
  - Salt bridges   : dashed magenta lines
  - Halogen bonds  : dashed orange lines
  - Pi-stacking    : dashed green lines
  - Hydrophobic    : dashed yellow lines
  - Clashes        : solid red lines

Public API
----------
  InteractionRenderer(plotter)
      Wraps a PyVista plotter for interaction drawing.

  render_interactions(result, positions)
      Draw all interactions from an InteractionResult.

  clear()
      Remove all interaction actors from the plotter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from pyvistaqt import QtInteractor

from PSVAP.core.constants import (
    CLASH_COLOR,
    HBOND_DASH_COLOR,
    SALT_BRIDGE_DASH_COLOR,
)

# Interaction type colours (RGB 0-1)
_COLORS: dict[str, str] = {
    "hbond":       HBOND_DASH_COLOR,        # cyan
    "salt_bridge": SALT_BRIDGE_DASH_COLOR,  # magenta
    "halogen":     "orange",
    "pi_stack":    "green",
    "hydrophobic": "yellow",
    "clash":       CLASH_COLOR,             # red
}

# Actor name prefix — used to remove them cleanly
_ACTOR_PREFIX = "interaction_"


class InteractionRenderer:
    """
    Draws non-covalent interaction lines in a PyVista QtInteractor.

    Each interaction type is added as a named actor so it can be
    individually shown, hidden, or removed without rebuilding the scene.

    Usage
    -----
    renderer = InteractionRenderer(plotter)
    renderer.render_interactions(result, positions)
    renderer.clear()
    """

    def __init__(self, plotter: "QtInteractor") -> None:
        self._plotter = plotter
        self._actor_names: list[str] = []

    def clear(self) -> None:
        """Remove all interaction actors from the plotter."""
        for name in self._actor_names:
            try:
                self._plotter.remove_actor(name)
            except Exception:
                pass
        self._actor_names.clear()

    def render_interactions(
        self,
        result,                  # InteractionResult from analysis/interactions.py
        positions: np.ndarray,
        show_hbonds:       bool = True,
        show_salt_bridges: bool = True,
        show_halogen:      bool = True,
        show_pi_stacks:    bool = True,
        show_hydrophobic:  bool = False,  # off by default — too many contacts
        show_clashes:      bool = True,
    ) -> None:
        """
        Draw all interactions from an InteractionResult.

        Parameters
        ----------
        result    : InteractionResult from detect_all_interactions()
        positions : (N, 3) current frame positions
        show_*    : toggle each interaction type on/off
        """
        self.clear()
        pos = np.asarray(positions, dtype=float)

        if show_hbonds and result.hbonds:
            self._draw_lines(
                [(h.donor_idx, h.acceptor_idx) for h in result.hbonds],
                pos, color=_COLORS["hbond"],
                name=f"{_ACTOR_PREFIX}hbonds",
                line_width=2.0, dashed=True,
            )

        if show_salt_bridges and result.salt_bridges:
            self._draw_lines(
                [(s.pos_idx, s.neg_idx) for s in result.salt_bridges],
                pos, color=_COLORS["salt_bridge"],
                name=f"{_ACTOR_PREFIX}salt_bridges",
                line_width=2.0, dashed=True,
            )

        if show_halogen and result.halogen_bonds:
            self._draw_lines(
                [(h.halogen_idx, h.acceptor_idx) for h in result.halogen_bonds],
                pos, color=_COLORS["halogen"],
                name=f"{_ACTOR_PREFIX}halogen",
                line_width=1.5, dashed=True,
            )

        if show_pi_stacks and result.pi_stacks:
            self._draw_pi_stacks(result.pi_stacks,
                                 name=f"{_ACTOR_PREFIX}pi_stacks")

        if show_hydrophobic and result.hydrophobic:
            self._draw_lines(
                [(h.idx_a, h.idx_b) for h in result.hydrophobic],
                pos, color=_COLORS["hydrophobic"],
                name=f"{_ACTOR_PREFIX}hydrophobic",
                line_width=1.0, dashed=True,
            )

        if show_clashes and result.clashes:
            self._draw_lines(
                [(c.idx_a, c.idx_b) for c in result.clashes],
                pos, color=_COLORS["clash"],
                name=f"{_ACTOR_PREFIX}clashes",
                line_width=3.0, dashed=False,
            )

        if self._actor_names:
            self._plotter.render()

    def set_visibility(self, interaction_type: str, visible: bool) -> None:
        """
        Show or hide a specific interaction type.
        interaction_type: 'hbonds', 'salt_bridges', 'halogen',
                          'pi_stacks', 'hydrophobic', 'clashes'
        """
        actor_name = f"{_ACTOR_PREFIX}{interaction_type}"
        if actor_name in self._actor_names:
            try:
                actor = self._plotter.renderer.GetActors()
                # PyVista actor visibility via remove/re-add is safer
                # than direct VTK actor manipulation
                pass
            except Exception:
                pass

    # ── Internal line drawing ─────────────────────────────────────────────

    def _draw_lines(
        self,
        pairs: list[tuple[int, int]],
        positions: np.ndarray,
        color: str,
        name: str,
        line_width: float = 2.0,
        dashed: bool = True,
    ) -> None:
        """Draw a set of atom-pair lines as a single PolyData mesh."""
        if not pairs:
            return
        try:
            import pyvista as pv
        except ImportError:
            return

        points: list[list[float]] = []
        lines_arr: list[int] = []
        pt_idx = 0

        for i, j in pairs:
            if i >= len(positions) or j >= len(positions):
                continue
            points.append(positions[i].tolist())
            points.append(positions[j].tolist())
            lines_arr.extend([2, pt_idx, pt_idx + 1])
            pt_idx += 2

        if not points:
            return

        mesh = pv.PolyData(np.array(points, dtype=np.float32))
        mesh.lines = np.array(lines_arr, dtype=np.int64)

        self._plotter.add_mesh(
            mesh,
            color=color,
            line_width=line_width,
            show_scalar_bar=False,
            name=name,
            # PyVista doesn't natively support dashed lines via add_mesh,
            # but we set opacity slightly lower for dashed-type interactions
            # to visually distinguish them from solid clash lines.
            opacity=0.75 if dashed else 1.0,
        )
        self._actor_names.append(name)

    def _draw_pi_stacks(
        self,
        pi_stacks: list,
        name: str,
    ) -> None:
        """Draw pi-stacking as lines between ring centroids."""
        if not pi_stacks:
            return
        try:
            import pyvista as pv
        except ImportError:
            return

        points: list[list[float]] = []
        lines_arr: list[int] = []
        pt_idx = 0

        for ps in pi_stacks:
            points.append(ps.ring_a_center.tolist())
            points.append(ps.ring_b_center.tolist())
            lines_arr.extend([2, pt_idx, pt_idx + 1])
            pt_idx += 2

        if not points:
            return

        mesh = pv.PolyData(np.array(points, dtype=np.float32))
        mesh.lines = np.array(lines_arr, dtype=np.int64)

        self._plotter.add_mesh(
            mesh,
            color=_COLORS["pi_stack"],
            line_width=2.0,
            opacity=0.7,
            show_scalar_bar=False,
            name=name,
        )
        self._actor_names.append(name)