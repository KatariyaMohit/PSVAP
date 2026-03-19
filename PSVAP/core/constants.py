"""
Scientific and application-wide constants.

Rule 6: All constants live here — no magic numbers in the middle of functions.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Interaction Detection Cutoffs (Angstroms / degrees)
# ---------------------------------------------------------------------------
HBOND_DISTANCE_CUTOFF: float = 3.5       # donor–acceptor max distance (Å)
HBOND_ANGLE_CUTOFF: float = 120.0         # D-H…A minimum angle (degrees)
SALT_BRIDGE_CUTOFF: float = 4.0           # oppositely-charged group distance (Å)
HALOGEN_BOND_CUTOFF: float = 3.5          # C-X…O/N/S max distance (Å)
CLASH_VDW_OVERLAP: float = 0.4            # overlap beyond vdW sum to flag clash (Å)
PI_STACKING_CUTOFF: float = 5.5           # ring centroid distance (Å)

# ---------------------------------------------------------------------------
# Rendering Defaults
# ---------------------------------------------------------------------------
DEFAULT_ATOM_POINT_SIZE: float = 6.0
DEFAULT_ATOM_COLOR: str = "royalblue"
DEFAULT_BACKGROUND_COLOR: str = "white"
DEFAULT_BOND_RADIUS: float = 0.1
HBOND_DASH_COLOR: str = "cyan"
SALT_BRIDGE_DASH_COLOR: str = "magenta"
CLASH_COLOR: str = "red"

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
FRAME_SWITCH_TARGET_MS: int = 100         # max ms for switching trajectory frames
MIN_RENDER_FPS: int = 30                  # target rendering FPS

# ---------------------------------------------------------------------------
# Water Probe / SASA
# ---------------------------------------------------------------------------
WATER_PROBE_RADIUS: float = 1.4           # Å — standard water probe radius
