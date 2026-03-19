"""
core/system_model.py — PATCH NOTES
------------------------------------
Add `bonds` field to SystemMetadata dataclass.
Add `_selection_mask` attribute to SystemModel.
Add `selection_changed` signal to SystemModel.

HOW TO APPLY
------------
Find the SystemMetadata dataclass in your existing system_model.py and
add the `bonds` field as shown below.

Also ensure SystemModel has:
  - selection_changed = Signal()     (Qt signal)
  - _selection_mask attribute        (np.ndarray | None)
  - apply_selection(mask) method
  - clear_selection() method

=== SYSTEMMETADATA — find this class and add `bonds=None` field ===

@dataclass
class SystemMetadata:
    source_path:  Path | None         = None
    box_bounds:   np.ndarray | None   = None
    timesteps:    list[int]           = field(default_factory=list)
    bonds:        list | None         = None   # ← ADD THIS LINE

=== SYSTEMMODEL — ensure these exist ===

class SystemModel(QObject):
    data_loaded       = Signal()
    frame_changed     = Signal(int)
    selection_changed = Signal()       # ← ADD IF MISSING

    def __init__(self):
        super().__init__()
        self.atoms: list       = []
        self.trajectory: list  = []
        self.metadata          = SystemMetadata()
        self._current_frame: int = 0
        self._selection_mask: np.ndarray | None = None   # ← ADD IF MISSING

    def get_frame(self, n: int) -> np.ndarray | None:
        if not self.trajectory or n >= len(self.trajectory):
            return None
        return self.trajectory[n]

    def n_frames(self) -> int:
        return len(self.trajectory)

    def apply_selection(self, mask) -> None:
        self._selection_mask = mask
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        self._selection_mask = None
        self.selection_changed.emit()

    def set_data(self, atoms, trajectory, metadata):
        self.atoms = atoms
        self.trajectory = trajectory
        self.metadata = metadata
        self._current_frame = 0
        self._selection_mask = None
        self.data_loaded.emit()
"""

# This file is documentation — the actual changes must be made to
# PSVAP/core/system_model.py manually or by reading it.
# The key attributes that must exist on SystemModel:
#   .atoms              list of Atom
#   .trajectory         list of np.ndarray frames
#   .metadata           SystemMetadata (with .bonds, .box_bounds, .timesteps)
#   ._selection_mask    np.ndarray | None
#   .get_frame(n)       → np.ndarray | None
#   .n_frames()         → int
#   selection_changed   Qt Signal
#   data_loaded         Qt Signal
#   frame_changed       Qt Signal(int)