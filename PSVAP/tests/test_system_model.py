import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata, SystemModel


def test_system_model_set_data_and_get_frame():
    m = SystemModel()
    atoms = [Atom(id=1, type_id=1), Atom(id=2, type_id=1)]
    traj = [np.zeros((2, 3), dtype=np.float64), np.ones((2, 3), dtype=np.float64)]
    md = SystemMetadata()
    m.set_data(atoms=atoms, trajectory=traj, metadata=md)
    assert m.n_frames() == 2
    f0 = m.get_frame(0)
    assert f0.shape == (2, 3)

