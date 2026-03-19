from pathlib import Path

from PSVAP.io.lammps_parser import LammpsParser


def test_lammps_parser_reads_two_lammpstrj_frames():
    path = Path(__file__).parent / "fixtures" / "sample.lammpstrj"
    atoms, traj, md = LammpsParser().parse(path)
    assert len(atoms) == 2
    assert len(traj) == 2
    assert md.timesteps == [0, 1]


def test_lammps_parser_reads_data_file_single_frame():
    path = Path(__file__).parent / "fixtures" / "sample.data"
    atoms, traj, md = LammpsParser().parse(path)
    assert len(atoms) == 3
    assert len(traj) == 1
    assert md.timesteps == [0]

