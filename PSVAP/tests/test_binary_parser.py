from __future__ import annotations

import numpy as np
import pytest

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import detect_parser
from PSVAP.io.binary_parser import (
    BinaryFormatError,
    BinaryParser,
    DumpCustomTrajectory,
    FormatMismatchError,
    MemmapTrajectory,
    RawBinaryConfig,
    format_mismatch_message,
)


def _mock_topology(monkeypatch, n_atoms: int = 2):
    atoms = [
        Atom(
            id=i,
            element="C" if i else "N",
            name=f"A{i + 1}",
            x=float(i),
            y=0.0,
            z=0.0,
        )
        for i in range(n_atoms)
    ]
    metadata = SystemMetadata(
        bonds=np.array([2, 0, 1], dtype=np.int64) if n_atoms >= 2 else None
    )

    def _fake_parse_topology(self, topo_path):
        return atoms, [np.zeros((n_atoms, 3), dtype=np.float64)], metadata

    monkeypatch.setattr(BinaryParser, "_parse_topology", _fake_parse_topology)
    return atoms, metadata


def _write_dumpcustom_bin(path, frames: np.ndarray) -> None:
    magic = b"DUMPCUSTOM"
    field_names = b"id type x y z"
    natoms = int(frames.shape[1])
    size_one = 5
    box = (-20.0, 20.0, -20.0, 20.0, -20.0, 20.0)

    with path.open("wb") as fh:
        for timestep, frame in enumerate(frames):
            rows: list[float] = []
            for atom_idx, pos in enumerate(frame, start=1):
                rows.extend([
                    float(atom_idx),
                    1.0,
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                ])

            fh.write(np.int64(-len(magic)).tobytes())
            fh.write(magic)
            fh.write(np.int32(1).tobytes())
            fh.write(np.int32(2).tobytes())
            fh.write(np.int64(timestep).tobytes())
            fh.write(np.int64(natoms).tobytes())
            fh.write(np.int32(0).tobytes())
            fh.write(np.zeros(6, dtype=np.int32).tobytes())
            fh.write(np.asarray(box, dtype="<f8").tobytes())
            fh.write(np.int32(size_one).tobytes())
            fh.write(np.int32(0).tobytes())
            fh.write(np.int32(len(field_names)).tobytes())
            fh.write(field_names)
            fh.write(np.int32(1).tobytes())
            fh.write(np.int32(natoms * size_one).tobytes())
            fh.write(np.asarray(rows, dtype="<f8").tobytes())


@pytest.mark.parametrize(
    ("header", "format_name"),
    [
        (b"CDF\x01\x00\x00\x00\x00", "Amber NetCDF"),
        (b"\x54\x00\x00\x00CORD", "CHARMM/NAMD DCD"),
        (b"\x00\x00\x07\xCB\x00\x00\x00\x01", "GROMACS XTC"),
        (b"\x00\x00\x07\xC9\x00\x00\x00\x01", "GROMACS TRR"),
    ],
)
def test_binary_parser_rejects_mislabeled_known_formats(tmp_path, header, format_name):
    bin_path = tmp_path / "trajectory.bin"
    bin_path.write_bytes(header + b"\x00" * 32)

    with pytest.raises(FormatMismatchError) as excinfo:
        BinaryParser().parse(bin_path)

    assert str(excinfo.value) == format_mismatch_message(format_name)


def test_detect_parser_supports_bin_extension(tmp_path):
    parser = detect_parser(tmp_path / "sample.bin")
    assert isinstance(parser, BinaryParser)


def test_binary_parser_requires_topology_for_raw_binary(tmp_path):
    bin_path = tmp_path / "trajectory.bin"
    bin_path.write_bytes(b"RAWBINF0" + (np.zeros((1, 1, 3), dtype="<f4")).tobytes())

    parser = BinaryParser(
        binary_config=RawBinaryConfig(
            atoms_per_frame=1,
            data_precision="float32",
            endianness="little",
            byte_offset=8,
        )
    )

    with pytest.raises(BinaryFormatError, match="topology file"):
        parser.parse(bin_path)


def test_binary_parser_memmaps_raw_binary_with_pdb_topology(tmp_path, monkeypatch):
    topo_path = tmp_path / "topology.pdb"
    topo_path.write_text("MOCK", encoding="utf-8")
    _mock_topology(monkeypatch, n_atoms=2)

    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.55, 0.0, 0.0]],
        ],
        dtype="<f4",
    )
    bin_path = tmp_path / "trajectory.bin"
    bin_path.write_bytes(b"RAWBINF0" + frames.tobytes())

    parser = BinaryParser(
        binary_config=RawBinaryConfig(
            atoms_per_frame=2,
            data_precision="float32",
            endianness="little",
            byte_offset=8,
        ),
        topology_path=topo_path,
    )
    atoms, trajectory, metadata = parser.parse(bin_path)

    assert len(atoms) == 2
    assert atoms[0].name == "A1"
    assert isinstance(trajectory, MemmapTrajectory)
    assert trajectory.shape == (2, 2, 3)
    np.testing.assert_allclose(trajectory[1], frames[1].astype(np.float64))
    assert metadata.source_path == bin_path
    assert metadata.timesteps == [0, 1]


def test_binary_parser_rejects_topology_atom_count_mismatch(tmp_path, monkeypatch):
    topo_path = tmp_path / "topology.pdb"
    topo_path.write_text("MOCK", encoding="utf-8")
    _mock_topology(monkeypatch, n_atoms=2)

    frames = np.zeros((1, 2, 3), dtype="<f4")
    bin_path = tmp_path / "trajectory.bin"
    bin_path.write_bytes(b"RAWBINF0" + frames.tobytes())

    parser = BinaryParser(
        binary_config=RawBinaryConfig(
            atoms_per_frame=3,
            data_precision="float32",
            endianness="little",
            byte_offset=8,
        ),
        topology_path=topo_path,
    )

    with pytest.raises(BinaryFormatError, match="topology atom count"):
        parser.parse(bin_path)


def test_dumpcustom_bin_loads_without_topology(tmp_path):
    frames = np.array(
        [
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]],
            [[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]],
        ],
        dtype=np.float64,
    )
    bin_path = tmp_path / "trajectory.bin"
    _write_dumpcustom_bin(bin_path, frames)

    atoms, trajectory, metadata = BinaryParser().parse(bin_path)

    assert len(atoms) == 2
    assert atoms[0].name == "Atom 1"
    assert atoms[0].element == "X"
    assert isinstance(trajectory, DumpCustomTrajectory)
    assert trajectory.shape == (2, 2, 3)
    np.testing.assert_allclose(trajectory[0], frames[0])
    np.testing.assert_allclose(trajectory[1], frames[1])
    assert metadata.timesteps == [0, 1]
    assert isinstance(metadata.bonds, np.ndarray)
    assert metadata.bonds.size == 0
