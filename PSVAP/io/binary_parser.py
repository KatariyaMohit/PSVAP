"""
io/binary_parser.py
-------------------
Strict validator and raw-binary fallback loader for `.bin` trajectories.

Workflow
--------
1. Read the first 8 bytes and reject files that match known binary
   simulation formats but were mislabeled as `.bin`.
2. If no known signature is found, treat the file as raw coordinate data.
3. Use a separate topology file (.pdb or .gro) for atom identities/bonds.
4. Back the trajectory with numpy.memmap so frame access stays lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Sequence

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class FormatMismatchError(ValueError):
    """Raised when a `.bin` file is really a different known format."""


class BinaryFormatError(ValueError):
    """Raised when a raw binary file cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class BinarySignatureMatch:
    format_name: str
    header: bytes


@dataclass(frozen=True, slots=True)
class DumpCustomHeader:
    timestep: int
    natoms: int
    box_bounds: np.ndarray
    size_one: int
    field_names: tuple[str, ...]
    header_size: int
    data_count: int
    frame_size: int
    id_column: int | None
    type_column: int | None
    position_columns: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RawBinaryConfig:
    atoms_per_frame: int
    data_precision: str
    endianness: str
    byte_offset: int = 0

    def validate(self) -> None:
        if self.atoms_per_frame <= 0:
            raise BinaryFormatError("Atoms per frame must be a positive integer.")
        if self.data_precision not in {"float32", "float64"}:
            raise BinaryFormatError(
                "Data precision must be either 'float32' or 'float64'."
            )
        if self.endianness not in {"little", "big"}:
            raise BinaryFormatError(
                "Endianness must be either 'little' or 'big'."
            )
        if self.byte_offset < 0:
            raise BinaryFormatError("Byte offset cannot be negative.")

        dtype = self.numpy_dtype()
        if self.byte_offset % dtype.itemsize != 0:
            raise BinaryFormatError(
                f"Byte offset {self.byte_offset} is not aligned to "
                f"{dtype.itemsize}-byte {self.data_precision} values."
            )

    def numpy_dtype(self) -> np.dtype:
        base = np.float32 if self.data_precision == "float32" else np.float64
        byte_order = "<" if self.endianness == "little" else ">"
        return np.dtype(base).newbyteorder(byte_order)

    def bytes_per_frame(self) -> int:
        return self.atoms_per_frame * 3 * self.numpy_dtype().itemsize


class MemmapTrajectory(Sequence[np.ndarray]):
    """
    Sequence-like wrapper over a `(n_frames, n_atoms, 3)` memmap.

    Frames are converted to native-endian float64 arrays on access so the rest
    of the application can treat them like the existing parser outputs.
    """

    def __init__(self, memmap_array: np.memmap) -> None:
        self._memmap = memmap_array

    def __len__(self) -> int:
        return int(self._memmap.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if not isinstance(index, int):
            raise TypeError("Trajectory indices must be integers or slices.")

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        frame = self._memmap[index]
        return np.asarray(frame, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self._memmap.shape)


class DumpCustomTrajectory(Sequence[np.ndarray]):
    """Sequence wrapper over a strided DUMPCUSTOM frame view."""

    def __init__(
        self,
        frame_view: np.ndarray,
        position_columns: tuple[int, int, int],
    ) -> None:
        self._frame_view = frame_view
        self._position_columns = position_columns

    def __len__(self) -> int:
        return int(self._frame_view.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if not isinstance(index, int):
            raise TypeError("Trajectory indices must be integers or slices.")

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        frame = self._frame_view[index][:, self._position_columns]
        return np.asarray(frame, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self._frame_view.shape[0]),
            int(self._frame_view.shape[1]),
            3,
        )


def inspect_binary_signature(path: Path) -> BinarySignatureMatch | None:
    """
    Read the first 8 bytes and detect mislabeled known binary formats.

    Supported signatures:
      - AMBER NetCDF: `CDF\\x01`
      - CHARMM/NAMD DCD: `CORD` within the first 8 bytes
      - GROMACS XTC: big-endian magic `0x000007CB`
      - GROMACS TRR: big-endian magic `0x000007C9`
    """
    header = _read_header(path)

    if header.startswith(b"CDF\x01"):
        return BinarySignatureMatch("Amber NetCDF", header)
    if b"CORD" in header:
        return BinarySignatureMatch("CHARMM/NAMD DCD", header)
    if header.startswith(b"\x00\x00\x07\xCB"):
        return BinarySignatureMatch("GROMACS XTC", header)
    if header.startswith(b"\x00\x00\x07\xC9"):
        return BinarySignatureMatch("GROMACS TRR", header)
    return None


def inspect_dumpcustom_header(path: Path) -> DumpCustomHeader | None:
    """
    Detect LAMMPS DUMPCUSTOM binary dumps and parse the first-frame header.

    The professor's file follows this layout: a short binary header per frame
    followed by `natoms * size_one` float64 values. We only need the first
    frame header to derive atom count, box, field names, and frame size.
    """
    with path.open("rb") as fh:
        raw = fh.read(512)
    if len(raw) < 170:
        return None

    magic_len = struct.unpack_from("<q", raw, 0)[0]
    if magic_len >= 0:
        return None

    magic = raw[8:8 - magic_len]
    if magic != b"DUMPCUSTOM":
        return None

    timestep = struct.unpack_from("<q", raw, 26)[0]
    natoms = struct.unpack_from("<q", raw, 34)[0]
    if natoms <= 0:
        raise BinaryFormatError("DUMPCUSTOM header reported a non-positive atom count.")

    box_values = struct.unpack_from("<6d", raw, 70)
    box_bounds = np.array(
        [
            [box_values[0], box_values[1]],
            [box_values[2], box_values[3]],
            [box_values[4], box_values[5]],
        ],
        dtype=np.float64,
    )

    size_one = struct.unpack_from("<i", raw, 118)[0]
    if size_one <= 0:
        raise BinaryFormatError("DUMPCUSTOM header reported an invalid column count.")

    field_text, field_end = _extract_dumpcustom_field_text(raw, 122)
    field_names = tuple(field_text.split())
    if len(field_names) != size_one:
        raise BinaryFormatError(
            "DUMPCUSTOM column count does not match the field-name list. "
            f"size_one={size_one}, parsed fields={len(field_names)}."
        )

    data_count = struct.unpack_from("<i", raw, field_end + 4)[0]
    if data_count != natoms * size_one:
        raise BinaryFormatError(
            "DUMPCUSTOM data count does not match natoms * size_one. "
            f"data_count={data_count}, expected={natoms * size_one}."
        )

    position_columns = _resolve_position_columns(field_names)
    id_column = field_names.index("id") if "id" in field_names else None
    type_column = field_names.index("type") if "type" in field_names else None
    header_size = field_end + 8
    frame_size = header_size + data_count * 8

    return DumpCustomHeader(
        timestep=int(timestep),
        natoms=int(natoms),
        box_bounds=box_bounds,
        size_one=int(size_one),
        field_names=field_names,
        header_size=int(header_size),
        data_count=int(data_count),
        frame_size=int(frame_size),
        id_column=id_column,
        type_column=type_column,
        position_columns=position_columns,
    )


def format_mismatch_message(format_name: str) -> str:
    return (
        "Header mismatch detected. "
        f"This file appears to be a {format_name} file, not a raw binary. "
        "Please rename the extension or select the correct parser."
    )


class BinaryParser(BaseParser):
    """
    Parser for raw binary coordinate trajectories stored as `.bin`.

    The file is only treated as raw binary when its first 8 bytes do not
    match a known trajectory signature.
    """

    def __init__(
        self,
        *,
        binary_config: RawBinaryConfig | None = None,
        topology_path: Path | None = None,
        progress_callback=None,
    ) -> None:
        self._binary_config = binary_config
        self._topology_path = topology_path
        self._progress = progress_callback or (lambda _pct: None)

    def parse(
        self,
        path: Path,
        topology_path: Path | None = None,
    ) -> tuple[list[Atom], Sequence[np.ndarray], SystemMetadata]:
        mismatch = inspect_binary_signature(path)
        if mismatch is not None:
            raise FormatMismatchError(
                format_mismatch_message(mismatch.format_name)
            )

        dumpcustom = inspect_dumpcustom_header(path)
        if dumpcustom is not None:
            return self._parse_dumpcustom(path, dumpcustom, topology_path)

        config = self._binary_config
        if config is None:
            raise BinaryFormatError(
                "Raw binary detected, but no binary configuration was provided. "
                "Please open the file from the UI and supply atoms/frame, "
                "precision, endianness, and byte offset."
            )

        config.validate()

        topo_path = topology_path or self._topology_path
        if topo_path is None:
            raise BinaryFormatError(
                "Raw binary trajectories require a separate topology file "
                "(.pdb or .gro)."
            )
        if topo_path.suffix.lower() not in {".pdb", ".gro"}:
            raise BinaryFormatError(
                "Raw binary trajectories require a topology file in PDB or GRO format."
            )

        self._progress(10)
        atoms, _topo_frames, topo_metadata = self._parse_topology(topo_path)
        self._progress(35)

        if not atoms:
            raise BinaryFormatError(
                f"Topology file '{topo_path.name}' did not produce any atoms."
            )
        if len(atoms) != config.atoms_per_frame:
            raise BinaryFormatError(
                "Atoms per frame does not match the topology atom count. "
                f"Topology has {len(atoms)} atoms, but the binary configuration "
                f"expects {config.atoms_per_frame}."
            )

        trajectory = self._build_memmap_trajectory(path, config)
        n_frames = len(trajectory)
        if n_frames == 0:
            raise BinaryFormatError(
                f"Raw binary file '{path.name}' contains no complete frames."
            )

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=topo_metadata.box_bounds,
            timesteps=list(range(n_frames)),
            bonds=topo_metadata.bonds,
        )
        self._progress(100)
        return atoms, trajectory, metadata

    def _parse_dumpcustom(
        self,
        path: Path,
        header: DumpCustomHeader,
        topology_path: Path | None,
    ) -> tuple[list[Atom], DumpCustomTrajectory, SystemMetadata]:
        topo_path = topology_path or self._topology_path

        self._progress(10)
        frame_view = self._build_dumpcustom_view(path, header)
        trajectory = DumpCustomTrajectory(frame_view, header.position_columns)
        timesteps = self._read_dumpcustom_timesteps(path, header, len(trajectory))

        if topo_path is None:
            atoms = self._build_skeleton_atoms(frame_view[0], header)
            bonds = np.array([], dtype=np.int64)
        else:
            if topo_path.suffix.lower() not in {".pdb", ".gro"}:
                raise BinaryFormatError(
                    "Optional topology for DUMPCUSTOM binary files must be PDB or GRO."
                )
            atoms, _topo_frames, topo_metadata = self._parse_topology(topo_path)
            if len(atoms) != header.natoms:
                raise BinaryFormatError(
                    "DUMPCUSTOM atom count does not match the topology atom count. "
                    f"Header has {header.natoms} atoms, topology has {len(atoms)}."
                )
            bonds = topo_metadata.bonds

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=header.box_bounds,
            timesteps=timesteps,
            bonds=bonds,
        )
        self._progress(100)
        return atoms, trajectory, metadata

    def _parse_topology(
        self, topo_path: Path
    ) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        from PSVAP.io.base_parser import detect_parser

        parser = detect_parser(topo_path)
        return parser.parse(topo_path)

    def _build_dumpcustom_view(
        self,
        path: Path,
        header: DumpCustomHeader,
    ) -> np.ndarray:
        file_size = path.stat().st_size
        if file_size % header.frame_size != 0:
            raise BinaryFormatError(
                "DUMPCUSTOM file size is not divisible by the parsed frame size. "
                f"file_size={file_size}, frame_size={header.frame_size}."
            )

        n_frames = file_size // header.frame_size
        raw = np.memmap(str(path), dtype=np.uint8, mode="r")
        view = np.ndarray(
            shape=(n_frames, header.natoms, header.size_one),
            dtype=np.dtype("<f8"),
            buffer=raw,
            offset=header.header_size,
            strides=(header.frame_size, header.size_one * 8, 8),
        )
        return view

    def _read_dumpcustom_timesteps(
        self,
        path: Path,
        header: DumpCustomHeader,
        n_frames: int,
    ) -> list[int]:
        timesteps: list[int] = []
        with path.open("rb") as fh:
            for frame_idx in range(n_frames):
                fh.seek(frame_idx * header.frame_size + 26)
                raw = fh.read(8)
                if len(raw) != 8:
                    raise BinaryFormatError(
                        f"Could not read timestep for DUMPCUSTOM frame {frame_idx}."
                    )
                timesteps.append(struct.unpack("<q", raw)[0])
        return timesteps

    def _build_skeleton_atoms(
        self,
        first_frame: np.ndarray,
        header: DumpCustomHeader,
    ) -> list[Atom]:
        atoms: list[Atom] = []
        for i in range(header.natoms):
            atom_id = i + 1
            if header.id_column is not None:
                atom_id = int(round(float(first_frame[i, header.id_column])))

            type_id = None
            if header.type_column is not None:
                type_id = int(round(float(first_frame[i, header.type_column])))

            x_idx, y_idx, z_idx = header.position_columns

            atoms.append(
                Atom(
                    id=atom_id,
                    type_id=type_id,
                    element="X",
                    x=float(first_frame[i, x_idx]),
                    y=float(first_frame[i, y_idx]),
                    z=float(first_frame[i, z_idx]),
                    name=f"Atom {i + 1}",
                )
            )
        return atoms

    def _build_memmap_trajectory(
        self,
        path: Path,
        config: RawBinaryConfig,
    ) -> MemmapTrajectory:
        file_size = path.stat().st_size
        payload_size = file_size - config.byte_offset
        frame_size = config.bytes_per_frame()
        if payload_size < 0:
            raise BinaryFormatError(
                f"Byte offset {config.byte_offset} exceeds file size {file_size}."
            )
        if frame_size <= 0:
            raise BinaryFormatError("Computed frame size is invalid.")
        if payload_size % frame_size != 0:
            raise BinaryFormatError(
                "Raw binary size is not divisible by the configured frame size. "
                f"Payload bytes={payload_size}, frame bytes={frame_size}."
            )

        n_frames = payload_size // frame_size
        dtype = config.numpy_dtype()
        memmap_array = np.memmap(
            str(path),
            dtype=dtype,
            mode="r",
            offset=config.byte_offset,
            shape=(n_frames, config.atoms_per_frame, 3),
            order="C",
        )
        return MemmapTrajectory(memmap_array)


def _read_header(path: Path, n_bytes: int = 8) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n_bytes)


def _extract_dumpcustom_field_text(data: bytes, start_offset: int) -> tuple[str, int]:
    """
    Locate the printable field-name text block inside the DUMPCUSTOM header.

    This is intentionally tolerant because the exact preamble before the text
    can vary slightly across LAMMPS revisions.
    """
    printable = set(range(32, 127))
    start = None
    for idx in range(start_offset, len(data)):
        if data[idx] in printable and chr(data[idx]).isalpha():
            start = idx
            break
    if start is None:
        raise BinaryFormatError("Could not locate the DUMPCUSTOM field-name block.")

    end = start
    while end < len(data) and data[end] in printable:
        end += 1

    text = data[start:end].decode("ascii", errors="strict").strip()
    if not text:
        raise BinaryFormatError("DUMPCUSTOM field-name block is empty.")
    return text, end


def _resolve_position_columns(field_names: tuple[str, ...]) -> tuple[int, int, int]:
    for labels in (("x", "y", "z"), ("xu", "yu", "zu")):
        if all(label in field_names for label in labels):
            return tuple(field_names.index(label) for label in labels)
    raise BinaryFormatError(
        "DUMPCUSTOM binary does not contain x/y/z or xu/yu/zu position columns."
    )
