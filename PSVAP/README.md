PSVAP — Particle Simulation Visualization & Analysis Package
===========================================================

This repository implements the PSVAP desktop application as defined in the
`PSVAP_Master_Plan` (Software Development Master Plan, v1.0).

PSVAP is a post-simulation analysis and visualization tool for particle-based
molecular dynamics data. It is **not** a simulation engine. It reads trajectory
and structure files from external engines (LAMMPS, GROMACS, AMBER, CHARMM, etc.)
and provides a Python-first, plugin-ready environment for analysis and
visualization.

The authoritative source of truth for architecture, modules, dependencies, and
development phases is the Master Plan document. This README only summarizes:

- Project purpose and scope
- High-level folder structure
- Basic setup and run instructions

Project Purpose
---------------

- High cohesion, low coupling across clearly separated layers.
- Python-first implementation of all scientific and application logic.
- Extensible analysis and modeling via a safe Python plugin sandbox.
- Research-grade reliability with validated calculations.
- Cross-platform desktop application targeting Windows 10/11 and Ubuntu 22.04+.

Folder Structure (Planned)
--------------------------

At the top level:

- `main.py` — entry point; starts the Qt application only.
- `README.md` — this summary.
- `CHANGELOG.md` — living changelog; update every significant change.
- `INSTALL.md` — setup instructions for new developers.
- `requirements.txt` — pinned runtime dependencies.
- `requirements-dev.txt` — testing and linting dependencies.
- `pyproject.toml` — package metadata and build configuration.
- `app/` — application controller layer.
- `core/` — core data layer, including `SystemModel`.
- `io/` — parsers, exporters, and external engine integration.
- `visualization/` — visualization engine wrapping PyVista.
- `analysis/` — scientific analysis modules.
- `modeling/` — structure modification and MD setup tools.
- `plugins/` — Python plugin sandbox and public API.
- `gui/` — PySide6 GUI layer.
- `tests/` — automated tests and fixtures.
- `docs/` — documentation and diagrams.

For full details, refer to Section 3 (Complete Project Folder Structure) of the
Master Plan.

Quick Start (Summary)
---------------------

1. Create and activate a Python 3.11 environment (Conda recommended).
2. Install RDKit from `conda-forge`.
3. Install remaining Python dependencies from `requirements.txt`.
4. Install developer tools from `requirements-dev.txt` (optional for development).
5. Run `python main.py` from the repository root to launch the application.

The exact commands, external tool checks, and supported CLI options are
specified in Section 10 (Setup, Build & Run Instructions) of the Master Plan and
must be kept in sync with this repository.

