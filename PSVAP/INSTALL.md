PSVAP Installation & Setup
==========================

These instructions summarize Section 10 (Setup, Build & Run Instructions) of the
PSVAP Master Plan. Whenever installation or run commands change, update both
this file and the changelog.

Prerequisites
-------------

- Python 3.11 or higher
- Git
- C/C++ build toolchain
  - Windows: Visual Studio Build Tools (including MSVC and CMake components)
  - Linux: `build-essential` (gcc, g++, make)
- Conda (Miniconda or Anaconda) — **strongly recommended** for RDKit and other
  compiled scientific packages.

Environment Setup
-----------------

From the PSVAP repository root:

```bash
# 1. (Optional here) Clone the repository if not already present
# git clone https://github.com/<your-org>/PSVAP.git
# cd PSVAP

# 2. Create a conda environment
conda create -n psvap python=3.11
conda activate psvap

# 3. Install RDKit via conda-forge
conda install -c conda-forge rdkit

# 4. Install remaining Python dependencies
pip install -r requirements.txt

# 5. Install developer tools (tests, linting)
pip install -r requirements-dev.txt
```

External Tools (PATH)
---------------------

The following executables are used by various PSVAP modules and must be
available on your `PATH`. Not all are required for Phase 0, but the Master Plan
defines them for the full system:

- AutoDock Vina — `vina`
- PACKMOL — `packmol`
- fpocket — `fpocket`
- propka — `propka`
- martinize2 — `martinize2`
- Open Babel — `obabel`
- SCWRL4 — `scwrl4` (optional)

You can verify availability, for example:

```bash
vina --version
fpocket --help
obabel --version
```

Running the Application
-----------------------

From the PSVAP root directory:

```bash
conda activate psvap
python main.py
```

To start with specific input files:

```bash
python main.py --topo path/to/topology.pdb --traj path/to/trajectory.xtc
```

Testing and Type Checking
-------------------------

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_geometry.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Static type checking
mypy . --ignore-missing-imports
```

