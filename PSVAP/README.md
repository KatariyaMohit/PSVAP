# PSVAP — Particle Simulation Visualization & Analysis Package

PSVAP is a post-simulation analysis and visualization tool for particle-based
molecular dynamics data. It is **not** a simulation engine. It reads trajectory
and structure files from external engines (LAMMPS, GROMACS, AMBER, CHARMM, etc.)
and provides a Python-first, plugin-ready environment for analysis and visualization.

This README covers:

- High-level folder structure
- Full setup and run instructions

## Folder Structure 

At the top level:

| Path                   | Description                                              |
|------------------------|----------------------------------------------------------|
| `main.py`              | Entry point; starts the Qt application only.             |
| `README.md`            | This summary.                                            |
| `CHANGELOG.md`         | Living changelog; update every significant change.       |
| `INSTALL.md`           | Setup instructions for new developers.                   |
| `requirements.txt`     | Pinned runtime dependencies.                             |
| `requirements-dev.txt` | Testing and linting dependencies.                        |
| `pyproject.toml`       | Package metadata and build configuration.                |
| `app/`                 | Application controller layer.                            |
| `core/`                | Core data layer, including `SystemModel`.                |
| `io/`                  | Parsers, exporters, and external engine integration.     |
| `visualization/`       | Visualization engine wrapping PyVista.                   |
| `analysis/`            | Scientific analysis modules.                             |
| `modeling/`            | Structure modification and MD setup tools.               |
| `plugins/`             | Python plugin sandbox and public API.                    |
| `gui/`                 | PySide6 GUI layer.                                       |
| `tests/`               | Automated tests and fixtures.                            |
| `docs/`                | Documentation and diagrams.                              |


---

## Installation & Setup

### Prerequisites

- Python 3.11 (recommended)
- Git
- Conda (Miniconda / Anaconda) — strongly recommended
- C/C++ build toolchain:
  - **Windows:** Visual Studio Build Tools (MSVC + CMake)
  - **Linux:** `build-essential` (gcc, g++, make)

---

### Clone the Repository

```bash
git clone https://github.com/KatariyaMohit/PSVAP.git
cd PSVAP
```

---

### Environment Setup (IMPORTANT)

> Always use **Anaconda Prompt** or a fresh terminal.

#### Create Conda Environment

```bash
conda create -n psvap python=3.11
```

Press `y` when prompted.

#### Activate Environment

```bash
conda activate psvap
```

You should see:

```
(psvap) C:\Users\username\Software Project\PSVAP>
```

---

### Install Dependencies (Correct Order)

> **IMPORTANT:** Install heavy libraries via `conda`, not `pip`

#### Install Core Scientific & GUI Libraries

```bash
conda install -c conda-forge rdkit pyvista vtk pyvistaqt pyside6=6.6 qt=6.6
```

This ensures:
- No DLL errors
- Compatible Qt + VTK + PyVista stack

#### Install Remaining Python Dependencies

```bash
pip install -r requirements.txt
```

#### Install Developer Tools (Optional)

```bash
pip install -r requirements-dev.txt
```

---

### Run the Application

```bash
conda activate psvap
python PSVAP\main.py
```

---

## Important Notes (VERY IMPORTANT)

Always follow this rule:

| Library Type       | Install Using |
|--------------------|---------------|
| RDKit              | conda         |
| PySide6 (Qt GUI)   | conda         |
| VTK / PyVista      | conda         |
| Other Python libs  | pip           |

### ❌ Avoid This (causes errors)

Mixing `pip` and `conda` for:
- PySide6
- VTK
- PyVista

---

## Common Errors & Fixes

### `conda not recognized`
- Add Miniconda to PATH
- Restart terminal

---

### `No module named PySide6`

```bash
conda install -c conda-forge pyside6
```

---

### `DLL load failed while importing QtWidgets`

**Fix:**

```bash
pip uninstall PySide6 PySide6-Essentials PySide6-Addons -y
conda remove pyside6 qt --force -y
conda install -c conda-forge pyside6=6.6 qt=6.6
```

---

### VTK / PyVista Import Errors

```bash
conda install -c conda-forge vtk pyvista pyvistaqt
```

---

## Clean Reset (If Things Break)

```bash
conda remove -n psvap --all
conda create -n psvap python=3.11
conda activate psvap
conda install -c conda-forge rdkit pyvista vtk pyvistaqt pyside6=6.6 qt=6.6
pip install -r requirements.txt
```

---

## Summary

- ✅ Use `conda` for heavy dependencies
- ✅ Use `pip` only for lightweight packages
- ✅ Keep environment clean to avoid DLL issues

---
