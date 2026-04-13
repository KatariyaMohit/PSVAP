# Right-Click Context Menu Feature - Implementation Summary

## Overview
Added right-click context menu functionality to the PSVAP visualizer that displays particle ID and type, with the ability to copy this information to the clipboard.

## Changes Made

### 1. **VisualizationEngine** (`PSVAP/visualization/viz_engine.py`)

#### New Signal
- Added `atom_right_clicked` signal that emits a dictionary with atom information

#### New Methods
- **`get_atom_info_at_position(picked_point)`**: Extracts atom information from a 3D position
  - Returns: `dict` with keys:
    - `index`: atom index in the atoms list
    - `atom`: the Atom object
    - `position`: 3D coordinates of the atom
    - `type_label`: element symbol or type ID label

- **`handle_atom_right_click(picked_point)`**: Handles right-click events and emits signal
  - Calls `get_atom_info_at_position()` and emits `atom_right_clicked` signal

- **`get_plotter()`**: Returns the attached plotter widget for event filter access

#### Refactored Code
- Refactored `_on_atom_picked()` to use the new `get_atom_info_at_position()` helper method
- Reduces code duplication and improves maintainability

### 2. **ViewportPanel** (`PSVAP/gui/panels/viewport_panel.py`)

#### Event Handling
- Installed event filter on plotter to detect right-click (MouseButtonPress + RightButton)
- Implemented `eventFilter()` method that:
  - Detects right-click events
  - Uses VTK's world point picker to convert 2D mouse coordinates to 3D world position
  - Calls `viz.handle_atom_right_click()` with the 3D position

#### Context Menu UI
- Implemented `_on_atom_right_clicked()` method that:
  - Creates a QMenu showing particle info (ID and Type)
  - Adds "Copy ID & Type" action to clipboard
  - Shows menu at cursor position

- Implemented `_copy_to_clipboard()` helper method
  - Uses QGuiApplication.clipboard() to copy text

#### Signal Connection
- Connected `viz.atom_right_clicked` signal to `_on_atom_right_clicked()` slot in `__init__`

## User Experience

### How to Use
1. Launch the application: `python PSVAP/main.py`
2. Load a trajectory file (via File → Open or command line argument)
3. Right-click on any particle in the visualizer
4. A context menu appears showing:
   - **Particle ID and Type** (e.g., "ID: 42  |  Type: C")
   - **Copy ID & Type** action to copy to clipboard

### Menu Information
- **ID**: The atom's index in the particles list
- **Type**: Either the element symbol (for PDB/CIF/GRO/XYZ/SDF files) or "TYPE N" for LAMMPS numeric type IDs

## Technical Details

### Coordinate System Conversion
- Mouse coordinates (2D screen space) are converted to 3D world space using VTK's `vtkWorldPointPicker`
- Y-axis is flipped because VTK uses bottom-left origin while Qt uses top-left

### Atom Info Extraction
- Element-based files: Use element symbol (e.g., "C", "N", "O")
- LAMMPS files: Use type ID (e.g., "TYPE 0", "TYPE 1")
- Falls back to type ID if element symbol not available

## Files Modified
1. `PSVAP/visualization/viz_engine.py`
   - Added signal, helper methods, refactored existing code

2. `PSVAP/gui/panels/viewport_panel.py`
   - Added complete right-click handling and context menu implementation

## Testing
Run the included test script to verify the implementation:
```bash
cd /home/bomore/Work/soft_proj/PSVAP
python test_right_click_feature.py
```

Or test manually by running the application and right-clicking on particles in the visualizer.

## Future Enhancements
- Add more options to context menu (e.g., select atom, highlight neighbors, show properties)
- Show extended atom information in tooltip on hover
- Add customizable copy format options
- Add keyboard shortcut support for copy action
