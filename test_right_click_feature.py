#!/usr/bin/env python3
"""
Simple test script to verify that the right-click context menu feature is properly integrated.
This tests the structure and method existence rather than the GUI interaction itself.
"""

import sys
from pathlib import Path

# Add parent directory to path
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

def test_visualization_engine_structure():
    """Test that VisualizationEngine has the required methods and signals."""
    import inspect
    from PSVAP.visualization.viz_engine import VisualizationEngine
    
    # Check that the class has the required attributes
    class_source = inspect.getsource(VisualizationEngine)
    
    assert 'atom_right_clicked' in class_source, "Missing atom_right_clicked signal"
    print("✓ VisualizationEngine has atom_right_clicked signal")
    
    assert 'get_atom_info_at_position' in class_source, "Missing get_atom_info_at_position method"
    print("✓ VisualizationEngine has get_atom_info_at_position method")
    
    assert 'handle_atom_right_click' in class_source, "Missing handle_atom_right_click method"
    print("✓ VisualizationEngine has handle_atom_right_click method")
    
    assert 'get_plotter' in class_source, "Missing get_plotter method"
    print("✓ VisualizationEngine has get_plotter method")

def test_viewport_panel_structure():
    """Test that ViewportPanel has the required methods and event filter."""
    import inspect
    from PSVAP.gui.panels.viewport_panel import ViewportPanel
    
    class_source = inspect.getsource(ViewportPanel)
    
    # Check methods exist
    assert 'eventFilter' in class_source, "Missing eventFilter method"
    print("✓ ViewportPanel has eventFilter method")
    
    assert '_on_atom_right_clicked' in class_source, "Missing _on_atom_right_clicked method"
    print("✓ ViewportPanel has _on_atom_right_clicked method")
    
    assert '_copy_to_clipboard' in class_source, "Missing _copy_to_clipboard method"
    print("✓ ViewportPanel has _copy_to_clipboard method")
    
    assert 'QMenu' in class_source, "Missing QMenu usage"
    print("✓ ViewportPanel uses QMenu for context menu")
    
    assert 'atom_right_clicked.connect' in class_source, "Signal not connected"
    print("✓ ViewportPanel connects to atom_right_clicked signal")

def test_atom_structure():
    """Test that Atom class has the expected properties."""
    from PSVAP.core.atom import Atom
    
    # Create a test atom
    test_atom = Atom(
        id=0,
        type_id=1,
        element='C',
        x=0.0,
        y=0.0,
        z=0.0,
        name='CA'
    )
    
    # Check properties
    assert test_atom.id == 0, "Atom id property not working"
    assert test_atom.element == 'C', "Atom element property not working"
    assert test_atom.type_id == 1, "Atom type_id property not working"
    print("✓ Atom class properties working correctly")

def main():
    """Run all tests."""
    print("Testing right-click context menu feature implementation...")
    print()
    
    try:
        test_visualization_engine_structure()
        print()
        test_viewport_panel_structure()
        print()
        test_atom_structure()
        print()
        print("✅ All structure tests passed!")
        print()
        print("To test the feature manually:")
        print("  1. Run: python PSVAP/main.py --traj path/to/trajectory.lammpstrj")
        print("  2. Load a trajectory file via File → Open")
        print("  3. Right-click on any particle in the visualizer")
        print("  4. A context menu should appear showing:")
        print("     - Particle ID and Type")
        print("     - Copy action to clipboard")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
