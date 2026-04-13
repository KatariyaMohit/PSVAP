from __future__ import annotations

from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu
from pyvistaqt import QtInteractor

from PSVAP.app.controller import ApplicationController


class ViewportPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        self.controller.viz.attach_plotter(self.plotter)
        
        # Install event filter on plotter to detect right-click
        self.plotter.installEventFilter(self)

    def eventFilter(self, obj, event: QEvent) -> bool:
        """Intercept right-click events on the plotter."""
        if obj == self.plotter:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.RightButton:
                    # Store mouse position and show context menu for atom
                    self._show_atom_context_menu(event.pos())
                    return True  # Consume the event
        
        return super().eventFilter(obj, event)

    def _show_atom_context_menu(self, mouse_pos: QPoint) -> None:
        """Show context menu for the atom at the given mouse position."""
        try:
            # Get atom info at this position
            atom_info = self._pick_atom_at_pos(mouse_pos)
            
            if atom_info is None:
                return
            
            idx = atom_info['index']
            type_label = atom_info['type_label']
            
            # Prepare display info
            info_text = f"ID: {idx}  |  Type: {type_label}"
            
            # Create context menu
            menu = QMenu(self)
            
            # Add info action (disabled, just for display)
            info_action = menu.addAction(info_text)
            info_action.setEnabled(False)
            menu.setStyleSheet("QMenu { background-color: #2B2B2B; color: white; }")
            
            menu.addSeparator()
            
            # Add copy to clipboard action
            copy_action = menu.addAction("Copy ID & Type")
            copy_action.triggered.connect(
                lambda: self._copy_to_clipboard(f"{idx}:{type_label}")
            )
            
            # Show menu at cursor position
            cursor_pos = self.plotter.mapToGlobal(mouse_pos)
            menu.exec(cursor_pos)
            
        except Exception as e:
            print(f"Error showing context menu: {e}")

    def _pick_atom_at_pos(self, mouse_pos: QPoint) -> dict | None:
        """Pick atom at the given mouse position and return its info."""
        try:
            renderer = self.plotter.renderer
            render_window = self.plotter.render_window
            
            if not renderer or not render_window:
                return None
            
            # Get render window dimensions
            width, height = render_window.GetSize()
            
            # Convert mouse position (flip Y because VTK uses bottom-left origin)
            x = int(mouse_pos.x())
            y = int(height - mouse_pos.y())
            
            # Use VTK picker
            import vtk
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, y, 0, renderer)
            picked_pos = picker.GetPickPosition()
            
            if not picked_pos:
                return None
            
            # Get atom at this position using the visualization engine's method
            atom_info = self.controller.viz.get_atom_info_at_position(picked_pos)
            return atom_info
            
        except Exception as e:
            print(f"Error picking atom: {e}")
            return None

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard."""
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(text)


