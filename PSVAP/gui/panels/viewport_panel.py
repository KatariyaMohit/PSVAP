from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout
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

