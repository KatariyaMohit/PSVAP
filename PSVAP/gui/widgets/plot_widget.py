"""
gui/widgets/plot_widget.py
---------------------------
Phase 7: Embeddable Matplotlib plot widget.

Wraps a Matplotlib FigureCanvas for embedding 2D charts inside Qt panels.
Falls back gracefully to a placeholder label if Matplotlib is not available.

Public API
----------
  PlotWidget(parent)
    .plot_line(x, y, title, xlabel, ylabel, color)
    .plot_bar(labels, values, title, ylabel, color)
    .plot_scatter(x, y, title, xlabel, ylabel)
    .clear()
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

BG_COLOR   = "#0A0A0A"
TEXT_COLOR = "#CCCCCC"
GRID_COLOR = "#2A2A2A"
ACCENT     = "#E8FF00"


class PlotWidget(QWidget):
    """
    Embeddable Matplotlib plot widget.

    If Matplotlib is not installed, shows a placeholder label.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._canvas = None
        self._fig    = None
        self._ax     = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._init_matplotlib()

    def _init_matplotlib(self) -> None:
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            matplotlib.rcParams.update({
                "figure.facecolor":  BG_COLOR,
                "axes.facecolor":    BG_COLOR,
                "axes.edgecolor":    GRID_COLOR,
                "axes.labelcolor":   TEXT_COLOR,
                "xtick.color":       TEXT_COLOR,
                "ytick.color":       TEXT_COLOR,
                "text.color":        TEXT_COLOR,
                "grid.color":        GRID_COLOR,
                "grid.linestyle":    "--",
                "grid.alpha":        0.5,
                "font.size":         9,
                "font.family":       "sans-serif",
            })

            self._fig    = Figure(figsize=(4, 3), dpi=90, tight_layout=True)
            self._ax     = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setStyleSheet(f"background:{BG_COLOR}; border:none;")
            self._layout.addWidget(self._canvas)

        except ImportError:
            placeholder = QLabel("Plot area\n(install matplotlib to enable)")
            placeholder.setStyleSheet(
                f"color:#555555; font-size:10px; "
                f"background:{BG_COLOR}; border:1px solid #2A2A2A; padding:12px;")
            self._layout.addWidget(placeholder)

    def _ensure_axes(self) -> bool:
        """Return True if Matplotlib axes are available."""
        return self._ax is not None and self._canvas is not None

    def plot_line(
        self,
        x,
        y,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        color: str = ACCENT,
    ) -> None:
        """Plot a line chart."""
        if not self._ensure_axes():
            return
        import numpy as np
        self._ax.clear()
        self._ax.plot(x, y, color=color, linewidth=1.5)
        self._ax.set_title(title, fontsize=10, color=TEXT_COLOR)
        self._ax.set_xlabel(xlabel, fontsize=9)
        self._ax.set_ylabel(ylabel, fontsize=9)
        self._ax.grid(True)
        self._canvas.draw()

    def plot_bar(
        self,
        labels,
        values,
        title: str = "",
        ylabel: str = "",
        color: str = ACCENT,
    ) -> None:
        """Plot a bar chart."""
        if not self._ensure_axes():
            return
        self._ax.clear()
        x_pos = range(len(labels))
        self._ax.bar(x_pos, values, color=color, alpha=0.8)
        self._ax.set_xticks(list(x_pos))
        if len(labels) <= 20:
            self._ax.set_xticklabels(
                [str(l) for l in labels], rotation=45, ha="right", fontsize=8
            )
        else:
            self._ax.set_xticklabels([])
        self._ax.set_title(title, fontsize=10, color=TEXT_COLOR)
        self._ax.set_ylabel(ylabel, fontsize=9)
        self._ax.grid(True, axis="y")
        self._canvas.draw()

    def plot_scatter(
        self,
        x,
        y,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        color: str = ACCENT,
        alpha: float = 0.6,
    ) -> None:
        """Plot a scatter chart (used for Ramachandran plots)."""
        if not self._ensure_axes():
            return
        self._ax.clear()
        self._ax.scatter(x, y, c=color, s=8, alpha=alpha)
        self._ax.set_title(title, fontsize=10, color=TEXT_COLOR)
        self._ax.set_xlabel(xlabel, fontsize=9)
        self._ax.set_ylabel(ylabel, fontsize=9)
        self._ax.grid(True)
        self._canvas.draw()

    def clear(self) -> None:
        """Clear the plot area."""
        if self._ensure_axes():
            self._ax.clear()
            self._canvas.draw()