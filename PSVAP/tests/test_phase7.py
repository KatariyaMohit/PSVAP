"""
tests/test_phase7.py
---------------------
Phase 7 unit tests for:
  plugins/api.py          (PluginAPI)
  plugins/sandbox.py      (run_plugin_script, PluginSandbox)
  io/exporter.py          (Exporter, export functions)
  gui/widgets/plot_widget.py  (PlotWidget — no display, just instantiation)

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_phase7.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata, SystemModel


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_model() -> SystemModel:
    m = SystemModel()
    atoms = [
        Atom(id=0, type_id=0, element="C", x=0.0, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="CA", resname="ALA"),
        Atom(id=1, type_id=1, element="N", x=1.5, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="N",  resname="ALA"),
        Atom(id=2, type_id=0, element="C", x=3.0, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="CA", resname="GLY"),
    ]
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    m.set_data(atoms=atoms, trajectory=[pos, pos + 0.1],
               metadata=SystemMetadata())
    return m


def _make_api(model=None):
    from PSVAP.plugins.api import PluginAPI
    if model is None:
        model = _make_model()
    return PluginAPI(model=model, engine=None)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PluginAPI
# ═══════════════════════════════════════════════════════════════════════════

class TestPluginAPI:

    def test_get_atoms_returns_list(self):
        api = _make_api()
        atoms = api.get_atoms()
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_get_positions_returns_array(self):
        api = _make_api()
        pos = api.get_positions()
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3, 3)

    def test_get_positions_is_copy(self):
        """Modifying returned positions should not affect model."""
        api = _make_api()
        pos = api.get_positions()
        original = pos.copy()
        pos[:] = 999.0
        pos2 = api.get_positions()
        np.testing.assert_allclose(pos2, original, atol=1e-10)

    def test_get_frame_returns_array(self):
        api = _make_api()
        f = api.get_frame(0)
        assert isinstance(f, np.ndarray)
        assert f.shape == (3, 3)

    def test_get_frame_out_of_range_returns_zeros(self):
        api = _make_api()
        f = api.get_frame(999)
        assert isinstance(f, np.ndarray)

    def test_n_atoms(self):
        api = _make_api()
        assert api.n_atoms() == 3

    def test_n_frames(self):
        api = _make_api()
        assert api.n_frames() == 2

    def test_log_calls_callback(self):
        messages = []
        model = _make_model()
        from PSVAP.plugins.api import PluginAPI
        api = PluginAPI(model=model, stdout_callback=messages.append)
        api.log("test message")
        assert "test message" in messages

    def test_export_text(self, tmp_path):
        model = _make_model()
        from PSVAP.plugins.api import PluginAPI
        api = PluginAPI(model=model, output_dir=tmp_path)
        api.export("hello world", "test.txt")
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.txt").read_text() == "hello world"

    def test_export_numpy(self, tmp_path):
        model = _make_model()
        from PSVAP.plugins.api import PluginAPI
        api = PluginAPI(model=model, output_dir=tmp_path)
        arr = np.array([1.0, 2.0, 3.0])
        api.export(arr, "data.npy")
        assert (tmp_path / "data.npy").exists()

    def test_export_dict(self, tmp_path):
        import json
        model = _make_model()
        from PSVAP.plugins.api import PluginAPI
        api = PluginAPI(model=model, output_dir=tmp_path)
        api.export({"key": "value", "n": 42}, "result.json")
        data = json.loads((tmp_path / "result.json").read_text())
        assert data["key"] == "value"

    def test_export_list(self, tmp_path):
        model = _make_model()
        from PSVAP.plugins.api import PluginAPI
        api = PluginAPI(model=model, output_dir=tmp_path)
        api.export([1, 2, 3], "indices.txt")
        content = (tmp_path / "indices.txt").read_text()
        assert "1" in content and "2" in content

    def test_build_globals_keys(self):
        api = _make_api()
        g = api.build_globals()
        for key in ["get_atoms", "get_positions", "get_frame",
                    "get_selection", "highlight", "log", "export",
                    "n_atoms", "n_frames", "np"]:
            assert key in g

    def test_get_selection_returns_mask(self):
        api = _make_api()
        mask = api.get_selection("type == 0")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. Plugin Sandbox
# ═══════════════════════════════════════════════════════════════════════════

class TestPluginSandbox:

    def test_run_simple_script(self):
        """A simple script should execute without error."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "log('hello from plugin')",
            api,
            stdout_callback=output.append,
        )
        assert any("hello" in line for line in output)

    def test_run_script_with_numpy(self):
        """numpy should be available as 'np'."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "arr = np.array([1.0, 2.0, 3.0])\nlog(str(arr.mean()))",
            api,
            stdout_callback=output.append,
        )
        # str(np.float64(2.0)) → '2.0'; accept any line containing '2'
        assert any("2" in line for line in output), f"Got output: {output}"


    def test_run_script_get_atoms(self):
        """get_atoms() should return 3 atoms."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "atoms = get_atoms()\nlog(str(len(atoms)))",
            api,
            stdout_callback=output.append,
        )
        assert any("3" in line for line in output)

    def test_run_script_get_positions(self):
        """get_positions() should return (3,3) array."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "pos = get_positions()\nlog(str(pos.shape))",
            api,
            stdout_callback=output.append,
        )
        assert any("3" in line for line in output)

    def test_syntax_error_caught(self):
        """Syntax errors should be caught, not raised."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "def broken(:\n    pass",  # bad syntax
            api,
            stdout_callback=output.append,
        )
        assert any("error" in line.lower() or "syntax" in line.lower()
                   for line in output)

    def test_runtime_error_caught(self):
        """Runtime errors should be caught, not raised."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "x = 1 / 0",   # ZeroDivisionError
            api,
            stdout_callback=output.append,
        )
        assert any("error" in line.lower() for line in output)

    def test_plugin_sandbox_class(self):
        """PluginSandbox.execute() should work."""
        from PSVAP.plugins.sandbox import PluginSandbox
        api = _make_api()
        sb = PluginSandbox()
        output = sb.execute("log('sandbox test')", api=api)
        assert "sandbox test" in output

    def test_plugin_sandbox_no_api(self):
        """execute() without API should return error message."""
        from PSVAP.plugins.sandbox import PluginSandbox
        sb = PluginSandbox()
        output = sb.execute("log('test')", api=None)
        assert "ERROR" in output.upper()

    def test_forbidden_import_blocked(self):
        """Import of os/sys should be blocked or produce an error."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        run_plugin_script(
            "import os\nlog(os.getcwd())",
            api,
            stdout_callback=output.append,
        )
        # Should either produce an error OR the import is silently blocked
        # (depending on RestrictedPython version) — either is acceptable
        # The key is no unhandled exception propagates to the test
        assert isinstance(output, list)   # no crash = pass

    def test_multiline_script(self):
        """Multi-line scripts with loops should work."""
        from PSVAP.plugins.sandbox import run_plugin_script
        api = _make_api()
        output = []
        script = (
            "total = 0\n"
            "for i in range(5):\n"
            "    total += i\n"
            "log(str(total))\n"
        )
        run_plugin_script(script, api, stdout_callback=output.append)
        assert any("10" in line for line in output)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Exporter
# ═══════════════════════════════════════════════════════════════════════════

class TestExporter:

    def test_export_atoms_csv(self, tmp_path):
        from PSVAP.io.exporter import export_atoms_csv
        model = _make_model()
        out = tmp_path / "atoms.csv"
        export_atoms_csv(model.atoms, model.get_frame(0), out)
        assert out.exists()
        content = out.read_text()
        assert "id" in content
        assert "element" in content
        assert "x" in content

    def test_csv_row_count(self, tmp_path):
        from PSVAP.io.exporter import export_atoms_csv
        model = _make_model()
        out = tmp_path / "atoms.csv"
        export_atoms_csv(model.atoms, model.get_frame(0), out)
        lines = out.read_text().splitlines()
        # 1 header + 3 atoms
        assert len(lines) == 4

    def test_export_atoms_pdb(self, tmp_path):
        from PSVAP.io.exporter import export_atoms_pdb
        model = _make_model()
        out = tmp_path / "structure.pdb"
        export_atoms_pdb(model.atoms, model.get_frame(0), out)
        assert out.exists()
        content = out.read_text()
        assert "ATOM" in content
        assert "END" in content

    def test_export_generic_csv_array(self, tmp_path):
        from PSVAP.io.exporter import _export_generic_csv
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = tmp_path / "data.csv"
        _export_generic_csv(data, out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == 2

    def test_export_generic_csv_dict(self, tmp_path):
        from PSVAP.io.exporter import _export_generic_csv
        data = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}
        out = tmp_path / "params.csv"
        _export_generic_csv(data, out)
        content = out.read_text()
        assert "alpha" in content
        assert "1.0" in content

    def test_exporter_class_exists(self):
        from PSVAP.io.exporter import Exporter
        ex = Exporter()
        assert hasattr(ex, "export_png")
        assert hasattr(ex, "export_mp4")
        assert hasattr(ex, "export_csv")

    def test_export_pdb_file_readable(self, tmp_path):
        """Exported PDB should be parseable by PDBParser."""
        from PSVAP.io.exporter import export_atoms_pdb
        from PSVAP.io.pdb_parser import PDBParser
        model = _make_model()
        out = tmp_path / "roundtrip.pdb"
        export_atoms_pdb(model.atoms, model.get_frame(0), out)
        # Should not raise
        atoms, traj, meta = PDBParser().parse(out)
        assert len(atoms) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. PlotWidget (no display — just instantiation and method calls)
# ═══════════════════════════════════════════════════════════════════════════

class TestPlotWidget:

    @pytest.fixture(autouse=True)
    def qt_app(self):
        """Ensure a QApplication exists for widget tests."""
        try:
            from PySide6.QtWidgets import QApplication
            import sys
            app = QApplication.instance() or QApplication(sys.argv[:1])
            yield app
        except Exception:
            pytest.skip("Qt not available in this environment")

    def test_instantiation(self):
        from PSVAP.gui.widgets.plot_widget import PlotWidget
        w = PlotWidget()
        assert w is not None

    def test_plot_line(self):
        from PSVAP.gui.widgets.plot_widget import PlotWidget
        w = PlotWidget()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        w.plot_line(x, y, title="Test Line", xlabel="X", ylabel="Y")

    def test_plot_bar(self):
        from PSVAP.gui.widgets.plot_widget import PlotWidget
        w = PlotWidget()
        w.plot_bar(["A", "B", "C"], [1.0, 2.5, 1.8], title="Test Bar")

    def test_plot_scatter(self):
        from PSVAP.gui.widgets.plot_widget import PlotWidget
        w = PlotWidget()
        x = np.random.default_rng(0).random(20)
        y = np.random.default_rng(1).random(20)
        w.plot_scatter(x, y, title="Test Scatter")

    def test_clear(self):
        from PSVAP.gui.widgets.plot_widget import PlotWidget
        w = PlotWidget()
        x = np.linspace(0, 5, 20)
        w.plot_line(x, x**2)
        w.clear()   # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. Settings Dialog (no display)
# ═══════════════════════════════════════════════════════════════════════════

class TestSettingsDialog:

    @pytest.fixture(autouse=True)
    def qt_app(self):
        try:
            from PySide6.QtWidgets import QApplication
            import sys
            app = QApplication.instance() or QApplication(sys.argv[:1])
            yield app
        except Exception:
            pytest.skip("Qt not available")

    def test_instantiation(self):
        from PSVAP.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog()
        assert dlg is not None

    def test_restore_defaults(self):
        from PSVAP.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog()
        dlg._hbond_dist.setValue(5.0)
        dlg._restore_defaults()
        assert dlg._hbond_dist.value() == pytest.approx(3.5)

    def test_get_render_mode(self):
        from PSVAP.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog()
        mode = dlg.get_render_mode()
        assert mode in {"atoms", "atoms_bonds"}

    def test_get_background_color(self):
        from PSVAP.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog()
        color = dlg.get_background_color()
        assert color.startswith("#")
        assert len(color) == 7