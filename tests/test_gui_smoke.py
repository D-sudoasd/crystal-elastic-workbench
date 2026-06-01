import os
import json

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_loads_example_and_analyzes(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_example_by_name("Si cubic")
    window.analyze_current_matrix()

    assert window.current_tensor is not None
    assert window.example_combo.currentText() == "Si cubic"
    assert window.current_summary is not None
    assert window.summary_table.rowCount() > 0
    assert window.model_table.rowCount() == 4
    assert window.export_model_table_button.text() == "Export Model Table"
    assert "overall_stable" in window.stability_text.toPlainText()
    window.close()
    app.processEvents()


def test_main_window_exposes_publication_plot_controls(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow
    from crystal_elastic_workbench.plot_styles import list_palette_names

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.theme_combo.currentText() == "Nature White"
    assert set(list_palette_names()).issuperset(
        {window.theme_combo.itemText(index) for index in range(window.theme_combo.count())}
    )
    assert set(list_palette_names()).issuperset(
        {window.palette_combo.itemText(index) for index in range(window.palette_combo.count())}
    )
    assert window.export_dpi_spin.value() == 300
    assert window.transparent_background_checkbox.isChecked() is False
    assert window.lighting_spin.value() == 100
    assert window.surface_smoothing_spin.value() == 0
    assert window.show_edges_checkbox.isChecked() is False
    assert window.cmap_combo.currentText() == "Nature Surface"
    assert window.cmap_combo.itemText(0) == "Nature Surface"
    assert window.cmap_combo.findText("Blue-White-Red") >= 0

    window.close()
    app.processEvents()


def test_main_window_exposes_scientific_dashboard_controls(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QHeaderView

    from crystal_elastic_workbench.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.tabs.tabText(0) == "Dashboard"
    assert "Ready" in window.workflow_status_label.text()
    assert "Material" in window.material_status_chip.text()
    assert "Crystal" in window.crystal_status_chip.text()
    assert "Unit" in window.unit_status_chip.text()
    assert "3D backend" in window.pyvista_status_chip.text()
    assert window.analyze_workflow_button.text() == "Analyze + Update Figures"
    assert window.paste_matrix_button.text() == "Paste Matrix"
    assert window.cij_table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Stretch
    assert window.export_paper_figures_button.text() == "Export Paper Figures"
    assert window.export_full_package_button.text() == "Export Full Package"

    window.close()
    app.processEvents()


def test_surface_plot_uses_high_quality_image_preview_when_pyvista_available(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow
    from crystal_elastic_workbench.render3d import pyvista_status

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_example_by_name("Si cubic")
    window.theta_spin.setValue(7)
    window.phi_spin.setValue(13)
    window.update_surface_plot()

    assert window.current_surface is not None
    if pyvista_status().available:
        assert window.surface_pane.preview_mode == "image"
    else:
        assert window.surface_pane.preview_mode == "figure"

    window.close()
    app.processEvents()


def test_analyze_workflow_updates_dashboard_and_default_figures(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow
    from crystal_elastic_workbench.render3d import pyvista_status

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_example_by_name("Si cubic")
    window.analyze_current_matrix()

    assert window.current_tensor is not None
    assert window.current_summary is not None
    assert window.current_line_data is not None
    assert window.current_polar_data is not None
    assert window.current_surface is not None
    assert "Si cubic" in window.material_status_chip.text()
    assert "cubic" in window.crystal_status_chip.text()
    assert "GPa" in window.unit_status_chip.text()
    assert "Stable" in window.workflow_status_label.text()
    assert "updated" in window.figure_status_label.text().lower()
    assert "Recommended model: Hill" in window.anisotropy_summary_label.text()
    assert window.model_table.item(0, 0).text() == "Voigt"
    assert window.model_table.item(2, 0).text() == "Hill"
    assert window.metric_labels["B_H"].text() != "-"
    assert window.metric_labels["G_H"].text() != "-"
    assert window.metric_labels["E_H"].text() != "-"
    assert window.metric_labels["A_U"].text() != "-"
    if pyvista_status().available:
        assert window.surface_pane.preview_mode == "image"
    else:
        assert window.surface_pane.preview_mode == "figure"

    window.close()
    app.processEvents()


def test_paste_matrix_from_clipboard_populates_cij_table(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
    from crystal_elastic_workbench.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    matrix = EXAMPLE_MATERIALS["Si cubic"].matrix
    clipboard_text = "\n".join("\t".join(f"{value:.8g}" for value in row) for row in matrix)
    QApplication.clipboard().setText(clipboard_text)

    window.populate_matrix(np.zeros((6, 6)))
    window.paste_matrix_from_clipboard()

    assert np.allclose(window.read_matrix(), matrix)

    window.close()
    app.processEvents()


def test_non_numeric_cij_cell_is_reported_and_highlighted(qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.cij_table.item(2, 4).setText("not-a-number")

    with pytest.raises(ValueError, match="C33-C13"):
        window.read_matrix()

    assert window.cij_table.item(2, 4).background().color() == QColor("#ffd6d6")

    window.close()
    app.processEvents()


def test_batch_paper_figure_export_writes_pngs_and_manifests(tmp_path, qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_example_by_name("Si cubic")
    window.theta_spin.setValue(7)
    window.phi_spin.setValue(13)
    window.analyze_current_matrix()

    exported = window.export_paper_figures_to_directory(tmp_path)

    assert set(exported) == {"line_png", "polar_png", "surface_png"}
    for path in exported.values():
        assert path.exists()
        assert path.stat().st_size > 1000
        manifest = json.loads(path.with_name(f"{path.name}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["parameters"]["theme"] == "Nature White"
        assert manifest["parameters"]["dpi"] == 300
    surface_manifest = json.loads(
        exported["surface_png"].with_name(f"{exported['surface_png'].name}.manifest.json").read_text(encoding="utf-8")
    )
    assert surface_manifest["parameters"]["backend"] in {"pyvista", "matplotlib"}

    window.close()
    app.processEvents()


def test_gui_mp4_export_forwards_surface_style_options(tmp_path, monkeypatch, qtbot=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench import gui as gui_module
    from crystal_elastic_workbench.core import ElasticTensor
    from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
    from crystal_elastic_workbench.gui import MainWindow
    from crystal_elastic_workbench.sampling import sample_sphere

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    window.current_tensor = tensor
    window.current_surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)
    window.lighting_spin.setValue(77)
    window.surface_smoothing_spin.setValue(35)
    window.show_edges_checkbox.setChecked(True)
    palette_index = window.cmap_combo.findText("Blue-Gold")
    assert palette_index >= 0
    window.cmap_combo.setCurrentIndex(palette_index)
    output = tmp_path / "surface.mp4"
    seen = {}

    def fake_get_save_file_name(*args, **kwargs):
        return str(output), "MP4 (*.mp4)"

    def fake_export(tensor_arg, surface_arg, output_path, *, options):
        seen["tensor"] = tensor_arg
        seen["surface"] = surface_arg
        seen["path"] = output_path
        seen["options"] = options
        output.write_bytes(b"mp4 bytes")
        return output

    monkeypatch.setattr(gui_module.QFileDialog, "getSaveFileName", fake_get_save_file_name)
    monkeypatch.setattr(gui_module.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_module, "export_surface_mp4_animation", fake_export)

    window.export_mp4()

    assert seen["tensor"] is tensor
    assert seen["surface"] is window.current_surface
    assert seen["path"] == str(output)
    assert seen["options"].palette_name == "Blue-Gold"
    assert seen["options"].lighting_intensity == pytest.approx(0.77)
    assert seen["options"].surface_smoothing == pytest.approx(0.35)
    assert seen["options"].show_edges is True

    window.close()
    app.processEvents()
