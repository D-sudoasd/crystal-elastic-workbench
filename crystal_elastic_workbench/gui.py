"""PySide6 desktop GUI for Crystal Elastic Workbench."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from crystal_elastic_workbench.animation_export import (
    AnimationExportOptions,
    export_surface_gif_animation,
    export_surface_mp4_animation,
)
from crystal_elastic_workbench.cij_matrix_table import CijMatrixTable
from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.dashboard_model import build_dashboard_state
from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.exporting import (
    export_elastic_model_table,
    export_analysis_package,
    export_sampled_data,
    write_export_manifest,
)
from crystal_elastic_workbench.elastic_models import (
    ElasticModelResult,
    elastic_model_results,
    elastic_model_table_frame,
)
from crystal_elastic_workbench.figure_export import SurfaceFigureExportOptions, export_surface_figure
from crystal_elastic_workbench.gui_services import matrix_from_frame, parse_clipboard_matrix
from crystal_elastic_workbench.gui_style import GUI_STYLE_SHEET, configure_platform_fonts
from crystal_elastic_workbench.paper_export import PaperFigureExportOptions, export_paper_figures
from crystal_elastic_workbench.plot_styles import (
    DEFAULT_3D_PALETTE_NAME,
    DEFAULT_THEME_NAME,
    get_theme,
    list_3d_palette_names,
    list_palette_names,
    list_theme_names,
)
from crystal_elastic_workbench.render3d import (
    PyVistaUnavailableError,
    Render3DOptions,
    pyvista_status,
    render_surface_image,
)
from crystal_elastic_workbench.sampling import (
    DirectionalSurface,
    sample_direction_path,
    sample_plane,
    sample_sphere,
)
from crystal_elastic_workbench.stability import StabilityResult, check_stability
from crystal_elastic_workbench.templates import CRYSTAL_SYSTEMS
from crystal_elastic_workbench.visualization import (
    plot_directional_surface,
    plot_direction_path,
    plot_line_slice,
    plot_plane_slice,
)


configure_platform_fonts()


PROPERTY_CHOICES = {
    "Young's modulus E(n)": "young",
    "Linear compressibility beta(n)": "compressibility",
    "Shear modulus G(n,m), mean": "shear",
    "Poisson ratio nu(n,m), mean": "poisson",
}


class ImagePreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class FigurePane(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._canvas: FigureCanvas | None = None
        self._toolbar: NavigationToolbar | None = None
        self._figure = None
        self._image_label: ImagePreviewLabel | None = None
        self._image_array: np.ndarray | None = None
        self.preview_mode = "empty"
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def _clear_current(self) -> None:
        if self._canvas is not None:
            if self._toolbar is not None:
                self._layout.removeWidget(self._toolbar)
                self._toolbar.setParent(None)
                self._toolbar = None
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
            if self._figure is not None:
                plt.close(self._figure)
        if self._image_label is not None:
            self._layout.removeWidget(self._image_label)
            self._image_label.setParent(None)
            self._image_label = None
        self._figure = None
        self._image_array = None
        self.preview_mode = "empty"

    def set_figure(self, figure) -> None:
        self._clear_current()
        self._figure = figure
        self._canvas = FigureCanvas(figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self.preview_mode = "figure"
        self._canvas.draw_idle()

    def set_image(self, image: np.ndarray) -> None:
        self._clear_current()
        array = np.ascontiguousarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("Preview image must be an RGB or RGBA array.")
        height, width, channels = array.shape
        image_format = QImage.Format_RGBA8888 if channels == 4 else QImage.Format_RGB888
        qimage = QImage(array.data, width, height, channels * width, image_format).copy()
        label = ImagePreviewLabel()
        label.set_preview_pixmap(QPixmap.fromImage(qimage))
        self._image_array = array.copy()
        self._image_label = label
        self._layout.addWidget(label)
        self.preview_mode = "image"

    def save_figure(self, path: str | Path, *, dpi: int = 300, transparent: bool = False) -> None:
        if self._figure is not None:
            self._figure.savefig(path, dpi=dpi, transparent=transparent)
            return
        if self._image_array is not None:
            plt.imsave(path, self._image_array)
            return
        raise RuntimeError("No figure is available to save.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Crystal Elastic Workbench")
        self.resize(1380, 860)

        self.current_tensor: ElasticTensor | None = None
        self.current_summary = None
        self.current_model_results: list[ElasticModelResult] | None = None
        self.current_stability: StabilityResult | None = None
        self.current_line_data = None
        self.current_polar_data = None
        self.current_surface: DirectionalSurface | None = None
        self.last_analysis_time: datetime | None = None

        input_panel = self._build_input_panel()
        tabs = self._build_tabs()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(input_panel)
        splitter.addWidget(tabs)
        splitter.setSizes([420, 960])

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(8)
        central_layout.addWidget(self._build_workflow_header())
        central_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)
        self._apply_app_style()
        self.load_example_by_name("Si cubic")
        self._update_workflow_status("Ready for analysis")

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        metadata = QGroupBox("Input")
        form = QFormLayout(metadata)
        self.material_edit = QLineEdit("Untitled")
        self.unit_edit = QLineEdit("GPa")
        self.system_combo = QComboBox()
        self.system_combo.addItems(CRYSTAL_SYSTEMS)
        self.example_combo = QComboBox()
        self.example_combo.addItems(EXAMPLE_MATERIALS.keys())
        load_example_button = QPushButton("Load Example")
        load_example_button.clicked.connect(lambda: self.load_example_by_name(self.example_combo.currentText()))
        example_row = QHBoxLayout()
        example_row.addWidget(self.example_combo)
        example_row.addWidget(load_example_button)
        form.addRow("Material", self.material_edit)
        form.addRow("Unit", self.unit_edit)
        form.addRow("Crystal system", self.system_combo)
        form.addRow("Example", example_row)
        layout.addWidget(metadata)
        layout.addWidget(self._build_plot_style_group())

        matrix_group = QGroupBox("Cij Matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        self.cij_table = CijMatrixTable()
        matrix_layout.addWidget(QLabel("Voigt order: [11, 22, 33, 23, 13, 12]"))
        matrix_layout.addWidget(self.cij_table, stretch=1)
        matrix_tools = QHBoxLayout()
        self.paste_matrix_button = self._action_button("Paste Matrix", QStyle.SP_DialogOpenButton)
        sym_button = self._action_button("Symmetrize", QStyle.SP_BrowserReload)
        self.paste_matrix_button.clicked.connect(self.paste_matrix_from_clipboard)
        sym_button.clicked.connect(self.symmetrize_table)
        matrix_tools.addWidget(self.paste_matrix_button)
        matrix_tools.addWidget(sym_button)
        matrix_tools.addStretch()
        matrix_layout.addLayout(matrix_tools)
        layout.addWidget(matrix_group, stretch=1)

        layout.addWidget(self._build_actions_group())
        return panel

    def _build_workflow_header(self) -> QWidget:
        group = QGroupBox("Workflow Status")
        layout = QHBoxLayout(group)
        self.workflow_status_label = QLabel("Ready")
        self.workflow_status_label.setObjectName("statusBanner")
        self.material_status_chip = QLabel("Material: Untitled")
        self.crystal_status_chip = QLabel("Crystal: -")
        self.unit_status_chip = QLabel("Unit: GPa")
        self.pyvista_status_chip = QLabel(self._pyvista_status_text())
        self.last_analysis_label = QLabel("Last analysis: not run")
        for chip in (
            self.material_status_chip,
            self.crystal_status_chip,
            self.unit_status_chip,
            self.pyvista_status_chip,
            self.last_analysis_label,
        ):
            chip.setObjectName("statusChip")
        layout.addWidget(self.workflow_status_label, stretch=2)
        layout.addWidget(self.material_status_chip)
        layout.addWidget(self.crystal_status_chip)
        layout.addWidget(self.unit_status_chip)
        layout.addWidget(self.pyvista_status_chip)
        layout.addWidget(self.last_analysis_label)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Actions")
        grid = QGridLayout(group)
        self.analyze_workflow_button = self._action_button("Analyze + Update Figures", QStyle.SP_MediaPlay)
        import_csv_button = self._action_button("Import CSV", QStyle.SP_DialogOpenButton)
        import_excel_button = self._action_button("Import Excel", QStyle.SP_DialogOpenButton)
        import_json_button = self._action_button("Import JSON", QStyle.SP_DialogOpenButton)
        export_results_button = self._action_button("Save Summary", QStyle.SP_DialogSaveButton)
        copy_button = self._action_button("Copy Summary", QStyle.SP_FileDialogDetailedView)
        self.analyze_workflow_button.setObjectName("primaryButton")
        self.analyze_workflow_button.clicked.connect(self.analyze_current_matrix)
        import_csv_button.clicked.connect(self.import_csv)
        import_excel_button.clicked.connect(self.import_excel)
        import_json_button.clicked.connect(self.import_json)
        export_results_button.clicked.connect(self.save_summary)
        copy_button.clicked.connect(self.copy_summary)
        grid.addWidget(self.analyze_workflow_button, 0, 0, 1, 2)
        for index, button in enumerate(
            [import_csv_button, import_excel_button, import_json_button, export_results_button, copy_button],
            start=2,
        ):
            grid.addWidget(button, index // 2, index % 2)
        return group

    def _build_plot_style_group(self) -> QWidget:
        group = QGroupBox("Publication Plot Style")
        form = QFormLayout(group)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list_theme_names())
        theme_index = self.theme_combo.findText(DEFAULT_THEME_NAME)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list_palette_names())
        palette_index = self.palette_combo.findText("Nature White")
        if palette_index >= 0:
            self.palette_combo.setCurrentIndex(palette_index)
        self.export_dpi_spin = QSpinBox()
        self.export_dpi_spin.setRange(72, 1200)
        self.export_dpi_spin.setValue(get_theme(DEFAULT_THEME_NAME).export_dpi)
        self.transparent_background_checkbox = QCheckBox("Transparent")
        self.transparent_background_checkbox.setChecked(False)
        form.addRow("Theme", self.theme_combo)
        form.addRow("Palette", self.palette_combo)
        form.addRow("Export dpi", self.export_dpi_spin)
        form.addRow("Background", self.transparent_background_checkbox)
        return group

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_summary_tab(), "Results")
        self.tabs.addTab(self._build_1d_tab(), "1D")
        self.tabs.addTab(self._build_2d_tab(), "2D")
        self.tabs.addTab(self._build_3d_tab(), "3D")
        return self.tabs

    def _build_dashboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.dashboard_stability_banner = QLabel("No analysis yet")
        self.dashboard_stability_banner.setObjectName("dashboardBanner")
        self.figure_status_label = QLabel("Figures: not generated")
        self.figure_status_label.setObjectName("mutedLabel")
        layout.addWidget(self.dashboard_stability_banner)

        metrics_group = QGroupBox("Key Elastic Parameters")
        metrics_grid = QGridLayout(metrics_group)
        metric_specs = [
            ("B_H", "Bulk Hill"),
            ("G_H", "Shear Hill"),
            ("E_H", "Young Hill"),
            ("A_U", "Universal anisotropy"),
        ]
        self.metric_labels: dict[str, QLabel] = {}
        for index, (key, label) in enumerate(metric_specs):
            title = QLabel(label)
            title.setObjectName("metricTitle")
            value = QLabel("-")
            value.setObjectName("metricValue")
            self.metric_labels[key] = value
            metrics_grid.addWidget(title, 0, index)
            metrics_grid.addWidget(value, 1, index)
        layout.addWidget(metrics_group)

        anisotropy_group = QGroupBox("Anisotropy and Figure State")
        anisotropy_layout = QVBoxLayout(anisotropy_group)
        self.anisotropy_summary_label = QLabel("Run analysis to inspect anisotropy and figure generation status.")
        self.anisotropy_summary_label.setWordWrap(True)
        anisotropy_layout.addWidget(self.anisotropy_summary_label)
        anisotropy_layout.addWidget(self.figure_status_label)
        layout.addWidget(anisotropy_group)

        exports_group = QGroupBox("Recommended Export")
        exports_layout = QHBoxLayout(exports_group)
        self.export_paper_figures_button = self._action_button("Export Paper Figures", QStyle.SP_DialogSaveButton)
        self.export_full_package_button = self._action_button("Export Full Package", QStyle.SP_DriveHDIcon)
        self.export_paper_figures_button.setObjectName("primaryButton")
        self.export_paper_figures_button.clicked.connect(self.export_paper_figures)
        self.export_full_package_button.clicked.connect(self.export_package)
        exports_layout.addWidget(self.export_paper_figures_button)
        exports_layout.addWidget(self.export_full_package_button)
        exports_layout.addStretch()
        layout.addWidget(exports_group)
        layout.addStretch()
        return tab

    def _action_button(self, text: str, icon: QStyle.StandardPixmap | None = None) -> QPushButton:
        button = QPushButton(text)
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        return button

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.stability_text = QTextEdit()
        self.stability_text.setReadOnly(True)
        self.stability_text.setMinimumHeight(150)
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.model_table = QTableWidget(0, 8)
        self.model_table.setHorizontalHeaderLabels(["Model", "Assumption", "B", "G", "E", "nu", "B/G", "Notes"])
        self.model_table.setAlternatingRowColors(True)
        self.export_model_table_button = self._action_button("Export Model Table", QStyle.SP_DialogSaveButton)
        self.export_model_table_button.clicked.connect(self.export_model_table)
        layout.addWidget(QLabel("Stability and warnings"))
        layout.addWidget(self.stability_text)
        layout.addWidget(QLabel("Scalar elastic parameters"))
        layout.addWidget(self.summary_table, stretch=1)
        layout.addWidget(QLabel("Elastic Model Comparison"))
        layout.addWidget(self.model_table, stretch=1)
        layout.addWidget(self.export_model_table_button)
        return tab

    def _property_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(PROPERTY_CHOICES.keys())
        return combo

    def _build_1d_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.line_property_combo = self._property_combo()
        self.line_mode_combo = QComboBox()
        self.line_mode_combo.addItems(["Angle in plane", "High-symmetry path", "Custom path"])
        self.line_plane_combo = QComboBox()
        self.line_plane_combo.addItems(["xy", "xz", "yz"])
        self.path_edit = QLineEdit("100; 110; 111; 100")
        line_button = QPushButton("Plot")
        save_button = QPushButton("Save Figure")
        save_data_button = QPushButton("Save Data")
        line_button.clicked.connect(self.update_line_plot)
        save_button.clicked.connect(lambda: self.save_current_figure(self.line_pane))
        save_data_button.clicked.connect(lambda: self.save_sampled_data("line"))
        controls.addWidget(QLabel("Property"))
        controls.addWidget(self.line_property_combo)
        controls.addWidget(QLabel("Mode"))
        controls.addWidget(self.line_mode_combo)
        controls.addWidget(QLabel("Plane"))
        controls.addWidget(self.line_plane_combo)
        controls.addWidget(QLabel("Path"))
        controls.addWidget(self.path_edit)
        controls.addWidget(line_button)
        controls.addWidget(save_button)
        controls.addWidget(save_data_button)
        controls.addStretch()
        self.line_pane = FigurePane()
        layout.addLayout(controls)
        layout.addWidget(self.line_pane, stretch=1)
        return tab

    def _build_2d_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.polar_property_combo = self._property_combo()
        self.polar_plane_combo = QComboBox()
        self.polar_plane_combo.addItems(["xy", "xz", "yz", "custom normal"])
        self.normal_edit = QLineEdit("0, 0, 1")
        polar_button = QPushButton("Plot")
        save_button = QPushButton("Save Figure")
        save_data_button = QPushButton("Save Data")
        polar_button.clicked.connect(self.update_polar_plot)
        save_button.clicked.connect(lambda: self.save_current_figure(self.polar_pane))
        save_data_button.clicked.connect(lambda: self.save_sampled_data("polar"))
        controls.addWidget(QLabel("Property"))
        controls.addWidget(self.polar_property_combo)
        controls.addWidget(QLabel("Plane"))
        controls.addWidget(self.polar_plane_combo)
        controls.addWidget(QLabel("Normal/Miller"))
        controls.addWidget(self.normal_edit)
        controls.addWidget(polar_button)
        controls.addWidget(save_button)
        controls.addWidget(save_data_button)
        controls.addStretch()
        self.polar_pane = FigurePane()
        layout.addLayout(controls)
        layout.addWidget(self.polar_pane, stretch=1)
        return tab

    def _build_3d_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.surface_property_combo = self._property_combo()
        self.cmap_combo = QComboBox()
        surface_palettes = list_3d_palette_names()
        self.cmap_combo.addItems(surface_palettes + [name for name in list_palette_names() if name not in surface_palettes])
        cmap_index = self.cmap_combo.findText(DEFAULT_3D_PALETTE_NAME)
        if cmap_index >= 0:
            self.cmap_combo.setCurrentIndex(cmap_index)
        self.theta_spin = QSpinBox()
        self.theta_spin.setRange(7, 121)
        self.theta_spin.setValue(31)
        self.phi_spin = QSpinBox()
        self.phi_spin.setRange(13, 241)
        self.phi_spin.setValue(61)
        self.gif_frames_spin = QSpinBox()
        self.gif_frames_spin.setRange(5, 240)
        self.gif_frames_spin.setValue(72)
        self.gif_dpi_spin = QSpinBox()
        self.gif_dpi_spin.setRange(72, 600)
        self.gif_dpi_spin.setValue(300)
        self.rotation_axis_combo = QComboBox()
        self.rotation_axis_combo.addItems(["z", "x", "y"])
        self.lighting_spin = QSpinBox()
        self.lighting_spin.setRange(0, 200)
        self.lighting_spin.setSuffix("%")
        self.lighting_spin.setValue(100)
        self.surface_smoothing_spin = QSpinBox()
        self.surface_smoothing_spin.setRange(0, 100)
        self.surface_smoothing_spin.setSuffix("%")
        self.surface_smoothing_spin.setValue(0)
        self.show_edges_checkbox = QCheckBox("Edges")
        self.render3d_status_label = QLabel()
        self._refresh_render3d_status_label()
        surface_button = QPushButton("Plot")
        gif_button = QPushButton("Export GIF")
        mp4_button = QPushButton("Export MP4")
        save_button = QPushButton("Save Figure")
        save_data_button = QPushButton("Save Data")
        surface_button.clicked.connect(self.update_surface_plot)
        gif_button.clicked.connect(self.export_gif)
        mp4_button.clicked.connect(self.export_mp4)
        save_button.clicked.connect(self.save_surface_figure)
        save_data_button.clicked.connect(lambda: self.save_sampled_data("surface"))
        controls.addWidget(QLabel("Property"))
        controls.addWidget(self.surface_property_combo)
        controls.addWidget(QLabel("Palette"))
        controls.addWidget(self.cmap_combo)
        controls.addWidget(QLabel("theta"))
        controls.addWidget(self.theta_spin)
        controls.addWidget(QLabel("phi"))
        controls.addWidget(self.phi_spin)
        controls.addWidget(surface_button)
        controls.addWidget(save_button)
        controls.addWidget(save_data_button)
        controls.addWidget(QLabel("GIF frames"))
        controls.addWidget(self.gif_frames_spin)
        controls.addWidget(QLabel("dpi"))
        controls.addWidget(self.gif_dpi_spin)
        controls.addWidget(QLabel("axis"))
        controls.addWidget(self.rotation_axis_combo)
        controls.addWidget(QLabel("Light"))
        controls.addWidget(self.lighting_spin)
        controls.addWidget(QLabel("Smooth"))
        controls.addWidget(self.surface_smoothing_spin)
        controls.addWidget(self.show_edges_checkbox)
        controls.addWidget(gif_button)
        controls.addWidget(mp4_button)
        controls.addStretch()
        self.surface_pane = FigurePane()
        layout.addLayout(controls)
        layout.addWidget(self.render3d_status_label)
        layout.addWidget(self.surface_pane, stretch=1)
        return tab

    def _refresh_render3d_status_label(self) -> None:
        status = pyvista_status()
        if status.available:
            self.render3d_status_label.setText("3D backend: PyVista/VTK static preview and export")
        else:
            self.render3d_status_label.setText(f"3D backend: Matplotlib fallback ({status.message})")

    def _pyvista_status_text(self) -> str:
        status = pyvista_status()
        backend = "PyVista" if status.available else "Matplotlib fallback"
        return f"3D backend: {backend}"

    def _update_workflow_status(self, message: str) -> None:
        if not hasattr(self, "workflow_status_label"):
            return
        self.workflow_status_label.setText(message)
        self.material_status_chip.setText(f"Material: {self.material_edit.text().strip() or 'Untitled'}")
        self.crystal_status_chip.setText(f"Crystal: {self.system_combo.currentText()}")
        self.unit_status_chip.setText(f"Unit: {self.unit_edit.text().strip() or 'GPa'}")
        self.pyvista_status_chip.setText(self._pyvista_status_text())
        if self.last_analysis_time is None:
            self.last_analysis_label.setText("Last analysis: not run")
        else:
            self.last_analysis_label.setText(f"Last analysis: {self.last_analysis_time:%Y-%m-%d %H:%M:%S}")

    def _apply_app_style(self) -> None:
        self.setStyleSheet(GUI_STYLE_SHEET)

    def _selected_property(self, combo: QComboBox) -> str:
        return PROPERTY_CHOICES[combo.currentText()]

    def _selected_plane(self, combo: QComboBox) -> str | np.ndarray:
        plane = combo.currentText()
        if plane != "custom normal":
            return plane
        raw = self.normal_edit.text().replace(",", " ").split()
        if len(raw) != 3:
            raise ValueError("Custom normal/Miller index must contain three numbers.")
        return np.array([float(value) for value in raw], dtype=float)

    def read_matrix(self) -> np.ndarray:
        return self.cij_table.matrix()

    def populate_matrix(self, matrix: np.ndarray) -> None:
        self.cij_table.set_matrix(matrix)

    def symmetrize_table(self) -> None:
        try:
            self.populate_matrix(0.5 * (self.read_matrix() + self.read_matrix().T))
        except Exception as exc:
            self.show_error("Cannot symmetrize matrix", exc)

    def load_example_by_name(self, name: str) -> None:
        example = EXAMPLE_MATERIALS[name]
        example_index = self.example_combo.findText(name)
        if example_index >= 0:
            self.example_combo.setCurrentIndex(example_index)
        self.material_edit.setText(example.name)
        self.unit_edit.setText(example.unit)
        index = self.system_combo.findText(example.crystal_system)
        if index >= 0:
            self.system_combo.setCurrentIndex(index)
        self.populate_matrix(example.matrix)
        self._update_workflow_status("Ready for analysis")

    def analyze_current_matrix(self) -> None:
        try:
            matrix = self.read_matrix()
            tensor = ElasticTensor(
                matrix,
                crystal_system=self.system_combo.currentText(),
                unit=self.unit_edit.text().strip() or "GPa",
                material_name=self.material_edit.text().strip() or "Untitled",
            )
            stability = check_stability(matrix, crystal_system=self.system_combo.currentText())
            summary = tensor.polycrystalline_summary()
            model_results = elastic_model_results(summary)
        except Exception as exc:
            self._update_workflow_status("Analysis failed")
            if hasattr(self, "dashboard_stability_banner"):
                self.dashboard_stability_banner.setText(f"Analysis failed: {exc}")
            self.show_error("Analysis failed", exc)
            return

        self.current_tensor = tensor
        self.current_stability = stability
        self.current_summary = summary
        self.current_model_results = model_results
        self.fill_stability(stability)
        self.fill_summary(summary.as_dict())
        self.fill_model_table(model_results)
        self.update_line_plot()
        self.update_polar_plot()
        self.update_surface_plot()
        self.last_analysis_time = datetime.now()
        self._update_dashboard_after_analysis()
        self._update_workflow_status("Stable: analysis complete" if stability.overall_stable else "Warning: check stability")

    def fill_stability(self, stability: StabilityResult) -> None:
        payload = stability.as_dict()
        lines = [f"{key}: {value}" for key, value in payload.items()]
        self.stability_text.setPlainText("\n".join(lines))

    def fill_summary(self, values: dict[str, float | None]) -> None:
        self.summary_table.setRowCount(len(values))
        for row, (key, value) in enumerate(values.items()):
            key_item = QTableWidgetItem(key)
            if value is None:
                value_text = "not applicable"
            elif isinstance(value, float):
                value_text = f"{value:.8g}"
            else:
                value_text = str(value)
            value_item = QTableWidgetItem(value_text)
            value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.summary_table.setItem(row, 0, key_item)
            self.summary_table.setItem(row, 1, value_item)
        self.summary_table.resizeColumnsToContents()

    def fill_model_table(self, results: list[ElasticModelResult]) -> None:
        frame = elastic_model_table_frame(results)
        self.model_table.setRowCount(len(frame))
        for row, record in frame.iterrows():
            for col, column in enumerate(frame.columns):
                value = record[column]
                if pd.isna(value):
                    value_text = "not applicable"
                elif isinstance(value, float):
                    value_text = f"{value:.8g}"
                else:
                    value_text = str(value)
                item = QTableWidgetItem(value_text)
                if column not in {"Model", "Assumption", "Notes"}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.model_table.setItem(row, col, item)
        self.model_table.resizeColumnsToContents()

    def _update_dashboard_after_analysis(self) -> None:
        if self.current_summary is None or self.current_stability is None:
            return
        summary = self.current_summary.as_dict()
        generated = [
            self.current_line_data is not None,
            self.current_polar_data is not None,
            self.current_surface is not None,
        ]
        dashboard = build_dashboard_state(
            summary,
            self.current_stability,
            figures_generated=all(generated),
        )
        self.dashboard_stability_banner.setText(dashboard.status_text)
        for key, value in dashboard.metrics.items():
            self.metric_labels[key].setText(value)
        self.anisotropy_summary_label.setText(dashboard.anisotropy_text)
        self.figure_status_label.setText(dashboard.figure_text)

    def _parse_clipboard_matrix(self, text: str) -> np.ndarray:
        return parse_clipboard_matrix(text)

    def paste_matrix_from_clipboard(self) -> None:
        try:
            text = QGuiApplication.clipboard().text()
            if not text.strip():
                raise ValueError("Clipboard is empty.")
            self.populate_matrix(self._parse_clipboard_matrix(text))
            self._update_workflow_status("Matrix pasted; ready for analysis")
        except Exception as exc:
            self.show_error("Paste matrix failed", exc)

    def require_tensor(self) -> ElasticTensor:
        if self.current_tensor is None:
            self.analyze_current_matrix()
        if self.current_tensor is None:
            raise RuntimeError("No valid tensor is available.")
        return self.current_tensor

    def update_line_plot(self) -> None:
        try:
            tensor = self.require_tensor()
            property_name = self._selected_property(self.line_property_combo)
            mode = self.line_mode_combo.currentText()
            if mode == "Angle in plane":
                plane = sample_plane(
                    tensor,
                    property_name=property_name,
                    plane=self.line_plane_combo.currentText(),
                    angle_count=361,
                )
                self.current_line_data = plane
                self.line_pane.set_figure(
                    plot_line_slice(
                        plane,
                        theme_name=self.theme_combo.currentText(),
                        palette_name=self.palette_combo.currentText(),
                    )
                )
            else:
                points = self._default_high_symmetry_path() if mode == "High-symmetry path" else self._parse_path()
                path = sample_direction_path(
                    tensor,
                    property_name=property_name,
                    points=points,
                    points_per_segment=51,
                )
                self.current_line_data = path
                self.line_pane.set_figure(
                    plot_direction_path(
                        path,
                        theme_name=self.theme_combo.currentText(),
                        palette_name=self.palette_combo.currentText(),
                    )
                )
        except Exception as exc:
            self.show_error("1D plot failed", exc)

    def update_polar_plot(self) -> None:
        try:
            tensor = self.require_tensor()
            plane = sample_plane(
                tensor,
                property_name=self._selected_property(self.polar_property_combo),
                plane=self._selected_plane(self.polar_plane_combo),
                angle_count=361,
            )
            self.current_polar_data = plane
            self.polar_pane.set_figure(
                plot_plane_slice(
                    plane,
                    theme_name=self.theme_combo.currentText(),
                    palette_name=self.palette_combo.currentText(),
                )
            )
        except Exception as exc:
            self.show_error("2D plot failed", exc)

    def update_surface_plot(self) -> None:
        try:
            tensor = self.require_tensor()
            surface = sample_sphere(
                tensor,
                property_name=self._selected_property(self.surface_property_combo),
                theta_count=self.theta_spin.value(),
                phi_count=self.phi_spin.value(),
            )
            self.current_surface = surface
            self._refresh_render3d_status_label()
            try:
                self.surface_pane.set_image(
                    render_surface_image(
                        surface,
                        options=self._render3d_options(window_size=(1400, 1100)),
                    )
                )
            except PyVistaUnavailableError:
                self.surface_pane.set_figure(
                    plot_directional_surface(
                        surface,
                        theme_name=self.theme_combo.currentText(),
                        palette_name=self.cmap_combo.currentText(),
                    )
                )
        except Exception as exc:
            self.show_error("3D plot failed", exc)

    def _default_high_symmetry_path(self):
        return [
            ("[100]", [1, 0, 0]),
            ("[110]", [1, 1, 0]),
            ("[111]", [1, 1, 1]),
            ("[100]", [1, 0, 0]),
        ]

    def _parse_path(self):
        points = []
        for chunk in self.path_edit.text().split(";"):
            raw = chunk.strip()
            if not raw:
                continue
            cleaned = raw.replace("[", "").replace("]", "").replace(",", " ")
            parts = cleaned.split()
            if len(parts) == 1 and len(parts[0]) == 3 and parts[0].lstrip("-").isdigit():
                vector = [float(char) for char in parts[0]]
            elif len(parts) == 3:
                vector = [float(value) for value in parts]
            else:
                raise ValueError("Path entries must look like 100, [1 1 0], or 1,1,1.")
            points.append((f"[{raw.strip('[]')}]", vector))
        if len(points) < 2:
            raise ValueError("Custom direction path needs at least two points separated by semicolons.")
        return points

    def import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Cij CSV", "", "CSV (*.csv);;All files (*)")
        if path:
            self._load_matrix_from_frame(pd.read_csv(path, header=None))

    def import_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Cij Excel", "", "Excel (*.xlsx *.xls);;All files (*)"
        )
        if path:
            self._load_matrix_from_frame(pd.read_excel(path, header=None))

    def import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Cij JSON", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            matrix = payload.get("stiffness_matrix", payload.get("matrix")) if isinstance(payload, dict) else payload
            self.populate_matrix(np.asarray(matrix, dtype=float).reshape(6, 6))
            if isinstance(payload, dict):
                if "material_name" in payload:
                    self.material_edit.setText(str(payload["material_name"]))
                if "unit" in payload:
                    self.unit_edit.setText(str(payload["unit"]))
                if "crystal_system" in payload:
                    index = self.system_combo.findText(str(payload["crystal_system"]))
                    if index >= 0:
                        self.system_combo.setCurrentIndex(index)
        except Exception as exc:
            self.show_error("JSON import failed", exc)

    def _load_matrix_from_frame(self, frame: pd.DataFrame) -> None:
        try:
            self.populate_matrix(matrix_from_frame(frame))
        except Exception as exc:
            self.show_error("Matrix import failed", exc)

    def save_summary(self) -> None:
        if self.current_summary is None:
            self.analyze_current_matrix()
        if self.current_summary is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save scalar summary",
            "elastic_summary.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return
        frame = pd.DataFrame([self.current_summary.as_dict()])
        try:
            if selected_filter.startswith("Excel") or path.lower().endswith(".xlsx"):
                frame.to_excel(path, index=False)
            else:
                frame.to_csv(path, index=False)
        except Exception as exc:
            self.show_error("Save summary failed", exc)

    def copy_summary(self) -> None:
        if self.current_summary is None:
            self.analyze_current_matrix()
        if self.current_summary is None:
            return
        frame = pd.DataFrame([self.current_summary.as_dict()])
        QGuiApplication.clipboard().setText(frame.to_csv(index=False))

    def export_model_table(self) -> None:
        try:
            tensor = self.require_tensor()
            path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save elastic model table",
                "elastic_model_summary.csv",
                "CSV (*.csv);;Excel (*.xlsx)",
            )
            if not path:
                return
            if selected_filter.startswith("Excel") and not path.lower().endswith((".xlsx", ".xls")):
                path = f"{path}.xlsx"
            export_elastic_model_table(tensor, path)
            QMessageBox.information(self, "Model table exported", f"Model table written:\n{path}")
        except Exception as exc:
            self.show_error("Export model table failed", exc)

    def export_paper_figures(self) -> None:
        try:
            directory = QFileDialog.getExistingDirectory(self, "Choose paper figure export directory")
            if not directory:
                return
            exported = self.export_paper_figures_to_directory(directory)
            QMessageBox.information(
                self,
                "Paper figures exported",
                "Figures written:\n" + "\n".join(str(path) for path in exported.values()),
            )
        except Exception as exc:
            self.show_error("Export paper figures failed", exc)

    def export_paper_figures_to_directory(self, directory: str | Path) -> dict[str, Path]:
        tensor = self.require_tensor()
        if self.current_line_data is None:
            self.update_line_plot()
        if self.current_polar_data is None:
            self.update_polar_plot()
        if self.current_surface is None:
            self.update_surface_plot()
        if self.current_line_data is None or self.current_polar_data is None or self.current_surface is None:
            raise RuntimeError("1D, 2D, and 3D figures must be generated before export.")

        exported = export_paper_figures(
            tensor,
            line_data=self.current_line_data,
            polar_data=self.current_polar_data,
            surface=self.current_surface,
            output_dir=directory,
            options=PaperFigureExportOptions(
                dpi=self.export_dpi_spin.value(),
                theme_name=self.theme_combo.currentText(),
                palette_name=self.palette_combo.currentText(),
                surface_palette_name=self.cmap_combo.currentText(),
                transparent_background=self.transparent_background_checkbox.isChecked(),
                lighting_intensity=self.lighting_spin.value() / 100.0,
                surface_smoothing=self.surface_smoothing_spin.value() / 100.0,
                show_edges=self.show_edges_checkbox.isChecked(),
                render3d_options=self._render3d_options(),
            ),
        )
        self.figure_status_label.setText("Figures: paper PNG set exported")
        return exported

    def export_package(self) -> None:
        try:
            tensor = self.require_tensor()
            directory = QFileDialog.getExistingDirectory(self, "Choose export directory")
            if not directory:
                return
            manifest = export_analysis_package(
                tensor,
                directory,
                sphere_theta_count=self.theta_spin.value(),
                sphere_phi_count=self.phi_spin.value(),
            )
            QMessageBox.information(self, "Export complete", f"Manifest written:\n{manifest}")
        except Exception as exc:
            self.show_error("Export package failed", exc)

    def export_gif(self) -> None:
        try:
            if self.current_surface is None:
                self.update_surface_plot()
            if self.current_surface is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export rotating GIF", "elastic_surface.gif", "GIF (*.gif)")
            if not path:
                return
            export_surface_gif_animation(
                self.require_tensor(),
                self.current_surface,
                path,
                options=AnimationExportOptions(
                    frames=self.gif_frames_spin.value(),
                    dpi=self.gif_dpi_spin.value(),
                    theme_name=self.theme_combo.currentText(),
                    palette_name=self.cmap_combo.currentText(),
                    axis=self.rotation_axis_combo.currentText(),
                    transparent_background=self.transparent_background_checkbox.isChecked(),
                    lighting_intensity=self.lighting_spin.value() / 100.0,
                    surface_smoothing=self.surface_smoothing_spin.value() / 100.0,
                    show_edges=self.show_edges_checkbox.isChecked(),
                ),
            )
            QMessageBox.information(self, "GIF export complete", f"Animation written:\n{path}")
        except Exception as exc:
            self.show_error("GIF export failed", exc)

    def export_mp4(self) -> None:
        try:
            if self.current_surface is None:
                self.update_surface_plot()
            if self.current_surface is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export rotating MP4", "elastic_surface.mp4", "MP4 (*.mp4)")
            if not path:
                return
            export_surface_mp4_animation(
                self.require_tensor(),
                self.current_surface,
                path,
                options=AnimationExportOptions(
                    frames=self.gif_frames_spin.value(),
                    dpi=self.gif_dpi_spin.value(),
                    theme_name=self.theme_combo.currentText(),
                    palette_name=self.cmap_combo.currentText(),
                    axis=self.rotation_axis_combo.currentText(),
                    lighting_intensity=self.lighting_spin.value() / 100.0,
                    surface_smoothing=self.surface_smoothing_spin.value() / 100.0,
                    show_edges=self.show_edges_checkbox.isChecked(),
                ),
            )
            QMessageBox.information(self, "MP4 export complete", f"Animation written:\n{path}")
        except Exception as exc:
            self.show_error("MP4 export failed", exc)

    def save_current_figure(self, pane: FigurePane) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            "elastic_plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            pane.save_figure(
                path,
                dpi=self.export_dpi_spin.value(),
                transparent=self.transparent_background_checkbox.isChecked(),
            )
            write_export_manifest(
                self.require_tensor(),
                path,
                export_type="figure",
                parameters={
                    "dpi": self.export_dpi_spin.value(),
                    "theme": self.theme_combo.currentText(),
                    "palette": self.palette_combo.currentText(),
                    "transparent_background": self.transparent_background_checkbox.isChecked(),
                },
            )
        except Exception as exc:
            self.show_error("Save figure failed", exc)

    def _render3d_options(self, *, window_size: tuple[int, int] | None = None) -> Render3DOptions:
        kwargs = {
            "theme_name": self.theme_combo.currentText(),
            "palette_name": self.cmap_combo.currentText(),
            "transparent_background": self.transparent_background_checkbox.isChecked(),
            "lighting_intensity": self.lighting_spin.value() / 100.0,
            "surface_smoothing": self.surface_smoothing_spin.value() / 100.0,
            "show_edges": self.show_edges_checkbox.isChecked(),
        }
        if window_size is not None:
            kwargs["window_size"] = window_size
        return Render3DOptions(**kwargs)

    def save_surface_figure(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D figure",
            "elastic_surface.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            if self.current_surface is None:
                self.update_surface_plot()
            if self.current_surface is None:
                return
            result = export_surface_figure(
                self.require_tensor(),
                self.current_surface,
                path,
                options=SurfaceFigureExportOptions(
                    dpi=self.export_dpi_spin.value(),
                    theme_name=self.theme_combo.currentText(),
                    palette_name=self.cmap_combo.currentText(),
                    transparent_background=self.transparent_background_checkbox.isChecked(),
                    lighting_intensity=self.lighting_spin.value() / 100.0,
                    surface_smoothing=self.surface_smoothing_spin.value() / 100.0,
                    show_edges=self.show_edges_checkbox.isChecked(),
                    render3d_options=self._render3d_options(),
                ),
            )
            if result.fallback_message:
                QMessageBox.warning(
                    self,
                    "PyVista unavailable",
                    f"{result.fallback_message}\nFalling back to Matplotlib 3D.",
                )
        except Exception as exc:
            self.show_error("Save 3D figure failed", exc)

    def save_sampled_data(self, kind: str) -> None:
        try:
            tensor = self.require_tensor()
            if kind == "line":
                data = self.current_line_data
                if data is None:
                    self.update_line_plot()
                    data = self.current_line_data
            elif kind == "polar":
                data = self.current_polar_data
                if data is None:
                    self.update_polar_plot()
                    data = self.current_polar_data
            else:
                data = self.current_surface
                if data is None:
                    self.update_surface_plot()
                    data = self.current_surface
            if data is None:
                return

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save sampled data",
                f"elastic_{kind}_data.csv",
                "CSV (*.csv)",
            )
            if not path:
                return
            export_sampled_data(tensor, data, path, kind=kind)
        except Exception as exc:
            self.show_error("Save sampled data failed", exc)

    def show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
