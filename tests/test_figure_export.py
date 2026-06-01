import json

import matplotlib

matplotlib.use("Agg")

from crystal_elastic_workbench import figure_export as figure_export_module
from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.figure_export import SurfaceFigureExportOptions, export_surface_figure
from crystal_elastic_workbench.render3d import PyVistaUnavailableError, Render3DOptions
from crystal_elastic_workbench.sampling import sample_sphere


def test_surface_figure_export_falls_back_to_matplotlib_and_records_manifest(tmp_path, monkeypatch):
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)

    def fail_pyvista(*args, **kwargs):
        raise PyVistaUnavailableError("forced unavailable")

    seen = {}
    original_plot = figure_export_module.plot_directional_surface

    def capture_matplotlib_surface(surface_arg, **kwargs):
        seen["palette_name"] = kwargs.get("palette_name")
        return original_plot(surface_arg, **kwargs)

    monkeypatch.setattr("crystal_elastic_workbench.figure_export.render_surface_png", fail_pyvista)
    monkeypatch.setattr(
        "crystal_elastic_workbench.figure_export.plot_directional_surface",
        capture_matplotlib_surface,
    )

    result = export_surface_figure(
        tensor,
        surface,
        tmp_path / "surface.png",
        options=SurfaceFigureExportOptions(
            dpi=120,
            theme_name="Nature White",
            palette_name="Viridis Refined",
            transparent_background=True,
            render3d_options=Render3DOptions(window_size=(320, 260)),
        ),
    )

    assert result.backend == "matplotlib"
    assert result.fallback_message == "forced unavailable"
    assert seen["palette_name"] == "Nature Surface"
    assert result.path.stat().st_size > 1000
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["export_type"] == "3d_figure"
    assert manifest["parameters"]["backend"] == "matplotlib"
    assert manifest["parameters"]["dpi"] == 120
    assert manifest["parameters"]["transparent_background"] is True
    assert manifest["parameters"]["palette"] == "Nature Surface"
    assert manifest["parameters"]["palette_category"] == "sequential"
    assert manifest["parameters"]["compose_annotations"] is True
    assert manifest["parameters"]["annotation_backend"] == "matplotlib"
    assert manifest["parameters"]["title_font_size"] == 18
    assert manifest["parameters"]["colorbar_tick_size"] == 10
    assert manifest["parameters"]["surface_subdivision"] == 1
    assert manifest["parameters"]["ambient"] == 0.28
    assert manifest["parameters"]["diffuse"] == 0.74
    assert manifest["parameters"]["specular"] == 0.32
    assert manifest["parameters"]["specular_power"] == 28.0
    assert manifest["parameters"]["property"] == "young"
