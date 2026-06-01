import json
from pathlib import Path

from crystal_elastic_workbench import paper_export as paper_export_module
from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.paper_export import PaperFigureExportOptions, export_paper_figures
from crystal_elastic_workbench.render3d import PyVistaUnavailableError, Render3DOptions
from crystal_elastic_workbench.sampling import sample_direction_path, sample_plane, sample_sphere


def test_export_paper_figures_writes_traceable_png_set(tmp_path):
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    line_data = sample_direction_path(
        tensor,
        property_name="young",
        points=[
            ("[100]", [1, 0, 0]),
            ("[110]", [1, 1, 0]),
            ("[111]", [1, 1, 1]),
        ],
        points_per_segment=5,
    )
    polar_data = sample_plane(tensor, property_name="young", plane="xy", angle_count=37)
    surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)

    exported = export_paper_figures(
        tensor,
        line_data=line_data,
        polar_data=polar_data,
        surface=surface,
        output_dir=tmp_path,
        options=PaperFigureExportOptions(
            dpi=120,
            theme_name="Nature White",
            palette_name="Nature White",
            surface_palette_name="Viridis Refined",
            render3d_options=Render3DOptions(window_size=(360, 300)),
        ),
    )

    assert set(exported) == {"line_png", "polar_png", "surface_png"}
    for key, path in exported.items():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size > 1000
        manifest = json.loads(path.with_name(f"{path.name}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["material_name"] == "Si cubic"
        assert manifest["parameters"]["dpi"] == 120
        assert manifest["parameters"]["theme"] == "Nature White"
        if key == "surface_png":
            assert manifest["parameters"]["backend"] in {"pyvista", "matplotlib"}
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


def test_paper_surface_fallback_uses_render_options_palette_instead_of_stale_option(tmp_path, monkeypatch):
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    line_data = sample_direction_path(
        tensor,
        property_name="young",
        points=[("[100]", [1, 0, 0]), ("[110]", [1, 1, 0])],
        points_per_segment=3,
    )
    polar_data = sample_plane(tensor, property_name="young", plane="xy", angle_count=13)
    surface = sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)
    seen = {}
    original_plot = paper_export_module.plot_directional_surface

    def fail_pyvista(*args, **kwargs):
        raise PyVistaUnavailableError("forced unavailable")

    def capture_matplotlib_surface(surface_arg, **kwargs):
        seen["palette_name"] = kwargs.get("palette_name")
        return original_plot(surface_arg, **kwargs)

    monkeypatch.setattr("crystal_elastic_workbench.paper_export._render_surface_png_isolated", fail_pyvista)
    monkeypatch.setattr(
        "crystal_elastic_workbench.paper_export.plot_directional_surface",
        capture_matplotlib_surface,
    )

    exported = export_paper_figures(
        tensor,
        line_data=line_data,
        polar_data=polar_data,
        surface=surface,
        output_dir=tmp_path,
        options=PaperFigureExportOptions(
            surface_palette_name="Viridis Refined",
            render3d_options=Render3DOptions(window_size=(360, 300), palette_name="Blue-Gold"),
        ),
    )

    assert seen["palette_name"] == "Blue-Gold"
    manifest = json.loads(
        exported["surface_png"].with_name(f"{exported['surface_png'].name}.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["parameters"]["backend"] == "matplotlib"
    assert manifest["parameters"]["palette"] == "Blue-Gold"
