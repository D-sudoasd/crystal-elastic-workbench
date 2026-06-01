import json

from crystal_elastic_workbench.animation_export import (
    AnimationExportOptions,
    export_surface_gif_animation,
    export_surface_mp4_animation,
)
from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.sampling import sample_sphere


def _si_tensor_and_surface():
    example = EXAMPLE_MATERIALS["Si cubic"]
    tensor = ElasticTensor(
        example.matrix,
        crystal_system=example.crystal_system,
        unit=example.unit,
        material_name=example.name,
    )
    return tensor, sample_sphere(tensor, property_name="young", theta_count=5, phi_count=9)


def test_gif_animation_export_writes_manifest_and_forwards_scientific_style_options(tmp_path, monkeypatch):
    tensor, surface = _si_tensor_and_surface()
    seen = {}

    def fake_export(surface_arg, output_path, **kwargs):
        seen["surface"] = surface_arg
        seen["kwargs"] = kwargs
        output_path.write_bytes(b"gif bytes")
        return output_path

    monkeypatch.setattr("crystal_elastic_workbench.animation_export.export_rotating_gif", fake_export)

    output = export_surface_gif_animation(
        tensor,
        surface,
        tmp_path / "surface.gif",
        options=AnimationExportOptions(
            frames=11,
            dpi=150,
            theme_name="Nature White",
            palette_name="Viridis Refined",
            axis="x",
            transparent_background=True,
            lighting_intensity=0.8,
            surface_smoothing=0.25,
            surface_subdivision=2,
            show_edges=True,
            specular=0.4,
        ),
    )

    assert seen["surface"] is surface
    assert seen["kwargs"]["frames"] == 11
    assert seen["kwargs"]["dpi"] == 150
    assert seen["kwargs"]["axis"] == "x"
    assert seen["kwargs"]["transparent_background"] is True
    assert seen["kwargs"]["surface_subdivision"] == 2
    assert seen["kwargs"]["specular"] == 0.4
    assert output.read_bytes() == b"gif bytes"
    manifest = json.loads(output.with_name(f"{output.name}.manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_type"] == "gif"
    assert manifest["parameters"]["property"] == "young"
    assert manifest["parameters"]["palette"] == "Viridis Refined"
    assert manifest["parameters"]["palette_category"] == "sequential"
    assert manifest["parameters"]["compose_annotations"] is False
    assert manifest["parameters"]["annotation_backend"] == "pyvista"
    assert manifest["parameters"]["title_font_size"] == 18
    assert manifest["parameters"]["colorbar_tick_size"] == 10
    assert manifest["parameters"]["surface_subdivision"] == 2
    assert manifest["parameters"]["show_edges"] is True
    assert manifest["parameters"]["specular"] == 0.4


def test_mp4_animation_export_writes_manifest_without_requiring_ffmpeg(tmp_path, monkeypatch):
    tensor, surface = _si_tensor_and_surface()
    seen = {}

    def fake_export(surface_arg, output_path, **kwargs):
        seen["surface"] = surface_arg
        seen["kwargs"] = kwargs
        output_path.write_bytes(b"mp4 bytes")
        return output_path

    monkeypatch.setattr("crystal_elastic_workbench.animation_export.export_rotating_mp4", fake_export)

    output = export_surface_mp4_animation(
        tensor,
        surface,
        tmp_path / "surface.mp4",
        options=AnimationExportOptions(frames=13, dpi=180, theme_name="Nature White", axis="y"),
    )

    assert seen["surface"] is surface
    assert seen["kwargs"]["frames"] == 13
    assert seen["kwargs"]["dpi"] == 180
    assert seen["kwargs"]["axis"] == "y"
    assert output.read_bytes() == b"mp4 bytes"
    manifest = json.loads(output.with_name(f"{output.name}.manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_type"] == "mp4"
    assert manifest["parameters"]["property"] == "young"
    assert manifest["parameters"]["frames"] == 13
    assert manifest["parameters"]["palette"] == "Nature Surface"
    assert manifest["parameters"]["palette_category"] == "sequential"
    assert manifest["parameters"]["compose_annotations"] is False
    assert manifest["parameters"]["annotation_backend"] == "pyvista"
    assert manifest["parameters"]["title_font_size"] == 18
    assert manifest["parameters"]["colorbar_tick_size"] == 10
    assert manifest["parameters"]["surface_subdivision"] == 1
    assert manifest["parameters"]["specular"] == 0.32
