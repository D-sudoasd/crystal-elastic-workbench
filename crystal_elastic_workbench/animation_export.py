"""Rotating 3D animation export services with standard manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.exporting import write_export_manifest
from crystal_elastic_workbench.plot_styles import DEFAULT_3D_PALETTE_NAME, get_palette
from crystal_elastic_workbench.sampling import DirectionalSurface
from crystal_elastic_workbench.visualization import export_rotating_gif, export_rotating_mp4


@dataclass(frozen=True)
class AnimationExportOptions:
    frames: int = 72
    dpi: int = 120
    theme_name: str = "Nature White"
    palette_name: str = DEFAULT_3D_PALETTE_NAME
    axis: str = "z"
    transparent_background: bool = False
    lighting_intensity: float = 1.0
    surface_smoothing: float = 0.0
    surface_subdivision: int = 1
    show_edges: bool = False
    title_font_size: int = 18
    label_font_size: int = 12
    colorbar_title_size: int = 12
    colorbar_tick_size: int = 10
    ambient: float = 0.28
    diffuse: float = 0.74
    specular: float = 0.32
    specular_power: float = 28.0


def _render_style_parameters(options: AnimationExportOptions) -> dict[str, object]:
    palette = get_palette(options.palette_name)
    return {
        "palette_category": palette.category,
        "compose_annotations": False,
        "annotation_backend": "pyvista",
        "title_font_size": options.title_font_size,
        "label_font_size": options.label_font_size,
        "colorbar_title_size": options.colorbar_title_size,
        "colorbar_tick_size": options.colorbar_tick_size,
        "lighting_intensity": options.lighting_intensity,
        "surface_smoothing": options.surface_smoothing,
        "surface_subdivision": options.surface_subdivision,
        "show_edges": options.show_edges,
        "ambient": options.ambient,
        "diffuse": options.diffuse,
        "specular": options.specular,
        "specular_power": options.specular_power,
    }


def _gif_manifest_parameters(surface: DirectionalSurface, options: AnimationExportOptions) -> dict[str, object]:
    return {
        "frames": options.frames,
        "dpi": options.dpi,
        "theme": options.theme_name,
        "palette": options.palette_name,
        "axis": options.axis,
        "property": surface.property_name,
        "transparent_background": options.transparent_background,
        **_render_style_parameters(options),
    }


def _mp4_manifest_parameters(surface: DirectionalSurface, options: AnimationExportOptions) -> dict[str, object]:
    return {
        "frames": options.frames,
        "dpi": options.dpi,
        "theme": options.theme_name,
        "palette": options.palette_name,
        "axis": options.axis,
        "property": surface.property_name,
        **_render_style_parameters(options),
    }


def export_surface_gif_animation(
    tensor: ElasticTensor,
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    options: AnimationExportOptions | None = None,
) -> Path:
    opts = options or AnimationExportOptions()
    output = Path(output_path)
    export_rotating_gif(
        surface,
        output,
        frames=opts.frames,
        dpi=opts.dpi,
        theme_name=opts.theme_name,
        palette_name=opts.palette_name,
        axis=opts.axis,
        transparent_background=opts.transparent_background,
        lighting_intensity=opts.lighting_intensity,
        surface_smoothing=opts.surface_smoothing,
        surface_subdivision=opts.surface_subdivision,
        show_edges=opts.show_edges,
        title_font_size=opts.title_font_size,
        label_font_size=opts.label_font_size,
        colorbar_title_size=opts.colorbar_title_size,
        colorbar_tick_size=opts.colorbar_tick_size,
        ambient=opts.ambient,
        diffuse=opts.diffuse,
        specular=opts.specular,
        specular_power=opts.specular_power,
    )
    write_export_manifest(
        tensor,
        output,
        export_type="gif",
        parameters=_gif_manifest_parameters(surface, opts),
    )
    return output


def export_surface_mp4_animation(
    tensor: ElasticTensor,
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    options: AnimationExportOptions | None = None,
) -> Path:
    opts = options or AnimationExportOptions()
    output = Path(output_path)
    export_rotating_mp4(
        surface,
        output,
        frames=opts.frames,
        dpi=opts.dpi,
        theme_name=opts.theme_name,
        palette_name=opts.palette_name,
        axis=opts.axis,
        lighting_intensity=opts.lighting_intensity,
        surface_smoothing=opts.surface_smoothing,
        surface_subdivision=opts.surface_subdivision,
        show_edges=opts.show_edges,
        title_font_size=opts.title_font_size,
        label_font_size=opts.label_font_size,
        colorbar_title_size=opts.colorbar_title_size,
        colorbar_tick_size=opts.colorbar_tick_size,
        ambient=opts.ambient,
        diffuse=opts.diffuse,
        specular=opts.specular,
        specular_power=opts.specular_power,
    )
    write_export_manifest(
        tensor,
        output,
        export_type="mp4",
        parameters=_mp4_manifest_parameters(surface, opts),
    )
    return output
