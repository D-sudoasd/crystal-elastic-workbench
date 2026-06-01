"""Single-figure export services independent of Qt dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.exporting import write_export_manifest
from crystal_elastic_workbench.plot_styles import DEFAULT_3D_PALETTE_NAME
from crystal_elastic_workbench.render3d import (
    PyVistaUnavailableError,
    Render3DOptions,
    render3d_style_parameters,
    render_surface_png,
)
from crystal_elastic_workbench.sampling import DirectionalSurface
from crystal_elastic_workbench.visualization import plot_directional_surface


@dataclass(frozen=True)
class SurfaceFigureExportOptions:
    dpi: int = 300
    theme_name: str = "Nature White"
    palette_name: str = DEFAULT_3D_PALETTE_NAME
    transparent_background: bool = False
    lighting_intensity: float = 1.0
    surface_smoothing: float = 0.0
    surface_subdivision: int = 1
    show_edges: bool = False
    ambient: float = 0.28
    diffuse: float = 0.74
    specular: float = 0.32
    specular_power: float = 28.0
    render3d_options: Render3DOptions | None = None


@dataclass(frozen=True)
class SurfaceFigureExportResult:
    path: Path
    manifest_path: Path
    backend: str
    fallback_message: str | None = None


def _render_options(options: SurfaceFigureExportOptions) -> Render3DOptions:
    if options.render3d_options is not None:
        return options.render3d_options
    return Render3DOptions(
        theme_name=options.theme_name,
        palette_name=options.palette_name,
        transparent_background=options.transparent_background,
        lighting_intensity=options.lighting_intensity,
        surface_smoothing=options.surface_smoothing,
        surface_subdivision=options.surface_subdivision,
        show_edges=options.show_edges,
        ambient=options.ambient,
        diffuse=options.diffuse,
        specular=options.specular,
        specular_power=options.specular_power,
    )


def _save_matplotlib_surface(
    surface: DirectionalSurface,
    output: Path,
    options: SurfaceFigureExportOptions,
    render_options: Render3DOptions,
) -> None:
    fig = plot_directional_surface(
        surface,
        theme_name=render_options.theme_name,
        palette_name=render_options.palette_name,
    )
    try:
        fig.savefig(output, dpi=options.dpi, transparent=options.transparent_background)
    finally:
        plt.close(fig)


def export_surface_figure(
    tensor: ElasticTensor,
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    options: SurfaceFigureExportOptions | None = None,
) -> SurfaceFigureExportResult:
    """Export a 3D surface figure and write its sidecar manifest."""

    opts = options or SurfaceFigureExportOptions()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_options = _render_options(opts)

    backend = "matplotlib"
    fallback_message: str | None = None
    if output.suffix.lower() == ".png":
        try:
            render_surface_png(surface, output, options=render_options)
            backend = "pyvista"
        except PyVistaUnavailableError as exc:
            fallback_message = str(exc)
            _save_matplotlib_surface(surface, output, opts, render_options)
    else:
        _save_matplotlib_surface(surface, output, opts, render_options)

    manifest_path = write_export_manifest(
        tensor,
        output,
        export_type="3d_figure",
        parameters={
            "backend": backend,
            "dpi": opts.dpi,
            "theme": opts.theme_name,
            "palette": opts.palette_name,
            "transparent_background": opts.transparent_background,
            **render3d_style_parameters(render_options),
            "property": surface.property_name,
        },
    )
    return SurfaceFigureExportResult(
        path=output,
        manifest_path=manifest_path,
        backend=backend,
        fallback_message=fallback_message,
    )
