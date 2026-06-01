"""Batch paper-figure export services independent of the Qt GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.exporting import write_export_manifest
from crystal_elastic_workbench.plot_styles import DEFAULT_3D_PALETTE_NAME
from crystal_elastic_workbench.render3d import PyVistaUnavailableError, Render3DOptions, render3d_style_parameters
from crystal_elastic_workbench.sampling import DirectionPath, DirectionalSurface, PlaneSlice
from crystal_elastic_workbench.visualization import (
    plot_direction_path,
    plot_directional_surface,
    plot_line_slice,
    plot_plane_slice,
)


@dataclass(frozen=True)
class PaperFigureExportOptions:
    dpi: int = 300
    theme_name: str = "Nature White"
    palette_name: str = "Nature White"
    surface_palette_name: str = DEFAULT_3D_PALETTE_NAME
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


def _surface_render_options(options: PaperFigureExportOptions) -> Render3DOptions:
    if options.render3d_options is not None:
        return options.render3d_options
    return Render3DOptions(
        theme_name=options.theme_name,
        palette_name=options.surface_palette_name,
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


def _render_surface_png_isolated(
    surface: DirectionalSurface,
    output_path: Path,
    options: Render3DOptions,
) -> None:
    """Render with PyVista in a child process so VTK cannot poison Qt in this process."""

    with tempfile.TemporaryDirectory(prefix="cij_render3d_") as tmp:
        payload_path = Path(tmp) / "payload.pkl"
        with payload_path.open("wb") as handle:
            pickle.dump({"surface": surface, "options": options}, handle)
        completed = subprocess.run(
            [sys.executable, "-m", "crystal_elastic_workbench.render3d_worker", str(payload_path), str(output_path)],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "PyVista worker failed."
        raise PyVistaUnavailableError(message)


def export_paper_figures(
    tensor: ElasticTensor,
    *,
    line_data: DirectionPath | PlaneSlice,
    polar_data: PlaneSlice,
    surface: DirectionalSurface,
    output_dir: str | Path,
    options: PaperFigureExportOptions | None = None,
) -> dict[str, Path]:
    """Export the default 1D, 2D, and 3D paper PNG set with sidecar manifests."""

    opts = options or PaperFigureExportOptions()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    line_path = out / "paper_1d.png"
    if isinstance(line_data, DirectionPath):
        line_fig = plot_direction_path(line_data, theme_name=opts.theme_name, palette_name=opts.palette_name)
    else:
        line_fig = plot_line_slice(line_data, theme_name=opts.theme_name, palette_name=opts.palette_name)
    line_fig.savefig(line_path, dpi=opts.dpi, transparent=opts.transparent_background)
    plt.close(line_fig)
    write_export_manifest(
        tensor,
        line_path,
        export_type="paper_figure_1d",
        parameters={
            "dpi": opts.dpi,
            "theme": opts.theme_name,
            "palette": opts.palette_name,
            "transparent_background": opts.transparent_background,
            "property": line_data.property_name,
        },
    )
    exported["line_png"] = line_path

    polar_path = out / "paper_2d_polar.png"
    polar_fig = plot_plane_slice(polar_data, theme_name=opts.theme_name, palette_name=opts.palette_name)
    polar_fig.savefig(polar_path, dpi=opts.dpi, transparent=opts.transparent_background)
    plt.close(polar_fig)
    write_export_manifest(
        tensor,
        polar_path,
        export_type="paper_figure_2d",
        parameters={
            "dpi": opts.dpi,
            "theme": opts.theme_name,
            "palette": opts.palette_name,
            "transparent_background": opts.transparent_background,
            "property": polar_data.property_name,
        },
    )
    exported["polar_png"] = polar_path

    surface_path = out / "paper_3d_surface.png"
    backend = "pyvista"
    render_options = _surface_render_options(opts)
    try:
        _render_surface_png_isolated(surface, surface_path, render_options)
    except PyVistaUnavailableError:
        backend = "matplotlib"
        surface_fig = plot_directional_surface(
            surface,
            theme_name=render_options.theme_name,
            palette_name=render_options.palette_name,
        )
        surface_fig.savefig(surface_path, dpi=opts.dpi, transparent=opts.transparent_background)
        plt.close(surface_fig)
    write_export_manifest(
        tensor,
        surface_path,
        export_type="paper_figure_3d",
        parameters={
            "backend": backend,
            "dpi": opts.dpi,
            "theme": opts.theme_name,
            "palette": opts.surface_palette_name,
            "transparent_background": opts.transparent_background,
            **render3d_style_parameters(render_options),
            "property": surface.property_name,
        },
    )
    exported["surface_png"] = surface_path
    return exported
