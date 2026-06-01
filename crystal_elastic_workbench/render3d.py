"""PyVista/VTK rendering helpers for high-quality 3D elastic surfaces."""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

from crystal_elastic_workbench.plot_styles import (
    DEFAULT_3D_PALETTE_NAME,
    DEFAULT_THEME_NAME,
    get_palette,
    get_theme,
    matplotlib_rc_params,
    palette_colormap,
)
from crystal_elastic_workbench.sampling import DirectionalSurface


_PROPERTY_LABELS = {
    "young": "E(n) [GPa]",
    "compressibility": "beta(n) [1/GPa]",
    "shear": "G(n,m) [GPa]",
    "poisson": "nu(n,m)",
}

_PROPERTY_TITLES = {
    "young": "E(n)",
    "compressibility": "beta(n)",
    "shear": "G(n,m)",
    "poisson": "nu(n,m)",
}


class PyVistaUnavailableError(RuntimeError):
    """Raised when the PyVista/VTK backend cannot be initialized."""


@dataclass(frozen=True)
class PyVistaStatus:
    available: bool
    message: str


@dataclass(frozen=True)
class Render3DOptions:
    theme_name: str = DEFAULT_THEME_NAME
    palette_name: str = DEFAULT_3D_PALETTE_NAME
    window_size: tuple[int, int] = (2400, 2000)
    transparent_background: bool = False
    parallel_projection: bool = True
    smooth_shading: bool = True
    show_edges: bool = False
    compose_annotations: bool = True
    title_font_size: int = 18
    label_font_size: int = 12
    colorbar_title_size: int = 12
    colorbar_tick_size: int = 10
    lighting_intensity: float = 1.0
    surface_smoothing: float = 0.0
    surface_subdivision: int = 1
    ambient: float = 0.28
    diffuse: float = 0.74
    specular: float = 0.32
    specular_power: float = 28.0
    background_color: str | None = None


def render3d_style_parameters(options: Render3DOptions) -> dict[str, object]:
    palette = get_palette(options.palette_name)
    return {
        "palette": options.palette_name,
        "palette_category": palette.category,
        "compose_annotations": options.compose_annotations,
        "annotation_backend": "matplotlib" if options.compose_annotations else "pyvista",
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


def pyvista_status() -> PyVistaStatus:
    if importlib.util.find_spec("pyvista") is None:
        return PyVistaStatus(False, "PyVista/VTK backend is unavailable: pyvista is not installed.")
    if importlib.util.find_spec("vtk") is None:
        return PyVistaStatus(False, "PyVista/VTK backend is unavailable: vtk is not installed.")
    return PyVistaStatus(True, "PyVista/VTK backend is available.")


def _import_pyvista():
    status = pyvista_status()
    if not status.available:
        raise PyVistaUnavailableError(
            f"{status.message} Install pyvista and vtk, then restart the application."
        )
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - depends on local VTK runtime
        raise PyVistaUnavailableError(
            "PyVista/VTK backend could not be initialized. "
            "The GUI will fall back to Matplotlib 3D for preview/export."
        ) from exc
    return pv


def _surface_mesh(pv, surface: DirectionalSurface, options: Render3DOptions):
    x = surface.x
    y = surface.y
    z = surface.z
    values = surface.values
    if x.shape[1] > 1 and np.allclose(x[:, 0], x[:, -1]) and np.allclose(y[:, 0], y[:, -1]):
        x = x[:, :-1]
        y = y[:, :-1]
        z = z[:, :-1]
        values = values[:, :-1]
    theta_count, phi_count = x.shape
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces: list[int] = []
    for theta_index in range(theta_count - 1):
        row = theta_index * phi_count
        next_row = (theta_index + 1) * phi_count
        for phi_index in range(phi_count):
            faces.extend(
                [
                    4,
                    row + phi_index,
                    row + (phi_index + 1) % phi_count,
                    next_row + (phi_index + 1) % phi_count,
                    next_row + phi_index,
                ]
            )
    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    mesh.point_data["value"] = values.ravel()
    subdivision = max(0, min(int(options.surface_subdivision), 4))
    if subdivision > 0:
        try:
            mesh = mesh.triangulate().subdivide(subdivision, subfilter="linear")
        except Exception:  # pragma: no cover - VTK subdivision support varies
            pass
    smoothing = max(0.0, min(float(options.surface_smoothing), 1.0))
    if smoothing > 0:
        mesh = mesh.smooth(n_iter=max(1, int(60 * smoothing)), relaxation_factor=0.015)
    try:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
        )
    except Exception:  # pragma: no cover - depends on VTK build
        pass
    return mesh


def _add_three_point_lighting(pv, plotter, options: Render3DOptions) -> None:
    try:
        plotter.remove_all_lights()
        for position, scale in (
            ((2.5, -3.5, 3.0), 0.90),
            ((-3.0, 2.2, 1.8), 0.45),
            ((0.0, 3.5, 4.0), 0.35),
        ):
            light = pv.Light(
                position=position,
                focal_point=(0.0, 0.0, 0.0),
                color="white",
                intensity=max(0.0, options.lighting_intensity) * scale,
            )
            plotter.add_light(light)
    except Exception:  # pragma: no cover - PyVista light APIs vary by VTK build
        plotter.enable_lightkit()


def _camera_for(surface: DirectionalSurface, *, azimuth_deg: float, elevation_deg: float, axis: str = "z"):
    coords = np.column_stack([surface.x.ravel(), surface.y.ravel(), surface.z.ravel()])
    center = coords.mean(axis=0)
    radius = max(float(np.linalg.norm(coords - center, axis=1).max()), 1.0) * 3.0
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = radius * math.cos(elevation)
    vertical = radius * math.sin(elevation)
    key = axis.lower()
    if key == "x":
        position = center + np.array([vertical, horizontal * math.cos(azimuth), horizontal * math.sin(azimuth)])
        view_up = (0.0, 0.0, 1.0)
    elif key == "y":
        position = center + np.array([horizontal * math.cos(azimuth), vertical, horizontal * math.sin(azimuth)])
        view_up = (0.0, 0.0, 1.0)
    else:
        position = center + np.array([horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), vertical])
        view_up = (0.0, 0.0, 1.0)
    return tuple(position), tuple(center), view_up


def _pyvista_scalar_bar_args(surface: DirectionalSurface, options: Render3DOptions, theme) -> dict[str, object]:
    return {
        "title": _PROPERTY_LABELS.get(surface.property_name, surface.property_name),
        "title_font_size": options.colorbar_title_size,
        "label_font_size": options.colorbar_tick_size,
        "color": theme.text_color,
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.20,
        "width": 0.08,
        "height": 0.62,
    }


def _build_plotter(
    surface: DirectionalSurface,
    options: Render3DOptions,
    *,
    azimuth: float = 35.0,
    elevation: float = 25.0,
):
    pv = _import_pyvista()
    theme = get_theme(options.theme_name)
    background = options.background_color or theme.figure_facecolor
    plotter = pv.Plotter(off_screen=True, window_size=options.window_size)
    plotter.set_background(background)
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:  # pragma: no cover - depends on VTK backend
        pass
    if options.parallel_projection:
        plotter.enable_parallel_projection()
    _add_three_point_lighting(pv, plotter, options)
    mesh = _surface_mesh(pv, surface, options)
    mesh_kwargs = {
        "scalars": "value",
        "cmap": palette_colormap(options.palette_name),
        "smooth_shading": options.smooth_shading,
        "show_edges": options.show_edges,
        "edge_color": "#404040",
        "line_width": 0.4,
        "ambient": max(0.0, min(float(options.ambient), 1.0)),
        "diffuse": max(0.0, min(float(options.diffuse), 1.0)),
        "specular": max(0.0, min(float(options.specular), 1.0)),
        "specular_power": max(1.0, float(options.specular_power)),
        "show_scalar_bar": not options.compose_annotations,
    }
    if not options.compose_annotations:
        mesh_kwargs["scalar_bar_args"] = _pyvista_scalar_bar_args(surface, options, theme)
    plotter.add_mesh(
        mesh,
        **mesh_kwargs,
    )
    if not options.compose_annotations:
        plotter.add_text(
            f"{_PROPERTY_TITLES.get(surface.property_name, surface.property_name)} directional surface",
            position="upper_edge",
            font_size=options.title_font_size,
            color=theme.text_color,
        )
        plotter.add_axes(line_width=1, labels_off=False)
    plotter.camera_position = _camera_for(surface, azimuth_deg=azimuth, elevation_deg=elevation)
    if options.parallel_projection:
        plotter.camera.parallel_projection = True
    plotter.reset_camera()
    return plotter


def _normalizer(surface: DirectionalSurface):
    vmin = float(np.nanmin(surface.values))
    vmax = float(np.nanmax(surface.values))
    if abs(vmax - vmin) < 1e-14:
        return colors.Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _crop_rendered_surface(image: np.ndarray, *, pad_fraction: float = 0.07) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        return image
    if image.shape[2] == 4 and np.nanmin(image[..., 3]) < 250:
        mask = image[..., 3] > 5
    else:
        rgb = image[..., :3].astype(float)
        mask = np.any(rgb < 248.0, axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return image
    height, width = image.shape[:2]
    pad = max(8, int(max(height, width) * pad_fraction))
    row0 = max(0, int(rows.min()) - pad)
    row1 = min(height, int(rows.max()) + pad + 1)
    col0 = max(0, int(cols.min()) - pad)
    col1 = min(width, int(cols.max()) + pad + 1)
    return image[row0:row1, col0:col1]


def _compose_surface_annotations(image: np.ndarray, surface: DirectionalSurface, options: Render3DOptions) -> np.ndarray:
    theme = get_theme(options.theme_name)
    width, height = int(options.window_size[0]), int(options.window_size[1])
    dpi = 200
    figure_facecolor = "none" if options.transparent_background else theme.figure_facecolor
    with plt.rc_context(matplotlib_rc_params(theme)):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=figure_facecolor)
        ax = fig.add_axes([0.065, 0.095, 0.685, 0.76])
        ax.imshow(_crop_rendered_surface(image))
        ax.set_axis_off()
        fig.suptitle(
            f"{_PROPERTY_TITLES.get(surface.property_name, surface.property_name)} directional surface",
            x=0.405,
            y=0.925,
            fontsize=options.title_font_size,
            color=theme.text_color,
            fontweight="regular",
        )
        cbar_ax = fig.add_axes([0.825, 0.245, 0.028, 0.50])
        mappable = ScalarMappable(norm=_normalizer(surface), cmap=palette_colormap(options.palette_name))
        mappable.set_array(surface.values)
        colorbar = fig.colorbar(mappable, cax=cbar_ax)
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        colorbar.ax.tick_params(
            colors=theme.axis_color,
            width=theme.spine_width,
            labelsize=options.colorbar_tick_size,
            length=3.5,
        )
        colorbar.outline.set_edgecolor(theme.axis_color)
        colorbar.outline.set_linewidth(theme.spine_width)
        colorbar.ax.set_title(
            _PROPERTY_LABELS.get(surface.property_name, surface.property_name),
            color=theme.text_color,
            fontsize=options.colorbar_title_size,
            pad=8,
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
    return rgba


def render_surface_image(
    surface: DirectionalSurface,
    *,
    options: Render3DOptions | None = None,
    azimuth: float = 35.0,
    elevation: float = 25.0,
) -> np.ndarray:
    opts = options or Render3DOptions()
    plotter = _build_plotter(surface, opts, azimuth=azimuth, elevation=elevation)
    try:
        image = plotter.screenshot(
            None,
            transparent_background=opts.transparent_background,
            return_img=True,
        )
        array = np.asarray(image).copy()
    finally:
        plotter.close()
    if opts.compose_annotations:
        return _compose_surface_annotations(array, surface, opts)
    return array


def render_surface_png(
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    options: Render3DOptions | None = None,
    azimuth: float = 35.0,
    elevation: float = 25.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = options or Render3DOptions()
    if opts.compose_annotations:
        plt.imsave(output, render_surface_image(surface, options=opts, azimuth=azimuth, elevation=elevation))
        return output
    plotter = _build_plotter(surface, opts, azimuth=azimuth, elevation=elevation)
    try:
        plotter.screenshot(str(output), transparent_background=opts.transparent_background, return_img=False)
    finally:
        plotter.close()
    return output


def render_surface_gif(
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    frames: int = 72,
    fps: int = 12,
    options: Render3DOptions | None = None,
    elevation: float = 25.0,
    axis: str = "z",
) -> Path:
    if frames < 2:
        raise ValueError("frames must be at least 2.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = options or Render3DOptions()
    if opts.compose_annotations:
        opts = Render3DOptions(**{**opts.__dict__, "compose_annotations": False})
    plotter = _build_plotter(surface, opts, azimuth=0.0, elevation=elevation)
    try:
        plotter.open_gif(str(output), fps=fps)
        for frame_index in range(frames):
            azimuth = 360.0 * frame_index / frames
            plotter.camera_position = _camera_for(surface, azimuth_deg=azimuth, elevation_deg=elevation, axis=axis)
            plotter.write_frame()
    finally:
        plotter.close()
    return output


def render_surface_mp4(
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    frames: int = 120,
    fps: int = 24,
    options: Render3DOptions | None = None,
    elevation: float = 25.0,
    axis: str = "z",
) -> Path:
    if frames < 2:
        raise ValueError("frames must be at least 2.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = options or Render3DOptions()
    if opts.compose_annotations:
        opts = Render3DOptions(**{**opts.__dict__, "compose_annotations": False})
    plotter = _build_plotter(surface, opts, azimuth=0.0, elevation=elevation)
    try:
        plotter.open_movie(str(output), framerate=fps)
        for frame_index in range(frames):
            azimuth = 360.0 * frame_index / frames
            plotter.camera_position = _camera_for(surface, azimuth_deg=azimuth, elevation_deg=elevation, axis=axis)
            plotter.write_frame()
    finally:
        plotter.close()
    return output
