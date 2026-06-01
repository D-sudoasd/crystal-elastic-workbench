"""Matplotlib plotting and animation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator

from crystal_elastic_workbench.plot_styles import (
    DEFAULT_3D_PALETTE_NAME,
    DEFAULT_PALETTE_NAME,
    DEFAULT_THEME_NAME,
    apply_axis_style,
    apply_colorbar_style,
    get_palette,
    get_theme,
    matplotlib_rc_params,
    palette_colormap,
)
from crystal_elastic_workbench.render3d import (
    PyVistaUnavailableError,
    Render3DOptions,
    render_surface_gif,
    render_surface_mp4,
)
from crystal_elastic_workbench.sampling import DirectionPath, DirectionalSurface, PlaneSlice


PROPERTY_LABELS = {
    "young": "Young's modulus E(n) [GPa]",
    "compressibility": "Linear compressibility beta(n) [1/GPa]",
    "shear": "Shear modulus G(n,m), transverse mean [GPa]",
    "poisson": "Poisson ratio nu(n,m), transverse mean",
}

PROPERTY_SHORT_LABELS = {
    "young": "E(n)",
    "compressibility": "beta(n)",
    "shear": "G(n,m)",
    "poisson": "nu(n,m)",
}


def property_label(property_name: str) -> str:
    return PROPERTY_LABELS.get(property_name, property_name)


def property_short_label(property_name: str) -> str:
    return PROPERTY_SHORT_LABELS.get(property_name, property_name)


def plot_line_slice(
    plane: PlaneSlice,
    *,
    title: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_PALETTE_NAME,
):
    theme = get_theme(theme_name)
    palette = get_palette(palette_name)
    with plt.rc_context(matplotlib_rc_params(theme)):
        fig, ax = plt.subplots(figsize=theme.figure_size_1d, constrained_layout=True, dpi=theme.figure_dpi)
    ax.plot(plane.angles_deg, plane.values, color=palette.colors[0], linewidth=theme.line_width)
    ax.set_xlabel("Angle in plane [deg]")
    ax.set_ylabel(property_label(plane.property_name))
    ax.set_title(title or f"{property_short_label(plane.property_name)} in {plane.plane_label.upper()} plane", pad=4)
    apply_axis_style(ax, theme)
    return fig


def plot_direction_path(
    path: DirectionPath,
    *,
    title: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_PALETTE_NAME,
):
    theme = get_theme(theme_name)
    palette = get_palette(palette_name)
    with plt.rc_context(matplotlib_rc_params(theme)):
        fig, ax = plt.subplots(figsize=theme.figure_size_1d, constrained_layout=True, dpi=theme.figure_dpi)
    ax.plot(path.distance, path.values, color=palette.colors[0], linewidth=theme.line_width)
    ax.set_xlabel("Direction path")
    ax.set_ylabel(property_label(path.property_name))
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels(path.tick_labels)
    for position in path.tick_positions:
        ax.axvline(position, color=theme.grid_color, linewidth=theme.grid_linewidth, alpha=0.26, zorder=0)
    ax.set_title(title or f"{property_short_label(path.property_name)} high-symmetry path", pad=4)
    apply_axis_style(ax, theme)
    return fig


def plot_plane_slice(
    plane: PlaneSlice,
    *,
    title: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_PALETTE_NAME,
):
    theme = get_theme(theme_name)
    palette = get_palette(palette_name)
    with plt.rc_context(matplotlib_rc_params(theme)):
        fig = plt.figure(figsize=theme.figure_size_2d, constrained_layout=True, dpi=theme.figure_dpi)
        ax = fig.add_subplot(111, projection="polar")
    radians = np.deg2rad(plane.angles_deg)
    ax.plot(radians, plane.values, color=palette.colors[0], linewidth=theme.line_width)
    ax.fill(radians, plane.values, color=palette.colors[0], alpha=0.075)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_rlabel_position(22.5)
    ax.set_title(title or f"{property_short_label(plane.property_name)} in {plane.plane_label.upper()} plane", pad=5)
    apply_axis_style(ax, theme)
    return fig


def _set_equal_3d_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    max_range = max(float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z)), 1e-12)
    mid_x = 0.5 * (float(np.max(x)) + float(np.min(x)))
    mid_y = 0.5 * (float(np.max(y)) + float(np.min(y)))
    mid_z = 0.5 * (float(np.max(z)) + float(np.min(z)))
    half = 0.5 * max_range
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_directional_surface(
    surface: DirectionalSurface,
    *,
    title: str | None = None,
    cmap: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_3D_PALETTE_NAME,
    alpha: float = 0.92,
    elev: float = 25.0,
    azim: float = 35.0,
):
    theme = get_theme(theme_name)
    with plt.rc_context(matplotlib_rc_params(theme)):
        fig = plt.figure(figsize=theme.figure_size_3d, constrained_layout=True, dpi=theme.figure_dpi)
        ax = fig.add_subplot(111, projection="3d")
    cmap_obj = plt.get_cmap(cmap) if cmap else palette_colormap(palette_name)
    vmin = float(np.min(surface.values))
    vmax = float(np.max(surface.values))
    if abs(vmax - vmin) < 1e-14:
        norm = colors.Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    facecolors = cmap_obj(norm(surface.values))
    facecolors[..., -1] = alpha
    ax.plot_surface(
        surface.x,
        surface.y,
        surface.z,
        facecolors=facecolors,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=True,
        rstride=1,
        cstride=1,
    )
    ax.scatter(
        [surface.min_value * surface.min_direction[0], surface.max_value * surface.max_direction[0]],
        [surface.min_value * surface.min_direction[1], surface.max_value * surface.max_direction[1]],
        [surface.min_value * surface.min_direction[2], surface.max_value * surface.max_direction[2]],
        color=["#d33f49", "#2a9d8f"],
        s=20,
        depthshade=False,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(surface.values)
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.72,
        pad=0.075,
        fraction=theme.colorbar_fraction,
        label=property_label(surface.property_name),
    )
    apply_colorbar_style(colorbar, theme)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or f"{property_short_label(surface.property_name)} directional surface", pad=4)
    _set_equal_3d_axes(ax, surface.x, surface.y, surface.z)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_facecolor(theme.axes_facecolor)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
            axis._axinfo["grid"]["linewidth"] = 0.0
        except Exception:
            pass
    ax.tick_params(colors=theme.axis_color, labelsize=theme.tick_size, width=theme.spine_width)
    return fig


def export_rotating_gif(
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    frames: int = 72,
    fps: int = 12,
    dpi: int = 120,
    elev: float = 25.0,
    cmap: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_3D_PALETTE_NAME,
    axis: str = "z",
    backend: str = "auto",
    transparent_background: bool = False,
    lighting_intensity: float = 1.0,
    surface_smoothing: float = 0.0,
    surface_subdivision: int = 1,
    show_edges: bool = False,
    title_font_size: int = 18,
    label_font_size: int = 12,
    colorbar_title_size: int = 12,
    colorbar_tick_size: int = 10,
    ambient: float = 0.28,
    diffuse: float = 0.74,
    specular: float = 0.32,
    specular_power: float = 28.0,
) -> Path:
    if frames < 2:
        raise ValueError("frames must be at least 2.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if backend in {"auto", "pyvista"}:
        try:
            return render_surface_gif(
                surface,
                output,
                frames=frames,
                fps=fps,
                elevation=elev,
                axis=axis,
                options=Render3DOptions(
                    theme_name=theme_name,
                    palette_name=palette_name,
                    window_size=(max(480, int(dpi * 8)), max(400, int(dpi * 6))),
                    transparent_background=transparent_background,
                    lighting_intensity=lighting_intensity,
                    surface_smoothing=surface_smoothing,
                    surface_subdivision=surface_subdivision,
                    show_edges=show_edges,
                    compose_annotations=False,
                    title_font_size=title_font_size,
                    label_font_size=label_font_size,
                    colorbar_title_size=colorbar_title_size,
                    colorbar_tick_size=colorbar_tick_size,
                    ambient=ambient,
                    diffuse=diffuse,
                    specular=specular,
                    specular_power=specular_power,
                ),
            )
        except PyVistaUnavailableError:
            if backend == "pyvista":
                raise
        except Exception:
            if backend == "pyvista":
                raise

    fig = plot_directional_surface(surface, cmap=cmap, theme_name=theme_name, palette_name=palette_name, elev=elev, azim=0.0)
    ax = fig.axes[0]

    def update(frame_index: int):
        azim = 360.0 * frame_index / frames
        key = axis.lower()
        if key == "x":
            ax.view_init(elev=azim, azim=0.0)
        elif key == "y":
            ax.view_init(elev=azim, azim=90.0)
        else:
            ax.view_init(elev=elev, azim=azim)
        return (ax,)

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output


def export_rotating_mp4(
    surface: DirectionalSurface,
    output_path: str | Path,
    *,
    frames: int = 120,
    fps: int = 24,
    dpi: int = 140,
    elev: float = 25.0,
    cmap: str | None = None,
    theme_name: str = DEFAULT_THEME_NAME,
    palette_name: str = DEFAULT_3D_PALETTE_NAME,
    axis: str = "z",
    lighting_intensity: float = 1.0,
    surface_smoothing: float = 0.0,
    surface_subdivision: int = 1,
    show_edges: bool = False,
    title_font_size: int = 18,
    label_font_size: int = 12,
    colorbar_title_size: int = 12,
    colorbar_tick_size: int = 10,
    ambient: float = 0.28,
    diffuse: float = 0.74,
    specular: float = 0.32,
    specular_power: float = 28.0,
    backend: str = "auto",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if frames < 2:
        raise ValueError("frames must be at least 2.")
    if backend in {"auto", "pyvista"}:
        try:
            return render_surface_mp4(
                surface,
                output,
                frames=frames,
                fps=fps,
                elevation=elev,
                axis=axis,
                options=Render3DOptions(
                    theme_name=theme_name,
                    palette_name=palette_name,
                    window_size=(max(480, int(dpi * 8)), max(400, int(dpi * 6))),
                    lighting_intensity=lighting_intensity,
                    surface_smoothing=surface_smoothing,
                    surface_subdivision=surface_subdivision,
                    show_edges=show_edges,
                    compose_annotations=False,
                    title_font_size=title_font_size,
                    label_font_size=label_font_size,
                    colorbar_title_size=colorbar_title_size,
                    colorbar_tick_size=colorbar_tick_size,
                    ambient=ambient,
                    diffuse=diffuse,
                    specular=specular,
                    specular_power=specular_power,
                ),
            )
        except PyVistaUnavailableError:
            if backend == "pyvista":
                raise
        except Exception:
            if backend == "pyvista":
                raise

    fig = plot_directional_surface(surface, cmap=cmap, theme_name=theme_name, palette_name=palette_name, elev=elev, azim=0.0)
    ax = fig.axes[0]

    def update(frame_index: int):
        azim = 360.0 * frame_index / frames
        key = axis.lower()
        if key == "x":
            ax.view_init(elev=azim, azim=0.0)
        elif key == "y":
            ax.view_init(elev=azim, azim=90.0)
        else:
            ax.view_init(elev=elev, azim=azim)
        return (ax,)

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output
