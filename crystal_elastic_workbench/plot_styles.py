"""Publication-oriented plotting themes and color palettes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from matplotlib import colors as mpl_colors


DEFAULT_THEME_NAME = "Nature White"
DEFAULT_PALETTE_NAME = "Nature White"
DEFAULT_3D_PALETTE_NAME = "Nature Surface"


@dataclass(frozen=True)
class Palette:
    name: str
    colors: tuple[str, ...]
    description: str
    category: str = "categorical"
    recommended_for_3d: bool = False


@dataclass(frozen=True)
class PlotTheme:
    name: str
    font_family: tuple[str, ...] = ("DejaVu Sans", "Arial", "sans-serif")
    font_size: float = 8.5
    label_size: float = 9.0
    title_size: float = 9.5
    tick_size: float = 8.0
    line_width: float = 1.25
    spine_width: float = 0.8
    grid_linewidth: float = 0.45
    grid_alpha: float = 0.18
    axes_facecolor: str = "#ffffff"
    figure_facecolor: str = "#ffffff"
    grid_color: str = "#8a8f98"
    text_color: str = "#202124"
    axis_color: str = "#30343b"
    figure_dpi: int = 150
    export_dpi: int = 300
    figure_size_1d: tuple[float, float] = (3.65, 2.45)
    figure_size_2d: tuple[float, float] = (3.35, 3.25)
    figure_size_3d: tuple[float, float] = (4.65, 4.05)
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.035


_PALETTES: dict[str, Palette] = {
    "Nature White": Palette(
        "Nature White",
        ("#2f6db3", "#d33f49", "#2a9d8f", "#e57a32", "#7b61a8", "#4d908e", "#6c757d"),
        "Clean white-background palette for default paper figures.",
    ),
    "Nature Muted": Palette(
        "Nature Muted",
        ("#496a81", "#a35f5f", "#5f8f72", "#b88746", "#806a9c", "#6f8f9f", "#52565c"),
        "Lower-saturation tones for dense multi-curve figures.",
    ),
    "Science High Contrast": Palette(
        "Science High Contrast",
        ("#004488", "#bb5566", "#228833", "#ddaa33", "#000000", "#66ccee", "#aa3377"),
        "High-contrast colors for talks, screens, and small panels.",
    ),
    "Okabe-Ito": Palette(
        "Okabe-Ito",
        ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"),
        "Color-vision-deficiency friendly categorical palette.",
    ),
    "Tol Bright": Palette(
        "Tol Bright",
        ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"),
        "Paul Tol bright palette for categorical scientific plots.",
    ),
    "Cividis": Palette(
        "Cividis",
        ("#00204D", "#31446B", "#666970", "#958F78", "#C8B866", "#FFEA46"),
        "Perceptually uniform sequential map with robust grayscale behavior.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Viridis Refined": Palette(
        "Viridis Refined",
        ("#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"),
        "Perceptually uniform sequential map for 2D and 3D scalar fields.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Gray Print": Palette(
        "Gray Print",
        ("#111111", "#343434", "#595959", "#7f7f7f", "#a6a6a6", "#cccccc"),
        "Print-safe grayscale palette for journals that reproduce figures in gray.",
        category="print_safe",
        recommended_for_3d=True,
    ),
    "Nature Surface": Palette(
        "Nature Surface",
        ("#15173d", "#234c7c", "#2178a6", "#36a6a6", "#8bc6a6", "#d9c474", "#e89a50"),
        "Balanced indigo-blue-teal-amber surface map for white-background 3D scalar fields.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Nature Thermal": Palette(
        "Nature Thermal",
        ("#251433", "#62366e", "#a44f6f", "#d9785f", "#efa75c", "#f6e27a"),
        "Muted thermal sequential map with restrained high-end yellow.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Blue-Gold": Palette(
        "Blue-Gold",
        ("#172a63", "#235c9c", "#3f95b6", "#86c8b8", "#d8d288", "#f3c44e"),
        "Blue-to-gold sequential map for modulus and intensity surfaces.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Teal-Amber": Palette(
        "Teal-Amber",
        ("#12343b", "#1e6f72", "#3aa99e", "#91cfa9", "#dfc27d", "#b97a2b"),
        "Teal-to-amber map for high-contrast surface gradients.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Deep Ocean": Palette(
        "Deep Ocean",
        ("#081d3a", "#123f6d", "#176c8c", "#21999a", "#6bc5a4", "#d9ed92"),
        "Dark-blue to sea-green sequential map with light high values.",
        category="sequential",
        recommended_for_3d=True,
    ),
    "Blue-White-Red": Palette(
        "Blue-White-Red",
        ("#234c9f", "#6f9ed6", "#d9e8f5", "#f7f7f7", "#f2c0a2", "#d6604d", "#9e1f28"),
        "Diverging blue-white-red map for signed or contrast-centered quantities.",
        category="diverging",
        recommended_for_3d=True,
    ),
    "Purple-Green": Palette(
        "Purple-Green",
        ("#432371", "#8e6bb3", "#d9d2e9", "#f7f7f7", "#d1e5c4", "#68a867", "#1b6e3b"),
        "Diverging purple-white-green map for contrast surfaces.",
        category="diverging",
        recommended_for_3d=True,
    ),
    "Brown-Blue": Palette(
        "Brown-Blue",
        ("#7f3b08", "#b97a32", "#dfc27d", "#f6ecd1", "#d1e5f0", "#67a9cf", "#2166ac"),
        "Diverging brown-white-blue map for print-friendly contrast.",
        category="diverging",
        recommended_for_3d=True,
    ),
    "Graphite": Palette(
        "Graphite",
        ("#111111", "#2b2b2b", "#4a4a4a", "#6f6f6f", "#9a9a9a", "#c8c8c8", "#eeeeee"),
        "High-resolution grayscale map for print-safe 3D surfaces.",
        category="print_safe",
        recommended_for_3d=True,
    ),
}

_PREFERRED_3D_PALETTE_ORDER = [
    "Nature Surface",
    "Nature Thermal",
    "Blue-Gold",
    "Teal-Amber",
    "Deep Ocean",
    "Blue-White-Red",
    "Purple-Green",
    "Brown-Blue",
    "Graphite",
]


_THEMES: dict[str, PlotTheme] = {
    "Nature White": PlotTheme(name="Nature White"),
    "Gray Print": PlotTheme(
        name="Gray Print",
        grid_alpha=0.16,
        axis_color="#1a1a1a",
        text_color="#111111",
    ),
}


def list_palette_names() -> list[str]:
    return list(_PALETTES)


def list_3d_palette_names() -> list[str]:
    ordered = [name for name in _PREFERRED_3D_PALETTE_ORDER if name in _PALETTES]
    remaining = [
        name
        for name, palette in _PALETTES.items()
        if palette.recommended_for_3d and name not in ordered
    ]
    return ordered + remaining


def list_theme_names() -> list[str]:
    return list(_THEMES)


def get_palette(name: str = DEFAULT_PALETTE_NAME) -> Palette:
    try:
        return _PALETTES[name]
    except KeyError as exc:
        valid = ", ".join(list_palette_names())
        raise ValueError(f"Unknown palette '{name}'. Valid palettes: {valid}") from exc


def get_theme(name: str = DEFAULT_THEME_NAME) -> PlotTheme:
    try:
        return _THEMES[name]
    except KeyError as exc:
        valid = ", ".join(list_theme_names())
        raise ValueError(f"Unknown plotting theme '{name}'. Valid themes: {valid}") from exc


def matplotlib_rc_params(theme: PlotTheme) -> dict[str, object]:
    return {
        "font.family": list(theme.font_family),
        "font.size": theme.font_size,
        "axes.labelsize": theme.label_size,
        "axes.titlesize": theme.title_size,
        "axes.linewidth": theme.spine_width,
        "axes.facecolor": theme.axes_facecolor,
        "figure.facecolor": theme.figure_facecolor,
        "figure.dpi": theme.figure_dpi,
        "savefig.dpi": theme.export_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "xtick.labelsize": theme.tick_size,
        "ytick.labelsize": theme.tick_size,
        "lines.linewidth": theme.line_width,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def apply_axis_style(ax, theme: PlotTheme, *, grid: bool = True) -> None:
    ax.tick_params(colors=theme.axis_color, width=theme.spine_width, labelsize=theme.tick_size)
    ax.xaxis.label.set_color(theme.text_color)
    ax.yaxis.label.set_color(theme.text_color)
    ax.title.set_color(theme.text_color)
    for spine in ax.spines.values():
        spine.set_color(theme.axis_color)
        spine.set_linewidth(theme.spine_width)
    if grid:
        ax.grid(True, color=theme.grid_color, linewidth=theme.grid_linewidth, alpha=theme.grid_alpha)


def apply_colorbar_style(colorbar, theme: PlotTheme) -> None:
    colorbar.ax.tick_params(colors=theme.axis_color, width=theme.spine_width, labelsize=theme.tick_size)
    colorbar.outline.set_edgecolor(theme.axis_color)
    colorbar.outline.set_linewidth(theme.spine_width)
    colorbar.set_label(colorbar.ax.get_ylabel(), color=theme.text_color, size=theme.label_size)


def palette_colormap(name: str, *, continuous: bool = True):
    palette = get_palette(name)
    if continuous:
        return mpl_colors.LinearSegmentedColormap.from_list(palette.name, palette.colors)
    return mpl_colors.ListedColormap(palette.colors, name=palette.name)


def cycle_colors(name: str) -> Iterable[str]:
    return get_palette(name).colors
