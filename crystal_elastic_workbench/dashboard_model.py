"""Presentation model for the GUI dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from crystal_elastic_workbench.stability import StabilityResult


@dataclass(frozen=True)
class DashboardState:
    status_text: str
    metrics: dict[str, str]
    anisotropy_text: str
    figure_text: str


def format_metric(value: float | None, unit: str = "") -> str:
    if value is None:
        return "-"
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.4g}{suffix}"


def build_dashboard_state(
    summary: dict[str, float | None],
    stability: StabilityResult,
    *,
    figures_generated: bool,
) -> DashboardState:
    if stability.overall_stable:
        status_text = "Stable: Cij passed symmetry, invertibility, positive-definite, and Born checks."
    else:
        problem_text = ", ".join(stability.failed_conditions) if stability.failed_conditions else "see stability details"
        status_text = f"Warning: stability checks need review ({problem_text})."

    metrics = {
        "B_H": format_metric(summary.get("bulk_hill_gpa"), "GPa"),
        "G_H": format_metric(summary.get("shear_hill_gpa"), "GPa"),
        "E_H": format_metric(summary.get("young_hill_gpa"), "GPa"),
        "A_U": format_metric(summary.get("universal_anisotropy")),
    }
    warning_text = f" Warnings: {', '.join(stability.warnings)}" if stability.warnings else ""
    anisotropy_text = (
        f"Pugh B/G={format_metric(summary.get('pugh_ratio'))}; "
        f"Zener anisotropy={format_metric(summary.get('zener_anisotropy'))}. "
        "Recommended model: Hill; see model comparison for Voigt/Reuss/Geometric spread."
        f"{warning_text}"
    )
    figure_text = "Figures: 1D/2D/3D updated" if figures_generated else "Figures: incomplete; check plot tabs"
    return DashboardState(
        status_text=status_text,
        metrics=metrics,
        anisotropy_text=anisotropy_text,
        figure_text=figure_text,
    )
