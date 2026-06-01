"""Polycrystalline elastic model comparison helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import pandas as pd

from crystal_elastic_workbench.core import ElasticTensor, PolycrystalSummary


MODEL_TABLE_COLUMNS = ["Model", "Assumption", "B", "G", "E", "nu", "B/G", "Notes"]
MODEL_NAMES = ["Voigt", "Reuss", "Hill", "Geometric"]


@dataclass(frozen=True)
class ElasticModelResult:
    model: str
    assumption: str
    bulk_gpa: float | None
    shear_gpa: float | None
    young_gpa: float | None
    poisson: float | None
    pugh_ratio: float | None
    notes: str

    def as_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def _summary_from_input(source: ElasticTensor | PolycrystalSummary) -> PolycrystalSummary:
    if isinstance(source, ElasticTensor):
        return source.polycrystalline_summary()
    return source


def _safe_isotropic_derived(
    bulk: float | None,
    shear: float | None,
) -> tuple[float | None, float | None, float | None]:
    if bulk is None or shear is None:
        return None, None, None
    denominator = 3.0 * bulk + shear
    if abs(denominator) <= 1e-14 or abs(shear) <= 1e-14:
        return None, None, None
    young = 9.0 * bulk * shear / denominator
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * denominator)
    pugh = bulk / shear
    return float(young), float(poisson), float(pugh)


def _model_result(
    *,
    model: str,
    assumption: str,
    bulk: float | None,
    shear: float | None,
    notes: str,
) -> ElasticModelResult:
    young, poisson, pugh = _safe_isotropic_derived(bulk, shear)
    return ElasticModelResult(
        model=model,
        assumption=assumption,
        bulk_gpa=None if bulk is None else float(bulk),
        shear_gpa=None if shear is None else float(shear),
        young_gpa=young,
        poisson=poisson,
        pugh_ratio=pugh,
        notes=notes,
    )


def elastic_model_results(source: ElasticTensor | PolycrystalSummary) -> list[ElasticModelResult]:
    """Return Voigt/Reuss/Hill/Geometric single-phase polycrystal estimates."""

    summary = _summary_from_input(source)
    results = [
        _model_result(
            model="Voigt",
            assumption="Uniform strain",
            bulk=summary.bulk_voigt_gpa,
            shear=summary.shear_voigt_gpa,
            notes="Upper-bound tendency for random polycrystal averaging.",
        ),
        _model_result(
            model="Reuss",
            assumption="Uniform stress",
            bulk=summary.bulk_reuss_gpa,
            shear=summary.shear_reuss_gpa,
            notes="Lower-bound tendency for random polycrystal averaging.",
        ),
        _model_result(
            model="Hill",
            assumption="Arithmetic VRH mean",
            bulk=summary.bulk_hill_gpa,
            shear=summary.shear_hill_gpa,
            notes="Recommended default central estimate.",
        ),
    ]

    if (
        summary.bulk_voigt_gpa > 0.0
        and summary.bulk_reuss_gpa > 0.0
        and summary.shear_voigt_gpa > 0.0
        and summary.shear_reuss_gpa > 0.0
    ):
        results.append(
            _model_result(
                model="Geometric",
                assumption="Geometric VR mean",
                bulk=sqrt(summary.bulk_voigt_gpa * summary.bulk_reuss_gpa),
                shear=sqrt(summary.shear_voigt_gpa * summary.shear_reuss_gpa),
                notes="Empirical center estimate; not a rigorous bound.",
            )
        )
    else:
        results.append(
            _model_result(
                model="Geometric",
                assumption="Geometric VR mean",
                bulk=None,
                shear=None,
                notes="Unavailable: Voigt/Reuss B and G must all be positive.",
            )
        )
    return results


def elastic_model_table_frame(results: Iterable[ElasticModelResult]) -> pd.DataFrame:
    rows = [
        {
            "Model": result.model,
            "Assumption": result.assumption,
            "B": result.bulk_gpa,
            "G": result.shear_gpa,
            "E": result.young_gpa,
            "nu": result.poisson,
            "B/G": result.pugh_ratio,
            "Notes": result.notes,
        }
        for result in results
    ]
    return pd.DataFrame(rows, columns=MODEL_TABLE_COLUMNS)


def elastic_model_diagnostics(summary: PolycrystalSummary) -> dict[str, object]:
    b_spread = float(summary.bulk_voigt_gpa - summary.bulk_reuss_gpa)
    g_spread = float(summary.shear_voigt_gpa - summary.shear_reuss_gpa)
    b_spread_percent = None
    g_spread_percent = None
    if abs(summary.bulk_hill_gpa) > 1e-14:
        b_spread_percent = float(100.0 * b_spread / summary.bulk_hill_gpa)
    if abs(summary.shear_hill_gpa) > 1e-14:
        g_spread_percent = float(100.0 * g_spread / summary.shear_hill_gpa)
    return {
        "recommended_model": "Hill",
        "included_models": MODEL_NAMES.copy(),
        "B_spread": b_spread,
        "G_spread": g_spread,
        "B_spread_percent": b_spread_percent,
        "G_spread_percent": g_spread_percent,
        "universal_anisotropy": summary.universal_anisotropy,
        "zener_anisotropy": summary.zener_anisotropy,
    }
