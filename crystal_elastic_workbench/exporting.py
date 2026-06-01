"""Traceable data export helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crystal_elastic_workbench import __version__
from crystal_elastic_workbench.core import ElasticTensor, VOIGT_LABELS
from crystal_elastic_workbench.elastic_models import (
    MODEL_NAMES,
    elastic_model_diagnostics,
    elastic_model_results,
    elastic_model_table_frame,
)
from crystal_elastic_workbench.sampling import (
    DirectionPath,
    DirectionalSurface,
    PlaneSlice,
    sample_plane,
    sample_sphere,
)
from crystal_elastic_workbench.stability import check_stability


def matrix_to_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=[f"C{i}" for i in VOIGT_LABELS], columns=[f"C{i}" for i in VOIGT_LABELS])


def plane_to_frame(plane: PlaneSlice) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "angle_deg": plane.angles_deg,
            "nx": plane.directions[:, 0],
            "ny": plane.directions[:, 1],
            "nz": plane.directions[:, 2],
            plane.property_name: plane.values,
        }
    )


def surface_to_frame(surface: DirectionalSurface) -> pd.DataFrame:
    flat_dirs = surface.directions.reshape(-1, 3)
    return pd.DataFrame(
        {
            "theta_rad": surface.theta.reshape(-1),
            "phi_rad": surface.phi.reshape(-1),
            "nx": flat_dirs[:, 0],
            "ny": flat_dirs[:, 1],
            "nz": flat_dirs[:, 2],
            surface.property_name: surface.values.reshape(-1),
            "x": surface.x.reshape(-1),
            "y": surface.y.reshape(-1),
            "z": surface.z.reshape(-1),
        }
    )


def path_to_frame(path: DirectionPath) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance": path.distance,
            "nx": path.directions[:, 0],
            "ny": path.directions[:, 1],
            "nz": path.directions[:, 2],
            path.property_name: path.values,
        }
    )


def sampled_data_to_frame(data: DirectionPath | PlaneSlice | DirectionalSurface) -> pd.DataFrame:
    if isinstance(data, DirectionPath):
        return path_to_frame(data)
    if isinstance(data, PlaneSlice):
        return plane_to_frame(data)
    if isinstance(data, DirectionalSurface):
        return surface_to_frame(data)
    raise TypeError("data must be DirectionPath, PlaneSlice, or DirectionalSurface.")


def export_sampled_data(
    tensor: ElasticTensor,
    data: DirectionPath | PlaneSlice | DirectionalSurface,
    output_path: str | Path,
    *,
    kind: str,
) -> Path:
    """Write sampled plot data to CSV with the standard sidecar manifest."""

    if kind not in {"line", "polar", "surface"}:
        raise ValueError("kind must be one of: line, polar, surface.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = sampled_data_to_frame(data)
    frame.to_csv(output, index=False)
    write_export_manifest(
        tensor,
        output,
        export_type=f"{kind}_sampled_data",
        parameters={"property": data.property_name, "rows": int(len(frame))},
    )
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_elastic_model_table(tensor: ElasticTensor, output_path: str | Path) -> Path:
    """Export the current Voigt/Reuss/Hill/Geometric comparison table."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = elastic_model_table_frame(elastic_model_results(tensor))
    if output.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(output, index=False)
    else:
        frame.to_csv(output, index=False)
    write_export_manifest(
        tensor,
        output,
        export_type="elastic_model_table",
        parameters={
            "recommended_model": "Hill",
            "included_models": MODEL_NAMES.copy(),
        },
    )
    return output


def write_export_manifest(
    tensor: ElasticTensor,
    exported_file: str | Path,
    *,
    export_type: str,
    parameters: dict[str, Any] | None = None,
) -> Path:
    """Write a sidecar manifest for a single exported figure or animation."""

    exported_path = Path(exported_file)
    manifest_path = exported_path.with_name(f"{exported_path.name}.manifest.json")
    payload = {
        "program": "Crystal Elastic Workbench",
        "version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_type": export_type,
        "exported_file": exported_path.name,
        "material_name": tensor.material_name,
        "unit": tensor.unit,
        "crystal_system": tensor.crystal_system,
        "voigt_order": ["11", "22", "33", "23", "13", "12"],
        "strain_convention": "[e11, e22, e33, 2e23, 2e13, 2e12]",
        "stiffness_matrix": tensor.stiffness_matrix.tolist(),
        "parameters": parameters or {},
    }
    _write_json(manifest_path, payload)
    return manifest_path


def export_analysis_package(
    tensor: ElasticTensor,
    output_dir: str | Path,
    *,
    plane_angle_count: int = 361,
    sphere_theta_count: int = 37,
    sphere_phi_count: int = 73,
) -> Path:
    """Export input, scalar results, sampled data, and a manifest.

    The manifest is intentionally redundant: it records the raw Cij matrix,
    unit, crystal system, sampling sizes, and every generated relative path.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    stiffness_csv = out / "stiffness_matrix.csv"
    compliance_csv = out / "compliance_matrix.csv"
    matrix_to_frame(tensor.stiffness_matrix).to_csv(stiffness_csv)
    matrix_to_frame(tensor.compliance_matrix).to_csv(compliance_csv)
    files["stiffness_matrix_csv"] = stiffness_csv.name
    files["compliance_matrix_csv"] = compliance_csv.name

    summary = tensor.polycrystalline_summary()
    summary_csv = out / "polycrystalline_summary.csv"
    pd.DataFrame([summary.as_dict()]).to_csv(summary_csv, index=False)
    files["polycrystalline_summary_csv"] = summary_csv.name

    model_results = elastic_model_results(summary)
    model_frame = elastic_model_table_frame(model_results)
    model_table_csv = out / "elastic_model_summary.csv"
    model_table_xlsx = out / "elastic_model_summary.xlsx"
    model_notes_json = out / "elastic_model_notes.json"
    model_frame.to_csv(model_table_csv, index=False)
    model_frame.to_excel(model_table_xlsx, index=False)
    _write_json(
        model_notes_json,
        {
            "recommended_model": "Hill",
            "included_models": MODEL_NAMES.copy(),
            "diagnostics": elastic_model_diagnostics(summary),
            "model_notes": [
                {
                    "model": result.model,
                    "assumption": result.assumption,
                    "notes": result.notes,
                }
                for result in model_results
            ],
        },
    )
    files["model_table_csv"] = model_table_csv.name
    files["model_table_xlsx"] = model_table_xlsx.name
    files["model_table_notes_json"] = model_notes_json.name

    stability = check_stability(tensor.stiffness_matrix, crystal_system=tensor.crystal_system)
    stability_json = out / "stability.json"
    _write_json(stability_json, stability.as_dict())
    files["stability_json"] = stability_json.name

    for prop in ("young", "compressibility"):
        plane = sample_plane(
            tensor,
            property_name=prop,
            plane="xy",
            angle_count=plane_angle_count,
        )
        plane_csv = out / f"plane_xy_{prop}.csv"
        plane_to_frame(plane).to_csv(plane_csv, index=False)
        files[f"plane_xy_{prop}_csv"] = plane_csv.name

    for prop in ("young", "compressibility", "shear", "poisson"):
        surface = sample_sphere(
            tensor,
            property_name=prop,
            theta_count=sphere_theta_count,
            phi_count=sphere_phi_count,
            transverse_mode="mean",
        )
        surface_csv = out / f"surface_{prop}.csv"
        surface_to_frame(surface).to_csv(surface_csv, index=False)
        files[f"surface_{prop}_csv"] = surface_csv.name

    manifest = {
        "program": "Crystal Elastic Workbench",
        "version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "material_name": tensor.material_name,
        "unit": tensor.unit,
        "crystal_system": tensor.crystal_system,
        "voigt_order": ["11", "22", "33", "23", "13", "12"],
        "strain_convention": "[e11, e22, e33, 2e23, 2e13, 2e12]",
        "stiffness_matrix": tensor.stiffness_matrix.tolist(),
        "sampling": {
            "plane_angle_count": plane_angle_count,
            "sphere_theta_count": sphere_theta_count,
            "sphere_phi_count": sphere_phi_count,
            "transverse_mode_for_shear_and_poisson": "mean",
        },
        "files": files,
        "recommended_model": "Hill",
        "included_models": MODEL_NAMES.copy(),
    }

    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
