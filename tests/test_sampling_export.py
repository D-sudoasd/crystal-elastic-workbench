import json

import numpy as np
import pytest

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.exporting import (
    export_analysis_package,
    export_elastic_model_table,
    export_sampled_data,
    path_to_frame,
    sampled_data_to_frame,
    write_export_manifest,
)
from crystal_elastic_workbench.sampling import sample_direction_path, sample_plane, sample_sphere


def isotropic_cubic_matrix(bulk_gpa: float = 160.0, shear_gpa: float = 80.0) -> np.ndarray:
    c11 = bulk_gpa + 4.0 * shear_gpa / 3.0
    c12 = bulk_gpa - 2.0 * shear_gpa / 3.0
    c44 = shear_gpa
    return np.array(
        [
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ],
        dtype=float,
    )


def test_plane_sampling_returns_unit_directions_in_requested_plane():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    result = sample_plane(tensor, property_name="young", plane="xy", angle_count=181)

    assert result.angles_deg.shape == (181,)
    assert result.values.shape == (181,)
    assert np.allclose(np.linalg.norm(result.directions, axis=1), 1.0)
    assert np.allclose(result.directions[:, 2], 0.0)
    assert np.ptp(result.values) == pytest.approx(0.0, abs=1e-8)
    assert result.values[0] == pytest.approx(205.71428571428572)


def test_sphere_sampling_has_grid_shape_and_reports_extrema():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    surface = sample_sphere(tensor, property_name="young", theta_count=13, phi_count=25)

    assert surface.values.shape == (13, 25)
    assert surface.x.shape == surface.values.shape
    assert surface.min_value == pytest.approx(205.71428571428572)
    assert surface.max_value == pytest.approx(205.71428571428572)
    assert np.linalg.norm(surface.min_direction) == pytest.approx(1.0)
    assert np.linalg.norm(surface.max_direction) == pytest.approx(1.0)


def test_direction_path_sampling_supports_high_symmetry_style_paths():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    path = sample_direction_path(
        tensor,
        property_name="young",
        points=[
            ("[100]", [1, 0, 0]),
            ("[110]", [1, 1, 0]),
            ("[111]", [1, 1, 1]),
        ],
        points_per_segment=9,
    )

    assert path.values.size == path.directions.shape[0]
    assert path.tick_labels == ["[100]", "[110]", "[111]"]
    assert np.all(np.diff(path.distance) >= 0.0)
    assert np.allclose(np.linalg.norm(path.directions, axis=1), 1.0)
    assert np.ptp(path.values) == pytest.approx(0.0, abs=1e-8)
    frame = path_to_frame(path)
    assert {"distance", "nx", "ny", "nz", "young"}.issubset(frame.columns)
    assert len(frame) == path.values.size


def test_export_analysis_package_writes_traceable_manifest_and_data(tmp_path):
    tensor = ElasticTensor(
        isotropic_cubic_matrix(),
        crystal_system="cubic",
        unit="GPa",
        material_name="isotropic-test",
    )

    manifest_path = export_analysis_package(
        tensor,
        tmp_path,
        plane_angle_count=91,
        sphere_theta_count=9,
        sphere_phi_count=17,
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["material_name"] == "isotropic-test"
    assert manifest["unit"] == "GPa"
    assert manifest["crystal_system"] == "cubic"
    assert "stiffness_matrix_csv" in manifest["files"]
    assert "surface_young_csv" in manifest["files"]
    assert "model_table_csv" in manifest["files"]
    assert "model_table_xlsx" in manifest["files"]
    assert "model_table_notes_json" in manifest["files"]
    assert manifest["recommended_model"] == "Hill"
    assert manifest["included_models"] == ["Voigt", "Reuss", "Hill", "Geometric"]
    assert (tmp_path / manifest["files"]["stiffness_matrix_csv"]).exists()
    assert (tmp_path / manifest["files"]["surface_young_csv"]).exists()
    assert (tmp_path / manifest["files"]["model_table_csv"]).exists()
    assert (tmp_path / manifest["files"]["model_table_xlsx"]).exists()
    assert (tmp_path / manifest["files"]["model_table_notes_json"]).exists()
    notes = json.loads((tmp_path / manifest["files"]["model_table_notes_json"]).read_text(encoding="utf-8"))
    assert notes["recommended_model"] == "Hill"
    assert notes["diagnostics"]["B_spread"] == pytest.approx(0.0)


def test_export_sampled_data_writes_csv_and_manifest(tmp_path):
    tensor = ElasticTensor(
        isotropic_cubic_matrix(),
        crystal_system="cubic",
        unit="GPa",
        material_name="sampled-export-test",
    )
    plane = sample_plane(tensor, property_name="young", plane="xy", angle_count=19)

    output = export_sampled_data(tensor, plane, tmp_path / "plane.csv", kind="polar")

    assert output.exists()
    assert len(sampled_data_to_frame(plane)) == 19
    manifest = json.loads(output.with_name(f"{output.name}.manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_type"] == "polar_sampled_data"
    assert manifest["parameters"]["property"] == "young"
    assert manifest["parameters"]["rows"] == 19


def test_export_sampled_data_rejects_unknown_kind(tmp_path):
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")
    plane = sample_plane(tensor, property_name="young", plane="xy", angle_count=19)

    with pytest.raises(ValueError, match="kind must be one of"):
        export_sampled_data(tensor, plane, tmp_path / "plane.csv", kind="unknown")


def test_export_elastic_model_table_writes_sidecar_manifest(tmp_path):
    tensor = ElasticTensor(
        isotropic_cubic_matrix(),
        crystal_system="cubic",
        unit="GPa",
        material_name="model-table-test",
    )

    output = export_elastic_model_table(tensor, tmp_path / "elastic_model_summary.csv")

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Voigt" in text
    assert "Geometric" in text
    manifest = json.loads(output.with_name(f"{output.name}.manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_type"] == "elastic_model_table"
    assert manifest["parameters"]["recommended_model"] == "Hill"
    assert manifest["parameters"]["included_models"] == ["Voigt", "Reuss", "Hill", "Geometric"]


def test_single_export_manifest_records_input_and_output_file(tmp_path):
    tensor = ElasticTensor(
        isotropic_cubic_matrix(),
        crystal_system="cubic",
        unit="GPa",
        material_name="sidecar-test",
    )
    exported_file = tmp_path / "plot.png"
    exported_file.write_bytes(b"fake image bytes")

    manifest_path = write_export_manifest(
        tensor,
        exported_file,
        export_type="figure",
        parameters={"dpi": 300},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["material_name"] == "sidecar-test"
    assert manifest["export_type"] == "figure"
    assert manifest["exported_file"] == "plot.png"
    assert manifest["parameters"]["dpi"] == 300
