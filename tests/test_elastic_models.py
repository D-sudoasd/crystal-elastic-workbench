import numpy as np
import pytest

from crystal_elastic_workbench.core import ElasticTensor, PolycrystalSummary
from crystal_elastic_workbench.elastic_models import (
    elastic_model_diagnostics,
    elastic_model_results,
    elastic_model_table_frame,
)


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


def test_elastic_model_results_have_expected_rows_and_columns():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    results = elastic_model_results(tensor)
    frame = elastic_model_table_frame(results)

    assert [result.model for result in results] == ["Voigt", "Reuss", "Hill", "Geometric"]
    assert list(frame.columns) == ["Model", "Assumption", "B", "G", "E", "nu", "B/G", "Notes"]


def test_isotropic_cubic_models_reduce_to_same_values():
    tensor = ElasticTensor(isotropic_cubic_matrix(160.0, 80.0), crystal_system="cubic")

    results = elastic_model_results(tensor)

    for result in results:
        assert result.bulk_gpa == pytest.approx(160.0)
        assert result.shear_gpa == pytest.approx(80.0)
        assert result.young_gpa == pytest.approx(205.71428571428572)
        assert result.poisson == pytest.approx(0.2857142857142857)
        assert result.pugh_ratio == pytest.approx(2.0)


def test_anisotropic_models_report_hill_between_voigt_and_reuss_and_geometric_notes():
    matrix = isotropic_cubic_matrix()
    matrix[0, 0] = 320.0
    matrix[3, 3] = 45.0
    tensor = ElasticTensor(matrix, crystal_system="triclinic", symmetrize=True)

    by_model = {result.model: result for result in elastic_model_results(tensor)}

    assert by_model["Reuss"].bulk_gpa <= by_model["Hill"].bulk_gpa <= by_model["Voigt"].bulk_gpa
    assert by_model["Reuss"].shear_gpa <= by_model["Hill"].shear_gpa <= by_model["Voigt"].shear_gpa
    assert by_model["Geometric"].bulk_gpa is not None
    assert by_model["Geometric"].shear_gpa is not None
    assert "empirical" in by_model["Geometric"].notes.lower()


def test_geometric_model_is_blank_when_bounds_are_not_positive():
    summary = PolycrystalSummary(
        bulk_voigt_gpa=100.0,
        bulk_reuss_gpa=80.0,
        bulk_hill_gpa=90.0,
        shear_voigt_gpa=30.0,
        shear_reuss_gpa=-2.0,
        shear_hill_gpa=14.0,
        young_hill_gpa=40.0,
        poisson_hill=0.3,
        pugh_ratio=6.0,
        universal_anisotropy=0.0,
        bulk_anisotropy_percent=0.0,
        shear_anisotropy_percent=0.0,
    )

    geometric = elastic_model_results(summary)[-1]

    assert geometric.model == "Geometric"
    assert geometric.bulk_gpa is None
    assert geometric.shear_gpa is None
    assert "positive" in geometric.notes.lower()


def test_elastic_model_diagnostics_report_spread_and_recommended_model():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    diagnostics = elastic_model_diagnostics(tensor.polycrystalline_summary())

    assert diagnostics["recommended_model"] == "Hill"
    assert diagnostics["included_models"] == ["Voigt", "Reuss", "Hill", "Geometric"]
    assert diagnostics["B_spread"] == pytest.approx(0.0)
    assert diagnostics["G_spread"] == pytest.approx(0.0)
