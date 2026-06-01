import math

import numpy as np
import pytest

from crystal_elastic_workbench.core import ElasticTensor
from crystal_elastic_workbench.stability import check_stability


def isotropic_cubic_matrix(bulk_gpa: float = 160.0, shear_gpa: float = 80.0) -> np.ndarray:
    """Return an exactly isotropic cubic stiffness matrix in Voigt notation."""
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


def test_vrh_reduces_to_input_bulk_and_shear_for_isotropic_cubic():
    tensor = ElasticTensor(isotropic_cubic_matrix(160.0, 80.0), crystal_system="cubic")

    summary = tensor.polycrystalline_summary()

    assert summary.bulk_voigt_gpa == pytest.approx(160.0)
    assert summary.bulk_reuss_gpa == pytest.approx(160.0)
    assert summary.bulk_hill_gpa == pytest.approx(160.0)
    assert summary.shear_voigt_gpa == pytest.approx(80.0)
    assert summary.shear_reuss_gpa == pytest.approx(80.0)
    assert summary.shear_hill_gpa == pytest.approx(80.0)
    assert summary.young_hill_gpa == pytest.approx(205.71428571428572)
    assert summary.poisson_hill == pytest.approx(0.2857142857142857)
    assert summary.universal_anisotropy == pytest.approx(0.0, abs=1e-12)


def test_directional_properties_use_engineering_shear_convention_correctly():
    tensor = ElasticTensor(isotropic_cubic_matrix(160.0, 80.0), crystal_system="cubic")

    n = np.array([1.0, 0.0, 0.0])
    m = np.array([0.0, 1.0, 0.0])

    assert tensor.youngs_modulus(n) == pytest.approx(205.71428571428572)
    assert tensor.shear_modulus(n, m) == pytest.approx(80.0)
    assert tensor.poisson_ratio(n, m) == pytest.approx(0.2857142857142857)
    assert tensor.linear_compressibility(n) == pytest.approx(1.0 / (3.0 * 160.0))


def test_directional_youngs_modulus_is_rotation_invariant_for_isotropic_cubic():
    tensor = ElasticTensor(isotropic_cubic_matrix(160.0, 80.0), crystal_system="cubic")
    expected = 205.71428571428572

    for direction in ([1, 0, 0], [1, 1, 0], [1, 1, 1], [2, -1, 3]):
        assert tensor.youngs_modulus(direction) == pytest.approx(expected)


def test_stability_reports_positive_definite_and_born_rules():
    stable = check_stability(isotropic_cubic_matrix(160.0, 80.0), crystal_system="cubic")

    assert stable.is_symmetric
    assert stable.is_invertible
    assert stable.is_positive_definite
    assert stable.born_stable
    assert stable.overall_stable
    assert stable.min_eigenvalue_gpa > 0

    unstable_cubic = isotropic_cubic_matrix(160.0, 80.0)
    unstable_cubic[3, 3] = -5.0
    unstable = check_stability(unstable_cubic, crystal_system="cubic")

    assert not unstable.is_positive_definite
    assert not unstable.born_stable
    assert not unstable.overall_stable
    assert any("C44 > 0" in item for item in unstable.failed_conditions)


def test_rejects_invalid_direction_and_non_orthogonal_transverse_direction():
    tensor = ElasticTensor(isotropic_cubic_matrix(), crystal_system="cubic")

    with pytest.raises(ValueError, match="non-zero"):
        tensor.youngs_modulus([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="orthogonal"):
        tensor.shear_modulus([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])


def test_roundtrip_compliance_inverts_stiffness_matrix():
    matrix = isotropic_cubic_matrix(160.0, 80.0)
    tensor = ElasticTensor(matrix, crystal_system="cubic")

    assert np.linalg.norm(matrix @ tensor.compliance_matrix - np.eye(6)) < 1e-10
    assert math.isfinite(tensor.condition_number)
