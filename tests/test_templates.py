import numpy as np

from crystal_elastic_workbench.examples import EXAMPLE_MATERIALS
from crystal_elastic_workbench.stability import check_stability
from crystal_elastic_workbench.templates import CRYSTAL_SYSTEMS, apply_crystal_template


def test_common_crystal_systems_are_available():
    assert {
        "cubic",
        "hexagonal",
        "tetragonal",
        "orthorhombic",
        "trigonal",
        "monoclinic",
        "triclinic",
    }.issubset(set(CRYSTAL_SYSTEMS))


def test_apply_cubic_template_fills_symmetry_related_entries():
    matrix = apply_crystal_template(
        "cubic",
        {
            "C11": 240.0,
            "C12": 140.0,
            "C44": 80.0,
        },
    )

    assert matrix.shape == (6, 6)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 0] == 240.0
    assert matrix[1, 1] == 240.0
    assert matrix[2, 2] == 240.0
    assert matrix[0, 1] == 140.0
    assert matrix[3, 3] == 80.0
    assert matrix[4, 4] == 80.0
    assert matrix[5, 5] == 80.0


def test_example_materials_include_stable_cubic_silicon():
    silicon = EXAMPLE_MATERIALS["Si cubic"]
    result = check_stability(silicon.matrix, crystal_system=silicon.crystal_system)

    assert silicon.unit == "GPa"
    assert silicon.matrix.shape == (6, 6)
    assert result.overall_stable
