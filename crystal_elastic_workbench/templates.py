"""Crystal-system input templates for stiffness matrices."""

from __future__ import annotations

from typing import Mapping

import numpy as np


CRYSTAL_SYSTEMS = (
    "cubic",
    "hexagonal",
    "tetragonal",
    "orthorhombic",
    "trigonal",
    "rhombohedral",
    "monoclinic",
    "triclinic",
)


def _get(constants: Mapping[str, float], name: str, *, default: float | None = None) -> float:
    if name in constants:
        return float(constants[name])
    if default is not None:
        return float(default)
    raise ValueError(f"Missing required elastic constant {name}.")


def _blank() -> np.ndarray:
    return np.zeros((6, 6), dtype=float)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return matrix + matrix.T - np.diag(np.diag(matrix))


def apply_crystal_template(crystal_system: str, constants: Mapping[str, float]) -> np.ndarray:
    """Build a 6x6 Voigt stiffness matrix from independent constants.

    The templates are practical input helpers, not a substitute for checking the
    exact point-group convention used by a source paper. Trigonal and monoclinic
    templates follow common conventions documented in the README.
    """

    system = crystal_system.lower().strip()
    if system == "rhombohedral":
        system = "trigonal"

    if system == "cubic":
        c11 = _get(constants, "C11")
        c12 = _get(constants, "C12")
        c44 = _get(constants, "C44")
        c = _blank()
        c[0, 0] = c[1, 1] = c[2, 2] = c11
        c[0, 1] = c[0, 2] = c[1, 2] = c12
        c[3, 3] = c[4, 4] = c[5, 5] = c44
        return _symmetrize(c)

    if system == "hexagonal":
        c11 = _get(constants, "C11")
        c12 = _get(constants, "C12")
        c13 = _get(constants, "C13")
        c33 = _get(constants, "C33")
        c44 = _get(constants, "C44")
        c66 = _get(constants, "C66", default=0.5 * (c11 - c12))
        c = _blank()
        c[0, 0] = c[1, 1] = c11
        c[2, 2] = c33
        c[0, 1] = c12
        c[0, 2] = c[1, 2] = c13
        c[3, 3] = c[4, 4] = c44
        c[5, 5] = c66
        return _symmetrize(c)

    if system == "tetragonal":
        c = _blank()
        c11 = _get(constants, "C11")
        c[0, 0] = c[1, 1] = c11
        c[2, 2] = _get(constants, "C33")
        c[0, 1] = _get(constants, "C12")
        c[0, 2] = c[1, 2] = _get(constants, "C13")
        c[3, 3] = c[4, 4] = _get(constants, "C44")
        c[5, 5] = _get(constants, "C66")
        return _symmetrize(c)

    if system == "orthorhombic":
        c = _blank()
        for i, key in enumerate(("C11", "C22", "C33", "C44", "C55", "C66")):
            c[i, i] = _get(constants, key)
        c[0, 1] = _get(constants, "C12")
        c[0, 2] = _get(constants, "C13")
        c[1, 2] = _get(constants, "C23")
        return _symmetrize(c)

    if system == "trigonal":
        c11 = _get(constants, "C11")
        c12 = _get(constants, "C12")
        c13 = _get(constants, "C13")
        c14 = _get(constants, "C14", default=0.0)
        c33 = _get(constants, "C33")
        c44 = _get(constants, "C44")
        c66 = _get(constants, "C66", default=0.5 * (c11 - c12))
        c = _blank()
        c[0, 0] = c[1, 1] = c11
        c[2, 2] = c33
        c[0, 1] = c12
        c[0, 2] = c[1, 2] = c13
        c[0, 3] = c14
        c[1, 3] = -c14
        c[3, 3] = c[4, 4] = c44
        c[4, 5] = c14
        c[5, 5] = c66
        return _symmetrize(c)

    if system == "monoclinic":
        c = _blank()
        for i, key in enumerate(("C11", "C22", "C33", "C44", "C55", "C66")):
            c[i, i] = _get(constants, key)
        for row, col, key in (
            (0, 1, "C12"),
            (0, 2, "C13"),
            (1, 2, "C23"),
            (0, 4, "C15"),
            (1, 4, "C25"),
            (2, 4, "C35"),
            (3, 5, "C46"),
        ):
            c[row, col] = _get(constants, key, default=0.0)
        return _symmetrize(c)

    if system == "triclinic":
        c = _blank()
        for row in range(6):
            for col in range(row, 6):
                key = f"C{row + 1}{col + 1}"
                c[row, col] = _get(constants, key)
        return _symmetrize(c)

    raise ValueError(f"Unsupported crystal system: {crystal_system}")
