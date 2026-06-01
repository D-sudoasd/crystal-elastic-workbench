"""Elastic stability checks and Born criteria for common crystal systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    is_symmetric: bool
    is_invertible: bool
    is_positive_definite: bool
    born_stable: bool
    overall_stable: bool
    condition_number: float
    min_eigenvalue_gpa: float
    failed_conditions: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _condition(label: str, value: bool, failed: list[str]) -> None:
    if not bool(value):
        failed.append(label)


def _born_failures(c: np.ndarray, crystal_system: str) -> tuple[list[str], list[str]]:
    system = crystal_system.lower().strip()
    failed: list[str] = []
    warnings: list[str] = []

    c11, c22, c33 = c[0, 0], c[1, 1], c[2, 2]
    c12, c13, c23 = c[0, 1], c[0, 2], c[1, 2]
    c44, c55, c66 = c[3, 3], c[4, 4], c[5, 5]

    if system == "cubic":
        _condition("C11 - C12 > 0", c11 - c12 > 0, failed)
        _condition("C11 + 2*C12 > 0", c11 + 2.0 * c12 > 0, failed)
        _condition("C44 > 0", c44 > 0, failed)
    elif system == "hexagonal":
        _condition("C44 > 0", c44 > 0, failed)
        _condition("C11 > |C12|", c11 > abs(c12), failed)
        _condition("(C11 + C12)*C33 > 2*C13^2", (c11 + c12) * c33 > 2.0 * c13**2, failed)
    elif system == "tetragonal":
        _condition("C11 > |C12|", c11 > abs(c12), failed)
        _condition("2*C13^2 < C33*(C11 + C12)", 2.0 * c13**2 < c33 * (c11 + c12), failed)
        _condition("C44 > 0", c44 > 0, failed)
        _condition("C66 > 0", c66 > 0, failed)
    elif system == "orthorhombic":
        _condition("C11 > 0", c11 > 0, failed)
        _condition("C22 > 0", c22 > 0, failed)
        _condition("C33 > 0", c33 > 0, failed)
        _condition("C44 > 0", c44 > 0, failed)
        _condition("C55 > 0", c55 > 0, failed)
        _condition("C66 > 0", c66 > 0, failed)
        _condition("C11 + C22 - 2*C12 > 0", c11 + c22 - 2.0 * c12 > 0, failed)
        _condition("C11 + C33 - 2*C13 > 0", c11 + c33 - 2.0 * c13 > 0, failed)
        _condition("C22 + C33 - 2*C23 > 0", c22 + c33 - 2.0 * c23 > 0, failed)
        _condition(
            "C11 + C22 + C33 + 2*(C12 + C13 + C23) > 0",
            c11 + c22 + c33 + 2.0 * (c12 + c13 + c23) > 0,
            failed,
        )
    elif system in {"trigonal", "rhombohedral"}:
        _condition("C11 > |C12|", c11 > abs(c12), failed)
        _condition("C44 > 0", c44 > 0, failed)
        _condition("(C11 + C12)*C33 > 2*C13^2", (c11 + c12) * c33 > 2.0 * c13**2, failed)
        _condition("(C11 - C12)*C44 > 2*C14^2", (c11 - c12) * c44 > 2.0 * c[0, 3] ** 2, failed)
        warnings.append("Trigonal criteria assume the common C14 convention in Voigt position C14.")
    elif system in {"monoclinic", "triclinic"}:
        warnings.append(
            f"No compact Born-criteria shortcut is applied for {system}; positive definiteness is used."
        )
    else:
        warnings.append(f"Unknown crystal system '{crystal_system}'; positive definiteness is used.")

    return failed, warnings


def check_stability(
    stiffness_matrix: Iterable[Iterable[float]],
    *,
    crystal_system: str = "triclinic",
    symmetry_tolerance: float = 1e-8,
    condition_warning_threshold: float = 1e10,
) -> StabilityResult:
    c = np.asarray(stiffness_matrix, dtype=float)
    if c.shape != (6, 6):
        raise ValueError("stiffness_matrix must be a 6x6 matrix in Voigt notation.")
    if not np.all(np.isfinite(c)):
        raise ValueError("stiffness_matrix must contain finite values.")

    is_symmetric = bool(np.allclose(c, c.T, atol=symmetry_tolerance, rtol=0.0))
    c_symmetric = 0.5 * (c + c.T)
    condition_number = float(np.linalg.cond(c_symmetric))

    warnings: list[str] = []
    if not is_symmetric:
        warnings.append("Cij is not symmetric within tolerance; checks use the symmetrized matrix.")
    if condition_number > condition_warning_threshold:
        warnings.append(f"Condition number is high ({condition_number:.3e}); inversion may be unstable.")

    try:
        np.linalg.inv(c_symmetric)
        is_invertible = True
    except np.linalg.LinAlgError:
        is_invertible = False

    eigenvalues = np.linalg.eigvalsh(c_symmetric)
    min_eigenvalue = float(np.min(eigenvalues))
    is_positive_definite = bool(np.all(eigenvalues > 0.0))

    failed_conditions, born_warnings = _born_failures(c_symmetric, crystal_system)
    warnings.extend(born_warnings)
    born_stable = len(failed_conditions) == 0
    overall_stable = bool(is_symmetric and is_invertible and is_positive_definite and born_stable)

    return StabilityResult(
        is_symmetric=is_symmetric,
        is_invertible=is_invertible,
        is_positive_definite=is_positive_definite,
        born_stable=born_stable,
        overall_stable=overall_stable,
        condition_number=condition_number,
        min_eigenvalue_gpa=min_eigenvalue,
        failed_conditions=failed_conditions,
        warnings=warnings,
    )
