"""Numerical core for elastic tensor analysis.

The package uses the standard Voigt order [11, 22, 33, 23, 13, 12].
Stiffness matrices map stress to engineering strain through the compliance
matrix: [e11, e22, e33, 2e23, 2e13, 2e12] = S [s11, s22, s33, s23, s13, s12].

Directional quantities are computed by applying physical stress tensors and
converting the engineering strain vector back to a symmetric strain tensor.
This avoids the common factor-of-two/four mistakes in shear terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


VOIGT_LABELS = ("11", "22", "33", "23", "13", "12")


@dataclass(frozen=True)
class PolycrystalSummary:
    """Voigt/Reuss/Hill scalar elastic parameters."""

    bulk_voigt_gpa: float
    bulk_reuss_gpa: float
    bulk_hill_gpa: float
    shear_voigt_gpa: float
    shear_reuss_gpa: float
    shear_hill_gpa: float
    young_hill_gpa: float
    poisson_hill: float
    pugh_ratio: float
    universal_anisotropy: float
    bulk_anisotropy_percent: float
    shear_anisotropy_percent: float
    cauchy_pressure_gpa: float | None = None
    zener_anisotropy: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def normalize_vector(vector: Iterable[float], *, name: str = "direction") -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError(f"{name} must be non-zero.")
    return arr / norm


def assert_orthogonal(n: np.ndarray, m: np.ndarray, *, tolerance: float = 1e-8) -> None:
    dot = float(np.dot(n, m))
    if abs(dot) > tolerance:
        raise ValueError("Transverse direction m must be orthogonal to direction n.")


def stress_tensor_to_voigt(stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=float).reshape(3, 3)
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ],
        dtype=float,
    )


def strain_voigt_to_tensor(strain: np.ndarray) -> np.ndarray:
    strain = np.asarray(strain, dtype=float).reshape(6)
    return np.array(
        [
            [strain[0], 0.5 * strain[5], 0.5 * strain[4]],
            [0.5 * strain[5], strain[1], 0.5 * strain[3]],
            [0.5 * strain[4], 0.5 * strain[3], strain[2]],
        ],
        dtype=float,
    )


def _safe_divide(numerator: float, denominator: float, label: str) -> float:
    if abs(denominator) <= 1e-14:
        raise ValueError(f"Cannot compute {label}; denominator is too close to zero.")
    return float(numerator / denominator)


class ElasticTensor:
    """A 6x6 elastic stiffness matrix with directional property methods."""

    def __init__(
        self,
        stiffness_matrix: Iterable[Iterable[float]],
        *,
        crystal_system: str = "triclinic",
        unit: str = "GPa",
        material_name: str = "Untitled",
        symmetrize: bool = False,
    ) -> None:
        matrix = np.asarray(stiffness_matrix, dtype=float)
        if matrix.shape != (6, 6):
            raise ValueError("stiffness_matrix must be a 6x6 matrix in Voigt notation.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("stiffness_matrix must contain only finite numbers.")
        if symmetrize:
            matrix = 0.5 * (matrix + matrix.T)

        self.stiffness_matrix = matrix
        self.crystal_system = crystal_system.lower().strip()
        self.unit = unit
        self.material_name = material_name
        self.compliance_matrix = np.linalg.inv(matrix)
        self.condition_number = float(np.linalg.cond(matrix))

    def strain_from_stress_tensor(self, stress_tensor: np.ndarray) -> np.ndarray:
        stress_voigt = stress_tensor_to_voigt(stress_tensor)
        strain_voigt = self.compliance_matrix @ stress_voigt
        return strain_voigt_to_tensor(strain_voigt)

    def polycrystalline_summary(self) -> PolycrystalSummary:
        c = self.stiffness_matrix
        s = self.compliance_matrix

        c11, c22, c33 = c[0, 0], c[1, 1], c[2, 2]
        c12, c13, c23 = c[0, 1], c[0, 2], c[1, 2]
        c44, c55, c66 = c[3, 3], c[4, 4], c[5, 5]

        s11, s22, s33 = s[0, 0], s[1, 1], s[2, 2]
        s12, s13, s23 = s[0, 1], s[0, 2], s[1, 2]
        s44, s55, s66 = s[3, 3], s[4, 4], s[5, 5]

        bulk_voigt = (c11 + c22 + c33 + 2.0 * (c12 + c13 + c23)) / 9.0
        shear_voigt = (
            c11 + c22 + c33 - c12 - c13 - c23 + 3.0 * (c44 + c55 + c66)
        ) / 15.0

        bulk_reuss = _safe_divide(
            1.0,
            s11 + s22 + s33 + 2.0 * (s12 + s13 + s23),
            "Reuss bulk modulus",
        )
        shear_reuss = _safe_divide(
            15.0,
            4.0 * (s11 + s22 + s33)
            - 4.0 * (s12 + s13 + s23)
            + 3.0 * (s44 + s55 + s66),
            "Reuss shear modulus",
        )

        bulk_hill = 0.5 * (bulk_voigt + bulk_reuss)
        shear_hill = 0.5 * (shear_voigt + shear_reuss)
        young_hill = _safe_divide(
            9.0 * bulk_hill * shear_hill,
            3.0 * bulk_hill + shear_hill,
            "Hill Young's modulus",
        )
        poisson_hill = _safe_divide(
            3.0 * bulk_hill - 2.0 * shear_hill,
            2.0 * (3.0 * bulk_hill + shear_hill),
            "Hill Poisson ratio",
        )
        pugh_ratio = _safe_divide(bulk_hill, shear_hill, "Pugh ratio")
        universal_anisotropy = 5.0 * shear_voigt / shear_reuss + bulk_voigt / bulk_reuss - 6.0
        bulk_anisotropy = 100.0 * (bulk_voigt - bulk_reuss) / (bulk_voigt + bulk_reuss)
        shear_anisotropy = 100.0 * (shear_voigt - shear_reuss) / (shear_voigt + shear_reuss)

        cauchy_pressure = None
        zener = None
        if self.crystal_system == "cubic":
            cauchy_pressure = float(c12 - c44)
            zener = _safe_divide(2.0 * c44, c11 - c12, "Zener anisotropy")

        return PolycrystalSummary(
            bulk_voigt_gpa=float(bulk_voigt),
            bulk_reuss_gpa=float(bulk_reuss),
            bulk_hill_gpa=float(bulk_hill),
            shear_voigt_gpa=float(shear_voigt),
            shear_reuss_gpa=float(shear_reuss),
            shear_hill_gpa=float(shear_hill),
            young_hill_gpa=float(young_hill),
            poisson_hill=float(poisson_hill),
            pugh_ratio=float(pugh_ratio),
            universal_anisotropy=float(universal_anisotropy),
            bulk_anisotropy_percent=float(bulk_anisotropy),
            shear_anisotropy_percent=float(shear_anisotropy),
            cauchy_pressure_gpa=cauchy_pressure,
            zener_anisotropy=zener,
        )

    def youngs_modulus(self, direction: Iterable[float]) -> float:
        n = normalize_vector(direction)
        stress = np.outer(n, n)
        strain = self.strain_from_stress_tensor(stress)
        axial_strain = float(n @ strain @ n)
        return _safe_divide(1.0, axial_strain, "directional Young's modulus")

    def linear_compressibility(self, direction: Iterable[float]) -> float:
        n = normalize_vector(direction)
        stress = -np.eye(3)
        strain = self.strain_from_stress_tensor(stress)
        return float(-(n @ strain @ n))

    def shear_modulus(self, direction: Iterable[float], transverse: Iterable[float]) -> float:
        n = normalize_vector(direction, name="direction n")
        m = normalize_vector(transverse, name="transverse direction m")
        assert_orthogonal(n, m)
        stress = np.outer(n, m) + np.outer(m, n)
        strain = self.strain_from_stress_tensor(stress)
        engineering_shear = float(2.0 * (m @ strain @ n))
        return _safe_divide(1.0, engineering_shear, "directional shear modulus")

    def poisson_ratio(self, direction: Iterable[float], transverse: Iterable[float]) -> float:
        n = normalize_vector(direction, name="direction n")
        m = normalize_vector(transverse, name="transverse direction m")
        assert_orthogonal(n, m)
        stress = np.outer(n, n)
        strain = self.strain_from_stress_tensor(stress)
        axial = float(n @ strain @ n)
        transverse_strain = float(m @ strain @ m)
        return _safe_divide(-transverse_strain, axial, "directional Poisson ratio")

    def transverse_basis(self, direction: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        n = normalize_vector(direction)
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(n, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, reference)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return u, v

    def transverse_scan(
        self,
        direction: Iterable[float],
        *,
        property_name: str,
        samples: int = 72,
    ) -> dict[str, float]:
        n = normalize_vector(direction)
        u, v = self.transverse_basis(n)
        values = []
        for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
            m = np.cos(angle) * u + np.sin(angle) * v
            if property_name == "shear":
                values.append(self.shear_modulus(n, m))
            elif property_name == "poisson":
                values.append(self.poisson_ratio(n, m))
            else:
                raise ValueError("property_name must be 'shear' or 'poisson'.")
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    def directional_property(
        self,
        direction: Iterable[float],
        *,
        property_name: str,
        transverse_mode: str = "mean",
    ) -> float:
        name = property_name.lower()
        if name in {"young", "youngs", "e"}:
            return self.youngs_modulus(direction)
        if name in {"compressibility", "beta"}:
            return self.linear_compressibility(direction)
        if name in {"shear", "g"}:
            return self.transverse_scan(direction, property_name="shear")[transverse_mode]
        if name in {"poisson", "nu"}:
            return self.transverse_scan(direction, property_name="poisson")[transverse_mode]
        raise ValueError(f"Unsupported directional property: {property_name}")
