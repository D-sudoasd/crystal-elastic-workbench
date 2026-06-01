"""Bundled example stiffness matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crystal_elastic_workbench.templates import apply_crystal_template


@dataclass(frozen=True)
class ExampleMaterial:
    name: str
    crystal_system: str
    unit: str
    matrix: np.ndarray
    note: str


EXAMPLE_MATERIALS = {
    "Al cubic": ExampleMaterial(
        name="Al cubic",
        crystal_system="cubic",
        unit="GPa",
        matrix=apply_crystal_template("cubic", {"C11": 108.2, "C12": 61.3, "C44": 28.5}),
        note="Representative cubic aluminum stiffness constants; verify against your data source.",
    ),
    "Si cubic": ExampleMaterial(
        name="Si cubic",
        crystal_system="cubic",
        unit="GPa",
        matrix=apply_crystal_template("cubic", {"C11": 165.7, "C12": 63.9, "C44": 79.6}),
        note="Representative cubic silicon stiffness constants; verify temperature and source.",
    ),
    "MgO cubic": ExampleMaterial(
        name="MgO cubic",
        crystal_system="cubic",
        unit="GPa",
        matrix=apply_crystal_template("cubic", {"C11": 297.0, "C12": 95.0, "C44": 155.0}),
        note="Representative cubic MgO stiffness constants; intended as a GUI example.",
    ),
}
