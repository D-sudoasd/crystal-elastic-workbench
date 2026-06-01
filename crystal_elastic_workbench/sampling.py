"""Sampling utilities for 1D, 2D, and 3D elastic property data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from crystal_elastic_workbench.core import ElasticTensor, normalize_vector


@dataclass(frozen=True)
class PlaneSlice:
    property_name: str
    plane_label: str
    angles_deg: np.ndarray
    directions: np.ndarray
    values: np.ndarray

    @property
    def min_value(self) -> float:
        return float(np.min(self.values))

    @property
    def max_value(self) -> float:
        return float(np.max(self.values))


@dataclass(frozen=True)
class DirectionalSurface:
    property_name: str
    theta: np.ndarray
    phi: np.ndarray
    directions: np.ndarray
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    min_value: float
    max_value: float
    min_direction: np.ndarray
    max_direction: np.ndarray


@dataclass(frozen=True)
class DirectionPath:
    property_name: str
    distance: np.ndarray
    directions: np.ndarray
    values: np.ndarray
    tick_positions: list[float]
    tick_labels: list[str]


def plane_basis(plane: str | Iterable[float]) -> tuple[str, np.ndarray, np.ndarray]:
    if isinstance(plane, str):
        key = plane.lower().strip()
        if key == "xy":
            return "xy", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        if key == "xz":
            return "xz", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        if key == "yz":
            return "yz", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        raise ValueError("plane must be one of 'xy', 'xz', 'yz' or a normal vector.")

    normal = normalize_vector(plane, name="plane normal")
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return f"normal_{normal[0]:.3f}_{normal[1]:.3f}_{normal[2]:.3f}", u, v


def sample_plane(
    tensor: ElasticTensor,
    *,
    property_name: str = "young",
    plane: str | Iterable[float] = "xy",
    angle_count: int = 361,
    transverse_mode: str = "mean",
) -> PlaneSlice:
    if angle_count < 3:
        raise ValueError("angle_count must be at least 3.")
    plane_label, u, v = plane_basis(plane)
    angles = np.linspace(0.0, 360.0, angle_count)
    radians = np.deg2rad(angles)
    directions = np.cos(radians)[:, None] * u + np.sin(radians)[:, None] * v
    values = np.array(
        [
            tensor.directional_property(
                direction,
                property_name=property_name,
                transverse_mode=transverse_mode,
            )
            for direction in directions
        ],
        dtype=float,
    )
    return PlaneSlice(
        property_name=property_name,
        plane_label=plane_label,
        angles_deg=angles,
        directions=directions,
        values=values,
    )


def sample_sphere(
    tensor: ElasticTensor,
    *,
    property_name: str = "young",
    theta_count: int = 37,
    phi_count: int = 73,
    transverse_mode: str = "mean",
) -> DirectionalSurface:
    if theta_count < 3:
        raise ValueError("theta_count must be at least 3.")
    if phi_count < 4:
        raise ValueError("phi_count must be at least 4.")

    theta = np.linspace(0.0, np.pi, theta_count)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_count)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.empty(theta_grid.shape + (3,), dtype=float)
    directions[..., 0] = np.sin(theta_grid) * np.cos(phi_grid)
    directions[..., 1] = np.sin(theta_grid) * np.sin(phi_grid)
    directions[..., 2] = np.cos(theta_grid)

    values = np.empty(theta_grid.shape, dtype=float)
    for index in np.ndindex(theta_grid.shape):
        values[index] = tensor.directional_property(
            directions[index],
            property_name=property_name,
            transverse_mode=transverse_mode,
        )

    min_index = np.unravel_index(int(np.argmin(values)), values.shape)
    max_index = np.unravel_index(int(np.argmax(values)), values.shape)

    x = values * directions[..., 0]
    y = values * directions[..., 1]
    z = values * directions[..., 2]
    return DirectionalSurface(
        property_name=property_name,
        theta=theta_grid,
        phi=phi_grid,
        directions=directions,
        values=values,
        x=x,
        y=y,
        z=z,
        min_value=float(values[min_index]),
        max_value=float(values[max_index]),
        min_direction=directions[min_index].copy(),
        max_direction=directions[max_index].copy(),
    )


def _coerce_path_point(point) -> tuple[str, np.ndarray]:
    if isinstance(point, tuple) and len(point) == 2:
        label, vector = point
        return str(label), normalize_vector(vector)
    vector = normalize_vector(point)
    label = f"[{vector[0]:.3g},{vector[1]:.3g},{vector[2]:.3g}]"
    return label, vector


def sample_direction_path(
    tensor: ElasticTensor,
    *,
    property_name: str = "young",
    points,
    points_per_segment: int = 41,
    transverse_mode: str = "mean",
) -> DirectionPath:
    """Sample a piecewise path between user-provided crystallographic directions."""

    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least 2.")
    coerced = [_coerce_path_point(point) for point in points]
    if len(coerced) < 2:
        raise ValueError("At least two path points are required.")

    labels = [label for label, _ in coerced]
    vectors = [vector for _, vector in coerced]
    sampled_directions: list[np.ndarray] = []
    distances: list[float] = []
    tick_positions: list[float] = [0.0]
    cumulative = 0.0

    for segment_index, (start, end) in enumerate(zip(vectors[:-1], vectors[1:])):
        dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
        segment_length = float(np.arccos(dot))
        fractions = np.linspace(0.0, 1.0, points_per_segment)
        if segment_index > 0:
            fractions = fractions[1:]
        for fraction in fractions:
            vector = (1.0 - fraction) * start + fraction * end
            direction = normalize_vector(vector)
            sampled_directions.append(direction)
            distances.append(cumulative + segment_length * float(fraction))
        cumulative += segment_length
        tick_positions.append(cumulative)

    directions = np.asarray(sampled_directions, dtype=float)
    values = np.array(
        [
            tensor.directional_property(
                direction,
                property_name=property_name,
                transverse_mode=transverse_mode,
            )
            for direction in directions
        ],
        dtype=float,
    )
    return DirectionPath(
        property_name=property_name,
        distance=np.asarray(distances, dtype=float),
        directions=directions,
        values=values,
        tick_positions=tick_positions,
        tick_labels=labels,
    )
