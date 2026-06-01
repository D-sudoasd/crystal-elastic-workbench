"""Small testable services used by the desktop GUI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_clipboard_matrix(text: str) -> np.ndarray:
    """Parse clipboard text into a 6 x 6 Cij matrix."""

    rows: list[list[float]] = []
    for line in text.strip().splitlines():
        cleaned = line.replace(",", " ").replace(";", " ")
        if not cleaned.strip():
            continue
        rows.append([float(value) for value in cleaned.split()])
    if len(rows) == 1 and len(rows[0]) == 36:
        return np.asarray(rows[0], dtype=float).reshape(6, 6)
    if len(rows) == 6 and all(len(row) == 6 for row in rows):
        return np.asarray(rows, dtype=float)
    raise ValueError("Clipboard must contain a 6 x 6 numeric Cij matrix.")


def parse_numeric_block(text: str) -> np.ndarray:
    """Parse rectangular spreadsheet-style numeric text."""

    rows: list[list[float]] = []
    for line in text.strip().splitlines():
        cleaned = line.replace(",", " ").replace(";", " ")
        if not cleaned.strip():
            continue
        try:
            rows.append([float(value) for value in cleaned.split()])
        except ValueError as exc:
            raise ValueError("Pasted block must contain only numeric values.") from exc

    if not rows:
        raise ValueError("Pasted block is empty.")

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("Pasted block must be rectangular.")
    return np.asarray(rows, dtype=float)


def matrix_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Extract the first complete numeric 6 x 6 matrix block from a data frame."""

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    matrix = numeric.dropna(how="all").dropna(axis=1, how="all").iloc[:6, :6].to_numpy(dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("Imported table must contain a complete numeric 6 x 6 Cij matrix.")
    return matrix
