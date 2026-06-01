import numpy as np
import pandas as pd
import pytest

from crystal_elastic_workbench.gui_services import matrix_from_frame, parse_clipboard_matrix, parse_numeric_block


def test_parse_clipboard_matrix_accepts_tabular_6_by_6_text():
    matrix = np.arange(36, dtype=float).reshape(6, 6)
    text = "\n".join("\t".join(f"{value:g}" for value in row) for row in matrix)

    parsed = parse_clipboard_matrix(text)

    assert np.allclose(parsed, matrix)


def test_parse_clipboard_matrix_accepts_flat_36_value_text():
    parsed = parse_clipboard_matrix(" ".join(str(value) for value in range(36)))

    assert parsed.shape == (6, 6)
    assert parsed[5, 5] == 35


def test_parse_clipboard_matrix_rejects_wrong_shape():
    with pytest.raises(ValueError, match="6 x 6"):
        parse_clipboard_matrix("1 2 3\n4 5 6")


def test_parse_numeric_block_accepts_rectangular_selection_text():
    block = parse_numeric_block("1\t2\n3,4")

    assert block.shape == (2, 2)
    assert np.allclose(block, [[1.0, 2.0], [3.0, 4.0]])


def test_parse_numeric_block_rejects_ragged_selection_text():
    with pytest.raises(ValueError, match="rectangular"):
        parse_numeric_block("1\t2\n3")


def test_parse_numeric_block_rejects_non_numeric_selection_text():
    with pytest.raises(ValueError, match="numeric"):
        parse_numeric_block("1\tbad")


def test_matrix_from_frame_extracts_first_numeric_6_by_6_block():
    frame = pd.DataFrame(np.arange(36, dtype=float).reshape(6, 6))

    matrix = matrix_from_frame(frame)

    assert matrix.shape == (6, 6)
    assert matrix[0, 0] == 0
    assert matrix[5, 5] == 35


def test_matrix_from_frame_rejects_incomplete_numeric_input():
    frame = pd.DataFrame([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="6 x 6"):
        matrix_from_frame(frame)
