import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _table():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystal_elastic_workbench.cij_matrix_table import CijMatrixTable

    app = QApplication.instance() or QApplication([])
    table = CijMatrixTable()
    return app, table


def test_single_cell_edit_updates_symmetric_mirror_cell():
    app, table = _table()

    table.item(0, 1).setText("61.3")

    assert table.item(1, 0).text() == "61.3"
    table.close()
    app.processEvents()


def test_set_matrix_does_not_force_symmetry_for_full_import():
    app, table = _table()
    matrix = np.zeros((6, 6), dtype=float)
    matrix[0, 1] = 11.0
    matrix[1, 0] = 22.0

    table.set_matrix(matrix)

    assert table.item(0, 1).text() == "11"
    assert table.item(1, 0).text() == "22"
    assert np.allclose(table.matrix(), matrix)
    table.close()
    app.processEvents()


def test_paste_text_at_selection_updates_block_and_symmetric_mirrors():
    app, table = _table()
    table.setCurrentCell(0, 1)

    table.paste_text_at_selection("7\t8\n9\t10")

    assert table.item(0, 1).text() == "7"
    assert table.item(0, 2).text() == "8"
    assert table.item(1, 1).text() == "9"
    assert table.item(1, 2).text() == "10"
    assert table.item(1, 0).text() == "7"
    assert table.item(2, 0).text() == "8"
    assert table.item(2, 1).text() == "10"
    table.close()
    app.processEvents()


def test_paste_full_matrix_from_origin_preserves_asymmetric_source_values():
    app, table = _table()
    matrix = np.zeros((6, 6), dtype=float)
    matrix[0, 1] = 11.0
    matrix[1, 0] = 22.0
    text = "\n".join("\t".join(f"{value:g}" for value in row) for row in matrix)
    table.setCurrentCell(0, 0)

    table.paste_text_at_selection(text)

    assert table.item(0, 1).text() == "11"
    assert table.item(1, 0).text() == "22"
    assert np.allclose(table.matrix(), matrix)
    table.close()
    app.processEvents()


def test_paste_text_at_selection_rejects_overflow():
    app, table = _table()
    table.setCurrentCell(5, 5)

    with pytest.raises(ValueError, match="does not fit"):
        table.paste_text_at_selection("1\t2")

    table.close()
    app.processEvents()


def test_copy_selection_text_returns_tab_separated_block():
    app, table = _table()
    from PySide6.QtWidgets import QTableWidgetSelectionRange

    table.set_matrix(np.arange(36, dtype=float).reshape(6, 6))
    table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)

    assert table.copy_selection_text() == "0\t1\n6\t7"
    table.close()
    app.processEvents()


def test_zero_selection_clears_values_and_symmetric_mirrors():
    app, table = _table()
    matrix = np.ones((6, 6), dtype=float)
    table.set_matrix(matrix)
    table.item(0, 2).setSelected(True)

    table.zero_selection()

    assert table.item(0, 2).text() == "0"
    assert table.item(2, 0).text() == "0"
    table.close()
    app.processEvents()


def test_delete_key_clears_selected_cells_and_symmetric_mirrors():
    app, table = _table()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    table.set_matrix(np.ones((6, 6), dtype=float))
    table.item(0, 3).setSelected(True)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Delete, Qt.KeyboardModifier.NoModifier)
    table.keyPressEvent(event)

    assert table.item(0, 3).text() == "0"
    assert table.item(3, 0).text() == "0"
    assert event.isAccepted()
    table.close()
    app.processEvents()


def test_ctrl_c_key_copies_selected_cells_to_clipboard():
    app, table = _table()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent
    from PySide6.QtWidgets import QTableWidgetSelectionRange

    table.set_matrix(np.arange(36, dtype=float).reshape(6, 6))
    table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_C, Qt.KeyboardModifier.ControlModifier)
    table.keyPressEvent(event)

    assert QGuiApplication.clipboard().text() == "0\t1\n6\t7"
    assert event.isAccepted()
    table.close()
    app.processEvents()


def test_ctrl_v_key_pastes_clipboard_block_at_current_cell():
    app, table = _table()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent

    table.setCurrentCell(0, 1)
    QGuiApplication.clipboard().setText("4\t5")

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_V, Qt.KeyboardModifier.ControlModifier)
    table.keyPressEvent(event)

    assert table.item(0, 1).text() == "4"
    assert table.item(0, 2).text() == "5"
    assert table.item(1, 0).text() == "4"
    assert table.item(2, 0).text() == "5"
    assert event.isAccepted()
    table.close()
    app.processEvents()
