"""Spreadsheet-style Qt table for 6 x 6 Cij matrix editing."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QHeaderView, QMessageBox, QTableWidget, QTableWidgetItem

from crystal_elastic_workbench.core import VOIGT_LABELS
from crystal_elastic_workbench.gui_services import parse_numeric_block


class CijMatrixTable(QTableWidget):
    """A compact 6 x 6 stiffness matrix table with Excel-like shortcuts."""

    def __init__(self, parent=None) -> None:
        super().__init__(6, 6, parent)
        self._bulk_update = False
        self._configure_table()
        self.set_matrix(np.zeros((6, 6), dtype=float))
        self.itemChanged.connect(self._sync_symmetric_item)

    def _configure_table(self) -> None:
        headers = [f"{label}" for label in VOIGT_LABELS]
        self.setMinimumHeight(210)
        self.setHorizontalHeaderLabels(headers)
        self.setVerticalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setMinimumSectionSize(34)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QTableWidget.ExtendedSelection)
        self.setSelectionBehavior(QTableWidget.SelectItems)

    def _format_value(self, value: float) -> str:
        return f"{float(value):.8g}"

    def _ensure_item(self, row: int, col: int) -> QTableWidgetItem:
        item = self.item(row, col)
        if item is None:
            item = QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, col, item)
        return item

    def _set_cell_text(self, row: int, col: int, text: str) -> None:
        item = self._ensure_item(row, col)
        item.setText(text)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        item.setBackground(QColor("white"))

    def _set_cell_value(self, row: int, col: int, value: float, *, sync_mirror: bool) -> None:
        text = self._format_value(value)
        previous = self._bulk_update
        self._bulk_update = True
        try:
            self._set_cell_text(row, col, text)
            if sync_mirror and row != col:
                self._set_cell_text(col, row, text)
        finally:
            self._bulk_update = previous

    def set_matrix(self, matrix: np.ndarray) -> None:
        array = np.asarray(matrix, dtype=float)
        if array.shape != (6, 6):
            raise ValueError("Cij matrix must be 6 x 6.")
        previous = self._bulk_update
        self._bulk_update = True
        try:
            for row in range(6):
                for col in range(6):
                    self._set_cell_text(row, col, self._format_value(array[row, col]))
        finally:
            self._bulk_update = previous

    def matrix(self) -> np.ndarray:
        matrix = np.zeros((6, 6), dtype=float)
        errors: list[str] = []
        for row in range(6):
            for col in range(6):
                item = self._ensure_item(row, col)
                text = item.text().strip()
                try:
                    matrix[row, col] = float(text) if text else 0.0
                    item.setBackground(QColor("white"))
                except ValueError:
                    item.setBackground(QColor("#ffd6d6"))
                    errors.append(f"C{VOIGT_LABELS[row]}-C{VOIGT_LABELS[col]}")
        if errors:
            raise ValueError("Non-numeric Cij cell(s): " + ", ".join(errors))
        return matrix

    def _sync_symmetric_item(self, item: QTableWidgetItem) -> None:
        if self._bulk_update:
            return
        row = item.row()
        col = item.column()
        text = item.text().strip()
        try:
            value = float(text) if text else 0.0
        except ValueError:
            item.setBackground(QColor("#ffd6d6"))
            return
        self._set_cell_value(row, col, value, sync_mirror=True)

    def _selection_anchor(self) -> tuple[int, int]:
        ranges = self.selectedRanges()
        if ranges:
            return ranges[0].topRow(), ranges[0].leftColumn()
        row = self.currentRow()
        col = self.currentColumn()
        return max(row, 0), max(col, 0)

    def paste_text_at_selection(self, text: str) -> None:
        block = parse_numeric_block(text)
        start_row, start_col = self._selection_anchor()
        row_count, col_count = block.shape
        if start_row + row_count > 6 or start_col + col_count > 6:
            raise ValueError("Pasted block does not fit inside the 6 x 6 Cij matrix.")
        if (start_row, start_col) == (0, 0) and block.shape == (6, 6):
            self.set_matrix(block)
            return
        for row_offset in range(row_count):
            for col_offset in range(col_count):
                self._set_cell_value(
                    start_row + row_offset,
                    start_col + col_offset,
                    float(block[row_offset, col_offset]),
                    sync_mirror=True,
                )

    def copy_selection_text(self) -> str:
        indexes = self.selectedIndexes()
        if not indexes:
            row = max(self.currentRow(), 0)
            col = max(self.currentColumn(), 0)
            indexes = [self.model().index(row, col)]
        rows = [index.row() for index in indexes]
        cols = [index.column() for index in indexes]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        copied_rows: list[str] = []
        for row in range(top, bottom + 1):
            values = [self._ensure_item(row, col).text().strip() for col in range(left, right + 1)]
            copied_rows.append("\t".join(values))
        return "\n".join(copied_rows)

    def zero_selection(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            row = self.currentRow()
            col = self.currentColumn()
            if row >= 0 and col >= 0:
                indexes = [self.model().index(row, col)]
        for index in indexes:
            self._set_cell_value(index.row(), index.column(), 0.0, sync_mirror=True)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            QGuiApplication.clipboard().setText(self.copy_selection_text())
            event.accept()
            return
        if event.matches(QKeySequence.Paste):
            try:
                self.paste_text_at_selection(QGuiApplication.clipboard().text())
            except ValueError as exc:
                QMessageBox.warning(self, "Paste Cij block failed", str(exc))
            event.accept()
            return
        if event.key() in {Qt.Key_Delete, Qt.Key_Backspace}:
            self.zero_selection()
            event.accept()
            return
        super().keyPressEvent(event)
