"""Desktop GUI style and platform setup."""

from __future__ import annotations

import os
from pathlib import Path


GUI_STYLE_SHEET = """
QWidget {
    font-family: "Segoe UI", "Arial", "Microsoft YaHei UI", sans-serif;
    font-size: 10pt;
    color: #202124;
}
QGroupBox {
    border: 1px solid #d8dde6;
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    border: 1px solid #c9d0dc;
    border-radius: 5px;
    padding: 6px 10px;
    background: #ffffff;
}
QPushButton:hover {
    background: #f2f6fb;
}
QPushButton#primaryButton {
    background: #255f9e;
    border-color: #255f9e;
    color: #ffffff;
    font-weight: 600;
}
QLabel#statusBanner, QLabel#dashboardBanner {
    border: 1px solid #bfd3ea;
    border-radius: 6px;
    padding: 8px 10px;
    background: #eef6ff;
    color: #174a7c;
    font-weight: 600;
}
QLabel#statusChip {
    border: 1px solid #d8dde6;
    border-radius: 5px;
    padding: 5px 8px;
    background: #ffffff;
}
QLabel#metricTitle, QLabel#mutedLabel {
    color: #687385;
}
QLabel#metricValue {
    font-size: 14pt;
    font-weight: 700;
    color: #1f2937;
}
QTableWidget {
    gridline-color: #d8dde6;
    alternate-background-color: #f7f9fc;
    selection-background-color: #dbeafe;
}
"""


def configure_platform_fonts() -> None:
    if os.name != "nt":
        return
    windows_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if windows_font_dir.exists():
        os.environ.setdefault("QT_QPA_FONTDIR", str(windows_font_dir))
