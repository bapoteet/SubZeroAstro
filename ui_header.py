"""
ui_header.py - the banner shown at the top of every SubZeroAstro window:

                 SubZeroAstro   by Brian A. Poteet
                  Astro Image Quality and Cull

It sits above the menus and toolbars, so it is always visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

APP_NAME = "SubZeroAstro"
AUTHOR = "Brian A. Poteet"

PAGE_CULL = "Astro Image Quality and Cull"
PAGE_STACK = "Calibrate and Stack"
PAGE_PREPARE = "Prepare for PixInsight"
PAGE_FINISH = "Finish the Image"


class AppHeader(QWidget):
    def __init__(self, page_title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("appHeader")
        self.setStyleSheet("QWidget#appHeader { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                           "stop:0 #10203a, stop:1 #16161c); border-bottom: 1px solid #2c3e5c; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        self.name = QLabel(APP_NAME)
        self.name.setStyleSheet("color:#cfe3ff; font-size:20pt; font-weight:700; letter-spacing:1px;"
                                "background:transparent; border:none;")
        row.addWidget(self.name, 0, Qt.AlignBottom)
        self.by = QLabel(f"by {AUTHOR}")
        self.by.setStyleSheet("color:#8a93a6; font-size:8pt; font-style:italic; padding-bottom:5px;"
                              "background:transparent; border:none;")
        row.addWidget(self.by, 0, Qt.AlignBottom)
        row.addStretch(1)
        v.addLayout(row)
        self.page = QLabel(page_title)
        self.page.setAlignment(Qt.AlignCenter)
        self.page.setStyleSheet("color:#9fc0ec; font-size:11pt; font-weight:600; background:transparent;"
                                "border:none;")
        v.addWidget(self.page)

    def set_page(self, title: str):
        self.page.setText(title)


def install_header(window, page_title: str, with_menu: bool = False) -> AppHeader:
    """Put the banner at the very top of a QMainWindow, above its menu bar and toolbars.
    Call this BEFORE building menus: Qt deletes a menu bar that is replaced, so the window
    gets a fresh menu bar under the banner and window.menuBar() returns that one."""
    from PySide6.QtWidgets import QMenuBar
    header = AppHeader(page_title)
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(header)
    mb = QMenuBar(box)
    lay.addWidget(mb)
    mb.setVisible(with_menu)
    window.setMenuWidget(box)
    window.menuBar = lambda: mb          # menus are added under the banner
    window.app_header = header
    return header
