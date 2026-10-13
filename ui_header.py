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


NEXT_STYLE = ("QPushButton { background:#2f6fd6; color:white; font-weight:700; font-size:10.5pt; "
              "padding:7px 16px; border-radius:6px; border:1px solid #5a92ee; }"
              "QPushButton:hover { background:#3b7ff0; }"
              "QPushButton:disabled { background:#2a2f3a; color:#6c7486; border:1px solid #363c48; }")

BACK_STYLE = ("QPushButton { background:#262b36; color:#c3cad8; font-size:9.5pt; padding:6px 12px; "
              "border-radius:6px; border:1px solid #3a4252; }"
              "QPushButton:hover { background:#2f3644; color:#e4e9f2; }")


class AppHeader(QWidget):
    def __init__(self, page_title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("appHeader")
        self.setStyleSheet("QWidget#appHeader { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                           "stop:0 #10203a, stop:1 #16161c); border-bottom: 1px solid #2c3e5c; }")
        from PySide6.QtWidgets import QGridLayout
        g = QGridLayout(self)
        g.setContentsMargins(10, 6, 10, 6)
        g.setHorizontalSpacing(8)
        # left: "◀ back" to the page you came from
        self.left = QHBoxLayout()
        self.left.setSpacing(6)
        lw = QWidget()
        lw.setStyleSheet("background:transparent;")
        lw.setLayout(self.left)
        g.addWidget(lw, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        # centre: name and page title
        mid = QWidget()
        mid.setStyleSheet("background:transparent;")
        v = QVBoxLayout(mid)
        v.setContentsMargins(0, 0, 0, 0)
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
        g.addWidget(mid, 0, 1)
        self.right = QHBoxLayout()
        rw = QWidget()
        rw.setStyleSheet("background:transparent;")
        rw.setLayout(self.right)
        g.addWidget(rw, 0, 2, Qt.AlignRight | Qt.AlignVCenter)
        g.setColumnStretch(0, 1)          # equal side columns keep the title centred
        g.setColumnStretch(2, 1)

    def set_page(self, title: str):
        self.page.setText(title)

    def add_next(self, button, text=None):
        """Show a 'next step' button at the top right, with a right arrow - like turning a page."""
        if text:
            button.setText(f"{text}  ➜")
        button.setStyleSheet(NEXT_STYLE)
        button.setCursor(Qt.PointingHandCursor)
        self.right.addWidget(button)
        return button

    def add_back(self, text, callback):
        """A quieter '◀ previous page' button at the top left."""
        from PySide6.QtWidgets import QPushButton
        b = QPushButton(f"◀  {text}")
        b.setStyleSheet(BACK_STYLE)
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip(f"Go back to the {text} window (nothing here is closed or lost)")
        b.clicked.connect(callback)
        self.left.addWidget(b)
        self.btn_back = b
        return b


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
    header.btn_back = None
    # "◀ back" goes to whichever SubZeroAstro window opened this one
    prev = window.parentWidget()
    prev = prev.window() if prev is not None else None
    prev_header = getattr(prev, "app_header", None) if prev is not None else None
    if prev_header is not None:
        def go_back():
            if prev.isMinimized():
                prev.showNormal()
            prev.show()
            prev.raise_()
            prev.activateWindow()
        header.add_back(prev_header.page.text(), go_back)
    return header
