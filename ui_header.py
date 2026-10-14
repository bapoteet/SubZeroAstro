"""
ui_header.py - the banner shown at the top of every SubZeroAstro window:

                         SubZeroAstro
                       by Brian A. Poteet
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
PAGE_PREPARE = "Astro Image Processing"
PAGE_FINISH = "Finishing Touches"


import sys as _sys
_MAC = _sys.platform == "darwin"          # macOS draws the same point size smaller than Windows
_PT = (lambda x: round(x * 1.25, 1)) if _MAC else (lambda x: x)

NEXT_STYLE = ("QPushButton { background:#2f6fd6; color:white; font-weight:700; font-size:10.5pt; "
              "padding:7px 16px; border-radius:6px; border:1px solid #5a92ee; }"
              "QPushButton:hover { background:#3b7ff0; }"
              "QPushButton:disabled { background:#2a2f3a; color:#6c7486; border:1px solid #363c48; }")

RUN_STYLE = ("QPushButton { background:#2e8b57; color:white; font-weight:700; padding:6px 18px; "
             "border-radius:5px; border:1px solid #47a873; }"
             "QPushButton:hover { background:#36a065; }"
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
        # three centred lines: name / by author (very small) / page title
        self.name = QLabel(APP_NAME)
        self.name.setAlignment(Qt.AlignCenter)
        self.name.setStyleSheet(f"color:#cfe3ff; font-size:{_PT(20)}pt; font-weight:700; letter-spacing:1px;"
                                "background:transparent; border:none;")
        v.addWidget(self.name)
        self.by = QLabel(f"by {AUTHOR}")
        self.by.setAlignment(Qt.AlignCenter)
        self.by.setStyleSheet(f"color:#8a93a6; font-size:{_PT(7.5)}pt; font-style:italic; margin-top:-3px;"
                              "background:transparent; border:none;")
        v.addWidget(self.by)
        self.page = QLabel(page_title)
        self.page.setAlignment(Qt.AlignCenter)
        self.page.setStyleSheet(f"color:#9fc0ec; font-size:{_PT(11)}pt; font-weight:600; background:transparent;"
                                "border:none; margin-top:1px;")
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


def present(window, like=None, ms: int = 700):
    """Bring a page on screen like turning a page: same place and size as the page it was
    opened from, fading in. Never fails - a window that cannot fade just appears."""
    try:
        from PySide6.QtCore import QEasingCurve, QPropertyAnimation
        src = like.window() if like is not None else None
        if src is not None and src is not window and not window.isVisible():
            if src.isMaximized():
                window.setGeometry(src.normalGeometry())
                window.setWindowOpacity(0.0)
                window.showMaximized()
            else:
                window.setGeometry(src.geometry())
                window.setWindowOpacity(0.0)
                window.show()
        else:
            if window.isMinimized():
                window.showNormal()
            window.setWindowOpacity(0.0)
            window.show()
        window.raise_()
        window.activateWindow()
        anim = QPropertyAnimation(window, b"windowOpacity", window)
        anim.setDuration(ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)
        anim.finished.connect(lambda: window.setWindowOpacity(1.0))
        window._page_fade = anim
        anim.start()
    except Exception:  # noqa: BLE001
        window.setWindowOpacity(1.0)
        window.show()
        window.raise_()


def alive(w) -> bool:
    """The window still exists (not closed and deleted)."""
    if w is None:
        return False
    try:
        import shiboken6
        return shiboken6.isValid(w)
    except Exception:  # noqa: BLE001
        return True


def busy(w) -> bool:
    for attr in ("worker", "job"):
        j = getattr(w, attr, None)
        try:
            if j is not None and j.isRunning():
                return True
        except RuntimeError:
            pass
    return False


def leave(window, ms: int = 550):
    """Fade a page out and hide it (it stays alive, so Next can bring it back as it was)."""
    try:
        from PySide6.QtCore import QEasingCurve, QPropertyAnimation
        anim = QPropertyAnimation(window, b"windowOpacity", window)
        anim.setDuration(ms)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)

        def done():
            window.hide()
            window.setWindowOpacity(1.0)
        anim.finished.connect(done)
        window._page_fade = anim
        anim.start()
    except Exception:  # noqa: BLE001
        window.hide()


def open_page(owner, attr: str, key, make):
    """Show the next page. If you went Back from it and its input is unchanged (or it is still
    working), the same page comes back exactly as you left it; otherwise a fresh one opens."""
    from PySide6.QtCore import Qt
    old = getattr(owner, attr, None)
    if alive(old):
        if getattr(old, "_page_key", None) == key or busy(old):
            present(old, owner)
            return old
        old.close()
    w = make()
    w._page_key = key
    w.setAttribute(Qt.WA_DeleteOnClose, True)
    setattr(owner, attr, w)
    present(w, owner)
    return w


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
            # Windows keeps a page in front of the page that opened it, so going back means
            # stepping this page aside (it stays alive for Next) to reveal the one beneath
            if prev.isMinimized():
                prev.showNormal()
            prev.show()
            prev.activateWindow()
            leave(window)
        header.add_back(prev_header.page.text(), go_back)
    return header
