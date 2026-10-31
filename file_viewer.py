"""
file_viewer.py - a simple FITS / XISF viewer: the picture (auto-stretched, in colour when it is colour), its size,
levels and the main capture facts, and every header card with a search box. Zoom with the wheel, drag to move,
double-click to fit; the arrow buttons (or Page Up / Page Down) step through the other FITS / XISF files in the
same folder. Nothing in the file is changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QFileDialog, QFrame,
                               QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

import fits_core as core

EXTS = (".fits", ".fit", ".fts", ".xisf")
FILTER = "FITS / XISF (*.fits *.fit *.fts *.xisf);;All files (*)"
MAX_SIDE = 4000          # the picture is shown at most this wide (box-averaged); values are read from the full data


def read_image(path):
    """(data (C, H, W) float32 in camera units, cards, look) of a FITS or XISF file."""
    import stacking as st
    src = st.ImageSource(path)
    d = np.asarray(src.read(), np.float32)
    if d.ndim == 2:
        d = d[None]
    return d, list(src.cards), dict(src.look)


def display_planes(d: np.ndarray, look: dict) -> list:
    """The planes to show, top row first: [grey] or [R, G, B]. Raw colour-camera (Bayer) frames are shown in
    colour from their 2 x 2 pixel groups (half size)."""
    import stacking as st
    if d.shape[0] >= 3:
        planes = [d[0], d[1], d[2]]
    elif core.is_bayer(look):
        pat = str(look.get("BAYERPAT", "RGGB") or "RGGB").strip().upper()
        planes = [st.debayer_channel(d[0], pat, c, half=True) for c in ("R", "G", "B")]
    else:
        planes = [d[0]]
    roworder = str(look.get("ROWORDER", "BOTTOM-UP") or "BOTTOM-UP").strip().upper()
    if roworder != "TOP-DOWN" and "_XISF_FORMAT" not in look:        # FITS: bottom row first unless it says so
        planes = [p[::-1] for p in planes]
    return planes


def _shrink(p: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return p
    h, w = p.shape[0] // f * f, p.shape[1] // f * f
    return p[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def to_qimage(planes: list, stretch: bool = True, linked: bool = False):
    """(QImage, shrink factor). Auto-stretch like PixInsight's screen stretch (each colour on its own, so a colour
    cast is evened out for viewing), or the plain linear levels from black to the brightest pixel."""
    h, w = planes[0].shape
    f = max(1, int(np.ceil(max(h, w) / MAX_SIDE)))
    small = [np.nan_to_num(_shrink(np.asarray(p, np.float32), f)) for p in planes]
    if stretch:
        if linked and len(small) == 3:
            prm = core.stretch_params(np.mean(small, axis=0)[::2, ::2])
            out = [core.apply_stretch(p, *prm) for p in small]
        else:
            out = [core.apply_stretch(p, *core.stretch_params(p[::2, ::2])) for p in small]
    else:
        lo = min(float(np.min(p)) for p in small)
        hi = max(float(np.max(p)) for p in small)
        out = [np.clip((p - lo) / max(hi - lo, 1e-12) * 255 + 0.5, 0, 255).astype(np.uint8) for p in small]
    if len(out) == 3:
        rgb = np.ascontiguousarray(np.stack(out, -1))
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
    else:
        g = np.ascontiguousarray(out[0])
        img = QImage(g.data, g.shape[1], g.shape[0], g.shape[1], QImage.Format_Grayscale8).copy()
    return img, f


def summary(path: Path, d: np.ndarray, look: dict) -> str:
    """Size, levels and the main capture facts, as HTML."""
    c, h, w = d.shape
    s = d[:, ::max(1, h // 700), ::max(1, w // 700)]
    s = s[np.isfinite(s)]
    kind = ("colour" if c >= 3 else ("raw colour camera (Bayer " + str(look.get("BAYERPAT", "")).strip() + ")"
                                     if core.is_bayer(look) else "mono"))
    fmt = look.get("_XISF_FORMAT") or {8: "8-bit", 16: "16-bit", 32: "32-bit", -32: "32-bit float",
                                        -64: "64-bit float"}.get(look.get("BITPIX"), "")
    facts = []
    for key, label in (("OBJECT", "Object"), ("IMAGETYP", "Type"), ("FILTER", "Filter"), ("EXPTIME", "Exposure"),
                       ("DATE-OBS", "Date"), ("INSTRUME", "Camera"), ("TELESCOP", "Telescope"),
                       ("FOCALLEN", "Focal length"), ("GAIN", "Gain"), ("CCD-TEMP", "Sensor temp"),
                       ("NCOMBINE", "Frames combined")):
        v = look.get(key)
        if v is None or str(v).strip() == "":
            continue
        if key == "EXPTIME":
            v = f"{float(v):g} s"
        elif key == "FOCALLEN":
            v = f"{float(v):g} mm"
        elif key == "CCD-TEMP":
            v = f"{float(v):g} °C"
        facts.append(f"{label}: <b>{v}</b>")
    lv = (f"min {float(s.min()):.4g} · median {float(np.median(s)):.4g} · max {float(s.max()):.4g}"
          if s.size else "no valid pixels")
    size = path.stat().st_size / 1e6 if path.exists() else 0
    return (f"<b>{path.name}</b> · {size:.1f} MB<br><span style='color:#9a9aa6'>{path.parent}</span><br>"
            f"{w} × {h} px · {kind}{' · ' + str(fmt) if fmt else ''}<br>{lv}"
            + ("<br>" + " · ".join(facts) if facts else ""))


class _View(QGraphicsView):
    """Zoom with the wheel, drag to move, double-click to fit / 100 %. Reports the pixel under the mouse."""

    def __init__(self, on_move=None):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.item)
        self.setBackgroundBrush(QColor(8, 8, 10))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.fit_mode = True
        self.on_move = on_move
        self.setToolTip("Scroll to zoom in, drag to move around, double-click to fit the window")

    def show_image(self, qimg: QImage, full_w: int, full_h: int):
        self.item.setPixmap(QPixmap.fromImage(qimg))
        self.item.setScale(full_w / max(qimg.width(), 1))
        self._scene.setSceneRect(QRectF(0, 0, full_w, full_h))
        if self.fit_mode:
            self.fit()

    def fit(self):
        self.fit_mode = True
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit_mode:
            self.fit()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.25 ** steps
        cur = self.transform().m11()
        if cur * factor > 16:
            factor = 16 / cur
        self.fit_mode = False
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, e):
        if self.fit_mode:
            self.fit_mode = False
            self.resetTransform()
        else:
            self.fit()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.on_move:
            p = self.mapToScene(e.position().toPoint())
            self.on_move(int(p.x()), int(p.y()))


class FileViewer(QDialog):
    """View any FITS or XISF file: the picture and its header."""

    def __init__(self, parent=None, path=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("FITS / XISF Viewer")
        self.resize(1300, 820)
        self.setAcceptDrops(True)
        self.path = None
        self.data = None
        self.look = {}
        self.cards = []
        self.planes = []
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.b_open = QPushButton("Open…")
        self.b_open.clicked.connect(self.open_dialog)
        self.b_prev = QPushButton("◀ Previous")
        self.b_prev.setToolTip("The previous FITS / XISF file in this folder (Page Up)")
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next = QPushButton("Next ▶")
        self.b_next.setToolTip("The next FITS / XISF file in this folder (Page Down)")
        self.b_next.clicked.connect(lambda: self.step(1))
        self.cb_stretch = QCheckBox("Auto-stretch")
        self.cb_stretch.setChecked(True)
        self.cb_stretch.setToolTip("Brighten the picture for viewing (like PixInsight's screen stretch). Off: the "
                                   "plain linear levels - most unprocessed data then looks black")
        self.cb_stretch.toggled.connect(self.redraw)
        self.cb_linked = QCheckBox("Keep colour balance")
        self.cb_linked.setToolTip("Stretch the colours together, so the picture's own colour balance shows (off: "
                                  "each colour is stretched on its own, which hides a colour cast)")
        self.cb_linked.toggled.connect(self.redraw)
        self.b_fit = QPushButton("Fit")
        self.b_fit.clicked.connect(lambda: self.view.fit())
        for wdg in (self.b_open, self.b_prev, self.b_next, self.cb_stretch, self.cb_linked, self.b_fit):
            top.addWidget(wdg)
        top.addStretch(1)
        self.pixel = QLabel("")
        self.pixel.setStyleSheet("color:#9a9aa6")
        top.addWidget(self.pixel)
        lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.view = _View(self.show_pixel)
        split.addWidget(self.view)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        self.info = QLabel("Open a FITS or XISF file (or drop one here).")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sl.addWidget(self.info)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the header…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.fill)
        row.addWidget(self.search, 1)
        b_copy = QPushButton("Copy Header")
        b_copy.setToolTip("Copy every header card as text")
        b_copy.clicked.connect(self.copy_header)
        row.addWidget(b_copy)
        sl.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Keyword", "Value", "Comment"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        sl.addWidget(self.table, 1)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 480])
        lay.addWidget(split, 1)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, activated=lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key_PageUp), self, activated=lambda: self.step(-1))
        if path:
            self.load(path)
        self._buttons()

    # ---- files ---------------------------------------------------------------
    def _siblings(self) -> list:
        if self.path is None:
            return []
        try:
            return sorted((p for p in self.path.parent.iterdir() if p.suffix.lower() in EXTS and p.is_file()),
                          key=lambda p: p.name.lower())
        except OSError:
            return []

    def _buttons(self):
        sib = self._siblings()
        self.b_prev.setEnabled(len(sib) > 1)
        self.b_next.setEnabled(len(sib) > 1)

    def step(self, d: int):
        sib = self._siblings()
        if len(sib) < 2 or self.path not in sib:
            return
        self.load(sib[(sib.index(self.path) + d) % len(sib)])

    def open_dialog(self):
        start = str(self.path.parent) if self.path else str(Path.home())
        p, _ = QFileDialog.getOpenFileName(self, "Open a FITS or XISF file", start, FILTER)
        if p:
            self.load(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in EXTS:
                self.load(p)
                break

    def load(self, path) -> bool:
        path = Path(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            d, cards, look = read_image(path)
            planes = display_planes(d, look)
        except Exception as e:  # noqa: BLE001 - say what is wrong with the file, keep the window
            QApplication.restoreOverrideCursor()
            self.info.setText(f"<b>{path.name}</b><br>Can't show this file: {e}")
            return False
        self.path, self.data, self.cards, self.look, self.planes = path, d, cards, look, planes
        self.setWindowTitle(f"FITS / XISF Viewer — {path.name}")
        self.cb_linked.setEnabled(len(planes) == 3)
        self.info.setText(summary(path, d, look))
        self.redraw()
        self.fill()
        self._buttons()
        QApplication.restoreOverrideCursor()
        return True

    # ---- picture ---------------------------------------------------------------
    def redraw(self, *_):
        if not self.planes:
            return
        img, _f = to_qimage(self.planes, self.cb_stretch.isChecked(), self.cb_linked.isChecked())
        h, w = self.planes[0].shape
        self.view.show_image(img, w, h)

    def show_pixel(self, x: int, y: int):
        if not self.planes:
            return
        h, w = self.planes[0].shape
        if not (0 <= x < w and 0 <= y < h):
            self.pixel.setText("")
            return
        vals = [float(p[y, x]) for p in self.planes]
        txt = " / ".join(f"{v:.5g}" for v in vals)
        self.pixel.setText(f"x {x}  y {y}   {'R / G / B ' if len(vals) == 3 else 'value '}{txt}")

    # ---- header ---------------------------------------------------------------
    def fill(self, *_):
        q = self.search.text().strip().lower()
        cards = [(str(k), v, str(c or "")) for k, v, c in self.cards if not str(k).startswith("_")]
        rows = [c for c in cards if not q or q in " ".join(str(x) for x in c).lower()]
        self.table.setRowCount(len(rows))
        for r, (k, v, c) in enumerate(rows):
            for col, text in enumerate((k, "" if v is None else str(v), c)):
                it = QTableWidgetItem(text)
                if col == 0:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.table.setItem(r, col, it)
        self.table.resizeColumnToContents(1)
        if self.table.columnWidth(1) > 300:
            self.table.setColumnWidth(1, 300)

    def header_text(self) -> str:
        out = []
        for k, v, c in self.cards:
            if str(k).startswith("_"):
                continue
            if k in ("COMMENT", "HISTORY"):
                out.append(f"{k:8s} {'' if v is None else v}")
                continue
            val = ("T" if v else "F") if isinstance(v, bool) else (
                "'" + v.replace("'", "''") + "'" if isinstance(v, str) else ("" if v is None else str(v)))
            out.append(f"{k:8s}= {val:>20s}" + (f" / {c}" if c else ""))
        return "\n".join(out)

    def copy_header(self):
        QApplication.clipboard().setText(self.header_text())


def open_viewer(parent, path=None, ask=True):
    """Open the viewer (asking for a file when none is given). Returns the window."""
    if path is None and ask:
        start = ""
        try:
            start = str(parent.settings.value("last_folder", str(Path.home())))
        except Exception:  # noqa: BLE001
            start = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(parent, "Open a FITS or XISF file", start, FILTER)
        if not path:
            return None
    dlg = FileViewer(parent, path)
    dlg.show()
    return dlg


# ==========================================================================
# Opening FITS / XISF files with SubZeroAstro (double-click in Explorer / Finder)
# ==========================================================================

def viewer_files(args) -> list:
    """The FITS / XISF files among command-line arguments (what Windows passes on a double-click)."""
    out = []
    for a in args:
        p = Path(a)
        if p.suffix.lower() in EXTS and p.is_file():
            out.append(p)
    return out


PROGID = "SubZeroAstro.AstroImage"


def register_windows(exe=None) -> str:
    """Windows: add SubZeroAstro to 'Open with' for .fits/.fit/.fts/.xisf (for this user only - no administrator
    needed) and return what to tell the user. Windows 10/11 lets only the user choose the default app, so the
    Default Apps page is opened next for that one click."""
    import sys
    import winreg  # noqa: PLC0415 - Windows only
    exe = exe or sys.executable
    cmd = f'"{exe}" "%1"'
    base = r"Software\Classes"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "Astronomy image (FITS / XISF)")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}\DefaultIcon") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{exe}",0')
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}\shell\open\command") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd)
    for ext in EXTS:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{ext}\OpenWithProgids") as k:
            winreg.SetValueEx(k, PROGID, 0, winreg.REG_NONE, b"")
    try:
        import ctypes
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)     # SHCNE_ASSOCCHANGED: refresh Explorer
    except Exception:  # noqa: BLE001
        pass
    return ("SubZeroAstro is now offered for FITS and XISF files.\n\nTo make it the default: right-click any .fits "
            "or .xisf file › Open with › Choose another app › SubZeroAstro, and tick 'Always use this app'. Do the "
            "same once for .xisf (and .fit if you use it).")


MAC_HOWTO = ("To open FITS and XISF files with SubZeroAstro on a double-click:\n\n"
             "1. In Finder, select any .fits file and press ⌘I (Get Info).\n"
             "2. Under 'Open with', choose SubZeroAstro.\n"
             "3. Click 'Change All…' and confirm.\n\n"
             "Do the same once for an .xisf file (and .fit if you use it). Double-clicking then opens the file in "
             "SubZeroAstro's viewer.")
