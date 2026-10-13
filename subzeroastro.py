"""
SubZeroAstro - From subs to masters - zero hassle.

Culling with star quality metrics, stacking, and preparing masters for PixInsight.

Open a folder of FITS frames, see star quality for every frame (FWHM,
eccentricity, SNR, star count, background) in the sidebar and on a session
chart, flag suspects (by hand or with Auto-Flag) and move bad frames into a
REJECTED subfolder. The culling view was inspired by Tom van Peer's FITS Blaster for macOS
(https://astrophoto-app.com/); written independently for Windows.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (QItemSelectionModel, QObject, QPointF, QRectF, QRunnable,
                            QSettings, QSize, Qt, QThreadPool, QTimer, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QFont,
                           QIcon, QImage, QKeySequence, QPainter, QPalette,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListView,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter,
                               QStyle, QTableWidget, QTableWidgetItem,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

import checkpoints as cp
import fits_core as core
from chart import ChartPoint, SessionChart, filter_color

APP_NAME = "SubZeroAstro"
APP_TAGLINE = "From subs to masters - zero hassle."
APP_VERSION = "0.8.11 beta"
APP_AUTHOR = "Brian A. Poteet"
APP_URL = "https://github.com/bapoteet/SubZeroAstro"
ICON_W, ICON_H = 120, 84
CACHE_SIZE = 8
ROLE_UID = Qt.UserRole + 1

BADGE_COLORS = {"green": QColor(70, 200, 95), "amber": QColor(235, 170, 50),
                "red": QColor(230, 70, 70), "grey": QColor(130, 130, 140)}

SORTS = ["File name", "Capture time", "Filter", "Quality score", "FWHM", "Eccentricity",
         "SNR", "Star count", "Background"]
SORT_METRIC = {"Quality score": "score", "FWHM": "fwhm", "Eccentricity": "eccentricity",
               "SNR": "snr", "Star count": "stars", "Background": "background"}
CHART_METRICS = ["score", "fwhm", "eccentricity", "snr", "stars", "background"]


# --------------------------------------------------------------------------
# Background work
# --------------------------------------------------------------------------

def to_qimage(arr) -> QImage:
    h, w = arr.shape
    buf = arr.tobytes()
    return QImage(buf, w, h, w, QImage.Format_Grayscale8).copy()


PREVIEW_MAX = 1600            # side of the previews kept on disk
PREVIEW_KEEP = 3000           # at most this many preview files
_preview_saves = 0


def _preview_path(f) -> Path:
    import hashlib
    key = hashlib.sha1(f"{os.path.normcase(str(f.path))}|{f.size}|{f.mtime_ns}".encode("utf-8")).hexdigest()
    return core.cache_dir() / "previews" / f"{key}.jpg"


def preview_load(f):
    """(QImage, factor) from the local preview cache, or None."""
    p = _preview_path(f)
    if not p.exists() or not f.display_size:
        return None
    img = QImage(str(p))
    if img.isNull() or img.width() == 0:
        return None
    return img, max(1, int(round(f.display_size[0] / img.width())))


def preview_save(f, img: QImage, factor: int):
    """Keep a screen-sized JPEG of a frame on this computer, so it shows instantly next time."""
    try:
        p = _preview_path(f)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        k = 1
        while max(img.width(), img.height()) / k > PREVIEW_MAX:
            k += 1
        small = img if k == 1 else img.scaled(img.width() // k, img.height() // k, Qt.IgnoreAspectRatio,
                                               Qt.SmoothTransformation)
        small.save(str(p), "JPG", 88)
        global _preview_saves
        _preview_saves += 1
        if _preview_saves % 100:
            return                                   # prune now and then, not on every frame
        files = list(p.parent.glob("*.jpg"))
        if len(files) > PREVIEW_KEEP:
            for old in sorted(files, key=lambda x: x.stat().st_mtime)[:len(files) - PREVIEW_KEEP]:
                old.unlink(missing_ok=True)
    except OSError:
        pass


class LoaderSignals(QObject):
    analyzed = Signal(int, int, object)          # generation, uid, result dict
    analyze_failed = Signal(int, int, str)
    display_ready = Signal(int, int, object, int, bool, str)   # gen, uid, QImage, factor, full, path
    display_failed = Signal(int, int, str, str)


class AnalyzeTask(QRunnable):
    def __init__(self, gen, uid, path, mode, signals):
        super().__init__()
        self.gen, self.uid, self.path, self.mode, self.signals = gen, uid, str(path), mode, signals

    def run(self):
        try:
            res = core.analyze_frame(self.path, self.mode)
            res["qthumb"] = to_qimage(res["thumb"])
            self.signals.analyzed.emit(self.gen, self.uid, res)
        except Exception as e:  # noqa: BLE001
            self.signals.analyze_failed.emit(self.gen, self.uid, str(e) or type(e).__name__)


class DisplayTask(QRunnable):
    def __init__(self, gen, uid, path, params, full, signals):
        super().__init__()
        self.gen, self.uid, self.path = gen, uid, str(path)
        self.params, self.full, self.signals = params, full, signals

    def run(self):
        try:
            arr, factor = core.load_frame_display(self.path, self.params, None if self.full else 25e6)
            self.signals.display_ready.emit(self.gen, self.uid, to_qimage(arr), factor, self.full, self.path)
        except Exception as e:  # noqa: BLE001
            self.signals.display_failed.emit(self.gen, self.uid, str(e) or type(e).__name__, self.path)


# --------------------------------------------------------------------------
# Image viewer
# --------------------------------------------------------------------------

class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.item)
        self.setBackgroundBrush(QColor(8, 8, 10))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.fit_mode = True
        self.has_image = False
        self.message = "Click Load Data… (Ctrl+O) to choose folders\nor drag folders onto this window"
        self.banner = ""
        self.subtitle = ""
        self.on_detail = None        # called when zoomed in enough to want full resolution

    def clear(self, message=None):
        self.item.setPixmap(QPixmap())
        self.has_image = False
        if message is not None:
            self.message = message
        self.banner = ""
        self.subtitle = ""
        self.viewport().update()

    def show_image(self, qimg: QImage, full_size):
        """Show an image scaled to cover full_size (w, h) in scene coordinates, so
        thumbnails, half- and full-resolution images all line up."""
        self.item.setPixmap(QPixmap.fromImage(qimg))
        w = qimg.width() or 1
        self.item.setScale(full_size[0] / w)
        rect = QRectF(0, 0, full_size[0], full_size[1])
        self._scene.setSceneRect(rect)
        self.has_image = True
        if self.fit_mode:
            self.fit()
        self.viewport().update()

    def fit(self):
        self.fit_mode = True
        if self.has_image:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.viewport().update()

    def actual_size(self):
        self.fit_mode = False
        self.resetTransform()
        self._check_detail()
        self.viewport().update()

    def zoom_percent(self) -> int:
        return int(round(self.transform().m11() * 100))

    def _check_detail(self):
        if self.on_detail and self.transform().m11() >= 0.5:
            self.on_detail()

    def wheelEvent(self, event):
        if not self.has_image:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 1.25 ** steps
        cur = self.transform().m11()
        if cur * factor > 16:
            factor = 16 / cur
        self.fit_mode = False
        self.scale(factor, factor)
        self._check_detail()
        self.viewport().update()

    def mouseDoubleClickEvent(self, event):
        if self.fit_mode:
            self.actual_size()
        else:
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()

    def drawForeground(self, painter, rect):
        painter.save()
        painter.resetTransform()
        vp = self.viewport().rect()
        f = QFont(painter.font())
        if not self.has_image:
            painter.setPen(QColor(150, 150, 160))
            f.setPointSize(13)
            painter.setFont(f)
            painter.drawText(vp, Qt.AlignCenter, self.message)
        else:
            if self.banner:
                f.setPointSize(16)
                f.setBold(True)
                painter.setFont(f)
                painter.setPen(QColor(255, 80, 80) if self.banner == "REJECTED" else QColor(255, 220, 60))
                painter.drawText(vp.adjusted(14, 10, -14, -10), Qt.AlignTop | Qt.AlignLeft, self.banner)
            f.setPointSize(9)
            f.setBold(False)
            painter.setFont(f)
            painter.setPen(QColor(170, 170, 180))
            text = f"{self.zoom_percent()}%"
            if self.subtitle:
                text = f"{self.subtitle}    {text}"
            painter.drawText(vp.adjusted(10, 6, -10, -6), Qt.AlignBottom | Qt.AlignRight, text)
        painter.restore()


# --------------------------------------------------------------------------
# Auto-Flag dialog
# --------------------------------------------------------------------------

class CheckpointDialog(QDialog):
    """Settings › Saved Progress: where checkpoints go and how long they are kept."""

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Saved Progress")
        self.resize(640, 300)
        lay = QVBoxLayout(self)
        intro = QLabel(f"{APP_NAME} saves your progress as it works — the culling session, each finished "
                       "master while stacking, and the image after every Prepare step — so you can carry on "
                       "after a crash and go back to an earlier step. Prepare copies are the big ones "
                       "(roughly 100–400 MB per filter per step), so a fast local drive is best.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        row = QHBoxLayout()
        self.dir_edit = QLineEdit(settings.value("checkpoints/dir", "") or "")
        self.dir_edit.setPlaceholderText(str(cp.default_root()))
        row.addWidget(self.dir_edit, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self.browse)
        row.addWidget(b)
        b2 = QPushButton("Default")
        b2.clicked.connect(self.dir_edit.clear)
        row.addWidget(b2)
        form.addRow("Folder:", row)
        self.days = QSpinBox()
        self.days.setRange(0, 365)
        self.days.setSpecialValueText("never")
        self.days.setSuffix(" days")
        self.days.setValue(int(settings.value("checkpoints/days", 30)))
        form.addRow("Delete progress not used for:", self.days)
        lay.addLayout(form)
        self.size_label = QLabel()
        lay.addWidget(self.size_label)
        b3 = QPushButton("Delete All Saved Progress Now")
        b3.clicked.connect(self.clear_all)
        lay.addWidget(b3, 0, Qt.AlignLeft)
        lay.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.update_size()

    def update_size(self):
        self.size_label.setText(f"Currently using <b>{cp.human(cp.total_size())}</b> in {cp.root()}")

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for saved progress",
                                             self.dir_edit.text() or str(cp.default_root()))
        if d:
            self.dir_edit.setText(d)

    def clear_all(self):
        if QMessageBox.question(self, "Saved Progress", "Delete all saved progress? You won't be able to resume "
                                "interrupted runs or go back to earlier Prepare steps.\n\n(Your images, masters "
                                "and the culling session are not touched.)") \
                == QMessageBox.StandardButton.Yes:
            freed = cp.clear_all()
            self.update_size()
            QMessageBox.information(self, "Saved Progress", f"Freed {cp.human(freed)}.")

    def save(self):
        d = self.dir_edit.text().strip()
        if d:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Saved Progress", f"Can't use that folder:\n{e}")
                return
        self.settings.setValue("checkpoints/dir", d)
        self.settings.setValue("checkpoints/days", self.days.value())
        cp.set_root(d)
        self.accept()


class AutoFlagDialog(QDialog):
    """Flag frames that fall outside thresholds relative to their filter's median."""

    def __init__(self, parent, frames, medians, settings):
        super().__init__(parent)
        self.setWindowTitle("Auto-Flag")
        self.frames = [f for f in frames if not f.rejected]
        self.medians = medians
        self.settings = settings
        lay = QVBoxLayout(self)
        intro = QLabel("Flag frames that are worse than the median of their own filter.\n"
                       "Flagged frames are only marked — review them in the Flagged view, "
                       "then reject the ones you don't want.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        grid = QGridLayout()
        self.rows = {}

        def row(r, key, text, spin, suffix):
            cb = QCheckBox(text)
            cb.setChecked(settings.value(f"af/{key}/on", key in ("fwhm", "ecc", "stars", "snr"), type=bool))
            spin.setValue(settings.value(f"af/{key}/val", spin.value(), type=float))
            grid.addWidget(cb, r, 0)
            grid.addWidget(spin, r, 1)
            grid.addWidget(QLabel(suffix), r, 2)
            cb.toggled.connect(self.update_count)
            spin.valueChanged.connect(self.update_count)
            self.rows[key] = (cb, spin)

        def dspin(lo, hi, val, step, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(val)
            return s

        row(0, "fwhm", "FWHM more than", dspin(1.0, 3.0, 1.20, 0.05), "× the median")
        row(1, "ecc", "Eccentricity above", dspin(0.2, 0.99, 0.60, 0.05), "")
        row(2, "stars", "Star count below", dspin(5, 100, 60, 5, 0), "% of the median")
        row(3, "snr", "SNR below", dspin(5, 100, 60, 5, 0), "% of the median")
        row(4, "bg", "Background above", dspin(101, 400, 140, 5, 0), "% of the median")
        self.red_cb = QCheckBox("Every frame with a red badge")
        self.red_cb.setChecked(settings.value("af/red", False, type=bool))
        self.red_cb.toggled.connect(self.update_count)
        grid.addWidget(self.red_cb, 5, 0, 1, 3)
        lay.addLayout(grid)

        self.count_label = QLabel()
        f = self.count_label.font()
        f.setBold(True)
        self.count_label.setFont(f)
        lay.addWidget(self.count_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.update_count()

    def matches(self):
        out = []
        on = {k: (cb.isChecked(), sp.value()) for k, (cb, sp) in self.rows.items()}
        for f in self.frames:
            m = f.metrics
            gm = self.medians.get(f.group)
            if not m or not gm:
                continue
            hit = False
            if on["fwhm"][0] and m.get("fwhm") and gm.get("fwhm"):
                hit |= m["fwhm"] > gm["fwhm"] * on["fwhm"][1]
            if on["ecc"][0] and m.get("eccentricity") is not None:
                hit |= m["eccentricity"] > on["ecc"][1]
            if on["stars"][0] and gm.get("stars"):
                hit |= m.get("stars", 0) < gm["stars"] * on["stars"][1] / 100
            if on["snr"][0] and m.get("snr") is not None and gm.get("snr"):
                hit |= m["snr"] < gm["snr"] * on["snr"][1] / 100
            if on["bg"][0] and gm.get("background"):
                hit |= m.get("background", 0) > gm["background"] * on["bg"][1] / 100
            if self.red_cb.isChecked():
                hit |= f.badge == "red"
            if hit:
                out.append(f.uid)
        return out

    def update_count(self, *_):
        n = len(self.matches())
        self.count_label.setText(f"{n} of {len(self.frames)} frames match")
        ok = self.buttons.button(QDialogButtonBox.Ok)
        if ok is not None:
            ok.setText(f"Flag {n} frames")
            ok.setEnabled(n > 0)

    def save(self):
        for k, (cb, sp) in self.rows.items():
            self.settings.setValue(f"af/{k}/on", cb.isChecked())
            self.settings.setValue(f"af/{k}/val", sp.value())
        self.settings.setValue("af/red", self.red_cb.isChecked())


# --------------------------------------------------------------------------
# Load Data dialog: collect folders / files from several places, then start
# --------------------------------------------------------------------------

class HeaderDialog(QDialog):
    """Every header card of a FITS / XISF file in its own window (a "header dump")."""

    def __init__(self, parent, path=None, cards=None, title=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(820, 680)
        lay = QVBoxLayout(self)
        self.head = QLabel()
        self.head.setWordWrap(True)
        self.head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.head)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search keywords, values and comments…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.fill)
        row.addWidget(self.search, 1)
        b_open = QPushButton("Open Another File…")
        b_open.clicked.connect(self.open_other)
        b_copy = QPushButton("Copy All")
        b_copy.setToolTip("Copy the whole header as text (80-column FITS cards)")
        b_copy.clicked.connect(self.copy_all)
        row.addWidget(b_open)
        row.addWidget(b_copy)
        lay.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Keyword", "Value", "Comment"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self.cards = []
        self.load(path, cards, title)

    def load(self, path=None, cards=None, title=None):
        self.path = Path(path) if path else None
        if cards is None and self.path is not None:
            try:
                import stacking as st
                cards = st.ImageSource(self.path).cards
            except Exception as e:  # noqa: BLE001
                cards = [("ERROR", str(e), "")]
        self.cards = [(str(k), v, str(c or "")) for k, v, c in (cards or [])
                      if not str(k).startswith("_")]
        name = title or (self.path.name if self.path else "header")
        self.setWindowTitle(f"FITS header — {name}")
        size = ""
        if self.path is not None and self.path.exists():
            size = f" · {self.path.stat().st_size / 1e6:.1f} MB"
        self.head.setText(f"<b>{name}</b>{size}<br><span style='color:#9a9aa6'>"
                          f"{self.path.parent if self.path else ''}</span><br>{len(self.cards)} header cards")
        self.fill()

    def fill(self, *_):
        q = self.search.text().strip().lower()
        rows = [c for c in self.cards if not q or q in " ".join(str(x) for x in c).lower()]
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
        if self.table.columnWidth(1) > 320:
            self.table.setColumnWidth(1, 320)

    def as_text(self):
        out = []
        for k, v, c in self.cards:
            if k in ("COMMENT", "HISTORY"):
                out.append(f"{k:8s} {'' if v is None else v}")
                continue
            if isinstance(v, bool):
                val = "T" if v else "F"
            elif isinstance(v, str):
                val = "'" + v.replace("'", "''") + "'"
            else:
                val = "" if v is None else str(v)
            out.append(f"{k:8s}= {val:>20s}" + (f" / {c}" if c else ""))
        return "\n".join(out)

    def copy_all(self):
        QApplication.clipboard().setText(self.as_text())

    def open_other(self):
        f, _ = QFileDialog.getOpenFileName(self, "Show the header of…", str(self.path.parent if self.path else ""),
                                           "FITS / XISF (*.fits *.fit *.fts *.xisf);;All files (*)")
        if f:
            self.load(f)


class Toast(QLabel):
    """A message that floats over the top of a window for a few seconds."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background:#3a2f12; color:#ffe2a0; border:1px solid #e0a030; "
                           "border-radius:6px; padding:8px 16px; font-size:10.5pt; }")
        self.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_message(self, text, seconds=6):
        self.setText(text)
        w = min(760, max(320, self.parent().width() - 80))
        self.setFixedWidth(w)
        self.adjustSize()
        self.move((self.parent().width() - w) // 2, 12)
        self.raise_()
        self.show()
        self.timer.start(int(seconds * 1000))


COMMON_FILTERS = ["L-Ultimate", "L-eXtreme", "L-eNhance", "L-Pro", "ALP-T", "NBZ", "UV/IR cut",
                  "L", "R", "G", "B", "Ha", "OIII", "SII", "No filter"]


class LoadDataDialog(QDialog):
    """Build a list of folders and files to process. Nothing is measured until
    Start Processing is pressed."""

    ROLE_SRC = Qt.UserRole + 2

    def __init__(self, parent, settings, recursive_default=False, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Load Data")
        self.setAcceptDrops(True)
        self.settings = settings
        self.resize(760, 440)
        lay = QVBoxLayout(self)
        intro = QLabel("Add the folders (or individual files) to process — from as many places as you like, "
                       "for example several nights on your NAS. Nothing is read until you press "
                       "<b>Start Processing</b>. You can also drag folders and files onto this window.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.list, 1)

        row = QHBoxLayout()
        b_folder = QToolButton()
        b_folder.setText("Add Folder…")
        b_folder.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b_folder.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        b_folder.clicked.connect(self.add_folder_dialog)
        b_files = QToolButton()
        b_files.setText("Add Files…")
        b_files.clicked.connect(self.add_files_dialog)
        b_remove = QToolButton()
        b_remove.setText("Remove Selected")
        b_remove.clicked.connect(self.remove_selected)
        b_clear = QToolButton()
        b_clear.setText("Clear All")
        b_clear.clicked.connect(self.clear_all)
        self.chk_sub = QCheckBox("Include subfolders")
        self.chk_sub.setToolTip("For folders added from now on: also load FITS files in their subfolders "
                                "(FLAT, DARK, BIAS, CALIB and OFFSET folders are skipped)")
        self.chk_sub.setChecked(recursive_default)
        for b in (b_folder, b_files, b_remove, b_clear):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.chk_sub)
        lay.addLayout(row)

        frow = QHBoxLayout()
        flab = QLabel("Filter used:")
        flab.setToolTip("If the camera software didn't record the filter (the ASIAIR can't), or recorded it "
                        "wrong, type or pick the real one here. It is used for grouping, quality "
                        "comparison, stacking and matching your flats - your files are not changed.")
        frow.addWidget(flab)
        self.filter_combo = QComboBox()
        self.filter_combo.setEditable(True)
        self.filter_combo.addItem("(as recorded in the files)")
        self.filter_combo.addItems(COMMON_FILTERS)
        self.filter_combo.setToolTip(flab.toolTip())
        last = settings.value("load/filter_override", "") or ""
        if last:
            self.filter_combo.setCurrentText(last)
        frow.addWidget(self.filter_combo, 1)
        frow.addWidget(QLabel("<span style='color:#9a9aa6'>for every frame loaded here</span>"))
        lay.addLayout(frow)

        self.total = QLabel()
        f = self.total.font()
        f.setBold(True)
        self.total.setFont(f)
        lay.addWidget(self.total)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        ok = self.buttons.button(QDialogButtonBox.Ok)
        if ok is not None:
            ok.setText("Start Processing")
        for src in initial or []:
            self.add_source(src)
        self.update_total()

    # -- adding
    def last_folder(self):
        return self.settings.value("last_folder", str(Path.home()))

    def add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder of FITS files", self.last_folder())
        if d:
            self.settings.setValue("last_folder", str(Path(d).parent))
            self.add_source(("folder", Path(d), self.chk_sub.isChecked()))

    def add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add FITS files", self.last_folder(),
                                                "FITS files (*.fits *.fit *.fts);;All files (*)")
        if files:
            self.settings.setValue("last_folder", str(Path(files[0]).parent))
            self.add_source(("files", [Path(x) for x in files]))

    def add_source(self, src):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if src[0] == "folder":
                path, rec = Path(src[1]), src[2]
                for i in range(self.list.count()):          # already listed?
                    s2 = self.list.item(i).data(self.ROLE_SRC)
                    if s2[0] == "folder" and Path(s2[1]) == path and s2[2] == rec:
                        return
                frames = core.scan_folder(path, recursive=rec)
                n_rej = sum(1 for f in frames if f.rejected)
                text = f"📁  {path}\n      {len(frames) - n_rej} FITS files" + \
                       (f" + {n_rej} already rejected" if n_rej else "") + \
                       ("  ·  including subfolders" if rec else "")
                if not frames:
                    text += "   ⚠ no FITS files found" + ("" if rec else " (try Include subfolders)")
            else:
                files = [Path(x) for x in src[1]]
                frames = core.frames_from_files(files)
                parents = sorted({str(f.path.parent) for f in frames})
                text = f"🗎  {len(frames)} selected file{'s' if len(frames) != 1 else ''}\n      from " + \
                       (parents[0] if len(parents) == 1 else f"{len(parents)} folders")
            it = QListWidgetItem(text)
            it.setData(self.ROLE_SRC, src)
            it.setData(self.ROLE_SRC + 1, len(frames))
            it.setToolTip(str(src[1]) if src[0] == "folder" else "\n".join(str(x) for x in src[1][:30]))
            self.list.addItem(it)
        finally:
            QApplication.restoreOverrideCursor()
        self.update_total()

    def remove_selected(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))
        self.update_total()

    def clear_all(self):
        self.list.clear()
        self.update_total()

    def update_total(self):
        n_src = self.list.count()
        n = sum(int(self.list.item(i).data(self.ROLE_SRC + 1) or 0) for i in range(n_src))
        self.total.setText(f"{n_src} source{'s' if n_src != 1 else ''}  ·  {n} frames" if n_src
                           else "Nothing added yet — click Add Folder…")
        ok = self.buttons.button(QDialogButtonBox.Ok)
        if ok is not None:
            ok.setEnabled(n > 0)

    def sources(self):
        return [self.list.item(i).data(self.ROLE_SRC) for i in range(self.list.count())]

    def filter_override(self):
        t = self.filter_combo.currentText().strip()
        val = "" if (not t or t.startswith("(as recorded")) else core.normalize_filter(t)
        self.settings.setValue("load/filter_override", val)
        return val or None

    # -- drag and drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        files = [p for p in paths if p.is_file()]
        for p in paths:
            if p.is_dir():
                self.add_source(("folder", p, self.chk_sub.isChecked()))
        if files:
            self.add_source(("files", files))


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Astro Image Quality and Cull")
        self.setAcceptDrops(True)
        # Settings keep their original storage name so nothing is lost after the rename
        self.settings = QSettings("FITSBlasterWin", "FITSBlasterWin")

        self.frames: dict[int, core.Frame] = {}
        self.order: list[int] = []
        self.items: dict[int, QListWidgetItem] = {}
        self.current_uid: int | None = None
        self.generation = 0
        self.loaded_count = 0
        self.to_load = 0
        self.load_started = 0.0
        self.undo_stack: list = []          # [(uid, rejected), ...] or {"label", "reject", "flags"}
        self.cache: OrderedDict[tuple[int, bool], tuple[QImage, int]] = OrderedDict()
        self.pending_display: set[tuple[int, bool]] = set()
        self.known_filters: list[str] = []
        self.root: Path | None = None
        self.sources: list = []              # what was loaded, for Reload / Re-measure
        self.filter_overrides: dict = {}     # path -> filter name set by the user
        self.medians: dict = {}
        self.disk_cache: dict[Path, dict] = {}      # folder -> {cache_key: entry}
        self.cache_dirty: set[Path] = set()
        self.network = False

        self.signals = LoaderSignals()
        self.signals.analyzed.connect(self.on_analyzed)
        self.signals.analyze_failed.connect(self.on_analyze_failed)
        self.signals.display_ready.connect(self.on_display_ready)
        self.signals.display_failed.connect(self.on_display_failed)

        self.analyze_pool = QThreadPool(self)
        self.display_pool = QThreadPool(self)
        self.display_pool.setMaxThreadCount(2)

        self.stats_timer = QTimer(self)
        self.stats_timer.setSingleShot(True)
        self.stats_timer.setInterval(600)
        self.stats_timer.timeout.connect(self.refresh_stats)
        self.session_timer = QTimer(self)        # autosave of the culling session (0.6)
        self.session_timer.setSingleShot(True)
        self.session_timer.setInterval(1500)
        self.session_timer.timeout.connect(self.save_session)

        cp.set_root(self.settings.value("checkpoints/dir", "") or None)
        import ui_header
        ui_header.install_header(self, ui_header.PAGE_CULL, with_menu=True)
        self._build_ui()
        self._build_actions()
        self._restore_settings()
        QTimer.singleShot(3000, self.cleanup_checkpoints)
        self.view.on_detail = self.request_detail
        self.update_status()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(6)

        row1 = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All filters")
        self.filter_combo.setToolTip("Show only frames taken with this filter")
        self.show_combo = QComboBox()
        self.show_combo.addItems(["All frames", "Kept", "Flagged", "Rejected"])
        self.show_combo.setToolTip("Which frames to list")
        row1.addWidget(self.filter_combo, 1)
        row1.addWidget(self.show_combo, 1)
        sl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(SORTS)
        row2.addWidget(self.sort_combo, 1)
        self.worst_first = QCheckBox("Worst first")
        self.worst_first.setToolTip("For quality sorts: put the worst frames at the top")
        row2.addWidget(self.worst_first)
        sl.addLayout(row2)

        self.list = QListWidget()
        self.list.setViewMode(QListView.ListMode)
        self.list.setIconSize(QSize(ICON_W, ICON_H))
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setSpacing(2)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        sl.addWidget(self.list, 1)

        self.filter_combo.currentIndexChanged.connect(self.on_view_changed)
        self.show_combo.currentIndexChanged.connect(self.on_view_changed)
        self.sort_combo.currentIndexChanged.connect(self.on_sort_changed)
        self.worst_first.toggled.connect(self.on_sort_changed)
        self.list.currentItemChanged.connect(self.on_current_changed)
        self.list.itemSelectionChanged.connect(self.on_selection_changed)

        # Centre: viewer above, chart below
        self.view = ImageView()
        chart_box = QWidget()
        cl = QVBoxLayout(chart_box)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 0)
        bar.addWidget(QLabel("Chart:"))
        self.chart_metric = QComboBox()
        for k in CHART_METRICS:
            self.chart_metric.addItem(core.METRIC_NAMES[k], k)
        self.chart_metric.currentIndexChanged.connect(self.update_chart)
        bar.addWidget(self.chart_metric)
        self.legend = QLabel()
        bar.addWidget(self.legend, 1)
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: #f0b040;")
        self.warning_label.setWordWrap(True)
        bar.addWidget(self.warning_label, 2)
        cl.addLayout(bar)
        self.chart = SessionChart()
        self.chart.frameClicked.connect(self.goto_uid)
        self.chart.rangeSelected.connect(self.select_uids)
        cl.addWidget(self.chart, 1)

        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.addWidget(self.view)
        self.vsplit.addWidget(chart_box)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.setSizes([620, 220])

        # Right: inspector
        insp = QWidget()
        il = QVBoxLayout(insp)
        il.setContentsMargins(8, 8, 8, 8)

        def heading(text):
            lab = QLabel(text)
            tf = lab.font()
            tf.setPointSize(tf.pointSize() + 1)
            tf.setBold(True)
            lab.setFont(tf)
            return lab

        il.addWidget(heading("Kept frames"))
        self.kept_label = QLabel("–")
        self.kept_label.setWordWrap(True)
        self.kept_label.setTextFormat(Qt.RichText)
        self.kept_label.setToolTip("Live summary of the frames that will go to the stacker: everything not "
                                   "rejected and not flagged. Updates as you flag and reject.")
        self.kept_label.setStyleSheet("QLabel { background:#18263a; border:1px solid #2c4466; border-radius:5px;"
                                      " padding:6px; }")
        il.addWidget(self.kept_label)
        il.addWidget(heading("Quality"))
        qform = QFormLayout()
        qform.setLabelAlignment(Qt.AlignRight)
        self.q_labels = {}
        for key, label in [("score", "Score"), ("fwhm", "FWHM"), ("eccentricity", "Eccentricity"),
                           ("snr", "SNR"), ("stars", "Stars"), ("background", "Background")]:
            lab = QLabel("–")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            qform.addRow(label + ":", lab)
            self.q_labels[key] = lab
        il.addLayout(qform)
        self.reasons_label = QLabel()
        self.reasons_label.setWordWrap(True)
        il.addWidget(self.reasons_label)

        il.addWidget(heading("Frame"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.info_labels = {}
        for key, label in [("file", "File"), ("status", "Status"), ("folder", "Folder"),
                           ("object", "Target"), ("filter", "Filter"), ("exptime", "Exposure"),
                           ("date_obs", "Captured"), ("night", "Night of"), ("camera", "Camera"),
                           ("gain", "Gain"), ("temp", "Sensor temp"), ("size", "Size"), ("bayer", "Bayer")]:
            lab = QLabel("–")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setWordWrap(True)
            form.addRow(label + ":", lab)
            self.info_labels[key] = lab
        il.addLayout(form)

        hrow = QHBoxLayout()
        hrow.addWidget(QLabel("FITS header"))
        hrow.addStretch(1)
        b_hdr = QToolButton()
        b_hdr.setText("Open in Window")
        b_hdr.setToolTip("Show the full header of this frame in its own window (Ctrl+H)")
        b_hdr.clicked.connect(self.show_header_window)
        hrow.addWidget(b_hdr)
        il.addLayout(hrow)
        self.header_search = QLineEdit()
        self.header_search.setPlaceholderText("Search header…")
        self.header_search.setClearButtonEnabled(True)
        self.header_search.textChanged.connect(self.fill_header_table)
        il.addWidget(self.header_search)
        self.header_table = QTableWidget(0, 2)
        self.header_table.setHorizontalHeaderLabels(["Keyword", "Value"])
        self.header_table.verticalHeader().setVisible(False)
        self.header_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.header_table.horizontalHeader().setStretchLastSection(True)
        self.header_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.header_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.header_table.setWordWrap(False)
        il.addWidget(self.header_table, 1)
        insp.setMinimumWidth(240)
        insp.setMaximumWidth(420)
        for lab in list(self.info_labels.values()) + list(self.q_labels.values()) + [self.reasons_label]:
            lab.setMinimumWidth(1)          # long text wraps instead of widening the panel

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(side)
        self.splitter.addWidget(self.vsplit)
        self.splitter.addWidget(insp)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([340, 900, 310])

        self.ready_bar = QFrame()
        self.ready_bar.setStyleSheet("QFrame { background: #1f3350; border-bottom: 1px solid #4a86e8; }"
                                     "QLabel { color: #e8eefc; font-size: 11pt; }")
        rb = QHBoxLayout(self.ready_bar)
        rb.setContentsMargins(12, 6, 8, 6)
        self.ready_label = QLabel()
        rb.addWidget(self.ready_label, 1)
        b_go = QToolButton()
        b_go.setText("Open Stacking Window")
        b_go.setStyleSheet("QToolButton { background: #4a86e8; color: white; border-radius: 4px; "
                           "padding: 4px 14px; font-weight: bold; }")
        b_go.clicked.connect(self.open_stacking)
        b_x = QToolButton()
        b_x.setText("Not yet")
        b_x.clicked.connect(lambda: self.ready_bar.hide())
        rb.addWidget(b_go)
        rb.addWidget(b_x)
        self.ready_bar.hide()
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.ready_bar)
        cv.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self.toast = Toast(central)

        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _build_actions(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        st = self.style()

        def act(text, slot, shortcut=None, icon=None, tip=None):
            a = QAction(text, self)
            if icon is not None:
                a.setIcon(st.standardIcon(icon))
            if shortcut:
                if isinstance(shortcut, (list, tuple)):
                    a.setShortcuts([QKeySequence(s) for s in shortcut])
                else:
                    a.setShortcut(QKeySequence(shortcut))
                a.setShortcutContext(Qt.WindowShortcut)
            if tip:
                a.setToolTip(tip)
            a.triggered.connect(slot)
            self.addAction(a)
            return a

        self.act_load = act("Load Data…", self.load_data, "Ctrl+O", QStyle.SP_DirOpenIcon,
                            "Choose folders and files from one or more places, then start processing (Ctrl+O)")
        self.act_open = act("Open One Folder…", self.choose_folder, "Ctrl+Shift+O", None,
                            "Quickly open a single folder (Ctrl+Shift+O)")
        self.act_reject = act("Reject / Restore", self.toggle_reject, ["X", "Delete"],
                              QStyle.SP_DialogDiscardButton,
                              "Move the frame (or selected frames) to REJECTED, or bring back (X)")
        self.act_flag = act("Flag", self.toggle_flag, "F", None,
                            "Flag / unflag the frame or selected frames for review (F)")
        self.act_autoflag = act("Auto-Flag…", self.auto_flag, "Ctrl+F", None,
                                "Flag frames worse than thresholds you choose (Ctrl+F)")
        self.act_reject_flagged = act("Reject Flagged", self.reject_flagged, "Shift+X", None,
                                      "Reject every flagged frame (Shift+X)")
        self.act_clear_flags = act("Clear All Flags", self.clear_flags, "D")
        self.act_undo = act("Undo", self.undo, "Ctrl+Z", QStyle.SP_ArrowBack,
                            "Undo the last reject/restore, flag change, Auto-Flag or Auto-cull (Ctrl+Z)")
        self.act_reopen = act("Reopen Last Session", self.reopen_last_session, "Ctrl+Shift+R", None,
                              "Load the folders of your last session again, with its flags (Ctrl+Shift+R)")
        self.act_fit = act("Fit", self.view.fit, "Ctrl+0", None, "Fit image to window (Ctrl+0)")
        self.act_100 = act("100%", self.view.actual_size, "1", None, "Actual pixels (1)")
        self.act_export = act("Export Keeper List…", self.export_keepers, "Ctrl+E")
        self.act_next = act("Next frame", lambda: self.step(1), ["Right", "Down"])
        self.act_prev = act("Previous frame", lambda: self.step(-1), ["Left", "Up"])
        self.act_reload = act("Reload Folder", self.reload, "F5")
        self.act_remeasure = act("Re-measure All (ignore saved results)", self.remeasure)
        self.act_quit = act("Exit", self.close, "Ctrl+Q")

        self.act_stack = act("Stack Kept Frames…", self.open_stacking, "Ctrl+K", QStyle.SP_MediaPlay,
                             "Open the Stacking window with the frames you kept (Ctrl+K)")
        self.act_header = act("Show FITS Header…", self.show_header_window, "Ctrl+H", None,
                              "Show every header card of the current frame in its own window (Ctrl+H)")
        self.act_header_file = act("Inspect a File's FITS Header…", self.inspect_file_header, None, None,
                                   "Show the header of any FITS or XISF file")
        self.act_set_filter = act("Set Filter Name…", self.set_filter_dialog, None, None,
                                  "Tell SubZeroAstro which filter was used, for the selected frames or all frames "
                                  "(when the camera didn't record it). Your files are not changed.")
        self.act_show_image = QAction("Show Image", self, checkable=True)
        self.act_show_image.setChecked(True)
        self.act_show_image.setShortcut("Ctrl+I")
        self.act_show_image.setToolTip("Show or hide the big image (Ctrl+I). Hiding it stops loading previews "
                                       "from the drive - faster on a NAS.")
        self.act_show_image.toggled.connect(self.toggle_image)
        self.addAction(self.act_show_image)
        self.act_start_over = act("Start Over – Load New Data…", self.start_over, "Ctrl+N",
                                  QStyle.SP_BrowserReload,
                                  "Clear this session and open a new folder (Ctrl+N)")
        self.preset_actions = {}
        for key, p in core.CULL_PRESETS.items():
            a = QAction(p["label"], self)
            a.triggered.connect(lambda _=False, k=key: self.apply_preset(k))
            self.addAction(a)
            self.preset_actions[key] = a

        self.chk_sub = QCheckBox("Include subfolders")
        self.chk_sub.setToolTip("Also load FITS files in subfolders (FLAT, DARK, BIAS, CALIB, OFFSET "
                                "folders are skipped) - the default for Load Data and Open One Folder")
        self.act_advance = QAction("Jump to next frame after rejecting", self, checkable=True)
        self.act_advance.setChecked(True)

        self.mode_group = QActionGroup(self)
        self.mode_actions = {}
        for key, text in [("fast", "Fast — sample ¼ of each frame (best for network drives)"),
                          ("full", "Full frame — most thorough, reads every pixel"),
                          ("off", "Off — thumbnails only, no star metrics")]:
            a = QAction(text, self, checkable=True)
            a.setData(key)
            self.mode_group.addAction(a)
            self.mode_actions[key] = a

        tb.addAction(self.act_load)
        tb.addWidget(self.chk_sub)
        tb.addSeparator()
        tb.addAction(self.act_reject)
        tb.addAction(self.act_flag)
        tb.addAction(self.act_autoflag)
        tb.addAction(self.act_undo)
        tb.addSeparator()
        tb.addAction(self.act_fit)
        tb.addAction(self.act_100)
        tb.addAction(self.act_show_image)

        # Second row: auto-cull presets and Start Over
        self.addToolBarBreak()
        tb2 = QToolBar("Auto-cull")
        tb2.setMovable(False)
        tb2.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb2)
        lab = QLabel("  Auto-cull:  ")
        lab.setToolTip("Flag frames for rejection using FWHM, eccentricity and star count, each compared "
                       "with the median of its own filter.\nFlagged frames are only marked: review them in "
                       "the Flagged view, then press Shift+X (Reject Flagged).")
        tb2.addWidget(lab)
        colors = {"aggressive": "#e04848", "semi": "#e0a030", "less": "#46b85a"}
        for key, a in self.preset_actions.items():
            tb2.addAction(a)
            btn = tb2.widgetForAction(a)
            if btn is not None:
                btn.setStyleSheet(f"QToolButton {{ border: 1px solid {colors[key]}; border-radius: 4px; "
                                  f"padding: 3px 10px; margin: 2px 3px; }}"
                                  f"QToolButton:hover {{ background: {colors[key]}40; }}")
        tb2.addAction(self.act_reject_flagged)
        tb2.addSeparator()
        tb2.addAction(self.act_start_over)
        self.update_preset_tooltips()
        # next step, top right of the banner (like turning a page)
        self.btn_next = QPushButton()
        self.btn_next.setToolTip("Open the Calibrate and Stack page with the frames you kept (Ctrl+K)")
        self.btn_next.clicked.connect(self.open_stacking)
        self.app_header.add_next(self.btn_next, "Calibrate and Stack")

        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.act_load, self.act_open, self.act_reopen, self.act_start_over, self.act_reload,
                  self.act_remeasure):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_export)
        m.addAction(self.act_header_file)
        m.addSeparator()
        m.addAction(self.act_quit)
        m = mb.addMenu("&Frame")
        for a in (self.act_reject, self.act_undo):
            m.addAction(a)
        m.addSeparator()
        for a in (self.act_flag, self.act_autoflag, self.act_reject_flagged, self.act_clear_flags):
            m.addAction(a)
        m.addAction(self.act_set_filter)
        m.addAction(self.act_header)
        cm = m.addMenu("Auto-cull")
        for a in self.preset_actions.values():
            cm.addAction(a)
        m.addSeparator()
        m.addAction(self.act_prev)
        m.addAction(self.act_next)
        m.addSeparator()
        m.addAction(self.act_advance)
        m = mb.addMenu("&Stack")
        m.addAction(self.act_stack)
        a = QAction("Prepare Masters for PixInsight…", self)
        a.triggered.connect(self.open_prepare)
        m.addAction(a)
        a = QAction("Finish the Image…", self)
        a.setToolTip("Colour, brighten, stars, palettes - pick from previews, step by step")
        a.triggered.connect(self.open_finish)
        m.addAction(a)
        m = mb.addMenu("&View")
        m.addAction(self.act_show_image)
        m.addAction(self.act_fit)
        m.addAction(self.act_100)
        m = mb.addMenu("&Settings")
        sm = m.addMenu("Star measurement")
        for a in self.mode_actions.values():
            sm.addAction(a)
        a = QAction("Saved Progress (crash recovery)…", self)
        a.triggered.connect(lambda: CheckpointDialog(self, self.settings).exec())
        m.addAction(a)
        m = mb.addMenu("&Help")
        a = QAction("Keyboard Shortcuts", self)
        a.triggered.connect(self.show_keys)
        m.addAction(a)
        a = QAction("About the Quality Metrics", self)
        a.triggered.connect(self.show_metrics_help)
        m.addAction(a)
        a = QAction("About", self)
        a.triggered.connect(self.show_about)
        m.addAction(a)

    # ---------------------------------------------------------- settings
    def _restore_settings(self):
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        else:
            self.resize(1600, 950)
        for key, w in (("splitter", self.splitter), ("vsplit", self.vsplit)):
            s = self.settings.value(key)
            if s is not None:
                w.restoreState(s)
        self.chk_sub.setChecked(self.settings.value("subfolders", False, type=bool))
        self.act_advance.setChecked(self.settings.value("advance", True, type=bool))
        mode = self.settings.value("measure_mode", "fast")
        self.mode_actions.get(mode, self.mode_actions["fast"]).setChecked(True)
        idx = self.chart_metric.findData(self.settings.value("chart_metric", "fwhm"))
        if idx >= 0:
            self.chart_metric.setCurrentIndex(idx)

    def measure_mode(self) -> str:
        for k, a in self.mode_actions.items():
            if a.isChecked():
                return k
        return "fast"

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
        self.settings.setValue("vsplit", self.vsplit.saveState())
        self.settings.setValue("subfolders", self.chk_sub.isChecked())
        self.settings.setValue("advance", self.act_advance.isChecked())
        self.settings.setValue("measure_mode", self.measure_mode())
        self.settings.setValue("chart_metric", self.chart_metric.currentData())
        self.generation += 1
        self.analyze_pool.clear()
        self.display_pool.clear()
        self.save_disk_cache()
        self.save_session(clean=True)
        super().closeEvent(event)

    # ------------------------------------------------------------ opening
    def load_data(self, initial=None):
        """Show the Load Data window; start processing what was collected."""
        dlg = LoadDataDialog(self, self.settings, self.chk_sub.isChecked(), initial)
        if dlg.exec():
            self.chk_sub.setChecked(dlg.chk_sub.isChecked())
            self.open_sources(dlg.sources(), filter_name=dlg.filter_override())

    def choose_folder(self):
        start = self.settings.value("last_folder", str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, "Open folder of FITS files", start)
        if d:
            self.open_folder(d)

    def open_folder(self, folder, use_cache=True):
        folder = Path(folder)
        self.settings.setValue("last_folder", str(folder))
        self.open_sources([("folder", folder, self.chk_sub.isChecked())], use_cache)

    def open_sources(self, sources, use_cache=True, filter_name=None):
        self.statusBar().showMessage("Scanning folders…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        dups = []
        try:
            QApplication.processEvents()
            frames, root = core.scan_sources(sources, dups)
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
        if not frames:
            where = "\n".join(str(s[1]) for s in sources if s[0] == "folder")
            QMessageBox.information(self, APP_NAME,
                                    "No FITS files (.fits, .fit, .fts) were found" + (f" in:\n{where}" if where else ".")
                                    + "\n\nTip: tick “Include subfolders” if your frames are in subfolders.")
            return
        self.sources = list(sources)
        if filter_name:
            for f in frames:
                self.filter_overrides[str(f.path)] = filter_name
        for f in frames:
            f.filter_override = self.filter_overrides.get(str(f.path))
        self.open_frames(frames, root, use_cache)
        self.session_timer.start()
        if dups:
            self.toast.show_message(f"{len(dups)} duplicate frame{'s were' if len(dups) != 1 else ' was'} skipped "
                                    "- the same file added twice, or the same frame in two folders.", 7)
            self.log_dups = dups
        n_src = len(sources)
        if n_src > 1:
            self.setWindowTitle(f"{APP_NAME} — {n_src} sources under {root}")

    def reload(self):
        if self.sources:
            self.open_sources(self.sources)

    def remeasure(self):
        if self.sources:
            self.open_sources(self.sources, use_cache=False)

    def open_frames(self, frames, root, use_cache=True):
        self.save_disk_cache()
        self.generation += 1
        self.analyze_pool.clear()
        self.display_pool.clear()
        self.root = Path(root)
        self.frames = {f.uid: f for f in frames}
        self.loaded_count = 0
        self.undo_stack.clear()
        self.cache.clear()
        self.pending_display.clear()
        self.known_filters = []
        self.medians = {}
        self.current_uid = None
        self.disk_cache = {}
        self.cache_dirty = set()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All filters")
        self.filter_combo.blockSignals(False)
        self.view.clear("Loading…")
        self.setWindowTitle(f"{APP_NAME} — {self.root}")

        self.network = core.is_network_path(self.root)
        cores = os.cpu_count() or 4
        self.analyze_pool.setMaxThreadCount(3 if self.network else max(2, min(cores - 1, 8)))

        # Results saved from earlier sessions
        mode = self.measure_mode()
        todo = []
        for f in frames:
            if f.folder not in self.disk_cache:
                self.disk_cache[f.folder] = core.load_cache(f.folder) if use_cache else {}
            entry = self.disk_cache[f.folder].get(f.cache_key)
            usable = entry is not None and (entry.get("mode") == mode or
                                            (mode == "fast" and entry.get("mode") == "full"))
            if usable:
                self.apply_result(f, dict(entry, qthumb=to_qimage(entry["thumb"])), from_cache=True)
            else:
                todo.append(f)
        cached = len(frames) - len(todo)
        self.to_load = len(todo)
        self.load_started = time.time()

        self.progress.setRange(0, max(1, len(frames)))
        self.progress.setValue(cached)
        self.progress.setVisible(bool(todo))
        self.loaded_count = cached
        self.rebuild_list()
        order_index = {u: i for i, u in enumerate(self.order)}
        todo.sort(key=lambda f: order_index.get(f.uid, 1 << 30))
        for f in todo:
            self.analyze_pool.start(AnalyzeTask(self.generation, f.uid, f.path, mode, self.signals))
        if self.order:
            self.list.setCurrentRow(0)
        self.refresh_stats()
        if cached:
            self.statusBar().showMessage(f"{cached} frames loaded instantly from saved results"
                                         + (f"; measuring {len(todo)} new frames…" if todo else ""), 6000)
        if not todo:
            self.loading_finished()

    # ------------------------------------------------------ drag and drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        if len(paths) == 1 and paths[0].is_dir():
            self.open_folder(paths[0])
            return
        rec = self.chk_sub.isChecked()
        initial = [("folder", p, rec) for p in paths if p.is_dir()]
        files = [p for p in paths if p.is_file()]
        if files:
            initial.append(("files", files))
        self.load_data(initial)

    # ----------------------------------------------------------- loading
    def apply_result(self, f: core.Frame, res: dict, from_cache=False):
        f.loaded = True
        f.error = None
        f.thumb = res["qthumb"]
        f.thumb_arr = res["thumb"]
        f.stretch = tuple(res["stretch"])
        f.display_size = tuple(res["display_size"])
        f.cards = res["cards"]
        f.info = dict(res["info"])
        if f.filter_override:
            f.info["filter"] = f.filter_override
        f.metrics = res.get("metrics") or {}
        f.night = core.night_of(f.info)
        filt = f.info.get("filter") or "No filter"
        if filt not in self.known_filters:
            self.known_filters.append(filt)
            self.filter_combo.blockSignals(True)
            cur = self.filter_combo.currentText()
            self.filter_combo.clear()
            self.filter_combo.addItem("All filters")
            self.filter_combo.addItems(sorted(self.known_filters))
            self.filter_combo.setCurrentText(cur)
            self.filter_combo.blockSignals(False)
        if not from_cache:
            self.disk_cache.setdefault(f.folder, {})[f.cache_key] = core.cache_entry(res)
            self.cache_dirty.add(f.folder)

    def on_analyzed(self, gen, uid, res):
        if gen != self.generation or uid not in self.frames:
            return
        f = self.frames[uid]
        self.apply_result(f, res)
        self.loaded_count += 1
        if self.loaded_count == min(10, len(self.frames)):
            QTimer.singleShot(0, self.suggest_filter_from_names)
        self.progress.setValue(self.loaded_count)
        self.refresh_item(uid)
        if uid == self.current_uid:
            self.show_current()
        self.stats_timer.start()
        self.check_loading_done()

    def on_analyze_failed(self, gen, uid, msg):
        if gen != self.generation or uid not in self.frames:
            return
        f = self.frames[uid]
        f.loaded = True
        f.error = msg
        self.loaded_count += 1
        self.progress.setValue(self.loaded_count)
        self.refresh_item(uid)
        if uid == self.current_uid:
            self.show_current()
        self.check_loading_done()

    def check_loading_done(self):
        n = len(self.frames)
        if self.loaded_count >= n:
            self.loading_finished()
        else:
            done = self.loaded_count
            elapsed = time.time() - self.load_started
            measured = done - (n - self.to_load)
            if measured > 3 and elapsed > 2:
                left = (n - done) * elapsed / measured
                self.progress.setFormat(f"%v / %m  ·  ~{left / 60:.0f} min left" if left > 90
                                        else f"%v / %m  ·  ~{left:.0f} s left")
            self.update_status()

    def suggest_filter_from_names(self):
        """The camera recorded no filter, but the file names mention one ("…Lult_0107.fit")."""
        if getattr(self, "_filter_asked", None) == self.generation:
            return
        measured = [f for f in self.frames.values() if f.loaded and not f.filter_override]
        if len(measured) < min(10, len(self.frames)) or len(measured) < 3:
            return                       # ask early: after the first few frames, not after all of them
        if any((f.info.get("filter") or "No filter") != "No filter" for f in measured):
            return                       # the files do record a filter
        self._filter_asked = self.generation
        # frames not measured yet are included too, so the answer covers the whole set
        frames = [f for f in self.frames.values() if not f.filter_override
                  and (not f.loaded or (f.info.get("filter") or "No filter") == "No filter")]
        found = {}
        for f in frames:
            name = core.filter_from_name(f.path.name) or core.filter_from_name(f.folder.name)
            if name:
                found.setdefault(name, []).append(f.uid)
        if not found:
            return
        name, uids = max(found.items(), key=lambda kv: len(kv[1]))
        if len(uids) < 0.6 * len(frames):
            return
        if getattr(self, "auto_answer_filter", None) is not None:
            ok = self.auto_answer_filter
        else:
            ok = QMessageBox.question(
                self, APP_NAME, f"The files record no filter, but {len(uids)} of their names mention "
                f"“{name}”.\n\nUse {name} as the filter for these frames? (Used for grouping, stacking, matching "
                "flats and choosing colour palettes. Your files are not changed. You can change it any time "
                "with Frame › Set Filter Name….)") == QMessageBox.StandardButton.Yes
        if ok:
            self.set_filter(uids, name)

    def loading_finished(self):
        QTimer.singleShot(300, self.suggest_filter_from_names)
        self.progress.hide()
        self.progress.setFormat("%v / %m")
        self.refresh_stats()
        if self.sort_combo.currentText() != "File name" or self.filter_combo.currentIndex() > 0:
            self.rebuild_list()
        self.save_disk_cache()
        if self.to_load:
            el = time.time() - self.load_started
            self.statusBar().showMessage(f"Measured {self.to_load} frames in {el:.0f} s"
                                         + (" (network drive)" if self.network else ""), 8000)
        self.update_status()

    def save_disk_cache(self):
        for folder in list(self.cache_dirty):
            try:
                core.save_cache(folder, self.disk_cache.get(folder, {}))
            except OSError:
                pass
        self.cache_dirty.clear()

    # ------------------------------------------------------------ stats
    def refresh_stats(self):
        """Recompute group medians, badges and scores; refresh what changed."""
        frames = list(self.frames.values())
        before = {f.uid: (f.badge, None if f.score is None else round(f.score)) for f in frames}
        self.medians = core.assess(frames)
        for f in frames:
            if before.get(f.uid) != (f.badge, None if f.score is None else round(f.score)):
                self.refresh_item(f.uid)
        self.update_chart()
        self.update_warnings()
        self.update_preset_tooltips()
        if self.current_uid is not None:
            self.fill_quality(self.frames.get(self.current_uid))
        self.update_status()

    def update_preset_tooltips(self):
        frames = list(self.frames.values())
        measured = sum(1 for f in frames if f.metrics and not f.rejected)
        for key, a in self.preset_actions.items():
            p = core.CULL_PRESETS[key]
            desc = (f"{p['label']}: flag frames with FWHM over {p['fwhm']:.2f}× the median, "
                    f"star count under {100 * p['stars']:.0f}% of the median, eccentricity over "
                    f"{p['ecc_abs']:.2f} (or {p['ecc_rel']:.2f} above the median), SNR under "
                    f"{100 * p['snr']:.0f}% of the median (cloud/haze), or that are somewhat worse "
                    "in several of these at once.")
            if measured:
                n = len(core.cull_preset(frames, self.medians, key))
                desc += f"\n\nWould flag {n} of {measured} frames ({100 * n / measured:.0f}%)."
            a.setToolTip(desc)
            a.setStatusTip(desc.replace("\n\n", "  "))

    def update_warnings(self):
        msgs = []
        groups = {}
        fwhms = {}
        for f in self.frames.values():
            if f.metrics and f.metrics.get("eccentricity") is not None and not f.rejected:
                groups.setdefault(f.info.get("filter") or "No filter", []).append(f.metrics["eccentricity"])
                if f.metrics.get("fwhm"):
                    fwhms.setdefault(f.info.get("filter") or "No filter", []).append(f.metrics["fwhm"])
        for filt, es in sorted(groups.items()):
            if len(es) >= 3:
                es.sort()
                med = es[len(es) // 2]
                fw = sorted(fwhms.get(filt, []))
                fmed = fw[len(fw) // 2] if fw else None
                if fmed is not None and fmed < 2.0:
                    # stars only ~1.5 px wide: roundness can't be measured reliably, round stars read 0.4-0.6
                    if med > 0.75:
                        msgs.append(f"⚠ {filt}: stars look elongated (median e {med:.2f}), but they are very small "
                                    f"(FWHM {fmed:.1f} px, undersampled) so this reading is uncertain — "
                                    "check a few frames at 100%")
                    continue
                if med > 0.7:
                    msgs.append(f"⚠ {filt}: stars elongated in most frames (median e {med:.2f}) — "
                                "check tracking, guiding or wind")
                elif med > 0.55:
                    msgs.append(f"⚠ {filt}: stars slightly out of round in most frames (median e {med:.2f}) — "
                                "possible tilt, collimation or pinched optics")
        self.warning_label.setText("\n".join(msgs))
        self.warning_label.setToolTip("\n".join(msgs))

    # ------------------------------------------------------------- chart
    def chart_frames(self):
        fi = self.filter_combo.currentIndex()
        filt = self.filter_combo.currentText()
        frames = [f for f in self.frames.values()
                  if fi <= 0 or (f.info.get("filter") or "No filter") == filt]
        frames.sort(key=lambda f: (str(f.info.get("date_obs") or "~"), core.natural_key(str(f.path))))
        return frames

    def update_chart(self, *_):
        key = self.chart_metric.currentData() or "fwhm"
        sel = set(self.selected_uids(multi_only=True))
        pts = []
        for f in self.chart_frames():
            filt = f.info.get("filter") or "No filter"
            pts.append(ChartPoint(f.uid, core.frame_value(f, key), filter_color(filt), f.rejected,
                                  f.flagged, f.uid in sel, f.night, f.name, f.group))
        medians = []
        if key != "score":
            fi = self.filter_combo.currentIndex()
            for g, gm in self.medians.items():
                filt = g.split("|", 1)[1]
                if fi > 0 and filt != self.filter_combo.currentText():
                    continue
                if gm.get(key) is not None:
                    medians.append((filter_color(filt), gm[key]))
        self.chart.set_data(pts, medians, self.current_uid, core.METRIC_NAMES[key],
                            core.METRIC_HIGHER_BETTER[key])
        filts = sorted({p.group.split("|", 1)[1] for p in pts})
        self.legend.setText("  ".join(f"<span style='color:{filter_color(x).name()}'>●</span> {x}"
                                      for x in filts))

    def goto_uid(self, uid):
        if uid not in self.items:
            # Not in the current list (e.g. rejected frame while showing Kept): show all
            self.show_combo.setCurrentIndex(0)
        it = self.items.get(uid)
        if it is not None:
            self.list.setCurrentItem(it)
            self.list.scrollToItem(it)

    def select_uids(self, uids):
        visible = [u for u in uids if u in self.items]
        if not visible:
            return
        self.list.blockSignals(True)
        self.list.clearSelection()
        self.list.setCurrentItem(self.items[visible[0]], QItemSelectionModel.SelectionFlag.NoUpdate)
        for u in visible:
            self.items[u].setSelected(True)
        self.list.blockSignals(False)
        self.list.scrollToItem(self.items[visible[0]])
        self.current_uid = visible[0]
        self.show_current()
        self.update_chart()
        self.statusBar().showMessage(f"{len(visible)} frames selected — press X to reject or F to flag", 6000)

    # ----------------------------------------------------- display loads
    def request_display(self, uid, full=False):
        key = (uid, full)
        if key in self.cache or key in self.pending_display or uid not in self.frames:
            return
        f = self.frames[uid]
        if f.error:
            return
        self.pending_display.add(key)
        self.display_pool.start(DisplayTask(self.generation, uid, f.path, f.stretch, full, self.signals),
                                2 if uid == self.current_uid else 0)

    def request_detail(self):
        uid = self.current_uid
        if uid is None or (uid, True) in self.cache:
            return
        half = self.cache.get((uid, False))
        if half is not None and half[1] == 1:
            return          # the "half" image is already full resolution
        self.request_display(uid, full=True)

    def on_display_ready(self, gen, uid, img, factor, full, path):
        if gen != self.generation:
            return
        key = (uid, full)
        self.pending_display.discard(key)
        self.cache[key] = (img, factor)
        if not full and uid in self.frames:
            preview_save(self.frames[uid], img, factor)
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            old = next(iter(self.cache))
            if old[0] == self.current_uid:
                self.cache.move_to_end(old)
                if all(k[0] == self.current_uid for k in self.cache):
                    break
                continue
            del self.cache[old]
        if uid == self.current_uid:
            f = self.frames[uid]
            if full or (uid, True) not in self.cache:
                self.view.show_image(img, self.full_size(f, img, factor))
            if not full and factor > 1 and not self.view.fit_mode and self.view.transform().m11() >= 0.5:
                self.request_display(uid, full=True)

    def on_display_failed(self, gen, uid, msg, path):
        if gen != self.generation:
            return
        self.pending_display.discard((uid, False))
        self.pending_display.discard((uid, True))
        f = self.frames.get(uid)
        if f is None:
            return
        if str(f.path) != path:
            self.request_display(uid)
            return
        if uid == self.current_uid:
            self.view.clear(f"Could not display {f.name}:\n{msg}")

    @staticmethod
    def full_size(f: core.Frame, img: QImage | None = None, factor: int = 1):
        """Size of the frame in display pixels. The display image can arrive
        before the frame's measurement, so fall back to the image itself."""
        if f.display_size:
            return f.display_size
        if img is not None and img.width() > 0:
            return (img.width() * factor, img.height() * factor)
        if f.thumb is not None and f.thumb.width() > 0:
            return (f.thumb.width(), f.thumb.height())
        return (1, 1)

    # -------------------------------------------------------------- list
    def frame_visible(self, f: core.Frame) -> bool:
        show = self.show_combo.currentText()
        if show == "Kept" and f.rejected:
            return False
        if show == "Rejected" and not f.rejected:
            return False
        if show == "Flagged" and not f.flagged:
            return False
        if self.filter_combo.currentIndex() > 0 and \
                (f.info.get("filter") or "No filter") != self.filter_combo.currentText():
            return False
        return True

    def sorted_frames(self, frames):
        mode = self.sort_combo.currentText()
        if mode in SORT_METRIC:
            key = SORT_METRIC[mode]
            hb = core.METRIC_HIGHER_BETTER[key]
            worst = self.worst_first.isChecked()

            def k(f):
                v = core.frame_value(f, key)
                if v is None:
                    return (1, 0.0, core.natural_key(f.name))
                good_first = -v if hb else v
                return (0, -good_first if worst else good_first, core.natural_key(f.name))
            return sorted(frames, key=k)
        if mode == "Capture time":
            return sorted(frames, key=lambda f: (str(f.info.get("date_obs") or "~"), core.natural_key(str(f.path))))
        if mode == "Filter":
            return sorted(frames, key=lambda f: (f.info.get("filter") or "~", core.natural_key(f.rel_folder),
                                                 core.natural_key(f.name)))
        return sorted(frames, key=lambda f: (core.natural_key(f.rel_folder), core.natural_key(f.name)))

    def on_sort_changed(self, *_):
        """A new sort order shows the top of the list (e.g. the worst frames)."""
        self.rebuild_list()
        if self.list.count():
            self.list.setCurrentRow(0)
            self.list.scrollToTop()

    def on_view_changed(self, *_):
        self.rebuild_list()
        self.update_chart()

    def rebuild_list(self, *_):
        keep = self.current_uid
        self.list.blockSignals(True)
        self.list.clear()
        self.items.clear()
        frames = self.sorted_frames([f for f in self.frames.values() if self.frame_visible(f)])
        self.order = [f.uid for f in frames]
        for f in frames:
            it = QListWidgetItem()
            it.setData(ROLE_UID, f.uid)
            self.list.addItem(it)
            self.items[f.uid] = it
            self.refresh_item(f.uid)
        self.list.blockSignals(False)
        if keep in self.items:
            self.list.setCurrentItem(self.items[keep])
            self.list.scrollToItem(self.items[keep])
            if self.current_uid != keep:
                self.current_uid = keep
                self.show_current()
        elif self.order:
            self.list.setCurrentRow(0)
            self.current_uid = self.order[0]
            self.show_current()
        else:
            self.current_uid = None
            self.show_current()
        self.update_status()

    def make_icon(self, f: core.Frame) -> QIcon:
        pm = QPixmap(ICON_W, ICON_H)
        pm.fill(QColor(20, 20, 24))
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if f.thumb is not None and not f.thumb.isNull():
            t = QPixmap.fromImage(f.thumb).scaled(ICON_W, ICON_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap((ICON_W - t.width()) // 2, (ICON_H - t.height()) // 2, t)
        else:
            p.setPen(QColor(140, 140, 150))
            p.drawText(pm.rect(), Qt.AlignCenter, "error" if f.error else "loading…")
        if f.rejected:
            p.fillRect(pm.rect(), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor(255, 70, 70), 3))
            p.drawRect(pm.rect().adjusted(1, 1, -2, -2))
            font = QFont(p.font())
            font.setBold(True)
            font.setPointSize(9)
            p.setFont(font)
            p.drawText(pm.rect(), Qt.AlignCenter, "REJECTED")
        if f.badge in BADGE_COLORS:
            p.setPen(QPen(QColor(0, 0, 0), 1.5))
            p.setBrush(BADGE_COLORS[f.badge])
            p.drawEllipse(ICON_W - 17, 4, 13, 13)
        if f.flagged:
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.setBrush(QColor(255, 220, 60))
            p.drawPolygon(QPolygonF([QPointF(4, 4), QPointF(18, 4), QPointF(4, 18)]))
        p.end()
        return QIcon(pm)

    def refresh_item(self, uid):
        it = self.items.get(uid)
        f = self.frames.get(uid)
        if it is None or f is None:
            return
        lines = [f.name]
        details = []
        if f.info:
            details.append(f.info.get("filter") or "No filter")
            exp = f.info.get("exptime")
            if isinstance(exp, (int, float)):
                details.append(f"{exp:g} s")
        if f.score is not None:
            details.append(f"score {f.score:.0f}")
        if f.error:
            details.append("⚠ unreadable")
        if details:
            lines.append(" · ".join(details))
        m = f.metrics
        if m and m.get("fwhm") is not None:
            lines.append(f"FWHM {m['fwhm']:.2f} · e {m['eccentricity']:.2f} · {m['stars']}★")
        elif m:
            lines.append("no stars measured")
        if f.rel_folder:
            lines.append(f"📁 {f.rel_folder}")
        it.setText("\n".join(lines))
        it.setIcon(self.make_icon(f))
        tip = str(f.path)
        if f.reasons:
            tip += "\n\n" + "\n".join(f.reasons)
        if f.error:
            tip += f"\n\n{f.error}"
        it.setToolTip(tip)
        it.setForeground(QBrush(QColor(230, 110, 110) if f.rejected else QColor(225, 225, 230)))

    def on_current_changed(self, cur, _prev):
        self.current_uid = cur.data(ROLE_UID) if cur is not None else None
        self.show_current()
        self.chart.set_current(self.current_uid)

    def on_selection_changed(self):
        self.update_chart()

    def step(self, delta):
        if not self.order:
            return
        row = self.list.currentRow()
        row = max(0, min(self.list.count() - 1, row + delta))
        self.list.setCurrentRow(row)

    # ------------------------------------------------------- show frame
    def show_current(self):
        uid = self.current_uid
        f = self.frames.get(uid) if uid is not None else None
        self.fill_inspector(f)
        if f is None:
            self.view.clear("No frames to show" if self.frames else None)
            return
        if not self.view.isVisible() and not self.act_show_image.isChecked():
            return                                    # image hidden: don't read previews from the drive
        if (uid, False) not in self.cache and (uid, True) not in self.cache and not f.error:
            pv = preview_load(f)
            if pv is not None:                        # screen-sized preview saved on this computer
                self.cache[(uid, False)] = pv
        # what you click comes first: drop queued look-ahead loads
        self.display_pool.clear()
        self.pending_display = {k for k in self.pending_display if k[0] == uid}
        if f.error:
            self.view.clear(f"Could not read {f.name}:\n{f.error}")
        elif (uid, True) in self.cache:
            self.cache.move_to_end((uid, True))
            img, fac = self.cache[(uid, True)]
            self.view.show_image(img, self.full_size(f, img, fac))
        elif (uid, False) in self.cache:
            self.cache.move_to_end((uid, False))
            img, fac = self.cache[(uid, False)]
            self.view.show_image(img, self.full_size(f, img, fac))
            if not self.view.fit_mode and self.view.transform().m11() >= 0.5:
                self.request_detail()
        elif f.thumb is not None:
            self.view.show_image(f.thumb, self.full_size(f))
            self.request_display(uid)
        else:
            self.view.clear("Loading…")
            self.request_display(uid)
        self.view.banner = "REJECTED" if f.rejected else ("FLAGGED" if f.flagged else "")
        self.view.subtitle = f.name
        self.view.viewport().update()
        if uid in self.order:
            i = self.order.index(uid)
            for j in (i + 1, i - 1, i + 2):
                if 0 <= j < len(self.order) and self.frames[self.order[j]].loaded:
                    self.request_display(self.order[j])

    def fill_quality(self, f):
        Q = self.q_labels
        if f is None or not f.metrics:
            for lab in Q.values():
                lab.setText("–")
            self.reasons_label.setText("")
            return
        m = f.metrics
        gm = self.medians.get(f.group, {})

        def vs(key, fmt, unit=""):
            v = m.get(key)
            if v is None:
                return "–"
            s = fmt.format(v) + unit
            med = gm.get(key)
            if med:
                s += f"   <span style='color:#8a8a96'>(median {fmt.format(med)}{unit})</span>"
            return s

        Q["score"].setText("–" if f.score is None else
                           f"<b style='color:{BADGE_COLORS.get(f.badge, QColor('#ccc')).name()}'>{f.score:.0f}</b>"
                           "   <span style='color:#8a8a96'>(100 = typical)</span>")
        fw = vs("fwhm", "{:.2f}", " px")
        sc = f.info.get("scale")
        if sc and m.get("fwhm"):
            fw = fw.replace(" px", f" px = {m['fwhm'] * sc:.1f}″", 1)
        Q["fwhm"].setText(fw)
        Q["eccentricity"].setText(vs("eccentricity", "{:.2f}"))
        Q["snr"].setText(vs("snr", "{:.0f}"))
        Q["stars"].setText(vs("stars", "{:.0f}"))
        Q["background"].setText(vs("background", "{:.0f}", " ADU"))
        lines = []
        if f.reasons:
            col = BADGE_COLORS.get(f.badge, QColor("#ccc")).name()
            lines += [f"<span style='color:{col}'>• {r}</span>" for r in f.reasons]
        elif f.badge == "green":
            lines.append("<span style='color:#46c85f'>• Within the normal range for this filter</span>")
        if f.flagged and f.flag_note:
            lines.append(f"<span style='color:#ffdc3c'>⚑ {f.flag_note}</span>")
        self.reasons_label.setText("<br>".join(lines))

    def fill_inspector(self, f):
        L = self.info_labels
        self.fill_quality(f)
        if f is None:
            for lab in L.values():
                lab.setText("–")
            self.header_table.setRowCount(0)
            return
        i = f.info or {}

        def fmt(v, suffix=""):
            if v in (None, ""):
                return "–"
            if isinstance(v, float):
                return f"{v:g}{suffix}"
            return f"{v}{suffix}"

        L["file"].setText(f.name)
        if f.rejected:
            L["status"].setText("<span style='color:#ff6060'><b>Rejected</b></span>")
        elif f.flagged:
            L["status"].setText("<span style='color:#ffdc3c'><b>Flagged</b></span> (kept)")
        else:
            L["status"].setText("<span style='color:#60d060'>Kept</span>")
        parent = f.path.parent
        short = str(Path(*parent.parts[-2:])) if len(parent.parts) > 2 else str(parent)
        L["folder"].setText(("…" + os.sep if len(parent.parts) > 2 else "") + short)
        L["folder"].setToolTip(str(parent))
        L["object"].setText(fmt(i.get("object")))
        L["filter"].setText(fmt(i.get("filter")))
        L["exptime"].setText(fmt(i.get("exptime"), " s"))
        L["date_obs"].setText(fmt(i.get("date_loc") or i.get("date_obs")) + ("" if i.get("date_loc") else " UTC"
                                                                             if i.get("date_obs") else ""))
        L["night"].setText(fmt(f.night))
        L["camera"].setText(fmt(i.get("camera")))
        L["gain"].setText(fmt(i.get("gain")))
        L["temp"].setText(fmt(i.get("temp"), " °C"))
        if f.display_size:
            w, h = f.display_size
            L["size"].setText(f"{w * 2} × {h * 2}" if i.get("bayer") else f"{w} × {h}")
        else:
            L["size"].setText("–")
        L["bayer"].setText(f"{i['bayer']} (colour camera, shown in greyscale)" if i.get("bayer")
                           else ("No (mono)" if f.info else "–"))
        self.fill_header_table()

    def fill_header_table(self, *_):
        f = self.frames.get(self.current_uid) if self.current_uid is not None else None
        rows = []
        if f is not None:
            q = self.header_search.text().strip().lower()
            for k, v, c in f.cards:
                if v is None:
                    val = c
                elif isinstance(v, bool):
                    val = "T" if v else "F"
                else:
                    val = str(v)
                if q and q not in k.lower() and q not in val.lower():
                    continue
                rows.append((k, val, c))
        self.header_table.setRowCount(len(rows))
        for r, (k, val, c) in enumerate(rows):
            ki = QTableWidgetItem(k)
            vi = QTableWidgetItem(val)
            if c:
                ki.setToolTip(c)
                vi.setToolTip(c)
            self.header_table.setItem(r, 0, ki)
            self.header_table.setItem(r, 1, vi)

    # ---------------------------------------------------- reject / undo
    def selected_uids(self, multi_only=False):
        uids = [it.data(ROLE_UID) for it in self.list.selectedItems()]
        if len(uids) <= 1:
            if multi_only:
                return []
            return [self.current_uid] if self.current_uid is not None else []
        return sorted(uids, key=lambda u: self.order.index(u) if u in self.order else 0)

    def toggle_reject(self):
        uids = self.selected_uids()
        if not uids:
            return
        reject = not all(self.frames[u].rejected for u in uids)
        done = self.apply_reject(uids, reject)
        if done:
            self.undo_stack.append({"label": "reject" if reject else "restore", "reject": done})
        if reject and len(uids) == 1 and self.act_advance.isChecked() and self.show_combo.currentText() == "All frames":
            self.step(1)

    def reject_flagged(self):
        uids = [u for u, f in self.frames.items() if f.flagged and not f.rejected]
        if not uids:
            QMessageBox.information(self, APP_NAME, "No frames are flagged.\n\n"
                                    "Flag frames with F, by dragging across the chart, or with Auto-Flag.")
            return
        if QMessageBox.question(self, APP_NAME, f"Move {len(uids)} flagged frames to REJECTED?\n\n"
                                "(You can undo this with Ctrl+Z.)") != QMessageBox.StandardButton.Yes:
            return
        flags = self.flag_snapshot()
        done = self.apply_reject(uids, True)
        for u, _r in done:
            self.frames[u].flagged = False
            self.refresh_item(u)
        if done:
            self.undo_stack.append({"label": "Reject Flagged", "reject": done, "flags": flags})
        self.rebuild_list()
        self.update_chart()
        if done:
            self.show_ready_bar()

    def apply_reject(self, uids, reject):
        done, errors = [], []
        for u in uids:
            f = self.frames[u]
            if f.rejected == reject:
                continue
            try:
                core.set_rejected(f, reject)
                done.append((u, reject))
            except OSError as e:
                errors.append(f"{f.name}: {e}")
        if errors:
            QMessageBox.warning(self, APP_NAME,
                                "Some files could not be moved (are they open in another program?):\n\n"
                                + "\n".join(errors[:10]) + ("\n…" if len(errors) > 10 else ""))
        if done:
            self.session_timer.start()
            if self.show_combo.currentText() != "All frames":
                row = self.list.currentRow()
                self.rebuild_list()
                if self.list.count():
                    self.list.setCurrentRow(min(max(row, 0), self.list.count() - 1))
            else:
                for u, _r in done:
                    self.refresh_item(u)
                self.show_current()
            self.refresh_stats()
        return done

    def undo(self):
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo", 2000)
            return
        last = self.undo_stack.pop()
        if isinstance(last, list):
            last = {"label": "change", "reject": last}
        rej = last.get("reject") or []
        uids = [u for u, _r in rej]
        by_state = {}
        for u, r in rej:
            by_state.setdefault(not r, []).append(u)
        for state, us in by_state.items():
            self.apply_reject(us, state)
        if last.get("flags") is not None:
            self.restore_flags(last["flags"])
        if uids and uids[0] in self.items:
            self.list.setCurrentItem(self.items[uids[0]])
        what = last.get("label", "change")
        self.statusBar().showMessage(f"Undid {what}" + (f" ({len(uids)} frame(s) moved back)" if uids else ""), 3000)

    # -------------------------------------------------------------- flags
    def flag_snapshot(self) -> dict:
        return {u: f.flag_note for u, f in self.frames.items() if f.flagged}

    def push_flag_undo(self, label):
        self.undo_stack.append({"label": label, "flags": self.flag_snapshot()})

    def restore_flags(self, snap: dict):
        changed = [u for u, f in self.frames.items() if f.flagged != (u in snap) or
                   (u in snap and f.flag_note != snap[u])]
        for u in changed:
            f = self.frames[u]
            f.flagged = u in snap
            f.flag_note = snap.get(u, "")
        if self.show_combo.currentText() == "Flagged":
            self.rebuild_list()
        else:
            for u in changed:
                self.refresh_item(u)
            self.show_current()
        self.update_chart()
        self.update_status()
        self.session_timer.start()

    def toggle_flag(self):
        uids = self.selected_uids()
        if not uids:
            return
        state = not all(self.frames[u].flagged for u in uids)
        self.push_flag_undo("flag" if state else "unflag")
        self.set_flags(uids, state)

    def set_flags(self, uids, state):
        for u in uids:
            self.frames[u].flagged = state
            if not state:
                self.frames[u].flag_note = ""
        if self.show_combo.currentText() == "Flagged":
            self.rebuild_list()
        else:
            for u in uids:
                self.refresh_item(u)
            self.show_current()
        self.update_chart()
        self.update_status()
        self.session_timer.start()

    def clear_flags(self):
        uids = [u for u, f in self.frames.items() if f.flagged]
        if uids:
            self.push_flag_undo("Clear All Flags")
            self.set_flags(uids, False)

    def apply_preset(self, level):
        """Flag exactly the frames this preset picks (replacing earlier flags)."""
        p = core.CULL_PRESETS[level]
        frames = list(self.frames.values())
        measured = [f for f in frames if f.metrics and not f.rejected]
        if not measured:
            QMessageBox.information(self, APP_NAME, "No quality metrics yet — wait for frames to be measured "
                                    "(and check Settings → Star measurement is not Off).")
            return
        if self.loaded_count < len(self.frames):
            if QMessageBox.question(self, APP_NAME, f"{len(self.frames) - self.loaded_count} frames are still being "
                                    "measured and will not be included.\n\nApply the preset anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        picked = set(core.cull_preset(frames, self.medians, level))
        self.push_flag_undo(p["label"])
        for f in frames:
            if f.uid in picked:
                f.flag_note = f"{p['label']}: " + "; ".join(
                    core.cull_reasons(f, self.medians.get(f.group, {}), p))
            elif f.flag_note:
                f.flag_note = ""
        cleared = [f.uid for f in frames if f.flagged and f.uid not in picked]
        if cleared:
            self.set_flags(cleared, False)
        self.set_flags(sorted(picked), True)
        if picked:
            self.show_combo.setCurrentText("Flagged")
            self.statusBar().showMessage(
                f"{p['label']}: flagged {len(picked)} of {len(measured)} frames "
                f"({100 * len(picked) / len(measured):.0f}%) — review them, then Shift+X to reject them all "
                "(Ctrl+Z undoes this)", 12000)
        else:
            self.statusBar().showMessage(f"{p['label']}: no frames need culling", 8000)

    def kept_lights(self):
        return [f for f in self.frames.values() if not f.rejected and not f.error]

    def show_ready_bar(self):
        kept = self.kept_lights()
        if not kept:
            return
        hours = sum(float((f.info or {}).get("exptime") or 0) for f in kept) / 3600
        self.ready_label.setText(f"<b>{len(kept)} frames kept</b> ({hours:.1f} h) — ready to stack?")
        self.ready_bar.show()

    def open_stacking(self):
        kept = self.kept_lights()
        if not kept:
            QMessageBox.information(self, APP_NAME, "Load and cull some light frames first — the Stacking "
                                    "window opens with the frames you kept.")
            return
        if self.loaded_count < len(self.frames):
            if QMessageBox.question(self, APP_NAME, f"{len(self.frames) - self.loaded_count} frames are still "
                                    "being measured. Open the Stacking window with the frames measured so far?") \
                    != QMessageBox.StandardButton.Yes:
                return
            kept = [f for f in kept if f.loaded]
        import stack_window
        self.ready_bar.hide()
        self.stack_win = stack_window.StackWindow(self, kept, self.settings)
        self.stack_win.setAttribute(Qt.WA_DeleteOnClose, True)
        self.stack_win.show()
        self.stack_win.raise_()

    def open_prepare(self):
        import prepare_window
        self.prep_win = prepare_window.PrepareWindow(self, self.settings)
        self.prep_win.setAttribute(Qt.WA_DeleteOnClose, True)
        self.prep_win.show()
        self.prep_win.raise_()

    def open_finish(self):
        import finish_window
        self.finish_win = finish_window.FinishWindow(self, self.settings)
        self.finish_win.setAttribute(Qt.WA_DeleteOnClose, True)
        self.finish_win.show()
        self.finish_win.raise_()

    def start_over(self):
        """Clear the session and open a new folder."""
        if self.frames:
            flagged = sum(1 for f in self.frames.values() if f.flagged and not f.rejected)
            busy = self.loaded_count < len(self.frames)
            if flagged or busy:
                msg = "Clear this session and load new data?"
                if flagged:
                    msg += f"\n\n{flagged} frames are flagged but not rejected — the flags will be cleared."
                if busy:
                    msg += "\n\nMeasuring is still in progress — finished results are kept for next time."
                msg += "\n\n(Rejected frames stay in their REJECTED folders.)"
                if QMessageBox.question(self, APP_NAME, msg) != QMessageBox.StandardButton.Yes:
                    return
        self.reset_session()
        self.load_data()

    def reset_session(self):
        self.save_disk_cache()
        self.generation += 1
        self.analyze_pool.clear()
        self.display_pool.clear()
        self.frames = {}
        self.order = []
        self.root = None
        self.sources = []
        self.filter_overrides = {}
        self.current_uid = None
        self.loaded_count = 0
        self.to_load = 0
        self.undo_stack.clear()
        self.cache.clear()
        self.pending_display.clear()
        self.known_filters = []
        self.medians = {}
        self.disk_cache = {}
        self.cache_dirty = set()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All filters")
        self.filter_combo.blockSignals(False)
        self.show_combo.blockSignals(True)
        self.show_combo.setCurrentIndex(0)
        self.show_combo.blockSignals(False)
        self.progress.hide()
        self.list.blockSignals(True)
        self.list.clear()
        self.items.clear()
        self.list.blockSignals(False)
        self.view.clear("Click Load Data… (Ctrl+O) to choose folders\nor drag folders onto this window")
        self.fill_inspector(None)
        self.warning_label.setText("")
        self.ready_bar.hide()
        self.update_chart()
        self.update_preset_tooltips()
        self.setWindowTitle(f"{APP_NAME} — Astro Image Quality and Cull")
        self.update_status()

    def auto_flag(self):
        if not any(f.metrics for f in self.frames.values()):
            QMessageBox.information(self, APP_NAME, "No quality metrics yet — wait for frames to be measured "
                                    "(and check Settings → Star measurement is not Off).")
            return
        dlg = AutoFlagDialog(self, list(self.frames.values()), self.medians, self.settings)
        if dlg.exec():
            dlg.save()
            uids = dlg.matches()
            self.push_flag_undo("Auto-Flag")
            self.set_flags(uids, True)
            self.show_combo.setCurrentText("Flagged")
            self.statusBar().showMessage(f"Flagged {len(uids)} frames — step through them and press X "
                                         "to reject, or Shift+X to reject them all", 8000)

    def cleanup_checkpoints(self):
        try:
            freed = cp.cleanup(int(self.settings.value("checkpoints/days", 30)))
        except (OSError, ValueError):
            return
        if freed:
            self.statusBar().showMessage(f"Deleted old saved progress ({cp.human(freed)})", 5000)

    # ------------------------------------------------------------ header dump, filter, image
    def show_header_window(self):
        f = self.frames.get(self.current_uid) if self.current_uid is not None else None
        if f is None:
            self.inspect_file_header()
            return
        dlg = HeaderDialog(self, f.path, f.cards or None, f.name)
        dlg.show()
        self.header_dialog = dlg

    def inspect_file_header(self):
        start = str(self.root) if self.root else self.settings.value("last_folder", str(Path.home()))
        p, _ = QFileDialog.getOpenFileName(self, "Show the header of…", start,
                                           "FITS / XISF (*.fits *.fit *.fts *.xisf);;All files (*)")
        if p:
            dlg = HeaderDialog(self, p)
            dlg.show()
            self.header_dialog = dlg

    def set_filter_dialog(self):
        if not self.frames:
            QMessageBox.information(self, APP_NAME, "Load some frames first.")
            return
        sel = self.selected_uids()
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Filter Name")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Which filter were these frames taken with?\n(Used for grouping, quality comparison, "
                             "stacking and matching flats. Your files are not changed.)"))
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(COMMON_FILTERS)
        cur = self.frames[sel[0]].info.get("filter") if sel else None
        if cur and cur != "No filter":
            combo.setCurrentText(cur)
        lay.addWidget(combo)
        r_sel = QCheckBox(f"Only the {len(sel)} selected frame(s)") if len(sel) > 1 else None
        if r_sel:
            r_sel.setChecked(True)
            lay.addWidget(r_sel)
        b_reset = QCheckBox("Go back to the filter recorded in the files")
        lay.addWidget(b_reset)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if not dlg.exec():
            return
        uids = sel if (r_sel and r_sel.isChecked()) else list(self.frames)
        name = None if b_reset.isChecked() else core.normalize_filter(combo.currentText())
        self.set_filter(uids, name)

    def set_filter(self, uids, name):
        for u in uids:
            f = self.frames[u]
            f.filter_override = name
            if name:
                self.filter_overrides[str(f.path)] = name
                f.info["filter"] = name
            else:
                self.filter_overrides.pop(str(f.path), None)
                f.info["filter"] = core.frame_info(core.cards_to_dict(f.cards)).get("filter") if f.cards else None
        self.known_filters = sorted({f.info.get("filter") or "No filter" for f in self.frames.values() if f.loaded})
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All filters")
        self.filter_combo.addItems(self.known_filters)
        self.filter_combo.blockSignals(False)
        self.refresh_stats()
        self.rebuild_list()
        self.session_timer.start()
        self.statusBar().showMessage(f"Filter set to {name or 'as recorded'} for {len(uids)} frame(s)", 5000)

    def toggle_image(self, on):
        self.view.setVisible(on)
        if on:
            self.show_current()

    def update_kept_summary(self):
        if not hasattr(self, "btn_next") or not hasattr(self, "kept_label"):
            return
        kept = [f for f in self.frames.values() if not f.rejected and not f.flagged and not f.error]
        if not self.frames:
            self.kept_label.setText("–")
            self.btn_next.setEnabled(False)
            return
        meas = [f for f in kept if f.metrics]
        hours = sum(float((f.info or {}).get("exptime") or 0) for f in kept) / 3600

        def avg(k):
            v = [f.metrics.get(k) for f in meas if f.metrics.get(k) is not None]
            return (sum(v) / len(v)) if v else None
        txt = (f"<b>{len(kept)}</b> of {len(self.frames)} frames · <b>{hours:.1f} h</b>"
               f"{'  ·  ' + str(len(meas)) + ' measured' if len(meas) < len(kept) else ''}<br>")
        parts = []
        fw = avg("fwhm")
        if fw is not None:
            scales = [float(f.info.get("scale")) for f in meas if (f.info or {}).get("scale")]
            arc = f" ({fw * (sum(scales) / len(scales)):.1f}″)" if scales else ""
            parts.append(f"FWHM <b>{fw:.2f}</b> px{arc}")
        for k, lab, fmt in (("eccentricity", "Ecc", "{:.2f}"), ("snr", "SNR", "{:.0f}"), ("stars", "Stars", "{:.0f}")):
            v = avg(k)
            if v is not None:
                parts.append(f"{lab} <b>{fmt.format(v)}</b>")
        txt += " · ".join(parts) if parts else "<span style='color:#9a9aa6'>measuring…</span>"
        filters = sorted({f.info.get("filter") or "No filter" for f in kept if f.info})
        if len(filters) > 1:
            txt += "<br><span style='color:#9a9aa6'>" + ", ".join(
                f"{fl}: {sum(1 for f in kept if (f.info.get('filter') or 'No filter') == fl)}" for fl in filters) + "</span>"
        self.kept_label.setText(txt)
        self.btn_next.setEnabled(bool(kept))

    # ------------------------------------------------------------ session (0.6)
    @staticmethod
    def session_path() -> Path:
        return cp.root() / "culling_session.json"

    def save_session(self, clean=False):
        """Remember what is loaded and flagged, so it can be reopened after a crash."""
        p = self.session_path()
        if not self.sources or not self.frames:
            if clean and p.exists():          # nothing loaded now: just record a clean exit
                d = cp.read_json(p, {}) or {}
                d["clean"] = True
                try:
                    cp.write_json(p, d)
                except OSError:
                    pass
            return
        srcs = []
        for src in self.sources:
            if src[0] == "folder":
                srcs.append(["folder", str(src[1]), bool(src[2])])
            else:
                srcs.append(["files", [str(x) for x in src[1]]])
        cur = self.frames.get(self.current_uid)
        d = {"sources": srcs,
             "flags": {str(f.path): f.flag_note for f in self.frames.values() if f.flagged and not f.rejected},
             "filters": dict(self.filter_overrides),
             "frames": len(self.frames), "rejected": sum(1 for f in self.frames.values() if f.rejected),
             "show": self.show_combo.currentText(), "current": str(cur.path) if cur else None,
             "time": time.strftime("%Y-%m-%d %H:%M"), "clean": bool(clean)}
        try:
            cp.write_json(p, d)
        except OSError:
            pass

    def last_session(self):
        d = cp.read_json(self.session_path(), None)
        return d if d and d.get("sources") else None

    def offer_resume(self):
        """At start-up after a crash: offer to reopen what was loaded."""
        d = self.last_session()
        if not d or d.get("clean"):
            return
        nfl = len(d.get("flags", {}))
        if QMessageBox.question(
                self, APP_NAME, f"{APP_NAME} didn't close normally last time.\n\nReopen your last session "
                f"({d.get('frames', '?')} frames, {nfl} flagged, {d.get('time', '')})?\n\n"
                "Star measurements are reused from the cache, so this is quick.") == QMessageBox.StandardButton.Yes:
            self.reopen_last_session()

    def reopen_last_session(self):
        d = self.last_session()
        if not d:
            QMessageBox.information(self, APP_NAME, "There is no saved session yet.")
            return
        sources = []
        for src in d["sources"]:
            if src[0] == "folder" and Path(src[1]).is_dir():
                sources.append(("folder", Path(src[1]), bool(src[2])))
            elif src[0] == "files":
                files = [Path(x) for x in src[1] if Path(x).exists()]
                if files:
                    sources.append(("files", files))
        if not sources:
            QMessageBox.information(self, APP_NAME, "The folders of the last session can't be found any more "
                                    "(is the NAS / drive connected?).")
            return
        flags = d.get("flags", {})
        self.filter_overrides.update(d.get("filters", {}) or {})
        self.open_sources(sources)
        if not self.frames:
            return
        by_path = {os.path.normcase(str(f.path)): u for u, f in self.frames.items()}
        uids = []
        for path, note in flags.items():
            u = by_path.get(os.path.normcase(path))
            if u is not None and not self.frames[u].rejected:
                self.frames[u].flag_note = note
                uids.append(u)
        if uids:
            self.set_flags(uids, True)
        if d.get("show"):
            self.show_combo.setCurrentText(d["show"])      # ignored if that view no longer exists
        cur = by_path.get(os.path.normcase(d.get("current") or ""))
        if cur is not None and cur in self.items:
            self.list.setCurrentItem(self.items[cur])
        self.statusBar().showMessage(f"Reopened the last session — {len(uids)} flagged frame(s) restored", 6000)

    # ------------------------------------------------------------ export
    def export_keepers(self):
        kept = sorted((f for f in self.frames.values() if not f.rejected and not f.error),
                      key=lambda f: core.natural_key(str(f.path)))
        if not kept:
            QMessageBox.information(self, APP_NAME, "There are no kept frames to export.")
            return
        start = str((self.root or Path.home()) / "keepers.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save list of kept frames", start, "Text files (*.txt)")
        if not path:
            return
        Path(path).write_text("\n".join(str(f.path) for f in kept) + "\n", encoding="utf-8")
        self.statusBar().showMessage(f"Saved {len(kept)} file paths to {path}", 4000)

    # ------------------------------------------------------------ status
    def update_status(self):
        self.update_kept_summary()
        self._update_status()

    def _update_status(self):
        n = len(self.frames)
        if n == 0:
            self.status_label.setText("No folder open")
            return
        fr = list(self.frames.values())
        rej = sum(1 for f in fr if f.rejected)
        flg = sum(1 for f in fr if f.flagged and not f.rejected)
        err = sum(1 for f in fr if f.error)
        badges = {b: sum(1 for f in fr if f.badge == b and not f.rejected) for b in ("green", "amber", "red")}
        parts = [f"{n} frames", f"{n - rej} kept", f"{rej} rejected"]
        if flg:
            parts.append(f"{flg} flagged")
        if any(badges.values()):
            parts.append(f"<span style='color:#46c85f'>●</span> {badges['green']}  "
                         f"<span style='color:#eba932'>●</span> {badges['amber']}  "
                         f"<span style='color:#e64646'>●</span> {badges['red']}")
        if err:
            parts.append(f"{err} unreadable")
        if len(self.order) != n:
            parts.append(f"{len(self.order)} shown")
        if self.loaded_count < n:
            parts.append(f"measuring {self.loaded_count}/{n}")
        self.status_label.setText("   ·   ".join(parts))

    # -------------------------------------------------------------- help
    def show_keys(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "↑ ↓ ← →\tPrevious / next frame\n"
            "Shift + arrows\tSelect a range\n"
            "Shift / Ctrl + click\tSelect frames\n"
            "X or Delete\tReject / restore frame(s)\n"
            "F\tFlag / unflag frame(s)\n"
            "Shift+X\tReject all flagged frames\n"
            "D\tClear all flags\n"
            "Ctrl+N\tStart over – load new data\n"
            "Ctrl+K\tStack kept frames…\n"
            "Ctrl+F\tAuto-Flag…\n"
            "Ctrl+Z\tUndo reject / restore\n"
            "Ctrl+0\tFit to window\n"
            "1\t100% (actual pixels)\n"
            "Mouse wheel\tZoom   ·   drag to pan\n"
            "Double-click\tToggle fit / 100%\n"
            "Chart: click\tJump to that frame\n"
            "Chart: drag\tSelect a range of frames\n"
            "Ctrl+O\tLoad Data (one or more folders)\n"
            "Ctrl+Shift+O\tOpen one folder\n"
            "Ctrl+E\tExport list of kept frames\n"
            "F5\tReload folder"))

    def show_metrics_help(self):
        QMessageBox.information(self, "About the quality metrics", (
            "FWHM — star size (sensor pixels; arcseconds too when the header has focal length "
            "and pixel size). Lower is sharper. Fitted with an elliptical Moffat profile (β = 4), "
            "the same default as PixInsight's SubframeSelector.\n\n"
            "Eccentricity — 0 = round, 1 = a line. Above ~0.5 is visible elongation (tracking, "
            "guiding, wind, tilt). A whole session that is high points to the rig, not the night.\n\n"
            "SNR — how far the brighter stars rise above the sky noise. Drops with clouds, haze, "
            "moonlight, light pollution or soft focus.\n\n"
            "Stars — number of stars detected. A sudden drop usually means cloud.\n\n"
            "Background — sky brightness. Rises with moonlight, twilight or haze.\n\n"
            "Badges and score compare each frame with the median of its own camera + filter, "
            "so Ha frames are never judged against luminance frames. Score 100 = a typical frame.\n\n"
            "Star measurement 'Fast' measures three bands across each frame (about a quarter "
            "of the file), which gives nearly the same numbers as 'Full frame' and is 3–4× "
            "faster on network drives."))

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}", (
            f"<h2 style='margin-bottom:0'>{APP_NAME}</h2>"
            f"<i>{APP_TAGLINE}</i><br><br>Version {APP_VERSION}<br><br>"
            "Cull your subframes with star-quality metrics, stack them with calibration, and prepare the "
            "masters for PixInsight — plate solving, gradient removal and the RC-Astro XTerminators included.<br>"
            "Rejected frames are moved to a <i>REJECTED</i> subfolder — nothing is ever deleted "
            "and your images are never modified.<br><br>"
            f"Created by {APP_AUTHOR}, built with Claude (Anthropic).<br>"
            f"<a href='{APP_URL}'>{APP_URL}</a><br>"
            "Free software under the MIT License. Provided as is, with no warranty.<br><br>"
            "The culling view was inspired by FITS Blaster for macOS by Tom van Peer "
            "(<a href='https://astrophoto-app.com/'>astrophoto-app.com</a>)."))


# --------------------------------------------------------------------------

def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    base = QColor(30, 30, 34)
    window = QColor(40, 40, 46)
    text = QColor(225, 225, 230)
    p.setColor(QPalette.Window, window)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, window)
    p.setColor(QPalette.ToolTipBase, QColor(50, 50, 56))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, window)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Highlight, QColor(196, 120, 40))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Link, QColor(110, 170, 255))
    p.setColor(QPalette.PlaceholderText, QColor(130, 130, 140))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, QColor(110, 110, 118))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_dark_theme(app)

    if "--selftest" in sys.argv:
        import selftest
        i = sys.argv.index("--selftest")
        out = Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 else Path("selftest_output")
        sys.exit(selftest.run(app, out))

    def excepthook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        try:
            sys.__stderr__.write(msg)
        except Exception:  # noqa: BLE001 - pythonw has no console
            pass
        QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{value}\n\n{msg[-1500:]}")
    sys.excepthook = excepthook

    w = MainWindow()
    w.show()
    if len(sys.argv) <= 1:
        QTimer.singleShot(400, w.offer_resume)
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        QTimer.singleShot(0, lambda: w.open_folder(target) if target.is_dir()
                          else w.open_sources([("files", [Path(a) for a in sys.argv[1:]])]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
