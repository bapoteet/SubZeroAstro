"""
FITS Blaster for Windows - Stage 1 (culling)

Open a folder of FITS frames, flip through them with the keyboard, and move
bad frames into a REJECTED subfolder. Inspired by Tom van Peer's FITS Blaster
for macOS (https://astrophoto-app.com/); written independently for Windows.

Keys
  Up / Down / Left / Right   previous / next frame
  Shift + arrows / click     select a range
  X or Delete                reject (or restore) the frame or selected range
  Ctrl+Z                     undo the last reject / restore
  Ctrl+O                     open folder
  F                          fit image to window      1  100% zoom
  Mouse wheel                zoom        drag         pan
"""

from __future__ import annotations

import sys
import traceback
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (QObject, QRectF, QRunnable, QSettings, QSize, Qt,
                            QThreadPool, QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage,
                           QKeySequence, QPainter, QPalette, QPen, QPixmap,
                           QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QFileDialog, QFormLayout, QFrame,
                               QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QSplitter, QStyle,
                               QTableWidget, QTableWidgetItem, QToolBar,
                               QVBoxLayout, QWidget)

import fits_core as core

APP_NAME = "FITS Blaster for Windows"
APP_VERSION = "0.1 (stage 1)"
ICON_W, ICON_H = 120, 84
CACHE_SIZE = 10

ROLE_UID = Qt.UserRole + 1


# --------------------------------------------------------------------------
# Background loading
# --------------------------------------------------------------------------

def to_qimage(arr) -> QImage:
    h, w = arr.shape
    buf = arr.tobytes()
    return QImage(buf, w, h, w, QImage.Format_Grayscale8).copy()


class LoaderSignals(QObject):
    preview_ready = Signal(int, int, object)   # generation, uid, result dict
    preview_failed = Signal(int, int, str)
    display_ready = Signal(int, int, object, str)   # generation, uid, QImage, path
    display_failed = Signal(int, int, str, str)


class PreviewTask(QRunnable):
    def __init__(self, gen, uid, path, signals):
        super().__init__()
        self.gen, self.uid, self.path, self.signals = gen, uid, str(path), signals

    def run(self):
        try:
            res = core.load_frame_preview(self.path)
            res["qthumb"] = to_qimage(res.pop("thumb"))
            self.signals.preview_ready.emit(self.gen, self.uid, res)
        except Exception as e:  # noqa: BLE001
            self.signals.preview_failed.emit(self.gen, self.uid, str(e) or type(e).__name__)


class DisplayTask(QRunnable):
    def __init__(self, gen, uid, path, params, signals):
        super().__init__()
        self.gen, self.uid, self.path = gen, uid, str(path)
        self.params, self.signals = params, signals

    def run(self):
        try:
            img = to_qimage(core.load_frame_display(self.path, self.params))
            self.signals.display_ready.emit(self.gen, self.uid, img, self.path)
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
        self.message = "Open a folder of FITS files  (Ctrl+O)\nor drag a folder onto this window"
        self.banner = ""
        self.subtitle = ""

    def clear(self, message=None):
        self.item.setPixmap(QPixmap())
        self.has_image = False
        if message is not None:
            self.message = message
        self.banner = ""
        self.subtitle = ""
        self.viewport().update()

    def show_image(self, qimg: QImage, full_size=None):
        """Show an image. If full_size (w, h) is given the image is a placeholder
        (thumbnail) scaled up to that size so zoom stays consistent."""
        self.item.setPixmap(QPixmap.fromImage(qimg))
        if full_size and qimg.width() > 0:
            scale = full_size[0] / qimg.width()
            self.item.setScale(scale)
            rect = QRectF(0, 0, full_size[0], full_size[1])
        else:
            self.item.setScale(1.0)
            rect = QRectF(0, 0, qimg.width(), qimg.height())
        self._scene.setSceneRect(rect)
        self.has_image = True
        if self.fit_mode:
            self.fit()
        self.viewport().update()

    def fit(self):
        self.fit_mode = True
        if self.has_image:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def actual_size(self):
        self.fit_mode = False
        self.resetTransform()

    def zoom_percent(self) -> int:
        return int(round(self.transform().m11() * 100))

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
        self.viewport().update()

    def mouseDoubleClickEvent(self, event):
        if self.fit_mode:
            self.actual_size()
        else:
            self.fit()
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()

    def drawForeground(self, painter, rect):
        painter.save()
        painter.resetTransform()
        vp = self.viewport().rect()
        if not self.has_image:
            painter.setPen(QColor(150, 150, 160))
            f = painter.font()
            f.setPointSize(13)
            painter.setFont(f)
            painter.drawText(vp, Qt.AlignCenter, self.message)
        else:
            f = painter.font()
            if self.banner:
                f.setPointSize(16)
                f.setBold(True)
                painter.setFont(f)
                painter.setPen(QColor(255, 80, 80))
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
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.settings = QSettings("FITSBlasterWin", "FITSBlasterWin")

        self.frames: dict[int, core.Frame] = {}
        self.order: list[int] = []          # uids in current list order
        self.items: dict[int, QListWidgetItem] = {}
        self.current_uid: int | None = None
        self.generation = 0
        self.loaded_count = 0
        self.undo_stack: list[list[tuple[int, bool]]] = []
        self.cache: OrderedDict[int, QImage] = OrderedDict()
        self.pending_display: set[int] = set()
        self.known_filters: list[str] = []
        self.root: Path | None = None

        self.signals = LoaderSignals()
        self.signals.preview_ready.connect(self.on_preview_ready)
        self.signals.preview_failed.connect(self.on_preview_failed)
        self.signals.display_ready.connect(self.on_display_ready)
        self.signals.display_failed.connect(self.on_display_failed)

        self.preview_pool = QThreadPool(self)
        self.preview_pool.setMaxThreadCount(max(2, QThreadPool.globalInstance().maxThreadCount() - 1))
        self.display_pool = QThreadPool(self)
        self.display_pool.setMaxThreadCount(2)

        self._build_ui()
        self._build_actions()
        self._restore_settings()
        self.update_status()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        # Left: sidebar
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(6)

        row1 = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All filters")
        self.filter_combo.setToolTip("Show only frames taken with this filter")
        self.show_combo = QComboBox()
        self.show_combo.addItems(["All frames", "Kept", "Rejected"])
        self.show_combo.setToolTip("Show all frames, only kept ones, or only rejected ones")
        row1.addWidget(self.filter_combo, 1)
        row1.addWidget(self.show_combo, 1)
        sl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["File name", "Capture time", "Filter"])
        row2.addWidget(self.sort_combo, 1)
        sl.addLayout(row2)

        self.list = QListWidget()
        self.list.setViewMode(QListView.ListMode)
        self.list.setIconSize(QSize(ICON_W, ICON_H))
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setSpacing(2)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        sl.addWidget(self.list, 1)

        self.filter_combo.currentIndexChanged.connect(self.rebuild_list)
        self.show_combo.currentIndexChanged.connect(self.rebuild_list)
        self.sort_combo.currentIndexChanged.connect(self.rebuild_list)
        self.list.currentItemChanged.connect(self.on_current_changed)

        # Centre: viewer
        self.view = ImageView()

        # Right: inspector
        insp = QWidget()
        il = QVBoxLayout(insp)
        il.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Frame")
        tf = title.font()
        tf.setPointSize(tf.pointSize() + 2)
        tf.setBold(True)
        title.setFont(tf)
        il.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.info_labels = {}
        for key, label in [("file", "File"), ("status", "Status"), ("folder", "Folder"),
                           ("object", "Target"), ("filter", "Filter"), ("exptime", "Exposure"),
                           ("date_obs", "Captured"), ("camera", "Camera"), ("gain", "Gain"),
                           ("temp", "Sensor temp"), ("size", "Size"), ("bayer", "Bayer")]:
            lab = QLabel("–")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setWordWrap(True)
            form.addRow(label + ":", lab)
            self.info_labels[key] = lab
        il.addLayout(form)

        note = QLabel("Quality metrics (FWHM, eccentricity, SNR, star count) arrive in stage 2.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a96; font-style: italic; padding: 6px 0;")
        il.addWidget(note)

        il.addWidget(QLabel("FITS header"))
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

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(side)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(insp)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([330, 900, 300])
        self.setCentralWidget(self.splitter)

        # Status bar
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
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

        self.act_open = act("Open Folder…", self.choose_folder, "Ctrl+O", QStyle.SP_DirOpenIcon,
                            "Open a folder of FITS files (Ctrl+O)")
        self.act_open_files = act("Open Files…", self.choose_files, "Ctrl+Shift+O")
        self.act_reject = act("Reject / Restore", self.toggle_reject, ["X", "Delete"],
                              QStyle.SP_DialogDiscardButton,
                              "Move the frame (or selected range) to REJECTED, or bring it back (X)")
        self.act_undo = act("Undo", self.undo, "Ctrl+Z", QStyle.SP_ArrowBack, "Undo last reject/restore (Ctrl+Z)")
        self.act_fit = act("Fit", self.view.fit, "F", None, "Fit image to window (F)")
        self.act_100 = act("100%", self.view.actual_size, "1", None, "Actual pixels (1)")
        self.act_export = act("Export Keeper List…", self.export_keepers, "Ctrl+E")
        self.act_next = act("Next frame", lambda: self.step(1), ["Right", "Down"])
        self.act_prev = act("Previous frame", lambda: self.step(-1), ["Left", "Up"])
        self.act_reload = act("Reload Folder", self.reload, "F5")
        self.act_quit = act("Exit", self.close, "Ctrl+Q")

        self.chk_sub = QCheckBox("Include subfolders")
        self.chk_sub.setToolTip("Also load FITS files in subfolders (FLAT, DARK, BIAS, CALIB folders are skipped)")
        self.act_advance = QAction("Jump to next frame after rejecting", self, checkable=True)
        self.act_advance.setChecked(True)

        tb.addAction(self.act_open)
        tb.addWidget(self.chk_sub)
        tb.addSeparator()
        tb.addAction(self.act_reject)
        tb.addAction(self.act_undo)
        tb.addSeparator()
        tb.addAction(self.act_fit)
        tb.addAction(self.act_100)

        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self.act_open)
        m.addAction(self.act_open_files)
        m.addAction(self.act_reload)
        m.addSeparator()
        m.addAction(self.act_export)
        m.addSeparator()
        m.addAction(self.act_quit)
        m = mb.addMenu("&Frame")
        m.addAction(self.act_reject)
        m.addAction(self.act_undo)
        m.addSeparator()
        m.addAction(self.act_prev)
        m.addAction(self.act_next)
        m.addSeparator()
        m.addAction(self.act_advance)
        m = mb.addMenu("&View")
        m.addAction(self.act_fit)
        m.addAction(self.act_100)
        m = mb.addMenu("&Help")
        a = QAction("Keyboard Shortcuts", self)
        a.triggered.connect(self.show_keys)
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
            self.resize(1500, 900)
        s = self.settings.value("splitter")
        if s is not None:
            self.splitter.restoreState(s)
        self.chk_sub.setChecked(self.settings.value("subfolders", False, type=bool))
        self.act_advance.setChecked(self.settings.value("advance", True, type=bool))

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
        self.settings.setValue("subfolders", self.chk_sub.isChecked())
        self.settings.setValue("advance", self.act_advance.isChecked())
        self.generation += 1
        self.preview_pool.clear()
        self.display_pool.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------ opening
    def choose_folder(self):
        start = self.settings.value("last_folder", str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, "Open folder of FITS files", start)
        if d:
            self.open_folder(d)

    def choose_files(self):
        start = self.settings.value("last_folder", str(Path.home()))
        files, _ = QFileDialog.getOpenFileNames(self, "Open FITS files", start,
                                                "FITS files (*.fits *.fit *.fts);;All files (*)")
        if files:
            self.open_frames(core.frames_from_files(files), Path(files[0]).parent)

    def open_folder(self, folder):
        folder = Path(folder)
        self.settings.setValue("last_folder", str(folder))
        frames = core.scan_folder(folder, recursive=self.chk_sub.isChecked())
        if not frames:
            QMessageBox.information(self, APP_NAME,
                                    f"No FITS files (.fits, .fit, .fts) were found in:\n{folder}"
                                    + ("" if self.chk_sub.isChecked() else
                                       "\n\nTip: tick “Include subfolders” if your frames are in subfolders."))
            return
        self.open_frames(frames, folder)

    def reload(self):
        if self.root is not None:
            self.open_folder(self.root)

    def open_frames(self, frames, root):
        self.generation += 1
        self.preview_pool.clear()
        self.display_pool.clear()
        self.root = Path(root)
        self.frames = {f.uid: f for f in frames}
        self.loaded_count = 0
        self.undo_stack.clear()
        self.cache.clear()
        self.pending_display.clear()
        self.known_filters = []
        self.current_uid = None
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All filters")
        self.filter_combo.blockSignals(False)
        self.view.clear("Loading…")
        self.setWindowTitle(f"{APP_NAME} — {self.root}")
        self.progress.setRange(0, len(frames))
        self.progress.setValue(0)
        self.progress.show()
        self.rebuild_list()
        # Queue previews in list order so the top of the list fills in first
        for uid in self.order + [u for u in self.frames if u not in self.order]:
            f = self.frames[uid]
            self.preview_pool.start(PreviewTask(self.generation, uid, f.path, self.signals))
        if self.order:
            self.list.setCurrentRow(0)
        self.update_status()

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
        files = []
        for p in paths:
            if p.is_dir():
                files += [f.path for f in core.scan_folder(p, recursive=self.chk_sub.isChecked())]
            else:
                files.append(p)
        frames = core.frames_from_files(files)
        if frames:
            self.open_frames(frames, frames[0].folder)

    # ----------------------------------------------------------- loading
    def on_preview_ready(self, gen, uid, res):
        if gen != self.generation or uid not in self.frames:
            return
        f = self.frames[uid]
        f.loaded = True
        f.error = None
        f.thumb = res["qthumb"]
        f.stretch = res["stretch"]
        f.display_size = res["display_size"]
        f.cards = res["cards"]
        f.info = res["info"]
        self.loaded_count += 1
        self.progress.setValue(self.loaded_count)

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

        self.refresh_item(uid)
        if uid == self.current_uid:
            self.show_current()
        self.check_loading_done()

    def on_preview_failed(self, gen, uid, msg):
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
        if self.loaded_count >= len(self.frames):
            self.progress.hide()
            if self.sort_combo.currentText() != "File name" or self.filter_combo.currentIndex() > 0:
                self.rebuild_list()
        self.update_status()

    def request_display(self, uid):
        if uid in self.cache or uid in self.pending_display or uid not in self.frames:
            return
        f = self.frames[uid]
        if f.error:
            return
        self.pending_display.add(uid)
        self.display_pool.start(DisplayTask(self.generation, uid, f.path, f.stretch, self.signals))

    def on_display_ready(self, gen, uid, img, path):
        if gen != self.generation:
            return
        self.pending_display.discard(uid)
        self.cache[uid] = img
        self.cache.move_to_end(uid)
        while len(self.cache) > CACHE_SIZE:
            old = next(iter(self.cache))
            if old == self.current_uid:
                self.cache.move_to_end(old)
                if len(self.cache) <= 1:
                    break
                continue
            del self.cache[old]
        if uid == self.current_uid:
            self.view.show_image(img)

    def on_display_failed(self, gen, uid, msg, path):
        if gen != self.generation:
            return
        self.pending_display.discard(uid)
        f = self.frames.get(uid)
        if f is None:
            return
        if str(f.path) != path:
            # File was moved (rejected/restored) while loading - try again.
            self.request_display(uid)
            return
        if uid == self.current_uid:
            self.view.clear(f"Could not display {f.name}:\n{msg}")

    # -------------------------------------------------------------- list
    def frame_visible(self, f: core.Frame) -> bool:
        show = self.show_combo.currentText()
        if show == "Kept" and f.rejected:
            return False
        if show == "Rejected" and not f.rejected:
            return False
        filt = self.filter_combo.currentText()
        if self.filter_combo.currentIndex() > 0 and (f.info.get("filter") or "No filter") != filt:
            return False
        return True

    def sort_key(self):
        mode = self.sort_combo.currentText()
        if mode == "Capture time":
            return lambda f: (str(f.info.get("date_obs") or "~"), core.natural_key(str(f.path)))
        if mode == "Filter":
            return lambda f: (f.info.get("filter") or "~", core.natural_key(f.rel_folder),
                              core.natural_key(f.name))
        return lambda f: (core.natural_key(f.rel_folder), core.natural_key(f.name))

    def rebuild_list(self, *_):
        keep = self.current_uid
        self.list.blockSignals(True)
        self.list.clear()
        self.items.clear()
        frames = sorted((f for f in self.frames.values() if self.frame_visible(f)), key=self.sort_key())
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
        elif self.order:
            self.list.setCurrentRow(0)
        else:
            self.current_uid = None
            self.show_current()
        self.update_status()

    def make_icon(self, f: core.Frame) -> QIcon:
        pm = QPixmap(ICON_W, ICON_H)
        pm.fill(QColor(20, 20, 24))
        p = QPainter(pm)
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
        if f.error:
            details.append("⚠ unreadable")
        if details:
            lines.append(" · ".join(details))
        if f.rel_folder:
            lines.append(f"📁 {f.rel_folder}")
        it.setText("\n".join(lines))
        it.setIcon(self.make_icon(f))
        it.setToolTip(str(f.path) + (f"\n\n{f.error}" if f.error else ""))
        it.setForeground(QBrush(QColor(230, 110, 110) if f.rejected else QColor(225, 225, 230)))

    def on_current_changed(self, cur, _prev):
        self.current_uid = cur.data(ROLE_UID) if cur is not None else None
        self.show_current()

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
        if f.error:
            self.view.clear(f"Could not read {f.name}:\n{f.error}")
        elif uid in self.cache:
            self.cache.move_to_end(uid)
            self.view.show_image(self.cache[uid])
        elif f.thumb is not None:
            self.view.show_image(f.thumb, f.display_size)
            self.request_display(uid)
        else:
            self.view.clear("Loading…")
            self.request_display(uid)
        self.view.banner = "REJECTED" if f.rejected else ""
        self.view.subtitle = f.name
        self.view.viewport().update()
        # Prefetch neighbours for instant stepping
        if uid in self.order:
            i = self.order.index(uid)
            for j in (i + 1, i - 1, i + 2):
                if 0 <= j < len(self.order) and self.frames[self.order[j]].loaded:
                    self.request_display(self.order[j])

    def fill_inspector(self, f):
        L = self.info_labels
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
        L["status"].setText("<span style='color:#ff6060'><b>Rejected</b></span>" if f.rejected
                            else "<span style='color:#60d060'>Kept</span>")
        L["folder"].setText(str(f.path.parent))
        L["object"].setText(fmt(i.get("object")))
        L["filter"].setText(fmt(i.get("filter")))
        L["exptime"].setText(fmt(i.get("exptime"), " s"))
        L["date_obs"].setText(fmt(i.get("date_obs")))
        L["camera"].setText(fmt(i.get("camera")))
        L["gain"].setText(fmt(i.get("gain")))
        L["temp"].setText(fmt(i.get("temp"), " °C"))
        if f.display_size:
            w, h = f.display_size
            if i.get("bayer"):
                L["size"].setText(f"{w * 2} × {h * 2}")
            else:
                L["size"].setText(f"{w} × {h}")
        else:
            L["size"].setText("–")
        L["bayer"].setText(f"{i['bayer']} (shown as greyscale)" if i.get("bayer") else "No (mono)"
                           if f.info else "–")
        self.fill_header_table()

    def fill_header_table(self, *_):
        f = self.frames.get(self.current_uid) if self.current_uid is not None else None
        rows = []
        if f is not None:
            q = self.header_search.text().strip().lower()
            for k, v, c in f.cards:
                val = "" if v is None else (("T" if v else "F") if isinstance(v, bool) else str(v))
                if v is None:
                    val = c
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
    def selected_uids(self):
        uids = [it.data(ROLE_UID) for it in self.list.selectedItems()]
        if len(uids) <= 1 and self.current_uid is not None:
            return [self.current_uid]
        return sorted(uids, key=lambda u: self.order.index(u) if u in self.order else 0)

    def toggle_reject(self):
        uids = self.selected_uids()
        if not uids:
            return
        reject = not all(self.frames[u].rejected for u in uids)
        done = self.apply_reject(uids, reject)
        if done:
            self.undo_stack.append(done)
        if reject and len(uids) == 1 and self.act_advance.isChecked() and self.show_combo.currentText() == "All frames":
            self.step(1)

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
            if self.show_combo.currentText() != "All frames":
                # Frames may leave the current view; keep position sensible
                row = self.list.currentRow()
                self.rebuild_list()
                if self.list.count():
                    self.list.setCurrentRow(min(row, self.list.count() - 1))
            else:
                for u, _r in done:
                    self.refresh_item(u)
                self.show_current()
            self.update_status()
        return done

    def undo(self):
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo", 2000)
            return
        last = self.undo_stack.pop()
        uids = [u for u, _r in last]
        by_state = {}
        for u, r in last:
            by_state.setdefault(not r, []).append(u)
        for state, us in by_state.items():
            self.apply_reject(us, state)
        if uids and uids[0] in self.items:
            self.list.setCurrentItem(self.items[uids[0]])
        self.statusBar().showMessage(f"Undid {len(uids)} change(s)", 2000)

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
        n = len(self.frames)
        if n == 0:
            self.status_label.setText("No folder open")
            return
        rej = sum(1 for f in self.frames.values() if f.rejected)
        err = sum(1 for f in self.frames.values() if f.error)
        parts = [f"{n} frames", f"{n - rej} kept", f"{rej} rejected"]
        if err:
            parts.append(f"{err} unreadable")
        if len(self.order) != n:
            parts.append(f"{len(self.order)} shown")
        if self.loaded_count < n:
            parts.append(f"loading {self.loaded_count}/{n}")
        self.status_label.setText("   ·   ".join(parts))

    # -------------------------------------------------------------- help
    def show_keys(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "↑ ↓ ← →\tPrevious / next frame\n"
            "Shift + arrows\tSelect a range\n"
            "Shift / Ctrl + click\tSelect frames\n"
            "X or Delete\tReject / restore frame(s)\n"
            "Ctrl+Z\tUndo\n"
            "F\tFit to window\n"
            "1\t100% (actual pixels)\n"
            "Mouse wheel\tZoom   ·   drag to pan\n"
            "Double-click\tToggle fit / 100%\n"
            "Ctrl+O\tOpen folder\n"
            "Ctrl+E\tExport list of kept frames\n"
            "F5\tReload folder"))

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}", (
            f"<b>{APP_NAME}</b> {APP_VERSION}<br><br>"
            "Fast culling of astrophotography FITS frames on Windows.<br>"
            "Rejected frames are moved to a <i>REJECTED</i> subfolder — nothing is ever deleted "
            "and your images are never modified.<br><br>"
            "Inspired by FITS Blaster for macOS by Tom van Peer "
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

    def excepthook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        sys.__stderr__.write(msg)
        QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{value}\n\n{msg[-1500:]}")
    sys.excepthook = excepthook

    w = MainWindow()
    w.show()
    # Allow "FITS Blaster.exe <folder>" or dropping a folder onto the exe
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        QTimer.singleShot(0, lambda: w.open_folder(target) if target.is_dir()
                          else w.open_frames(core.frames_from_files(sys.argv[1:]), target.parent))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
