"""
prepare_window.py - the "Prepare for PixInsight" window (Stage 4).

Pick masters (preloaded after stacking), set a crop on the preview, choose the
steps (plate solve with ASTAP, GraXpert, BlurXTerminator / NoiseXTerminator /
StarXTerminator with controls generated from your installed RC-Astro CLI),
then run. Ends with a "ready for final processing" summary and a tailored
PixInsight plan.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

import checkpoints as cp
import prepare as P
import tools

XT_KEYS = ("bxt", "nxt", "sxt")
# Starting values the first time (users change them; the choice is remembered)
XT_START = {"bxt": {"enabled": True, "values": {"ss": 0.25, "sn": 0.50}},
            "nxt": {"enabled": False, "values": {"dn": 0.60}},
            "sxt": {"enabled": False, "values": {"stars": True}}}


def save_rgb8(path, rgb8: np.ndarray) -> bool:
    h, w, _ = rgb8.shape
    arr = np.ascontiguousarray(rgb8)
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return img.save(str(path), quality=92) if str(path).lower().endswith((".jpg", ".jpeg")) \
        else img.save(str(path))


# --------------------------------------------------------------------------
# Controls built from an RC-Astro parameter document
# --------------------------------------------------------------------------

class ParamPanel(QWidget):
    """Controls for one RC-Astro product, generated from `rc-astro <key> --json`
    (README-DEVS sections 3-8): groups, modes, visibleIf / disabledIf."""

    def __init__(self, spec: tools.ProductSpec, values=None, mode=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.widgets = {}
        self.rows = {}
        self.mode_boxes = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        start = spec.defaults()
        start.update({k: v for k, v in (values or {}).items() if spec.param(k)})
        for g in spec.groups:
            box = QGroupBox(g.get("label", ""))
            form = QFormLayout(box)
            form.setContentsMargins(6, 4, 6, 4)
            for name in g.get("params", []):
                p = spec.param(name)
                if p is None:
                    continue
                w = self._make(p, start.get(name, p.get("default")))
                tip = p.get("detailedDescriptionHtml") or p.get("description") or p.get("label")
                w.setToolTip(tip)
                lab = QLabel(p.get("label", name))
                lab.setToolTip(tip)
                form.addRow(lab, w)
                self.widgets[name] = w
                self.rows[name] = (lab, w)
            for mname in g.get("modes", []):
                m = next((x for x in spec.modes if x.get("name") == mname), None)
                if m is None:
                    continue
                cb = QCheckBox(m.get("label", mname))
                cb.setToolTip(m.get("description", ""))
                cb.setChecked(mode == mname)
                cb.toggled.connect(lambda on, n=mname: self._mode_toggled(n, on))
                form.addRow(cb)
                self.mode_boxes[mname] = cb
            if form.rowCount():
                lay.addWidget(box)
        self.refresh()

    def _make(self, p, value):
        t = p.get("type")
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(self.refresh)
        elif t == "int":
            w = QSpinBox()
            w.setRange(int(p.get("min", 0)), int(p.get("max", 100)))
            w.setValue(int(value))
            w.valueChanged.connect(self.refresh)
        elif t == "enum":
            w = QComboBox()
            for o in p.get("options", []):
                w.addItem(str(o.get("label", o.get("value"))) if isinstance(o, dict) else str(o),
                          o.get("value") if isinstance(o, dict) else o)
            i = w.findData(value)
            w.setCurrentIndex(max(i, 0))
            w.currentIndexChanged.connect(self.refresh)
        else:
            w = QDoubleSpinBox()
            w.setDecimals(int(p.get("precision", 2)))
            w.setRange(float(p.get("min", 0.0)), float(p.get("max", 1.0)))
            w.setSingleStep(10 ** -min(int(p.get("precision", 2)), 2))
            if p.get("unit"):
                w.setSuffix(" " + p["unit"])
            w.setValue(float(value))
            w.valueChanged.connect(self.refresh)
        return w

    def _value(self, name):
        w = self.widgets[name]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentData()
        return w.value()

    def _set(self, name, value):
        w = self.widgets[name]
        w.blockSignals(True)
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            i = w.findData(value)
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            w.setValue(value)
        w.blockSignals(False)

    def _mode_toggled(self, name, on):
        if on:                                       # modes are mutually exclusive
            for n, cb in self.mode_boxes.items():
                if n != name and cb.isChecked():
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
        self.refresh()

    def mode(self):
        return next((n for n, cb in self.mode_boxes.items() if cb.isChecked()), None)

    def values(self) -> dict:
        return {n: self._value(n) for n in self.widgets}

    def refresh(self, *_):
        mode = self.mode()
        vals = self.values()
        eff = self.spec.effective(vals, mode)
        for name, (lab, w) in self.rows.items():
            p = self.spec.param(name)
            vis = self.spec.is_visible(p, eff)
            lab.setVisible(vis)
            w.setVisible(vis)
            dis = self.spec.is_disabled(p, eff, mode)
            w.setEnabled(not dis)
            if dis and eff.get(name) is not None and self._value(name) != eff[name]:
                self._set(name, eff[name])


# --------------------------------------------------------------------------
# Preview with a crop rectangle
# --------------------------------------------------------------------------

class CropView(QGraphicsView):
    rectChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(10, 10, 12))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setFrameShape(QFrame.NoFrame)
        self.pix = QGraphicsPixmapItem()
        self.scene().addItem(self.pix)
        self.rect_item = QGraphicsRectItem()
        self.rect_item.setPen(QPen(QColor(255, 200, 60), 0))
        self.rect_item.setBrush(QBrush(QColor(255, 200, 60, 25)))
        self.scene().addItem(self.rect_item)
        self.full = (0, 0)
        self.scale_f = 1.0
        self._drag = None
        self.locked = False                  # True while showing a saved step (no crop box)
        self.on_click = None                 # callable(x, y, button) while locked (e.g. Spot Fix)
        self.on_wheel = None                 # callable(steps)
        self.message = "Select a master on the left"

    def show_still(self, rgb8):
        """Show a saved step's preview: no crop box, no dragging."""
        h, w, _ = rgb8.shape
        self.set_image(rgb8, (w, h), 1.0)
        self.locked = True
        self.rect_item.setVisible(False)

    def set_image(self, rgb8, full_size, scale_f):
        self.locked = False
        self.rect_item.setVisible(True)
        h, w, _ = rgb8.shape
        arr = np.ascontiguousarray(rgb8)
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.pix.setPixmap(QPixmap.fromImage(img))
        self.pix.setScale(scale_f)
        self.full = full_size
        self.scene().setSceneRect(QRectF(0, 0, full_size[0], full_size[1]))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def set_crop(self, rect):
        if rect is None:
            rect = (0, 0, self.full[0], self.full[1])
        x0, y0, x1, y1 = rect
        self.rect_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self.rectChanged.emit()

    def crop(self):
        r = self.rect_item.rect()
        w, h = self.full
        x0, y0 = max(0, int(r.left())), max(0, int(r.top()))
        x1, y1 = min(w, int(r.right())), min(h, int(r.bottom()))
        return (x0, y0, x1, y1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.full[0]:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        if self.on_wheel:
            self.on_wheel(e.angleDelta().y() / 120.0)
            return
        super().wheelEvent(e)

    def mousePressEvent(self, e):
        if self.locked and self.on_click and self.full[0]:
            p = self.mapToScene(e.position().toPoint())
            self.on_click(p.x(), p.y(), e.button())
            return
        if e.button() == Qt.LeftButton and self.full[0] and not self.locked:
            p = self.mapToScene(e.position().toPoint())
            r = self.rect_item.rect()
            tol = 12 / max(self.transform().m11(), 1e-6)
            edges = {"l": abs(p.x() - r.left()) < tol, "r": abs(p.x() - r.right()) < tol,
                     "t": abs(p.y() - r.top()) < tol, "b": abs(p.y() - r.bottom()) < tol}
            if any(edges.values()) and r.adjusted(-tol, -tol, tol, tol).contains(p):
                self._drag = ("edge", edges, p, QRectF(r))
            elif r.contains(p):
                self._drag = ("move", None, p, QRectF(r))
            else:
                self._drag = ("new", None, p, QRectF(p, p))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag:
            kind, edges, p0, r0 = self._drag
            p = self.mapToScene(e.position().toPoint())
            w, h = self.full
            r = QRectF(r0)
            if kind == "move":
                r.translate(p.x() - p0.x(), p.y() - p0.y())
                if r.left() < 0:
                    r.translate(-r.left(), 0)
                if r.right() > w:
                    r.translate(w - r.right(), 0)
                if r.top() < 0:
                    r.translate(0, -r.top())
                if r.bottom() > h:
                    r.translate(0, h - r.bottom())
            elif kind == "edge":
                if edges["l"]:
                    r.setLeft(min(max(0, p.x()), r.right() - 16))
                if edges["r"]:
                    r.setRight(max(min(w, p.x()), r.left() + 16))
                if edges["t"]:
                    r.setTop(min(max(0, p.y()), r.bottom() - 16))
                if edges["b"]:
                    r.setBottom(max(min(h, p.y()), r.top() + 16))
            else:
                r = QRectF(p0, p).normalized().intersected(QRectF(0, 0, w, h))
            if r.width() >= 16 and r.height() >= 16:
                self.rect_item.setRect(r)
                self.rectChanged.emit()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = None
        super().mouseReleaseEvent(e)

    def drawForeground(self, painter, rect):
        if not self.full[0]:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor(150, 150, 160))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, self.message)
            painter.restore()


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

class PrepareWorker(QThread):
    progress = Signal(str, float)
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, jobs, opts_for):
        super().__init__()
        self.jobs = jobs                  # list of lists of master paths
        self.opts_for = opts_for          # callable(set_index) -> PrepareOptions
        self.cancel_requested = False
        self.results = []

    def run(self):
        try:
            n = len(self.jobs)
            for i, paths in enumerate(self.jobs):
                self.progress.emit(f"Loading masters ({len(paths)})", i / n)
                masters = [P.load_master(p) for p in paths]
                res = P.run_prepare(masters, self.opts_for(i),
                                    progress=lambda t, f, i=i: self.progress.emit(t, (i + f) / n),
                                    log=self.log.emit, cancel=lambda: self.cancel_requested,
                                    save_image=save_rgb8)
                res["target"] = masters[0].target
                res["masters"] = masters
                self.results.append(res)
        except P.Cancelled:
            self.log.emit("Cancelled.")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1500:]}")


# --------------------------------------------------------------------------
# Ready dialog
# --------------------------------------------------------------------------

class ReadyDialog(QDialog):
    def __init__(self, parent, results):
        super().__init__(parent)
        self.setWindowTitle("Ready for final processing")
        self.resize(820, 700)
        lay = QVBoxLayout(self)
        head = QLabel("<h2>Your images now appear to be ready for final processing in PixInsight</h2>"
                      "Below is a recommended order of PixInsight processes and scripts — steps already "
                      "done here are ticked. The same plan is saved next to your prepared files.")
        head.setWordWrap(True)
        lay.addWidget(head)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        html = "".join(Path(r["report"]).read_text(encoding="utf-8").split("<body>", 1)[-1]
                       .replace("</body></html>", "") for r in results)
        view.setHtml(f"<div style='font-size:10pt'>{html}</div>")
        lay.addWidget(view, 1)
        row = QHBoxLayout()
        self.results = results
        b_folder = QPushButton("Stop Here – Continue in PixInsight")
        b_folder.setToolTip("Open the folder with the prepared masters selected, ready to load into PixInsight")
        b_folder.clicked.connect(lambda: reveal_prepared(results))
        b_fin = QPushButton("Finish the Image Here…")
        b_fin.setToolTip("Carry on in SubZeroAstro: colour, brighten, stars - pick from previews")
        b_fin.setStyleSheet("QPushButton { background:#4a86e8; color:white; font-weight:bold; padding:5px 12px; "
                            "border-radius:4px; }")
        b_fin.clicked.connect(lambda: (self.accept(), open_finish(parent, results)))
        b_report = QPushButton("Open Plan in Browser")
        b_report.clicked.connect(lambda: _open(results[0]["report"]))
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        row.addWidget(b_folder)
        row.addWidget(b_fin)
        row.addWidget(b_report)
        row.addStretch(1)
        row.addWidget(b_close)
        lay.addLayout(row)


def prepared_masters(results) -> list:
    """The prepared master files (one per filter, XISF preferred, stars images left out)."""
    by = {}
    for r in results or []:
        for f in r.get("files", []):
            f = Path(f)
            if f.suffix.lower() in (".xisf", ".fits") and f.stem.endswith("_prepared"):
                if f.stem not in by or f.suffix.lower() == ".xisf":
                    by[f.stem] = f
    return list(by.values())


def reveal_prepared(results):
    files = prepared_masters(results)
    tools.reveal(files or [r["out_dir"] for r in results or []])


def open_finish(parent, results):
    files = prepared_masters(results)
    if not files:
        return None
    import finish_window
    first_target = results[0].get("target")
    files = [f for f in files if not first_target or f.name.startswith(first_target)] or files
    win = finish_window.FinishWindow(parent, parent.settings if hasattr(parent, "settings") else None, files)
    win.setAttribute(Qt.WA_DeleteOnClose, True)
    win.show()
    win.raise_()
    if parent is not None:
        parent.finish_win = win
    return win


def _open(path):
    if path and sys.platform.startswith("win"):
        try:
            os.startfile(str(Path(path)))  # noqa: S606 - Path() fixes '//nas/share' style network paths
        except OSError:
            pass


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

class PrepareWindow(QMainWindow):
    def __init__(self, parent, settings, master_files=None):
        super().__init__(parent)
        self.setWindowTitle("SubZeroAstro — Prepare for PixInsight")
        self.settings = settings
        self.sets = []                       # list of lists of Master (preview copies)
        self.crops = {}                      # set index -> rect
        self.specs = {}
        self.panels = {}
        self.rc_products = None
        self.worker = None
        self.cps = None                      # checkpoints of the selected target
        self.last_results = []
        import ui_header
        ui_header.install_header(self, ui_header.PAGE_PREPARE, with_menu=False)
        self._build_ui()
        self.resize(1450, 900)
        self.detect_tools()
        if master_files:
            self.add_master_paths(master_files)

    # ------------------------------------------------------------------ UI
    def _tool_row(self, form, label, key, finder, tip):
        row = QHBoxLayout()
        edit = QLineEdit(self.settings.value(key, "") or "")
        edit.setPlaceholderText("auto-detect")
        edit.setToolTip(tip)
        btn = QPushButton("Browse…")
        status = QLabel()
        status.setMinimumWidth(260)

        def browse():
            f, _ = QFileDialog.getOpenFileName(self, f"Locate {label}", edit.text() or "C:\\Program Files",
                                               "Programs (*.exe);;All files (*)")
            if f:
                edit.setText(f)
                self.settings.setValue(key, f)
                self.detect_tools()
        btn.clicked.connect(browse)
        edit.editingFinished.connect(lambda: (self.settings.setValue(key, edit.text().strip()), self.detect_tools()))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        row.addWidget(status)
        form.addRow(label + ":", row)
        return edit, status

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        tbox = QGroupBox("Your tools")
        tform = QFormLayout(tbox)
        self.astap_edit, self.astap_status = self._tool_row(
            tform, "ASTAP", "prep/astap", tools.find_astap, "ASTAP program (astap.exe or astap_cli.exe) or its folder")
        self.gx_edit, self.gx_status = self._tool_row(
            tform, "GraXpert", "prep/graxpert", tools.find_graxpert, "GraXpert program (GraXpert-win64.exe)")
        self.rc_edit, self.rc_status = self._tool_row(
            tform, "RC-Astro CLI", "prep/rcastro", tools.find_rcastro,
            r"rc-astro.exe (normally C:\Program Files\RC-Astro\CLI)")
        root.addWidget(tbox)

        split = QSplitter(Qt.Horizontal)
        # masters
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Masters</b> (one row per target)"))
        self.set_list = QListWidget()
        self.set_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.set_list.currentRowChanged.connect(self.show_set)
        ll.addWidget(self.set_list, 1)
        brow = QHBoxLayout()
        b_add = QPushButton("Add Masters…")
        b_add.clicked.connect(self.add_masters_dialog)
        b_dir = QPushButton("Add Folder…")
        b_dir.clicked.connect(self.add_folder_dialog)
        brow.addWidget(b_add)
        brow.addWidget(b_dir)
        ll.addLayout(brow)
        split.addWidget(left)

        # preview + crop
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        self.view = CropView()
        self.view.rectChanged.connect(self.crop_changed)
        ml.addWidget(self.view, 1)
        crow = QHBoxLayout()
        b_auto = QPushButton("Auto-crop")
        b_auto.setToolTip("The area covered by (nearly) every frame — removes the ragged stacking edges")
        b_auto.clicked.connect(self.auto_crop)
        b_full = QPushButton("Full image")
        b_full.clicked.connect(lambda: self.view.set_crop(None))
        self.crop_label = QLabel()
        crow.addWidget(QLabel("Crop (drag the yellow box or its edges):"))
        crow.addWidget(b_auto)
        crow.addWidget(b_full)
        crow.addWidget(self.crop_label, 1)
        ml.addLayout(crow)
        self.crop_row_widgets = (b_auto, b_full)

        # step history (0.6): saved progress after each step
        hbox = QGroupBox("Step history — saved after each step")
        hl = QVBoxLayout(hbox)
        hr = QHBoxLayout()
        hr.addWidget(QLabel("Show:"))
        self.hist_combo = QComboBox()
        self.hist_combo.setToolTip("Look at the image after any finished step")
        self.hist_combo.currentIndexChanged.connect(self.show_history_step)
        hr.addWidget(self.hist_combo, 1)
        self.btn_before = QPushButton("Hold for Before")
        self.btn_before.setToolTip("Press and hold to see the image before this step")
        self.btn_before.pressed.connect(lambda: self.show_history_step(before=True))
        self.btn_before.released.connect(lambda: self.show_history_step())
        hr.addWidget(self.btn_before)
        self.btn_redo = QPushButton("Go Back && Redo This Step")
        self.btn_redo.setToolTip("Throw away this step and the ones after it. Change its settings on the right, "
                                 "then press Prepare — the earlier steps are not run again.")
        self.btn_redo.clicked.connect(self.redo_from_step)
        hr.addWidget(self.btn_redo)
        self.btn_hclear = QPushButton("Start Fresh")
        self.btn_hclear.setToolTip("Delete the saved progress for this target and run every step again")
        self.btn_hclear.clicked.connect(self.clear_history)
        hr.addWidget(self.btn_hclear)
        hl.addLayout(hr)
        self.hist_label = QLabel()
        self.hist_label.setWordWrap(True)
        self.hist_label.setStyleSheet("color:#9a9aa6;")
        hl.addWidget(self.hist_label)
        ml.addWidget(hbox)
        split.addWidget(mid)

        # steps
        right = QScrollArea()
        right.setWidgetResizable(True)
        steps = QWidget()
        sl = QVBoxLayout(steps)
        s = self.settings
        self.chk_crop = QCheckBox("Crop (same box for every filter)")
        self.chk_crop.setChecked(s.value("prep/crop", True, type=bool))
        sl.addWidget(self.chk_crop)
        self.chk_solve = QCheckBox("Plate solve with ASTAP")
        self.chk_solve.setChecked(s.value("prep/solve", True, type=bool))
        sl.addWidget(self.chk_solve)
        sv = QHBoxLayout()
        sv.addSpacing(22)
        sv.addWidget(QLabel("Search radius"))
        self.solve_radius = QDoubleSpinBox()
        self.solve_radius.setRange(1, 180)
        self.solve_radius.setValue(float(s.value("prep/solve_radius", 30.0)))
        self.solve_radius.setSuffix(" °")
        sv.addWidget(self.solve_radius)
        self.chk_blind = QCheckBox("Search the whole sky")
        self.chk_blind.setToolTip("Blind solve: ignore the pointing in the header and search the whole sky. "
                                  "Slower - use it when the normal solve fails (wrong or missing coordinates).")
        self.chk_blind.setChecked(s.value("prep/blind", False, type=bool))
        self.chk_blind.toggled.connect(lambda on: self.solve_radius.setEnabled(not on))
        self.solve_radius.setEnabled(not self.chk_blind.isChecked())
        sv.addWidget(self.chk_blind)
        sv.addStretch(1)
        sl.addLayout(sv)
        self.chk_grad = QCheckBox("Remove gradients")
        self.chk_grad.setChecked(s.value("prep/gradient", True, type=bool))
        sl.addWidget(self.chk_grad)
        gv = QFormLayout()
        self.grad_tool = QComboBox()
        self.grad_tool.addItems(["GraXpert (AI)", "Built-in (polynomial)"])
        self.grad_tool.setCurrentIndex(int(s.value("prep/grad_tool", 0)))
        self.gx_smooth = QDoubleSpinBox()
        self.gx_smooth.setRange(0, 1)
        self.gx_smooth.setSingleStep(0.05)
        self.gx_smooth.setValue(float(s.value("prep/gx_smoothing", 0.1)))
        self.gx_corr = QComboBox()
        self.gx_corr.addItems(["Subtraction", "Division"])
        self.gx_corr.setCurrentText(s.value("prep/gx_correction", "Subtraction"))
        self.gx_corr.setToolTip("Subtraction for light pollution / moon gradients; Division for vignetting-like "
                                "residuals")
        gv.addRow("Method:", self.grad_tool)
        gv.addRow("GraXpert smoothing:", self.gx_smooth)
        gv.addRow("Correction:", self.gx_corr)
        gw = QWidget()
        gw.setLayout(gv)
        gw.setContentsMargins(18, 0, 0, 0)
        sl.addWidget(gw)

        self.xt_boxes = {}
        self.xt_holders = {}
        for key, label in (("bxt", "BlurXTerminator"), ("nxt", "NoiseXTerminator (linear, light)"),
                           ("sxt", "StarXTerminator (starless + stars images)")):
            box = QGroupBox(label)
            box.setCheckable(True)
            cfg = self._xt_cfg(key)
            box.setChecked(bool(cfg.get("enabled")))
            v = QVBoxLayout(box)
            holder = QLabel("Set the RC-Astro CLI above to load the settings")
            holder.setStyleSheet("color:#8a8a96;")
            v.addWidget(holder)
            sl.addWidget(box)
            self.xt_boxes[key] = box
            self.xt_holders[key] = (v, holder)
        dv = QHBoxLayout()
        dv.addWidget(QLabel("RC-Astro device:"))
        self.device = QComboBox()
        self.device.setToolTip("Which processor runs the XTerminators. A dedicated NVIDIA/AMD card is much faster "
                               "than integrated graphics.")
        dv.addWidget(self.device, 1)
        sl.addLayout(dv)

        # ---- final output: where, and in which formats
        outbox = QGroupBox("Save final images")
        ov = QVBoxLayout(outbox)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Folder:"))
        self.out_edit = QLineEdit(s.value("prep/out_dir", "") or "")
        self.out_edit.setPlaceholderText("blank = a 'Prepared' folder next to the masters")
        self.out_edit.setToolTip("Where the finished images go. Leave blank to use a 'Prepared' folder next to "
                                 "each set of master lights. If you pick a folder, each target gets its own "
                                 "subfolder named after it (e.g. …\\IC1805), so targets never mix. "
                                 "Your original masters are never changed.")
        orow.addWidget(self.out_edit, 1)
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self.browse_out_dir)
        orow.addWidget(b_out)
        b_clear = QPushButton("Default")
        b_clear.setToolTip("Go back to a 'Prepared' folder next to the masters")
        b_clear.clicked.connect(lambda: self.out_edit.clear())
        orow.addWidget(b_clear)
        ov.addLayout(orow)

        def fmt_box(label, key, default, tip):
            cb = QCheckBox(label)
            cb.setChecked(s.value(key, default, type=bool))
            cb.setToolTip(tip)
            return cb
        lin = QHBoxLayout()
        lin.addWidget(QLabel("Linear masters (for PixInsight):"))
        self.fmt_xisf = fmt_box("XISF", "prep/xisf", True, "PixInsight's own format — recommended")
        self.fmt_fits = fmt_box("FITS", "prep/fits", False, "Standard astronomy format, opens anywhere")
        lin.addWidget(self.fmt_xisf)
        lin.addWidget(self.fmt_fits)
        lin.addStretch(1)
        ov.addLayout(lin)
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Stretched colour preview:"))
        self.fmt_tif = fmt_box("TIFF (16-bit)", "prep/img_tif", True, "Full quality, good for Photoshop/Affinity")
        self.fmt_jpg = fmt_box("JPG", "prep/img_jpg", True, "Small, for sharing and phone viewing")
        self.fmt_png = fmt_box("PNG", "prep/img_png", False, "Lossless 8-bit")
        for cb in (self.fmt_tif, self.fmt_jpg, self.fmt_png):
            qrow.addWidget(cb)
        qrow.addStretch(1)
        ov.addLayout(qrow)
        qv = QHBoxLayout()
        qv.addWidget(QLabel("Preview palette:"))
        self.palette = QComboBox()
        qv.addWidget(self.palette, 1)
        ov.addLayout(qv)
        self.chk_checkpoints = QCheckBox("Save progress after each step (resume after a crash, go back to a step)")
        self.chk_checkpoints.setChecked(s.value("prep/checkpoints", True, type=bool))
        self.chk_checkpoints.setToolTip("Keeps a copy of the image after each step in the checkpoint folder "
                                        "(Settings). Uses disk space; old copies are cleaned up automatically.")
        ov.addWidget(self.chk_checkpoints)
        ov.addWidget(QLabel("<span style='color:#9a9aa6'>The linear masters are what you process in PixInsight; "
                            "the stretched preview is a quick auto-stretched look at the result.</span>"))
        sl.addWidget(outbox)
        sl.addStretch(1)
        right.setWidget(steps)
        split.addWidget(right)
        split.setSizes([300, 700, 450])
        root.addWidget(split, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        root.addWidget(self.log_view)
        brow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("")
        self.progress.setTextVisible(True)
        brow.addWidget(self.progress, 1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_start = QPushButton("  Prepare for PixInsight  ")
        self.btn_start.setStyleSheet("QPushButton { font-weight: bold; padding: 6px 18px; }")
        self.btn_start.clicked.connect(self.start)
        self.btn_pi = QPushButton("Stop Here – Continue in PixInsight")
        self.btn_pi.setToolTip("Open the folder with the prepared masters (selected) to load them into PixInsight")
        self.btn_pi.setEnabled(False)
        self.btn_pi.clicked.connect(lambda: reveal_prepared(self.last_results))
        self.btn_finish = QPushButton("Finish the Image…")
        self.btn_finish.setToolTip("Carry on here: colour balance, brighten, stars, palettes - pick from "
                                   "previews, step by step")
        self.btn_finish.setEnabled(False)
        self.btn_finish.clicked.connect(lambda: open_finish(self, self.last_results))
        brow.addWidget(self.btn_pi)
        self.app_header.add_next(self.btn_finish, "Finish the Image")          # top right, page-turn style
        brow.addWidget(self.btn_cancel)
        brow.addWidget(self.btn_start)
        root.addLayout(brow)
        self.setCentralWidget(central)

    def log(self, t):
        self.log_view.appendPlainText(t)

    def _xt_cfg(self, key):
        try:
            cfg = json.loads(self.settings.value(f"prep/xt_{key}", "") or "null")
        except (ValueError, TypeError):
            cfg = None
        return cfg or dict(XT_START[key])

    # ------------------------------------------------------------ tools
    def detect_tools(self):
        astap = tools.find_astap(self.astap_edit.text().strip() or None)
        self.astap_path = astap
        self.astap_status.setText(f"<span style='color:#5fd27a'>✓ found</span> {Path(astap).name}" if astap
                                  else "<span style='color:#eba932'>not found — plate solving off</span>")
        gx = tools.find_graxpert(self.gx_edit.text().strip() or None)
        self.gx_path = gx
        self.gx_status.setText(f"<span style='color:#5fd27a'>✓ found</span> {Path(gx).name}" if gx
                               else "<span style='color:#eba932'>not found — built-in method used</span>")
        rc = tools.find_rcastro(self.rc_edit.text().strip() or None)
        self.rc_path = rc
        self.rc_products = None
        if rc:
            try:
                r = tools.RCAstro(rc)
                self.rc_products = r.products()
                lic = {p["key"]: p["license"].get("valid") for p in self.rc_products.get("products", [])}
                names = {p["key"]: p.get("displayName", p["key"]) for p in self.rc_products.get("products", [])}
                txt = " · ".join(f"{names[k]} {'✓' if v else '✗ (not licensed)'}" for k, v in lic.items())
                self.rc_status.setText(f"<span style='color:#5fd27a'>✓ v{r.cli_version}</span> {txt}")
                if r.schema and r.schema != tools.SCHEMA_VERSION:
                    self.log(f"Note: rc-astro reports format version {r.schema}; this app was written for "
                             f"{tools.SCHEMA_VERSION}. Things should still work — please report any problem.")
                for key in XT_KEYS:
                    if key in lic:
                        self.load_panel(r, key, lic[key])
                self.device.clear()
                for d in self.rc_products.get("devices", []):
                    label = d["id"] + (f"  –  {d['name']}" if d.get("name") else "")
                    if d["id"] == "auto" and d.get("device"):
                        label = f"auto  –  RC-Astro's choice (currently {d['device']})"
                    self.device.addItem(label, d["id"])
                want = self.settings.value("prep/device", "") or tools.preferred_device(
                    self.rc_products.get("devices", []))
                i = self.device.findData(want)
                self.device.setCurrentIndex(max(i, 0))
            except tools.ToolError as e:
                self.rc_status.setText(f"<span style='color:#e65050'>error: {e}</span>")
        else:
            self.rc_status.setText("<span style='color:#eba932'>not found — XTerminator steps off</span>")
        for key in XT_KEYS:
            if key not in self.panels:
                self.xt_boxes[key].setEnabled(False)

    def load_panel(self, rc, key, licensed):
        v, holder = self.xt_holders[key]
        try:
            spec = rc.spec(key)
        except tools.ToolError as e:
            holder.setText(str(e))
            return
        self.specs[key] = spec
        if key in self.panels:
            self.panels[key].setParent(None)
        cfg = self._xt_cfg(key)
        panel = ParamPanel(spec, cfg.get("values"), cfg.get("mode"))
        holder.setVisible(not licensed)
        holder.setText("Not licensed on this computer — activate it in your RC-Astro account")
        v.addWidget(panel)
        self.panels[key] = panel
        self.xt_boxes[key].setEnabled(licensed)

    # ------------------------------------------------------------ masters
    def add_masters_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Master lights", self.settings.value("stack/masters_root", ""),
                                                "Masters (*.fits *.fit *.fts *.xisf)")
        if files:
            self.add_master_paths(files)

    def add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with master lights",
                                             self.settings.value("stack/masters_root", ""))
        if d:
            self.add_master_paths([d])

    def add_master_paths(self, paths):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = P.find_masters(paths)
            known = {str(m.path) for s in self.sets for m in s}
            new = [P.load_master(f) for f in found if str(f) not in known]
        finally:
            QApplication.restoreOverrideCursor()
        if not new:
            self.log("No (new) master lights found there.")
            return
        allm = [m for s in self.sets for m in s] + new
        self.sets = P.target_sets(allm)
        self.set_list.clear()
        for s in self.sets:
            m0 = s[0]
            filters = ", ".join(m.filter for m in s)
            it = QListWidgetItem(f"{m0.target}   [{filters}]\n{m0.shape[1]} × {m0.shape[0]}"
                                 + ("  ·  solved" if all(m.solved for m in s) else ""))
            it.setToolTip("\n".join(str(m.path) for m in s))
            self.set_list.addItem(it)
        self.log(f"Loaded {len(new)} master(s): " + ", ".join(m.name for m in new))
        self.set_list.setCurrentRow(len(self.sets) - 1)

    def show_set(self, row):
        if not 0 <= row < len(self.sets):
            return
        s = self.sets[row]
        pals = P.palettes_for(s)
        self.palette.clear()
        self.palette.addItems(pals)
        step = max(1, int(np.ceil(max(s[0].shape) / 1400)))
        small = [type(m)(**{**m.__dict__, "data": m.data[:, ::step, ::step]}) for m in s]
        rgb = P.quick_look(small, pals[0]) if pals else np.zeros((10, 10, 3))
        h, w = s[0].shape
        self.view.set_image((rgb * 255 + 0.5).astype(np.uint8), (w, h), step)
        self.view.set_crop(self.crops.get(row, P.common_autocrop(s)))
        self.refresh_history(keep_view=True)

    # ------------------------------------------------------------ step history
    def current_checkpoints(self):
        row = self.set_list.currentRow()
        if not 0 <= row < len(self.sets):
            return None
        s = self.sets[row]
        return P.PrepareCheckpoints([m.path for m in s], s[0].target)

    def refresh_history(self, keep_view=False, select=None):
        cps = self.current_checkpoints()
        self.cps = cps
        steps = cps.steps if cps else []
        self.hist_combo.blockSignals(True)
        self.hist_combo.clear()
        self.hist_combo.addItem("Original masters (set the crop here)")
        for i, st_ in enumerate(steps, 1):
            self.hist_combo.addItem(f"After {i}. {st_['title']}  —  {st_.get('detail', '')}"[:140])
        if select is not None and 0 <= select < self.hist_combo.count():
            self.hist_combo.setCurrentIndex(select)
        self.hist_combo.blockSignals(False)
        if steps:
            self.hist_label.setText(
                f"{len(steps)} step(s) saved ({', '.join(t['title'] for t in steps)}; "
                f"{cp.human(cps.size())}, last {steps[-1].get('time', '')}). Pressing Prepare continues from "
                "here — any step whose settings you changed is redone, together with the steps after it.")
        else:
            self.hist_label.setText("No saved progress for this target yet. Each finished step will appear here, "
                                    "so you can look at it, compare it with the step before, or go back and "
                                    "redo it with other settings.")
        self.hist_combo.setEnabled(bool(steps))
        self.btn_hclear.setEnabled(bool(steps))
        if not keep_view or select is not None:
            self.show_history_step()
        else:
            self._update_history_buttons()

    def _update_history_buttons(self):
        i = self.hist_combo.currentIndex()
        self.btn_before.setEnabled(i >= 1)
        self.btn_redo.setEnabled(i >= 1 and not (self.worker and self.worker.isRunning()))
        for w in self.crop_row_widgets:
            w.setEnabled(i <= 0)

    def show_history_step(self, *_a, before=False):
        i = self.hist_combo.currentIndex()
        self._update_history_buttons()
        if before:
            i -= 1
        if i <= 0:
            if self.view.locked:
                row = self.set_list.currentRow()
                self.hist_combo.blockSignals(True)
                self.show_set(row) if not before else self._show_original()
                self.hist_combo.blockSignals(False)
            return
        rgb8 = self.cps.preview(i - 1) if self.cps else None
        if rgb8 is not None:
            self.view.show_still(rgb8)

    def _show_original(self):
        row = self.set_list.currentRow()
        if 0 <= row < len(self.sets):
            s = self.sets[row]
            pals = P.palettes_for(s)
            step = max(1, int(np.ceil(max(s[0].shape) / 1400)))
            small = [type(m)(**{**m.__dict__, "data": m.data[:, ::step, ::step]}) for m in s]
            rgb = P.quick_look(small, pals[0]) if pals else np.zeros((10, 10, 3))
            self.view.show_still((rgb * 255 + 0.5).astype(np.uint8))

    def redo_from_step(self):
        i = self.hist_combo.currentIndex()
        if i < 1 or not self.cps:
            return
        st_ = self.cps.steps[i - 1]
        later = [t["title"] for t in self.cps.steps[i:]]
        msg = (f"Go back to before <b>{st_['title']}</b>?<br><br>The saved result of this step"
               + (f" and of the steps after it ({', '.join(later)})" if later else "")
               + " will be thrown away. Change the settings on the right if you like, then press "
                 "<b>Prepare for PixInsight</b> — the steps before it are not run again.")
        if QMessageBox.question(self, "Go back", msg) != QMessageBox.Yes:
            return
        self.cps.truncate(i - 1)
        self.log(f"Went back to before {st_['title']} — it will be redone on the next run.")
        self.refresh_history(select=i - 1)

    def clear_history(self):
        if not self.cps or not self.cps.steps:
            return
        if QMessageBox.question(self, "Start fresh", "Delete the saved progress for this target? Every step "
                                "will run again on the next Prepare.") != QMessageBox.Yes:
            return
        self.cps.clear()
        self.refresh_history(select=0)

    def auto_crop(self):
        row = self.set_list.currentRow()
        if 0 <= row < len(self.sets):
            r = P.common_autocrop(self.sets[row])
            if r is None:
                self.log("These masters have no auto-crop information (stacked with an older version?) — "
                         "drag a crop box instead.")
            self.view.set_crop(r)

    def crop_changed(self):
        row = self.set_list.currentRow()
        if 0 <= row < len(self.sets):
            self.crops[row] = self.view.crop()
            x0, y0, x1, y1 = self.crops[row]
            self.crop_label.setText(f"x {x0}–{x1}, y {y0}–{y1}  ({x1 - x0} × {y1 - y0})")

    # ------------------------------------------------------------ run
    def _save_settings(self):
        s = self.settings
        for k, w in (("prep/crop", self.chk_crop), ("prep/solve", self.chk_solve), ("prep/gradient", self.chk_grad),
                     ("prep/xisf", self.fmt_xisf), ("prep/fits", self.fmt_fits), ("prep/img_tif", self.fmt_tif),
                     ("prep/img_jpg", self.fmt_jpg), ("prep/img_png", self.fmt_png),
                     ("prep/checkpoints", self.chk_checkpoints)):
            s.setValue(k, w.isChecked())
        s.setValue("prep/out_dir", self.out_edit.text().strip())
        s.setValue("prep/solve_radius", self.solve_radius.value())
        s.setValue("prep/blind", self.chk_blind.isChecked())
        s.setValue("prep/grad_tool", self.grad_tool.currentIndex())
        s.setValue("prep/gx_smoothing", self.gx_smooth.value())
        s.setValue("prep/gx_correction", self.gx_corr.currentText())
        if self.device.count():
            s.setValue("prep/device", self.device.currentData())
        for key in XT_KEYS:
            panel = self.panels.get(key)
            cfg = {"enabled": self.xt_boxes[key].isChecked()}
            if panel:
                cfg.update(values=panel.values(), mode=panel.mode())
            s.setValue(f"prep/xt_{key}", json.dumps(cfg))

    def options_for(self, index):
        fmts = tuple(f for f, cb in (("xisf", self.fmt_xisf), ("fits", self.fmt_fits)) if cb.isChecked())
        imgs = tuple(f for f, cb in (("tif", self.fmt_tif), ("jpg", self.fmt_jpg), ("png", self.fmt_png))
                     if cb.isChecked())
        xt = {}
        for key in XT_KEYS:
            panel = self.panels.get(key)
            if panel and self.xt_boxes[key].isChecked() and self.xt_boxes[key].isEnabled():
                xt[key] = {"enabled": True, "values": panel.values(), "mode": panel.mode(), "spec": self.specs[key]}
        return P.PrepareOptions(
            crop=self.chk_crop.isChecked(), crop_rect=self.crops.get(index), solve=self.chk_solve.isChecked(),
            astap=self.astap_path, solve_radius=180.0 if self.chk_blind.isChecked() else self.solve_radius.value(),
            gradient=self.chk_grad.isChecked(),
            gradient_tool="graxpert" if self.grad_tool.currentIndex() == 0 and self.gx_path else "builtin",
            graxpert=self.gx_path, gx_smoothing=self.gx_smooth.value(), gx_correction=self.gx_corr.currentText(),
            rcastro=self.rc_path, xt=xt, device=self.device.currentData() or "auto", formats=fmts,
            quicklook=bool(imgs), image_formats=imgs, checkpoints=self.chk_checkpoints.isChecked(), out_dir=self.out_edit.text().strip() or None,
            palette=self.palette.currentText() if index == self.set_list.currentRow() else None)

    def browse_out_dir(self):
        start = self.out_edit.text().strip() or self.settings.value("stack/masters_root", "") or ""
        d = QFileDialog.getExistingDirectory(self, "Save final images to", start)
        if d:
            self.out_edit.setText(d)
            self.settings.setValue("prep/out_dir", d)

    def start(self, show_dialog=True):
        if not self.sets:
            QMessageBox.information(self, "Prepare", "Add some master lights first.")
            return
        if not any(cb.isChecked() for cb in (self.fmt_xisf, self.fmt_fits, self.fmt_tif, self.fmt_jpg, self.fmt_png)):
            QMessageBox.information(self, "Prepare", "Tick at least one format under 'Save final images'.")
            return
        out = self.out_edit.text().strip()
        if out:
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Prepare", f"Can't use the output folder:\n{out}\n\n{e}")
                return
        self.show_dialog = show_dialog
        self._save_settings()
        jobs = [[m.path for m in s] for s in self.sets]
        self.btn_start.setEnabled(False)
        self.btn_redo.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker = PrepareWorker(jobs, self.options_for)
        self.worker.progress.connect(lambda t, f: (self.progress.setValue(int(1000 * f)), self.progress.setFormat(t)))
        self.worker.log.connect(self.log)
        self.worker.failed.connect(lambda m: (self.log("ERROR: " + m),
                                              QMessageBox.critical(self, "Prepare", m[:800])))
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel_requested = True
            self.btn_cancel.setEnabled(False)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        res = self.worker.results if self.worker else []
        self.last_results = res
        self.btn_pi.setEnabled(bool(res))
        self.btn_finish.setEnabled(bool(res))
        self.refresh_history(keep_view=True)
        if self.hist_combo.count() > 1:
            self.hist_combo.setCurrentIndex(self.hist_combo.count() - 1)
        if res:
            self.progress.setFormat("Ready for final processing in PixInsight")
            if getattr(self, "show_dialog", True):
                ReadyDialog(self, res).exec()

    def closeEvent(self, ev):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_requested = True
            self.worker.wait(20000)
        self._save_settings()
        super().closeEvent(ev)
