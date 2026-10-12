"""
finish_window.py - Stage 5 window: "Finish the Image".

A guided path from prepared linear masters to a finished picture:
  * a "Recommended next step" card that knows the data (OSC, mono RGB/LRGB,
    narrowband, smart telescope) and what has been done;
  * every tool shows 4-5 small versions to pick from - click the one you like;
  * every step is saved (step history, go back, resume after a crash, save / open
    a project);
  * "Stop Here - Continue in PixInsight" at any point.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QFrame, QGraphicsEllipseItem, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QSplitter, QToolButton, QVBoxLayout, QWidget)

import checkpoints as cp
import finish as F
import tools
from prepare_window import CropView, save_rgb8

STATUS_MARK = {"done": "✓", "next": "➜", "later": "○", "skipped": "⤼", "n/a": "–", "optional": "◇"}
PREVIEW_SIDE = 1400                         # live preview size (fast, still sharp on screen)


class CurveWidget(QWidget):
    """Shows the tone curve; the three inner points can be dragged."""
    changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 170)
        self.points = [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1, 1]]
        self.drag = None
        self.setToolTip("Drag the points: up = brighter, down = darker. Left = shadows, right = highlights.")

    def set_points(self, pts):
        self.points = [list(map(float, q)) for q in pts]
        self.update()

    def _xy(self, q):
        m = 8
        w, h = self.width() - 2 * m, self.height() - 2 * m
        return QPointF(m + q[0] * w, m + (1 - q[1]) * h)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        p.setPen(QPen(QColor(60, 60, 70), 1))
        for f in (0.25, 0.5, 0.75):
            p.drawLine(self._xy([f, 0]), self._xy([f, 1]))
            p.drawLine(self._xy([0, f]), self._xy([1, f]))
        p.setPen(QPen(QColor(90, 90, 100), 1, Qt.DashLine))
        p.drawLine(self._xy([0, 0]), self._xy([1, 1]))
        xs = np.linspace(0, 1, 120)
        pts = np.asarray(self.points)
        ys = np.clip(F.pchip(pts[:, 0], pts[:, 1], xs), 0, 1)
        path = QPainterPath(self._xy([0, ys[0]]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._xy([x, y]))
        p.setPen(QPen(QColor(120, 170, 255), 2))
        p.drawPath(path)
        p.setBrush(QColor(255, 200, 60))
        for q in self.points[1:-1]:
            p.drawEllipse(self._xy(q), 5, 5)

    def mousePressEvent(self, e):
        pos = e.position()
        best = min(range(1, len(self.points) - 1),
                   key=lambda i: (self._xy(self.points[i]) - pos).manhattanLength())
        if (self._xy(self.points[best]) - pos).manhattanLength() < 25:
            self.drag = best

    def mouseMoveEvent(self, e):
        if self.drag is None:
            return
        m = 8
        w, h = self.width() - 2 * m, self.height() - 2 * m
        x = min(max((e.position().x() - m) / w, 0.02), 0.98)
        y = min(max(1 - (e.position().y() - m) / h, 0), 1)
        i = self.drag
        x = min(max(x, self.points[i - 1][0] + 0.03), self.points[i + 1][0] - 0.03)
        y = min(max(y, self.points[i - 1][1]), self.points[i + 1][1])        # keep tones in order
        self.points[i] = [x, y]
        self.update()
        self.changed.emit([list(q) for q in self.points])

    def mouseReleaseEvent(self, _e):
        self.drag = None


def to_pixmap(rgb8: np.ndarray) -> QPixmap:
    h, w, _ = rgb8.shape
    a = np.ascontiguousarray(rgb8)
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class Job(QThread):
    """Run one function off the GUI thread."""
    ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1500:]}")


# --------------------------------------------------------------------------
# StarXTerminator for "Take Stars Out"
# --------------------------------------------------------------------------

def make_star_runner(settings):
    """callable(img (C,h,w) 0..1, linear) -> (starless, stars), or None if SXT isn't available."""
    path = settings.value("prep/rcastro", "") or tools.find_rcastro()
    if not path:
        return None
    try:
        rc = tools.RCAstro(path)
        prods = rc.products()
        sxt = next((p for p in prods.get("products", []) if p.get("key") == "sxt"), None)
        if sxt is None or not (sxt.get("license") or {}).get("valid", True):
            return None
    except Exception:  # noqa: BLE001
        return None
    device = settings.value("prep/device", "") or tools.preferred_device(prods.get("devices", []))

    def run(img, linear):
        import fits_core as core
        import prepare as P
        spec = rc.spec("sxt")
        vals = {"stars": True}
        if not linear and spec.param("unscreen"):
            vals["unscreen"] = True          # stretched image: separate so a screen blend puts stars back
        args = spec.args(vals)
        work = Path(tempfile.mkdtemp(prefix="szfin_"))
        src = work / "in.fits"
        core.write_fits(src, np.clip(img[0] if img.shape[0] == 1 else img, 0, 1).astype(np.float32),
                        [("IMAGETYP", "Master Light", ""), ("PIXRANGE", 65535, "values x this = ADU (0..1 range)")],
                        bitpix=-32)
        written = rc.run("sxt", src, work / "out.fits", args, device=device)
        like = img * 65535.0
        main = next((w for w in written if "_stars" not in Path(w).stem), str(work / "out.fits"))
        starless = P._read_result(main, like) / 65535.0
        st_f = [w for w in written if "_stars" in Path(w).stem]
        stars = (P._read_result(st_f[0], like) / 65535.0) if st_f else np.clip(img - starless, 0, None)
        return np.clip(starless, 0, None), np.clip(stars, 0, None)
    return run


# --------------------------------------------------------------------------
# Export dialog
# --------------------------------------------------------------------------

class ExportDialog(QDialog):
    def __init__(self, parent, settings, default_dir):
        super().__init__(parent)
        self.setWindowTitle("Save the Picture")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.dir = QLineEdit(settings.value("finish/export_dir", "") or str(default_dir))
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse())
        row.addWidget(self.dir, 1)
        row.addWidget(b)
        form.addRow("Folder:", row)
        fr = QHBoxLayout()
        self.boxes = {}
        for key, label, on in (("tif", "TIFF (16-bit)", True), ("jpg", "JPG", True), ("png", "PNG", False),
                               ("xisf", "XISF", False), ("fits", "FITS", False)):
            cb = QCheckBox(label)
            cb.setChecked(settings.value(f"finish/fmt_{key}", on, type=bool))
            self.boxes[key] = cb
            fr.addWidget(cb)
        form.addRow("Formats:", fr)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.settings = settings

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Save the picture to", self.dir.text())
        if d:
            self.dir.setText(d)

    def formats(self):
        return [k for k, cb in self.boxes.items() if cb.isChecked()]

    def accept(self):
        self.settings.setValue("finish/export_dir", self.dir.text().strip())
        for k, cb in self.boxes.items():
            self.settings.setValue(f"finish/fmt_{k}", cb.isChecked())
        super().accept()


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

class FinishWindow(QMainWindow):
    def __init__(self, parent, settings, master_files=None):
        super().__init__(parent)
        self.setWindowTitle("SubZeroAstro — Finish the Image")
        self.settings = settings
        self.state: F.State | None = None
        self.project: F.Project | None = None
        self.tool: F.Tool | None = None
        self.variants = []
        self.job = None
        self.synchronous = False           # self-test: run jobs inline
        F.TOOLS["stars_out"].runner = make_star_runner(settings)
        self._build_ui()
        self.resize(1500, 950)
        self.refresh()
        if master_files:
            self.load_masters(master_files)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        self.title = QLabel("<b>Finish the Image</b> — add your prepared masters to start")
        top.addWidget(self.title, 1)
        for text, slot, tip in (
                ("Add Masters…", self.add_masters_dialog, "Open the master lights of one target (prepared ones "
                                                          "are best)"),
                ("Open Project…", self.open_project_dialog, "Carry on with a project you saved earlier"),
                ("Save Project As…", self.save_project_dialog, "Keep this project (every step) in a folder of "
                                                                 "your choice"),
                ("Save the Picture…", self.export_dialog, "Save the image as it looks now (TIFF, JPG, PNG, "
                                                          "XISF, FITS)")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            top.addWidget(b)
        self.btn_pi = QPushButton("  Stop Here – Continue in PixInsight  ")
        self.btn_pi.setToolTip("Save the image exactly as it is now (XISF) with a note on what to do next in "
                               "PixInsight, and open the folder")
        self.btn_pi.setStyleSheet("QPushButton { background:#6d4bb8; color:white; font-weight:bold; "
                                  "padding:5px 12px; border-radius:4px; }")
        self.btn_pi.clicked.connect(self.stop_here_pixinsight)
        top.addWidget(self.btn_pi)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        # left: workflow
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Your workflow</b>"))
        self.flow = QListWidget()
        self.flow.setMinimumWidth(215)
        self.flow.itemClicked.connect(self.flow_clicked)
        self.flow.setToolTip("✓ done   ➜ recommended next   ○ later   ⤼ skipped   – not needed / not "
                             "possible yet\nClick a step to open it")
        ll.addWidget(self.flow, 1)
        self.kind_label = QLabel()
        self.kind_label.setWordWrap(True)
        self.kind_label.setStyleSheet("color:#9a9aa6;")
        ll.addWidget(self.kind_label)
        split.addWidget(left)

        # middle: next-step card, image, history
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        self.card = QFrame()
        self.card.setStyleSheet("QFrame#card { background:#1f2a3a; border:1px solid #35507a; border-radius:6px; }")
        self.card.setObjectName("card")
        cl = QHBoxLayout(self.card)
        self.card_text = QLabel()
        self.card_text.setWordWrap(True)
        cl.addWidget(self.card_text, 1)
        self.btn_do = QPushButton("Do It")
        self.btn_do.setStyleSheet("QPushButton { background:#4a86e8; color:white; font-weight:bold; "
                                  "padding:6px 18px; border-radius:4px; }")
        self.btn_do.clicked.connect(lambda: self.open_tool(F.next_step(self.state).key))
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setToolTip("Leave this step out and move on")
        self.btn_skip.clicked.connect(self.skip_next)
        cl.addWidget(self.btn_do)
        cl.addWidget(self.btn_skip)
        ml.addWidget(self.card)
        self.view = CropView()
        self.view.message = "Add the prepared masters of one target to start"
        ml.addWidget(self.view, 1)
        hr = QHBoxLayout()
        hr.addWidget(QLabel("Show:"))
        self.hist = QComboBox()
        self.hist.currentIndexChanged.connect(self.show_history)
        hr.addWidget(self.hist, 1)
        self.btn_before = QPushButton("Hold for Before")
        self.btn_before.pressed.connect(lambda: self.show_history(before=True))
        self.btn_before.released.connect(lambda: self.show_history())
        hr.addWidget(self.btn_before)
        self.btn_back = QPushButton("Go Back to Here")
        self.btn_back.setToolTip("Undo every step after the one shown, and carry on from there")
        self.btn_back.clicked.connect(self.go_back)
        hr.addWidget(self.btn_back)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last step (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_last)
        hr.addWidget(self.btn_undo)
        self.btn_start = QPushButton("Back to Start")
        self.btn_start.setToolTip("Return to the original masters, before anything was changed")
        self.btn_start.clicked.connect(self.back_to_start)
        hr.addWidget(self.btn_start)
        self.undo_shortcut = QShortcut(QKeySequence(QKeySequence.Undo), self)
        self.undo_shortcut.activated.connect(self.undo_last)
        self.chk_linked = QCheckBox("Natural colours")
        self.chk_linked.setChecked(True)
        self.chk_linked.setToolTip("Preview of unstretched data: ticked = real colours (linked); unticked = each "
                                   "colour balanced on its own (unlinked) to judge detail")
        self.chk_linked.toggled.connect(lambda _on: self.show_current())
        hr.addWidget(self.chk_linked)
        ml.addLayout(hr)
        split.addWidget(mid)

        # right: the open tool
        self.tool_box = QWidget()
        tl = QVBoxLayout(self.tool_box)
        self.tool_title = QLabel()
        self.tool_title.setWordWrap(True)
        tl.addWidget(self.tool_title)
        self.method_row = QWidget()
        mr = QHBoxLayout(self.method_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.addWidget(QLabel("Method:"))
        self.method = QComboBox()
        for k, v in F.Brighten.methods.items():
            self.method.addItem(v, k)
        self.method.setToolTip("Automatic: balanced, like PixInsight's STF.\nColour-preserving: keeps star "
                               "colours (arcsinh).\nHyperbolic (GHS): smooth and strong, best for starless "
                               "nebulae.")
        self.method.currentIndexChanged.connect(lambda _i: self.render_choices())
        mr.addWidget(self.method, 1)
        self.chk_tool_linked = QCheckBox("Keep colours linked")
        self.chk_tool_linked.setChecked(True)
        self.chk_tool_linked.setToolTip("Linked keeps the real colour balance; unlinked stretches each colour "
                                        "separately (removes casts, can shift colours)")
        self.chk_tool_linked.toggled.connect(lambda _on: self.render_choices())
        mr.addWidget(self.chk_tool_linked)
        tl.addWidget(self.method_row)
        self.choices = QWidget()
        self.grid = QGridLayout(self.choices)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.choices)
        tl.addWidget(sc, 1)
        self.choice_note = QLabel()
        self.choice_note.setWordWrap(True)
        self.choice_note.setStyleSheet("color:#9a9aa6;")
        tl.addWidget(self.choice_note)
        self.fine_box = QGroupBox("Fine-tune (optional)")
        self.fine_box.setCheckable(True)
        self.fine_box.setChecked(False)
        self.fine_box.setToolTip("Small adjustments to the version you picked. The big picture updates as you "
                                 "move a slider; nothing is changed until you press Apply.")
        self.fine_box.toggled.connect(self._fine_toggled)
        fl = QVBoxLayout(self.fine_box)
        self.fine_inner = QWidget()
        self.fine_grid = QGridLayout(self.fine_inner)
        self.fine_grid.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.fine_inner)
        self.curve = CurveWidget()
        self.curve.changed.connect(self._curve_dragged)
        fl.addWidget(self.curve)
        rr = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Back to the version you picked")
        self.btn_reset.clicked.connect(self.reset_fine)
        rr.addStretch(1)
        rr.addWidget(self.btn_reset)
        fl.addLayout(rr)
        tl.addWidget(self.fine_box)
        self.preview_hint = QLabel()
        self.preview_hint.setStyleSheet("color:#eba932;")
        self.preview_hint.setWordWrap(True)
        tl.addWidget(self.preview_hint)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(self.run_preview)
        self.preview_job = None
        self.preview_pending = False
        self.sliders = {}
        self.fine_values = {}
        self.spots = []
        self.spot_r = 10.0
        self.spot_items = []
        br = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet("QPushButton { font-weight:bold; padding:6px 18px; }")
        self.btn_apply.clicked.connect(self.apply_choice)
        self.btn_close_tool = QPushButton("Cancel")
        self.btn_close_tool.setToolTip("Close without changing anything - the picture goes back to how it was")
        self.btn_close_tool.clicked.connect(self.cancel_tool)
        br.addStretch(1)
        br.addWidget(self.btn_close_tool)
        br.addWidget(self.btn_apply)
        tl.addLayout(br)
        split.addWidget(self.tool_box)
        split.setSizes([270, 760, 470])
        root.addWidget(split, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        root.addWidget(self.log_view)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.setCentralWidget(central)

    def log(self, t):
        self.log_view.appendPlainText(t)

    # ------------------------------------------------------------ jobs
    def run_job(self, fn, done, busy_text="Working…"):
        if self.synchronous:
            try:
                done(fn())
            except Exception as e:  # noqa: BLE001
                self.job_failed(f"{e}\n\n{traceback.format_exc()[-1500:]}")
            return
        if self.job and self.job.isRunning():
            return
        self.progress.setVisible(True)
        self.progress.setFormat(busy_text)
        self.setEnabled_all(False)
        self.job = Job(fn)
        self.job.ok.connect(lambda r: (self.progress.setVisible(False), self.setEnabled_all(True), done(r)))
        self.job.failed.connect(self.job_failed)
        self.job.start()

    def job_failed(self, msg):
        self.progress.setVisible(False)
        self.setEnabled_all(True)
        self.log("ERROR: " + msg)
        QMessageBox.critical(self, "Finish", msg[:900])

    def setEnabled_all(self, on):
        for w in (self.flow, self.card, self.tool_box, self.btn_pi, self.hist, self.btn_back):
            w.setEnabled(on)

    # ------------------------------------------------------------ loading
    def add_masters_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Master lights of one target", self.settings.value("stack/masters_root", ""),
            "Masters (*.fits *.fit *.fts *.xisf)")
        if files:
            self.load_masters(files)

    def load_masters(self, files):
        files = [str(f) for f in files if "_stars" not in Path(f).stem and "quicklook" not in Path(f).stem]
        import prepare as P
        sets = P.target_sets([P.load_master(f) for f in files]) if len(files) > 1 else None
        if sets and len(sets) > 1:
            names = ", ".join(s[0].target for s in sets)
            QMessageBox.information(self, "Finish", f"These masters belong to {len(sets)} targets ({names}).\n\n"
                                    f"Finishing {sets[0][0].target} now - open the others afterwards.")
            files = [str(m.path) for m in sets[0]]

        def work():
            s = F.load_state(files)
            proj = F.Project.for_sources(s.sources, s.target)
            return s, proj
        self.run_job(work, self._loaded, "Loading masters…")

    def _loaded(self, res):
        s, proj = res
        if proj.steps:
            last = proj.steps[-1]
            if self.synchronous or QMessageBox.question(
                    self, "Finish", f"You worked on {s.target} before ({len(proj.steps) - 1} step(s), last: "
                    f"{last['label']} at {last['time']}).\n\nCarry on where you left off?") == QMessageBox.Yes:
                self.project = proj
                self.state = proj.load(len(proj.steps) - 1)
                self.log(f"Resumed {s.target}: {', '.join(x['label'] for x in proj.steps[1:]) or 'just loaded'}")
                self.after_change()
                return
            proj.truncate(0)
        self.project = proj
        self.state = s
        self.project.save(s, "load", "Masters loaded")
        self.log(f"Loaded {s.target} — {F.KIND_LABEL.get(s.kind, s.kind)}"
                 + (", stars already separated by Prepare" if s.stars_out else ""))
        self.after_change()

    def open_project_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open a saved project folder",
                                             self.settings.value("finish/project_dir", ""))
        if not d:
            return
        proj = F.Project(Path(d))
        if not proj.steps:
            QMessageBox.information(self, "Finish", "That folder doesn't contain a SubZeroAstro project.")
            return
        self.project = proj
        self.state = proj.load(len(proj.steps) - 1)
        self.log(f"Opened project {d}")
        self.after_change()

    def save_project_dialog(self):
        if not self.project:
            return
        d = QFileDialog.getExistingDirectory(self, "Save the project into this folder",
                                             self.settings.value("finish/project_dir", ""))
        if not d:
            return
        self.settings.setValue("finish/project_dir", d)

        def work():
            return self.project.save_as(Path(d))
        self.run_job(work, lambda p: (setattr(self, "project", p),
                                      self.log(f"Project saved to {p.dir} ({cp.human(p.size())})")),
                     "Saving project…")

    # ------------------------------------------------------------ state display
    def after_change(self):
        self.refresh()
        self.refresh_history(select_last=True)
        self.close_tool()

    def refresh(self):
        s = self.state
        self.flow.clear()
        has = s is not None
        for w in (self.btn_pi, self.hist, self.btn_back, self.btn_before, self.chk_linked):
            w.setEnabled(has)
        if not has:
            self.card_text.setText("Add the prepared masters of one target to start. SubZeroAstro will then "
                                   "suggest each next step for your kind of data.")
            self.btn_do.setVisible(False)
            self.btn_skip.setVisible(False)
            self.tool_box.setVisible(False)
            return
        self.title.setText(f"<b>{s.target}</b> — {F.KIND_LABEL.get(s.kind, s.kind)}, "
                           f"{'linear (not stretched yet)' if s.linear else 'stretched'}"
                           + ("  ·  stars separated" if s.stars_out else ""))
        for t, status in F.workflow(s):
            it = QListWidgetItem(f"{STATUS_MARK[status]}  {t.title}")
            it.setData(Qt.UserRole, t.key)
            ok, why = t.applicable(s)
            it.setToolTip(t.tip + ("" if ok or status in ("done", "skipped") else f"\n\nNot available: {why}"))
            if status in ("n/a",):
                it.setForeground(Qt.gray)
            if status == "next":
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.flow.addItem(it)
        nxt = F.next_step(s)
        self.btn_do.setVisible(nxt is not None)
        self.btn_skip.setVisible(nxt is not None)
        if nxt:
            self.card_text.setText(f"<b>Recommended next: {nxt.title}</b><br>{F.why_next(s, nxt)}")
        else:
            self.card_text.setText("<b>Your picture is finished.</b><br>Save it with <i>Save the Picture…</i>, "
                                   "or continue polishing in PixInsight (curves, local contrast, a final noise "
                                   "pass) with <i>Stop Here – Continue in PixInsight</i>.")
        self.kind_label.setText(f"Workflow for {F.KIND_LABEL.get(s.kind, s.kind)} data. Steps adapt to what "
                                "your data needs; you can open any available step yourself.")
        self.chk_linked.setVisible(s.linear and s.is_color)

    def show_current(self):
        if self.state is None:
            return
        small = self.state.small(PREVIEW_SIDE)
        h, w = self.state.shape
        self.disp_factor = max(h, w) / max(small.shape)
        self.view.show_still(F.state_display(small, linked=self.chk_linked.isChecked()))

    def refresh_history(self, select_last=False):
        self.hist.blockSignals(True)
        self.hist.clear()
        if self.project:
            for i, st_ in enumerate(self.project.steps):
                self.hist.addItem(f"{i}. {st_['label']}" if i else "Start (masters loaded)")
            if select_last:
                self.hist.setCurrentIndex(self.hist.count() - 1)
        self.hist.blockSignals(False)
        n = len(self.project.steps) if self.project else 0
        self.btn_undo.setEnabled(n > 1)
        self.btn_start.setEnabled(n > 1)
        self.show_history()

    def show_history(self, *_a, before=False):
        if not self.project:
            return
        i = self.hist.currentIndex()
        last = self.hist.count() - 1
        self.btn_back.setEnabled(0 <= i < last)
        self.btn_before.setEnabled(i >= 1)
        if before:
            i -= 1
        if i == last and not before:
            self.show_current()
            return
        rgb8 = self.project.preview(i) if i >= 0 else None
        if rgb8 is not None:
            self.view.show_still(rgb8)

    def go_back(self):
        i = self.hist.currentIndex()
        if not self.project or i < 0 or i >= len(self.project.steps) - 1:
            return
        later = [x["label"] for x in self.project.steps[i + 1:]]
        if not self.synchronous and QMessageBox.question(
                self, "Go back", f"Undo these steps and carry on from here?\n\n  • " + "\n  • ".join(later)) \
                != QMessageBox.Yes:
            return
        self.project.truncate(i + 1)
        self.state = self.project.load(i)
        self.log("Went back to: " + (self.project.steps[i]["label"]))
        self.after_change()

    # ------------------------------------------------------------ tools
    def flow_clicked(self, item):
        key = item.data(Qt.UserRole)
        t = F.TOOLS[key]
        ok, why = t.applicable(self.state)
        if key in self.state.done:
            QMessageBox.information(self, t.title, "Already done. To redo it, pick the step before it under "
                                    "“Show:” and click Go Back to Here.")
            return
        if not ok:
            QMessageBox.information(self, t.title, why)
            return
        self.open_tool(key)

    def skip_next(self):
        t = F.next_step(self.state)
        if not t:
            return
        s = self.state.copy()
        s.skipped.append(t.key)
        self.commit(s, t.key + "_skip", f"Skipped {t.title}")

    def open_tool(self, key):
        self.close_tool()
        self.tool = F.TOOLS[key]
        self.tool_box.setVisible(True)
        self.tool_title.setText(f"<h3 style='margin:0'>{self.tool.title}</h3>{self.tool.tip}")
        self.method_row.setVisible(key == "stretch")
        self.fine_box.setChecked(False)
        self.fine_box.setVisible(False)
        self.preview_hint.setText("")
        if key == "stretch":
            self.method.blockSignals(True)
            self.method.setCurrentIndex(self.method.findData(self.tool.recommended_method(self.state)))
            self.method.blockSignals(False)
            self.chk_tool_linked.setVisible(self.state.is_color)
        if key == "spotfix":
            self.spots = []
            self.show_current()
            self.view.on_click = self._spot_click
            self.view.on_wheel = self._spot_wheel
        self.render_choices()

    def close_tool(self):
        self.tool = None
        self.variants = []
        self.tool_box.setVisible(False)
        self.view.on_click = None
        self.view.on_wheel = None
        for it in self.spot_items:
            self.view.scene().removeItem(it)
        self.spot_items = []
        self.spots = []
        self.preview_timer.stop()

    def cancel_tool(self):
        """Close the tool without changing anything; the picture goes back to how it was."""
        was = self.tool
        self.close_tool()
        self.show_current()
        if was:
            self.log(f"{was.title}: cancelled - nothing changed")

    def _clear_choices(self):
        for b in self.group.buttons():
            self.group.removeButton(b)
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)

    def render_choices(self):
        t, s = self.tool, self.state
        if t is None or s is None:
            return
        self._clear_choices()
        self.choice_note.setText("")
        if not t.pick:
            v = t.variants(s)
            self.variants = v
            lab = QLabel(f"<i>{v[0].note}</i>")
            self.grid.addWidget(lab, 0, 0)
            self.btn_apply.setText(v[0].label)
            return
        self.btn_apply.setText("Apply")
        method = self.method.currentData() if t.key == "stretch" else None
        linked = self.chk_tool_linked.isChecked() if t.key == "stretch" else True

        def work():
            vs = t.variants(s, method, linked) if t.key == "stretch" else t.variants(s)
            small = F.thumb_state(t, s, 300)
            thumbs = [F.state_display(t.apply(small, v.params)) for v in vs]
            return vs, thumbs
        self.run_job(work, self._choices_ready, "Making the choices…")

    def _choices_ready(self, res):
        vs, thumbs = res
        self.variants = vs
        self._clear_choices()
        cols = 2
        avail = max(240, self.tool_box.width() - 60)
        side = max(110, min(220, avail // cols - 16))
        for i, (v, th) in enumerate(zip(vs, thumbs)):
            b = QToolButton()
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setIcon(QIcon(to_pixmap(th)))
            b.setIconSize(QSize(side, int(side * th.shape[0] / max(1, th.shape[1]))))
            b.setText(v.label + ("  ★" if v.recommended else ""))
            b.setToolTip(v.note or v.label)
            b.setStyleSheet("QToolButton:checked { border:3px solid #4a86e8; border-radius:4px; }")
            self.group.addButton(b, i)
            self.grid.addWidget(b, i // cols, i % cols)
            if v.recommended:
                b.setChecked(True)
        if not any(b.isChecked() for b in self.group.buttons()) and self.group.buttons():
            self.group.buttons()[0].setChecked(True)
        try:
            self.group.idClicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.group.idClicked.connect(self.on_choice)
        crop = " (shown at full size on a small area, so you see the real effect)" if self.tool.spatial else ""
        self.choice_note.setText("Click a version to see it big (★ = suggested). Nothing changes until you press "
                                 "Apply; Cancel leaves the picture as it was." + crop)
        self.on_choice(max(0, self.group.checkedId()))

    # ------------------------------------------------------------ preview & fine-tune
    def selected(self):
        if not self.variants:
            return None
        return self.variants[max(0, self.group.checkedId()) if self.tool and self.tool.pick else 0]

    def on_choice(self, i):
        """A version was picked: show it big and set the fine-tune controls to it."""
        self.fine_values = {}
        v = self.selected()
        if v is None:
            return
        self.build_sliders(v.params)
        self.schedule_preview()

    def build_sliders(self, params):
        while self.fine_grid.count():
            w = self.fine_grid.takeAt(0).widget()
            if w:
                w.setParent(None)
        self.sliders = {}
        t = self.tool
        specs = t.sliders(self.state, params) if t else []
        self.fine_box.setVisible(bool(specs) or (t is not None and t.editor == "curve"))
        for row, sp in enumerate(specs):
            lab = QLabel(sp.label)
            lab.setToolTip(sp.tip)
            sl = QSlider(Qt.Horizontal)
            n = max(1, int(round((sp.hi - sp.lo) / sp.step)))
            sl.setRange(0, n)
            sl.setValue(int(round((sp.value - sp.lo) / sp.step)))
            sl.setToolTip(sp.tip)
            val = QLabel(self._fmt(sp, sp.value))
            val.setMinimumWidth(42)
            sl.valueChanged.connect(lambda k, sp=sp, val=val: self._slider_moved(sp, k, val))
            self.fine_grid.addWidget(lab, row, 0)
            self.fine_grid.addWidget(sl, row, 1)
            self.fine_grid.addWidget(val, row, 2)
            self.sliders[sp.name] = (sl, sp)
        self.curve.setVisible(t is not None and t.editor == "curve")
        if t is not None and t.editor == "curve":
            self.curve.set_points(params.get("points", self.curve.points))

    @staticmethod
    def _fmt(sp, v):
        if sp.step >= 1:
            return f"{v:.0f}"
        return f"{v:.2f}"

    def _slider_moved(self, sp, k, val):
        v = sp.lo + k * sp.step
        val.setText(self._fmt(sp, v))
        self.fine_values[sp.name] = v
        self.fine_values.pop("points", None)            # sliders re-shape the curve
        if self.tool and self.tool.editor == "curve":
            self.curve.set_points(self.current_params().get("points", self.curve.points))
        if not self.fine_box.isChecked():
            self.fine_box.setChecked(True)
        self.schedule_preview()

    def _curve_dragged(self, pts):
        self.fine_values["points"] = pts
        if not self.fine_box.isChecked():
            self.fine_box.setChecked(True)
        self.schedule_preview()

    def _fine_toggled(self, on):
        self.fine_inner.setEnabled(True)
        self.schedule_preview()

    def reset_fine(self):
        v = self.selected()
        self.fine_values = {}
        if v is not None:
            self.build_sliders(v.params)
        self.schedule_preview()

    def current_params(self) -> dict:
        """The params Apply would use: the picked version, plus fine-tuning if switched on."""
        t, v = self.tool, self.selected()
        if t is None or v is None:
            return {}
        if t.key == "spotfix":
            return {"spots": [list(q) for q in self.spots]}
        if not self.fine_box.isChecked() or not self.fine_values:
            return v.params
        vals = {name: sp.lo + sl.value() * sp.step for name, (sl, sp) in self.sliders.items()}
        if "points" in self.fine_values:
            vals["points"] = self.fine_values["points"]
        return t.tuned(self.state, v.params, vals)

    def schedule_preview(self):
        if self.tool is None:
            return
        if self.synchronous:
            self.run_preview()
        else:
            self.preview_timer.start()

    def run_preview(self):
        t, s = self.tool, self.state
        if t is None or s is None:
            return
        if self.preview_job and self.preview_job.isRunning():
            self.preview_pending = True
            return
        params = self.current_params()
        tuned = self.fine_box.isChecked() and bool(self.fine_values)

        def work():
            if t.spatial:
                ps = s.crop(900, F.busiest_spot(s, 900))
            else:
                ps = s.small(PREVIEW_SIDE)
            if t.key == "spotfix":
                f = max(s.shape) / max(ps.shape)
                return F.state_display(t.apply(ps, {"spots": [[x / f, y / f, r / f] for x, y, r in params["spots"]]}))
            return t.preview(ps, params)

        def done(img):
            self.view.show_still(img)
            for it in self.spot_items:                   # keep the spot circles on top
                if it.scene() is None:
                    self.view.scene().addItem(it)
            v = self.selected()
            name = "your marked spots" if t.key == "spotfix" else f"“{v.label}”" + (" (fine-tuned)" if tuned else "")
            where = " — full-size view of a small area" if t.spatial else ""
            self.preview_hint.setText(f"Preview of {name}{where} — not applied yet. Apply to keep it, "
                                      "Cancel to go back.")
            if self.preview_pending:
                self.preview_pending = False
                self.schedule_preview()
        if self.synchronous:
            done(work())
            return
        self.preview_job = Job(work)
        self.preview_job.ok.connect(done)
        self.preview_job.failed.connect(lambda m: self.log("Preview failed: " + m[:300]))
        self.preview_job.start()

    # ------------------------------------------------------------ Spot Fix clicks
    def _spot_click(self, x, y, button):
        f = getattr(self, "disp_factor", 1.0)
        if button == Qt.RightButton:
            if self.spots:
                self.spots.pop()
                it = self.spot_items.pop()
                self.view.scene().removeItem(it)
        else:
            self.spots.append([x * f, y * f, self.spot_r])
            r = self.spot_r / f
            it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            it.setPen(QPen(QColor(255, 200, 60), 0))
            self.view.scene().addItem(it)
            self.spot_items.append(it)
        self.choice_note.setText(f"{len(self.spots)} spot(s) marked. Circle size {self.spot_r:.0f} px (mouse wheel). "
                                 "Right-click removes the last one.")
        self.schedule_preview()

    def _spot_wheel(self, steps):
        self.spot_r = float(min(80, max(3, self.spot_r * (1.15 ** steps))))
        self.choice_note.setText(f"Circle size {self.spot_r:.0f} px - click a blemish to fix it.")

    def apply_choice(self):
        t, s = self.tool, self.state
        if t is None or not self.variants:
            return
        v = self.selected()
        params = self.current_params()
        if t.key == "spotfix" and not params.get("spots"):
            QMessageBox.information(self, t.title, "Click the blemishes on the image first.")
            return
        tuned = (self.fine_box.isChecked() and bool(self.fine_values)) or t.key == "spotfix"
        label = (f"{len(params['spots'])} spot(s)" if t.key == "spotfix"
                 else v.label + (" (fine-tuned)" if tuned else ""))

        def work():
            t0 = time.time()
            new = t.apply(s, params)
            return new, time.time() - t0
        self.run_job(work, lambda r: (self.log(f"{t.title}: {label}  ({r[1]:.1f} s)"),
                                      self.commit(r[0], t.key, f"{t.title} — {label}", params)),
                     f"{t.title}…")

    # ------------------------------------------------------------ undo
    def undo_last(self):
        if not self.project or len(self.project.steps) <= 1 or (self.job and self.job.isRunning()):
            return
        last = self.project.steps[-1]["label"]
        n = len(self.project.steps) - 1
        self.project.truncate(n)
        self.state = self.project.load(n - 1)
        self.log(f"Undid: {last}")
        self.after_change()

    def back_to_start(self):
        if not self.project or len(self.project.steps) <= 1:
            return
        if not self.synchronous and QMessageBox.question(
                self, "Back to Start", "Go back to the original masters? All steps will be undone.") \
                != QMessageBox.Yes:
            return
        self.project.truncate(1)
        self.state = self.project.load(0)
        self.log("Back to the start - all steps undone")
        self.after_change()

    def commit(self, new_state, key, label, params=None):
        self.state = new_state

        def work():
            self.project.save(new_state, key, label, params)
            return True
        self.run_job(work, lambda _r: self.after_change(), "Saving the step…")

    # ------------------------------------------------------------ output
    def default_out_dir(self) -> Path:
        src = Path(self.state.sources[0]) if self.state and self.state.sources else Path.home()
        return src.parent / "Finished"

    def export_dialog(self):
        if not self.state or self.state.rgb is None:
            QMessageBox.information(self, "Finish", "Combine the colour channels first.")
            return
        dlg = ExportDialog(self, self.settings, self.default_out_dir())
        if not dlg.exec():
            return
        folder, fmts = Path(dlg.dir.text().strip()), dlg.formats()
        name = f"{cp.safe_name(self.state.target)}_{'finished' if not self.state.linear else 'linear'}"

        def work():
            return F.export(self.state, folder / name, fmts, save_rgb8=save_rgb8)
        self.run_job(work, lambda files: (self.log("Saved " + ", ".join(str(f) for f in files)),
                                          tools.reveal(files)), "Saving the picture…")

    def stop_here_pixinsight(self):
        if not self.state:
            return
        folder = Path(self.settings.value("finish/pi_dir", "") or (self.default_out_dir().parent / "PixInsight"))

        def work():
            return F.export_for_pixinsight(self.state, folder)

        def done(files):
            self.last_pi_files = files
            note = files[0].read_text(encoding="utf-8")
            self.log(f"Saved for PixInsight in {folder}")
            tools.reveal(files[1] if len(files) > 1 else files[0])
            if not self.synchronous:
                box = QMessageBox(self)
                box.setWindowTitle("Continue in PixInsight")
                box.setText(f"<b>Saved for PixInsight</b> in<br>{folder}<br><br>Open the .xisf file(s) in "
                            "PixInsight (File › Open, or drag them in). The folder is open for you.")
                box.setDetailedText(note)
                box.exec()
        self.run_job(work, done, "Saving for PixInsight…")

    def closeEvent(self, ev):
        if self.job and self.job.isRunning():
            self.job.wait(60000)
        super().closeEvent(ev)
