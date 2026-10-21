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
                               QFileDialog, QFormLayout, QFrame, QGraphicsEllipseItem, QGraphicsRectItem, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QSplitter, QToolButton, QVBoxLayout, QWidget)

import checkpoints as cp
import finish as F
import tools
from prepare_window import CropView, save_rgb8

SIGN_DRAGGED = "Where I dragged it"
LAST_EXPORT_DIR = "finish/export_dir_last"     # the folder the last picture was saved to
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


def changed_share(a: np.ndarray, b: np.ndarray, levels: int = 3) -> float:
    """Share of pixels whose displayed colour differs by more than a few levels (0..1)."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    return float(np.mean(d > levels))


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
        main = next((w for w in written if "star" not in Path(w).stem.lower()), str(work / "out.fits"))
        starless = P._read_result(main, like) / 65535.0
        st_f = [w for w in written if "star" in Path(w).stem.lower()]
        stars = (P._read_result(st_f[0], like) / 65535.0) if st_f else None
        # rc-astro may hand the picture back upside down: compare with what was sent and turn it back
        how = P.orientation_of(starless if stars is None else starless + stars, img)
        if how != "same":
            starless, stars = P.turn_back(how, starless, stars)
        if stars is None:
            stars = np.clip(img - starless, 0, None)
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        return np.clip(starless, 0, None), np.clip(stars, 0, None)
    return run


def make_noise_runner(settings):
    """callable(img (C,h,w) 0..1) -> NoiseXTerminator at full strength, or None if NXT isn't available."""
    path = settings.value("prep/rcastro", "") or tools.find_rcastro()
    if not path:
        return None
    try:
        rc = tools.RCAstro(path)
        prods = rc.products()
        nxt = next((p for p in prods.get("products", []) if p.get("key") == "nxt"), None)
        if nxt is None or not (nxt.get("license") or {}).get("valid", True):
            return None
    except Exception:  # noqa: BLE001
        return None
    device = settings.value("prep/device", "") or tools.preferred_device(prods.get("devices", []))

    def run(img):
        import fits_core as core
        import prepare as P
        spec = rc.spec("nxt")
        vals = {}
        if spec.param("dn"):
            vals["dn"] = float(spec.param("dn").get("max", 1.0))
        args = spec.args(vals)
        work = Path(tempfile.mkdtemp(prefix="sznxt_"))
        src = work / "in.fits"
        core.write_fits(src, np.clip(img[0] if img.shape[0] == 1 else img, 0, 1).astype(np.float32),
                        [("IMAGETYP", "Master Light", ""), ("PIXRANGE", 65535, "values x this = ADU (0..1 range)")],
                        bitpix=-32)
        written = rc.run("nxt", src, work / "out.fits", args, device=device)
        like = img * 65535.0
        main = next((w for w in written if "star" not in Path(w).stem.lower()), str(work / "out.fits"))
        out = P._read_result(main, like) / 65535.0
        how = P.orientation_of(out, img)          # rc-astro may hand it back upside down
        if how != "same":
            out = P.turn_back(how, out, None)[0]
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        return np.clip(out, 0, 1).astype(np.float32)
    return run


# --------------------------------------------------------------------------
# Export dialog
# --------------------------------------------------------------------------

class ExportDialog(QDialog):
    def __init__(self, parent, settings, default_dir, target=""):
        super().__init__(parent)
        self.setWindowTitle("Save the Picture")
        self.key = f"finish/export_dir/{cp.safe_name(target)}" if target else "finish/export_dir"
        lay = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        # each target remembers its own folder; a new target starts in the folder used last time
        # (only the very first picture starts next to its masters)
        self.dir = QLineEdit(settings.value(self.key, "") or settings.value(LAST_EXPORT_DIR, "") or str(default_dir))
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
        self.sub_note = QLabel()
        self.sub_note.setStyleSheet("color:#9a9aa6;")
        self.sub_note.setWordWrap(True)
        form.addRow("", self.sub_note)
        self.chk_about = QCheckBox("Also save a short description of the target (from Wikipedia)")
        self.chk_about.setToolTip("The opening paragraphs of the target's Wikipedia article, saved as a text file "
                                  "next to the pictures. You see it (and can edit it) before it is saved. "
                                  "Needs an internet connection.")
        self.chk_about.setChecked(settings.value("finish/save_about", False, type=bool))
        form.addRow("", self.chk_about)
        self.chk_credit = QCheckBox("Add a small “Processed with SubZeroAstro” credit (bottom-left corner)")
        self.chk_credit.setToolTip("A small, light, partly see-through line in the bottom-left corner of the "
                                   "TIFF / JPG / PNG pictures (bottom-right if your signature is there). "
                                   "XISF and FITS files are never marked. Thank you for spreading the word!")
        self.chk_credit.setChecked(settings.value("finish/credit", True, type=bool))
        form.addRow("", self.chk_credit)
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
        self.settings.setValue(self.key, self.dir.text().strip())
        self.settings.setValue(LAST_EXPORT_DIR, self.dir.text().strip())
        for k, cb in self.boxes.items():
            self.settings.setValue(f"finish/fmt_{k}", cb.isChecked())
        self.settings.setValue("finish/save_about", self.chk_about.isChecked())
        self.settings.setValue("finish/credit", self.chk_credit.isChecked())
        self.settings.sync()                     # written now, not only when the app quits
        super().accept()


class DescriptionDialog(QDialog):
    """A short description of the target from Wikipedia: looked up, shown (editable), then saved as
    <target>_about.txt next to the pictures."""

    def __init__(self, parent, target, folder):
        super().__init__(parent)
        self.target, self.folder = target, Path(folder)
        self.saved = None
        self.setWindowTitle("Target Description")
        self.setMinimumSize(640, 460)
        lay = QVBoxLayout(self)
        self.status = QLabel("Looking up the target on Wikipedia…")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("The description appears here - or type your own.")
        lay.addWidget(self.text, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(f"<small>Saved as <i>{cp.safe_name(target) or 'target'}_about.txt</i> in "
                             f"{self.folder}</small>"), 1)
        b_skip = QPushButton("Don't Save")
        b_skip.clicked.connect(self.reject)
        self.b_save = QPushButton("Save Description")
        self.b_save.setStyleSheet(__import__("ui_header").RUN_STYLE)
        self.b_save.clicked.connect(self.save)
        row.addWidget(b_skip)
        row.addWidget(self.b_save)
        lay.addLayout(row)
        self.job = None

    def start_lookup(self, sync=False):
        import target_info as TI
        t = self.target

        def work():
            return TI.lookup_all(t)
        if sync:
            try:
                self._found(work())
            except Exception as e:  # noqa: BLE001
                self._failed(str(e))
            return
        self.job = Job(work)
        self.job.ok.connect(self._found)
        self.job.failed.connect(lambda m: self._failed(m.split("\n")[0]))
        self.job.start()

    def _found(self, res):
        import html
        import target_info as TI
        wiki, facts, problems = res
        if wiki is None and facts is None:
            why = ("; ".join(problems) if problems else
                   f"neither Wikipedia nor SIMBAD knows {self.target!r} by that name")
            self.status.setText(f"No description found: {html.escape(why)}. Type your own description if you like.")
            par = self.parent()
            if problems and par is not None and hasattr(par, "log"):
                par.log("Target description: " + "; ".join(problems))
            return
        title, text, url = wiki if wiki else ("", "", "")
        got = [x for x, ok in (("Wikipedia" + (f" (<b>{html.escape(title)}</b>)" if title else ""), wiki),
                               ("SIMBAD facts", facts)) if ok]
        note = " and ".join(got)
        if problems:
            note += " - " + html.escape("; ".join(problems))
        self.status.setText(f"From {note}. Edit it if you like, then save. (The sources are credited at the end.)")
        self.text.setPlainText(TI.note_text(self.target, title, text, url, facts))

    def _failed(self, msg):
        import html
        import target_info as TI
        why = TI.explain(msg)
        self.status.setText("Could not get the description from Wikipedia: " + html.escape(why) + ". Type your own "
                            "description if you like.<br><small>" + html.escape(msg[:240]) + "</small>")
        par = self.parent()
        if par is not None and hasattr(par, "log"):
            par.log(f"Target description: could not reach Wikipedia - {msg[:300]}")

    def save(self):
        import target_info as TI
        content = self.text.toPlainText().strip()
        if not content:
            self.reject()
            return
        try:
            self.saved = TI.save(self.folder, self.target, content + "\n")
        except OSError as e:
            QMessageBox.warning(self, "Target Description", f"Could not save the description:\n{e}")
            return
        self.accept()

    def done(self, r):
        if self.job is not None and self.job.isRunning():
            self.job.wait(15000)
        super().done(r)


class AstroBinDialog(QDialog):
    """Share to AstroBin: the picture plus an acquisition CSV for AstroBin's upload page."""

    def __init__(self, parent, settings, sessions, default_dir):
        super().__init__(parent)
        import astrobin as AB
        self.settings, self.sessions = settings, sessions
        self.setWindowTitle("Share to AstroBin")
        self.setMinimumWidth(640)
        lay = QVBoxLayout(self)
        intro = QLabel("SubZeroAstro saves your picture and an <b>acquisition file</b> with every night's subs, "
                       "exposure, gain, temperature and FWHM, then opens AstroBin's upload page.<br>"
                       "On AstroBin: upload the picture, and on the <i>Acquisition</i> step click "
                       "<b>Import CSV</b> and choose the .csv file.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        tbl = QPlainTextEdit()
        tbl.setReadOnly(True)
        tbl.setMaximumHeight(150)
        lines = [f"{'Night':<12}{'Filter':<14}{'Subs':>6}{'Exp s':>8}{'Gain':>6}{'Temp':>6}" + f"{'FWHM(as)':>9}"]
        for x in sessions:
            lines.append(f"{x.date:<12}{x.filter[:13]:<14}{x.number:>6}{AB._fmt(x.duration):>8}"
                         f"{AB._fmt(x.gain):>6}{AB._fmt(x.cooling, 0):>6}{AB._fmt(x.fwhm):>9}")
        lines += ["", AB.summary(sessions)]
        tbl.setPlainText("\n".join(lines))
        tbl.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(tbl)
        form = QFormLayout()
        # AstroBin wants its own filter ID numbers; remembered per filter name
        self.fid = {}
        for filt in sorted({x.filter for x in sessions}):
            row = QHBoxLayout()
            e = QLineEdit(settings.value(f"astrobin/filter_id/{filt}", "") or "")
            e.setPlaceholderText("optional - AstroBin filter number")
            e.setToolTip("AstroBin identifies filters by a number: open your filter's page in AstroBin's "
                         "equipment database and copy the number from the page address. Remembered "
                         "for next time. Leave blank to choose the filter on AstroBin instead.")
            row.addWidget(e, 1)
            b = QPushButton("Find…")
            b.setToolTip("Open AstroBin's filter database in your browser")
            b.clicked.connect(lambda _=False: self._open(AB.FILTER_SEARCH_URL))
            row.addWidget(b)
            self.fid[filt] = e
            form.addRow(f"{filt} filter ID:", row)
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
        self.bortle = QSpinBox()
        self.bortle.setRange(0, 9)
        self.bortle.setSpecialValueText("(not set)")
        self.bortle.setValue(int(settings.value("astrobin/bortle", 0) or 0))
        form.addRow("Bortle class:", self.bortle)
        self.sqm = QDoubleSpinBox()
        self.sqm.setRange(0, 23)
        self.sqm.setDecimals(2)
        self.sqm.setSpecialValueText("(not set)")
        self.sqm.setValue(float(settings.value("astrobin/sqm", 0) or 0))
        form.addRow("Sky quality (SQM):", self.sqm)
        self.chk_cal = QCheckBox("Include calibration frame counts (darks, flats, dark-flats, bias)")
        self.chk_cal.setChecked(settings.value("astrobin/cal", True, type=bool))
        form.addRow("", self.chk_cal)
        self.chk_tif = QCheckBox("Also save a 16-bit TIFF (JPG is always saved)")
        self.chk_tif.setToolTip("AstroBin accepts JPG, PNG and TIFF. The JPG is smaller and uploads quickly; "
                                "the TIFF keeps full quality but may exceed your AstroBin plan's file size limit.")
        self.chk_tif.setChecked(settings.value("astrobin/tif", False, type=bool))
        form.addRow("", self.chk_tif)
        row = QHBoxLayout()
        self.dir = QLineEdit(str(default_dir))
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(self.dir, 1)
        row.addWidget(b)
        form.addRow("Folder:", row)
        self.chk_open = QCheckBox("Open AstroBin's upload page when done")
        self.chk_open.setChecked(settings.value("astrobin/open", True, type=bool))
        form.addRow("", self.chk_open)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Save for AstroBin")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _open(url):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Save the AstroBin files to", self.dir.text())
        if d:
            self.dir.setText(d)

    def filter_ids(self):
        return {f: e.text().strip() for f, e in self.fid.items()}

    def accept(self):
        for f, v in self.filter_ids().items():
            self.settings.setValue(f"astrobin/filter_id/{f}", v)
        self.settings.setValue("astrobin/bortle", self.bortle.value())
        self.settings.setValue("astrobin/sqm", self.sqm.value())
        self.settings.setValue("astrobin/cal", self.chk_cal.isChecked())
        self.settings.setValue("astrobin/tif", self.chk_tif.isChecked())
        self.settings.setValue("astrobin/open", self.chk_open.isChecked())
        super().accept()


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

class FinishWindow(QMainWindow):
    def __init__(self, parent, settings, master_files=None):
        super().__init__(parent)
        self.setWindowTitle("SubZeroAstro — Finishing Touches")
        self.settings = settings
        self.state: F.State | None = None
        self.project: F.Project | None = None
        self.tool: F.Tool | None = None
        self.variants = []
        self.job = None
        self.synchronous = False           # self-test: run jobs inline
        F.TOOLS["stars_out"].runner = make_star_runner(settings)
        F.TOOLS["denoise"].runner = make_noise_runner(settings)
        import ui_header
        ui_header.install_header(self, ui_header.PAGE_FINISH, with_menu=False)
        self.picture_saved = False
        self.btn_new_project = QPushButton("↺  End this Project and Start New Project")
        self.btn_new_project.setStyleSheet(ui_header.NEW_PROJECT_STYLE)
        self.btn_new_project.setCursor(Qt.PointingHandCursor)
        self.btn_new_project.setToolTip("Close every page and image and begin a new target from the start: "
                                        "Quality and Cull, then stacking. Nothing on disk is deleted.")
        self.btn_new_project.clicked.connect(lambda: self.end_project())
        self.app_header.right.addWidget(self.btn_new_project)
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
        self.title = QLabel("<b>Finishing Touches</b> — add your processed masters to start")
        top.addWidget(self.title, 1)
        self._top_widgets = [self.title]
        for text, slot, tip in (
                ("Add Masters…", self.add_masters_dialog, "Open the master lights of one target (prepared ones "
                                                          "are best)"),
                ("Open Project…", self.open_project_dialog, "Carry on with a project you saved earlier"),
                ("Save Project As…", self.save_project_dialog, "Keep this project (every step) in a folder of "
                                                                 "your choice"),
                ("Save the Picture…", self.export_dialog, "Save the image as it looks now (TIFF, JPG, PNG, "
                                                          "XISF, FITS)"),
                ("Share to AstroBin…", self.share_astrobin, "Save the picture and an acquisition file that "
                                                            "AstroBin's upload page imports (every night's subs, "
                                                            "exposure, gain, temperature, FWHM), then open "
                                                            "AstroBin's upload page")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            top.addWidget(b)
            self._top_widgets.append(b)
        self.btn_pi = QPushButton("  Stop Here – Continue in PixInsight  ")
        self.btn_pi.setToolTip("Save the image exactly as it is now (XISF) with a note on what to do next in "
                               "PixInsight, and open the folder")
        self.btn_pi.setStyleSheet("QPushButton { background:#6d4bb8; color:white; font-weight:bold; "
                                  "padding:5px 12px; border-radius:4px; }")
        self.btn_pi.clicked.connect(self.stop_here_pixinsight)
        top.addWidget(self.btn_pi)
        self._top_widgets.append(self.btn_pi)
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
        self._left_panel = left

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
        self.btn_again = QPushButton("↻ Do It Again")
        self.btn_again.clicked.connect(self.do_again)
        self.btn_again.setVisible(False)
        cl.addWidget(self.btn_again)
        cl.addWidget(self.btn_do)
        cl.addWidget(self.btn_skip)
        ml.addWidget(self.card)
        self.view = CropView()
        self.view.message = "Add the prepared masters of one target to start"
        self.view.zoomable = True                 # wheel zooms, drag pans, double-click fits
        self.view.on_zoom_detail = self._zoom_detail
        self.view.setToolTip("Scroll to zoom in, drag to move around, double-click to fit the window")
        self._detail_job = None
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
        self.btn_full = QPushButton("⛶  Full Window")
        self.btn_full.setCheckable(True)
        self.btn_full.setToolTip("Show the picture as big as the window (the step list, panels and log are "
                                 "hidden). Press again or Esc to go back.")
        self.btn_full.toggled.connect(self.set_full_window)
        hr.addWidget(self.btn_full)
        self.full_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.full_esc.activated.connect(lambda: self.btn_full.setChecked(False))
        self._full_saved = None
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
        self.chk_mask = QCheckBox("Show what will change")
        self.chk_mask.setToolTip("Tints the area this tool works on (the mask SubZeroAstro made for you) in red")
        self.chk_mask.toggled.connect(lambda _on: self.schedule_preview())
        self.chk_mask.setVisible(False)
        tl.addWidget(self.chk_mask)
        self.choices = QWidget()
        self.grid = QGridLayout(self.choices)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        # Sign Your Picture: the words, corner and colour (fonts are the picture choices)
        self.sign_row = QWidget()
        sg = QFormLayout(self.sign_row)
        sg.setContentsMargins(0, 0, 0, 0)
        self.sign_target = QLineEdit()
        self.sign_target.setPlaceholderText('e.g. NGC 281 "Pacman Nebula" - leave empty for none')
        self.sign_target.setToolTip("A line above your signature naming the object - remembered for each target")
        sg.addRow("Target name:", self.sign_target)
        self.sign_text = QLineEdit(self.settings.value("finish/signature", "") or "")
        self.sign_text.setPlaceholderText("Type your name as you want it to appear")
        self.sign_text.setToolTip("What to write - remembered for next time")
        sg.addRow("Signature:", self.sign_text)
        self.sign_corner = QComboBox()
        self.sign_corner.addItems([c.capitalize() for c in F.SIGN_CORNERS] + [SIGN_DRAGGED])
        corner0 = str(self.settings.value("finish/sign_corner", "Bottom right"))
        self.sign_corner.setCurrentText(corner0 if corner0 != SIGN_DRAGGED else "Bottom right")
        self.sign_corner.setToolTip("Pick a corner - or drag the signature on the picture to put it anywhere")
        sg.addRow("Place:", self.sign_corner)
        self.sign_pos = None                    # (fx, fy) top-left as fractions, once dragged
        self.sign_rect = None                   # where the signature is on the preview (x, y, w, h)
        self.sign_box_item = None
        self.sign_colour = QComboBox()
        self.sign_colour.addItems(list(F.SIGN_COLOURS))
        self.sign_colour.setCurrentText(str(self.settings.value("finish/sign_colour", "Soft white")))
        sg.addRow("Colour:", self.sign_colour)
        self.sign_row.setVisible(False)
        tl.addWidget(self.sign_row)
        # Final Crop: buttons for the box drawn on the picture
        self.crop_row = QWidget()
        cr = QHBoxLayout(self.crop_row)
        cr.setContentsMargins(0, 0, 0, 0)
        b_found = QPushButton("Found Edges")
        b_found.setToolTip("Put the box back where SubZeroAstro found ragged or off-colour edges")
        b_found.clicked.connect(lambda: self._set_crop_box("found"))
        b_all = QPushButton("Whole Picture")
        b_all.setToolTip("Box around the whole picture (then drag its sides)")
        b_all.clicked.connect(lambda: self._set_crop_box("all"))
        cr.addWidget(QLabel("Drag the yellow box or its sides:"))
        cr.addStretch(1)
        cr.addWidget(b_found)
        cr.addWidget(b_all)
        self.crop_row.setVisible(False)
        tl.addWidget(self.crop_row)
        # Flip the Picture: two toggles; each press flips the preview (again = back)
        self.flip_row = QWidget()
        fr_ = QHBoxLayout(self.flip_row)
        fr_.setContentsMargins(0, 0, 0, 0)
        self.btn_flip_h = QPushButton("⇆  Flip Horizontally")
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setToolTip("Mirror left ↔ right. Press again to flip it back.")
        self.btn_flip_v = QPushButton("⇅  Flip Vertically")
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setToolTip("Upside down. Press again to flip it back.")
        for b in (self.btn_flip_h, self.btn_flip_v):
            b.setStyleSheet("QPushButton { padding:6px 14px; } QPushButton:checked { background:#2f5d9e; "
                            "color:white; }")
            b.toggled.connect(lambda _on: self.schedule_preview())
            fr_.addWidget(b)
        self.flip_rot = 0                 # quarter turns clockwise
        self.btn_rot_cw = QPushButton("↻  Rotate 90° Clockwise")
        self.btn_rot_cw.setToolTip("Turn the picture a quarter turn clockwise. Press again to keep turning.")
        self.btn_rot_ccw = QPushButton("↺  Rotate 90° Counter-clockwise")
        self.btn_rot_ccw.setToolTip("Turn the picture a quarter turn counter-clockwise. Press again to keep turning.")
        self.btn_rot_cw.clicked.connect(lambda: self._rotate(1))
        self.btn_rot_ccw.clicked.connect(lambda: self._rotate(-1))
        for b in (self.btn_rot_cw, self.btn_rot_ccw):
            b.setStyleSheet("QPushButton { padding:6px 14px; }")
            fr_.addWidget(b)
        self.rot_label = QLabel("")
        fr_.addWidget(self.rot_label)
        fr_.addStretch(1)
        self.flip_row.setVisible(False)
        tl.addWidget(self.flip_row)
        self.spot_wheel_timer = QTimer(self)
        self.spot_wheel_timer.setSingleShot(True)
        self.spot_wheel_timer.setInterval(350)
        self.spot_wheel_timer.timeout.connect(self.schedule_preview)
        self.sign_timer = QTimer(self)
        self.sign_timer.setSingleShot(True)
        self.sign_timer.setInterval(450)
        self.sign_timer.timeout.connect(self._sign_changed)
        self.sign_text.textChanged.connect(lambda _t: self.sign_timer.start())
        self.sign_target.textChanged.connect(lambda _t: self.sign_timer.start())
        self.sign_corner.currentTextChanged.connect(self._sign_corner_picked)
        self.sign_colour.currentTextChanged.connect(lambda _t: self.sign_timer.start())
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
        self._before_cache = {}
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
        self.btn_done_tool = QPushButton("Done – Next Step  ➜")
        self.btn_done_tool.setToolTip("Finished with this tool - the fixes so far are kept; go on to the next step")
        self.btn_done_tool.clicked.connect(self.done_tool)
        self.btn_done_tool.setVisible(False)
        br.addStretch(1)
        br.addWidget(self.btn_close_tool)
        br.addWidget(self.btn_apply)
        br.addWidget(self.btn_done_tool)
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
        head, _, tb = msg.partition("\n\nTraceback")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Finishing Touches")
        box.setText(head.strip()[:600] or "Something went wrong.")
        if tb:
            box.setInformativeText("The technical details are in the log at the bottom of the page.")
            box.setDetailedText(("Traceback" + tb)[-1500:])
        box.exec()

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
                 + (f" ({len(s.channels)} filters: {', '.join(sorted(s.channels))})" if s.channels else "")
                 + (" — Prepare already separated the stars; 'Take Stars Out' hides them instantly"
                    if s.stars_hidden else ""))
        if getattr(s, "load_note", ""):
            self.log(s.load_note)
        if getattr(s, "stars_missing", False):
            self.log("This master is already starless, but its stars-only file (…_stars.xisf) was not found "
                     "next to it — the picture will be finished without stars. Re-run Prepare to get both.")
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
        typ = F.target_type(s)
        self.title.setText(f"<b>{s.target}</b> — " + (f"{typ}, " if typ else "") + f"{F.KIND_LABEL.get(s.kind, s.kind)}, "
                           f"{'linear (not stretched yet)' if s.linear else 'stretched'}"
                           + ("  ·  stars out (they go back at 'Put Stars Back')"
                              if s.stars_out and not s.stars_hidden else ""))
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
        again = self.again_tool()
        self.btn_again.setVisible(again is not None)
        again_note = ""
        if again is not None:
            self.btn_again.setToolTip(f"Run {again.title} once more, on the picture as it is now "
                                      "(Undo Last Step takes it back)")
            again_note = (f"<br><i>Not quite enough? <b>↻ Do It Again</b> runs {again.title} once more on "
                          "this picture.</i>")
        if nxt:
            self.card_text.setText(f"<b>Recommended next: {nxt.title}</b><br>{F.why_next(s, nxt)}" + again_note)
        else:
            self.card_text.setText("<b>Your picture is finished.</b><br>Save it with <i>Save the Picture…</i>, "
                                   "or continue polishing in PixInsight (curves, local contrast, a final noise "
                                   "pass) with <i>Stop Here – Continue in PixInsight</i>. Want to sign it? Pick <i>Sign Your "
                                   "Picture</i> in the list on the left.")
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
        if key in self.state.done and not getattr(t, "repeatable", False):
            QMessageBox.information(self, t.title, "Already done. To redo it, pick the step before it under "
                                    "“Show:” and click Go Back to Here.")
            return
        if not ok:
            QMessageBox.information(self, t.title, why)
            return
        self.open_tool(key)

    def again_tool(self):
        """The tool of the last step, when it can be run again on the picture as it is now."""
        if not self.project or len(self.project.steps) <= 1 or self.state is None:
            return None
        t = F.TOOLS.get(self.project.steps[-1].get("tool", ""))
        if t is None or not getattr(t, "repeatable", False) or not t.applicable(self.state)[0]:
            return None
        return t

    def do_again(self):
        t = self.again_tool()
        if t is not None:
            self.open_tool(t.key)

    def skip_next(self):
        t = F.next_step(self.state)
        if not t:
            return
        s = self.state.copy()
        s.skipped.append(t.key)
        self.commit(s, t.key + "_skip", f"Skipped {t.title}")

    def open_tool(self, key, again=False):
        """again: the tool was just applied and stays open on the new picture (Spot Fix, Mote Buster)."""
        self.close_tool()
        self.tool = F.TOOLS[key]
        self.tool_box.setVisible(True)
        self.tool_title.setText(f"<h3 style='margin:0'>{self.tool.title}</h3>{self.tool.tip}")
        self.method_row.setVisible(key == "stretch")
        self.chk_mask.blockSignals(True)
        self.chk_mask.setChecked(False)
        self.chk_mask.blockSignals(False)
        has_mask = key == "core" or getattr(self.tool, "has_mask", False)
        self.chk_mask.setVisible(has_mask)
        self.chk_mask.setText(getattr(self.tool, "mask_label", "Show what will change"))
        self.chk_mask.setToolTip(getattr(self.tool, "mask_tip", "Tints the area this tool works on (the mask "
                                                                   "SubZeroAstro made for you) in red"))
        self.sign_row.setVisible(key == "signature")
        self.crop_row.setVisible(key == "final_crop")
        self.flip_row.setVisible(key == "flip")
        if key == "flip":
            for b in (self.btn_flip_h, self.btn_flip_v):
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
            self.flip_rot = 0
            self.rot_label.setText("")
        if key == "signature":
            F.TOOLS["signature"].thumb_corner = self.sign_corner.currentText().lower()
            import targets as _tg
            tgt = self.state.target if self.state else ""
            self.sign_target.blockSignals(True)
            self.sign_target.setText(str(self.settings.value(f"finish/sign_target/{cp.safe_name(tgt)}",
                                                             _tg.display_name(tgt)) or ""))
            self.sign_target.blockSignals(False)
            self.sign_pos = None
            self.sign_corner.blockSignals(True)
            if self.sign_corner.currentText() == SIGN_DRAGGED:
                self.sign_corner.setCurrentText("Bottom right")
            self.sign_corner.blockSignals(False)
            self.view.on_drag = self._sign_drag
        self.fine_box.blockSignals(True)
        self.fine_box.setChecked(False)
        self.fine_box.blockSignals(False)
        self.fine_box.setVisible(False)
        self.preview_hint.setText("")
        if key == "stretch":
            self.method.blockSignals(True)
            self.method.setCurrentIndex(self.method.findData(self.tool.recommended_method(self.state)))
            self.method.blockSignals(False)
            self.chk_tool_linked.setVisible(self.state.is_color)
        if getattr(self.tool, "editor", None) == "spots":
            self.spots = []
            # start at a size you can see: at least ~10 screen pixels on a big, shrunk-to-fit picture
            lo_, hi_ = getattr(self.tool, "spot_range", (3.0, 80.0))
            self.spot_r = float(min(hi_, max(lo_, getattr(self.tool, "spot_start", 10.0),
                                              10.0 * getattr(self, "disp_factor", 1.0))))
            self.show_current()
            self.view.on_click = self._spot_click
            self.view.on_drag = self._spot_drag
            self.view.on_hover = self._spot_hover
            self.view.on_wheel = self._spot_wheel
            self.spot_active = None
            spot_note = ("Click a blemish to circle it, then turn the mouse wheel to size that circle. Click a "
                         "circle to pick it again (drag to move); right-click removes it.")
            if again:
                spot_note = ("<b>Fixed - this is the new picture.</b> Circle any spots that are left and press "
                             "the button again, or press <b>Done – Next Step</b> when you are happy. "
                             "(Undo Last Step takes back the last round.)")
            elif hasattr(self.tool, "found_spots"):
                try:
                    found = self.tool.found_spots(self.state)
                except Exception as e:  # noqa: BLE001 - finding is a help, never a blocker
                    found = []
                    self.log(f"{self.tool.title}: could not search automatically ({e})")
                for x, y, r in found:
                    self._add_spot(x, y, r)
                spot_note = ((f"Found {len(found)} donut(s) - circled in yellow. " if found else
                              "No donuts found automatically (ones on bright nebulosity are left to you). ") +
                             "Click a blemish to circle it, then turn the mouse wheel to size that circle. Click a "
                             "circle to pick it again (drag to move); right-click removes it.")
        self.render_choices()
        if getattr(self.tool, "editor", None) == "spots":
            self.choice_note.setText(spot_note)
            self.btn_done_tool.setVisible(again or key in self.state.done)
            self.btn_close_tool.setText("Close" if again else "Cancel")
        if key == "final_crop":
            self._set_crop_box("found")
        if key == "flip" and self.state is not None and "signature" in self.state.done:
            self.choice_note.setText("<b>Your picture is already signed.</b> The preview turns the signature too, "
                                     "but when you keep it, the signature is put back on top the right way round, "
                                     "in the same place.")

    def close_tool(self):
        self.tool = None
        self.btn_done_tool.setVisible(False)
        self.btn_close_tool.setText("Cancel")
        self.variants = []
        self.tool_box.setVisible(False)
        self.view.on_click = None
        self.view.on_wheel = None
        self.view.on_drag = None
        self.view.on_hover = None
        self._drop_hover_ring()
        self.spot_active = None
        self._drop_sign_box()
        for it in self.spot_items:
            self.view.scene().removeItem(it)
        self.spot_items = []
        self.spots = []
        self.preview_timer.stop()

    def set_full_window(self, on):
        """Picture as big as the window: hide the step list, the card, the tool panel, the log and the
        banner (the Show: row stays, to look through the steps); everything comes back as it was."""
        widgets = [self._left_panel, self.card, self.tool_box, self.log_view] + list(self._top_widgets)
        mw = self.menuWidget()
        if mw is not None:
            widgets.append(mw)
        if on:
            if self._full_saved is None:
                self._full_saved = {w: w.isVisible() for w in widgets}
            for w in widgets:
                w.setVisible(False)
            self.btn_full.setText("✕  Exit Full Window")
            if not (self.isMaximized() or self.isFullScreen()):
                self._full_was_normal = True
                self.showMaximized()
        else:
            for w, vis in (self._full_saved or {}).items():
                w.setVisible(vis)
            self._full_saved = None
            self.btn_full.setText("⛶  Full Window")
            if getattr(self, "_full_was_normal", False):
                self._full_was_normal = False
                self.showNormal()
        QTimer.singleShot(0, self._refit_view)

    def _refit_view(self):
        if not self.view._zoomed and self.view.full[0]:
            self.view._fit()

    def _zoom_detail(self):
        """Zoomed in past the preview's own pixels: draw the picture shown at full resolution."""
        if self.tool is not None or self.project is None or self.state is None:
            return
        if self._detail_job is not None and self._detail_job.isRunning():
            return
        i = self.hist.currentIndex()
        last = self.hist.count() - 1
        linked = self.chk_linked.isChecked()
        project, state = self.project, self.state

        def work():
            s = state if i == last else project.load(i)
            return i, F.state_display(s, linked=linked)

        def done(res):
            j, img = res
            if j == self.hist.currentIndex() and self.tool is None and self.view._zoomed:
                self.view.upgrade_pixmap(img)
        if self.synchronous:
            done(work())
            return
        self._detail_job = Job(work)
        self._detail_job.ok.connect(done)
        self._detail_job.start()

    def _rotate(self, quarter):
        """A quarter turn (+1 clockwise, -1 counter-clockwise); the preview follows."""
        self.flip_rot = (self.flip_rot + quarter) % 4
        self.rot_label.setText({0: "", 1: "turned 90° clockwise", 2: "turned 180°",
                                3: "turned 90° counter-clockwise"}[self.flip_rot])
        self.schedule_preview()

    def done_tool(self):
        """Finished with a tool that can be run several times: keep what was done and move on."""
        was = self.tool
        self.close_tool()
        self.show_current()
        self.refresh()
        if was:
            self.log(f"{was.title}: done")

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
            extra = self.extra_params()
            extra.pop("pos", None)                      # font samples: shown in the corner view
            thumbs = [F.state_display(t.apply(small, {**v.params, **extra}),
                                      with_stars=True if t.shows_stars else None) for v in vs]
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
        keep = getattr(self, "_keep_choice", None)
        if keep is not None and 0 <= keep < len(self.group.buttons()):
            self.group.buttons()[keep].setChecked(True)
        self._keep_choice = None
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

    def _set_crop_box(self, which):
        """Show the full picture with a draggable crop box (Final Crop)."""
        if self.state is None or not self.variants:
            return
        small = self.state.small(PREVIEW_SIDE)
        rgb8 = F.state_display(small, linked=self.chk_linked.isChecked())
        h, w = self.state.shape
        self.view.set_image(rgb8, (w, h), w / rgb8.shape[1])
        self.view.set_crop(tuple(self.variants[0].params["rect"]) if which == "found" else (0, 0, w, h))
        self.preview_hint.setText("Drag the box, then press Crop to the Box. Nothing changes until then.")

    def extra_params(self) -> dict:
        """Settings that live outside the picture choices (the signature's words, corner, colour)."""
        if self.tool is not None and self.tool.key == "signature":
            corner = self.sign_corner.currentText().lower()
            p = {"text": self.sign_text.text().strip(), "target_text": self.sign_target.text().strip(),
                 "corner": corner if corner in F.SIGN_CORNERS else "bottom right",
                 "colour": self.sign_colour.currentText()}
            if self.sign_pos is not None:
                p["pos"] = [float(self.sign_pos[0]), float(self.sign_pos[1])]
            return p
        return {}

    def _sign_corner_picked(self, text):
        if text != SIGN_DRAGGED:
            self.sign_pos = None                 # a corner again
        self.sign_timer.start()

    def _drop_sign_box(self):
        if self.sign_box_item is not None and self.sign_box_item.scene() is not None:
            self.view.scene().removeItem(self.sign_box_item)
        self.sign_box_item = None

    def _show_sign_box(self, rect):
        """A dashed outline round the signature on the preview - drag it to move the signature."""
        self.sign_rect = rect
        self._drop_sign_box()
        if rect is None:
            return
        x, y, w, h = rect
        it = QGraphicsRectItem(x, y, w, h)
        pen = QPen(QColor(255, 200, 60), 0, Qt.DashLine)
        it.setPen(pen)
        self.view.scene().addItem(it)
        self.sign_box_item = it

    def _sign_drag(self, x, y, phase):
        """Drag the signature (or click anywhere to move it there)."""
        if self.sign_rect is None or self.sign_box_item is None or not self.view.full[0]:
            return
        pw, ph = self.view.full
        rx, ry, rw, rh = self.sign_rect
        if phase == "press":
            inside = rx <= x <= rx + rw and ry <= y <= ry + rh
            self._sign_grab = (x - rx, y - ry) if inside else (rw / 2, rh / 2)
            phase = "move"
        gx, gy = getattr(self, "_sign_grab", (rw / 2, rh / 2))
        nx = min(max(0.0, x - gx), max(0.0, pw - rw))
        ny = min(max(0.0, y - gy), max(0.0, ph - rh))
        self.sign_box_item.setRect(nx, ny, rw, rh)
        if phase == "release":
            self.sign_rect = (nx, ny, rw, rh)
            self.sign_pos = (nx / pw, ny / ph)
            self.sign_corner.blockSignals(True)
            self.sign_corner.setCurrentText(SIGN_DRAGGED)
            self.sign_corner.blockSignals(False)
            self.preview_hint.setText("Signature moved - Apply to keep it there.")
            self.schedule_preview()

    def _sign_changed(self):
        self.settings.setValue("finish/signature", self.sign_text.text().strip())
        if self.state is not None:
            self.settings.setValue(f"finish/sign_target/{cp.safe_name(self.state.target)}",
                                   self.sign_target.text().strip())
        self.settings.setValue("finish/sign_corner", self.sign_corner.currentText())
        self.settings.setValue("finish/sign_colour", self.sign_colour.currentText())
        if self.tool is not None and self.tool.key == "signature":
            F.TOOLS["signature"].thumb_corner = self.sign_corner.currentText().lower()
            keep = max(0, self.group.checkedId())
            self._keep_choice = keep
            self.render_choices()

    def current_params(self) -> dict:
        """The params Apply would use: the picked version, plus fine-tuning if switched on."""
        t, v = self.tool, self.selected()
        if t is None or v is None:
            return {}
        if getattr(t, "editor", None) == "spots":
            return {"spots": [list(q) for q in self.spots]}
        if t.key == "final_crop":
            return {"rect": [int(x) for x in self.view.crop()]}
        if t.key == "flip":
            return {"h": self.btn_flip_h.isChecked(), "v": self.btn_flip_v.isChecked(), "rot": self.flip_rot % 4}
        if t.key == "signature":
            p = dict(v.params)
            if self.fine_box.isChecked():
                p.update({name: sp.lo + sl.value() * sp.step for name, (sl, sp) in self.sliders.items()})
            p.update(self.extra_params())
            return p
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
        if t is None or s is None or (not self.variants and getattr(t, "editor", None) != "spots"):
            return                      # choices not ready yet
        if t.key == "final_crop":       # the box on the picture is the preview
            return
        if self.preview_job and self.preview_job.isRunning():
            self.preview_pending = True
            return
        params = self.current_params()
        tuned = self.fine_box.isChecked() and bool(self.fine_values)
        if (t.key == "core" or getattr(t, "has_mask", False)) and self.chk_mask.isChecked():
            params = dict(params, show_mask=True)

        def work():
            if t.spatial:
                ps = s.crop(900, F.busiest_spot(s, 900))
            else:
                ps = s.small(PREVIEW_SIDE)
            if getattr(t, "editor", None) == "spots":
                f = max(s.shape) / max(ps.shape)
                return F.state_display(t.apply(ps, {"spots": [[x / f, y / f, r / f] for x, y, r in params["spots"]]}))
            if t.key == "signature":
                b = t.block(ps.shape, params)
                rect = None if b is None else (b[1], b[2], b[0].shape[1], b[0].shape[0])
                return t.preview(ps, params), rect
            img = t.preview(ps, params)
            if params.get("show_mask"):
                return img
            key = (t.key, id(s), t.spatial)
            base = self._before_cache.get(key)
            if base is None:                             # the picture as it is now, at the same size
                base = F.state_display(ps, with_stars=True if t.shows_stars else None)
                self._before_cache = {key: base}
            return img, changed_share(base, img)

        def done(img):
            rect = None
            change = None
            if t.key == "signature" and isinstance(img, tuple):
                img, rect = img
            elif isinstance(img, tuple):
                img, change = img
            self.view.show_still(img)
            if t.key == "signature":
                self._show_sign_box(rect)
            for it in self.spot_items:                   # keep the spot circles on top
                if it.scene() is None:
                    self.view.scene().addItem(it)
            v = self.selected()
            name = "your marked circles" if getattr(t, "editor", None) == "spots" else f"“{v.label}”" + (" (fine-tuned)" if tuned else "")
            where = " — full-size view of a small area" if t.spatial else ""
            how = ""
            if change is not None:
                if change < 0.005:
                    how = (" <b>This barely changes your picture</b> (under 0.5% of it visibly) - there is "
                           "little for this step to do here.")
                else:
                    how = f" It visibly changes about {change * 100:.0f}% of the picture."
            self.preview_hint.setText(f"Preview of {name}{where} — not applied yet. Apply to keep it, "
                                      "Cancel to go back." + how)
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

    # ------------------------------------------------------------ Spot Fix / Mote Buster circles
    # Click on a blemish: a circle appears there, and the mouse wheel resizes THAT circle while you
    # watch. Click inside a circle to pick it again (drag to move it); right-click removes the picked one.
    # A dashed ring follows the mouse, showing the size the next circle will have.
    SPOT_PEN = QColor(255, 200, 60)
    SPOT_ACTIVE = QColor(80, 220, 255)

    SPOT_MIN_DRAWN = 3.0               # a tiny circle is still drawn this big (shown pixels) so it can be seen

    @staticmethod
    def _spot_pen(color, dashed=False):
        pen = QPen(color, 2)
        pen.setCosmetic(True)             # 2 screen pixels thick at any zoom
        if dashed:
            pen.setStyle(Qt.DashLine)
        return pen

    def _spot_disp(self, i):
        f = getattr(self, "disp_factor", 1.0)
        x, y, r = self.spots[i]
        return x / f, y / f, r / f

    def _redraw_spot(self, i):
        x, y, r = self._spot_disp(i)
        r = max(r, self.SPOT_MIN_DRAWN)
        it = self.spot_items[i]
        it.setRect(x - r, y - r, 2 * r, 2 * r)

    def _set_active_spot(self, i):
        self.spot_active = i
        for k, it in enumerate(self.spot_items):
            it.setPen(self._spot_pen(self.SPOT_ACTIVE if k == i else self.SPOT_PEN))

    def _add_spot(self, x_full, y_full, r_full):
        """A circle at full-size coordinates (drawn on the shown, possibly smaller, picture)."""
        f = getattr(self, "disp_factor", 1.0)
        self.spots.append([float(x_full), float(y_full), float(r_full)])
        x, y, r = x_full / f, y_full / f, max(r_full / f, self.SPOT_MIN_DRAWN)
        it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        it.setPen(self._spot_pen(self.SPOT_PEN))
        self.view.scene().addItem(it)
        self.spot_items.append(it)

    def _remove_spot(self, i):
        if 0 <= i < len(self.spots):
            self.spots.pop(i)
            it = self.spot_items.pop(i)
            if it.scene() is not None:
                self.view.scene().removeItem(it)
        self._set_active_spot(len(self.spots) - 1 if self.spots else None)

    def _spot_note(self):
        n = len(self.spots)
        act = getattr(self, "spot_active", None)
        size = (f"Picked circle: {self.spots[act][2]:.0f} px" if act is not None and act < n
                else f"Next circle: {self.spot_r:.0f} px")
        self.choice_note.setText(f"{n} circle(s) marked. {size} - turn the mouse wheel to resize it. Click a "
                                 "circle to pick it (drag to move), right-click removes the picked one.")

    def _spot_hit(self, x, y):
        """Index of the circle under (x, y) in shown pixels, or None (the smallest one wins)."""
        hits = []
        for i in range(len(self.spots)):
            cx, cy, r = self._spot_disp(i)
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= max(r, 4.0):
                hits.append((r, i))
        return min(hits)[1] if hits else None

    def _spot_drag(self, x, y, phase):
        f = getattr(self, "disp_factor", 1.0)
        if phase == "press":
            i = self._spot_hit(x, y)
            if i is None:
                self._add_spot(x * f, y * f, self.spot_r)
                i = len(self.spots) - 1
                self._spot_grab = None              # a new circle: nothing to drag
                self.schedule_preview()
            else:
                cx, cy, _r = self._spot_disp(i)
                self._spot_grab = (x - cx, y - cy)
            self._set_active_spot(i)
            self._spot_note()
            return
        i = getattr(self, "spot_active", None)
        grab = getattr(self, "_spot_grab", None)
        if i is None or grab is None or i >= len(self.spots):
            return
        self.spots[i][0] = (x - grab[0]) * f
        self.spots[i][1] = (y - grab[1]) * f
        self._redraw_spot(i)
        if phase == "release":
            self._spot_grab = None
            self.schedule_preview()

    def _spot_click(self, x, y, button):
        """Right-click: remove the picked circle (or the one under the mouse)."""
        if button == Qt.RightButton and self.spots:
            i = self._spot_hit(x, y)
            if i is None:
                i = getattr(self, "spot_active", None)
            self._remove_spot(len(self.spots) - 1 if i is None else i)
            self._spot_note()
            self.schedule_preview()

    def _spot_wheel(self, steps):
        lo, hi = getattr(self.tool, "spot_range", (3.0, 80.0))
        k = 1.12 ** steps
        i = getattr(self, "spot_active", None)
        if i is not None and i < len(self.spots):
            self.spots[i][2] = float(min(hi, max(lo, self.spots[i][2] * k)))
            self.spot_r = self.spots[i][2]           # the next circle starts at this size
            self._redraw_spot(i)
            self.spot_wheel_timer.start()            # redraw the preview once the wheel stops
        else:
            self.spot_r = float(min(hi, max(lo, self.spot_r * k)))
        self._update_hover_ring()
        self._spot_note()

    def _spot_hover(self, x, y):
        self._hover_xy = (x, y)
        self._update_hover_ring()

    def _update_hover_ring(self):
        xy = getattr(self, "_hover_xy", None)
        if xy is None or self.tool is None or getattr(self.tool, "editor", None) != "spots":
            return
        f = getattr(self, "disp_factor", 1.0)
        r = max(self.spot_r / f, self.SPOT_MIN_DRAWN)
        if getattr(self, "hover_ring", None) is None or self.hover_ring.scene() is None:
            self.hover_ring = QGraphicsEllipseItem()
            self.hover_ring.setPen(self._spot_pen(QColor(255, 255, 255, 170), dashed=True))
            self.view.scene().addItem(self.hover_ring)
        over = self._spot_hit(*xy) is not None
        self.hover_ring.setVisible(not over)
        self.hover_ring.setRect(xy[0] - r, xy[1] - r, 2 * r, 2 * r)

    def _drop_hover_ring(self):
        hr = getattr(self, "hover_ring", None)
        if hr is not None and hr.scene() is not None:
            self.view.scene().removeItem(hr)
        self.hover_ring = None
        self._hover_xy = None

    def apply_choice(self):
        t, s = self.tool, self.state
        if t is None or not self.variants:
            return
        v = self.selected()
        params = self.current_params()
        if t.key == "flip" and not params.get("h") and not params.get("v") and not params.get("rot"):
            if not self.synchronous:
                QMessageBox.information(self, t.title, "Press a Flip or Rotate button first.")
            return
        if t.key == "signature" and not params.get("text") and not params.get("target_text"):
            if not self.synchronous:
                QMessageBox.information(self, t.title, "Type your signature (or the target name) first.")
            return
        spots_tool = getattr(t, "editor", None) == "spots"
        if spots_tool and not params.get("spots"):
            if not self.synchronous:
                QMessageBox.information(self, t.title, "Click on the picture to mark them first.")
            return
        tuned = (self.fine_box.isChecked() and bool(self.fine_values)) or spots_tool
        label = (f"{len(params['spots'])} spot(s)" if spots_tool
                 else v.label + (" (fine-tuned)" if tuned else ""))
        if t.key == "flip":
            label = F.FlipPicture.describe(params)

        if t.key == "flip" and "signature" in s.done and self.project is not None:
            plan = F.resign_plan(self.project.steps)
            if plan == "blocked":
                if not self.synchronous:
                    QMessageBox.information(
                        self, t.title, "Your signature is already on the picture and would be turned with it. "
                        "Click the Sign Your Picture step in the list and use Go Back to Here, flip or turn the "
                        "picture, then sign it again.")
                return
            if isinstance(plan, int):
                self._flip_under_signature(params, label)
                return

        def work():
            t0 = time.time()
            new = t.apply(s, params)
            return new, time.time() - t0
        again = (lambda: self.open_tool(t.key, again=True)) if getattr(t, "repeatable", False) else None
        self.run_job(work, lambda r: (self.log(f"{t.title}: {label}  ({r[1]:.1f} s)"),
                                      self.commit(r[0], t.key, f"{t.title} — {label}", params, then=again)),
                     f"{t.title}…")

    def _flip_under_signature(self, params, label):
        """Flip / turn a signed picture: the signature is put back on top afterwards, reading the right way."""
        proj = self.project
        k = F.resign_plan(proj.steps)

        def work():
            redo = F.flip_under_signature(proj, params)
            proj.truncate(k)
            for st, tool, lab, prm in redo:
                proj.save(st, tool, lab, prm)
            return redo[-1][0]

        def done(new):
            self.state = new
            self.log(f"Flip or Rotate: {label} - your signature was put back on top, the right way round")
            self.close_tool()
            self.after_change()
        self.run_job(work, done, "Turning the picture under your signature…")

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

    def commit(self, new_state, key, label, params=None, then=None):
        """then: called after the step is saved and the window updated (e.g. reopen the tool)."""
        self.state = new_state

        def work():
            self.project.save(new_state, key, label, params)
            return True
        self.run_job(work, lambda _r: (self.after_change(), then() if then else None), "Saving the step…")

    # ------------------------------------------------------------ output
    def default_out_dir(self) -> Path:
        src = Path(self.state.sources[0]) if self.state and self.state.sources else Path.home()
        return src.parent / "Finished"

    def export_dialog(self):
        if not self.state or self.state.rgb is None:
            QMessageBox.information(self, "Finish", "Combine the colour channels first.")
            return
        dlg = ExportDialog(self, self.settings, self.default_out_dir(), self.state.target)
        sub = F.picture_folder_name(self.state.target, self.state.sources)
        dlg.sub_note.setText(f"The pictures go in their own folder inside it: <b>{sub}</b>")
        if not dlg.exec():
            return
        fmts = dlg.formats()
        # every file of this picture in one folder: <chosen folder>/<target>_<scope>/
        folder = Path(dlg.dir.text().strip()) / F.picture_folder_name(self.state.target, self.state.sources)
        name = f"{cp.safe_name(self.state.target)}_{'finished' if not self.state.linear else 'linear'}"
        about = dlg.chk_about.isChecked()
        credit = self.credit_corner() if dlg.chk_credit.isChecked() else None
        target = self.state.target

        def work():
            return F.export(self.state, folder / name, fmts, save_rgb8=save_rgb8, credit=credit)

        def done(files):
            self.log("Saved " + ", ".join(str(f) for f in files))
            self.picture_saved = True
            if about and not self.synchronous:
                d = DescriptionDialog(self, target, folder)
                d.start_lookup()
                d.exec()
                if d.saved:
                    self.log(f"Saved the target description: {d.saved}")
                    files = list(files) + [d.saved]
            tools.reveal(files)
        self.run_job(work, done, "Saving the picture…")

    def credit_corner(self):
        """Corner for the 'Processed with SubZeroAstro' credit (None when switched off in Save the Picture)."""
        if not self.settings.value("finish/credit", True, type=bool):
            return None
        return F.credit_corner(self.project.steps if self.project else [])

    def share_astrobin(self, auto_accept=False):
        """Picture + acquisition CSV for AstroBin's upload page (its API cannot upload)."""
        import astrobin as AB
        if not self.state or self.state.rgb is None:
            QMessageBox.information(self, "Finish", "Combine the colour channels first.")
            return None
        if self.state.linear and not auto_accept:
            if QMessageBox.question(self, "Share to AstroBin", "The picture has not been brightened (stretched) "
                                    "yet, so it will look almost black on AstroBin. Share it anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return None
        sessions = AB.acquisition(self.state.sources)
        dlg = AstroBinDialog(self, self.settings, sessions, self.default_out_dir().parent / "AstroBin")
        if not auto_accept and not dlg.exec():
            return None
        folder = Path(dlg.dir.text().strip())
        name = cp.safe_name(self.state.target)
        fmts = ["jpg"] + (["tif"] if dlg.chk_tif.isChecked() else [])
        ids, bortle = dlg.filter_ids(), dlg.bortle.value() or None
        sqm, cal, open_page = dlg.sqm.value() or None, dlg.chk_cal.isChecked(), dlg.chk_open.isChecked()

        def work():
            files = F.export(self.state, folder / f"{name}_AstroBin", fmts, save_rgb8=save_rgb8,
                             credit=self.credit_corner())
            files.append(AB.write_csv(folder / f"{name}_astrobin_acquisition.csv", sessions, ids, bortle, sqm, cal))
            return files

        def done(files):
            self.last_astrobin_files = files
            self.picture_saved = True
            self.log(f"Saved for AstroBin in {folder}: " + ", ".join(Path(f).name for f in files))
            if self.synchronous:
                return
            tools.reveal(files)
            if open_page:
                AstroBinDialog._open(AB.UPLOAD_URL)
            QMessageBox.information(self, "Share to AstroBin",
                                    "<b>Ready for AstroBin.</b><br><br>1. Drag <b>" + Path(files[0]).name +
                                    "</b> onto AstroBin's upload page.<br>2. On the <i>Acquisition</i> step, "
                                    "choose <b>Import CSV</b> and pick <b>" + Path(files[-1]).name + "</b>.<br>"
                                    "<br>The folder is open for you.")
        self.run_job(work, done, "Saving for AstroBin…")
        return dlg

    def stop_here_pixinsight(self):
        if not self.state:
            return
        folder = Path(self.settings.value("finish/pi_dir", "") or (self.default_out_dir().parent / "PixInsight"))

        def work():
            return F.export_for_pixinsight(self.state, folder)

        def done(files):
            self.last_pi_files = files
            self.picture_saved = True
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

    def end_project(self, ask=True, load=True):
        """'End this Project and Start New Project': confirm, then close every page and go back to page 1."""
        import ui_header
        if ask:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Start New Project")
            saved = ("Your picture has been saved." if self.picture_saved else
                     "<b>You have not saved the picture in this session.</b> Use <i>Save the Picture…</i> first "
                     "if you want to keep it.")
            box.setText("<b>End this project and start a new one?</b><br><br>"
                        "All pages and images close and SubZeroAstro starts again at "
                        "<b>Quality and Cull</b> with the Load Light Frame Data window, ready for your next target.<br><br>"
                        f"{saved}<br><br><small>Nothing on disk is deleted. Every step here is kept: opening "
                        "these masters on Finishing Touches later carries on where you stopped.</small>")
            b_save = None if self.picture_saved or not self.state or self.state.rgb is None else \
                box.addButton("Save the Picture First…", QMessageBox.ActionRole)
            b_go = box.addButton("Start New Project", QMessageBox.AcceptRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(b_go)
            box.exec()
            if b_save is not None and box.clickedButton() is b_save:
                self.export_dialog()
                return False
            if box.clickedButton() is not b_go:
                return False
        return ui_header.new_project(self, load=load)

    def closeEvent(self, ev):
        if self.job and self.job.isRunning():
            self.job.wait(60000)
        super().closeEvent(ev)
