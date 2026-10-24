"""
stack_window.py - the Stacking window.

Opened from the culling window with the kept frames already loaded. Shows the
light groups (target + filter + camera + rig), the calibration that will be
used for each, lets you add calibration frames or masters, checks for missing
calibration before starting, then stacks in the background.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

import fits_core as core
import stacking as st

OK, WARN, BAD = "✓", "⚠", "✗"


def lights_from_session(frames) -> tuple[list, list]:
    """LightFrames from the culling session's kept frames. Returns (lights, skipped_names)."""
    lights, skipped = [], []
    for f in frames:
        if f.rejected or f.error or not f.cards:
            continue
        look = core.cards_to_dict(f.cards)
        typ, master = st.frame_type(look, f.name)
        if typ not in (None, "light") or master:
            skipped.append(f.name)
            continue
        w, h = int(look.get("NAXIS1", 0) or 0), int(look.get("NAXIS2", 0) or 0)
        info = st.info_from_look(f.path, look, w, h, "light", cards=f.cards)
        if getattr(f, "filter_override", None):
            info.filter = f.filter_override          # the user said which filter was really used
        lights.append(st.LightFrame(info, f.score, (f.metrics or {}).get("fwhm")))
    return lights, skipped


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------

class StackWorker(QThread):
    progress = Signal(str, float)
    log = Signal(str)
    group_done = Signal(object)
    failed = Signal(str)
    ask_sky = Signal(object)            # {"filter", "problems", "built", "answer", "event"} - answered on the UI thread

    def __init__(self, groups, items, opts, rigs, runlog=None, skip=()):
        super().__init__()
        self.groups, self.items, self.opts, self.rigs = groups, items, opts, rigs
        self.runlog = runlog or st.StackRunLog(opts)
        self.skip = set(skip)               # id(group) already saved by an interrupted run
        self.completed = False
        self.cancel_requested = False
        self.results = []
        self.saved_cal = []
        self.sky_answer = None              # tests: "without" / "use" / "cancel" instead of asking
        self.sky_dropped = set()            # sky flats the user chose to stack without

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            self._run()
        except st.Cancelled:
            self.log.emit("Cancelled.")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1500:]}")

    def _sky_flat(self, item, helper_b, helper_d, prog) -> bool:
        """Build a SkyFlat (Mote Buster) and check it. When it may not be good enough, ask whether to use it
        anyway or stack without flats. True = use it; False = stack without flats."""
        try:
            st.build_master(item, helper_b, helper_d, self.opts.kappa,
                            progress=lambda t, f: prog(t, 0.15 * f), cancel=lambda: self.cancel_requested)
            problems = st.sky_flat_problems(item)
        except st.Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - asked about below, never a crash
            item.data = None
            problems = [str(e)]
        if not problems:
            return True
        built = item.data is not None
        self.log.emit(f"WARNING: the SkyFlat (Mote Buster) for {item.filter} may not be good enough: "
                      + "; ".join(problems))
        q = {"filter": item.filter, "problems": problems, "built": built, "answer": self.sky_answer}
        if q["answer"] is None:
            import threading
            q["event"] = threading.Event()
            self.ask_sky.emit(q)
            while not q["event"].wait(0.2):
                if self.cancel_requested:
                    raise st.Cancelled()
        ans = q["answer"] or "without"
        if ans == "cancel":
            self.cancel_requested = True
            raise st.Cancelled()
        if ans == "use" and built:
            self.log.emit(f"Using the SkyFlat for {item.filter} anyway, as chosen")
            return True
        item.data = None
        self.log.emit(f"Stacking {item.filter} without flats, as chosen (no SkyFlat)")
        return False

    def _run(self):
        opts = self.opts
        n = len(self.groups)
        cams = {}
        for g in self.groups:
            cams.setdefault(g.target, set()).add((g.camera or "").lower())
        built = set()
        common_for = {}
        if opts.align_common:
            for gset in st.alignment_sets(self.groups):
                if len(gset) > 1:
                    try:
                        c = st.common_reference(gset, progress=lambda t, f: self.progress.emit(t, 0.0))
                    except Exception as e:  # noqa: BLE001
                        self.log.emit(f"Could not prepare a common reference: {e}")
                        c = None
                    if c:
                        self.log.emit(f"Common reference for {gset[0].target} "
                                      f"({', '.join(g.filter for g in gset)}): {c['name']}")
                        for g in gset:
                            common_for[id(g)] = c
        for gi, g in enumerate(self.groups):
            def prog(text, frac, gi=gi):
                self.progress.emit(text, (gi + max(0.0, min(frac, 1.0))) / n)
                if self.cancel_requested:
                    raise st.Cancelled()
            if id(g) in self.skip:
                d = self.runlog.done(g)
                if d:
                    res = st.StackResult(g, files=[Path(f) for f in d["files"]], used=d.get("used", 0))
                    self.log.emit(f"\n=== {g.label()} — already saved by the interrupted run ({d.get('time', '')}) ===")
                    for f in d["files"]:
                        self.log.emit(f"Kept {f}")
                    self.results.append(res)
                    self.group_done.emit(res)
                    continue
            p = g.plan
            # build / load the calibration masters this group needs
            for item, helper_b, helper_d in [(p.bias, None, None), (p.dark, None, None)] + p.flat_sets():
                if item is None:
                    continue
                if not item.master and id(item) not in built:
                    prev = self.runlog.cal_saved(item)
                    if prev and st.use_saved_calibration(item, prev):
                        self.log.emit(f"Using {Path(prev[0]).name} saved by the interrupted run")
                first = id(item) not in built
                if item.sky and id(item) in self.sky_dropped:
                    p.flat = None
                    self.log.emit(f"No flat for {g.label()} - stacking without flats, as chosen")
                    p.warnings.append(("flat", "no flats - the SkyFlat was not used, as chosen"))
                    continue
                if item.sky and first:
                    if not self._sky_flat(item, helper_b, helper_d, prog):
                        self.sky_dropped.add(id(item))
                        p.flat = None
                        p.warnings.append(("flat", "no flats - the SkyFlat was not used, as chosen"))
                        built.add(id(item))
                        continue
                else:
                    st.build_master(item, helper_b, helper_d, opts.kappa,
                                    progress=lambda t, f: prog(t, 0.15 * f), cancel=lambda: self.cancel_requested)
                if first:
                    built.add(id(item))
                    if not item.master:
                        self.log.emit(f"Built {item.label()}")
                        fg = (getattr(item, "sky_stats", None) or {}).get("flip_groups")
                        if item.sky and fg:
                            self.log.emit(f"  SkyFlat: the frames come from both sides of a meridian flip ({fg[0]} and "
                                          f"{fg[1]} frames, the nebula turned 180° between them) - each pixel's flat "
                                          "was taken from the side that shows plain sky there, so no ghost of the "
                                          "nebula gets into it")
                        if opts.save_cal_masters and opts.cal_root and not item.sky:
                            files = st.save_calibration_master(item, opts.cal_root, g.rig, opts.formats)
                            self.saved_cal += files
                            try:
                                self.runlog.mark_cal(item, files)
                            except OSError:
                                pass
                            for f in files:
                                self.log.emit(f"  saved {f}")
            self.log.emit(f"\n=== {g.label()} — {len(g.frames)} frames ===")
            try:
                res = st.stack_group(g, opts, multi_camera=len(cams[g.target]) > 1,
                                     progress=lambda t, f: prog(t, 0.15 + 0.8 * f),
                                     cancel=lambda: self.cancel_requested, common=common_for.get(id(g)))
                files = st.save_master_light(res, opts, multi_camera=len(cams[g.target]) > 1)
                try:
                    self.runlog.mark(g, res)
                except OSError:
                    pass
                for line in res.log:
                    self.log.emit(line)
                for f in files:
                    self.log.emit(f"Saved {f}")
                res.data = None                     # free memory
            except st.Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                res = st.StackResult(g, error=str(e))
                self.log.emit(f"FAILED: {e}")
            self.results.append(res)
            self.group_done.emit(res)
        self.completed = True
        self.runlog.finish(self.groups)
        self.progress.emit("Done", 1.0)


# --------------------------------------------------------------------------
# Missing-calibration dialog
# --------------------------------------------------------------------------

class PreflightDialog(QDialog):
    ADD = 2

    def __init__(self, parent, problems):
        super().__init__(parent)
        self.setWindowTitle("Check calibration")
        self.resize(720, 420)
        lay = QVBoxLayout(self)
        head = QLabel("<b>Some calibration frames are missing or don't match.</b><br>"
                      "You can stack anyway, or add the missing calibration first.")
        head.setWordWrap(True)
        lay.addWidget(head)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(f"{label}\n" + "\n".join(f"   {WARN} {w}" for w in ws)
                                      for label, ws in problems))
        lay.addWidget(text, 1)
        row = QHBoxLayout()
        b_add = QPushButton("Add Calibration Frames First…")
        b_go = QPushButton("Stack Anyway")
        b_cancel = QPushButton("Cancel")
        b_add.clicked.connect(lambda: self.done(self.ADD))
        b_go.clicked.connect(self.accept)
        b_cancel.clicked.connect(self.reject)
        row.addWidget(b_add)
        row.addStretch(1)
        row.addWidget(b_cancel)
        row.addWidget(b_go)
        lay.addLayout(row)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

class StackWindow(QMainWindow):
    COLS = ["Stack", "Target", "Filter", "Camera", "Telescope (rig)", "Frames", "Exposure", "Nights",
            "Bias", "Darks", "Flats"]

    def __init__(self, parent, frames, settings):
        super().__init__(parent)
        self.setWindowTitle("SubZeroAstro — Calibrate and Stack")
        self.settings = settings
        self.worker = None
        self.rigs = st.Rigs(self._load_json("stack/rigs", []))
        self.added_infos: list = []          # calibration frames the user added
        self.items: list = []                # all calibration items (library + added)
        self.lights, skipped = lights_from_session(frames)
        import ui_header
        ui_header.install_header(self, ui_header.PAGE_STACK, with_menu=False)
        self._build_ui()
        self.resize(1300, 860)
        self.refresh_all()
        if skipped:
            self.log(f"Not stacked (not light frames): {', '.join(skipped[:10])}"
                     + (" …" if len(skipped) > 10 else ""))

    # ------------------------------------------------------------ settings
    def _load_json(self, key, default):
        try:
            return json.loads(self.settings.value(key, "") or "null") or default
        except (ValueError, TypeError):
            return default

    def _save_settings(self):
        s = self.settings
        s.setValue("stack/rigs", json.dumps(self.rigs.entries))
        s.setValue("stack/cal_root", self.cal_root.text())
        s.setValue("stack/masters_root", self.masters_root.text())
        s.setValue("stack/kappa", self.kappa.value())
        s.setValue("stack/weighting", self.weighting.isChecked())
        s.setValue("stack/osc_half", self.osc_mode.currentIndex() == 1)
        s.setValue("stack/fits", self.fmt_fits.isChecked())
        s.setValue("stack/xisf", self.fmt_xisf.isChecked())
        s.setValue("stack/save_cal", self.save_cal.isChecked())
        s.setValue("stack/align_common", self.align_common.isChecked())

    # ------------------------------------------------------------------ UI
    def _folder_row(self, key, tip):
        row = QHBoxLayout()
        edit = QLineEdit(self.settings.value(key, ""))
        edit.setPlaceholderText(tip)
        edit.setToolTip(tip)
        btn = QPushButton("Browse…")

        def browse():
            d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or str(Path.home()))
            if d:
                edit.setText(d)
                self._save_settings()
                self.refresh_all()
        btn.clicked.connect(browse)
        edit.editingFinished.connect(lambda: (self._save_settings(), self.refresh_all()))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        return row, edit

    def _build_ui(self):
        central = QWidget()
        lay = QVBoxLayout(central)

        # destinations
        box = QGroupBox("Save locations")
        form = QFormLayout(box)
        r1, self.cal_root = self._folder_row(
            "stack/cal_root", "Your calibration frames folder (with a subfolder per telescope rig)")
        r2, self.masters_root = self._folder_row(
            "stack/masters_root", "Where master lights are saved (a subfolder per target is created)")
        form.addRow("Calibration masters:", r1)
        form.addRow("Master lights:", r2)
        hint = QLabel("Calibration masters are saved as  <i>&lt;calibration folder&gt;\\&lt;rig&gt;\\Masters\\"
                      "&lt;camera&gt;\\Bias | Darks | Flats\\&lt;filter&gt;</i>  and found there automatically "
                      "next time. Master lights go to  <i>&lt;master lights folder&gt;\\&lt;target&gt;</i>.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9a9aa6;")
        box.setToolTip(hint.text())          # the full explanation on hover; one short line on the page
        short = QLabel("Calibration masters are saved by rig, camera and filter and found again automatically; "
                       "master lights get a subfolder per target. (Hover for details.)")
        short.setStyleSheet("color:#9a9aa6;")
        short.setToolTip(hint.text())
        form.addRow(short)
        lay.addWidget(box)

        split = QSplitter(Qt.Vertical)
        # light groups
        gbox = QGroupBox("Light frames to stack")
        gl = QVBoxLayout(gbox)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)          # telescope column takes spare width
        for c in (8, 9, 10):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            self.table.setColumnWidth(c, 64)
        self.table.itemSelectionChanged.connect(self.show_plan_details)
        # one click includes or leaves out a whole kind of filter
        brow = QHBoxLayout()
        brow.addWidget(QLabel("Stack:"))
        self.chk_broad = QCheckBox("Broadband (L, R, G, B)")
        self.chk_narrow = QCheckBox("Narrowband (Ha, OIII, SII)")
        for cb_, band in ((self.chk_broad, "broad"), (self.chk_narrow, "narrow")):
            cb_.setTristate(False)
            cb_.setChecked(True)
            cb_.setToolTip("Tick or untick every row of this kind at once (blue rows: broadband, red rows: "
                           "narrowband)")
            cb_.clicked.connect(lambda v, band=band: self.set_band_enabled(band, v))
            brow.addWidget(cb_)
        brow.addStretch(1)
        gl.addLayout(brow)
        gl.addWidget(self.table, 2)
        self.details = QLabel("Select a row to see which calibration will be used.")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.RichText)
        self.table.setMinimumHeight(70)
        det = QScrollArea()                   # long notes scroll instead of making the page taller
        det.setWidget(self.details)
        det.setWidgetResizable(True)
        det.setFrameShape(QFrame.NoFrame)
        det.setMinimumHeight(48)
        det.setMaximumHeight(130)
        gl.addWidget(det)
        split.addWidget(gbox)

        # calibration
        cbox = QGroupBox("Calibration frames")
        cl = QVBoxLayout(cbox)
        crow = QHBoxLayout()
        b1 = QPushButton("Add Calibration Folder…")
        b1.setToolTip("A folder of bias, dark or flat subframes (or masters). Frame types are read from the "
                      "FITS/XISF headers.")
        b1.clicked.connect(self.add_cal_folder)
        b2 = QPushButton("Add Calibration Files…")
        b2.clicked.connect(self.add_cal_files)
        b3 = QPushButton("Search Calibration Folder for Subframes")
        b3.setToolTip("Look through every rig folder inside your calibration folder for bias / dark / flat "
                      "subframes (can take a while on a network drive)")
        b3.clicked.connect(self.search_cal_root)
        b4 = QPushButton("Remove Selected")
        b4.clicked.connect(self.remove_cal)
        b5 = QPushButton("Build Calibration Masters…")
        b5.setToolTip("Turn a folder of bias, dark, flat and dark-flat subframes into masters in your calibration "
                      "library - no light frames needed")
        b5.clicked.connect(self.build_cal_masters)
        self.btn_build_masters = b5
        for b in (b1, b2, b3, b5):
            crow.addWidget(b)
        crow.addStretch(1)
        crow.addWidget(b4)
        cl.addLayout(crow)
        self.cal_list = QListWidget()
        self.cal_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cal_list.setMinimumHeight(40)
        cl.addWidget(self.cal_list, 1)
        split.addWidget(cbox)

        # log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        self.log_view.setFont(f)
        self.log_view.setMinimumHeight(40)
        split.addWidget(self.log_view)
        split.setSizes([330, 220, 160])
        lay.addWidget(split, 1)

        # options + start
        obox = QHBoxLayout()
        obox.addWidget(QLabel("Reject outliers beyond"))
        self.kappa = QDoubleSpinBox()
        self.kappa.setRange(1.5, 6.0)
        self.kappa.setSingleStep(0.5)
        self.kappa.setValue(float(self.settings.value("stack/kappa", 3.0)))
        self.kappa.setSuffix(" σ")
        obox.addWidget(self.kappa)
        self.weighting = QCheckBox("Weight frames by noise")
        self.weighting.setChecked(self.settings.value("stack/weighting", True, type=bool))
        obox.addWidget(self.weighting)
        obox.addWidget(QLabel("Colour cameras:"))
        self.osc_mode = QComboBox()
        self.osc_mode.addItems(["Full resolution", "Half resolution (faster, less disk)"])
        self.osc_mode.setCurrentIndex(1 if self.settings.value("stack/osc_half", False, type=bool) else 0)
        obox.addWidget(self.osc_mode)
        self.fmt_fits = QCheckBox("FITS")
        self.fmt_fits.setChecked(self.settings.value("stack/fits", True, type=bool))
        self.fmt_xisf = QCheckBox("XISF")
        self.fmt_xisf.setChecked(self.settings.value("stack/xisf", False, type=bool))
        obox.addStretch(1)
        lay.addLayout(obox)
        obox = QHBoxLayout()                  # second row: keeps the page narrow enough for a laptop
        self.save_cal = QCheckBox("Save new calibration masters")
        self.save_cal.setChecked(self.settings.value("stack/save_cal", True, type=bool))
        obox.addWidget(self.save_cal)
        self.align_common = QCheckBox("Line up all filters of a target")
        self.align_common.setToolTip("Put every master of the same target, camera and telescope on one pixel grid "
                                     "(e.g. L, R, G, B or Ha, OIII, SII), ready to combine without re-aligning")
        self.align_common.setChecked(self.settings.value("stack/align_common", True, type=bool))
        obox.addWidget(self.align_common)
        self.sky_flats = QCheckBox("SkyFlats (Mote Buster) when flats are missing")
        self.sky_flats.setToolTip("No flats for a filter? Make a flat from your light frames instead (best when "
                                  "you shot several targets with that filter): stars hidden, light-pollution "
                                  "tilt removed, leaving vignetting and dust shadows. Real flats are better.")
        self.sky_flats.setChecked(self.settings.value("stack/sky_flats", True, type=bool))
        self.sky_flats.toggled.connect(lambda on: (self.settings.setValue("stack/sky_flats", on), self.refresh_all()))
        obox.addWidget(self.sky_flats)
        self.cosmetic = QCheckBox("Remove hot pixels")
        self.cosmetic.setToolTip("Cosmetic correction: single pixels far brighter than all their neighbours (hot "
                                 "pixels, cosmic-ray hits) are replaced by their neighbours in every frame. "
                                 "Needed when the frames weren't dithered - stacking can't remove those.")
        self.cosmetic.setChecked(self.settings.value("stack/cosmetic", True, type=bool))
        self.cosmetic.toggled.connect(lambda on: self.settings.setValue("stack/cosmetic", on))
        obox.addWidget(self.cosmetic)
        obox.addSpacing(16)
        obox.addWidget(QLabel("Save as:"))
        obox.addWidget(self.fmt_fits)
        obox.addWidget(self.fmt_xisf)
        obox.addStretch(1)
        lay.addLayout(obox)

        brow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        brow.addWidget(self.progress, 1)
        self.btn_prepare = QPushButton("Continue to Image Processing…")
        self.btn_prepare.setToolTip("Crop, plate solve, remove gradients and run the XTerminators on the masters "
                                    "just made, then get a step-by-step PixInsight plan")
        self.btn_prepare.setEnabled(False)
        self.btn_prepare.clicked.connect(self.open_prepare)
        self.app_header.add_next(self.btn_prepare, "Continue to Image Processing")   # top right, page-turn style
        self.btn_pi = QPushButton("Stop Here – Continue in PixInsight")
        self.btn_pi.setToolTip("Open the folder with the new master lights selected, plus a note of the "
                               "recommended PixInsight steps")
        self.btn_pi.setEnabled(False)
        self.btn_pi.clicked.connect(self.stop_here_pixinsight)
        brow.addWidget(self.btn_pi)
        self.btn_open = QPushButton("Open Master Lights Folder")
        self.btn_open.clicked.connect(self.open_masters_folder)
        brow.addWidget(self.btn_open)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        brow.addWidget(self.btn_cancel)
        self.btn_start = QPushButton("▶  Check && Stack")
        self.btn_start.setStyleSheet(__import__("ui_header").RUN_STYLE)
        self.btn_start.clicked.connect(self.start)
        brow.addWidget(self.btn_start)
        lay.addLayout(brow)
        self.setCentralWidget(central)

    # ------------------------------------------------------------ helpers
    def log(self, text):
        self.log_view.appendPlainText(text)

    def options(self) -> st.StackOptions:
        fmts = tuple(f for f, cb in (("fits", self.fmt_fits), ("xisf", self.fmt_xisf)) if cb.isChecked()) or ("fits",)
        return st.StackOptions(kappa=self.kappa.value(), weighting=self.weighting.isChecked(),
                               osc_half=self.osc_mode.currentIndex() == 1, formats=fmts,
                               masters_root=self.masters_root.text().strip(),
                               cal_root=self.cal_root.text().strip(),
                               save_cal_masters=self.save_cal.isChecked(),
                               align_common=self.align_common.isChecked(),
                               cosmetic=self.cosmetic.isChecked())

    @staticmethod
    def open_folder(path):
        """Open a folder in Explorer. str(Path(...)) turns '//nas/share' into '\\\\nas\\share',
        which Windows needs for network folders."""
        import tools
        if path and st._is_dir(path):
            if not tools.reveal(Path(path)):
                QMessageBox.information(None, "SubZeroAstro", f"Could not open {Path(path)}")

    def open_masters_folder(self):
        """Open the folder of the masters just stacked (selected), or the master lights folder."""
        import tools
        res = getattr(getattr(self, "worker", None), "results", None) or []
        files = [Path(f) for r in res for f in (getattr(r, "files", None) or []) if Path(f).exists()]
        if files:
            tools.reveal(files)
            return
        root = self.masters_root.text().strip()
        if root and st._is_dir(root):
            self.open_folder(root)
        else:
            QMessageBox.information(self, "SubZeroAstro", f"The master lights folder was not found:\n{root}")

    def rig_choices(self):
        root = Path(self.cal_root.text().strip()) if self.cal_root.text().strip() else None
        names = sorted(p.name for p in st._safe_listdir(root) if st._is_dir(p)) if root and st._is_dir(root) else []
        return names

    # ------------------------------------------------------------ data
    def refresh_all(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                lib = st.scan_library(self.cal_root.text().strip()) if self.cal_root.text().strip() else []
            except Exception as e:  # noqa: BLE001 - the library is a help; added frames must still work
                lib = []
                self.log(f"Could not read the calibration masters folder ({e}) - using the added files only")
            added = st.group_calibration(self.added_infos)
            self.items = lib + added
            self.groups = st.group_lights(self.lights, self.rigs)
            pool = None
            if getattr(self, "sky_flats", None) is None or self.sky_flats.isChecked():
                pool = []
                for g in self.groups:
                    for lf in g.frames:
                        lf.info._rig = g.rig
                        pool.append(lf.info)
            for g in self.groups:
                g.plan = st.plan_group(g, self.items, sky_pool=pool)
        finally:
            QApplication.restoreOverrideCursor()
        self.fill_table()
        self.fill_cal_list()

    def fill_cal_list(self):
        self.cal_list.clear()
        if not self.items:
            it = QListWidgetItem("No calibration frames yet — add a folder of bias / darks / flats, or set your "
                                 "calibration masters folder above.")
            it.setFlags(Qt.NoItemFlags)
            self.cal_list.addItem(it)
            return
        used = set()
        for g in self.groups:
            for c in g.plan.all_items():
                used.add(id(c))
        for c in self.items:
            mark = "●" if id(c) in used else "○"
            it = QListWidgetItem(f"{mark}  {c.label()}")
            it.setData(Qt.UserRole, id(c))
            cam = c.camera or "unknown camera"
            it.setToolTip(f"{cam}\n{c.ref.path if c.master else c.ref.path.parent}\n"
                          + ("Used by a group below" if id(c) in used else "Not needed by any group"))
            if id(c) not in used:
                it.setForeground(QColor(140, 140, 150))
            self.cal_list.addItem(it)

    def status_cell(self, plan: st.CalPlan, kind):
        if plan.precal:
            return OK, f"not needed - the lights are already calibrated ({plan.precal})"
        if kind == "bias":
            if plan.bias:
                return OK, plan.bias.label()
            if plan.dark:
                return OK, "not needed (darks include the bias)"
            return BAD, next((w for k, w in plan.warnings if k == "bias"), "missing")
        if kind == "dark":
            if plan.dark:
                return (OK if plan.dark_mode == "exact" else WARN), plan.dark.label() + (
                    "  (scaled to the light exposure)" if plan.dark_mode == "scaled" else "")
            return (WARN if plan.bias else BAD), next((w for k, w in plan.warnings if k == "dark"), "missing")
        if plan.flat:
            fb = next((w for k, w in plan.warnings if k == "flatbias"), None)
            return (WARN if fb or plan.notes else OK), plan.flat.label() + (f"\n{fb}" if fb else "") + \
                ("\n" + "\n".join(plan.notes) if plan.notes else "")
        return BAD, next((w for k, w in plan.warnings if k == "flat"), "missing")

    def fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.groups))
        colors = {OK: QColor(80, 200, 100), WARN: QColor(235, 170, 50), BAD: QColor(230, 80, 80)}
        for r, g in enumerate(self.groups):
            cb = QCheckBox()
            cb.setChecked(g.enabled)
            cb.toggled.connect(lambda v, g=g: (setattr(g, "enabled", v), self._sync_band_boxes()))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(8, 0, 0, 0)
            hl.addWidget(cb)
            self.table.setCellWidget(r, 0, holder)
            nights = g.nights
            vals = [g.target, g.filter, st.short_camera(g.camera), None, str(len(g.frames)),
                    f"{g.total_exposure / 3600:.2f} h",
                    (nights[0] if len(nights) == 1 else f"{nights[0]} … {nights[-1]} ({len(nights)})") if nights else "?"]
            band = st.filter_band(g.filter)
            tint = QColor(70, 110, 190, 45) if band == "broad" else QColor(200, 70, 70, 45)
            for c, v in enumerate(vals, start=1):
                if v is None:
                    continue
                it_ = QTableWidgetItem(v)
                it_.setBackground(tint)                    # blue = broadband, red = narrowband
                if c == 1:
                    it_.setToolTip("Broadband filter (L, R, G, B…)" if band == "broad" else
                                   "Narrowband filter (Ha, OIII, SII, dual-band)")
                self.table.setItem(r, c, it_)
            # filter: editable combo - force the filter when the header has it wrong / missing
            fcombo = QComboBox()
            fcombo.setEditable(True)
            fchoices = sorted({c.filter for c in self.items if c.kind == "flat" and c.filter} | {g.filter})
            for name in fchoices:
                fcombo.addItem(name)
            fcombo.setCurrentText(g.filter)
            fcombo.setToolTip("The filter these lights were taken with. If the camera didn't record it (or got it "
                              "wrong), pick the filter of your flats here and they will be used. Your files are "
                              "not changed.")
            fcombo.activated.connect(lambda _i, g=g, cbx=fcombo: self.set_group_filter(g, cbx.currentText()))
            fcombo.lineEdit().editingFinished.connect(lambda g=g, cbx=fcombo: self.set_group_filter(g, cbx.currentText()))
            self.table.setCellWidget(r, 2, fcombo)
            # rig name: editable combo
            combo = QComboBox()
            combo.setEditable(True)
            choices = self.rig_choices()
            for name in choices:
                combo.addItem(name)
            if g.rig not in choices:
                combo.insertItem(0, g.rig)
            combo.setCurrentText(g.rig)
            combo.setToolTip(f"{g.camera} at {g.ref.focal or '?'} mm — the folder name used for this rig's "
                             "calibration masters. Pick one of your rig folders or type a new name.")
            combo.activated.connect(lambda _i, g=g, cbx=combo: self.set_rig(g, cbx.currentText()))
            combo.lineEdit().editingFinished.connect(lambda g=g, cbx=combo: self.set_rig(g, cbx.currentText()))
            self.table.setCellWidget(r, 4, combo)
            for c, kind in ((8, "bias"), (9, "dark"), (10, "flat")):
                sym, tip = self.status_cell(g.plan, kind)
                it = QTableWidgetItem(sym)
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(colors[sym])
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                it.setToolTip(tip)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self._sync_band_boxes()
        self.table.setColumnWidth(2, max(self.table.columnWidth(2), 130))   # room for the filter name
        if self.groups:
            self.table.selectRow(0)
        self.show_plan_details()

    def set_band_enabled(self, band, on):
        """Tick / untick every light group of one kind (broadband or narrowband)."""
        for g in self.groups:
            if st.filter_band(g.filter) == band:
                g.enabled = bool(on)
        self.fill_table()

    def _sync_band_boxes(self):
        for cb_, band in ((getattr(self, "chk_broad", None), "broad"), (getattr(self, "chk_narrow", None), "narrow")):
            if cb_ is None:
                continue
            gs = [g for g in self.groups if st.filter_band(g.filter) == band]
            cb_.blockSignals(True)
            cb_.setVisible(bool(gs))
            cb_.setChecked(bool(gs) and all(g.enabled for g in gs))
            cb_.blockSignals(False)

    def set_group_filter(self, g, name):
        import fits_core as core
        name = core.normalize_filter(name.strip()) if name.strip() else ""
        if not name or name == g.filter:
            return
        for lf in g.frames:
            lf.info.filter = name
        self.log(f"{g.target}: filter set to {name} for {len(g.frames)} frames - calibration re-matched")
        self.refresh_all()

    def set_rig(self, g, name):
        name = name.strip()
        if not name or name == g.rig:
            return
        self.rigs.set(g.camera, g.ref.focal, name)
        self._save_settings()
        self.refresh_all()

    def show_plan_details(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows or rows[0] >= len(self.groups):
            return
        g = self.groups[rows[0]]
        p = g.plan
        lines = [f"<b>{g.label()}</b> — {len(g.frames)} frames, {g.total_exposure / 3600:.2f} h"]
        for name, kind in (("Bias", "bias"), ("Darks", "dark"), ("Flats", "flat")):
            sym, tip = self.status_cell(p, kind)
            col = {"✓": "#50c864", "⚠": "#eba932", "✗": "#e65050"}[sym]
            lines.append(f"<span style='color:{col}'>{sym}</span> <b>{name}:</b> {tip.replace(chr(10), '<br>')}")
        import targets
        typ = targets.classify(g.target)
        if typ == targets.COMET:
            lines.append("<span style='color:#9fc0ec'>☄ <b>Comet:</b> besides the usual stack (aligned on the stars), "
                         "a second stack follows the comet; the main master joins the sharp stars and the sharp "
                         "comet. If the comet barely moved, the usual stack is already sharp for both.</span>")
        elif typ:
            lines.append(f"<span style='color:#8a93a6'>Target: {targets.DESCRIBE[typ]} - Finishing Touches "
                         "recommends the steps that suit it.</span>")
        name = st.master_light_name(g, False)
        dest = Path(self.masters_root.text().strip() or "(next to the lights)") / st.safe_name(g.target)
        lines.append(f"Will save: <i>{dest}\\{name}</i>")
        self.details.setText("<br>".join(lines))

    # ------------------------------------------------------------ calibration
    def _add_cal_paths(self, paths):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            infos, problems = st.scan_paths(paths, progress=lambda t, f: (self.progress.setFormat(t),
                                                                         QApplication.processEvents()))
        finally:
            QApplication.restoreOverrideCursor()
            self.progress.setFormat("")
        cal = [i for i in infos if i.type in st.KIND_FOLDER]
        lights = [i for i in infos if i.type == "light"]
        known = {str(i.path) for i in self.added_infos}
        self.added_infos += [i for i in cal if str(i.path) not in known]
        self.log(f"Added {len(cal)} calibration files"
                 + (f"; ignored {len(lights)} light frames" if lights else "")
                 + (f"; {len(problems)} unreadable" if problems else ""))
        for p in problems[:10]:
            self.log("   " + p)
        unknown = [i for i in infos if i.type is None]
        if unknown:
            self.log(f"   {len(unknown)} files had no frame type in their header or name and were skipped "
                     f"(e.g. {unknown[0].path.name})")
        self.refresh_all()

    def add_cal_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of calibration frames",
                                             self.cal_root.text() or str(Path.home()))
        if d:
            self._add_cal_paths([d])

    def build_cal_masters(self, folder=None, modal=True):
        """The Build Calibration Masters window; the library is re-read when it closes."""
        import cal_builder_window as cbw
        o = self.options()
        dlg = cbw.CalBuilderDialog(self, self.cal_root.text().strip(), self.rigs, o.formats, o.kappa)
        self.cal_builder = dlg
        if folder:
            dlg.choose_folder(folder)
        if not modal:
            dlg.show()
            return dlg
        dlg.exec()
        if dlg.lib_edit.text().strip() and not self.cal_root.text().strip():
            self.cal_root.setText(dlg.lib_edit.text().strip())
        if dlg.built:
            self.log(f"Built {len(dlg.built)} calibration master(s) into your library")
            self.refresh_all()
        return dlg

    def add_cal_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Calibration frames or masters", self.cal_root.text() or
                                                str(Path.home()), "FITS / XISF (*.fits *.fit *.fts *.xisf)")
        if files:
            self._add_cal_paths(files)

    def search_cal_root(self):
        root = self.cal_root.text().strip()
        if not root or not st._is_dir(root):
            QMessageBox.information(self, "Stacking", "Set your calibration masters folder first.")
            return
        rigs = {g.rig for g in self.groups}
        dirs = [Path(root) / r for r in rigs if st._is_dir(Path(root) / r)]
        if not dirs:
            QMessageBox.information(self, "Stacking", "None of the rig folders for these lights exist yet in "
                                    f"{root}.\n\nRigs: {', '.join(sorted(rigs))}")
            return
        self._add_cal_paths([d for d in dirs])

    def remove_cal(self):
        ids = {it.data(Qt.UserRole) for it in self.cal_list.selectedItems()}
        drop = set()
        for c in self.items:
            if id(c) in ids and c.origin == "added":
                drop |= {str(i.path) for i in c.infos}
        if not drop:
            QMessageBox.information(self, "Stacking", "Only calibration frames you added can be removed here — "
                                    "masters in your calibration folder are always available.")
            return
        self.added_infos = [i for i in self.added_infos if str(i.path) not in drop]
        self.refresh_all()

    # ------------------------------------------------------------ run
    def start(self, skip_preflight=False):
        groups = [g for g in self.groups if g.enabled]
        if not groups:
            QMessageBox.information(self, "Stacking", "Tick at least one group to stack.")
            return
        ui = __import__("ui_header")
        if getattr(self.btn_start, "_done", False) and not skip_preflight:
            if not ui.confirm_again(self, "Stacking", "<b>These frames were already stacked</b> in this window.<br><br>"
                                    "Stack them again? The masters already saved are kept; the new ones are "
                                    "saved as <i>_2</i>, <i>_3</i>, …"):
                return
        ui.mark_ready(self.btn_start, "▶  Check && Stack")
        opts = self.options()
        if not opts.masters_root:
            QMessageBox.information(self, "Stacking", "Choose where master lights should be saved "
                                    "(Master lights folder at the top).")
            return
        if opts.save_cal_masters and not opts.cal_root and any(
                c is not None and not c.master for g in groups
                for c in g.plan.all_items()):
            if QMessageBox.question(self, "Stacking", "New calibration masters will be built, but no calibration "
                                    "masters folder is set, so they won't be saved for next time.\n\n"
                                    "Continue anyway?") != QMessageBox.StandardButton.Yes:
                return
        problems = [(g.label(), [w for _, w in g.plan.warnings]) for g in groups if g.plan.warnings]
        if problems and not skip_preflight:
            dlg = PreflightDialog(self, problems)
            r = dlg.exec()
            if r == PreflightDialog.ADD:
                self.add_cal_folder()
                return
            if r != 1:
                return
        few = [g.label() for g in groups if len(g.frames) < 5]
        if few:
            self.log("Note: fewer than 5 frames in " + "; ".join(few) +
                     " — outlier rejection needs at least 5, so these are averaged.")
        runlog = st.StackRunLog(opts)
        skip = set()
        done = [(g, runlog.done(g)) for g in groups]
        done = [(g, d) for g, d in done if d]
        if done:
            r = self.ask_resume(done, len(groups))
            if r is None:
                return
            if r:
                skip = {id(g) for g, _d in done}
            else:
                runlog.finish([g for g, _d in done])
        self._save_settings()
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log(f"Stacking {len(groups)} group(s)…" + (f" ({len(skip)} already saved, skipped)" if skip else ""))
        self.worker = StackWorker(groups, self.items, opts, self.rigs, runlog=runlog, skip=skip)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.failed.connect(self.on_failed)
        self.worker.ask_sky.connect(self.ask_sky_flat)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def ask_sky_flat(self, q):
        """The SkyFlat (Mote Buster) may not be good enough: use it anyway, stack without flats, or stop."""
        box = QMessageBox(self)
        box.setWindowTitle("SkyFlat (Mote Buster)")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"<b>The SkyFlat made for filter {q['filter']} may not be good enough to use.</b>")
        why = "\n".join(f"  • {p}" for p in q["problems"])
        box.setInformativeText(why + "\n\nA poor flat can add blotches or rings instead of removing them. "
                               "Do you want to proceed without flats for this filter?")
        b_without = box.addButton("Proceed Without Flats", QMessageBox.AcceptRole)
        b_use = box.addButton("Use the SkyFlat Anyway", QMessageBox.DestructiveRole) if q["built"] else None
        b_stop = box.addButton("Cancel Stacking", QMessageBox.RejectRole)
        box.setDefaultButton(b_without)
        box.exec()
        c = box.clickedButton()
        q["answer"] = "use" if (b_use is not None and c is b_use) else "cancel" if c is b_stop else "without"
        q["event"].set()

    def ask_resume(self, done, total):
        """True = skip the masters an interrupted run saved, False = stack all again, None = cancel."""
        names = "\n".join(f"  • {g.label()}  ({d.get('time', '')})" for g, d in done)
        box = QMessageBox(self)
        box.setWindowTitle("Continue where you left off?")
        box.setIcon(QMessageBox.Question)
        box.setText(f"An earlier stacking run didn't finish, but {len(done)} of these {total} masters were "
                    "already saved:")
        box.setInformativeText(names + "\n\nContinue with the rest, or stack everything again "
                               "(new versions are saved as _2, nothing is overwritten)?")
        b_cont = box.addButton("Continue With the Rest", QMessageBox.AcceptRole)
        b_all = box.addButton("Stack Everything Again", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(b_cont)
        if getattr(self, "auto_answer_resume", None) is not None:      # self-test
            return self.auto_answer_resume
        box.exec()
        if box.clickedButton() is b_cont:
            return True
        if box.clickedButton() is b_all:
            return False
        return None

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.progress.setFormat("Cancelling…")

    def on_progress(self, text, frac):
        self.progress.setValue(int(1000 * frac))
        self.progress.setFormat(text)

    def on_failed(self, msg):
        self.log("ERROR: " + msg)
        QMessageBox.critical(self, "Stacking", f"Stacking stopped:\n\n{msg[:800]}")

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        w = self.worker
        if w is None:
            return
        ok = [r for r in w.results if not r.error]
        bad = [r for r in w.results if r.error]
        cal_warn = []
        for r in ok:
            for line in r.log:
                if line.startswith("WARNING: the ") and " master " in line and line not in cal_warn:
                    cal_warn.append(line)
        self.last_cal_warnings = cal_warn
        if cal_warn:
            box = QMessageBox(QMessageBox.Warning, "Check your calibration frames",
                              "<b>The masters were made, but a calibration frame does not match your lights:</b>"
                              "<br><br>" + "<br><br>".join(x.replace("WARNING: ", "• ") for x in cal_warn[:4])
                              + "<br><br>This is why a picture can come out with bright bands, borders or blotches. "
                              "Remove that frame from the <i>Calibration frames</i> list (or take new ones) and "
                              "stack again.", QMessageBox.Ok, self)
            box.setWindowModality(Qt.NonModal)           # never blocks - the log keeps the details too
            self._cal_box = box
            QTimer.singleShot(300, box.open)
        self.progress.setFormat(f"Done — {len(ok)} master{'s' if len(ok) != 1 else ''} saved"
                                + (f", {len(bad)} failed" if bad else "")
                                + ("  ·  next: Continue to Image Processing" if ok else ""))
        self.btn_prepare.setEnabled(bool(ok))
        self.btn_pi.setEnabled(bool(ok))
        if ok:
            __import__("ui_header").mark_done(self.btn_start, "✓  Stacked – Run Again?")
            self.log("\nMasters are ready. Next: click “Continue to Image Processing ➜” (top right) to crop, plate solve, remove "
                     "gradients and run the XTerminators.")
        self.last_results = w.results
        self.last_saved_cal = w.saved_cal
        built_paths = set()
        for g in self.groups:              # release calibration data
            for c in g.plan.all_items():
                if c is not None:
                    c.data = None
                    if not c.master and c.saved:
                        built_paths |= {str(i.path) for i in c.infos}
                    if getattr(c, "replaced", None):     # swapped for a master saved by an earlier run
                        built_paths |= set(c.replaced)
        if built_paths:                    # use the saved library masters from now on
            self.added_infos = [i for i in self.added_infos if str(i.path) not in built_paths]
        self.refresh_all()                 # newly saved masters now come from the library

    def master_files(self):
        files = []
        for r in getattr(self, "last_results", []) or []:
            if not r.error:
                files += [f for f in r.files if Path(f).suffix.lower() in (".fits", ".xisf")
                          and not Path(f).stem.endswith(("_Comet", "_Stars"))]   # comet extras stay in the folder
        return files

    def stop_here_pixinsight(self):
        files = self.master_files()
        if not files:
            return
        import prepare as P
        try:
            masters = [P.load_master(f) for f in files]
            for group in P.target_sets(masters):
                steps = P.pixinsight_plan(group)
                note = group[0].path.parent / f"{group[0].target}_continue_in_PixInsight.txt"
                note.write_text(P.plan_text(group[0].target, steps), encoding="utf-8")
                self.log(f"Saved PixInsight steps: {note}")
        except Exception as e:  # noqa: BLE001
            self.log(f"Could not write the PixInsight note: {e}")
        import tools
        tools.reveal(files)

    def open_prepare(self):
        import prepare_window
        files = self.master_files()
        __import__("ui_header").open_page(self, "prep_win", tuple(str(f) for f in files),
                                          lambda: prepare_window.PrepareWindow(self, self.settings, files))

    def closeEvent(self, ev):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Stacking", "Stacking is still running. Cancel it and close?") \
                    != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self.worker.cancel()
            self.worker.wait(30000)
        self._save_settings()
        super().closeEvent(ev)
