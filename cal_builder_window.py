"""
cal_builder_window.py - the "Build Calibration Masters" window on Calibrate and Stack: pick a folder of
bias / dark / flat / dark-flat subframes (mixed together is fine), see how they were sorted, and build
every master into the calibration library - no light frames needed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QVBoxLayout)

import cal_builder as cb
import stacking as st
import ui_header


class _Work(QThread):
    progress = Signal(str, float)
    log = Signal(str)
    done = Signal(object, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def run(self):
        try:
            self.done.emit(self.fn(self), "")
        except st.Cancelled:
            self.done.emit(None, "Cancelled.")
        except Exception as e:  # noqa: BLE001 - shown in the window, never a crash
            self.done.emit(None, f"{type(e).__name__}: {e}")


class CalBuilderDialog(QDialog):
    def __init__(self, parent=None, cal_root="", rigs=None, formats=("fits",), kappa=3.0):
        super().__init__(parent)
        self.setWindowTitle("Build Calibration Masters")
        self.setMinimumSize(860, 640)
        self.rigs, self.formats, self.kappa = rigs, formats, kappa
        self.folders, self.infos, self.jobs, self.library = [], [], [], []
        self.work = None
        self.built = []                          # jobs saved in this window
        lay = QVBoxLayout(self)
        intro = QLabel("Put your <b>bias, dark, flat and dark-flat subframes</b> in one folder (mixed together or "
                       "in subfolders). SubZeroAstro sorts them by frame type, camera, exposure, gain, offset, "
                       "temperature and filter, builds a master of each set, checks it, and saves it to your "
                       "calibration library - where stacking finds it automatically. No light frames needed.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        frow = QHBoxLayout()
        self.folder_label = QLabel("<i>no folder chosen</i>")
        self.folder_label.setWordWrap(True)
        b_add = QPushButton("Choose Folder…")
        b_add.clicked.connect(self.choose_folder)
        frow.addWidget(self.folder_label, 1)
        frow.addWidget(b_add)
        form.addRow("Calibration subframes:", frow)
        lrow = QHBoxLayout()
        self.lib_edit = QLineEdit(cal_root or "")
        self.lib_edit.setPlaceholderText("your calibration library folder")
        self.lib_edit.editingFinished.connect(self.fill_rigs)
        b_lib = QPushButton("Browse…")
        b_lib.clicked.connect(self.choose_library)
        lrow.addWidget(self.lib_edit, 1)
        lrow.addWidget(b_lib)
        form.addRow("Save masters into:", lrow)
        self.rig = QComboBox()
        self.rig.setEditable(True)
        self.rig.setToolTip("The telescope (rig) these were taken with. Flats only fit lights from the same rig.")
        form.addRow("Telescope (rig):", self.rig)
        lay.addLayout(form)
        self.summary = QLabel("Choose the folder with your calibration subframes.")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        lay.addWidget(self.progress)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(QFont("Menlo" if ui_header._MAC else "Consolas", 9))
        self.out.setMaximumHeight(170)
        lay.addWidget(self.out)
        bottom = QHBoxLayout()
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(self.cancel)
        bottom.addWidget(self.b_cancel)
        bottom.addStretch(1)
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.accept)
        bottom.addWidget(self.b_close)
        self.b_build = QPushButton("▶  Build Masters")
        self.b_build.setStyleSheet(ui_header.RUN_STYLE)
        self.b_build.setEnabled(False)
        self.b_build.clicked.connect(self.build)
        bottom.addWidget(self.b_build)
        lay.addLayout(bottom)
        self.fill_rigs()

    # ------------------------------------------------------------ setup
    def fill_rigs(self, suggestion=None):
        cur = suggestion or self.rig.currentText()
        self.rig.clear()
        root = Path(self.lib_edit.text().strip()) if self.lib_edit.text().strip() else None
        names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root and root.is_dir() else []
        if cur and cur not in names:
            names.insert(0, cur)
        self.rig.addItems(names)
        if cur:
            self.rig.setCurrentText(cur)

    def choose_library(self):
        d = QFileDialog.getExistingDirectory(self, "Calibration library folder", self.lib_edit.text() or str(Path.home()))
        if d:
            self.lib_edit.setText(d)
            self.fill_rigs()

    def choose_folder(self, folder=None):
        d = folder or QFileDialog.getExistingDirectory(self, "Folder of calibration subframes",
                                                      self.lib_edit.text() or str(Path.home()))
        if not d:
            return
        self.folders = [d]
        self.folder_label.setText(d)
        lib_root = self.lib_edit.text().strip()

        def scan(w):
            infos, problems = st.scan_paths(self.folders, progress=lambda t, f: w.progress.emit(t, f))
            library = st.scan_library(lib_root) if lib_root else []
            return infos, problems, library
        self._start(scan, self._scanned)

    def _scanned(self, res, err):
        if err or res is None:
            self.summary.setText(f"<span style='color:#e65050'>Could not read the folder: {err}</span>")
            return
        self.infos, problems, self.library = res
        rig = cb.suggest_rig(self.folders, self.infos, self.rigs)
        self.library = [c for c in self.library if (c.rig or "").lower() == rig.lower()]
        self.fill_rigs(rig)
        self.jobs, skipped = cb.plan(self.infos, self.library)
        self.list.clear()
        for j in self.jobs:
            it = QListWidgetItem(j.label() + ("".join(f"\n      Note: {n}" for n in j.notes))
                                 + ("".join(f"\n      ⚠ {w}" for w in j.warnings)))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.list.addItem(it)
        lights = sum(1 for i in skipped if i.type == "light")
        masters = sum(1 for i in self.infos if i.master)
        other = len(skipped) - lights
        bits = [f"<b>{len(self.jobs)} master{'s' if len(self.jobs) != 1 else ''} to build</b> from "
                f"{len(self.infos) - len(skipped) - masters} subframes"]
        if masters:
            bits.append(f"{masters} existing master{'s' if masters != 1 else ''} (used to calibrate flats only)")
        if lights:
            bits.append(f"{lights} light frame{'s' if lights != 1 else ''} ignored")
        if other:
            bits.append(f"{other} file{'s' if other != 1 else ''} of unknown type ignored (no IMAGETYP in the header, "
                        "and neither the file nor its folder name says bias, dark or flat)")
        if problems:
            bits.append(f"{len(problems)} unreadable")
        self.summary.setText(" · ".join(bits) + ". Untick any you don't want.")
        self.b_build.setEnabled(bool(self.jobs))
        for p in problems[:20]:
            self.out.appendPlainText(f"Could not read {p}")

    # ------------------------------------------------------------ build
    def build(self):
        root = self.lib_edit.text().strip()
        rig = st.safe_name(self.rig.currentText().strip() or "My scope")
        if not root:
            QMessageBox.information(self, "Build Calibration Masters", "Choose the calibration library folder to "
                                    "save the masters into first.")
            return
        chosen = [j for k, j in enumerate(self.jobs) if self.list.item(k).checkState() == Qt.Checked]
        if not chosen:
            return
        # a flat's bias / dark-flats are built even when unticked - but not saved twice
        lib = self.library

        def run(w):
            return cb.build(chosen, root, rig, self.formats, self.kappa,
                            progress=lambda t, f: w.progress.emit(t, f), cancel=lambda: w.cancelled,
                            log=lambda s: w.log.emit(s), library=lib)
        self.out.appendPlainText(f"Building {len(chosen)} master(s) into {Path(root) / rig} …")
        self._start(run, self._built)

    def _built(self, jobs, err):
        if err:
            self.out.appendPlainText(err)
            return
        self.built += [j for j in jobs if j.files]
        bad = [j for j in jobs if j.error]
        warn = [j for j in jobs if j.warnings]
        text = f"{sum(1 for j in jobs if j.files)} master(s) saved to your calibration library."
        detail = ""
        if warn:
            detail += "Check these before you use them:\n\n" + "\n\n".join(
                f"• {j.label()}\n   " + "\n   ".join(j.warnings) for j in warn)
        if bad:
            detail += ("\n\n" if detail else "") + "Not built:\n" + "\n".join(f"• {j.label()}: {j.error}" for j in bad)
        box = QMessageBox(self)
        box.setWindowTitle("Build Calibration Masters")
        box.setIcon(QMessageBox.Warning if (warn or bad) else QMessageBox.Information)
        box.setText(text)
        box.setInformativeText(detail or "They are used automatically when you stack lights with matching settings.")
        box.setModal(False)
        box.show()
        self._box = box

    # ------------------------------------------------------------ worker plumbing
    def _start(self, fn, then):
        if self.work is not None and self.work.isRunning():
            return
        self.b_build.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.work = _Work(fn)
        self.work.progress.connect(lambda t, f: (self.progress.setValue(int(1000 * f)), self.progress.setFormat(t)))
        self.work.log.connect(self.out.appendPlainText)

        def finished(res, err):
            self.b_cancel.setEnabled(False)
            self.b_build.setEnabled(bool(self.jobs))
            then(res, err)
        self.work.done.connect(finished)
        self.work.start()

    def busy(self):
        return self.work is not None and self.work.isRunning()

    def cancel(self):
        if self.work is not None:
            self.work.cancelled = True

    def done(self, r):
        if self.busy():
            if QMessageBox.question(self, "Build Calibration Masters", "Masters are still being built. Stop and "
                                    "close?") != QMessageBox.StandardButton.Yes:
                return
            self.work.cancelled = True
            self.work.wait(30000)
        super().done(r)
