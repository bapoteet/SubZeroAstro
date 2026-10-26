"""File › Organize My Astro Files… - the window for organizer.py: pick a folder (a drive or a NAS share
works too), see the plan, then copy or move, with undo."""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QDialog, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

import checkpoints as cp
import organizer as O

PREVIEW_ROWS = 3000
KIND_TEXT = {"light": "light", "flat": "flat", "darkflat": "dark flat", "dark": "dark", "bias": "bias",
             "master": "master light", "calmaster": "calibration master", "unsorted": "unsorted"}


class _Job(QThread):
    ok = Signal(object)
    failed = Signal(str)
    note = Signal(str, int, int)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop = False

    def run(self):
        try:
            self.ok.emit(self.fn(self))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1200:]}")


class OrganizeDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Organize My Astro Files")
        self.items = []
        self.job = None
        self.last_log = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Sorts your FITS and XISF files - subs and masters, lights, flats, darks, dark flats and bias - into "
            "one tidy layout, read from each file's own header: <i>Targets › target › night › LIGHT / FLAT</i>, "
            "a darks library by camera, gain, temperature and exposure, and each target's masters together. "
            "Nothing changes until you have seen the plan; every run can be undone; nothing is ever deleted. "
            "Saved projects are left alone. A NAS works too - pick its shared folder.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        def row(label, key, tip):
            h = QHBoxLayout()
            lab = QLabel(label)
            lab.setMinimumWidth(170)
            ed = QLineEdit(self.settings.value(key, "") or "")
            ed.setToolTip(tip)
            b = QPushButton("Browse…")
            b.clicked.connect(lambda: self._browse(ed))
            h.addWidget(lab)
            h.addWidget(ed, 1)
            h.addWidget(b)
            lay.addLayout(h)
            return ed, h
        self.src, _h = row("Folder to organize:", "organize/src",
                           "Every subfolder is included. A whole drive or a NAS share works.")
        self.dst, hd = row("Put the organized files in:", "organize/dst",
                           "Your astro library. Can be the same folder (organize in place).")
        self.same = QCheckBox("Same folder (organize in place)")
        self.same.setChecked(self.settings.value("organize/same", True, type=bool))
        self.same.toggled.connect(self._same_toggled)
        lay.addWidget(self.same)
        h = QHBoxLayout()
        self.copy = QRadioButton("Copy (the originals stay where they are)")
        self.move = QRadioButton("Move (the files end up only in the new place)")
        grp = QButtonGroup(self)
        grp.addButton(self.copy)
        grp.addButton(self.move)
        (self.move if self.settings.value("organize/move", False, type=bool) else self.copy).setChecked(True)
        h.addWidget(self.copy)
        h.addWidget(self.move)
        h.addStretch(1)
        lay.addLayout(h)
        h = QHBoxLayout()
        self.b_scan = QPushButton("Look Through the Files (nothing is changed)")
        self.b_scan.clicked.connect(lambda: self.scan())
        self.b_go = QPushButton("Organize")
        self.b_go.setEnabled(False)
        self.b_go.clicked.connect(lambda: self.go())
        self.b_undo = QPushButton("Undo Last Run…")
        self.b_undo.clicked.connect(lambda: self.undo_last())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setVisible(False)
        self.b_stop.clicked.connect(lambda: setattr(self.job, "stop", True) if self.job else None)
        for b in (self.b_scan, self.b_go, self.b_undo, self.b_stop):
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.status = QLabel("Pick a folder, then Look Through the Files.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "File", "Goes to", "Why"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Interactive)
        self.table.setColumnWidth(1, 260)
        self.table.setColumnWidth(3, 260)
        lay.addWidget(self.table, 1)
        b = QPushButton("Close")
        b.clicked.connect(self.close)
        hb = QHBoxLayout()
        hb.addStretch(1)
        hb.addWidget(b)
        lay.addLayout(hb)
        self.resize(1100, 680)
        self._same_toggled(self.same.isChecked())

    # ---------------------------------------------------------------- helpers
    def _browse(self, ed):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder", ed.text() or str(Path.home()))
        if d:
            ed.setText(d)
            if ed is self.src and self.same.isChecked():
                self.dst.setText(d)

    def _same_toggled(self, on):
        self.dst.setEnabled(not on)
        if on:
            self.dst.setText(self.src.text())

    def library(self) -> Path:
        return Path((self.src.text() if self.same.isChecked() else self.dst.text()).strip())

    def _busy(self, on):
        for w in (self.b_scan, self.b_go, self.b_undo, self.src, self.dst, self.same, self.copy, self.move):
            w.setEnabled(not on)
        if not on:
            self.dst.setEnabled(not self.same.isChecked())
            self.b_go.setEnabled(bool([i for i in self.items if i.status == "move"]))
        self.b_stop.setVisible(on)
        self.bar.setVisible(on)

    def _run(self, fn, done, wait=False):
        self._busy(True)
        self.job = _Job(fn)
        self.job.note.connect(self._note)
        self.job.ok.connect(lambda r: (self._busy(False), done(r)))
        self.job.failed.connect(lambda e: (self._busy(False), self.status.setText("Stopped: " + e.split("\n")[0]),
                                           QMessageBox.warning(self, "Organize My Astro Files", e[:1500])))
        self.job.start()
        if wait:                                   # the self-test runs it to the end
            while self.job.isRunning():
                QApplication.processEvents()
                self.job.wait(20)
            for _ in range(5):                     # deliver the finished signal queued from the thread
                QApplication.processEvents()

    def _note(self, text, i, n):
        self.status.setText(text)
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(min(i, max(n, 1)))

    # ---------------------------------------------------------------- scan
    def scan(self, wait=False):
        src, lib = Path(self.src.text().strip()), self.library()
        if not src.is_dir() or not str(lib):
            QMessageBox.information(self, "Organize My Astro Files", "Pick the folder to organize first.")
            return
        for k, v in (("organize/src", str(src)), ("organize/dst", self.dst.text()),
                     ("organize/same", self.same.isChecked()), ("organize/move", self.move.isChecked())):
            self.settings.setValue(k, v)

        def work(job):
            paths = O.scan(src, progress=lambda t, n: job.note.emit(t, 0, 0), stop=lambda: job.stop)
            items = O.classify(paths, progress=lambda t, i: job.note.emit(t, i, len(paths)), stop=lambda: job.stop)
            job.note.emit("Checking for duplicates and name clashes…", 0, 0)
            return O.plan(items, lib)
        self._run(work, self._show_plan, wait)

    def _show_plan(self, items):
        self.items = items
        sm = O.summary(items)
        kinds = ", ".join(f"{n} {KIND_TEXT.get(k, k)}{'s' if n != 1 and not k.endswith('s') else ''}"
                          for k, n in sorted(sm["by_kind"].items(), key=lambda kv: -kv[1]))
        parts = [f"<b>{sm['files']} file(s) to {'move' if self.move.isChecked() else 'copy'}</b> "
                 f"({cp.human(sm['bytes'])})" + (f": {kinds}" if kinds else "")]
        if sm["targets"]:
            parts.append(f"{len(sm['targets'])} target(s): " + ", ".join(sorted(sm["targets"])[:12])
                         + (" …" if len(sm["targets"]) > 12 else ""))
        if sm.get("here"):
            parts.append(f"{sm['here']} already in the right place")
        if sm.get("duplicate"):
            parts.append(f"{sm['duplicate']} duplicate(s) - left where they are, so you can check and delete them")
        if sm.get("error"):
            parts.append(f"{sm['error']} file(s) could not be read - left where they are")
        self.status.setText("<br>".join(parts) + ("<br>Check the list, then press <b>Organize</b>."
                                                  if sm["files"] else "<br>Nothing to do."))
        lib = self.library()
        order = {"move": 0, "duplicate": 1, "error": 2, "here": 3}
        rows = sorted(items, key=lambda i: (order.get(i.status, 9), str(i.dst or "")))[:PREVIEW_ROWS]
        self.table.setRowCount(len(rows))
        mark = {"move": "→", "duplicate": "≡", "error": "!", "here": "✓"}
        for r, it in enumerate(rows):
            try:
                to = str(Path(it.dst).relative_to(lib)) if it.dst else ""
            except ValueError:
                to = str(it.dst)
            if it.status in ("duplicate", "error"):
                to = "(stays where it is)"
            vals = [mark.get(it.status, ""), it.src.name, to, it.why]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 1:
                    cell.setToolTip(str(it.src))
                self.table.setItem(r, c, cell)
        self.b_go.setEnabled(sm["files"] > 0)

    # ---------------------------------------------------------------- go / undo
    def go(self, ask=True, wait=False):
        todo = [i for i in self.items if i.status == "move"]
        if not todo:
            return
        move = self.move.isChecked()
        lib = self.library()
        if ask:
            verb = "Move" if move else "Copy"
            msg = (f"{verb} {len(todo)} file(s) into {lib}?\n\n"
                   + ("Moved files are no longer in their old place (empty folders left behind are removed). "
                      if move else "The originals stay where they are. ")
                   + "You can undo this with Undo Last Run.")
            if QMessageBox.question(self, "Organize My Astro Files", msg,
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return

        def work(job):
            return O.execute(self.items, lib, move=move,
                             progress=lambda t, i, n: job.note.emit(t, i, n), stop=lambda: job.stop)

        def done(log):
            import json
            self.last_log = log
            d = json.loads(Path(log).read_text(encoding="utf-8"))
            self.status.setText(f"Done: {len(d['done'])} file(s) {'moved' if move else 'copied'}"
                                + (f"; {len(d['failed'])} could not be - see {log}" if d["failed"] else "")
                                + ". Undo Last Run puts everything back.")
            self.items = []
            self.table.setRowCount(0)
            self.b_go.setEnabled(False)
        self._run(work, done, wait)

    def undo_last(self, ask=True, wait=False):
        log = self.last_log if self.last_log and Path(self.last_log).exists() else O.last_log(self.library())
        if log is None:
            QMessageBox.information(self, "Undo Last Run", "No run to undo in this library.")
            return
        if ask and QMessageBox.question(
                self, "Undo Last Run", f"Undo the run of {Path(log).stem.replace('organize_', '')}? Moved files go "
                "back where they were; copies are removed.", QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return

        def work(job):
            return O.undo(log, progress=lambda t, i, n: job.note.emit(t, i, n))
        self._run(work, lambda r: self.status.setText(f"Undone: {r[0]} file(s) put back"
                                                      + (f", {r[1]} skipped (changed or missing since)"
                                                         if r[1] else "") + "."), wait)
        self.last_log = None
