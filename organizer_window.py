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
        self.b_scan = QPushButton("Step 1 – Look Through the Files (nothing is changed)")
        self.b_scan.clicked.connect(lambda: self.scan())
        self.b_go = QPushButton("Step 3 – Organize")
        self.b_go.setEnabled(False)
        self.b_go.setToolTip("Lights up after Step 1, once you have seen the plan")
        self.b_go.clicked.connect(lambda: self.go())
        self.b_dups = QPushButton("Step 2 – Duplicates… (if any)")
        self.b_dups.setEnabled(False)
        self.b_dups.setToolTip("See the duplicates found, and keep them, move them to a Duplicates folder or "
                               "delete them")
        self.b_dups.clicked.connect(lambda: self.open_duplicates())
        self.b_undo = QPushButton("Undo Last Run…")
        self.b_undo.clicked.connect(lambda: self.undo_last())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setVisible(False)
        self.b_stop.clicked.connect(lambda: setattr(self.job, "stop", True) if self.job else None)
        for b in (self.b_scan, self.b_dups, self.b_go, self.b_undo, self.b_stop):
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.status = QLabel("Pick a folder, then Step 1 – Look Through the Files. Step 2 deals with any duplicates it finds; Step 3 – Organize lights up once you have seen the plan.")
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
        for w in (self.b_scan, self.b_go, self.b_dups, self.b_undo, self.src, self.dst, self.same, self.copy, self.move):
            w.setEnabled(not on)
        if not on:
            self.dst.setEnabled(not self.same.isChecked())
            self.b_go.setEnabled(bool([i for i in self.items if i.status == "move"]))
            self.b_dups.setEnabled(bool(O.duplicates(self.items)))
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
            items = O.classify(paths, progress=lambda t, i: job.note.emit(t, i, len(paths)), stop=lambda: job.stop,
                               library=lib)
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
            parts.append(f"{sm['duplicate']} duplicate(s) - left where they are; press <b>Step 2 – Duplicates…</b> to "
                         "keep, set aside or delete them")
        if sm.get("error"):
            parts.append(f"{sm['error']} file(s) could not be read - left where they are")
        self.status.setText("<br>".join(parts) + ("<br>Check the list, deal with any duplicates (<b>Step 2</b>), then press <b>Step 3 – Organize</b>."
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
        self.b_dups.setEnabled(bool(O.duplicates(items)))

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
            self.b_dups.setEnabled(False)
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

    # ---------------------------------------------------------------- duplicates
    def open_duplicates(self):
        dups = O.duplicates(self.items)
        if not dups:
            return None
        d = DuplicatesDialog(self, dups)
        d.show()
        return d

    def handle_duplicates(self, dups, delete, wait=False):
        lib = self.library()

        def work(job):
            return O.handle_duplicates(dups, lib, delete=delete, progress=lambda t, i, n: job.note.emit(t, i, n),
                                       stop=lambda: job.stop)

        def done(r):
            n, kept_back, log = r
            if not delete:
                self.last_log = log
            msg = (f"{n} duplicate(s) {'deleted' if delete else f'moved to the {O.DUP_DIR} folder (Undo Last Run puts them back)'}.")
            if kept_back:
                msg += f" {len(kept_back)} kept: " + "; ".join(f"{i.src.name} - {why}" for i, why in kept_back[:5]) \
                       + (" …" if len(kept_back) > 5 else "")
            self.status.setText(msg)
            self.b_dups.setEnabled(bool(O.duplicates(self.items)))
        self._run(work, done, wait)


class DuplicatesDialog(QDialog):
    """The duplicates found: tick the ones to deal with, then keep, set aside or delete them. Deleting only
    removes a file that is byte for byte the same as the copy that is kept."""

    def __init__(self, owner: OrganizeDialog, dups):
        super().__init__(owner)
        self.owner, self.dups = owner, dups
        self.setWindowTitle("Duplicates")
        lay = QVBoxLayout(self)
        lab = QLabel(f"<b>{len(dups)} duplicate(s)</b> - the same frame saved twice. The copy in the <i>Kept</i> "
                     "column stays (it is the one organized). Tick the duplicates to deal with. <b>Delete</b> "
                     "checks every byte first and only deletes a file that is exactly the same as its kept copy; "
                     "anything that differs is left for you to look at. Not sure? <b>Move to a Duplicates "
                     "Folder</b> sets them aside (Undo puts them back) and you can delete that folder later.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        self.table = QTableWidget(len(dups), 3)
        self.table.setHorizontalHeaderLabels(["Duplicate", "Kept", "Size"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for r, it in enumerate(dups):
            c = QTableWidgetItem(it.src.name)
            c.setFlags(c.flags() | Qt.ItemIsUserCheckable)
            c.setCheckState(Qt.Checked)
            c.setToolTip(str(it.src))
            k = QTableWidgetItem(str(it.same_as.name))
            k.setToolTip(str(it.same_as))
            self.table.setItem(r, 0, c)
            self.table.setItem(r, 1, k)
            self.table.setItem(r, 2, QTableWidgetItem(cp.human(it.size)))
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        for text, slot in (("Tick All", lambda: self._tick(True)), ("Untick All", lambda: self._tick(False))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        self.b_keep = QPushButton("Keep Them")
        self.b_keep.clicked.connect(self.close)
        self.b_aside = QPushButton("Move to a Duplicates Folder")
        self.b_aside.clicked.connect(lambda: self.act(False))
        self.b_delete = QPushButton("Delete Ticked Duplicates…")
        self.b_delete.clicked.connect(lambda: self.act(True))
        for b in (self.b_keep, self.b_aside, self.b_delete):
            h.addWidget(b)
        lay.addLayout(h)
        self.resize(820, 460)

    def _tick(self, on):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def ticked(self):
        return [it for r, it in enumerate(self.dups) if self.table.item(r, 0).checkState() == Qt.Checked]

    def act(self, delete, ask=True, wait=False):
        todo = self.ticked()
        if not todo:
            return
        if delete and ask and QMessageBox.question(
                self, "Delete Duplicates",
                f"Delete {len(todo)} duplicate file(s) ({cp.human(sum(i.size for i in todo))})?\n\nEach one is "
                "compared byte for byte with its kept copy first; only exact copies are deleted. This can't be "
                "undone.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.close()
        self.owner.handle_duplicates(todo, delete, wait)
