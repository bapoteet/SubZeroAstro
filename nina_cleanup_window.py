"""nina_cleanup_window.py - File › Clean Up My N.I.N.A. Computers…: the computers that run N.I.N.A. (each one's
shared image folder), Check (what is safe to delete - filed in your organized library, same contents - and what
stays, and why), then Delete; optionally after every Organize Automatically run."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

import nina_cleanup as NC


class NinaCleanupDialog(QDialog):
    def __init__(self, owner, settings):
        super().__init__(owner)
        self.owner, self.settings = owner, settings
        self.setWindowTitle("Clean Up My N.I.N.A. Computers")
        self.reps, self.dels, self.job = None, None, None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "N.I.N.A.'s Remote Copy copies every frame off the rig's computer but never deletes it there - the "
            "drive fills up, and frames left behind are copied again on later nights. This deletes a frame from a "
            "N.I.N.A. computer <b>only when the very same file is already filed in your organized library</b> "
            "(same name, size and contents) - copied <i>and</i> organized. Everything else stays, with the reason. "
            "Nothing is deleted until you have seen the check and press Delete.<br>Share each computer's image "
            "folder on your network (with permission to change files) and add it here, e.g. "
            "<code>\\\\EAGLE1\\NINA Images</code>.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Shared image folder", "Keep (days)", "Last clean-up"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        for text, fn in (("Add Computer", lambda: self.add_row()), ("Remove", self.remove_row),
                         ("Browse for Folder…", self.browse_row)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.lib_note = QLabel("")
        self.lib_note.setWordWrap(True)
        lay.addWidget(self.lib_note)
        keep_note = QLabel("<b>Keep (days)</b>: how long a frame stays on that computer after it is filed in your "
                           "library - set it by the computer's drive: a 500 GB drive imaging all night fills in 2-3 "
                           "nights (keep 2-3), a big one can keep a week. Frames still there are copied again by "
                           "Remote Copy each night; Organize sets those duplicates aside.")
        keep_note.setWordWrap(True)
        lay.addWidget(keep_note)
        self.with_auto = QCheckBox("Clean up after every Organize Automatically run (checks, then deletes only "
                                   "what is safe)")
        self.with_auto.setChecked(settings.value("nina_clean/with_auto", False, type=bool))
        lay.addWidget(self.with_auto)
        self.status = QLabel(settings.value("nina_clean/last", "") or "")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        h = QHBoxLayout()
        self.b_check = QPushButton("Check")
        self.b_check.setToolTip("Look at every frame on each computer: safe to delete (filed in your library, same "
                                "contents) or kept, and why. Nothing is deleted.")
        self.b_check.clicked.connect(lambda: self.check())
        self.b_delete = QPushButton("Delete Safe Frames…")
        self.b_delete.setEnabled(False)
        self.b_delete.clicked.connect(lambda: self.delete())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setVisible(False)
        self.b_stop.clicked.connect(self._stop)
        b_close = QPushButton("Save && Close")
        b_close.clicked.connect(self.accept)
        for b in (self.b_check, self.b_delete, self.b_stop):
            h.addWidget(b)
        h.addStretch(1)
        h.addWidget(b_close)
        lay.addLayout(h)
        for c in NC.load_list(settings):
            self.add_row(c)
        if not self.table.rowCount():
            self.add_row()
        self._lib_note()
        for b in self.findChildren(QPushButton):
            b.setAutoDefault(False)
        self.resize(900, 560)

    # ---------------------------------------------------------------- rows
    def add_row(self, c=None):
        c = c or {}
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(c.get("name") or f"N.I.N.A. computer {r + 1}"))
        self.table.setItem(r, 1, QTableWidgetItem(c.get("folder") or ""))
        sp = QSpinBox()
        sp.setRange(0, 90)
        sp.setValue(int(c.get("keep_days", NC.DEFAULT_KEEP) if c.get("keep_days") is not None else NC.DEFAULT_KEEP))
        sp.setToolTip("Days a frame stays on this computer after it is filed in your library (0 = delete at once)")
        self.table.setCellWidget(r, 2, sp)
        last = QTableWidgetItem(c.get("last") or "")
        last.setFlags(last.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 3, last)
        return r

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def browse_row(self):
        r = self.table.currentRow()
        if r < 0:
            r = self.add_row()
        d = QFileDialog.getExistingDirectory(self, "The N.I.N.A. computer's shared image folder")
        if d:
            self.table.item(r, 1).setText(d)

    def computers(self) -> list:
        out = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            folder = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            sp = self.table.cellWidget(r, 2)
            last = self.table.item(r, 3).text() if self.table.item(r, 3) else ""
            if folder:
                out.append({"name": name or folder, "folder": folder, "last": last,
                            "keep_days": sp.value() if sp is not None else NC.DEFAULT_KEEP})
        return out

    def _lib_note(self):
        lib = NC.library_of(self.settings)
        self.lib_note.setText(f"Frames must be filed in your organized library: <b>{lib}</b>" if lib else
                              "<b>No organized library yet</b> - set one up in File › Organize My Astro Files… first "
                              "(nothing can be checked without it).")

    def save(self):
        NC.save_list(self.settings, self.computers())
        self.settings.setValue("nina_clean/with_auto", self.with_auto.isChecked())

    def accept(self):
        self.save()
        super().accept()

    # ---------------------------------------------------------------- check / delete
    def _busy(self, on):
        self.b_check.setEnabled(not on)
        self.b_delete.setEnabled(not on and bool(self.reps and any(r["safe"] for r in self.reps)))
        self.b_stop.setVisible(on)
        self.bar.setVisible(on)

    def _note(self, text, i, n):
        self.status.setText(text)
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(i)

    def _stop(self):
        if self.job is not None:
            self.job.stop = True

    def _wait(self):
        while self.job is not None and self.job.isRunning():
            QApplication.processEvents()
            self.job.wait(20)
        for _ in range(5):
            QApplication.processEvents()

    def check(self, wait=False, now=None):
        self.save()
        comps = self.computers()
        lib = NC.library_of(self.settings)
        if not comps:
            QMessageBox.information(self, self.windowTitle(), "Add a N.I.N.A. computer's shared image folder first.")
            return None
        if not lib:
            self._lib_note()
            return None
        from organizer_window import _Job
        self.reps = None
        self.job = _Job(lambda job: NC.check(comps, lib, now=now,
                                             progress=lambda t, i, n: job.note.emit(t, i, n),
                                             stop=lambda: job.stop))
        self.job.note.connect(self._note)

        def done(reps):
            self.reps = reps
            self._busy(False)
            safe = sum(len(r["safe"]) for r in reps)
            self.status.setText(NC.describe_check(reps).replace("\n", "<br>")
                                + ("<br><b>Press Delete Safe Frames… to free the space.</b>" if safe else
                                   "<br>Nothing to delete."))
            try:
                import session_log as _sl
                _sl.write("nina clean-up", "check: " + NC.describe_check(reps).replace("\n", "; "))
            except Exception:  # noqa: BLE001
                pass
        self.job.ok.connect(done)
        self.job.failed.connect(lambda e: (self._busy(False), self.status.setText("Stopped: " + e.split("\n")[0])))
        self._busy(True)
        self.job.start()
        if wait:
            self._wait()
        return self.job

    def delete(self, confirm=True, wait=False):
        reps = [r for r in (self.reps or []) if r["safe"]]
        if not reps:
            return None
        n = sum(len(r["safe"]) for r in reps)
        gb = sum(r["bytes"] for r in reps) / 1e9
        if confirm and QMessageBox.question(
                self, self.windowTitle(),
                f"Delete <b>{n} frame(s)</b> ({gb:.1f} GB) from {', '.join(r['name'] for r in reps)}?<br><br>Each one "
                "is filed in your organized library with the same contents, so nothing is lost - the library's copy "
                "is the one you keep. They can't be brought back on the N.I.N.A. computer.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return None
        from organizer_window import _Job
        self.job = _Job(lambda job: NC.delete(reps, progress=lambda t, i, n_: job.note.emit(t, i, n_),
                                              stop=lambda: job.stop))
        self.job.note.connect(self._note)

        def done(dels):
            self.dels = dels
            self.reps = None
            self._busy(False)
            text = NC.describe_delete(dels)
            stamp = time.strftime("%Y-%m-%d %H:%M")
            self.status.setText(text.replace("\n", "<br>"))
            self.settings.setValue("nina_clean/last", f"{stamp}: " + text.replace("\n", "; "))
            by = {d["name"]: d for d in dels}
            for r in range(self.table.rowCount()):
                d = by.get(self.table.item(r, 0).text().strip() if self.table.item(r, 0) else "")
                if d is not None:
                    self.table.item(r, 3).setText(f"{stamp}: {d['deleted']} deleted ({d['bytes'] / 1e9:.1f} GB)")
            self.save()
            try:
                import session_log as _sl
                _sl.write("nina clean-up", "deleted: " + text.replace("\n", "; "))
            except Exception:  # noqa: BLE001
                pass
        self.job.ok.connect(done)
        self.job.failed.connect(lambda e: (self._busy(False), self.status.setText("Stopped: " + e.split("\n")[0])))
        self._busy(True)
        self.job.start()
        if wait:
            self._wait()
        return self.job
