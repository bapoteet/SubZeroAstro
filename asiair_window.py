"""
asiair_window.py - File › Fetch From My ASIAIRs…: your ASIAIRs (name, IP address, rig), Fetch Now, and fetching
every few minutes through the night so the subs are already here in the morning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import asiair as A

NO_RIG = "(no rig - use its name)"


def riglist(settings) -> list:
    try:
        return [r["name"] for r in json.loads(settings.value("organize/riglist", "[]") or "[]") if r.get("name")]
    except (ValueError, TypeError):
        return []


def default_incoming(settings) -> str:
    return (settings.value("asiair/incoming", "") or settings.value("organize/src", "") or "").strip()


class AsiairDialog(QDialog):
    def __init__(self, owner, settings):
        super().__init__(owner)
        self.owner, self.settings = owner, settings
        self.setWindowTitle("Fetch From My ASIAIRs")
        lay = QVBoxLayout(self)
        intro = QLabel(
            "No more USB sticks: SubZeroAstro copies the new subs from each ASIAIR over your network into your "
            "incoming folder, each in a folder named after its rig, so <b>Organize</b> files them under the right "
            "rig. The .jpg previews stay behind. A sub is never fetched twice, and a copy is checked before it "
            "counts.<br>Each ASIAIR's <b>IP address</b> is in the ASIAIR app (its Wi-Fi settings); a shared folder "
            "or a Mac's mounted share (Browse…) works too.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "IP address or folder", "Rig", "Last fetch"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        for text, fn in (("Add ASIAIR", lambda: self.add_row()), ("Remove", self.remove_row),
                         ("Browse for Folder…", self.browse_row), ("Test", lambda: self.test()),
                         ("Delete .jpgs on This ASIAIR…", lambda: self.delete_jpgs())):
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Incoming folder:"))
        self.incoming = QLineEdit(default_incoming(settings))
        h.addWidget(self.incoming, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_incoming)
        h.addWidget(b)
        lay.addLayout(h)
        self.free = QCheckBox("Free space on the ASIAIR: delete each sub from it once its copy is checked")
        self.free.setChecked(settings.value("asiair/free", False, type=bool))
        lay.addWidget(self.free)
        h = QHBoxLayout()
        self.night = QCheckBox("Keep fetching through the night, every")
        self.night.setToolTip("While SubZeroAstro is open (on the computer the ASIAIRs' network reaches), each new "
                              "sub comes over soon after it's taken - over Wi-Fi that spreads the slow copying over "
                              "the night, so everything is already here in the morning")
        self.every = QSpinBox()
        self.every.setRange(5, 120)
        self.every.setValue(int(settings.value("asiair/every", 10) or 10))
        self.night.setChecked(settings.value("asiair/night", False, type=bool))
        h.addWidget(self.night)
        h.addWidget(self.every)
        h.addWidget(QLabel("minutes"))
        h.addStretch(1)
        lay.addLayout(h)
        self.with_auto = QCheckBox("Fetch first whenever Organize Automatically runs")
        self.with_auto.setChecked(settings.value("asiair/with_auto", False, type=bool))
        lay.addWidget(self.with_auto)

        self.status = QLabel(settings.value("asiair/last", "") or "")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        h = QHBoxLayout()
        self.b_fetch = QPushButton("Fetch Now")
        self.b_fetch.clicked.connect(lambda: self.fetch_now())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setVisible(False)
        self.b_stop.clicked.connect(self._stop)
        b_close = QPushButton("Save && Close")
        b_close.clicked.connect(self.accept)
        h.addWidget(self.b_fetch)
        h.addWidget(self.b_stop)
        h.addStretch(1)
        h.addWidget(b_close)
        lay.addLayout(h)
        for u in A.load_units(settings):
            self.add_row(u)
        if not self.table.rowCount():
            self.add_row()
        self.job = None
        for b in self.findChildren(QPushButton):     # no orange "default" outline on the last button clicked
            b.setAutoDefault(False)
        self.resize(900, 520)

    # ---------------------------------------------------------------- rows
    def add_row(self, u=None):
        u = u or {}
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(u.get("name") or f"ASIAIR {r + 1}"))
        self.table.setItem(r, 1, QTableWidgetItem(u.get("address") or ""))
        cb = QComboBox()
        cb.addItem(NO_RIG, "")
        for n in riglist(self.settings):
            cb.addItem(n, n)
        if u.get("rig") and cb.findData(u["rig"]) < 0:
            cb.addItem(u["rig"], u["rig"])
        cb.setCurrentIndex(max(0, cb.findData(u.get("rig") or "")))
        self.table.setCellWidget(r, 2, cb)
        last = QTableWidgetItem(u.get("last") or "")
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
        d = QFileDialog.getExistingDirectory(self, "The ASIAIR's shared folder (e.g. EMMC Images)")
        if d:
            self.table.item(r, 1).setText(d)

    def units(self) -> list:
        out = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            addr = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            cb = self.table.cellWidget(r, 2)
            last = self.table.item(r, 3).text() if self.table.item(r, 3) else ""
            if addr or name:
                out.append({"name": name or addr, "address": addr, "rig": cb.currentData() if cb else "",
                            "last": last})
        return out

    def save(self):
        A.save_units(self.settings, self.units())
        self.settings.setValue("asiair/incoming", self.incoming.text().strip())
        self.settings.setValue("asiair/free", self.free.isChecked())
        self.settings.setValue("asiair/every", self.every.value())
        self.settings.setValue("asiair/with_auto", self.with_auto.isChecked())
        self.settings.setValue("asiair/night", self.night.isChecked())
        if hasattr(self.owner, "set_asiair_night"):
            self.owner.set_asiair_night(self.night.isChecked(), self.every.value())

    def accept(self):
        self.save()
        super().accept()

    def _pick_incoming(self):
        d = QFileDialog.getExistingDirectory(self, "Incoming folder", self.incoming.text() or str(Path.home()))
        if d:
            self.incoming.setText(d)

    # ---------------------------------------------------------------- test / fetch
    def test(self, row=None):
        r = self.table.currentRow() if row is None else row
        units = self.units()
        if not (0 <= r < len(units)):
            r = 0
        if not units:
            return None
        u = units[r]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            roots = A.shares(u["address"])
            subs = A.scan(roots) if roots else []
        finally:
            QApplication.restoreOverrideCursor()
        seen = A.load_manifest(A.safe_name(u["name"]))
        new = sum(1 for s in subs if A._mkey(s[1], s[2], s[3]) not in seen)
        txt = (f"<b>{u['name']}</b>: found {', '.join(str(p) for p in roots)} - {len(subs)} sub(s), {new} not "
               "fetched yet." if roots else
               f"<b>{u['name']}</b>: can't reach {u['address'] or '(no address)'}. Is it on, on the same network, "
               "and is the IP address right? (ASIAIR app › Wi-Fi settings.) Try opening \\\\<IP> in File Explorer.")
        self.status.setText(txt)
        return roots, len(subs), new

    def delete_jpgs(self, row=None, confirm=True):
        """On demand only: delete the .jpg previews on the selected ASIAIR (its subs are never touched)."""
        r = self.table.currentRow() if row is None else row
        units = self.units()
        if not units:
            return None
        u = units[r if 0 <= r < len(units) else 0]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            roots = A.shares(u["address"])
            found = A.jpgs(roots) if roots else []
        finally:
            QApplication.restoreOverrideCursor()
        if not roots:
            self.status.setText(f"<b>{u['name']}</b>: can't reach {u['address'] or '(no address)'}.")
            return None
        if not found:
            self.status.setText(f"<b>{u['name']}</b>: no .jpg files on it.")
            return 0
        if confirm and QMessageBox.question(
                self, "Delete .jpgs on the ASIAIR?",
                f"Delete these from <b>{u['name']}</b>?<br><br>{A.jpg_summary(found, roots)}<br><br>Only the .jpg "
                "pictures go - the FITS subs stay. They can't be brought back.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return None
        from organizer_window import _Job
        self.job = _Job(lambda job: A.delete_jpgs(found, progress=lambda t, i, n: job.note.emit(t, i, n),
                                                  stop=lambda: job.stop))
        self.job.note.connect(self._note)

        def done(res):
            self._busy(False)
            self.jpg_result = res
            self.status.setText(f"<b>{u['name']}</b>: {res[0]} .jpg deleted"
                                + (f", {res[1]} couldn't be deleted" if res[1] else "") + ".")
        self.job.ok.connect(done)
        self.job.failed.connect(lambda e: (self._busy(False), self.status.setText("Stopped: " + e.split("\n")[0])))
        self._busy(True)
        self.job.start()
        if not confirm:
            while self.job.isRunning():
                QApplication.processEvents()
                self.job.wait(20)
            for _ in range(5):
                QApplication.processEvents()
        return self.job

    def _stop(self):
        if self.job is not None:
            self.job.stop = True

    def fetch_now(self, wait=False):
        self.save()
        inc = self.incoming.text().strip()
        if not inc:
            QMessageBox.information(self, "Fetch From My ASIAIRs", "Choose the incoming folder first.")
            return None
        units = [u for u in self.units() if u["address"]]
        if not units:
            QMessageBox.information(self, "Fetch From My ASIAIRs", "Add an ASIAIR and its IP address first.")
            return None
        from organizer_window import _Job
        free = self.free.isChecked()
        self.job = _Job(lambda job: A.fetch_all(units, inc, progress=lambda t, i, n: job.note.emit(t, i, n),
                                                stop=lambda: job.stop, free=free))
        self.job.note.connect(self._note)
        self.job.ok.connect(self._done)
        self.job.failed.connect(lambda e: (self._busy(False), self.status.setText("Stopped: " + e.split("\n")[0])))
        self._busy(True)
        self.job.start()
        if wait:
            while self.job.isRunning():
                QApplication.processEvents()
                self.job.wait(20)
            for _ in range(5):
                QApplication.processEvents()
        return self.job

    def _busy(self, on):
        self.b_fetch.setEnabled(not on)
        self.b_stop.setVisible(on)
        self.bar.setVisible(on)

    def _note(self, text, i, n):
        self.status.setText(text)
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(i)

    def _done(self, reps):
        self._busy(False)
        self.results = reps
        stamp = time.strftime("%Y-%m-%d %H:%M")
        text = f"{stamp}: " + A.describe(reps).replace("\n", "; ")
        self.settings.setValue("asiair/last", text)
        names = {r["name"]: r for r in reps}
        for row in range(self.table.rowCount()):
            it = self.table.item(row, 0)
            r = names.get(A.safe_name(it.text() if it else ""))
            if r is not None and self.table.item(row, 3) is not None:
                self.table.item(row, 3).setText(f"{stamp}: {r['copied']} new" if not r["error"] else
                                                f"{stamp}: not reached")
        A.save_units(self.settings, self.units())
        self.status.setText(A.describe(reps).replace("\n", "<br>"))
