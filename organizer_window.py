"""File › Organize My Astro Files… - the window for organizer.py: pick a folder (a drive or a NAS share
works too), see the plan, then copy or move, with undo."""
from __future__ import annotations

import re
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

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
        self._last_note = 0.0
        self.note.connect(self._log_note)

    def _log_note(self, text, i, n):
        # the session log on disk gets the progress now and then - after a freeze it shows how far it got
        now = time.time()
        if now - self._last_note >= 60:
            self._last_note = now
            _slog(f"{text}" + (f"  ({i:,} of {n:,})" if n else ""))

    def run(self):
        t0 = time.time()
        _slog(f"started: {getattr(self.fn, '__name__', 'job')}")
        try:
            res = self.fn(self)
            _slog(f"finished in {time.time() - t0:.0f} s")
            self.ok.emit(res)
        except Exception as e:  # noqa: BLE001
            _slog(f"failed after {time.time() - t0:.0f} s: {e}\n{traceback.format_exc()[-1200:]}")
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1200:]}")


def _slog(text):
    try:
        import session_log
        session_log.write("organizer", text)
    except Exception:  # noqa: BLE001
        pass


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
            "one tidy layout, read from each file's own header (target, rig, night, filter, camera settings), "
            "so file names and the current mess don't matter. Pick the layout below. Nothing changes until you "
            "have seen the plan; every run can be undone; only exact duplicates can be deleted, and only when "
            "you say so. Saved projects are left alone. A NAS works too - pick its shared folder.")
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
        self.src, _h = row("Folders / files to organize:", "organize/src",
                           "Browse… picks one or several folders (Shift-click for a run of them, Ctrl/⌘-click for "
                           "separate ones); Add Files… picks single files. Every subfolder is included. A whole "
                           "drive or a NAS share works too.")
        bf = QPushButton("Add Files…")
        bf.setToolTip("Pick individual FITS / XISF files (Shift-click for a run, Ctrl/⌘-click for separate ones)")
        bf.clicked.connect(lambda: self._add_files())
        _h.addWidget(bf)
        self.dst, hd = row("Put the organized files in:", "organize/dst",
                           "Your astro library. Can be the same folder (organize in place).")
        self.same = QCheckBox("Same folder (organize in place) - untick to put the organized files somewhere else")
        self.same.setChecked(self.settings.value("organize/same", True, type=bool))
        self.same.toggled.connect(self._same_toggled)
        lay.addWidget(self.same)
        h = QHBoxLayout()
        h.addWidget(QLabel("Layout:"))
        self.layout_box = QComboBox()
        for key, text in O.LAYOUTS.items():
            self.layout_box.addItem(text, key)
        cur = self.settings.value("organize/layout", "rig") or "rig"
        self.layout_box.setCurrentIndex(max(0, self.layout_box.findData(cur)))
        self.layout_box.setToolTip("By rig: everything one telescope (and its camera) took, together - lights by "
                                   "target and night, its own flats, darks and bias.\nBy target: all of one "
                                   "object's nights together, darks and bias in libraries per rig.")
        h.addWidget(self.layout_box, 1)
        lay.addLayout(h)
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
        self.b_rigs = QPushButton("Your Rigs…")
        self.b_rigs.setToolTip("Your telescopes and the focal length(s) each one writes in the FITS header - set "
                               "up once; every file is matched to its rig by focal length (±3 %). Add New Rig when "
                               "you get another one.")
        self.b_rigs.clicked.connect(lambda: self.edit_rigs())
        self.b_which = QPushButton("Which Rig?…")
        self.b_which.setEnabled(False)
        self.b_which.setToolTip("Files whose focal length fits more than one rig, or whose header has no focal "
                                "length: you say which rig, by folder - remembered")
        self.b_which.clicked.connect(lambda: self.which_rig())
        self.b_dups = QPushButton("Step 2 – Duplicates… (if any)")
        self.b_dups.setEnabled(False)
        self.b_dups.setToolTip("See the duplicates found, and keep them, move them to a Duplicates folder or "
                               "delete them")
        self.b_dups.clicked.connect(lambda: self.open_duplicates())
        self.b_safe = QPushButton("Safe to Delete?…")
        self.b_safe.setToolTip("After organizing: checks every file still in the folders you organized - is each one "
                               "safely in Organized Astro Files (byte for byte)? Reads only, deletes nothing.")
        self.b_safe.clicked.connect(lambda: self.safe_to_delete())
        self.b_undo = QPushButton("Undo Last Run…")
        self.b_undo.clicked.connect(lambda: self.undo_last())
        self.b_stop = QPushButton("Stop")
        self.b_stop.setVisible(False)
        self.b_stop.clicked.connect(lambda: setattr(self.job, "stop", True) if self.job else None)
        for b in (self.b_rigs, self.b_scan, self.b_which, self.b_dups, self.b_go, self.b_safe, self.b_undo,
                  self.b_stop):
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.status = QLabel("Start with <b>Your Rigs…</b> (once): name each telescope and the focal length it writes "
                             "in your files. Then pick a folder and Step 1 – Look Through the Files; Step 2 deals with "
                             "any duplicates; Step 3 – Organize lights up once you have seen the plan.")
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
        self._organized = False
        self._labels = {b: b.text() for b in (self.b_rigs, self.b_scan, self.b_which, self.b_dups, self.b_go)}
        last = self.settings.value("organize/last_done", "") or ""
        if last:
            self.status.setText(self.status.text() + f"<br><i>Last time: {last}</i>")
        self._update_steps()

    def _update_steps(self):
        """Where you are: steps done get a ✓ and go grey, the next one is highlighted - also after a break."""
        rigs_done = bool(self.riglist())
        scanned = bool(self.items)
        issues = [g for g in O.rig_issues(self.items) if g["reason"] != "nomatch"] if scanned else []
        dups = bool(O.duplicates(self.items)) if scanned else False
        tomove = bool([i for i in self.items if i.status == "move"]) if scanned else False
        done = {self.b_rigs: rigs_done, self.b_scan: scanned or self._organized,
                self.b_which: scanned and not issues, self.b_dups: scanned and not dups,
                self.b_go: self._organized}
        if not rigs_done:
            nxt = self.b_rigs
        elif not scanned:
            nxt = self.b_scan
        elif issues:
            nxt = self.b_which
        elif dups:
            nxt = self.b_dups
        elif tomove:
            nxt = self.b_go
        else:
            nxt = None
        for b, label in self._labels.items():
            if b is nxt:
                b.setText("▶ " + label)
                b.setStyleSheet("QPushButton { font-weight: bold; border: 2px solid #3d8be0; border-radius: 4px; "
                                "padding: 3px 10px; }")
            elif done.get(b) and (b in (self.b_rigs, self.b_scan, self.b_go) or scanned):
                b.setText("✓ " + label)
                b.setStyleSheet("QPushButton { color: gray; }")
            else:
                b.setText(label)
                b.setStyleSheet("")

    # ---------------------------------------------------------------- helpers
    def _browse(self, ed):
        start = (self.sources() or [Path.home()])[0] if ed is self.src else Path(ed.text() or Path.home())
        if ed is self.src:
            # several folders at once: Shift-click a run, Ctrl/⌘-click separate ones (Qt's own dialog allows it)
            from PySide6.QtWidgets import QAbstractItemView as _AIV, QListView, QTreeView
            dlg = QFileDialog(self, "Choose one or more folders", str(start if Path(start).is_dir() else
                                                                     Path(start).parent))
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.DontUseNativeDialog, True)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            for v in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
                v.setSelectionMode(_AIV.ExtendedSelection)
            if not dlg.exec():
                return
            picks = [p for p in dlg.selectedFiles() if p]
            if not picks:
                return
            self.set_sources(picks)
            return
        d = QFileDialog.getExistingDirectory(self, "Choose a folder", str(start))
        if d:
            import os
            ed.setText(os.path.normpath(d))
            if ed is self.dst and self.same.isChecked():
                self.same.setChecked(False)        # picking a destination means: not the same folder

    def _add_files(self):
        start = (self.sources() or [Path.home()])[0]
        files, _ = QFileDialog.getOpenFileNames(self, "Choose files to organize",
                                                str(start if Path(start).is_dir() else Path(start).parent),
                                                "FITS / XISF (*.fits *.fit *.fts *.xisf)")
        if files:
            self.set_sources(self.sources() + [Path(f) for f in files] if self.sources() else files)

    def sources(self) -> list:
        """The folders and files picked (separated by ';' in the box)."""
        return [Path(x.strip()) for x in (self.src.text() or "").split(";") if x.strip()]

    def set_sources(self, paths):
        seen = []
        for p in paths:
            if str(p) not in [str(q) for q in seen]:
                seen.append(Path(p))
        import os
        self.src.setText("; ".join(os.path.normpath(str(p)) for p in seen))
        if self.same.isChecked():
            self.dst.setText(str(self._common()))

    def _common(self) -> Path:
        """The folder holding everything picked (a folder itself when only one folder is picked)."""
        import os
        srcs = self.sources()
        if not srcs:
            return Path("")
        if len(srcs) == 1 and srcs[0].is_dir():
            return srcs[0]
        try:
            return Path(os.path.commonpath([str(p if p.is_dir() else p.parent) for p in srcs]))
        except ValueError:
            return srcs[0].parent

    def _same_toggled(self, on):
        self.dst.setEnabled(not on)
        if on:
            if self.dst.text().strip() and Path(self.dst.text().strip()) != self._common():
                self._own_dst = self.dst.text()            # keep the folder you picked for when you untick
            self.dst.setText(str(self._common()))
        elif getattr(self, "_own_dst", ""):
            self.dst.setText(self._own_dst)

    def library(self) -> Path:
        return Path(str(self._common()) if self.same.isChecked() else self.dst.text().strip())

    def _busy(self, on):
        for w in (self.b_scan, self.b_rigs, self.b_which, self.b_go, self.b_dups, self.b_safe, self.b_undo, self.src,
                  self.dst,
                  self.same,
                  self.copy, self.move,
                  self.layout_box):
            w.setEnabled(not on)
        if not on:
            self.dst.setEnabled(not self.same.isChecked())
            self.b_go.setEnabled(bool([i for i in self.items if i.status == "move"]))
            self.b_dups.setEnabled(bool(O.duplicates(self.items)))
            self.b_which.setEnabled(bool([g for g in O.rig_issues(self.items) if g["reason"] != "nomatch"]))
        self.b_stop.setVisible(on)
        self.bar.setVisible(on)
        if not on and hasattr(self, "_labels"):
            self._update_steps()

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
        srcs = [p for p in self.sources() if p.exists()]
        src, lib = self._common(), self.library()
        if not srcs or not str(lib) or str(lib) == ".":
            QMessageBox.information(self, "Organize My Astro Files", "Pick the folders or files to organize first.")
            return
        for k, v in (("organize/src", self.src.text()), ("organize/dst", self.dst.text()),
                     ("organize/same", self.same.isChecked()), ("organize/move", self.move.isChecked()),
                     ("organize/layout", self.layout_box.currentData())):
            self.settings.setValue(k, v)

        riglist, tol = self.riglist(), self.tol()
        if not riglist:
            QMessageBox.information(self, "Organize My Astro Files", "First tell SubZeroAstro about your rigs - "
                                    "each telescope's name and the focal length it writes in your files. "
                                    "Find My Rigs in My Files fills most of it in for you.")
            if not self.edit_rigs():
                return
            riglist = self.riglist()
        layout = self.layout_box.currentData() or "rig"
        self._plan_args = (lib, layout, src)

        def work(job):
            paths = []
            O.SKIPPED.clear()
            O.scan._keep_skipped = True
            for sp in srcs:
                if sp.is_dir():
                    paths += O.scan(sp, progress=lambda t, n: job.note.emit(t, 0, 0), stop=lambda: job.stop)
                elif sp.suffix.lower() in O.EXTS:
                    paths.append(sp)
            O.scan._keep_skipped = False
            paths = list(dict.fromkeys(paths))
            _slog(f"Step 1: {len(paths):,} files found in {', '.join(str(x) for x in srcs)}")
            items = O.classify(paths, progress=lambda t, i: job.note.emit(t, i, len(paths)), stop=lambda: job.stop,
                               library=lib, layout=layout, root=src, riglist=riglist, tol=tol)
            job.note.emit("Checking for duplicates and name clashes…", 0, 0)
            return O.plan(items, lib)
        self._run(work, self._show_plan, wait)

    def _show_plan(self, items):
        self.items = items
        self._organized = False
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
        if O.SKIPPED:
            parts.append(f"{len(O.SKIPPED)} folder(s) not looked into on purpose (saved projects, PixInsight/Siril "
                         "work folders, hidden folders) - their files stay where they are")
        self.status.setText("<br>".join(parts) + ("<br>Check the list, deal with any duplicates (<b>Step 2</b>), then press <b>Step 3 – Organize</b>."
                                                  if sm["files"] else "<br>Nothing to do."))
        lib = self.library()
        order = {"move": 0, "norig": 1, "duplicate": 2, "error": 3, "here": 4}
        rows = sorted(items, key=lambda i: (order.get(i.status, 9), str(i.dst or "")))[:PREVIEW_ROWS]
        self.table.setRowCount(len(rows))
        mark = {"move": "→", "duplicate": "≡", "error": "!", "here": "✓", "norig": "?"}
        for r, it in enumerate(rows):
            try:
                to = str(Path(it.dst).relative_to(lib)) if it.dst else ""
            except ValueError:
                to = str(it.dst)
            if it.status in ("duplicate", "error", "norig"):
                to = "(stays where it is)"
            vals = [mark.get(it.status, ""), it.src.name, to, it.why]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 1:
                    cell.setToolTip(str(it.src))
                if c >= 2:
                    cell.setToolTip(v)                     # the full text when the column is too narrow
                self.table.setItem(r, c, cell)
        self.b_go.setEnabled(sm["files"] > 0)
        self.b_dups.setEnabled(bool(O.duplicates(items)))
        issues = O.rig_issues(items, (getattr(self, "_plan_args", None) or (None, None, None))[2])
        self.b_which.setEnabled(bool([g for g in issues if g["reason"] != "nomatch"]))
        if issues:
            nm = [g for g in issues if g["reason"] == "nomatch"]
            other = [g for g in issues if g["reason"] != "nomatch"]
            msg = []
            if nm:
                fls = sorted({f"{g['focal']:.0f} mm" for g in nm if g["focal"]})
                msg.append(f"<b>{sum(g['files'] for g in nm)} file(s) match none of your rigs</b> ("
                           + ", ".join(fls[:6]) + ") - press <b>Your Rigs…</b> to add them")
            if other:
                msg.append(f"<b>{sum(g['files'] for g in other)} file(s) need you to say which rig</b> (a focal "
                           "length that fits two rigs, or none in the header) - press <b>Which Rig?…</b>")
            self.status.setText(self.status.text() + "<br>" + "<br>".join(msg)
                                + "<br>Those files stay where they are until then.")
        self._update_steps()

    # ---------------------------------------------------------------- go / undo
    def go(self, ask=True, wait=False):
        todo = [i for i in self.items if i.status == "move"]
        if not todo:
            return
        move = self.move.isChecked()
        lib = self.library()
        if ask:
            verb = "Move" if move else "Copy"
            first_move = move and not self.settings.value("organize/moved_before", False, type=bool)
            msg = (("This is your first Move. Have you tried these folders with Copy first and checked the "
                    "result in Organized Astro Files? If not, choose No, switch to Copy and try that first.\n\n"
                    if first_move else "")
                   + f"{verb} {len(todo)} file(s) into {lib}?\n\n"
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
            left = O.left_behind(self.items)
            rows = []
            for why, its in left.items():
                rows += [(it.src, why + (f" - {it.why}" if it.status == "norig" else "")) for it in its]
            rows += [(Path(f["src"]), f"could not be {'moved' if move else 'copied'}: {f.get('error', '')}")
                     for f in d["failed"]]
            nleft = len(rows)
            msg = [f"<b>Done: {len(d['done'])} file(s) {'moved' if move else 'copied'}</b> into "
                   f"<b>{O.org_base(lib)}</b>. Undo Last Run puts everything back."]
            if nleft:
                parts = [f"{len(v)} {k}" for k, v in left.items()]
                if d["failed"]:
                    parts.append(f"{len(d['failed'])} could not be {'moved' if move else 'copied'}")
                msg.append(f"<b>{nleft} file(s) were left where they are</b> - listed below with the reason: "
                           + "; ".join(parts) + ".")
            if O.SKIPPED:
                msg.append(f"{len(O.SKIPPED)} folder(s) were not looked into on purpose: "
                           + "; ".join(f"{Path(p).name} ({w})" for p, w in O.SKIPPED[:4])
                           + (" …" if len(O.SKIPPED) > 4 else ""))
            self.status.setText("<br>".join(msg))
            self.items = []
            self.table.setRowCount(min(len(rows), PREVIEW_ROWS))
            for r, (pth, why) in enumerate(rows[:PREVIEW_ROWS]):
                for c, v in enumerate(("!", pth.name, f"(still in {pth.parent})", why)):
                    cell = QTableWidgetItem(v)
                    if c == 1:
                        cell.setToolTip(str(pth))
                    if c >= 2:
                        cell.setToolTip(v)                 # the full text when the column is too narrow
                    self.table.setItem(r, c, cell)
            self.b_go.setEnabled(False)
            self.b_dups.setEnabled(False)
            self._organized = True
            if move:
                self.settings.setValue("organize/moved_before", True)
            import time as _time
            srcs = self.sources()
            self.settings.setValue("organize/last_done",
                                   f"{_time.strftime('%b %d %H:%M')} - {len(d['done'])} file(s) "
                                   f"{'moved' if move else 'copied'} from {srcs[0] if len(srcs) == 1 else f'{len(srcs)} folders'}"
                                   f" into {O.org_base(lib)}")
            self._update_steps()
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
            self._update_steps()
        self._run(work, done, wait)


    # ---------------------------------------------------------------- first time
    def show_intro(self):
        """The first time: try it on a little, with Copy, before moving anything - must be clicked through."""
        box = QMessageBox(self)
        box.setWindowTitle("Organize My Astro Files - before you start")
        box.setIcon(QMessageBox.Information)
        box.setText("<b>Try it on a small piece first, with Copy.</b>")
        box.setInformativeText(
            "1. <b>Your Rigs…</b> - name each telescope and the focal length it writes in your files (once).<br>"
            "2. Pick <b>one or two folders</b> (subfolders are included) and choose <b>Copy</b> - your originals "
            "stay where they are.<br>"
            "3. <b>Step 1</b>, check the plan, <b>Step 3</b>, then look through <b>Organized Astro Files</b>.<br>"
            "4. Happy with it? <b>Undo Last Run</b> (it removes the copies), switch to <b>Move</b>, and run it "
            "again.<br>"
            "5. Before clearing out old folders, press <b>Safe to Delete?</b><br><br>"
            "Every run can be undone; nothing is ever deleted unless you delete it.")
        cb = QCheckBox("Don't show this again")
        box.setCheckBox(cb)
        box.addButton("I Understand", QMessageBox.AcceptRole)
        box.exec()
        if cb.isChecked():
            self.settings.setValue("organize/intro_ok", True)
        self.copy.setChecked(True)                 # start on Copy

    # ---------------------------------------------------------------- Safe to delete?
    def safe_to_delete(self, wait=False):
        """Is it safe to delete what is left in the folders you organized? A verdict plus a list - reads only."""
        srcs = [p for p in self.sources() if p.exists()]
        lib = self.library()
        if not srcs:
            QMessageBox.information(self, "Safe to Delete?", "Pick the folders you organized first.")
            return

        def work(job):
            return O.check_leftovers(srcs, lib, progress=lambda t, i, n: job.note.emit(t, i, n),
                                     stop=lambda: job.stop)

        def done(r):
            self.safe_result = r
            n_safe, n_keep, n_work = len(r["safe"]), len(r["keep"]), len(r["work"])
            n_proj, n_other = len(r["project"]), len(r["other"])
            where = ", ".join(str(p) for p in srcs[:3]) + (" …" if len(srcs) > 3 else "")
            if n_keep == 0 and n_proj == 0 and n_other == 0 and n_work == 0:
                verdict = (f"<b style='color:#3c9a5f'>✓ Safe to delete.</b> Every astro file left in {where} has an "
                           f"identical copy in Organized Astro Files ({n_safe} file(s)); "
                           f"{r['empty_folders']} folder(s) are empty.")
            elif n_keep == 0:
                verdict = (f"<b style='color:#3c9a5f'>✓ No astro data would be lost:</b> every astro file left in "
                           f"{where} is safely in Organized Astro Files ({n_safe}).")
            else:
                verdict = (f"<b style='color:#d9534f'>✗ Not yet - {n_keep} astro file(s) exist only here.</b> "
                           "Don't delete these folders: run Step 1 to see why those files were not organized.")
            extra = []
            if n_work:
                extra.append(f"{n_work} PixInsight / Siril work copies (calibrated, registered …) - can be made again "
                             "from the subs, delete them if you don't need them")
            if n_proj:
                extra.append(f"{n_proj} file(s) in saved SubZeroAstro projects - keep them")
            if n_other:
                extra.append(f"{n_other} other file(s) (JPG, notes, logs …) the organizer never touches - "
                             "look through them before deleting")
            self.status.setText(verdict + ("<br>" + "<br>".join(extra) if extra else ""))
            rows = ([("⚠", p, n) for p, n in r["keep"]] + [("ⓘ", p, n) for p, n in r["project"]]
                    + [("ⓘ", p, n) for p, n in r["other"]] + [("·", p, n) for p, n in r["work"]]
                    + [("✓", p, n) for p, n in r["safe"]])
            self.table.setRowCount(min(len(rows), PREVIEW_ROWS))
            for k, (m, pth, note) in enumerate(rows[:PREVIEW_ROWS]):
                for c, v in enumerate((m, pth.name, str(pth.parent), note)):
                    cell = QTableWidgetItem(v)
                    if c == 1:
                        cell.setToolTip(str(pth))
                    if c >= 2:
                        cell.setToolTip(v)                 # the full text when the column is too narrow
                    self.table.setItem(k, c, cell)
        self._run(work, done, wait)

    # ---------------------------------------------------------------- Your Rigs
    def riglist(self):
        import json as _json
        try:
            v = _json.loads(self.settings.value("organize/riglist", "[]") or "[]")
            return [r for r in v if r.get("name") and (r.get("focals") or r.get("folders"))]
        except Exception:  # noqa: BLE001
            return []

    def tol(self):
        try:
            if not self.settings.value("organize/rig_tol_v3", False, type=bool):    # new default: ±3 %
                self.settings.setValue("organize/rig_tol", 3)
                self.settings.setValue("organize/rig_tol_v3", True)
            return float(self.settings.value("organize/rig_tol", 3)) / 100.0
        except (TypeError, ValueError):
            return 0.03

    def save_riglist(self, riglist, tol_pct=None):
        import json as _json
        import stacking as _st
        self.settings.setValue("organize/riglist", _json.dumps(riglist))
        if tol_pct is not None:
            self.settings.setValue("organize/rig_tol", tol_pct)
            self.settings.setValue("organize/rig_tol_v3", True)
        # the same names in Calibrate and Stack (by camera + focal length)
        try:
            rigs = _st.Rigs(_json.loads(self.settings.value("stack/rigs", "[]") or "[]"))
            for r in riglist:
                if r.get("camera"):
                    for f in r.get("focals") or []:
                        rigs.set(r["camera"], float(f), r["name"])
            self.settings.setValue("stack/rigs", _json.dumps(rigs.entries))
        except Exception:  # noqa: BLE001
            pass

    def _found(self):
        """Rigs found in the files looked through (or a sample of the chosen folder) that none of yours matches."""
        riglist, tol = self.riglist(), self.tol()
        root = (getattr(self, "_plan_args", None) or (None, None, None))[2]
        pairs = [(it.info, it.src) for it in self.items if it.info is not None]
        if not pairs:
            return []
        clusters = O.focal_clusters(pairs, tol)
        out = []
        for c in clusters:
            probe = type("I", (), {"focal": c["focal"], "camera": c["camera"]})()
            name, why = O.match_rig(probe, None, riglist, tol)
            if name:
                continue
            others = [o["paths"] for o in clusters if o is not c]
            c["suggest"] = O.suggest_rig_name(c["paths"], root, others)
            out.append(c)
        return out

    def find_in_files(self, folder=None, sample=600):
        """Find My Rigs in My Files: read a spread-out sample of headers from the folder (fast even on a NAS)."""
        import stacking as _st
        folder = Path(folder or self._common())
        if not folder.is_dir():
            return []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths = O.scan(folder)
            step = max(1, len(paths) // sample)
            pick = paths[::step]
            pairs = []
            for k, p in enumerate(pick):
                try:
                    pairs.append((_st.read_info(p), p))
                except Exception:  # noqa: BLE001
                    pass
                if k % 25 == 0:
                    QApplication.processEvents()
            clusters = O.focal_clusters(pairs, self.tol())
            riglist = self.riglist()
            out = []
            for c in clusters:
                probe = type("I", (), {"focal": c["focal"], "camera": c["camera"]})()
                if O.match_rig(probe, None, riglist, self.tol())[0]:
                    continue
                c["suggest"] = O.suggest_rig_name(c["paths"], folder, [o["paths"] for o in clusters if o is not c])
                c["files"] = round(c["files"] * step)
                out.append(c)
            return out
        finally:
            QApplication.restoreOverrideCursor()

    def edit_rigs(self, rows=None, tol_pct=None):
        """Your Rigs…: the table of rigs (name, focal lengths, camera). rows given = save those (self-test)."""
        if rows is None:
            d = YourRigsDialog(self, self.riglist(), round(self.tol() * 100), self._found())
            if not d.exec():
                return False
            rows, tol_pct = d.result_rows(), d.tol.value()
        old = {r["name"]: r for r in self.riglist()}
        merged = {}
        for r in rows:
            m = merged.setdefault(r["name"], {"name": r["name"], "focals": [], "camera": r.get("camera", ""),
                                              "folders": list((old.get(r["name"]) or {}).get("folders") or [])})
            for f in r.get("focals") or []:
                if f not in m["focals"]:
                    m["focals"].append(f)
            m["folders"] += [f for f in r.get("folders") or [] if f not in m["folders"]]
        self.save_riglist(list(merged.values()), tol_pct)
        self.replan()
        self._update_steps()
        return True

    def which_rig(self, answers=None):
        """Which Rig?…: the files a focal length can't place (fits two rigs, or none in the header) - you pick
        the rig for each folder; remembered for that folder."""
        root = (getattr(self, "_plan_args", None) or (None, None, None))[2]
        groups = [g for g in O.rig_issues(self.items, root) if g["reason"] != "nomatch"]
        if not groups:
            return False
        riglist = self.riglist()
        if answers is None:
            d = WhichRigDialog(self, groups, [r["name"] for r in riglist])
            if not d.exec():
                return False
            answers = d.answers()
        for g, name in zip(groups, answers):
            if not name:
                continue
            for r in riglist:
                if r["name"] == name:
                    r.setdefault("folders", [])
                    if g["folder"] not in r["folders"]:
                        r["folders"].append(g["folder"])
        self.save_riglist(riglist)
        self.replan()
        return True

    def replan(self):
        """Work the plan out again with the rigs as they are now - from the headers already read."""
        if not self.items or not getattr(self, "_plan_args", None):
            return
        lib, layout, src = self._plan_args
        cache = {str(it.src): it.info for it in self.items}

        def cached(p):
            v = cache.get(str(p))
            if v is None:
                raise OSError("could not be read")
            return v
        items = O.plan(O.classify([it.src for it in self.items], read_info=cached, library=lib, layout=layout,
                                  root=src, riglist=self.riglist(), tol=self.tol()), lib)
        self._show_plan(items)


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
        # a whole drive can hold tens of thousands of duplicates: list the first ones, the rest follow Tick All /
        # Untick All (a table row for every one of them made the window crawl)
        self.shown = dups[:PREVIEW_ROWS]
        self.extra = dups[PREVIEW_ROWS:]
        self.extra_on = True
        if self.extra:
            more = QLabel(f"The first {len(self.shown):,} are listed; the other {len(self.extra):,} are "
                          "handled together with them - <b>Tick All</b> / <b>Untick All</b> includes or leaves them out.")
            more.setWordWrap(True)
            lay.addWidget(more)
        self.table = QTableWidget(len(self.shown), 3)
        self.table.setHorizontalHeaderLabels(["Duplicate", "Kept", "Size"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for r, it in enumerate(self.shown):
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
        self.extra_on = bool(on)
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def ticked(self):
        return ([it for r, it in enumerate(self.shown) if self.table.item(r, 0).checkState() == Qt.Checked]
                + (list(self.extra) if self.extra_on else []))

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

class YourRigsDialog(QDialog):
    """Your rigs: one row per telescope - its short name, the focal length(s) it writes in your FITS headers
    (several when you use a reducer: 360, 290) and, if two rigs share a focal length, the camera."""

    def __init__(self, owner, riglist, tol_pct, found):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Your Rigs")
        lay = QVBoxLayout(self)
        lab = QLabel("Name each rig - a telescope with the camera that lives on it - and give the focal length it "
                     "writes in your FITS headers. Files are matched to a rig by focal length, within the "
                     "tolerance below (the header value drifts a few % from night to night). A scope with and "
                     "without a reducer: one rig, two focal lengths (e.g. <i>360, 290</i>). Add the camera only "
                     "when two rigs have about the same focal length. Set up once - remembered, and Calibrate "
                     "and Stack uses the same names.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Rig name", "Focal length(s), mm", "Camera (optional)", "Found in your files"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        for r in riglist:
            self.add_row(r["name"], ", ".join(f"{f:g}" for f in r.get("focals") or []), r.get("camera", ""),
                         "your rig" + (f" (+ {len(r.get('folders') or [])} folder(s) you assigned)"
                                       if r.get("folders") else ""))
        for c in found:
            self.add_row(c.get("suggest", ""), f"{c['focal']:g}", "",
                         f"found: {c['files']} file(s), {c['lo']:.0f}–{c['hi']:.0f} mm, {c['camera']} - "
                         f"name it or remove the row", new=True)
        h = QHBoxLayout()
        b = QPushButton("Add New Rig")
        b.clicked.connect(lambda: self.add_row("", "", "", "new", new=True, edit=True))
        h.addWidget(b)
        b = QPushButton("Remove Selected")
        b.clicked.connect(self.remove_selected)
        h.addWidget(b)
        b = QPushButton("Find My Rigs in My Files…")
        b.setToolTip("Reads a sample of headers from the folder to organize and adds a row for each focal length "
                     "none of your rigs matches")
        b.clicked.connect(self.find)
        h.addWidget(b)
        h.addStretch(1)
        h.addWidget(QLabel("Match within ±"))
        self.tol = QSpinBox()
        self.tol.setRange(1, 15)
        self.tol.setSuffix(" %")
        self.tol.setValue(int(tol_pct))
        h.addWidget(self.tol)
        lay.addLayout(h)
        self.err = QLabel("")
        self.err.setStyleSheet("color: #d9534f")
        lay.addWidget(self.err)
        h = QHBoxLayout()
        h.addStretch(1)
        ok = QPushButton("Save My Rigs")
        ok.setDefault(True)
        ok.clicked.connect(self.save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        h.addWidget(ok)
        h.addWidget(cancel)
        lay.addLayout(h)
        self.resize(980, 460)

    def add_row(self, name, focals, camera, note, new=False, edit=False):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for k, v in enumerate((name, focals, camera, note)):
            cell = QTableWidgetItem(v)
            if k == 3:
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, k, cell)
        if edit:
            self.table.setCurrentCell(r, 0)
            self.table.editItem(self.table.item(r, 0))

    def remove_selected(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def find(self):
        have = {(self.table.item(r, 1).text() or "").strip() for r in range(self.table.rowCount())}
        found = self.owner.find_in_files()
        added = 0
        for c in found:
            if f"{c['focal']:g}" in have:
                continue
            self.add_row(c.get("suggest", ""), f"{c['focal']:g}", "",
                         f"found: about {c['files']} file(s), {c['lo']:.0f}–{c['hi']:.0f} mm, {c['camera']}", new=True)
            added += 1
        self.err.setText("" if added else "Nothing new found - every focal length in the sample matches a rig.")

    def result_rows(self):
        rows = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() or "").strip()
            fl = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", self.table.item(r, 1).text() or "")]
            cam = (self.table.item(r, 2).text() or "").strip()
            if name or fl:
                rows.append({"name": name, "focals": fl, "camera": cam})
        return rows

    def save(self):
        rows = self.result_rows()
        bad = [r for r in rows if not r["name"] or not r["focals"]]
        if bad:
            self.err.setText("Every rig needs a name and at least one focal length (or remove that row).")
            return
        self.accept()


class WhichRigDialog(QDialog):
    """Files a focal length can't place: say which rig each folder belongs to (remembered for that folder)."""

    def __init__(self, owner, groups, names):
        super().__init__(owner)
        self.setWindowTitle("Which Rig?")
        lay = QVBoxLayout(self)
        lab = QLabel("These files can't be placed by focal length alone - it fits more than one of your rigs, or the "
                     "header has none. Pick the rig for each folder; SubZeroAstro remembers it for that folder.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        self.table = QTableWidget(len(groups), 4)
        self.table.setHorizontalHeaderLabels(["Rig", "Folder", "Files", "Why"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.boxes = []
        for r, g in enumerate(groups):
            box = QComboBox()
            box.addItem("- not now -", "")
            choices = [n.strip() for n in g["names"].split(",")] if g.get("names") else names
            for n in choices + [n for n in names if n not in choices]:
                box.addItem(n, n)
            if g.get("names"):
                box.setCurrentIndex(0)
            self.table.setCellWidget(r, 0, box)
            self.boxes.append(box)
            why = (f"{g['focal']:.0f} mm fits {g['names']}" if g["reason"] == "ambiguous" else
                   "no focal length in the header")
            for k, v in ((1, g["folder"]), (2, str(g["files"])), (3, f"{why} ({g['camera']})")):
                cell = QTableWidgetItem(v)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, k, cell)
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        h.addStretch(1)
        ok = QPushButton("Use These Rigs")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        h.addWidget(ok)
        h.addWidget(cancel)
        lay.addLayout(h)
        self.resize(980, 380)

    def answers(self):
        return [b.currentData() for b in self.boxes]
