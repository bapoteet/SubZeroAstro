"""
nina_window.py - Settings › Work With N.I.N.A.…: your rigs from N.I.N.A.'s profiles, a check of its image file
pattern, and the command that runs Fast Track by itself after the night.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

import framing as FR
import nina_link as NL


class ProfilesTable(QWidget):
    """N.I.N.A. profiles found on this computer (and any profile files added): one row each, tick the rigs."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Rig name", "Focal length", "Camera", "Filters", "File name pattern",
                                              "Profile"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        b = QPushButton("Add Profile Files…")
        b.setToolTip("N.I.N.A. runs on each rig's own computer? Copy its profile (on that computer: "
                     "%LOCALAPPDATA%\\NINA\\Profiles\\*.profile) here, or open it over the network")
        b.clicked.connect(lambda: self.add_files())
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.profiles = []
        extra = [p for p in (settings.value("nina/profile_files", []) or []) if Path(p).is_file()] \
            if settings is not None else []
        self.load(NL.read_profiles() + NL.read_profiles(extra))

    def load(self, profiles):
        seen = {d["file"] for d in self.profiles}
        for d in profiles:
            if d["file"] not in seen:
                self.profiles.append(d)
                seen.add(d["file"])
        self.table.setRowCount(0)
        for d in self.profiles:
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = QTableWidgetItem(NL.rig_name(d))
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked if d.get("focal") else Qt.Unchecked)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(f"{d['focal']:g} mm" if d.get("focal") else "not set"))
            self.table.setItem(r, 2, QTableWidgetItem(d.get("camera") or ""))
            self.table.setItem(r, 3, QTableWidgetItem(", ".join(d.get("filters") or [])))
            issues = []
            for kind, pat in (d.get("patterns") or {}).items():
                issues += [f"{kind}: {x}" for x in NL.check_pattern(pat)]
            it = QTableWidgetItem("⚠ " + "; ".join(issues) if issues else "✓ fine")
            it.setToolTip("\n".join(f"{k}: {v}" for k, v in (d.get("patterns") or {}).items()))
            self.table.setItem(r, 4, it)
            self.table.setItem(r, 5, QTableWidgetItem(d.get("profile") or Path(d["file"]).stem))
            for c in range(1, 6):
                self.table.item(r, c).setFlags(self.table.item(r, c).flags() & ~Qt.ItemIsEditable)
        self.note.setText(
            f"<b>{len(self.profiles)} N.I.N.A. profile(s)</b> found." if self.profiles else
            "No N.I.N.A. profile on this computer. If N.I.N.A. runs on your rigs' own computers, copy each one's "
            "profile file over (on that computer: <i>%LOCALAPPDATA%\\NINA\\Profiles</i>) and press "
            "<b>Add Profile Files…</b>.")

    def add_files(self, paths=None):
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "N.I.N.A. profile files", str(Path.home()),
                                                    "N.I.N.A. profiles (*.profile)")
        if not paths:
            return 0
        got = NL.read_profiles(paths)
        if self.settings is not None:
            keep = list(dict.fromkeys(list(self.settings.value("nina/profile_files", []) or [])
                                      + [str(p) for p in paths]))[-30:]
            self.settings.setValue("nina/profile_files", keep)
        self.load(got)
        return len(got)

    def checked(self) -> list:
        out = []
        for r, d in enumerate(self.profiles):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked and d.get("focal"):
                dd = dict(d)
                dd["telescope"] = it.text().strip() or NL.rig_name(d)
                out.append(dd)
        return out


class ProfilesPicker(QDialog):
    """Your Rigs › From N.I.N.A.…: tick the profiles to add as rigs."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rigs From N.I.N.A.")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Each N.I.N.A. profile names its telescope and focal length (Options › Equipment › "
                             "Telescope). Tick the ones to add to Your Rigs; you can change a rig's name here first."))
        self.t = ProfilesTable(settings, self)
        lay.addWidget(self.t, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Add to Your Rigs")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(980, 420)


class NinaDialog(QDialog):
    """Settings › Work With N.I.N.A.…"""

    def __init__(self, owner, settings):
        super().__init__(owner)
        self.owner, self.settings = owner, settings
        self.setWindowTitle("Work With N.I.N.A.")
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        # 1 - rigs and file names
        w1 = QWidget()
        l1 = QVBoxLayout(w1)
        self.profiles = ProfilesTable(settings, w1)
        l1.addWidget(self.profiles, 1)
        pat = QLabel("<b>File name pattern:</b> SubZeroAstro reads everything it needs from each file's header, so "
                     "any pattern works - but a token N.I.N.A. doesn't know ends up in the file names as typed, or "
                     "as a bare <i>$$</i>. Fix it in N.I.N.A.: Options › Imaging › Image File Pattern. A clear "
                     f"pattern if you want one:<br><code>{NL.SUGGESTED_PATTERN}</code>")
        pat.setWordWrap(True)
        pat.setTextInteractionFlags(Qt.TextSelectableByMouse)
        l1.addWidget(pat)
        h = QHBoxLayout()
        b = QPushButton("Add Ticked to Your Rigs…")
        b.setToolTip("Opens Your Rigs with these added - check them, then Save My Rigs")
        b.clicked.connect(lambda: self.add_to_rigs())
        h.addWidget(b)
        b2 = QPushButton("Copy Suggested Pattern")
        b2.clicked.connect(lambda: QApplication.clipboard().setText(NL.SUGGESTED_PATTERN))
        h.addWidget(b2)
        h.addStretch(1)
        l1.addLayout(h)
        tabs.addTab(w1, "Your Rigs && File Names")

        # 2 - after the night
        w2 = QWidget()
        l2 = QVBoxLayout(w2)
        intro = QLabel(
            "Let N.I.N.A. start SubZeroAstro's <b>Fast Track</b> when the night's imaging is done: the night's "
            "frames are checked, calibrated, stacked and finished on the suggested settings, and the picture is "
            "saved - waiting for you in the morning.<br><br>In N.I.N.A.'s Advanced Sequencer, at the end of the "
            "sequence (after warming the camera and parking), add <b>Utility › External Script</b> and paste the "
            "line below into it. It runs on the computer N.I.N.A. runs on, so SubZeroAstro must be installed there "
            "and the folder is where that N.I.N.A. saves its frames.<br><br>Only the frames changed in the last "
            "hours below are used (the night's), the target with the most frames is processed, and the picture "
            "goes in the usual Pictures folder (add <code>--target \"M 31\"</code> or <code>--out \"D:\\Pictures\""
            "</code> to choose).")
        intro.setWordWrap(True)
        l2.addWidget(intro)
        h = QHBoxLayout()
        h.addWidget(QLabel("N.I.N.A.'s image folder:"))
        prof = self.profiles.profiles[0] if self.profiles.profiles else {}
        self.folder = QLineEdit(settings.value("nina/ft_folder", "") or prof.get("image_path", "") or "")
        self.folder.textChanged.connect(self._cmd)
        h.addWidget(self.folder, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick)
        h.addWidget(b)
        l2.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(QLabel("Frames from the last"))
        self.hours = QDoubleSpinBox()
        self.hours.setRange(1, 72)
        self.hours.setDecimals(0)
        self.hours.setValue(float(settings.value("nina/ft_hours", 18) or 18))
        self.hours.valueChanged.connect(self._cmd)
        h.addWidget(self.hours)
        h.addWidget(QLabel("hours"))
        h.addStretch(1)
        l2.addLayout(h)
        self.cmd = QLineEdit()
        self.cmd.setReadOnly(True)
        l2.addWidget(self.cmd)
        b = QPushButton("Copy for N.I.N.A.")
        b.clicked.connect(lambda: QApplication.clipboard().setText(self.cmd.text()))
        h = QHBoxLayout()
        h.addWidget(b)
        h.addStretch(1)
        l2.addLayout(h)
        l2.addStretch(1)
        tabs.addTab(w2, "After the Night")

        # 3 - targets
        w3 = QWidget()
        l3 = QVBoxLayout(w3)
        t3 = QLabel("<b>Tonight's Target Picker</b> (Stack › Tonight's Target Picker…) already talks to N.I.N.A.: it "
                    "reads your site and horizon from a N.I.N.A. profile, <b>Send to N.I.N.A.</b> puts the picked "
                    "targets straight into N.I.N.A.'s target list (Advanced Sequencer), and <b>Save List for "
                    "N.I.N.A.…</b> writes a list N.I.N.A. imports.")
        t3.setWordWrap(True)
        l3.addWidget(t3)
        if hasattr(owner, "open_target_picker"):
            b = QPushButton("Open Tonight's Target Picker")
            b.clicked.connect(lambda: owner.open_target_picker())
            h = QHBoxLayout()
            h.addWidget(b)
            h.addStretch(1)
            l3.addLayout(h)
        l3.addStretch(1)
        tabs.addTab(w3, "Targets")

        # 4 - same framing next session
        w4 = QWidget()
        l4 = QVBoxLayout(w4)
        t4 = QLabel("Pick a <b>plate-solved master</b> (Astro Image Processing solves them when ASTAP is installed) "
                    "or any solved picture of the target. SubZeroAstro works out its centre and the camera's "
                    "<b>position angle</b> the way N.I.N.A. measures them, and <b>Send to N.I.N.A.</b> puts a "
                    "target with both into N.I.N.A.'s target list. Next session, use it with <i>Slew, center and "
                    "rotate</i> (a rotator, or N.I.N.A.'s manual rotator prompts) or <i>Slew and center</i> "
                    "without one - and the new frames line up with the old ones. Not solved yet? ASTAP solves a "
                    "copy first (your file isn't changed).")
        t4.setWordWrap(True)
        l4.addWidget(t4)
        h = QHBoxLayout()
        b = QPushButton("Choose Master or Picture…")
        b.clicked.connect(lambda: self.choose_framing())
        h.addWidget(b)
        h.addStretch(1)
        l4.addLayout(h)
        self.fr_label = QLabel("")
        self.fr_label.setWordWrap(True)
        self.fr_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        l4.addWidget(self.fr_label)
        h = QHBoxLayout()
        self.b_fr_send = QPushButton("Send to N.I.N.A.")
        self.b_fr_send.clicked.connect(lambda: self.send_framing())
        self.b_fr_copy = QPushButton("Copy Centre && Angle")
        self.b_fr_copy.clicked.connect(lambda: QApplication.clipboard().setText(FR.describe(self.framing)))
        for x in (self.b_fr_send, self.b_fr_copy):
            x.setEnabled(False)
            h.addWidget(x)
        h.addStretch(1)
        l4.addLayout(h)
        l4.addStretch(1)
        self.fr_tab = tabs.addTab(w4, "Same Framing Next Session")
        self.tabs = tabs
        self.framing = None

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._cmd()
        self.resize(1040, 560)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "N.I.N.A.'s image folder", self.folder.text() or str(Path.home()))
        if d:
            self.folder.setText(d)

    def _cmd(self, *_a):
        f = self.folder.text().strip() or "<N.I.N.A.'s image folder>"
        self.cmd.setText(NL.fast_track_command(f, self.hours.value()))
        if self.settings is not None:
            self.settings.setValue("nina/ft_folder", self.folder.text().strip())
            self.settings.setValue("nina/ft_hours", self.hours.value())

    def add_to_rigs(self, wait=False):
        picked = self.profiles.checked()
        if not picked:
            return None
        org = self.owner.open_organizer() if hasattr(self.owner, "open_organizer") else None
        org = org or getattr(self.owner, "organizer_win", None)
        if org is None:
            return None
        return org.edit_rigs(preload=picked, show=not wait)

    # ---------------------------------------------------------------- same framing next session
    def choose_framing(self, path=None, astap=None):
        """Work out the framing of a solved picture (solving it with ASTAP first if needed)."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "A plate-solved master or picture", self.settings.value("nina/frame_dir", "") or
                str(Path.home()), "FITS / XISF (*.fit *.fits *.fts *.xisf)")
        if not path:
            return None
        if self.settings is not None:
            self.settings.setValue("nina/frame_dir", str(Path(path).parent))
        if astap is None:
            import tools
            astap = tools.find_astap(self.settings.value("prep/astap", "") or None if self.settings is not None
                                     else None)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.fr_label.setText("Reading" + (" and plate solving…" if astap else "…"))
            QApplication.processEvents()
            fr = FR.from_file(path, astap)
        except Exception as e:  # noqa: BLE001
            fr = None
            self.fr_label.setText(f"<span style='color:#eba932'>{Path(path).name}: {e}</span>")
        finally:
            QApplication.restoreOverrideCursor()
        self.framing = fr
        for x in (self.b_fr_send, self.b_fr_copy):
            x.setEnabled(fr is not None)
        if fr is not None:
            self.fr_label.setText(f"<b>{Path(path).name}</b>{' (solved just now by ASTAP)' if fr['solved_now'] else ''}"
                                  f"<br>{FR.describe(fr)}")
        return fr

    def send_framing(self, folder=None):
        """Write the framing as a N.I.N.A. target (with the saved Target Picker template, if there is one)."""
        if not self.framing:
            return None
        import target_picker as tp
        tpl = (self.settings.value("picker/nina_template", "") or "") if self.settings is not None else ""
        template = Path(tpl).read_text(encoding="utf-8-sig") if tpl and Path(tpl).is_file() else None
        if folder is None:
            profs = tp.nina_profiles() if hasattr(tp, "nina_profiles") else []
            folder = tp.nina_targets_folder(profs[0] if profs else None)
        try:
            paths = tp.save_nina_targets([FR.as_target(self.framing)], folder, "same framing", template)
        except OSError as e:
            self.fr_label.setText(self.fr_label.text() + f"<br><span style='color:#eba932'>Couldn't write to "
                                  f"{folder}: {e}</span>")
            return None
        self.fr_label.setText(self.fr_label.text() + f"<br>✓ Sent to N.I.N.A.: {paths[0]} - in N.I.N.A.: "
                              "Sequencer › Advanced › Targets › SubZeroAstro. (N.I.N.A. on another computer: copy "
                              "the file into its Documents\\N.I.N.A\\Targets folder.)")
        return paths

