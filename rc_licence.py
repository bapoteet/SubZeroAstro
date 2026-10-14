"""
rc_licence.py - "RC-Astro Licences" dialog on the Astro Image Processing page.

Shows which RC-Astro tools (BlurXTerminator, StarXTerminator, NoiseXTerminator…)
are licensed on this computer and activates them without a terminal: RC-Astro's
own procedure is to run `rc-astro license` and paste the activation code from
the rc-astro.com account page into it - this dialog does exactly that. One code
activates every tool the account owns. A Terminal-window route and the CLI's
own updater (`rc-astro update`) are offered too.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (QDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QVBoxLayout)

import tools
import ui_header


_orphans = []          # jobs still running when their window closed


class _Job(QThread):
    done = Signal(object, object)        # (result, error text or None)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except tools.ToolError as e:
            self.done.emit(None, str(e))
        except Exception as e:  # noqa: BLE001 - reported in the dialog, never a crash
            self.done.emit(None, f"{type(e).__name__}: {e}")


class LicenceDialog(QDialog):
    def __init__(self, exe: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RC-Astro Licences")
        self.setMinimumWidth(620)
        self.rc = tools.RCAstro(exe)
        self.job = None
        self._busy = False
        self.changed = False                 # licences changed - the page re-reads the tools
        self.before = None

        lay = QVBoxLayout(self)
        intro = QLabel("SubZeroAstro uses RC-Astro's command-line tools (BlurXTerminator, StarXTerminator, "
                       "NoiseXTerminator). Each must be activated once on this computer. "
                       "<b>One code from your rc-astro.com account activates every tool you own.</b>")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        box = QGroupBox("On this computer")
        bl = QVBoxLayout(box)
        self.status = QLabel("checking…")
        self.status.setWordWrap(True)
        bl.addWidget(self.status)
        lay.addWidget(box)

        act = QGroupBox("Activate")
        al = QVBoxLayout(act)
        s1 = QHBoxLayout()
        s1.addWidget(QLabel("<b>1.</b> Sign in to your RC-Astro account and click <i>Get Activation Code</i> "
                            "(click it again to copy the code)."), 1)
        b_acc = QPushButton("Open My RC-Astro Account")
        b_acc.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(tools.RC_ACCOUNT_URL)))
        s1.addWidget(b_acc)
        al.addLayout(s1)
        al.addWidget(QLabel("<b>2.</b> Paste the code here and click Activate:"))
        s2 = QHBoxLayout()
        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("activation code from rc-astro.com")
        self.code_edit.returnPressed.connect(self.activate)
        s2.addWidget(self.code_edit, 1)
        self.b_act = QPushButton("Activate")
        self.b_act.setStyleSheet(ui_header.RUN_STYLE)
        self.b_act.clicked.connect(self.activate)
        s2.addWidget(self.b_act)
        al.addLayout(s2)
        s3 = QHBoxLayout()
        hint = QLabel("<small>Prefer RC-Astro's own way? This opens a terminal window running "
                      "<code>rc-astro license</code>; paste the code there and press Enter.</small>")
        hint.setWordWrap(True)
        s3.addWidget(hint, 1)
        b_term = QPushButton("Do It in a Terminal Window")
        b_term.clicked.connect(self.in_terminal)
        s3.addWidget(b_term)
        al.addLayout(s3)
        lay.addWidget(act)

        upd = QGroupBox("RC-Astro CLI updates")
        ul = QHBoxLayout(upd)
        self.b_check = QPushButton("Check for Updates")
        self.b_check.clicked.connect(self.check_update)
        self.b_install = QPushButton("Download && Install Update")
        self.b_install.clicked.connect(self.install_update)
        ul.addWidget(self.b_check)
        ul.addWidget(self.b_install)
        ul.addStretch(1)
        lay.addWidget(upd)

        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(QFont("Menlo" if ui_header._MAC else "Consolas", 9))
        self.out.setPlaceholderText("Messages from rc-astro appear here.")
        self.out.setMaximumHeight(150)
        lay.addWidget(self.out)

        bottom = QHBoxLayout()
        b_re = QPushButton("Check Again")
        b_re.setToolTip("Re-read the licence status (e.g. after activating in a terminal window)")
        b_re.clicked.connect(self.refresh)
        bottom.addWidget(b_re)
        bottom.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        bottom.addWidget(b_close)
        lay.addLayout(bottom)
        self.refresh()

    # ------------------------------------------------------------ helpers
    def busy(self) -> bool:
        return self._busy

    def _start(self, fn, then, note):
        if self.busy():
            return
        for b in (self.b_act, self.b_check, self.b_install):
            b.setEnabled(False)
        if note:
            self.out.appendPlainText(note)
        self._then = then
        self._busy = True
        self.job = _Job(fn)
        self.job.done.connect(self._finished)       # a method of this dialog: runs on the UI thread
        self.job.start()

    def _finished(self, res, err):
        self._busy = False
        for b in (self.b_act, self.b_check, self.b_install):
            b.setEnabled(True)
        self._then(res, err)

    def _show_status(self, lic, err):
        if err:
            self.status.setText(f"<span style='color:#e65050'>Could not read the licences: {err}</span>")
            return
        rows = []
        for key, name, ok, msg in lic or []:
            mark = ("<span style='color:#5fd27a'>✓ licensed</span>" if ok
                    else "<span style='color:#eba932'>✗ not licensed</span>")
            rows.append(f"<b>{name}</b> &nbsp; {mark} &nbsp; <small>{msg}</small>")
        ver = f"RC-Astro CLI v{self.rc.cli_version}" if self.rc.cli_version else "RC-Astro CLI"
        self.status.setText(f"<small>{ver}</small><br>" + ("<br>".join(rows) or "no RC-Astro tools reported"))
        self.lic = lic

    def refresh(self):
        self._start(self.rc.licences, self._after_refresh, None)

    def _after_refresh(self, res, err):
        if res is not None and self.before is not None and \
                [r[2] for r in res] != [r[2] for r in self.before]:
            self.changed = True
        if res is not None and self.before is None:
            self.before = res
        self._show_status(res, err)

    # ------------------------------------------------------------ actions
    def activate(self):
        code = self.code_edit.text().strip()
        if not code:
            QMessageBox.information(self, "RC-Astro Licences",
                                    "Paste the activation code from your RC-Astro account first "
                                    "(step 1 opens the account page).")
            return
        rc = self.rc

        def work():
            exit_code, said = rc.activate(code)
            return exit_code, said, rc.licences()
        self._start(work, self._after_activate, "Activating…")

    def _after_activate(self, res, err):
        if err:
            self.out.appendPlainText(f"Not activated: {err}")
            return
        exit_code, said, lic = res
        if said:
            self.out.appendPlainText(said)
        was = sum(1 for r in (self.before or []) if r[2])
        now = sum(1 for r in lic if r[2])
        self._show_status(lic, None)
        if exit_code == 0 and lic and (now > was or now == len(lic)):
            self.changed = True
            self.code_edit.clear()
            self.out.appendPlainText(f"✓ Activated - {now} of {len(lic)} RC-Astro tools licensed on this computer.")
        else:
            self.out.appendPlainText("✗ The licences did not change. Check the code (copy it again from "
                                     "the account page - codes expire after a while), or try "
                                     "'Do It in a Terminal Window'.")
        self.before = lic

    def in_terminal(self):
        if not tools.open_in_terminal(tools._command(self.rc.exe) + ["license"]):
            QMessageBox.warning(self, "RC-Astro Licences",
                                "Could not open a terminal window. Open one yourself and type:\n\n"
                                f"\"{self.rc.exe}\" license")
            return
        self.out.appendPlainText("A terminal window opened with 'rc-astro license'. Paste your code there, "
                                 "press Enter, then click 'Check Again' here.")

    def check_update(self):
        self._start(self.rc.check_update, self._after_check, "Checking for RC-Astro CLI updates…")

    def _after_check(self, res, err):
        if err:
            self.out.appendPlainText(f"Update check failed: {err}")
            return
        self.out.appendPlainText(res[1] or "(rc-astro said nothing)")

    def install_update(self):
        if QMessageBox.question(
                self, "Update RC-Astro CLI",
                "This downloads the newest RC-Astro CLI installer and starts it in a terminal window.\n\n"
                "Let the installer finish, then come back and click 'Check Again'. "
                "Don't run XTerminator steps while it installs.\n\nContinue?") != QMessageBox.Yes:
            return
        if self.rc.install_update():
            self.changed = True
            self.out.appendPlainText("The RC-Astro updater is running in a terminal window.")
        else:
            self.out.appendPlainText("Could not open a terminal window. Run this yourself: "
                                     f"\"{self.rc.exe}\" update --install")

    def done(self, r):
        if self.job is not None and self.job.isRunning():
            self.job.wait(3000)
            if self.job.isRunning():            # still talking to RC-Astro: let it finish on its own
                _orphans.append(self.job)
                self.job.finished.connect(lambda j=self.job: _orphans.remove(j) if j in _orphans else None)
        super().done(r)
