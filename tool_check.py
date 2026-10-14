"""
tool_check.py - "Your Tools": which optional helper programs (ASTAP, GraXpert,
RC-Astro CLI) are installed and ready, checked in the background at startup.

None of them is required: pages 1-2 never use them and page 3 has a fallback
for each. So page 1 shows a slim amber banner only when one is missing or not
licensed, with "Set Up…" (the Your Tools window) and "Don't Remind Me". The
same window is in Settings > Your Tools….
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QToolButton, QVBoxLayout)

import tools

MAC = sys.platform == "darwin"

# key, name, settings key, finder, what it does, what happens without it, download page
TOOLS = [
    ("rcastro", "RC-Astro CLI", "prep/rcastro", tools.find_rcastro,
     "BlurXTerminator, NoiseXTerminator and StarXTerminator (sharpening, noise reduction, star removal)",
     "sharpening, noise reduction and star removal are unavailable",
     "https://www.rc-astro.com/?p=173572"),
    ("graxpert", "GraXpert", "prep/graxpert", tools.find_graxpert,
     "AI gradient and light-pollution removal (free)",
     "the built-in gradient removal is used instead",
     "https://graxpert.com/"),
    ("astap", "ASTAP", "prep/astap", tools.find_astap,
     "plate solving - works out exactly where the picture points (free; also install a star database, e.g. D50)",
     "plate solving is skipped",
     "https://www.hnsky.org/astap.htm"),
]
BANNER_OFF = "ui/tools_banner_off"        # the problems the user said not to be reminded about


def check(paths: dict) -> list:
    """paths: {settings key: configured path}. Returns one dict per tool:
    key, name, path, ok, problem (short), detail (sentence)."""
    out = []
    for key, name, skey, finder, does, without, url in TOOLS:
        r = {"key": key, "name": name, "path": None, "ok": False, "problem": "", "detail": "",
             "does": does, "without": without, "url": url, "skey": skey}
        try:
            r["path"] = finder((paths.get(skey) or "").strip() or None)
        except Exception:  # noqa: BLE001 - a check must never stop the app
            r["path"] = None
        if not r["path"]:
            r["problem"] = "not found"
            r["detail"] = f"not found - {without}"
        elif key == "rcastro":
            try:
                lic = tools.RCAstro(r["path"]).licences()
                off = [n for _, n, ok, _ in lic if not ok]
                if not lic:
                    r["problem"], r["detail"] = "no tools", "found, but it reports no RC-Astro tools"
                elif off:
                    r["problem"] = "not licensed"
                    r["detail"] = f"found, but not licensed on this computer: {', '.join(off)}"
                    r["ok"] = len(off) < len(lic)        # usable, some tools missing
                else:
                    r["ok"] = True
                    r["detail"] = "ready - " + ", ".join(n for _, n, _, _ in lic)
            except tools.ToolError as e:
                r["problem"], r["detail"] = "not working", f"found, but it did not answer: {e}"
            except Exception as e:  # noqa: BLE001
                r["problem"], r["detail"] = "not working", f"found, but it did not answer ({type(e).__name__})"
        else:
            r["ok"] = True
            r["detail"] = f"ready - {Path(r['path']).name}"
        out.append(r)
    return out


def problems(results) -> str:
    """A fingerprint of what is wrong ('' when everything is ready)."""
    return ";".join(f"{r['key']}:{r['problem']}" for r in results if r["problem"])


def settings_paths(settings) -> dict:
    return {t[2]: settings.value(t[2], "") or "" for t in TOOLS}


def browse_program(parent, label, current=""):
    """Pick a program: an .exe on Windows, an app or the rc-astro program on a Mac."""
    start = current or ("/Applications" if MAC else "C:\\Program Files")
    filt = "Programs (*.exe);;All files (*)" if sys.platform == "win32" else "All files (*)"
    f, _ = QFileDialog.getOpenFileName(parent, f"Locate {label}", start, filt)
    return f


def refresh_open_pages():
    """Page 3 re-reads its tools after a change here."""
    for w in QApplication.topLevelWidgets():
        if hasattr(w, "detect_tools") and hasattr(w, "rc_status"):
            try:
                w.detect_tools()
            except Exception:  # noqa: BLE001
                pass


class Checker(QThread):
    done = Signal(list)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.result = None

    def run(self):
        self.result = check(self.paths)
        self.done.emit(self.result)


# --------------------------------------------------------------------------

class ToolsBanner(QFrame):
    """Slim amber notice on page 1; hidden while all helpers are ready or the user dismissed these problems."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.results = []
        self.checker = None
        self.setStyleSheet("QFrame { background:#3a2e14; border-bottom:1px solid #d69e2e; }"
                           "QLabel { color:#f3e3c0; font-size:10.5pt; background:transparent; border:none; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 5, 8, 5)
        self.label = QLabel()
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        b_set = QToolButton()
        b_set.setText("Set Up…")
        b_set.setStyleSheet("QToolButton { background:#b7791f; color:white; border-radius:4px; padding:4px 14px; "
                            "font-weight:bold; } QToolButton:hover { background:#c98a2a; }")
        b_set.clicked.connect(self.set_up)
        b_off = QToolButton()
        b_off.setText("Don't Remind Me")
        b_off.setToolTip("Hide this until something else changes (Settings > Your Tools… is always there)")
        b_off.clicked.connect(self.dismiss)
        b_x = QToolButton()
        b_x.setText("✕")
        b_x.setToolTip("Hide for now")
        b_x.clicked.connect(self.hide)
        for b in (b_set, b_off, b_x):
            lay.addWidget(b)
        self.hide()

    def start_check(self):
        if self.checker is not None and self.checker.isRunning():
            return
        self.checker = Checker(settings_paths(self.settings))
        self.checker.done.connect(self.show_results)
        self.checker.start()

    def show_results(self, results):
        self.results = results
        bad = problems(results)
        if not bad or bad == (self.settings.value(BANNER_OFF, "") or ""):
            self.hide()
            return
        parts = []
        for r in results:
            if r["ok"] and not r["problem"]:
                parts.append(f"{r['name']} <span style='color:#5fd27a'>✓</span>")
            else:
                parts.append(f"{r['name']} <span style='color:#ffb347'>✗</span> <i>({r['problem']})</i>")
        hint = "; ".join(f"{r['without']}" for r in results if r["problem"] == "not found")
        self.label.setText("<b>Optional helpers for Image Processing:</b> &nbsp;" + " &nbsp;·&nbsp; ".join(parts)
                           + (f"<br><small>Without them, {hint}. You can still cull and stack.</small>"
                              if hint else ""))
        self.show()

    def dismiss(self):
        self.settings.setValue(BANNER_OFF, problems(self.results))
        self.hide()

    def set_up(self):
        dlg = ToolsDialog(self.settings, self.window(), self.results or None)
        dlg.exec()
        self.show_results(dlg.results)


# --------------------------------------------------------------------------

class ToolsDialog(QDialog):
    def __init__(self, settings, parent=None, results=None):
        super().__init__(parent)
        self.setWindowTitle("Your Tools")
        self.setMinimumWidth(760)
        self.settings = settings
        self.results = results or []
        self.checker = None
        self.rows = {}
        lay = QVBoxLayout(self)
        intro = QLabel("SubZeroAstro works on its own. These free or paid helper programs add extra steps on the "
                       "<b>Astro Image Processing</b> page. Install the ones you want, then click "
                       "<b>Check Again</b>. If one is installed somewhere unusual, use <b>Browse…</b>.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for i, (key, name, skey, finder, does, without, url) in enumerate(TOOLS):
            title = QLabel(f"<b>{name}</b><br><small>{does}</small>")
            title.setWordWrap(True)
            title.setMinimumWidth(260)
            status = QLabel("checking…")
            status.setWordWrap(True)
            edit = QLineEdit(settings.value(skey, "") or "")
            edit.setPlaceholderText("auto-detect")
            b_browse = QPushButton("Browse…")
            b_get = QPushButton("Download Page")
            b_get.setToolTip(url)
            b_get.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))

            def browse(_=False, e=edit, k=skey, n=name):
                f = browse_program(self, n, e.text())
                if f:
                    e.setText(f)
                    self.settings.setValue(k, f)
                    self.recheck()
            b_browse.clicked.connect(browse)
            edit.editingFinished.connect(lambda e=edit, k=skey: (self.settings.setValue(k, e.text().strip()),
                                                                 self.recheck()))
            r0 = i * 2
            grid.addWidget(title, r0, 0, 2, 1)
            grid.addWidget(status, r0, 1, 1, 3)
            grid.addWidget(edit, r0 + 1, 1)
            grid.addWidget(b_browse, r0 + 1, 2)
            grid.addWidget(b_get, r0 + 1, 3)
            self.rows[key] = (status, edit)
            if key == "rcastro":
                self.b_lic = QPushButton("RC-Astro Licences…")
                self.b_lic.setToolTip("See and activate your RC-Astro licences, or update the RC-Astro CLI")
                self.b_lic.clicked.connect(self.licences)
                grid.addWidget(self.b_lic, r0 + 1, 4)
        lay.addLayout(grid)
        bottom = QHBoxLayout()
        b_re = QPushButton("Check Again")
        b_re.clicked.connect(self.recheck)
        bottom.addWidget(b_re)
        bottom.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        bottom.addWidget(b_close)
        lay.addLayout(bottom)
        if self.results:
            self.show_results(self.results)
        self.recheck()

    def busy(self):
        return self.checker is not None and self.checker.isRunning()

    def recheck(self):
        if self.busy():
            return
        for status, _ in self.rows.values():
            status.setText("<span style='color:#8a93a6'>checking…</span>")
        self.checker = Checker(settings_paths(self.settings))
        self.checker.done.connect(self.show_results)
        self.checker.start()

    def show_results(self, results):
        self.results = results
        for r in results:
            status, _ = self.rows[r["key"]]
            if r["ok"] and not r["problem"]:
                status.setText(f"<span style='color:#5fd27a'>✓ {r['detail']}</span>")
            elif r["ok"]:
                status.setText(f"<span style='color:#eba932'>⚠ {r['detail']}</span>")
            else:
                status.setText(f"<span style='color:#eba932'>✗ {r['detail']}</span>")
            status.setToolTip(r["path"] or "")
            if r["key"] == "rcastro":
                self.b_lic.setEnabled(bool(r["path"]))

    def licences(self):
        r = next((r for r in self.results if r["key"] == "rcastro"), None)
        if not r or not r["path"]:
            return
        import rc_licence
        dlg = rc_licence.LicenceDialog(r["path"], self)
        dlg.exec()
        if dlg.changed:
            self.recheck()

    def done(self, code):
        if self.busy():
            self.checker.wait(15000)
        if self.checker is not None and self.checker.result is not None:
            self.results = self.checker.result
        refresh_open_pages()
        super().done(code)
