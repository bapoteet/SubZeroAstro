"""In-app updates: is a newer SubZeroAstro on GitHub, and 'Update Now' to install it.

New versions are published as GitHub Releases of bapoteet/SubZeroAstro, each with a zipped app per
platform (SubZeroAstro-Windows.zip, SubZeroAstro-macOS-AppleSilicon.zip). On start-up (unless turned
off) the newest release is looked up in the background; when it is newer than this copy, a slim blue
banner on page 1 offers What's New / Update Now / Skip This Version.

Update Now downloads the zip, unpacks it next to the app, and hands over to a tiny helper script that
waits for SubZeroAstro to close, puts the new files in place and starts it again. That only works for
the packaged app in a folder the user can write to; otherwise Update Now opens the download page.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressDialog,
                               QToolButton)

REPO = "bapoteet/SubZeroAstro"
API = f"https://api.github.com/repos/{REPO}/releases?per_page=20"
PAGE = f"https://github.com/{REPO}/releases/latest"
UA = "SubZeroAstro-updater"
AUTO_KEY = "update/auto_check"          # check at start-up (on unless turned off)
SKIP_KEY = "update/skip"                # a version the user chose to skip


# ------------------------------------------------------------------ versions and releases

def parse_version(text) -> tuple:
    """'0.9.119 beta', 'v0.9.119-beta', '1.0' -> (0, 9, 119) / (1, 0); () when there are no numbers."""
    m = re.search(r"\d+(?:\.\d+)*", str(text or ""))
    return tuple(int(x) for x in m.group(0).split(".")) if m else ()


def is_newer(candidate, current) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    n = max(len(a), len(b))
    return bool(a) and a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def asset_name(platform: str | None = None) -> str | None:
    p = platform or sys.platform
    if p.startswith("win"):
        return "SubZeroAstro-Windows.zip"
    if p == "darwin":
        return "SubZeroAstro-macOS-AppleSilicon.zip"
    return None


def newest_release(releases, current, platform: str | None = None) -> dict | None:
    """The newest published release that is newer than `current` and has this platform's zip."""
    want = asset_name(platform)
    best = None
    for r in releases or []:
        if r.get("draft") or not want:
            continue
        asset = next((a for a in r.get("assets") or [] if a.get("name") == want), None)
        tag = r.get("tag_name") or ""
        if asset is None or not is_newer(tag, current):
            continue
        if best is None or is_newer(tag, best["tag"]):
            best = {"tag": tag, "version": ".".join(map(str, parse_version(tag))),
                    "name": r.get("name") or tag, "notes": (r.get("body") or "").strip(),
                    "url": asset.get("browser_download_url"), "size": int(asset.get("size") or 0),
                    "page": r.get("html_url") or PAGE}
    return best


def fetch_releases(timeout=15) -> list:
    from target_info import ssl_context       # certifi-backed: works inside the packaged Mac app too
    req = urllib.request.Request(API, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ where this copy lives

def _untranslocated(app: Path) -> Path:
    """macOS 'App Translocation': an app opened straight from Downloads (still marked as downloaded) runs
    from a hidden read-only copy. Ask macOS where the real SubZeroAstro.app is, so that one gets updated."""
    if "/AppTranslocation/" not in str(app):
        return app
    try:
        import ctypes
        cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        sec = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Security.framework/Security")
        cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long,
                                                               ctypes.c_bool]
        cf.CFURLGetFileSystemRepresentation.restype = ctypes.c_bool
        cf.CFURLGetFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_char_p,
                                                        ctypes.c_long]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        sec.SecTranslocateCreateOriginalPathForURL.restype = ctypes.c_void_p
        sec.SecTranslocateCreateOriginalPathForURL.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        raw = str(app).encode()
        url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), True)
        orig = sec.SecTranslocateCreateOriginalPathForURL(url, None) if url else None
        if url:
            cf.CFRelease(url)
        if orig:
            buf = ctypes.create_string_buffer(4096)
            ok = cf.CFURLGetFileSystemRepresentation(orig, True, buf, 4096)
            cf.CFRelease(orig)
            if ok and buf.value:
                return Path(buf.value.decode())
    except Exception:  # noqa: BLE001
        pass
    return app


def install_target(platform: str | None = None, exe: str | None = None, frozen: bool | None = None) -> Path | None:
    """What gets replaced: the app folder (Windows) or SubZeroAstro.app (Mac). None when running from source."""
    p = platform or sys.platform
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if not frozen:
        return None
    exe_p = Path(exe or sys.executable)
    if p.startswith("win"):
        return exe_p.parent
    if p == "darwin":
        for parent in exe_p.parents:
            if parent.suffix == ".app":
                return _untranslocated(parent) if exe is None else parent
    return None


def _writable(folder: Path) -> bool:
    try:
        probe = folder / f".szaupdate_{os.getpid()}"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def why_not(platform: str | None = None) -> str:
    """'' when Update Now can install by itself, else the reason (shown in the banner's tooltip and the logs)."""
    p = platform or sys.platform
    if asset_name(p) is None:
        return "no download for this kind of computer"
    target = install_target(p)
    if target is None:
        return "running from Python, not the packaged app"
    if "/AppTranslocation/" in str(target):
        return ("macOS is running SubZeroAstro from a hidden read-only copy (it was opened straight from the "
                "download) - move SubZeroAstro.app into Applications and open it from there")
    if p.startswith("win"):
        return "" if _writable(target) else f"can't write to {target}"
    # Mac: the new version goes inside the existing SubZeroAstro.app, so only the app itself must be writable
    return "" if _writable(target / "Contents") else f"can't write to {target}"


def can_self_update(platform: str | None = None) -> bool:
    return why_not(platform) == ""


# ------------------------------------------------------------------ download, unpack, hand over

def download(url: str, dest: Path, progress=None, timeout=60) -> Path:
    """Fetch `url` into `dest`; progress(done_bytes, total_bytes) may return False to cancel."""
    if url.startswith("file:"):
        req, ctx = urllib.request.Request(url), None
    else:
        from target_info import ssl_context
        req, ctx = urllib.request.Request(url, headers={"User-Agent": UA}), ssl_context()
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress is not None and progress(done, total) is False:
                raise InterruptedError("download cancelled")
    part.replace(dest)
    return dest


def unpack_windows(zip_path: Path, stage: Path) -> Path:
    """Unzip the Windows build; returns the folder that holds SubZeroAstro.exe."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(stage)
    hit = next(iter(sorted(stage.rglob("SubZeroAstro.exe"), key=lambda p: len(p.parts))), None)
    if hit is None:
        raise FileNotFoundError("SubZeroAstro.exe is not in the downloaded zip")
    return hit.parent


def windows_script(pid: int, new_dir: Path, target: Path, stage: Path) -> str:
    """Wait for this copy to close, copy the new files over it, start it again, tidy up."""
    return (
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)\r\n'
        f'robocopy "{new_dir}" "{target}" /E /R:10 /W:1 /NFL /NDL /NJH /NJS /NP >nul\r\n'
        f'start "" "{target / "SubZeroAstro.exe"}"\r\n'
        f'rmdir /S /Q "{stage}" >nul 2>&1\r\n'
    )


def mac_script(pid: int, zip_path: Path, target: Path, stage: Path) -> str:
    """Same on the Mac: ditto keeps the app's links and permissions, which plain unzipping loses. The new
    Contents are copied inside the existing SubZeroAstro.app first and only swapped in once complete, so a
    failed copy leaves the old version working (and only the app itself needs to be writable)."""
    new_app = stage / "x" / target.name
    c, cn, co = target / "Contents", target / "Contents.new", target / "Contents.old"
    return (
        "#!/bin/sh\n"
        f"while kill -0 {pid} 2>/dev/null; do sleep 1; done\n"
        f'rm -rf "{cn}" "{co}"\n'
        f'if ditto -x -k "{zip_path}" "{stage / "x"}" && [ -d "{new_app / "Contents"}" ] && '
        f'ditto "{new_app / "Contents"}" "{cn}"; then\n'
        f'  mv "{c}" "{co}" && mv "{cn}" "{c}" && rm -rf "{co}"\n'
        f'  [ -d "{c}" ] || mv "{co}" "{c}"\n'
        "fi\n"
        f'rm -rf "{cn}"\n'
        f'xattr -dr com.apple.quarantine "{target}" 2>/dev/null\n'
        f'touch "{target}"\n'
        f'open "{target}"\n'
        f'rm -rf "{stage}"\n'
    )


def prepare_handover(zip_path: Path, stage: Path, platform: str | None = None,
                     pid: int | None = None, target: Path | None = None) -> list:
    """Unpack (Windows) and write the helper script; returns the command that runs it."""
    p = platform or sys.platform
    pid = os.getpid() if pid is None else pid
    target = target or install_target(p)
    if target is None:
        raise RuntimeError("this copy of SubZeroAstro can't update itself")
    if p.startswith("win"):
        new_dir = unpack_windows(zip_path, stage / "new")
        script = stage / "update.bat"
        script.write_text(windows_script(pid, new_dir, target, stage), encoding="utf-8")
        return ["cmd", "/c", str(script)]
    script = stage / "update.sh"
    script.write_text(mac_script(pid, zip_path, target, stage), encoding="utf-8")
    return ["/bin/sh", str(script)]


def launch(cmd: list):
    if sys.platform.startswith("win"):
        # a hidden console (not DETACHED_PROCESS: then tasklist and robocopy would each pop up a window)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(cmd, start_new_session=True, close_fds=True)


def staging_dir() -> Path:
    """Next to the app when possible (same disk, so nothing big is copied twice), else the temp folder."""
    target = install_target()
    bases = ([target.parent] if target is not None else []) + [Path(tempfile.gettempdir())]
    for base in bases:
        d = base / "SubZeroAstro_update"
        try:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    raise OSError("no folder to download the update into")


# ------------------------------------------------------------------ Qt: background check and the banner

class CheckThread(QThread):
    done = Signal(object, str)          # (release or None, error text)

    def __init__(self, current, fetch=None):
        super().__init__()
        self.current = current
        self.fetch = fetch or fetch_releases

    def run(self):
        try:
            self.done.emit(newest_release(self.fetch(), self.current), "")
        except Exception as e:  # noqa: BLE001 - no network is normal; never bother the user about it
            self.done.emit(None, str(e) or e.__class__.__name__)


class DownloadThread(QThread):
    progress = Signal(int, int)
    done = Signal(str, str)             # (zip path, error text)

    def __init__(self, url, dest):
        super().__init__()
        self.url, self.dest, self.cancel = url, Path(dest), False

    def run(self):
        try:
            download(self.url, self.dest, lambda d, t: (self.progress.emit(d, t), not self.cancel)[1])
            self.done.emit(str(self.dest), "")
        except Exception as e:  # noqa: BLE001
            self.done.emit("", str(e) or e.__class__.__name__)


class UpdateBanner(QFrame):
    """Slim blue notice on page 1 when a newer SubZeroAstro is out."""

    def __init__(self, settings, current, parent=None):
        super().__init__(parent)
        self.settings, self.current = settings, current
        self.release = None
        self.checker = None
        self.downloader = None
        self.manual = False
        self.setStyleSheet("QFrame { background:#14304a; border-bottom:1px solid #3b82c4; }"
                           "QLabel { color:#d8e9fb; font-size:10.5pt; background:transparent; border:none; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 5, 8, 5)
        self.label = QLabel()
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        self.b_news = QToolButton()
        self.b_news.setText("What's New")
        self.b_news.clicked.connect(self.show_notes)
        self.b_update = QToolButton()
        self.b_update.setText("Update Now")
        self.b_update.setStyleSheet("QToolButton { background:#2b6cb0; color:white; border-radius:4px; "
                                    "padding:4px 14px; font-weight:bold; } QToolButton:hover { background:#3182ce; }")
        self.b_update.clicked.connect(self.update_now)
        self.b_skip = QToolButton()
        self.b_skip.setText("Skip This Version")
        self.b_skip.setToolTip("Don't tell me about this version again (Help › Check for Updates still finds it)")
        self.b_skip.clicked.connect(self.skip)
        b_x = QToolButton()
        b_x.setText("✕")
        b_x.setToolTip("Hide for now")
        b_x.clicked.connect(self.hide)
        for b in (self.b_news, self.b_update, self.b_skip, b_x):
            lay.addWidget(b)
        self.hide()

    # -- checking
    def auto_enabled(self) -> bool:
        return self.settings.value(AUTO_KEY, True, type=bool)

    def start_check(self, manual=False, fetch=None):
        if not manual and not self.auto_enabled():
            return
        if self.checker is not None and self.checker.isRunning():
            return
        self.manual = manual
        self.checker = CheckThread(self.current, fetch)
        self.checker.done.connect(self.on_checked)
        self.checker.start()

    def on_checked(self, release, error=""):
        manual, self.manual = self.manual, False
        if release is None:
            self.hide()
            if manual:
                self._info("Check for Updates",
                           f"Couldn't reach GitHub to check for updates.\n\n({error})" if error
                           else f"You have the newest version of SubZeroAstro ({self.current}).")
            return
        self.show_release(release, force=manual)

    def show_release(self, release, force=False):
        self.release = release
        if not force and release["tag"] == (self.settings.value(SKIP_KEY, "") or ""):
            self.hide()
            return
        size = f" ({release['size'] / 1e6:.0f} MB)" if release.get("size") else ""
        self.label.setText(f"<b>SubZeroAstro {release['version']} is available</b> — you have {self.current}."
                           f"<br><small>Update Now downloads it{size}, closes SubZeroAstro and starts the new "
                           f"version.</small>" if can_self_update() else
                           f"<b>SubZeroAstro {release['version']} is available</b> — you have {self.current}.")
        reason = why_not()
        self.b_update.setText("Update Now" if not reason else "Download…")
        self.b_update.setToolTip("Download, install and restart" if not reason else
                                 f"Opens the download page - this copy can't update itself: {reason}")
        self.show()

    def skip(self):
        if self.release:
            self.settings.setValue(SKIP_KEY, self.release["tag"])
        self.hide()

    def show_notes(self):
        if not self.release:
            return
        box = QMessageBox(self.window())
        box.setWindowTitle(f"What's New in {self.release['version']}")
        box.setText(f"<b>{self.release['name']}</b>")
        box.setInformativeText(self.release["notes"] or "No notes for this version.")
        box.setStandardButtons(QMessageBox.Close)
        box.exec()

    # -- installing
    def update_now(self, ask=True):
        if not self.release:
            return
        reason = why_not()
        if reason:
            if "AppTranslocation" in str(install_target()) or "read-only copy" in reason:
                self._info("Update SubZeroAstro", "SubZeroAstro can't update itself here: " + reason
                           + ".\n\nAfter that, Update Now will work. The download page opens now.")
            QDesktopServices.openUrl(QUrl(self.release.get("page") or PAGE))
            return
        if ask and QMessageBox.question(
                self.window(), "Update SubZeroAstro",
                f"Download SubZeroAstro {self.release['version']} and install it now?\n\n"
                "SubZeroAstro will close and start again. Anything still running (stacking, processing) "
                "will be stopped, so let it finish first. Your settings and projects are kept.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
            return
        self.stage = staging_dir()
        dest = self.stage / (asset_name() or "update.zip")
        self.dlg = QProgressDialog("Downloading the update…", "Cancel", 0, 100, self.window())
        self.dlg.setWindowTitle("Update SubZeroAstro")
        self.dlg.setMinimumDuration(0)
        self.downloader = DownloadThread(self.release["url"], dest)
        self.downloader.progress.connect(self._progress)
        self.downloader.done.connect(self._downloaded)
        self.dlg.canceled.connect(lambda: setattr(self.downloader, "cancel", True))
        self.downloader.start()

    def _progress(self, done, total):
        if total:
            self.dlg.setValue(int(100 * done / total))
            self.dlg.setLabelText(f"Downloading the update… {done / 1e6:.0f} of {total / 1e6:.0f} MB")

    def _downloaded(self, path, error):
        self.dlg.close()
        if not path:
            shutil.rmtree(self.stage, ignore_errors=True)
            if "cancelled" not in error:
                self._info("Update SubZeroAstro", f"The download failed:\n\n{error}\n\n"
                                                  "Try again later, or download it from the GitHub page.")
            return
        try:
            cmd = prepare_handover(Path(path), self.stage)
            launch(cmd)
        except Exception as e:  # noqa: BLE001
            self._info("Update SubZeroAstro", f"The update couldn't be installed:\n\n{e}")
            return
        QApplication.closeAllWindows()
        QApplication.quit()

    def _info(self, title, text):
        QMessageBox.information(self.window(), title, text)
