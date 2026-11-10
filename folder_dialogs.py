"""
folder_dialogs.py - help for SubZeroAstro's own folder windows (the ones that take several folders at once).

Qt's own folder window can't browse the network (Windows' "Network" place is missing), so a NAS share that is not
mapped to a drive letter could not be picked. This adds:
  * a "Network / NAS Folder…" button that opens the system's folder window (which can browse the network) and
    jumps to the folder picked there;
  * the network shares used before, in the sidebar;
  * typing \\\\server\\share (or smb://server/share on a Mac) into the folder box works too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QUrl
from PySide6.QtWidgets import QFileDialog, QGridLayout, QPushButton

_KEY = "net/recent_shares"


def _settings():
    return QSettings("FITSBlasterWin", "FITSBlasterWin")


def share_root(p) -> str | None:
    """\\\\server\\share of a network path (Windows), or /Volumes/<share> on a Mac; None for a local folder."""
    s = str(p)
    if s.startswith(("\\\\", "//")):
        parts = [x for x in s.replace("/", "\\").split("\\") if x]
        if len(parts) >= 2:
            return "\\\\" + parts[0] + "\\" + parts[1]
    if sys.platform == "darwin" and s.startswith("/Volumes/"):
        parts = Path(s).parts
        if len(parts) >= 3:
            return str(Path(*parts[:3]))
    return None


def remember(paths):
    """Keep the network shares of these folders for the sidebar next time."""
    roots = [r for r in (share_root(p) for p in paths) if r]
    if not roots:
        return
    try:
        st = _settings()
        old = st.value(_KEY, [], type=list) or []
        new = list(dict.fromkeys(roots + [x for x in old if x not in roots]))[:8]
        st.setValue(_KEY, new)
    except Exception:  # noqa: BLE001 - only a convenience
        pass


def network_places() -> list:
    """Network shares used before (still reachable), for the sidebar."""
    out = []
    try:
        for s in _settings().value(_KEY, [], type=list) or []:
            try:
                if Path(s).exists():
                    out.append(Path(s))
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


def add_network(dlg: QFileDialog):
    """Add the remembered shares to a Qt folder window's sidebar and a 'Network / NAS Folder…' button."""
    try:
        urls = list(dlg.sidebarUrls())
        have = {u.toLocalFile() for u in urls}
        for p in network_places():
            if str(p) not in have:
                urls.append(QUrl.fromLocalFile(str(p)))
        dlg.setSidebarUrls(urls)
    except Exception:  # noqa: BLE001
        pass
    btn = QPushButton("Network / NAS Folder…")
    btn.setToolTip("Browse the network (a NAS or another computer) with the system's folder window. You can also "
                   "type the address into the folder box, e.g. \\\\DS1522\\Astro"
                   + (" or smb://DS1522/Astro" if sys.platform == "darwin" else ""))

    def pick():
        d = QFileDialog.getExistingDirectory(dlg, "Choose a network folder", "")
        if d:
            d = os.path.normpath(d)
            remember([d])
            dlg.setDirectory(str(Path(d).parent) if Path(d).parent != Path(d) else d)
            dlg.selectFile(d)
    btn.clicked.connect(pick)
    lay = dlg.layout()
    if isinstance(lay, QGridLayout):
        lay.addWidget(btn, lay.rowCount(), 0, 1, max(1, lay.columnCount()))
    elif lay is not None:
        lay.addWidget(btn)
    return btn


# ---------------------------------------------------------------- "Captured" column
# The folder window's Date Modified is when a folder was last changed on this computer - after a re-sort of the
# library that's the re-sort's date for every folder. "Captured" is when the frames inside were actually taken:
# from the night folders (2026-07-05 …) under it, or failing those from DATE-OBS in the frames' own headers.
import re as _re
import threading as _threading

_NIGHT = _re.compile(r"^(\d{4}-\d{2}-\d{2})")
_DATEOBS = _re.compile(rb"DATE-OBS=\s*'?\s*(\d{4}-\d{2}-\d{2})")
_FITS = (".fits", ".fit", ".fts")
_SLOTS = _threading.Semaphore(6)          # a few looks at a time (a NAS answers slower with many at once)


def _header_date(p: Path):
    try:
        with open(p, "rb") as f:
            m = _DATEOBS.search(f.read(2880 * 6))
        return m.group(1).decode() if m else None
    except OSError:
        return None


def captured_range(folder) -> str:
    """When the frames in this folder were taken: '2026-07-05', or '2026-07-05 → 07-22 (6 nights)', or '' when
    no frames or dates are found. Night folders are read first (one look at a few folders); only without them
    are a couple of frames' headers read."""
    folder = Path(folder)
    nights, fits, seen = set(), [], 0
    m = _NIGHT.match(folder.name)
    if m:
        nights.add(m.group(1))
    level = [folder]
    for _depth in range(4):
        nxt = []
        for d in level:
            try:
                with os.scandir(d) as it:
                    for e in it:
                        n = e.name
                        if n.startswith((".", "#", "@", "$")):
                            continue
                        try:
                            isdir = e.is_dir()
                        except OSError:
                            continue
                        if isdir:
                            mm = _NIGHT.match(n)
                            if mm:
                                nights.add(mm.group(1))
                            else:
                                nxt.append(Path(e.path))
                        elif n.lower().endswith(_FITS) and len(fits) < 4000:
                            fits.append(Path(e.path))
            except OSError:
                pass
            seen += 1
            if seen > 400:
                break
        if nights or seen > 400:
            break
        level = nxt
    if not nights and fits:
        fits.sort()
        for p in (fits[0], fits[-1]) if len(fits) > 1 else (fits[0],):
            d = _header_date(p)
            if d:
                nights.add(d)
    if not nights:
        return ""
    lo, hi = min(nights), max(nights)
    if lo == hi:
        return lo
    end = hi[5:] if hi[:4] == lo[:4] else hi
    many = f"  ({len(nights)} nights)" if len(nights) > 2 else ""
    return f"{lo} → {end}{many}"


def add_captured_column(dlg: QFileDialog):
    """Show 'Captured' (when the frames inside were taken) in a Qt folder window's detail view, sortable. It takes
    the Kind column's place - in a folder window every row is a 'Folder' - so Date Modified stays."""
    from PySide6.QtCore import QObject, QSortFilterProxyModel, Qt, QTimer, Signal
    from PySide6.QtWidgets import QFileSystemModel

    COL = 2                                         # Name, Size, Kind -> Captured, Date Modified

    class _Relay(QObject):
        done = Signal(str)

    class CapturedProxy(QSortFilterProxyModel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._cache = {}
            self._pending = set()
            self._lock = _threading.Lock()
            self._relay = _Relay()
            self._relay.done.connect(self._ready)
            self._resort = QTimer(self)
            self._resort.setSingleShot(True)
            self._resort.setInterval(400)
            self._resort.timeout.connect(self._sort_again)
            self.setSortCaseSensitivity(Qt.CaseInsensitive)

        def _fs(self):
            m = self.sourceModel()
            return m if isinstance(m, QFileSystemModel) else None

        def captured(self, src) -> "str | None":
            """The text for a source row (None while it is being worked out)."""
            fs = self._fs()
            if fs is None or not src.isValid() or not fs.isDir(src):
                return ""
            path = fs.filePath(src)
            with self._lock:
                if path in self._cache:
                    return self._cache[path]
                if path not in self._pending:
                    self._pending.add(path)
                    _threading.Thread(target=self._work, args=(path,), daemon=True).start()
            return None

        def _work(self, path):
            try:
                with _SLOTS:
                    txt = captured_range(path)
            except Exception:  # noqa: BLE001 - a date is a nicety
                txt = ""
            with self._lock:
                self._cache[path] = txt
                self._pending.discard(path)
            self._relay.done.emit(path)

        def _ready(self, path):
            fs = self._fs()
            if fs is None:
                return
            src = fs.index(path)
            if src.isValid():
                cell = self.mapFromSource(src.siblingAtColumn(COL))
                if cell.isValid():
                    self.dataChanged.emit(cell, cell)
            if self.sortColumn() == COL:
                self._resort.start()                 # dates still coming in: put the rows in order again

        def _sort_again(self):
            if self.sortColumn() == COL:
                self.sort(COL, self.sortOrder())

        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if orientation == Qt.Horizontal and section == COL:
                if role == Qt.DisplayRole:
                    return "Captured"
                if role == Qt.ToolTipRole:
                    return ("When the frames inside were taken (from the night folders, or the frames' own "
                            "DATE-OBS) - not when the folder was made or changed on this computer. "
                            "Click to sort by it.")
            return super().headerData(section, orientation, role)

        def data(self, idx, role=Qt.DisplayRole):
            if idx.isValid() and idx.column() == COL and role in (Qt.DisplayRole, Qt.ToolTipRole):
                try:
                    txt = self.captured(self.mapToSource(idx).siblingAtColumn(0))
                except Exception:  # noqa: BLE001
                    txt = ""
                return "…" if txt is None else txt
            return super().data(idx, role)

        def lessThan(self, left, right):
            try:
                return self._less(left, right)
            except Exception:  # noqa: BLE001 - a sort must never fail
                return False

        def _less(self, left, right):
            fs = self._fs()
            if fs is None:
                return False
            col = left.column()
            if col == COL:
                a = self.captured(left.siblingAtColumn(0)) or ""
                b = self.captured(right.siblingAtColumn(0)) or ""
                # folders without a date (or still being read) always go to the bottom
                ka = (0, a) if a else (1, "")
                kb = (0, b) if b else (1, "")
                if ka[0] != kb[0]:
                    return (ka[0] < kb[0]) == (self.sortOrder() == Qt.AscendingOrder)
                if a != b:
                    return a < b
                return fs.fileName(left).lower() < fs.fileName(right).lower()
            if col == 1:
                return fs.size(left) < fs.size(right)
            if col == 3:
                return fs.lastModified(left) < fs.lastModified(right)
            return fs.fileName(left).lower() < fs.fileName(right).lower()

    proxy = CapturedProxy(dlg)
    dlg.setProxyModel(proxy)
    return proxy
