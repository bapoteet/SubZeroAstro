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
    """Add a 'Captured' column (when the frames inside were taken) to a Qt folder window's detail view."""
    from PySide6.QtCore import QIdentityProxyModel, QModelIndex, QObject, QPersistentModelIndex, Qt, Signal
    from PySide6.QtWidgets import QFileSystemModel, QTreeView

    class _Relay(QObject):
        done = Signal(str)

    class CapturedProxy(QIdentityProxyModel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._map = {}
            self._cache = {}
            self._pending = set()
            self._lock = _threading.Lock()
            self._relay = _Relay()
            self._relay.done.connect(self._ready)

        def _extra(self):
            src = self.sourceModel()
            return src.columnCount(QModelIndex()) if src is not None else 4

        def columnCount(self, parent=QModelIndex()):
            return self._extra() + 1

        def index(self, row, column, parent=QModelIndex()):
            if column != self._extra():
                return super().index(row, column, parent)
            base = super().index(row, 0, parent)
            if not base.isValid():
                return QModelIndex()
            src = super().mapToSource(base)
            key = src.internalId()
            self._map[key] = QPersistentModelIndex(src)
            return self.createIndex(row, column, key)

        def mapToSource(self, proxy):
            if proxy.isValid() and proxy.column() == self._extra():
                p = self._map.get(proxy.internalId())
                if p is None or not p.isValid():
                    return QModelIndex()
                return self.sourceModel().index(p.row(), 0, p.parent())
            return super().mapToSource(proxy)

        def mapFromSource(self, src):
            return super().mapFromSource(src)

        def parent(self, child=None):
            if child is None:
                return super().parent()
            if child.isValid() and child.column() == self._extra():
                src = self.mapToSource(child)
                return super().mapFromSource(src.parent()) if src.isValid() else QModelIndex()
            return super().parent(child)

        def sibling(self, row, column, idx):
            if idx.isValid() and (column == self._extra() or idx.column() == self._extra()):
                return self.index(row, column, self.parent(idx))
            return super().sibling(row, column, idx)

        def hasChildren(self, parent=QModelIndex()):
            if parent.isValid() and parent.column() == self._extra():
                return False
            return super().hasChildren(parent)

        def rowCount(self, parent=QModelIndex()):
            if parent.isValid() and parent.column() == self._extra():
                return 0
            return super().rowCount(parent)

        def flags(self, idx):
            if idx.isValid() and idx.column() == self._extra():
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
            return super().flags(idx)

        def headerData(self, section, orientation, role=Qt.DisplayRole):
            if orientation == Qt.Horizontal and section == self._extra():
                if role == Qt.DisplayRole:
                    return "Captured"
                if role == Qt.ToolTipRole:
                    return ("When the frames inside were taken (from the night folders, or the frames' own "
                            "DATE-OBS) - not when the folder was made or changed on this computer")
                return None
            return super().headerData(section, orientation, role)

        def data(self, idx, role=Qt.DisplayRole):
            if idx.isValid() and idx.column() == self._extra():
                if role not in (Qt.DisplayRole, Qt.ToolTipRole):
                    return None
                src = self.mapToSource(idx)
                model = self.sourceModel()
                if not src.isValid() or not isinstance(model, QFileSystemModel) or not model.isDir(src):
                    return None
                path = model.filePath(src)
                with self._lock:
                    if path in self._cache:
                        return self._cache[path]
                    if path not in self._pending:
                        self._pending.add(path)
                        _threading.Thread(target=self._work, args=(path,), daemon=True).start()
                return "…"
            return super().data(idx, role)

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
            model = self.sourceModel()
            if model is None:
                return
            try:
                src = model.index(path)
            except Exception:  # noqa: BLE001
                return
            if not src.isValid():
                return
            col0 = super().mapFromSource(src)
            if col0.isValid():
                cell = self.index(col0.row(), self._extra(), col0.parent())
                self.dataChanged.emit(cell, cell)

    proxy = CapturedProxy(dlg)
    dlg.setProxyModel(proxy)
    for v in dlg.findChildren(QTreeView):
        try:
            v.header().resizeSection(proxy._extra(), 230)
        except Exception:  # noqa: BLE001
            pass
    return proxy
