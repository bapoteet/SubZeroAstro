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
