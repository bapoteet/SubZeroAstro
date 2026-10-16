"""
zip_logs.py - "Zip My Logs": everything needed to report a problem, in one zip.

Collects the log of every open page, the stack logs of the masters just made, page 3's
processing report, and a short note about this computer (SubZeroAstro version, system,
screen, which helper programs were found). Saved to the Desktop and shown in the folder.
No pictures are included - only text.
"""

from __future__ import annotations

import platform
import sys
import time
import zipfile
from pathlib import Path


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip() or "page"


def system_note() -> str:
    import ui_header
    lines = [f"SubZeroAstro {ui_header.APP_VERSION or '?'}",
             f"Saved {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"System: {platform.system()} {platform.release()} ({platform.machine()})"
             + (f", macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else ""),
             f"Python {platform.python_version()}" + ("  (packaged app)" if getattr(sys, "frozen", False) else "")]
    try:
        import numpy
        import PySide6
        lines.append(f"numpy {numpy.__version__}, PySide6 {PySide6.__version__}")
    except Exception:  # noqa: BLE001
        pass
    try:
        from PySide6.QtWidgets import QApplication
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            lines.append(f"Screen: {g.width()} x {g.height()} available, scale {scr.devicePixelRatio():g}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import tool_check
        from PySide6.QtCore import QSettings
        st = QSettings("FITSBlasterWin", "FITSBlasterWin")
        for r in tool_check.check(tool_check.settings_paths(st)):
            lines.append(f"{r['name']}: {r['detail']}" + (f"  [{r['path']}]" if r["path"] else ""))
    except Exception as e:  # noqa: BLE001
        lines.append(f"(tool check failed: {e})")
    return "\n".join(lines) + "\n"


def collect(windows) -> dict:
    """{name in zip: text or Path} from the open pages."""
    out = {"00_about_this_computer.txt": system_note()}
    n = 1
    for w in windows:
        title = getattr(getattr(w, "app_header", None), "page", None)
        title = title.text() if title is not None else w.windowTitle()
        view = getattr(w, "log_view", None)
        if view is not None:
            try:
                out[f"{n:02d}_{_safe(title)}_log.txt"] = view.toPlainText() + "\n"
                n += 1
            except RuntimeError:
                pass
        if hasattr(w, "frames") and hasattr(w, "sources"):         # page 1: what is loaded
            try:
                fr = list(w.frames.values())
                info = [f"Sources: {[str(s[1]) if s[0] == 'folder' else f'{len(s[1])} files' for s in w.sources]}",
                        f"Frames: {len(fr)}, rejected {sum(1 for f in fr if f.rejected)}, "
                        f"flagged {sum(1 for f in fr if f.flagged and not f.rejected)}",
                        f"Filters: {sorted({(f.info or {}).get('filter') or 'No filter' for f in fr})}",
                        f"Status: {w.status_label.text()}" if hasattr(w, "status_label") else ""]
                out[f"{n:02d}_{_safe(title)}_summary.txt"] = "\n".join(info) + "\n"
                n += 1
            except Exception:  # noqa: BLE001
                pass
        for r in getattr(w, "last_results", None) or []:
            files = getattr(r, "files", None) if not isinstance(r, dict) else r.get("files")
            for f in files or []:
                p = Path(f)
                if p.suffix.lower() == ".txt" and p.exists():
                    out[f"files/{p.name}"] = p
            if isinstance(r, dict) and r.get("out_dir"):
                for p in Path(r["out_dir"]).glob("*_PixInsight_next_steps.txt"):
                    out[f"files/{p.name}"] = p
    return out


def zip_my_logs(parent=None, folder=None) -> Path | None:
    """Write the zip (Desktop by default) and show it. Returns its path."""
    from PySide6.QtWidgets import QApplication, QMessageBox
    import ui_header
    main = ui_header.main_page()
    wins = [main] if main is not None else []
    wins += [w for w in ui_header.other_pages(main) if w.isVisible()] if main is not None else []
    if not wins:
        wins = [w for w in QApplication.topLevelWidgets() if getattr(w, "app_header", None) is not None]
    items = collect(wins)
    base = Path(folder) if folder else (Path.home() / "Desktop" if (Path.home() / "Desktop").is_dir() else Path.home())
    path = base / f"SubZeroAstro_Logs_{time.strftime('%Y-%m-%d_%H%M%S')}.zip"
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, val in items.items():
                if isinstance(val, Path):
                    z.write(val, name)
                else:
                    z.writestr(name, val)
    except OSError as e:
        if parent is not None:
            QMessageBox.warning(parent, "Zip My Logs", f"Could not save the logs:\n\n{e}")
        return None
    if parent is not None and not getattr(parent, "synchronous", False) and folder is None:
        try:
            import tools
            tools.reveal([path])
        except Exception:  # noqa: BLE001
            pass
        QMessageBox.information(parent, "Zip My Logs",
                                f"<b>Your logs are zipped</b> ({len(items)} files, text only - no pictures):"
                                f"<br><br>{path}<br><br>The folder is open for you. Attach the zip to your "
                                "message or bug report.")
    return path
