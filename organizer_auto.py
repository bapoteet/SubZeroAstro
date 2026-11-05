"""
organizer_auto.py - Organize My Astro Files on a schedule, with nobody at the computer.

A folder where new frames arrive (for example the NAS folder N.I.N.A.'s copies land in) is organized into your
library every day, twice a week or once a week: the same rules as the Organize window (Your Rigs, the layout you
chose), always a move, so the incoming folder empties out. Files that can't be filed - no rig matches, a
duplicate, unreadable - stay where they are and are listed in the run's report. Every run can be undone with
Undo Last Run in the Organize window.

The schedule is the computer's own: Windows Task Scheduler (schtasks) or a macOS LaunchAgent, which starts
SubZeroAstro with --auto-organize; it runs without a window and quits when done.
"""
from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

import organizer as O

TASK_NAME = "SubZeroAstro Auto-Organize"
MAC_LABEL = "io.github.bapoteet.subzeroastro.autoorganize"
SETTLE_MIN = 60            # a file changed in the last hour may still be arriving: it waits for the next run
LOCK_STALE_H = 12
FREQS = {"daily": "Every day", "twice": "Twice a week (Monday and Thursday)", "weekly": "Once a week (Sunday)"}
_DAYS = {"twice": ["MON", "THU"], "weekly": ["SUN"]}
_MAC_DAY = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings("FITSBlasterWin", "FITSBlasterWin")


def load_config(settings=None) -> dict:
    s = settings or _settings()
    return {"src": s.value("auto_org/src", "") or "", "dst": s.value("auto_org/dst", "") or "",
            "layout": s.value("auto_org/layout", "") or s.value("organize/layout", "rig") or "rig",
            "freq": s.value("auto_org/freq", "daily") or "daily", "time": s.value("auto_org/time", "11:00") or "11:00",
            "on": s.value("auto_org/on", False, type=bool)}


def save_config(cfg, settings=None):
    s = settings or _settings()
    for k in ("src", "dst", "layout", "freq", "time", "on"):
        if k in cfg:
            s.setValue(f"auto_org/{k}", cfg[k])


def problems(cfg) -> list:
    """Why a run can't go ahead (empty = fine)."""
    out = []
    src, dst = cfg.get("src", "").strip(), cfg.get("dst", "").strip()
    if not src:
        out.append("no incoming folder chosen")
    if not dst:
        out.append("no library folder chosen")
    if src and dst:
        a, b = os.path.normcase(os.path.abspath(src)), os.path.normcase(os.path.abspath(dst))
        org = os.path.normcase(os.path.abspath(str(O.org_base(dst))))
        if a == b or a == org or a.startswith(org + os.sep) or org.startswith(a + os.sep):
            out.append("the incoming folder must be outside the organized library (to re-sort the library itself, "
                       "use Change My Library's Layout in the Organize window)")
    return out


def run(cfg=None, settings=None, progress=None, now=None) -> dict:
    """One automatic run. Returns a report {time, filed, left: {why: n}, log, error}; also saved as text in the
    library's organize folder and as the 'last run' line shown in the app."""
    s = settings or _settings()
    cfg = cfg or load_config(s)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    rep = {"time": stamp, "filed": 0, "left": {}, "log": "", "error": ""}
    bad = problems(cfg)
    src = Path(cfg.get("src", ""))
    if not bad and not src.is_dir():
        bad = [f"the incoming folder {src} can't be reached (is the NAS on?)"]
    riglist = []
    try:
        riglist = json.loads(s.value("organize/riglist", "[]") or "[]")
    except (TypeError, ValueError):
        pass
    if not bad and not riglist:
        bad = ["Your Rigs is empty - set up your rigs once in the Organize window"]
    if bad:
        rep["error"] = "; ".join(bad)
        _finish(rep, s, None)
        return rep
    lib = Path(cfg["dst"])
    base = O.org_base(lib)
    lock = base / O.LOG_DIR / "auto_organize.lock"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_H * 3600:
            rep["error"] = "another automatic run is still going"
            _finish(rep, s, None)
            return rep
        lock.write_text(stamp, encoding="utf-8")
    except OSError as e:
        rep["error"] = f"the library {lib} can't be written ({e})"
        _finish(rep, s, None)
        return rep
    try:
        try:
            tol = float(s.value("organize/rig_tol", 3)) / 100.0
        except (TypeError, ValueError):
            tol = 0.03
        t_now = now if now is not None else time.time()
        paths, settling = [], 0
        for p in O.scan(src):
            try:
                if t_now - p.stat().st_mtime < SETTLE_MIN * 60:
                    settling += 1                       # still arriving, perhaps: next run
                    continue
            except OSError:
                continue
            paths.append(p)
        if progress:
            progress(f"{len(paths)} file(s) to look at")
        items = O.plan(O.classify(paths, library=lib, layout=cfg.get("layout") or "rig", root=src,
                                  riglist=riglist, tol=tol), lib)
        todo = [i for i in items if i.status == "move"]
        if todo:
            log = O.execute(items, lib, move=True,
                            progress=(lambda t, i, n: progress(t)) if progress else None)
            rep["log"] = str(log)
            try:
                d = json.loads(log.read_text(encoding="utf-8"))
                rep["filed"] = len(d.get("done", []))
                if d.get("failed"):
                    rep["left"]["could not be copied (see the log)"] = len(d["failed"])
            except (OSError, ValueError):
                rep["filed"] = len(todo)
        for i in items:
            if i.status == "move" or i.status == "here":
                continue
            why = {"norig": "no rig matched - Your Rigs / Which Rig? in the Organize window",
                   "duplicate": "duplicates - Step 2 in the Organize window",
                   "error": "could not be read"}.get(i.status, i.why or i.status)
            rep["left"][why] = rep["left"].get(why, 0) + 1
        if settling:
            rep["left"][f"changed in the last {SETTLE_MIN} min (next run)"] = settling
    except Exception as e:  # noqa: BLE001 - a scheduled run must report, never hang
        rep["error"] = str(e)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    _finish(rep, s, base)
    return rep


def describe(rep) -> str:
    if rep.get("error"):
        return f"{rep['time']}: not run - {rep['error']}"
    t = f"{rep['time']}: {rep['filed']} file(s) filed"
    if rep["left"]:
        t += "; left in the incoming folder: " + ", ".join(f"{n} {w}" for w, n in rep["left"].items())
    return t


def _finish(rep, s, base):
    text = describe(rep)
    try:
        s.setValue("auto_org/last", text)
        s.sync()
    except Exception:  # noqa: BLE001
        pass
    if base is not None:
        try:
            with open(base / O.LOG_DIR / "auto_organize_history.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


# ------------------------------------------------------------------ the computer's scheduler
def _command() -> list:
    """How the scheduler starts SubZeroAstro: the installed app, or Python + the script when run from source."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--auto-organize"]
    return [sys.executable, str(Path(__file__).resolve().parent / "subzeroastro.py"), "--auto-organize"]


def _hhmm(t) -> tuple:
    try:
        h, m = (int(x) for x in str(t).split(":")[:2])
        return max(0, min(23, h)), max(0, min(59, m))
    except ValueError:
        return 11, 0


def windows_args(cfg) -> list:
    cmd = _command()
    tr = " ".join(f'"{c}"' if " " in c or c.endswith(".exe") else c for c in cmd)
    h, m = _hhmm(cfg.get("time"))
    a = ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/TR", tr, "/ST", f"{h:02d}:{m:02d}"]
    days = _DAYS.get(cfg.get("freq"))
    a += ["/SC", "WEEKLY", "/D", ",".join(days)] if days else ["/SC", "DAILY"]
    return a


def mac_plist(cfg) -> dict:
    h, m = _hhmm(cfg.get("time"))
    days = _DAYS.get(cfg.get("freq"))
    when = ([{"Hour": h, "Minute": m, "Weekday": _MAC_DAY[d]} for d in days] if days else [{"Hour": h, "Minute": m}])
    return {"Label": MAC_LABEL, "ProgramArguments": _command(), "StartCalendarInterval": when,
            "RunAtLoad": False}


def _mac_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{MAC_LABEL}.plist"


def _run(args) -> tuple:
    try:
        kw = {"capture_output": True, "text": True, "timeout": 60}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000           # no console window flashing up
        r = subprocess.run(args, **kw)
        return r.returncode == 0, (r.stderr or r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


def schedule(cfg) -> tuple:
    """Create (or replace) the scheduled task. Returns (ok, message)."""
    if sys.platform == "win32":
        ok, msg = _run(windows_args(cfg))
        return ok, (f"Windows Task Scheduler: '{TASK_NAME}' - {FREQS.get(cfg.get('freq'), '')} at "
                    f"{cfg.get('time')}" if ok else f"Windows Task Scheduler said: {msg}")
    if sys.platform == "darwin":
        p = _mac_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _run(["launchctl", "unload", str(p)])
            with open(p, "wb") as f:
                plistlib.dump(mac_plist(cfg), f)
        except OSError as e:
            return False, str(e)
        ok, msg = _run(["launchctl", "load", str(p)])
        return ok, (f"macOS: {FREQS.get(cfg.get('freq'), '')} at {cfg.get('time')}" if ok else msg)
    return False, "automatic runs are set up on Windows and macOS only"


def unschedule() -> tuple:
    if sys.platform == "win32":
        ok, msg = _run(["schtasks", "/Delete", "/F", "/TN", TASK_NAME])
        return True, ("removed" if ok else msg)
    if sys.platform == "darwin":
        p = _mac_path()
        _run(["launchctl", "unload", str(p)])
        try:
            p.unlink()
        except OSError:
            pass
        return True, "removed"
    return True, ""


def main_auto() -> int:
    """subzeroastro --auto-organize: one run, no window."""
    try:
        import session_log
        session_log.start("(automatic organize run)")
        session_log.write("organizer", "automatic run started")
    except Exception:  # noqa: BLE001
        pass
    rep = run()
    try:
        import session_log
        session_log.write("organizer", "automatic run: " + describe(rep))
    except Exception:  # noqa: BLE001
        pass
    return 1 if rep.get("error") else 0
