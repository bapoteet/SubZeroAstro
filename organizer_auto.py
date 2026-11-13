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
import re
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
            "on": s.value("auto_org/on", False, type=bool),
            "jpgs": s.value("organize/delete_jpgs", True, type=bool),
            "empty": s.value("organize/delete_empty", False, type=bool),
            "other": s.value("organize/delete_other", False, type=bool)}


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


def _note(text):
    """A line in the session log (so Zip My Logs shows which files an automatic run left, and why)."""
    try:
        import session_log
        session_log.write("organizer", text)
    except Exception:  # noqa: BLE001
        pass


def run(cfg=None, settings=None, progress=None, now=None) -> dict:
    """One automatic run. Returns a report {time, filed, left: {why: n}, log, error}; also saved as text in the
    library's organize folder and as the 'last run' line shown in the app."""
    s = settings or _settings()
    cfg = cfg or load_config(s)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    rep = {"time": stamp, "filed": 0, "left": {}, "log": "", "error": ""}
    bad = problems(cfg)
    src = Path(cfg.get("src", ""))
    for k in ("src", "dst"):                         # a Mac: NAS folders must be mounted - remember / reconnect
        if cfg.get(k):
            ensure_mounted(cfg[k], s)
    lib0 = Path(cfg.get("dst", ""))
    if not bad and not src.is_dir():
        bad = [f"the incoming folder {src} can't be reached ({_unreachable(src)})"]
    if not bad and not lib0.is_dir():
        bad = [f"your library {lib0} can't be reached ({_unreachable(lib0)})"]
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
        if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_H * 3600 and _lock_alive(lock):
            rep["error"] = "another automatic run is still going"
            _finish(rep, s, None)
            return rep
        lock.write_text(f"{stamp}|{os.getpid()}|{_host()}", encoding="utf-8")
    except OSError as e:
        rep["error"] = f"the library {lib} can't be written ({e})"
        _finish(rep, s, None)
        return rep
    try:
        try:
            tol = float(s.value("organize/rig_tol", 3)) / 100.0
        except (TypeError, ValueError):
            tol = 0.03
        if s.value("asiair/with_auto", False, type=bool):       # the ASIAIRs' new subs first
            try:
                import asiair as A
                units = A.load_units(s)
                inc = (s.value("asiair/incoming", "") or "").strip() or str(src)
                if units:
                    if progress:
                        progress("Fetching from your ASIAIRs…")
                    reps = A.fetch_all(units, inc, free=s.value("asiair/free", False, type=bool))
                    rep["asiair"] = A.describe(reps).replace("\n", "; ")
                    try:
                        A.record_last(s, reps)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as e:  # noqa: BLE001
                rep["asiair"] = f"ASIAIR fetch failed: {e}"
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
        if cfg.get("jpgs", True):
            rep["jpgs"] = O.delete_previews(items, lib, roots=[src])   # the ASIAIR's .jpg previews
        if cfg.get("empty", False):                          # frames that are all zeros (never written)
            gone = O.delete_empty(items)
            rep["empty"] = len(gone)
            for g in gone:
                _note(f"deleted empty frame (all zeros): {g}")
        if todo:
            log = O.execute(items, lib, move=True, keep=[src],
                            progress=(lambda t, i, n: progress(t)) if progress else None)
            rep["log"] = str(log)
            try:
                d = json.loads(log.read_text(encoding="utf-8"))
                rep["filed"] = len(d.get("done", []))
                if cfg.get("other", False):                  # everything left that isn't a frame
                    gone = O.delete_other([src], lib, min_age_s=SETTLE_MIN * 60)
                    rep["other"] = len(gone)
                    for g in gone:
                        _note(f"deleted (not a frame): {g}")
                if d.get("failed"):
                    rep["left"]["could not be copied (see the log)"] = len(d["failed"])
                    _note(f"could not be copied ({len(d['failed'])}; full list in {log}):")
                    for fl in d["failed"][:40]:
                        _note(f"  {fl.get('src')}  -  {fl.get('error')}")
            except (OSError, ValueError):
                rep["filed"] = len(todo)
        unread = [i for i in items if i.status == "error"]
        if unread:
            _note(f"could not be read ({len(unread)}):")
            for i in unread[:40]:
                _note(f"  {i.src}  -  {i.why}")
        for i in items:
            if i.status == "move" or i.status == "here":
                continue
            why = {"norig": "no rig matched - Your Rigs / Which Rig? in the Organize window",
                   "duplicate": "duplicates - Step 2 in the Organize window",
                   "error": "could not be read"}.get(i.status, i.why or i.status)
            rep["left"][why] = rep["left"].get(why, 0) + 1
        if settling:
            rep["left"][f"changed in the last {SETTLE_MIN} min (next run)"] = settling
        if s.value("nina_clean/with_auto", False, type=bool):    # then free space on the N.I.N.A. computers
            try:
                import nina_cleanup as NC
                if progress:
                    progress("Cleaning up your N.I.N.A. computers…")
                txt = NC.run(s)
                if txt:
                    rep["nina"] = txt
                    _note(f"N.I.N.A. clean-up: {txt}")
            except Exception as e:  # noqa: BLE001
                rep["nina"] = f"N.I.N.A. clean-up failed: {e}"
    except Exception as e:  # noqa: BLE001 - a scheduled run must report, never hang
        rep["error"] = str(e)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    _finish(rep, s, base)
    return rep


def _host() -> str:
    import socket
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _pid_alive(pid) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259        # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _lock_alive(lock) -> bool:
    """A run's marker counts only while that run is really going: SubZeroAstro closed or the computer switched off
    mid-run leaves the marker behind, and it must not block the next run. A run on another computer (the same
    library on a NAS) can't be checked from here, so it counts until it is stale."""
    try:
        parts = lock.read_text(encoding="utf-8").split("|")
        if len(parts) >= 3 and parts[2] and parts[2] != _host():
            return True
        return _pid_alive(int(parts[1])) if len(parts) >= 2 else False
    except (OSError, ValueError):
        return False


def _volume(path):
    """macOS: '/Volumes/AstroDataOrganizedSAZ/x' -> '/Volumes/AstroDataOrganizedSAZ' (else None)."""
    parts = Path(str(path)).parts
    if sys.platform == "darwin" and len(parts) >= 3 and parts[1] == "Volumes":
        return str(Path(*parts[:3]))
    return None


def _unreachable(path) -> str:
    vol = _volume(path)
    if vol and not Path(vol).exists():
        return (f"the NAS folder {Path(vol).name} isn't connected on this Mac - Finder › Go › Connect to Server…, "
                f"or add it to System Settings › General › Login Items so it connects at login")
    return "is the NAS on and connected?"


def mount_urls(text) -> dict:
    """`mount` output -> {'/Volumes/Share': 'smb://user@server/Share'} for the network shares mounted now."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"^//(\S+?) on (/Volumes/.+?) \((smbfs|afpfs|nfs)", line)
        if m:
            scheme = {"smbfs": "smb", "afpfs": "afp", "nfs": "nfs"}[m.group(3)]
            out[m.group(2)] = f"{scheme}://{m.group(1)}"
    return out


def ensure_mounted(path, settings=None) -> bool:
    """macOS: a library or incoming folder on a NAS lives under /Volumes/<share>, which exists only while the
    share is connected. Remember how each share was connected; if it isn't connected at run time, reconnect it
    (the password comes from the keychain, as when Finder connects). Windows network paths need nothing."""
    vol = _volume(path)
    if not vol:
        return True
    s = settings
    key = "auto_org/mount/" + Path(vol).name
    try:
        r = subprocess.run(["mount"], capture_output=True, text=True, timeout=15)
        urls = mount_urls(r.stdout)
    except (OSError, subprocess.SubprocessError):
        urls = {}
    if vol in urls:
        if s is not None:
            s.setValue(key, urls[vol])
        return True
    if Path(vol).exists():
        return True
    url = s.value(key, "") if s is not None else ""
    if not url:
        return False
    try:
        subprocess.run(["osascript", "-e", f'mount volume "{url}"'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass
    for _ in range(20):
        if Path(vol).exists():
            return True
        time.sleep(0.5)
    return False


def describe(rep) -> str:
    if rep.get("error"):
        return f"{rep['time']}: not run - {rep['error']}"
    t = f"{rep['time']}: {rep['filed']} file(s) filed"
    if rep.get("asiair"):
        t += f" [ASIAIRs: {rep['asiair']}]"
    if rep.get("other"):
        t += f", {rep['other']} other file(s) deleted"
    if rep.get("empty"):
        t += f", {rep['empty']} empty frame(s) deleted"
    if rep.get("jpgs"):
        t += f", {rep['jpgs']} .jpg preview(s) deleted"
    if rep["left"]:
        t += "; left in the incoming folder: " + ", ".join(f"{n} {w}" for w, n in rep["left"].items())
    if rep.get("nina"):
        t += f" [N.I.N.A. computers: {rep['nina']}]"
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


def start_background() -> tuple:
    """Run Now: start one automatic run as its own process (no window), exactly like a scheduled run."""
    try:
        kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "close_fds": True}
        if sys.platform == "win32":
            kw["creationflags"] = 0x00000008 | 0x08000000 | 0x00000200   # detached, no window, new group
        else:
            kw["start_new_session"] = True
        subprocess.Popen(_command(), **kw)
        return True, ""
    except OSError as e:
        return False, str(e)


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
