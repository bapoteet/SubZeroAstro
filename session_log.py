"""
session_log.py - a log file on disk for every session, so a crash or a frozen computer leaves a trail.

Every page's log lines are copied here as they happen, with the time; every 30 seconds a line notes how
much memory the app uses and how much the computer has left; an unexpected error or a hard crash (the
app killed by the system) writes its traceback here too. The newest few sessions are kept, and
"Zip My Logs" includes the last one before this, so a log from a session that ended badly is not lost.

  Windows: %LOCALAPPDATA%\\SubZeroAstro\\Logs      macOS: ~/Library/Logs/SubZeroAstro
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

KEEP = 8                  # session logs kept
_f = None
_path: Path | None = None
_lock = threading.Lock()
_timer = None


def logs_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "SubZeroAstro" / "Logs"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "SubZeroAstro"
    else:
        base = Path.home() / ".subzeroastro" / "logs"
    return base


def current() -> Path | None:
    return _path


def previous(n: int = 2) -> list:
    """The newest n session logs before this one."""
    try:
        logs = sorted(logs_dir().glob("session_*.log"), key=lambda p: p.name, reverse=True)
    except OSError:
        return []
    return [p for p in logs if p != _path][:n]


# ------------------------------------------------------------------ memory
def _gb(b) -> float:
    return float(b) / 1024 ** 3


def memory() -> dict:
    """{'app': GB this app uses, 'free': GB the computer has left, 'total': GB} - what can be found."""
    out = {}
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                out["total"], out["free"] = _gb(ms.ullTotalPhys), _gb(ms.ullAvailPhys)

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            get = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
            get.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            get.restype = wintypes.BOOL
            if get(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                out["app"] = _gb(pmc.WorkingSetSize)
        elif sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                       timeout=5).stdout.strip() or 0)
            if total:
                out["total"] = _gb(total)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = 16384
            stats = {}
            for line in vm.splitlines():
                if "page size of" in line:
                    page = int("".join(ch for ch in line.split("page size of")[1] if ch.isdigit()) or page)
                elif ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip().rstrip(".")
                    if v.isdigit():
                        stats[k.strip()] = int(v)
            freeish = sum(stats.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative",
                                                     "Pages purgeable"))
            out["free"] = _gb(freeish * page)
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if rss.isdigit():
                out["app"] = _gb(int(rss) * 1024)
        else:
            with open("/proc/meminfo") as fh:
                info = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in fh if ln.split()[1:2]}
            out["total"], out["free"] = _gb(info.get("MemTotal", 0)), _gb(info.get("MemAvailable", 0))
            with open(f"/proc/{os.getpid()}/status") as fh:
                for ln in fh:
                    if ln.startswith("VmRSS:"):
                        out["app"] = _gb(int(ln.split()[1]) * 1024)
    except Exception:  # noqa: BLE001 - memory notes are best effort
        pass
    return out


def memory_text() -> str:
    m = memory()
    bits = []
    if "app" in m:
        bits.append(f"app uses {m['app']:.1f} GB")
    if "free" in m:
        bits.append(f"{m['free']:.1f} GB free")
    if "total" in m:
        bits.append(f"of {m['total']:.0f} GB")
    return ", ".join(bits) or "unknown"


# ------------------------------------------------------------------ writing
def start(version: str = "") -> Path | None:
    """Open this session's log (call once, at start-up)."""
    global _f, _path
    if _f is not None:
        return _path
    try:
        d = logs_dir()
        d.mkdir(parents=True, exist_ok=True)
        _path = d / f"session_{time.strftime('%Y-%m-%d_%H%M%S')}.log"
        _f = open(_path, "a", encoding="utf-8", buffering=1)
        for old in sorted(d.glob("session_*.log"), key=lambda p: p.name, reverse=True)[KEEP:]:
            try:
                old.unlink()
            except OSError:
                pass
        write("session", f"SubZeroAstro {version} started - {platform.system()} {platform.release()} "
                         f"({platform.machine()}) - memory: {memory_text()}")
        try:
            import faulthandler
            faulthandler.enable(_f)          # a hard crash still writes where it happened
        except Exception:  # noqa: BLE001
            pass
    except OSError:
        _f, _path = None, None
    return _path


def write(page: str, text: str):
    if _f is None:
        return
    try:
        stamp = time.strftime("%H:%M:%S")
        with _lock:
            for line in str(text).splitlines() or [""]:
                _f.write(f"{stamp} [{page}] {line}\n")
    except Exception:  # noqa: BLE001 - logging never stops the app
        pass


def heartbeat(seconds: int = 30):
    """Note the memory every `seconds` (a QTimer: call after the QApplication exists)."""
    global _timer
    if _f is None or _timer is not None:
        return
    try:
        from PySide6.QtCore import QTimer
        _timer = QTimer()
        _timer.setInterval(seconds * 1000)

        def tick():
            threading.Thread(target=lambda: write("memory", memory_text()), daemon=True).start()
        _timer.timeout.connect(tick)
        _timer.start()
    except Exception:  # noqa: BLE001
        _timer = None


def stop():
    global _f
    if _f is not None:
        write("session", "SubZeroAstro closed normally")
        try:
            import faulthandler
            faulthandler.disable()
        except Exception:  # noqa: BLE001
            pass
        try:
            _f.close()
        except OSError:
            pass
        _f = None


def ended_badly(p: Path) -> bool:
    """A session log without the 'closed normally' line: the app crashed, froze or was killed."""
    try:
        with open(p, "rb") as fh:
            fh.seek(max(0, p.stat().st_size - 400))
            return b"closed normally" not in fh.read()
    except OSError:
        return False
