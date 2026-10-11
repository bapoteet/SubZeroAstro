"""
checkpoints.py - crash recovery and step history (SubZeroAstro 0.6).

Everything lives under one folder (default %LOCALAPPDATA%\\SubZeroAstro\\checkpoints,
changeable in the settings) so it is easy to see how much space it takes and to
clear it:

    prepare_<target>_<key>/     one per set of masters in the Prepare window
        manifest.json           finished steps, their settings and titles
        01_crop/ m0.npy m0.pkl preview.npy ...
    stack_<key>.json            which masters an interrupted stacking run already saved
    culling_session.json        the last culling session (folders, flags, sort)

Writes are atomic (write to *.tmp, then rename) so a crash or power cut never
leaves a half-written checkpoint that looks valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

_root_override: str | None = None


def set_root(path: str | None):
    """Use another folder for checkpoints (None/'' = the default)."""
    global _root_override
    _root_override = path or None


def default_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "SubZeroAstro" / "checkpoints" if base else Path.home() / ".cache" / "subzeroastro" / "checkpoints"


def root() -> Path:
    d = Path(_root_override) if _root_override else default_root()
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_key(paths, extra="") -> str:
    """Identity of a set of input files: path + size + modification time (+ extra text)."""
    h = hashlib.sha1()
    for p in sorted(os.path.normcase(os.path.abspath(str(x))) for x in paths):
        try:
            stt = os.stat(p)
            sig = f"{p}|{stt.st_size}|{stt.st_mtime_ns}\n"
        except OSError:
            sig = f"{p}|missing\n"
        h.update(sig.encode("utf-8"))
    h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:40] or "x"


def write_atomic(path: Path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def write_json(path: Path, obj):
    write_atomic(path, json.dumps(obj, indent=1, default=str).encode("utf-8"))


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    n = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n


def total_size() -> int:
    try:
        return _size(root())
    except OSError:
        return 0


def human(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("bytes", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _last_used(p: Path) -> float:
    m = p / "manifest.json" if p.is_dir() else p
    try:
        return m.stat().st_mtime if m.exists() else p.stat().st_mtime
    except OSError:
        return 0.0


def cleanup(max_age_days: float, keep=()) -> int:
    """Delete checkpoints not used for max_age_days (0 = keep forever). Returns bytes freed."""
    if not max_age_days or max_age_days <= 0:
        return 0
    cutoff = time.time() - max_age_days * 86400
    freed = 0
    keep = {os.path.normcase(str(k)) for k in keep}
    for p in list(root().iterdir()):
        if os.path.normcase(str(p)) in keep or p.name == "culling_session.json":
            continue
        if _last_used(p) < cutoff:
            freed += _size(p)
            remove(p)
    return freed


def clear_all(keep_session=True) -> int:
    freed = 0
    for p in list(root().iterdir()):
        if keep_session and p.name == "culling_session.json":
            continue
        freed += _size(p)
        remove(p)
    return freed


def remove(p: Path):
    p = Path(p)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except OSError:
            pass
