"""
nina_meta.py - what N.I.N.A. knew about each frame when it took it.

N.I.N.A.'s free Session Metadata plugin (by Tom Palmer) writes an ImageMetaData.csv (or .json) next to the frames:
one row per frame with the guiding RMS during that exposure, the pier side, HFR, star count, airmass and more.
Guiding is the one thing a frame itself can't tell us afterwards, so when the file is there each frame gets:

    metrics["guide_rms"]   total guiding RMS in arcseconds (also guide_ra / guide_dec)
    info["pier"]           East / West

Unguided rigs (a 10Micron or an AP mount with a model) write 0 for guiding - that is read as "no guiding data",
so nothing changes for them.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

MAX_BYTES = 30 * 1024 * 1024
_SIGN = ("FilePath", "GuidingRMSArcSec")


def _f(v):
    try:
        x = float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _key(path) -> str:
    """Frames are matched by file name (the CSV holds the path on the rig's own computer)."""
    name = re.split(r"[\\/]", str(path))[-1].lower()
    return name


def _stem(name: str) -> str:
    n = name.lower()
    for ext in (".fits", ".fit", ".fts", ".xisf", ".fz"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


def record(row: dict) -> dict:
    """One plugin row -> {guide_rms, guide_ra, guide_dec, pier, hfr, stars, airmass} (missing ones left out)."""
    out = {}
    for k, col in (("guide_rms", "GuidingRMSArcSec"), ("guide_ra", "GuidingRMSRAArcSec"),
                   ("guide_dec", "GuidingRMSDECArcSec"), ("hfr", "HFR"), ("stars", "DetectedStars"),
                   ("airmass", "Airmass")):
        v = _f(row.get(col))
        if v is not None and v > 0:              # 0 = not guided / not measured
            out[k] = v
    pier = str(row.get("PierSide") or "").strip()
    if pier.lower() in ("east", "west"):
        out["pier"] = pier.capitalize()
    return out


def parse_text(text: str) -> dict:
    """The plugin's CSV or JSON -> {file name (lower case): record}."""
    t = text.lstrip("﻿").strip()
    rows = []
    if t.startswith("["):
        try:
            rows = [r for r in json.loads(t) if isinstance(r, dict)]
        except ValueError:
            rows = []
    elif t:
        rows = list(csv.DictReader(io.StringIO(t)))
    out = {}
    for r in rows:
        fp = r.get("FilePath")
        if not fp:
            continue
        rec = record(r)
        if rec:
            out[_key(fp)] = rec
    return out


def looks_like(path: Path) -> bool:
    """Is this one of the plugin's image metadata files (whatever the user renamed it to)?"""
    if path.suffix.lower() not in (".csv", ".json"):
        return False
    try:
        if path.stat().st_size > MAX_BYTES:
            return False
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return False
    return all(s in head for s in _SIGN)


class MetaIndex:
    """Finds and reads the metadata files for frames' folders (the folder itself and up to two above it), and
    re-reads a file only when it has changed - so live watching picks up new rows as N.I.N.A. adds them."""

    def __init__(self, levels=2):
        self.levels = levels
        self.files: dict[str, tuple] = {}        # path -> (mtime, size, {name: rec})
        self.dirs: dict[str, list] = {}          # folder -> [metadata files in it]

    def _scan_dir(self, d: Path) -> list:
        k = str(d)
        if k not in self.dirs:
            found = []
            try:
                for p in d.iterdir():
                    if p.is_file() and looks_like(p):
                        found.append(p)
            except OSError:
                pass
            self.dirs[k] = found
        return self.dirs[k]

    def _read(self, p: Path) -> dict:
        try:
            st = p.stat()
        except OSError:
            return {}
        old = self.files.get(str(p))
        if old and old[0] == st.st_mtime_ns and old[1] == st.st_size:
            return old[2]
        try:
            data = parse_text(p.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError:
            data = old[2] if old else {}
        self.files[str(p)] = (st.st_mtime_ns, st.st_size, data)
        return data

    def forget_dirs(self):
        """Look for new metadata files next time (a new night's folder may have one by now)."""
        self.dirs.clear()

    def lookup(self, frame_path) -> dict:
        p = Path(frame_path)
        name = p.name.lower()
        d = p.parent
        for _ in range(self.levels + 1):
            for mf in self._scan_dir(d):
                data = self._read(mf)
                rec = data.get(name)
                if rec is None:                     # saved as .fits, listed as .fit (or the other way)
                    st = _stem(name)
                    rec = next((v for k, v in data.items() if _stem(k) == st), None)
                if rec is not None:
                    return rec
            if d.parent == d:
                break
            d = d.parent
        return {}


def attach(f, rec: dict):
    """Put a frame's N.I.N.A. record on it (metrics guide_*, info pier). Returns True if anything was added."""
    if not rec:
        return False
    if f.metrics is None:
        f.metrics = {}
    for k in ("guide_rms", "guide_ra", "guide_dec"):
        if k in rec:
            f.metrics[k] = rec[k]
    if f.info is not None and rec.get("pier"):
        f.info["pier"] = rec["pier"]
    for k in ("hfr", "airmass"):
        if k in rec and f.info is not None:
            f.info["nina_" + k] = rec[k]
    return "guide_rms" in rec or "pier" in rec
