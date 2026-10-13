"""
astrobin.py - "Share to AstroBin": the finished picture plus an acquisition CSV that
AstroBin's upload page can import (Acquisition › Import CSV), so nobody has to type in
every night's subs, exposure, gain and temperature by hand.

AstroBin's public API is read-only, so the upload itself happens in the browser; this
module prepares everything for it.

The acquisition details come from the ACQnn / ACQCAL keywords SubZeroAstro writes into each
master when it stacks (they survive Prepare and the Finish exports). Masters stacked
elsewhere get one row per master from their standard header keywords instead.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

UPLOAD_URL = "https://www.astrobin.com/upload/"
FILTER_SEARCH_URL = "https://app.astrobin.com/equipment/explorer/filter"

# AstroBin's "Long exposure" acquisition CSV, in its documented column order
COLUMNS = ["date", "filter", "number", "duration", "iso", "binning", "gain", "sensorCooling", "fNumber",
           "darks", "flats", "flatDarks", "bias", "bortle", "meanSqm", "meanFwhm", "temperature"]


@dataclass
class Session:
    date: str
    filter: str
    number: int
    duration: float
    gain: float | None = None
    cooling: int | None = None
    fwhm: float | None = None           # arcsec
    binning: int = 1
    cal: dict = field(default_factory=dict)   # darks / flats / flatDarks / bias (first row of a filter)


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _header(path):
    import stacking as st
    src = st.ImageSource(path)
    return {k: v for k, v, _c in src.cards if k not in ("HISTORY", "COMMENT")}


def _count_in(text) -> int:
    """'25 flat subframes' -> 25; a master file name -> 0 (count unknown)."""
    m = re.match(r"\s*(\d+)\s", str(text or ""))
    return int(m.group(1)) if m else 0


def sessions_from_header(look: dict) -> list[Session]:
    import fits_core as core
    filt = core.normalize_filter(look.get("FILTER")) if look.get("FILTER") else "No filter"
    rows = []
    for k in sorted(k for k in look if re.fullmatch(r"ACQ\d\d", str(k))):
        parts = str(look[k]).split("|")
        if len(parts) < 7 or not _num(parts[1]):
            continue
        rows.append(Session(date=parts[0], filter=filt, number=int(float(parts[1])),
                            duration=_num(parts[2]) or 0.0, gain=_num(parts[3]),
                            cooling=int(_num(parts[4])) if _num(parts[4]) is not None else None,
                            fwhm=_num(parts[5]), binning=int(_num(parts[6]) or 1)))
    cal = {}
    if look.get("ACQCAL"):
        vals = (str(look["ACQCAL"]).split("|") + ["0"] * 4)[:4]
        cal = dict(zip(("darks", "flats", "flatDarks", "bias"), (int(_num(v) or 0) for v in vals)))
    if not rows:
        # a master stacked elsewhere, or before SubZeroAstro recorded nights: one row
        info = core.frame_info(look)
        night = core.night_of(info) or str(look.get("DATE-OBS") or "")[:10]
        n = int(_num(look.get("STACKCNT")) or _num(look.get("NCOMBINE")) or 0)
        exp = _num(look.get("EXPTIME")) or _num(look.get("EXPOSURE")) or 0.0
        if _num(look.get("STACKCNT")) and _num(look.get("EXPTOTAL")):
            exp = _num(look.get("EXPTOTAL")) / n
        if n:
            rows.append(Session(date=night, filter=filt, number=n, duration=exp, gain=_num(look.get("GAIN")),
                                cooling=round(_num(look.get("CCD-TEMP"))) if _num(look.get("CCD-TEMP")) is not None
                                else None, binning=int(_num(look.get("XBINNING")) or 1)))
        if not cal:
            cal = {"darks": _count_in(look.get("CALDARK")), "flats": _count_in(look.get("CALFLAT")),
                   "flatDarks": 0, "bias": _count_in(look.get("CALBIAS"))}
    if rows and cal:
        rows[0].cal = cal
    return rows


def acquisition(paths) -> list[Session]:
    """All sessions behind a picture made from these masters (one list across filters)."""
    out, seen = [], set()
    for p in paths:
        try:
            look = _header(p)
        except Exception:  # noqa: BLE001
            continue
        for s in sessions_from_header(look):
            key = (s.date, s.filter, s.number, s.duration)
            if key not in seen:                 # the same master opened twice
                seen.add(key)
                out.append(s)
    return sorted(out, key=lambda s: (s.filter, s.date))


def total_hours(sessions) -> float:
    return sum(s.number * s.duration for s in sessions) / 3600


def _fmt(v, nd=2):
    if v is None or v == "":
        return ""
    if isinstance(v, int):
        return str(v)
    return f"{float(v):.{nd}f}".rstrip("0").rstrip(".")


def csv_rows(sessions, filter_ids=None, bortle=None, sqm=None, include_cal=True) -> list[list[str]]:
    filter_ids = filter_ids or {}
    rows = []
    for s in sessions:
        fid = str(filter_ids.get(s.filter, "") or "").strip()
        cal = s.cal if include_cal else {}
        rows.append([s.date, fid if fid.isdigit() else "", str(s.number), _fmt(s.duration, 4), "",
                     str(s.binning or 1), _fmt(s.gain), _fmt(s.cooling, 0), "",
                     _fmt(cal.get("darks") or ""), _fmt(cal.get("flats") or ""),
                     _fmt(cal.get("flatDarks") or ""), _fmt(cal.get("bias") or ""),
                     _fmt(bortle) if bortle else "", _fmt(sqm) if sqm else "", _fmt(s.fwhm), ""])
    return rows


def write_csv(path, sessions, filter_ids=None, bortle=None, sqm=None, include_cal=True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(csv_rows(sessions, filter_ids, bortle, sqm, include_cal))
    return path


def summary(sessions) -> str:
    if not sessions:
        return "No acquisition details were found in the masters."
    lines = []
    for filt in sorted({s.filter for s in sessions}):
        ss = [s for s in sessions if s.filter == filt]
        subs = sum(s.number for s in ss)
        lines.append(f"{filt}: {subs} subs over {len({s.date for s in ss})} night(s), "
                     f"{total_hours(ss):.1f} h")
    return "\n".join(lines) + f"\nTotal: {total_hours(sessions):.1f} h"
