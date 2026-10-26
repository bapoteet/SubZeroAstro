"""
target_picker.py - Tonight's Target Picker: which deep-sky objects are up, dark and clear of your trees
tonight, from where your rig is. Plain astronomy (no internet, no AI): the Sun and Moon from low-precision
almanac formulas (good to a few arc-minutes / ~0.3 degree, plenty for planning), each object's altitude
every 5 minutes through the night, checked against your horizon.

The object list is N.I.N.A.'s deep-sky database - read straight from N.I.N.A. when it is installed
(%LOCALAPPDATA%\\NINA\\NINA.sqlite), otherwise the copy of its imaging-worthy objects built into
SubZeroAstro (dso_data.py; N.I.N.A. database, Mozilla Public License 2.0, github.com/isbeorn/nina).
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import targets as _targets

STEP_MIN = 5
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

GALAXY, NEBULA, SNR, PN, DARK, OPEN, GLOBULAR, GCLUSTER = ("Galaxy", "Bright nebula", "Supernova remnant",
                                                          "Planetary nebula", "Dark nebula", "Open cluster",
                                                          "Globular cluster", "Galaxy cluster")
TYPES = (GALAXY, NEBULA, SNR, PN, DARK, OPEN, GLOBULAR, GCLUSTER)
_NINA_TYPE = {"GALXY": GALAXY, "GX+DN": GALAXY, "BRTNB": NEBULA, "CL+NB": NEBULA, "SNREM": SNR, "PLNNB": PN,
              "DRKNB": DARK, "OPNCL": OPEN, "GLOCL": GLOBULAR, "GALCL": GCLUSTER}
NARROWBAND_OK = {NEBULA, SNR, PN}          # shine in H-alpha / OIII: a bright Moon matters much less
_NAME_ORDER = ["M", "NGC", "IC", "Sh2", "Barnard", "Abell", "vdB", "LDN", "LBN", "Collinder", "Melotte",
               "Caldwell", "Arp", "Hickson", "RCW", "PK", "UGC"]


# ---------------------------------------------------------------- catalogue
@dataclass
class Target:
    name: str
    common: str
    kind: str
    ra: float                  # degrees, J2000
    dec: float
    mag: float | None
    size: float                # arc-minutes, longest
    size_min: float | None
    constellation: str = ""
    aka: list = field(default_factory=list)

    @property
    def size_text(self) -> str:
        if not self.size:
            return ""
        a = f"{self.size:.0f}'" if self.size >= 10 else f"{self.size:.1f}'"
        if self.size_min and abs(self.size_min - self.size) > 0.05:
            b = f"{self.size_min:.0f}'" if self.size_min >= 10 else f"{self.size_min:.1f}'"
            return f"{a} × {b}"
        return a


def _keep(kind_code, mag, size, names) -> bool:
    """The imaging-worthy part of N.I.N.A.'s 16 000 objects: big enough and bright enough to photograph."""
    messier = any(n.startswith(("M ", "Caldwell ")) for n in names)
    if kind_code not in _NINA_TYPE:
        return False
    if messier:
        return True
    if kind_code in ("GALXY",):
        return size >= 2 and (mag is None or mag <= 13)
    if kind_code == "SNREM":                  # 'SNR G132.7+01.3' alone = a faint radio remnant
        return size >= 2 and any(not n.startswith("SNR ") for n in names)
    if kind_code in ("BRTNB", "CL+NB", "GX+DN"):
        return size >= 2
    if kind_code == "PLNNB":
        return (size >= 0.4 and (mag is None or mag <= 13.5)) or any(n.startswith("Abell ") for n in names)
    if kind_code == "DRKNB":
        return size >= 15
    if kind_code == "OPNCL":
        return size >= 8 and (mag is None or mag <= 8)
    if kind_code == "GLOCL":
        return mag is None or mag <= 10
    if kind_code == "GALCL":
        return size >= 10
    return False


def _pick_names(names, kind):
    cat = [n for n in names if not n.startswith("NAME ")]
    common = [n[5:] for n in names if n.startswith("NAME ")]

    def rank(n):
        c = n.split(" ")[0]
        return (_NAME_ORDER.index(c) if c in _NAME_ORDER else 99, len(n))
    cat.sort(key=rank)
    word = {GALAXY: "Galaxy", NEBULA: "Nebula", SNR: "Nebula", PN: "Nebula", OPEN: "Cluster", GLOBULAR: "Cluster",
            DARK: "Nebula", GCLUSTER: "Cluster"}.get(kind, "")
    # drop "Ori Nebula"-style abbreviations; prefer a name that says what it is, then the shortest
    good = [c for c in common if not re.match(r"^[A-Z][a-z]{2} ", c) or len(c.split()[0]) > 3] or common
    good.sort(key=lambda c: (0 if word and word in c else 1, len(c)))
    return (cat[0] if cat else (names[0] if names else "?")), (good[0] if good else ""), cat[1:]


def from_rows(rows) -> list:
    """rows: (id, ra, dec, magnitude, sizemax ", sizemin ", dsotype, 'cat nr|cat nr|NAME x', constellation)."""
    out = []
    for _id, ra, dec, mag, smax, smin, code, names, const in rows:
        if ra is None or dec is None:
            continue
        names = [n for n in (names or "").split("|") if n] or [_id]
        size = float(smax or smin or 0) / 60.0
        if not _keep(code, mag, size, names):
            continue
        kind = _NINA_TYPE[code]
        name, common, aka = _pick_names(names, kind)
        for n in [name] + aka:                       # SubZeroAstro's own list knows the names imagers use
            known = _targets.common_name(n)
            if known:
                common = known
                break
        if mag is not None and float(mag) >= 50:      # 99.9 = unknown
            mag = None
        out.append(Target(name, common, kind, float(ra), float(dec), None if mag is None else float(mag), size,
                          None if smin in (None, "") else float(smin) / 60.0, (const or "").title(), aka))
    return out


NINA_QUERY = ("SELECT d.id, d.ra, d.dec, d.magnitude, d.sizemax, d.sizemin, d.dsotype, "
              "(SELECT group_concat(n.catalogue || ' ' || n.designation, '|') FROM cataloguenr n "
              "WHERE n.dsodetailid = d.id), d.constellation FROM dsodetail d")


def nina_folder() -> Path | None:
    la = os.environ.get("LOCALAPPDATA")
    return Path(la) / "NINA" if la else None


def nina_db_path() -> Path | None:
    f = nina_folder()
    p = f / "NINA.sqlite" if f else None
    return p if p and p.is_file() else None


def load_catalog(db: Path | None = None):
    """(targets, where-from text). N.I.N.A.'s own database when there is one, else the built-in copy."""
    db = db or nina_db_path()
    if db:
        try:
            con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
            try:
                rows = con.execute(NINA_QUERY).fetchall()
            finally:
                con.close()
            t = from_rows(rows)
            if t:
                return t, f"your N.I.N.A. database ({len(t)} objects worth imaging)"
        except Exception:  # noqa: BLE001 - locked or another layout: fall back to the built-in copy
            pass
    import dso_data
    t = from_rows(dso_data.rows())
    return t, f"the built-in copy of N.I.N.A.'s database ({len(t)} objects worth imaging)"


# ---------------------------------------------------------------- N.I.N.A. profiles and horizon files
def _tag(xml, tag):
    m = re.search(rf"<(?:\w+:)?{tag}>([^<]*)</(?:\w+:)?{tag}>", xml)
    return m.group(1).strip() if m else ""


def nina_profiles(folder: Path | None = None) -> list:
    """Site and rig from N.I.N.A.'s profiles, newest used first: [{name, lat, lon, horizon, focal, pixel}]."""
    folder = folder or (nina_folder() / "Profiles" if nina_folder() else None)
    out = []
    if not folder or not Path(folder).is_dir():
        return out
    for p in Path(folder).glob("*.profile"):
        try:
            xml = p.read_text(encoding="utf-8", errors="replace")

            def num(tag):
                try:
                    return float(_tag(xml, tag))
                except ValueError:
                    return None
            d = {"name": _tag(xml, "Name") or p.stem, "lat": num("Latitude"), "lon": num("Longitude"),
                 "horizon": _tag(xml, "HorizonFilePath"), "focal": num("FocalLength"), "pixel": num("PixelSize"),
                 "used": _tag(xml, "LastUsed"), "file": str(p),
                 "targets": _tag(xml, "SequencerTargetsFolder")}
            if d["lat"] is not None and d["lon"] is not None and not (d["lat"] == 0 and d["lon"] == 0):
                out.append(d)
        except OSError:
            continue
    out.sort(key=lambda d: d["used"], reverse=True)
    return out


class Horizon:
    """The lowest altitude you can image at, for every degree of azimuth (0 = N, 90 = E)."""

    def __init__(self, alt=None):
        self.alt = np.zeros(360, np.float32) if alt is None else np.asarray(alt, np.float32)

    @classmethod
    def from_points(cls, pts):
        """[(azimuth, altitude)] as in a N.I.N.A. horizon file, joined by straight lines."""
        pts = sorted((float(a) % 360, float(h)) for a, h in pts)
        if not pts:
            return cls()
        az = np.array([p[0] for p in pts])
        h = np.array([p[1] for p in pts])
        az = np.concatenate([az - 360, az, az + 360])
        h = np.concatenate([h, h, h])
        return cls(np.interp(np.arange(360), az, h))

    @classmethod
    def from_file(cls, path):
        pts = []
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = [c for c in re.split(r"[\t ,;]+", line) if c]
            if len(cols) == 2:
                try:
                    pts.append((float(cols[0]), float(cols[1])))
                except ValueError:
                    pass
        if not pts:
            raise ValueError("no 'azimuth altitude' lines found")
        return cls.from_points(pts)

    @classmethod
    def from_sectors(cls, sectors):
        """16 compass sectors (N, NNE … NNW), each the altitude it is blocked to (0 = open, 90 = blocked)."""
        alt = np.zeros(360, np.float32)
        for d in range(360):
            alt[d] = sectors[int(((d + 11.25) % 360) // 22.5)]
        return cls(alt)

    def sectors(self):
        return [float(np.max(self.alt[[int((i * 22.5 + k) % 360) for k in range(-11, 12)]])) for i in range(16)]


def _point(word):
    w = word.strip().upper().replace("NORTH", "N").replace("SOUTH", "S").replace("EAST", "E").replace("WEST", "W")
    w = re.sub(r"[^NSEW]", "", w)
    return COMPASS.index(w) if w in COMPASS else None


def parse_view(text, blocked=90.0, base=None):
    """'open sky from NNW to SSW' / 'trees to the east up to 30 degrees' -> 16 sector altitudes.
    'open/clear/see … from A to B' opens A..B clockwise and blocks the rest; 'blocked/trees/house … from A to
    B [up to N degrees]' or '… to the E' blocks those sectors; anything else is left as it was."""
    sec = list(base) if base is not None else [0.0] * 16
    t = text.lower().replace("-", " to ")
    opened_any = False
    for clause in re.split(r"[.;\n]|,(?=\s*(?:and\s+)?(?:open|clear|blocked|trees|house|see|but))|\bbut\b", t):
        c = clause.strip()
        if not c:
            continue
        m = re.search(r"\bfrom\s+([nsew]+\w*)\s+(?:to|through|thru)\s+([nsew]+\w*)", c) or \
            re.search(r"\b([nsew]{1,3})\s+(?:to|through|thru)\s+([nsew]{1,3})\b", c)
        rng = None
        if m:
            a, b = _point(m.group(1)), _point(m.group(2))
            if a is not None and b is not None:          # the shorter way round (E to W: through S)
                cw = (b - a) % 16
                rng = ([(a + i) % 16 for i in range(cw + 1)] if cw <= 8 else
                       [(b + i) % 16 for i in range(16 - cw + 1)])
        else:
            m = re.search(r"\b(?:to|in|on)\s+the\s+([nsew]+\w*)|\b([nsew]{1,3})\b", c)
            if m:
                p = _point(m.group(1) or m.group(2))
                if p is not None:
                    rng = [(p + i) % 16 for i in (-1, 0, 1)] if len(COMPASS[p]) == 1 else [p]
        if not rng:
            continue
        h = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|deg|degrees)", c)
        if re.search(r"\b(open|clear|see|visible|view)\b", c) and not re.search(r"\b(block|tree|house|roof|hill|"
                                                                               r"build|wall|obstruct)", c):
            if not opened_any:
                sec = [blocked] * 16
                opened_any = True
            for i in rng:
                sec[i] = float(h.group(1)) if h else 0.0
        else:
            for i in rng:
                sec[i] = float(h.group(1)) if h else blocked
    return sec


# ---------------------------------------------------------------- Sun, Moon, altitude
def jd(t: _dt.datetime) -> float:
    t = t.astimezone(_dt.timezone.utc)
    return 2440587.5 + t.timestamp() / 86400.0


def sun_radec(j):
    n = np.asarray(j) - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))) % 360
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    return ra, dec


def moon_radec(j):
    """Geocentric Moon (Astronomical Almanac low-precision series, ~0.3°) and its horizontal parallax."""
    T = (np.asarray(j) - 2451545.0) / 36525.0
    s = lambda a, b: np.sin(np.radians(a + b * T))  # noqa: E731
    c = lambda a, b: np.cos(np.radians(a + b * T))  # noqa: E731
    lam = (218.32 + 481267.881 * T + 6.29 * s(135.0, 477198.87) - 1.27 * s(259.3, -413335.36)
           + 0.66 * s(235.7, 890534.22) + 0.21 * s(269.9, 954397.74) - 0.19 * s(357.5, 35999.05)
           - 0.11 * s(186.5, 966404.03))
    beta = (5.13 * s(93.3, 483202.02) + 0.28 * s(228.2, 960400.89) - 0.28 * s(318.3, 6003.15)
            - 0.17 * s(217.6, -407332.21))
    par = (0.9508 + 0.0518 * c(135.0, 477198.87) + 0.0095 * c(259.3, -413335.36) + 0.0078 * c(235.7, 890534.22)
           + 0.0028 * c(269.9, 954397.74))
    lam, beta = np.radians(lam % 360), np.radians(beta)
    eps = np.radians(23.439 - 0.0130 * T)
    x = np.cos(beta) * np.cos(lam)
    y = np.cos(eps) * np.cos(beta) * np.sin(lam) - np.sin(eps) * np.sin(beta)
    z = np.sin(eps) * np.cos(beta) * np.sin(lam) + np.cos(eps) * np.sin(beta)
    return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arcsin(z)), par


def lst(j, lon):
    return (280.46061837 + 360.98564736629 * (np.asarray(j) - 2451545.0) + lon) % 360


def altaz(ra, dec, lat, lst_deg):
    """Altitude and azimuth (degrees; azimuth from north through east). Broadcasts ra/dec against lst."""
    H = np.radians(np.asarray(lst_deg) - np.asarray(ra))
    d, p = np.radians(dec), math.radians(lat)
    sa = np.sin(p) * np.sin(d) + np.cos(p) * np.cos(d) * np.cos(H)
    alt = np.degrees(np.arcsin(np.clip(sa, -1, 1)))
    az = np.degrees(np.arctan2(-np.cos(d) * np.sin(H), np.sin(d) * math.cos(p) - np.cos(d) * math.sin(p) * np.cos(H)))
    return alt, az % 360


def separation(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ---------------------------------------------------------------- the night
@dataclass
class Night:
    times: list                    # UTC datetimes, every STEP_MIN minutes from local noon to noon
    dark: np.ndarray               # bool per sample: sun below -18 (or -12 when it never gets that dark)
    sun_alt: np.ndarray
    moon_alt: np.ndarray
    moon_ra: np.ndarray
    moon_dec: np.ndarray
    moon_lit: float                # 0..1 at midnight
    twilight: str                  # "astronomical" / "nautical" / "none"
    lst: np.ndarray

    @property
    def dark_start(self):
        i = np.flatnonzero(self.dark)
        return self.times[i[0]] if i.size else None

    @property
    def dark_end(self):
        i = np.flatnonzero(self.dark)
        return self.times[i[-1]] if i.size else None

    @property
    def dark_hours(self):
        return float(self.dark.sum()) * STEP_MIN / 60.0


def night(lat, lon, date: _dt.date) -> Night:
    """The night that starts on the evening of `date` at the site."""
    noon = _dt.datetime(date.year, date.month, date.day, 12, tzinfo=_dt.timezone.utc) - _dt.timedelta(hours=lon / 15)
    n = 24 * 60 // STEP_MIN + 1
    times = [noon + _dt.timedelta(minutes=STEP_MIN * i) for i in range(n)]
    j = np.array([jd(t) for t in times])
    L = lst(j, lon)
    sra, sdec = sun_radec(j)
    salt, _ = altaz(sra, sdec, lat, L)
    mra, mdec, par = moon_radec(j)
    malt, _ = altaz(mra, mdec, lat, L)
    malt = malt - par * np.cos(np.radians(malt))          # seen from the ground, not the Earth's centre
    mid = n // 2
    elong = separation(sra[mid], sdec[mid], mra[mid], mdec[mid])
    lit = float((1 - math.cos(math.radians(elong))) / 2)
    dark, tw = salt < -18, "astronomical"
    if not dark.any():
        dark, tw = salt < -12, "nautical"
    if not dark.any():
        tw = "none"
    return Night(times, dark, salt, malt, mra, mdec, lit, tw, L)


@dataclass
class Plan:
    target: Target
    hours: float                   # dark and above your horizon
    peak_alt: float                # highest while dark and visible (or while dark, if never visible)
    peak_time: object
    start: object                  # first and last moment it is imageable tonight
    end: object
    moon_sep: float                # degrees, at the middle of its window
    moon_up: float                 # fraction of its window with the Moon up
    max_alt: float                 # highest it ever gets from here
    fill: float | None = None      # longest size / long side of the frame
    score: float = 0.0
    alt: np.ndarray | None = None  # altitude every STEP_MIN through the night (for the chart)
    az: np.ndarray | None = None


def plan_night(targets, lat, lon, date, horizon: Horizon | None = None, min_alt=30.0, fov=None, nt=None,
               keep_curves=False):
    """Every target's night: hours imageable, best window, peak altitude, Moon. Returns (Night, [Plan])."""
    nt = nt or night(lat, lon, date)
    if not targets:
        return nt, []
    ra = np.array([t.ra for t in targets])[:, None]
    dec = np.array([t.dec for t in targets])[:, None]
    alt, az = altaz(ra, dec, lat, nt.lst[None, :])
    hz = (horizon or Horizon()).alt[(az.astype(int)) % 360]
    ok = (alt >= np.maximum(hz, min_alt)) & nt.dark[None, :]
    hours = ok.sum(1) * STEP_MIN / 60.0
    dark_alt = np.where(nt.dark[None, :], alt, -90)
    vis_alt = np.where(ok, alt, -90)
    plans = []
    for k, t in enumerate(targets):
        idx = np.flatnonzero(ok[k])
        src = vis_alt[k] if idx.size else dark_alt[k]
        p = int(np.argmax(src))
        if idx.size:
            m = idx[idx.size // 2]
            sep = float(separation(t.ra, t.dec, nt.moon_ra[m], nt.moon_dec[m]))
            mup = float((nt.moon_alt[idx] > 0).mean())
            start, end = nt.times[idx[0]], nt.times[idx[-1]] + _dt.timedelta(minutes=STEP_MIN)
        else:
            sep, mup, start, end = 180.0, 0.0, None, None
        fill = None
        if fov and t.size:
            fill = t.size / max(fov)
        pl = Plan(t, float(hours[k]), float(src[p]), nt.times[p], start, end, sep, mup,
                  90 - abs(lat - t.dec), fill)
        pl.score = score(pl, nt.moon_lit)
        if keep_curves:
            pl.alt, pl.az = alt[k], az[k]
        plans.append(pl)
    return nt, plans


def score(p: Plan, moon_lit: float) -> float:
    """'Best tonight': hours you get, how high it climbs, the Moon (much less for narrowband targets) and how it
    fills your frame. Higher is better; 0 = not up tonight."""
    if p.hours <= 0:
        return 0.0
    s = p.hours * (0.4 + 0.6 * math.sin(math.radians(max(p.peak_alt, 0)))) * fame(p.target)
    moon = moon_lit * p.moon_up
    s *= 1 - (0.25 if p.target.kind in NARROWBAND_OK else 0.7) * moon
    if p.moon_sep < 30 and p.moon_up > 0:
        s *= 0.6 + 0.4 * (1 - moon_lit)
    if p.fill is not None:
        if p.fill < 0.04:
            s *= 0.35
        elif p.fill < 0.1:
            s *= 0.7
        elif p.fill > 1.05:
            s *= 0.6
    return round(s, 3)


def fame(t: Target) -> float:
    """Well-known targets first: a common name or a Messier / Caldwell number beats an obscure catalogue entry."""
    names = [t.name] + list(t.aka)
    if t.common or any(n.startswith(("M ", "Caldwell ")) for n in names):
        return 1.0
    if any(n.split(" ")[0] in ("NGC", "IC", "Sh2", "Abell", "Barnard", "vdB", "Arp") for n in names):
        return 0.8
    return 0.5


def fov_arcmin(focal_mm, pixel_um, width_px, height_px):
    """Field of view (long, short) in arc-minutes, or None."""
    try:
        if not (focal_mm and pixel_um and width_px and height_px):
            return None
        scale = 206.265 * float(pixel_um) / float(focal_mm)          # arcsec per pixel
        a, b = width_px * scale / 60, height_px * scale / 60
        return (max(a, b), min(a, b))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def fit_text(fill) -> str:
    if fill is None:
        return ""
    if fill > 1.05:
        return f"mosaic ({fill:.1f}× the frame)"
    if fill < 0.1:
        return f"small ({fill * 100:.0f}% of the frame)"
    return f"fits ({fill * 100:.0f}% of the frame)"


def nina_coords(t: Target) -> str:
    """RA / Dec as N.I.N.A. accepts them (J2000): '00h 42m 44s  +41° 16' 08"'."""
    h = t.ra / 15
    hh, mm = int(h), int((h % 1) * 60)
    ss = round(((h * 60) % 1) * 60)
    if ss == 60:
        ss, mm = 0, mm + 1
    if mm == 60:
        mm, hh = 0, (hh + 1) % 24
    sign = "-" if t.dec < 0 else "+"
    d = abs(t.dec)
    dd, dm = int(d), int((d % 1) * 60)
    ds = round(((d * 60) % 1) * 60)
    if ds == 60:
        ds, dm = 0, dm + 1
    if dm == 60:
        dm, dd = 0, dd + 1
    return f"{hh:02d}h {mm:02d}m {ss:02d}s  {sign}{dd:02d}° {dm:02d}' {ds:02d}\""


def site_from_fits(path):
    """(lat, lon, focal mm, pixel µm, width px, height px) from a FITS / XISF header - whatever it has."""
    import stacking as st
    src = st.ImageSource(path)
    h = dict(src.look)
    h.setdefault("NAXIS1", src.w)
    h.setdefault("NAXIS2", src.h)

    def f(*keys):
        for k in keys:
            v = h.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    try:
                        return _sexagesimal(str(v))
                    except ValueError:
                        pass
        return None
    return (f("SITELAT", "LAT-OBS", "OBSGEO-B"), f("SITELONG", "LONG-OBS", "OBSGEO-L"), f("FOCALLEN"),
            f("XPIXSZ", "PIXSIZE1"), f("NAXIS1"), f("NAXIS2"))


def _sexagesimal(s):
    parts = [float(x) for x in re.split(r"[\s:°'\"dms]+", s.strip()) if x not in ("", "+", "-")]
    if not parts:
        raise ValueError(s)
    v = abs(parts[0]) + (parts[1] / 60 if len(parts) > 1 else 0) + (parts[2] / 3600 if len(parts) > 2 else 0)
    return -v if s.strip().startswith("-") else v


def nina_csv(targets, path):
    """A Telescopius-style target list that N.I.N.A. imports: Sequencer › (simple sequencer) › Import Targets.
    N.I.N.A. reads 'Familiar Name', 'Catalogue Entry', 'Right Ascension (J2000)', 'Declination (J2000)' and
    'Position Angle (East)' (it needs the '(J2000)' column names - a plain 'Right Ascension' column would be
    read as the declination too)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Familiar Name", "Catalogue Entry", "Right Ascension (J2000)", "Declination (J2000)",
                    "Position Angle (East)"])
        for t in targets:
            h = t.ra / 15
            ra = f"{int(h):02d} {int(h * 60) % 60:02d} {(h * 3600) % 60:04.1f}"
            d = abs(t.dec)
            dec = f"{'-' if t.dec < 0 else '+'}{int(d):02d} {int(d * 60) % 60:02d} {(d * 3600) % 60:04.1f}"
            label = t.name if not t.common else f"{t.name} {t.common}"
            w.writerow([label, t.name, ra.strip(), dec.strip(), 0])
    return Path(path)


# ---------------------------------------------------------------- N.I.N.A. advanced sequencer targets
def nina_targets_folder(profile: dict | None = None) -> Path:
    """Where N.I.N.A. keeps its saved targets (it watches the folder and its subfolders, so a file written
    there appears in the Sequencer's target list straight away)."""
    if profile and profile.get("targets"):
        return Path(profile["targets"])
    return Path.home() / "Documents" / "N.I.N.A" / "Targets"


def _coords(t: Target) -> dict:
    h = (t.ra % 360) / 15
    hh, mm = int(h), int((h * 60) % 60)
    ss = round((h * 3600) % 60, 5)
    d = abs(t.dec)
    dd, dm = int(d), int((d * 60) % 60)
    ds = round((d * 3600) % 60, 5)
    return {"RAHours": hh, "RAMinutes": mm, "RASeconds": ss, "NegativeDec": t.dec < 0,
            "DecDegrees": dd, "DecMinutes": dm, "DecSeconds": ds}


def nina_target_json(t: Target, name: str, template: str | None = None) -> str:
    """A N.I.N.A. Deep Sky Object container for the target. With a template (one of your own saved N.I.N.A.
    targets) everything you set up in it - exposures, autofocus, dithering, checks - is kept and only the
    name and every set of coordinates change; without one, an empty container with just the target."""
    import json
    title = t.common or t.name
    if template:
        doc = json.loads(template)

        def walk(o):
            if isinstance(o, dict):
                if "InputCoordinates" in str(o.get("$type", "")) or {"RAHours", "DecDegrees"} <= set(o):
                    o.update(_coords(t))
                if "TargetName" in o:
                    o["TargetName"] = title
                    if "PositionAngle" in o:
                        o["PositionAngle"] = 0.0
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(doc)
        doc["Name"] = name
        return json.dumps(doc, indent=2)
    coll = "System.Collections.ObjectModel.ObservableCollection`1[[NINA.Sequencer.{}, NINA.Sequencer]], System.ObjectModel"
    doc = {
        "$id": "1", "$type": "NINA.Sequencer.Container.DeepSkyObjectContainer, NINA.Sequencer",
        "Target": {"$id": "2", "$type": "NINA.Astrometry.InputTarget, NINA.Astrometry", "Expanded": True,
                   "TargetName": title, "PositionAngle": 0.0,
                   "InputCoordinates": dict({"$id": "3", "$type": "NINA.Astrometry.InputCoordinates, NINA.Astrometry"},
                                            **_coords(t))},
        "ExposureInfoListExpanded": False,
        "ExposureInfoList": {"$id": "4", "$type": "NINA.Core.Utility.AsyncObservableCollection`1[[NINA.Sequencer."
                             "Utility.ExposureInfo, NINA.Sequencer]], NINA.Core", "$values": []},
        "Strategy": {"$type": "NINA.Sequencer.Container.ExecutionStrategy.SequentialStrategy, NINA.Sequencer"},
        "Name": name,
        "Conditions": {"$id": "5", "$type": coll.format("Conditions.ISequenceCondition"), "$values": []},
        "IsExpanded": True,
        "Items": {"$id": "6", "$type": coll.format("SequenceItem.ISequenceItem"), "$values": []},
        "Triggers": {"$id": "7", "$type": coll.format("Trigger.ISequenceTrigger"), "$values": []},
        "Parent": None, "ErrorBehavior": 0, "Attempts": 1,
    }
    return json.dumps(doc, indent=2)


def save_nina_targets(targets, folder, night_label="", template: str | None = None) -> list:
    """Write each target into N.I.N.A.'s targets folder (a 'SubZeroAstro' subfolder). Returns the paths."""
    folder = Path(folder) / "SubZeroAstro"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for t in targets:
        name = f"{t.common or t.name}{' ' + night_label if night_label else ''}"
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip() or "Target"
        p = folder / f"{stem}.json"
        n = 1
        while p.exists():
            n += 1
            p = folder / f"{stem} ({n}).json"
        p.write_text(nina_target_json(t, name, template), encoding="utf-8")
        out.append(p)
    return out
