"""
stacking.py - calibration masters, star alignment and stacking (NumPy only).

Pipeline for each light group (target + filter + camera):
  1. Choose calibration: bias, dark (exact exposure, or scaled with bias),
     flat (+ bias or dark-flat for the flat itself). Masters come from the
     calibration library, from masters the user adds, or are built from
     subframes the user adds (and then saved to the library).
  2. Calibrate each light: (light - dark) / normalised flat, then replace hot
     pixels found in the master dark. Colour (OSC) frames are debayered.
  3. Align every frame to the best-scoring frame by matching star triangles,
     then resample with bilinear interpolation.
  4. Normalise (sky level and star brightness) and weight by noise.
  5. Combine with sigma-clipped weighted mean (outliers such as satellite
     trails, planes and hot pixels are rejected).
  6. Save as 32-bit float FITS and/or XISF with a full record in the header
     and a text log next to the file.
Everything works on the raw row order of the files; masters keep the
ROWORDER of their frames so viewers show them the right way up.
"""

from __future__ import annotations

import math
import os
import sys
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path

import numpy as np

import checkpoints as cp
import fits_core as core
import metrics as M
import xisf

IMAGE_EXTS = core.FITS_EXTS | {".xisf"}


class Cancelled(Exception):
    pass


def open_files_for(n: int) -> bool:
    """Make room for n more open files. Combining keeps every cached frame open (memory-mapped); macOS and Linux
    allow only 256 or 1024 open files by default, so 3046 Seestar frames failed with 'Too many open files'.
    Raises this process's soft limit (up to the system's hard limit). True if there is room now."""
    try:
        import resource
    except ImportError:                                  # Windows: memory-mapped files don't use up this limit
        return True
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = n + 256
        if soft != resource.RLIM_INFINITY and soft < want:
            cap = hard if hard != resource.RLIM_INFINITY else want
            if sys.platform == "darwin":
                cap = min(cap, 1_000_000)
                try:
                    import subprocess
                    mx = int(subprocess.run(["sysctl", "-n", "kern.maxfilesperproc"], capture_output=True,
                                            text=True, timeout=5).stdout.strip() or 0)
                    if mx > 0:
                        cap = min(cap, mx)
                except Exception:  # noqa: BLE001
                    pass
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, cap), hard))
            soft = min(want, cap)
        return soft == resource.RLIM_INFINITY or soft >= want
    except Exception:  # noqa: BLE001
        return False


# ==========================================================================
# Image access (FITS and XISF, whole image or blocks of rows)
# ==========================================================================

class ImageSource:
    def __init__(self, path):
        self.path = Path(path)
        self.kind = "xisf" if self.path.suffix.lower() == ".xisf" else "fits"
        self.scale = 1.0
        self._data = None
        if self.kind == "xisf":
            _r, _img, self.cards, self.look = xisf.read_header(self.path)
            self.w, self.h = int(self.look["NAXIS1"]), int(self.look["NAXIS2"])
            self.channels = int(self.look.get("_XISF_CHANNELS", 1))
            # PixInsight stores floating-point images in the range 0..1. Without a stated range,
            # decide from the values (a master dark or bias of 0..1 must not be taken as 0..1 ADU)
            if self.look.get("_XISF_FORMAT", "").startswith("Float"):
                b = str(self.look.get("_XISF_BOUNDS", "") or "")
                self.scale = 65535.0 if b.startswith("0:1") else None     # None: decided from the values
        else:
            with open(self.path, "rb") as f:
                self.lay = core.read_layout(f)
            self.cards, self.look = self.lay.cards, self.lay.look
            self.h, self.w = self.lay.shape[-2], self.lay.shape[-1]
            self.channels = self.lay.shape[0] if len(self.lay.shape) == 3 else 1
            # Floating-point masters saved in PixInsight's 0..1 convention
            pr = self.look.get("PIXRANGE")
            if isinstance(pr, (int, float)) and pr > 0:
                self.scale = float(pr)
            elif self.lay.bitpix < 0:
                # floating point - masters, and lights calibrated in PixInsight (the comet's _c_d_r frames) - may be
                # on the 0..1 scale: decide from the data (0..1 or ADU). Left at 0..1 the comet master was 65535x
                # too dark and every level after it (sky noise floors, the stretch's white point) went wrong
                self.scale = None

    def _scaled(self, d):
        if self.scale is None:
            self.scale = 65535.0 if float(np.nanmax(d[::7, ::7] if d.ndim == 2 else d[0, ::7, ::7])) <= 1.5 else 1.0
        if self.scale != 1.0:
            d = d * np.float32(self.scale)
        return d

    def read(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        if self.kind == "xisf":
            return self._scaled(xisf.read_xisf(self.path)[0])
        return self._scaled(core.read_fits(self.path)[0])

    def keep_in_memory(self):
        self._data = self.read()

    def read_rows(self, r0: int, n: int) -> np.ndarray:
        if self._data is not None or self.kind == "xisf" or self.channels != 1:
            if self._data is None:
                self._data = self.read()
            d = self._data if self._data.ndim == 2 else self._data[0]
            return d[r0:r0 + n]
        if self.scale is None:
            self._data = self.read()
            return self._data[r0:r0 + n]
        with open(self.path, "rb") as f:
            rows = core.read_rows(f, self.lay, r0, n)
        return rows * np.float32(self.scale) if self.scale != 1.0 else rows


# ==========================================================================
# What kind of frame is this?
# ==========================================================================

def frame_type(look: dict, name: str):
    """Return (type, is_master); type is light / dark / flat / bias / darkflat or None."""
    t = str(look.get("IMAGETYP") or look.get("FRAME") or look.get("FRAMETYP") or "").lower()
    n = name.lower()
    master = "master" in t or "master" in n or (isinstance(look.get("NCOMBINE"), int) and look["NCOMBINE"] > 1)

    def kind(s):
        if ("dark" in s and "flat" in s) or "flatdark" in s.replace(" ", "") or "darkflat" in s.replace(" ", ""):
            return "darkflat"
        if "flat" in s:
            return "flat"
        if "dark" in s:
            return "dark"
        if "bias" in s or "offset" in s or "zero" in s:
            return "bias"
        if "light" in s or "science" in s or "object" in s:
            return "light"
        return None

    k = kind(t) or kind(n)
    if k is None and (core.smart_scope(look) or look.get("STACKCNT")):
        k = "light"                     # smart telescopes don't write IMAGETYP on their lights
    return k, master


@dataclass
class ImgInfo:
    path: Path
    type: str | None
    master: bool
    camera: str | None
    gain: float | None
    offset: float | None
    exptime: float | None
    temp: float | None
    filter: str
    w: int
    h: int
    x0: int | None
    y0: int | None
    binning: int | None
    bayer: str | None
    night: str | None
    date_obs: str | None
    focal: float | None
    telescope: str | None
    rotator: float | None
    target: str | None
    roworder: str | None
    ncombine: int | None = None
    planes: int = 1                     # 3 = already-debayered colour (e.g. a Seestar stack)
    stackcnt: int | None = None         # frames inside a smart-telescope stack
    totalexp: float | None = None
    smart: str | None = None
    precal: str | None = None           # already calibrated (ready-made data): why we think so

    @property
    def exposure_total(self) -> float:
        """Real integration time this file holds (a stack counts all its frames)."""
        if self.totalexp:
            return float(self.totalexp)
        if self.stackcnt and self.exptime:
            return float(self.stackcnt) * float(self.exptime)
        return float(self.exptime or 0)


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def info_from_look(path, look, w, h, typ=None, master=False, planes=1, cards=None) -> ImgInfo:
    fi = core.frame_info(look)
    xb = _num(look.get("XBINNING") or look.get("CCDXBIN"))
    return ImgInfo(
        path=Path(path), type=typ, master=master,
        camera=(str(fi["camera"]).strip() if fi.get("camera") else None),
        gain=_num(fi.get("gain")), offset=_num(fi.get("offset")),
        exptime=_num(fi.get("exptime")), temp=_num(fi.get("temp")),
        filter=fi.get("filter") or "No filter", w=int(w), h=int(h),
        x0=(int(look["XORGSUBF"]) if _num(look.get("XORGSUBF")) is not None else None),
        y0=(int(look["YORGSUBF"]) if _num(look.get("YORGSUBF")) is not None else None),
        # a 3-plane image is already colour, whatever BAYERPAT still says (PixInsight keeps it after debayering:
        # the comet's _c_d_r frames were stacked as one Bayer plane - grey)
        binning=int(xb) if xb else None, bayer=fi.get("bayer") if int(planes or 1) == 1 else None,
        night=core.night_of(fi), date_obs=(str(fi.get("date_obs")) if fi.get("date_obs") else None),
        focal=_num(look.get("FOCALLEN")), telescope=fi.get("telescope"),
        rotator=_num(fi.get("rotator")),
        target=(str(fi["object"]).strip() if fi.get("object") else None),
        roworder=(str(look.get("ROWORDER")).strip().upper() if look.get("ROWORDER") else None),
        ncombine=int(look["NCOMBINE"]) if isinstance(look.get("NCOMBINE"), int) else None,
        planes=int(planes or 1), stackcnt=int(_num(fi.get("stackcnt"))) if _num(fi.get("stackcnt")) else None,
        totalexp=_num(fi.get("totalexp")), smart=fi.get("smart"),
        precal=(core.precalibrated(cards, look, Path(path).name) if (typ in (None, "light") and not master)
                else None))


_FOLDER_KIND = [("darkflat", r"\b(dark[\s_-]*flats?|flat[\s_-]*darks?)\b"), ("flat", r"\bflats?\b"),
                ("dark", r"\bdarks?\b"), ("bias", r"\b(bias|biases|offsets?)\b")]


def folder_kind(folder: str):
    """bias / dark / flat / darkflat from a folder name such as 'Darks' or 'Flat_H', or None."""
    f = re.sub(r"[_\-.]", " ", (folder or "").lower())
    for k, pat in _FOLDER_KIND:
        if re.search(pat, f):
            return k
    return None


_NOT_TARGET = re.compile(r"^(light|lights|flat|flats|dark|darks|bias|biases|darkflat|darkflats|autorun|plan|live|preview|"
                         r"rejected|calibrated|registered|masters?|stack(ed)?|\d{4}-\d{2}-\d{2}.*|[lrgbhaoiis]+|"
                         r"luminance|red|green|blue|h-?alpha|oiii|sii|ha|session.*|night.*|\d+s?)$", re.I)


def target_from_name(path) -> str | None:
    """The target when the header does not say (no OBJECT): from the file name the capture program wrote, or
    from the folder it is in. ASIAIR / Seestar: Light_M51_180.0s_Bin1_6200MC_gain0_20240525-212957_-6.6C_0001.fit,
    in .../Autorun/Light/M51/. N.I.N.A. and others often: M51_2024-05-25_..."""
    p = Path(path)
    stem = p.stem
    m = re.match(r"^(?:light|lights)[_ -](.+?)[_ ]\d+(?:\.\d+)?s(?:ec)?[_ ]", stem, re.I)       # ASIAIR / Seestar
    if m and not _NOT_TARGET.match(m.group(1).strip()):
        return m.group(1).strip()
    m = re.match(r"^([A-Za-z][A-Za-z0-9 +\-]*?\d[A-Za-z0-9 +\-]*?)[_ ]\d{4}-?\d{2}-?\d{2}", stem)   # M51_2024-05-25…
    if m and not _NOT_TARGET.match(m.group(1).strip()):
        return m.group(1).strip()
    # the folder: ASIAIR keeps lights in .../Light/<target>/
    for anc in list(p.parents)[:2]:
        if anc.parent.name.lower() in ("light", "lights") and not _NOT_TARGET.match(anc.name):
            return anc.name
    # a folder named after a known target (M51, NGC 1097, C_2025_R2, 'Comet PANSTARRS'…), a level or two up
    import targets as _tg
    for anc in list(p.parents)[:3]:
        if anc.name and not _NOT_TARGET.match(anc.name) and _tg.classify(anc.name):
            return anc.name
    return None


def read_info(path) -> ImgInfo:
    src = ImageSource(path)
    typ, master = frame_type(src.look, src.path.name)
    if typ is None:                      # no IMAGETYP and nothing in the file name: the folder may say
        typ = folder_kind(src.path.parent.name)
    info = info_from_look(path, src.look, src.w, src.h, typ, master, planes=src.channels, cards=src.cards)
    if typ in (None, "light") and (not info.target or _GENERIC_TARGET.match(str(info.target).strip())):
        # no OBJECT in the header (some ASIAIR files), or a placeholder like 'Snapshot' (Telescope Live comet
        # frames): the file name or the folder says
        info.target = target_from_name(path) or info.target
    return info


# OBJECT values that are not a target's name
_GENERIC_TARGET = re.compile(r"^(snapshot|snap|target|object|unknown|untitled|none|test|light|lights|image|"
                             r"frame|default|new target|my target)?$", re.I)


# ==========================================================================
# Names: rigs, targets, cameras
# ==========================================================================

def safe_name(s: str) -> str:
    s = core._ascii(s)
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().rstrip(".")
    return s or "Unknown"


def short_camera(cam: str | None) -> str:
    if not cam:
        return "Camera"
    s = re.sub(r"^(ZWO|QHY|Player\s*One|ToupTek|Atik|SBIG|FLI|Moravian|Altair)\s+", "", cam.strip(), flags=re.I)
    s = re.sub(r"\s+(Pro|Mini|Cool(ed)?)$", "", s, flags=re.I)
    return safe_name(s.replace(" ", ""))


def target_name(t: str | None, filt: str | None = None) -> str:
    """The target's name as a folder-safe word. With the frame's filter, a filter tacked onto the name is
    dropped: Telescope Live (and some sequencers) write OBJECT = 'M78_B' / 'M78_Ha' / 'NGC 7000 - Red', which
    would otherwise make every filter a target of its own (so its colours were never combined)."""
    if not t:
        return "Unknown"
    s = str(t).strip()
    if filt:
        f = core.normalize_filter(filt)
        m = re.match(r"^(.*?[^\s_\-])[\s_\-]+([^\s_\-]+)$", s)
        if m and f not in ("No filter", "") and core.normalize_filter(m.group(2)) == f and m.group(2) != m.group(1):
            s = m.group(1)
    return safe_name(s.replace(" ", ""))


class Rigs:
    """Telescope (rig) names, remembered per camera + focal length."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])     # [{"camera":..., "focal":..., "name":...}]

    def lookup(self, camera, focal):
        for e in self.entries:
            if (e.get("camera") or "").lower() != (camera or "").lower():
                continue
            ef, f = e.get("focal"), focal
            if ef is None or f is None:
                if ef is None and f is None:
                    return e["name"]
                continue
            if abs(float(ef) - float(f)) <= 0.03 * max(float(ef), 1):
                return e["name"]
        return None

    def set(self, camera, focal, name):
        self.entries = [e for e in self.entries
                        if not ((e.get("camera") or "").lower() == (camera or "").lower()
                                and self._same(e.get("focal"), focal))]
        self.entries.append({"camera": camera, "focal": focal, "name": name})

    @staticmethod
    def _same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= 0.03 * max(float(a), 1)

    @staticmethod
    def suggestion(info: ImgInfo) -> str:
        t = (info.telescope or "").strip()
        # mount drivers often land in TELESCOP (ASCOM / N.I.N.A.): AvalonStarGo.NET, EQMOD, OnStep, Simulator …
        mountish = re.search(r"\b(AM\d|EQ\d|HEQ|CEM|GEM|RST|mount|AZ-|iOptron|Rainbow|Sky-?Watcher\s+EQ|StarGo|"
                             r"Avalon|EQMOD|EQ-?ASCOM|GSServer|Green\s*Swamp|OnStep|10\s*Micron|Paramount|"
                             r"TheSkyX|Losmandy|Gemini|Astro-?Physics\s+GTO|AP\s*V2|Simulator|ASCOM|Telescope\s*\d|"
                             r"Alpaca|INDI|LX200|NexStar|SynScan|Pegasus\s*NYX|ZWO\s*AM)|\.NET\b", t, re.I)
        if t and not mountish:
            return safe_name(t)
        return f"{info.focal:g}mm scope" if info.focal else "My scope"

    def name_for(self, info: ImgInfo) -> str:
        return self.lookup(info.camera, info.focal) or self.suggestion(info)


# ==========================================================================
# Calibration items (existing masters, or sets of subframes to build)
# ==========================================================================

KIND_FOLDER = {"bias": "Bias", "dark": "Darks", "darkflat": "Darks", "flat": "Flats"}
KIND_LABEL = {"bias": "Bias", "dark": "Dark", "darkflat": "Dark-flat", "flat": "Flat"}


@dataclass
class CalItem:
    kind: str
    infos: list                         # ImgInfo of each subframe, or the one master
    master: bool                        # True = an existing master file
    origin: str = "added"               # "library" or "added"
    rig: str | None = None              # from the library folder, if known
    data: np.ndarray | None = None      # built / loaded master (raw ADU; flats not yet normalised)
    saved: list = field(default_factory=list)
    hot: tuple | None = None            # hot pixel coordinates (darks)
    sky: bool = False                   # a sky flat made from light frames (never saved to the library)
    sky_stats: dict = field(default_factory=dict)   # how good a sky flat came out (see sky_flat_problems)
    skipped: list = field(default_factory=list)     # subframes that were gone or unreadable when building

    @property
    def ref(self) -> ImgInfo:
        return self.infos[0]

    def __getattr__(self, name):        # camera, gain, ... come from the first frame
        if name in ("camera", "gain", "offset", "exptime", "temp", "filter", "w", "h", "x0", "y0",
                    "binning", "bayer", "night", "focal", "telescope", "rotator", "roworder"):
            return getattr(self.infos[0], name)
        raise AttributeError(name)

    @property
    def count(self):
        if self.master:
            return self.ref.ncombine or 0
        return len(self.infos)

    def label(self) -> str:
        if self.sky:
            n_t = len({target_name(i.target, i.filter) for i in self.infos})
            return (f"SkyFlat (Mote Buster) · {self.filter}  (made from {len(self.infos)} light frames, "
                    f"{n_t} target{'s' if n_t != 1 else ''})")
        bits = [KIND_LABEL[self.kind]]
        if self.kind in ("dark", "darkflat") and self.exptime is not None:
            bits.append(f"{self.exptime:g} s")
        if self.kind == "flat":
            bits.append(self.filter)
        if self.gain is not None:
            bits.append(f"gain {self.gain:g}")
        if self.temp is not None and self.kind in ("dark", "bias", "darkflat"):
            bits.append(f"{self.temp:g} °C")
        if self.kind == "flat" and self.night:
            bits.append(self.night)
        what = (f"master{' of ' + str(self.count) if self.count else ''}" if self.master
                else f"{len(self.infos)} subframes")
        src = "library" if self.origin == "library" else "added"
        return f"{' · '.join(bits)}  ({what}, {src}: {self.ref.path.name if self.master else self.ref.path.parent.name})"


def _temp_bucket(t):
    return None if t is None else round(t / 2.0)


def group_calibration(infos) -> list[CalItem]:
    """Turn calibration file infos into CalItems: one per master, and one per
    set of compatible subframes."""
    items, sets = [], {}
    for i in infos:
        if i.type not in KIND_FOLDER:
            continue
        if i.master:
            items.append(CalItem(i.type, [i], True))
            continue
        key = (i.type, (i.camera or "").lower(), i.gain, i.offset,
               round(i.exptime, 2) if (i.exptime is not None and i.type in ("dark", "darkflat")) else None,
               _temp_bucket(i.temp) if i.type in ("dark", "darkflat", "bias") else None,
               i.filter if i.type == "flat" else None,
               i.w, i.h, i.x0, i.y0, i.binning,
               i.night if i.type == "flat" else None)
        sets.setdefault(key, []).append(i)
    for key, lst in sets.items():
        lst.sort(key=lambda x: core.natural_key(x.path.name))
        items.append(CalItem(key[0], lst, False))
    return items


def _is_dir(p) -> bool:
    """Path.is_dir() that never raises: a network drive can answer some names with 'Invalid argument'
    (EINVAL) - a broken alias, a name the share cannot handle - and one such entry must not stop the rest."""
    try:
        return Path(p).is_dir()
    except OSError:
        return False


def _safe_listdir(p) -> list:
    """Path.iterdir() as a list, empty when the folder cannot be read."""
    try:
        return list(Path(p).iterdir())
    except OSError:
        return []


def scan_paths(paths, progress=None):
    """Read headers of all FITS/XISF files in the given files/folders (recursive).
    Returns (infos, problems)."""
    files = []
    for p in paths:
        p = Path(p)
        if _is_dir(p):
            for dirpath, dirnames, filenames in os.walk(p, onerror=lambda e: None):
                dirnames[:] = [d for d in dirnames if d.upper() != core.REJECTED_DIR and not d.startswith(".")]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and not core.is_hidden_file(fn):
                        files.append(Path(dirpath) / fn)
        elif p.suffix.lower() in IMAGE_EXTS and not core.is_hidden_file(p.name):
            files.append(p)
    infos, problems = [], []
    for k, f in enumerate(sorted(set(files), key=lambda x: core.natural_key(str(x)))):
        if progress and k % 10 == 0:
            progress(f"Reading headers… {k}/{len(files)}", k / max(len(files), 1))
        try:
            infos.append(read_info(f))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f.name}: {e}")
    return infos, problems


def library_dir(cal_root, rig, camera, kind, filt=None) -> Path:
    d = Path(cal_root) / safe_name(rig) / "Masters" / safe_name(camera or "Camera") / KIND_FOLDER[kind]
    if kind == "flat":
        d = d / safe_name(filt or "No filter")
    return d


def scan_library(cal_root, progress=None) -> list[CalItem]:
    """Masters already saved in <cal_root>/<rig>/Masters/..."""
    root = Path(cal_root) if cal_root else None
    if not root or not _is_dir(root):
        return []
    items = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for rig_dir in sorted(p for p in entries if _is_dir(p)):
        mdir = rig_dir / "Masters"
        if not _is_dir(mdir):
            continue
        infos, _ = scan_paths([mdir], progress)
        # the same master saved as both .fits and .xisf counts once (FITS preferred)
        by_base = {}
        for i in infos:
            b = i.path.with_suffix("")
            if b not in by_base or i.path.suffix.lower() != ".xisf":
                by_base[b] = i
        infos = list(by_base.values())
        for i in infos:
            if i.type in KIND_FOLDER:
                i.master = True
                items.append(CalItem(i.type, [i], True, origin="library", rig=rig_dir.name))
    return items


# ==========================================================================
# Light groups and calibration plans
# ==========================================================================

@dataclass
class LightFrame:
    info: ImgInfo
    score: float | None = None
    fwhm: float | None = None


@dataclass
class LightGroup:
    target: str
    filter: str
    camera: str | None
    rig: str
    frames: list
    plan: "CalPlan | None" = None
    enabled: bool = True

    @property
    def ref(self) -> ImgInfo:
        return self.frames[0].info

    @property
    def total_exposure(self):
        return sum(f.info.exposure_total for f in self.frames)

    @property
    def nights(self):
        return sorted({f.info.night for f in self.frames if f.info.night})

    def label(self):
        cam = short_camera(self.camera)
        return f"{self.target} · {self.filter} · {cam} · {self.rig}"


NARROW_LINES = ("Ha", "OIII", "SII", "Hβ")
BROAD_ORDER = ("L", "R", "G", "B")


def filter_band(name) -> str:
    """'broad' (L, R, G, B, no filter, light-pollution filters) or 'narrow' (Ha, OIII, SII, dual-band)."""
    n = name or ""
    return "narrow" if n in NARROW_LINES or core.is_dual_band(n) else "broad"


def _filter_order(name):
    n = name or ""
    if n in BROAD_ORDER:
        return (0, BROAD_ORDER.index(n), n)
    if filter_band(n) == "broad":
        return (0, 9, n)
    return (1, NARROW_LINES.index(n) if n in NARROW_LINES else 9, n)


def group_lights(lights: list[LightFrame], rigs: Rigs) -> list[LightGroup]:
    groups = {}
    for lf in lights:
        i = lf.info
        key = (target_name(i.target, i.filter), i.filter, (i.camera or "").lower(), bool(i.bayer) or i.planes > 1,
               i.w, i.h, i.x0, i.y0, i.binning, rigs.name_for(i), bool(i.precal))
        groups.setdefault(key, []).append(lf)
    out = []
    for key, lst in groups.items():
        lst.sort(key=lambda f: (f.info.date_obs or "", str(f.info.path)))
        out.append(LightGroup(key[0], key[1], lst[0].info.camera, key[9], lst))
    # per target: broadband first (L, R, G, B), then narrowband (Ha, OIII, SII)
    out.sort(key=lambda g: (g.target, g.camera or "", _filter_order(g.filter)))
    return out


@dataclass
class CalPlan:
    bias: CalItem | None = None
    dark: CalItem | None = None
    dark_mode: str = "none"             # exact / scaled / none
    flat: CalItem | None = None
    flat_bias: CalItem | None = None    # how the flat subframes are calibrated
    flat_dark: CalItem | None = None
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)     # (kind, text) kind in bias/dark/flat/other
    precal: str | None = None           # the lights are already calibrated: nothing is applied
    night_flats: dict = field(default_factory=dict)  # night -> (flat, flat_bias, flat_dark): each night its own flat

    def missing(self):
        return [k for k, _ in self.warnings]

    def flat_sets(self):
        """(flat, flat_bias, flat_dark) for every flat this plan uses: the main one and the per-night ones."""
        out = [(self.flat, self.flat_bias, self.flat_dark)] if self.flat is not None else []
        for trio in self.night_flats.values():
            if all(trio[0] is not t[0] for t in out):
                out.append(trio)
        return out

    def all_items(self):
        """Every calibration item this plan uses (each once)."""
        seen, out = set(), []
        for c in [self.bias, self.dark] + [x for t in self.flat_sets() for x in t]:
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                out.append(c)
        return out


def _same_cam(a, b):
    return a is None or b is None or a.strip().lower() == b.strip().lower()


def _eq(a, b, tol=0.0):
    return a is None or b is None or abs(a - b) <= tol


def geometry_ok(cal: ImgInfo | CalItem, light: ImgInfo) -> bool:
    if cal.binning and light.binning and cal.binning != light.binning:
        return False
    if cal.w == light.w and cal.h == light.h:
        return (cal.x0 or 0) == (light.x0 or 0) and (cal.y0 or 0) == (light.y0 or 0) \
            or cal.x0 is None or light.x0 is None
    if light.x0 is None or light.y0 is None:
        return False
    cx, cy = (cal.x0 or 0), (cal.y0 or 0)
    return (light.x0 - cx >= 0 and light.y0 - cy >= 0 and
            light.x0 - cx + light.w <= cal.w and light.y0 - cy + light.h <= cal.h)


def crop_to(cal_data: np.ndarray, cal: ImgInfo | CalItem, light: ImgInfo) -> np.ndarray:
    if cal_data.shape[-2:] == (light.h, light.w):
        return cal_data
    ox, oy = light.x0 - (cal.x0 or 0), light.y0 - (cal.y0 or 0)
    return cal_data[..., oy:oy + light.h, ox:ox + light.w]


def _rig_ok(item: CalItem, rig: str):
    return item.rig is None or item.rig.lower() == rig.lower()


SKY_FLAT_MIN = 12                       # light frames needed for a sky flat


def sky_flat_frames(g: "LightGroup", pool) -> list:
    """Light frames (ImgInfo) that can go into a sky flat for this group: same camera, filter, rig and
    sensor area - from every target and night that qualifies (different fields make it better)."""
    L = g.ref
    out = []
    for i in pool or []:
        if (_same_cam(i.camera, L.camera) and i.filter == L.filter and i.w == L.w and i.h == L.h
                and (i.x0 or 0) == (L.x0 or 0) and (i.y0 or 0) == (L.y0 or 0) and i.binning == L.binning
                and bool(i.bayer) == bool(L.bayer) and not i.smart and not i.precal and i.planes == 1
                and getattr(i, "_rig", g.rig) == g.rig):
            out.append(i)
    return out


def plan_group(g: LightGroup, items: list[CalItem], sky_pool=None) -> CalPlan:
    L = g.ref
    p = CalPlan()
    if L.precal:
        # ready-made data (Telescope Live and the like): calibrating again would subtract the dark
        # twice and divide by the flat twice, and a SkyFlat would correct what is already corrected
        p.precal = L.precal
        p.notes.append(f"These lights are already calibrated ({L.precal}) - no bias, darks, flats or "
                       "SkyFlat are applied again.")
        return p
    cam_items = [c for c in items if _same_cam(c.camera, L.camera) and geometry_ok(c, L)]

    def pick(cands, key):
        cands = sorted(cands, key=key)
        return cands[0] if cands else None

    def unknown(c):
        return sum(v is None for v in (c.camera, c.gain, c.offset))

    # bias for the lights
    biases = [c for c in cam_items if c.kind == "bias" and _eq(c.gain, L.gain) and _eq(c.offset, L.offset)]
    p.bias = pick(biases, lambda c: (unknown(c), 0 if c.origin == "added" else 1, -c.count))

    # darks
    darks = [c for c in cam_items if c.kind == "dark" and _eq(c.gain, L.gain) and _eq(c.offset, L.offset)
             and _eq(c.temp, L.temp, 3.0)]
    exact = [c for c in darks if c.exptime is not None and L.exptime is not None
             and abs(c.exptime - L.exptime) <= 0.02 * L.exptime]
    if exact:
        p.dark = pick(exact, lambda c: (unknown(c), abs((c.temp or 0) - (L.temp or 0)),
                                        0 if c.origin == "added" else 1, -c.count))
        p.dark_mode = "exact"
    elif darks and p.bias is not None:
        p.dark = pick(darks, lambda c: (abs((c.exptime or 0) - (L.exptime or 0)), -c.count))
        p.dark_mode = "scaled"
        p.notes.append(f"Darks are {p.dark.exptime:g} s, lights {L.exptime:g} s — dark current is scaled "
                       "using the bias.")
    elif darks:
        other = sorted({f"{c.exptime:g} s" for c in darks if c.exptime})
        p.warnings.append(("dark", f"No darks match the {L.exptime:g} s exposure "
                                   f"(found {', '.join(other) or 'other exposures'}) and there is no bias to "
                                   "scale them with."))
    if p.dark is None and not any(k == "dark" for k, _ in p.warnings):
        temp = f", {L.temp:g} °C" if L.temp is not None else ""
        gain = f", gain {L.gain:g}" if L.gain is not None else ""
        p.warnings.append(("dark", f"No darks for {L.exptime or 0:g} s{gain}{temp}."
                                   + (" The bias will be used instead; hot pixels are removed by stacking "
                                      "if you dithered." if p.bias else "")))
    if p.dark is None and p.bias is None:
        p.warnings.append(("bias", "No bias either — the camera offset will not be removed, so the "
                                   "flat correction will be inaccurate." if True else ""))

    # flats
    flats = [c for c in cam_items if c.kind == "flat" and c.filter == L.filter and _rig_ok(c, g.rig)
             and (not L.bayer or not c.bayer or c.bayer == L.bayer)]
    if flats:
        nights = g.nights

        def dist(c):
            if not c.night or not nights:
                return 9999
            from datetime import date
            d = date.fromisoformat(c.night)
            return min(abs((d - date.fromisoformat(n)).days) for n in nights)
        p.flat = pick(flats, lambda c: (dist(c), 0 if c.origin == "added" else 1, -c.count))
        dd = dist(p.flat)
        if 9999 > dd > 30:
            p.notes.append(f"Heads-up only — this flat WILL be used. It is from {p.flat.night}, {dd} days from "
                           "the lights, so dust may have moved; if dust spots look wrong, take new flats.")
        if p.flat.rotator is not None and L.rotator is not None and abs(p.flat.rotator - L.rotator) > 2:
            p.notes.append(f"Flats were taken at camera angle {p.flat.rotator:g}°, lights at {L.rotator:g}° — "
                           "dust shadows may not line up.")
        def helpers(flat):
            """(bias, dark-flat) that calibrate the flat's own subframes."""
            if flat.master:
                return None, None
            fl = flat.ref
            fdarks = [c for c in items if c.kind in ("darkflat", "dark") and _same_cam(c.camera, fl.camera)
                      and c.exptime is not None and fl.exptime is not None
                      and abs(c.exptime - fl.exptime) <= 0.05 * max(fl.exptime, 0.01)
                      and _eq(c.gain, fl.gain) and geometry_ok(c, fl)]
            fbias = [c for c in items if c.kind == "bias" and _same_cam(c.camera, fl.camera)
                     and _eq(c.gain, fl.gain) and _eq(c.offset, fl.offset) and geometry_ok(c, fl)]
            fd = pick(fdarks, lambda c: (unknown(c), -c.count))
            return (None if fd is not None else pick(fbias, lambda c: (unknown(c), -c.count))), fd

        # flats taken every session: each night's lights get that night's flat (dust moves between nights)
        if len(nights) > 1:
            per = {}
            for n in nights:
                same = [c for c in flats if c.night == n]
                if same:
                    f_ = pick(same, lambda c: (0 if c.origin == "added" else 1, -c.count))
                    per[n] = (f_,) + helpers(f_)
            if len(per) >= 2:
                p.night_flats = per
                rest = [n for n in nights if n not in per]
                p.notes.append(f"Per-night flats: {len(per)} of {len(nights)} nights each use their own flat "
                               f"({', '.join(per)})" + (f"; {', '.join(rest)} use{'s' if len(rest) == 1 else ''} "
                                                        f"the flat from {p.flat.night}" if rest else ""))
        if not p.flat.master:
            fl = p.flat.ref
            p.flat_bias, p.flat_dark = helpers(p.flat)
            if p.flat_dark is None and p.flat_bias is None:
                p.warnings.append(("flatbias", f"The flats (gain {fl.gain if fl.gain is not None else '?'}) have "
                                               "no matching bias or dark-flats, so the camera offset stays in "
                                               "them and the correction will be too strong (dark rings / "
                                               "bright centre)."))
    else:
        sky = sky_flat_frames(g, sky_pool) if sky_pool is not None and not L.smart and L.planes == 1 else []
        if len(sky) >= SKY_FLAT_MIN:
            p.flat = CalItem("flat", sky, False, origin="sky", sky=True)
            if p.dark is not None and p.dark_mode == "exact":
                p.flat_dark = p.dark
            else:
                p.flat_bias = p.bias
            targets = sorted({target_name(i.target, i.filter) for i in sky})
            p.notes.append(f"No flats for filter {L.filter} on {g.rig} - a SkyFlat (Mote Buster) will be made from {len(sky)} "
                           f"of your light frames ({len(targets)} target{'s' if len(targets) != 1 else ''}): "
                           "their stars are hidden and each frame's light-pollution tilt is taken out, leaving "
                           "the vignetting and dust shadows. Real flats are still better.")
            if len(targets) == 1:
                p.notes.append("All these frames are of one target: if its nebula fills most of the picture, "
                               "part of it could be taken for vignetting. Lights of other targets (same filter "
                               "and rig) make a better SkyFlat.")
        else:
            p.warnings.append(("flat", f"No flats for filter {L.filter} on {g.rig} — vignetting and dust "
                                       "shadows will not be corrected."
                               + (f" (A SkyFlat needs at least {SKY_FLAT_MIN} light frames with this filter - "
                                  f"there are {len(sky)}.)" if sky_pool is not None and 0 < len(sky) else "")))
    if L.smart or L.planes > 1:
        # smart telescopes calibrate on board (or the input is already a stack): missing
        # calibration is normal here, so don't stop the user with warnings
        if p.warnings:
            p.notes.append(f"{L.smart or 'Stacked colour input'}: no separate calibration frames needed "
                           "(add darks/flats from the scope if you have them)")
        if L.planes > 1:
            p.bias = p.dark = p.flat = p.flat_bias = p.flat_dark = None
            p.night_flats = {}
            p.dark_mode = "none"
        p.warnings = []
    return p


# ==========================================================================
# Building masters
# ==========================================================================

def _robust_mean(stack: np.ndarray, kappa: float, weights=None, frame_sigma=None):
    """Sigma-clipped (median / MAD) weighted mean along axis 0; NaN = missing.
    frame_sigma: optional per-frame noise; a pixel is only rejected if it is
    further than kappa x max(spread across frames, that frame's own noise) from
    the median, so frames that are simply noisier are not penalised.
    Returns (result, rejected_count)."""
    n = stack.shape[0]
    valid = ~np.isnan(stack)
    k = valid.sum(axis=0)
    if weights is None:
        weights = np.ones(n, np.float32)
    w = weights.reshape((n,) + (1,) * (stack.ndim - 1)).astype(np.float32)
    if n < 5:
        ww = w * valid
        sw = ww.sum(axis=0)
        out = (np.nan_to_num(stack) * ww).sum(axis=0) / np.maximum(sw, 1e-12)
        return np.where(k > 0, out, 0).astype(np.float32), 0
    srt = np.sort(stack, axis=0)                       # NaN sort last
    lo = np.maximum((k - 1) // 2, 0)[None]
    hi = np.maximum(k // 2, 0)[None]
    med = 0.5 * (np.take_along_axis(srt, lo, 0)[0] + np.take_along_axis(srt, hi, 0)[0])
    del srt
    dev = np.abs(stack - med)
    dsrt = np.sort(dev, axis=0)
    mad = 0.5 * (np.take_along_axis(dsrt, lo, 0)[0] + np.take_along_axis(dsrt, hi, 0)[0])
    del dsrt
    sig = np.maximum(1.4826 * mad, 1e-3 * np.maximum(np.abs(med), 1.0))
    if frame_sigma is not None:
        fs = np.asarray(frame_sigma, np.float32).reshape((n,) + (1,) * (stack.ndim - 1))
        keep = valid & (dev <= kappa * np.maximum(sig, fs))
    else:
        keep = valid & (dev <= kappa * sig)
    ww = w * keep
    sw = ww.sum(axis=0)
    out = (np.nan_to_num(stack) * ww).sum(axis=0) / np.maximum(sw, 1e-12)
    out = np.where(sw > 0, out, med)
    rejected = int(valid.sum() - keep.sum())
    return np.where(k > 0, out, 0).astype(np.float32), rejected


def integrate_sources(sources, kappa=3.0, pre=None, progress=None, cancel=None, label="Combining",
                      mem_budget=4e8):
    """Combine equally sized single-channel images in blocks of rows."""
    n = len(sources)
    h, w = sources[0].h, sources[0].w
    rows = int(max(4, min(h, mem_budget / max(n * w * 4 * 3, 1))))
    out = np.empty((h, w), np.float32)
    rejected = 0
    for r0 in range(0, h, rows):
        if cancel and cancel():
            raise Cancelled()
        nr = min(rows, h - r0)
        stack = np.empty((n, nr, w), np.float32)
        for i, s in enumerate(sources):
            block = s.read_rows(r0, nr)
            stack[i] = pre(i, block, r0) if pre else block
        out[r0:r0 + nr], rej = _robust_mean(stack, kappa)
        rejected += rej
        if progress:
            progress(f"{label}: {min(r0 + nr, h)}/{h} rows", (r0 + nr) / h)
    return out, rejected


def _cfa_phases():
    return ((0, 0), (0, 1), (1, 0), (1, 1))


def normalize_flat(flat: np.ndarray, bayer: bool) -> np.ndarray:
    """Divide a flat by its median (per colour-filter position for OSC, which
    keeps the colour balance of the lights)."""
    f = flat.astype(np.float32, copy=True)
    if bayer:
        for i, j in _cfa_phases():
            ph = f[i::2, j::2]
            ph /= max(float(np.median(ph[::4, ::4])), 1e-6)
    else:
        f /= max(float(np.median(f[::4, ::4])), 1e-6)
    np.clip(f, 0.05, None, out=f)
    return f


def build_master(item: CalItem, helper_bias: CalItem | None = None, helper_dark: CalItem | None = None,
                 kappa=3.0, progress=None, cancel=None) -> np.ndarray:
    """Load an existing master or combine subframes into one (raw ADU)."""
    if item.data is not None:
        return item.data
    if item.master:
        item.data = ImageSource(item.ref.path).read().astype(np.float32)
        if item.data.ndim == 3:
            item.data = item.data[0]
        return item.data
    sources, gone = [], set()
    for i in item.infos:
        # a file moved, renamed or deleted since it was added (or a drive that dropped out): leave it out
        # and say so, rather than stopping the whole stack - as long as enough are left
        try:
            sources.append(ImageSource(i.path))
        except OSError as e:
            item.skipped.append((Path(i.path).name, getattr(e, "strerror", None) or str(e)))
            gone.add(str(i.path))
    if item.skipped and len(sources) < max(3, len(item.infos) // 2):
        names = ", ".join(n for n, _ in item.skipped[:3]) + (" …" if len(item.skipped) > 3 else "")
        raise FileNotFoundError(f"{len(item.skipped)} of the {len(item.infos)} {KIND_LABEL[item.kind].lower()} "
                                f"subframes can't be opened any more ({names}) - were they moved, renamed or "
                                "is the drive disconnected? Add the calibration folder again.")
    if item.skipped:
        item.infos = [i for i in item.infos if str(i.path) not in gone]
    sizes = {(s.h, s.w) for s in sources}
    if len(sizes) > 1:
        raise ValueError(f"{KIND_LABEL[item.kind]} subframes have different sizes")
    label = f"Building master {KIND_LABEL[item.kind].lower()} ({len(sources)} frames)"
    if item.kind != "flat":
        item.data, _ = integrate_sources(sources, kappa, None, progress, cancel, label)
        return item.data

    # flats: subtract dark-flat or bias, then scale each frame to median 1
    sub = None
    helper = helper_dark or helper_bias
    if helper is not None:
        hd = build_master(helper, kappa=kappa, cancel=cancel)
        sub = crop_to(hd, helper, item.ref)
    if item.sky:
        item.data = build_sky_flat(item, sub, progress, cancel)
        return item.data
    scales = []
    for k, s in enumerate(sources):
        if cancel and cancel():
            raise Cancelled()
        sample = np.concatenate([s.read_rows(r, 1) for r in range(0, s.h, max(1, s.h // 64))])
        if sub is not None:
            sample = sample - sub[::max(1, s.h // 64)][:len(sample)]
        scales.append(1.0 / max(float(np.median(sample)), 1e-6))
        if progress:
            progress(f"Measuring flats: {k + 1}/{len(sources)}", (k + 1) / len(sources) * 0.2)

    def pre(i, block, r0):
        if sub is not None:
            block = block - sub[r0:r0 + block.shape[0]]
        return block * np.float32(scales[i])

    item.data, _ = integrate_sources(sources, kappa, pre, progress, cancel, label)
    return item.data


# --------------------------------------------------------------------------
# Sky flats: a flat made from the light frames themselves
# --------------------------------------------------------------------------
def _box(a: np.ndarray, r: int) -> np.ndarray:
    if r < 1:
        return a.astype(np.float32)
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)).astype(np.float32)


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(sigma * 0.95)))
    for _ in range(3):
        a = _box(a, r)
    return a


def _minf(a: np.ndarray, r: int) -> np.ndarray:
    h, w = a.shape
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    b = np.min(np.stack([p[:, i:i + w] for i in range(2 * r + 1)]), axis=0)
    p = np.pad(b, ((r, r), (0, 0)), mode="edge")
    return np.min(np.stack([p[i:i + h, :] for i in range(2 * r + 1)]), axis=0)


def _bin(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)).astype(np.float32)


def _upsample(a: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = a.shape
    ys = np.clip((np.arange(h) + 0.5) * ih / h - 0.5, 0, ih - 1)
    xs = np.clip((np.arange(w) + 0.5) * iw / w - 0.5, 0, iw - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
    wy, wx = (ys - y0)[:, None].astype(np.float32), (xs - x0)[None, :].astype(np.float32)
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _tilt_coef(p: np.ndarray):
    """A plane a + b*u + c*v (u, v from -1 to 1 across the frame) fitted robustly to a NaN-masked
    frame. Returns (b/a, c/a): the tilt as a fraction of the level."""
    h, w = p.shape
    yy, xx = np.mgrid[:h, :w]
    u, v = (xx / max(w - 1, 1) * 2 - 1), (yy / max(h - 1, 1) * 2 - 1)
    ok = np.isfinite(p)
    st = max(1, int(np.sqrt(ok.sum() / 20000)))
    sel = np.zeros_like(ok)
    sel[::st, ::st] = True
    sel &= ok
    if sel.sum() < 30:
        return 0.0, 0.0
    A = np.stack([np.ones(int(sel.sum())), u[sel], v[sel]], 1)
    z = p[sel]
    keep = np.ones(z.size, bool)
    for _ in range(3):
        coef, *_r = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        sd = float(np.median(np.abs(res[keep]))) * 1.4826 + 1e-9
        keep = np.abs(res) < 3 * sd
    a = max(float(coef[0]), 1e-9)
    return float(coef[1]) / a, float(coef[2]) / a


def _tilt_plane(shape, b, c):
    h, w = shape
    u = (np.arange(w) / max(w - 1, 1) * 2 - 1)[None, :]
    v = (np.arange(h) / max(h - 1, 1) * 2 - 1)[:, None]
    return (1.0 + b * u + c * v).astype(np.float32)


def _sky_frame(b: np.ndarray):
    """One binned light frame ready for the sky flat: scaled to median 1, its stars hidden (NaN) and
    its light-pollution tilt divided out. Returns (frame, pixel noise) or None. Only star-sized bright
    things are hidden ('top-hat'): anything bigger - vignetting, donuts - is what the flat must keep."""
    med = float(np.median(b))
    if not np.isfinite(med) or med <= 0:
        return None
    p = (b / med).astype(np.float32)
    dif = (p[:, 1:] - p[:, :-1]).ravel()
    noise = float(np.median(np.abs(dif - np.median(dif)))) * 1.4826 / np.sqrt(2) + 1e-6
    sm = _smooth(p, 0.8)
    opened = -_minf(-_minf(sm, 2), 2)
    stars = (sm - opened) > 4.0 * noise * 0.35
    stars = -_minf(-stars.astype(np.float32), 2) > 0.5
    p[stars] = np.nan
    return p, noise, _tilt_coef(p)


def _fill_holes(a: np.ndarray) -> np.ndarray:
    """Fill NaN gaps smoothly from their surroundings (wider and wider, until none are left)."""
    ok = np.isfinite(a)
    if ok.all():
        return a
    out = a.copy()
    val = np.where(ok, a, 0).astype(np.float32)
    wgt = ok.astype(np.float32)
    for sig in (2, 4, 8, 16, 32, 64):
        num, den = _smooth(val, sig), _smooth(wgt, sig)
        fill = ~np.isfinite(out) & (den > 0.02)
        out[fill] = num[fill] / den[fill]
        if np.isfinite(out).all():
            break
    out[~np.isfinite(out)] = 1.0
    return out


def _orientation_groups(arr: np.ndarray):
    """Frames taken on both sides of a meridian flip (camera turned 180° with the telescope) show the
    nebula at two different places on the sensor. Finds the two groups from the frames themselves:
    after the shared pattern (vignetting, dust) is taken out, what is left is the nebula - one way up in
    one group, the other way up in the other. Returns a boolean array (group of each frame), or None when
    the frames do not split clearly into two such groups."""
    n = arr.shape[0]
    if n < 6:
        return None
    h, w = arr.shape[1:]
    b = max(1, int(np.ceil(max(h, w) / 64)))
    hh, ww = h // b * b, w // b * b
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = np.nanmean(arr[:, :hh, :ww].reshape(n, hh // b, b, ww // b, b), axis=(2, 4))
        t = t / np.nanmedian(t.reshape(n, -1), axis=1)[:, None, None]
        r = t - np.nanmedian(t, axis=0)[None]
    r = np.nan_to_num(r).reshape(n, -1)
    r = r - r.mean(axis=1, keepdims=True)
    if not np.isfinite(r).all() or float(np.abs(r).sum()) == 0:
        return None
    u, sv, _vt = np.linalg.svd(r, full_matrices=False)
    explained = float(sv[0] ** 2 / max(float((sv ** 2).sum()), 1e-30))
    sc = u[:, 0] * sv[0]
    g = sc > 0
    if explained < 0.35 or g.sum() < 3 or (~g).sum() < 3:
        return None
    # two clear clusters, not one cloud split down the middle
    a, c = sc[g], sc[~g]
    spread = max(float(a.std()), float(c.std()), 1e-12)
    if (float(a.mean()) - float(c.mean())) < 4 * spread:
        return None
    return g


def build_sky_flat(item: "CalItem", sub: np.ndarray | None = None, progress=None, cancel=None,
                   max_frames: int = 40) -> np.ndarray:
    """A flat made from light frames (best from several targets and nights): each frame is scaled to
    the same brightness, its stars are hidden and its own light-pollution tilt taken out; the lower half
    of every pixel's values (a nebula only ever adds light) then shows what the optics and dust do to an
    even sky - a flat. Raw-sized, median ~1."""
    infos = list(item.infos)
    if len(infos) > max_frames:
        infos = [infos[i] for i in np.linspace(0, len(infos) - 1, max_frames).astype(int)]
    ref = item.ref
    H, W = ref.h, ref.w
    bayer = bool(ref.bayer)
    phases = list(_cfa_phases()) if bayer else [None]
    k = 2 if bayer else 4
    stacks = {ph: [] for ph in phases}
    noises = {ph: [] for ph in phases}
    tilts = {ph: [] for ph in phases}
    no_sky = 0
    for n, inf in enumerate(infos):
        if cancel and cancel():
            raise Cancelled()
        raw = ImageSource(inf.path).read().astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape != (H, W):
            continue
        if sub is not None and sub.shape == raw.shape:
            before = float(np.median(raw[::8, ::8]))
            raw = raw - sub
            if float(np.median(raw[::8, ::8])) < max(3.0, 0.03 * before):
                no_sky += 1                       # the dark/bias took the whole sky away
        for ph in phases:
            plane = raw if ph is None else raw[ph[0]::2, ph[1]::2]
            f = _sky_frame(_bin(plane, k))
            if f is not None:
                stacks[ph].append(f[0])
                noises[ph].append(f[1])
                tilts[ph].append(f[2])
        if progress:
            progress(f"SkyFlat (Mote Buster) - building a flat from your lights: {n + 1}/{len(infos)}", (n + 1) / len(infos) * 0.9)
    item.sky_stats = {"tried": len(infos), "no_sky": no_sky}
    if no_sky > len(infos) / 2:
        raise ValueError("SkyFlat (Mote Buster) can't be made: after the dark/bias is taken away almost no sky "
                         "is left in the light frames - the master dark or bias is as bright as the sky. "
                         "Check that dark (a light leak? the wrong exposure or gain?) or stack without it.")
    out = np.ones((H, W), np.float32)
    import warnings
    for ph in phases:
        if len(stacks[ph]) < 3:
            raise ValueError("Too few usable light frames for a sky flat")
        # light pollution tilts each frame differently; the optics (off-centre vignetting) tilt every
        # frame the same way - so only each frame's difference from the average tilt is taken out
        tb = np.array(tilts[ph])
        mb, mc = float(np.median(tb[:, 0])), float(np.median(tb[:, 1]))
        for i_f, (b_, c_) in enumerate(tb):
            stacks[ph][i_f] = stacks[ph][i_f] / _tilt_plane(stacks[ph][i_f].shape, b_ - mb, c_ - mc)
        arr = np.stack(stacks[ph])

        def lower_half(a):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mid = np.nanmedian(a, axis=0)
                # the lower half of each pixel's values: a nebula on one of the targets, or a star that
                # slipped through, only ever makes a pixel brighter - so it lands in the upper half
                return np.nanmean(np.where(a <= mid[None], a, np.nan), axis=0)
        groups = _orientation_groups(arr)
        if groups is None:
            low = lower_half(arr)
        else:
            # both sides of a meridian flip: where one side has nebula on a pixel, the other side (turned
            # 180°) usually has plain sky there - take the darker of the two sides, pixel by pixel
            la, lb = lower_half(arr[groups]), lower_half(arr[~groups])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low = np.fmin(la, lb)
            item.sky_stats["flip_groups"] = (int(groups.sum()), int((~groups).sum()))
        med = low + 0.798 * float(np.mean(noises[ph]))       # mean of a lower half sits 0.8 sigma low
        valid = np.isfinite(arr).sum(axis=0)
        med[valid < max(3, int(0.2 * len(arr)))] = np.nan
        holes = float(np.mean(~np.isfinite(med)))
        med = _fill_holes(med)
        rough = _pixel_noise(med) / max(float(np.median(med)), 1e-6)    # the flat's own grain
        med = _smooth(med, 1.0)
        med = med / max(float(np.median(med)), 1e-6)
        big = _smooth(med, max(3.0, min(med.shape) / 40))
        st_ = item.sky_stats
        st_["used"] = min(st_.get("used", 10 ** 9), len(arr))
        st_["grain"] = max(st_.get("grain", 0.0), rough)
        st_["holes"] = max(st_.get("holes", 0.0), holes)
        st_["low"] = min(st_.get("low", 9.0), float(np.percentile(big, 0.5)))
        st_["high"] = max(st_.get("high", 0.0), float(np.percentile(big, 99.5)))
        if ph is None:
            out = _upsample(med, H, W)
        else:
            ph_h, ph_w = out[ph[0]::2, ph[1]::2].shape
            out[ph[0]::2, ph[1]::2] = _upsample(med, ph_h, ph_w)
    if progress:
        progress("SkyFlat ready", 1.0)
    return np.clip(out, 0.2, 5.0).astype(np.float32)


def sky_flat_problems(item: "CalItem") -> list:
    """Reasons a SkyFlat (Mote Buster) may do more harm than good ([] when it looks fine)."""
    s = item.sky_stats or {}
    out = []
    if not s or item.data is None:
        return ["it could not be made"]
    if s.get("used", 0) < SKY_FLAT_MIN:
        out.append(f"only {s.get('used', 0)} light frames were usable (at least {SKY_FLAT_MIN} are needed "
                   "for a smooth flat)")
    if s.get("grain", 0) > 0.02:
        out.append(f"it is grainy ({100 * s['grain']:.1f}% noise) - it would add noise to every frame")
    if s.get("low", 1) < 0.4 or s.get("high", 1) > 1.8:
        out.append(f"its brightness range is unusual for vignetting ({s.get('low', 1):.2f} - {s.get('high', 1):.2f})"
                   " - nebulosity or a light-pollution gradient may have got into it")
    if s.get("holes", 0) > 0.05:
        out.append(f"{100 * s['holes']:.0f}% of it had to be filled in (always covered by stars or nebula)")
    return out


def hot_pixels(dark: np.ndarray, bias: np.ndarray | None, limit=0.002):
    """Hot pixel coordinates from a master dark (top outliers, at most 0.2%)."""
    d = dark if bias is None else dark - bias
    s = d[::4, ::4]
    med = float(np.median(s))
    sig = float(np.median(np.abs(s - med)) * 1.4826) + 1e-3
    thr = med + 8 * sig
    ys, xs = np.nonzero(d > thr)
    if ys.size > limit * d.size:
        vals = d[ys, xs]
        keep = np.argsort(-vals)[: int(limit * d.size)]
        ys, xs = ys[keep], xs[keep]
    return ys, xs


ADU_FULL = 65535.0


def save_master_file(path_base: Path, data: np.ndarray, header: list, formats=("fits",), image_type=None):
    """Write 32-bit float FITS and/or XISF in PixInsight's 0..1 convention
    (1.0 = 65535 ADU). PIXRANGE records the factor so values can be restored."""
    path_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(data, np.float32) / np.float32(ADU_FULL), 0.0, 1.0)
    header = [h for h in header if h[0] != "PIXRANGE"] + [("PIXRANGE", ADU_FULL, "values x this = ADU (0..1 range)")]
    written = []
    if "fits" in formats:
        p = path_base.with_name(path_base.name + ".fits")    # names may contain dots ("20.0s")
        core.write_fits(p, data, header, bitpix=-32)
        written.append(p)
    if "xisf" in formats:
        p = path_base.with_name(path_base.name + ".xisf")
        xisf.write_xisf(p, data, [(k, v, c) for k, v, c in header if k not in ("HISTORY", "COMMENT")],
                        image_type, bounds=(0.0, 1.0))
        written.append(p)
    return written


def save_calibration_master(item: CalItem, cal_root, rig: str, formats=("fits",)) -> list:
    """Save a master built from subframes into the calibration library."""
    i = item.ref
    cam = short_camera(i.camera)
    parts = {"bias": "MasterBias", "dark": "MasterDark", "darkflat": "MasterDarkFlat",
             "flat": "MasterFlat"}[item.kind]
    name = [parts]
    if item.kind == "flat":
        name.append(safe_name(i.filter))
    name.append(cam)
    if item.kind in ("dark", "darkflat") and i.exptime is not None:
        name.append(f"{i.exptime:g}s")
    if i.gain is not None:
        name.append(f"G{i.gain:g}")
    if i.offset is not None and item.kind != "flat":
        name.append(f"O{i.offset:g}")
    if i.temp is not None and item.kind != "flat":
        name.append(f"{i.temp:g}C")
    if item.kind == "flat" and i.night:
        name.append(i.night)
    if i.x0 or i.y0:
        name.append(f"{i.w}x{i.h}")
    folder = library_dir(cal_root, rig, i.camera, item.kind, i.filter)
    header = [("IMAGETYP", f"Master {KIND_LABEL[item.kind]}", ""), ("INSTRUME", i.camera, ""),
              ("TELESCOP", rig, "rig name"), ("FILTER", i.filter if item.kind == "flat" else None, ""),
              ("EXPTIME", i.exptime, "[s] subframe exposure"), ("GAIN", i.gain, ""), ("OFFSET", i.offset, ""),
              ("CCD-TEMP", i.temp, "[C]"), ("XBINNING", i.binning, ""), ("YBINNING", i.binning, ""),
              ("BAYERPAT", i.bayer, ""), ("XORGSUBF", i.x0, ""), ("YORGSUBF", i.y0, ""),
              ("FOCALLEN", i.focal, ""), ("ROWORDER", i.roworder, ""), ("DATE-OBS", i.date_obs, "first subframe"),
              ("NCOMBINE", len(item.infos), "subframes combined"),
              ("XPIXSZ", _num(ImageSource(i.path).look.get("XPIXSZ")), "[um]"),
              ("SWCREATE", "SubZeroAstro", ""),
              ("HISTORY", f"Sigma-clipped mean of {len(item.infos)} subframes", "")]
    header += [("HISTORY", f"  {x.path.name}", "") for x in item.infos[:200]]
    base = folder / "_".join(name)
    base = _unique_base(base, formats)
    item.saved = save_master_file(base, item.data, header, formats, f"Master{KIND_LABEL[item.kind].replace('-', '')}")
    return item.saved


def _unique_base(base: Path, formats) -> Path:
    def exists(b):
        return any(b.with_name(f"{b.name}.{f}").exists() for f in formats)
    if not exists(base):
        return base
    k = 2
    while exists(base.with_name(f"{base.name}_{k}")):
        k += 1
    return base.with_name(f"{base.name}_{k}")


# ==========================================================================
# Light calibration and debayering
# ==========================================================================

class Calibrator:
    def __init__(self, light: ImgInfo, plan: CalPlan):
        self.L = light
        self.sub = None          # what to subtract (dark, scaled dark, or bias)
        self.flat = None
        self.hot = None
        bias = crop_to(plan.bias.data, plan.bias, light) if plan.bias and plan.bias.data is not None else None
        if plan.dark is not None and plan.dark.data is not None:
            dark = crop_to(plan.dark.data, plan.dark, light)
            if plan.dark_mode == "scaled" and bias is not None:
                k = (light.exptime or 1) / max(plan.dark.exptime or 1, 1e-6)
                self.sub = bias + (dark - bias) * np.float32(k)
            else:
                self.sub = dark
            ys, xs = hot_pixels(dark, bias)
            self.hot = (ys, xs)
        elif bias is not None:
            self.sub = bias
        if plan.flat is not None and plan.flat.data is not None:
            self.flat = normalize_flat(crop_to(plan.flat.data, plan.flat, light), bool(light.bayer))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        x = raw.astype(np.float32, copy=True)
        if self.sub is not None:
            x -= self.sub
        if self.flat is not None:
            x /= self.flat
        if self.hot is not None and self.hot[0].size:
            fix_pixels(x, *self.hot, step=2 if self.L.bayer else 1)
        return x


def _pixel_noise(a: np.ndarray) -> float:
    d = (a[:, 1:] - a[:, :-1]).ravel()
    return float(np.median(np.abs(d - np.median(d)))) * 1.4826 / np.sqrt(2) + 1e-6


def master_unevenness(m: np.ndarray):
    """(large-scale spread, pixel noise) of a dark or bias master. A real dark or bias is flat apart
    from noise (and some amp glow at the edge); one with a picture in it - brighter in the middle like
    vignetting - had light leaking in."""
    st_ = max(1, min(m.shape) // 500)                    # ~500 px on the short side
    s = m[::st_, ::st_].astype(np.float32)
    k = max(2, min(s.shape) // 24)                       # ~24 blocks on the short side
    g = _smooth(_bin(s, k), 1.5)
    return float(np.percentile(g, 95) - np.percentile(g, 5)), _pixel_noise(s)


def calibration_check(light: ImgInfo, cal: "Calibrator") -> list:
    """Log lines checking the dark / bias subtraction on one light frame. With narrowband
    filters there may be only a few ADU of sky above the camera offset, so a dark or bias that does
    not match the lights (other scale, gain / offset, temperature, or light leaking in) leaves a
    'pedestal' or a pattern that the flat and the stretch then turn into bright bands and borders.
    Says so plainly when the numbers don't add up."""
    raw = ImageSource(light.path).read()
    raw = (raw[0] if raw.ndim == 3 else raw)[::8, ::8].astype(np.float32)
    sub = cal.sub[::8, ::8].astype(np.float32)
    if sub.shape != raw.shape:
        return []
    raw_med, sub_med = float(np.median(raw)), float(np.median(sub))
    sky = float(np.median(raw - sub))
    noise = _pixel_noise(raw)
    what = "dark" if cal.hot is not None else "bias"
    out = [f"Calibration check: light background {raw_med:.0f} ADU - {what} {sub_med:.0f} ADU = {sky:.1f} ADU of sky "
           f"(pixel noise {noise:.1f} ADU)"]
    spread, mnoise = master_unevenness(cal.sub)
    if raw_med > 100 and sub_med < 1.0:
        out.append(f"WARNING: the {what} master is practically zero ({sub_med:.3f}) while the lights are at "
                   f"{raw_med:.0f} ADU - it is on a different scale (0..1) or empty, so the camera offset stays in "
                   "the image and the flat will over-correct the edges (bright borders). "
                   f"Save the {what} in camera units, or use another one.")
    elif sub_med > raw_med + max(5.0, 0.5 * noise):
        out.append(f"WARNING: the {what} master is BRIGHTER than the lights ({sub_med:.0f} vs {raw_med:.0f} ADU), "
                   "which a real one never is - subtracting it leaves the picture below zero with false bright and "
                   f"dark areas. Check that the {what} has the same gain, offset, temperature and exposure as the "
                   "lights, and that no light got in while taking it.")
    elif sky < 0.5 * noise and raw_med > 100:
        out.append("Note: very little sky signal above the camera offset (typical for narrowband) - the result "
                   "depends strongly on matching darks, bias and flats.")
    if spread > max(30.0, 6 * mnoise):
        out.append(f"WARNING: the {what} master is uneven like a picture (it varies by {spread:.0f} ADU across the "
                   f"frame; a real {what} is flat to within a few ADU) - light probably leaked in while it was taken "
                   "(lens cap, filter wheel, focuser or room light). Its pattern ends up in your picture as bright "
                   f"and dark areas. Take new {what}s with the scope fully covered, or remove this {what} from the "
                   "calibration list and stack without it.")
    return out


def fix_pixels(img, ys, xs, step=1):
    """Replace pixels with the mean of their same-colour neighbours."""
    h, w = img.shape
    acc = np.zeros(ys.size, np.float32)
    cnt = np.zeros(ys.size, np.float32)
    for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        acc[ok] += img[yy[ok], xx[ok]]
        cnt[ok] += 1
    img[ys, xs] = acc / np.maximum(cnt, 1)


def _pattern(bayer: str, look_x0=0, look_y0=0):
    p = (bayer or "RGGB").upper()
    grid = [[p[0], p[1]], [p[2], p[3]]]
    return grid


def _conv3(img, k):
    """3x3 convolution with edge replication (k is a 3x3 list)."""
    p = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img)
    h, w = img.shape
    for dy in range(3):
        for dx in range(3):
            if k[dy][dx]:
                out += np.float32(k[dy][dx]) * p[dy:dy + h, dx:dx + w]
    return out


def debayer_channel(cfa: np.ndarray, bayer: str, ch: str, half=False) -> np.ndarray:
    """One colour channel ("R", "G" or "B") from a CFA image.
    half=True: 2x2 superpixel (half size); else bilinear at full size."""
    grid = _pattern(bayer)
    if half:
        h2, w2 = cfa.shape[0] // 2 * 2, cfa.shape[1] // 2 * 2
        parts = [cfa[i:h2:2, j:w2:2] for i in (0, 1) for j in (0, 1) if grid[i][j] == ch]
        return (sum(parts) / len(parts)).astype(np.float32)
    mask = np.zeros(cfa.shape, np.float32)
    for i in (0, 1):
        for j in (0, 1):
            if grid[i][j] == ch:
                mask[i::2, j::2] = 1
    k = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    num = _conv3(cfa * mask, k)
    den = _conv3(mask, k)
    out = num / np.maximum(den, 1e-6)
    return np.where(mask > 0, cfa, out).astype(np.float32)


def debayer_bilinear(cfa: np.ndarray, bayer: str) -> list:
    """Bilinear demosaic of a CFA image into [R, G, B] full-size planes.
    Each missing value is the mean of the nearest pixels of that colour."""
    grid = _pattern(bayer)
    c = cfa.astype(np.float32, copy=False)
    P = np.pad(c, 1, mode="edge")
    N, S_, W, E = P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]
    NW, NE, SW, SE = P[:-2, :-2], P[:-2, 2:], P[2:, :-2], P[2:, 2:]
    out = {ch: np.empty_like(c) for ch in "RGB"}
    for i in (0, 1):
        for j in (0, 1):
            col = grid[i][j]
            sl = (slice(i, None, 2), slice(j, None, 2))
            out[col][sl] = c[sl]
            if col == "G":
                out[grid[i][1 - j]][sl] = (W[sl] + E[sl]) * 0.5       # colour beside it in the row
                out[grid[1 - i][j]][sl] = (N[sl] + S_[sl]) * 0.5      # colour above / below
            else:
                out["G"][sl] = (N[sl] + S_[sl] + W[sl] + E[sl]) * 0.25
                out["B" if col == "R" else "R"][sl] = (NW[sl] + NE[sl] + SW[sl] + SE[sl]) * 0.25
    return [out["R"], out["G"], out["B"]]


def despike(a: np.ndarray, ratio: float = 0.2, k: float = 8.0, replace: str = "max") -> tuple[np.ndarray, int]:
    """Hot pixels and other single-pixel spikes replaced by their brightest neighbour, for finding
    stars. A real star always spills light into the next pixel (even a sharp one keeps 30-70% of its
    peak there); a hot pixel has none. Without this, sensors with many hot pixels (calibrated
    frames from remote observatories) give star lists that are mostly sensor defects - which sit
    in the same place in every frame and can make the matcher line up the sensor instead of the
    sky. Returns (cleaned, number replaced). Mono data only (in a Bayer mosaic the neighbours are
    other colours)."""
    h, w = a.shape
    if h < 3 or w < 3:
        return a, 0
    p = np.pad(a, 1, mode="edge")
    nb = None
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            v = p[dy:dy + h, dx:dx + w]
            nb = v.copy() if nb is None else np.maximum(nb, v, out=nb)
    sub = a[::7, ::7]
    med = float(np.median(sub))
    sig = float(np.median(np.abs(sub - med)) * 1.4826) + 1e-6
    peak = a - med
    spike = (peak > k * sig) & ((nb - med) < ratio * peak)
    n = int(spike.sum())
    if n == 0:
        return a, 0
    out = a.copy()
    if replace == "median":                  # for the picture itself: the typical neighbour, not the brightest
        ys, xs = np.nonzero(spike)
        neigh = np.stack([p[ys + dy, xs + dx] for dy in (0, 1, 2) for dx in (0, 1, 2) if (dy, dx) != (1, 1)])
        out[ys, xs] = np.median(neigh, axis=0)
    else:
        out[spike] = nb[spike]
    return out, n


def cosmetic(img: np.ndarray, bayer: bool = False) -> tuple[np.ndarray, int]:
    """Cosmetic correction (like PixInsight's CosmeticCorrection): hot pixels and cosmic-ray hits -
    single pixels far brighter than all their neighbours - replaced by the median of the neighbours.
    Without dithering, stacking cannot remove hot pixels (they sit on the same pixel in every frame),
    and they end up as coloured specks once the filters are combined. A Bayer mosaic is corrected
    one colour at a time. Returns (image, pixels replaced)."""
    if img.ndim != 2:
        return img, 0
    limit = COSMETIC_MAX_SHARE * img.size
    if not bayer:
        out, n = despike(img.astype(np.float32, copy=False), replace="median")
        return (img, -n) if n > limit else (out, n)
    out = img.astype(np.float32, copy=True)
    total = 0
    for dy in (0, 1):
        for dx in (0, 1):
            sub, n = despike(out[dy::2, dx::2], replace="median")
            out[dy::2, dx::2] = sub
            total += n
    return (img, -total) if total > limit else (out, total)


# More 'hot pixels' than this share of the frame means the stars themselves are that sharp (binned or
# undersampled frames, ~1 px stars): hot pixels can't be told apart from stars, so nothing is changed.
# A typical sensor has well under 0.05% hot pixels.
COSMETIC_MAX_SHARE = 0.002


def luminance(cal: np.ndarray, bayer: bool) -> tuple[np.ndarray, int]:
    """Image used for finding stars, and its scale factor to full resolution."""
    if not bayer and cal.ndim == 2:
        cal = despike(cal.astype(np.float32, copy=False))[0]
    h2, w2 = cal.shape[0] // 2 * 2, cal.shape[1] // 2 * 2
    return cal[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)), 2


# ==========================================================================
# Star alignment
# ==========================================================================

def _median3(a: np.ndarray) -> np.ndarray:
    """3x3 median: removes hot pixels and single-pixel noise spikes, keeps stars."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return np.median(np.stack([p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]),
                     axis=0).astype(np.float32)


def _gauss1(a: np.ndarray) -> np.ndarray:
    """Separable Gaussian blur, sigma 1 px (a matched filter for small stars)."""
    k = np.exp(-np.arange(-3, 4) ** 2 / 2.0).astype(np.float32)
    k /= k.sum()
    h, w = a.shape
    p = np.pad(a, ((0, 0), (3, 3)), mode="edge")
    b = sum(k[i] * p[:, i:i + w] for i in range(7))
    p = np.pad(b, ((3, 3), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + h, :] for i in range(7)).astype(np.float32)


def find_stars_robust(img: np.ndarray, factor: int, max_stars=150, k=5.0):
    """Star finder for faint, noisy, uncalibrated subs (smart telescopes: 10-20 s at
    high gain, hot pixels): median filter, Gaussian matched filter, and centroids
    from the upper half of each star so noise doesn't pull them to pixel centres."""
    clean = _median3(img)
    bg, _sky, _s = M._background(clean)
    sm = _gauss1(clean - bg)
    s_ = sm[::3, ::3]
    med = float(np.median(s_))
    sig = float(np.median(np.abs(s_ - med)) * 1.4826) + 1e-6
    h, w = sm.shape
    m = 8
    if h <= 2 * m + 2 or w <= 2 * m + 2:
        return np.empty((0, 2)), np.empty(0)
    c = sm[m:h - m, m:w - m]
    cand = c > med + k * sig
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                cand &= c >= sm[m + dy:h - m + dy, m + dx:w - m + dx]
    ys, xs = np.nonzero(cand)
    ys += m
    xs += m
    if ys.size == 0:
        return np.empty((0, 2)), np.empty(0)
    v = sm[ys, xs]
    order = np.argsort(-v)
    r = 3
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    pts, pk, taken = [], [], set()
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        cell = (y >> 2, x >> 2)
        if any((cell[0] + a, cell[1] + b) in taken for a in (-1, 0, 1) for b in (-1, 0, 1)):
            continue                                  # merge maxima closer than ~4 px
        cut = np.clip(sm[y - r:y + r + 1, x - r:x + r + 1] - 0.5 * sm[y, x], 0, None)
        sv = cut.sum()
        if sv <= 0:
            continue
        taken.add(cell)
        cy, cx = (cut * yy).sum() / sv, (cut * xx).sum() / sv
        pts.append(((x + cx) * factor + (factor - 1) / 2, (y + cy) * factor + (factor - 1) / 2))
        pk.append(float(v[i]))
        if len(pts) >= max_stars:
            break
    return np.array(pts, dtype=np.float64), np.array(pk, dtype=np.float64)


ROBUST_MIN_STARS = 60          # fewer stars than this in the reference -> use the robust finder


def find_stars(img: np.ndarray, factor: int, max_stars=150, robust=False):
    """Centroids (x, y) in full-resolution pixel coordinates and peak values,
    brightest first."""
    if robust:
        return find_stars_robust(img, factor, max_stars)
    bg, _sky, sigma = M._background(img)
    resid = img - bg
    ys, xs, peaks = M._find_stars(resid, 6)
    if ys.size == 0:
        return np.empty((0, 2)), np.empty(0)
    top = float(np.percentile(img, 99.99))
    order = np.argsort(-peaks)
    pts, pk = [], []
    r = 3
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for idx in order:
        y, x = ys[idx], xs[idx]
        if img[y, x] >= top and len(pts) < 5:
            continue                                   # skip the very brightest (often saturated)
        cut = np.clip(resid[y - r:y + r + 1, x - r:x + r + 1], 0, None)
        s = cut.sum()
        if s <= 0:
            continue
        cy, cx = (cut * yy).sum() / s, (cut * xx).sum() / s
        pts.append(((x + cx) * factor + (factor - 1) / 2, (y + cy) * factor + (factor - 1) / 2))
        pk.append(resid[y, x])
        if len(pts) >= max_stars:
            break
    return np.array(pts, dtype=np.float64), np.array(pk, dtype=np.float64)


def _triangles(pts, k=5):
    """Triangle invariants (two side ratios) and vertex indices ordered by
    the length of the opposite side."""
    n = len(pts)
    if n < 3:
        return np.empty((0, 2)), np.empty((0, 3), int)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    inv, tri = [], []
    seen = set()
    for i in range(n):
        nb = np.argsort(d[i])[1:k + 1]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                t = tuple(sorted((i, int(nb[a]), int(nb[b]))))
                if t in seen:
                    continue
                seen.add(t)
                p, q, r = t
                opp = {p: d[q, r], q: d[p, r], r: d[p, q]}
                order = sorted(t, key=lambda v: opp[v])
                s0, s1, s2 = sorted(opp.values())
                if s0 <= 1e-6:
                    continue
                inv.append((s2 / s1, s1 / s0))
                tri.append(order)
    return np.array(inv), np.array(tri, dtype=int)


def _fit_similarity(src, dst):
    """Least-squares similarity transform (rotation, scale, shift): dst ≈ A @ src + t."""
    n = len(src)
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    A[0::2] = np.c_[src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)]
    A[1::2] = np.c_[src[:, 1], src[:, 0], np.zeros(n), np.ones(n)]
    b[0::2], b[1::2] = dst[:, 0], dst[:, 1]
    (a, c, tx, ty), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[a, -c, tx], [c, a, ty]])


def _fit_affine(src, dst):
    n = len(src)
    X = np.c_[src, np.ones(n)]
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return sol.T


def _apply(T, pts):
    return pts @ T[:, :2].T + T[:, 2]


def _match_once(ref_pts, tgt_pts, tol, n_tri, mirrored):
    """One attempt: triangles from the n_tri brightest stars; mirrored=True also allows a
    left-right mirror image (a frame saved flipped, or with the other row order)."""
    M = np.array([[-1.0, 0, 0], [0, 1.0, 0]]) if mirrored else None
    tp = tgt_pts.copy()
    if mirrored:
        tp[:, 0] = -tp[:, 0]
    R, T_ = ref_pts[:n_tri], tp[:n_tri]
    inv_r, tri_r = _triangles(R)
    inv_t, tri_t = _triangles(T_)
    if len(inv_r) == 0 or len(inv_t) == 0:
        raise ValueError("too few stars")
    dists = np.sqrt(((inv_t[:, None, :] - inv_r[None, :, :]) ** 2).sum(-1))
    best = np.argmin(dists, axis=1)
    good = np.nonzero(dists[np.arange(len(best)), best] < 0.03)[0]
    if good.size == 0:
        raise ValueError("no matching star patterns")
    good = good[np.argsort(dists[good, best[good]])][:400]

    def count_inliers(T):
        p = _apply(T, tgt_pts)
        d = np.sqrt(((p[:, None, :] - ref_pts[None, :, :]) ** 2).sum(-1))
        j = np.argmin(d, axis=1)
        dmin = d[np.arange(len(j)), j]
        ok = dmin < tol
        return ok, j

    best_T, best_n = None, 0
    for gi in good:
        src = T_[tri_t[gi]]
        dst = R[tri_r[best[gi]]]
        Tm = _fit_similarity(src, dst)
        sc = math.hypot(Tm[0, 0], Tm[1, 0])
        if not 0.8 < sc < 1.25:
            continue
        T = _compose(Tm, M) if mirrored else Tm          # in the frame's own coordinates
        ok, _ = count_inliers(T)
        n = int(ok.sum())
        if n > best_n:
            best_T, best_n = T, n
            if n > 0.8 * min(len(tgt_pts), len(ref_pts)):
                break
    if best_T is None or best_n < 6:
        raise ValueError("could not match the star field")
    T = best_T
    for _ in range(3):
        ok, j = count_inliers(T)
        src, dst = tgt_pts[ok], ref_pts[j[ok]]
        if len(src) < 6:
            break
        # affine keeps a mirror image; a similarity fit would turn it back
        T = _fit_affine(src, dst) if (len(src) >= 20 or mirrored) else _fit_similarity(src, dst)
    ok, j = count_inliers(T)
    res = _apply(T, tgt_pts[ok]) - ref_pts[j[ok]]
    rms = float(np.sqrt((res ** 2).sum(1).mean())) if ok.any() else 99.0
    return T, int(ok.sum()), rms, ok, j


def match_frames(ref_pts, tgt_pts, tol=3.0):
    """Transform T (2x3) mapping target coordinates to reference coordinates,
    plus (inliers, rms, ok, j). Tries the 40 brightest stars, then 80, then the frame
    mirrored (frames saved flipped by other capture software / settings).
    Raises ValueError if the frames cannot be matched."""
    if len(ref_pts) < 6 or len(tgt_pts) < 6:
        raise ValueError("too few stars")
    last = None
    for n_tri, mirrored in ((40, False), (80, False), (40, True), (80, True)):
        if n_tri > 40 and min(len(ref_pts), len(tgt_pts)) <= 40:
            continue
        try:
            T, n, rms, ok, j = _match_once(ref_pts, tgt_pts, tol, n_tri, mirrored)
            few = min(len(ref_pts), len(tgt_pts))
            # a real match shares a good part of its stars and fits tightly; a mirror image needs
            # stronger evidence still (never accept a handful of chance coincidences)
            need = max(10, int(0.2 * few)) if mirrored else max(8, int((0.15 if n_tri == 40 else 0.1) * few))
            if n >= need and rms <= 0.6 * tol:
                return T, n, rms, ok, j
            last = ValueError(f"too few stars in common ({n} of {few})")
        except ValueError as e:
            last = e
    raise last or ValueError("could not match the star field")


# --------------------------------------------------------------------------
# Plate solutions as a guide: many frames carry one (PinPoint at remote observatories writes it into
# every frame; our masters copy the reference frame's; ASTAP adds one on page 3). Where it says each
# pixel points on the sky, two frames can be lined up without any star pattern - the stars then only
# fine-tune it. Used when star matching fails, and to catch a star match that is plainly wrong.
# --------------------------------------------------------------------------

def wcs_of(look: dict):
    """Linear (TAN) plate solution from a header, or None."""
    try:
        ct = str(look.get("CTYPE1", "RA---TAN") or "RA---TAN").upper()
        if "RA" not in ct:
            return None
        crval = (float(look["CRVAL1"]), float(look["CRVAL2"]))
        crpix = (float(look["CRPIX1"]), float(look["CRPIX2"]))
        if look.get("CD1_1") not in (None, ""):
            cd = np.array([[float(look["CD1_1"]), float(look.get("CD1_2") or 0.0)],
                           [float(look.get("CD2_1") or 0.0), float(look["CD2_2"])]])
        else:
            c1, c2 = float(look["CDELT1"]), float(look["CDELT2"])
            r = np.radians(float(look.get("CROTA2") or 0.0))
            cd = np.array([[c1 * np.cos(r), -c2 * np.sin(r)], [c1 * np.sin(r), c2 * np.cos(r)]])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(cd)) or abs(np.linalg.det(cd)) < 1e-14:
        return None
    return {"crval": np.radians(crval), "crpix": np.array(crpix), "cd": cd}


def _wcs_sky(w, x, y):
    """0-based pixel (x, y) -> (ra, dec) radians, gnomonic projection."""
    xi, eta = np.radians(w["cd"] @ np.vstack([x + 1 - w["crpix"][0], y + 1 - w["crpix"][1]]))
    a0, d0 = w["crval"]
    den = np.cos(d0) - eta * np.sin(d0)
    return a0 + np.arctan2(xi, den), np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, den))


def _wcs_pix(w, ra, dec):
    a0, d0 = w["crval"]
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - a0)
    xi = np.cos(dec) * np.sin(ra - a0) / cosc
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - a0)) / cosc
    p = np.linalg.solve(w["cd"], np.degrees(np.vstack([xi, eta])))
    return p[0] + w["crpix"][0] - 1, p[1] + w["crpix"][1] - 1


def wcs_transform(ref_look: dict, look: dict, shape):
    """T (2x3) mapping this frame's pixels to the reference frame's, from the two plate solutions;
    None when either has none."""
    wr, wi = wcs_of(ref_look or {}), wcs_of(look or {})
    if wr is None or wi is None:
        return None
    h, w = shape[-2], shape[-1]
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 5), np.linspace(0, h - 1, 5))
    x, y = gx.ravel(), gy.ravel()
    ra, dec = _wcs_sky(wi, x, y)
    xr, yr = _wcs_pix(wr, ra, dec)
    if not (np.all(np.isfinite(xr)) and np.all(np.isfinite(yr))):
        return None
    return _fit_affine(np.c_[x, y], np.c_[xr, yr])


def _corner_gap(T1, T2, shape) -> float:
    h, w = shape[-2], shape[-1]
    c = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]], dtype=np.float64)
    return float(np.max(np.hypot(*(_apply(T1, c) - _apply(T2, c)).T)))


def match_guided(ref_pts, tgt_pts, T0, tols=(15.0, 8.0, 4.0, 2.5, 1.8)):
    """Star match that starts from a known transform (a plate solution): pair each star with the
    nearest reference star through it, refit, tighten. Same results as match_frames."""
    ref_pts, tgt_pts = np.asarray(ref_pts, float), np.asarray(tgt_pts, float)
    if len(ref_pts) < 6 or len(tgt_pts) < 6:
        raise ValueError("too few stars")
    T = np.asarray(T0, float)
    ok = np.zeros(len(tgt_pts), bool)
    j = np.zeros(len(tgt_pts), int)
    for tol in tols:
        p = _apply(T, tgt_pts)
        d = np.hypot(p[:, None, 0] - ref_pts[None, :, 0], p[:, None, 1] - ref_pts[None, :, 1])
        j = d.argmin(1)
        ok = d[np.arange(len(p)), j] < tol
        if ok.sum() < 6:
            raise ValueError(f"the plate solutions point the same way, but too few stars agree ({int(ok.sum())})")
        T = _fit_affine(tgt_pts[ok], ref_pts[j[ok]])
    res = np.hypot(*(_apply(T, tgt_pts[ok]) - ref_pts[j[ok]]).T)
    rms = float(np.sqrt(np.mean(res ** 2)))
    n = int(ok.sum())
    if n < max(8, int(0.06 * min(len(ref_pts), len(tgt_pts)))) or rms > 1.2:
        raise ValueError(f"too few stars agree with the plate solutions ({n}, {rms:.2f} px)")
    return T, n, rms, ok, j


def match_frames_wcs(ref_pts, tgt_pts, ref_look=None, look=None, shape=None, tol=3.0):
    """match_frames, checked against (and rescued by) the plate solutions when both frames have one.
    Returns (T, inliers, rms, ok, j, how) - how is "stars", or "plate solution" when the plate
    solutions found it (star matching failed, or agreed with a false pattern)."""
    Tw = wcs_transform(ref_look, look, shape) if shape is not None else None
    try:
        T, n, rms, ok, j = match_frames(ref_pts, tgt_pts, tol=tol)
        err = None
    except ValueError as e:
        T, err = None, e
    if Tw is None:
        if T is None:
            raise err
        return T, n, rms, ok, j, "stars"
    if T is not None and _corner_gap(T, Tw, shape) <= 15.0:
        return T, n, rms, ok, j, "stars"
    try:
        Tg, ng, rmsg, okg, jg = match_guided(ref_pts, tgt_pts, Tw)
        return Tg, ng, rmsg, okg, jg, "plate solution"
    except ValueError as e2:
        if T is not None:
            return T, n, rms, ok, j, "stars"
        raise ValueError(f"{err}; {e2}")


# --------------------------------------------------------------------------
# Distortion-correcting registration: images from different cameras / telescopes don't line up
# with a shift-rotate-scale alone - the optics bend the field a little differently, most at the
# edges. A smooth polynomial fitted to many matched stars takes that out.
# --------------------------------------------------------------------------

def _poly_terms(order):
    return [(i, j) for n in range(order + 1) for i in range(n + 1) for j in [n - i]]


def _poly_design(x, y, order, norm):
    cx, cy, sc = norm
    u, v = (x - cx) / sc, (y - cy) / sc
    return np.stack([u ** i * v ** j for i, j in _poly_terms(order)], axis=-1)


def fit_poly(src, dst, order, shape):
    """dst ≈ P(src): a polynomial of the given order (1 = affine), coordinates normalised to the
    image size so higher orders stay well-behaved."""
    h, w = shape
    norm = (w / 2.0, h / 2.0, max(w, h) / 2.0)
    D = _poly_design(src[:, 0], src[:, 1], order, norm)
    cxy, *_ = np.linalg.lstsq(D, dst, rcond=None)
    return {"order": order, "norm": norm, "coef": cxy}


def poly_apply(model, pts):
    D = _poly_design(pts[:, 0], pts[:, 1], model["order"], model["norm"])
    return D @ model["coef"]


def warp_poly(planes, model, out_shape, rows=256) -> list:
    """Resample so each output (reference) pixel takes its value from where model(ref xy) says it
    is in the image (bilinear; NaN outside)."""
    h, w = out_shape
    H, W = planes[0].shape
    flats = [np.ascontiguousarray(p, dtype=np.float32).ravel() for p in planes]
    outs = [np.empty((h, w), np.float32) for _ in planes]
    xs = np.arange(w, dtype=np.float64)
    for r0 in range(0, h, rows):
        r1 = min(r0 + rows, h)
        yy, xx = np.meshgrid(np.arange(r0, r1, dtype=np.float64), xs, indexing="ij")
        D = _poly_design(xx.ravel(), yy.ravel(), model["order"], model["norm"])
        sxy = D @ model["coef"]
        sx = sxy[:, 0].reshape(r1 - r0, w).astype(np.float32)
        sy = sxy[:, 1].reshape(r1 - r0, w).astype(np.float32)
        x0 = np.floor(sx)
        y0 = np.floor(sy)
        fx, fy = sx - x0, sy - y0
        x0 = x0.astype(np.int32)
        y0 = y0.astype(np.int32)
        valid = (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1)
        idx = np.clip(y0, 0, H - 2) * W + np.clip(x0, 0, W - 2)
        w11 = fx * fy
        w10 = fy - w11
        w01 = fx - w11
        w00 = 1 - fx - fy + w11
        for f, o in zip(flats, outs):
            v = f[idx] * w00 + f[idx + 1] * w01 + f[idx + W] * w10 + f[idx + W + 1] * w11
            o[r0:r1] = np.where(valid, v, np.nan)
    return outs


def distortion_model(ref_all, frame_all, T, T_ref_out, C_plane, out_shape):
    """For stacking: a polynomial from output pixels to the frame's plane pixels, fitted to the stars
    - or None when the plain (affine) alignment T (frame -> reference) is already good at the edges.
    Stars are paired again through T with a wider tolerance, so edge stars that a plain alignment
    puts a few pixels off still count. T_ref_out: reference -> output; C_plane: full-resolution ->
    plane coordinates (half-size colour), or None."""
    ref_all = np.asarray(ref_all, np.float64)
    frame_all = np.asarray(frame_all, np.float64)
    if len(ref_all) < 20 or len(frame_all) < 20:
        return None
    pred = _apply(T, frame_all)
    d = np.sqrt(((pred[:, None, :] - ref_all[None, :, :]) ** 2).sum(-1))
    k = np.argmin(d, axis=1)
    pair = d[np.arange(len(k)), k] < 5.0
    # one frame star per reference star
    _u, first = np.unique(k[pair], return_index=True)
    idx = np.nonzero(pair)[0][first]
    ref_xy, frame_xy = ref_all[k[idx]], frame_all[idx]
    n = len(ref_xy)
    if n < 20:
        return None
    out_xy = _apply(T_ref_out, np.asarray(ref_xy, np.float64))
    fr = np.asarray(frame_xy, np.float64)
    if C_plane is not None:
        fr = _apply(C_plane, fr)
    aff = fit_poly(out_xy, fr, 1, out_shape)
    r_aff = np.sqrt(((poly_apply(aff, out_xy) - fr) ** 2).sum(1))
    order = 3 if n >= 40 else 2
    pol = fit_poly(out_xy, fr, order, out_shape)
    r_pol = np.sqrt(((poly_apply(pol, out_xy) - fr) ** 2).sum(1))
    # keep only stars that fit (a mismatched star would bend the model), then refit
    good = r_pol < max(0.6, 3 * float(np.median(r_pol)))
    if good.sum() >= max(20, 3 * len(_poly_terms(order))):
        pol = fit_poly(out_xy[good], fr[good], order, out_shape)
        r_pol = np.sqrt(((poly_apply(pol, out_xy[good]) - fr[good]) ** 2).sum(1))
        r_aff = r_aff[good]
    else:
        return None
    rms_aff, rms_pol = float(np.sqrt(np.mean(r_aff ** 2))), float(np.sqrt(np.mean(r_pol ** 2)))
    edge = float(np.percentile(r_aff, 95))
    if edge < 0.6 or rms_pol > 0.75 * rms_aff:
        return None                     # the plain alignment is fine, or the model would not help
    pol["edge_affine"] = edge
    pol["rms"] = rms_pol
    return pol


def register_distortion(ref_lum, img_lum, ref_shape, factor=2, max_stars=400, ref_look=None, img_look=None):
    """How img lines up with ref, with distortion: (model ref->img, info). info: stars matched, rms
    of a plain (affine) fit and of the distortion fit, and the largest affine error at the edge.
    Raises ValueError when the star fields can't be matched."""
    def binned(a):
        a = despike(np.asarray(a, np.float32))[0]           # hot pixels are not stars (see despike)
        h, w = a.shape
        return a[:h // factor * factor, :w // factor * factor].reshape(h // factor, factor, w // factor,
                                                                         factor).mean(axis=(1, 3))
    rp, _ = find_stars(binned(ref_lum).astype(np.float32), factor, max_stars=max_stars)
    tp, _ = find_stars(binned(img_lum).astype(np.float32), factor, max_stars=max_stars)
    T, n, rms, ok, j, _how = match_frames_wcs(rp, tp, ref_look, img_look, np.shape(img_lum), tol=4.0)
    Ti = np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2]           # ref -> img
    # pair every star through the transform (edge stars an affine fit misses come in on later rounds)
    model = {"order": 1, "norm": (ref_shape[1] / 2.0, ref_shape[0] / 2.0, max(ref_shape) / 2.0)}
    model["coef"] = fit_poly(rp[j[ok]], tp[ok], 1, ref_shape)["coef"]
    aff = dict(model)
    pairs = None
    for step, order in enumerate((1, 2, 3, 3, 3)):
        pred = poly_apply(model, rp)
        d = np.sqrt(((pred[:, None, :] - tp[None, :, :]) ** 2).sum(-1))
        k = np.argmin(d, axis=1)
        good = d[np.arange(len(k)), k] < (6.0, 4.0, 2.5, 1.5, 1.0)[step]
        if good.sum() < 12:
            break
        need = {1: 3, 2: 20, 3: 40}[order]
        use = order if good.sum() >= need else 1
        model = fit_poly(rp[good], tp[k[good]], use, ref_shape)
        if order == 1:
            aff = model
        pairs = (rp[good], tp[k[good]])
    if pairs is None:
        raise ValueError("too few stars to measure the distortion")
    src, dst = pairs
    r_aff = np.sqrt(((poly_apply(aff, src) - dst) ** 2).sum(1))
    r_pol = np.sqrt(((poly_apply(model, src) - dst) ** 2).sum(1))
    info = {"stars": int(len(src)), "rms_affine": float(np.sqrt(np.mean(r_aff ** 2))),
            "rms": float(np.sqrt(np.mean(r_pol ** 2))), "edge_affine": float(np.percentile(r_aff, 95)),
            "order": model["order"], "shift": float(np.median(np.sqrt(((src - dst) ** 2).sum(1))))}
    return model, info


def is_mirrored(T) -> bool:
    return float(np.linalg.det(np.asarray(T)[:, :2])) < 0


def warp_planes(planes, T: np.ndarray, out_shape, rows=256) -> list:
    """Resample images so they line up with the reference (bilinear; NaN outside).
    T maps image coordinates -> reference coordinates. The sampling positions
    are computed once and reused for every plane (e.g. R, G and B)."""
    A = T[:, :2]
    t = T[:, 2]
    Ai = np.linalg.inv(A).astype(np.float32)
    h, w = out_shape
    H, W = planes[0].shape
    flats = [np.ascontiguousarray(p, dtype=np.float32).ravel() for p in planes]
    outs = [np.empty((h, w), np.float32) for _ in planes]
    xs = np.arange(w, dtype=np.float32) - np.float32(t[0])
    for r0 in range(0, h, rows):
        ys = np.arange(r0, min(r0 + rows, h), dtype=np.float32)[:, None] - np.float32(t[1])
        sx = Ai[0, 0] * xs[None, :] + Ai[0, 1] * ys
        sy = Ai[1, 0] * xs[None, :] + Ai[1, 1] * ys
        x0 = np.floor(sx)
        y0 = np.floor(sy)
        fx = sx - x0
        fy = sy - y0
        x0 = x0.astype(np.int32)
        y0 = y0.astype(np.int32)
        valid = (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1)
        idx = np.clip(y0, 0, H - 2) * W + np.clip(x0, 0, W - 2)
        w11 = fx * fy
        w10 = fy - w11
        w01 = fx - w11
        w00 = 1 - fx - fy + w11
        for f, o in zip(flats, outs):
            v = f[idx] * w00
            v += f[idx + 1] * w01
            v += f[idx + W] * w10
            v += f[idx + W + 1] * w11
            v[~valid] = np.nan
            o[r0:r0 + v.shape[0]] = v
    return outs


def warp(img: np.ndarray, T: np.ndarray, out_shape) -> np.ndarray:
    return warp_planes([img], T, out_shape)[0]


def pixel_noise(img: np.ndarray) -> float:
    """Noise from pixel-to-pixel differences (unaffected by gradients and nebulae)."""
    s = img[::4]
    d = s[:, 1:] - s[:, :-1]
    d = d[np.isfinite(d)]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2)) + 1e-6


# ==========================================================================
# Stacking a group
# ==========================================================================

@dataclass
class StackOptions:
    kappa: float = 3.0
    weighting: bool = True
    osc_half: bool = False                  # colour at half resolution (superpixel)
    formats: tuple = ("fits",)
    masters_root: str = ""
    cal_root: str = ""
    save_cal_masters: bool = True
    align_common: bool = True               # line up all masters of a target (same camera + rig)
    work_dir: str = ""                      # local folder for temporary files
    comet: bool = True                      # comets: also stack on the comet (and combine)
    cosmetic: bool = True                   # replace hot pixels / cosmic rays in each calibrated frame
    pattern: bool = True                    # field rotation: take the camera's fixed pattern out (no swirls)
    local_norm: bool = True                 # local normalization: every frame's background matched to the reference
    drizzle: bool = False                   # 2x drizzle: finer detail from dithered frames (master at double size)
    pixfrac: float = 0.8                    # drizzle drop size (share of an input pixel)
    drizzle_if_dithered: bool = False       # Fast Track: drizzle only frames that were dithered (else it adds
                                            # nothing but a 4x larger master and slower steps - the comet)


@dataclass
class StackResult:
    group: LightGroup
    files: list = field(default_factory=list)
    used: int = 0
    skipped: list = field(default_factory=list)
    log: list = field(default_factory=list)
    data: np.ndarray | None = None
    error: str | None = None
    ref_path: Path | None = None            # frame whose pixel grid the master uses
    common_ref: str | None = None
    autocrop: tuple | None = None           # (x0, y0, x1, y1) covered by ~all frames
    extra: dict = field(default_factory=dict)   # more masters to save, e.g. {"Comet": …, "Stars": …}
    dither: dict | None = None              # were the frames dithered (see dither_report)


def coverage_map(transforms, out_shape, in_shape, step=8):
    """Fraction of frames covering each point of a coarse grid over the output.
    transforms map frame-plane coordinates -> output coordinates."""
    h, w = out_shape
    ih, iw = in_shape
    ys = np.arange(0, h, step, dtype=np.float64)
    xs = np.arange(0, w, step, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    cnt = np.zeros(X.shape, np.int32)
    for T in transforms:
        Ai = np.linalg.inv(T[:, :2])
        dx, dy = X - T[0, 2], Y - T[1, 2]
        sx = Ai[0, 0] * dx + Ai[0, 1] * dy
        sy = Ai[1, 0] * dx + Ai[1, 1] * dy
        cnt += (sx >= 0) & (sy >= 0) & (sx <= iw - 1) & (sy <= ih - 1)
    return cnt / max(len(transforms), 1)


def largest_rectangle(mask: np.ndarray):
    """Largest axis-aligned all-True rectangle: (row0, col0, row1, col1) inclusive, or None."""
    rows, cols = mask.shape
    heights = np.zeros(cols, np.int32)
    best, best_area = None, 0
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []                      # (start column, height)
        for c in range(cols + 1):
            h = int(heights[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s0, sh = stack.pop()
                area = sh * (c - s0)
                if area > best_area:
                    best_area, best = area, (r - sh + 1, s0, r, c - 1)
                start = s0
            stack.append((start, h))
    return best


def auto_crop_rect(transforms, out_shape, in_shape, min_fraction=0.9):
    """Rectangle (x0, y0, x1, y1) of the output, exclusive end, covered by at least
    min_fraction of the frames - the ragged stacking edges lie outside it."""
    h, w = out_shape
    step = max(4, int(math.ceil(max(h, w) / 600)))
    cov = coverage_map(transforms, out_shape, in_shape, step)
    r = largest_rectangle(cov >= min_fraction - 1e-9)
    if r is None:
        return None
    r0, c0, r1, c1 = r
    # stay one grid step inside so partly covered cells are excluded
    x0 = min(w, (c0 + 1) * step) if c0 > 0 else 0
    y0 = min(h, (r0 + 1) * step) if r0 > 0 else 0
    x1 = max(x0, c1 * step) if c1 < cov.shape[1] - 1 else w
    y1 = max(y0, r1 * step) if r1 < cov.shape[0] - 1 else h
    if (x1 - x0) < w * 0.3 or (y1 - y0) < h * 0.3:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def common_reference(group_list: list, progress=None) -> dict | None:
    """Pick one reference frame for several groups of the same target, camera
    and rig (e.g. L, R, G, B or Ha, OIII, SII): the best-scored frame of the
    group with the most frames. Returns {"name", "path", "pts"}."""
    if len(group_list) < 2:
        return None
    big = max(group_list, key=lambda g: (len(g.frames), g.total_exposure))
    best = max(big.frames, key=lambda f: (f.score if f.score is not None else 0))
    if progress:
        progress(f"Finding stars in the common reference {best.info.path.name}", 0.0)
    raw = ImageSource(best.info.path).read()
    if raw.ndim == 3:
        raw = raw.mean(axis=0) if raw.shape[0] == 3 and not best.info.bayer else raw[0]
    lum, fac = luminance(raw.astype(np.float32), bool(best.info.bayer))
    robust = bool(best.info.smart)
    pts, _pk = find_stars(lum, fac, robust=robust)
    if not robust and len(pts) < ROBUST_MIN_STARS:
        r_pts, _ = find_stars(lum, fac, robust=True)
        if len(r_pts) > len(pts):
            robust, pts = True, r_pts
    if len(pts) < 6:
        return None
    return {"name": best.info.path.name, "path": best.info.path, "pts": pts, "robust": robust}


def alignment_sets(groups: list) -> list:
    """Groups that can share a pixel grid: same target, camera, rig and frame size."""
    sets = {}
    for g in groups:
        r = g.ref
        sets.setdefault((g.target, (g.camera or "").lower(), g.rig, r.w, r.h, r.binning,
                         bool(r.bayer)), []).append(g)
    return list(sets.values())


def master_light_name(g: LightGroup, multi_camera: bool) -> str:
    parts = [safe_name(g.target)]                          # no '/' from comet names like 12P/Pons-Brooks
    if g.filter and g.filter != "No filter":
        parts.append(safe_name(core._ascii(g.filter)))
    if multi_camera:
        parts.append(short_camera(g.camera))
    parts.append("Master")
    n = g.nights
    if n:
        parts.append(n[0] if len(n) == 1 else f"{n[0]}_to_{n[-1]}")
    return "_".join(parts)


def _invert_T(T: np.ndarray) -> np.ndarray:
    """The 2x3 transform that undoes T."""
    Ai = np.linalg.inv(T[:, :2])
    return np.hstack([Ai, (-Ai @ T[:, 2])[:, None]])


def _rotation_spread(transforms) -> float:
    """How far (degrees) the frames are turned against each other - field rotation of an alt-az mount.
    A meridian flip (an equatorial mount turning the camera half a turn, 180 degrees) is not field rotation:
    angles are compared modulo 180, so frames from both sides of the meridian count as not turned."""
    ang = [float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))) for T in transforms]
    if not ang:
        return 0.0
    a0 = ang[0]
    folded = [((a - a0 + 90.0) % 180.0) - 90.0 for a in ang]
    return max(folded) - min(folded)


PATTERN_PASSES = 2          # a second measurement from the cleaned stack catches what the first left


def sensor_pattern(frames_cal, model_planes, transforms, ks, bgs_list, ref_bg, bayer, shape):
    """The camera's own fixed pattern (dark-current pattern, warm pixels, colour blotches that stay put on
    the sensor) measured from the lights themselves: each frame minus the stacked sky, put back where it lay on
    the sensor, averaged over frames. With field rotation that pattern turns against the sky from frame to
    frame and stacks into circular swirls; taken out of every frame, they go. frames_cal yields the calibrated
    frames (CFA or mono) in the order of transforms."""
    h, w = shape
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float32)
    grid = _pattern(bayer) if bayer else None
    for cal_img, T, k, bgs in zip(frames_cal, transforms, ks, bgs_list):
        back = warp_planes([np.asarray(p, np.float32) for p in model_planes], _invert_T(T), (h, w))
        back = [(b - np.float32(ref_bg[c])) / np.float32(k) + np.float32(bgs[c]) for c, b in enumerate(back)]
        if grid is not None:
            model = np.empty((h, w), np.float32)
            idx = {"R": 0, "G": 1, "B": 2}
            for i in (0, 1):
                for j in (0, 1):
                    model[i::2, j::2] = back[idx[grid[i][j]]][i::2, j::2]
        else:
            model = back[0]
        r = cal_img.astype(np.float32) - model
        fin = np.isfinite(r)
        if fin.sum() < 1000:
            continue
        med = float(np.median(r[fin][::7]))
        sig = float(np.median(np.abs(r[fin][::7] - med))) * 1.4826 + 1e-9
        ok = fin & (np.abs(r - med) < 4 * sig)          # stars a little off, satellites, cosmic rays: left out
        acc[ok] += (r[ok] - med)
        cnt[ok] += 1
    pat = np.where(cnt >= 5, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
    # only the fine, pixel-to-pixel part: broad shapes in it are mostly the stacked sky not quite matching
    # (each colour of a colour camera on its own, so colours don't mix)
    if grid is not None:
        for i in (0, 1):
            for j in (0, 1):
                sub = pat[i::2, j::2]
                pat[i::2, j::2] = sub - _smooth(sub, 2.0)
    else:
        pat = pat - _smooth(pat, 4.0)
    return pat, int((cnt >= 5).mean() * 100)


def object_mask(plane: np.ndarray, block: int) -> np.ndarray:
    """Where the target and its glow are (and bright stars), on the same 4x-thinned grid background_map
    samples: anything clearly above the smooth sky, a little widened. Leaving it out of the sky estimate keeps
    the target out of 'Even out the sky' - otherwise a bright target (a planetary nebula, a galaxy) that is not
    scaled exactly like the stars leaves block-shaped dark or bright borders around itself in the stack."""
    sub = np.asarray(plane[::4, ::4], np.float32)
    m = np.zeros(sub.shape, bool)
    fin = np.isfinite(sub)
    if fin.sum() < 100:
        return m
    for _ in range(2):
        bg = background_map(plane, block, m)[::4, ::4][:sub.shape[0], :sub.shape[1]]
        r = np.where(fin, sub - bg, 0).astype(np.float32)
        sm = _box(r, 2)
        v = sm[fin & ~m]
        if v.size < 100:
            break
        med = float(np.median(v))
        sig = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
        m = sm > med + 3 * sig
        # stars are not the target (and hardly move a block median): opening - shrink away everything star-sized,
        # grow back what is left - keeps the nebula / galaxy and its glow, drops the stars (M27's dense star field
        # otherwise covered over half the picture)
        re = 4
        core = _box(m.astype(np.float32), re) > 0.999
        m = _box(core.astype(np.float32), re + max(2, block // 32)) > 0  # grown back and widened: the glow's edge
    if m.mean() > 0.6:                    # the whole field is nebula: nothing to tell sky from target by
        return np.zeros(sub.shape, bool)
    return m


def background_map(plane: np.ndarray, block: int, mask: np.ndarray | None = None, want_grid=False) -> np.ndarray:
    """The smooth background of an aligned frame (NaN outside it): the median of large blocks, gently smoothed
    and spread back to full size. Stars hardly move a block median. mask (from object_mask, the same for every
    frame) leaves the target out; a block that is mostly target takes its neighbours' sky."""
    h, w = plane.shape
    nby, nbx = -(-h // block), -(-w // block)
    pad = np.full((nby * block, nbx * block), np.nan, np.float32)
    pad[:h, :w] = plane
    blocks = pad.reshape(nby, block, nbx, block)[:, ::4, :, ::4]
    if mask is not None and mask.any():
        bh, bw = blocks.shape[1], blocks.shape[3]
        mp = np.zeros((nby * bh, nbx * bw), bool)
        mh, mw = min(mask.shape[0], mp.shape[0]), min(mask.shape[1], mp.shape[1])
        mp[:mh, :mw] = mask[:mh, :mw]
        mb = mp.reshape(nby, bh, nbx, bw)
        blocks = np.where(mb, np.nan, blocks)
        keep = (~mb & np.isfinite(blocks)).mean(axis=(1, 3))
    else:
        keep = None
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grid = np.nanmedian(blocks, axis=(1, 3)).astype(np.float32)
    if keep is not None:
        grid = np.where(keep >= 0.25, grid, np.nan).astype(np.float32)   # mostly target: from the neighbours
    ok = np.isfinite(grid)
    if not ok.any():
        return np.zeros((nby, nbx) if want_grid else (h, w), np.float32)
    for _ in range(max(nby, nbx)):                       # empty blocks (outside the frame): from their neighbours
        if ok.all():
            break
        p = np.pad(np.where(ok, grid, 0), 1)
        q = np.pad(ok.astype(np.float32), 1)
        s_ = sum(p[1 + dy:1 + dy + nby, 1 + dx:1 + dx + nbx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        n_ = sum(q[1 + dy:1 + dy + nby, 1 + dx:1 + dx + nbx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        fill = (~ok) & (n_ > 0)
        grid = np.where(fill, s_ / np.maximum(n_, 1), grid)
        ok = ok | fill
    if nby >= 3 and nbx >= 3:                             # a little smoothing between blocks
        p = np.pad(grid, 1, mode="edge")
        grid = sum(p[1 + dy:1 + dy + nby, 1 + dx:1 + dx + nbx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    if want_grid:
        return grid.astype(np.float32)
    return _upsample(grid.astype(np.float32), nby * block, nbx * block)[:h, :w]


def grid_to_map(grid: np.ndarray, block: int, h: int, w: int) -> np.ndarray:
    """A background_map grid spread to full size (h, w)."""
    nby, nbx = grid.shape
    return _upsample(grid.astype(np.float32), nby * block, nbx * block)[:h, :w]


def drizzle_add(acc, wacc, planes_, T, add_of, keep_of, scale=2, pixfrac=0.8, weight=1.0, rows=96):
    """Add one frame to a drizzle: every output pixel (scale x finer than the reference grid) takes the input
    pixels whose shrunken 'drops' (pixfrac of a pixel, centred on it) overlap it, weighted by the overlap - a gather
    over output rows, so no huge scatter arrays. T maps this frame -> reference. planes_: the frame's planes, already
    scaled and with their own sky taken off; add_of(c, r0, r1): what is added back (the reference's sky) for output
    rows r0..r1; keep_of(r0, r1): 1 where the regular stack did not reject this frame (satellites, planes).
    acc (nc, oh, ow), wacc (oh, ow); rows must be a multiple of scale."""
    H, W = planes_[0].shape
    nc = len(planes_)
    oh, ow = wacc.shape
    Ti = _invert_T(T).astype(np.float32)
    half = np.float32(0.5 / scale)                         # an output pixel's half size, in input pixels
    hp = np.float32(0.5 * pixfrac)
    flat = [np.ascontiguousarray(p, dtype=np.float32).ravel() for p in planes_]
    xs = ((np.arange(ow, dtype=np.float32) + 0.5) / scale - 0.5)[None, :]

    def band(r0):
        r1 = min(oh, r0 + rows)
        ry = ((np.arange(r0, r1, dtype=np.float32) + 0.5) / scale - 0.5)[:, None]
        qx = Ti[0, 0] * xs + Ti[0, 1] * ry + Ti[0, 2]
        qy = Ti[1, 0] * xs + Ti[1, 1] * ry + Ti[1, 2]
        keep = keep_of(r0, r1)
        fx0, fy0 = np.floor(qx).astype(np.int32), np.floor(qy).astype(np.int32)
        s_ = np.zeros((nc, r1 - r0, ow), np.float32)
        w_ = np.zeros((r1 - r0, ow), np.float32)
        for dy in (0, 1):
            iy = fy0 + dy
            oy = np.clip(np.minimum(iy + hp, qy + half) - np.maximum(iy - hp, qy - half), 0, None)
            oy *= (iy >= 0) & (iy < H)
            iyw = np.clip(iy, 0, H - 1) * W
            for dx in (0, 1):
                ix = fx0 + dx
                ox = np.clip(np.minimum(ix + hp, qx + half) - np.maximum(ix - hp, qx - half), 0, None)
                ox *= (ix >= 0) & (ix < W)
                w = ox * oy * keep
                idx = iyw + np.clip(ix, 0, W - 1)
                for c in range(nc):
                    s_[c] += flat[c][idx] * w
                w_ += w
        for c in range(nc):
            a_ = add_of(c, r0, r1)
            acc[c, r0:r1] += (s_[c] + a_ * w_) * np.float32(weight)
        wacc[r0:r1] += w_ * np.float32(weight)
    # bands of rows in parallel (numpy lets go of the interpreter lock): each band writes only its own rows
    import os
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2)))) as ex:
        list(ex.map(band, range(0, oh, rows)))


def dither_report(transforms, fwhms) -> dict:
    """Were the frames dithered? From how far apart they sit (their shifts against the reference) and how well
    their sub-pixel positions cover a pixel - drizzle needs both. Star size (FWHM, px) from page 1 when measured."""
    t = np.array([[T[0, 2], T[1, 2]] for T in transforms], np.float64)
    n = len(t)
    spread = float(np.max(np.ptp(t, axis=0))) if n else 0.0
    fr = np.mod(t, 1.0)
    occ = len({(int(a * 4), int(b * 4)) for a, b in fr}) / 16.0 if n else 0.0
    fw = [float(f) for f in fwhms if f]
    fwhm = float(np.median(fw)) if fw else None
    return {"dithered": n >= 10 and spread >= 2.0 and occ >= 0.5, "spread": spread, "phases": occ,
            "fwhm": fwhm, "small_stars": fwhm is None or fwhm < 2.2, "frames": n}   # 2.2 px: really undersampled


def frame_workers(h: int, w: int, nc: int = 1) -> int:
    """How many frames to calibrate and line up at once: about half the processor cores (the warp already
    uses several), at most 6, and only as many as the free memory holds (a frame in flight needs about seven
    copies of itself)."""
    cpus = os.cpu_count() or 2
    if cpus < 4:
        return 1
    per = float(h) * w * 4 * max(nc, 1) * 7
    try:
        import session_log
        free = session_log.memory().get("free")
    except Exception:  # noqa: BLE001
        free = None
    by_mem = int(free * 1e9 * 0.45 // per) if free else 2
    return max(1, min(cpus // 2, 6, by_mem))


def stack_group(g: LightGroup, opts: StackOptions, multi_camera=False, progress=None, cancel=None,
                common: dict | None = None) -> StackResult:
    """Calibrate, align and combine one group. Calibration masters in g.plan must
    already be built (CalItem.data).
    common: {"name", "pts"} of a shared reference frame - every frame is then
    placed on that frame's pixel grid so all masters of a target line up."""
    res = StackResult(g)
    log = res.log
    t_start = time.time()
    plan = g.plan or CalPlan()

    def prog(text, frac):
        if progress:
            progress(text, frac)
        if cancel and cancel():
            raise Cancelled()

    frames = sorted(g.frames, key=lambda f: (-(f.score if f.score is not None else 0), -f.info.exposure_total))
    ref_info = frames[0].info
    bayer = ref_info.bayer
    rgb_in = ref_info.planes == 3 and not bayer      # already-debayered colour stacks
    channels = ["R", "G", "B"] if (bayer or rgb_in) else ["L"]
    cal = Calibrator(ref_info, plan)
    night_cals = {}
    if plan.night_flats:
        for n_, (f_, _b, _d) in plan.night_flats.items():
            if f_ is not None and f_.data is not None:
                night_cals[n_] = Calibrator(ref_info, _dc_replace(plan, flat=f_))
    log.append(f"Group: {g.label()}  ({len(frames)} frames, {g.total_exposure / 3600:.2f} h)")
    log.append(f"Reference frame: {ref_info.path.name}")
    log.append("Calibration: " + "; ".join(
        f"{name} = {it.label()}" for name, it in (("bias", plan.bias), ("dark", plan.dark), ("flat", plan.flat))
        if it is not None) + (f"  [dark {plan.dark_mode}]" if plan.dark else ""))
    if night_cals:
        log.append(f"Flats per night: {', '.join(f'{n_} -> {plan.night_flats[n_][0].label()}' for n_ in night_cals)}")
    for k, w in plan.warnings:
        log.append(f"WARNING: {w}")
    for n in plan.notes:
        log.append(f"Note: {n}")

    if not rgb_in and cal.sub is not None:
        try:
            for line in calibration_check(ref_info, cal):
                log.append(line)
        except Exception as e:  # noqa: BLE001 - a check must never stop stacking
            log.append(f"(calibration check skipped: {e})")

    hot_counts = []        # hot pixels replaced per frame (cosmetic correction)

    def calibrated(info):
        raw = ImageSource(info.path).read()
        if rgb_in:
            if raw.ndim != 3 or raw.shape[0] != 3:
                raise ValueError("not a 3-plane colour image like the reference")
            return raw.astype(np.float32, copy=False)      # stacked colour: no calibration, no debayer
        if raw.ndim == 3:
            raw = raw[0]
        out = night_cals.get(info.night, cal).apply(raw)
        if getattr(opts, "cosmetic", True):
            out, nhot = cosmetic(out, bool(bayer))
            hot_counts.append(nhot)
        return out

    def lumin(img):
        return luminance(img.mean(axis=0) if img.ndim == 3 else img, bool(bayer))

    def planes(cal_img):
        if rgb_in:
            return [cal_img[0], cal_img[1], cal_img[2]]
        if bayer:
            if opts.osc_half:
                return [debayer_channel(cal_img, bayer, c, half=True) for c in channels]
            return debayer_bilinear(cal_img, bayer)
        return [cal_img]

    # --- reference -------------------------------------------------------------
    prog(f"{g.label()}: reference frame", 0.0)
    ref_cal = calibrated(ref_info)
    if rgb_in:
        log.append("Input: colour images that are already stacked (e.g. by a smart telescope) — "
                   "aligned and combined without calibration or debayering")
    lum, fac = lumin(ref_cal)
    robust = bool(ref_info.smart)
    ref_pts, ref_peaks = find_stars(lum, fac, robust=robust)
    if not robust and len(ref_pts) < ROBUST_MIN_STARS:
        r_pts, r_peaks = find_stars(lum, fac, robust=True)
        if len(r_pts) > len(ref_pts):
            robust, ref_pts, ref_peaks = True, r_pts, r_peaks
    if robust:
        log.append("Star finder: robust mode for " + (f"{ref_info.smart} frames" if ref_info.smart else "faint frames")
                   + " (hot pixels and noise filtered out)")
    if len(ref_pts) < 6:
        raise ValueError(f"Too few stars in the reference frame {ref_info.path.name}")
    half = bayer and opts.osc_half
    out_h, out_w = (ref_cal.shape[-2] // 2, ref_cal.shape[-1] // 2) if half else ref_cal.shape[-2:]
    del ref_cal
    res.ref_path = ref_info.path
    T_g2c = None
    looks = {}

    def look_of(path):
        k = str(path)
        if k not in looks:
            try:
                looks[k] = ImageSource(path).look
            except Exception:  # noqa: BLE001 - a header is only a guide here
                looks[k] = {}
        return looks[k]
    ref_shape = (out_h * 2, out_w * 2) if half else (out_h, out_w)
    if common and common.get("name") != ref_info.path.name:
        try:
            T_g2c, inl, rms, _ok, _j, how = match_frames_wcs(common["pts"], ref_pts, look_of(common["path"]),
                                                             look_of(ref_info.path), ref_shape)
            log.append(f"Aligned to the common reference {common['name']} (stars matched {inl}, rms {rms:.2f} px"
                       + (", mirror image of it" if is_mirrored(T_g2c) else "")
                       + (", guided by the plate solutions" if how != "stars" else "") + ") "
                       "so all masters of this target line up")
            res.ref_path = common.get("path", ref_info.path)
        except ValueError as e:
            log.append(f"WARNING: could not align to the common reference {common['name']} ({e}); "
                       "this master uses its own reference frame")
    elif common:
        log.append("This group holds the common reference frame for the target")
    res.common_ref = common.get("name") if common and (T_g2c is not None or
                                                      common.get("name") == ref_info.path.name) else None
    C_half = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25]])
    Cinv_half = np.array([[2.0, 0, 0.5], [0, 2.0, 0.5]])

    def final_T(T):
        """Frame -> output grid: via the group reference, then the common reference,
        then (for half-resolution colour) into half-size pixel coordinates."""
        Tf = T if T_g2c is None else _compose(T_g2c, T)
        return _compose(C_half, _compose(Tf, Cinv_half)) if half else Tf

    def is_identity(T):
        return np.allclose(T, [[1, 0, 0], [0, 1, 0]], atol=1e-9)

    # --- local cache of aligned frames --------------------------------------------
    work = Path(opts.work_dir) if opts.work_dir else Path(tempfile.gettempdir())
    work.mkdir(parents=True, exist_ok=True)
    need = len(frames) * len(channels) * out_h * out_w * 2
    try:
        free = shutil.disk_usage(work).free
    except OSError:
        free = 0
    use_cache = need < 0.6 * free
    cache_dir = Path(tempfile.mkdtemp(prefix="fbstack_", dir=work)) if use_cache else None
    log.append(f"Aligned frames cached in {cache_dir} ({need / 1e9:.1f} GB)" if use_cache else
               f"Not enough free space for the cache ({need / 1e9:.1f} GB needed) — using two passes")

    ref_bg = None
    entries = []          # (info, output transform, scale k, bg list, weight)
    distortion_fixed = []  # frames lined up with a distortion model (edge error it removed)
    wcs_used = []          # frames lined up with the help of their plate solutions
    last_model = [None]
    frame_models = []      # per entry: the distortion model, or None

    def prepare(idx, lf, write):
        info = lf.info
        tm = {}
        _t = time.perf_counter()
        img = calibrated(info)
        tm["read and calibrate"] = time.perf_counter() - _t
        _t = time.perf_counter()
        lum, fac = lumin(img)
        pts, peaks = find_stars(lum, fac, robust=robust)
        tm["find stars"] = time.perf_counter() - _t
        _t = time.perf_counter()
        if info is ref_info:
            T = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            k_phot = 1.0
            inl, rms = len(pts), 0.0
        else:
            T, inl, rms, ok, j, how = match_frames_wcs(ref_pts, pts, look_of(ref_info.path), look_of(info.path),
                                                       img.shape[-2:])
            if how != "stars":
                wcs_used.append(info.path.name)
            # photometric scale from matched stars
            pr, pt = ref_peaks[j[ok]], peaks[ok]
            good = (pr > 0) & (pt > 0)
            if rgb_in and good.sum() >= 12:
                # stacks from smart telescopes are each scaled differently, and their bright
                # stars are clipped: use the fainter three quarters of the matched stars
                good &= (pr < np.percentile(pr[good], 75)) & (pt < np.percentile(pt[good], 75))
            k_phot = float(np.median(pr[good] / pt[good])) if good.sum() >= 5 else 1.0
            k_phot = float(np.clip(k_phot, 0.01, 100.0) if rgb_in else np.clip(k_phot, 0.3, 3.0))
        Tout = final_T(T)
        tm["match stars"] = time.perf_counter() - _t
        _t = time.perf_counter()
        pls = planes(img)
        del img
        noises = [pixel_noise(p) for p in pls]
        model = None
        if info is not ref_info:
            model = distortion_model(ref_pts, pts, T, final_T(np.array([[1.0, 0, 0], [0, 1.0, 0]])),
                                     C_half if half else None, (out_h, out_w))
        if model is not None:
            aligned = warp_poly(pls, model, (out_h, out_w))
        elif is_identity(Tout):
            aligned = [p.astype(np.float32, copy=False) for p in pls]
        else:
            aligned = warp_planes(pls, Tout, (out_h, out_w))
        del pls
        bgs = []
        for a in aligned:
            sm = a[::8, ::8]
            bgs.append(float(np.median(sm[~np.isnan(sm)])))
        tm["line up"] = time.perf_counter() - _t
        aligned_tm[id(aligned)] = tm
        return aligned, Tout, k_phot, bgs, noises, inl, rms, model

    aligned_tm = {}                     # per frame: seconds in each part of preparing it (for the log)
    stage_t = {}

    ln_block = max(64, int(max(out_h, out_w) / 16))
    ref_maps = []
    ln_masks = []
    last_grids = []                     # this frame's sky (block grid, reference coordinates)
    frame_grids = []

    def normalize(aligned, k_phot, bgs):
        if getattr(opts, "local_norm", True) and ref_maps:
            # local normalization: this frame's own background (gradients from the moon, haze, light pollution
            # that change through the night) is replaced by the reference frame's, so every frame matches it
            # everywhere - outlier rejection then sees only real outliers
            grids = [background_map(a, ln_block, ln_masks[c], want_grid=True) for c, a in enumerate(aligned)]
            return [((a - grid_to_map(gr, ln_block, *a.shape)) * np.float32(k_phot) + ref_maps[c])
                    for c, (a, gr) in enumerate(zip(aligned, grids))], grids
        return [((a - bgs[c]) * np.float32(k_phot) + np.float32(ref_bg[c])) for c, a in enumerate(aligned)], []

    n = len(frames)
    used = []
    cache_files = []
    frame_noise = []
    n_mirrored = 0
    welford = None
    workers = 1                                   # decided once the reference frame shows how many planes

    def work(lf):
        """Calibrate, line up and (once the reference is in) normalise one frame - several frames at once
        on computers with the cores and memory for it (numpy lets go of the interpreter lock)."""
        try:
            aligned, T, k_phot, bgs, noises, inl, rms, model = prepare(0, lf, True)
        except ValueError as e:
            return lf, None, str(e)
        tm = aligned_tm.pop(id(aligned), {})
        normed, grids = None, None
        if ref_bg is not None:
            _t = time.perf_counter()
            normed, grids = normalize(aligned, k_phot, bgs)
            tm["even out the sky"] = time.perf_counter() - _t
            aligned = None
        for k_, v_ in tm.items():
            stage_t[k_] = stage_t.get(k_, 0.0) + v_
        return lf, (aligned, T, k_phot, bgs, noises, inl, rms, model, normed, grids), None

    def consume(lf, r, err):
        """In frame order, on this thread: the reference's sky, the log, the weights, the stack itself."""
        nonlocal ref_bg, ref_maps, welford, n_mirrored
        if r is None:
            res.skipped.append((lf.info.path.name, err))
            log.append(f"SKIPPED {lf.info.path.name}: {err}")
            return
        aligned, T, k_phot, bgs, noises, inl, rms, model, normed, grids = r
        if model is not None:
            distortion_fixed.append(model["edge_affine"])
        if ref_bg is None:
            ref_bg = bgs
            if getattr(opts, "local_norm", True):
                ln_masks[:] = [object_mask(a, ln_block) for a in aligned]
                ref_maps = [background_map(a, ln_block, ln_masks[c]) for c, a in enumerate(aligned)]
                _mpct = 100.0 * float(np.mean([m.mean() for m in ln_masks]))
                log.append(f"Even out the sky between frames: every frame's sky matched to the reference frame's, "
                           f"in {ln_block} px areas (PixInsight: LocalNormalization); the target and its glow "
                           f"({_mpct:.0f}% of the picture) are left out of the sky")
        if normed is None:
            normed, grids = normalize(aligned, k_phot, bgs)
        del aligned
        noise = float(np.mean(noises)) * k_phot
        weight = 1.0 / noise ** 2 if opts.weighting else 1.0
        entries.append((lf.info, T, k_phot, bgs, weight))
        frame_grids.append(list(grids or []))
        frame_models.append(model)
        frame_noise.append(noise)
        mir = is_mirrored(T)
        n_mirrored += int(mir)
        log.append(f"  {lf.info.path.name}: stars matched {inl}, rms {rms:.2f} px, scale {k_phot:.3f}, "
                   f"noise {noise:.1f}" + (f", score {lf.score:.0f}" if lf.score is not None else "")
                   + (" (mirror image — lined up anyway)" if mir else ""))
        _t = time.perf_counter()
        if use_cache:
            p = cache_dir / f"f{len(used):05d}.npy"
            np.save(p, np.stack([np.clip(x, -65000, 65000).astype(np.float16) for x in normed]))
            cache_files.append(p)
        else:
            welford = _welford_add(welford, normed)
        stage_t["write to the cache"] = stage_t.get("write to the cache", 0.0) + time.perf_counter() - _t
        used.append(lf)

    t_frames = time.perf_counter()

    def step_text(idx):
        return f"{g.label()}: calibrating and aligning {idx + 1}/{n}" + (f" ({workers} at a time)" if workers > 1 else "")
    i = 0
    while i < n and ref_bg is None:               # the reference first: every other frame is matched to it
        prog(step_text(i), 0.05 + 0.6 * i / n)
        consume(*work(frames[i]))
        i += 1
    rest = list(range(i, n))
    workers = frame_workers(out_h, out_w, len(ref_bg or [0]))
    if workers > 1 and len(rest) > 1:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            todo = iter(rest)
            pending = deque()
            for j in todo:
                pending.append((j, ex.submit(work, frames[j])))
                if len(pending) >= workers:
                    break
            try:
                while pending:
                    j, fut = pending.popleft()
                    prog(step_text(j), 0.05 + 0.6 * j / n)
                    consume(*fut.result())
                    nxt = next(todo, None)
                    if nxt is not None:
                        pending.append((nxt, ex.submit(work, frames[nxt])))
            finally:
                for _j, fut in pending:
                    fut.cancel()
    else:
        for j in rest:
            prog(step_text(j), 0.05 + 0.6 * j / n)
            consume(*work(frames[j]))
    if not entries:
        raise ValueError("No frames could be aligned")
    if distortion_fixed:
        log.append(f"Field distortion corrected in {len(distortion_fixed)} of {len(entries)} frames (their stars "
                   f"were up to {max(distortion_fixed):.1f} px off at the edges with a plain alignment - e.g. after "
                   "a meridian flip, or another night's setup)")
    done_hot = [n for n in hot_counts if n >= 0]
    skipped_hot = [n for n in hot_counts if n < 0]
    if done_hot and max(done_hot) > 0:
        log.append(f"Hot pixels removed (cosmetic correction): about {int(np.median(done_hot))} per frame")
    if skipped_hot:
        log.append(f"Hot-pixel removal skipped in {len(skipped_hot)} frame(s): the stars are as sharp as hot pixels "
                   "there (binned or undersampled frames), so the two can't be told apart")
    if wcs_used:
        log.append(f"Lined up with the help of their plate solutions: {len(wcs_used)} frame(s) whose star match "
                   "failed or disagreed with where the telescope was pointing")
    res.used = len(entries)
    if entries and stage_t:
        log.append("Time for the frames, " + (f"{workers} at a time, " if workers > 1 else "one at a time, ")
                   + "summed over all frames: " + ", ".join(f"{k} {v:.0f} s" for k, v in stage_t.items())
                   + f"; {time.perf_counter() - t_frames:.0f} s on the clock for {len(entries)} frames")
    try:
        res.autocrop = auto_crop_rect([e[1] for e in entries], (out_h, out_w), (out_h, out_w))
    except Exception:  # noqa: BLE001 - auto-crop is only a suggestion
        res.autocrop = None
    if res.autocrop:
        x0, y0, x1, y1 = res.autocrop
        log.append(f"Auto-crop suggestion (area covered by 90% of frames): x {x0}-{x1}, y {y0}-{y1} "
                   f"({100 * (x1 - x0) * (y1 - y0) / (out_w * out_h):.0f}% of the image)")
    weights = np.array([e[4] for e in entries], np.float32)
    weights /= weights.mean()
    # were the frames dithered (moved by small random amounts between frames)? and are the stars small? Then
    # Double the resolution (drizzle 2x) can add detail - say so; and say so when it was on without dithering
    try:
        res.dither = dither_report([e[1] for e in entries], [f.fwhm for f in g.frames])
        d_ = res.dither
        if d_["dithered"] and d_["small_stars"] and not getattr(opts, "drizzle", False):
            log.append(f"TIP: drizzle - these frames are dithered (moved up to {d_['spread']:.0f} px between frames, "
                       f"{d_['phases']:.0%} of sub-pixel positions covered)"
                       + (f" and the stars are small ({d_['fwhm']:.1f} px)" if d_["fwhm"] else "")
                       + ": stacking again with 'Double the resolution (drizzle 2x)' can give finer detail. It "
                       "makes stacking take noticeably longer and the masters 4x larger.")
        elif getattr(opts, "drizzle", False) and not d_["dithered"]:
            log.append("TIP: drizzle - these frames were hardly dithered, so drizzle could add little detail; "
                       "stacking without it is faster and gives smaller masters.")
    except Exception:  # noqa: BLE001 - advice only
        pass

    def drizzle_stack(regular):
        """2x drizzle of all frames (dithered frames give finer detail), with the regular stack's rejection."""
        sc = 2
        pf = float(getattr(opts, "pixfrac", 0.8))
        oh, ow = out_h * sc, out_w * sc
        acc = np.zeros((nc, oh, ow), np.float32)
        wacc = np.zeros((oh, ow), np.float32)
        dz = list(range(len(entries)))
        for j, i in enumerate(dz):
            info, T, k_phot, bgs, weight = entries[i]
            prog(f"{g.label()}: drizzle {j + 1}/{len(dz)}", 0.9 + 0.08 * j / max(len(dz), 1))
            pls = planes(calibrated(info))
            cached = np.load(cache_files[i], mmap_mode="r")
            # this frame's own sky, measured in reference coordinates (target left out) when the frames were
            # aligned - taken off after drizzling, so the sky blocks line up with the reference's (sky blocks
            # measured on the turned frame left box-shaped borders around a bright target)
            fg = frame_grids[i] if (getattr(opts, "local_norm", True) and ref_maps and i < len(frame_grids)
                                     and frame_grids[i]) else None
            sig = max(frame_noise[i], 1e-6) * opts.kappa

            # rejection at reference resolution once per frame (each reference pixel = sc x sc output pixels)
            okr = np.ones((out_h, out_w), np.float32)
            for c in range(nc):
                cv = np.asarray(cached[c], np.float32)
                tol = np.maximum(sig, 0.25 * np.abs(regular[c] - np.float32(ref_bg[c])))   # stars vary with seeing
                okr *= (np.isnan(cv) | (np.abs(cv - regular[c]) <= tol)).astype(np.float32)
                del cv

            def up_rows(a2, r0, r1):
                blk = a2[r0 // sc:(r1 + sc - 1) // sc]
                return np.repeat(np.repeat(blk, sc, axis=0), sc, axis=1)[:r1 - r0, :ow]

            def keep_of(r0, r1):
                return up_rows(okr, r0, r1)
            # each plane scaled and with its own sky taken off once; the reference's sky is added back per row
            if fg is not None:
                pn = [p_.astype(np.float32) * np.float32(k_phot) for p_ in pls]
                sky_add = [ref_maps[c] - grid_to_map(fg[c], ln_block, out_h, out_w) * np.float32(k_phot)
                           for c in range(len(pls))]

                def add_of(c, r0, r1):
                    return up_rows(sky_add[c], r0, r1)
            else:
                pn = [(p_.astype(np.float32) - np.float32(bgs[c])) * np.float32(k_phot) for c, p_ in enumerate(pls)]

                def add_of(c, r0, r1):
                    return up_rows(ref_maps[c], r0, r1) if ref_maps else np.float32(ref_bg[c])
            drizzle_add(acc, wacc, pn, T, add_of, keep_of, sc, pf, float(weights[i]))
            del pls, cached
        cover = wacc > 0
        wacc = np.maximum(wacc, 1e-12)
        for c in range(nc):                                  # in place: a double-size colour master is large
            acc[c] /= wacc
            acc[c] = np.where(cover, acc[c], _upsample(regular[c], oh, ow))
        drz = acc
        log.append(f"Drizzle 2x (drop size {pf:g}): master is {ow} x {oh} px; "
                   f"{100 * (1 - cover.mean()):.1f}% of it had no drop and was filled from the regular stack"
                   + (" - dither between frames for drizzle to add detail" if cover.mean() < 0.97 else ""))
        if res.autocrop:
            res.autocrop = tuple(int(v) * sc for v in res.autocrop)
        return drz.astype(np.float32)

    # --- combine ---------------------------------------------------------------
    nc = len(channels)
    out = np.empty((nc, out_h, out_w), np.float32)
    rejected = 0

    def combine_cache(label="combining", f0=0.65, span=0.3):
        nonlocal rejected
        rejected = 0
        open_files_for(len(cache_files))
        mm = [np.load(p, mmap_mode="r") for p in cache_files]
        rows = int(max(8, min(out_h, 3e8 / max(len(mm) * out_w * 4 * 4, 1))))
        for c in range(nc):
            for r0 in range(0, out_h, rows):
                prog(f"{g.label()}: {label} {('channel ' + channels[c] + ', ') if nc > 1 else ''}"
                     f"rows {r0}/{out_h}", f0 + span * (c + r0 / out_h) / nc)
                nr = min(rows, out_h - r0)
                stack = np.empty((len(mm), nr, out_w), np.float32)
                for i, m in enumerate(mm):
                    stack[i] = m[c, r0:r0 + nr]
                out[c, r0:r0 + nr], rej = _robust_mean(stack, opts.kappa, weights, frame_noise)
                rejected += rej
        del mm

    if use_cache:
        rot = _rotation_spread([e[1] for e in entries])
        do_pattern = (getattr(opts, "pattern", True) and not rgb_in and not half and rot > 0.5
                      and plan.dark is None and len(entries) >= 10)
        combine_cache(f0=0.65, span=0.15 if do_pattern else 0.3)
        if do_pattern:
            # field rotation (alt-az mount) without darks: the camera's fixed pattern turns against the sky from
            # frame to frame and stacks into circular swirls. Measure it from the lights, take it out, stack again.
            try:
                sel = list(range(len(entries)))[::max(1, len(entries) // 40)][:40]
                pat_all = None
                for npass in range(PATTERN_PASSES):
                    model = [np.nan_to_num(out[c], nan=float(np.nanmedian(out[c][::16, ::16]))) for c in range(nc)]

                    def cal_iter():
                        for j, i in enumerate(sel):
                            prog(f"{g.label()}: measuring the camera pattern {j + 1}/{len(sel)}",
                                 0.8 + 0.05 * j / len(sel))
                            c_img = calibrated(entries[i][0])
                            yield c_img if pat_all is None else c_img - pat_all
                    pat, cover = sensor_pattern(cal_iter(), model, [entries[i][1] for i in sel],
                                                [entries[i][2] for i in sel], [entries[i][3] for i in sel],
                                                ref_bg, bayer, (out_h, out_w))
                    pat_all = pat if pat_all is None else pat_all + pat
                    pat_planes = planes(pat)
                    for i, p in enumerate(cache_files):
                        prog(f"{g.label()}: taking the camera pattern out {i + 1}/{len(cache_files)}",
                             0.85 + 0.05 * i / len(cache_files))
                        T, k = entries[i][1], entries[i][2]
                        if frame_models[i] is not None:
                            ap = warp_poly(pat_planes, frame_models[i], (out_h, out_w))
                        else:
                            ap = pat_planes if is_identity(T) else warp_planes(pat_planes, T, (out_h, out_w))
                        arr = np.load(p).astype(np.float32)
                        arr -= np.nan_to_num(np.stack(ap)) * np.float32(k)
                        np.save(p, np.clip(arr, -65000, 65000).astype(np.float16))
                    combine_cache("combining again", 0.9, 0.08)
                prms = float(np.std(pat_all[::3, ::3]))
                log.append(f"Field rotation of {rot:.1f}° between frames (alt-az mount): the camera's fixed pattern "
                           f"(measured from the lights, strength {prms:.1f} ADU) was taken out of every frame, so it "
                           "does not stack into circular swirls")
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001 - the ordinary stack is kept
                log.append(f"Camera-pattern removal skipped: {e}")
    else:
        mean, m2, cnt = welford
        sig = np.sqrt(m2 / np.maximum(cnt - 1, 1))
        sw = np.zeros_like(mean)
        swx = np.zeros_like(mean)
        for i, (info, T, k_phot, bgs, weight) in enumerate(entries):
            prog(f"{g.label()}: second pass {i + 1}/{len(entries)}", 0.65 + 0.3 * i / len(entries))
            img = calibrated(info)
            pls = planes(img)
            if frame_models[i] is not None:
                aligned = warp_poly(pls, frame_models[i], (out_h, out_w))
            else:
                aligned = pls if is_identity(T) else warp_planes(pls, T, (out_h, out_w))
            normed = np.stack(normalize(aligned, k_phot, bgs)[0])
            valid = ~np.isnan(normed)
            keep = valid & ((np.abs(normed - mean) <= opts.kappa * sig) | (cnt < 5))
            w = np.float32(weights[i]) * keep
            sw += w
            swx += np.nan_to_num(normed) * w
            rejected += int(valid.sum() - keep.sum())
        out[:] = np.where(sw > 0, swx / np.maximum(sw, 1e-12), np.nan_to_num(mean))
    out = np.nan_to_num(out)
    want_dz = getattr(opts, "drizzle", False)
    if want_dz and getattr(opts, "drizzle_if_dithered", False) and res.dither is not None \
            and not res.dither.get("dithered", True):
        want_dz = False
        log.append("Drizzle left out: these frames were hardly dithered, so it would add nothing but a 4x larger "
                   "master (Fast Track)")
    if want_dz and use_cache and not half:
        try:
            out = drizzle_stack(out)
        except Cancelled:
            raise
        except MemoryError:
            log.append("Drizzle skipped: not enough memory for a double-size master - the regular master is kept")
        except Exception as e:  # noqa: BLE001 - the regular master is kept
            log.append(f"Drizzle skipped: {e}")
    res.data = out[0] if nc == 1 else out
    if use_cache:
        try:
            import targets
            if opts.comet and targets.is_comet(g.target):
                comet_stack(res, out, entries, cache_files, weights, frame_noise, opts, cache_dir, prog)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - the ordinary master is never lost
            log.append(f"Comet stack skipped: {e}")
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    elif opts.comet:
        try:
            import targets
            if targets.is_comet(g.target):
                log.append("Comet: not enough free disk space for the second (comet-aligned) stack")
        except Exception:  # noqa: BLE001
            pass
    total = sum(1 for _ in entries) * out.size
    if n_mirrored:
        log.append(f"Note: {n_mirrored} frame(s) were mirror images of the reference (saved flipped, e.g. by a "
                   "different capture setting) — they were lined up anyway")
    log.append(f"Combined {len(entries)} frames (sigma clip k={opts.kappa:g}, "
               f"{'noise-weighted' if opts.weighting else 'equal weights'}); "
               f"{100 * rejected / max(total, 1):.3f}% of pixels rejected as outliers")
    log.append(f"Time: {time.time() - t_start:.0f} s")
    return res


# --------------------------------------------------------------------------
# Comets: a second stack that follows the comet, and the two combined
# --------------------------------------------------------------------------
def _time_s(date_obs):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(date_obs).replace("Z", "")).timestamp()
    except (TypeError, ValueError):
        return None


def comet_position(lum: np.ndarray, guess=None, search=None):
    """Where the comet's head is in a star-aligned frame: the stars are taken away (an 'opening'),
    each fuzzy blob is measured against its own surroundings (so gradients and frame edges can't win),
    the outer edge of the frame is ignored and the middle - where the target was put - is gently
    preferred. Returns ((x, y), significance) in full-size pixels; significance = how many times
    the blob stands above the noise (below ~5 it is not reliably the comet)."""
    h, w = lum.shape
    f = max(1, int(np.ceil(max(h, w) / 1000)))
    small = _bin(np.nan_to_num(lum, nan=float(np.nanmedian(lum))), f)
    op = -_minf(-_minf(small, 5), 5)                       # stars (even bright ones) gone, the coma stays
    bl = _smooth(op, 3.0)
    bg = _smooth(_minf(bl, 12), 30.0)                      # the local sky around each blob
    resid = bl - bg
    sh, sw = resid.shape
    dif = (bl[:, 1:] - bl[:, :-1]).ravel()
    noise = float(np.median(np.abs(dif - np.median(dif)))) * 1.4826 + 1e-9
    yy, xx = np.mgrid[:sh, :sw]
    score = resid.copy()
    m = 0.06
    score[(yy < m * sh) | (yy > (1 - m) * sh) | (xx < m * sw) | (xx > (1 - m) * sw)] = -np.inf
    if guess is not None:
        gx, gy = guess[0] / f, guess[1] / f
        r = max(6, (search or 0.06 * max(h, w)) / f)
        score = np.where((xx - gx) ** 2 + (yy - gy) ** 2 <= r * r, score, -np.inf)
    else:
        d2 = ((xx - sw / 2) / (sw / 2)) ** 2 + ((yy - sh / 2) / (sh / 2)) ** 2
        score = score * (1.0 - 0.35 * np.clip(d2, 0, 1))     # gently prefer the middle of the frame
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    peak = float(resid[iy, ix])
    y0, y1 = max(0, iy - 6), min(sh, iy + 7)
    x0, x1 = max(0, ix - 6), min(sw, ix + 7)
    win = np.clip(resid[y0:y1, x0:x1] - 0.5 * peak, 0, None)
    sig = peak / (noise * 0.25)                             # noise after the smoothing
    if win.sum() <= 0:
        return ((ix + 0.5) * f, (iy + 0.5) * f), sig
    wy, wx = np.mgrid[y0:y1, x0:x1]
    return (((wx * win).sum() / win.sum() + 0.5) * f, ((wy * win).sum() / win.sum() + 0.5) * f), sig


def comet_stack(res, stars_out, entries, cache_files, weights, frame_noise, opts, cache_dir, prog):
    """Follow the comet: find its head in every star-aligned frame, fit its straight-line motion over
    time, shift each frame so the comet stays put and combine again (now the stars move and are
    rejected). The comet image and the star image are then joined: stars from the star stack, the
    comet (head and tail) from the comet stack. Saves 'Comet' and 'Stars' masters next to the main one."""
    log = res.log
    times = [_time_s(e[0].date_obs) for e in entries]
    if any(t is None for t in times) or len(entries) < 3:
        log.append("Comet: the frames have no capture times (DATE-OBS), so the comet's motion cannot be followed")
        return
    t = np.array(times) - min(times)
    span = float(t.max())
    lum_all = stars_out.mean(axis=0) if stars_out.ndim == 3 else stars_out
    guess, sig0 = comet_position(lum_all)
    if sig0 < 5:
        log.append(f"Comet: it is too faint to find automatically in this stack (signal {sig0:.1f}x the noise) - "
                   "only the ordinary stack is saved. Longer exposures or more frames help.")
        return
    pos = []
    for i, cf in enumerate(cache_files):
        prog("Comet: finding the comet in each frame", 0.95 + 0.01 * i / len(cache_files))
        fr = np.load(cf, mmap_mode="r")
        lum = np.asarray(fr, dtype=np.float32).mean(axis=0)
        (x, y), sg = comet_position(lum, guess, 0.05 * max(lum.shape))
        pos.append((x, y, 1.0 if sg >= 3 else 0.0))
    P = np.array(pos)
    A = np.stack([np.ones_like(t), t], 1)
    keep = P[:, 2] > 0
    for _ in range(3):
        if keep.sum() < 3:
            log.append("Comet: could not find the comet in enough frames - only the ordinary stack is saved")
            return
        cx, *_r = np.linalg.lstsq(A[keep], P[keep, 0], rcond=None)
        cy, *_r = np.linalg.lstsq(A[keep], P[keep, 1], rcond=None)
        rx, ry = P[:, 0] - A @ cx, P[:, 1] - A @ cy
        d = np.hypot(rx, ry)
        sd = float(np.median(d[keep])) * 1.4826 + 0.3
        keep = keep & (d < 3 * sd + 1.0)
    vx, vy = float(cx[1]), float(cy[1])                     # px per second
    fit_rms = float(np.sqrt(np.mean(d[keep] ** 2))) if keep.any() else 99.0
    if keep.sum() < max(3, 0.6 * len(entries)) or fit_rms > 3.0:
        log.append(f"Comet: its position could not be followed reliably (found in {int(keep.sum())} of "
                   f"{len(entries)} frames, scatter {fit_rms:.1f} px) - only the ordinary stack is saved")
        return
    moved = float(np.hypot(vx, vy) * span)
    rate = float(np.hypot(vx, vy) * 3600)
    log.append(f"Comet: head found in {int(keep.sum())} of {len(entries)} frames; it moved {moved:.1f} px over "
               f"{span / 60:.0f} min ({rate:.0f} px per hour)")
    if moved < 3.0:
        log.append("Comet: it barely moved during these frames, so the ordinary stack is already sharp for both "
                   "the stars and the comet - no separate comet stack is needed")
        return
    # second pass: shift every frame so the comet stays where it is in the middle of the session
    t_mid = float(np.median(t))
    nc = stars_out.shape[0] if stars_out.ndim == 3 else 1
    out_h, out_w = stars_out.shape[-2:]
    # the stars, masked out of each frame before it is lined up on the comet: in 25 short frames the rejection
    # alone left every faint star as a dotted trail across the comet's tail (the comet of 16 April 2026)
    star_m, star_part = _comet_star_mask(stars_out, cx, cy, span)
    shifted = []
    for i, cf in enumerate(cache_files):
        prog("Comet: lining the frames up on the comet", 0.96 + 0.02 * i / len(cache_files))
        fr = np.array(np.load(cf, mmap_mode="r"), dtype=np.float32)
        if star_m is not None:
            fr[:, star_m] = np.nan
        dx, dy = vx * (t[i] - t_mid), vy * (t[i] - t_mid)
        T = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        pl = warp_planes(list(fr), T, (out_h, out_w))
        p = cache_dir / f"c{i:05d}.npy"
        np.save(p, np.stack([np.clip(x, -65000, 65000).astype(np.float16) for x in pl]))
        shifted.append(p)
        del fr, pl
    open_files_for(len(shifted))
    mm = [np.load(p, mmap_mode="r") for p in shifted]
    comet = np.empty((nc, out_h, out_w), np.float32)
    rows = int(max(8, min(out_h, 3e8 / max(len(mm) * out_w * 4 * 4, 1))))
    kappa = min(opts.kappa, 2.5)                           # the trailing stars are the outliers now
    for c in range(nc):
        for r0 in range(0, out_h, rows):
            nr = min(rows, out_h - r0)
            stack = np.empty((len(mm), nr, out_w), np.float32)
            for i, m in enumerate(mm):
                stack[i] = m[c, r0:r0 + nr]
            comet[c, r0:r0 + nr], _rej = _robust_mean(stack, kappa, weights, frame_noise)
    del mm
    comet = np.nan_to_num(comet)
    # join: the comet (head + tail) from the comet stack, everything else from the star stack
    lum_c = comet.mean(axis=0)
    f = max(1, int(np.ceil(max(out_h, out_w) / 1000)))
    small = _bin(lum_c, f)
    op = _smooth(-_minf(-_minf(small, 3), 3), 3.0)
    bg = float(np.median(op))
    dif = (small[:, 1:] - small[:, :-1]).ravel()
    noise = float(np.median(np.abs(dif - np.median(dif)))) * 1.4826 / np.sqrt(2) + 1e-6
    m = np.clip((op - bg - 2.0 * noise * 0.3) / (4.0 * noise * 0.3), 0, 1)
    m = _smooth(-_minf(-m, 2), 3.0)
    M = _upsample(np.clip(m, 0, 1), out_h, out_w)[None]
    # near the edges the comet stack is made of fewer frames (each was shifted by up to half the comet's motion):
    # there the star stack is used, and the suggested crop leaves that strip out (it made blue bands and blobs)
    edge = int(np.ceil(0.5 * moved)) + 4
    M[..., :edge, :] = 0
    M[..., -edge:, :] = 0
    M[..., :, :edge] = 0
    M[..., :, -edge:] = 0
    if getattr(res, "autocrop", None):
        x0, y0, x1, y1 = res.autocrop
        res.autocrop = (max(x0, edge), max(y0, edge), min(x1, out_w - edge), min(y1, out_h - edge))
    else:
        res.autocrop = (edge, edge, out_w - edge, out_h - edge)
    stars3 = stars_out if stars_out.ndim == 3 else stars_out[None]
    # the comet stack has no stars now: inside the comet's area they come from the star stack, on top of it
    comet_st = comet + star_part if star_part is not None else comet
    joined = stars3 * (1 - M) + comet_st * M
    res.extra = {"Comet": comet[0] if nc == 1 else comet, "Stars": stars3[0] if nc == 1 else stars3.copy()}
    res.data = joined[0] if nc == 1 else joined
    res.comet = {"vx": vx, "vy": vy, "moved": moved, "rate": rate}
    log.append(f"Comet: a second stack follows the comet ({kappa:g}-sigma rejection removes the trailing stars); "
               "the main master joins the sharp stars and the sharp comet. The comet-only and stars-only masters "
               "are saved next to it (_Comet, _Stars).")


def _comet_star_mask(stars_out, cx, cy, span):
    """(mask, stars) in the star-aligned frame: where the stars are (small, bright, round things above the local
    sky, grown by a few pixels), and the stars themselves above that sky (to put them back on the comet stack).
    Nothing near the comet head's track is masked (its nucleus is compact too). (None, None) if it can't tell."""
    try:
        st3 = stars_out if stars_out.ndim == 3 else stars_out[None]
        L = st3.mean(axis=0).astype(np.float32)
        h, w = L.shape
        loc = _smooth(L, 5.0)
        hp = L - loc
        noise = float(np.median(np.abs(hp[::4, ::4] - np.median(hp[::4, ::4])))) * 1.4826 + 1e-9
        core_ = _box((hp > 4.0 * noise).astype(np.float32), 1) > 0.5      # 3x3 majority: a star, not a noise pixel
        m = _box(core_.astype(np.float32), 4) > 0.02                    # grown by about 4 px
        # keep the comet head's path free
        yy, xx = np.mgrid[0:h:4, 0:w:4]
        ts = np.linspace(0.0, max(span, 1.0), 12)
        hx, hy = cx[0] + cx[1] * ts, cy[0] + cy[1] * ts
        d = np.min(np.hypot(xx[None] - hx[:, None, None], yy[None] - hy[:, None, None]), axis=0)
        free = np.repeat(np.repeat(d < max(40.0, 0.015 * max(h, w)), 4, axis=0), 4, axis=1)[:h, :w]
        m &= ~free
        if m.mean() > 0.25:                                             # not stars: something else is going on
            return None, None
        part = np.stack([(c - _smooth(c.astype(np.float32), 5.0)) for c in st3]) * m[None]
        return m, np.clip(part, 0, None).astype(np.float32)
    except Exception:  # noqa: BLE001 - without the mask the comet stack is made as before
        return None, None


def _compose(A, B):
    """A ∘ B for 2x3 affine matrices (apply B first)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def _welford_add(state, normed):
    x = np.stack(normed)
    valid = ~np.isnan(x)
    if state is None:
        mean = np.where(valid, x, 0).astype(np.float32)
        m2 = np.zeros_like(mean)
        cnt = valid.astype(np.float32)
        return mean, m2, cnt
    mean, m2, cnt = state
    cnt += valid
    delta = np.where(valid, x - mean, 0)
    mean += delta / np.maximum(cnt, 1)
    m2 += np.where(valid, delta * (x - mean), 0)
    return mean, m2, cnt


_HINT_KEYS = ("RA", "DEC", "OBJCTRA", "OBJCTDEC", "EQUINOX", "RADESYS", "SITELAT", "SITELONG", "SITEELEV",
              "XBINNING", "YBINNING")
_WCS_KEYS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
             "CDELT1", "CDELT2", "CROTA1", "CROTA2")


def reference_keywords(ref_path, half=False) -> list:
    """Pointing, pixel size and (if present) the linear WCS of the reference frame,
    so plate solvers (PixInsight ImageSolver, ASTAP …) start from the right place.
    The master has the reference frame's pixel grid, so its WCS still applies."""
    try:
        look = ImageSource(ref_path).look
    except Exception:  # noqa: BLE001
        return []
    out = [(k, look[k], "from reference frame") for k in _HINT_KEYS if look.get(k) not in (None, "")]
    f = 2.0 if half else 1.0
    if not look.get("FOCALLEN") and _num(look.get("FOCAL")):            # Vespera writes FOCAL
        out.append(("FOCALLEN", _num(look.get("FOCAL")), "[mm] from reference frame"))
    for k in ("XPIXSZ", "YPIXSZ"):
        v = _num(look.get(k)) or _num(look.get("PIXSZ"))
        if v:
            out.append((k, v * f, "[um] effective pixel size" + (" (half-resolution colour)" if half else "")))
    if all(look.get(k) not in (None, "") for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")):
        out.append(("CTYPE1", str(look["CTYPE1"]).replace("-SIP", ""), "from reference frame (approximate)"))
        out.append(("CTYPE2", str(look["CTYPE2"]).replace("-SIP", ""), ""))
        for k in _WCS_KEYS:
            v = _num(look.get(k))
            if v is None:
                continue
            if half and k in ("CRPIX1", "CRPIX2"):
                v = (v - 1 - 0.5) / 2 + 1
            elif half and (k.startswith("CD") or k.startswith("CDELT")):
                v = v * 2
            out.append((k, v, ""))
    return out


# --------------------------------------------------------------------------
# Resume after a crash (0.6): remember each master as soon as it is saved
# --------------------------------------------------------------------------

def _group_key(g, opts) -> str:
    plan = g.plan or CalPlan()
    cal = []
    for c in (plan.bias, plan.dark, plan.flat):
        cal.append(sorted(str(i.path) for i in c.infos) if c is not None else None)
    for n_ in sorted(plan.night_flats):
        cal.append((n_, sorted(str(i.path) for i in plan.night_flats[n_][0].infos)))
    extra = repr((g.label(), cal, opts.kappa, opts.weighting, opts.osc_half, tuple(opts.formats),
                  os.path.normcase(str(opts.masters_root)), opts.align_common, getattr(opts, "cosmetic", True),
                  getattr(opts, "local_norm", True), getattr(opts, "drizzle", False)))
    return cp.file_key([f.info.path for f in g.frames], extra)


def _cal_key(item) -> str:
    return cp.file_key([i.path for i in item.infos], item.kind)


class StackRunLog:
    """Masters (and calibration masters) saved by a stacking run that did not finish,
    so the next run can skip them. Cleared when a run completes."""

    def __init__(self, opts):
        self.opts = opts
        self.written = []
        self._keys = {}                     # keys are fixed on first use (calibration may be swapped later)

    def _gpath(self, g):
        if id(g) not in self._keys:
            self._keys[id(g)] = _group_key(g, self.opts)
        return cp.root() / f"stack_{self._keys[id(g)]}.json"

    def _cpath(self, item):
        return cp.root() / f"stackcal_{_cal_key(item)}.json"

    def done(self, g):
        """The saved entry for this group if all its files still exist, else None."""
        d = cp.read_json(self._gpath(g))
        if d and d.get("files") and all(Path(f).exists() for f in d["files"]):
            return d
        return None

    def mark(self, g, res):
        p = self._gpath(g)
        cp.write_json(p, {"label": g.label(), "files": [str(f) for f in res.files], "used": res.used,
                          "time": time.strftime("%Y-%m-%d %H:%M")})
        self.written.append(p)

    def cal_saved(self, item):
        d = cp.read_json(self._cpath(item))
        files = [f for f in (d or {}).get("files", []) if Path(f).exists()]
        return files or None

    def mark_cal(self, item, files):
        p = self._cpath(item)
        cp.write_json(p, {"label": item.label(), "files": [str(f) for f in files]})
        self.written.append(p)

    def finish(self, groups=()):
        """The run completed: forget it (a later run stacks again, as _2 versions)."""
        for g in groups:
            cp.remove(self._gpath(g))
        for p in self.written:
            cp.remove(p)
        self.written = []


def use_saved_calibration(item, files) -> bool:
    """Swap a calibration item built from subframes for the master an interrupted run
    already saved (much faster than rebuilding it)."""
    f = next((Path(x) for x in files if Path(x).suffix.lower() in (".fits", ".fit", ".xisf")), None)
    if f is None:
        return False
    try:
        info = read_info(f)
    except Exception:  # noqa: BLE001
        return False
    item.replaced = [str(i.path) for i in item.infos]      # the subframes it stands in for
    item.infos, item.master, item.origin, item.data = [info], True, "library", None
    item.saved = [Path(x) for x in files]
    return True


def acquisition_keywords(g, res, header, half=False) -> list:
    """Per-night acquisition summary, kept in the master (and every file made from it) so the
    final picture can be shared to AstroBin with its acquisition details filled in.
      ACQ01 = 'night|subs|sub exposure s|gain|sensor C|FWHM arcsec|binning'
      ACQCAL = 'darks|flats|flatDarks|bias'"""
    skipped = {s for s, _ in res.skipped}
    look = {k: v for k, v, _c in header}
    pix = _num(look.get("XPIXSZ"))
    focal = _num(look.get("FOCALLEN"))
    scale = 206.265 * (pix / (2 if half else 1)) / focal if pix and focal else None
    nights = {}
    for f in g.frames:
        i = f.info
        if i.path.name in skipped:
            continue
        n = int(i.stackcnt or 1)
        sub = (i.exposure_total / n) if n > 1 else (i.exptime or 0)
        key = (i.night or "", round(float(sub), 3), i.gain, i.binning or 1)
        d = nights.setdefault(key, {"n": 0, "temps": [], "fwhm": []})
        d["n"] += n
        if i.temp is not None:
            d["temps"].append(i.temp)
        if f.fwhm and scale:
            d["fwhm"].append(f.fwhm * scale)
    out = []

    def fmt(v, nd=2):
        return "" if v is None else f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    for j, (key, d) in enumerate(sorted(nights.items(), key=lambda kv: kv[0][0]), 1):
        if j > 99:
            break
        night, sub, gain, b = key
        temp = round(float(np.mean(d["temps"]))) if d["temps"] else None
        fw = float(np.mean(d["fwhm"])) if d["fwhm"] else None
        out.append((f"ACQ{j:02d}", "|".join([night, str(d["n"]), fmt(sub, 4), fmt(gain), fmt(temp, 0), fmt(fw),
                                             str(int(b))]), "night|subs|exp|gain|cool|fwhm|bin"))
    plan = g.plan or CalPlan()
    darks = plan.dark.count if plan.dark else 0
    flats = plan.flat.count if plan.flat else 0
    fdarks = plan.flat_dark.count if plan.flat_dark else 0
    bias = (plan.bias.count if plan.bias else 0) or (plan.flat_bias.count if plan.flat_bias else 0)
    out.append(("ACQCAL", f"{darks}|{flats}|{fdarks}|{bias}", "darks|flats|flatDarks|bias"))
    return out


def save_master_light(res: StackResult, opts: StackOptions, multi_camera=False) -> list:
    g = res.group
    i0 = g.ref
    name = master_light_name(g, multi_camera)
    folder = Path(opts.masters_root or i0.path.parent) / safe_name(g.target)     # 'C/2023 A3' has a slash
    base = _unique_base(folder / name, opts.formats)
    infos = [f.info for f in g.frames]
    dates = sorted(i.date_obs for i in infos if i.date_obs)
    temps = [i.temp for i in infos if i.temp is not None]
    plan = g.plan or CalPlan()
    header = [("IMAGETYP", "Master Light", ""), ("OBJECT", g.target or i0.target, ""), ("FILTER", g.filter, ""),
              ("INSTRUME", g.camera, ""), ("TELESCOP", g.rig, "rig name"), ("FOCALLEN", i0.focal, "[mm]"),
              ("EXPTIME", i0.exptime, "[s] subframe exposure"),
              ("EXPTOTAL", round(sum(f.info.exposure_total for f in g.frames
                                     if f.info.path.name not in {s for s, _ in res.skipped}), 1),
               "[s] total exposure combined"),
              ("NCOMBINE", res.used, "files combined"), ("GAIN", i0.gain, ""),
              ("STACKCNT", sum((f.info.stackcnt or 1) for f in g.frames
                               if f.info.path.name not in {s for s, _ in res.skipped})
               if any(f.info.stackcnt for f in g.frames) else None, "subframes inside the combined stacks"),
              ("CCD-TEMP", round(float(np.mean(temps)), 1) if temps else None, "[C] mean"),
              ("DATE-OBS", dates[0] if dates else None, "first frame"),
              ("DATE-END", dates[-1] if dates else None, "last frame"),
              ("ROWORDER", i0.roworder, ""),
              ("CALBIAS", plan.bias.ref.path.name if plan.bias and plan.bias.master else
               (f"{plan.bias.count} bias subframes" if plan.bias else "none"), ""),
              ("CALDARK", (plan.dark.ref.path.name if plan.dark.master else f"{plan.dark.count} dark subframes")
               if plan.dark else "none", plan.dark_mode if plan.dark else ""),
              ("CALFLAT", (f"per-night flats ({len(plan.night_flats)} nights)" if plan.night_flats else
                           plan.flat.ref.path.name if plan.flat.master else f"{plan.flat.count} flat subframes")
               if plan.flat else "none", ""),
              ("PRECAL", str(plan.precal)[:60] if plan.precal else None, "lights were already calibrated"),
              ("STACKREJ", f"sigma clip k={opts.kappa:g}", ""),
              ("STACKWGT", "noise" if opts.weighting else "equal", ""),
              ("SWCREATE", "SubZeroAstro", "")]
    if g.ref.bayer:
        header.append(("COLORSPC", "RGB", "debayered" + (" (superpixel)" if opts.osc_half else " (bilinear)")))
    header += reference_keywords(res.ref_path or g.ref.path, half=bool(g.ref.bayer) and opts.osc_half)
    try:
        header += acquisition_keywords(g, res, header, half=bool(g.ref.bayer) and opts.osc_half)
    except Exception:  # noqa: BLE001 - a summary must never stop a master being saved
        pass
    if res.common_ref:
        header.append(("ALIGNREF", res.common_ref, "all masters of this target share this grid"))
    if res.autocrop:
        header.append(("AUTOCROP", " ".join(str(v) for v in res.autocrop), "x0 y0 x1 y1 covered by 90% of frames"))
    header += [("HISTORY", line, "") for line in res.log[:40]]
    files = save_master_file(base, res.data, header, opts.formats, "MasterLight")
    for tag, arr in (getattr(res, "extra", None) or {}).items():
        files += save_master_file(base.with_name(base.name + "_" + tag), arr,
                                  header + [("COMETSTK", tag, "comet stack: " + ("comet-aligned" if tag == "Comet"
                                                                                 else "star-aligned"))],
                                  opts.formats, "MasterLight")
    log_path = base.with_name(base.name + "_stacklog.txt")
    lines = [f"{name}", "=" * len(name), "", *res.log, "", "Frames combined:"]
    skipped = {s for s, _ in res.skipped}
    for f in g.frames:
        if f.info.path.name in skipped:
            continue
        lines.append(f"  {f.info.path}  ({f.info.date_obs or ''}"
                     + (f", FWHM {f.fwhm:.2f}" if f.fwhm else "") + ")")
    if res.skipped:
        lines += ["", "Skipped:"] + [f"  {a}: {b}" for a, b in res.skipped]
    lines += ["", "Saved:"] + [f"  {p}" for p in files]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res.files = files + [log_path]
    return res.files
