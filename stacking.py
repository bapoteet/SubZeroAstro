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
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import checkpoints as cp
import fits_core as core
import metrics as M
import xisf

IMAGE_EXTS = core.FITS_EXTS | {".xisf"}


class Cancelled(Exception):
    pass


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
            # PixInsight stores floating-point images in the range 0..1
            if self.look.get("_XISF_FORMAT", "").startswith("Float") and \
                    str(self.look.get("_XISF_BOUNDS", "")).startswith("0:1"):
                self.scale = 65535.0
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
            elif self.lay.bitpix < 0 and "master" in str(self.look.get("IMAGETYP", "")).lower():
                self.scale = None            # decide from the data (0..1 or ADU)

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


def info_from_look(path, look, w, h, typ=None, master=False, planes=1) -> ImgInfo:
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
        binning=int(xb) if xb else None, bayer=fi.get("bayer"),
        night=core.night_of(fi), date_obs=(str(fi.get("date_obs")) if fi.get("date_obs") else None),
        focal=_num(look.get("FOCALLEN")), telescope=fi.get("telescope"),
        rotator=_num(fi.get("rotator")),
        target=(str(fi["object"]).strip() if fi.get("object") else None),
        roworder=(str(look.get("ROWORDER")).strip().upper() if look.get("ROWORDER") else None),
        ncombine=int(look["NCOMBINE"]) if isinstance(look.get("NCOMBINE"), int) else None,
        planes=int(planes or 1), stackcnt=int(_num(fi.get("stackcnt"))) if _num(fi.get("stackcnt")) else None,
        totalexp=_num(fi.get("totalexp")), smart=fi.get("smart"))


def read_info(path) -> ImgInfo:
    src = ImageSource(path)
    typ, master = frame_type(src.look, src.path.name)
    return info_from_look(path, src.look, src.w, src.h, typ, master, planes=src.channels)


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


def target_name(t: str | None) -> str:
    return safe_name((t or "Unknown").replace(" ", "")) if t else "Unknown"


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
        mountish = re.search(r"\b(AM\d|EQ\d|HEQ|CEM|GEM|RST|mount|AZ-|iOptron|Rainbow|Sky-?Watcher\s+EQ)", t, re.I)
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


def scan_paths(paths, progress=None):
    """Read headers of all FITS/XISF files in the given files/folders (recursive).
    Returns (infos, problems)."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d.upper() != core.REJECTED_DIR]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                        files.append(Path(dirpath) / fn)
        elif p.suffix.lower() in IMAGE_EXTS:
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
    if not root or not root.is_dir():
        return []
    items = []
    for rig_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mdir = rig_dir / "Masters"
        if not mdir.is_dir():
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


def group_lights(lights: list[LightFrame], rigs: Rigs) -> list[LightGroup]:
    groups = {}
    for lf in lights:
        i = lf.info
        key = (target_name(i.target), i.filter, (i.camera or "").lower(), bool(i.bayer) or i.planes > 1,
               i.w, i.h, i.x0, i.y0, i.binning, rigs.name_for(i))
        groups.setdefault(key, []).append(lf)
    out = []
    for key, lst in groups.items():
        lst.sort(key=lambda f: (f.info.date_obs or "", str(f.info.path)))
        out.append(LightGroup(key[0], key[1], lst[0].info.camera, key[9], lst))
    out.sort(key=lambda g: (g.target, g.camera or "", g.filter))
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

    def missing(self):
        return [k for k, _ in self.warnings]


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


def plan_group(g: LightGroup, items: list[CalItem]) -> CalPlan:
    L = g.ref
    p = CalPlan()
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
        if not p.flat.master:
            fl = p.flat.ref
            fdarks = [c for c in items if c.kind in ("darkflat", "dark") and _same_cam(c.camera, fl.camera)
                      and c.exptime is not None and fl.exptime is not None
                      and abs(c.exptime - fl.exptime) <= 0.05 * max(fl.exptime, 0.01)
                      and _eq(c.gain, fl.gain) and geometry_ok(c, fl)]
            fbias = [c for c in items if c.kind == "bias" and _same_cam(c.camera, fl.camera)
                     and _eq(c.gain, fl.gain) and _eq(c.offset, fl.offset) and geometry_ok(c, fl)]
            p.flat_dark = pick(fdarks, lambda c: (unknown(c), -c.count))
            if p.flat_dark is None:
                p.flat_bias = pick(fbias, lambda c: (unknown(c), -c.count))
            if p.flat_dark is None and p.flat_bias is None:
                p.warnings.append(("flatbias", f"The flats (gain {fl.gain if fl.gain is not None else '?'}) have "
                                               "no matching bias or dark-flats, so the camera offset stays in "
                                               "them and the correction will be too strong (dark rings / "
                                               "bright centre)."))
    else:
        p.warnings.append(("flat", f"No flats for filter {L.filter} on {g.rig} — vignetting and dust "
                                   "shadows will not be corrected."))
    if L.smart or L.planes > 1:
        # smart telescopes calibrate on board (or the input is already a stack): missing
        # calibration is normal here, so don't stop the user with warnings
        if p.warnings:
            p.notes.append(f"{L.smart or 'Stacked colour input'}: no separate calibration frames needed "
                           "(add darks/flats from the scope if you have them)")
        if L.planes > 1:
            p.bias = p.dark = p.flat = p.flat_bias = p.flat_dark = None
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
    sources = [ImageSource(i.path) for i in item.infos]
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


def luminance(cal: np.ndarray, bayer: bool) -> tuple[np.ndarray, int]:
    """Image used for finding stars, and its scale factor to full resolution."""
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


def match_frames(ref_pts, tgt_pts, tol=3.0):
    """Transform T (2x3) mapping target coordinates to reference coordinates,
    plus (inliers, rms). Raises ValueError if the frames cannot be matched."""
    if len(ref_pts) < 6 or len(tgt_pts) < 6:
        raise ValueError("too few stars")
    R, T_ = ref_pts[:40], tgt_pts[:40]
    inv_r, tri_r = _triangles(R)
    inv_t, tri_t = _triangles(T_)
    if len(inv_r) == 0 or len(inv_t) == 0:
        raise ValueError("too few stars")
    # nearest reference triangle for every target triangle
    dists = np.sqrt(((inv_t[:, None, :] - inv_r[None, :, :]) ** 2).sum(-1))
    best = np.argmin(dists, axis=1)
    good = np.nonzero(dists[np.arange(len(best)), best] < 0.03)[0]
    if good.size == 0:
        raise ValueError("no matching star patterns")
    good = good[np.argsort(dists[good, best[good]])][:300]

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
        T = _fit_similarity(src, dst)
        sc = math.hypot(T[0, 0], T[1, 0])
        if not 0.8 < sc < 1.25:
            continue
        ok, _ = count_inliers(T)
        n = int(ok.sum())
        if n > best_n:
            best_T, best_n = T, n
            if n > 0.8 * min(len(tgt_pts), len(ref_pts)):
                break
    if best_T is None or best_n < 6:
        raise ValueError("could not match the star field")
    # refine with all inliers (affine when there are plenty)
    T = best_T
    for _ in range(3):
        ok, j = count_inliers(T)
        src, dst = tgt_pts[ok], ref_pts[j[ok]]
        if len(src) < 6:
            break
        T = _fit_affine(src, dst) if len(src) >= 20 else _fit_similarity(src, dst)
    ok, j = count_inliers(T)
    res = _apply(T, tgt_pts[ok]) - ref_pts[j[ok]]
    rms = float(np.sqrt((res ** 2).sum(1).mean())) if ok.any() else 99.0
    return T, int(ok.sum()), rms, ok, j


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
    parts = [g.target]
    if g.filter and g.filter != "No filter":
        parts.append(safe_name(core._ascii(g.filter)))
    if multi_camera:
        parts.append(short_camera(g.camera))
    parts.append("Master")
    n = g.nights
    if n:
        parts.append(n[0] if len(n) == 1 else f"{n[0]}_to_{n[-1]}")
    return "_".join(parts)


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
    log.append(f"Group: {g.label()}  ({len(frames)} frames, {g.total_exposure / 3600:.2f} h)")
    log.append(f"Reference frame: {ref_info.path.name}")
    log.append("Calibration: " + "; ".join(
        f"{name} = {it.label()}" for name, it in (("bias", plan.bias), ("dark", plan.dark), ("flat", plan.flat))
        if it is not None) + (f"  [dark {plan.dark_mode}]" if plan.dark else ""))
    for k, w in plan.warnings:
        log.append(f"WARNING: {w}")
    for n in plan.notes:
        log.append(f"Note: {n}")

    def calibrated(info):
        raw = ImageSource(info.path).read()
        if rgb_in:
            if raw.ndim != 3 or raw.shape[0] != 3:
                raise ValueError("not a 3-plane colour image like the reference")
            return raw.astype(np.float32, copy=False)      # stacked colour: no calibration, no debayer
        if raw.ndim == 3:
            raw = raw[0]
        return cal.apply(raw)

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
    if common and common.get("name") != ref_info.path.name:
        try:
            T_g2c, inl, rms, _ok, _j = match_frames(common["pts"], ref_pts)
            log.append(f"Aligned to the common reference {common['name']} (stars matched {inl}, rms {rms:.2f} px) "
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

    def prepare(idx, lf, write):
        info = lf.info
        img = calibrated(info)
        lum, fac = lumin(img)
        pts, peaks = find_stars(lum, fac, robust=robust)
        if info is ref_info:
            T = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            k_phot = 1.0
            inl, rms = len(pts), 0.0
        else:
            T, inl, rms, ok, j = match_frames(ref_pts, pts)
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
        pls = planes(img)
        del img
        noises = [pixel_noise(p) for p in pls]
        if is_identity(Tout):
            aligned = [p.astype(np.float32, copy=False) for p in pls]
        else:
            aligned = warp_planes(pls, Tout, (out_h, out_w))
        del pls
        bgs = []
        for a in aligned:
            sm = a[::8, ::8]
            bgs.append(float(np.median(sm[~np.isnan(sm)])))
        return aligned, Tout, k_phot, bgs, noises, inl, rms

    def normalize(aligned, k_phot, bgs):
        return [((a - bgs[c]) * np.float32(k_phot) + np.float32(ref_bg[c])) for c, a in enumerate(aligned)]

    n = len(frames)
    used = []
    cache_files = []
    frame_noise = []
    welford = None
    for idx, lf in enumerate(frames):
        prog(f"{g.label()}: calibrating and aligning {idx + 1}/{n}", 0.05 + 0.6 * idx / n)
        try:
            aligned, T, k_phot, bgs, noises, inl, rms = prepare(idx, lf, True)
        except ValueError as e:
            res.skipped.append((lf.info.path.name, str(e)))
            log.append(f"SKIPPED {lf.info.path.name}: {e}")
            continue
        if ref_bg is None:
            ref_bg = bgs
        normed = normalize(aligned, k_phot, bgs)
        noise = float(np.mean(noises)) * k_phot
        weight = 1.0 / noise ** 2 if opts.weighting else 1.0
        entries.append((lf.info, T, k_phot, bgs, weight))
        frame_noise.append(noise)
        log.append(f"  {lf.info.path.name}: stars matched {inl}, rms {rms:.2f} px, scale {k_phot:.3f}, "
                   f"noise {noise:.1f}" + (f", score {lf.score:.0f}" if lf.score is not None else ""))
        if use_cache:
            p = cache_dir / f"f{len(used):05d}.npy"
            np.save(p, np.stack([np.clip(x, -65000, 65000).astype(np.float16) for x in normed]))
            cache_files.append(p)
        else:
            welford = _welford_add(welford, normed)
        used.append(lf)
    if not entries:
        raise ValueError("No frames could be aligned")
    res.used = len(entries)
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

    # --- combine ---------------------------------------------------------------
    nc = len(channels)
    out = np.empty((nc, out_h, out_w), np.float32)
    rejected = 0
    if use_cache:
        mm = [np.load(p, mmap_mode="r") for p in cache_files]
        rows = int(max(8, min(out_h, 3e8 / max(len(mm) * out_w * 4 * 4, 1))))
        for c in range(nc):
            for r0 in range(0, out_h, rows):
                prog(f"{g.label()}: combining {('channel ' + channels[c] + ', ') if nc > 1 else ''}"
                     f"rows {r0}/{out_h}", 0.65 + 0.3 * (c + r0 / out_h) / nc)
                nr = min(rows, out_h - r0)
                stack = np.empty((len(mm), nr, out_w), np.float32)
                for i, m in enumerate(mm):
                    stack[i] = m[c, r0:r0 + nr]
                out[c, r0:r0 + nr], rej = _robust_mean(stack, opts.kappa, weights, frame_noise)
                rejected += rej
        del mm
        shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        mean, m2, cnt = welford
        sig = np.sqrt(m2 / np.maximum(cnt - 1, 1))
        sw = np.zeros_like(mean)
        swx = np.zeros_like(mean)
        for i, (info, T, k_phot, bgs, weight) in enumerate(entries):
            prog(f"{g.label()}: second pass {i + 1}/{len(entries)}", 0.65 + 0.3 * i / len(entries))
            img = calibrated(info)
            pls = planes(img)
            aligned = pls if is_identity(T) else warp_planes(pls, T, (out_h, out_w))
            normed = np.stack(normalize(aligned, k_phot, bgs))
            valid = ~np.isnan(normed)
            keep = valid & ((np.abs(normed - mean) <= opts.kappa * sig) | (cnt < 5))
            w = np.float32(weights[i]) * keep
            sw += w
            swx += np.nan_to_num(normed) * w
            rejected += int(valid.sum() - keep.sum())
        out[:] = np.where(sw > 0, swx / np.maximum(sw, 1e-12), np.nan_to_num(mean))
    out = np.nan_to_num(out)
    res.data = out[0] if nc == 1 else out
    total = sum(1 for _ in entries) * out.size
    log.append(f"Combined {len(entries)} frames (sigma clip k={opts.kappa:g}, "
               f"{'noise-weighted' if opts.weighting else 'equal weights'}); "
               f"{100 * rejected / max(total, 1):.3f}% of pixels rejected as outliers")
    log.append(f"Time: {time.time() - t_start:.0f} s")
    return res


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
    extra = repr((g.label(), cal, opts.kappa, opts.weighting, opts.osc_half, tuple(opts.formats),
                  os.path.normcase(str(opts.masters_root)), opts.align_common))
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
    folder = Path(opts.masters_root or i0.path.parent) / g.target
    base = _unique_base(folder / name, opts.formats)
    infos = [f.info for f in g.frames]
    dates = sorted(i.date_obs for i in infos if i.date_obs)
    temps = [i.temp for i in infos if i.temp is not None]
    plan = g.plan or CalPlan()
    header = [("IMAGETYP", "Master Light", ""), ("OBJECT", i0.target, ""), ("FILTER", g.filter, ""),
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
              ("CALFLAT", (plan.flat.ref.path.name if plan.flat.master else f"{plan.flat.count} flat subframes")
               if plan.flat else "none", ""),
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
