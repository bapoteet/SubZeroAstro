"""
prepare.py - Stage 4: get master lights ready for PixInsight.

Linear pipeline (each step optional):
    crop -> plate solve (ASTAP) -> gradient removal (GraXpert or built-in)
    -> BlurXTerminator -> NoiseXTerminator -> StarXTerminator -> save
plus a quick-look colour image (auto-stretch, RGB / LRGB / SHO / HOO …) and a
step-by-step PixInsight plan tailored to what was done.

Pixel data is kept in camera units (ADU) internally and written in
PixInsight's 0..1 convention (see stacking.save_master_file).
"""

from __future__ import annotations

import json
import pickle
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import checkpoints as cp
import fits_core as core
import stacking as st
import tools

STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END",
              "PIXRANGE", "_XISF_CHANNELS", "_XISF_BOUNDS", "_XISF_FORMAT"}
WCS_PREFIXES = ("CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_", "CDELT", "CROTA", "CUNIT", "PC1_", "PC2_", "A_", "B_",
                "AP_", "BP_", "PLTSOLVD", "LONPOLE", "LATPOLE")


class Cancelled(Exception):
    pass


# ==========================================================================
# Masters
# ==========================================================================

@dataclass
class Master:
    path: Path
    data: np.ndarray                      # (channels, h, w) float32, ADU
    cards: list                           # (key, value, comment) to carry forward
    look: dict
    target: str
    filter: str
    alignref: str | None
    autocrop: tuple | None
    steps: list = field(default_factory=list)      # what was done, for the log and the plan
    outputs: list = field(default_factory=list)
    solved: bool = False

    @property
    def is_color(self):
        return self.data.shape[0] == 3

    @property
    def name(self):
        st_ = self.path.stem
        try:
            n = int(self.look.get("STACKCNT") or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 1 and "_master" not in st_.lower():         # a smart-telescope stack: give it a tidy name
            day = core.night_of(core.frame_info(self.look)) or "undated"
            dev = core.safe_filename(core.smart_scope(self.look) or "Stack") if hasattr(core, "safe_filename") \
                else re.sub(r"[^A-Za-z0-9]+", "", core.smart_scope(self.look) or "Stack")
            return f"{self.target}_{re.sub(r'[^A-Za-z0-9]+', '', self.filter)}_{dev}_{n}x_{day}"
        return st_

    @property
    def shape(self):
        return self.data.shape[1:]

    def set_card(self, key, value, comment=""):
        self.cards = [c for c in self.cards if c[0] != key] + [(key, value, comment)]

    def get(self, key, default=None):
        for k, v, _c in self.cards:
            if k == key:
                return v
        return default


def is_master_light(look: dict, name: str) -> bool:
    t = str(look.get("IMAGETYP", "")).lower()
    try:
        smart_stack = int(look.get("STACKCNT") or 0) > 1          # e.g. a Seestar "Stacked_…" file
    except (TypeError, ValueError):
        smart_stack = False
    return "master" in t and "light" in t or smart_stack or (
        "_master" in name.lower() and "flat" not in t and "dark" not in t and "bias" not in t)


_VERSION = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d+)$")


def newest_versions(files) -> list:
    """Restacking the same target saves ..._2026-07-03_2, _3, ... next to the first master: from a folder only
    the newest of each is used (pick an older one with Add Masters… to work on it)."""
    groups = {}
    for f in files:
        m = _VERSION.search(f.stem)
        key = (f.parent, f.stem[:m.start(2) - 1] if m else f.stem)
        groups.setdefault(key, []).append(f)
    out = []
    for fs in groups.values():
        def ver(f):
            m = _VERSION.search(f.stem)
            return int(m.group(2)) if m else 1
        top = max(ver(f) for f in fs)
        out += [f for f in fs if ver(f) == top]
    return out


def find_masters(paths) -> list:
    """Master light files in the given files / folders (one per base name; FITS preferred)."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            here = [f for f in p.iterdir() if f.suffix.lower() in st.IMAGE_EXTS and not f.name.startswith(".")]
            stems = {f.stem for f in here}
            # a comet's extra masters (_Comet, _Stars) stay out when the joined master is there too -
            # pick one of them with Add Masters… to work on it
            here = [f for f in here if not (f.stem.endswith(("_Comet", "_Stars")) and
                                            f.stem.rsplit("_", 1)[0] in stems)]
            files += newest_versions(here)
        elif p.suffix.lower() in st.IMAGE_EXTS:
            files.append(p)
    by_base = {}
    for f in sorted(files, key=lambda x: core.natural_key(str(x))):
        b = f.with_suffix("")
        if b not in by_base or f.suffix.lower() != ".xisf":
            by_base[b] = f
    out = []
    for f in by_base.values():
        try:
            src = st.ImageSource(f)
        except Exception:  # noqa: BLE001
            continue
        if is_master_light(src.look, f.name) and not f.stem.endswith("_prepared"):
            out.append(f)
    return out


def load_master(path) -> Master:
    src = st.ImageSource(path)
    d = src.read().astype(np.float32)
    if d.ndim == 2:
        d = d[None]
    cards = [(k, v, c) for k, v, c in src.cards if k not in STRUCTURAL and not k.startswith("_") and v is not None]
    hist = [(k, v, c) for k, v, c in src.cards if k in ("HISTORY", "COMMENT")]
    look = src.look
    ac = None
    try:
        ac = tuple(int(x) for x in str(look.get("AUTOCROP", "")).split()) or None
        if ac and len(ac) != 4:
            ac = None
    except ValueError:
        ac = None
    return Master(Path(path), d, cards + [(k, c, "") for k, _v, c in hist], look,
                  st.target_name(look.get("OBJECT"), look.get("FILTER")) if look.get("OBJECT") else "Unknown",
                  core.normalize_filter(look.get("FILTER")) if look.get("FILTER") else
                  ("RGB" if d.shape[0] == 3 else "No filter"),
                  look.get("ALIGNREF"), ac, solved=bool(look.get("PLTSOLVD")) or bool(look.get("CTYPE1")))


def target_sets(masters: list) -> list:
    """Masters that belong together (same target and pixel grid)."""
    sets = {}
    for m in masters:
        # same target and image size belong together, even when stacked in separate runs:
        # run_prepare lines them up (line_up) if they were not aligned to each other
        key = (re.sub(r"[^a-z0-9]", "", str(m.target).lower()), m.shape)
        sets.setdefault(key, []).append(m)
    return list(sets.values())


def needs_line_up(masters) -> bool:
    ref = masters[0]
    return any(not (m.alignref and m.alignref == ref.alignref) for m in masters[1:])


_WCS_KEYS = re.compile(r"^(CTYPE\d|CRVAL\d|CRPIX\d|CD\d_\d|CDELT\d|CROTA\d|PC\d_\d|CUNIT\d|PLTSOLVD|"
                       r"[AB]P?_\d_\d|[AB]P?_ORDER|EQUINOX|RADESYS|LONPOLE|LATPOLE)$")


def _binned_lum(m, f=2):
    lum = m.data.mean(axis=0) if m.data.shape[0] > 1 else st.despike(np.asarray(m.data[0], np.float32))[0]
    h, w = lum.shape
    return lum[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.float32)


def _valid_rect(valid: np.ndarray):
    """Largest-ish rectangle with valid pixels all round: shrink the worst border until clean."""
    h, w = valid.shape
    y0, y1, x0, x1 = 0, h, 0, w
    while y1 - y0 > 2 and x1 - x0 > 2:
        bad = {"top": 1 - valid[y0, x0:x1].mean(), "bottom": 1 - valid[y1 - 1, x0:x1].mean(),
               "left": 1 - valid[y0:y1, x0].mean(), "right": 1 - valid[y0:y1, x1 - 1].mean()}
        side = max(bad, key=bad.get)
        if bad[side] <= 0:
            break
        if side == "top":
            y0 += 1
        elif side == "bottom":
            y1 -= 1
        elif side == "left":
            x0 += 1
        else:
            x1 -= 1
    return (x0, y0, x1, y1) if y1 - y0 > 2 and x1 - x0 > 2 else None


def _intersect(a, b):
    if a is None:
        return b
    if b is None:
        return a
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[2] > r[0] and r[3] > r[1] else a


def line_up(masters, log=None):
    """Masters of one target that were stacked in separate runs (e.g. Ha on one night,
    SII and OIII later) are not on the same pixel grid: match their stars to the first
    master's and resample them onto its grid. Returns the area all of them cover
    (x0, y0, x1, y1), or None when nothing needed lining up."""
    lg = log or (lambda t: None)
    if len(masters) < 2 or not needs_line_up(masters):
        return None
    def depth(m):                         # the deepest master is the best reference (most stars, least noise)
        for k in ("EXPTOTAL", "NCOMBINE"):
            try:
                v = float(m.get(k) or m.look.get(k) or 0)
            except (TypeError, ValueError):
                v = 0
            if v > 0:
                return v
        return 0.0
    ref = max(masters, key=depth) if any(depth(m) > 0 for m in masters) else masters[0]
    if not ref.alignref:
        ref.alignref = str(ref.path)
    ref.set_card("ALIGNREF", Path(ref.alignref).name, "lined up on this grid (Astro Image Processing)")
    ref_look = core.cards_to_dict(ref.cards)
    ref_pts, _ = st.find_stars(_binned_lum(ref), 2, max_stars=200)
    overlap = tuple(ref.autocrop) if ref.autocrop else None
    for m in masters:
        if m is ref or (m.alignref and m.alignref == ref.alignref):
            continue
        pts, _ = st.find_stars(_binned_lum(m), 2, max_stars=200)
        try:
            T, n_in, rms, _ok, _j, how = st.match_frames_wcs(ref_pts, pts, ref_look, core.cards_to_dict(m.cards),
                                                             m.shape, tol=4.0)
        except ValueError as e:
            lg(f"Could not line up {m.name} with {ref.name} ({e}) — it is used as it is")
            continue
        A = np.asarray(T)[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        if abs(scale - 1.0) > 0.03:              # same telescope and camera: the scale cannot change
            lg(f"Not lining up {m.name}: the star match would resize it by {scale:.3f}x, which cannot be right "
               "for the same telescope - it is used as it is")
            continue
        own = m.autocrop                      # its own well-covered area (ragged stacking edges outside)
        planes = st.warp_planes(list(m.data), T, ref.shape)
        valid = np.isfinite(planes[0])
        m.data = np.stack([np.nan_to_num(p, nan=0.0) for p in planes]).astype(np.float32)
        overlap = _intersect(overlap, _valid_rect(valid))
        if own:                               # carry that area over to the reference grid
            x0, y0, x1, y1 = own
            c = st._apply(T, np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64))
            inner = (int(np.ceil(max(c[0, 0], c[2, 0]))), int(np.ceil(max(c[0, 1], c[1, 1]))),
                     int(np.floor(min(c[1, 0], c[3, 0]))), int(np.floor(min(c[2, 1], c[3, 1]))))
            overlap = _intersect(overlap, inner)
        m.alignref = ref.alignref or str(ref.path)
        m.set_card("ALIGNREF", Path(m.alignref).name, "lined up on this grid (Astro Image Processing)")
        m.autocrop = ref.autocrop
        m.cards = [c for c in m.cards if not _WCS_KEYS.match(str(c[0]))]   # its old plate solution no longer fits
        m.cards += [c for c in ref.cards if _WCS_KEYS.match(str(c[0]))]    # it now sits on the reference's grid
        m.solved = False
        m.steps.append(("align", f"Lined up with the {ref.filter} master ({n_in} stars matched, "
                                 f"{rms:.2f} px)"))
        lg(f"Lined up {m.name} with {ref.name}: {n_in} stars matched, accuracy {rms:.2f} px"
           + (" (guided by the plate solutions)" if how != "stars" else ""))
    return overlap


# ==========================================================================
# Crop
# ==========================================================================

def common_autocrop(masters: list):
    rects = [m.autocrop for m in masters if m.autocrop]
    if not rects:
        return None
    x0 = max(r[0] for r in rects)
    y0 = max(r[1] for r in rects)
    x1 = min(r[2] for r in rects)
    y1 = min(r[3] for r in rects)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def crop_outside_note(rect, covered):
    """A warning when a hand-drawn crop reaches past the area ~all frames cover (the stack's auto-crop):
    there only some frames - often only one filter - fill the picture, so it shows up as a coloured, noisier
    band along that edge."""
    if rect is None or covered is None:
        return ""
    x0, y0, x1, y1 = (int(v) for v in rect)
    cx0, cy0, cx1, cy1 = (int(v) for v in covered)
    area = max(1, (x1 - x0) * (y1 - y0))
    ix = max(0, min(x1, cx1) - max(x0, cx0))
    iy = max(0, min(y1, cy1) - max(y0, cy0))
    out = 1.0 - ix * iy / area
    if out < 0.02:
        return ""
    sides = [n for n, bad in (("top", y0 < cy0), ("bottom", y1 > cy1), ("left", x0 < cx0), ("right", x1 > cx1)) if bad]
    return (f"Note: {100 * out:.0f}% of this crop ({', '.join(sides)} edge) is outside the area 90% of the frames "
            f"cover (x {cx0}-{cx1}, y {cy0}-{cy1}) - fewer frames there, so it is noisier, and a coloured band can "
            f"show at the very edge where only one filter reached; crop inside that area to avoid it")


def crop(m: Master, rect):
    x0, y0, x1, y1 = (int(v) for v in rect)
    h, w = m.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return
    m.data = np.ascontiguousarray(m.data[:, y0:y1, x0:x1])
    for k, d in (("CRPIX1", x0), ("CRPIX2", y0)):
        v = m.get(k)
        if isinstance(v, (int, float)):
            m.set_card(k, float(v) - d, "adjusted for crop")
    m.cards = [c for c in m.cards if c[0] != "AUTOCROP"]
    m.autocrop = None
    m.steps.append(("crop", f"Cropped to x {x0}-{x1}, y {y0}-{y1} ({x1 - x0} x {y1 - y0})"))


# ==========================================================================
# Temporary files for external tools (0..1 float FITS)
# ==========================================================================

def ready_made(m) -> str | None:
    """Why this master's lights were already calibrated (Telescope Live and similar ready-made data), or None."""
    look = getattr(m, "look", None) or {}
    if look.get("PRECAL"):
        return str(look.get("PRECAL"))
    try:                         # masters stacked before the PRECAL keyword existed: their stack log says so
        log = Path(m.path).with_name(Path(m.path).stem + "_stacklog.txt")
        if log.is_file() and "already calibrated" in log.read_text(encoding="utf-8", errors="ignore"):
            return "ready-made data"
    except (OSError, TypeError, ValueError):
        pass
    return None


# BlurXTerminator on ready-made data (dark sites, big apertures, many hours): its high signal-to-noise takes
# firmer star sharpening than a backyard stack - tighter star edges without a cut-out look
PRO_BXT = {"ss": 0.55, "ash": 0.50}


def bxt_values_for(masters, values: dict) -> tuple:
    """(values to use, note or ''): the usual backyard settings become PRO_BXT for ready-made data.
    Values the user changed are kept as they are."""
    v = dict(values or {})
    try:
        # the app's usual settings (0.50 / 0.40 from the start, 0.40 / 0.40 in older versions)
        usual = abs(float(v.get("ash", -1)) - 0.40) < 1e-6 and \
            any(abs(float(v.get("ss", -1)) - x) < 1e-6 for x in (0.40, 0.50))
    except (TypeError, ValueError):
        usual = False
    if usual and masters and all(ready_made(m) for m in masters):
        v.update(PRO_BXT)
        return v, (f"ready-made data: BlurXTerminator Sharpen Stars {PRO_BXT['ss']:.2f}, Adjust Star Halos "
                   f"{PRO_BXT['ash']:.2f} (instead of {float(values.get('ss')):.2f} / {float(values.get('ash')):.2f}) for "
                   "tighter star edges")
    return v, ""


def gradient_correction_for(m) -> tuple:
    """GraXpert's correction for one master, and why.
    Subtraction - the lights were flat-calibrated (your flats, a SkyFlat, or ready-made data such as Telescope
    Live): what is left is light pollution or moon glow, which adds to the picture.
    Division - the master was stacked without any flats: vignetting is still in it, and it multiplies."""
    look = getattr(m, "look", None) or {}
    why = ready_made(m)
    if why:
        return "Subtraction", f"the lights were already calibrated ({why})"
    calflat = str(look.get("CALFLAT") or "").strip()
    if calflat.lower() != "none":
        return "Subtraction", ("the lights were flat-calibrated" if calflat else
                               "light pollution and moon glow add to the picture")
    return "Division", "stacked without flats, so vignetting is still in the picture"


def _temp_scale(d) -> float:
    """What a picture is divided by before an outside tool gets it (they all work in 0..1).
    Normally 65535 (the app's ADU scale). A picture on another scale - a master saved as 0..1, or values
    pushed past 65535 by a division (GraXpert 'Division') - is divided by its own maximum instead: sent as
    65535ths it would reach the tool as almost pure black (or clipped flat white), and the AI tools then
    invent a repeating texture in every tile."""
    try:
        top = float(np.nanmax(d))
    except ValueError:
        return 65535.0
    if not np.isfinite(top) or top <= 0:
        return 65535.0
    return 65535.0 if 1000.0 <= top <= 65535.0 * 1.02 else top


def _write_temp(m: Master, path: Path, data=None, mono=False) -> float:
    d = m.data if data is None else data
    if mono and d.shape[0] == 3:
        d = d.mean(axis=0, keepdims=True)
    scale = _temp_scale(d)
    arr = np.clip(d / scale, 0, 1).astype(np.float32)
    arr = arr[0] if arr.shape[0] == 1 else arr
    hdr = [(k, v, c) for k, v, c in m.cards if k not in ("HISTORY", "COMMENT") and k not in STRUCTURAL]
    core.write_fits(path, arr, hdr, bitpix=-32)
    return scale


# Outside tools do not all agree on which way up a FITS file is (the ROWORDER keyword):
# rc-astro was seen to hand images back upside down. Every result is compared with what was
# sent and turned back if needed, so the picture never flips between steps.
_TURNS = {"same": lambda a: a,
          "upside down": lambda a: a[..., ::-1, :],
          "mirrored left-right": lambda a: a[..., :, ::-1],
          "rotated 180°": lambda a: a[..., ::-1, ::-1]}


def _signature(a, side=160):
    a = np.asarray(a, np.float32)
    lum = a.mean(axis=0) if a.ndim == 3 else a
    h, w = lum.shape
    f = max(1, max(h, w) // side)
    hh, ww = h // f * f, w // f * f
    small = lum[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    small = np.arcsinh((small - np.median(small)) / (np.std(small) + 1e-12))
    small -= small.mean()
    sd = float(small.std())
    return small / sd if sd > 0 else small


def orientation_of(result, ref) -> str:
    """How a tool turned the picture around compared with what it was given ('same' if not)."""
    result, ref = np.asarray(result), np.asarray(ref)
    if result.shape[-2:] != ref.shape[-2:]:
        return "same"
    r = _signature(ref)
    score = {name: float((_signature(fn(result)) * r).mean()) for name, fn in _TURNS.items()}
    best = max(score, key=score.get)
    if best != "same" and score[best] > 0.3 and score[best] > score["same"] + 0.1:
        return best
    return "same"


def turn_back(how, *arrays):
    """Undo orientation_of's result (every turn in _TURNS is its own inverse)."""
    fn = _TURNS.get(how, _TURNS["same"])
    return [None if a is None else np.ascontiguousarray(fn(a)) for a in arrays]


def _level(a) -> float:
    a = np.asarray(a, np.float32)
    v = a[..., ::7, ::7] if a.ndim >= 2 else a
    v = v[np.isfinite(v)]
    return float(np.median(np.abs(v))) if v.size else 0.0


def _read_result(path, like: np.ndarray, scale: float | None = None, log=None, check_level=True) -> np.ndarray:
    """A tool's answer, back on the app's scale. With `scale` (what _write_temp divided by) the file is
    taken as the 0..1 picture that was sent; without it, as before (0..1 is taken as 65535ths)."""
    d = st.ImageSource(path)
    if scale is not None:
        d.scale = 1.0                         # the tool wrote what it was given: 0..1 of `scale`
    arr = np.asarray(d.read(), np.float32)
    if scale is not None:
        arr = arr * np.float32(scale)
    elif float(np.nanmax(arr)) <= 1.5:        # tools write 0..1
        arr = arr * np.float32(65535.0)
    if check_level:
        # a tool hands back the same picture, a little changed - never one hundreds of times brighter or
        # darker. If it does, the answer came back on another scale: bring it to the level that was sent.
        ref, got = _level(like), _level(arr)
        if ref > 0 and got > 0 and not (1 / 30 < got / ref < 30):
            f = np.float32(ref / got)
            arr = arr * f
            if log:
                log(f"    {Path(path).name}: came back on another brightness scale (x{1 / float(f):.3g}) — "
                    "brought back to the level that was sent")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape != like.shape:
        if arr.shape[0] == 1 and like.shape[0] == 3 and arr.shape[1:] == like.shape[1:]:
            arr = np.repeat(arr, 3, axis=0)
        else:
            raise tools.ToolError(f"tool returned an image of a different size {arr.shape} vs {like.shape}")
    return np.nan_to_num(arr).astype(np.float32)


# ==========================================================================
# Plate solving hints
# ==========================================================================

def _sexa(v, hours=False):
    if isinstance(v, (int, float)):
        return float(v)
    if not v:
        return None
    parts = re.split(r"[\s:hdms°'\"]+", str(v).strip())
    parts = [p for p in parts if p]
    try:
        sign = -1.0 if parts[0].startswith("-") else 1.0
        nums = [abs(float(parts[0]))] + [float(x) for x in parts[1:3]]
    except (ValueError, IndexError):
        return None
    val = nums[0] + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
    return sign * val * (15.0 if hours else 1.0)


def solve_hints(m: Master):
    L = m.look
    ra = L.get("CRVAL1") if L.get("CTYPE1") else None
    dec = L.get("CRVAL2") if L.get("CTYPE1") else None
    if ra is None:
        ra = _num(L.get("RA")) if _num(L.get("RA")) is not None else _sexa(L.get("OBJCTRA"), hours=True)
        dec = _num(L.get("DEC")) if _num(L.get("DEC")) is not None else _sexa(L.get("OBJCTDEC"))
    fov = None
    scale = core.frame_info(L).get("scale")         # arcsec/pixel (FOCALLEN/FOCAL, XPIXSZ/PIXSZ)
    if scale:
        fov = m.shape[0] * scale / 3600.0
    return ra, dec, fov


def solve_with_retries(astap, path, ra, dec, fov, radius, cancel=None, progress=None, memo=None):
    """Plate solve, trying again when the field size from the header is wrong - the usual cause of
    'No solution found': binning not counted in the pixel size (field half or double the real one), or
    a focal reducer / Barlow the header doesn't know about - or missing (no FOCALLEN / XPIXSZ): then
    typical field sizes are tried. memo (a dict shared by the masters of one run) remembers the field
    size that worked, and that nothing worked, so the other masters don't repeat the search.
    Returns (cards, note); raises tools.ToolError with a plain explanation when every try fails."""
    memo = memo if memo is not None else {}
    r_near = min(radius, 10.0) if ra is not None else 180.0
    tries = []
    if memo.get("fov"):
        tries.append((memo["fov"], r_near, "the field size that solved the other masters"))
    tries.append((fov, radius if ra is not None else 180.0, ""))
    if memo.get("failed"):
        tries = tries[:2]                         # the search already failed for a master of this run
    elif fov:
        for k, why in ((0.5, "binning"), (2.0, "binning"), (0.7, "a focal extender"), (1.4, "a focal reducer"),
                       (0.25, "4x binning"), (4.0, "4x binning")):
            tries.append((fov * k, r_near, f"the field is {k:g}x what the header says ({why}?)"))
    else:                                         # no focal length / pixel size in the header
        for f in (0.25, 0.35, 0.5, 1.0, 1.4, 2.0, 3.0, 4.5):
            tries.append((f, r_near, f"the header has no pixel scale; a field {f:g}° tall fits"))
    last = None
    for n, (f, r, why) in enumerate(tries):
        if n and progress:
            progress(f"retry {n} of {len(tries) - 1}" + (f" - field {f:.2f}°" if f else " - field from the header"))
        try:
            cards = astap.solve(path, ra, dec, f, radius=r, cancel=cancel)
            if f:
                memo["fov"] = f
            return cards, (f"Plate solved on retry {n}: {why}" if n and why else "")
        except tools.ToolCancelled:
            raise
        except tools.ToolError as e:
            last = e
    memo["failed"] = True
    dbs = tools.astap_databases(getattr(astap, "exe", None))
    hint = []
    if not fov:
        hint.append("the header has no focal length / pixel size; fields from 0.25° to 4.5° tall were tried")
    if fov:
        hint.append(f"the header gives a field {fov:.2f}° tall; fields from {fov / 4:.2f}° to {fov * 4:.2f}° were tried")
    small = [d for d in dbs if tools.ASTAP_DB_FOV.get(d, 99) <= 0.3]
    if dbs and not small and (fov is None or fov < 1.0):
        hint.append(f"ASTAP has only the {', '.join(x.upper() for x in dbs)} star database(s), made for wide fields - "
                    "install D50 (or D80) from the ASTAP website for small fields like this")
    elif not dbs:
        hint.append("no ASTAP star database was found - install D50 from the ASTAP website")
    if ra is None:
        hint.append("the header has no pointing (RA/Dec) - tick 'Search the whole sky'")
    raise tools.ToolError(f"{last} - " + "; ".join(hint) if hint else str(last))


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def apply_wcs(m: Master, cards):
    m.cards = [c for c in m.cards if not c[0].startswith(WCS_PREFIXES)]
    m.cards += [(k, v, c) for k, v, c in cards if k != "SIMPLE"]
    m.solved = True


# ==========================================================================
# Built-in gradient removal
# ==========================================================================

def builtin_gradient(plane: np.ndarray, degree=3, grid=(24, 16)):
    """Fit a smooth 2-D polynomial to the sky (stars and bright objects ignored)
    and subtract it, keeping the overall sky level."""
    h, w = plane.shape
    gx, gy = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    pts, vals = [], []
    for i in range(gy):
        for j in range(gx):
            box = plane[ys[i]:ys[i + 1]:2, xs[j]:xs[j + 1]:2]
            med = np.median(box)
            mad = np.median(np.abs(box - med)) * 1.4826 + 1e-6
            good = box[np.abs(box - med) < 2.5 * mad]
            if good.size:
                pts.append(((xs[j] + xs[j + 1]) / 2 / w - 0.5, (ys[i] + ys[i + 1]) / 2 / h - 0.5))
                vals.append(np.median(good))
    pts = np.array(pts)
    vals = np.array(vals)
    # drop samples sitting on bright nebulae / galaxies
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    keep = vals < med + 3 * mad
    pts, vals = pts[keep], vals[keep]

    def terms(x, y):
        return np.stack([x ** i * y ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)
    coef, *_ = np.linalg.lstsq(terms(pts[:, 0], pts[:, 1]), vals, rcond=None)
    Y, X = np.mgrid[0:h:8, 0:w:8]
    model_small = terms(X / w - 0.5, Y / h - 0.5) @ coef
    # upsample the smooth model
    ry = np.minimum(np.arange(h) // 8, model_small.shape[0] - 1)
    rx = np.minimum(np.arange(w) // 8, model_small.shape[1] - 1)
    model = model_small[ry][:, rx]
    return (plane - model + np.median(model)).astype(np.float32)


# ==========================================================================
# Quick-look image
# ==========================================================================

def stretch(x: np.ndarray, target=0.22, black_sigma=-2.8):
    s = x[::4, ::4]
    s = s[np.isfinite(s)]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826 + 1e-6
    lo = med + black_sigma * mad
    hi = float(np.percentile(s, 99.98))
    hi = max(hi, lo + 1e-6)
    n = np.clip((x - lo) / (hi - lo), 0, 1)
    nm = min(max((med - lo) / (hi - lo), 1e-6), 0.999)
    mbal = nm * (1 - target) / (nm - 2 * nm * target + target)
    return ((mbal - 1) * n / ((2 * mbal - 1) * n - mbal)).astype(np.float32)


def palettes_for(masters: list) -> list:
    f = {m.filter for m in masters}
    color = [m for m in masters if m.is_color]
    out = []
    if color:
        out.append("RGB (colour camera)")
        import finish
        if any(finish.dual_band_filter(m.filter, m.look) for m in color):
            line = core.band_lines(color[0].filter)[0]
            out = [f"{'SOO' if line == 'SII' else 'HOO'} (dual-band)", "Gold & blue (dual-band)"] + out
    if {"R", "G", "B"} <= f:
        out += ["LRGB"] if "L" in f else []
        out += ["RGB"]
    if {"Ha", "OIII", "SII"} <= f:
        out += ["SHO (Hubble)", "HOO"]
    elif {"Ha", "OIII"} <= f:
        out += ["HOO"]
    out += [f"{m.filter} (mono)" for m in masters if not m.is_color]
    return out


def quick_look(masters: list, palette: str) -> np.ndarray:
    """(h, w, 3) float image in 0..1 for display/export."""
    by = {m.filter: m for m in masters if not m.is_color}

    def ch(f):
        return stretch(by[f].data[0])

    if palette.endswith("(dual-band)"):
        import finish
        m = next(m for m in masters if m.is_color)
        ha, o3 = finish.split_dual_band(m.data)
        h, o = stretch(ha), stretch(o3)
        rgb = np.stack([h, 0.85 * h + 0.15 * o, o] if palette.startswith("Gold") else [h, o, o], -1)
    elif palette.startswith("RGB (colour"):
        m = next(m for m in masters if m.is_color)
        rgb = np.stack([stretch(m.data[i]) for i in range(3)], -1)
    elif palette == "RGB":
        rgb = np.stack([ch("R"), ch("G"), ch("B")], -1)
    elif palette == "LRGB":
        rgb = np.stack([ch("R"), ch("G"), ch("B")], -1)
        lum = ch("L")
        cur = rgb.mean(-1, keepdims=True) + 1e-6
        rgb = np.clip(rgb * (lum[..., None] / cur), 0, 1)
    elif palette.startswith("SHO"):
        rgb = np.stack([ch("SII"), ch("Ha"), ch("OIII")], -1)
    elif palette == "HOO":
        o = ch("OIII")
        rgb = np.stack([ch("Ha"), o, o], -1)
    else:
        f = palette.split(" ")[0]
        g = ch(f)
        rgb = np.stack([g, g, g], -1)
    return np.clip(rgb, 0, 1)


def write_tiff16(path, rgb: np.ndarray):
    """Uncompressed 16-bit RGB TIFF (baseline), no extra libraries needed."""
    import struct
    h, w, _ = rgb.shape
    data = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype("<u2").tobytes()
    entries = []

    def ent(tag, typ, count, value):
        entries.append((tag, typ, count, value))
    n_entries = 11
    ifd_offset = 8
    ifd_size = 2 + n_entries * 12 + 4
    bps_offset = ifd_offset + ifd_size
    data_offset = bps_offset + 6
    ent(256, 4, 1, w)                       # ImageWidth
    ent(257, 4, 1, h)                       # ImageLength
    ent(258, 3, 3, bps_offset)              # BitsPerSample -> 16,16,16
    ent(259, 3, 1, 1)                       # no compression
    ent(262, 3, 1, 2)                       # RGB
    ent(273, 4, 1, data_offset)             # StripOffsets
    ent(277, 3, 1, 3)                       # SamplesPerPixel
    ent(278, 4, 1, h)                       # RowsPerStrip
    ent(279, 4, 1, len(data))               # StripByteCounts
    ent(284, 3, 1, 1)                       # chunky
    ent(339, 3, 3, 0)                       # SampleFormat (placeholder -> uint, below)
    entries[-1] = (339, 3, 1, 1)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_offset))
        f.write(struct.pack("<H", len(entries)))
        for tag, typ, count, value in entries:
            if typ == 3 and count == 1:
                f.write(struct.pack("<HHIHH", tag, typ, count, value, 0))
            else:
                f.write(struct.pack("<HHII", tag, typ, count, value))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<HHH", 16, 16, 16))
        f.write(data)


# ==========================================================================
# The PixInsight plan
# ==========================================================================

def pixinsight_plan(masters: list, palette_hint: str | None = None) -> list:
    """Ordered list of dicts: {title, where, why, done(bool), note}."""
    done = {k for m in masters for k, _t in m.steps}
    solved = all(m.solved for m in masters)
    color = any(m.is_color for m in masters)
    filters = {m.filter for m in masters}
    narrow = bool(filters & {"Ha", "OIII", "SII"}) and not color
    starsplit = "sxt" in done
    multi = len(masters) > 1
    aligned = multi and all(m.alignref for m in masters)
    steps = []

    def add(title, where, why, is_done=False, note=""):
        steps.append({"title": title, "where": where, "why": why, "done": is_done, "note": note})

    add("Open the prepared masters", "File › Open (the *_prepared.xisf files)",
        "These are linear 32-bit images — use the STF auto-stretch (Ctrl+A) only to look at them.")
    if multi and not aligned:
        add("StarAlignment", "Process › ImageRegistration › StarAlignment",
            "These masters do not share a pixel grid yet — align them to one reference before combining.")
    add("Crop the stacking edges", "Process › Geometry › DynamicCrop",
        "Ragged edges from dithering confuse gradient tools and colour calibration.", "crop" in done)
    add("Plate solve", "Script › Image Analysis › ImageSolver",
        "Needed for SPCC and for annotation.", solved,
        "Solved by ASTAP — the solution is in the file header." if solved else "")
    add("Remove gradients", "GraXpert, or Process › BackgroundModelization › DynamicBackgroundExtraction",
        "Light pollution and moon gradients must go before colour calibration.", "gradient" in done)
    if multi and not color:
        if narrow:
            add("Combine the narrowband channels", "Process › PixelMath (or the NarrowbandNormalization script)",
                "e.g. SHO: R = SII, G = Ha, B = OIII; HOO: R = Ha, G = B = OIII. "
                "Linear-fit the weaker channels to Ha first (Process › ColorCalibration › LinearFit).")
        else:
            add("Combine the colour channels", "Process › ChannelManagement › ChannelCombination",
                "Combine R, G and B into one colour image (keep L separate for LRGB).")
    add("Sharpen with BlurXTerminator", "Process › RC-Astro › BlurXTerminator",
        "Deconvolution works best on linear data, before colour calibration and noise reduction.",
        "bxt" in done)
    if color or (multi and not narrow):
        add("Colour calibration", "Process › ColorCalibration › SpectrophotometricColorCalibration (SPCC)",
            "Uses Gaia star colours — requires the plate solution. Run BackgroundNeutralization first if "
            "SPCC asks for it.", False,
            "" if solved else "Plate solve first (step above).")
    elif narrow:
        add("Balance the narrowband colours", "Process › ColorCalibration › BackgroundNeutralization",
            "SPCC is not meant for narrowband palettes; neutralise the background instead "
            "(SPCC's narrowband mode is an option for Ha/OIII/SII filters).")
    add("Reduce noise (linear, light)", "Process › RC-Astro › NoiseXTerminator",
        "A light pass now (denoise ~0.5–0.7) keeps the stretch clean; a final pass can follow.", "nxt" in done)
    add("Separate the stars", "Process › RC-Astro › StarXTerminator (Generate star image)",
        "Lets you stretch the nebula/galaxy hard without bloating the stars.", starsplit,
        "Starless and stars-only images were already saved." if starsplit else "Optional.")
    add("Stretch to non-linear", "Script › Utilities › GeneralizedHyperbolicStretch  (or HistogramTransformation "
        "from the STF)", "Stretch the starless image (and the stars image gently) — this is where the image "
        "comes to life.")
    if not color and "L" in filters:
        add("Add the luminance", "Process › ColorSpaces › LRGBCombination",
            "After both L and RGB are stretched to similar brightness.")
    add("Fine-tune contrast and colour", "Process › IntensityTransformations › CurvesTransformation, "
        "LocalHistogramEqualization", "Saturation, contrast and local detail on the stretched image.")
    if starsplit:
        add("Put the stars back", "Process › PixelMath:  ~((~starless)*(~stars))",
            "Screen-blend the stretched stars onto the processed starless image.")
    add("Final noise pass and export", "NoiseXTerminator (light) › File › Save As (TIFF/JPG)",
        "Keep a 16-bit TIFF and an XISF project for later.")
    return steps


def plan_html(target: str, masters: list, steps: list, files: list) -> str:
    rows = []
    for i, s in enumerate(steps, 1):
        mark = "✅" if s["done"] else "➡️"
        state = "<span class='done'>done here</span>" if s["done"] else ""
        rows.append(f"<tr class='{'d' if s['done'] else ''}'><td>{i}</td><td>{mark}</td>"
                    f"<td><b>{s['title']}</b> {state}<br><span class='where'>{s['where']}</span><br>"
                    f"{s['why']}" + (f"<br><i>{s['note']}</i>" if s['note'] else "") + "</td></tr>")
    flist = "".join(f"<li>{Path(f).name}</li>" for f in files)
    order, who = [], {}
    for m in masters:
        for _k, t in m.steps:
            if t not in who:
                order.append(t)
                who[t] = []
            who[t].append(m.filter)
    all_f = [m.filter for m in masters]
    did = "".join(f"<li>{t}" + ("" if len(masters) == 1 or who[t] == all_f else
                                f" <span class='where'>({', '.join(who[t])})</span>") + "</li>" for t in order)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{target} – next steps in PixInsight</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#1e1e22;color:#e4e4ea;max-width:900px;margin:24px auto;padding:0 16px}}
h1{{font-size:22px}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #3a3a42;padding:8px;vertical-align:top}}
tr.d td{{opacity:.6}} .where{{color:#8fb4ff}} .done{{color:#5fd27a;font-size:12px}} li{{margin:2px 0}}</style></head><body>
<h1>{target}: ready for Finishing Touches (or PixInsight)</h1>
<p>SubZeroAstro prepared these linear masters. Open them in PixInsight and continue with the steps below —
steps marked ✅ were already done here.</p>
<h3>Prepared files</h3><ul>{flist}</ul>
<h3>What was done</h3><ul>{did or '<li>(nothing — masters copied as they were)</li>'}</ul>
<h3>Recommended PixInsight steps, in order</h3><table>{''.join(rows)}</table>
</body></html>"""


def plan_text(target: str, steps: list) -> str:
    lines = [f"{target}: next steps in PixInsight", ""]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i:2d}. [{'done' if s['done'] else '    '}] {s['title']}  —  {s['where']}")
        lines.append(f"      {s['why']}" + (f"  ({s['note']})" if s['note'] else ""))
    return "\n".join(lines) + "\n"


# ==========================================================================
# Checkpoints: crash recovery and "go back to a step" (0.6)
# ==========================================================================

STEP_ORDER = ("crop", "solve", "gradient", "bxt", "nxt", "sxt")
STEP_TITLES = {"crop": "Crop", "solve": "Plate solve", "gradient": "Gradient removal",
               "bxt": "BlurXTerminator", "nxt": "NoiseXTerminator", "sxt": "StarXTerminator"}


class PrepareCheckpoints:
    """The image after each finished step, for one set of masters. A step is only
    trusted once its 'done' marker exists, so a crash mid-save is harmless."""

    def __init__(self, paths, target):
        self.paths = [str(x) for x in paths]
        self.dir = cp.root() / f"prepare_{cp.safe_name(target)}_{cp.file_key(self.paths)}"
        man = cp.read_json(self.dir / "manifest.json", None) or {}
        good = []
        for st_ in man.get("steps", []):
            if (self.dir / st_.get("dir", "?") / "done").exists():
                good.append(st_)
            else:
                break
        self.manifest = {"target": target, "masters": self.paths, "steps": good}

    @property
    def steps(self) -> list:
        return self.manifest["steps"]

    def _write_manifest(self):
        self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cp.write_json(self.dir / "manifest.json", self.manifest)

    def save(self, step, sig, detail, masters, stars, preview=None):
        idx = len(self.steps)
        sub = f"{idx + 1:02d}_{step}"
        d = self.dir / sub
        cp.remove(d)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(masters):
            np.save(d / f"m{i}.npy", m.data, allow_pickle=False)
            cp.write_atomic(d / f"m{i}.pkl", pickle.dumps(
                {"cards": m.cards, "steps": m.steps, "solved": m.solved, "autocrop": m.autocrop,
                 "look": m.look}, protocol=4))
            if i in stars:
                np.save(d / f"m{i}_stars.npy", stars[i], allow_pickle=False)
        if preview is not None:
            np.save(d / "preview.npy", preview, allow_pickle=False)
        (d / "done").write_text("ok", encoding="utf-8")
        self.steps.append({"step": step, "sig": sig, "dir": sub, "title": STEP_TITLES.get(step, step),
                           "detail": detail, "time": time.strftime("%Y-%m-%d %H:%M")})
        self._write_manifest()

    def restore(self, n, masters, stars):
        """Put the state after the first n steps back into masters / stars."""
        if n <= 0:
            return
        d = self.dir / self.steps[n - 1]["dir"]
        stars.clear()
        for i, m in enumerate(masters):
            m.data = np.load(d / f"m{i}.npy", allow_pickle=False)
            state = pickle.loads((d / f"m{i}.pkl").read_bytes())
            m.cards, m.steps, m.solved = state["cards"], state["steps"], state["solved"]
            m.autocrop, m.look = state["autocrop"], state["look"]
            sp = d / f"m{i}_stars.npy"
            if sp.exists():
                stars[i] = np.load(sp, allow_pickle=False)

    def truncate(self, n):
        """Forget step n and everything after it (index from 0)."""
        for st_ in self.steps[n:]:
            cp.remove(self.dir / st_["dir"])
        del self.steps[n:]
        if self.dir.exists():
            if self.steps:
                self._write_manifest()
            else:
                cp.remove(self.dir)

    def preview(self, n):
        try:
            return np.load(self.dir / self.steps[n]["dir"] / "preview.npy", allow_pickle=False)
        except (OSError, IndexError, ValueError):
            return None

    def size(self) -> int:
        return cp._size(self.dir) if self.dir.exists() else 0

    def clear(self):
        cp.remove(self.dir)
        self.manifest["steps"] = []


def plan_steps(masters: list, opts) -> list:
    """The steps this run will do, in order: [(step, signature, extra)], where the
    signature captures every setting that changes the result of that step."""
    out = []
    if opts.crop:
        rect = opts.crop_rect or common_autocrop(masters)
        if rect:
            out.append(("crop", json.dumps([int(v) for v in rect]), rect))
    if opts.solve and opts.astap:
        out.append(("solve", json.dumps([round(float(opts.solve_radius), 3)]), None))
    if opts.gradient:
        tool = "graxpert" if opts.gradient_tool == "graxpert" and opts.graxpert else "builtin"
        sig = [tool] + ([round(float(opts.gx_smoothing), 4), opts.gx_correction] if tool == "graxpert" else [])
        out.append(("gradient", json.dumps(sig), tool))
    rc = None
    for key in ("bxt", "nxt", "sxt"):
        cfg = opts.xt.get(key) or {}
        if not cfg.get("enabled") or not opts.rcastro:
            continue
        rc = rc or tools.RCAstro(opts.rcastro)
        spec = cfg.get("spec") or rc.spec(key)
        vals = cfg.get("values", {})
        if key == "bxt":
            vals, _note = bxt_values_for(masters, vals)
        args = spec.args(vals, cfg.get("mode"))
        out.append((key, json.dumps([list(args), cfg.get("mode")], default=str), (spec, args)))
    return out


def resume_point(cps: PrepareCheckpoints, plan: list) -> int:
    """How many leading steps of plan are already done with the same settings."""
    n = 0
    for (step, sig, _x), done in zip(plan, cps.steps):
        if done["step"] != step or done["sig"] != sig:
            break
        n += 1
    return n


def preview_rgb8(masters: list, palette=None, max_side=1400):
    """Small auto-stretched colour preview of a set (what the window shows)."""
    try:
        pals = palettes_for(masters)
        pal = palette if palette in pals else (pals[0] if pals else None)
        if pal is None:
            return None
        step = max(1, int(np.ceil(max(masters[0].shape) / max_side)))
        small = [type(m)(**{**m.__dict__, "data": m.data[:, ::step, ::step]}) for m in masters]
        return (quick_look(small, pal) * 255 + 0.5).astype(np.uint8)
    except Exception:  # noqa: BLE001 - a preview must never stop the run
        return None


# ==========================================================================
# Running everything
# ==========================================================================

@dataclass
class PrepareOptions:
    crop: bool = True
    crop_rect: tuple | None = None          # None = automatic
    solve: bool = True
    astap: str | None = None
    solve_radius: float = 30.0
    gradient: bool = True
    gradient_tool: str = "graxpert"         # "graxpert" or "builtin"
    graxpert: str | None = None
    gx_smoothing: float = 0.1
    gx_correction: str = "Automatic"     # Automatic: Subtraction, or Division for masters stacked without flats
    rcastro: str | None = None
    xt: dict = field(default_factory=dict)  # key -> {"enabled", "values", "mode"}
    device: str = "auto"
    formats: tuple = ("xisf",)              # linear prepared masters: "xisf" and/or "fits" (may be empty)
    quicklook: bool = True
    image_formats: tuple = ("tif", "jpg", "png")   # stretched quick-look colour image
    palette: str | None = None
    out_dir: str | None = None              # default: <master folder>/Prepared
    work_dir: str | None = None
    checkpoints: bool = True                # save after each step; resume / go back


_TARGETISH = re.compile(r"^(NGC|IC|M|SH2|SH|LDN|LBN|VDB|ABELL|ARP|UGC|PGC|C|B|CED|GUM|RCW|HICKSON|HCG)"
                        r"[\s_-]*\d+", re.I)


def _norm_target(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def target_out_dir(chosen, target: str) -> Path:
    """Where to save when the user picked an output folder. Each target gets its own folder,
    so IC 1805 never lands in a folder left over from NGC 891:
      .../IC1805            -> used as is
      .../NGC891            -> .../IC1805   (the folder is named after a different target)
      .../FromSubZeroStacker-> .../FromSubZeroStacker/IC1805"""
    p = Path(chosen)
    if not target or _norm_target(p.name) == _norm_target(target):
        return p
    safe = re.sub(r'[<>:"/\\|?*]+', "_", str(target)).strip() or "Target"
    if _TARGETISH.match(p.name):
        return p.parent / safe
    return p / safe


def run_prepare(masters: list, opts: PrepareOptions, progress=None, log=None, cancel=None,
                save_image=None, step_preview=None) -> dict:
    """Prepare one set of masters (same target / grid). save_image(path, rgb8) writes
    JPG/PNG (the GUI supplies it). Returns {"files", "plan", "report"}."""
    t0 = time.time()
    lg = log or (lambda s: None)

    def pr(text, frac):
        if progress:
            progress(text, frac)
        if cancel and cancel():
            raise Cancelled()

    def can():
        return bool(cancel and cancel())

    target = masters[0].target
    out_dir = target_out_dir(opts.out_dir, target) if opts.out_dir else masters[0].path.parent / "Prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fbprep_", dir=opts.work_dir or None))
    n = len(masters)
    files = []

    plan = plan_steps(masters, opts)
    if any(step == "bxt" for step, _s, _x in plan):
        _v, _note = bxt_values_for(masters, (opts.xt.get("bxt") or {}).get("values", {}))
        if _note:
            lg(_note)
    cps = PrepareCheckpoints([m.path for m in masters], target) if opts.checkpoints else None
    stars_images = {}                      # master index -> stars-only image (StarXTerminator)
    start = 0
    if cps is not None:
        start = resume_point(cps, plan)
        if len(cps.steps) > start:
            lg("Settings changed since step " + cps.steps[start]["title"] + " — redoing it and the steps after it")
        cps.truncate(start)
        if start:
            cps.restore(start, masters, stars_images)
            lg("Resuming from saved progress — already done: " + ", ".join(t["title"] for t in cps.steps))
    overlap = None
    if start == 0 and needs_line_up(masters):
        pr("Lining up the masters", 0.0)
        overlap = line_up(masters, lg)
        if overlap is not None and not any(step == "crop" for step, _s, _e in plan):
            for m in masters:
                crop(m, overlap)
            lg(f"Trimmed to the area all masters cover: {overlap}")
            overlap = None
    if opts.solve and not opts.astap:
        lg("Plate solve skipped — set the ASTAP folder to enable it")
    for key in ("bxt", "nxt", "sxt"):
        if (opts.xt.get(key) or {}).get("enabled") and not opts.rcastro:
            lg(f"{key.upper()} skipped — set the RC-Astro folder to enable it")
    rc = tools.RCAstro(opts.rcastro) if opts.rcastro else None
    span = {"crop": (0.0, 0.03), "solve": (0.03, 0.12), "gradient": (0.15, 0.15),
            "bxt": (0.3, 0.15), "nxt": (0.45, 0.15), "sxt": (0.6, 0.15)}

    for idx, (step, sig, extra) in enumerate(plan):
        if idx < start:
            continue
        s0, ds = span[step]
        before = len(masters[0].steps)
        if step == "crop":
            pr("Cropping", s0)
            if overlap is not None:                  # stay inside the area all lined-up masters cover
                extra = _intersect(tuple(int(v) for v in extra), overlap)
            edge_note = crop_outside_note(extra, common_autocrop(masters))
            for m in masters:
                crop(m, extra)
            lg(f"Cropped {n} master(s) to {extra}")
            if edge_note:
                lg(edge_note)
        elif step == "solve":
            astap = tools.ASTAP(opts.astap)
            solve_memo = {}
            for i, m in enumerate(masters):
                pr(f"Plate solving {m.name}", s0 + ds * i / n)
                ra, dec, fov = solve_hints(m)
                tmp = work / f"solve_{i}.fits"
                _write_temp(m, tmp, mono=True)
                try:
                    cards, note = solve_with_retries(astap, tmp, ra, dec, fov, opts.solve_radius, can,
                                                     lambda t: pr(f"Plate solving {m.name}: {t}", s0 + ds * i / n),
                                                     solve_memo)
                    if note:
                        lg(note)
                    apply_wcs(m, cards)
                    c = {k: v for k, v, _ in cards}
                    m.steps.append(("solve", f"Plate solved with ASTAP (RA {c.get('CRVAL1', 0):.4f}°, "
                                             f"Dec {c.get('CRVAL2', 0):.4f}°)"))
                    lg(f"Solved {m.name}: RA {c.get('CRVAL1', 0):.4f}, Dec {c.get('CRVAL2', 0):.4f}")
                except tools.ToolCancelled:
                    raise Cancelled()
                except tools.ToolError as e:
                    lg(f"Plate solve failed for {m.name}: {e}")
                except Exception as e:  # noqa: BLE001 - plate solving is optional: never stop the run
                    lg(f"Plate solve skipped for {m.name} (unexpected problem: {type(e).__name__}: {e})")
        elif step == "gradient":
            for i, m in enumerate(masters):
                pr(f"Removing gradients: {m.name}", s0 + ds * i / n)
                if extra == "graxpert":
                    tmp = work / f"bge_{i}.fits"
                    sc = _write_temp(m, tmp)
                    try:
                        corr = opts.gx_correction
                        if corr not in ("Subtraction", "Division"):
                            corr, why = gradient_correction_for(m)
                            lg(f"    {m.name}: GraXpert correction {corr} — {why}")
                        res = tools.GraXpert(opts.graxpert).background_extraction(
                            tmp, work / f"bge_{i}_out", smoothing=opts.gx_smoothing, correction=corr,
                            cancel=can)
                        out = _read_result(res, m.data, sc, lg)
                        how = orientation_of(out, m.data)
                        if how != "same":
                            out, = turn_back(how, out)
                            lg(f"    GraXpert returned the picture {how} — turned it back")
                        m.data = out
                        m.steps.append(("gradient", f"Gradients removed with GraXpert "
                                                    f"({corr.lower()}, smoothing {opts.gx_smoothing:g})"))
                    except tools.ToolCancelled:
                        raise Cancelled()
                    except tools.ToolError as e:
                        lg(f"GraXpert failed for {m.name}: {e} — using the built-in method instead")
                        m.data = np.stack([builtin_gradient(p) for p in m.data])
                        m.steps.append(("gradient", "Gradients removed (built-in polynomial model)"))
                else:
                    m.data = np.stack([builtin_gradient(p) for p in m.data])
                    m.steps.append(("gradient", "Gradients removed (built-in polynomial model)"))
        else:                               # RC-Astro bxt / nxt / sxt
            key = step
            spec, args = extra
            for i, m in enumerate(masters):
                base = s0 + ds * i / n
                pr(f"{spec.name}: {m.name}", base)
                tmp_in = work / f"{key}_{i}.fits"
                tmp_out = work / f"{key}_{i}_out.fits"
                sc = _write_temp(m, tmp_in)
                try:
                    written = rc.run(key, tmp_in, tmp_out, args, device=opts.device,
                                     progress=lambda f, t, b=base, mm=m: pr(f"{spec.name}: {mm.name} {t}",
                                                                            b + ds / n * f),
                                     log=lg, cancel=can)
                except tools.ToolCancelled:
                    raise Cancelled()
                main = next((w for w in written if "star" not in Path(w).stem.lower()), str(tmp_out))
                out = _read_result(main, m.data, sc, lg)
                extra_out = [w for w in written if "star" in Path(w).stem.lower()]
                stars_out = _read_result(extra_out[0], m.data, sc, check_level=False) if extra_out else None
                if key == "sxt" and stars_out is None:
                    lg(f"    {spec.name} did not hand back a stars-only image — making it from the "
                       "difference")
                # starless + stars should add up to what was sent: the best check of which way up it is
                how = orientation_of(out if stars_out is None else out + stars_out, m.data)
                if how != "same":
                    out, stars_out = turn_back(how, out, stars_out)
                    lg(f"    {spec.name} returned the picture {how} — turned it back")
                if key == "sxt" and stars_out is None:
                    stars_out = np.clip(m.data - out, 0, None).astype(np.float32)
                m.data = out
                if stars_out is not None:
                    stars_images[i] = stars_out
                desc = " ".join(args) or "default settings"
                m.steps.append((key, f"{spec.name} ({desc})"))
                lg(f"{spec.name} done: {m.name}  [{desc}]")
        rgb_step = None
        if step_preview is not None or cps is not None:
            try:
                rgb_step = preview_rgb8(masters, opts.palette)
            except Exception:  # noqa: BLE001 - a preview never stops the run
                rgb_step = None
        if step_preview is not None and rgb_step is not None:
            step_preview(STEP_TITLES[step], rgb_step)      # the picture after each step, while the run goes on
        if cps is not None:
            pr(f"Saving progress ({STEP_TITLES[step]})", s0 + ds)
            detail = "; ".join(t for _k, t in masters[0].steps[before:]) or STEP_TITLES[step]
            try:
                cps.save(step, sig, detail, masters, stars_images,
                         rgb_step if rgb_step is not None else preview_rgb8(masters, opts.palette))
            except OSError as e:            # e.g. disk full - keep going without checkpoints
                lg(f"Could not save progress ({e}) — continuing without checkpoints")
                cps.truncate(len(cps.steps))
                cps = None

    # 7. save the prepared masters
    for i, m in enumerate(masters):
        pr(f"Saving {m.name}", 0.8 + 0.1 * i / n)
        hdr = [(k, v, c) for k, v, c in m.cards if k not in ("HISTORY",)]
        hdr += [("HISTORY", f"SubZeroAstro prepare: {t}", "") for _k, t in m.steps]
        hdr.append(("PREPARED", True, "prepared for PixInsight by SubZeroAstro"))
        try:
            hdr.append(("ORIGFILE", Path(m.path).name[:68], "stacked master this was prepared from"))
        except Exception:  # noqa: BLE001
            pass
        if i in stars_images:
            hdr.append(("STARLESS", True, "stars removed (StarXTerminator); stars in *_stars file"))
        base = out_dir / f"{m.name}_prepared"
        data = m.data[0] if m.data.shape[0] == 1 else m.data
        if not opts.formats:
            continue
        written = st.save_master_file(base, data, hdr, opts.formats, "MasterLight")
        m.outputs += written
        files += written
        if i in stars_images:
            sd = stars_images[i]
            sw = st.save_master_file(out_dir / f"{m.name}_prepared_stars", sd[0] if sd.shape[0] == 1 else sd,
                                     hdr + [("STARONLY", True, "stars-only image from StarXTerminator")],
                                     opts.formats, "MasterLight")
            files += sw

    # 8. quick look
    img_fmts = tuple(f.lower().lstrip(".") for f in (opts.image_formats or ()))
    if opts.quicklook and img_fmts:
        pal = opts.palette or (palettes_for(masters) or [None])[0]
        if pal:
            pr(f"Quick-look image ({pal})", 0.92)
            rgb = quick_look(masters, pal)
            stem = f"{target}_{re.sub(r'[^A-Za-z0-9]+', '', pal.split(' ')[0])}_quicklook"
            if "tif" in img_fmts or "tiff" in img_fmts:
                tif = out_dir / f"{stem}.tif"
                write_tiff16(tif, rgb)
                files.append(tif)
            if save_image:
                rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
                for ext in [e for e in ("jpg", "png") if e in img_fmts]:
                    p = out_dir / f"{stem}.{ext}"
                    if save_image(p, rgb8):
                        files.append(p)

    # 9. the PixInsight plan
    steps = pixinsight_plan(masters)
    report = out_dir / f"{target}_PixInsight_next_steps.html"
    report.write_text(plan_html(target, masters, steps, files), encoding="utf-8")
    (out_dir / f"{target}_PixInsight_next_steps.txt").write_text(plan_text(target, steps), encoding="utf-8")
    files.append(report)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    lg(f"Prepared {n} master(s) for {target} in {time.time() - t0:.0f} s → {out_dir}")
    pr("Done", 1.0)
    return {"files": files, "plan": steps, "report": report, "out_dir": out_dir,
            "checkpoints": cps, "resumed": start}
