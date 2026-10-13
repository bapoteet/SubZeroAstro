"""
finish.py - Stage 5 (0.7): from prepared linear masters to a finished picture.

Every tool is written the same way so the window can stay simple:

    tool.applicable(state) -> (ok, reason)
    tool.variants(state)   -> [Variant(label, params, note)]   4-5 choices to pick from
    tool.apply(state, params) -> new State

Statistics (black points, star colours, ...) are measured on the full image inside
variants() and stored in the params, so a thumbnail and the full-size result use
exactly the same numbers - what you pick is what you get.

Images are float32 in 0..1, shape (channels, h, w).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import checkpoints as cp


# ==========================================================================
# The image being finished
# ==========================================================================

@dataclass
class State:
    rgb: np.ndarray | None = None          # main image (3|1, h, w) 0..1
    channels: dict = field(default_factory=dict)       # filter -> (h, w), mono sets before combining
    channel_stars: dict = field(default_factory=dict)  # filter -> (h, w) stars-only, if Prepare separated them
    stars: np.ndarray | None = None        # stars-only image (3|1, h, w)
    linear: bool = True
    stars_linear: bool = True
    stars_out: bool = False                # stars currently separated from rgb
    done: list = field(default_factory=list)       # tool keys applied
    skipped: list = field(default_factory=list)
    kind: str = "osc"                      # osc / smart / mono_rgb / mono_lrgb / mono_nb / mono
    target: str = ""
    sources: list = field(default_factory=list)     # master file paths
    palette: str | None = None
    filter: str | None = None              # filter of the masters (as set by the user if overridden)
    lin_lum: np.ndarray | None = None      # (h, w) unstretched luminance, kept when the image is brightened
    sat_core: bool = False                 # the brightest area was clipped in the masters themselves
    stars_missing: bool = False            # starless master whose stars-only file was not found
    stars_hidden: bool = False             # stars separated by Prepare but still shown in the picture:
                                           # "Take Stars Out" makes the separation visible

    def copy(self) -> "State":
        return replace(self, channels=dict(self.channels), channel_stars=dict(self.channel_stars),
                       done=list(self.done), skipped=list(self.skipped), sources=list(self.sources))

    @property
    def is_color(self) -> bool:
        return self.rgb is not None and self.rgb.shape[0] == 3

    @property
    def shape(self):
        if self.rgb is not None:
            return self.rgb.shape[1:]
        return next(iter(self.channels.values())).shape

    def _map(self, fn) -> "State":
        s = self.copy()
        if s.rgb is not None:
            s.rgb = fn(s.rgb)
        if s.stars is not None:
            s.stars = fn(s.stars)
        if s.lin_lum is not None:
            s.lin_lum = fn(s.lin_lum[None])[0]
        s.channels = {k: fn(v[None])[0] for k, v in s.channels.items()}
        s.channel_stars = {k: fn(v[None])[0] for k, v in s.channel_stars.items()}
        return s

    def small(self, max_side=480) -> "State":
        """A downsized copy for thumbnails (pixel averaging, so small stars stay visible)."""
        h, w = self.shape
        f = max(1, int(np.ceil(max(h, w) / max_side)))
        if f == 1:
            return self.copy()

        def shrink(a):
            c, hh, ww = a.shape
            hh2, ww2 = hh // f * f, ww // f * f
            return np.ascontiguousarray(a[:, :hh2, :ww2].reshape(c, hh2 // f, f, ww2 // f, f).mean(axis=(2, 4)),
                                        dtype=np.float32)
        return self._map(shrink)

    def crop(self, size=420, center=None) -> "State":
        """A full-resolution piece (for tools whose effect depends on pixel scale)."""
        h, w = self.shape
        cy, cx = center or (h // 2, w // 2)
        y0, x0 = max(0, min(h - size, cy - size // 2)), max(0, min(w - size, cx - size // 2))
        return self._map(lambda a: np.ascontiguousarray(a[:, y0:y0 + size, x0:x0 + size]))


KIND_LABEL = {"osc": "colour camera", "smart": "smart telescope", "osc_dual": "colour camera + dual-band filter",
              "osc_sho": "colour camera, two dual-band filters (Ha+OIII and SII+OIII)", "mono_rgb": "mono RGB",
              "mono_lrgb": "mono LRGB", "mono_nb": "mono narrowband", "mono": "mono (one filter)"}


# ==========================================================================
# Loading prepared masters
# ==========================================================================

def _stars_sibling(path: Path):
    for ext in (path.suffix, ".xisf", ".fits", ".fit"):
        p = path.with_name(path.stem + "_stars" + ext)
        if p.exists():
            return p
    return None


def dual_band_filter(filter_name, look=None) -> bool:
    """A filter that passes only Ha and OIII (and maybe Hb) on a colour camera:
    L-Ultimate, L-eXtreme, L-eNhance, ALP-T, NBZ ... and the Seestar's "LP" filter."""
    import fits_core as core
    if core.is_dual_band(filter_name):
        return True
    smart = core.smart_scope(look or {}) or ""
    return smart.startswith("Seestar") and str(filter_name or "").strip().upper() == "LP"


def load_state(paths) -> State:
    """Build the starting state from one target's masters (prepared or not)."""
    import prepare as P
    masters = [P.load_master(p) for p in paths]
    s = State(target=masters[0].target, sources=[str(m.path) for m in masters])
    color = [m for m in masters if m.is_color]
    smart = any(bool(__import__("fits_core").smart_scope(m.look)) for m in masters)

    def stars_of(m):
        sp = _stars_sibling(m.path)
        if sp is None:
            return None
        try:
            return np.clip(P.load_master(sp).data / 65535.0, 0, None).astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    duals = [m for m in color if dual_band_filter(m.filter, m.look)]
    if len(duals) >= 2 and len({__import__("fits_core").band_lines(m.filter)[0] for m in duals}) >= 2:
        # e.g. L-Ultimate (Ha+OIII) + L-Synergy (SII+OIII): a full SHO set from a colour camera
        import fits_core as core
        o3s, o3_stars = [], []
        for m in duals:
            rgb = np.clip(m.data / 65535.0, 0, None).astype(np.float32)
            line, o3 = split_dual_band(rgb)
            name = core.band_lines(m.filter)[0]
            s.channels[name] = line
            o3s.append(o3)
            st_img = stars_of(m)
            if st_img is not None and st_img.shape == rgb.shape:
                s.channel_stars[name] = st_img[0]
                o3_stars.append((st_img[1] + st_img[2]) / 2)
        s.channels["OIII"] = np.mean(o3s, axis=0).astype(np.float32)        # OIII from both filters
        if len(o3_stars) == len(duals):
            s.channel_stars["OIII"] = np.mean(o3_stars, axis=0).astype(np.float32)
        else:
            s.channel_stars = {}
        s.kind = "osc_sho"
        s.filter = " + ".join(m.filter for m in duals)
        return s
    if color:
        m = color[0]
        s.rgb = np.clip(m.data / 65535.0, 0, None).astype(np.float32)
        s.kind = "smart" if smart else "osc"
        s.filter = m.filter
        if dual_band_filter(m.filter, m.look):
            s.kind = "osc_dual"
        st_img = stars_of(m)
        if st_img is not None and st_img.shape == s.rgb.shape:
            s.stars, s.stars_out, s.stars_hidden = st_img, True, True
        elif m.look.get("STARLESS"):
            # already starless but the stars file is missing: never run star removal twice
            s.done.append("stars_out")
            s.stars_missing = True
        return s
    for m in masters:
        s.channels[m.filter] = np.clip(m.data[0] / 65535.0, 0, None).astype(np.float32)
        st_img = stars_of(m)
        if st_img is not None:
            s.channel_stars[m.filter] = st_img[0]
    f = set(s.channels)
    if {"Ha", "OIII"} <= f:
        s.kind = "mono_nb"
    elif {"R", "G", "B"} <= f:
        s.kind = "mono_lrgb" if "L" in f else "mono_rgb"
    else:
        s.kind = "mono"
        k = sorted(f)[0]
        s.rgb = s.channels[k][None].copy()
        if k in s.channel_stars:
            s.stars, s.stars_out, s.stars_hidden = s.channel_stars[k][None].copy(), True, True
        s.channels = {}
        s.channel_stars = {}
    return s


# ==========================================================================
# Small helpers
# ==========================================================================

def _stats(x: np.ndarray):
    """Robust background (median) and noise (MAD sigma) of a plane."""
    s = x[::3, ::3]                       # odd step: never locks onto a 2x2 Bayer pattern
    s = s[np.isfinite(s)]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826 + 1e-7
    return med, mad


def _lum(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=0) if img.shape[0] > 1 else img[0]


def _mtf(m, x):
    """PixInsight's midtones transfer function."""
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def _min_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Grey-level erosion with a (2r+1)^2 square (separable)."""
    h, w = a.shape
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    b = np.min(np.stack([p[:, i:i + w] for i in range(2 * r + 1)]), axis=0)
    p = np.pad(b, ((r, r), (0, 0)), mode="edge")
    return np.min(np.stack([p[i:i + h, :] for i in range(2 * r + 1)]), axis=0)


def display_rgb8(img: np.ndarray, linear: bool, linked=True) -> np.ndarray:
    """(h, w, 3) uint8 for showing a state on screen (auto-stretched if still linear)."""
    if linear:
        out = []
        if linked:
            med, mad = _stats(_lum(img))
            lo = med - 2.8 * mad
            hi = max(float(np.percentile(_lum(img)[::3, ::3], 99.98)), lo + 1e-6)
            nm = min(max((med - lo) / (hi - lo), 1e-6), 0.999)
            m = nm * (1 - 0.22) / (nm - 2 * nm * 0.22 + 0.22)
            for c in img:
                out.append(_mtf(m, np.clip((c - lo) / (hi - lo), 0, 1)))
        else:
            for c in img:
                med, mad = _stats(c)
                lo = med - 2.8 * mad
                hi = max(float(np.percentile(c[::3, ::3], 99.98)), lo + 1e-6)
                nm = min(max((med - lo) / (hi - lo), 1e-6), 0.999)
                m = nm * (1 - 0.22) / (nm - 2 * nm * 0.22 + 0.22)
                out.append(_mtf(m, np.clip((c - lo) / (hi - lo), 0, 1)))
        img = np.stack(out)
    img = np.clip(img, 0, 1)
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    return (img.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)


def display_fixed(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Auto-stretch img with the stretch worked out on ref, so several versions of one
    image are shown with the same settings and can be compared."""
    out = []
    for c, r in zip(img, ref):
        med, mad = _stats(r)
        lo = med - 2.8 * mad
        hi = max(float(np.percentile(r[::3, ::3], 99.98)), lo + 1e-6)
        nm = min(max((med - lo) / (hi - lo), 1e-6), 0.999)
        m = nm * (1 - 0.22) / (nm - 2 * nm * 0.22 + 0.22)
        out.append(_mtf(m, np.clip((c - lo) / (hi - lo), 0, 1)))
    return (np.clip(np.stack(out), 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)


STAR_TOOLS = ("star_stretch", "shrink", "star_color")


def state_display(s: State, linked=True, with_stars=None) -> np.ndarray:
    """What the user should see for a state. With the stars taken out you see the nebula on
    its own; the stars are shown when they were not taken out yet (stars_hidden), while the
    star steps are worked on, or when with_stars=True."""
    if s.rgb is None:                          # mono set not combined yet: show a quick colour mix
        ch = {k: v + s.channel_stars.get(k, 0) for k, v in s.channels.items()}
        keys = [k for k in ("R", "G", "B") if k in ch] if s.kind in ("mono_rgb", "mono_lrgb") else []
        if len(keys) == 3:
            img = np.stack([ch["R"], ch["G"], ch["B"]])
        elif "Ha" in ch and "OIII" in ch:
            img = np.stack([ch.get("SII", ch["Ha"]), ch["Ha"], ch["OIII"]])
        else:
            img = next(iter(ch.values()))[None]
        return display_rgb8(img, True, linked=False)
    if with_stars is None:
        with_stars = s.stars_hidden or bool(s.done and s.done[-1] in STAR_TOOLS)
    img = s.rgb
    if not (with_stars and s.stars_out and s.stars is not None):
        return display_rgb8(img, s.linear, linked)
    st_ = s.stars if s.stars.shape[0] == img.shape[0] else np.repeat(s.stars, img.shape[0], 0)
    if s.linear and s.stars_linear:
        return display_rgb8(img + st_, True, linked)
    if not s.linear and not s.stars_linear:
        return display_rgb8(1 - (1 - img) * (1 - np.clip(st_, 0, 1)), False, linked)
    # one brightened, the other not: put both on screen scale, then screen-blend
    a = display_rgb8(img, s.linear, linked).astype(np.float32) / 255
    b = display_rgb8(st_, s.stars_linear, linked).astype(np.float32) / 255
    if s.stars_linear:                         # unstretched stars: keep only the stars, not the noise floor
        b = np.clip((b - 0.12) / 0.88, 0, 1)
    return (np.clip(1 - (1 - a) * (1 - b), 0, 1) * 255 + 0.5).astype(np.uint8)


def merge_hidden_stars(s: State) -> State:
    """The stars were never taken out (step skipped): put them back into the picture for good."""
    if not s.stars_hidden or s.stars is None or s.rgb is None:
        return s
    s = s.copy()
    st_ = s.stars if s.stars.shape[0] == s.rgb.shape[0] else np.repeat(s.stars, s.rgb.shape[0], 0)
    if s.linear and s.stars_linear:
        s.rgb = (s.rgb + st_).astype(np.float32)
    else:
        s.rgb = (1 - (1 - np.clip(s.rgb, 0, 1)) * (1 - np.clip(st_, 0, 1))).astype(np.float32)
    s.stars, s.stars_out, s.stars_hidden = None, False, False
    return s


# tools that work on picture and stars side by side, so stars from Prepare can stay separate
KEEPS_STARS_APART = {"stars_out", "palette", "combine", "balance", "scnr", "ha_glow"}


# ==========================================================================
# Tools
# ==========================================================================

@dataclass
class Variant:
    label: str
    params: dict
    note: str = ""
    recommended: bool = False


@dataclass
class Slider:
    """One plain-language fine-tune control. value = where the picked version is."""
    name: str
    label: str
    lo: float
    hi: float
    value: float
    tip: str = ""
    step: float = 0.01


class Tool:
    key = ""
    title = ""
    tip = ""
    pick = True                    # show thumbnails to choose from
    spatial = False                # effect depends on pixel scale -> thumbnails are 1:1 crops
    optional = False               # offered, but not pushed by "Recommended next"
    editor = None                  # "curve" -> the window also shows a curve you can drag
    shows_stars = False            # previews show the stars too (star steps)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "apply" in cls.__dict__ and cls.__dict__.get("key", "") not in KEEPS_STARS_APART:
            orig = cls.__dict__["apply"]

            def apply(self, s, params, _orig=orig):
                return _orig(self, merge_hidden_stars(s), params)
            apply.__doc__ = orig.__doc__
            cls.apply = apply

    def sliders(self, s: "State", params: dict) -> list:
        """Fine-tune controls, starting at the picked version's values."""
        return []

    def tuned(self, s: "State", params: dict, values: dict) -> dict:
        """Picked version's params with the fine-tune values applied."""
        p = dict(params)
        p.update(values)
        return p

    def applicable(self, s: "State"):
        return True, ""

    def possible(self, s: "State") -> bool:
        """Can this step happen at some point (maybe after earlier steps)?"""
        return True

    def variants(self, s: State) -> list:
        return [Variant("Apply", {})]

    def apply(self, s: State, params: dict) -> State:
        raise NotImplementedError

    def why(self, s: "State") -> str:
        return self.tip

    def is_optional(self, s: "State") -> bool:
        return self.optional

    def preview(self, s: "State", params: dict) -> np.ndarray:
        """What a version looks like (thumbnails and the live fine-tune preview)."""
        return state_display(self.apply(s, params), with_stars=True if self.shows_stars else None)


# --------------------------------------------------------------------------
def nebula_top(x: np.ndarray) -> float:
    """Brightness of the brightest nebula / galaxy (not stars): stars are eroded away first."""
    sm = x[::2, ::2]
    er = _min_filter(sm, 2)
    return float(np.percentile(er[::2, ::2], 99.8))


def channel_stats(x: np.ndarray):
    med, mad = _stats(x)
    return med, mad, max(nebula_top(x), med + 8 * mad)


PALETTE_INFO = {
    "HOO": ("HOO", "Ha red, OIII teal - the natural bicolour look"),
    "HOO+": ("HOO (vivid)", "Bicolour with a synthetic green for warmer tones"),
    "HHO": ("Gold & blue", "Hubble-like gold and blue from two filters (green is turned gold by Remove Green Cast)"),
    "FORAXX": ("Foraxx (dynamic)", "Colours shift with brightness: gold where Ha dominates, blue where OIII does - "
                                   "a popular modern palette"),
    "SHO": ("Hubble (SHO)", "SII red, Ha green, OIII blue - the classic Hubble look (green is tamed later)"),
    "HSO": ("HSO", "Ha red, SII green, OIII blue"),
    "HOO+S": ("HOO + SII", "Natural-looking reds and teals"),
}


def build_palette(pal: str, n: dict) -> np.ndarray:
    """Colour image (3, h, w) from normalised narrowband channels {'Ha', 'OIII', 'SII'?}."""
    h, o = n["Ha"], n["OIII"]
    sii = n.get("SII", h)
    if pal == "SHO":
        return np.stack([sii, h, o])
    if pal == "HSO":
        return np.stack([h, sii, o])
    if pal == "HOO+S":
        return np.stack([0.7 * h + 0.3 * sii, o, o])
    if pal == "HOO+":
        return np.stack([h, 0.5 * h + 0.5 * o, o])
    if pal == "HHO":
        return np.stack([h, 0.85 * h + 0.15 * o, o])
    if pal == "FORAXX":                                   # Foraxx dynamic palette (S = Ha when there's no SII)
        oc, oh = np.clip(o, 0, 1), np.clip(o * h, 0, 1)
        t = oc ** (1 - oc)
        u = oh ** (1 - oh)
        return np.stack([t * sii + (1 - t) * h, u * h + (1 - u) * o, o])
    return np.stack([h, o, o])                           # HOO


def split_dual_band(rgb: np.ndarray):
    """Ha and OIII from a colour camera behind a dual-band filter: Ha is recorded by the
    red pixels, OIII by the green and blue ones. Backgrounds removed."""
    r, g, b = rgb
    ha = r - _stats(r)[0]
    o3 = (g - _stats(g)[0] + b - _stats(b)[0]) / 2
    return ha.astype(np.float32), o3.astype(np.float32)


class CombineChannels(Tool):
    key = "combine"
    title = "Combine Colour Channels"
    tip = ("Builds one colour picture from your separate filter images (R, G, B, L, Ha, OIII, SII). "
           "Pick the colour mix you like.")

    def applicable(self, s):
        if s.rgb is not None:
            return False, "Already a colour image"
        if len(s.channels) < 2:
            return False, "Needs at least two filter images"
        return True, ""

    def variants(self, s):
        f = set(s.channels)
        out = []
        if {"R", "G", "B"} <= f:
            if "L" in f:
                out.append(Variant("LRGB", {"palette": "LRGB"}, "Colour from RGB, detail from L", True))
            out.append(Variant("RGB", {"palette": "RGB"}, "True colour", "L" not in f))
        if {"Ha", "OIII", "SII"} <= f:
            out += [Variant(*PALETTE_INFO[k][:1], {"palette": k}, PALETTE_INFO[k][1], k == "SHO")
                    for k in ("SHO", "FORAXX", "HSO", "HOO+S")]
        elif {"Ha", "OIII"} <= f:
            out += [Variant(PALETTE_INFO[k][0], {"palette": k}, PALETTE_INFO[k][1], k == "HOO")
                    for k in ("HOO", "HOO+", "HHO", "FORAXX")]
        if {"Ha"} <= f and {"R", "G", "B"} <= f:
            out.append(Variant("RGB + Ha glow", {"palette": "RGB+Ha"}, "True colour with the hydrogen glow "
                               "added to red"))
        # pre-measure each channel so thumbnails match the full result
        stats = {k: channel_stats(v) for k, v in s.channels.items()}
        for v in out:
            v.params["stats"] = stats
        return out

    def sliders(self, s, params):
        m = params.get("mix", (1.0, 1.0, 1.0))
        return [Slider("r", "Red strength", 0.4, 1.6, m[0], "How strong the filter shown as red is"),
                Slider("g", "Green strength", 0.4, 1.6, m[1], "How strong the filter shown as green is"),
                Slider("b", "Blue strength", 0.4, 1.6, m[2], "How strong the filter shown as blue is")]

    def tuned(self, s, params, values):
        p = dict(params)
        p["mix"] = [values.get("r", 1.0), values.get("g", 1.0), values.get("b", 1.0)]
        return p

    @staticmethod
    def _norm(x, st_):
        """Each filter scaled so its sky sits at 0.01 and its brightest real signal near 0.9
        (a 'balanced' mix: weak filters like SII are lifted to match)."""
        med, mad, hi = st_
        k = 0.5 / max(hi - med, 1e-6)                    # nebula at ~0.5, stars above
        ped = 0.01 + 5.0 * mad * k                       # sky sits clear of zero, so its noise isn't clipped
        return np.clip((x - med) * k + ped, 0, None)

    def apply(self, s, params):
        s = s.copy()
        pal = params["palette"]
        stats = params.get("stats") or {k: channel_stats(v) for k, v in s.channels.items()}

        def build(ch, raw=False):
            n = ch if raw else {k: self._norm(v, stats[k]) for k, v in ch.items()}
            if pal in ("RGB", "LRGB", "RGB+Ha"):
                r, g, b = n["R"], n["G"], n["B"]
                if pal == "RGB+Ha" and "Ha" in n:
                    r = np.maximum(r, 0.6 * r + 0.6 * n["Ha"])
                img = np.stack([r, g, b])
                if pal == "LRGB" and "L" in n:
                    y = img.mean(0) + 1e-6
                    img = img * (np.clip(n["L"], 0, None) / np.clip(y, 1e-6, None))[None]
                return img
            return build_palette(pal, n)

        mix = np.asarray(params.get("mix", (1.0, 1.0, 1.0)), np.float32)[:, None, None]
        s.rgb = np.clip(build(s.channels) * mix, 0, None).astype(np.float32)
        if s.channel_stars and set(s.channel_stars) >= set(s.channels):
            s.stars = np.clip(build({k: v * 0.5 / max(stats[k][2] - stats[k][0], 1e-6)
                                     for k, v in s.channel_stars.items()}, raw=True) * mix, 0, None).astype(np.float32)
            s.stars_out = True
            s.stars_linear = True
            s.stars_hidden = "stars_out" not in s.done
        s.channels, s.channel_stars = {}, {}
        s.palette = pal
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class DualBandPalette(Tool):
    key = "palette"
    title = "Choose a Colour Palette"
    tip = ("Your dual-band filter records hydrogen (Ha) in the red pixels and oxygen (OIII) in the green and "
           "blue ones. SubZeroAstro separates the two and lets you colour them - pick the look you like.")

    def applicable(self, s):
        if s.kind != "osc_dual":
            return False, "For colour cameras with a dual-band filter (L-Ultimate, L-eXtreme, ALP-T, ...)"
        if not s.linear or s.rgb is None or s.rgb.shape[0] != 3:
            return False, "Choose the palette before brightening"
        return True, ""

    def possible(self, s):
        return s.kind == "osc_dual"

    def why(self, s):
        import fits_core as core
        line = core.band_lines(s.filter)[0]
        what = "sulphur (SII, red)" if line == "SII" else "hydrogen (Ha, red)"
        return (f"Your {s.filter or 'dual-band'} filter passes {what} and oxygen (OIII, teal) light. Separate "
                "them and pick how to colour them - from natural to Hubble-like gold and blue.")

    def variants(self, s):
        ha, o3 = split_dual_band(s.rgb)
        stats = {"Ha": channel_stats(ha), "OIII": channel_stats(o3)}
        out = [Variant("As captured", {"palette": "NATURAL"}, "The colours the camera recorded, sky neutralised")]
        out += [Variant(PALETTE_INFO[k][0], {"palette": k, "stats": stats}, PALETTE_INFO[k][1], k == "HOO")
                for k in ("HOO", "HOO+", "HHO", "FORAXX")]
        import fits_core as core
        if core.band_lines(s.filter)[0] == "SII":           # L-Synergy: SII in the red pixels
            for v in out[1:]:
                v.label = v.label.replace("HOO", "SOO")
                v.note = v.note.replace("Ha red", "SII red").replace("where Ha", "where SII")
        return out

    def sliders(self, s, params):
        return CombineChannels.sliders(self, s, params)

    def tuned(self, s, params, values):
        return CombineChannels.tuned(self, s, params, values)

    def _build(self, img, params, stats=None):
        mix = np.asarray(params.get("mix", (1.0, 1.0, 1.0)), np.float32)[:, None, None]
        if params["palette"] == "NATURAL":
            bgs = [_stats(c)[0] for c in img]
            ref = float(np.mean(bgs))
            return np.clip(np.stack([c - b + ref for c, b in zip(img, bgs)]) * mix, 0, None).astype(np.float32)
        ha, o3 = split_dual_band(img)
        st_ = stats or params["stats"]
        n = {"Ha": CombineChannels._norm(ha, st_["Ha"]), "OIII": CombineChannels._norm(o3, st_["OIII"])}
        return np.clip(build_palette(params["palette"], n) * mix, 0, None).astype(np.float32)

    def apply(self, s, params):
        s = s.copy()
        stats = params.get("stats")
        s.rgb = self._build(s.rgb, params)
        if s.stars is not None and s.stars.shape[0] == 3 and s.stars_linear:
            if params["palette"] == "NATURAL":
                s.stars = s.stars
            else:
                ha, o3 = s.stars[0], (s.stars[1] + s.stars[2]) / 2
                n = {"Ha": np.clip(ha * 0.5 / max(stats["Ha"][2] - stats["Ha"][0], 1e-6), 0, None),
                     "OIII": np.clip(o3 * 0.5 / max(stats["OIII"][2] - stats["OIII"][0], 1e-6), 0, None)}
                s.stars = np.clip(build_palette(params["palette"], n), 0, None).astype(np.float32)
        s.palette = params["palette"]
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
def _star_colour_factors(img: np.ndarray):
    """Median R/G and B/G of unsaturated stars (aperture photometry), or None."""
    import stacking as st
    lum = _lum(img)
    try:
        pts, _pk = st.find_stars((lum * 65535.0).astype(np.float32), 1, max_stars=300)
    except Exception:  # noqa: BLE001
        return None
    if len(pts) < 10:
        return None
    h, w = lum.shape
    rs, bs = [], []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (8 <= xi < w - 8 and 8 <= yi < h - 8):
            continue
        box = img[:, yi - 8:yi + 9, xi - 8:xi + 9]
        if box.max() > 0.9:                       # saturated
            continue
        yy, xx = np.mgrid[-8:9, -8:9]
        rr = np.hypot(yy, xx)
        inner, ring = rr <= 3.5, (rr >= 6) & (rr <= 8)
        f = [float((c[inner] - np.median(c[ring])).sum()) for c in box]
        if min(f) <= 0:
            continue
        rs.append(f[0] / f[1])
        bs.append(f[2] / f[1])
    if len(rs) < 8:
        return None
    return float(np.median(rs)), float(np.median(bs)), len(rs)


class BalanceColours(Tool):
    key = "balance"

    def possible(self, s):
        return (s.is_color or (s.rgb is None and len(s.channels) >= 2)) and s.linear
    title = "Balance Colours"
    tip = ("Removes colour casts: makes the empty sky neutral grey and the average star white - "
           "close to what PixInsight's colour calibration does.")

    def applicable(self, s):
        if not s.is_color:
            return False, "Needs a colour image"
        if not s.linear:
            return False, "Works on the unstretched image - use it before Brighten"
        return True, ""

    def variants(self, s):
        bgs = [_stats(c)[0] for c in s.rgb]
        fac = _star_colour_factors(s.rgb)
        base = {"bgs": bgs}
        out = [Variant("Neutral sky only", {**base, "white": None}, "Only removes the background colour cast")]
        if fac:
            rg, bg_, n = fac
            out.insert(0, Variant("Stars white", {**base, "white": [1 / rg, 1.0, 1 / bg_]},
                                  f"Average of {n} stars made white", True))
            out.append(Variant("Stars white, warmer", {**base, "white": [1.06 / rg, 1.0, 0.94 / bg_]},
                               "A touch warmer"))
            out.append(Variant("Stars white, cooler", {**base, "white": [0.95 / rg, 1.0, 1.05 / bg_]},
                               "A touch cooler"))
        else:
            out[0].recommended = True
            out[0].note += " (not enough clean stars to judge star colour)"
        return out

    def sliders(self, s, params):
        return [Slider("warmth", "Warmth", -1.0, 1.0, params.get("warmth", 0.0), "Cooler (bluer) <-> warmer (redder)"),
                Slider("tint", "Tint", -1.0, 1.0, params.get("tint", 0.0), "Magenta <-> green")]

    def apply(self, s, params):
        s = s.copy()
        bgs = params["bgs"]
        w, t = float(params.get("warmth", 0.0)), float(params.get("tint", 0.0))
        if w or t:
            base = np.asarray(params.get("white") or [1.0, 1.0, 1.0], np.float32)
            params = dict(params, white=list(base * np.array([1 + 0.08 * w, 1 + 0.06 * t, 1 - 0.08 * w])))
        ref = float(np.mean(bgs))
        img = np.stack([c - b for c, b in zip(s.rgb, bgs)])
        if params.get("white"):
            img = img * np.asarray(params["white"], np.float32)[:, None, None]
        s.rgb = np.clip(img + ref, 0, None).astype(np.float32)
        if s.stars is not None and s.stars.shape[0] == 3 and params.get("white"):
            s.stars = (s.stars * np.asarray(params["white"], np.float32)[:, None, None]).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class RemoveGreen(Tool):
    key = "scnr"

    def possible(self, s):
        return s.is_color or (s.rgb is None and len(s.channels) >= 2)
    title = "Remove Green Cast"
    tip = ("Removes the green tint colour cameras and the Hubble palette often leave. Real green "
           "almost never exists in deep-sky objects. (PixInsight: SCNR)")
    fine = ("amount", "Amount", 0.0, 1.0, 0.05)

    def applicable(self, s):
        return (True, "") if s.is_color else (False, "Needs a colour image")

    def variants(self, s):
        g = s.rgb[1] - np.maximum(s.rgb[0], s.rgb[2])
        excess = float(np.mean(g[::3, ::3] > 0))
        best = 1.0 if s.kind in ("mono_nb", "osc_dual", "osc_sho") and s.palette not in ("NATURAL", None) \
            else (0.75 if excess > 0.5 else 0.5)
        return [Variant(f"{int(a * 100)}%", {"amount": a}, "", a == best) for a in (0.25, 0.5, 0.75, 1.0)]

    def sliders(self, s, params):
        return [Slider("amount", "Amount", 0.0, 1.0, params["amount"], "How much of the green cast to remove")]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        r, g, b = s.rgb
        neutral = np.minimum(g, (r + b) / 2)
        s.rgb = np.stack([r, g * (1 - a) + neutral * a, b]).astype(np.float32)
        s.done.append(self.key)
        return s

    def why(self, s):
        if s.kind in ("mono_nb", "osc_dual", "osc_sho") and s.palette not in ("NATURAL", None):
            return "Narrowband palettes are very green straight out of combination - this gives the classic gold and blue."
        return "Colour cameras usually leave a slight green tint after colour balancing."


# --------------------------------------------------------------------------
class TakeStarsOut(Tool):
    key = "stars_out"

    def possible(self, s):
        return s.stars_hidden or (not s.stars_out and self.runner is not None)
    title = "Take Stars Out"
    tip = ("Splits the stars from the nebula or galaxy (StarXTerminator) so each can be brightened "
           "separately - you get bright detail without bloated stars. They go back in at the end.")
    pick = False
    runner = None                  # set by the window: callable(img (C,h,w) 0..1) -> (starless, stars)

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.stars_hidden:
            return True, ""
        if s.stars_out:
            return False, "Stars are already taken out"
        if self.runner is None:
            return False, "Needs StarXTerminator (set the RC-Astro CLI in the Prepare window)"
        return True, ""

    def variants(self, s):
        if s.stars_hidden:
            return [Variant("Take the stars out", {}, "Uses the stars StarXTerminator already separated in "
                                                      "Prepare - instant", True)]
        return [Variant("Separate the stars", {}, "Runs StarXTerminator on the full image", True)]

    def why(self, s):
        if s.stars_hidden:
            return ("Prepare already separated the stars - this takes them out of view (instantly), so the "
                    "nebula can be brightened strongly. They go back in at 'Put Stars Back'.")
        return self.tip

    def apply(self, s, params):
        s = s.copy()
        if s.stars_hidden:
            s.stars_hidden = False
        else:
            starless, stars = self.runner(s.rgb, s.linear)
            s.rgb, s.stars = starless.astype(np.float32), stars.astype(np.float32)
            s.stars_out, s.stars_linear = True, s.linear
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
def ghs(x: np.ndarray, D: float, b: float, SP: float) -> np.ndarray:
    """Generalised hyperbolic stretch (Payne & Cranfield), symmetric about SP,
    normalised to 0..1."""
    D = max(D, 1e-6)

    def g(u):
        if abs(b) < 1e-6:
            return 1 - np.exp(-D * u)
        return 1 - (1 + D * b * u) ** (-1 / b)
    xs = np.clip(x, 0, 1)
    up = g(np.clip(xs - SP, 0, None))
    dn = g(np.clip(SP - xs, 0, None))
    f = np.where(xs >= SP, up, -dn)
    f0, f1 = -g(SP), g(1 - SP)
    return ((f - f0) / (f1 - f0)).astype(np.float32)


class Brighten(Tool):
    key = "stretch"
    title = "Brighten the Image"
    tip = ("Turns the dark, 'linear' data into a visible picture (the stretch). Pick the brightness you "
           "like - stronger shows fainter detail but more noise.")
    methods = {"auto": "Automatic (balanced)", "colour": "Colour-preserving", "ghs": "Hyperbolic (GHS)"}

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if not s.linear:
            return False, "Already brightened"
        return True, ""

    def recommended_method(self, s):
        if s.stars_out:
            return "ghs"                 # starless: strong, smooth stretch for faint nebulosity
        if s.is_color:
            return "colour"              # stars keep their colour
        return "auto"

    LEVELS = (("Gentle", 0.06), ("Light", 0.09), ("Medium", 0.12), ("Strong", 0.16), ("Very strong", 0.21))

    def variants(self, s, method=None, linked=True):
        """Five strengths, the same for every method: the sky ends up at 6 % ... 21 % grey."""
        method = method or self.recommended_method(s)
        img = s.rgb
        planes = [_lum(img)] if linked else list(img)
        st_ = []
        for p in planes:
            med, mad = _stats(p)
            lo = max(0.0, med - 2.8 * mad)
            hi = max(float(np.percentile(p[::3, ::3], 99.995)), lo + 1e-5)
            st_.append((lo, hi, med))
        lo, hi, med = st_[0]
        nm = min(max((med - lo) / (hi - lo), 1e-6), 0.5)
        # neutral sky: line the channel backgrounds up before a linked stretch
        shifts = [med - _stats(c)[0] for c in img] if (linked and img.shape[0] > 1) else None
        out = []
        for i, (name, t) in enumerate(self.LEVELS):
            prm = self._with_target({"method": method, "linked": linked, "stats": st_, "shifts": shifts,
                                     "nm": nm, "protect": 0.0}, t)
            out.append(Variant(name, prm, f"sky at {int(t * 100)}% grey", i == 2))
        return out

    def _with_target(self, prm, t):
        prm = dict(prm, target=float(t))
        nm = prm["nm"]
        if prm["method"] == "ghs":
            prm["lnD"] = self._solve(lambda x: float(ghs(np.array([nm]), np.exp(x) - 1, 6.0, nm / 2)[0]),
                                     t, 0.0, 14.0)
            prm["b"] = 6.0
        elif prm["method"] == "colour":
            prm["beta"] = float(np.exp(self._solve(
                lambda x: float(np.arcsinh(np.exp(x) * nm) / np.arcsinh(np.exp(x))), t, 0.0, 14.0)))
        return prm

    def sliders(self, s, params):
        return [Slider("brightness", "Brightness", 0.03, 0.30, params["target"],
                       "How bright the sky ends up - more shows fainter detail but also more noise", 0.005),
                Slider("protect", "Protect bright areas", 0.0, 1.0, params.get("protect", 0.0),
                       "Keeps galaxy cores and bright nebula from turning white")]

    def tuned(self, s, params, values):
        p = self._with_target(params, values.get("brightness", params["target"]))
        p["protect"] = float(values.get("protect", params.get("protect", 0.0)))
        return p

    @staticmethod
    def _solve(f, target, a, b):
        """x in [a, b] with f(x) = target (f increasing), by bisection."""
        for _ in range(48):
            m = (a + b) / 2
            if f(m) < target:
                a = m
            else:
                b = m
        return (a + b) / 2

    def _stretch_planes(self, img, params):
        method, linked, stt = params["method"], params["linked"], params["stats"]
        shifts = params.get("shifts")
        out = []
        for c, plane in enumerate(img):
            lo, hi, med = stt[0] if linked else stt[c]
            if shifts:
                plane = plane + np.float32(shifts[c])
            n = np.clip((plane - lo) / (hi - lo), 0, 1)
            nm = min(max((med - lo) / (hi - lo), 1e-6), 0.5)
            if method == "auto":
                t = params["target"]
                m = nm * (1 - t) / (nm - 2 * nm * t + t)
                out.append(_mtf(m, n))
            elif method == "ghs":
                out.append(ghs(n, float(np.exp(params["lnD"]) - 1), params["b"], nm / 2))
            else:
                out.append(n)
        out = np.stack(out)
        if method == "colour":            # arcsinh on luminance, colour ratios kept
            k = float(params["beta"])
            y = out.mean(0) if out.shape[0] > 1 else out[0]
            y2 = np.arcsinh(k * y) / np.arcsinh(k)
            ratio = np.where(y > 1e-7, y2 / np.maximum(y, 1e-7), 0)
            out = out * ratio[None]
            mx = out.max(0)
            out = np.where(mx > 1, out / np.maximum(mx, 1e-6), out)
        return np.clip(out, 0, 1).astype(np.float32)

    def apply(self, s, params):
        s = s.copy()
        out = self._stretch_planes(s.rgb, params)
        pr = float(params.get("protect", 0.0))
        if pr > 0:                      # blend bright areas towards a gentler stretch
            mild = self._stretch_planes(s.rgb, self._with_target(params, max(0.02, params["target"] * 0.45)))
            y = out.mean(0) if out.shape[0] > 1 else out[0]
            w = (pr * np.clip((y - params["target"]) / max(1e-3, 1 - params["target"]), 0, 1) ** 0.7)[None]
            out = (out * (1 - w) + mild * w).astype(np.float32)
        lin = _lum(s.rgb)
        s.lin_lum = lin.astype(np.float32)
        s.sat_core = _core_saturated(s.rgb)
        s.rgb = out
        s.linear = False
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class BrightenStars(Tool):
    def possible(self, s):
        return s.stars_out or bool(s.channel_stars) or TOOLS["stars_out"].possible(s)

    key = "star_stretch"
    shows_stars = True
    title = "Brighten Stars Separately"
    tip = ("Brightens the stars-only image gently, so stars stay small and colourful instead of "
           "turning into big white blobs.")

    def applicable(self, s):
        if not s.stars_out or s.stars is None or s.stars_hidden:
            return False, "Take the stars out first"
        if not s.stars_linear:
            return False, "Stars already brightened"
        return True, ""

    def variants(self, s):
        # black point just above the noise, so the stretch brightens stars, not background grain
        bgs = [max(0.0, m + 2.5 * n) for m, n in (_stats(c) for c in s.stars)]
        hi = float(np.percentile(s.stars[:, ::4, ::4], 99.99)) + 1e-6
        return [Variant(["Subtle", "Light", "Medium", "Strong", "Bold"][i], {"s": v, "bgs": bgs, "hi": hi},
                        "", i == 2) for i, v in enumerate((3.0, 4.0, 5.0, 6.0, 7.0))]

    def sliders(self, s, params):
        return [Slider("s", "Star brightness", 2.0, 8.0, params["s"], "How bright the stars become", 0.1)]

    def apply(self, s, params):
        s = s.copy()
        k = 3.0 ** float(params["s"])
        x = np.clip(np.stack([(c - b) for c, b in zip(s.stars, params["bgs"])]) / params["hi"], 0, 1)
        s.stars = np.clip((k * x) / ((k - 1) * x + 1), 0, 1).astype(np.float32)
        s.stars_linear = False
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class ShrinkStars(Tool):
    def possible(self, s):
        return s.stars_out or bool(s.channel_stars) or TOOLS["stars_out"].possible(s)

    key = "shrink"
    spatial = True
    shows_stars = True
    title = "Shrink Stars"
    tip = "Makes stars smaller and less dominant, so the nebula or galaxy stands out."

    def applicable(self, s):
        if not s.stars_out or s.stars is None or s.stars_hidden:
            return False, "Take the stars out first"
        if s.stars_linear:
            return False, "Brighten the stars first"
        return True, ""

    def variants(self, s):
        return [Variant(n, {"a": a, "r": r}, "", i == 1) for i, (n, a, r) in
                enumerate((("Slight", 0.35, 1), ("Moderate", 0.6, 1), ("Strong", 0.8, 2), ("Very strong", 1.0, 2)))]

    def sliders(self, s, params):
        return [Slider("a", "Amount", 0.0, 1.0, params["a"], "How much smaller the stars get"),
                Slider("halo", "Halo reduction", 0.0, 1.0, params.get("halo", 0.6 * params["a"]),
                       "Dims the faint glow around stars")]

    def apply(self, s, params):
        s = s.copy()
        a, r = float(params["a"]), int(params["r"])
        halo = float(params.get("halo", 0.6 * a))
        out = []
        for c in s.stars:
            er = _min_filter(c, r)
            red = (1 - a) * c + a * er
            out.append(np.clip(red ** (1 + halo), 0, 1))     # also soften faint halos
        s.stars = np.stack(out).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
NARROWBAND_KINDS = ("osc_dual", "osc_sho", "mono_nb")
STAR_BLUE, STAR_WHITE, STAR_ORANGE = (0.66, 0.78, 1.0), (1.0, 0.97, 0.92), (1.0, 0.70, 0.42)


def star_colour_map(stars: np.ndarray, spread=1.0):
    """Per-pixel position on a hot-to-cool star colour scale (-1 blue .. 0 white .. +1 orange),
    estimated from how much brighter each star is in the red channel than in green/blue.
    Behind a dual-band filter that compares the star's light at Ha (656 nm) with OIII
    (500 nm) - two points on its spectrum, enough to tell hot blue stars from cool orange ones."""
    st_ = np.clip(stars, 0, None)
    lum = st_.max(axis=0)
    ok = (lum < 0.85).astype(np.float32)                 # clipped cores carry no colour - use the rims
    r = _blur(st_[0] * ok, 1.5)
    gb = _blur(0.5 * (st_[1] + st_[2]) * ok, 1.5)
    ratio = np.log((r + 1e-4) / (gb + 1e-4))
    lm, lmad = _stats(lum)
    thr = max(lm + 6 * lmad, lm + 0.02)                  # clearly part of a star, not background
    sel = (lum > thr) & (ok > 0)
    vals = ratio[sel] if sel.sum() > 30 else ratio.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-3
    x = np.clip((ratio - med) / (2.2 * mad / max(spread, 0.1)), -1, 1)   # scaled to this image's stars
    # one colour per star: spread the estimate from its rim over the whole disc
    wgt = lum * ok + 1e-6
    x = _blur(x * wgt, 2.5) / _blur(wgt, 2.5)
    return np.clip(x, -1, 1).astype(np.float32), lum


def _star_ramp(x):
    b, w, o = (np.asarray(c, np.float32)[:, None, None] for c in (STAR_BLUE, STAR_WHITE, STAR_ORANGE))
    xm, xp = np.clip(-x, 0, 1)[None], np.clip(x, 0, 1)[None]
    return w + (b - w) * xm + (o - w) * xp


class StarColours(Tool):
    def possible(self, s):
        return (s.stars_out or bool(s.channel_stars) or TOOLS["stars_out"].possible(s)) and \
            (s.rgb is None or s.rgb.shape[0] == 3 or bool(s.channels))

    key = "star_color"
    spatial = True
    shows_stars = True
    title = "Star Colours"
    tip = ("Gives stars natural-looking colours - blue-white, white, yellow and orange - instead of the "
           "single tint a narrowband palette paints them. Or makes them clean white.")

    def is_optional(self, s):
        return s.kind not in NARROWBAND_KINDS

    def why(self, s):
        if s.kind in NARROWBAND_KINDS:
            return ("A dual-band or narrowband palette colours every star the same (often teal or blue). "
                    "This estimates each star's real colour from your data and paints it back in.")
        return self.tip

    def applicable(self, s):
        if not s.stars_out or s.stars is None or s.stars_hidden:
            return False, "Take the stars out first"
        if s.stars.shape[0] != 3:
            return False, "Needs a colour image"
        return True, ""

    def variants(self, s):
        nb = s.kind in NARROWBAND_KINDS
        return [Variant("Natural", {"mode": "natural", "sat": 0.85, "spread": 1.0},
                        "Each star's colour estimated from your data", nb),
                Variant("Natural, vivid", {"mode": "natural", "sat": 1.3, "spread": 1.4},
                        "Stronger star colours", False),
                Variant("White", {"mode": "natural", "sat": 0.0, "spread": 1.0},
                        "Clean white stars - the classic narrowband look", False),
                Variant("As they are", {"mode": "keep", "sat": 1.0, "spread": 1.0},
                        "Leave the star colours unchanged", not nb)]

    def sliders(self, s, params):
        if params.get("mode") == "keep":
            return []
        return [Slider("sat", "Colour strength", 0.0, 1.6, params["sat"], "0 = white stars"),
                Slider("spread", "Colour range", 0.4, 2.0, params["spread"],
                       "How different hot and cool stars look from each other")]

    def apply(self, s, params):
        s = s.copy()
        if params.get("mode") != "keep":
            x, lum = star_colour_map(s.stars, float(params.get("spread", 1.0)))
            col = _star_ramp(x)
            sat = float(params.get("sat", 1.0))
            col = 1.0 + (col - 1.0) * sat                        # sat 0 -> white
            col = col / np.max(col, axis=0, keepdims=True)
            s.stars = np.clip(lum[None] * col, 0, None).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class PutStarsBack(Tool):
    def possible(self, s):
        return s.stars_out or bool(s.channel_stars) or TOOLS["stars_out"].possible(s)

    key = "stars_back"
    title = "Put Stars Back"
    tip = "Blends the stars back over the processed nebula or galaxy (screen blend), at the strength you pick."

    def applicable(self, s):
        if not s.stars_out or s.stars is None or s.stars_hidden:
            return False, "Stars are not separated"
        if s.linear != s.stars_linear:
            return False, ("Brighten the stars first" if s.stars_linear else "Brighten the image first")
        return True, ""

    def variants(self, s):
        return [Variant(f"Stars {int(k * 100)}%", {"k": k}, "", k == 1.0) for k in (0.6, 0.8, 1.0, 1.2)]

    def sliders(self, s, params):
        return [Slider("k", "Star strength", 0.3, 1.5, params["k"], "How bright the stars are in the final picture")]

    def apply(self, s, params):
        s = s.copy()
        k = float(params["k"])
        st_ = s.stars if s.stars.shape[0] == s.rgb.shape[0] else np.repeat(s.stars, s.rgb.shape[0], 0)
        if s.linear:
            s.rgb = (s.rgb + k * st_).astype(np.float32)
        else:
            s.rgb = (1 - (1 - s.rgb) * (1 - np.clip(k * st_, 0, 1))).astype(np.float32)
        s.stars, s.stars_out = None, False
        s.done.append(self.key)
        return s



# --------------------------------------------------------------------------
def pchip(xs, ys, x):
    """Monotone cubic through the points (Fritsch-Carlson) - curves never overshoot."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros_like(xs)
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2 / (1 / np.where(d[:-1] == 0, 1e-9, d[:-1]) +
                                                1 / np.where(d[1:] == 0, 1e-9, d[1:])), 0)
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(h) - 1)
    t = (x - xs[i]) / h[i]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]


def _apply_lum(img, fn):
    """Change brightness through the luminance only, so colours (hue) stay the same."""
    y = _lum(img)
    y2 = fn(y)
    ratio = np.where(y > 1e-6, y2 / np.maximum(y, 1e-6), 0).astype(np.float32)
    out = img * ratio[None]
    mx = out.max(0)
    return np.where(mx > 1, out / np.maximum(mx, 1e-6), out).astype(np.float32)


class Curves(Tool):
    key = "curves"
    title = "Adjust Brightness & Contrast (Curves)"
    tip = ("Reshapes the tones: lift faint detail, deepen the background, add punch. Pick a preset, then "
           "fine-tune with the sliders or drag the curve. (PixInsight: CurvesTransformation)")
    editor = "curve"
    PRESETS = (("More contrast", dict(contrast=0.6)), ("Lift faint detail", dict(shadows=0.6, midtones=0.2)),
               ("Darker background", dict(shadows=-0.6)), ("Brighter highlights", dict(highlights=0.5)),
               ("Punchy (S-curve + lift)", dict(contrast=0.5, midtones=0.25)))

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first - curves work on the stretched picture"
        return True, ""

    def possible(self, s):
        return True

    @staticmethod
    def points(contrast=0.0, shadows=0.0, midtones=0.0, highlights=0.0, **_):
        """Five control points from the four plain-language controls."""
        pts = [[0.0, 0.0],
               [0.15, 0.15 + 0.07 * shadows - 0.04 * contrast],
               [0.35, 0.35 + 0.06 * shadows + 0.08 * midtones - 0.05 * contrast],
               [0.65, 0.65 + 0.08 * midtones + 0.05 * contrast + 0.05 * highlights],
               [1.0, 1.0]]
        ys = np.clip([q[1] for q in pts], 0, 1)
        ys = np.maximum.accumulate(ys)                      # never reverse tones
        return [[q[0], float(y)] for q, y in zip(pts, ys)]

    def variants(self, s):
        return [Variant(name, {**{"contrast": 0.0, "shadows": 0.0, "midtones": 0.0, "highlights": 0.0}, **v,
                               "points": self.points(**v)}, "", i == 0) for i, (name, v) in enumerate(self.PRESETS)]

    def sliders(self, s, params):
        return [Slider("contrast", "Contrast", -1, 1, params.get("contrast", 0), "Flatter <-> punchier"),
                Slider("shadows", "Shadows", -1, 1, params.get("shadows", 0), "Darker background <-> lift faint detail"),
                Slider("midtones", "Midtones", -1, 1, params.get("midtones", 0), "Darker <-> brighter main object"),
                Slider("highlights", "Highlights", -1, 1, params.get("highlights", 0), "Softer <-> brighter bright areas")]

    def tuned(self, s, params, values):
        p = dict(params)
        p.update(values)
        if "points" not in values:
            p["points"] = self.points(**{k: p.get(k, 0.0) for k in ("contrast", "shadows", "midtones", "highlights")})
        return p

    def apply(self, s, params):
        s = s.copy()
        pts = np.asarray(params["points"])
        s.rgb = _apply_lum(s.rgb, lambda y: np.clip(pchip(pts[:, 0], pts[:, 1], np.clip(y, 0, 1)), 0, 1))
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class Saturation(Tool):
    key = "saturation"
    title = "Colour Intensity (Saturation)"
    tip = ("Makes colours richer (or calmer). The empty sky is protected so its noise doesn't turn into "
           "coloured speckles.")

    def applicable(self, s):
        if not s.is_color:
            return False, "Needs a colour image"
        if s.linear:
            return False, "Brighten the image first"
        return True, ""

    def possible(self, s):
        return s.is_color or (s.rgb is None and len(s.channels) >= 2)

    def variants(self, s):
        sky = float(np.median(_lum(s.rgb)[::3, ::3]))
        return [Variant(n, {"amount": a, "protect": 0.8, "sky": sky}, "", i == 2) for i, (n, a) in
                enumerate((("Calmer", 0.85), ("Slight", 1.15), ("Rich", 1.35), ("Vivid", 1.6), ("Bold", 1.9)))]

    def sliders(self, s, params):
        return [Slider("amount", "Colour intensity", 0.5, 2.5, params["amount"], "Grey <-> vivid", 0.05),
                Slider("protect", "Protect the sky", 0.0, 1.0, params["protect"],
                       "Keeps the dark background from getting colourful noise")]

    def apply(self, s, params):
        s = s.copy()
        f, pr, sky = float(params["amount"]), float(params["protect"]), float(params["sky"])
        y = _lum(s.rgb)
        w = np.clip((y - sky * 1.15) / 0.25, 0, 1)
        eff = 1 + (f - 1) * ((1 - pr) + pr * w)
        s.rgb = np.clip(y[None] + (s.rgb - y[None]) * eff[None], 0, 1).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
def clahe(y: np.ndarray, grid=8, clip=2.5, bins=256) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation of a 0..1 plane
    (PixInsight: LocalHistogramEqualization)."""
    h, w = y.shape
    gy, gx = max(2, int(round(grid * h / max(h, w)))), max(2, int(round(grid * w / max(h, w))))
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    q = np.clip((y * (bins - 1)).astype(np.int32), 0, bins - 1)
    lut = np.empty((gy, gx, bins), np.float32)
    for i in range(gy):
        for j in range(gx):
            tile = q[ys[i]:ys[i + 1]:2, xs[j]:xs[j + 1]:2]
            hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float64)
            lim = clip * hist.sum() / bins
            excess = np.clip(hist - lim, 0, None).sum()
            hist = np.minimum(hist, lim) + excess / bins
            cdf = np.cumsum(hist)
            lut[i, j] = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-9)
    cy = (ys[:-1] + ys[1:]) / 2
    cx = (xs[:-1] + xs[1:]) / 2
    fy = np.clip(np.interp(np.arange(h), cy, np.arange(gy)), 0, gy - 1)
    fx = np.clip(np.interp(np.arange(w), cx, np.arange(gx)), 0, gx - 1)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1, x1 = np.minimum(y0 + 1, gy - 1), np.minimum(x0 + 1, gx - 1)
    wy = (fy - y0).astype(np.float32)[:, None]
    wx = (fx - x0).astype(np.float32)[None, :]
    out = np.empty((h, w), np.float32)
    for r0 in range(0, h, 512):                       # in strips, to keep memory modest
        r1 = min(h, r0 + 512)
        qq = q[r0:r1]
        a = lut[y0[r0:r1, None], x0[None, :], qq]
        b = lut[y0[r0:r1, None], x1[None, :], qq]
        c = lut[y1[r0:r1, None], x0[None, :], qq]
        d = lut[y1[r0:r1, None], x1[None, :], qq]
        wyy = wy[r0:r1]
        out[r0:r1] = (a * (1 - wx) + b * wx) * (1 - wyy) + (c * (1 - wx) + d * wx) * wyy
    return out


class LocalContrast(Tool):
    key = "local"
    title = "Bring Out Hidden Structure"
    tip = ("Boosts contrast locally, so detail inside bright cores and faint dust lanes shows up. "
           "(PixInsight: LocalHistogramEqualization)")
    optional = True

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first"
        return True, ""

    def variants(self, s):
        return [Variant(n, {"amount": a, "grid": 8, "clip": 2.5}, "", i == 1) for i, (n, a) in
                enumerate((("Subtle", 0.15), ("Moderate", 0.3), ("Strong", 0.45), ("Very strong", 0.65)))]

    def sliders(self, s, params):
        return [Slider("amount", "Strength", 0.0, 1.0, params["amount"], "How much local contrast to add"),
                Slider("grid", "Detail size", 4, 16, params["grid"], "Large structures (4) <-> fine detail (16)", 1)]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        eq = None

        def fn(y):
            nonlocal eq
            eq = clahe(np.clip(y, 0, 1), int(round(params["grid"])), float(params["clip"]))
            return (1 - a) * y + a * eq
        s.rgb = _apply_lum(s.rgb, fn)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
def _flux(img, pts, r_in=3.5, r_out=(6, 8)):
    h, w = img.shape
    yy, xx = np.mgrid[-8:9, -8:9]
    rr = np.hypot(yy, xx)
    inner, ring = rr <= r_in, (rr >= r_out[0]) & (rr <= r_out[1])
    out = np.full(len(pts), np.nan)
    for k, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if 8 <= xi < w - 8 and 8 <= yi < h - 8:
            box = img[yi - 8:yi + 9, xi - 8:xi + 9]
            out[k] = float((box[inner] - np.median(box[ring])).sum())
    return out


class HaGlow(Tool):
    key = "ha_glow"
    title = "Add Hydrogen Glow"
    tip = ("Blends your Ha image into the red channel so nebulae glow in a natural-colour picture. The "
           "starlight is removed from the Ha first (continuum subtraction), so stars don't turn pink.")

    def applicable(self, s):
        if s.rgb is not None or not {"Ha", "R", "G", "B"} <= set(s.channels):
            return False, "Needs separate R, G, B and Ha images, before combining"
        return True, ""

    def possible(self, s):
        return s.rgb is None and {"Ha", "R", "G", "B"} <= set(s.channels)

    def variants(self, s):
        import stacking as st
        ha, r = s.channels["Ha"], s.channels["R"]
        bg_h, bg_r = _stats(ha)[0], _stats(r)[0]
        c = None
        try:
            pts, _ = st.find_stars((r * 65535.0).astype(np.float32), 1, max_stars=200)
            fh, fr = _flux(ha, pts), _flux(r, pts)
            ok = np.isfinite(fh) & np.isfinite(fr) & (fr > 0)
            if ok.sum() >= 8:
                c = float(np.median(fh[ok] / fr[ok]))
        except Exception:  # noqa: BLE001
            c = None
        if not c or c <= 0:
            c = float(np.percentile(ha[::3, ::3], 99) - bg_h) / max(float(np.percentile(r[::3, ::3], 99) - bg_r), 1e-6)
        base = {"c": c, "bg_h": bg_h, "bg_r": bg_r}
        return [Variant(n, {**base, "k": k}, f"starlight factor {c:.2f} removed from Ha", i == 1) for i, (n, k) in
                enumerate((("Subtle", 0.5), ("Natural", 1.0), ("Strong", 1.6), ("Very strong", 2.5)))]

    def sliders(self, s, params):
        return [Slider("k", "Glow strength", 0.0, 4.0, params["k"], "How much hydrogen glow to add to red", 0.05)]

    def preview(self, s, params):
        ch = s.channels
        base = np.stack([ch["R"], ch["G"], ch["B"]])
        r = self.apply(s, params).channels
        return display_fixed(np.stack([r["R"], r["G"], r["B"]]), base)

    def apply(self, s, params):
        s = s.copy()
        ch = dict(s.channels)
        c, k = float(params["c"]), float(params["k"])
        line = np.clip((ch["Ha"] - params["bg_h"]) - c * (ch["R"] - params["bg_r"]), 0, None)  # Ha without starlight
        ch["R"] = (ch["R"] + k * line / max(c, 1e-6)).astype(np.float32)
        ch.pop("Ha")
        s.channels = ch
        if "Ha" in s.channel_stars:
            cs = dict(s.channel_stars)
            cs.pop("Ha")
            s.channel_stars = cs
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class SpotFix(Tool):
    key = "spotfix"
    title = "Spot Fix"
    tip = ("Click on a dust donut, a satellite remnant or a hot pixel and it is patched from its "
           "surroundings. (PixInsight: CloneStamp)")
    pick = False
    optional = True

    def applicable(self, s):
        return (True, "") if s.rgb is not None else (False, "Combine the colour channels first")

    def variants(self, s):
        return [Variant("Fix the marked spots", {"spots": []}, "Click spots on the image (right-click removes "
                        "the last one); the mouse wheel changes the circle size", True)]

    @staticmethod
    def heal(img: np.ndarray, x, y, r, seed=0) -> np.ndarray:
        c, h, w = img.shape
        R = int(r * 1.6) + 3
        x0, x1, y0, y1 = max(0, int(x) - R), min(w, int(x) + R + 1), max(0, int(y) - R), min(h, int(y) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - y, xx - x)
        mask = d <= r
        ring = (d > r) & (d <= r * 1.6 + 2)
        if not mask.any() or not ring.any():
            return img
        out = img.copy()
        rng = np.random.default_rng(seed)
        for k in range(c):
            patch = out[k, y0:y1, x0:x1].astype(np.float64)
            fill = patch.copy()
            fill[mask] = np.median(patch[ring])
            for _ in range(80):                 # smooth (harmonic) fill from the edge inwards
                avg = (np.roll(fill, 1, 0) + np.roll(fill, -1, 0) + np.roll(fill, 1, 1) + np.roll(fill, -1, 1)) / 4
                fill[mask] = avg[mask]
            sd = float(np.std(patch[ring] - np.median(patch[ring])))
            fill[mask] += rng.normal(0, sd * 0.8, int(mask.sum()))
            out[k, y0:y1, x0:x1] = fill.astype(out.dtype)
        return out

    def apply(self, s, params):
        s = s.copy()
        img = s.rgb
        for i, (x, y, r) in enumerate(params.get("spots", [])):
            img = self.heal(img, x, y, r, seed=i)
        s.rgb = img.astype(np.float32)
        if "spotfix" not in s.done:
            s.done.append(self.key)
        return s



# --------------------------------------------------------------------------
def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Box blur with radius r (cumulative sums - fast at any size)."""
    if r < 1:
        return a
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (out / (k * k)).astype(np.float32)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-like blur (three box passes)."""
    r = max(1, int(round(sigma * 0.95)))
    for _ in range(3):
        a = _box_blur(a, r)
    return a


def _resize(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of a plane."""
    ih, iw = a.shape
    ys = np.clip((np.arange(h) + 0.5) * ih / h - 0.5, 0, ih - 1)
    xs = np.clip((np.arange(w) + 0.5) * iw / w - 0.5, 0, iw - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
    wy, wx = (ys - y0)[:, None].astype(np.float32), (xs - x0)[None, :].astype(np.float32)
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _small_plane(a: np.ndarray, side=900):
    f = max(1, int(np.ceil(max(a.shape) / side)))
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)), f


def _bright_blob(y: np.ndarray, level: float) -> np.ndarray:
    """Areas brighter than level that are bigger than a star (opening removes stars)."""
    sm, _f = _small_plane(y, 600)
    m = (sm > level).astype(np.float32)
    opened = -_min_filter(-_min_filter(m, 2), 2)          # erode then dilate
    return opened


def _core_saturated(lin_rgb: np.ndarray) -> bool:
    """Was the brightest extended area clipped in the masters (the camera maxed out)?"""
    mx = lin_rgb.max(axis=0)
    blob = _bright_blob(mx, 0.97)
    return bool(blob.sum() >= 12)


def hot_core(s: "State") -> bool:
    """A white, featureless area bigger than a star after brightening."""
    if s.rgb is None or s.linear:
        return False
    y = _lum(s.rgb)
    return bool(_bright_blob(y, 0.90).sum() >= 12)


class TameCore(Tool):
    key = "core"
    title = "Tame Bright Core"
    tip = ("Brings back detail in a burnt-out galaxy core or bright nebula centre. SubZeroAstro finds the "
           "bright area itself and makes a soft-edged mask, so the rest of the image is not touched. "
           "(PixInsight: a luminance mask + HDRMultiscaleTransform)")
    optional = True

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first"
        return True, ""

    def is_optional(self, s):
        return not hot_core(s)            # recommended only when a burnt core is detected

    def why(self, s):
        if s.sat_core:
            return ("The core looks burnt out - and it was already saturated in the masters, so its detail "
                    "can't be recovered; this makes it look smooth and natural instead of a flat white disc.")
        return "The core looks burnt out (pure white) - bring its detail back without touching the rest."

    def variants(self, s):
        note = "core saturated in the masters: smooths rather than recovers" if s.sat_core else ""
        return [Variant(n, {"amount": a, "size": 0.78, "detail": 30.0}, note, i == 1) for i, (n, a) in
                enumerate((("Gentle", 0.45), ("Medium", 0.7), ("Strong", 0.9), ("Maximum", 1.0)))]

    def sliders(self, s, params):
        return [Slider("amount", "Strength", 0.0, 1.0, params["amount"], "How much the core is toned down"),
                Slider("size", "Core size", 0.5, 0.95, params["size"],
                       "Bigger area (left) <-> only the very brightest part (right)"),
                Slider("detail", "Detail", 3.0, 150.0, params["detail"],
                       "More = more structure brought out inside the core", 1.0)]

    def mask(self, s, params) -> np.ndarray:
        """Soft 0..1 mask of the bright core (full size)."""
        y = _lum(s.rgb)
        small, f = _small_plane(y, 700)
        h, w = small.shape
        sig = max(2.0, max(h, w) / 140)
        b = _blur(small, sig)
        top = float(b.max())                                      # the core's (blurred) brightness
        sky = float(np.median(b))
        t1 = sky + (top - sky) * float(params["size"])           # size is relative to this image
        t0 = sky + (t1 - sky) * 0.72
        m = np.clip((b - t0) / max(t1 - t0, 1e-3), 0, 1)
        m = m * m * (3 - 2 * m)                                  # smoothstep
        m = _blur(m, sig)                                        # feather the edge
        return np.clip(_resize(m, *y.shape), 0, 1)

    def _target(self, s, params, m):
        """Core luminance re-made from the linear data (or by compressing the stretched one)."""
        y = _lum(s.rgb)
        c = float(params["detail"])
        inside = m > 0.5
        edge = (m > 0.25) & (m < 0.75)
        if not inside.any():
            return y
        if s.lin_lum is not None and s.lin_lum.shape == y.shape:
            n = s.lin_lum
            n_e = float(np.median(n[edge])) if edge.any() else float(np.percentile(n[inside], 5))
            n_max = float(np.percentile(n[inside], 99.9))
            e = float(np.median(y[edge])) if edge.any() else float(np.percentile(y[inside], 5))
            top = 0.93
            u = np.clip((n - n_e) / max(n_max - n_e, 1e-9), 0, None)
            rec = e + (top - e) * np.log1p(c * u) / np.log1p(c)
            return np.where(n > n_e, np.minimum(rec, 1.0), y).astype(np.float32)
        # no linear data kept: compress the large-scale brightness, keep the fine detail
        base = _blur(y, 6)
        a = 0.6
        return np.clip(base * (1 - a * base ** 2) + (y - base) * 1.6, 0, 1).astype(np.float32)

    def apply(self, s, params):
        s = s.copy()
        m = self.mask(s, params)
        a = float(params["amount"])
        target = self._target(s, params, m)
        w = (a * m).astype(np.float32)
        s.rgb = _apply_lum(s.rgb, lambda y: y * (1 - w) + target * w)
        if self.key not in s.done:
            s.done.append(self.key)
        return s

    def preview(self, s, params):
        img = state_display(self.apply(s, params))
        if params.get("show_mask"):
            m = self.mask(s, params)[..., None]
            tint = np.array([255, 70, 40], np.float32)
            img = (img * (1 - 0.45 * m) + tint * 0.45 * m).astype(np.uint8)
        return img

TOOLS = {t.key: t for t in (HaGlow(), CombineChannels(), DualBandPalette(), BalanceColours(), RemoveGreen(), TakeStarsOut(), Brighten(),
                            TameCore(), Curves(), Saturation(), LocalContrast(), BrightenStars(), ShrinkStars(),
                            StarColours(), PutStarsBack(),
                            SpotFix())}


def thumb_state(tool: Tool, s: State, size=420) -> State:
    """The small image a tool's choices are rendered on."""
    if tool.spatial:
        return s.crop(size, busiest_spot(s, size))
    return s.small(size)


def busiest_spot(s: State, size=420):
    """Centre of the area with the most stars / detail (for 1:1 crops)."""
    img = s.stars if s.stars is not None else s.rgb
    if img is None:
        return None
    lum = _lum(img)
    h, w = lum.shape
    if h <= size or w <= size:
        return None
    b = max(8, size // 4)
    hh, ww = h // b * b, w // b * b
    blocks = lum[:hh, :ww].reshape(hh // b, b, ww // b, b)
    score = blocks.std(axis=(1, 3))
    iy, ix = np.unravel_index(int(np.argmax(score[1:-1, 1:-1])), score[1:-1, 1:-1].shape)
    return (iy + 1) * b + b // 2, (ix + 1) * b + b // 2


# ==========================================================================
# What's next
# ==========================================================================

_POLISH = ["core", "curves", "saturation", "local"]
_STARS = ["star_stretch", "shrink", "star_color", "stars_back"]
WORKFLOWS = {
    "osc": ["balance", "scnr", "stars_out", "stretch"] + _POLISH + _STARS + ["spotfix"],
    "osc_dual": ["palette", "stars_out", "stretch", "scnr"] + _POLISH + _STARS + ["spotfix"],
    "osc_sho": ["combine", "stars_out", "stretch", "scnr"] + _POLISH + _STARS + ["spotfix"],
    "smart": ["balance", "scnr", "stars_out", "stretch"] + _POLISH + _STARS + ["spotfix"],
    "mono_rgb": ["ha_glow", "combine", "balance", "scnr", "stars_out", "stretch"] + _POLISH + _STARS + ["spotfix"],
    "mono_lrgb": ["ha_glow", "combine", "balance", "scnr", "stars_out", "stretch"] + _POLISH + _STARS + ["spotfix"],
    "mono_nb": ["combine", "stars_out", "stretch", "scnr"] + _POLISH + _STARS + ["spotfix"],
    "mono": ["stars_out", "stretch", "core", "curves", "local"] + _STARS + ["spotfix"],
}

WHY = {
    "combine": "Your filters are still separate images - combine them into one colour picture first.",
    "balance": "Colour balance works best on the unstretched data, before anything else changes the colours.",
    "stars_out": "Separating the stars now lets you brighten the nebula strongly without bloating the stars.",
    "stretch": "Time to turn the linear data into a visible picture.",
    "star_stretch": "The stars-only image needs its own gentle brighten so stars stay small and colourful.",
    "shrink": "Optional: smaller stars make the main object stand out.",
    "stars_back": "Last step: put the stars back over the processed image.",
    "curves": "Now shape the tones: a little contrast or lifting the faint parts makes a big difference.",
    "saturation": "Bring the colours out - the sky is protected from colour noise.",
    "ha_glow": "You have an Ha image with your RGB - add its glow to the red channel before combining.",
}


def workflow(s: State) -> list:
    """[(tool, status)] in the recommended order; status: done / skipped / next / later / n/a."""
    steps = WORKFLOWS.get(s.kind, WORKFLOWS["osc"])
    out, found_next = [], False
    for key in steps:
        t = TOOLS[key]
        if key in s.done:
            out.append((t, "done"))
            continue
        if key in s.skipped:
            out.append((t, "skipped"))
            continue
        ok, _why = t.applicable(s)
        if not ok:
            out.append((t, "later" if t.possible(s) else "n/a"))
            continue
        if t.is_optional(s):
            out.append((t, "optional"))
            continue
        out.append((t, "later" if found_next else "next"))
        found_next = True
    return out


def next_step(s: State):
    for t, status in workflow(s):
        if status == "next":
            return t
    return None


def why_next(s: State, t: Tool) -> str:
    base = t.why(s) if type(t).why is not Tool.why else WHY.get(t.key, t.tip)
    return base


# ==========================================================================
# Project: every step saved (resume, go back, save / open)
# ==========================================================================

class Project:
    """A folder with manifest.json and one sub-folder per finished step."""

    def __init__(self, folder: Path):
        self.dir = Path(folder)
        man = cp.read_json(self.dir / "manifest.json", None) or {}
        good = [x for x in man.get("steps", []) if (self.dir / x.get("dir", "?") / "done").exists()]
        self.manifest = {"app": "SubZeroAstro", "kind": "finish-project", "sources": man.get("sources", []),
                         "target": man.get("target", ""), "steps": good}

    @classmethod
    def for_sources(cls, sources, target):
        return cls(cp.root() / f"finish_{cp.safe_name(target)}_{cp.file_key(sources)}")

    @property
    def steps(self):
        return self.manifest["steps"]

    def _write(self):
        self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cp.write_json(self.dir / "manifest.json", self.manifest)

    def save(self, s: State, tool: str, label: str, params=None):
        idx = len(self.steps)
        sub = f"{idx:02d}_{tool}"
        d = self.dir / sub
        cp.remove(d)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        dt = np.float32 if s.linear else np.float16           # stretched data keeps fine in 16-bit float
        if s.rgb is not None:
            arrays["rgb"] = s.rgb.astype(dt)
        if s.stars is not None:
            arrays["stars"] = s.stars.astype(np.float32 if s.stars_linear else np.float16)
        if s.lin_lum is not None:
            arrays["lin_lum"] = s.lin_lum.astype(np.float32)
        for k, v in s.channels.items():
            arrays["ch_" + k] = v
        for k, v in s.channel_stars.items():
            arrays["cs_" + k] = v
        np.savez(d / "state.npz", **arrays)
        meta = {"linear": s.linear, "stars_linear": s.stars_linear, "stars_out": s.stars_out, "done": s.done,
                "skipped": s.skipped, "kind": s.kind, "target": s.target, "sources": s.sources,
                "palette": s.palette, "sat_core": s.sat_core, "filter": s.filter,
                "stars_hidden": s.stars_hidden}
        cp.write_json(d / "meta.json", meta)
        thumb = state_display(s.small(700))
        np.save(d / "preview.npy", thumb, allow_pickle=False)
        (d / "done").write_text("ok", encoding="utf-8")
        clean = {k: v for k, v in (params or {}).items() if k not in ("stats", "bgs")}
        self.steps.append({"tool": tool, "label": label, "dir": sub, "time": time.strftime("%Y-%m-%d %H:%M"),
                           "params": json.loads(json.dumps(clean, default=str))})
        if not self.manifest["sources"]:
            self.manifest["sources"], self.manifest["target"] = s.sources, s.target
        self._write()

    def load(self, n: int) -> State:
        """State after step n (index into steps)."""
        d = self.dir / self.steps[n]["dir"]
        meta = cp.read_json(d / "meta.json", {})
        z = np.load(d / "state.npz")
        s = State(rgb=z["rgb"].astype(np.float32) if "rgb" in z else None,
                  stars=z["stars"].astype(np.float32) if "stars" in z else None,
                  channels={k[3:]: z[k] for k in z.files if k.startswith("ch_")},
                  channel_stars={k[3:]: z[k] for k in z.files if k.startswith("cs_")},
                  linear=meta.get("linear", True), stars_linear=meta.get("stars_linear", True),
                  stars_out=meta.get("stars_out", False), done=meta.get("done", []),
                  skipped=meta.get("skipped", []), kind=meta.get("kind", "osc"), target=meta.get("target", ""),
                  sources=meta.get("sources", []), palette=meta.get("palette"),
                  lin_lum=z["lin_lum"].astype(np.float32) if "lin_lum" in z else None,
                  sat_core=bool(meta.get("sat_core", False)), filter=meta.get("filter"),
                  stars_hidden=bool(meta.get("stars_hidden", False)))
        return s

    def preview(self, n):
        try:
            return np.load(self.dir / self.steps[n]["dir"] / "preview.npy", allow_pickle=False)
        except (OSError, IndexError, ValueError):
            return None

    def truncate(self, n):
        """Keep steps [0, n)."""
        for x in self.steps[n:]:
            cp.remove(self.dir / x["dir"])
        del self.steps[n:]
        if self.steps:
            self._write()

    def save_as(self, folder: Path) -> "Project":
        folder = Path(folder)
        if folder.exists() and any(folder.iterdir()):
            folder = folder / f"{cp.safe_name(self.manifest.get('target') or 'project')}.szproj"
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(self.dir, folder)
        return Project(folder)

    def size(self):
        return cp._size(self.dir) if self.dir.exists() else 0


# ==========================================================================
# Export
# ==========================================================================

def export(s: State, base: Path, formats=("tif", "jpg"), save_rgb8=None) -> list:
    """Write the current image. Stars still separated -> also a stars file and a
    combined preview."""
    import fits_core as core
    import prepare as P
    import xisf
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    img = s.rgb
    if s.stars_out and s.stars is not None and s.linear == s.stars_linear:
        st_ = s.stars if s.stars.shape[0] == img.shape[0] else np.repeat(s.stars, img.shape[0], 0)
        img = img + st_ if s.linear else 1 - (1 - img) * (1 - np.clip(st_, 0, 1))
    img = np.clip(img, 0, 1).astype(np.float32)
    files = []
    hdr = [("OBJECT", s.target, ""), ("SWCREATE", "SubZeroAstro", ""),
           ("HISTORY", "SubZeroAstro finish: " + ", ".join(TOOLS[k].title for k in s.done if k in TOOLS), "")]
    data = img[0] if img.shape[0] == 1 else img
    for f in formats:
        p = base.with_name(base.name + "." + f)
        if f == "tif":
            rgb = img if img.shape[0] == 3 else np.repeat(img, 3, 0)
            P.write_tiff16(p, rgb.transpose(1, 2, 0))
        elif f in ("jpg", "png"):
            if save_rgb8 is None:
                continue
            rgb = img if img.shape[0] == 3 else np.repeat(img, 3, 0)
            if not save_rgb8(p, (rgb.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)):
                continue
        elif f == "fits":
            core.write_fits(p, data, hdr, bitpix=-32)
        elif f == "xisf":
            xisf.write_xisf(p, data, [h for h in hdr if h[0] != "HISTORY"], "Light", bounds=(0.0, 1.0))
        files.append(p)
    return files


# ==========================================================================
# Stop here - continue in PixInsight
# ==========================================================================

PI_EQUIVALENT = {
    "combine": "ChannelCombination (RGB / SHO / HOO) or LRGBCombination - or PixelMath for custom palettes",
    "palette": "Split channels (Ha = R, OIII = (G+B)/2), then PixelMath / ChannelCombination for HOO, Foraxx ...",
    "balance": "SpectrophotometricColorCalibration (SPCC) - plate solve first (ImageSolver); "
               "BackgroundNeutralization if no SPCC",
    "scnr": "SCNR (Average Neutral, green)",
    "stars_out": "StarXTerminator (tick 'Generate star image')",
    "stretch": "GeneralizedHyperbolicStretch (GHS) or HistogramTransformation (use STF as a starting point)",
    "star_stretch": "Star Stretch (Bill Blanshan's PixelMath / script) or a gentle HistogramTransformation",
    "shrink": "Star Reduction (Bill Blanshan's script) or MorphologicalTransformation (Erosion)",
    "curves": "CurvesTransformation",
    "saturation": "CurvesTransformation (saturation curve) or ColorSaturation",
    "local": "LocalHistogramEqualization",
    "ha_glow": "Continuum subtraction (PixelMath: Ha - k*R) then NBRGBCombination or PixelMath into R",
    "spotfix": "CloneStamp",
    "core": "HDRMultiscaleTransform through a luminance (range) mask on the core",
    "star_color": "PixelMath on the stars image (colour from the R/(G+B) ratio), or Narrowband Normalization's star options",
        "stars_back": "PixelMath screen blend:  ~((~starless)*(~stars))",
}


def export_for_pixinsight(s: State, folder: Path) -> list:
    """Save everything needed to carry on in PixInsight from exactly this point,
    plus a short note on what to do next. Returns the files written (note first)."""
    import xisf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = "linear" if s.linear else "stretched"
    base = f"{cp.safe_name(s.target or 'image')}_{stage}"
    hdr = [("OBJECT", s.target, ""), ("SWCREATE", "SubZeroAstro", "")]
    files = []

    def put(name, data, what):
        p = folder / f"{name}.xisf"
        d = np.clip(np.asarray(data, np.float32), 0, 1)
        xisf.write_xisf(p, d[0] if d.ndim == 3 and d.shape[0] == 1 else d,
                        hdr + [("COMMENT", what, "")], "Light", bounds=(0.0, 1.0))
        files.append(p)
        return p.name

    names = []
    if s.rgb is None:                                    # mono filters not combined yet
        for k, v in s.channels.items():
            names.append(put(f"{base}_{k}", v, f"{k} filter, {stage}"))
        for k, v in s.channel_stars.items():
            names.append(put(f"{base}_{k}_stars", v, f"{k} stars only"))
    else:
        main = f"{base}_starless" if s.stars_out else base
        names.append(put(main, s.rgb, ("starless, " if s.stars_out else "") + stage))
        if s.stars_out and s.stars is not None:
            names.append(put(f"{base}_stars", s.stars, "stars only, " + ("linear" if s.stars_linear else "stretched")))

    sxt = TOOLS["stars_out"]
    had = sxt.runner
    sxt.runner = had or (lambda *a: None)          # PixInsight has StarXTerminator even if we don't
    try:
        remaining = [t for t, st_ in workflow(s) if st_ in ("next", "later")]
    finally:
        sxt.runner = had
    done = [TOOLS[k].title for k in s.done if k in TOOLS]
    lines = [f"{s.target} - continue in PixInsight", "=" * 40, "",
             f"Data: {KIND_LABEL.get(s.kind, s.kind)}, currently {stage.upper()}"
             + (" (still linear - use an STF auto-stretch to view it)" if s.linear else ""), "",
             "Files:"] + [f"  {n}" for n in names] + ["",
             "Done in SubZeroAstro: " + (", ".join(done) if done else "nothing yet"), "",
             "Suggested next steps in PixInsight:"]
    if remaining:
        for i, t in enumerate(remaining, 1):
            lines.append(f"  {i}. {t.title}  ->  {PI_EQUIVALENT.get(t.key, '')}")
    else:
        lines.append("  The image is finished. Optional polish: CurvesTransformation (contrast, saturation), "
                     "LocalHistogramEqualization, a final NoiseXTerminator pass.")
    if s.linear:
        lines += ["", "Tip: everything here is linear - keep it that way until the stretch step."]
    note = folder / f"{cp.safe_name(s.target or 'image')}_continue_in_PixInsight.txt"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [note] + files
