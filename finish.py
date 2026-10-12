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


KIND_LABEL = {"osc": "colour camera", "smart": "smart telescope", "mono_rgb": "mono RGB",
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

    if color:
        m = color[0]
        s.rgb = np.clip(m.data / 65535.0, 0, None).astype(np.float32)
        s.kind = "smart" if smart else "osc"
        st_img = stars_of(m)
        if st_img is not None and st_img.shape == s.rgb.shape:
            s.stars, s.stars_out = st_img, True
            s.done.append("stars_out")
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
            s.stars, s.stars_out = s.channel_stars[k][None].copy(), True
            s.done.append("stars_out")
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


def state_display(s: State, linked=True) -> np.ndarray:
    """What the user should see for a state: the full picture (stars added back for
    viewing if they are separated)."""
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
    img = s.rgb
    if s.stars_out and s.stars is not None and s.linear == s.stars_linear:
        st_ = s.stars if s.stars.shape[0] == img.shape[0] else np.repeat(s.stars, img.shape[0], 0)
        img = img + st_ if s.linear else 1 - (1 - img) * (1 - np.clip(st_, 0, 1))
    return display_rgb8(img, s.linear, linked)


# ==========================================================================
# Tools
# ==========================================================================

@dataclass
class Variant:
    label: str
    params: dict
    note: str = ""
    recommended: bool = False


class Tool:
    key = ""
    title = ""
    tip = ""
    pick = True                    # show thumbnails to choose from
    fine = None                    # (param name, label, min, max, step) for "fine-tune"
    spatial = False                # effect depends on pixel scale -> thumbnails are 1:1 crops

    def applicable(self, s: State):
        return True, ""

    def possible(self, s: State) -> bool:
        """Can this step happen at some point (maybe after earlier steps)?"""
        return True

    def variants(self, s: State) -> list:
        return [Variant("Apply", {})]

    def apply(self, s: State, params: dict) -> State:
        raise NotImplementedError

    def why(self, s: State) -> str:
        return self.tip


# --------------------------------------------------------------------------
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
            out += [Variant("Hubble (SHO)", {"palette": "SHO"}, "SII red, Ha green, OIII blue - the classic "
                            "Hubble look (green is tamed later)", True),
                    Variant("HSO", {"palette": "HSO"}, "Ha red, SII green, OIII blue"),
                    Variant("HOO + SII", {"palette": "HOO+S"}, "Natural-looking reds and teals")]
        if {"Ha", "OIII"} <= f:
            out += [Variant("HOO", {"palette": "HOO"}, "Ha red, OIII teal - natural bicolour",
                            not ({"SII"} <= f)),
                    Variant("HOO (vivid)", {"palette": "HOO+"}, "Bicolour with a synthetic green for "
                            "warmer tones")]
        if {"Ha"} <= f and {"R", "G", "B"} <= f:
            out.append(Variant("RGB + Ha glow", {"palette": "RGB+Ha"}, "True colour with the hydrogen glow "
                               "added to red"))
        # pre-measure each channel so thumbnails match the full result
        stats = {k: _stats(v) + (float(np.percentile(v[::3, ::3], 99.99)),) for k, v in s.channels.items()}
        for v in out:
            v.params["stats"] = stats
        return out

    @staticmethod
    def _norm(x, st_):
        """Each filter scaled so its sky sits at 0.01 and its brightest real signal near 0.9
        (a 'balanced' mix: weak filters like SII are lifted to match)."""
        med, mad, hi = st_
        return np.clip((x - med) / max(hi - med, 1e-6) * 0.89 + 0.01, 0, 1)

    def apply(self, s, params):
        s = s.copy()
        pal = params["palette"]
        stats = params.get("stats") or {k: _stats(v) + (float(np.percentile(v[::3, ::3], 99.99)),)
                                         for k, v in s.channels.items()}

        def build(ch):
            n = {k: self._norm(v, stats[k]) for k, v in ch.items()}
            if pal in ("RGB", "LRGB", "RGB+Ha"):
                r, g, b = n["R"], n["G"], n["B"]
                if pal == "RGB+Ha" and "Ha" in n:
                    r = np.maximum(r, 0.6 * r + 0.6 * n["Ha"])
                img = np.stack([r, g, b])
                if pal == "LRGB" and "L" in n:
                    y = img.mean(0) + 1e-6
                    img = img * (np.clip(n["L"], 0, None) / np.clip(y, 1e-6, None))[None]
                return img
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
            return np.stack([h, o, o])                       # HOO

        s.rgb = np.clip(build(s.channels), 0, None).astype(np.float32)
        if s.channel_stars and set(s.channel_stars) >= set(s.channels):
            s.stars = np.clip(build(s.channel_stars) - 0.01, 0, None).astype(np.float32)
            s.stars_out = True
            s.stars_linear = True
            if "stars_out" not in s.done:
                s.done.append("stars_out")
        s.channels, s.channel_stars = {}, {}
        s.palette = pal
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

    def apply(self, s, params):
        s = s.copy()
        bgs = params["bgs"]
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
        best = 1.0 if s.kind == "mono_nb" else (0.75 if excess > 0.5 else 0.5)
        return [Variant(f"{int(a * 100)}%", {"amount": a}, "", a == best) for a in (0.25, 0.5, 0.75, 1.0)]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        r, g, b = s.rgb
        neutral = np.minimum(g, (r + b) / 2)
        s.rgb = np.stack([r, g * (1 - a) + neutral * a, b]).astype(np.float32)
        s.done.append(self.key)
        return s

    def why(self, s):
        if s.kind == "mono_nb":
            return "The Hubble palette is very green straight out of combination - this gives the classic gold and blue."
        return "Colour cameras usually leave a slight green tint after colour balancing."


# --------------------------------------------------------------------------
class TakeStarsOut(Tool):
    key = "stars_out"

    def possible(self, s):
        return not s.stars_out and self.runner is not None
    title = "Take Stars Out"
    tip = ("Splits the stars from the nebula or galaxy (StarXTerminator) so each can be brightened "
           "separately - you get bright detail without bloated stars. They go back in at the end.")
    pick = False
    runner = None                  # set by the window: callable(img (C,h,w) 0..1) -> (starless, stars)

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.stars_out:
            return False, "Stars are already separated"
        if self.runner is None:
            return False, "Needs StarXTerminator (set the RC-Astro CLI in the Prepare window)"
        return True, ""

    def variants(self, s):
        return [Variant("Separate the stars", {}, "Runs StarXTerminator on the full image", True)]

    def apply(self, s, params):
        s = s.copy()
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
            prm = {"method": method, "linked": linked, "stats": st_, "target": t, "shifts": shifts}
            if method == "ghs":
                prm["lnD"] = self._solve(lambda x: float(ghs(np.array([nm]), np.exp(x) - 1, 6.0, nm / 2)[0]),
                                         t, 0.0, 14.0)
                prm["b"] = 6.0
            elif method == "colour":
                prm["beta"] = float(np.exp(self._solve(
                    lambda x: float(np.arcsinh(np.exp(x) * nm) / np.arcsinh(np.exp(x))), t, 0.0, 14.0)))
            out.append(Variant(name, prm, f"sky at {int(t * 100)}% grey", i == 2))
        return out

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
        s.rgb = self._stretch_planes(s.rgb, params)
        s.linear = False
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class BrightenStars(Tool):
    def possible(self, s):
        return s.stars_out or bool(s.channel_stars) or TOOLS["stars_out"].possible(s)

    key = "star_stretch"
    title = "Brighten Stars Separately"
    tip = ("Brightens the stars-only image gently, so stars stay small and colourful instead of "
           "turning into big white blobs.")

    def applicable(self, s):
        if not s.stars_out or s.stars is None:
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
    title = "Shrink Stars"
    tip = "Makes stars smaller and less dominant, so the nebula or galaxy stands out."

    def applicable(self, s):
        if not s.stars_out or s.stars is None:
            return False, "Take the stars out first"
        if s.stars_linear:
            return False, "Brighten the stars first"
        return True, ""

    def variants(self, s):
        return [Variant(n, {"a": a, "r": r}, "", i == 1) for i, (n, a, r) in
                enumerate((("Slight", 0.35, 1), ("Moderate", 0.6, 1), ("Strong", 0.8, 2), ("Very strong", 1.0, 2)))]

    def apply(self, s, params):
        s = s.copy()
        a, r = float(params["a"]), int(params["r"])
        out = []
        for c in s.stars:
            er = _min_filter(c, r)
            red = (1 - a) * c + a * er
            out.append(np.clip(red ** (1 + 0.6 * a), 0, 1))     # also soften faint halos
        s.stars = np.stack(out).astype(np.float32)
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
        if not s.stars_out or s.stars is None:
            return False, "Stars are not separated"
        if s.linear != s.stars_linear:
            return False, ("Brighten the stars first" if s.stars_linear else "Brighten the image first")
        return True, ""

    def variants(self, s):
        return [Variant(f"Stars {int(k * 100)}%", {"k": k}, "", k == 1.0) for k in (0.6, 0.8, 1.0, 1.2)]

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


TOOLS = {t.key: t for t in (CombineChannels(), BalanceColours(), RemoveGreen(), TakeStarsOut(), Brighten(),
                            BrightenStars(), ShrinkStars(), PutStarsBack())}


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

WORKFLOWS = {
    "osc": ["balance", "scnr", "stars_out", "stretch", "star_stretch", "shrink", "stars_back"],
    "smart": ["balance", "scnr", "stars_out", "stretch", "star_stretch", "shrink", "stars_back"],
    "mono_rgb": ["combine", "balance", "scnr", "stars_out", "stretch", "star_stretch", "shrink", "stars_back"],
    "mono_lrgb": ["combine", "balance", "scnr", "stars_out", "stretch", "star_stretch", "shrink", "stars_back"],
    "mono_nb": ["combine", "stars_out", "stretch", "scnr", "star_stretch", "shrink", "stars_back"],
    "mono": ["stars_out", "stretch", "star_stretch", "shrink", "stars_back"],
}

WHY = {
    "combine": "Your filters are still separate images - combine them into one colour picture first.",
    "balance": "Colour balance works best on the unstretched data, before anything else changes the colours.",
    "stars_out": "Separating the stars now lets you brighten the nebula strongly without bloating the stars.",
    "stretch": "Time to turn the linear data into a visible picture.",
    "star_stretch": "The stars-only image needs its own gentle brighten so stars stay small and colourful.",
    "shrink": "Optional: smaller stars make the main object stand out.",
    "stars_back": "Last step: put the stars back over the processed image.",
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
        for k, v in s.channels.items():
            arrays["ch_" + k] = v
        for k, v in s.channel_stars.items():
            arrays["cs_" + k] = v
        np.savez(d / "state.npz", **arrays)
        meta = {"linear": s.linear, "stars_linear": s.stars_linear, "stars_out": s.stars_out, "done": s.done,
                "skipped": s.skipped, "kind": s.kind, "target": s.target, "sources": s.sources,
                "palette": s.palette}
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
                  sources=meta.get("sources", []), palette=meta.get("palette"))
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
    "balance": "SpectrophotometricColorCalibration (SPCC) - plate solve first (ImageSolver); "
               "BackgroundNeutralization if no SPCC",
    "scnr": "SCNR (Average Neutral, green)",
    "stars_out": "StarXTerminator (tick 'Generate star image')",
    "stretch": "GeneralizedHyperbolicStretch (GHS) or HistogramTransformation (use STF as a starting point)",
    "star_stretch": "Star Stretch (Bill Blanshan's PixelMath / script) or a gentle HistogramTransformation",
    "shrink": "Star Reduction (Bill Blanshan's script) or MorphologicalTransformation (Erosion)",
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
