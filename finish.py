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
import re
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
    load_note: str = ""                    # message for the user about how the masters were read
    stars_hidden: bool = False             # stars separated by Prepare but still shown in the picture:
                                           # "Take Stars Out" makes the separation visible
    step_note: str = ""                    # what the last step found out, for the log (not saved)

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


def _sky_noise(x):
    """Robust (median, noise) of the sky in a plane."""
    sm = x[::4, ::4]
    med = float(np.median(sm))
    d = (sm[:, 1:] - sm[:, :-1]).ravel()
    return med, float(np.median(np.abs(d - np.median(d)))) * 1.4826 / np.sqrt(2) + 1e-9


def _patch_shift(a, b, search=48):
    """Shift (dx, dy) that moves patch b onto patch a (phase correlation, sub-pixel), or None."""
    def prep(p):
        p = np.nan_to_num(np.asarray(p, np.float32))
        p = p - np.median(p)
        hi = np.percentile(p, 99.7)
        p = np.clip(p, 0, hi if hi > 0 else None)
        return p * np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))
    A, B = prep(a), prep(b)
    if not A.any() or not B.any():
        return None
    F = np.fft.fft2(A) * np.conj(np.fft.fft2(B))
    r = np.fft.ifft2(F / np.maximum(np.abs(F), 1e-12)).real
    r = np.fft.fftshift(r)
    cy, cx = r.shape[0] // 2, r.shape[1] // 2
    win = r[cy - search:cy + search + 1, cx - search:cx + search + 1]
    py, px = np.unravel_index(int(np.argmax(win)), win.shape)
    if win[py, px] < 4 * np.std(win):           # no clear peak: nothing to measure here
        return None

    def sub(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    dy = dx = 0.0
    if 0 < py < win.shape[0] - 1:
        dy = sub(win[py - 1, px], win[py, px], win[py + 1, px])
    if 0 < px < win.shape[1] - 1:
        dx = sub(win[py, px - 1], win[py, px], win[py, px + 1])
    return (px - search + dx, py - search + dy)


def _edge_offset(ref_img, img, shape):
    """How far a lined-up master still is from the reference in the four corners of the picture and
    in the centre, measured by comparing image patches (stars included) directly - so any offset
    shows, however large. Returns (worst corner, centre) in pixels, or None if nothing to measure."""
    h, w = shape
    s = int(min(384, h // 3, w // 3))
    if s < 64:
        return None
    m_ = int(0.04 * min(h, w))
    spots = [(m_, m_), (m_, w - m_ - s), (h - m_ - s, m_), (h - m_ - s, w - m_ - s)]
    worst, n = 0.0, 0
    for y0, x0 in spots:
        d = _patch_shift(ref_img[y0:y0 + s, x0:x0 + s], img[y0:y0 + s, x0:x0 + s])
        if d is not None:
            worst = max(worst, float(np.hypot(*d)))
            n += 1
    if n == 0:
        return None
    y0, x0 = h // 2 - s // 2, w // 2 - s // 2
    c = _patch_shift(ref_img[y0:y0 + s, x0:x0 + s], img[y0:y0 + s, x0:x0 + s])
    return worst, (float(np.hypot(*c)) if c is not None else 0.0)


EDGE_LIMIT = 1.5          # px: a master whose corner stars are further off than this would show double stars


def _cards_look(m) -> dict:
    """A master's header as a dict (for its plate solution), or {}."""
    try:
        import fits_core as _core
        return _core.cards_to_dict(getattr(m, "cards", None) or []) or dict(getattr(m, "look", None) or {})
    except Exception:  # noqa: BLE001
        return {}


def assemble_mono(masters, stars_of=lambda m: None, log=None, crop_edges=True):
    """Mono masters -> one plane per filter, all on ONE pixel grid.
    * Every master is checked against a reference master by its stars. Masters from another
      camera or telescope are bent a little differently by their optics, most at the edges; a
      polynomial fitted to the matched stars puts them exactly on the reference grid (no coloured
      double stars in the corners).
    * Several masters of the same filter (other nights, other setups) are all used: brought to the
      same brightness scale and averaged, the less noisy ones counting more.
    Returns (channels, channel_stars, notes)."""
    import stacking as st
    lg = log or (lambda t: None)
    notes = []
    pref = ("L", "Ha", "G", "R", "OIII", "SII", "B")
    order = sorted(range(len(masters)), key=lambda i: (pref.index(masters[i].filter)
                                                       if masters[i].filter in pref else 9, i))

    def depth(m):          # the deepest master is the best reference: most stars, least noise
        lk = _cards_look(m)
        for k in ("EXPTOTAL", "NCOMBINE"):
            try:
                v = float(lk.get(k) or 0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                return v
        return 0.0
    deepest = max(order, key=lambda i: (depth(masters[i]), -order.index(i)))
    if depth(masters[deepest]) > 0:
        if order[0] != deepest:
            notes.append(f"lined up to {getattr(masters[deepest], 'path', Path('the deepest master')).stem} "
                         "(the deepest master)")
        order = [deepest] + [i for i in order if i != deepest]
    ref = masters[order[0]]
    ref_plane = np.clip(ref.data[0], 0, None).astype(np.float32)
    shape = ref_plane.shape
    # starless masters (StarXTerminator on page 3) have no stars to line up by: measure on the
    # starless image plus its stars-only image, i.e. the picture as it really is
    _rs = stars_of(ref)
    ref_full = ref_plane + _rs[0] * 65535.0 if _rs is not None and _rs.shape[1:] == shape else ref_plane
    ref_grid = str(_cards_look(ref).get("ALIGNREF") or "")
    planes = {}          # filter -> [(plane, stars or None, weight)]
    fixed, failed = [], []
    checks = []          # (name, worst corner offset) after lining up
    shared = 0           # masters page 3 already put on the reference's grid
    worst = 0.0
    for i in order:
        m = masters[i]
        img = np.clip(m.data[0], 0, None).astype(np.float32)
        stars = stars_of(m)
        star_plane = stars[0] * 65535.0 if stars is not None else None        # same units as the data
        full = img + star_plane if star_plane is not None and star_plane.shape == img.shape else img
        same_grid = (m is not ref and img.shape == shape and ref_grid
                     and str(_cards_look(m).get("ALIGNREF") or "") == ref_grid)
        if same_grid:
            # page 3 already lined this master up with the others: keep it unless the check says otherwise
            pre = None
            try:
                pre = _edge_offset(ref_full, full, shape)
            except Exception:  # noqa: BLE001
                pre = None
            if pre is None or (pre[0] <= 0.6 and pre[1] <= 0.6):
                # page 3 put it on the same grid; a check that can't be made (too few stars) is no reason to move it
                shared += 1
                if pre is not None:
                    checks.append((getattr(m, "path", Path(m.filter)).stem, pre[0]))
                planes.setdefault(m.filter, []).append((img, star_plane))
                continue
            kept = (img, star_plane, full, pre)        # to fall back on if lining it up again makes it worse
        else:
            kept = None
        if m is not ref:
            try:
                model, info = st.register_distortion(ref_full, full, shape,
                                                     ref_look=_cards_look(ref), img_look=_cards_look(m))
                off = max(info["shift"], info["edge_affine"]) if img.shape != shape else info["edge_affine"]
                if img.shape != shape or info["shift"] > 0.25 or info["edge_affine"] > 0.6:
                    warped = st.warp_poly([img] + ([star_plane] if star_plane is not None else []), model, shape)
                    img = np.nan_to_num(warped[0], nan=0.0)
                    if star_plane is not None:
                        star_plane = np.nan_to_num(warped[1], nan=0.0)
                    full = img + star_plane if star_plane is not None and star_plane.shape == img.shape else img
                    fixed.append(f"{m.filter} ({m.path.stem if hasattr(m, 'path') else m.filter}: off by up to "
                                 f"{off:.1f} px, now {info['rms']:.2f} px)")
                    worst = max(worst, off)
            except ValueError as e:
                same_filter = sum(1 for x in masters if x.filter == m.filter) > 1
                if img.shape != shape or same_filter:
                    # an unmatched master would put its stars somewhere else: double stars. Leave it out
                    # when the filter has other masters; keep it (with a warning) when it is the only one
                    failed.append(f"{getattr(m, 'path', Path(m.filter)).stem}: its stars could not be matched to the "
                                  f"others ({e}) - left out so it can't cause double stars")
                    continue
                failed.append(f"{m.filter}: its stars could not be matched to the others ({e}) - used as it is; if "
                              "the stars look doubled, line this filter up in PixInsight (StarAlignment)")
        elif star_plane is not None and star_plane.shape != shape:
            star_plane = None
        if m is not ref:
            # check the result: are this master's stars really on the reference's, out to the corners?
            try:
                off = _edge_offset(ref_full, full, shape)
            except Exception:  # noqa: BLE001
                off = None
            name = getattr(m, "path", Path(m.filter)).stem
            if kept is not None and (off is None or off[0] > kept[3][0]):
                # it was already on the reference's grid (page 3) and the new fit only moved it further away (a
                # faint filter with few stars, e.g. SII): keep it as page 3 left it - no shift, no narrower crop
                img, star_plane, full, pre = kept
                fixed = [x for x in fixed if not x.startswith(f"{m.filter} (")]
                off = pre
            if off is not None:
                checks.append((name, off[0]))
                if off[0] > EDGE_LIMIT:
                    same_filter = sum(1 for x in masters if x.filter == m.filter) > 1
                    if same_filter:
                        failed.append(f"{name}: after lining up, its stars are still {off[0]:.1f} px off in a corner "
                                      "- left out so it can't cause double stars")
                        continue
                    failed.append(f"{m.filter}: after lining up, its stars are still {off[0]:.1f} px off in a corner "
                                  "(it is the only master of this filter, so it is used) - expect coloured edges on "
                                  "stars near the corners")
        planes.setdefault(m.filter, []).append((img, star_plane))
    # the area every master covers (lined-up masters leave empty wedges at their edges)
    cover = None
    for items in planes.values():
        for img, _stp in items:
            # real gaps are wedges, not single noise pixels that happened to clip to 0
            v = _box_blur((img > 0).astype(np.float32), 2) > 0.5
            cover = v if cover is None else (cover & v)
    crop = None
    if crop_edges and cover is not None and not cover.all():
        import prepare as P
        r = P._valid_rect(cover[::4, ::4])
        if r:
            x0, y0, x1, y1 = (int(r[0] * 4) + 4, int(r[1] * 4) + 4, int(r[2] * 4) - 4, int(r[3] * 4) - 4)
            if (x1 - x0) * (y1 - y0) > 0.3 * shape[0] * shape[1]:
                crop = (x0, y0, x1, y1)
    if crop:
        x0, y0, x1, y1 = crop
        planes = {f: [(img[y0:y1, x0:x1], None if stp is None else stp[y0:y1, x0:x1]) for img, stp in items]
                  for f, items in planes.items()}
        shape = (y1 - y0, x1 - x0)
        notes.append(f"cropped to the area all masters cover ({x1 - x0} x {y1 - y0} px)")
    chans, cstars = {}, {}
    for filt, items in planes.items():
        if len(items) == 1:
            img, stp = items[0]
        else:
            # same brightness scale as the first of this filter, then a noise-weighted average
            b0, n0 = _sky_noise(items[0][0])
            s0 = float(np.percentile(items[0][0][::4, ::4], 99.5)) - b0
            acc = np.zeros(shape, np.float64)
            accs = np.zeros(shape, np.float64) if all(x[1] is not None for x in items) else None
            wsum = 0.0
            for img, stp in items:
                b, n = _sky_noise(img)
                sc = s0 / max(float(np.percentile(img[::4, ::4], 99.5)) - b, 1e-6)
                wgt = 1.0 / max(n * sc, 1e-9) ** 2
                acc += wgt * ((img - b) * sc + b0)
                if accs is not None:
                    accs += wgt * (stp * sc)
                wsum += wgt
            img = (acc / wsum).astype(np.float32)
            stp = (accs / wsum).astype(np.float32) if accs is not None else None
            notes.append(f"{len(items)} {filt} masters combined")
        chans[filt] = np.clip(img / 65535.0, 0, None).astype(np.float32)
        if stp is not None:
            cstars[filt] = np.clip(stp / 65535.0, 0, None).astype(np.float32)
    if len(cstars) != len(chans):
        cstars = {}
    if shared:
        notes.append(f"{shared} master(s) were already lined up on page 3 and match the reference - used as they are")
    if checks:
        kept = [c for c in checks if c[1] <= EDGE_LIMIT]
        if kept:
            notes.append(f"alignment check: {len(kept)} master(s) match the reference to within "
                         f"{max(c[1] for c in kept):.1f} px in every corner")
    if fixed:
        notes.append(f"{len(fixed)} master(s) corrected for a different field distortion (edges were off by up to "
                     f"{worst:.1f} px)")
        for x in fixed:
            lg("Lined up with distortion correction: " + x)
    notes += failed
    return chans, cstars, notes


def master_depth(m) -> float:
    """Total exposure (or frame count) a master holds, from its header; 0 when unknown."""
    lk = _cards_look(m)
    for k in ("EXPTOTAL", "NCOMBINE"):
        try:
            v = float(lk.get(k) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return v
    return 0.0


def pick_colour_masters(color, is_dual):
    """Which colour-camera masters make the picture. With a dual-band filter, the narrowband
    masters (unfiltered ones would bloat the stars and wash out the nebula colours); otherwise
    the filter with the most exposure. Returns (used, left_out)."""
    duals = [m for m in color if is_dual(m)]
    if duals:
        used = duals
    else:
        by = {}
        for m in color:
            by.setdefault(m.filter, []).append(m)
        best = max(by, key=lambda f: (sum(master_depth(m) for m in by[f]), len(by[f])))
        used = by[best]
    return used, [m for m in color if m not in used]


def assemble_colour(masters, stars_of=lambda m: None, log=None):
    """Colour masters -> one (3, h, w) picture per filter, all on ONE grid: every master of a filter
    is used (lined up and noise-weighted like mono masters, the deepest one as the reference),
    instead of only the first one found. Each colour is lined up with the same colour of the deepest
    master (red with red …: with a dual-band filter red is Ha and blue OIII, which look different), so
    the three colours stay on the deepest master's grid. Returns ({filter: rgb}, {filter: stars rgb},
    notes), images scaled 0..1."""
    from types import SimpleNamespace
    import re as _re
    cache = {}

    def st_of(m):
        if id(m) not in cache:
            cache[id(m)] = stars_of(m)
        return cache[id(m)]
    per_c, star_c, notes = [], [], []
    for c in range(3):
        planes = [SimpleNamespace(path=m.path, data=m.data[c:c + 1], cards=m.cards, look=m.look,
                                  filter=m.filter, orig=m, ch=c) for m in masters]

        def plane_stars(w):
            st_img = st_of(w.orig)
            return None if st_img is None or st_img.shape[0] != 3 else st_img[w.ch:w.ch + 1]
        ch_, cs_, nt_ = assemble_mono(planes, plane_stars, log, crop_edges=False)
        per_c.append(ch_)
        star_c.append(cs_)
        for n in nt_:
            if n not in notes:
                notes.append(n)
    rgbs, stars = {}, {}
    for f in dict.fromkeys(m.filter for m in masters):
        if all(f in per_c[c] for c in range(3)) and len({per_c[c][f].shape for c in range(3)}) == 1:
            rgbs[f] = np.stack([per_c[c][f] for c in range(3)])
            if all(f in star_c[c] for c in range(3)):
                stars[f] = np.stack([star_c[c][f] for c in range(3)])
    # the area every master covers, the same for all three colours (lined-up masters leave wedges)
    for f, img in list(rgbs.items()):
        v = _box_blur((img.min(axis=0) > 0).astype(np.float32), 2) > 0.5
        if not v.all():
            import prepare as P
            r = P._valid_rect(v[::4, ::4])
            if r:
                x0, y0, x1, y1 = (int(r[0] * 4) + 4, int(r[1] * 4) + 4, int(r[2] * 4) - 4, int(r[3] * 4) - 4)
                if (x1 - x0) * (y1 - y0) > 0.3 * v.size:
                    rgbs[f] = img[:, y0:y1, x0:x1]
                    if f in stars:
                        stars[f] = stars[f][:, y0:y1, x0:x1]
                    notes.append(f"cropped to the area all masters cover ({x1 - x0} x {y1 - y0} px)")
    clean, seen = [], set()
    for n in notes:                      # the same note for each colour: keep it once
        n = _re.sub(r"\d+ master\(s\)", f"{len(masters) - 1} other master(s)", n) if len(masters) > 1 else n
        key = _re.sub(r"[\d.]+", "#", n)
        if key not in seen:
            seen.add(key)
            clean.append(n)
    return rgbs, stars, clean


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
        is_dual = lambda m: dual_band_filter(m.filter, m.look)  # noqa: E731
        used, left = pick_colour_masters(color, is_dual)
        notes = []
        if len(used) > 1:
            try:
                rgbs, srgbs, notes = assemble_colour(used, stars_of)
                if not rgbs:
                    raise ValueError("no colour could be matched across the masters")
            except Exception as e:  # noqa: BLE001 - never stop on this: fall back to the deepest master
                deepest = max(used, key=master_depth)
                rgbs = {deepest.filter: np.clip(deepest.data, 0, None).astype(np.float32)}
                st0 = stars_of(deepest)
                srgbs = {deepest.filter: st0 * 65535.0} if st0 is not None else {}
                notes = [f"could not combine the masters ({e}) - used the deepest one"]
                rgbs = {k: v / 65535.0 for k, v in rgbs.items()}
                srgbs = {k: v / 65535.0 for k, v in srgbs.items()}
        else:
            m = used[0]
            rgbs = {m.filter: np.clip(m.data / 65535.0, 0, None).astype(np.float32)}
            st0 = stars_of(m)
            srgbs = {m.filter: st0} if st0 is not None and st0.shape == m.data.shape else {}
        if left:
            notes.append("left out " + ", ".join(sorted({f"{m.filter} ({m.path.stem})" for m in left}))
                         + (" - unfiltered data would bloat the stars and wash out the dual-band colours"
                            if any(is_dual(m) for m in used) else " - another filter with less exposure"))
        if notes:
            s.load_note = "; ".join(notes)
        dual_f = [f for f in rgbs if any(is_dual(m) and m.filter == f for m in used)]
        import fits_core as core
        if len(dual_f) >= 2 and len({core.band_lines(f)[0] for f in dual_f}) >= 2:
            # e.g. L-Ultimate (Ha+OIII) + L-Synergy (SII+OIII): a full SHO set from a colour camera
            o3s, o3_stars = [], []
            for f in dual_f:
                line, o3 = split_dual_band(rgbs[f])
                name = core.band_lines(f)[0]
                s.channels[name] = line
                o3s.append(o3)
                st_img = srgbs.get(f)
                if st_img is not None and st_img.shape == rgbs[f].shape:
                    s.channel_stars[name] = st_img[0]
                    o3_stars.append((st_img[1] + st_img[2]) / 2)
            o3, o3_note = combine_oiii(o3s, [s.channels[core.band_lines(f)[0]] for f in dual_f],
                                       [sum(master_depth(m) for m in used if m.filter == f) for f in dual_f])
            s.channels["OIII"] = o3                                               # OIII from both filters
            if o3_note:
                s.load_note = "; ".join(x for x in (s.load_note, o3_note) if x)
            if len(o3_stars) == len(dual_f):
                s.channel_stars["OIII"] = np.mean(o3_stars, axis=0).astype(np.float32)
            else:
                s.channel_stars = {}
            s.kind = "osc_sho"
            s.filter = " + ".join(dual_f)
            return s
        # one picture: the filter with the most exposure among those used
        f = max(rgbs, key=lambda k: sum(master_depth(m) for m in used if m.filter == k))
        m = next(x for x in used if x.filter == f)
        s.rgb = rgbs[f]
        s.kind = "smart" if smart else "osc"
        s.filter = f
        if is_dual(m):
            s.kind = "osc_dual"
        st_img = srgbs.get(f)
        if st_img is not None and st_img.shape == s.rgb.shape:
            s.stars, s.stars_out, s.stars_hidden = st_img, True, True
        elif m.look.get("STARLESS"):
            # already starless but the stars file is missing: never run star removal twice
            s.done.append("stars_out")
            s.stars_missing = True
        return s
    chans, cstars, notes = assemble_mono(masters, stars_of)
    import targets as _tg
    if _tg.is_comet(s.target) and len(chans) >= 2:
        try:
            notes = notes + align_comet_channels(chans)
        except Exception as e:  # noqa: BLE001 - never stop loading over this
            notes = notes + [f"comet: could not line up the filters on the comet ({e})"]
    s.channels, s.channel_stars = chans, cstars
    if notes:
        s.load_note = "; ".join(notes)
    f = set(s.channels)
    known = {"Ha", "OIII", "SII", "Hβ", "L", "R", "G", "B"}
    unknown = sorted(f - known)
    if len(f) >= 2 and unknown and not ({"Ha", "OIII"} <= f or {"R", "G", "B"} <= f):
        # filters we cannot name: never drop channels - combine them in a sensible order instead
        names = sorted(f)
        role = ("Ha", "OIII", "SII") if len(names) <= 3 else ("R", "G", "B", "L")
        ren = dict(zip(names, role))
        s.channels = {ren[k]: v for k, v in s.channels.items() if k in ren}
        s.channel_stars = {ren[k]: v for k, v in s.channel_stars.items() if k in ren}
        s.load_note = ("Filter names not recognised (" + ", ".join(names) + ") - treated as "
                       + ", ".join(f"{k} = {v}" for k, v in ren.items())
                       + ". To fix, set the filter names on the first page (Frame > Set Filter Name).")
        f = set(s.channels)
    broadband_galaxy = {"R", "G", "B"} <= f and is_galaxy_target(s.target)
    if {"Ha", "OIII"} <= f and not broadband_galaxy:
        s.kind = "mono_nb"
    elif {"R", "G", "B"} <= f:
        # a galaxy with RGB (and L) is processed as LRGB even when narrowband masters came along too:
        # Ha goes in as HII regions; the narrowband palettes are still offered under Combine Colour Channels
        if {"Ha", "OIII"} <= f:
            s.load_note = "; ".join(x for x in (getattr(s, "load_note", ""), (
                "a galaxy with R, G, B" + (", L" if "L" in f else "") + " and narrowband: processed as "
                + ("LRGB" if "L" in f else "RGB") + " - Ha adds the HII regions (Add Hydrogen Glow); the "
                "narrowband palettes (SHO, HOO...) are still offered under Combine Colour Channels")) if x)
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


STAR_KEEP_SMALL = 0.85


def _local_max(a: np.ndarray, r: int) -> np.ndarray:
    """Largest value within about r pixels (fast: on 4x4 blocks, then grown back), never below a itself."""
    f = 4
    h, w = a.shape
    hh, ww = max(f, h // f * f), max(f, w // f * f)
    if h < f or w < f:
        return a.copy()
    b = a[:hh, :ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3))
    for k in range(int(np.ceil(r / f))):               # square and cross steps in turn: a round (octagon) reach
        p = np.pad(b, 1, mode="edge")
        nb = [p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]]
        if k % 2 == 0:
            nb += [p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]]
        b = np.maximum.reduce(nb)
    m = np.repeat(np.repeat(b, f, axis=0), f, axis=1)
    m = np.pad(m, ((0, h - hh), (0, w - ww)), mode="edge")
    return np.maximum(m, a)


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
STARS_ALONE = True        # the star steps show the stars on their own (the window's tick box can add the nebula)


def star_view():
    """How the star steps show the picture: the stars alone, or over the nebula."""
    return "only" if STARS_ALONE else True


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
        with_stars = star_view() if (s.done and s.done[-1] in STAR_TOOLS) else s.stars_hidden
    img = s.rgb
    if not (with_stars and s.stars_out and s.stars is not None):
        return display_rgb8(img, s.linear, linked)
    st_ = s.stars if s.stars.shape[0] == img.shape[0] else np.repeat(s.stars, img.shape[0], 0)
    if with_stars == "only":                   # the stars on their own (while the star steps are worked on)
        return display_rgb8(st_, s.stars_linear, linked)
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
    repeatable = False             # may be run again after it is done (each run is its own step)

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
        return state_display(self.apply(s, params), with_stars=star_view() if self.shows_stars else None)


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


OIII_BOOST = 2.0       # how much the faint, OIII-only light is lifted at full strength


def boost_faint_oiii(ha: np.ndarray, o3: np.ndarray, strength: float) -> np.ndarray:
    """OIII with its faint OIII-only parts lifted (the Squid in the Flying Bat, the outer shells of planetaries).
    The part of the OIII that simply follows the Ha (the same gas glowing in both) is taken away first - measured
    as the typical OIII-to-Ha ratio where the Ha is bright - and what is left, smoothed so single noisy pixels
    don't count and lifted above the sky's own noise, is added back on top of the OIII."""
    if strength <= 0:
        return o3
    hm, _hn = _stats(ha)
    om, _on = _stats(o3)
    hr, orr = ha - hm, o3 - om
    sel = hr > np.percentile(hr[::4, ::4], 95)
    a = float(np.median(np.clip(orr[sel], 0, None) / np.maximum(hr[sel], 1e-9))) if sel.sum() > 50 else 0.0
    d = _blur((orr - a * np.clip(hr, 0, None)).astype(np.float32), 4.0)
    dm, dn = _stats(d)
    w = np.clip((d - dm - dn) / (2 * dn + 1e-12), 0, 1)       # faded in above the sky's own ripple
    ex = w * np.clip(d - dm, 0, None)
    return (o3 + float(strength) * OIII_BOOST * ex).astype(np.float32)


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


LEAK_MAX = 0.10      # most of a red line a colour camera's green/blue pixels catch (typically 5-9 %)


def _line_leak(o, line, inner):
    """How much of a red line (Ha or SII) the green/blue pixels caught, from the data itself. Real OIII is never
    negative, and among the pixels brightest in the line some always have next to no OIII (dust-bound fronts),
    so the low end of OIII/line there is the leak. Never more than LEAK_MAX; 0 when it can't be told."""
    m = inner & (line > np.percentile(line[inner], 95))
    if m.sum() < 100:
        return 0.0
    return float(np.clip(np.percentile(o[m] / np.maximum(line[m], 1e-12), 5), 0.0, LEAK_MAX))


def combine_oiii(o3s, lines, depths=None):
    """OIII from two dual-band filters (e.g. Askar D1 = Ha+OIII, D2 = SII+OIII). The camera's green and blue
    pixels also catch a little of the red line, so each OIII estimate carries some Ha (D1) or SII (D2): that leak
    is measured per filter and taken out, filter 2's OIII is put on filter 1's scale, and the two are combined
    weighted by exposure (or noise). Returns (oiii, note)."""
    if len(o3s) != 2 or len(lines) != 2:
        return np.mean(o3s, axis=0).astype(np.float32), ""
    o1, o2 = (np.asarray(x, np.float64) for x in o3s)
    l1, l2 = (np.asarray(x, np.float64) for x in lines)
    st_ = max(1, int(np.sqrt(o1.size / 400000)))
    # measured on smoothed copies (noise would scatter the ratios) and away from the edges, where fewer frames -
    # often of one filter only - reached
    sm_ = lambda x: _blur(x.astype(np.float32), 3.0)[::st_, ::st_].astype(np.float64)  # noqa: E731
    a1, a2, b1, b2 = sm_(o1), sm_(o2), sm_(l1), sm_(l2)
    hh, ww = a1.shape
    yy_, xx_ = np.mgrid[:hh, :ww]
    inner = (yy_ > 0.1 * hh) & (yy_ < 0.9 * hh) & (xx_ > 0.1 * ww) & (xx_ < 0.9 * ww)
    if inner.sum() < 2000:
        return (np.mean(o3s, axis=0).astype(np.float32),
                "OIII from both filters: averaged (too little picture to measure the line leakage)")
    f1, f2 = _line_leak(a1, b1, inner), _line_leak(a2, b2, inner)
    c1s, c2s = (a1 - f1 * b1)[inner], (a2 - f2 * b2)[inner]
    span = lambda x: float(np.percentile(x, 97) - np.percentile(x, 50))  # noqa: E731 - the OIII's brightness range
    p_ = span(c1s) / span(c2s) if span(c2s) > 0 else 1.0
    if not (0.05 < p_ < 20):
        p_ = 1.0
    c1 = o1 - f1 * l1                                       # OIII from filter 1 without its line's leak
    c2 = p_ * (o2 - f2 * l2)                                # filter 2's, on filter 1's scale, without its leak
    if depths and len(depths) == 2 and min(depths) > 0:
        # by exposure: the masters have been through noise reduction, which leaves pixel noise that says little
        # about how deep each one is; filter 2's OIII is scaled by p, which scales its noise too
        w1, w2 = float(depths[0]), float(depths[1]) / p_ ** 2
        how = "by exposure"
    else:
        w1 = 1.0 / max(_sky_noise(c1)[1], 1e-12) ** 2
        w2 = 1.0 / max(_sky_noise(c2)[1], 1e-12) ** 2
        how = "noise-weighted"
    share = min(max(w1 / (w1 + w2), 0.2), 0.8)              # both filters hold real OIII - never drop one
    out = c1 * share + c2 * (1 - share)
    note = (f"OIII from both filters: line leakage taken out ({100 * f1:.0f}% of the Ha, {100 * f2:.0f}% of the SII), "
            f"combined {how} ({100 * share:.0f}% / {100 * (1 - share):.0f}%)")
    return out.astype(np.float32), note


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
            weak_l = bool("L" in f and self._weak_l(s))
            if "L" in f:
                out.append(Variant("LRGB", {"palette": "LRGB"},
                                   "Colour from RGB, detail from L" + (" - but your L is much noisier than the colour "
                                                                       "filters (too few frames?), so RGB is suggested"
                                                                       if weak_l else ""), not weak_l))
            out.append(Variant("RGB", {"palette": "RGB"}, "True colour" + (" - your L is left out (much noisier "
                                                                          "than R, G and B)" if weak_l else ""),
                               "L" not in f or weak_l))
        nb_first = not ({"R", "G", "B"} <= f)          # with RGB too, the broadband mix is the suggestion
        if {"Ha", "OIII", "SII"} <= f:
            out += [Variant(*PALETTE_INFO[k][:1], {"palette": k}, PALETTE_INFO[k][1], k == "SHO" and nb_first)
                    for k in ("SHO", "FORAXX", "HSO", "HOO+S", "HOO")]
        elif {"Ha", "OIII"} <= f:
            out += [Variant(PALETTE_INFO[k][0], {"palette": k}, PALETTE_INFO[k][1], k == "HOO" and nb_first)
                    for k in ("HOO", "HOO+", "HHO", "FORAXX")]
        if {"Ha", "OIII"} <= f:
            out.append(Variant("HOO + faint OIII", {"palette": "HOO", "o3boost": 1.0},
                               "HOO with the faint OIII-only glow lifted (e.g. the Squid in the Flying Bat): "
                               "OIII that just follows the Ha is left as it is"))
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
        out = [Slider("r", "Red strength", 0.4, 1.6, m[0], "How strong the filter shown as red is"),
               Slider("g", "Green strength", 0.4, 1.6, m[1], "How strong the filter shown as green is"),
               Slider("b", "Blue strength", 0.4, 1.6, m[2], "How strong the filter shown as blue is")]
        if "o3boost" in params:
            out.append(Slider("o3boost", "Faint OIII boost", 0.0, 2.0, params["o3boost"],
                              "How much the faint OIII-only glow is lifted"))
        return out

    def tuned(self, s, params, values):
        p = dict(params)
        p["mix"] = [values.get("r", 1.0), values.get("g", 1.0), values.get("b", 1.0)]
        if "o3boost" in p:
            p["o3boost"] = values.get("o3boost", p["o3boost"])
        return p

    @staticmethod
    def _weak_l(s) -> bool:
        """True when the L master is much noisier than the colour filters (e.g. one L frame against hours of RGB):
        its detail would make the picture worse, not better. Signal-to-noise of each filter, from the data."""
        def snr(x):
            med, mad, hi = channel_stats(x)
            return (hi - med) / max(mad, 1e-9)
        try:
            rgb = np.mean([snr(s.channels[k]) for k in ("R", "G", "B")])
            return snr(s.channels["L"]) < 0.6 * rgb
        except Exception:  # noqa: BLE001 - a guess only; LRGB stays the suggestion
            return False

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
            if not raw and params.get("o3boost") and "Ha" in n and "OIII" in n:
                n["OIII"] = boost_faint_oiii(n["Ha"], n["OIII"], float(params["o3boost"]))
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
        if ({"Ha", "R", "G", "B"} <= set(s.channels) and pal != "RGB+Ha" and "ha_glow" not in s.done
                and "ha_glow" not in s.skipped):
            s.skipped.append("ha_glow")          # stays in the list, so it is clear it was passed by
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

    def _narrowband(self, s):
        return s.kind in ("mono_nb", "osc_dual", "osc_sho") and s.palette not in ("NATURAL", None)

    GOLD_PALETTES = ("SHO", "HHO")         # palettes whose green is the Ha

    def variants(self, s):
        if self._narrowband(s) and s.palette not in self.GOLD_PALETTES:
            return [Variant(f"{int(a * 100)}%", {"amount": a}, "", a == 1.0) for a in (0.25, 0.5, 0.75, 1.0)]
        if self._narrowband(s):
            # narrowband: the green is real signal (Ha in SHO) - turn it gold instead of deleting it, so the
            # brightest structure keeps its brightness and the classic gold and blue appears
            return ([Variant(f"Gold {int(a * 100)}%", {"amount": a, "gold": True}, "", a == 1.0) for a in (0.5, 0.75, 1.0)]
                    + [Variant("Remove only", {"amount": 1.0, "gold": False},
                               "plain SCNR: takes the green away - darker, more blue and magenta")])
        g = s.rgb[1] - np.maximum(s.rgb[0], s.rgb[2])
        excess = float(np.mean(g[::3, ::3] > 0))
        best = 0.75 if excess > 0.5 else 0.5
        return [Variant(f"{int(a * 100)}%", {"amount": a}, "", a == best) for a in (0.25, 0.5, 0.75, 1.0)]

    def sliders(self, s, params):
        return [Slider("amount", "Amount", 0.0, 1.0, params["amount"], "How much of the green cast to remove")]

    GOLD_RED, GOLD_GREEN = 0.75, 0.45     # pure green (0,1,0) at 100% -> (0.75, 0.55, 0): gold

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        r, g, b = s.rgb
        neutral = np.minimum(g, (r + b) / 2)
        if params.get("gold"):
            ex = (g - neutral) * a                               # the green above the neutral level
            s.rgb = np.stack([np.minimum(r + self.GOLD_RED * ex, 1.0), g - self.GOLD_GREEN * ex, b]).astype(np.float32)
        else:
            s.rgb = np.stack([r, g * (1 - a) + neutral * a, b]).astype(np.float32)
        s.done.append(self.key)
        return s

    def why(self, s):
        if s.kind in ("mono_nb", "osc_dual", "osc_sho") and s.palette not in ("NATURAL", None):
            if s.palette in self.GOLD_PALETTES:
                return ("The green here is your Ha - the strongest signal. Gold turns it gold instead of removing it, "
                        "so the bright structure keeps its brightness: the classic Hubble gold and blue.")
            return "Narrowband palettes are very green straight out of combination - this tames it."
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
        if (s.stars is not None and s.rgb is not None and s.stars.shape[1:] == s.rgb.shape[1:]
                and s.stars_linear == s.linear and target_type(s) == "planetary nebula"):
            s.rgb, s.stars, _moved = rescue_object(s.rgb, s.stars)
            s.step_note = (f"moved {_moved:.0%} of the stars image's light - the part of the planetary nebula star "
                           "removal had taken for a star - back to the nebula" if _moved > 0.001 else
                           "checked the stars image for parts of the planetary nebula - none found")
        s.done.append(self.key)
        return s


def _grain_map(C: np.ndarray, k: int = 3) -> np.ndarray:
    """Pixel-to-pixel grain: how much each pixel differs from the mean of its four neighbours, smoothed."""
    g = np.zeros_like(C, dtype=np.float32)
    g[1:-1, 1:-1] = np.abs(C[1:-1, 1:-1] - 0.25 * (C[:-2, 1:-1] + C[2:, 1:-1] + C[1:-1, :-2] + C[1:-1, 2:]))
    return _box_blur(g, k)


def find_star_leftovers(starless: np.ndarray, stars: np.ndarray, max_stars: int = 30) -> tuple:
    """The boxes star removal painted over the brightest stars. StarXTerminator replaces a big star with a smooth,
    grain-free square patch; once the stars and the nebula are brightened differently its edges show as a square,
    often with bright bars along it, and a leftover core can stay in it (a close double star). A painted box is
    found by its missing grain (real nebula always has grain), around each bright star of the stars image.
    Returns ([(fx0, fy0, fx1, fy1)] as fractions of the picture's size, number left for Spot Fix): boxes inside
    bright, structured nebula (a reflection nebula hugging the star) are left alone - patching would smooth it."""
    L = _lum(starless).astype(np.float32)
    h, w = L.shape
    if min(h, w) < 128:
        return [], 0
    Ssm, f = _small_plane(_lum(stars).astype(np.float32), 1600)
    sm = _box_blur(Ssm, 2)
    pk = sm >= SpotFix._minmax(sm, 15, np.max) - 1e-12
    ys, xs = np.nonzero(pk & (sm > 0))
    order = np.argsort(-sm[ys, xs])[:max_stars]
    R = int(np.clip(600 * w / 12298, 120, 700))
    min_half = max(10, w // 400)
    out, busy = [], 0
    seen = []                            # boxes already found (a star and its leftover dot share one)
    for i in order:
        cy, cx = int(ys[i] * f + f // 2), int(xs[i] * f + f // 2)
        if any(a0 <= cx < a1 and b0 <= cy < b1 for a0, b0, a1, b1 in seen):
            continue
        y0, y1, x0, x1 = max(0, cy - R), min(h, cy + R), max(0, cx - R), min(w, cx + R)
        C = L[y0:y1, x0:x1]
        q = 4
        gm = _grain_map(C, 3)
        hh, ww = C.shape[0] // q * q, C.shape[1] // q * q
        if hh < 8 or ww < 8:
            continue
        g4 = gm[:hh, :ww].reshape(hh // q, q, ww // q, q).mean((1, 3))
        yy, xx = np.mgrid[:g4.shape[0], :g4.shape[1]]
        ccy, ccx = min((cy - y0) // q, g4.shape[0] - 1), min((cx - x0) // q, g4.shape[1] - 1)
        d = np.maximum(abs(yy - ccy), abs(xx - ccx))
        far = d > 0.7 * R / q
        if not far.any():
            continue
        ref = float(np.median(g4[far]))
        low = g4 < 0.45 * ref
        if not low[ccy, ccx]:
            continue
        comp = np.zeros_like(low)
        comp[ccy, ccx] = True
        for _ in range(R // q):
            g2 = (SpotFix._minmax(comp.astype(np.float32), 3, np.max) > 0) & low
            if (g2 == comp).all():
                break
            comp = g2
        if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
            continue                     # runs out of the window: a large smooth area, not a painted box
        ry, rx = np.nonzero(comp)
        by0, by1 = ry.min() * q + y0, (ry.max() + 1) * q + y0
        bx0, bx1 = rx.min() * q + x0, (rx.max() + 1) * q + x0
        if max(by1 - by0, bx1 - bx0) / 2 < min_half:
            continue
        seen.append((bx0, by0, bx1, by1))
        # busy surroundings (bright, structured nebula): leave it - Spot Fix by hand if it shows
        bw, bh = bx1 - bx0, by1 - by0
        m = int(max(bw, bh) * 0.8)
        X0, X1, Y0, Y1 = max(0, bx0 - m), min(w, bx1 + m), max(0, by0 - m), min(h, by1 + m)
        A = L[Y0:Y1, X0:X1]
        yy2, xx2 = np.mgrid[Y0:Y1, X0:X1]
        ring = ~((yy2 >= by0 - bh * 0.45) & (yy2 < by1 + bh * 0.45) & (xx2 >= bx0 - bw * 0.45) & (xx2 < bx1 + bw * 0.45))
        if ring.sum() < 50:
            continue
        grain = SpotFix._grain(A, ring) + 1e-12
        struct = float(np.std((_blur(A, 3) - _blur(A, 24))[ring])) / grain
        if struct > 1.1:
            busy += 1
            continue
        out.append((bx0 / w, by0 / h, bx1 / w, by1 / h))
    return out, busy


GLOW_KEEP = 0.7                     # how much of a cleaned star glow is kept (made round)


def heal_box(img: np.ndarray, x0, y0, x1, y1, grow=0.45, seed=0) -> np.ndarray:
    """Clean one painted box and the square glow around it. Star removal leaves a big star's glow in the starless
    picture cut square - a plate with straight edges and bright bars along them, and a grain-free box in the middle.
    Patching only the box leaves the square plate showing once the picture is brightened. So, around the box, the
    smooth part of the picture is rebuilt as background (filled in from well outside the glow) plus a ROUND glow made
    from the glow's own average fall-off with distance from the star (a little fainter), and the grain-free box gets
    the fine texture of an equal area right next to it (as PixInsight's CloneStamp would)."""
    c, h, w = img.shape
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return img
    rng = np.random.default_rng(seed)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    q = max(1, int(round(half / 24)))                     # the glow is smooth: worked on a smaller copy
    Rmax = 5.0 * half
    Rw = int(Rmax * 1.2)
    wx0, wy0 = max(0, int(cx - Rw)), max(0, int(cy - Rw))
    wx1, wy1 = min(w, int(cx + Rw)), min(h, int(cy + Rw))
    wx1, wy1 = wx0 + (wx1 - wx0) // q * q, wy0 + (wy1 - wy0) // q * q
    Hs, Ws = (wy1 - wy0) // q, (wx1 - wx0) // q
    if Hs < 8 or Ws < 8:
        return img
    yy, xx = np.mgrid[:Hs, :Ws]
    r = np.hypot((yy + 0.5) * q + wy0 - cy, (xx + 0.5) * q + wx0 - cx) / q
    border = np.zeros((Hs, Ws), bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    hq = half / q
    # the donor area for the box's texture (the same for every colour)
    bw, bh = x1 - x0, y1 - y0
    gx, gy = int(bw * grow) + 2, int(bh * grow) + 2
    px0, px1, py0, py1 = max(0, x0 - gx), min(w, x1 + gx), max(0, y0 - gy), min(h, y1 + gy)
    pw, ph = px1 - px0, py1 - py0
    cands = [(px0 - pw, py0), (px1, py0), (px0, py0 - ph), (px0, py1)]
    cands = [(dx, dy) for dx, dy in cands if 0 <= dx and dx + pw <= w and 0 <= dy and dy + ph <= h]
    donor = cands[int(rng.integers(0, len(cands)))] if cands else None
    sig = max(3.0, min(pw, ph) / 12)
    out = img.copy()
    Rh = None
    for k in range(c):
        W = out[k, wy0:wy1, wx0:wx1].astype(np.float64)
        Ls = _blur(W.reshape(Hs, q, Ws, q).mean((1, 3)).astype(np.float32), max(1.5, hq / 8)).astype(np.float64)

        def solve(R):
            m = (r < R) & ~border
            Bg = Ls.copy()
            ringv = Ls[(r >= R) & (r < R * 1.2)]
            Bg[m] = np.median(ringv) if ringv.size else np.median(Ls)
            Bg = SpotFix._harmonic(Bg, m, 200)
            E = Ls - Bg
            nb = int(R) + 2
            ri = np.minimum(r.astype(int), nb - 1)
            prof = np.zeros(nb)
            for i in range(nb):
                v = E[(ri == i) & m]
                prof[i] = np.median(v) if v.size else (prof[i - 1] if i else 0.0)
            return Bg, prof

        if Rh is None:                                   # how far the glow reaches (found once, on the first colour)
            _bg, prof = solve(Rmax / q)
            pk = prof[:int(hq * 1.2) + 1].max()
            cut = Rmax / q
            for i in range(int(hq * 1.5), len(prof)):
                if prof[i] < 0.06 * pk:
                    cut = i
                    break
            Rh = float(np.clip(cut * 1.25, hq * 2.2, Rmax / q))
        Bg, prof = solve(Rh)
        prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, "valid")
        prof = np.minimum.accumulate(prof)               # a real glow only fades outwards ...
        sg = max(1.0, 0.5 * hq)
        kk = np.arange(-int(3 * sg), int(3 * sg) + 1)
        ker = np.exp(-0.5 * (kk / sg) ** 2)
        prof = np.convolve(np.pad(prof, len(kk) // 2, mode="edge"), ker / ker.sum(), "valid")   # ... softly
        prof = np.clip(prof - prof[min(int(Rh), len(prof) - 1)], 0, None)          # down to the surroundings
        ri = np.clip(r, 0, len(prof) - 1.001)
        i0 = ri.astype(int)
        fr = ri - i0
        E0 = prof[i0] * (1 - fr) + prof[np.minimum(i0 + 1, len(prof) - 1)] * fr
        t = np.clip((Rh - r) / (0.12 * Rh), 0, 1)
        t = t * t * (3 - 2 * t)
        fade = np.clip(1 - (r / Rh) ** 2, 0, 1)
        newLs = Ls + t * ((Bg + GLOW_KEEP * E0 * (0.5 + 0.5 * fade)) - Ls)
        W = W + _resize((newLs - Ls).astype(np.float32), W.shape[0], W.shape[1]).astype(np.float64)
        # the grain-free box (grown to take in the bars along its edges): texture from right next to it
        fy0, fy1, fx0, fx1 = py0 - wy0, py1 - wy0, px0 - wx0, px1 - wx0
        if donor is not None and fy0 >= 0 and fx0 >= 0 and fy1 <= W.shape[0] and fx1 <= W.shape[1]:
            dx, dy = donor
            mg = 12                                                      # filtered with a margin: no edge lines
            ex0, ey0, ex1, ey1 = max(0, dx - mg), max(0, dy - mg), min(w, dx + pw + mg), min(h, dy + ph + mg)
            big = img[k, ey0:ey1, ex0:ex1].astype(np.float32)
            tex = (big - _blur(big, sig))[dy - ey0:dy - ey0 + ph, dx - ex0:dx - ex0 + pw].astype(np.float64)
            sd = 1.4826 * float(np.median(np.abs(tex - np.median(tex)))) + 1e-12
            tex = np.where(np.abs(tex) > 3.5 * sd, 0.0, tex)                 # no stars or specks copied along
            mg2 = int(3 * sig) + 2
            a0, a1 = max(0, fy0 - mg2), min(W.shape[0], fy1 + mg2)
            b0, b1 = max(0, fx0 - mg2), min(W.shape[1], fx1 + mg2)
            As = _blur(W[a0:a1, b0:b1].astype(np.float32), sig).astype(np.float64)[fy0 - a0:fy1 - a0, fx0 - b0:fx1 - b0]
            y2, x2 = np.mgrid[:ph, :pw]
            dist = np.minimum.reduce([y2, ph - 1 - y2, x2, pw - 1 - x2]).astype(np.float64)
            soft = np.clip(dist / max(4.0, 0.08 * min(pw, ph)), 0, 1)
            W[fy0:fy1, fx0:fx1] = W[fy0:fy1, fx0:fx1] * (1 - soft) + (As + tex) * soft
        out[k, wy0:wy1, wx0:wx1] = W.astype(out.dtype)
    return out


class CleanStarLeftovers(Tool):
    key = "leftovers"
    title = "Clean Bright-Star Leftovers"
    tip = ("Star removal (StarXTerminator) paints a smooth, grain-free square over each big star. Once the nebula "
           "and the stars are brightened differently, those squares show - often with bright bars along their "
           "edges, and sometimes with a leftover dot where a close double star was. This finds the painted squares "
           "by their missing grain, patches each with the surroundings' own texture, and turns the square glow "
           "around it into a soft round one, as a real star's glow is. Squares inside bright, structured nebula are "
           "left alone (patching would smooth the nebula) - use Spot "
           "Fix on those if they show. (PixInsight: CloneStamp)")
    pick = True

    def possible(self, s):
        return s.stars_out or s.stars_hidden or TOOLS["stars_out"].possible(s)

    def applicable(self, s):
        if not s.stars_out or s.stars is None or s.stars_hidden or s.rgb is None:
            return False, "Take the stars out first"
        if s.stars.shape[1:] != s.rgb.shape[1:]:
            return False, "The stars image does not match the picture"
        return True, ""

    def _found(self, s):
        key = (id(s.rgb), id(s.stars))
        if getattr(self, "_cache_key", None) != key:
            try:
                self._cache = find_star_leftovers(s.rgb, s.stars)
            except Exception:  # noqa: BLE001 - nothing found is the safe answer
                self._cache = ([], 0)
            self._cache_key = key
        return self._cache

    def is_optional(self, s):
        return not self._found(s)[0]

    def why(self, s):
        boxes, busy = self._found(s)
        if boxes:
            return (f"<b>Cleaning is recommended for this picture.</b> Star removal left {len(boxes)} painted "
                    "square(s) around bright stars - they show once the picture is brightened. Press <b>Do It</b> "
                    "to clean them now, before brightening." + (f" ({busy} inside bright nebula are left for Spot "
                                                                  "Fix, if they show.)" if busy else ""))
        return self.tip

    def variants(self, s):
        boxes, busy = self._found(s)
        note = f"{busy} inside bright nebula left alone" if busy else ""
        if not boxes:
            return [Variant("Nothing to clean - next step", {"boxes": []},
                            "no painted squares found around the bright stars" + (f"; {note}" if note else ""), True)]
        return [Variant(f"Clean the squares ({len(boxes)})", {"boxes": [list(b) for b in boxes]},
                        "each patched, and its square glow made round" + (f"; {note}" if note else ""), True),
                Variant("Don't clean", {"boxes": []}, "leave the picture as it is", False)]

    def apply(self, s, params):
        s = s.copy()
        img = s.rgb
        h, w = img.shape[1:]
        for i, b in enumerate(params.get("boxes") or []):
            if len(b) != 4:
                continue
            fx0, fy0, fx1, fy1 = b
            img = heal_box(img, int(fx0 * w), int(fy0 * h), int(np.ceil(fx1 * w)), int(np.ceil(fy1 * h)), seed=i)
        s.rgb = img.astype(np.float32)
        s.done.append(self.key)
        return s


def _round_star_mask(R: np.ndarray, noise: float, limit: int = 4000) -> np.ndarray:
    """Where the round, compact stars are in R (the detail left after the smooth glow is taken away): 1 on and
    just around each star, 0 elsewhere. Streaks and spikes - long, not round - are left out."""
    h, w = R.shape
    if h < 20 or w < 20:
        return np.zeros((h, w), np.float32)
    p = np.pad(R, 2, mode="edge")
    mx = np.maximum.reduce([p[2 + dy:2 + dy + h, 2 + dx:2 + dx + w] for dy in range(-2, 3) for dx in range(-2, 3)])
    cand = np.argwhere((R >= mx) & (R > 8 * noise))
    if len(cand) > limit:
        cand = cand[np.argsort(-R[cand[:, 0], cand[:, 1]])[:limit]]
    m = np.zeros((h, w), np.float32)
    q = 8
    yy, xx = np.mgrid[-q:q + 1, -q:q + 1]
    for y, x in cand:
        if y < q or x < q or y >= h - q or x >= w - q:
            continue
        win = np.clip(R[y - q:y + q + 1, x - q:x + q + 1], 0, None)
        win = np.where(win > 0.25 * R[y, x], win, 0)
        sw = float(win.sum())
        if sw <= 0:
            continue
        cy, cx = float((win * yy).sum()) / sw, float((win * xx).sum()) / sw
        a = float((win * (yy - cy) ** 2).sum()) / sw
        b = float((win * (xx - cx) ** 2).sum()) / sw
        c = float((win * (yy - cy) * (xx - cx)).sum()) / sw
        t = np.sqrt(((a - b) / 2) ** 2 + c * c)
        l1, l2 = (a + b) / 2 + t, max((a + b) / 2 - t, 1e-6)
        # a star is round AND compact (its light gathered in a few pixels); spike fragments and noise on the glow
        # are broad and flat
        if l1 / l2 < 2.5 and abs(cy) < 1.0 and abs(cx) < 1.0 and l1 < 6.0:
            # the whole star out to where its skirt meets the sky (moving the skirt alone left a ring with a dark
            # middle in the nebula image)
            Q = 3 * q
            y0, y1, x0, x1 = max(0, y - Q), min(h, y + Q + 1), max(0, x - Q), min(w, x + Q + 1)
            dd = np.hypot(*np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x])
            di = dd.astype(int)
            pr = np.bincount(di.ravel(), R[y0:y1, x0:x1].ravel()) / np.maximum(np.bincount(di.ravel()), 1)
            base = float(np.median(pr[max(0, min(len(pr), Q + 1) - 6):Q + 1]))   # the sky / glow around it
            low = np.nonzero(pr[:Q + 1] < base + 2 * noise)[0]
            rr = min(max((float(low[0]) if low.size else float(q)) * 1.3 + 2.0, 4.5 * np.sqrt(l1) + 3.0), float(Q))
            m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], np.clip((rr - dd) / 2.0, 0, 1))
    return np.maximum(m, _blur(m, 1.5))


def rescue_object(rgb: np.ndarray, stars: np.ndarray):
    """Star removal (StarXTerminator) can take a small, bright nebula - a planetary nebula like M57 - for a
    star and put (part of) it in the stars-only image, where no nebula step reaches it and Put Stars Back
    makes it white again. Anything in the central part of the stars image that is much wider than a star
    is moved back into the nebula image. Returns (rgb, stars, share of the stars image's light moved)."""
    lum = stars.mean(axis=0) if stars.shape[0] > 1 else stars[0]
    h, w = lum.shape
    f = max(1, int(np.ceil(max(h, w) / 1500)))
    sm, _f = _small_plane(lum, 1500)
    r = max(3, int(round(12 / f)))                        # wider than ~25 px: not a star
    opened = -_min_filter(-_min_filter(sm, r), r)
    med = float(np.median(sm))
    noise = float(np.median(np.abs(sm - med))) * 1.4826 + 1e-6
    if float(opened.max()) <= med + 5 * noise:
        return rgb, stars, 0.0
    m = opened > max(med + 5 * noise, med + 0.08 * (float(opened.max()) - med))
    sh, sw = m.shape
    box = np.zeros_like(m)
    box[int(sh * 0.3):int(sh * 0.7), int(sw * 0.3):int(sw * 0.7)] = True   # the target sits near the middle
    m &= box
    if not m.any():
        return rgb, stars, 0.0
    # the part to move follows the object's own light, not a box: how bright the (smoothed) stars image is
    # there, faded smoothly to nothing at the sky - inside a round area around the object (a square-shaped
    # weight left a visible square around NGC 7662's core)
    yy, xx = np.nonzero(m)
    cy, cx = float(yy.mean()), float(xx.mean())
    gy, gx = np.mgrid[:sh, :sw]
    dpx = np.hypot(gy - cy, gx - cx)
    # how far the object's glow reaches in the stars image: out to where its average brightness around the
    # centre falls to the sky - star removal also puts the faint wings of a bright core there, and leaving them
    # behind dug a dark rim around NGC 7662's core once the stars were processed
    ri = dpx.astype(int)
    # measured on the opened image: stars around the object must not stretch the reach
    prof = np.bincount(ri.ravel(), opened.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    # out to where the glow is truly down to the sky: a share of a very bright core's peak still shows as a ring
    # once the stars are stretched
    lo = med + max(1.0 * noise, 1e-6 * (float(opened.max()) - med))
    below = np.nonzero(prof[int(np.sqrt(m.sum() / np.pi)):] < lo)[0]
    reach = (int(np.sqrt(m.sum() / np.pi)) + int(below[0])) if below.size else int(np.sqrt(m.sum() / np.pi)) * 3
    rad = max(float(reach) * 1.15, 2.0 * r)
    dist = dpx / rad
    # a long, gentle fade at the edge (a short one left a visible ring where the moved part ended)
    near = np.clip((1.3 - dist) / 0.6, 0, 1) ** 2
    near = near * near * (3 - 2 * near)
    # only smooth glow above the sky moves (below), so the round fade alone decides how much: a weight that
    # also followed the glow's brightness left the faint outer wings behind as a ring
    wgt = np.clip(_resize(near.astype(np.float32),
                          sh * f, sw * f), 0, 1)
    # the bright body itself moves whole, detail and all; only its faint wings are taken without the stars on them
    bright = np.nonzero(prof < med + 0.01 * (float(prof[0]) - med))[0]
    rc = float(bright[0]) if bright.size else rad
    body = np.clip((rc * 1.3 - dpx) / max(rc * 0.3, 1.0), 0, 1).astype(np.float32)
    body = np.clip(_resize(body, sh * f, sw * f), 0, 1)
    fb = np.zeros((h, w), np.float32)
    fb[:min(h, body.shape[0]), :min(w, body.shape[1])] = body[:h, :w]
    full = np.zeros((h, w), np.float32)
    full[:min(h, wgt.shape[0]), :min(w, wgt.shape[1])] = wgt[:h, :w]
    # only the smooth glow moves: stars sitting on the object's wings stay in the stars image (an opening
    # wider than a star takes them out of what is moved)
    # only the glow above the sky moves: moving the sky level too left a disc darker than the sky around it
    sky = np.array([float(np.median(c)) for c in stars], np.float32)[:, None, None]
    # (the smooth glow itself, not capped by each pixel: capping took the noise's peaks and left speckle)
    smooth = np.stack([_blur(_opening(c, 8).astype(np.float32), 2.0) for c in stars])
    # what is left over the glow that is not a round star - the spikes and streaks star removal leaves around a
    # bright core - goes too: left in the stars image they showed as spokes once the stars were brightened
    rest = np.maximum(stars - smooth, 0)
    rl, fr = _small_plane(rest.mean(axis=0), 1500)
    keep = _round_star_mask(rl, noise)
    keep = np.clip(_resize(keep, rl.shape[0] * fr, rl.shape[1] * fr), 0, 1)
    ks = np.ones((h, w), np.float32)
    ks[:min(h, keep.shape[0]), :min(w, keep.shape[1])] = keep[:h, :w]
    smooth = smooth + rest * (1 - ks)[None]
    moved = np.maximum((fb[None] * stars + (1 - fb[None]) * smooth) - sky, 0) * full[None]
    moved = np.minimum(moved, np.maximum(stars, 0))       # never leave the stars image below nothing
    share = float(moved.sum() / max(float(stars.sum()), 1e-9))
    return (rgb + (moved if moved.shape[0] == rgb.shape[0] else moved.mean(0, keepdims=True))).astype(np.float32), \
        (stars - moved).astype(np.float32), share


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
    title = "Brighten the Image (Stretch)"
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
        if s.is_color and target_type(s) == "planetary nebula":
            # small and very bright: a per-colour stretch pushes all three colours to the top together and the
            # nebula turns grey; brightening the luminance and keeping each pixel's colour mix keeps it colourful
            return "colour"
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
        return [Slider("s", "Star brightness", 2.0, 8.0, params["s"], "How bright the stars become", 0.1),
                Slider("small", "Keep bright stars small", 0.0, 1.0, params.get("small", STAR_KEEP_SMALL),
                       "Bright stars keep their real size instead of swelling into big soft discs; faint "
                       "stars are brightened as before")]

    def apply(self, s, params):
        s = s.copy()
        k = 3.0 ** float(params["s"])
        x = np.clip(np.stack([(c - b) for c, b in zip(s.stars, params["bgs"])]) / params["hi"], 0, 1)
        y = (k * x) / ((k - 1) * x + 1)
        keep = float(params.get("small", STAR_KEEP_SMALL))
        if keep > 0:
            # one curve for every pixel lifts a bright star's faint outer glow (0.5 % of its peak) as much as
            # a faint star's core - the star swells into a big soft disc. Around bright stars, brighten the
            # peak and keep the star's own profile instead, so it stays its real size.
            xm = x.max(axis=0)
            ym = y.max(axis=0)
            # each bright star's own area: grown out from its core for as long as the brightened glow is still
            # visible (so the change ends where the glow has already faded - no dark moat, no square), carrying
            # the star's peak with it. Done on a 4x smaller grid, then smoothed.
            f = 4
            h, w_ = xm.shape
            hh, ww = max(f, h // f * f), max(f, w_ // f * f)
            pk = xm[:hh, :ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3))
            allow = ym[:hh, :ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3)) > 0.02
            P = np.where(pk > 0.15, pk, 0.0).astype(np.float32)
            for it in range(int(max(xm.shape) / 25 / f) + 4):            # at most ~4 % of the picture
                q = np.pad(P, 1)
                nb = [q[1:-1, :-2], q[1:-1, 1:-1], q[1:-1, 2:], q[:-2, 1:-1], q[2:, 1:-1]]
                if it % 2 == 0:
                    nb += [q[:-2, :-2], q[:-2, 2:], q[2:, :-2], q[2:, 2:]]
                grown = np.where(allow, np.maximum(P, np.maximum.reduce(nb)), P)
                if np.array_equal(grown, P):
                    break
                P = grown
            P = np.repeat(np.repeat(P, f, axis=0), f, axis=1)
            P = np.pad(P, ((0, h - P.shape[0]), (0, w_ - P.shape[1])), mode="edge")[:h, :w_]
            P = _blur(P.astype(np.float32), 1.5 * f)                       # no 4x4 blocks
            peak = np.maximum(P, xm)
            w = keep * _blur((np.clip((P - 0.15) / 0.45, 0, 1) ** 2).astype(np.float32), 2.0 * f)
            shaped = ((k * peak) / ((k - 1) * peak + 1))[None] * x / np.maximum(peak, 1e-9)[None]
            y = (1 - w)[None] * y + w[None] * shaped
        s.stars = np.clip(y, 0, 1).astype(np.float32)
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
    # one colour per star: spread the estimate from its rim over the whole disc. A big overexposed star
    # (its clipped core wider than the small blur) is filled from further out, step by step; whatever is
    # still far from any usable rim stays white - never a guess (that painted a big star's core pure blue)
    wgt = lum * ok
    est = _blur(x * wgt, 2.5)
    cover = _blur(wgt, 2.5)
    out = np.where(cover > 1e-3, est / np.maximum(cover, 1e-6), 0.0)
    filled = cover > 1e-3
    for rad in (8.0, 24.0, 64.0):
        if filled.all():
            break
        e2, c2 = _blur(x * wgt, rad), _blur(wgt, rad)
        new = ~filled & (c2 > 1e-3)
        out = np.where(new, e2 / np.maximum(c2, 1e-6), out)
        filled |= new
    # one colour per star: a big star gets the colour of its WHOLE light (red total against green + blue total).
    # Measured pixel by pixel, a star whose red sits a pixel or two off its green and blue (colour masters not
    # perfectly on top of each other) came out half orange, half blue-white.
    try:
        out = _one_colour_per_star(out, st_, lum, ok, thr, med, mad, spread)
    except Exception:  # noqa: BLE001 - keep the per-pixel colours
        pass
    # the overexposed core itself: white, fading into the rim's colour (as a real overexposed star looks)
    core_w = np.clip((lum - 0.85) / 0.15, 0, 1)
    out = out * (1.0 - 0.7 * _blur(core_w, 1.5))
    return np.clip(out, -1, 1).astype(np.float32), lum


def _one_colour_per_star(x, st_, lum, ok, thr, med, mad, spread, side=2000, min_area=6):
    """Give every big star (at least min_area pixels of a working copy at most `side` wide) one colour: the
    ratio of its total red light to its total green + blue light, on the same scale as the per-pixel estimate."""
    h, w = lum.shape
    F = max(1, int(np.ceil(max(h, w) / side)))
    hs, ws = h // F, w // F
    if hs < 8 or ws < 8:
        return x

    def small(a):
        return a[:hs * F, :ws * F].reshape(hs, F, ws, F).mean(axis=(1, 3))
    m = small(lum) > thr
    if not m.any():
        return x
    lab = np.where(m, np.arange(1, hs * ws + 1, dtype=np.int64).reshape(hs, ws), 0)
    for _ in range(400):                                     # connected stars: spread the biggest label over each
        grown = SpotFix._minmax(lab.astype(np.float64), 3, np.max).astype(np.int64) * m
        if np.array_equal(grown, lab):
            break
        lab = grown
    ids, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    area = np.bincount(inv.ravel())
    rs = np.bincount(inv.ravel(), weights=small(st_[0] * ok).ravel())
    gbs = np.bincount(inv.ravel(), weights=small(0.5 * (st_[1] + st_[2]) * ok).ravel())
    ratio = np.log((rs + 1e-4 * area) / (gbs + 1e-4 * area))
    xl = np.clip((ratio - med) / (2.2 * mad / max(spread, 0.1)), -1, 1)
    big = (ids > 0) & (area >= min_area) & ((rs + gbs) > 0)
    if not big.any():
        return x
    table = np.where(big, xl, np.nan).astype(np.float32)
    yi = np.minimum(np.arange(h) // F, hs - 1)
    xi = np.minimum(np.arange(w) // F, ws - 1)
    full = table[inv[yi][:, xi]]
    return np.where(np.isfinite(full), full, x).astype(np.float32)


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
        return False                # every colour picture: natural star colours (broadband too - LRGB, OSC)

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
        return [Variant("Natural", {"mode": "natural", "sat": 0.85, "spread": 1.0},
                        "Each star's colour estimated from your data", True),
                Variant("Natural, vivid", {"mode": "natural", "sat": 1.3, "spread": 1.4},
                        "Stronger star colours", False),
                Variant("White", {"mode": "natural", "sat": 0.0, "spread": 1.0},
                        "Clean white stars - the classic narrowband look", False),
                Variant("As they are", {"mode": "keep", "sat": 1.0, "spread": 1.0},
                        "Leave the star colours unchanged", False)]

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
def bad_edges(s: "State", max_frac=0.15):
    """Edges that are ragged or discoloured (e.g. a strip only one filter covered): walk in from
    each side while the line's colour or brightness is clearly off. Full-size (x0, y0, x1, y1)."""
    h, w = s.shape
    sm = s.small(800)
    a = state_display(sm, with_stars=True).astype(np.float32) / 255.0      # (h, w, 3) as seen
    sh, sw = a.shape[:2]
    fy, fx = h / sh, w / sw

    def inward(lines):                       # lines: (n, L, 3) from the edge inward
        med = np.median(lines, axis=1)        # (n, 3)
        n = len(med)
        core = med[n // 4: 3 * n // 4] if n >= 8 else med
        ref = np.median(core, axis=0)
        spread = 1.4826 * np.median(np.abs(core - ref), axis=0) + 0.006
        dev = np.max(np.abs(med - ref) / spread, axis=1)
        cut, good, limit = 0, 0, int(n * max_frac)
        for i in range(min(n, limit + 3)):
            if dev[i] > 6.0:
                cut, good = i + 1, 0
            else:
                good += 1
                if good >= 3:
                    break
        return min(cut + (1 if cut else 0), limit)
    left = inward(a.transpose(1, 0, 2))
    right = inward(a.transpose(1, 0, 2)[::-1])
    top = inward(a)
    bottom = inward(a[::-1])
    return (int(left * fx), int(top * fy), int(w - right * fx), int(h - bottom * fy))


class FinalCrop(Tool):
    key = "final_crop"
    title = "Final Crop"
    tip = ("A last clean-up crop: trims ragged or discoloured edges and frames the picture. The yellow box "
           "starts at the edges SubZeroAstro found - drag it or its sides to adjust.")
    optional = True
    pick = False
    editor = "crop"                  # the window shows a draggable box on the picture

    def applicable(self, s):
        if s.rgb is None and not s.channels:
            return False, "Load masters first"
        return True, ""

    def variants(self, s):
        x0, y0, x1, y1 = bad_edges(s)
        h, w = s.shape
        found = (x0, y0, x1, y1) != (0, 0, w, h)
        note = ("The box starts where the edges look ragged or off-colour" if found else
                "No bad edges found - drag the box to frame the picture")
        return [Variant("Crop to the Box", {"rect": [x0, y0, x1, y1]}, note, True)]

    def apply(self, s, params):
        h, w = s.shape
        x0, y0, x1, y1 = (int(v) for v in params["rect"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError("The crop box is too small")
        s = s._map(lambda a: np.ascontiguousarray(a[:, y0:y1, x0:x1]))
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
def sky_level(y: np.ndarray):
    """Brightness and noise of the true sky: the darkest quarter of the picture in blocks, so a
    nebula that fills most of the frame does not count as 'sky'. (small plane in, floats out)"""
    h, w = y.shape
    b = max(8, min(h, w) // 16)
    hh, ww = h // b * b, w // b * b
    blocks = y[:hh, :ww].reshape(hh // b, b, ww // b, b)
    med = np.median(blocks, axis=(1, 3))
    dark = med <= np.percentile(med, 25)
    px = blocks.transpose(0, 2, 1, 3)[dark].ravel()
    sky = float(np.median(px))
    noise = float(np.median(np.abs(px - sky))) * 1.4826 + 1e-6
    return sky, noise, dark, b


def sky_glow_map(x: np.ndarray, grid: int = 16) -> np.ndarray:
    """A smooth map of the sky glow under the picture (one colour plane), made the way PixInsight's
    ABE / DBE do it: sample the faint end of each area (stars removed), then fit a gently curved
    surface to those samples, throwing out every sample that sticks up above it - that is nebula,
    not sky - and fit again. A nebula can never pull the surface up, so its glow is kept."""
    small, _f = _small_plane(x, 600)
    starless = -_min_filter(-_min_filter(small, 2), 2)
    h, w = small.shape
    b = max(8, max(h, w) // grid)
    gh, gw = max(3, h // b), max(3, w // b)
    hh, ww = gh * b, gw * b
    blocks = starless[:hh, :ww].reshape(gh, b, gw, b).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    g = np.percentile(blocks, 20, axis=2).astype(np.float64).ravel()
    yy, xx = np.mgrid[:gh, :gw]
    u = ((xx + 0.5) / gw * 2 - 1).ravel()
    v = ((yy + 0.5) / gh * 2 - 1).ravel()

    def terms(u, v):
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u * u * u, u * u * v, u * v * v,
                         v * v * v], axis=-1)
    A = terms(u, v)
    keep = g <= np.percentile(g, 60)                      # start from the darker samples
    for _ in range(8):
        coef, *_r = np.linalg.lstsq(A[keep], g[keep], rcond=None)
        res = g - A @ coef
        sd = float(np.median(np.abs(res[keep]))) * 1.4826 + 1e-6
        new = res < 1.0 * sd                              # samples above the surface = nebula
        if new.sum() < 12 or (new == keep).all():
            break
        keep = new
    H, W = x.shape
    sh, sw = max(2, H // 8), max(2, W // 8)              # smooth: work it out small, then enlarge
    uu = ((np.arange(sw) + 0.5) / sw * 2 - 1)[None, :]
    vv = ((np.arange(sh) + 0.5) / sh * 2 - 1)[:, None]
    surf = terms(np.broadcast_to(uu, (sh, sw)).ravel(), np.broadcast_to(vv, (sh, sw)).ravel()) @ coef
    return _resize(np.clip(surf.reshape(sh, sw), 0, None).astype(np.float32), H, W)


def _toe(v: np.ndarray, t: float) -> np.ndarray:
    """Values near black roll off softly instead of clipping (slope 1 above t)."""
    t = max(t, 1e-4)
    return np.where(v >= t, v, t * np.exp((v - t) / t)).astype(np.float32)


class DarkenSky(Tool):
    key = "sky"
    title = "Darken the Sky Background"
    tip = ("Makes the sky darker (or lighter) without dimming the nebula, faint dust or the stars: "
           "SubZeroAstro maps the sky glow across the picture and takes part of it away. Everything "
           "that shines above the sky keeps exactly the brightness it had above it - and brighter "
           "corners lose more, so the sky also evens out.")
    optional = True

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first"
        return True, ""

    def _sky_now(self, s):
        small, _f = _small_plane(_lum(s.rgb), 700)
        return sky_level(small)[0]

    def is_optional(self, s):
        return self._sky_now(s) < 0.13            # recommended only when the sky looks grey

    def why(self, s):
        sky = self._sky_now(s)
        return (f"The sky background is at {sky * 100:.0f}% brightness - a little grey. This takes part of "
                "the sky glow away, keeping the faint dust and stars just as bright above the sky."
                if sky >= 0.13 else self.tip)

    def variants(self, s):
        sky = self._sky_now(s)
        crushed = sky < 0.03

        def note(a):
            return f"sky {sky * 100:.0f}% -> about {max(sky * (1 - a), 0.01) * 100:.0f}%"
        calm = 0.6 if s.kind in NARROWBAND_KINDS else 0.5
        # 'Much darker' unless that would take the sky below about 3% (black, faint dust lost): then 'Darker'
        much = not crushed and sky * (1 - 0.7) >= 0.03
        return [Variant("Slightly darker", {"amount": 0.3, "calm": calm}, note(0.3), False),
                Variant("Darker", {"amount": 0.5, "calm": calm}, note(0.5), not crushed and not much),
                Variant("Much darker", {"amount": 0.7, "calm": calm}, note(0.7), much),
                Variant("Lighter", {"amount": -0.3, "calm": calm},
                        note(-0.3) + (" - the sky looks crushed to black" if crushed else ""), crushed)]

    def sliders(self, s, params):
        return [Slider("amount", "Sky darkness", -0.5, 0.9, params["amount"],
                       "Right = darker sky, left = lighter. What shines above the sky keeps its brightness "
                       "above it."),
                Slider("calm", "Calm sky colours", 0.0, 1.0, params.get("calm", 0.0),
                       "Takes the colour noise (purple, blue or green blotches) out of the darkest areas only; "
                       "the nebula's colours stay")]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        out = np.empty_like(s.rgb)
        for c in range(s.rgb.shape[0]):
            x = s.rgb[c]
            glow = sky_glow_map(x)
            _sky, noise, _d, _b = sky_level(x[::2, ::2])          # per-pixel noise (not averaged)
            v = x - a * glow
            out[c] = _toe(v, 2.5 * noise) if a > 0 else v
        calm = float(params.get("calm", 0.0))
        if calm > 0 and out.shape[0] == 3:
            # weak channels (OIII / SII in narrowband, colour in a short RGB set) leave blotchy colour in the faint
            # sky: pull only the darkest areas toward grey, judged on a smoothed brightness so noise doesn't decide
            y = _lum(out)
            ys = _blur(y.astype(np.float32), 4.0)
            # the sky is everything near the sky's own (smoothed) level - not only the darkest part of it; red or
            # green blotches sit right at that level, so a percentile cut left most of them
            sm_ = ys[::4, ::4]
            sky = float(np.median(sm_))
            sn = float(np.median(np.abs(sm_ - sky))) * 1.4826 + 1e-6
            hi = sky + 4.0 * sn
            w = np.clip((hi - ys) / (hi - sky + 2.0 * sn), 0, 1).astype(np.float32)
            w = np.clip(_blur(w, 6.0) * 1.2, 0, 1)       # soft edge round galaxies and nebulae, no halos
            # in the sky: the colour is smoothed far wider than the blotches (true average tint kept) and pulled
            # toward grey by 'calm'; the brightness and its fine detail are not touched
            chroma = out - y[None]
            smooth = np.stack([_blur(c_.astype(np.float32), 24.0) for c_ in chroma])
            want = (1.0 - calm) * smooth
            out = y[None] + chroma + (w * min(1.0, 0.5 + calm))[None] * (want - chroma)
        s.rgb = np.clip(out, 0, 1).astype(np.float32)
        if self.key not in s.done:
            s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
# Dust donuts: round shadows that dim everything inside a ring by the same fraction
# --------------------------------------------------------------------------
def _opening(a: np.ndarray, r: int) -> np.ndarray:
    """Remove bright points smaller than r (stars) - erosion then dilation."""
    return -_min_filter(-_min_filter(a, r), r)


def donut_profile(plane: np.ndarray, x: float, y: float, r: float, sectors: int = 16, bins: int = 28,
                  harmonics: bool = False):
    """How much light a donut at (x, y) of radius r lets through, ring by ring: (rho, through) with
    rho in units of r and through = 1 outside the shadow. Measured in every direction against the sky
    just outside the circle *in that direction*, then the median over directions - so a nebula or a
    gradient crossing the donut does not fool it. Also returns how many directions agree."""
    h, w = plane.shape
    R = int(r * 1.75) + 4
    x0, x1, y0, y1 = max(0, int(x) - R), min(w, int(x) + R + 1), max(0, int(y) - R), min(h, int(y) + R + 1)
    crop = plane[y0:y1, x0:x1].astype(np.float32)
    if crop.size < 64:
        return None
    st = max(1, int(r / 40))
    small = crop[::st, ::st]
    small = _blur(_opening(small, 2), max(1.0, r / st / 25))
    yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
    yy, xx = yy * st + y0, xx * st + x0
    rho = np.hypot(yy - y, xx - x) / r
    th = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    edges = np.linspace(0, 1.25, bins + 1)
    ref_ring = (rho >= 1.3) & (rho <= 1.7)
    # the sky without the donut: a tilted plane (in log) fitted to the ring just outside, so a gradient or a
    # nebula's edge across the donut is not taken for part of it
    ratio = None
    if ref_ring.sum() >= 20 and float(np.median(small[ref_ring])) > 0:
        lg = np.log(np.maximum(small, 1e-9))
        A = np.stack([np.ones(int(ref_ring.sum())), (xx[ref_ring] - x) / r, (yy[ref_ring] - y) / r], 1)
        cf, *_ = np.linalg.lstsq(A, lg[ref_ring], rcond=None)
        ratio = small / np.exp(cf[0] + cf[1] * (xx - x) / r + cf[2] * (yy - y) / r)
    prof = np.full((sectors, bins), np.nan)
    for k in range(sectors):
        sel = th == k
        ref = small[sel & ref_ring]
        if ref.size < 3 or ratio is None:
            continue
        for i in range(bins):
            v = ratio[sel & (rho >= edges[i]) & (rho < edges[i + 1])]
            if v.size:
                prof[k, i] = float(np.median(v))
    ok_rows = ~np.all(np.isnan(prof), axis=1)
    if ok_rows.sum() < sectors // 2:
        return None
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            through = np.nanmedian(prof[ok_rows], axis=0)
            dark_rows = np.nanmedian(prof[ok_rows][:, :max(1, int(bins * 0.76))], axis=1)
    through = np.where(np.isnan(through), 1.0, through)
    mid = 0.5 * (edges[1:] + edges[:-1])
    dark_dirs = int(np.sum(dark_rows < 1.0))
    if not harmonics:
        return mid, through, dark_dirs, int(ok_rows.sum())
    # a donut is rarely perfectly round in its light: dust that moved between flats and lights leaves a
    # brighter crescent on one side. Per ring, the profile around the circle is fitted with its average
    # and the first two angular waves - enough for an off-centre ring, too smooth to follow a nebula's detail
    ang = (np.arange(sectors) + 0.5) / sectors * 2 * np.pi - np.pi
    basis = np.stack([np.ones(sectors), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], 1)
    coefs = np.zeros((bins, 5))
    coefs[:, 0] = through
    for i in range(bins):
        ok = ~np.isnan(prof[:, i])
        if ok.sum() >= 10:
            coefs[i], *_ = np.linalg.lstsq(basis[ok], prof[ok, i], rcond=None)
    return mid, through, dark_dirs, int(ok_rows.sum()), coefs


def donut_flat(shape, x, y, r, mid, through, coefs=None):
    """The synthetic flat for one donut: 1 everywhere except the donut (smooth) - below 1 where dust
    shadows the light, above 1 where a flat over-corrected it (dust that moved between flats and lights)."""
    t = np.convolve(np.pad(through, 1, mode="edge"), np.ones(3) / 3, mode="valid")     # gentle smoothing
    t = np.clip(t, 0.3, 2.0)        # a shadow (< 1) or a bright over-corrected ring (> 1)
    taper = np.clip((1.18 - mid) / 0.18, 0, 1)                    # back to exactly 1 at the edge
    t = 1.0 - (1.0 - t) * taper
    h, w = shape
    R = int(r * 1.3) + 2
    x0, x1, y0, y1 = max(0, int(x) - R), min(w, int(x) + R + 1), max(0, int(y) - R), min(h, int(y) + R + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rho = np.hypot(yy - y, xx - x) / r
    if coefs is None:
        patch = np.interp(rho, mid, t, left=t[0], right=1.0).astype(np.float32)
        return (y0, y1, x0, x1), patch
    sm = np.stack([np.convolve(np.pad(coefs[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid")
                   for j in range(coefs.shape[1])], 1)
    a = np.arctan2(yy - y, xx - x)
    parts = [np.interp(rho, mid, sm[:, j], left=sm[0, j], right=1.0 if j == 0 else 0.0) for j in range(5)]
    damp = np.clip(rho / 0.45, 0, 1) ** 2          # near the centre the direction means little: round there
    val = parts[0] + damp * (parts[1] * np.cos(a) + parts[2] * np.sin(a) + parts[3] * np.cos(2 * a)
                             + parts[4] * np.sin(2 * a))
    val = np.clip(val, 0.3, 2.0)
    tp = np.clip((1.25 - rho) / 0.15, 0, 1)        # measured out to 1.25 r: fade to 1 over its last part
    return (y0, y1, x0, x1), (1.0 - (1.0 - val) * tp).astype(np.float32)


def _sector_depths(plane, x, y, r, sectors=16):
    """Per direction: how much darker the inside (rho < 0.9) is than the sky just outside - after a
    tilted plane fitted to that outside sky is taken away, so a gradient across the donut (the edge of
    a nebula, light pollution) does not make the directions disagree."""
    h, w = plane.shape
    R = int(r * 1.75) + 3
    x0, x1, y0, y1 = max(0, int(x) - R), min(w, int(x) + R + 1), max(0, int(y) - R), min(h, int(y) + R + 1)
    crop = np.log(np.maximum(plane[y0:y1, x0:x1], 1e-6))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rho = np.hypot(yy - y, xx - x) / r
    th = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    ring = (rho >= 1.3) & (rho <= 1.7)
    if ring.sum() < 20:
        return np.array([])
    A = np.stack([np.ones(int(ring.sum())), (xx[ring] - x) / r, (yy[ring] - y) / r], 1)
    coef, *_ = np.linalg.lstsq(A, crop[ring], rcond=None)
    res = crop - (coef[0] + coef[1] * (xx - x) / r + coef[2] * (yy - y) / r)
    out = []
    for k in range(sectors):
        sel = th == k
        a, b = res[sel & (rho < 0.9)], res[sel & ring]
        if a.size > 2 and b.size > 2:
            out.append(1.0 - float(np.exp(np.median(a) - np.median(b))))
    return np.array(out)


def centre_donut(plane, x, y, r):
    """Nudge a circle (found, or clicked by hand) onto the donut's real centre: the spot where the
    shadow is deepest and most even in every direction. Returns (x, y)."""
    best, bx, by = -1e9, x, y
    for step in (0.12, 0.05):
        cx0, cy0 = bx, by
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                cx, cy = cx0 + dx * step * r, cy0 + dy * step * r
                d = _sector_depths(plane, cx, cy, r)
                if d.size < 12:
                    continue
                med = float(np.median(d))
                sc = abs(med) - 1.5 * float(np.median(np.abs(d - med))) * 1.4826      # dark or bright
                if sc > best:
                    best, bx, by = sc, cx, cy
    return bx, by


def find_donuts(y: np.ndarray, max_n: int = 12, bright: bool = False) -> list:
    """Dust donuts in an unstretched (linear) luminance plane: [[x, y, r], ...] full-size.
    Candidates come from round 'disc' and 'ring' shaped dips at several sizes; one is kept only when
    every direction around it agrees on how much darker the inside is (a dark gap in a nebula does
    not), then its true outer edge is measured."""
    small, f = _small_plane(y, 700)
    base = _opening(small, 2)
    sky, noise, _d, _b = sky_level(small)
    lg = np.log(np.maximum(base, sky * 0.2) + noise).astype(np.float32)
    sg = -1.0 if bright else 1.0                     # bright: over-corrected rings, found as dips of -log
    lg = sg * lg
    h, w = small.shape
    dif = (small[:, 1:] - small[:, :-1]).ravel()                  # pixel-to-pixel noise (gradients cancel)
    pix_noise = float(np.median(np.abs(dif - np.median(dif)))) * 1.4826 / np.sqrt(2) / max(sky, 1e-9)

    def mean_sq(a):                                   # mean over a (2a+1)^2 square, and its area
        a = max(1, int(round(a)))
        return _box_blur(lg, a), float((2 * a + 1) ** 2)

    def band(a_in, a_out):
        mo, ao = mean_sq(a_out)
        mi, ai = mean_sq(a_in)
        return (mo * ao - mi * ai) / max(ao - ai, 1.0)
    found = []
    taken = np.zeros((h, w), dtype=bool)
    for R in (45, 32, 22, 15, 10, 6.5):              # outer radius in small-plane pixels (squares ~ circles)
        if R * 3.4 > min(h, w):
            continue
        outer = band(1.05 * R, 1.45 * R)
        disc = mean_sq(0.72 * R)[0] - outer
        ring = band(0.4 * R, 0.85 * R) - outer
        resp = np.minimum(disc, ring)
        sd = float(np.median(np.abs(resp - np.median(resp)))) * 1.4826 + 1e-9
        e = int(R * 1.5) + 2
        cand = (resp < -4.0 * sd) & (resp <= _min_filter(resp, max(1, int(R * 0.7))) + 1e-12) & ~taken
        cand[:e], cand[-e:], cand[:, :e], cand[:, -e:] = False, False, False, False    # whole circle inside
        ys, xs = np.nonzero(cand)
        for i in np.argsort(resp[ys, xs])[:30]:
            cy, cx = int(ys[i]), int(xs[i])
            if taken[cy, cx]:
                continue
            yl, yh, xl, xh = max(0, int(cy - 1.7 * R)), int(cy + 1.7 * R) + 1, max(0, int(cx - 1.7 * R)), int(cx + 1.7 * R) + 1
            loc = float(np.median(base[yl:yh, xl:xh]))
            if loc > sky * 1.25 + 3 * noise:
                continue                       # inside bright nebulosity: too easy to mistake - click those
            d = sg * _sector_depths(base, cx, cy, R)
            if d.size < 12:
                continue
            med = float(np.median(d))
            spread = float(np.median(np.abs(d - med))) * 1.4826
            chance = 1.25 * pix_noise / np.sqrt(np.pi * (0.9 * R) ** 2)      # noise of the inside median
            if med < max(0.015, 6 * chance) or med > 0.6 or np.mean(d > 0) < 0.94 or spread > 0.35 * med:
                continue
            # the true outer edge: where the shadow has faded, measured outwards from the centre
            # (a bright over-corrected ring is often found at a smaller scale than its full size: look wider)
            reach = 3.0 if bright else 1.5
            res = donut_profile(small, cx, cy, R * reach)
            r_out = R
            if res is not None:
                mid, thr, _dk, _tt = res
                dep = sg * (1 - thr)
                inside = np.nonzero(dep > 0.2 * max(dep.max(), 1e-6))[0]
                if inside.size:
                    r_out = float(np.clip(mid[inside[-1]] * R * reach * 1.08, 0.7 * R, (2.8 if bright else 1.4) * R))
            ex, ey = centre_donut(base, cx, cy, R)
            found.append([ex * f, ey * f, r_out * f, med])
            yy, xx = np.ogrid[:h, :w]
            taken |= (yy - cy) ** 2 + (xx - cx) ** 2 <= (r_out * 1.5) ** 2
            if len(found) >= max_n:
                break
    found.sort(key=lambda d: -d[3])
    return [[float(a), float(b), float(c)] for a, b, c, _dep in found[:max_n]]


class DustDonuts(Tool):
    key = "dust"
    title = "Mote Buster – Fix Dust Donuts"
    tip = ("Dust on the filter or camera window casts round shadows ('donuts') that dim everything inside "
           "them by the same fraction. SubZeroAstro finds them, measures each shadow ring by ring and divides "
           "it out with a synthetic flat made just for that donut - so the nebula underneath comes back "
           "instead of being painted over. Flats are still the best cure.")
    pick = False
    optional = True
    editor = "spots"                 # circles on the picture: click to add, right-click to remove
    repeatable = True                # fix some, look again, fix more
    spot_range = (8.0, 700.0)
    spot_start = 60.0

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if not s.linear:
            return False, "Works on the unstretched picture - use Go Back to before Brighten the Image"
        return True, ""

    def found_spots(self, s):
        # shadows and over-corrected bright rings (dust that moved between the flats and the lights) alike;
        # one circle per donut. Each colour is then measured on its own when the circles are applied.
        lum = _lum(s.rgb)
        out = []
        for br in (False, True):
            for x, y, r in find_donuts(lum, bright=br):
                if all(np.hypot(x - a, y - b) > 0.6 * (r + c) for a, b, c in out):
                    out.append([x, y, r])
        return out[:14]

    def variants(self, s):
        return [Variant("Fix the marked donuts", {"spots": []},
                        "Found donuts are circled. Click to add one (mouse wheel = circle size), right-click "
                        "removes the last; the circle should just cover the donut", True)]

    def apply(self, s, params):
        s = s.copy()
        img = s.rgb.copy()
        # each filter has its own dust - and the same speck sits a little differently in each - so every
        # colour is centred and measured on its own (clicks need not be exact)
        bases = []
        for c in range(img.shape[0]):
            small, f = _small_plane(img[c], 900)
            bases.append((_opening(small, 2), f))
        for x, y, r in params.get("spots", []):
            for c in range(img.shape[0]):
                base, f = bases[c]
                ex, ey = centre_donut(base, x / f, y / f, r / f)
                if np.hypot(ex * f - x, ey * f - y) > 0.3 * r:          # wandered off: keep the click
                    ex, ey = x / f, y / f
                x_c, y_c = ex * f, ey * f
                res = donut_profile(img[c], x_c, y_c, r, harmonics=True)
                if res is None:
                    continue
                mid, thr, _dark, _tot, coefs = res
                (y0, y1, x0, x1), flat = donut_flat(img[c].shape, x_c, y_c, r, mid, thr, coefs)
                img[c, y0:y1, x0:x1] = img[c, y0:y1, x0:x1] / flat
        s.rgb = img.astype(np.float32)
        if self.key not in s.done:
            s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class FinalNoise(Tool):
    key = "denoise"
    title = "Final Noise Reduction (NoiseXTerminator)"
    tip = ("A last, light NoiseXTerminator pass on the finished picture: smooths the grain in the sky and "
           "faint areas (more visible once the sky is darker) while keeping detail.")
    optional = True
    spatial = True                   # choices are 1:1 crops - grain only shows at full size
    runner = None                    # set by the window: callable(img (C,h,w) 0..1) -> denoised (full strength)
    _cache: dict = {}

    def possible(self, s):
        return self.runner is not None

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if self.runner is None:
            return False, "Needs NoiseXTerminator (the RC-Astro CLI, licensed) - see Settings > Your Tools"
        return True, ""

    def variants(self, s):
        return [Variant("Light", {"amount": 0.3}, "a gentle touch - keeps a natural texture", False),
                Variant("Medium", {"amount": 0.5}, "the usual final pass", True),
                Variant("Strong", {"amount": 0.75}, "for very grainy pictures", False)]

    def sliders(self, s, params):
        return [Slider("amount", "Strength", 0.0, 1.0, params["amount"],
                       "How much of NoiseXTerminator's smoothing is used")]

    def _denoised(self, img):
        """NoiseXTerminator at full strength - run once per image, then blended for each strength."""
        sub = img[:, ::max(1, img.shape[1] // 64), ::max(1, img.shape[2] // 64)]
        key = (img.shape, float(sub.sum()), float((sub * sub).sum()))
        if key not in FinalNoise._cache:
            if len(FinalNoise._cache) > 4:
                FinalNoise._cache.pop(next(iter(FinalNoise._cache)))
            FinalNoise._cache[key] = self.runner(img)
        return FinalNoise._cache[key]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        if a > 0:
            den = self._denoised(s.rgb)
            s.rgb = np.clip(s.rgb + a * (den - s.rgb), 0, 1).astype(np.float32)
        if self.key not in s.done:
            s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
import sys as _sys
SIGN_FONTS = ([("Elegant script", "Snell Roundhand", False), ("Handwritten", "Bradley Hand", False),
               ("Classic", "Georgia", True)] if _sys.platform == "darwin" else
              [("Elegant script", "Segoe Script", False), ("Handwritten", "Ink Free", False),
               ("Classic", "Georgia", True)])
SIGN_COLOURS = {"Soft white": (0.95, 0.95, 0.93), "Warm white": (1.0, 0.93, 0.82), "Light grey": (0.78, 0.80, 0.84),
                "Ice blue": (0.74, 0.86, 1.0)}
SIGN_CORNERS = ("bottom right", "bottom left", "top right", "top left")
SIGN_TARGET_FONT = ("Georgia", True)          # the target name above the signature: clean and readable


def signature_alpha(text, family, italic, px):
    """The signature as a coverage mask (h, w) 0..1, drawn with Qt's font engine."""
    from PySide6.QtCore import QPointF as _P
    from PySide6.QtGui import QFont as _F, QFontMetricsF as _FM, QImage as _I, QPainter as _Pa, QColor as _C
    f = _F(family)
    f.setPixelSize(max(6, int(px)))
    f.setItalic(bool(italic))
    fm = _FM(f)
    pad = int(px * 0.6) + 2
    w = int(fm.horizontalAdvance(text) + fm.averageCharWidth() + 2 * pad)
    h = int(fm.height() + 2 * pad)
    img = _I(max(w, 1), max(h, 1), _I.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = _Pa(img)
    p.setRenderHint(_Pa.Antialiasing, True)
    p.setRenderHint(_Pa.TextAntialiasing, True)
    p.setFont(f)
    p.setPen(_C(255, 255, 255))
    p.drawText(_P(pad, pad + fm.ascent()), text)
    p.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())[:, :w * 4]
    a = arr.reshape(h, w, 4)[:, :, 3].astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.01)
    if len(ys) == 0:
        return np.zeros((1, 1), np.float32)
    return np.ascontiguousarray(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


class SignPicture(Tool):
    key = "signature"
    title = "Sign Your Picture"
    tip = ("Adds your signature in a corner of the picture - small, tasteful and in the font you pick. "
           "It becomes part of the saved picture.")
    optional = True
    editor = "signature"             # the window shows: your signature text, corner and colour

    def applicable(self, s):
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first"
        if s.stars_out and not s.stars_hidden:
            return False, "Put the stars back first"
        return True, ""

    def variants(self, s):
        return [Variant(name, {"font": fam, "italic": it}, f"{fam}", i == 0)
                for i, (name, fam, it) in enumerate(SIGN_FONTS)]

    def sliders(self, s, params):
        return [Slider("size", "Size", 1.0, 8.0, params.get("size", 2.8),
                       "Height of the signature, in % of the picture's shorter side", 0.1),
                Slider("opacity", "Opacity", 0.3, 1.0, params.get("opacity", 0.85), "How solid the signature is"),
                Slider("margin", "Distance from edge", 0.5, 8.0, params.get("margin", 2.5),
                       "Gap to the picture's edges, in % of the shorter side", 0.1)]

    @staticmethod
    def corner_focus(shape, corner):
        """Centre of the area around the signature (for zoomed thumbnails)."""
        h, w = shape
        cy = int(h * (0.86 if corner.startswith("bottom") else 0.14))
        cx = int(w * (0.80 if corner.endswith("right") else 0.20))
        return cy, cx

    @staticmethod
    def block(shape, params):
        """The signature (and the target name above it) as one coverage mask, and where it goes:
        (alpha (h, w), x0, y0) in pixels of an image of this shape - or None when there is no text.
        params: text, target_text, font, italic, size, margin, corner, pos (fx, fy of the top-left
        corner as fractions of the picture - set by dragging; overrides the corner)."""
        h, w = shape
        short = min(h, w)
        px = short * float(params.get("size", 2.8)) / 100.0
        text = str(params.get("text") or "").strip()
        target = str(params.get("target_text") or "").strip()
        if not text and not target:
            return None
        lines = []
        if target:
            lines.append(signature_alpha(target, SIGN_TARGET_FONT[0], SIGN_TARGET_FONT[1], px * 0.62))
        if text:
            lines.append(signature_alpha(text, params.get("font", "Segoe Script"), params.get("italic", False), px))
        pos = params.get("pos")
        corner = params.get("corner", "bottom right")
        right = (float(pos[0]) > 0.5) if pos else corner.endswith("right")
        gap = int(px * 0.25)
        bw = max(a.shape[1] for a in lines)
        bh = sum(a.shape[0] for a in lines) + gap * (len(lines) - 1)
        a = np.zeros((bh, bw), np.float32)
        y = 0
        for ln in lines:
            lh, lw = ln.shape
            x = bw - lw if right else 0                      # lines line up on the side nearest the edge
            a[y:y + lh, x:x + lw] = ln
            y += lh + gap
        ah, aw = a.shape
        if ah >= h or aw >= w:                                  # never bigger than the picture
            k = min((h - 2) / ah, (w - 2) / aw)
            a = _resize(a, max(1, int(ah * k)), max(1, int(aw * k)))
            ah, aw = a.shape
        if pos:
            x0, y0 = int(round(float(pos[0]) * w)), int(round(float(pos[1]) * h))
        else:
            m = int(short * float(params.get("margin", 2.5)) / 100.0)
            y0 = h - m - ah if corner.startswith("bottom") else m
            x0 = w - m - aw if corner.endswith("right") else m
        y0, x0 = max(0, min(h - ah, y0)), max(0, min(w - aw, x0))
        return a, x0, y0

    def apply(self, s, params):
        s = s.copy()
        b = self.block(s.shape, params)
        if b is None:
            s.done.append(self.key)
            return s
        a, x0, y0 = b
        ah, aw = a.shape
        px = min(s.shape) * float(params.get("size", 2.8)) / 100.0
        op = float(params.get("opacity", 0.85))
        col = np.asarray(SIGN_COLOURS.get(params.get("colour"), SIGN_COLOURS["Soft white"]), np.float32)
        rgb = s.rgb.copy()
        region = rgb[:, y0:y0 + ah, x0:x0 + aw]
        # a soft shadow keeps the signature readable over bright nebulosity
        sh = _blur(a, max(1.0, px * 0.05))
        dy = dx = max(1, int(px * 0.04))
        shadow = np.zeros_like(a)
        shadow[dy:, dx:] = sh[:ah - dy, :aw - dx]
        region *= (1 - 0.45 * op * shadow)[None]
        c = col[:, None, None] if rgb.shape[0] == 3 else np.asarray([col.mean()], np.float32)[:, None, None]
        region[:] = region * (1 - op * a)[None] + c * (op * a)[None]
        rgb[:, y0:y0 + ah, x0:x0 + aw] = region
        s.rgb = np.clip(rgb, 0, 1).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
class FlipPicture(Tool):
    key = "flip"
    title = "Flip or Rotate the Picture"
    tip = ("Flip Horizontally or Vertically (press again to flip it back), or turn the picture 90° clockwise or "
           "counter-clockwise (each press turns it once more). When it looks right, keep it. The saved pictures "
           "are turned too.")
    optional = True
    pick = False
    editor = "flip"                  # the window shows two toggle buttons

    def applicable(self, s):
        if s.rgb is None and not s.channels:
            return False, "Load masters first"
        return True, ""

    def variants(self, s):
        return [Variant("Keep This Orientation", {"h": False, "v": False, "rot": 0},
                        "Use the buttons above to flip or turn the picture, then keep it", True)]

    @staticmethod
    def describe(params) -> str:
        """'flipped horizontally and turned 90° clockwise' …"""
        rot = int(params.get("rot") or 0) % 4
        bits = [x for x, on in (("flipped horizontally", params.get("h")), ("flipped vertically", params.get("v")))
                if on]
        if rot:
            bits.append({1: "turned 90° clockwise", 2: "turned 180°", 3: "turned 90° counter-clockwise"}[rot])
        return " and ".join(bits)

    def apply(self, s, params):
        h, v = bool(params.get("h")), bool(params.get("v"))
        how = params.get("how")                       # projects saved by v0.9.29-0.9.31
        if how:
            h, v = how in ("lr", "both"), how in ("ud", "both")
        rot = int(params.get("rot") or 0) % 4         # quarter turns clockwise, after any flips

        def fn(a):
            if h:
                a = a[..., ::-1]
            if v:
                a = a[..., ::-1, :]
            if rot:
                a = np.rot90(a, -rot, axes=(-2, -1))  # row 0 is the top: negative = clockwise on screen
            return np.ascontiguousarray(a)
        s = s._map(fn) if (h or v or rot) else s.copy()
        s.done.append(self.key)
        return s


# steps that can safely be done again, in order, under a moved signature (all depend only on their settings)
RESIGN_SAFE = ("flip", "final_crop", "spotfix", "colours", "saturation")


def resign_plan(steps):
    """When the picture is flipped or turned after it was signed, the signature must not turn with it.
    Returns the index of the (last) Sign Your Picture step when the steps after it can be done again
    underneath it, 'blocked' when they cannot, or None when the picture is not signed."""
    k = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].get("tool") == "signature"), None)
    if k is None:
        return None
    if k < 1 or any(st.get("tool") not in RESIGN_SAFE for st in steps[k + 1:]):
        return "blocked"
    return k


def flip_under_signature(project, flip_params):
    """Flip / turn a signed picture with the signature kept the right way round: from the picture as it
    was before signing, the later steps are done again, then the flip, then the same signature on top
    (in the same place on screen). Returns [(state, tool, label, params), …] - the steps that replace
    the signature step and everything after it."""
    steps = project.steps
    k = resign_plan(steps)
    if not isinstance(k, int):
        raise ValueError("the signature cannot be moved over this flip")
    s = project.load(k - 1)
    out = []
    for st in steps[k + 1:]:
        s = TOOLS[st["tool"]].apply(s, st.get("params") or {})
        out.append((s, st["tool"], st["label"], st.get("params") or {}))
    s = TOOLS["flip"].apply(s, flip_params)
    out.append((s, "flip", f"{FlipPicture.title} — {FlipPicture.describe(flip_params)}", flip_params))
    sig = steps[k]
    s = TOOLS["signature"].apply(s, sig.get("params") or {})
    out.append((s, "signature", sig["label"], sig.get("params") or {}))
    return out


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
        # the sky is protected from BOOSTS (they would colour its noise); calming a colour works everywhere,
        # so purple or blue blotches in the sky can be taken down too
        eff = np.where(f > 1, 1 + (f - 1) * ((1 - pr) + pr * w), f)
        s.rgb = np.clip(y[None] + (s.rgb - y[None]) * eff[None], 0, 1).astype(np.float32)
        s.done.append(self.key)
        return s


# --------------------------------------------------------------------------
COLOUR_BANDS = (("reds", 0.0), ("yellows", 60.0), ("greens", 120.0), ("cyans", 180.0), ("blues", 240.0),
                ("magentas", 300.0))


def hue_degrees(rgb: np.ndarray):
    """Hue (0-360, red = 0) and chroma (max - min) of a (3, h, w) image."""
    r, g, b = rgb
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    safe = np.where(c > 1e-9, c, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    return (h * 60.0) % 360.0, c, mx


def smooth_colour(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Blend the colours over a few pixels while keeping each pixel's brightness - abrupt colour edges and colour
    noise soften, detail (which lives in the brightness) stays sharp. amount 0..1; the width follows the
    picture's size, so a preview and the full picture look alike."""
    if amount <= 0 or rgb.shape[0] != 3:
        return rgb
    h, w = rgb.shape[1:]
    sig = max(0.6, amount * max(h, w) / 150.0)           # 1.0: about the width of a planetary's colour zones
    y = rgb.mean(0)
    by = _blur(y.astype(np.float32), sig)
    col = np.stack([_blur(c.astype(np.float32), sig) for c in rgb]) / np.maximum(by, 1e-6)[None]
    out = col * y[None]
    mx = out.max(0)
    out = np.where(mx > 1, out / np.maximum(mx, 1e-6), out)
    return np.clip(out, 0, 1).astype(np.float32)


class AdjustColours(Tool):
    key = "colours"
    title = "Adjust Colours"
    tip = ("Brings out (or tones down) one colour at a time - reds, yellows, greens, cyans, blues, magentas - "
           "without touching the others. Grey and white (the background, white stars) stay as they are. "
           "(PixInsight: ColorSaturation)")
    optional = True

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
        base = {name: 1.0 for name, _h in COLOUR_BANDS}
        # colour smoothing: on for planetary nebulae (their colour zones meet in hard bands), off otherwise
        base.update({"bright": 0.5, "protect": 0.8, "sky": sky,
                     "smooth": 0.4 if target_type(s) == "planetary nebula" else 0.0})
        looks = (("Boost blues", {"blues": 1.7, "cyans": 1.35}),
                 ("Boost reds", {"reds": 1.6, "magentas": 1.2}),
                 ("Blues and reds", {"blues": 1.6, "cyans": 1.3, "reds": 1.4}),
                 ("Boost magentas", {"magentas": 1.7, "reds": 1.15}),
                 ("Calm the greens", {"greens": 0.4, "yellows": 0.8}),
                 ("Calm blues and purples", {"magentas": 0.35, "blues": 0.6, "smooth": 0.5}))
        return [Variant(n, {**base, **ch}, "", i == 2) for i, (n, ch) in enumerate(looks)]

    def sliders(self, s, params):
        out = [Slider(name, name.capitalize(), 0.0, 2.5, params[name],
                      f"How strong the {name} are: below 1 calmer, above 1 stronger", 0.05)
               for name, _h in COLOUR_BANDS]
        out.append(Slider("bright", "Brighten boosted colours", 0.0, 1.0, params["bright"],
                          "Also brighten the colours you boost (faint blue or red nebulosity shows more)", 0.05))
        out.append(Slider("protect", "Protect the sky", 0.0, 1.0, params["protect"],
                          "Keeps the dark background from getting colourful noise"))
        out.append(Slider("smooth", "Smooth colour transitions", 0.0, 1.0, params.get("smooth", 0.0),
                          "Blends the colours where they change abruptly (and calms colour noise); detail stays sharp",
                          0.05))
        return out

    def apply(self, s, params):
        s = s.copy()
        rgb = s.rgb
        hue, c, mx = hue_degrees(rgb)
        f = np.zeros_like(hue)
        for name, centre in COLOUR_BANDS:          # triangular weights 60 degrees wide: they add up to 1
            d = np.abs((hue - centre + 180.0) % 360.0 - 180.0)
            f += np.clip(1.0 - d / 60.0, 0, 1) * float(params.get(name, 1.0))
        y = _lum(rgb)
        pr, sky = float(params.get("protect", 0.8)), float(params.get("sky", 0.0))
        w = np.clip((y - sky * 1.15) / 0.25, 0, 1)
        # the sky is protected from BOOSTS (they would colour its noise); calming a colour works everywhere,
        # so purple or blue blotches in the sky can be taken down too
        eff = np.where(f > 1, 1 + (f - 1) * ((1 - pr) + pr * w), f)
        out = y[None] + (rgb - y[None]) * eff[None]
        br = float(params.get("bright", 0.0))
        if br > 0:                                  # lift the boosted colours a little too
            colourful = np.clip(c / np.maximum(mx, 1e-6), 0, 1)
            out = out * (1 + br * np.clip(eff - 1, 0, None) * colourful * 0.5)[None]
        out = smooth_colour(np.clip(out, 0, 1).astype(np.float32), float(params.get("smooth", 0.0)))
        s.rgb = np.clip(out, 0, 1).astype(np.float32)
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
        sky = float(np.median(_lum(s.rgb)[::3, ::3]))
        return [Variant(n, {"amount": a, "grid": 8, "clip": 2.5, "protect": 0.85, "sky": sky}, "", i == 1)
                for i, (n, a) in
                enumerate((("Subtle", 0.15), ("Moderate", 0.3), ("Strong", 0.45), ("Very strong", 0.65)))]

    def sliders(self, s, params):
        return [Slider("amount", "Strength", 0.0, 1.0, params["amount"], "How much local contrast to add"),
                Slider("grid", "Detail size", 4, 16, params["grid"], "Large structures (4) <-> fine detail (16)", 1),
                Slider("protect", "Protect the sky", 0.0, 1.0, params.get("protect", 0.85),
                       "Leaves the empty background alone - boosting it only shows faint blotches of leftover "
                       "noise (they can look like a moire pattern)")]

    def apply(self, s, params):
        s = s.copy()
        a = float(params["amount"])
        pr = float(params.get("protect", 0.0))
        sky = params.get("sky")
        eq = None

        def fn(y):
            nonlocal eq
            eq = clahe(np.clip(y, 0, 1), int(round(params["grid"])), float(params["clip"]))
            if pr <= 0:
                return (1 - a) * y + a * eq
            # only where there is something above the sky (judged on a smoothed copy, so noise doesn't decide)
            sk = float(np.median(y[::3, ::3])) if sky is None else float(sky)
            w = np.clip((_blur(y.astype(np.float32), 3.0) - sk * 1.1 - 0.01) / 0.15, 0, 1)
            aa = a * ((1 - pr) + pr * w)
            return (1 - aa) * y + aa * eq
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


def is_galaxy_target(name: str) -> bool:
    """A galaxy by its name (catalogue number or common name) - see targets.py."""
    import targets
    return targets.is_galaxy(name)


class HaGlow(Tool):
    key = "ha_glow"
    title = "Add Hydrogen Glow"
    tip = ("Blends your Ha image into the red channel so nebulae glow in a natural-colour picture. The "
           "starlight is removed from the Ha first (continuum subtraction), so stars don't turn pink.")
    has_mask = True
    mask_label = "Show the hydrogen regions (red)"
    mask_tip = "Tints in red exactly what this step adds to the picture"

    def applicable(self, s):
        if s.rgb is not None and "ha_glow" in s.skipped:
            return False, ("The colour channels are already combined - the hydrogen glow is added before that. "
                           "Use Back to Start (or pick 'Start (masters loaded)' in Show and Go Back to Here), "
                           "then Add Hydrogen Glow, then Combine Colour Channels.")
        if s.rgb is not None or not {"Ha", "R", "G", "B"} <= set(s.channels):
            return False, "Needs separate R, G, B and Ha images, before combining"
        return True, ""

    def possible(self, s):
        return s.rgb is None and {"Ha", "R", "G", "B"} <= set(s.channels)

    def why(self, s):
        if is_galaxy_target(s.target):
            return ("A galaxy with Ha: the hydrogen regions (HII) - the pink-red knots along the spiral arms - are "
                    "added to red on their own, so they stand out crisply against the blue arms without tinting "
                    "the whole galaxy pink.")
        return self.tip

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
        note = f"starlight factor {c:.2f} removed from Ha"
        if is_galaxy_target(s.target):
            return [Variant("HII regions - natural", {**base, "k": 1.2, "mode": "knots"},
                            "only the compact hydrogen knots of the galaxy; " + note, True),
                    Variant("HII regions - strong", {**base, "k": 2.2, "mode": "knots"},
                            "the knots stand out boldly", False),
                    Variant("HII regions - very strong", {**base, "k": 3.5, "mode": "knots"},
                            "show-piece: every knot vivid", False),
                    Variant("All hydrogen glow", {**base, "k": 1.0, "mode": "all"},
                            "also any smooth Ha glow (for galaxies inside nebulosity)", False)]
        return [Variant(n, {**base, "k": k}, note, i == 1) for i, (n, k) in
                enumerate((("Subtle", 0.5), ("Natural", 1.0), ("Strong", 1.6), ("Very strong", 2.5)))]

    def sliders(self, s, params):
        return [Slider("k", "Glow strength", 0.0, 5.0, params["k"], "How much hydrogen glow to add to red", 0.05)]

    def line(self, s, params) -> np.ndarray:
        """The hydrogen emission alone (Ha without starlight) - and for 'knots' only the compact HII regions
        clearly above the noise: the galaxy's smooth glow and any leftover sky are left out."""
        ch = s.channels
        c = float(params["c"])
        line = np.clip((ch["Ha"] - params["bg_h"]) - c * (ch["R"] - params["bg_r"]), 0, None).astype(np.float32)
        if params.get("mode") != "knots":
            return line
        sm = _blur(line, 1.0)
        h, w = sm.shape
        r = int(max(4, min(24, max(h, w) / 160)))           # HII knots: up to about this radius
        small, f = _small_plane(sm, 1200)
        rr = max(2, int(round(r / f)))
        base_small = -_min_filter(-_min_filter(small, rr), rr)   # the smooth part (opening)
        base_small = _blur(base_small, rr)
        knots = np.clip(sm - _resize(base_small, h, w), 0, None)
        med, noise = _stats(line)
        nz = noise / 2.5                                        # noise after the light smoothing
        sig = np.clip((knots - 2.5 * nz) / (2.5 * nz), 0, 1)   # only what clearly stands out
        return (knots * sig * sig * (3 - 2 * sig)).astype(np.float32)

    def preview(self, s, params):
        ch = s.channels
        base = np.stack([ch["R"], ch["G"], ch["B"]])
        r = self.apply(s, params).channels
        img = display_fixed(np.stack([r["R"], r["G"], r["B"]]), base)
        if params.get("show_mask"):
            ln = self.line(s, params)
            top = float(np.percentile(ln[ln > 0], 99)) if (ln > 0).any() else 1.0
            m = np.clip(ln / max(top, 1e-9), 0, 1) ** 0.5
            if m.shape != img.shape[:2]:
                m = _resize(m, *img.shape[:2])
            tint = np.array([255, 40, 60], np.float32)
            img = (img * (1 - 0.6 * m[..., None]) + tint * 0.6 * m[..., None]).astype(np.uint8)
        return img

    def apply(self, s, params):
        s = s.copy()
        ch = dict(s.channels)
        c, k = float(params["c"]), float(params["k"])
        line = self.line(s, params)
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
    editor = "spots"
    spot_range = (2.0, 700.0)        # same reach as Mote Buster: big satellite blobs and halos too
    spot_start = 10.0
    title = "Spot Fix"
    tip = ("Click on a dust donut, a satellite remnant or a hot pixel and it is patched from its "
           "surroundings. Stars inside the circle are kept - only the thing at its middle is removed. "
           "(PixInsight: CloneStamp)")
    pick = False
    optional = True
    repeatable = True

    def applicable(self, s):
        return (True, "") if s.rgb is not None else (False, "Combine the colour channels first")

    def variants(self, s):
        return [Variant("Fix the marked spots", {"spots": []}, "Click spots on the image (right-click removes "
                        "the last one); the mouse wheel changes the circle size", True)]

    @staticmethod
    def _minmax(a: np.ndarray, size: int, fn) -> np.ndarray:
        """Square min or max filter (separable, edges padded)."""
        k = size // 2
        from numpy.lib.stride_tricks import sliding_window_view as _sw
        p = np.pad(a, ((0, 0), (k, k)), mode="edge")
        a = fn(_sw(p, size, axis=1), axis=-1)
        p = np.pad(a, ((k, k), (0, 0)), mode="edge")
        return fn(_sw(p, size, axis=0), axis=-1)

    @staticmethod
    def _grain(patch: np.ndarray, where: np.ndarray) -> float:
        """Pixel-to-pixel noise in an area (so slopes and nebula edges are not counted as noise)."""
        dx = patch[:, 1:] - patch[:, :-1]
        pair = where[:, 1:] & where[:, :-1]
        if pair.sum() > 8:
            dv = dx[pair]
            return float(1.4826 * np.median(np.abs(dv - np.median(dv))) / np.sqrt(2))
        return float(np.std(patch[where] - np.median(patch[where])))

    @staticmethod
    def star_layer(lum: np.ndarray, d: np.ndarray, r: float):
        """Which pixels of a patch belong to stars to keep: small bright points well above the
        smooth background (a top-hat), except the thing at the middle of the circle - that is
        what was clicked, and it goes. Returns (keep mask 0..1, star element size)."""
        size = int(np.clip(r * 0.2, 2, 15)) * 2 + 1                   # 5 … 31 px
        if min(lum.shape) <= size * 2:
            return np.zeros_like(lum, np.float32), size
        bg = SpotFix._minmax(SpotFix._minmax(lum, size, np.min), size, np.max)
        sig = lum - bg
        # noise and the filter's own lift, measured on the sky just outside the circle
        ring = (d > r) & (d <= r * 1.6 + 2)
        if ring.sum() < 50:
            ring = np.ones_like(lum, bool)
        sd = SpotFix._grain(lum, ring)
        lift = float(np.median(sig[ring]))
        core = sig > lift + max(5.0 * sd, 1e-4)
        if not core.any():
            return np.zeros_like(lum, np.float32), size
        n3 = _box_blur(core.astype(np.float32), 1) * 9                 # no lone hot pixels
        core &= n3 >= 2.5
        # the clicked object: whatever star-like pixels touch the middle of the circle
        centre = core & (d <= max(1.5, min(0.35 * r, 10.0)))   # a big donut keeps the stars near its middle too
        for _ in range(size * 2):
            grown = (SpotFix._minmax(centre.astype(np.float32), 3, np.max) > 0) & core
            if (grown == centre).all():
                break
            centre = grown
        keep = core & ~centre
        if not keep.any():
            return np.zeros_like(lum, np.float32), size
        keep = SpotFix._minmax(keep.astype(np.float32), 5, np.max)        # include the faint edges
        keep = np.clip(_box_blur(keep, 1), 0, 1)
        return keep.astype(np.float32), size

    @staticmethod
    def heal(img: np.ndarray, x, y, r, seed=0, keep_stars=True) -> np.ndarray:
        """Patch a circle from its surroundings. keep_stars: stars inside the circle (other than the
        thing at its middle) are lifted off first and put back on top of the patch afterwards,
        brightened by as much as the blemish had dimmed them."""
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
        keep = None
        if keep_stars:
            lum = out[:, y0:y1, x0:x1].astype(np.float64).mean(0)
            keep, size = SpotFix.star_layer(lum, d, r)
            if not keep.any():
                keep = None
        for k in range(c):
            patch = out[k, y0:y1, x0:x1].astype(np.float64)
            stars = None
            base = patch
            if keep is not None:
                smooth = SpotFix._minmax(SpotFix._minmax(patch, size, np.min), size, np.max)
                stars = np.clip(patch - smooth, 0, None) * keep
                base = patch - stars                                       # the sky without its stars
            fill = base.copy()
            fill[mask] = np.median(base[ring])
            fill = SpotFix._harmonic(fill, mask)   # smooth fill from the edge inwards, at any size
            # the grain to put back: from neighbour-to-neighbour differences in the ring, so a sky
            # gradient or nebula edge around a big circle is not mistaken for noise
            sd = SpotFix._grain(patch, ring)
            fill[mask] += rng.normal(0, sd * 0.8, int(mask.sum()))
            if stars is not None:
                # a dust shadow dims the stars in it as much as the sky: brighten them back by as much
                rad = max(3, size)
                gain = np.clip(_box_blur(fill.astype(np.float32), rad) /
                               np.maximum(_box_blur(base.astype(np.float32), rad), 1e-6), 0.5, 2.5)
                fill = fill + stars * gain
            res = patch.copy()
            res[mask] = fill[mask]
            out[k, y0:y1, x0:x1] = res.astype(out.dtype)
        return out

    @staticmethod
    def _harmonic(fill: np.ndarray, mask: np.ndarray, iters: int = 80) -> np.ndarray:
        """Smoothly fill the masked pixels from their surroundings. Big holes are solved on a
        coarser copy first (coarse to fine), so the middle of a large circle blends in too,
        not just a band near its edge."""
        h, w = fill.shape
        if min(h, w) >= 48 and mask.any():
            ys, xs = np.nonzero(mask)
            if max(np.ptp(ys), np.ptp(xs)) > 30:
                h2, w2 = h // 2 * 2, w // 2 * 2
                small = fill[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean((1, 3))
                smask = mask[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).any((1, 3))
                small = SpotFix._harmonic(small, smask, iters)
                up = np.repeat(np.repeat(small, 2, 0), 2, 1)
                guess = fill.copy()
                guess[:h2, :w2] = up
                fill = fill.copy()
                fill[mask] = guess[mask]
        for _ in range(iters):
            avg = (np.roll(fill, 1, 0) + np.roll(fill, -1, 0) + np.roll(fill, 1, 1) + np.roll(fill, -1, 1)) / 4
            fill[mask] = avg[mask]
        return fill

    def apply(self, s, params):
        s = s.copy()
        img = s.rgb
        keep = bool(params.get("keep_stars", True))
        for i, (x, y, r) in enumerate(params.get("spots", [])):
            img = self.heal(img, x, y, r, seed=i, keep_stars=keep)
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
    return bool(_bright_blob(_lum(s.rgb), 0.90).sum() >= 12)


def burnt_area(s: "State") -> float:
    """How much of the picture is a white, featureless area (stars not counted): 0..1."""
    if s.rgb is None or s.linear:
        return 0.0
    b = _bright_blob(_lum(s.rgb), 0.90)
    return float(b.sum()) / max(b.size, 1)


BIG_CORE = 0.0025       # a burnt area bigger than this (about 30 x 30 px of a 600 px view): 'Strong' by default


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
        if self.ring_target(s):
            return ("A planetary nebula: its ring is far brighter than everything around it, so it turns white and "
                    "loses its colour. This lowers the ring below white and leaves the faint centre as it is - check "
                    "with 'Show what will change' that the green tint covers the whole ring.")
        if s.sat_core:
            return ("The core looks burnt out - and it was already saturated in the masters, so its detail "
                    "can't be recovered; this makes it look smooth and natural instead of a flat white disc.")
        return ("The core looks burnt out (pure white) - bring its detail back without touching the rest. Best done "
                "now, before Curves and Saturation brighten it further.")

    @staticmethod
    def ring_target(s) -> bool:
        """A planetary nebula: its bright part is a ring or shell around a faint centre."""
        try:
            return target_type(s) == "planetary nebula"
        except Exception:  # noqa: BLE001
            return False

    def variants(self, s):
        if self.ring_target(s):
            # Tame the Ring: lower everything brighter than the ring level, leave the faint centre as it is
            return [Variant(n, {"amount": 1.0, "size": 0.5, "detail": 30.0, "ring": k},
                            "the nebula brightened again, gently: ring below white, room for colour", i == 1)
                    for i, (n, k) in enumerate((("Gentle", 0.85), ("Medium", 0.75), ("Strong", 0.65),
                                                ("Maximum", 0.55)))]
        note = "core saturated in the masters: smooths rather than recovers" if s.sat_core else ""
        # a big burnt area (the bright heart of a nebula, a large galaxy core) needs 'Strong' to show its
        # detail; a small one (a galaxy nucleus) is fine with 'Medium'
        pick = 2 if burnt_area(s) >= BIG_CORE else 1
        # stronger versions also take in more of the area around the core, so a peaked galaxy nucleus
        # fades smoothly into the tamed core instead of staying a small bright dot
        return [Variant(n, {"amount": a, "size": sz, "detail": 30.0}, note, i == pick) for i, (n, a, sz) in
                enumerate((("Gentle", 0.45, 0.82), ("Medium", 0.7, 0.72), ("Strong", 0.9, 0.62),
                           ("Maximum", 1.0, 0.55)))]

    def sliders(self, s, params):
        if params.get("ring"):
            return [Slider("ring", "Ring brightness", 0.35, 0.95, params["ring"],
                           "How bright the brightest part of the ring becomes (left = darker, more room for colour)"),
                    Slider("size", "Area", 0.2, 0.95, params["size"],
                           "Bigger area (left) <-> only the very brightest part (right) - make sure the tint covers "
                           "the whole ring"),
                    Slider("amount", "Strength", 0.0, 1.0, params["amount"], "How much of the change is used")]
        return [Slider("amount", "Strength", 0.0, 1.0, params["amount"], "How much the core is toned down"),
                Slider("size", "Core size", 0.2, 0.95, params["size"],
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
        if params.get("ring"):
            # a ring: grow the area past the ring's outer edge (a third of its radius), so the whole ring is
            # well inside it and the feathered edge falls on the faint halo, not on the ring
            core = m > 0.5
            if core.any():
                rc = float(np.sqrt(core.sum() / np.pi))
                g = max(1, int(round(rc / 4)))
                m = np.maximum(m, (-_min_filter(-core.astype(np.float32), g) > 0.5).astype(np.float32))
        m = _blur(m, sig)                                        # feather the edge
        return np.clip(_resize(m, *y.shape), 0, 1)

    GLOW_DIV = 6.0             # the glow is blurred at the core radius / this
    PEAK_ROOM = 0.3            # the glow peak lands this share of the way from the edge brightness to white
    DETAIL_CAP = 0.08          # structure brighter than this above the lowered glow is squeezed (smoothly)
    SHOULDER = (0.80, 0.15)    # roll-off: starts here, approaches start + this
    RING_CURVE = 6.0           # ring mode: how much the bright end is compressed (log curve)
    TOP_SOFTEN = 1.5           # a clipped flat top is rounded off with a blur of its radius x this
    CORE_FLATTEN = 0.85        # share of the core's glow (above its edge's brightness) taken away at full strength

    def _target(self, s, params, m):
        """The core re-made like HDRMultiscaleTransform: its structure comes from the linear data when
        kept (so detail lost in a white core comes back), then the core's big smooth glow is lowered
        towards the brightness around it, with the fine structure kept - and lifted a little - on top.
        The core may end up darker than its edge, which is what makes the detail show."""
        y = _lum(s.rgb)
        c = float(params["detail"])
        inside = m > 0.5
        edge = (m > 0.25) & (m < 0.75)
        if not inside.any():
            return y
        src = y
        if s.lin_lum is not None and s.lin_lum.shape == y.shape:
            # how the picture was brightened around the core (picture ~ a * linear + b where it is not
            # white yet), used to re-make the white pixels from the linear data at their true scale -
            # above 1 at first; lowering the glow below brings them back into range with their detail
            n = s.lin_lum
            near = (m > 0.15) & (y > 0.15) & (y < 0.93)
            fit = None
            if near.sum() > 200:
                nn, yy_ = n[near].astype(np.float64), y[near].astype(np.float64)
                if float(np.std(nn)) > 1e-9:
                    a_, b_ = np.polyfit(nn, yy_, 1)
                    if a_ > 0:
                        fit = (a_, b_)
            if fit is not None:
                re_made = fit[0] * n + fit[1]
                white = np.clip((y - 0.9) / 0.08, 0, 1)                # blend in where the picture is white
                src = (y * (1 - white) + np.maximum(re_made, y) * white).astype(np.float32)
            else:
                n_e = float(np.median(n[edge])) if edge.any() else float(np.percentile(n[inside], 5))
                n_max = float(np.percentile(n[inside], 99.9))
                e = float(np.median(y[edge])) if edge.any() else float(np.percentile(y[inside], 5))
                top = 0.93
                u = np.clip((n - n_e) / max(n_max - n_e, 1e-9), 0, None)
                rec = e + (top - e) * np.log1p(c * u) / np.log1p(c)
                src = np.where(n > n_e, np.minimum(rec, 1.0), y).astype(np.float32)
        ring_top = float(params.get("ring", 0) or 0)
        if ring_top > 0:
            # a ring or shell (planetary nebula): the nebula inside the mask is brightened again, more gently,
            # from the linear data - its edge keeps the brightness it has now, its brightest part goes to
            # ring_top, and everything between follows the real brightness of the nebula (a log curve)
            n = s.lin_lum if (s.lin_lum is not None and s.lin_lum.shape == y.shape) else y
            e_y = float(np.median(y[edge])) if edge.any() else float(np.percentile(y[inside], 5))
            e_n = float(np.median(n[edge])) if edge.any() else float(np.percentile(n[inside], 5))
            n_top = float(np.percentile(n[inside], 99.9))
            top = max(ring_top, e_y + 0.05)
            if n_top <= e_n:
                return y
            u = np.clip((n - e_n) / (n_top - e_n), 0, None)
            k = self.RING_CURVE
            v = e_y + (top - e_y) * np.log1p(k * u) / np.log1p(k)
            v = np.where(n > e_n, v, y)
            v = np.where(v > 0.92, 0.92 + 0.06 * np.tanh((v - 0.92) / 0.06), v)   # never pure white
            # only the bright part is lowered: the faint glow around a very bright core (NGC 7662) has a tiny
            # share of the linear range and would otherwise be flattened to the edge level - a dark moat
            q = max(e_y + 0.1, 0.55)
            wb = np.clip((y - q) / max(0.9 - q, 1e-3), 0, 1) ** 1.5
            v = y * (1 - wb) + np.minimum(v, y) * wb
            return np.clip(v, 0, 1).astype(np.float32)
        # a nucleus burnt out even in the masters is a flat disc with a hard rim: round it off into a soft
        # peak (no detail is lost - there is none in a clipped area)
        flat_top = (src >= float(src[inside].max()) - 0.004) & inside
        if s.lin_lum is not None and s.lin_lum.shape == y.shape:
            n_ = s.lin_lum
            flat_top |= (n_ >= float(n_[inside].max()) * 0.995) & inside
        if 8 < flat_top.sum() < 0.15 * inside.sum():      # a small burnt-out nucleus - not a whole clipped ring
            rp = float(np.sqrt(flat_top.sum() / np.pi))
            sig = max(1.5, rp * self.TOP_SOFTEN)
            wgt = np.clip(_blur(flat_top.astype(np.float32), sig) * 2.0, 0, 1)
            src = (src * (1 - wgt) + _blur(src.astype(np.float32), sig) * wgt).astype(np.float32)
        e = float(np.median(src[edge])) if edge.any() else float(np.percentile(src[inside], 5))
        # the core's glow: a blur on the scale of the core itself
        rc = float(np.sqrt(inside.sum() / np.pi))
        small, f = _small_plane(src, 700)
        glow = _resize(_blur(small, max(1.0, rc / self.GLOW_DIV / f)), *y.shape)
        detail = src - glow
        # keep this much of the glow above the edge: at most 15%, and less when the brightest part
        # would still end up near white (a strongly brightened or re-made core)
        g_top = float(glow[inside].max())                      # the true peak (a small nucleus is a tiny share)
        room = (0.97 - e) * self.PEAK_ROOM                      # the glow peak lands this share of the way to white
        keep = 1 - self.CORE_FLATTEN
        if g_top > e:
            keep = min(keep, max(room, 0.0) / (g_top - e))
        flat = np.where(glow > e, e + (glow - e) * keep, glow)
        # structure re-made at a bigger scale is brought down with the glow, but never below 30%
        d_scale = float(np.clip(keep / (1 - self.CORE_FLATTEN), 0.3, 1.0))
        gain = (1.0 + 0.5 * min(1.0, c / 60.0)) * d_scale      # the Detail slider lifts the structure too
        # the structure on top of the lowered glow - but a lone very bright spot (the centre, often
        # burnt out in the masters too) is squeezed smoothly, so no bright dot with hard edges is left
        dl = detail * gain
        dl = np.where(dl > 0, self.DETAIL_CAP * np.tanh(dl / self.DETAIL_CAP), dl)   # dark detail (dust lanes) untouched
        v = flat + dl
        # a soft shoulder instead of a hard clip: the brightest structure rolls off gently below white
        s0, s1 = self.SHOULDER
        v = np.where(v > s0, s0 + s1 * np.tanh((v - s0) / s1), v)
        return np.clip(v, 0, 1).astype(np.float32)

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
            m2 = self.mask(s, params)
            m = m2[..., None]
            tint = np.array([40, 230, 110], np.float32)           # green: shows on red nebulae too
            img = img * (1 - 0.18 * m) + tint * 0.18 * m           # light, so the result shows through
            edge = (m2 > 0.4) & (m2 < 0.6)                         # a yellow line along the mask's edge
            img[edge] = np.array([255, 225, 40], np.float32)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

# --------------------------------------------------------------------------
# Comets: find the nucleus, reveal jets (Larson-Sekanina) and the tail (take away the round coma)

def comet_nucleus(y: np.ndarray):
    """Where the comet's head is, as fractions (fx, fy) of the width and height. Stars are taken away
    first (an opening keeps only blobs bigger than a star), then the brightest smooth spot is found and
    refined at full size. None when nothing comet-like stands out."""
    sm, f = _small_plane(y.astype(np.float32), 700)
    blob = _blur(_opening(sm, 3), 3.0)
    sky = float(np.median(blob))
    noise = float(np.median(np.abs(blob - sky))) * 1.4826 + 1e-6
    iy, ix = np.unravel_index(int(np.argmax(blob)), blob.shape)
    if (blob[iy, ix] - sky) / noise < 8:
        return None
    # refine on the full-size picture, in a small window around the coarse spot
    h, w = y.shape
    cy, cx = int((iy + 0.5) * f), int((ix + 0.5) * f)
    r = max(12, 3 * f)
    y0, y1, x0, x1 = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
    raw = y[y0:y1, x0:x1].astype(np.float32)
    if min(raw.shape) < 7:
        return (cx + 0.5) / w, (cy + 0.5) / h
    # a 3x3 median takes out noise without smearing the head toward the tail (a blur would)
    win = np.median(np.stack([np.roll(np.roll(raw, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), 0)
    inner = win[2:-2, 2:-2]
    jy, jx = np.unravel_index(int(np.argmax(inner)), inner.shape)
    jy, jx = jy + 2, jx + 2

    def _sub(a, b, c):                       # parabola through three samples: where its top is
        d = a - 2 * b + c
        return 0.0 if d == 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    px = jx + _sub(win[jy, jx - 1], win[jy, jx], win[jy, jx + 1])
    py = jy + _sub(win[jy - 1, jx], win[jy, jx], win[jy + 1, jx])
    return (x0 + px + 0.5) / w, (y0 + py + 0.5) / h


def _rotate_about(a: np.ndarray, cx: float, cy: float, deg: float) -> np.ndarray:
    """The plane turned by deg degrees around (cx, cy) - bilinear, in row bands to keep memory low."""
    h, w = a.shape
    t = np.deg2rad(deg)
    c, sn = np.float32(np.cos(t)), np.float32(np.sin(t))
    out = np.empty_like(a, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32) - np.float32(cx)
    band = max(1, int(4e6 // max(w, 1)))
    for r0 in range(0, h, band):
        ys = np.arange(r0, min(h, r0 + band), dtype=np.float32)[:, None] - np.float32(cy)
        sx = c * xs[None, :] + sn * ys + np.float32(cx)          # where each output pixel comes from
        sy = -sn * xs[None, :] + c * ys + np.float32(cy)
        sx = np.clip(sx, 0, w - 1.001)
        sy = np.clip(sy, 0, h - 1.001)
        x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
        fx, fy = sx - x0, sy - y0
        out[r0:r0 + band] = ((a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx) * (1 - fy)
                             + (a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx) * fy)
    return out


def larson_sekanina(y: np.ndarray, cx: float, cy: float, angle: float) -> np.ndarray:
    """Larson-Sekanina rotational gradient: 2*I - I(turned +a) - I(turned -a) around the nucleus.
    Anything round about the nucleus (the coma) cancels out; jets, shells and tail rays remain."""
    return 2 * y - _rotate_about(y, cx, cy, angle) - _rotate_about(y, cx, cy, -angle)


def radial_profile_model(y: np.ndarray, cx: float, cy: float, bins: int = 400) -> np.ndarray:
    """The round part of the coma: at each distance from the nucleus, the median brightness all the way
    round (a tail covers only a few directions, so the median ignores it)."""
    h, w = y.shape
    sm, f = _small_plane(y.astype(np.float32), 900)
    hh, ww = sm.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    rr = np.hypot((xx + 0.5) * f - cx, (yy + 0.5) * f - cy)
    rmax = float(rr.max()) + 1
    edges = np.linspace(0, np.sqrt(rmax), bins + 1) ** 2               # finer steps near the nucleus
    idx = np.clip(np.searchsorted(edges, rr.ravel()) - 1, 0, bins - 1)
    vals = sm.ravel()
    order = np.argsort(idx, kind="stable")
    sidx, sval = idx[order], vals[order]
    starts = np.searchsorted(sidx, np.arange(bins))
    ends = np.searchsorted(sidx, np.arange(bins), side="right")
    prof = np.array([np.median(sval[a:b]) if b > a else np.nan for a, b in zip(starts, ends)], np.float32)
    good = ~np.isnan(prof)
    mids = (edges[:-1] + edges[1:]) / 2
    prof = np.interp(mids, mids[good], prof[good]).astype(np.float32)
    # keep it falling outward (a median ring can wobble)
    prof = np.minimum.accumulate(prof)
    ry = (np.arange(h, dtype=np.float32) + 0.5)[:, None] - cy
    out = np.empty((h, w), np.float32)
    xs2 = ((np.arange(w, dtype=np.float32) + 0.5) - cx) ** 2
    band = max(1, int(4e6 // max(w, 1)))
    for r0 in range(0, h, band):
        rad = np.sqrt(ry[r0:r0 + band] ** 2 + xs2[None, :])
        out[r0:r0 + band] = np.interp(rad, mids, prof)
    return out


def _shift_plane(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The plane moved by (dx, dy) pixels (bilinear; the edges repeat)."""
    h, w = a.shape
    xs = np.clip(np.arange(w, dtype=np.float32) - dx, 0, w - 1.001)
    ys = np.clip(np.arange(h, dtype=np.float32) - dy, 0, h - 1.001)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = (xs - x0)[None, :], (ys - y0)[:, None]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x0 + 1] * fx
    bot = a[y0 + 1][:, x0] * (1 - fx) + a[y0 + 1][:, x0 + 1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def comet_mask(y: np.ndarray) -> np.ndarray:
    """Soft mask (0..1, full size) of the comet's head and tail: blobs bigger than stars that stand out of
    the sky, grown a little so the faint edges are included."""
    h, w = y.shape
    sm, f = _small_plane(y.astype(np.float32), 700)
    op = _blur(_opening(sm, 3), 3.0)
    sky = float(np.median(op))
    noise = float(np.median(np.abs(op - sky))) * 1.4826 + 1e-9
    m = np.clip((op - sky - 3 * noise) / (6 * noise), 0, 1)
    m = _blur(-_min_filter(-m, 4), 3.0)
    return np.clip(_resize(m, h, w), 0, 1)


def align_comet_channels(chans: dict, ref_key: str | None = None) -> list:
    """Comets imaged one filter after another (e.g. NINA's Orbitals: R, G, B, then L) sit in a slightly
    different place in each filter's master once the masters are lined up on the stars - the comet moved in
    between. This moves only the comet (head and tail) in each filter onto the reference filter's comet, so
    the colours of the comet line up; the stars stay where they are. Changes chans in place; returns notes."""
    if len(chans) < 2:
        return []
    keys = list(chans)
    ref_key = ref_key if ref_key in chans else ("L" if "L" in chans else keys[0])
    ref = chans[ref_key]
    h, w = ref.shape
    pr = comet_nucleus(ref)
    if pr is None:
        return ["comet: its head could not be found in the " + ref_key + " master, so the filters were not "
                "lined up on the comet"]
    prx, pry = pr[0] * w - 0.5, pr[1] * h - 0.5
    mref = comet_mask(ref)
    moved, notes = [], []
    for k in keys:
        if k == ref_key:
            continue
        pk = comet_nucleus(chans[k])
        if pk is None:
            notes.append(f"comet: not found in {k}, left as it is")
            continue
        dx, dy = prx - (pk[0] * w - 0.5), pry - (pk[1] * h - 0.5)
        dist = float(np.hypot(dx, dy))
        if dist < 0.7:
            continue
        if dist > 0.1 * max(h, w):
            notes.append(f"comet: in {k} it seems {dist:.0f} px away from {ref_key} - too far to be the same "
                         "night, left as it is")
            continue
        shifted = _shift_plane(chans[k], dx, dy)
        region = np.maximum(mref, _shift_plane(mref, -dx, -dy))        # where the comet is now and was
        chans[k] = (chans[k] * (1 - region) + shifted * region).astype(np.float32)
        moved.append(f"{k} {dist:.1f} px")
    if moved:
        notes.insert(0, f"comet lined up across the filters on {ref_key} (it had moved: " + ", ".join(moved)
                     + ") - the stars stay where they are")
    return notes


class _CometTool(Tool):
    optional = True

    def possible(self, s):
        return target_type(s) == "comet"

    def applicable(self, s):
        if target_type(s) != "comet":
            return False, "Only for comets"
        if s.rgb is None:
            return False, "Combine the colour channels first"
        if s.linear:
            return False, "Brighten the image first"
        return True, ""

    @staticmethod
    def nucleus(s):
        return comet_nucleus(_lum(s.rgb)) or (0.5, 0.5)

    @staticmethod
    def centre(s, params):
        h, w = s.shape
        return float(params["fx"]) * w - 0.5, float(params["fy"]) * h - 0.5


class CometJets(_CometTool):
    key = "comet_jets"
    title = "Reveal Comet Jets"
    tip = ("Brings out jets, shells and fine rays in the comet's head that the bright, round glow of the coma "
           "hides. It compares the picture with copies of itself turned a little around the nucleus: whatever is "
           "round cancels out, and the structure stays. Works best with the stars taken out. "
           "(PixInsight: LarsonSekanina)")

    def variants(self, s):
        fx, fy = self.nucleus(s)
        base = {"fx": fx, "fy": fy, "angle": 10.0, "detail": 1.5}
        return [Variant("Subtle", dict(base, amount=0.35)),
                Variant("Moderate", dict(base, amount=0.7), "", True),
                Variant("Strong", dict(base, amount=1.2, angle=15.0)),
                Variant("Jets map only", dict(base, amount=0.0, map=True),
                        "The structure on its own, grey - a study picture, not for the final image")]

    def sliders(self, s, params):
        if params.get("map"):
            return [Slider("angle", "Turn (degrees)", 3, 40, params["angle"],
                           "Small turns show fine jets near the nucleus; bigger turns show broad features", 1)]
        return [Slider("amount", "Strength", 0.0, 2.0, params["amount"], "How strongly the structure is added"),
                Slider("angle", "Turn (degrees)", 3, 40, params["angle"],
                       "Small turns show fine jets near the nucleus; bigger turns show broad features", 1),
                Slider("detail", "Smoothing", 0.5, 4.0, params["detail"],
                       "Smooths the structure before adding it, so noise isn't brought out with it", 0.1)]

    def apply(self, s, params):
        s = s.copy()
        cx, cy = self.centre(s, params)
        sc = max(s.shape) / 2000.0                           # angles act the same at any picture size;
        y = _lum(s.rgb)                                      # the smoothing scales with it
        d = _blur(larson_sekanina(y, cx, cy, float(params["angle"])), max(0.5, float(params["detail"]) * sc))
        if params.get("map"):
            spread = float(np.percentile(np.abs(d), 99.5)) + 1e-6
            g = np.clip(0.5 + 0.45 * d / spread, 0, 1).astype(np.float32)
            s.rgb = np.repeat(g[None], 3 if s.is_color else 1, axis=0)
            if s.stars is not None:
                s.stars = np.zeros_like(s.stars)
        else:
            a = float(params["amount"])
            sky = float(np.median(y[::3, ::3]))
            # only on the comet, not in the empty sky (where it would only bring out noise)
            wgt = np.clip((_blur(y, 4.0 * max(1, sc)) - sky - 0.02) / 0.1, 0, 1)
            s.rgb = _apply_lum(s.rgb, lambda yy: np.clip(yy + a * wgt * d, 0, 1))
        s.done.append(self.key)
        return s


class CometTail(_CometTool):
    key = "comet_tail"
    title = "Bring Out the Tail"
    tip = ("Takes away part of the bright, round glow around the comet's head (the coma), so the tail, "
           "the faint ion streamers and any jets stand out. The round glow is measured all the way round the "
           "nucleus at every distance; the tail points one way only, so it is left alone.")

    def variants(self, s):
        fx, fy = self.nucleus(s)
        return [Variant(n, {"fx": fx, "fy": fy, "amount": a}, "", i == 1)
                for i, (n, a) in enumerate((("Subtle", 0.35), ("Moderate", 0.6), ("Strong", 0.85)))]

    def sliders(self, s, params):
        return [Slider("amount", "How much of the round glow to take away", 0.0, 1.0, params["amount"],
                       "More shows more of the tail's structure, but the head looks less natural")]

    def apply(self, s, params):
        s = s.copy()
        cx, cy = self.centre(s, params)
        a = float(params["amount"])
        y = _lum(s.rgb)
        sky = float(np.median(y[::3, ::3]))
        model = np.maximum(radial_profile_model(y, cx, cy) - sky, 0)

        def fn(yy):
            out = yy - a * model
            # keep the head bright: lift what is left back up a little
            return np.clip(sky + (out - sky) / max(0.35, 1 - 0.55 * a), 0, 1)
        s.rgb = _apply_lum(s.rgb, fn)
        s.done.append(self.key)
        return s


TOOLS = {t.key: t for t in (HaGlow(), CombineChannels(), DualBandPalette(), BalanceColours(), RemoveGreen(), TakeStarsOut(),
                            CleanStarLeftovers(), Brighten(),
                            TameCore(), Curves(), Saturation(), AdjustColours(), LocalContrast(), BrightenStars(), ShrinkStars(),
                            StarColours(), PutStarsBack(), CometTail(), CometJets(),
                            DustDonuts(), DarkenSky(), FinalNoise(), FlipPicture(), FinalCrop(), SpotFix(), SignPicture())}

# steps that can sensibly be run again on their own result ("Do It Again"); one-way steps (combining,
# brightening, taking the stars out or putting them back, signing …) are not on the list
REPEATABLE = ("balance", "scnr", "core", "curves", "saturation", "colours", "local", "shrink", "star_color", "dust",
              "sky", "denoise", "flip", "final_crop", "spotfix")
for _k in REPEATABLE:
    TOOLS[_k].repeatable = True


def thumb_state(tool: Tool, s: State, size=420) -> State:
    """The small image a tool's choices are rendered on."""
    if tool.key == "signature":                 # zoom into the corner where the signature goes
        sm = s.small(size * 2)
        corner = getattr(tool, "thumb_corner", "bottom right")
        if corner not in SIGN_CORNERS:
            corner = "bottom right"
        return sm.crop(int(size * 1.1), SignPicture.corner_focus(sm.shape, corner))
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

_POLISH = ["core", "curves", "saturation", "colours", "local"]
_STARS = ["star_stretch", "shrink", "star_color", "stars_back"]
WORKFLOWS = {
    "osc": ["balance", "scnr", "dust", "stars_out", "leftovers", "stretch"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "osc_dual": ["palette", "dust", "stars_out", "leftovers", "stretch", "scnr"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "osc_sho": ["combine", "dust", "stars_out", "leftovers", "stretch", "scnr"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "smart": ["balance", "scnr", "dust", "stars_out", "leftovers", "stretch"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "mono_rgb": ["ha_glow", "combine", "balance", "scnr", "dust", "stars_out", "leftovers", "stretch"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "mono_lrgb": ["ha_glow", "combine", "balance", "scnr", "dust", "stars_out", "leftovers", "stretch"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "mono_nb": ["combine", "dust", "stars_out", "leftovers", "stretch", "scnr"] + _POLISH + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
    "mono": ["dust", "stars_out", "leftovers", "stretch", "core", "curves", "local"] + _STARS + ["sky", "denoise", "flip", "final_crop", "spotfix", "signature"],
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
    "colours": "Optional: bring out (or calm) one colour at a time - e.g. the blue of OIII next to the red of Ha.",
    "ha_glow": "You have an Ha image with your RGB - add its glow to the red channel before combining.",
}


# What kind of object it is changes only what is RECOMMENDED - every step stays available
TYPE_TUNING = {
    "galaxy": {"recommend": {"local", "core"}, "optional": set()},
    "planetary nebula": {"recommend": {"core"}, "optional": set()},
    "star cluster": {"recommend": set(), "optional": {"stars_out", "shrink", "local"}},
    "comet": {"recommend": {"sky"}, "optional": {"stars_out", "shrink", "local"}},
}
TYPE_WHY = {
    ("galaxy", "local"): "A galaxy: this brings out the dust lanes and the structure of the spiral arms.",
    ("galaxy", "core"): ("A galaxy: its core is far brighter than the arms, so after brightening it is usually "
                         "burnt out. Tame it now, before Curves and Saturation brighten it further - or Skip it "
                         "if the core already shows its detail."),
    ("comet", "sky"): "A comet: a darker, even sky lets the faint tail show.",
    ("comet", "comet_tail"): "A comet: take away some of the round glow of the head so the tail stands out.",
    ("comet", "comet_jets"): "A comet: bring out jets and shells near the nucleus (best with the stars out).",
}


def target_type(s: "State") -> str:
    import targets
    return targets.classify(s.target)


def workflow(s: State) -> list:
    """[(tool, status)] in the recommended order; status: done / skipped / next / later / n/a.
    The kind of object (galaxy, star cluster, comet …) only moves steps between recommended and
    optional - none is ever taken away."""
    steps = list(WORKFLOWS.get(s.kind, WORKFLOWS["osc"]))
    tune = TYPE_TUNING.get(target_type(s), {})
    if target_type(s) == "comet" and "stretch" in steps:
        i = steps.index("stretch") + 1
        steps[i:i] = ["comet_tail", "comet_jets"]
    if target_type(s) == "planetary nebula" and "core" in steps and "curves" in steps:
        # a planetary's ring is tamed after Curves (whose contrast would make it white again), before Saturation
        steps.remove("core")
        steps.insert(steps.index("curves") + 1, "core")
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
        optional = t.is_optional(s)
        if key in tune.get("optional", ()):
            optional = True
        elif key in tune.get("recommend", ()):
            optional = False
        if optional:
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
    special = TYPE_WHY.get((target_type(s), t.key))
    if special:
        return special
    base = t.why(s) if type(t).why is not Tool.why else WHY.get(t.key, t.tip)
    return base


# ==========================================================================
# Project: every step saved (resume, go back, save / open)
# ==========================================================================

def find_project(folder) -> "Path | None":
    """The saved project in this folder - the folder itself, or (when a project was saved into a folder
    that had other things in it) its <target>.szproj subfolder. Several: the most recently changed one."""
    folder = Path(folder)

    def ok(d):
        man = cp.read_json(d / "manifest.json", None) or {}
        return bool(man.get("steps")) and man.get("kind", "finish-project") == "finish-project"
    if ok(folder):
        return folder
    try:
        subs = [d for d in folder.iterdir() if d.is_dir() and ok(d)]
    except OSError:
        return None
    if not subs:
        return None
    return max(subs, key=lambda d: (d / "manifest.json").stat().st_mtime)


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
                  sources=self.resolve_sources(meta.get("sources", [])), palette=meta.get("palette"),
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

    MASTERS = "Masters"

    @staticmethod
    def _name(p) -> str:
        """A file's name from a path written on any computer (Windows paths use backslashes)."""
        return str(p).replace("\\", "/").rstrip("/").split("/")[-1]

    def resolve_sources(self, sources) -> list:
        """The masters, as paths that exist: a master that is no longer where it was (the project folder was
        moved, or the masters were copied into it) is looked for in the project's own Masters folder."""
        out = []
        for p in sources or []:
            q = Path(p)
            if not q.exists():
                alt = self.dir / self.MASTERS / self._name(p)
                if alt.exists():
                    q = alt
            out.append(str(q))
        return out

    def masters_to_keep(self) -> list:
        """Every file that belongs with the project's masters: each master plus its companions in the same
        folder (its stars-only image, stack log, comet/stars masters, notes - all named after it)."""
        files = []

        def add(q: Path):
            if q.exists() and q not in files:
                files.append(q)
                for sib in sorted(q.parent.glob(q.stem + "_*")):
                    if sib.is_file() and sib not in files:
                        files.append(sib)
        srcs = [Path(p) for p in self.resolve_sources(self.manifest.get("sources", []))]
        srcs += [Path(p) for p in self.manifest.get("origins", {}).values()]
        for q in srcs:
            add(q)
            # a page-3 ("_prepared") master: also keep the stacked master it was made from - needed to redo
            # page 3 from scratch or to add more nights. It sits one folder up (…/Prepared) unless page 3
            # saved elsewhere; newer prepared masters name it in their header (ORIGFILE)
            if q.stem.endswith("_prepared") or "_prepared_" in q.stem:
                base = q.stem.split("_prepared")[0]
                names = [base]
                try:
                    import prepare as P
                    org = str(P.load_master(q).look.get("ORIGFILE") or "")
                    if org:
                        names.insert(0, Path(org).stem)
                except Exception:  # noqa: BLE001
                    pass
                for folder in (q.parent.parent, q.parent):
                    for nm in names:
                        for ext in (".fits", ".fit", ".xisf", ".tif", ".tiff"):
                            add(folder / (nm + ext))
        return [f for f in files if not f.stem.endswith("_prepared_stars") or f.exists()]

    def keep_masters(self) -> tuple:
        """Copy the masters (and their companions) into <project>/Masters and use those copies from now on,
        so the project folder holds everything. The originals are left where they are. Returns (files, bytes)."""
        dest = self.dir / self.MASTERS
        files = [f for f in self.masters_to_keep() if dest.resolve() not in f.resolve().parents]
        if not files:
            return 0, 0
        dest.mkdir(parents=True, exist_ok=True)
        total = 0
        for f in files:
            t = dest / f.name
            if not t.exists() or t.stat().st_size != f.stat().st_size:
                tmp = t.with_name(t.name + ".copying")
                shutil.copy2(f, tmp)
                tmp.replace(t)
            total += t.stat().st_size

        def moved(lst):
            return [str(dest / self._name(p)) if (dest / self._name(p)).exists() else p for p in lst or []]
        origins = dict(self.manifest.get("origins", {}))
        for f in files:
            origins.setdefault(f.name, str(f))              # where each copied file came from
        self.manifest["origins"] = origins
        self.manifest["sources"] = moved(self.manifest.get("sources"))
        for st in self.steps:
            mp = self.dir / st["dir"] / "meta.json"
            meta = cp.read_json(mp, None)
            if meta is not None:
                meta["sources"] = moved(meta.get("sources"))
                cp.write_json(mp, meta)
        self._write()
        return len(files), total

    def save_as(self, folder: Path, keep_masters: bool = False) -> "Project":
        proj = self._save_as(folder)
        if keep_masters:
            proj.kept = proj.keep_masters()
        return proj

    def _save_as(self, folder: Path) -> "Project":
        """Copy the project into a folder of the user's choice (a non-empty folder gets a
        <target>.szproj subfolder). Saving again to the same place keeps it as it is - every
        step is already written there - and an older copy is only replaced once the new one is
        complete, so a failed save never loses anything."""
        folder = Path(folder)
        src = self.dir.resolve()
        if folder.exists() and any(folder.iterdir()) and folder.resolve() != src:
            folder = folder / f"{cp.safe_name(self.manifest.get('target') or 'project')}.szproj"
        dst = folder.resolve()
        if dst == src:
            self._write()
            return self                                  # already saved here, step by step
        if src in dst.parents:
            raise ValueError("Choose a folder outside the project's own folder.")
        if dst in src.parents:
            raise ValueError("That folder contains this project; choose another folder.")
        if not self.dir.exists():
            raise FileNotFoundError(f"The project's working folder is missing ({self.dir}). Your picture is "
                                    "still on screen - keep it with Save the Picture….")
        tmp = folder.with_name(folder.name + ".saving")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(self.dir, tmp)
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
        return Project(folder)

    def size(self):
        return cp._size(self.dir) if self.dir.exists() else 0


# ==========================================================================
# Export
# ==========================================================================

def scope_names(sources) -> list:
    """The telescope (rig) names in the masters' headers, in order, without repeats."""
    import fits_core as core
    out = []
    for p in sources or []:
        try:
            _d, _c, look = core.read_fits(p, header_only=True)
        except Exception:  # noqa: BLE001 - a missing or odd file just gives no name
            continue
        t = str(look.get("TELESCOP") or "").strip()
        if t and t.lower() not in ("my scope", "unknown") and t not in out:
            out.append(t)
    return out


def picture_folder_name(target: str, sources) -> str:
    """'M101_RedCat51' - the folder that holds every file of one finished picture."""
    import checkpoints as cp
    name = cp.safe_name(target) if target else "Picture"
    scopes = scope_names(sources)
    if scopes:
        name += "_" + "+".join(cp.safe_name(x) for x in scopes[:2])
    return name


CREDIT_TEXT = "Processed with SubZeroAstro"
CREDIT_FONT = "Arial"


def credit_corner(steps) -> str:
    """Where the small 'Processed with SubZeroAstro' credit goes: bottom left, unless the picture's
    signature (or target name) is there - then bottom right."""
    for st_ in reversed(list(steps or [])):
        if st_.get("tool") != "signature":
            continue
        p = st_.get("params") or {}
        pos = p.get("pos")
        if pos:
            left, bottom = float(pos[0]) <= 0.5, float(pos[1]) > 0.5
        else:
            c = str(p.get("corner") or "bottom right")
            left, bottom = c.endswith("left"), c.startswith("bottom")
        return "bottom right" if (left and bottom) else "bottom left"
    return "bottom left"


def add_credit(img: np.ndarray, corner: str = "bottom left", text: str = CREDIT_TEXT) -> np.ndarray:
    """The small credit line drawn into a (c, h, w) 0..1 picture: light grey, partly see-through,
    about 1.3% of the picture's shorter side high, in the corner given."""
    c, h, w = img.shape
    short = min(h, w)
    a = signature_alpha(text, CREDIT_FONT, False, max(9.0, short * 0.013))
    ah, aw = a.shape
    if ah >= h // 4 or aw >= w // 2:
        return img
    m = max(4, int(short * 0.015))
    y0 = h - m - ah
    x0 = m if corner.endswith("left") else w - m - aw
    out = img.copy()
    k = 0.55 * a[None]
    out[:, y0:y0 + ah, x0:x0 + aw] = out[:, y0:y0 + ah, x0:x0 + aw] * (1 - k) + 0.85 * k
    return out


def export(s: State, base: Path, formats=("tif", "jpg"), save_rgb8=None, credit=None) -> list:
    """Write the current image. Stars still separated -> also a stars file and a
    combined preview. credit: a corner ('bottom left' / 'bottom right') to put the small
    'Processed with SubZeroAstro' line on the pictures (TIFF, JPG, PNG - never on the XISF / FITS data),
    or None."""
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
           ("HISTORY", "SubZeroAstro finish: " + ", ".join(TOOLS[k].title for k in dict.fromkeys(s.done) if k in TOOLS), "")]
    data = img[0] if img.shape[0] == 1 else img
    pic = img
    if credit and any(f in ("tif", "jpg", "png") for f in formats):
        try:
            pic = add_credit(img, credit)
        except Exception:  # noqa: BLE001 - a credit line must never stop a save
            pic = img
    for f in formats:
        p = base.with_name(base.name + "." + f)
        if f == "tif":
            rgb = pic if pic.shape[0] == 3 else np.repeat(pic, 3, 0)
            P.write_tiff16(p, rgb.transpose(1, 2, 0))
        elif f in ("jpg", "png"):
            if save_rgb8 is None:
                continue
            rgb = pic if pic.shape[0] == 3 else np.repeat(pic, 3, 0)
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
    "colours": "ColorSaturation (hue-selective saturation curve)",
    "local": "LocalHistogramEqualization",
    "ha_glow": "Continuum subtraction (PixelMath: Ha - k*R) then NBRGBCombination or PixelMath into R",
    "spotfix": "CloneStamp",
    "leftovers": "CloneStamp on what star removal left around bright stars",
    "core": "HDRMultiscaleTransform through a luminance (range) mask on the core",
    "star_color": "PixelMath on the stars image (colour from the R/(G+B) ratio), or Narrowband Normalization's star options",
        "stars_back": "PixelMath screen blend:  ~((~starless)*(~stars))",
    "signature": "Annotation, or a PixelMath-blended signature image",
    "flip": "FastRotation (Horizontal / Vertical Mirror, Rotate 180°)",
    "final_crop": "DynamicCrop",
    "dust": "A new flat is the real cure; otherwise DynamicBackgroundExtraction with samples around the donut "
            "(division), or a synthetic flat built in PixelMath",
    "denoise": "NoiseXTerminator - a light pass (denoise about 0.3-0.5) on the stretched image",
    "sky": "AutomaticBackgroundExtractor or DynamicBackgroundExtraction to model the sky glow, then PixelMath "
           "$T - k*model (k about 0.5) so only part of the glow is taken away",
    "comet_jets": "LarsonSekanina (centre on the nucleus, angle about 10 degrees), blended back with PixelMath",
    "comet_tail": "A 1/rho or azimuthal-median coma model (e.g. the CometAlignment / Comet suite scripts), "
                  "subtracted in part with PixelMath",
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
    done = [TOOLS[k].title for k in dict.fromkeys(s.done) if k in TOOLS]
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
