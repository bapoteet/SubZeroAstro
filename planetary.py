"""
planetary.py - lucky imaging of planets (and later the Moon and Sun): a SER video of thousands of
very short frames is graded frame by frame, the sharpest few percent are lined up and stacked, the
colour channels are lined up (atmospheric dispersion) and the stack is sharpened with wavelets -
the AutoStakkert! + RegiStax workflow, written from the published methods (numpy only).

Phase 1: whole-disc alignment. Multi-point (local) alignment comes next.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# SER colour ids
_MONO, _BAYER, _RGB, _BGR = 0, (8, 9, 10, 11, 16, 17, 18, 19), 100, 101
_BAYER_NAME = {8: "RGGB", 9: "GRBG", 10: "GBRG", 11: "BGGR"}


# ==========================================================================
# Reading SER videos
# ==========================================================================

@dataclass
class SerVideo:
    path: Path
    width: int
    height: int
    frames: int
    depth: int
    color_id: int
    little_endian: bool
    camera: str = ""
    telescope: str = ""
    observer: str = ""
    times: np.ndarray | None = None          # per-frame UTC timestamps (100 ns ticks) if stored
    _mm: np.ndarray | None = field(default=None, repr=False)

    @property
    def planes(self) -> int:
        return 3 if self.color_id in (_RGB, _BGR) else 1

    @property
    def bayer(self) -> str | None:
        return _BAYER_NAME.get(self.color_id)

    @property
    def colour(self) -> bool:
        return self.planes == 3 or self.bayer is not None

    def raw(self, i: int) -> np.ndarray:
        return self._mm[i]

    def frame(self, i: int) -> np.ndarray:
        """Frame i as float32: (3, h, w) RGB for colour videos, (h, w) for mono, 0..1."""
        a = np.asarray(self._mm[i], dtype=np.float32)
        scale = float((1 << self.depth) - 1) if self.depth <= 16 else 65535.0
        a /= scale
        if self.planes == 3:
            a = np.moveaxis(a, -1, 0)
            return a[::-1].copy() if self.color_id == _BGR else a
        if self.bayer:
            return debayer_superpixel_full(a, self.bayer)
        return a


def open_ser(path) -> SerVideo:
    path = Path(path)
    with open(path, "rb") as f:
        h = f.read(178)
    if len(h) < 178 or not h[:14].startswith(b"LUCAM-RECORDER"):
        raise ValueError(f"{path.name} is not a SER video")
    _lu, cid, little, w, hh, depth, n = struct.unpack("<7i", h[14:42])
    planes = 3 if cid in (_RGB, _BGR) else 1
    bpp = 1 if depth <= 8 else 2
    size = path.stat().st_size
    fb = w * hh * planes * bpp
    if fb <= 0 or 178 + fb > size:
        raise ValueError(f"{path.name}: damaged SER header")
    n = min(n, (size - 178) // fb)                    # a capture cut short still opens
    # SER's "LittleEndian" flag is famously written the wrong way round by most programs:
    # 16-bit data is little-endian in practice almost everywhere
    dt = np.uint8 if bpp == 1 else np.dtype("<u2")
    shape = (n, hh, w, 3) if planes == 3 else (n, hh, w)
    mm = np.memmap(path, dtype=dt, mode="r", offset=178, shape=shape)
    times = None
    tail = size - 178 - n * fb
    if tail >= 8 * n:
        try:
            times = np.fromfile(path, dtype="<i8", count=n, offset=178 + n * fb)
        except (OSError, ValueError):
            times = None
    txt = lambda b: b.split(b"\0")[0].decode("latin-1", "replace").strip()  # noqa: E731
    return SerVideo(path, w, hh, n, depth, cid, bool(little), camera=txt(h[82:122]), telescope=txt(h[122:162]),
                    observer=txt(h[42:82]), times=times, _mm=mm)


def debayer_superpixel_full(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """Simple full-size RGB from a Bayer frame (bilinear) - good enough for grading and stacking;
    stacking many frames smooths the rest."""
    p = pattern.upper()
    h, w = cfa.shape
    out = np.zeros((3, h, w), np.float32)
    cnt = np.zeros((3, h, w), np.float32)
    idx = {"R": 0, "G": 1, "B": 2}
    for i in (0, 1):
        for j in (0, 1):
            c = idx[p[2 * i + j]]
            out[c, i::2, j::2] = cfa[i::2, j::2]
            cnt[c, i::2, j::2] = 1
    k = np.array([0.25, 0.5, 0.25], np.float32)
    for c in range(3):
        num, den = out[c], cnt[c]
        for ax in (0, 1):
            num = _conv1(num, k, ax)
            den = _conv1(den, k, ax)
        out[c] = np.where(cnt[c] > 0, out[c], num / np.maximum(den, 1e-6))
    return out


def _conv1(a, k, axis):
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    n = a.shape[axis]
    for i, kv in enumerate(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += kv * p[tuple(sl)]
    return out


def _gauss(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    return _conv1(_conv1(a, k, 0), k, 1)


def luminance(fr: np.ndarray) -> np.ndarray:
    return fr.mean(axis=0) if fr.ndim == 3 else fr


# ==========================================================================
# Grading, alignment, stacking
# ==========================================================================

def planet_box(lum: np.ndarray, pad: float = 0.25):
    """(y0, y1, x0, x1) around the bright object, with some sky round it."""
    sm = _gauss(lum, 2.0)
    bg = float(np.median(sm))
    thr = bg + 0.25 * (float(sm.max()) - bg)
    ys, xs = np.nonzero(sm > thr)
    if ys.size == 0:
        return 0, lum.shape[0], 0, lum.shape[1]
    h, w = lum.shape
    py = int((ys.max() - ys.min()) * pad) + 8
    px = int((xs.max() - xs.min()) * pad) + 8
    return max(0, ys.min() - py), min(h, ys.max() + py + 1), max(0, xs.min() - px), min(w, xs.max() + px + 1)


def centroid(lum: np.ndarray):
    sm = _gauss(lum, 2.0)
    bg = float(np.median(sm))
    m = np.clip(sm - bg - 0.1 * (float(sm.max()) - bg), 0, None)
    s = m.sum()
    if s <= 0:
        return lum.shape[1] / 2, lum.shape[0] / 2
    yy, xx = np.mgrid[:lum.shape[0], :lum.shape[1]]
    return float((xx * m).sum() / s), float((yy * m).sum() / s)


def sharpness(lum: np.ndarray, noise_sigma: float = 1.2) -> float:
    """How sharp a frame is: energy of fine detail (gradients of a lightly smoothed frame - the
    smoothing keeps noise from counting as detail), relative to the object's brightness."""
    s = _gauss(lum, noise_sigma)
    gy = s[2:, 1:-1] - s[:-2, 1:-1]
    gx = s[1:-1, 2:] - s[1:-1, :-2]
    bright = s[1:-1, 1:-1]
    m = bright > (np.median(bright) + 0.2 * (bright.max() - np.median(bright)))
    if m.sum() < 20:
        m = np.ones_like(bright, bool)
    e = float(np.mean((gx * gx + gy * gy)[m]))
    lev = float(np.mean(bright[m])) + 1e-6
    return e / (lev * lev)


def _window(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def phase_shift(ref: np.ndarray, img: np.ndarray, win=None):
    """(dx, dy) to move img onto ref, to a fraction of a pixel (phase correlation)."""
    if win is None:
        win = _window(*ref.shape)
    A = np.fft.rfft2((ref - ref.mean()) * win)
    B = np.fft.rfft2((img - img.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfft2(R, s=ref.shape)
    c = np.fft.fftshift(c)
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    h, w = c.shape

    def sub(a, b, cc):
        d = a - 2 * b + cc
        return 0.0 if abs(d) < 1e-12 else 0.5 * (a - cc) / d
    dy = iy - h // 2 + (sub(c[iy - 1, ix], c[iy, ix], c[(iy + 1) % h, ix]) if 0 < iy < h - 1 else 0)
    dx = ix - w // 2 + (sub(c[iy, ix - 1], c[iy, ix], c[iy, (ix + 1) % w]) if 0 < ix < w - 1 else 0)
    return float(dx), float(dy)


def fourier_shift(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move an image by a fraction of a pixel without blurring it."""
    h, w = a.shape[-2:]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy)).astype(np.complex64)
    if a.ndim == 3:
        return np.stack([np.fft.irfft2(np.fft.rfft2(p) * ph, s=(h, w)) for p in a]).astype(np.float32)
    return np.fft.irfft2(np.fft.rfft2(a) * ph, s=(h, w)).astype(np.float32)


@dataclass
class Graded:
    quality: np.ndarray            # per frame, higher = sharper
    centres: np.ndarray            # (n, 2) x, y
    box: tuple                     # crop around the planet (y0, y1, x0, x1)
    order: np.ndarray              # frame indices, best first


def grade(video: SerVideo, progress=None, cancel=None) -> Graded:
    n = video.frames
    lum0 = luminance(video.frame(n // 2))
    q = np.zeros(n, np.float32)
    cen = np.zeros((n, 2), np.float32)
    for i in range(n):
        if cancel and cancel():
            raise InterruptedError()
        lum = luminance(video.frame(i))
        cen[i] = centroid(lum)
        y0, y1, x0, x1 = planet_box(lum, 0.08)
        q[i] = sharpness(lum[y0:y1, x0:x1])
        if progress and i % 20 == 0:
            progress(f"Grading frames: {i + 1}/{n}", (i + 1) / n)
    # a box that holds the planet in every frame (it wanders in poor seeing / tracking)
    y0, y1, x0, x1 = planet_box(lum0, 0.3)
    cx0, cy0 = centroid(lum0)
    spread_x = float(np.percentile(np.abs(cen[:, 0] - cx0), 98))
    spread_y = float(np.percentile(np.abs(cen[:, 1] - cy0), 98))
    half_h = int(max(cy0 - y0, y1 - cy0) + spread_y + 4)
    half_w = int(max(cx0 - x0, x1 - cx0) + spread_x + 4)
    half = max(half_h, half_w)
    half = min(half, video.height // 2, video.width // 2)
    box = (half, half)
    order = np.argsort(-q)
    return Graded(q, cen, box, order)


def _cut(fr, cx, cy, half):
    """A (…, 2*half, 2*half) cut centred on (cx, cy), padded with the sky where it runs off the frame."""
    h, w = fr.shape[-2:]
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    out_shape = fr.shape[:-2] + (2 * half, 2 * half)
    sky = float(np.median(fr))
    out = np.full(out_shape, sky, np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = fr[..., sy0:sy1, sx0:sx1]
    return out, (cx - x0 - half, cy - y0 - half)


def stack(video: SerVideo, g: Graded, keep: float = 0.10, progress=None, cancel=None, log=None):
    """Average the sharpest `keep` (fraction, or a count when > 1) of the frames, each lined up on a
    reference made from the best frames. Returns (image, info)."""
    n_keep = int(keep) if keep > 1 else max(3, int(round(keep * len(g.order))))
    n_keep = min(n_keep, len(g.order))
    pick = g.order[:n_keep]
    half = g.box[0]
    # reference: the best ~5% (at least 5) lined up on their centres
    n_ref = max(5, min(50, n_keep // 2))
    ref = None
    for i in pick[:n_ref]:
        cut, _ = _cut(luminance(video.frame(int(i))), *g.centres[i], half)
        ref = cut if ref is None else ref + cut
    ref /= n_ref
    win = _window(*ref.shape)
    acc = None
    shifts = []
    for k, i in enumerate(pick):
        if cancel and cancel():
            raise InterruptedError()
        fr = video.frame(int(i))
        cut, _ = _cut(fr, *g.centres[i], half)
        dx, dy = phase_shift(ref, luminance(cut), win)
        shifts.append((dx, dy))
        moved = fourier_shift(cut, dx, dy)
        acc = moved if acc is None else acc + moved
        if progress and k % 10 == 0:
            progress(f"Stacking the best {n_keep} frames: {k + 1}/{n_keep}", (k + 1) / n_keep)
    img = acc / len(pick)
    info = {"frames": len(pick), "of": len(g.order), "shift_rms": float(np.sqrt(np.mean(np.square(shifts))))}
    if log:
        log(f"Stacked the best {len(pick)} of {len(g.order)} frames ({100 * len(pick) / len(g.order):.0f}%)")
    return img, info


def align_channels(img: np.ndarray):
    """Line red and blue up on green (the atmosphere spreads a low planet into a little rainbow).
    Returns (image, {"R": (dx, dy), "B": (dx, dy)})."""
    if img.ndim != 3:
        return img, {}
    win = _window(*img.shape[-2:])
    out = img.copy()
    moved = {}
    for c, name in ((0, "R"), (2, "B")):
        dx, dy = phase_shift(img[1], img[c], win)
        out[c] = fourier_shift(img[c], dx, dy)
        moved[name] = (dx, dy)
    return out, moved


# ==========================================================================
# Wavelet sharpening (RegiStax-style 'a trous' layers)
# ==========================================================================

_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16


def wavelet_layers(a: np.ndarray, n: int = 6):
    """Detail layers 1..n (finest first) and the residual: a == sum(layers) + residual."""
    layers = []
    c = a.astype(np.float32)
    for j in range(n):
        step = 1 << j
        k = np.zeros(4 * step + 1, np.float32)
        k[::step] = _B3
        s = _conv1(_conv1(c, k, 0), k, 1)
        layers.append(c - s)
        c = s
    return layers, c


def sharpen(img: np.ndarray, gains=(1.0, 2.5, 2.0, 1.4, 1.1, 1.0), denoise: float = 0.0) -> np.ndarray:
    """Boost wavelet layers: gain 1 = unchanged. `denoise` (0..1) softly removes noise from the
    finest layer before boosting."""
    planes = img if img.ndim == 3 else img[None]
    out = []
    for p in planes:
        layers, res = wavelet_layers(p, len(gains))
        acc = res.copy()
        for j, (L, gn) in enumerate(zip(layers, gains)):
            if denoise > 0 and j < 2:
                t = denoise * 3.0 * float(np.median(np.abs(L))) / 0.6745 * (1.0 if j == 0 else 0.5)
                L = np.sign(L) * np.maximum(np.abs(L) - t, 0)
            acc += gn * L
        out.append(acc)
    o = np.stack(out)
    return o if img.ndim == 3 else o[0]


def to_8bit(img: np.ndarray, black=None, white=None) -> np.ndarray:
    a = img if img.ndim == 3 else img[None]
    lo = float(np.percentile(a, 1)) if black is None else black
    hi = float(np.percentile(a, 99.98)) if white is None else white
    b = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    b = (b * 255 + 0.5).astype(np.uint8)
    return np.moveaxis(b, 0, -1) if img.ndim == 3 else b[0]
