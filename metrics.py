"""
metrics.py - star quality measurements for one frame (NumPy only).

measure(image, bayer=False, saturation=None) returns:
    fwhm          median star FWHM in sensor pixels (elliptical Moffat, beta = 4)
    eccentricity  median star eccentricity (0 = round, 1 = a line)
    snr           90th-percentile star peak / background noise (all detected stars)
    stars         number of stars detected in the measured area
    background    median sky level (ADU)
    noise         background noise (ADU, robust sigma)
    fitted        number of stars that were fitted

Method
  1. Sky background is estimated on a 64-pixel grid and subtracted.
  2. Stars are local maxima of a 3x3-smoothed image more than 5 sigma above the sky.
  3. Up to 150 of the brightest unsaturated stars are fitted with an elliptical
     Moffat profile (beta = 4, as PixInsight's SubframeSelector default) using a
     vectorised Levenberg-Marquardt solver.
  4. FWHM = geometric mean of the major and minor axis FWHM.

For one-shot-colour (Bayer) frames the raw mosaic is smoothed with a 2x2 box,
which mixes one red, two green and one blue pixel at every position, giving a
pattern-free image at full resolution. The small widening this causes is
removed from the FWHM afterwards.
"""

from __future__ import annotations

import numpy as np

BETA = 4.0
FWHM_PER_ALPHA = 2.0 * np.sqrt(2.0 ** (1.0 / BETA) - 1.0)     # 0.8698 for beta = 4
BOX2_VAR_FWHM2 = 8 * np.log(2) * 0.25                          # FWHM^2 added by a 2x2 box
TILE = 64
MAX_FIT = 150
HALF = 10                                                      # cutout radius -> 21 x 21


def _box3(img):
    """3x3 mean filter (edges left as the original values)."""
    out = img.copy()
    c = img[1:-1, 1:-1]
    s = (img[:-2, :-2] + img[:-2, 1:-1] + img[:-2, 2:] +
         img[1:-1, :-2] + c + img[1:-1, 2:] +
         img[2:, :-2] + img[2:, 1:-1] + img[2:, 2:])
    out[1:-1, 1:-1] = s / 9.0
    return out


def _background(img):
    """Return (background map, sky median, noise sigma)."""
    h, w = img.shape
    sub = img[::4, ::4]
    t = TILE // 4
    ny, nx = max(1, sub.shape[0] // t), max(1, sub.shape[1] // t)
    blocks = sub[:ny * t, :nx * t].reshape(ny, t, nx, t).transpose(0, 2, 1, 3).reshape(ny, nx, t * t)
    grid = np.median(blocks, axis=2).astype(np.float32)
    # Expand the grid to full size (nearest tile); pad the ragged edge.
    ry = np.minimum(np.arange(h) // TILE, ny - 1)
    rx = np.minimum(np.arange(w) // TILE, nx - 1)
    bg = grid[ry][:, rx]
    sky = float(np.median(grid))
    resid = sub - bg[::4, ::4]
    med = np.median(resid)
    sigma = float(np.median(np.abs(resid - med)) * 1.4826)
    return bg, sky, max(sigma, 1e-6)


def _find_stars(resid, margin):
    sm = _box3(resid)
    # Noise of the smoothed image itself (correct even when pixels are correlated)
    s = sm[::3, ::3]
    med = np.median(s)
    sig_sm = float(np.median(np.abs(s - med)) * 1.4826) + 1e-6
    thresh = med + 5.0 * sig_sm
    h, w = sm.shape
    inner = np.zeros_like(sm, dtype=bool)
    inner[margin:h - margin, margin:w - margin] = True
    ys, xs = np.nonzero((sm > thresh) & inner)
    if ys.size == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    v = sm[ys, xs]
    is_max = np.ones(ys.size, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = sm[ys + dy, xs + dx]
            # strict on one side so flat-topped peaks give exactly one maximum
            is_max &= (v > nb) if (dy, dx) < (0, 0) else (v >= nb)
    ys, xs, v = ys[is_max], xs[is_max], v[is_max]
    # Merge maxima closer than 4 px (keep the brighter), using a coarse grid
    order = np.argsort(-v)
    ys, xs, v = ys[order], xs[order], v[order]
    taken = {}
    keep = np.zeros(ys.size, bool)
    for i in range(ys.size):
        cy, cx = ys[i] >> 2, xs[i] >> 2
        clash = False
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                j = taken.get((gy, gx))
                if j is not None and abs(ys[j] - ys[i]) < 4 and abs(xs[j] - xs[i]) < 4:
                    clash = True
                    break
            if clash:
                break
        if not clash:
            taken[(cy, cx)] = i
            keep[i] = True
    return ys[keep], xs[keep], v[keep]


def _moffat_fit(cuts, iters=14):
    """
    Vectorised Levenberg-Marquardt fit of
        I = B + A * (1 + a dx^2 + 2 b dx dy + c dy^2) ^ -beta
    to N cutouts of shape (S, S). Returns params (N, 7) and a success mask.
    """
    n, s, _ = cuts.shape
    r = s // 2
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64) - r
    xx = xx.ravel()[None, :]
    yy = yy.ravel()[None, :]
    data = cuts.reshape(n, -1).astype(np.float64)

    edge = np.concatenate([cuts[:, 0, :], cuts[:, -1, :], cuts[:, :, 0], cuts[:, :, -1]], axis=1)
    B = np.median(edge, axis=1)
    A = cuts[:, r, r] - B
    pos = np.clip(data - B[:, None], 0, None)
    above = (data - B[:, None]) > (A[:, None] / 2)
    fw0 = np.clip(2 * np.sqrt(above.sum(1) / np.pi), 1.2, 15)
    alpha = fw0 / FWHM_PER_ALPHA
    tot = pos.sum(1) + 1e-9
    x0 = (pos * xx).sum(1) / tot
    y0 = (pos * yy).sum(1) / tot
    x0 = np.clip(x0, -1.5, 1.5)
    y0 = np.clip(y0, -1.5, 1.5)
    a = 1.0 / alpha ** 2
    p = np.stack([B, A, x0, y0, a, np.zeros(n), a], axis=1)

    def model(p):
        Bp, Ap, xp, yp, ap, bp, cp = [p[:, i:i + 1] for i in range(7)]
        dx = xx - xp
        dy = yy - yp
        q = ap * dx * dx + 2 * bp * dx * dy + cp * dy * dy
        u = 1.0 + np.clip(q, -0.99, None)
        ub = u ** -BETA
        return Bp + Ap * ub, (dx, dy, u, ub, Ap, ap, bp, cp)

    def jacobian(parts):
        dx, dy, u, ub, Ap, ap, bp, cp = parts
        g = -BETA * Ap * ub / u          # dI/dq
        J = np.empty((n, dx.shape[1], 7))
        J[:, :, 0] = 1.0
        J[:, :, 1] = ub
        J[:, :, 2] = g * (-2 * ap * dx - 2 * bp * dy)
        J[:, :, 3] = g * (-2 * bp * dx - 2 * cp * dy)
        J[:, :, 4] = g * dx * dx
        J[:, :, 5] = g * 2 * dx * dy
        J[:, :, 6] = g * dy * dy
        return J

    m, parts = model(p)
    chi = ((data - m) ** 2).sum(1)
    lam = np.full(n, 1e-3)
    eye = np.eye(7)[None]
    for _ in range(iters):
        J = jacobian(parts)
        res = data - m
        Jt = J.transpose(0, 2, 1)
        JTJ = Jt @ J
        JTr = (Jt @ res[..., None])[..., 0]
        diag = np.einsum("nkk->nk", JTJ)
        H = JTJ + lam[:, None, None] * (diag[:, :, None] * eye + 1e-12 * eye)
        try:
            dp = np.linalg.solve(H, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            H = H + 1e-6 * eye * (diag.max(1)[:, None, None] + 1)
            dp = np.linalg.solve(H, JTr[..., None])[..., 0]
        pn = p + dp
        mn, partsn = model(pn)
        chin = ((data - mn) ** 2).sum(1)
        better = np.isfinite(chin) & (chin < chi)
        p = np.where(better[:, None], pn, p)
        chi = np.where(better, chin, chi)
        m, parts = model(p)
        lam = np.where(better, lam * 0.3, lam * 10)
        lam = np.clip(lam, 1e-7, 1e7)

    a_, b_, c_ = p[:, 4], p[:, 5], p[:, 6]
    det = a_ * c_ - b_ * b_
    ok = (np.isfinite(p).all(1) & (a_ > 0) & (c_ > 0) & (det > 0) & (p[:, 1] > 0) &
          (np.abs(p[:, 2]) < 3) & (np.abs(p[:, 3]) < 3))
    return p, ok


def measure(img, bayer: bool = False, saturation: float | None = None, area_fraction: float = 1.0):
    """Measure star quality on a 2-D float image (raw orientation is fine)."""
    img = np.asarray(img, dtype=np.float32)
    if bayer:
        # Balance the four colour channels on the sky level, then smooth 2x2
        img = img.copy()
        meds = [float(np.median(img[i::2, j::2][::4, ::4])) for i in (0, 1) for j in (0, 1)]
        ref = float(np.mean(meds))
        for k, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            if meds[k] > 0:
                img[i::2, j::2] *= ref / meds[k]
        h2, w2 = img.shape[0] - 1, img.shape[1] - 1
        img = (img[:h2, :w2] + img[1:, :w2] + img[:h2, 1:] + img[1:, 1:]) * 0.25
    if saturation is None:
        saturation = float(img.max())
    sat_level = 0.92 * saturation

    bg, sky, sigma = _background(img)
    resid = img - bg
    margin = HALF + 2
    ys, xs, peaks = _find_stars(resid, margin)
    n_det = int(ys.size)
    out = {"background": sky, "noise": sigma, "stars": int(round(n_det / max(area_fraction, 1e-3))),
           "fwhm": None, "eccentricity": None, "snr": None, "fitted": 0}
    if n_det == 0:
        return out

    # Peak height of every detected star (raw residual, 3x3 max)
    raw_peak = np.max(np.stack([img[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    res_peak = np.max(np.stack([resid[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    good = raw_peak < sat_level
    # SNR: 90th percentile of star peak / sky noise over all unsaturated detections.
    # Using all stars (not just the fitted ones) makes it independent of how much
    # of the frame is measured; it drops with clouds, haze, bright sky or soft focus.
    if good.sum() >= 3:
        out["snr"] = float(np.percentile(res_peak[good] / sigma, 90))
        out["snr_n"] = int(good.sum())
    ys, xs, peaks = ys[good], xs[good], peaks[good]
    if ys.size == 0:
        return out
    # Brightest unsaturated stars give the most reliable fits
    sel = np.argsort(-peaks)[:MAX_FIT]
    ys, xs = ys[sel], xs[sel]

    off = np.arange(-HALF, HALF + 1)
    cuts = resid[(ys[:, None, None] + off[None, :, None]), (xs[:, None, None] + off[None, None, :])]
    p, ok = _moffat_fit(cuts)

    a_, b_, c_ = p[:, 4], p[:, 5], p[:, 6]
    tr = a_ + c_
    disc = np.sqrt(np.maximum((a_ - c_) ** 2 / 4 + b_ * b_, 0))
    lam_max = tr / 2 + disc            # largest curvature -> minor axis
    lam_min = tr / 2 - disc
    with np.errstate(invalid="ignore", divide="ignore"):
        f_minor = FWHM_PER_ALPHA / np.sqrt(lam_max)
        f_major = FWHM_PER_ALPHA / np.sqrt(lam_min)
        fwhm = np.sqrt(f_minor * f_major)
        if bayer:
            fwhm = np.sqrt(np.maximum(fwhm ** 2 - BOX2_VAR_FWHM2, 0.25))
            f_major = np.sqrt(np.maximum(f_major ** 2 - BOX2_VAR_FWHM2, 0.25))
            f_minor = np.sqrt(np.maximum(f_minor ** 2 - BOX2_VAR_FWHM2, 0.25))
        ecc = np.sqrt(np.clip(1 - (f_minor / f_major) ** 2, 0, 1))
        snr = p[:, 1] / sigma

    ok &= np.isfinite(fwhm) & (fwhm > 0.9) & (fwhm < 2 * HALF) & (snr > 5)
    if ok.sum() < 3:
        return out
    fwhm, ecc = fwhm[ok], ecc[ok]
    # Drop stars far from the typical size (galaxy cores, blends, hot pixels)
    med = np.median(fwhm)
    mad = np.median(np.abs(fwhm - med)) * 1.4826 + 1e-6
    keep = np.abs(fwhm - med) < 3 * mad + 0.1 * med
    if keep.sum() >= 3:
        fwhm, ecc = fwhm[keep], ecc[keep]
    out.update(fwhm=float(np.median(fwhm)), eccentricity=float(np.median(ecc)), fitted=int(fwhm.size))
    return out
