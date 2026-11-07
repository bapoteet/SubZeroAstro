"""
framing.py - keep the same framing next session.

A plate-solved master (or any solved FITS / XISF) knows exactly where it points and how the camera is turned. From
its solution we work out the centre of the picture and the camera's position angle the way N.I.N.A. reports them,
and write a N.I.N.A. target (Deep Sky Object container) with both - so next session "Slew, center and rotate" (or
the Framing Assistant) puts the camera back on the same patch of sky. Not solved yet: ASTAP solves it first.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def wcs_of(h: dict) -> dict | None:
    """{crval1, crval2, crpix1, crpix2, cd: (cd11, cd12, cd21, cd22)} from a header, or None if not solved."""
    if not h:
        return None
    c1, c2 = _f(h.get("CRVAL1")), _f(h.get("CRVAL2"))
    p1, p2 = _f(h.get("CRPIX1")), _f(h.get("CRPIX2"))
    if None in (c1, c2, p1, p2):
        return None
    cd = tuple(_f(h.get(k)) for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    if None in cd or not any(cd):
        d1, d2 = _f(h.get("CDELT1")), _f(h.get("CDELT2"))
        if not d1 or not d2:
            return None
        r = math.radians(_f(h.get("CROTA2")) or 0.0)
        cd = (d1 * math.cos(r), -d2 * math.sin(r), d1 * math.sin(r), d2 * math.cos(r))
    return {"crval1": c1, "crval2": c2, "crpix1": p1, "crpix2": p2, "cd": cd}


def pixel_to_sky(w: dict, x: float, y: float) -> tuple:
    """FITS pixel (1-based) -> (RA, Dec) in degrees, gnomonic (TAN) projection."""
    a, b, c, d = w["cd"]
    dx, dy = x - w["crpix1"], y - w["crpix2"]
    xi = math.radians(a * dx + b * dy)
    eta = math.radians(c * dx + d * dy)
    ra0, dec0 = math.radians(w["crval1"]), math.radians(w["crval2"])
    den = math.cos(dec0) - eta * math.sin(dec0)
    ra = ra0 + math.atan2(xi, den)
    dec = math.atan2(math.sin(dec0) + eta * math.cos(dec0), math.hypot(xi, den))
    return math.degrees(ra) % 360.0, math.degrees(dec)


def nina_position_angle(w: dict) -> tuple:
    """(position angle °, flipped) exactly as N.I.N.A. turns an ASTAP solution into one
    (NINA.Astrometry.WorldCoordinateSystem + ASTAPSolver: PA = 360 - (rotation - 180))."""
    cd11, cd12, cd21, cd22 = w["cd"]
    det = cd11 * cd22 - cd12 * cd21
    sign = -1 if det < 0 else 1
    cdelta1 = sign * math.hypot(cd11, cd21)
    cdelta2 = math.hypot(cd12, cd22)
    flipped = cdelta1 >= 0 or cdelta2 < 0
    rot2 = math.atan2(sign * cd11, cd21) - math.pi / 2
    rotation = math.degrees(-rot2 if flipped else rot2) % 360.0
    pa = (360.0 - (rotation - 180.0)) % 360.0
    return round(pa, 2) % 360.0, not flipped


def pixel_scale(w: dict) -> float:
    """arcsec per pixel."""
    _a, b, _c, d = w["cd"]
    return math.hypot(b, d) * 3600.0


def framing(h: dict, width: int, height: int, name: str = "") -> dict | None:
    """Centre, position angle and size of a solved picture: {ra, dec, pa, flipped, scale, fov_w, fov_h, name}."""
    w = wcs_of(h)
    if w is None:
        return None
    ra, dec = pixel_to_sky(w, (width + 1) / 2.0, (height + 1) / 2.0)
    pa, flipped = nina_position_angle(w)
    sc = pixel_scale(w)
    nm = (name or str(h.get("OBJECT") or "")).strip() or "Target"
    return {"ra": ra, "dec": dec, "pa": pa, "flipped": flipped, "scale": sc,
            "fov_w": width * sc / 60.0, "fov_h": height * sc / 60.0, "name": nm}


def read(path) -> tuple:
    """(header dict, width, height) of a FITS / XISF file."""
    import fits_core as core
    import stacking as st
    src = st.ImageSource(path)
    look = dict(core.cards_to_dict(getattr(src, "cards", None) or []) or {})
    look.update({k: v for k, v in dict(src.look or {}).items() if k not in look})
    return look, int(src.w), int(src.h)


def hints(h: dict) -> tuple:
    """(RA°, Dec°) the file says it points at (mount position), else (None, None)."""
    import target_picker as tp
    for rk, dk in (("RA", "DEC"), ("OBJCTRA", "OBJCTDEC")):
        ra, dec = h.get(rk), h.get(dk)
        if ra in (None, "") or dec in (None, ""):
            continue
        r, d = _f(ra), _f(dec)
        try:
            if r is None:
                r = tp._sexagesimal(str(ra)) * 15.0
            if d is None:
                d = tp._sexagesimal(str(dec))
        except ValueError:
            continue
        return r, d
    return None, None


def solve(path, astap_exe, log=None, cancel=None) -> dict:
    """Plate solve a copy of the file with ASTAP (so nothing is written next to your master) -> WCS header dict."""
    import fits_core as core
    import tools
    h, _w, ht = read(path)
    ra, dec = hints(h)
    fl, px = _f(h.get("FOCALLEN")), _f(h.get("XPIXSZ") or h.get("PIXSIZE1"))
    fov = (ht * px / 1000.0 / fl) * 57.29578 if fl and px else None
    tmp = Path(tempfile.mkdtemp(prefix="saz_frame_"))
    try:
        copy = tmp / ("frame" + Path(path).suffix.lower())
        shutil.copy2(path, copy)
        cards = tools.ASTAP(astap_exe).solve(copy, ra, dec, fov, radius=30.0 if ra is not None else 180.0,
                                              log=log, cancel=cancel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return core.cards_to_dict(cards)


def from_file(path, astap_exe=None, log=None) -> dict:
    """framing() of a file, solving it with ASTAP first if it isn't solved. Raises ValueError with the reason."""
    h, w, ht = read(path)
    fr = framing(h, w, ht)
    if fr is not None:
        fr["solved_now"] = False
        return fr
    if not astap_exe:
        raise ValueError("this picture isn't plate solved, and ASTAP (which would solve it) isn't installed. "
                         "Use a master from Astro Image Processing with plate solving on, or install ASTAP.")
    h2 = dict(h)
    h2.update(solve(path, astap_exe, log=log))
    fr = framing(h2, w, ht)
    if fr is None:
        raise ValueError("ASTAP couldn't solve it.")
    fr["solved_now"] = True
    return fr


def as_target(fr: dict):
    """Something target_picker.nina_target_json can write (name, coordinates, position angle)."""
    return SimpleNamespace(ra=fr["ra"], dec=fr["dec"], name=fr["name"], common="", pa=fr["pa"])


def describe(fr: dict) -> str:
    import target_picker as tp
    return (f"{fr['name']}: centre {tp.nina_coords(as_target(fr))} (J2000), position angle {fr['pa']:.1f}°"
            f"; {fr['fov_w']:.0f}′ × {fr['fov_h']:.0f}′ at "
            f"{fr['scale']:.2f}″/px")
