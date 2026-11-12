"""A colour picture of the sky around a target, for Tonight's Target Picker: a cut-out of the DSS2 colour survey
made by the CDS hips2fits service (Strasbourg), at the target's position and a field just bigger than the target
or your frame. Pictures are kept on disk, so a target already seen shows at once, also without internet.

This research made use of hips2fits, a service provided by CDS."""
from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

SERVERS = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
           "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits")
SURVEY = "CDS/P/DSS2/color"
SIDE = 480                       # pixels of the (square) picture
CREDIT = "DSS2 colour survey · CDS hips2fits"


def field_deg(size_arcmin, frame_arcmin=None) -> float:
    """How much sky the picture shows (degrees, square): the target with room around it, and your whole frame
    when that is known - so you see both how big the target is and where your frame's edges fall."""
    want = max(float(size_arcmin or 0) * 1.6, 12.0)               # at least 12' - a small galaxy still shows
    if frame_arcmin:
        want = max(want, max(frame_arcmin) * 1.25)
    return float(min(max(want / 60.0, 0.2), 10.0))


def url(ra, dec, fov_deg, side=SIDE, server=SERVERS[0]) -> str:
    q = {"hips": SURVEY, "ra": f"{float(ra):.5f}", "dec": f"{float(dec):.5f}", "fov": f"{float(fov_deg):.4f}",
         "width": int(side), "height": int(side), "projection": "TAN", "coordsys": "icrs", "format": "jpg"}
    return server + "?" + urllib.parse.urlencode(q)


def cache_file(ra, dec, fov_deg, side=SIDE) -> Path:
    import fits_core as core
    d = core.cache_dir() / "sky_images"
    d.mkdir(parents=True, exist_ok=True)
    key = f"{SURVEY}|{float(ra):.4f}|{float(dec):.4f}|{float(fov_deg):.3f}|{int(side)}"
    return d / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".jpg")


def fetch(ra, dec, fov_deg, side=SIDE, timeout=25.0, opener=None):
    """(jpeg bytes, None) or (None, why). From the disk cache when it is there; else asks CDS (the mirror if the
    main server fails). opener(url, timeout) -> bytes stands in for the network in the self-test."""
    cf = cache_file(ra, dec, fov_deg, side) if opener is None else None   # (the self-test's pictures aren't kept)
    try:
        if cf is not None and cf.is_file() and cf.stat().st_size > 1000:
            return cf.read_bytes(), None
    except OSError:
        pass
    why = ""
    for srv in SERVERS:
        u = url(ra, dec, fov_deg, side, srv)
        try:
            if opener is not None:
                data = opener(u, timeout)
            else:
                import target_info as ti
                req = urllib.request.Request(u, headers={"User-Agent": ti.UA})
                with urllib.request.urlopen(req, timeout=timeout, context=ti.ssl_context()) as r:
                    data = r.read()
        except Exception as e:  # noqa: BLE001 - try the mirror, then say why
            why = str(e) or e.__class__.__name__
            continue
        if not data or data[:2] != b"\xff\xd8":          # not a JPEG: an error page
            why = (data[:120].decode("utf-8", "replace").strip() if data else "empty answer") or "not a picture"
            continue
        if cf is not None:
            try:
                tmp = cf.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(cf)
            except OSError:
                pass
        return data, None
    return None, why or "no answer"
