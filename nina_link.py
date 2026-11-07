"""
nina_link.py - working with N.I.N.A. (Nighttime Imaging 'N' Astronomy), the imaging program most of us use.

* Your rigs from N.I.N.A.'s profiles: every profile names its telescope, focal length, camera and filters - read
  from this computer's profiles, or from profile files copied from your rigs' own computers.
* The image file pattern check: a token N.I.N.A. doesn't know is left in the file names as it was typed (or as a
  bare '$$'), which usually means a typo in Options › Imaging › Image File Pattern.
* After the night: N.I.N.A.'s External Script instruction can start SubZeroAstro's Fast Track on the night's
  frames (subzeroastro --fast-track <folder>), so a finished picture is waiting in the morning.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# every token N.I.N.A. fills into a file name (Options › Imaging › Image File Pattern), plus the ones plugins add
TOKENS = {
    "APPLICATIONSTARTDATE", "DATE", "DATEMINUS12", "DATETIME", "DATEUTC", "TIME", "TIMEUTC",
    "BINNING", "CAMERA", "GAIN", "OFFSET", "READOUTMODE", "READOUTMODES", "SENSORTEMP", "TEMPERATURESETPOINT",
    "USBLIMIT", "ECCENTRICITY", "FWHM", "EXPOSURETIME", "FRAMENR", "HFR", "IMAGETYPE", "SEQUENCETITLE",
    "STARCOUNT", "TARGETNAME", "FILTER", "FOCUSERPOSITION", "FOCUSERTEMP", "PEAKDEC", "PEAKDECARCSEC", "PEAKRA",
    "PEAKRAARCSEC", "RMS", "RMSARCSEC", "ROTATORANGLE", "SQM", "TELESCOPE", "TSPROJECTNAME", "TSSESSIONID",
    "TSTARGETNAME", "TSFILTERNAME", "PIERSIDE", "RADEC", "MECHANICALROTATORANGLE", "SEQUENCETITLE",
}
SUGGESTED_PATTERN = r"$$TARGETNAME$$\$$DATEMINUS12$$\$$IMAGETYPE$$\$$FILTER$$\$$TARGETNAME$$_$$DATETIME$$_$$FILTER$$_$$EXPOSURETIME$$s_$$FRAMENR$$"


def _section(xml, tag):
    m = re.search(rf"<(?:\w+:)?{tag}\b[^>]*>(.*?)</(?:\w+:)?{tag}>", xml, re.S)
    return m.group(1) if m else ""


def _tag(xml, tag):
    m = re.search(rf"<(?:\w+:)?{tag}>([^<]*)</(?:\w+:)?{tag}>", xml)
    return _unxml(m.group(1).strip()) if m else ""


def _unxml(s):
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
            .replace("&amp;", "&"))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _top_name(xml):
    """The profile's own name: the <Name> that is a direct child of <Profile>, not a settings section's."""
    depth_names = re.finditer(r"<(/?)(?:\w+:)?(\w+)\b[^>]*?(/?)>", xml)
    depth = 0
    for m in depth_names:
        close, tag, selfclose = m.group(1), m.group(2), m.group(3)
        if close:
            depth -= 1
            continue
        if selfclose:
            continue
        if depth == 1 and tag == "Name":
            end = xml.find("<", m.end())
            return _unxml(xml[m.end():end].strip())
        depth += 1
    return ""


def parse_profile(xml: str, file: str = "") -> dict:
    """A N.I.N.A. .profile (XML) -> {profile, telescope, focal, camera, filters, patterns, image_path, used}."""
    tel = _section(xml, "TelescopeSettings")
    cam = _section(xml, "CameraSettings")
    fw = _section(xml, "FilterWheelSettings")
    img = _section(xml, "ImageFileSettings")
    filters = []
    for blk in re.findall(r"<(?:\w+:)?FilterInfo\b[^>]*>(.*?)</(?:\w+:)?FilterInfo>", fw, re.S):
        n = _tag(blk, "_name") or _tag(blk, "Name")
        if n and n not in filters:
            filters.append(n)
    camera = _tag(cam, "LastDeviceName") or _tag(cam, "Id")
    if camera.upper().startswith("ASCOM.") or camera.lower() in ("no_device", "none"):
        camera = ""
    patterns = {k: _tag(img, t) for k, t in (("light", "FilePattern"), ("flat", "FilePatternFLAT"),
                                             ("dark", "FilePatternDARK"), ("bias", "FilePatternBIAS"))}
    return {"profile": _top_name(xml) or (Path(file).stem if file else ""),
            "telescope": _tag(tel, "Name"), "focal": _num(_tag(tel, "FocalLength")), "camera": camera,
            "filters": filters, "patterns": {k: v for k, v in patterns.items() if v},
            "image_path": _tag(img, "FilePath"), "used": _tag(xml, "LastUsed"), "file": str(file)}


def profiles_folder() -> Path | None:
    la = os.environ.get("LOCALAPPDATA")
    return Path(la) / "NINA" / "Profiles" if la else None


def read_profiles(paths=None) -> list:
    """Profiles from these files (or folders), else from this computer's N.I.N.A.; newest used first."""
    files = []
    for p in (paths if paths is not None else [profiles_folder()] if profiles_folder() else []):
        p = Path(p)
        if p.is_dir():
            files += sorted(p.glob("*.profile"))
        elif p.is_file():
            files.append(p)
    out = []
    for f in files:
        try:
            d = parse_profile(f.read_text(encoding="utf-8", errors="replace"), str(f))
        except OSError:
            continue
        if d["focal"] or d["telescope"] or d["filters"]:
            out.append(d)
    out.sort(key=lambda d: d["used"], reverse=True)
    return out


def rig_name(d) -> str:
    """A rig name for a profile: its telescope name (Options › Equipment › Telescope), else the profile's."""
    n = (d.get("telescope") or "").strip() or (d.get("profile") or "").strip()
    return re.sub(r"[\\/:*?\"<>|]+", " ", n).strip() or "N.I.N.A. rig"


def rigs_from_profiles(profiles, existing=(), tol=0.03) -> list:
    """Your Rigs rows for these profiles: name, focal length, and the camera only when another rig has nearly
    the same focal length (the only time Your Rigs needs it)."""
    rows = []
    for d in profiles:
        if not d.get("focal") or d["focal"] <= 0:
            continue
        rows.append({"name": rig_name(d), "focals": [round(float(d["focal"]), 1)], "camera": d.get("camera") or ""})
    allr = list(existing) + rows
    for r in rows:
        f = r["focals"][0]
        twins = [x for x in allr if x is not r and x.get("name") != r["name"]
                 and any(abs(float(g) - f) <= tol * f for g in x.get("focals") or [])]
        if not twins:
            r["camera"] = ""
    return rows


def check_pattern(pattern: str) -> list:
    """Problems in a N.I.N.A. image file pattern: tokens N.I.N.A. doesn't know (they end up in the file names as
    typed) and stray '$$'. [] = fine."""
    if not pattern:
        return []
    out = []
    rest = pattern
    for m in re.finditer(r"\$\$([A-Za-z0-9_]*)\$\$", pattern):
        tok = m.group(1)
        if tok.upper() not in TOKENS:
            out.append(f"$${tok}$$ is not a N.I.N.A. token" if tok else "'$$$$' (an empty token)")
    rest = re.sub(r"\$\$[A-Za-z0-9_]*\$\$", "", pattern)
    if "$$" in rest:
        out.append("a '$$' without its partner (an unfinished token)")
    return out


def odd_file_names(paths, limit=5) -> list:
    """File names with N.I.N.A. token leftovers in them ('$$')."""
    out = []
    for p in paths:
        if "$$" in Path(p).name:
            out.append(Path(p).name)
            if len(out) >= limit:
                break
    return out


def fast_track_command(folder, hours=18, exe=None) -> str:
    """What to put in N.I.N.A.'s External Script instruction (at the end of the sequence)."""
    if exe is None:
        if getattr(sys, "frozen", False):
            exe = f'"{sys.executable}"'
        else:
            exe = f'"{sys.executable}" "{Path(__file__).resolve().parent / "subzeroastro.py"}"'
    else:
        exe = f'"{exe}"'
    return f'{exe} --fast-track "{folder}" --hours {int(hours)}'


def parse_cli(argv) -> dict | None:
    """subzeroastro --fast-track <folder> [--hours N] [--target NAME] [--out DIR] -> options (None if absent)."""
    if "--fast-track" not in argv:
        return None
    a = list(argv)
    i = a.index("--fast-track")
    opt = {"folder": a[i + 1] if len(a) > i + 1 and not a[i + 1].startswith("--") else "", "hours": 18.0,
           "target": "", "out": ""}
    for key, flag in (("hours", "--hours"), ("target", "--target"), ("out", "--out")):
        if flag in a and len(a) > a.index(flag) + 1:
            opt[key] = a[a.index(flag) + 1]
    try:
        opt["hours"] = float(opt["hours"])
    except (TypeError, ValueError):
        opt["hours"] = 18.0
    return opt


def recent_frames(folder, hours=18.0, now=None) -> list:
    """The night's frames: FITS/XISF files under folder changed in the last `hours` (0 = all of them)."""
    import time
    import fits_core as core
    t0 = (now if now is not None else time.time()) - hours * 3600 if hours and hours > 0 else None
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_SubZeroAstro"))]
        for n in files:
            p = Path(root) / n
            if p.suffix.lower() not in core.LIGHT_EXTS:
                continue
            try:
                if t0 is None or p.stat().st_mtime >= t0:
                    out.append(p)
            except OSError:
                continue
    return sorted(out)
