"""Organize My Astro Files: sort a folder (or a whole drive or NAS share) of FITS / XISF files into one tidy
layout, read from each file's own header - the layout the book recommends:

    <library>/
      Targets/<Target>/<night>/LIGHT/<filter>/       subs (rejected ones in LIGHT/<filter>/REJECTED)
      Targets/<Target>/<night>/FLAT/<filter>/        flats of that night (and DARKFLAT/)
      Targets/<Target>/Masters/                      master lights (page 2 / page 3) and their companions
      Darks library/<camera>_G<gain>_<temp>C/<exp>s/ darks, by camera, gain, temperature and exposure
      Bias library/<camera>_G<gain>_<temp>C/
      Calibration masters/                           master darks, flats, bias
      Flats/<night>/<filter>/                        flats that can't be tied to one target
      Unsorted/                                      files whose header doesn't say what they are

Nothing is touched until you have seen the plan. Copy or move; every run writes an undo log, and
undo puts moved files back (or removes the copies). Nothing is ever deleted: duplicates are skipped and
reported, name clashes get a number added. Saved SubZeroAstro projects and the library's own folders are
left alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import fits_core as core

LOG_DIR = "_SubZeroAstro_organize"
ORG_ROOT = "Organized Astro Files"     # everything the organizer makes goes in here, so you can see where


def org_base(library) -> Path:
    """The folder the organized files go into: <library>/Organized Astro Files (or the library itself when you
    picked that folder already)."""
    library = Path(library)
    return library if library.name == ORG_ROOT else library / ORG_ROOT
TOP = ("Light Frames", "Calibration Frames", "Targets", "Darks library", "Bias library", "Calibration masters", "Flats", "Unsorted", LOG_DIR)
EXTS = {e.lower() for e in core.LIGHT_EXTS}
# in-between files of PixInsight WBPP / Siril / APP runs (calibrated, registered … copies of your subs) and
# system folders: left alone - they are not your originals and can be made again
WORK_DIRS = {"calibrated", "registered", "debayered", "cosmetized", "measured", "fastintegration", "drizzle",
             "process", "aligned", "normalized", "localnormalization", "system volume information"}
COMPANION_EXTS = {".txt", ".json", ".csv", ".log"}


@dataclass
class Item:
    src: Path
    dst: Path | None = None
    kind: str = ""                      # light / flat / darkflat / dark / bias / master / calmaster / unsorted
    target: str = ""
    night: str = ""
    filter: str = ""
    why: str = ""                       # why it goes there (shown in the preview)
    status: str = "move"                # move / here (already in place) / duplicate / error
    size: int = 0
    companions: list = field(default_factory=list)
    key: tuple | None = None             # same capture (time, exposure, filter, camera, size) = the same frame
    same_as: Path | None = None         # for a duplicate: the copy that is kept
    info: object = field(default=None, repr=False, compare=False)   # the header read (kept for re-planning)
    rig_issue: str = ""                  # why no rig could be matched: nofocal / nomatch / ambiguous: A, B


def _safe(s) -> str:
    s = core._ascii(str(s or ""))
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().rstrip(".")
    return s or "Unknown"


def _num_text(v, fmt="{:g}") -> str:
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "?"


def _known(v):
    try:
        return v is not None and v != "" and float(v) == float(v)
    except (TypeError, ValueError):
        return False


def _cal_folder(info) -> str:
    """ASI2600MM_G100_O50_-10C - only the parts the header gives (PixInsight masters often have no gain or
    temperature), never a '?' (Windows can't name a folder with one)."""
    import stacking as st
    cam = st.short_camera(info.camera)
    parts = [cam]
    if _known(info.gain):
        parts.append(f"G{_num_text(info.gain)}")
    if _known(getattr(info, "offset", None)):
        parts.append(f"O{_num_text(info.offset)}")
    if _known(info.temp):
        parts.append(f"{_num_text(info.temp, '{:.0f}')}C")
    return _safe("_".join(parts))


def _exp_folder(info, name="") -> str:
    """180s - from the header, or from a name such as masterDark_..._EXPOSURE-120.00s.xisf (PixInsight)."""
    v = info.exptime if _known(info.exptime) else None
    if v is None:
        m = re.search(r"(?i)(?:exposure|exp)[-_ ]?(\d+(?:\.\d+)?)\s*s", name or "") or \
            re.search(r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*s(?:ec)?(?![a-z])", name or "")
        v = float(m.group(1)) if m else None
    return f"{_num_text(v)}s" if v is not None else "Unknown exposure"


_RIGS = None          # SubZeroAstro's own rig names (Calibrate and Stack › rig names), set by classify()
_GUESS = {}           # (camera, focal) -> rig name read from your folder names, set by classify()


# ---------------------------------------------------------------- Your Rigs (name ↔ focal length)
_RIGLIST = None       # [{"name", "focals": [mm], "camera": "", "folders": [paths]}] - when set, the only source
_TOL = 0.03           # a rig's focal length in the headers drifts a few % night to night (focusing, solving)


def _cam_norm(c):
    import stacking as st
    return re.sub(r"[\W_]", "", st.short_camera(c or "")).lower() if c else ""


def match_rig(info, path, riglist, tol=0.03):
    """(rig name or None, reason). A folder you assigned to a rig wins; otherwise the focal length in the header
    within ±tol of one of a rig's focal lengths (and the camera, when the rig lists one). reason:
    'folder' / 'focal' / 'nofocal' / 'nomatch' / 'ambiguous: A, B'."""
    sp = str(path).replace("\\", "/").lower() if path is not None else ""
    best = None
    for r in riglist or []:
        for f in r.get("folders") or []:
            fp = str(f).replace("\\", "/").rstrip("/").lower() + "/"
            if sp.startswith(fp) and (best is None or len(fp) > best[1]):
                best = (r["name"], len(fp))
    if best:
        return best[0], "folder"
    fl = getattr(info, "focal", None)
    if not fl:
        return None, "nofocal"
    cam = _cam_norm(getattr(info, "camera", None))
    hits = {}
    for r in riglist or []:
        if r.get("camera") and cam and _cam_norm(r["camera"]) != cam:
            continue
        ds = [abs(float(fl) - float(f)) / float(f) for f in r.get("focals") or [] if f]
        ds = [d for d in ds if d <= tol]
        if ds:
            hits[r["name"]] = min(hits.get(r["name"], 9), min(ds))
    if len(hits) == 1:
        return next(iter(hits)), "focal"
    if len(hits) > 1:
        # within reach of two rigs (e.g. 506 and 530 mm scopes): the clearly nearer one wins (at most half the
        # distance of the next); a reading right between them is asked about
        near = sorted(hits.items(), key=lambda kv: kv[1])
        if near[0][1] <= 0.5 * near[1][1]:
            return near[0][0], "focal"
    hits = list(hits)
    if not hits:
        return None, "nomatch"
    return None, "ambiguous: " + ", ".join(hits)


def _rig(info, path=None) -> str:
    """Which rig (telescope + the camera that lives on it) a file comes from: the name you gave the rig in
    SubZeroAstro, else the telescope in the header (not a mount driver such as EQMOD), else the focal length,
    else - for calibration frames sorted by hand - the folder above the Darks / Flats / Bias folder."""
    import stacking as st
    if _RIGLIST is not None:                       # Your Rigs: the list you made, nothing guessed
        name, _why = match_rig(info, path, _RIGLIST, _TOL)
        return _safe(re.sub(r"\s+", "", name))[:24] if name else ""
    try:
        if _RIGS is not None:
            named = _RIGS.lookup(info.camera, info.focal)
            if named:
                return _safe(re.sub(r"\s+", "", named))[:24]
    except Exception:  # noqa: BLE001
        pass
    t = (info.telescope or "").strip()
    if t and st.Rigs.suggestion(info) == st.safe_name(t):          # a telescope, not a mount driver
        return _safe(re.sub(r"\s+", "", t))[:24]
    g = _GUESS.get(_rig_key(info))
    if g:                                                           # your own folder name for this rig
        return _safe(re.sub(r"\s+", "", g))[:24]
    if info.focal:
        return f"{info.focal:.0f}mm"
    if path is not None:
        ps = list(Path(path).parents)
        for k, anc in enumerate(ps[:5]):
            if st.folder_kind(anc.name) and k + 1 < len(ps):
                up = ps[k + 1].name
                if up and not re.match(r"(?i)^(calibration|cal|calib|astro.*|incoming.*|data|frames?)$", up) \
                        and not re.match(r"^[A-Za-z]:\\?$", up):
                    return _safe(re.sub(r"\s+", "", up))[:24]
                break
    return ""


def _exists(p) -> bool:
    """exists() that never raises: some drives answer 'Invalid argument' (Errno 22) for odd entries."""
    try:
        return Path(p).exists()
    except OSError:
        return False


def _same_path(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def _is_project_dir(d: Path) -> bool:
    m = d / "manifest.json"
    try:
        if not m.is_file():
            return False
        return json.loads(m.read_text(encoding="utf-8")).get("kind") == "finish-project"
    except (OSError, ValueError):
        return False


SKIPPED = []          # [(folder, why)] - folders the last scan() did not look into (shown after a run)


def _why_skipped(p: Path, n: str) -> str:
    if n == LOG_DIR:
        return ""
    if n.startswith((".", "$")):
        return "hidden / system folder"
    if n.endswith((".szproj", ".saving")) or _is_project_dir(p):
        return "a saved SubZeroAstro project"
    if n.lower() in WORK_DIRS:
        return "a work folder of PixInsight / Siril / APP (calibrated, registered … copies)"
    return ""


def scan(root, progress=None, stop=None) -> list:
    """Every FITS / XISF file under root (all subfolders), skipping saved projects, SubZeroAstro's own
    work folders and the organize logs. Returns [Path]."""
    root = Path(root)
    out = []
    if not getattr(scan, "_keep_skipped", False):
        SKIPPED.clear()
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        keep = []
        for n in sorted(dirnames, key=core.natural_key):
            p = d / n
            if (n.startswith((".", "$")) or n == LOG_DIR or n.endswith((".szproj", ".saving"))
                    or n.lower() in WORK_DIRS or _is_project_dir(p)):
                why = _why_skipped(p, n)
                if why and n != ORG_ROOT:
                    SKIPPED.append((p, why))
                continue
            keep.append(n)
        dirnames[:] = keep
        for f in sorted(filenames, key=core.natural_key):
            if os.path.splitext(f)[1].lower() in EXTS and not core.is_hidden_file(f):
                out.append(d / f)
        if progress:
            progress(f"Looking in {d}", len(out))
        if stop and stop():
            break
    return out


def _companions(p: Path) -> list:
    """Small files that belong with a master (stack log, notes) - named after it."""
    out = []
    try:
        for sib in p.parent.glob(p.stem + "_*"):
            if sib.suffix.lower() in COMPANION_EXTS and _exists(sib) and sib.is_file():
                out.append(sib)
    except OSError:
        pass
    return out


LAYOUTS = {"rig": "By rig: Light Frames / rig / target / night  and  Calibration Frames / rig / Flats, Darks, Bias",
           "target": "By target: Targets / target / night_rig, with darks and bias libraries"}


def classify(paths, read_info=None, progress=None, stop=None, library=None, rigs=None, layout="rig",
             root=None, riglist=None, tol=0.03) -> list:
    """Read each header and decide where the file belongs (relative to the library). Returns [Item].
    rigs: SubZeroAstro's saved rig names (stacking.Rigs), so a rig you named keeps its name."""
    import stacking as st
    global _RIGS, _RIGLIST, _TOL
    _RIGS = rigs
    _RIGLIST, _TOL = riglist, tol
    read_info = read_info or st.read_info
    items, infos = [], []
    for i, p in enumerate(paths):
        if progress and i % 20 == 0:
            progress(f"Reading headers {i + 1} of {len(paths)}", i)
        if stop and stop():
            break
        it = Item(src=Path(p))
        try:
            it.size = Path(p).stat().st_size
            info = read_info(p)
        except Exception as e:  # noqa: BLE001 - unreadable file: leave it where it is
            it.status, it.why = "error", f"could not read it ({e})"
            items.append(it)
            infos.append(None)
            continue
        it.info = info
        items.append(it)
        infos.append(info)

    # rigs without a name or a telescope in the header: the folder name their files sit under (SQA60Pro/…)
    global _GUESS
    _GUESS = {}
    for it in items:
        it.target = st.target_name(it.info.target) if (it.info is not None and it.info.target) else ""
    for c in rig_candidates(items, rigs, root=root if root is not None else library):
        if not c["named"] and c["guess"]:
            _GUESS[c["key"]] = c["guess"]

    # which targets had lights on which night (with which telescope) - flats without a target go there
    lights_on, rigs, existing = {}, {}, {}
    for it, info in zip(items, infos):
        if info is not None and info.type == "light" and not info.master and info.target and info.night:
            lights_on.setdefault(info.night, set()).add((st.target_name(info.target), _rig(info, it.src)))
            rigs.setdefault(st.target_name(info.target), set()).add(_rig(info, it.src))
    # nights already in the library (organized before): flats sorted on their own - a folder of calibration
    # frames only - still find the target that was shot that night
    if library is not None:
        tdir = org_base(library) / "Targets"
        try:
            for td in (tdir.iterdir() if tdir.is_dir() else []):
                if not td.is_dir():
                    continue
                for nd in td.iterdir():
                    m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:_(.+))?$", nd.name)
                    if m and nd.is_dir() and (nd / "LIGHT").is_dir():
                        lights_on.setdefault(m.group(1), set()).add((td.name, m.group(2) or ""))
                        existing.setdefault((td.name, m.group(1)), []).append((m.group(2) or "", nd.name))
                        if m.group(2):
                            rigs.setdefault(td.name, set()).add(m.group(2))
        except OSError:
            pass

    def session(tgt, info, path=None):
        """The night's folder, always named with the rig when it is known (2026-10-01_SQA60): a target shot with
        several rigs never mixes their lights and flats, and you can see at a glance which scope took them."""
        r = _rig(info, path)
        return f"{night}_{r}" if r else night

    for it, info in zip(items, infos):
        if info is None:
            continue
        if _RIGLIST is not None:
            rname, why = match_rig(info, it.src, _RIGLIST, _TOL)
            if not rname:                        # not filed under a guess: it waits until you say which rig
                fl = f"{info.focal:.0f} mm" if info.focal else "no focal length"
                it.status, it.dst, it.rig_issue = "norig", None, why
                it.why = {"nofocal": "no focal length in the header - which rig? (Which Rig…)",
                          "nomatch": f"no rig in Your Rigs matches {fl} - add it (Your Rigs…)"}.get(
                    why, f"{fl} fits more than one rig ({why.split(': ', 1)[-1]}) - which one? (Which Rig…)")
                continue
        name = it.src.name
        rejected = it.src.parent.name.upper() == core.REJECTED_DIR
        typ = info.type
        if typ is None:                 # no IMAGETYP: a folder above it may say (Flats/Ha/…, Darks/300s/…)
            for anc in list(it.src.parents)[1:4]:
                typ = st.folder_kind(anc.name)
                if typ:
                    break
        flt = _safe(info.filter or "No filter")
        night = info.night or "Unknown date"
        tgt = st.target_name(info.target) if info.target else ""
        it.night, it.filter, it.target = night, flt, tgt
        # masters and processed results (a colour 3-plane image, several frames combined, or a name that says so)
        # go to the target's Masters folder - never into a night's LIGHT folder as if they were subs
        looks_master = bool(info.master or "_prepared" in it.src.stem or re.search(r"master", name, re.I)
                            or (getattr(info, "planes", 1) or 1) >= 3 or (info.ncombine or 0) > 1
                            or re.search(r"(final|processed|stretched|integration|drizzle|stack(ed)?|_linear|"
                                         r"starless|_stars)", name, re.I))
        if info.date_obs and not looks_master:
            # (masters are only ever duplicates when byte for byte the same - two integrations of the same
            #  flats share their capture date but are different files, and both are kept)
            # rigs shooting at the same moment with the same camera model are not duplicates of each other
            it.key = (info.date_obs, info.exptime, flt, info.camera, it.size, typ, looks_master,
                      info.telescope, info.focal, str(it.src.parent.parent.parent.name) if not info.telescope
                      and not info.focal else "")
        if layout == "rig":
            _rig_layout(it, info, typ, looks_master, tgt, night, flt, name, rejected)
            continue
        if typ in (None, "light") and looks_master and tgt:
            it.kind, it.dst = "master", Path("Targets") / tgt / "Masters" / name
            it.why = "master light"
            it.companions = _companions(it.src)
        elif typ in ("dark", "flat", "darkflat", "bias") and looks_master:
            # per rig: two cameras of the same model are still two different sensors (hot pixels, amp glow)
            base = Path(_rig(info, it.src) or ".") / _cal_folder(info)
            sub = base if typ in ("dark", "bias") else base / f"Flats {night}" / flt
            it.kind, it.dst = "calmaster", Path("Calibration masters") / sub / name
            it.why = f"master {typ}"
            it.companions = _companions(it.src)
        elif typ == "light" or (typ is None and tgt and info.exptime):
            if not tgt:
                it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "a light without a target name"
            else:
                sub = Path("Targets") / tgt / session(tgt, info, it.src) / "LIGHT" / flt
                if rejected:
                    sub = sub / core.REJECTED_DIR
                it.kind, it.dst = "light", sub / name
                it.why = "light" + (" (rejected on page 1)" if rejected else "")
        elif typ in ("flat", "darkflat"):
            folder = "FLAT" if typ == "flat" else "DARKFLAT"
            owners = sorted(lights_on.get(night, set()))
            frig = _rig(info, it.src)
            same_scope = {t for t, r in owners if frig and r == frig}
            # a target shot that night with this scope - or with a scope we can't tell apart from it
            maybe = {t for t, r in owners if not frig or not r or r == frig}
            if tgt and any(t == tgt for t, _ in owners):
                owner = tgt
            elif len(same_scope) == 1:
                owner = next(iter(same_scope))
            elif len(maybe) == 1:
                owner = next(iter(maybe))
            else:
                owner = None
            if owner:
                there = existing.get((owner, night), [])
                pick = [d for r, d in there if r == frig] or ([there[0][1]] if len(there) == 1 else [])
                nightdir = pick[0] if pick else session(owner, info, it.src)
                it.kind, it.dst = typ, Path("Targets") / owner / nightdir / folder / flt / name
                it.why = f"{typ} of the night ({owner})"
            else:
                scope = frig or st.short_camera(info.camera)
                it.kind, it.dst = typ, Path("Flats") / scope / night / (folder if typ == "darkflat" else "") / flt / name
                it.why = (f"{typ} - several targets that night, kept by scope" if owners else
                          f"{typ} - no lights from that night here yet, kept by scope and night")
        elif typ == "dark":
            it.kind = "dark"
            it.dst = Path("Darks library") / (_rig(info, it.src) or ".") / _cal_folder(info) / \
                _exp_folder(info, name) / name
            it.why = "dark (library by rig, then camera settings and exposure)"
        elif typ == "bias":
            it.kind, it.dst = "bias", Path("Bias library") / (_rig(info, it.src) or ".") / _cal_folder(info) / name
            it.why = "bias (library by rig, then camera settings)"
        else:
            it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "the header doesn't say what it is"
    if progress:
        progress("Headers read", len(paths))
    return items


def _rig_layout(it, info, typ, looks_master, tgt, night, flt, name, rejected):
    """Light Frames / <rig> / <target> / <night> / <filter>   and   Calibration Frames / <rig> / Flats | Dark flats | Darks |
    Bias | Masters - everything a rig (one telescope with its own camera) took, in one place."""
    import stacking as st
    rig = _rig(info, it.src) or "Unknown rig"
    cal = Path("Calibration Frames") / rig
    if typ in (None, "light") and looks_master and tgt:
        it.kind, it.dst, it.why = "master", Path("Light Frames") / rig / tgt / "Masters" / name, "master light"
        it.companions = _companions(it.src)
    elif typ in ("dark", "flat", "darkflat", "bias") and looks_master:
        sub = Path(_cal_folder(info)) if typ in ("dark", "bias") else Path(f"Flats {night}") / flt
        it.kind, it.dst, it.why = "calmaster", cal / "Masters" / sub / name, f"master {typ} ({rig})"
        it.companions = _companions(it.src)
    elif typ == "light" or (typ is None and tgt and info.exptime):
        if not tgt:
            it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "a light without a target name"
            return
        sub = Path("Light Frames") / rig / tgt / night / flt
        if rejected:
            sub = sub / core.REJECTED_DIR
        it.kind, it.dst = "light", sub / name
        it.why = f"light ({rig})" + (" - rejected on page 1" if rejected else "")
    elif typ in ("flat", "darkflat"):
        it.kind = typ
        it.dst = cal / ("Flats" if typ == "flat" else "Dark flats") / st.short_camera(info.camera) / night / flt / name
        it.why = f"{'flat' if typ == 'flat' else 'dark flat'} of {night} ({rig})"
    elif typ == "dark":
        it.kind = "dark"
        it.dst = cal / "Darks" / _cal_folder(info) / _exp_folder(info, name) / name
        it.why = f"dark ({rig}: camera settings, exposure)"
    elif typ == "bias":
        it.kind, it.dst, it.why = "bias", cal / "Bias" / _cal_folder(info) / name, f"bias ({rig}: camera settings)"
    else:
        it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "the header doesn't say what it is"


def _quick_hash(p: Path) -> str:
    """Size + the first and last MB: enough to tell real duplicates from different files of one name."""
    h = hashlib.sha1()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()


def plan(items, library) -> list:
    """Final destinations: already in place / duplicate of a file already there / a numbered name if a
    different file of that name is in the way (also between files of this same run)."""
    library = Path(library)
    taken = {}
    # the same frame twice (a copy under another name): the one already in the library - or else the
    # first one found - is kept, the others are left where they are and reported
    seen = {}
    def keep_first(x):
        in_place = x.dst is not None and _same_path(org_base(library) / x.dst, x.src)
        copyish = bool(re.search(r"copy|\(\d+\)|_\d+$", x.src.stem, re.I))
        return (0 if in_place else 1, 1 if copyish else 0, core.natural_key(str(x.src)))
    for it in sorted(items, key=keep_first):
        if it.key is None or it.status == "error":
            continue
        if it.key in seen:
            it.status, it.why = "duplicate", f"the same frame as {seen[it.key].name}"
            it.same_as = seen[it.key]
        else:
            seen[it.key] = it.src
    for it in items:
        if it.dst is None or it.status in ("error", "duplicate", "norig"):
            continue
        it.dst = Path(*[_safe(x) if k < len(Path(it.dst).parts) - 1 else x
                        for k, x in enumerate(Path(it.dst).parts)])     # no ? : * … in folder names (Windows)
        dst = org_base(library) / it.dst
        if _same_path(dst, it.src):
            it.status, it.dst = "here", dst
            continue
        n = 1
        cand = dst
        while True:
            key = str(cand).lower()
            if key in taken:
                if taken[key] == it.size and _quick_hash(it.src) == _quick_hash(Path(taken[key + "#src"])):
                    it.status, it.same_as = "duplicate", Path(taken[key + "#src"])
                    it.why = f"the same file as {it.same_as.name}"
                    break
            elif _exists(cand):
                try:
                    if cand.stat().st_size == it.size and _quick_hash(cand) == _quick_hash(it.src):
                        it.status, it.same_as = "duplicate", cand
                        it.why = f"the same file as the one already in {cand.parent.name}"
                        break
                except OSError:
                    pass
            else:
                break
            n += 1
            cand = dst.with_name(f"{dst.stem}_{n}{dst.suffix}")
        it.dst = cand
        if it.status == "move":
            taken[str(cand).lower()] = it.size
            taken[str(cand).lower() + "#src"] = str(it.src)
    return items


DUP_DIR = "Duplicates"


def duplicates(items) -> list:
    """The duplicates found by plan(): [Item] (each with .same_as = the copy that is kept)."""
    return [i for i in items if i.status == "duplicate" and i.same_as is not None]


def same_bytes(a, b) -> bool:
    """True only if the two files are byte for byte identical (read in full)."""
    import filecmp
    try:
        a, b = Path(a), Path(b)
        return a.resolve() != b.resolve() and a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def handle_duplicates(dups, library, delete=False, progress=None, stop=None):
    """Deal with the duplicates the user ticked. delete=False moves them into <library>/Duplicates (undo with
    Undo Last Run); delete=True deletes each one - but only when it is byte for byte the same as the copy that
    is kept, which must still be there. Returns (done, kept_back, log_path); kept_back = [(Item, reason)]."""
    library = Path(library)
    logd = org_base(library) / LOG_DIR
    logd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = logd / f"organize_{stamp}_{'deleted' if delete else 'duplicates'}.json"
    done, kept_back, failed = [], [], []
    for i, it in enumerate(dups):
        if stop and stop():
            break
        if progress:
            progress(f"Checking {it.src.name}", i, len(dups))
        if not _exists(it.src):
            kept_back.append((it, "it is no longer there"))
            continue
        if not (it.same_as and _exists(it.same_as)):
            kept_back.append((it, "the copy to keep is no longer there"))
            continue
        if not same_bytes(it.src, it.same_as):
            kept_back.append((it, "not byte for byte the same as the kept copy - check it yourself"))
            continue
        try:
            if delete:
                it.src.unlink()
                done.append({"src": str(it.src), "same_as": str(it.same_as), "deleted": True})
            else:
                dst = org_base(library) / DUP_DIR / it.src.name
                n = 1
                while _exists(dst):
                    n += 1
                    dst = org_base(library) / DUP_DIR / f"{it.src.stem}_{n}{it.src.suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(it.src), str(dst))
                done.append({"src": str(it.src), "dst": str(dst), "same_as": str(it.same_as)})
            it.status = "handled"
        except OSError as e:
            failed.append({"src": str(it.src), "error": str(e)})
            kept_back.append((it, str(e)))
    log = {"app": "SubZeroAstro", "kind": "organize-duplicates", "move": True, "library": str(library),
           "time": stamp, "done": [] if delete else done, "failed": failed}
    if delete:
        log["deleted"] = done                 # deleted files can't be put back; the log says what went
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if done:
        _remove_empty_dirs({Path(d["src"]).parent for d in done}, stop_at=library)
    return len(done), kept_back, log_path


def summary(items) -> dict:
    out = {"files": 0, "bytes": 0, "here": 0, "duplicate": 0, "error": 0, "by_kind": {}, "targets": set()}
    for it in items:
        if it.status == "move":
            out["files"] += 1
            out["bytes"] += it.size
            out["by_kind"][it.kind] = out["by_kind"].get(it.kind, 0) + 1
            if it.target and it.kind in ("light", "master"):
                out["targets"].add(it.target)
        else:
            out[it.status] = out.get(it.status, 0) + 1
    return out


def _same_drive(a, b) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def execute(items, library, move=False, progress=None, stop=None) -> Path:
    """Copy or move the planned files; returns the undo log. A copy is checked (size) before a move removes
    the original; a file that fails is skipped and noted, the rest carry on."""
    library = Path(library)
    logd = org_base(library) / LOG_DIR
    logd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = logd / f"organize_{stamp}.json"
    done, failed = [], []
    todo = [it for it in items if it.status == "move"]

    def write_log():
        log_path.write_text(json.dumps({"app": "SubZeroAstro", "kind": "organize", "move": move,
                                        "library": str(library), "time": stamp, "done": done,
                                        "failed": failed}, indent=1), encoding="utf-8")
    for i, it in enumerate(todo):
        if stop and stop():
            break
        if progress:
            progress(f"{'Moving' if move else 'Copying'} {it.src.name}", i, len(todo))
        pairs = [(it.src, it.dst)] + [(c, it.dst.parent / c.name) for c in it.companions]
        for src, dst in pairs:
            try:
                if _exists(dst):
                    raise OSError(f"a different file named {dst.name} is already there - left where it is")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if move and _same_drive(src, dst.parent):
                    os.rename(src, dst)            # same drive: just a new folder entry, instant even for 20 TB
                    done.append({"src": str(src), "dst": str(dst)})
                    continue
                tmp = dst.with_name(dst.name + ".organizing")
                shutil.copy2(src, tmp)
                if tmp.stat().st_size != src.stat().st_size:
                    tmp.unlink()
                    raise OSError("the copy came out a different size")
                tmp.replace(dst)
                if move:
                    src.unlink()
                done.append({"src": str(src), "dst": str(dst)})
            except Exception as e:  # noqa: BLE001
                failed.append({"src": str(src), "dst": str(dst), "error": str(e)})
        if i % 50 == 0:
            write_log()
    write_log()
    if move:
        _remove_empty_dirs({Path(d["src"]).parent for d in done}, stop_at=library)
    return log_path


def _remove_empty_dirs(dirs, stop_at=None):
    """After a move, folders left completely empty are removed (never one that still has anything in it)."""
    for d in sorted(dirs, key=lambda p: -len(p.parts)):
        p = d
        while p and _exists(p) and (stop_at is None or p.resolve() != Path(stop_at).resolve()):
            try:
                if any(p.iterdir()):
                    break
                p.rmdir()
            except OSError:
                break
            p = p.parent


def undo(log_path, progress=None) -> tuple:
    """Reverse a run: moved files go back where they were; copies are removed (only if they are still the
    same size as the original, so nothing you changed since is lost). Returns (undone, skipped)."""
    log = json.loads(Path(log_path).read_text(encoding="utf-8"))
    move = bool(log.get("move"))
    undone = skipped = 0
    entries = list(reversed(log.get("done", [])))
    for i, d in enumerate(entries):
        src, dst = Path(d["src"]), Path(d["dst"])
        if progress:
            progress(f"Undoing {dst.name}", i, len(entries))
        try:
            if move:
                if dst.exists() and not src.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(src))
                    undone += 1
                else:
                    skipped += 1
            else:
                if dst.exists() and src.exists() and dst.stat().st_size == src.stat().st_size:
                    dst.unlink()
                    undone += 1
                else:
                    skipped += 1
        except OSError:
            skipped += 1
    _remove_empty_dirs({Path(d["dst"]).parent for d in entries}, stop_at=log.get("library"))
    Path(log_path).rename(Path(log_path).with_suffix(".undone.json"))
    return undone, skipped


def last_log(library) -> Path | None:
    """The newest run not yet undone - logs live in Organized Astro Files (older versions: the library itself)."""
    logs = []
    for d in {org_base(library) / LOG_DIR, Path(library) / LOG_DIR}:
        if _exists(d):
            logs += [p for p in d.glob("organize_*.json") if not p.name.endswith((".undone.json", "_deleted.json"))]
    logs.sort(key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


# ---------------------------------------------------------------- rig names
_NOT_RIG = re.compile(r"(?i)^(\d{4}[-_]?\d{2}[-_]?\d{2}.*|lights?|light[ _]?frames?|calibration( frames)?|cal(ib)?|"
                      r"flats?|darks?|bias(es)?|dark[ _]?flats?|flat[ _]?darks?|masters?|rejected|ha|oiii|sii|[lrgb]|"
                      r"h[- _]?alpha|lum|red|green|blue|no ?filter|\d+(\.\d+)?s|unsorted|incoming.*|astro.*|data|"
                      r"images?|captures?|n\.?i\.?n\.?a\.?|sequence.*|[a-z]:)$")


def _tnorm(n):
    return re.sub(r"[\W_]", "", n or "").lower()


def _ancestor_names(p, stop=None, targets=()):
    """Folder names above a file that could name a rig (not a date, filter, frame type, target …), nearest
    first, up to the scanned folder (stop, included)."""
    out = []
    for anc in list(Path(p).parents)[:8]:
        n = anc.name
        if stop is not None and len(anc.parts) < len(Path(stop).parts):
            break
        tn = _tnorm(n)
        if n and not _NOT_RIG.match(n) and not any(t and len(tn) >= 3 and (tn in t or t in tn) for t in targets):
            out.append(n)
        if stop is not None and anc == Path(stop):
            break
    return out


def _guess_names(groups, root=None) -> dict:
    """For each rig group: the folder name its files share that the other rigs' files don't - the one nearest
    the top (rig folders sit above target and night folders), e.g. SQA60Pro in SQA60Pro/Veil/2026-09-01/."""
    import os
    from collections import Counter
    allp = [str(p) for g in groups.values() for p in g["paths"]]
    if root is not None:
        stop = Path(root)                  # the folder you picked: its own name counts (e.g. SQA60Pro)
    else:
        try:
            stop = Path(os.path.commonpath(allp)) if allp else None
        except ValueError:
            stop = None
    targets = {_tnorm(t) for g in groups.values() for t in g["targets"]}
    per = {}
    for k, g in groups.items():
        c, depth = Counter(), {}
        for p in g["paths"]:
            names = _ancestor_names(p, stop, targets)
            for d, n in enumerate(names):
                c[n] += 1
                depth[n] = max(depth.get(n, 0), d)
        per[k] = (c, depth, len(g["paths"]))
    out = {}
    for k, (c, depth, n) in per.items():
        others = set()
        for k2, (c2, _, _) in per.items():
            if k2 != k:
                others |= set(c2)
        cands = [m for m, v in c.items() if v >= max(1, (2 * n) // 3) and (m not in others or len(per) == 1)]
        if cands:
            out[k] = max(cands, key=lambda m: (depth[m], c[m]))
    return out


def _rig_key(info):
    return ((info.camera or "").strip(), round(float(info.focal), 0) if info.focal else None,
            str(info.telescope or "").strip())


def rig_candidates(items, rigs=None, root=None) -> list:
    """Every rig found (camera + focal length, as Calibrate and Stack names them), with how many files, the
    telescope the header names, a few example folders and the name it gets now - for Name Your Rigs."""
    import stacking as st
    groups = {}
    for it in items:
        info = it.info
        if info is None or it.status == "error":
            continue
        k = _rig_key(info)
        g = groups.setdefault(k, {"camera": k[0], "focal": k[1], "telescopes": set(), "paths": [], "targets": set()})
        if info.telescope:
            g["telescopes"].add(str(info.telescope).strip())
        g["paths"].append(it.src)
        if it.target:
            g["targets"].add(it.target)
    guesses = _guess_names(groups, root)
    out = []
    for k, g in groups.items():
        named = None
        try:
            named = (rigs or st.Rigs()).lookup(g["camera"], g["focal"])
        except Exception:  # noqa: BLE001
            pass
        probe = type("I", (), {"camera": g["camera"], "focal": g["focal"],
                               "telescope": next(iter(g["telescopes"]), None)})()
        guess = guesses.get(k, "")
        tel = probe.telescope or ""
        tel_ok = bool(tel) and st.Rigs.suggestion(probe) == st.safe_name(tel)
        now = named or (_safe(re.sub(r"\s+", "", tel))[:24] if tel_ok else (f"{g['focal']:.0f}mm" if g["focal"] else ""))
        ex = []
        for p in g["paths"]:
            d = str(Path(p).parent)
            if d not in ex:
                ex.append(d)
            if len(ex) >= 3:
                break
        out.append({"key": k, "guess": guess, "camera": g["camera"], "focal": g["focal"],
                    "telescopes": sorted(g["telescopes"]),
                    "files": len(g["paths"]), "examples": ex, "named": named, "now": now,
                    "suggest": named or guess or now})
    out.sort(key=lambda r: (-r["files"]))
    return out


def rig_issues(items, root=None) -> list:
    """Files no rig could be matched to, grouped: [{reason, camera, focal, files, folder, items, names}] -
    folder is the deepest folder all of the group's files share (the one you can assign to a rig)."""
    import os
    import stacking as st
    groups = {}
    for it in items:
        if it.status != "norig" or it.info is None:
            continue
        info = it.info
        reason = it.rig_issue.split(":")[0]
        fl = round(float(info.focal), 0) if info.focal else None
        # one group per (reason, camera, focal) and the top folder under the scanned one
        top = ""
        try:
            rel = Path(it.src).relative_to(root) if root else Path(it.src)
            top = rel.parts[0] if len(rel.parts) > 1 else ""
        except ValueError:
            pass
        k = (reason, st.short_camera(info.camera), fl, top)
        g = groups.setdefault(k, {"reason": reason, "camera": info.camera or "", "focal": fl, "items": [],
                                  "names": it.rig_issue.split(": ", 1)[-1] if reason == "ambiguous" else ""})
        g["items"].append(it)
    out = []
    for g in groups.values():
        dirs = [str(Path(i.src).parent) for i in g["items"]]
        try:
            g["folder"] = os.path.commonpath(dirs)
        except ValueError:
            g["folder"] = dirs[0]
        g["files"] = len(g["items"])
        out.append(g)
    out.sort(key=lambda g: -g["files"])
    return out


def focal_clusters(infos_paths, tol=0.03) -> list:
    """Group the focal lengths found in headers into rigs: [{camera, focal (median), lo, hi, files, paths}] -
    readings within ±tol of each other are one rig (the same scope drifts a few % night to night)."""
    import stacking as st
    by_cam = {}
    for info, p in infos_paths:
        if info is not None and info.focal:
            by_cam.setdefault(st.short_camera(info.camera), []).append((float(info.focal), p, info.camera or ""))
    out = []
    for cam, vals in by_cam.items():
        vals.sort(key=lambda v: v[0])
        cur = [vals[0]]
        for v in vals[1:]:
            med = sorted(x[0] for x in cur)[len(cur) // 2]
            if abs(v[0] - med) <= tol * med:
                cur.append(v)
            else:
                out.append(cur)
                cur = [v]
        out.append(cur)
    res = []
    for c in out:
        fs = sorted(x[0] for x in c)
        res.append({"camera": c[0][2], "focal": round(fs[len(fs) // 2]), "lo": fs[0], "hi": fs[-1],
                    "files": len(c), "paths": [x[1] for x in c]})
    res.sort(key=lambda r: -r["files"])
    return res


def suggest_rig_name(paths, root=None, others=()) -> str:
    """A rig name from your own folders (the folder these files share that other rigs' files don't)."""
    groups = {"me": {"paths": list(paths), "targets": set()}}
    for i, o in enumerate(others):
        groups[i] = {"paths": list(o), "targets": set()}
    return _guess_names(groups, root).get("me", "")


def left_behind(items) -> dict:
    """What a run leaves where it was, by reason: {reason: [Item]} (needs a rig, duplicate, unreadable)."""
    out = {}
    for it in items:
        if it.status == "norig":
            k = "needs a rig (Your Rigs… / Which Rig?…)"
        elif it.status == "duplicate":
            k = "duplicate of a file that was organized (Step 2 deletes it)"
        elif it.status == "error":
            k = "could not be read"
        else:
            continue
        out.setdefault(k, []).append(it)
    return out


# ---------------------------------------------------------------- Safe to delete?
def check_leftovers(sources, library, progress=None, stop=None) -> dict:
    """After organizing: is it safe to delete what is left in the folders you organized? Reads only.
    Every file still there is put in one group:
      safe    - an astro file with a byte-for-byte identical copy in Organized Astro Files
      keep    - an astro file that is NOT in Organized Astro Files (only copy) - don't delete
      work    - in a PixInsight / Siril work folder (calibrated, registered … copies, can be made again)
      project - in a saved SubZeroAstro project
      other   - not an astro file (JPG, notes, logs, compressed FITS …) - the organizer never touches these
    Returns {"safe": [...], "keep": [...], "work": [...], "project": [...], "other": [...], "folders": n,
    "empty_folders": n} with (path, note) pairs."""
    base = org_base(library)
    out = {k: [] for k in ("safe", "keep", "work", "project", "other")}
    out["folders"] = out["empty_folders"] = 0
    # what is in Organized Astro Files, by size (so only files of the same size are ever compared)
    by_size = {}
    if _exists(base):
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d != LOG_DIR]
            for f in fns:
                p = Path(dp) / f
                try:
                    by_size.setdefault(p.stat().st_size, []).append(p)
                except OSError:
                    pass
            if stop and stop():
                break
    files = []
    for src in sources:
        src = Path(src)
        if src.is_file():
            files.append((src, src.parent))
            continue
        for dp, dns, fns in os.walk(src):
            d = Path(dp)
            if _same_path(d, base) or d.name == LOG_DIR:
                dns[:] = []
                continue
            dns[:] = [n for n in dns if not _same_path(d / n, base) and n != LOG_DIR]
            out["folders"] += 1
            if not fns and not dns:
                out["empty_folders"] += 1
            for f in fns:
                files.append((d / f, d))
    files = list(dict((str(p), (p, d)) for p, d in files).values())     # folders picked inside each other
    for k, (p, d) in enumerate(files):
        if stop and stop():
            break
        if progress and k % 20 == 0:
            progress(f"Checking {k + 1} of {len(files)}: {p.name}", k, len(files))
        name = p.name
        if name.startswith("._") or name in (".DS_Store", "Thumbs.db", "desktop.ini"):
            continue                                       # system clutter, never data
        parts = [x.lower() for x in p.parts]
        if any(x.endswith((".szproj", ".saving")) for x in parts) or any(
                _is_project_dir(Path(*p.parts[:i])) for i in range(len(p.parts) - 1, max(0, len(p.parts) - 4), -1)):
            out["project"].append((p, "in a saved SubZeroAstro project - keep it"))
            continue
        if any(x in WORK_DIRS for x in parts[:-1]):
            out["work"].append((p, "a PixInsight / Siril work copy (calibrated, registered …) - can be made again "
                                   "from the subs"))
            continue
        if os.path.splitext(name)[1].lower() not in EXTS:
            out["other"].append((p, "not an astro file - the organizer never touches it; check it yourself"))
            continue
        try:
            size = p.stat().st_size
        except OSError:
            out["keep"].append((p, "could not be read - keep it"))
            continue
        cands = sorted(by_size.get(size, []), key=lambda c: (c.name != name, str(c)))
        twin = next((c for c in cands if same_bytes(p, c)), None)
        if twin is not None:
            out["safe"].append((p, f"identical copy in {twin.parent}"))
        else:
            out["keep"].append((p, "NOT in Organized Astro Files - this is the only copy; run Step 1 to see why"))
    return out
