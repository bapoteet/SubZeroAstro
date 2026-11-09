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
LIGHT_DIR, CAL_DIR = "Light Frames", "Calibration Frames"
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
_KEEP_RIG = {}        # re-sorting a library: file path -> the rig folder it is already filed under
_TOL = 0.03           # a rig's focal length in the headers drifts a few % night to night (focusing, solving)


def _cam_norm(c):
    """A camera name reduced to its model: 'ZWO ASI2600MM Pro', 'ASI2600MM', '2600MMPro' and '2600 mm pro' are all
    '2600mm' - so the camera typed in Your Rigs matches the header however it is written (MM and MC still differ)."""
    if not c:
        return ""
    s = re.sub(r"[\W_]", "", str(c)).lower()
    s = re.sub(r"^(zwo|qhy|playerone|touptek|atik|sbig|fli|moravian|altair|svbony)", "", s)
    s = re.sub(r"^(asi|qhy)", "", s)
    s = re.sub(r"(pro|mini|cooled|cool)$", "", s)
    return s


def _rig_folder_key(name) -> str:
    """'RASA 11' / 'rasa-11' / 'RASA11' -> 'rasa11' (short names like 'C' are ignored)."""
    k = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    return k if len(k) >= 3 else ""


REMOTE_RIGS = True          # Organize: data from rented telescopes filed under the service's telescope


def remote_rig(info, path, by_name=False) -> str:
    """Frames from a telescope you rent time on (not one of your rigs): 'Telescope Live CHI-1', 'iTelescope T21',
    or the telescope named in an already-calibrated file's header - so they're filed under that telescope
    instead of waiting in Which Rig?. '' when it isn't such a file."""
    if not REMOTE_RIGS:
        return ""
    n = Path(str(path)).name if path is not None else ""
    m = re.match(r"^([A-Z]{3}-\d{1,2})(?:-(?:CCD|CMOS|[A-Z]{2,5}))?_", n)          # CHI-1-CCD_2019-12-02T…
    if m:
        return f"Telescope Live {m.group(1)}"
    m = re.match(r"^(?:raw|calibrated)-(T\d{1,2})-", n, re.I)                      # calibrated-T21-user-M31-…
    if m:
        return f"iTelescope {m.group(1).upper()}"
    if by_name:
        return ""
    pre = str(getattr(info, "precal", "") or "")
    if "telescope live" in pre.lower():
        tel = str(getattr(info, "telescope", "") or "").strip()
        return f"Telescope Live {tel}".strip() if tel else "Telescope Live"
    return ""


def match_rig(info, path, riglist, tol=0.03):
    """(rig name or None, reason). A folder you assigned to a rig wins; otherwise the focal length in the header
    within ±tol of one of a rig's focal lengths (and the camera, when the rig lists one). reason:
    'folder' / 'focal' / 'nofocal' / 'nomatch' / 'ambiguous: A, B'."""
    if path is not None and _KEEP_RIG:             # already filed in the library: it keeps that rig
        kept = _KEEP_RIG.get(os.path.normcase(os.path.abspath(str(path))))
        if kept:
            return kept, "filed"
    sp = str(path).replace("\\", "/").lower() if path is not None else ""
    best = None
    for r in riglist or []:
        for f in r.get("folders") or []:
            fp = str(f).replace("\\", "/").rstrip("/").lower() + "/"
            if sp.startswith(fp) and (best is None or len(fp) > best[1]):
                best = (r["name"], len(fp))
    if best:
        return best[0], "folder"
    # a Telescope Live / iTelescope file name says outright whose telescope took it - even when its focal
    # length happens to match one of your rigs
    rem = remote_rig(None, path, by_name=True)
    if rem:
        return rem, "remote"
    # a folder on the way named after a rig ("RASA11/Light/M31/…", e.g. one USB stick filled from several
    # ASIAIRs, each into a folder named after its rig) settles what the header can't: two rigs with the same
    # scope and camera, or no focal length at all. A header that names another rig, or none of them, still wins.
    parts = {_rig_folder_key(x) for x in re.split(r"[\\/]", str(path or ""))[:-1]} - {""}
    named = [r["name"] for r in riglist or [] if _rig_folder_key(r.get("name")) in parts]
    by_folder = named[0] if len(named) == 1 else None
    fl = getattr(info, "focal", None)
    if not fl:
        if by_folder:
            return by_folder, "folder"
        rem = remote_rig(info, path)
        return (rem, "remote") if rem else (None, "nofocal")
    cam = _cam_norm(getattr(info, "camera", None))
    hits, cam_off = {}, []
    for r in riglist or []:
        ds = [abs(float(fl) - float(f)) / float(f) for f in r.get("focals") or [] if f]
        ds = [d for d in ds if d <= tol]
        if not ds:
            continue
        if r.get("camera") and cam and _cam_norm(r["camera"]) != cam:
            cam_off.append(f"{r['name']} ({r['camera']})")      # right focal length, another camera
            continue
        hits[r["name"]] = min(hits.get(r["name"], 9), min(ds))
    if len(hits) == 1:
        return next(iter(hits)), "focal"
    if len(hits) > 1:
        # within reach of two rigs (e.g. 506 and 530 mm scopes): the clearly nearer one wins (at most half the
        # distance of the next); a reading right between them is asked about
        near = sorted(hits.items(), key=lambda kv: kv[1])
        if near[0][1] <= 0.5 * near[1][1] and near[0][1] < near[1][1]:     # (a tie is no answer)
            return near[0][0], "focal"
    hits = list(hits)
    if by_folder and by_folder in hits:            # a tie the folder settles (a focal length no rig has is not)
        return by_folder, "folder"
    if not hits and cam_off:
        return None, "camera: " + ", ".join(cam_off)
    if not hits:
        rem = remote_rig(info, path)
        return (rem, "remote") if rem else (None, "nomatch")
    return None, "ambiguous: " + ", ".join(hits)


def _rig(info, path=None) -> str:
    """Which rig (telescope + the camera that lives on it) a file comes from: the name you gave the rig in
    SubZeroAstro, else the telescope in the header (not a mount driver such as EQMOD), else the focal length,
    else - for calibration frames sorted by hand - the folder above the Darks / Flats / Bias folder."""
    import stacking as st
    if path is not None and _KEEP_RIG:             # re-sorting the library: the rig it's already filed under
        kept = _KEEP_RIG.get(os.path.normcase(os.path.abspath(str(path))))
        if kept:
            return kept
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


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def _same_path(a, b) -> bool:
    """The same file? Compared by name only - no trip to the disk (on a NAS or a USB drive every
    resolve() asks the drive about each folder on the way, and over 100,000 files that took hours)."""
    return _norm(a) == _norm(b)


class _Listing:
    """Which names are in a folder, asked once per folder (not once per file)."""

    def __init__(self):
        self.dirs = {}

    def exists(self, p) -> bool:
        p = Path(p)
        d = _norm(p.parent)
        names = self.dirs.get(d)
        if names is None:
            try:
                names = {os.path.normcase(n) for n in os.listdir(p.parent)}
            except OSError:              # no such folder yet (the usual case for a new layout), or unreadable
                names = set()
            self.dirs[d] = names
        return os.path.normcase(p.name) in names


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
            if n == DUP_DIR and d.name == ORG_ROOT:
                # the frames set aside as duplicates stay set aside: looked through again, each one was
                # offered as a duplicate again on every re-sort (2,000 of them, every time)
                SKIPPED.append((p, "frames set aside as duplicates - kept apart, not organized again"))
                continue
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
           "target_rig": "By target, then rig: Light Frames / target / rig / night  and  Calibration Frames / rig / "
                         "Flats, Darks, Bias",
           "date": "By date, then target, then rig: Light Frames / year / night / target / rig  and  Calibration "
                   "Frames / rig / Flats, Darks, Bias",
           "target": "By target (classic): Targets / target / night_rig, with darks and bias libraries"}


READERS = 8


def library_rigs(paths, library) -> dict:
    """Files already inside the organized library keep the rig they're filed under - re-sorting (Change My
    Library's Layout…) or organizing the library again only moves them, it never asks Which Rig? again.
    Calibration Frames/<rig>/…; Light Frames/<rig>/<target>/… or Light Frames/<target>/<rig>/… - whichever level
    holds fewer different names (a handful of rigs, many targets) and more of Your Rigs is the rig."""
    if library is None or str(library) in ("", "."):
        return {}
    try:
        base = os.path.normcase(os.path.abspath(str(org_base(library))))
    except OSError:
        return {}
    cal, lights = {}, []
    for p in paths:
        ap = os.path.normcase(os.path.abspath(str(p)))
        if not ap.startswith(base + os.sep):
            continue
        parts = Path(os.path.relpath(ap, base)).parts
        if len(parts) >= 3 and parts[0] == os.path.normcase(CAL_DIR):
            cal[ap] = Path(os.path.relpath(os.path.abspath(str(p)), str(org_base(library)))).parts[1]
        elif len(parts) >= 4 and parts[0] == os.path.normcase(LIGHT_DIR):
            lights.append((ap, Path(os.path.relpath(os.path.abspath(str(p)), str(org_base(library)))).parts))
    out = dict(cal)
    if lights:
        known = {_rig_folder_key(r.get("name")) for r in (_RIGLIST or [])} | \
                {_rig_folder_key(v) for v in cal.values()}
        lvl1 = {pp[1] for _a, pp in lights}
        lvl2 = {pp[2] for _a, pp in lights}
        hit1 = sum(1 for n in lvl1 if _rig_folder_key(n) in known)
        hit2 = sum(1 for n in lvl2 if _rig_folder_key(n) in known)
        lvl = 1 if (hit1, -len(lvl1)) >= (hit2, -len(lvl2)) else 2
        for a, pp in lights:
            # by date (Light Frames/year/night/target/rig/…) the rig is deeper: a folder named after one of
            # your rigs says it directly; otherwise the level that holds the rigs in the other layouts
            hit = [x for x in pp[1:-1] if _rig_folder_key(x) in known]
            if hit:
                out[a] = hit[0]
            elif len(pp) >= 6 and re.match(r"^\d{4}$", pp[1]) and re.match(r"^\d{4}-\d{2}-\d{2}$", pp[2]):
                out[a] = pp[4]                                     # year / night / target / rig
            elif len(pp) >= 5 and pp[1] == "Masters":
                out[a] = pp[3]                                     # Masters / target / rig
            else:
                out[a] = pp[lvl]
    return out


class HeaderCache:
    """What each file's header said, kept in the library (Organized Astro Files/_SubZeroAstro_organize/
    headers.json), so the next look - a re-sort into another layout, the next Organize - only lists the folders
    instead of opening every file again over the network. A file is known by its name, size and modified time,
    which a move keeps, so it's still known after Organize has filed it."""

    MAX = 400_000
    _open = {}

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.data, self.dirty = {}, False
        if self.path is not None:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if d.get("v") == 2:
                    self.data = d.get("files") or {}
            except (OSError, ValueError, AttributeError):
                self.data = {}

    @classmethod
    def for_library(cls, library):
        if library is None or str(library) in ("", "."):
            return None
        p = org_base(library) / LOG_DIR / "headers.json"
        if str(p) not in cls._open:
            cls._open[str(p)] = cls(p)
        return cls._open[str(p)]

    @staticmethod
    def _key(p, stt) -> str:
        # to the nanosecond (a move or a careful copy keeps it): a file rewritten in the same second with the
        # same name and size is still told apart
        return f"{Path(p).name.lower()}|{stt.st_size}|{stt.st_mtime_ns}"

    def get(self, p, stt):
        import stacking as st
        d = self.data.get(self._key(p, stt))
        if not d:
            return None
        try:
            return st.ImgInfo(**dict(d, path=Path(p)))
        except TypeError:                  # a field added or removed since: read the file again
            return None

    def put(self, p, stt, info):
        import dataclasses
        try:
            d = dataclasses.asdict(info)
        except TypeError:
            return
        d.pop("path", None)
        self.data[self._key(p, stt)] = d
        self.dirty = True

    def save(self):
        # only into a library that exists already (a first look at a new library leaves no trace; the first
        # Organize saves it)
        if not self.dirty or self.path is None or not self.path.parent.parent.is_dir():
            return
        if len(self.data) > self.MAX:
            for k in list(self.data)[: len(self.data) - self.MAX]:
                del self.data[k]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"v": 2, "files": self.data}), encoding="utf-8")
            tmp.replace(self.path)
            self.dirty = False
        except OSError:
            pass


def _empty_reason(p) -> str:
    """Why a file can't be read, when it's plain. All zeros (or 0 bytes): it was never written - the capture or a
    copy was cut short (a crash, the power, the network). A blank start with data further in: the copy wrote
    some pieces but never the header. Either way there is no usable frame in it."""
    try:
        size = Path(p).stat().st_size
        if size == 0:
            return "empty file (0 bytes) - the capture or a copy was cut short; nothing to rescue"
        blk = 64 << 10
        with open(p, "rb") as f:
            head = f.read(blk)
            rest = []
            for frac in (0.25, 0.5, 0.75, 1.0):
                f.seek(max(0, min(int(size * frac), size - blk)))
                rest.append(f.read(blk))
        if not head[:2880].strip(b"\0"):
            if not head.strip(b"\0") and not any(r.strip(b"\0") for r in rest):
                return ("all zeros - the file was made but the picture was never written into it (the capture or a "
                        "copy was cut short); nothing to rescue - check the copy on the imaging computer, else "
                        "delete it")
            return ("all zeros at the start - the header was never written (a copy cut short); the frame can't be "
                    "used - check the copy on the imaging computer or a backup, else delete it")
    except OSError:
        pass
    return ""


def _starts_with(p) -> str:
    """The first bytes of a file nobody can read, for the log: '53 49 4D 50' (SIMPLE) or 'PK..' tells what it is."""
    try:
        with open(p, "rb") as f:
            b = f.read(16)
        return " ".join(f"{x:02X}" for x in b) + "  " + "".join(chr(x) if 32 <= x < 127 else "." for x in b)
    except OSError:
        return ""


def classify(paths, read_info=None, progress=None, stop=None, library=None, rigs=None, layout="rig",
             root=None, riglist=None, tol=0.03) -> list:
    """Read each header and decide where the file belongs (relative to the library). Returns [Item].
    rigs: SubZeroAstro's saved rig names (stacking.Rigs), so a rig you named keeps its name."""
    import stacking as st
    global _RIGS, _RIGLIST, _TOL, _KEEP_RIG
    _RIGS = rigs
    _RIGLIST, _TOL = riglist, tol
    paths = list(paths)
    _KEEP_RIG = library_rigs(paths, library)
    use_cache = read_info is None
    read_info = read_info or st.read_info
    cache = HeaderCache.for_library(library) if use_cache else None
    items, infos = [], []

    def one(p):
        try:
            stt = Path(p).stat()
            info = cache.get(p, stt) if cache is not None else None
            if info is None:
                info = read_info(p)
                if cache is not None:
                    cache.put(p, stt, info)
            return stt.st_size, info, None
        except Exception as e:  # noqa: BLE001 - unreadable file: leave it where it is
            return 0, None, e

    # several headers at once: on a NAS each file mostly waits for the network, not the disk (8x faster on Wi-Fi)
    from concurrent.futures import ThreadPoolExecutor
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=READERS) as pool:
        futs = []
        for p in paths:
            futs.append(pool.submit(one, p))
        for i, (p, fu) in enumerate(zip(paths, futs)):
            if progress and i % 20 == 0:
                progress(f"Reading headers {i + 1} of {len(paths)}", i)
            if stop and stop():
                for f2 in futs[i:]:
                    f2.cancel()
                break
            size, info, err = fu.result()
            it = Item(src=Path(p))
            it.size = size
            if err is not None:
                it.status, it.why = "error", (_empty_reason(p) or f"could not read it ({err}) - it starts with "
                                                                     f"{_starts_with(p)}")
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
        it.target = st.target_name(it.info.target, it.info.filter) if (it.info is not None and it.info.target) else ""
    for c in rig_candidates(items, rigs, root=root if root is not None else library):
        if not c["named"] and c["guess"]:
            _GUESS[c["key"]] = c["guess"]

    # which targets had lights on which night (with which telescope) - flats without a target go there
    lights_on, rigs, existing = {}, {}, {}
    for it, info in zip(items, infos):
        if info is not None and info.type == "light" and not info.master and info.target and info.night:
            lights_on.setdefault(info.night, set()).add((st.target_name(info.target, info.filter), _rig(info, it.src)))
            rigs.setdefault(st.target_name(info.target, info.filter), set()).add(_rig(info, it.src))
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
                if why.startswith("camera: "):
                    it.why = (f"{fl} is {why[8:]} in Your Rigs, but this file's camera is "
                              f"{getattr(info, 'camera', None) or 'not in the header'} - fix or clear that rig's "
                              f"camera (Your Rigs…)")
                    continue
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
        tgt = st.target_name(info.target, info.filter) if info.target else ""
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
        if layout in ("rig", "target_rig", "date"):
            _rig_layout(it, info, typ, looks_master, tgt, night, flt, name, rejected, target_first=layout == "target_rig",
                        date_first=layout == "date")
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
    if cache is not None:
        cache.save()
    return items


def _rig_layout(it, info, typ, looks_master, tgt, night, flt, name, rejected, target_first=False, date_first=False):
    """Light Frames / <rig> / <target> / <night> / <filter>   and   Calibration Frames / <rig> / Flats | Dark flats | Darks |
    Bias | Masters - everything a rig (one telescope with its own camera) took, in one place. target_first: the
    lights go Light Frames / <target> / <rig> / <night> / <filter> instead (every rig's nights of one object
    together); calibration frames stay by rig either way - they belong to a rig, not to a target. date_first:
    Light Frames / <year> / <night> / <target> / <rig> / <filter> (what was shot each night, across every rig);
    master lights, which belong to no one night, go Light Frames / Masters / <target> / <rig>."""
    import stacking as st
    rig = _rig(info, it.src) or "Unknown rig"
    cal = Path("Calibration Frames") / rig
    top = (Path(LIGHT_DIR) / tgt / rig) if target_first else (Path(LIGHT_DIR) / rig / tgt)
    if typ in (None, "light") and looks_master and tgt:
        mdir = (Path(LIGHT_DIR) / "Masters" / tgt / rig) if date_first else top / "Masters"
        it.kind, it.dst, it.why = "master", mdir / name, "master light"
        it.companions = _companions(it.src)
    elif typ in ("dark", "flat", "darkflat", "bias") and looks_master:
        sub = Path(_cal_folder(info)) if typ in ("dark", "bias") else Path(f"Flats {night}") / flt
        it.kind, it.dst, it.why = "calmaster", cal / "Masters" / sub / name, f"master {typ} ({rig})"
        it.companions = _companions(it.src)
    elif typ == "light" or (typ is None and tgt and info.exptime):
        if not tgt:
            it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "a light without a target name"
            return
        if date_first:
            year = night[:4] if re.match(r"^\d{4}-", night) else night
            sub = Path(LIGHT_DIR) / year / night / tgt / rig / flt
        else:
            sub = top / night / flt
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


_HASHES = {}
HASH_HEAD, HASH_TAIL = 128 << 10, 32 << 10


def _quick_hash(p: Path) -> str:
    """Size + the start (the whole header - capture time, target, settings - and the first pixels) and the end
    of the file: enough to tell real duplicates from different files of one name, and only ~160 KB read per
    file (it used to be 2 MB, which over a NAS on Wi-Fi made 'Checking for duplicates' take most of an hour
    when a folder had been organized before)."""
    p = Path(p)
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k in _HASHES:
        return _HASHES[k]
    h = hashlib.sha1()
    size = st.st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(HASH_HEAD))
        if size > HASH_HEAD + HASH_TAIL:
            f.seek(-HASH_TAIL, os.SEEK_END)
            h.update(f.read(HASH_TAIL))
    if len(_HASHES) > 200_000:
        _HASHES.clear()
    _HASHES[k] = h.hexdigest()
    return _HASHES[k]


def plan(items, library, progress=None, stop=None) -> list:
    """Final destinations: already in place / duplicate of a file already there / a numbered name if a
    different file of that name is in the way (also between files of this same run)."""
    library = Path(library)
    taken = {}
    listing = _Listing()
    # re-sorting the library itself (every file already inside it): only moves, never a duplicates step - the
    # library was checked for duplicates when its files came in; two files headed for one name both stay
    # (the second numbered)
    base_ = os.path.normcase(str(org_base(library))) + os.sep
    resort = bool(items) and all(os.path.normcase(str(i.src)).startswith(base_) for i in items)
    # the same frame twice (a copy under another name): the one already in the library - or else the
    # first one found - is kept, the others are left where they are and reported
    seen = {}
    def keep_first(x):
        in_place = x.dst is not None and _same_path(org_base(library) / x.dst, x.src)
        # Windows' "name (1).fits" / "name - Copy.fits", or our own "_2": the copy, not the original (a
        # 4-digit frame number like _0042 is not a copy mark)
        copyish = bool(re.search(r"(?:\s\(\d+\)|[ _-]+copy(?:\s*\(\d+\))?|(?<!\d)_\d{1,2})$", x.src.stem, re.I))
        return (0 if in_place else 1, 1 if copyish else 0, core.natural_key(str(x.src)))
    for it in ([] if resort else sorted(items, key=keep_first)):
        if it.key is None or it.status == "error":
            continue
        if it.key in seen:
            it.status, it.why = "duplicate", f"the same frame as {seen[it.key].name}"
            it.same_as = seen[it.key]
        else:
            seen[it.key] = it.src
    for i, it in enumerate(items):
        if stop and stop():            # half a plan is no plan: nothing unchecked may be moved
            raise RuntimeError("you pressed Stop while duplicates were being checked - nothing was changed; "
                               "press Step 1 again to look through the files")
        if progress and i % 20 == 0:
            progress(f"Checking for duplicates and name clashes… {i:,} of {len(items):,}", i, len(items))
        if it.dst is None or it.status in ("error", "duplicate", "norig"):
            continue
        it.dst = Path(*[_safe(x) if k < len(Path(it.dst).parts) - 1 else x
                        for k, x in enumerate(Path(it.dst).parts)])     # no ? : * … in folder names (Windows)
        dst = org_base(library) / it.dst
        if _same_path(dst, it.src):
            it.status, it.dst = "here", dst
            continue
        clean = COPY_MARK.sub("", dst.stem)          # "M31_0042 (1).fits" is filed as "M31_0042.fits"
        if clean and clean != dst.stem:
            dst = dst.with_name(clean + dst.suffix)
            it.dst = dst.relative_to(org_base(library))
        n = 1
        cand = dst
        while True:
            key = str(cand).lower()
            if key in taken:
                if not resort and taken[key] == it.size and _quick_hash(it.src) == _quick_hash(Path(taken[key + "#src"])):
                    it.status, it.same_as = "duplicate", Path(taken[key + "#src"])
                    it.why = f"the same file as {it.same_as.name}"
                    break
            elif listing.exists(cand):
                try:
                    if not resort and cand.stat().st_size == it.size and _quick_hash(cand) == _quick_hash(it.src):
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
    # a chain: B is the same frame as A, but A itself turned out to be a copy of the file already in the
    # library - so B's copy to keep is the library's (A is about to go too)
    by_src = {os.path.normcase(str(i.src)): i for i in items}
    for it in items:
        k = it.same_as
        for _ in range(5):
            nxt = by_src.get(os.path.normcase(str(k))) if (it.status == "duplicate" and k is not None) else None
            if nxt is None or nxt is it or nxt.status != "duplicate" or nxt.same_as is None:
                break
            k = nxt.same_as
        if k is not it.same_as:
            it.same_as = k
            it.why = f"the same frame as the copy already in {Path(k).parent.name}"
    return items


DUP_DIR = "Duplicates"


def duplicates(items) -> list:
    """The duplicates found by plan(): [Item] (each with .same_as = the copy that is kept)."""
    return [i for i in items if i.status == "duplicate" and i.same_as is not None]


def same_bytes(a, b, chunk=8 << 20) -> bool:
    """True only if the two files are byte for byte identical (read in full). Both files are read at the same
    time, in big pieces, and it stops at the first difference - on a NAS that halves the wait."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        a, b = Path(a), Path(b)
        if _same_path(a, b) or a.stat().st_size != b.stat().st_size:
            return False
        if _quick_hash(a) != _quick_hash(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb, ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                ra, rb = pool.submit(fa.read, chunk), pool.submit(fb.read, chunk)
                xa, xb = ra.result(), rb.result()
                if xa != xb:
                    return False
                if not xa:
                    return True
    except OSError:
        return False


SAFE_MOVED = ("the same frame but not the same file (one copy was probably edited) - moved to the Duplicates folder "
              "instead of deleted; Undo Last Run puts it back")


def handle_duplicates(dups, library, delete=False, progress=None, stop=None):
    """Deal with the duplicates the user ticked. delete=False moves them into <library>/Duplicates (undo with
    Undo Last Run); delete=True deletes each one - but only when it is byte for byte the same as the copy that
    is kept, which must still be there. Returns (done, kept_back, log_path); kept_back = [(Item, reason)]."""
    library = Path(library)
    logd = org_base(library) / LOG_DIR
    logd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = logd / f"organize_{stamp}_{'deleted' if delete else 'duplicates'}.json"
    done, kept_back, failed, moved_safe = [], [], [], []
    for i, it in enumerate(dups):
        if stop and stop():
            break
        if progress:
            progress(f"Checking {it.src.name}", i, len(dups))
        if not _exists(it.src):
            kept_back.append((it, "it is no longer there"))
            continue
        if it.same_as and not _exists(it.same_as):        # its kept copy was itself a duplicate, handled just now
            for d in done + moved_safe:
                if _same_path(d["src"], it.same_as) and d.get("same_as") and _exists(d["same_as"]):
                    it.same_as = Path(d["same_as"])
                    break
        if not (it.same_as and _exists(it.same_as)):
            kept_back.append((it, "the copy to keep is no longer there"))
            continue
        # deleting: only a byte-for-byte twin goes. Anything else - moving to Duplicates, or a "duplicate" that is
        # the same frame (same capture time, target, filter) whose file differs, e.g. one copy had its header
        # edited - goes into the Duplicates folder, where Undo Last Run can put it back
        to_dups = not delete or not same_bytes(it.src, it.same_as)
        try:
            if not to_dups:
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
                entry = {"src": str(it.src), "dst": str(dst), "same_as": str(it.same_as)}
                if delete:                     # asked to delete, but not identical: kept safe instead
                    entry["not_identical"] = True
                    moved_safe.append(entry)
                    kept_back.append((it, SAFE_MOVED))
                else:
                    done.append(entry)
            it.status = "handled"
        except OSError as e:
            failed.append({"src": str(it.src), "error": str(e)})
            kept_back.append((it, str(e)))
    log = {"app": "SubZeroAstro", "kind": "organize-duplicates", "move": True, "library": str(library),
           "time": stamp, "done": moved_safe if delete else done, "failed": failed}
    if delete:
        log["deleted"] = done                 # deleted files can't be put back; the log says what went
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if done or moved_safe:
        _remove_empty_dirs({Path(d["src"]).parent for d in done + moved_safe}, stop_at=library)
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


def execute(items, library, move=False, progress=None, stop=None, keep=None) -> Path:
    """Copy or move the planned files; returns the undo log. A copy is checked (size) before a move removes
    the original; a file that fails is skipped and noted, the rest carry on."""
    library = Path(library)
    logd = org_base(library) / LOG_DIR
    logd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = logd / f"organize_{stamp}.json"
    done, failed, extra = [], [], {}
    todo = [it for it in items if it.status == "move"]

    def write_log():
        log_path.write_text(json.dumps(dict({"app": "SubZeroAstro", "kind": "organize", "move": move,
                                             "library": str(library), "time": stamp, "done": done,
                                             "failed": failed}, **extra)), encoding="utf-8")
    # Over a network drive every check is a round trip, so each folder is listed once (not one look per file),
    # each folder is made once, and the undo log is saved every 30 s instead of every 50 files (rewriting a
    # 150,000-file log thousands of times was most of the time a big re-sort took).
    names, devs = {}, {}

    def there(d):
        if d not in names:
            try:
                names[d] = {os.path.normcase(n) for n in os.listdir(d)}
            except FileNotFoundError:
                d.mkdir(parents=True, exist_ok=True)
                names[d] = set()
            except OSError:
                d.mkdir(parents=True, exist_ok=True)
                names[d] = None                       # cannot list it: fall back to a look per file
        return names[d]

    def dev(d):
        if d not in devs:
            try:
                devs[d] = os.stat(d).st_dev
            except OSError:
                devs[d] = None
        return devs[d]
    last = time.monotonic() - 31           # first save right away, so an Undo exists even if the run is cut short
    for i, it in enumerate(todo):
        if stop and stop():
            break
        if progress:
            progress(f"{'Moving' if move else 'Copying'} {it.src.name}", i, len(todo))
        pairs = [(it.src, it.dst)] + [(c, it.dst.parent / c.name) for c in it.companions]
        for src, dst in pairs:
            try:
                have = there(dst.parent)
                key = os.path.normcase(dst.name)
                if (key in have) if have is not None else _exists(dst):
                    raise OSError(f"a different file named {dst.name} is already there - left where it is")
                sd, dd = dev(src.parent), dev(dst.parent)
                if move and sd is not None and sd == dd:
                    os.rename(src, dst)            # same drive: just a new folder entry, instant even for 20 TB
                    done.append({"src": str(src), "dst": str(dst)})
                    if have is not None:
                        have.add(key)
                    continue
                tmp = dst.with_name(dst.name + ".organizing")
                shutil.copy2(src, tmp)
                if tmp.stat().st_size != src.stat().st_size:
                    tmp.unlink()
                    raise OSError("the copy came out a different size")
                tmp.replace(dst)
                if have is not None:
                    have.add(key)
                if move:
                    src.unlink()
                done.append({"src": str(src), "dst": str(dst)})
            except FileNotFoundError as e:
                gone = not _exists(src)
                failed.append({"src": str(src), "dst": str(dst), "error": (
                    "it was gone when its turn came - moved or deleted by something else while this run was going "
                    "(another Organize, on this or another computer, or a sync)" if gone else str(e))})
            except Exception as e:  # noqa: BLE001
                failed.append({"src": str(src), "dst": str(dst), "error": str(e)})
        if time.monotonic() - last > 30:
            write_log()
            last = time.monotonic()
    hc = HeaderCache.for_library(library)
    if hc is not None:                    # what the headers said, for the next look at this library
        hc.save()
    meta = carry_nina_metadata(done, move)
    if meta:
        extra["meta"] = meta
    write_log()
    if move:
        _remove_empty_dirs({Path(d["src"]).parent for d in done}, stop_at=library, keep=keep)
    return log_path


def carry_nina_metadata(done, move=False) -> list:
    """N.I.N.A.'s Session Metadata file (ImageMetaData.csv - guiding RMS per frame) goes with the frames: a copy
    into every folder they were filed into. After a move, the original goes too once its folder has no frames
    left. Returns what was done, for Undo: [{src, copies, removed}]."""
    import nina_meta as NM
    by_src = {}
    for d in done:
        by_src.setdefault(Path(d["src"]).parent, set()).add(Path(d["dst"]).parent)
    out = []
    for sdir, ddirs in by_src.items():
        try:
            metas = [p for p in sdir.iterdir() if p.is_file() and NM.looks_like(p)]
        except OSError:
            continue
        for mf in metas:
            rec = {"src": str(mf), "copies": [], "removed": False}
            try:
                body = mf.read_bytes()
            except OSError:
                continue
            for dd in sorted(ddirs):
                t = dd / mf.name
                try:
                    if t.exists():
                        old = t.read_bytes()
                        if old == body:
                            continue
                        if not body.startswith(old):          # another night's file of the same name
                            n = 2
                            while t.exists() and t.read_bytes() != body:
                                t = dd / f"{mf.stem} ({n}){mf.suffix}"
                                n += 1
                            if t.exists():
                                continue
                    t.write_bytes(body)
                    rec["copies"].append(str(t))
                except OSError:
                    pass
            if move and rec["copies"]:
                try:
                    if not any(p.suffix.lower() in EXTS for p in sdir.iterdir() if p.is_file()):
                        mf.unlink()
                        rec["removed"] = True
                except OSError:
                    pass
            if rec["copies"]:
                out.append(rec)
    return out


PREVIEW_EXTS = {".jpg", ".jpeg"}
COPY_MARK = re.compile(r"(?:\s\(\d+\)|\s+-\s+copy(?:\s*\(\d+\))?)+$", re.I)   # Windows' copy names
ASIAIR_JPG = re.compile(r"^(light|flat|dark|bias|darkflat|preview|live|stack)_.+\.jpe?g$|_thn\.jpe?g$", re.I)


def preview_files(items, library=None, roots=None) -> list:
    """The .jpg previews (an ASIAIR saves one with every frame) in the folders these frames came from, and
    anywhere under the incoming folders being organized (`roots`: previews left behind on their own once
    their frames had been filed) - never anything inside the organized library."""
    base = None
    if library is not None and str(library) not in ("", "."):
        try:
            base = org_base(library).resolve()
        except OSError:
            base = None
    out, seen = [], set()
    frame_dirs = {Path(it.src).parent for it in items if getattr(it, "src", None) is not None}
    dirs = [(d, True) for d in frame_dirs]
    for r in roots or []:
        for dp, dns, _fns in os.walk(Path(r)):
            dns[:] = [x for x in dns if not x.startswith((".", "#", "@"))]
            if Path(dp) not in frame_dirs:
                dirs.append((Path(dp), False))
    for d, with_frames in dirs:
        try:
            rd = d.resolve()
        except OSError:
            continue
        if rd in seen or (base is not None and (rd == base or base in rd.parents)):
            continue
        seen.add(rd)
        try:
            # beside frames: every .jpg; elsewhere under the incoming folder only an ASIAIR's own previews
            # (Light_…/Flat_…/…_thn.jpg) - never your own pictures that happen to be there
            out += [p for p in d.iterdir() if p.suffix.lower() in PREVIEW_EXTS and p.is_file()
                    and (with_frames or ASIAIR_JPG.search(p.name))]
        except OSError:
            continue
    return sorted(out)


def delete_previews(items, library=None, roots=None) -> int:
    """Delete those .jpg previews (Organize › 'Delete the .jpg previews'). Returns how many went."""
    n, parents = 0, set()
    for p in preview_files(items, library, roots):
        try:
            p.unlink()
            n += 1
            parents.add(p.parent)
        except OSError:
            pass
    if roots and parents:                 # folders left holding nothing but those previews: tidied away too
        _remove_empty_dirs(parents, keep=[Path(r) for r in roots])
    return n


def other_files(roots, library=None, min_age_s=600) -> list:
    """Everything under the incoming folders that isn't a FITS / XISF frame (.jpg, .csv, .txt, Thumbs.db,
    .DS_Store …) - never anything inside the organized library, nothing changed in the last `min_age_s` seconds
    (a copy still coming in), and never N.I.N.A.'s ImageMetaData file while frames it describes are still
    waiting in its folder."""
    import time
    import fits_core as core
    import nina_meta as NM
    now = time.time()
    keep = {e.lower() for e in core.LIGHT_EXTS}
    base = None
    if library is not None and str(library) not in ("", "."):
        try:
            base = org_base(library).resolve()
        except OSError:
            base = None
    out = []
    for r in roots or []:
        for dp, dns, fns in os.walk(Path(r)):
            d = Path(dp)
            try:
                rd = d.resolve()
            except OSError:
                continue
            if base is not None and (rd == base or base in rd.parents):
                dns[:] = []
                continue
            dns[:] = [x for x in dns if not x.startswith(("#", "@"))]
            frames_here = any(Path(n).suffix.lower() in keep for n in fns)
            for n in fns:
                p = d / n
                if p.suffix.lower() in keep:
                    continue
                try:
                    if min_age_s and now - p.stat().st_mtime < min_age_s:
                        continue
                except OSError:
                    continue
                if frames_here and NM.looks_like(p):
                    continue
                out.append(p)
    return sorted(out)


def describe_files(paths) -> str:
    """'30 .jpg, 2 .csv, 1 Thumbs.db (4.1 MB)'."""
    import collections
    c = collections.Counter((p.suffix.lower() or p.name) for p in paths)
    size = 0
    for p in paths:
        try:
            size += p.stat().st_size
        except OSError:
            pass
    return ", ".join(f"{n} {k}" for k, n in c.most_common(6)) + (", …" if len(c) > 6 else "") + \
        f" ({size / 1e6:.1f} MB)"


def delete_other(roots, library=None, min_age_s=600) -> list:
    """Delete everything under the incoming folders that isn't a FITS / XISF frame, then the folders left empty
    (the incoming folders themselves stay). Returns the paths deleted."""
    gone, parents = [], set()
    for p in other_files(roots, library, min_age_s):
        try:
            p.unlink()
            gone.append(str(p))
            parents.add(p.parent)
        except OSError:
            pass
    if parents:
        _remove_empty_dirs(parents, keep=[Path(r) for r in roots])
    return gone


def empty_frames(items) -> list:
    """The frames found to be empty - all zeros or 0 bytes, never written (see _empty_reason)."""
    return [it for it in items if it.status == "error" and (it.why or "").startswith(("all zeros", "empty file"))]


def delete_empty(items) -> list:
    """Delete the empty frames (Organize › 'Delete empty frames'): checked again just before - only a file that is
    still all zeros goes. Returns the paths deleted."""
    gone = []
    for it in empty_frames(items):
        if _empty_reason(it.src):
            try:
                it.src.unlink()
                it.status, it.why = "handled", "deleted: empty (never written)"
                gone.append(str(it.src))
            except OSError:
                pass
    return gone


def _remove_empty_dirs(dirs, stop_at=None, keep=None):
    """After a move, folders left completely empty are removed (never one that still has anything in it). keep:
    the folders you chose to organize - they stay (an incoming folder stays ready for the next night's frames),
    and nothing above them is ever touched."""
    roots = []
    for k in keep or []:
        try:
            roots.append(Path(k).resolve())
        except OSError:
            pass
    for d in sorted(dirs, key=lambda p: -len(p.parts)):
        p = d
        while p and _exists(p) and (stop_at is None or p.resolve() != Path(stop_at).resolve()):
            if roots:
                rp = p.resolve()
                if rp in roots or not any(r in rp.parents for r in roots):
                    break
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
    for m in log.get("meta") or []:                # N.I.N.A.'s metadata file: back in place, copies removed
        src, copies = Path(m["src"]), [Path(c) for c in m.get("copies") or []]
        try:
            if m.get("removed") and not src.exists():
                first = next((c for c in copies if c.exists()), None)
                if first is not None:
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(first, src)
            for c in copies:
                if c.exists():
                    c.unlink()
        except OSError:
            pass
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
