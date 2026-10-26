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
TOP = ("Lights", "Calibration", "Targets", "Darks library", "Bias library", "Calibration masters", "Flats", "Unsorted", LOG_DIR)
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


def _safe(s) -> str:
    s = core._ascii(str(s or ""))
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().rstrip(".")
    return s or "Unknown"


def _num_text(v, fmt="{:g}") -> str:
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "?"


def _cal_folder(info) -> str:
    import stacking as st
    cam = st.short_camera(info.camera)
    off = f"_O{_num_text(info.offset)}" if getattr(info, "offset", None) is not None else ""
    return f"{cam}_G{_num_text(info.gain)}{off}_{_num_text(info.temp, '{:.0f}')}C"


_RIGS = None          # SubZeroAstro's own rig names (Calibrate and Stack › rig names), set by classify()


def _rig(info, path=None) -> str:
    """Which rig (telescope + the camera that lives on it) a file comes from: the name you gave the rig in
    SubZeroAstro, else the telescope in the header (not a mount driver such as EQMOD), else the focal length,
    else - for calibration frames sorted by hand - the folder above the Darks / Flats / Bias folder."""
    import stacking as st
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


def _is_project_dir(d: Path) -> bool:
    m = d / "manifest.json"
    if not m.is_file():
        return False
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("kind") == "finish-project"
    except (OSError, ValueError):
        return False


def scan(root, progress=None, stop=None) -> list:
    """Every FITS / XISF file under root (all subfolders), skipping saved projects, SubZeroAstro's own
    work folders and the organize logs. Returns [Path]."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        keep = []
        for n in sorted(dirnames, key=core.natural_key):
            p = d / n
            if (n.startswith((".", "$")) or n == LOG_DIR or n.endswith((".szproj", ".saving"))
                    or n.lower() in WORK_DIRS or _is_project_dir(p)):
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
            if sib.is_file() and sib.suffix.lower() in COMPANION_EXTS:
                out.append(sib)
    except OSError:
        pass
    return out


LAYOUTS = {"rig": "By rig: Lights / rig / target / night  and  Calibration / rig / Flats, Darks, Bias",
           "target": "By target: Targets / target / night_rig, with darks and bias libraries"}


def classify(paths, read_info=None, progress=None, stop=None, library=None, rigs=None, layout="rig") -> list:
    """Read each header and decide where the file belongs (relative to the library). Returns [Item].
    rigs: SubZeroAstro's saved rig names (stacking.Rigs), so a rig you named keeps its name."""
    import stacking as st
    global _RIGS
    _RIGS = rigs
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
        items.append(it)
        infos.append(info)

    # which targets had lights on which night (with which telescope) - flats without a target go there
    lights_on, rigs, existing = {}, {}, {}
    for it, info in zip(items, infos):
        if info is not None and info.type == "light" and not info.master and info.target and info.night:
            lights_on.setdefault(info.night, set()).add((st.target_name(info.target), _rig(info)))
            rigs.setdefault(st.target_name(info.target), set()).add(_rig(info))
    # nights already in the library (organized before): flats sorted on their own - a folder of calibration
    # frames only - still find the target that was shot that night
    if library is not None:
        tdir = Path(library) / "Targets"
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
        if info.date_obs:
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
                f"{_num_text(info.exptime)}s" / name
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
    """Lights / <rig> / <target> / <night> / <filter>   and   Calibration / <rig> / Flats | Dark flats | Darks |
    Bias | Masters - everything a rig (one telescope with its own camera) took, in one place."""
    import stacking as st
    rig = _rig(info, it.src) or "Unknown rig"
    cal = Path("Calibration") / rig
    if typ in (None, "light") and looks_master and tgt:
        it.kind, it.dst, it.why = "master", Path("Lights") / rig / tgt / "Masters" / name, "master light"
        it.companions = _companions(it.src)
    elif typ in ("dark", "flat", "darkflat", "bias") and looks_master:
        sub = Path(_cal_folder(info)) if typ in ("dark", "bias") else Path(f"Flats {night}") / flt
        it.kind, it.dst, it.why = "calmaster", cal / "Masters" / sub / name, f"master {typ} ({rig})"
        it.companions = _companions(it.src)
    elif typ == "light" or (typ is None and tgt and info.exptime):
        if not tgt:
            it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "a light without a target name"
            return
        sub = Path("Lights") / rig / tgt / night / flt
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
        it.dst = cal / "Darks" / _cal_folder(info) / f"{_num_text(info.exptime)}s" / name
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
        in_place = x.dst is not None and (library / x.dst).resolve() == x.src.resolve()
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
        if it.dst is None or it.status in ("error", "duplicate"):
            continue
        dst = library / it.dst
        if dst.resolve() == it.src.resolve():
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
            elif cand.exists():
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
    logd = library / LOG_DIR
    logd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_path = logd / f"organize_{stamp}_{'deleted' if delete else 'duplicates'}.json"
    done, kept_back, failed = [], [], []
    for i, it in enumerate(dups):
        if stop and stop():
            break
        if progress:
            progress(f"Checking {it.src.name}", i, len(dups))
        if not it.src.exists():
            kept_back.append((it, "it is no longer there"))
            continue
        if not (it.same_as and Path(it.same_as).exists()):
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
                dst = library / DUP_DIR / it.src.name
                n = 1
                while dst.exists():
                    n += 1
                    dst = library / DUP_DIR / f"{it.src.stem}_{n}{it.src.suffix}"
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
    logd = library / LOG_DIR
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
                if dst.exists():
                    continue
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
        while p and p.exists() and (stop_at is None or p.resolve() != Path(stop_at).resolve()):
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
    d = Path(library) / LOG_DIR
    if not d.is_dir():
        return None
    logs = sorted((p for p in d.glob("organize_*.json")
                   if not p.name.endswith((".undone.json", "_deleted.json"))), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None
