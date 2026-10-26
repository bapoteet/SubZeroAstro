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
TOP = ("Targets", "Darks library", "Bias library", "Calibration masters", "Flats", "Unsorted", LOG_DIR)
EXTS = {e.lower() for e in core.LIGHT_EXTS}
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
    return f"{cam}_G{_num_text(info.gain)}_{_num_text(info.temp, '{:.0f}')}C"


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
            if n.startswith(".") or n == LOG_DIR or n.endswith((".szproj", ".saving")) or _is_project_dir(p):
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


def classify(paths, read_info=None, progress=None, stop=None) -> list:
    """Read each header and decide where the file belongs (relative to the library). Returns [Item]."""
    import stacking as st
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
    lights_on = {}
    for it, info in zip(items, infos):
        if info is not None and info.type == "light" and not info.master and info.target and info.night:
            lights_on.setdefault(info.night, set()).add((st.target_name(info.target), info.telescope or ""))

    for it, info in zip(items, infos):
        if info is None:
            continue
        name = it.src.name
        rejected = it.src.parent.name.upper() == core.REJECTED_DIR
        typ = info.type
        flt = _safe(info.filter or "No filter")
        night = info.night or "Unknown date"
        tgt = st.target_name(info.target) if info.target else ""
        it.night, it.filter, it.target = night, flt, tgt
        if info.date_obs:
            it.key = (info.date_obs, info.exptime, flt, info.camera, it.size, typ)
        looks_master = info.master or "_prepared" in it.src.stem or re.search(r"master", name, re.I)
        if typ in (None, "light") and looks_master and tgt:
            it.kind, it.dst = "master", Path("Targets") / tgt / "Masters" / name
            it.why = "master light"
            it.companions = _companions(it.src)
        elif typ in ("dark", "flat", "darkflat", "bias") and looks_master:
            it.kind, it.dst = "calmaster", Path("Calibration masters") / name
            it.why = f"master {typ}"
            it.companions = _companions(it.src)
        elif typ == "light" or (typ is None and tgt and info.exptime):
            if not tgt:
                it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "a light without a target name"
            else:
                sub = Path("Targets") / tgt / night / "LIGHT" / flt
                if rejected:
                    sub = sub / core.REJECTED_DIR
                it.kind, it.dst = "light", sub / name
                it.why = "light" + (" (rejected on page 1)" if rejected else "")
        elif typ in ("flat", "darkflat"):
            folder = "FLAT" if typ == "flat" else "DARKFLAT"
            owners = sorted(lights_on.get(night, set()))
            same_scope = [t for t, tel in owners if tel == (info.telescope or "")]
            if tgt and any(t == tgt for t, _ in owners):
                owner = tgt
            elif len(set(same_scope)) == 1:
                owner = same_scope[0]
            elif len({t for t, _ in owners}) == 1:
                owner = owners[0][0]
            else:
                owner = None
            if owner:
                it.kind, it.dst = typ, Path("Targets") / owner / night / folder / flt / name
                it.why = f"{typ} of the night ({owner})"
            else:
                it.kind, it.dst = typ, Path("Flats") / night / (folder if typ == "darkflat" else "") / flt / name
                it.why = f"{typ} - several (or no) targets that night"
        elif typ == "dark":
            it.kind = "dark"
            it.dst = Path("Darks library") / _cal_folder(info) / f"{_num_text(info.exptime)}s" / name
            it.why = "dark (library by camera, gain, temperature, exposure)"
        elif typ == "bias":
            it.kind, it.dst = "bias", Path("Bias library") / _cal_folder(info) / name
            it.why = "bias (library by camera, gain, temperature)"
        else:
            it.kind, it.dst, it.why = "unsorted", Path("Unsorted") / name, "the header doesn't say what it is"
    if progress:
        progress("Headers read", len(paths))
    return items


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
                    it.status = "duplicate"
                    break
            elif cand.exists():
                try:
                    if cand.stat().st_size == it.size and _quick_hash(cand) == _quick_hash(it.src):
                        it.status = "duplicate"
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
    logs = sorted(p for p in d.glob("organize_*.json") if not p.name.endswith(".undone.json"))
    return logs[-1] if logs else None
