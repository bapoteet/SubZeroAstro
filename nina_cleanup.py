"""Clean Up My N.I.N.A. Computers: free space on the computers that run N.I.N.A. (PrimaLuceLab Eagle, RBFocus
Gaius, a mini PC …). N.I.N.A.'s Remote Copy plugin copies every frame to another computer or the NAS but never
deletes it from the rig's computer - and frames left there are copied again on later nights. This deletes a
frame from a N.I.N.A. computer only when the very same file is already filed in your organized library (same
name, same size, same contents) - so it has been copied AND organized - and it is older than the days you keep.
Everything else stays, with the reason."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import organizer as O

DAY = 86400.0


def load_list(settings) -> list:
    try:
        v = json.loads(settings.value("nina_clean/list", "[]") or "[]")
        return [c for c in v if isinstance(c, dict) and (c.get("folder") or "").strip()]
    except (ValueError, TypeError):
        return []


def save_list(settings, computers):
    settings.setValue("nina_clean/list", json.dumps(computers))


def library_of(settings) -> str:
    """The organized library the frames must be filed in: Organize Automatically's, else Organize's."""
    for k in ("auto_org/dst", "organize/dst"):
        v = (settings.value(k, "") or "").strip()
        if v:
            return v
    return ""


_NUMBERED = re.compile(r"^(.*) \((\d+)\)(\.[^.]+)$")


def _base_name(name: str) -> str:
    """'x (2).fits' -> 'x.fits': the organizer numbers a second file of the same name."""
    m = _NUMBERED.match(name)
    return (m.group(1) + m.group(3)) if m else name


def library_index(library, progress=None, stop=None) -> dict:
    """lower-case file name -> [(path, size)] of every frame filed in the organized library."""
    base = O.org_base(Path(library))
    idx: dict = {}
    n = 0
    for top in (O.LIGHT_DIR, O.CAL_DIR):
        root = base / top
        if not root.is_dir():
            continue
        for dp, dns, fns in os.walk(root):
            if stop and stop():
                return idx
            dns[:] = [d for d in dns if not O._nas_dir(d)]
            for fn in fns:
                if Path(fn).suffix.lower() not in O.EXTS:
                    continue
                p = Path(dp) / fn
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                idx.setdefault(_base_name(fn).lower(), []).append((p, size))
                n += 1
                if progress and n % 500 == 0:
                    progress(f"Reading your library: {n} frames", 0, 0)
    return idx


DEFAULT_KEEP = 3          # days a frame stays on a N.I.N.A. computer after it is filed (each computer its own)


def check_one(comp, idx, keep_days=None, now=None, progress=None, stop=None) -> dict:
    """{name, folder, error, safe: [(path, size, library copy)], keep: [(path, why)], bytes}. keep_days: the
    computer's own (a small drive fills in 2-3 nights, a big one can keep a week), else this, else 3."""
    now = time.time() if now is None else now
    try:
        keep_days = float(comp["keep_days"]) if comp.get("keep_days") is not None else (
            float(keep_days) if keep_days is not None else float(DEFAULT_KEEP))
    except (TypeError, ValueError):
        keep_days = float(DEFAULT_KEEP)
    rep = {"name": comp.get("name") or comp.get("folder"), "folder": comp.get("folder"), "error": "",
           "safe": [], "keep": [], "bytes": 0}
    root = Path(comp.get("folder") or "")
    if not str(root).strip() or not root.is_dir():
        rep["error"] = (f"can't reach {root} - is the computer on, is the folder shared, and is the path right? "
                        "(Try opening it in File Explorer / Finder.)")
        return rep
    files = []
    for dp, dns, fns in os.walk(root):
        if stop and stop():
            break
        dns[:] = [d for d in dns if not O._nas_dir(d)]
        for fn in fns:
            if Path(fn).suffix.lower() in O.EXTS:
                files.append(Path(dp) / fn)
    for i, p in enumerate(files):
        if stop and stop():
            break
        if progress and i % 25 == 0:
            progress(f"{rep['name']}: checking {i + 1} of {len(files)}", i, len(files))
        try:
            st = p.stat()
        except OSError as e:
            rep["keep"].append((p, f"couldn't read it ({e})"))
            continue
        if keep_days and now - st.st_mtime < keep_days * DAY:
            rep["keep"].append((p, f"newer than {keep_days:g} day(s)"))
            continue
        cands = [(q, s) for q, s in idx.get(p.name.lower(), []) if s == st.st_size]
        if not cands:
            why = ("not in your organized library yet (not copied, or not organized yet)"
                   if not idx.get(p.name.lower()) else "the library's file of that name is a different size")
            rep["keep"].append((p, why))
            continue
        try:
            mine = O._quick_hash(p)
        except OSError as e:
            rep["keep"].append((p, f"couldn't read it ({e})"))
            continue
        match = None
        for q, _s in cands:
            try:
                if O._quick_hash(q) == mine:
                    match = q
                    break
            except OSError:
                continue
        if match is None:
            rep["keep"].append((p, "the library's file of that name has different contents"))
            continue
        rep["safe"].append((p, st.st_size, match))
        rep["bytes"] += st.st_size
    return rep


def check(computers, library, keep_days=None, now=None, progress=None, stop=None) -> list:
    if not library:
        return [{"name": c.get("name") or c.get("folder"), "folder": c.get("folder"), "safe": [], "keep": [],
                 "bytes": 0, "error": "no organized library set (Organize › your library)"} for c in computers]
    idx = library_index(library, progress=progress, stop=stop)
    return [check_one(c, idx, keep_days, now, progress, stop) for c in computers]


def delete(reps, progress=None, stop=None) -> list:
    """Deletes the files each report found safe (checked again just before: still there, same size), then the
    folders that became empty - never the shared folder itself. [{name, deleted, bytes, failed: [(path, why)]}]"""
    out = []
    for rep in reps:
        done = {"name": rep["name"], "deleted": 0, "bytes": 0, "failed": []}
        root = Path(rep.get("folder") or "")
        dirs = set()
        for i, (p, size, match) in enumerate(rep.get("safe") or []):
            if stop and stop():
                break
            if progress and i % 25 == 0:
                progress(f"{rep['name']}: deleting {i + 1} of {len(rep['safe'])}", i, len(rep["safe"]))
            try:
                if not Path(match).is_file() or Path(match).stat().st_size != size:
                    done["failed"].append((p, "the library copy changed since the check - kept"))
                    continue
                os.remove(p)
                done["deleted"] += 1
                done["bytes"] += size
                dirs.add(Path(p).parent)
            except OSError as e:
                done["failed"].append((p, str(e)))
        for d in sorted(dirs, key=lambda x: len(str(x)), reverse=True):     # deepest first
            while d != root and root in d.parents:
                try:
                    d.rmdir()                                   # only when empty
                except OSError:
                    break
                d = d.parent
        out.append(done)
    return out


def describe_check(reps) -> str:
    out = []
    for r in reps:
        if r["error"]:
            out.append(f"{r['name']}: {r['error']}")
            continue
        t = f"{r['name']}: {len(r['safe'])} frame(s) safe to delete ({r['bytes'] / 1e9:.1f} GB)"
        if r["keep"]:
            why = {}
            for _p, w in r["keep"]:
                why[w] = why.get(w, 0) + 1
            t += ", kept " + ", ".join(f"{n} {w}" for w, n in sorted(why.items(), key=lambda kv: -kv[1]))
        out.append(t)
    return "\n".join(out) if out else "No N.I.N.A. computers set up."


def describe_delete(dels) -> str:
    out = []
    for d in dels:
        t = f"{d['name']}: {d['deleted']} frame(s) deleted ({d['bytes'] / 1e9:.1f} GB freed)"
        if d["failed"]:
            t += f", {len(d['failed'])} not deleted ({d['failed'][0][1]})"
        out.append(t)
    return "\n".join(out) if out else "Nothing to delete."


def run(settings, progress=None, stop=None) -> str:
    """Check and delete in one go (after Organize Automatically): the text for its report and the log."""
    comps = load_list(settings)
    if not comps:
        return ""
    reps = check(comps, library_of(settings), progress=progress, stop=stop)
    dels = delete(reps, progress=progress, stop=stop)
    by = {d["name"]: d for d in dels}
    parts = []
    for r in reps:
        if r["error"]:
            parts.append(f"{r['name']}: {r['error']}")
        else:
            d = by.get(r["name"], {"deleted": 0, "bytes": 0, "failed": []})
            parts.append(f"{r['name']}: {d['deleted']} deleted ({d['bytes'] / 1e9:.1f} GB), {len(r['keep'])} kept"
                         + (f", {len(d['failed'])} not deleted" if d["failed"] else ""))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    settings.setValue("nina_clean/last", f"{stamp}: " + "; ".join(parts))
    return "; ".join(parts)
