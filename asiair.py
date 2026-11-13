"""
asiair.py - fetch the night's subs from your ASIAIRs over the network, no USB stick.

An ASIAIR shares its storage on the network as Windows folders (\\\\<its IP>\\EMMC Images, \\Udisk Images …; its IP
is in the ASIAIR app's Wi-Fi settings). Fetch copies every FITS file that hasn't been fetched before into your
incoming folder, in a folder named after that ASIAIR's rig - so Organize files it under the right rig even when two
rigs have the same scope and camera. The .jpg previews are left behind. Each copy is checked (size) before it
counts; only then, and only if you ask, is the sub deleted from the ASIAIR to free its space. What was fetched is
remembered, so organizing (which empties the incoming folder) never makes a sub come over twice.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SETTLE_S = 120                       # a sub changed in the last 2 minutes may still be being written
GUESS_SHARES = ("EMMC Images", "Udisk Images", "TF Images", "SD Images")


def _exts():
    import fits_core as core
    return {e.lower() for e in core.LIGHT_EXTS}


def safe_name(name) -> str:
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", str(name or "")).strip().rstrip(".")
    return n or "ASIAIR"


def is_ip(address) -> bool:
    return bool(re.fullmatch(r"\s*\d{1,3}(\.\d{1,3}){3}\s*", str(address or "")))


def unc(address) -> str:
    """'192.168.1.50' -> '\\\\192.168.1.50'; anything else (a share or folder path) unchanged."""
    a = str(address or "").strip()
    return "\\\\" + a if is_ip(a) else a


def parse_net_view(text) -> list:
    """Disk share names from Windows' 'net view \\\\host' output."""
    out, table = [], False
    for ln in str(text or "").splitlines():
        if ln.startswith("---"):
            table = True
            continue
        if not table or not ln.strip():
            continue
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) >= 2 and parts[1].lower() in ("disk", "datenträger", "disque", "disco"):
            out.append(parts[0])
    return out


def shares(address) -> list:
    """The folders to look through for an ASIAIR: its image shares for an IP address, else the folder given."""
    a = str(address or "").strip()
    m = re.fullmatch(r"[\\/]{2}(\d{1,3}(?:\.\d{1,3}){3})[\\/]?", a)          # \\192.168.0.111 = the whole ASIAIR
    if m:
        a = m.group(1)
    if not a:
        return []
    if not is_ip(a):
        try:                                           # a folder: only if it's there (a typo is "can't reach")
            return [Path(a)] if Path(a).is_dir() else []
        except OSError:
            return []
    host = unc(a)
    names = []
    if sys.platform.startswith("win"):
        try:
            r = subprocess.run(["net", "view", host], capture_output=True, text=True, timeout=25,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            names = [n for n in parse_net_view(r.stdout) if "image" in n.lower()] or parse_net_view(r.stdout)
        except (OSError, subprocess.SubprocessError):
            names = []
    found = []
    for n in names or GUESS_SHARES:
        p = Path(host + "\\" + n)
        try:
            if p.is_dir():
                found.append(p)
        except OSError:
            pass
    return found


def scan(roots, now=None, stop=None) -> list:
    """[(file, share root, relative path, size, mtime)] for the subs on the ASIAIR (FITS / XISF, not the .jpgs);
    a sub written in the last couple of minutes is left for next time."""
    now = time.time() if now is None else now
    exts = _exts()
    out = []
    for root in roots:
        root = Path(root)
        for dp, dns, fns in os.walk(root):
            if stop and stop():
                return out
            dns[:] = sorted(d for d in dns if not d.startswith((".", "$")))
            for n in sorted(fns):
                p = Path(dp) / n
                if p.suffix.lower() not in exts:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if now - st.st_mtime < SETTLE_S:
                    continue
                out.append((p, root, p.relative_to(root).as_posix(), st.st_size, st.st_mtime))
    return out


def manifest_path(name) -> Path:
    import fits_core as core
    d = core.cache_dir() / "asiair"
    d.mkdir(parents=True, exist_ok=True)
    return d / (re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_name(name)) + ".json")


def load_manifest(name) -> dict:
    try:
        return json.loads(manifest_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_manifest(name, data):
    p = manifest_path(name)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(p)


def _tail(p) -> str:
    r"""A sub's place on its ASIAIR, the same however the row points at it (the IP, \\IP\EMMC Images or
    \\IP\EMMC Images\Plan): 'EMMC Images/Plan/M31/x.fit'."""
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.lower().rstrip("\\/").endswith("images"):
            return "/".join(x.strip("\\/") for x in parts[i:])
    t = str(p).replace("\\", "/").lstrip("/")
    return t.split("/", 1)[1] if re.match(r"^\d{1,3}(\.\d{1,3}){3}/", t) else t


def _mkey(root, rel, size, p=None) -> str:
    if p is not None:
        return f"{_tail(p)}|{size}"
    return f"{Path(root).name}/{rel}|{size}"


def _seen(seen, root, rel, size, p) -> bool:
    """Fetched before - under today's key, or the key of an earlier version (the row's last folder + path)."""
    return _mkey(root, rel, size, p) in seen or _mkey(root, rel, size) in seen


def _drop_previews(sub: Path) -> int:
    """The ASIAIR's .jpg previews of a sub just freed from it (same name, and its _thn thumbnail)."""
    n = 0
    for name in (sub.stem + ".jpg", sub.stem + ".jpeg", sub.stem + "_thn.jpg", sub.stem + "_thn.jpeg",
                 sub.stem.upper() + ".JPG"):
        q = sub.with_name(name)
        try:
            if q.is_file():
                q.unlink()
                n += 1
        except OSError:
            pass
    return n


def _wait_if_paused(pause, stop):
    """Pause button: hold here (between subs - never halfway through one) until resumed or stopped."""
    while pause and pause() and not (stop and stop()):
        time.sleep(0.4)


def fetch(unit: dict, incoming, progress=None, stop=None, free=False, now=None, roots=None, pause=None) -> dict:
    """Copy one ASIAIR's new subs into incoming/<rig>/<share>/…  unit = {name, address, rig}.
    Returns {name, copied, already, failed: [(file, why)], deleted, bytes, error}."""
    name = safe_name(unit.get("name") or unit.get("address"))
    rep = {"name": name, "copied": 0, "already": 0, "failed": [], "deleted": 0, "jpgs": 0, "bytes": 0, "error": ""}
    roots = shares(unit.get("address")) if roots is None else roots
    if not roots:
        rep["error"] = (f"can't reach {unit.get('address') or '(no address)'} - is the ASIAIR on and on the same "
                        "network? (Its IP is in the ASIAIR app's Wi-Fi settings.)")
        return rep
    dest_top = Path(incoming) / safe_name(unit.get("rig") or name)
    seen = load_manifest(name)
    if progress:
        progress(f"{name}: looking for new subs…", 0, 0)
    subs = scan(roots, now=now, stop=stop)
    todo = [s for s in subs if not _seen(seen, s[1], s[2], s[3], s[0])]
    rep["already"] = len(subs) - len(todo)
    if free:                                     # fetched (and checked) on an earlier run: free them now
        for p, root, rel, size, mt in subs:
            if _seen(seen, root, rel, size, p):
                try:
                    p.unlink()
                    rep["deleted"] += 1
                    rep["jpgs"] += _drop_previews(p)
                except OSError:
                    pass
    for i, (p, root, rel, size, mt) in enumerate(todo):
        _wait_if_paused(pause, stop)
        if stop and stop():
            break
        if progress:
            progress(f"{name}: {p.name}", i, len(todo))
        dst = dest_top / _tail(p)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not (dst.exists() and dst.stat().st_size == size):
                tmp = dst.with_name(dst.name + ".fetching")
                shutil.copy2(p, tmp)
                if tmp.stat().st_size != size:
                    tmp.unlink()
                    raise OSError("the copy came out a different size")
                tmp.replace(dst)
                rep["copied"] += 1
                rep["bytes"] += size
            seen[_mkey(root, rel, size, p)] = mt
            if free:
                try:
                    p.unlink()
                    rep["deleted"] += 1
                    rep["jpgs"] += _drop_previews(p)
                except OSError as e:
                    rep["failed"].append((rel, f"copied, but couldn't delete it from the ASIAIR: {e}"))
        except Exception as e:  # noqa: BLE001
            rep["failed"].append((rel, str(e)))
        if i % 25 == 0:
            save_manifest(name, seen)
    save_manifest(name, seen)
    return rep


def fetch_all(units, incoming, progress=None, stop=None, free=False, together=True, pause=None) -> list:
    """Every ASIAIR's new subs. together: all of them at the same time - each ASIAIR has its own Wi-Fi radio,
    and that radio (not the network) is what's slow, so four ASIAIRs come over in about the time of one."""
    units = [u for u in units if u.get("address")]
    if not together or len(units) < 2:
        return [fetch(u, incoming, progress, stop, free, pause=pause) for u in units]
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    state = {}

    def prog_for(name):
        def p(text, i, n):
            if progress is None:
                return
            with lock:
                state[name] = (i, n)
                done = sum(a for a, _b in state.values())
                total = sum(b for _a, b in state.values())
                progress(f"Fetching from {len(units)} ASIAIRs at once - " + ", ".join(
                    f"{k}: {a}/{b}" for k, (a, b) in state.items()), done, total)
        return p
    # rows of the same ASIAIR (same name - e.g. its Plan and Autorun folders as two rows) go one after the other:
    # they share its radio and its record of what's been fetched
    groups = {}
    for u in units:
        groups.setdefault(safe_name(u.get("name") or u.get("address")), []).append(u)

    def run_group(name, us):
        return [fetch(u, incoming, prog_for(name), stop, free, pause=pause) for u in us]
    with ThreadPoolExecutor(max_workers=min(8, len(groups))) as pool:
        futs = [pool.submit(run_group, n, us) for n, us in groups.items()]
        return [r for f in futs for r in f.result()]


def describe(reps) -> str:
    out = []
    for r in reps:
        if r["error"]:
            out.append(f"{r['name']}: not fetched - {r['error']}")
            continue
        t = f"{r['name']}: {r['copied']} new sub(s) ({r['bytes'] / 1e9:.1f} GB)"
        if r["deleted"]:
            t += f", {r['deleted']} freed on the ASIAIR" + (f" (+{r['jpgs']} .jpg)" if r.get("jpgs") else "")
        if r["failed"]:
            t += f", {len(r['failed'])} failed ({r['failed'][0][1]})"
        out.append(t)
    return "\n".join(out) if out else "No ASIAIRs set up."


def load_units(settings) -> list:
    try:
        v = json.loads(settings.value("asiair/list", "[]") or "[]")
        return [u for u in v if isinstance(u, dict)]
    except (ValueError, TypeError):
        return []


def save_units(settings, units):
    settings.setValue("asiair/list", json.dumps(units))


def last_text(rep, stamp) -> str:
    """What the Last fetch column shows for one ASIAIR after a fetch round."""
    return f"{stamp}: not reached" if rep.get("error") else f"{stamp}: {rep.get('copied', 0)} new"


def record_last(settings, reps, stamp=None) -> list:
    """Every fetch - Fetch Now, the overnight rounds, before Organize Automatically - writes each ASIAIR's Last fetch
    (the overnight rounds once left the column at the last Fetch Now, days old). Returns the saved units."""
    import time as _t
    stamp = stamp or _t.strftime("%Y-%m-%d %H:%M")
    by = {r.get("name"): r for r in reps or []}
    units = load_units(settings)
    for u in units:
        r = by.get(safe_name(u.get("name") or u.get("address") or ""))
        if r is not None:
            u["last"] = last_text(r, stamp)
    save_units(settings, units)
    return units


JPG_EXTS = (".jpg", ".jpeg")


def jpgs(roots, stop=None) -> list:
    """The .jpg pictures on the ASIAIR (its previews, saved next to every sub) - [(path, size)]."""
    out = []
    for root in roots:
        for dp, dns, fns in os.walk(Path(root)):
            if stop and stop():
                return out
            dns[:] = [d for d in dns if not d.startswith((".", "$"))]
            for n in fns:
                if n.lower().endswith(JPG_EXTS):
                    p = Path(dp) / n
                    try:
                        out.append((p, p.stat().st_size))
                    except OSError:
                        pass
    return out


def jpg_summary(found, roots) -> str:
    """'812 .jpg (1.4 GB): Autorun 790, Live 22'."""
    import collections
    c = collections.Counter()
    for p, _s in found:
        top = ""
        for r in roots:
            try:
                top = p.relative_to(r).parts[0] if len(p.relative_to(r).parts) > 1 else "(top folder)"
                break
            except ValueError:
                continue
        c[top] += 1
    size = sum(s for _p, s in found)
    return f"{len(found)} .jpg ({size / 1e9:.2f} GB)" + (": " + ", ".join(f"{k} {n}" for k, n in c.most_common())
                                                        if c else "")


def delete_jpgs(found, progress=None, stop=None) -> tuple:
    """Delete these .jpgs from the ASIAIR -> (deleted, failed)."""
    done = failed = 0
    for i, (p, _s) in enumerate(found):
        if stop and stop():
            break
        if progress and i % 25 == 0:
            progress(f"Deleting .jpgs on the ASIAIR… {i} of {len(found)}", i, len(found))
        try:
            p.unlink()
            done += 1
        except OSError:
            failed += 1
    return done, failed

