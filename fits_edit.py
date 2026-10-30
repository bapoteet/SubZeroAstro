"""
fits_edit.py - change header keywords in FITS files, many at once, safely.

Only the header is rewritten; the image data is copied byte for byte. When the new header still fits in the
blocks the old one used, the file is changed in place; otherwise it is written to a temporary file next to it and
swapped in, so a failure never leaves a half-written frame. Every change is recorded in a small undo file in the
frames' folder (.subzeroastro_header_undo.json) with the values it replaced, so it can be put back.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

BLOCK, CARD = 2880, 80
UNDO_NAME = ".subzeroastro_header_undo.json"
# structural keywords that describe the data itself: changing them would corrupt the file
PROTECTED = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END",
             "PCOUNT", "GCOUNT", "XTENSION"}
FITS_EXT = (".fits", ".fit", ".fts")


def _ascii(s) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(s))


def format_card(key: str, value, comment: str = "") -> str:
    """One 80-character FITS card."""
    key = key.upper()[:8]
    if isinstance(value, bool):
        v = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        v = f"{value:>20}"
    elif isinstance(value, float):
        s = f"{value:.10G}"
        if not any(ch in s for ch in ".EN"):
            s += "."
        v = f"{s:>20}"
    else:
        s = _ascii(value).replace("'", "''")[:68]
        v = f"'{s:<8}'"
    line = f"{key:<8}= {v}"
    if comment:
        line += f" / {_ascii(comment)}"
    return f"{line:<80}"[:80]


def typed(text: str):
    """What the user typed, as the FITS type it looks like: 300 -> int, 300.5 -> float, T/F -> bool, else text."""
    t = str(text).strip()
    if t.upper() in ("T", "F"):
        return t.upper() == "T"
    try:
        return int(t)
    except ValueError:
        pass
    if any(c.isdigit() for c in t):
        try:
            return float(t)
        except ValueError:
            pass
    return t


def _read_header(f):
    """(list of 80-char cards without END, header length in bytes)."""
    cards, n = [], 0
    while True:
        block = f.read(BLOCK)
        if len(block) < BLOCK:
            raise ValueError("not a FITS file (the header has no END)")
        n += BLOCK
        text = block.decode("ascii", errors="replace")
        for i in range(0, BLOCK, CARD):
            c = text[i:i + CARD]
            if c[:8].strip() == "END" and c[8:].strip() == "":
                return cards, n
            cards.append(c)


def _value_of(card: str):
    import fits_core as core
    return core.parse_card(card)


def read_values(path, keys) -> dict:
    """{key: value or None} from a FITS file's primary header."""
    with open(path, "rb") as f:
        cards, _n = _read_header(f)
    out = {k.upper(): None for k in keys}
    for c in cards:
        k, v, _cm = _value_of(c)
        if k in out and out[k] is None:
            out[k] = v
    return out


def edit_header(path, changes: dict, undo: dict | None = None) -> dict:
    """Set (value) or remove (None) keywords in a FITS file's primary header. Returns {key: old value} for the
    keywords it changed (None = it was not there). `undo`, when given, gets the same record under the file name."""
    path = Path(path)
    if path.suffix.lower() not in FITS_EXT:
        raise ValueError(f"{path.name}: only FITS files can be edited (XISF headers are left alone)")
    bad = [k for k in changes if k.upper() in PROTECTED]
    if bad:
        raise ValueError(f"{', '.join(bad)} describe the image data itself and can't be changed")
    with open(path, "rb") as f:
        cards, old_len = _read_header(f)
    old = {}
    for key, value in changes.items():
        key = key.upper()[:8]
        idx = next((i for i, c in enumerate(cards) if c[:8].strip() == key and c[8:10] == "= "), None)
        prev = _value_of(cards[idx]) if idx is not None else None
        old[key] = prev[1] if prev else None
        if value is None:
            if idx is not None:
                del cards[idx]
            continue
        comment = prev[2] if prev else ""
        new = format_card(key, value, comment)
        if idx is not None:
            cards[idx] = new
        else:
            cards.append(new)
    body = "".join(cards) + "END".ljust(CARD)
    n = (len(body) + BLOCK - 1) // BLOCK * BLOCK
    head = body.ljust(n).encode("ascii")
    if n == old_len:
        with open(path, "r+b") as f:                      # same size: only the header blocks are rewritten
            f.write(head)
    else:
        tmp = path.with_name(path.name + ".sza_tmp")
        try:
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                src.seek(old_len)
                dst.write(head)
                shutil.copyfileobj(src, dst, 1 << 22)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    if undo is not None:
        undo[path.name] = {k: v for k, v in old.items()}
    return old


def edit_many(paths, changes: dict, progress=None) -> tuple[list, list]:
    """Edit every file; records the old values per folder in its undo file. (done paths, [(path, error)])."""
    done, failed = [], []
    by_folder: dict = {}
    paths = list(paths)
    for i, p in enumerate(paths):
        p = Path(p)
        rec = by_folder.setdefault(p.parent, {})
        try:
            edit_header(p, changes, rec)
            done.append(p)
        except Exception as e:  # noqa: BLE001 - one bad file never stops the rest
            failed.append((p, str(e)))
        if progress:
            progress(i + 1, len(paths))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for folder, rec in by_folder.items():
        if not rec:
            continue
        uf = folder / UNDO_NAME
        try:
            hist = json.loads(uf.read_text(encoding="utf-8")) if uf.exists() else []
        except (OSError, ValueError):
            hist = []
        hist.append({"time": stamp, "changes": {k: v for k, v in changes.items()}, "old": rec})
        try:
            uf.write_text(json.dumps(hist, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass
    return done, failed


def undo_last(folder) -> tuple[int, str]:
    """Put back the values the last edit in this folder replaced. (files restored, what was undone)."""
    uf = Path(folder) / UNDO_NAME
    hist = json.loads(uf.read_text(encoding="utf-8")) if uf.exists() else []
    if not hist:
        return 0, ""
    last = hist.pop()
    n = 0
    for name, old in last["old"].items():
        p = Path(folder) / name
        if p.exists():
            try:
                edit_header(p, old)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    uf.write_text(json.dumps(hist, indent=1, default=str), encoding="utf-8")
    what = ", ".join(f"{k}" for k in last["changes"])
    return n, f"{what} (changed {last['time']})"
