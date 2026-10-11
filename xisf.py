"""
xisf.py - read and write PixInsight XISF images (NumPy only).

Reading supports monolithic XISF 1.0 files with uncompressed, zlib or
zlib+sh (byte-shuffled) image data, which covers PixInsight's defaults.
LZ4 / Zstandard compressed files give a clear error asking to re-save them.
Writing produces uncompressed 32-bit float images that PixInsight opens directly.

Images are returned as float32 arrays shaped (h, w) or (channels, h, w), row 0
first as stored (XISF stores the top row first; PixInsight reads FITS files the
same way, so XISF masters line up with FITS subframes without flipping).
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

SIGNATURE = b"XISF0100"
NS = "{http://www.pixinsight.com/xisf}"
_FORMATS = {"UInt8": "u1", "UInt16": "u2", "UInt32": "u4", "Float32": "f4", "Float64": "f8"}


def _parse_value(text: str):
    t = (text or "").strip()
    if t.startswith("'"):
        return t[1:-1].replace("''", "'").rstrip() if t.endswith("'") and len(t) > 1 else t[1:]
    if t in ("T", "F"):
        return t == "T"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t or None


def read_header(path):
    """Return (xml_root, image_element, cards, lookup) without reading pixel data."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != SIGNATURE:
            raise ValueError("Not an XISF file")
        hlen = struct.unpack("<I", f.read(4))[0]
        f.read(4)
        xml = f.read(hlen).decode("utf-8", errors="replace")
    root = ET.fromstring(xml)
    img = root.find(f"{NS}Image")
    if img is None:
        img = root.find("Image")
    if img is None:
        raise ValueError("XISF file contains no image")
    cards, look = [], {}
    for kw in list(img.iter(f"{NS}FITSKeyword")) + list(img.iter("FITSKeyword")):
        name = kw.get("name", "").strip()
        if not name:
            continue
        val = _parse_value(kw.get("value", ""))
        if name in ("COMMENT", "HISTORY"):
            cards.append((name, None, kw.get("comment", "") or kw.get("value", "")))
            continue
        cards.append((name, val, kw.get("comment", "")))
        look.setdefault(name, val)
    # Useful XISF properties when FITS keywords are missing
    props = {p.get("id"): (p.get("value") if p.get("value") is not None else (p.text or ""))
             for p in list(img.iter(f"{NS}Property")) + list(img.iter("Property"))}
    if "INSTRUME" not in look and props.get("Instrument:Camera:Name"):
        look["INSTRUME"] = props["Instrument:Camera:Name"]
    if "FILTER" not in look and props.get("Instrument:Filter:Name"):
        look["FILTER"] = props["Instrument:Filter:Name"]
    if "EXPTIME" not in look and props.get("Instrument:FrameExposureTime"):
        try:
            look["EXPTIME"] = float(props["Instrument:FrameExposureTime"])
        except ValueError:
            pass
    if "IMAGETYP" not in look and img.get("imageType"):
        look["IMAGETYP"] = img.get("imageType")
    if "BAYERPAT" not in look and props.get("PCL:CFASourcePattern"):
        look["BAYERPAT"] = props["PCL:CFASourcePattern"]
    w, h, c = (int(x) for x in img.get("geometry").split(":")[:3])
    look["NAXIS1"], look["NAXIS2"] = w, h
    look["_XISF_CHANNELS"] = c
    look["_XISF_BOUNDS"] = img.get("bounds", "")
    look["_XISF_FORMAT"] = img.get("sampleFormat", "Float32")
    return root, img, cards, look


def read_xisf(path):
    """Return (data float32, cards, lookup)."""
    _root, img, cards, look = read_header(path)
    w, h, c = (int(x) for x in img.get("geometry").split(":")[:3])
    fmt = img.get("sampleFormat", "Float32")
    if fmt not in _FORMATS:
        raise ValueError(f"Unsupported XISF sample format {fmt}")
    order = ">" if img.get("byteOrder", "little") == "big" else "<"
    dt = np.dtype(order + _FORMATS[fmt])
    loc = img.get("location", "")
    m = re.match(r"attachment:(\d+):(\d+)", loc)
    if not m:
        raise ValueError("Only XISF files with attached image data are supported")
    pos, size = int(m.group(1)), int(m.group(2))
    with open(path, "rb") as f:
        f.seek(pos)
        raw = f.read(size)
    comp = img.get("compression", "")
    if comp:
        parts = comp.split(":")
        codec = parts[0].lower()
        if codec in ("zlib", "zlib+sh"):
            raw = zlib.decompress(raw)
            if codec == "zlib+sh":
                item = int(parts[2]) if len(parts) > 2 else dt.itemsize
                n = len(raw) // item
                raw = np.frombuffer(raw, np.uint8)[: n * item].reshape(item, n).T.tobytes()
        else:
            raise ValueError(f"XISF compression '{codec}' is not supported - re-save the file in "
                             "PixInsight without compression (or with zlib)")
    data = np.frombuffer(raw, dtype=dt, count=w * h * c)
    if img.get("pixelStorage", "Planar").lower() == "normal":
        data = data.reshape(h, w, c).transpose(2, 0, 1)
    else:
        data = data.reshape(c, h, w)
    data = data.astype(np.float32)
    if c == 1:
        data = data[0]
    return data, cards, look


def _fmt_value(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = f"{float(v):.10G}"
        return s if any(ch in s for ch in ".EN") else s + "."
    return "'" + str(v).replace("'", "''") + "'"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def write_xisf(path, data: np.ndarray, keywords: list | dict | None = None, image_type: str | None = None):
    """
    Write an uncompressed Float32 XISF file. data is (h, w) or (3, h, w) RGB.
    keywords: list of (name, value, comment) or dict name -> value.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        c, h, w = 1, data.shape[0], data.shape[1]
        planes = data[None]
    else:
        c, h, w = data.shape
        planes = data
    body = np.ascontiguousarray(planes.astype("<f4")).tobytes()
    if isinstance(keywords, dict):
        keywords = [(k, v, "") for k, v in keywords.items()]
    kw_xml = "".join(
        f'<FITSKeyword name="{_esc(k)}" value="{_esc(_fmt_value(v)) if v is not None else ""}" '
        f'comment="{_esc(cmt or "")}"/>'
        for k, v, cmt in (keywords or []))
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    color = "RGB" if c == 3 else "Gray"
    itype = f' imageType="{_esc(image_type)}"' if image_type else ""

    def header(pos):
        return ('<?xml version="1.0" encoding="UTF-8"?>'
                '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f'<Image geometry="{w}:{h}:{c}" sampleFormat="Float32" bounds="{lo:.8g}:{hi:.8g}" '
                f'colorSpace="{color}"{itype} location="attachment:{pos}:{len(body)}">'
                f'{kw_xml}</Image>'
                '<Metadata><Property id="XISF:CreatorApplication" type="String">'
                'FITS Blaster for Windows</Property></Metadata></xisf>').encode("utf-8")

    pos = 4096
    while True:
        xml = header(pos)
        if 16 + len(xml) <= pos:
            break
        pos += 4096
    with open(path, "wb") as f:
        f.write(SIGNATURE)
        f.write(struct.pack("<I", len(xml)))
        f.write(b"\0\0\0\0")
        f.write(xml)
        f.write(b"\0" * (pos - 16 - len(xml)))
        f.write(body)
