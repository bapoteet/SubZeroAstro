"""
xisf.py - read and write PixInsight XISF images (NumPy only).

Reading supports monolithic XISF 1.0 files, uncompressed or compressed with any of PixInsight's
codecs: zlib, LZ4, LZ4-HC and Zstandard, each with or without byte shuffling ("+sh").
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

try:                                    # fast LZ4 (a pure-Python fallback is below)
    import lz4.block as _lz4block
except ImportError:                     # pragma: no cover - depends on the installation
    _lz4block = None
try:
    import zstandard as _zstd
except ImportError:                     # pragma: no cover
    _zstd = None

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
        raw = decompress(raw, comp, dt.itemsize)
    data = np.frombuffer(raw, dtype=dt, count=w * h * c)
    if img.get("pixelStorage", "Planar").lower() == "normal":
        data = data.reshape(h, w, c).transpose(2, 0, 1)
    else:
        data = data.reshape(c, h, w)
    data = data.astype(np.float32)
    if c == 1:
        data = data[0]
    return data, cards, look


def lz4_block_decompress(src: bytes, size: int) -> bytes:
    """Decode one raw LZ4 block (no frame header) - used when the lz4 package is missing."""
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0:
            raise ValueError("damaged LZ4 data")
        ml = token & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(out) - off
        if start < 0:
            raise ValueError("damaged LZ4 data")
        if off >= ml:
            out += out[start:start + ml]
        else:                                   # overlapping copy: repeat the pattern
            pat = bytes(out[start:])
            out += (pat * (ml // off + 1))[:ml]
    if len(out) != size:
        raise ValueError("damaged LZ4 data (wrong size)")
    return bytes(out)


def decompress(raw: bytes, comp: str, itemsize: int) -> bytes:
    """Undo XISF compression 'codec:uncompressed-size[:item-size]'."""
    parts = comp.split(":")
    codec = parts[0].strip().lower()
    size = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
    shuffled = codec.endswith("+sh")
    base = codec[:-3] if shuffled else codec
    if base == "zlib":
        out = zlib.decompress(raw)
    elif base in ("lz4", "lz4hc"):
        if size is None:
            raise ValueError("XISF LZ4 data without its size")
        out = _lz4block.decompress(raw, uncompressed_size=size) if _lz4block is not None else \
            lz4_block_decompress(raw, size)
    elif base == "zstd":
        if _zstd is not None:
            out = _zstd.ZstdDecompressor().decompress(raw, max_output_size=size or 0)
        else:
            try:
                from compression import zstd as _std_zstd      # Python 3.14+
            except ImportError:
                raise ValueError("this XISF file is Zstandard-compressed and the zstandard module is "
                                 "missing - re-save it in PixInsight with zlib or LZ4 compression")
            out = _std_zstd.decompress(raw)
    else:
        raise ValueError(f"XISF compression '{codec}' is not supported - re-save the file in PixInsight "
                         "without compression (or with zlib)")
    if shuffled:
        item = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else itemsize
        n = len(out) // item
        out = np.frombuffer(out, np.uint8)[: n * item].reshape(item, n).T.tobytes()
    return out


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


def write_xisf(path, data: np.ndarray, keywords: list | dict | None = None, image_type: str | None = None,
               bounds: tuple | None = None, sample_format: str = "Float32", compression: str | None = None):
    """
    Write an XISF file (Float32 by default; "UInt16" like a camera frame). data is (h, w) or (3, h, w) RGB.
    keywords: list of (name, value, comment) or dict name -> value.
    compression: None, or "zlib" / "lz4" / "lz4hc" / "zstd", optionally with "+sh" (byte shuffling).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        c, h, w = 1, data.shape[0], data.shape[1]
        planes = data[None]
    else:
        c, h, w = data.shape
        planes = data
    if sample_format == "UInt16":
        body = np.ascontiguousarray(np.clip(np.round(planes), 0, 65535).astype("<u2")).tobytes()
        item = 2
    else:
        sample_format = "Float32"
        body = np.ascontiguousarray(planes.astype("<f4")).tobytes()
        item = 4
    comp_attr = ""
    if compression:
        codec = compression.lower()
        base = codec[:-3] if codec.endswith("+sh") else codec
        src = body
        if codec.endswith("+sh"):
            src = np.frombuffer(body, np.uint8).reshape(-1, item).T.tobytes()
        if base == "zlib":
            packed = zlib.compress(src, 6)
        elif base in ("lz4", "lz4hc"):
            if _lz4block is None:
                raise ValueError("LZ4 compression needs the lz4 module")
            packed = _lz4block.compress(src, mode="high_compression" if base == "lz4hc" else "default",
                                        store_size=False)
        elif base == "zstd":
            if _zstd is None:
                raise ValueError("Zstandard compression needs the zstandard module")
            packed = _zstd.ZstdCompressor(level=3).compress(src)
        else:
            raise ValueError(f"unknown XISF compression {compression}")
        comp_attr = f' compression="{codec}:{len(body)}' + (f':{item}"' if codec.endswith("+sh") else '"')
        body = packed
    if isinstance(keywords, dict):
        keywords = [(k, v, "") for k, v in keywords.items()]
    kw_xml = "".join(
        f'<FITSKeyword name="{_esc(k)}" value="{_esc(_fmt_value(v)) if v is not None else ""}" '
        f'comment="{_esc(cmt or "")}"/>'
        for k, v, cmt in (keywords or []))
    lo, hi = bounds if bounds else (float(np.nanmin(data)), float(np.nanmax(data)))
    color = "RGB" if c == 3 else "Gray"
    itype = f' imageType="{_esc(image_type)}"' if image_type else ""

    def header(pos):
        return ('<?xml version="1.0" encoding="UTF-8"?>'
                '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f'<Image geometry="{w}:{h}:{c}" sampleFormat="{sample_format}" '
                + (f'bounds="{lo:.8g}:{hi:.8g}" ' if sample_format == "Float32" else '')
                + f'colorSpace="{color}"{itype}{comp_attr} location="attachment:{pos}:{len(body)}">'
                f'{kw_xml}</Image>'
                '<Metadata><Property id="XISF:CreatorApplication" type="String">'
                'SubZeroAstro</Property></Metadata></xisf>').encode("utf-8")

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
