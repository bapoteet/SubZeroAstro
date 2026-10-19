"""
fits_core.py - the non-GUI parts of SubZeroAstro.

Contains:
  * a small, dependency-free FITS reader/writer (NumPy only)
  * the auto-stretch used for thumbnails and the viewer
  * folder scanning and the reject / restore file moves

Nothing in here imports Qt, so it can be tested on its own.
"""

from __future__ import annotations

import os
import sys
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FITS_EXTS = {".fits", ".fit", ".fts"}
LIGHT_EXTS = FITS_EXTS | {".xisf"}       # what page 1 loads (XISF: PixInsight / N.I.N.A.)


def is_hidden_file(name: str) -> bool:
    """Files that are never pictures: macOS '._name' AppleDouble copies (written next to every file
    on a NAS or external drive), '.DS_Store' and other dot-files."""
    return Path(str(name)).name.startswith(".")
REJECTED_DIR = "REJECTED"
DEFAULT_SKIP_PREFIXES = ["FLAT", "DARK", "BIAS", "CALIB", "OFFSET"]

BLOCK = 2880
CARD = 80

_BITPIX_DTYPE = {
    8: ">u1",
    16: ">i2",
    32: ">i4",
    64: ">i8",
    -32: ">f4",
    -64: ">f8",
}


# --------------------------------------------------------------------------
# FITS header parsing
# --------------------------------------------------------------------------

def _convert_value(text: str):
    t = text.strip()
    if t == "":
        return None
    if t == "T":
        return True
    if t == "F":
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t.replace("D", "E").replace("d", "e"))
    except ValueError:
        return t


def parse_card(card: str):
    """Return (keyword, value, comment). Commentary cards have value None."""
    key = card[:8].strip()
    if key == "HIERARCH" and "=" in card:
        k, rest = card[8:].split("=", 1)
        key = k.strip()
        valstr = rest
    elif card[8:10] == "= ":
        valstr = card[10:]
    else:
        return key, None, card[8:].rstrip()

    valstr = valstr.strip()
    if valstr.startswith("'"):
        out = []
        i = 1
        while i < len(valstr):
            c = valstr[i]
            if c == "'":
                if i + 1 < len(valstr) and valstr[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                i += 1
                break
            out.append(c)
            i += 1
        value = "".join(out).rstrip()
        rest = valstr[i:]
        comment = rest.split("/", 1)[1].strip() if "/" in rest else ""
    else:
        if "/" in valstr:
            v, comment = valstr.split("/", 1)
            comment = comment.strip()
        else:
            v, comment = valstr, ""
        value = _convert_value(v)
    return key, value, comment


def _read_header_cards(f):
    """Read one header (a list of parsed cards) from an open binary file."""
    cards = []
    while True:
        block = f.read(BLOCK)
        if len(block) < BLOCK:
            raise ValueError("File ended inside a FITS header (truncated or not a FITS file)")
        text = block.decode("ascii", errors="replace")
        for i in range(0, BLOCK, CARD):
            card = text[i:i + CARD]
            if card[:8].strip() == "END":
                return cards
            if card.strip() == "":
                continue
            cards.append(parse_card(card))


def _data_nbytes(lookup: dict) -> int:
    naxis = int(lookup.get("NAXIS", 0) or 0)
    if naxis == 0:
        return 0
    bitpix = int(lookup["BITPIX"])
    n = 1
    for i in range(1, naxis + 1):
        n *= int(lookup.get(f"NAXIS{i}", 0) or 0)
    pcount = int(lookup.get("PCOUNT", 0) or 0)
    gcount = int(lookup.get("GCOUNT", 1) or 1)
    return abs(bitpix) // 8 * gcount * (pcount + n)


def _padded(n: int) -> int:
    return ((n + BLOCK - 1) // BLOCK) * BLOCK


def cards_to_dict(cards) -> dict:
    d = {}
    for k, v, _c in cards:
        if v is not None and k and k not in d:
            d[k] = v
    return d


class Layout:
    """Where the image lives inside a FITS file, plus its header."""
    __slots__ = ("cards", "look", "offset", "shape", "bitpix", "bscale", "bzero")

    def __init__(self, cards, look, offset, shape, bitpix, bscale, bzero):
        self.cards, self.look, self.offset = cards, look, offset
        self.shape, self.bitpix, self.bscale, self.bzero = shape, bitpix, bscale, bzero

    @property
    def bytes_per_pixel(self):
        return abs(self.bitpix) // 8

    @property
    def row_bytes(self):
        return self.shape[-1] * self.bytes_per_pixel


def read_layout(f) -> Layout:
    """Parse headers from an open binary file and locate the first image."""
    first = f.read(9)
    f.seek(0)
    if not first.startswith(b"SIMPLE"):
        raise ValueError("Not a FITS file (missing SIMPLE keyword)")
    primary = _read_header_cards(f)
    plook = cards_to_dict(primary)
    cards = list(primary)
    look = plook
    nbytes = _data_nbytes(plook)
    if nbytes == 0:
        while True:
            try:
                ext = _read_header_cards(f)
            except ValueError:
                raise ValueError("No image data found in file")
            elook = cards_to_dict(ext)
            enbytes = _data_nbytes(elook)
            xt = str(elook.get("XTENSION", "")).strip().upper()
            if xt == "IMAGE" and enbytes > 0:
                cards += [c for c in ext if c[0] not in plook]
                look = dict(plook)
                look.update(elook)
                break
            if xt == "BINTABLE" and elook.get("ZIMAGE"):
                raise ValueError("Compressed FITS (fpack/.fz) is not supported")
            f.seek(_padded(enbytes), os.SEEK_CUR)
    bitpix = int(look["BITPIX"])
    if bitpix not in _BITPIX_DTYPE:
        raise ValueError(f"Unsupported BITPIX {bitpix}")
    naxis = int(look["NAXIS"])
    shape = tuple(int(look[f"NAXIS{i}"]) for i in range(naxis, 0, -1))
    offset = f.tell()
    _check_not_text_mangled(f, offset, _data_nbytes(look))
    return Layout(cards, look, offset, shape, bitpix,
                  float(look.get("BSCALE", 1.0) or 1.0), float(look.get("BZERO", 0.0) or 0.0))


TEXT_MANGLED = ("This file is damaged: its picture data was converted to text somewhere along the way (a "
                "download or copy that treated it as text), so the image is scrambled - the header still "
                "reads fine, which hides the damage. Download or copy the file again.")


def _check_not_text_mangled(f, offset: int, nbytes: int):
    """Binary data saved as UTF-8 text: every byte above 127 became the 3-byte replacement character
    EF BF BD, so the file grows by more than half and the pixels are noise. Seen with downloaded
    calibrated subframes. Raises ValueError with a plain explanation."""
    try:
        size = os.fstat(f.fileno()).st_size
    except (OSError, AttributeError, ValueError):
        return
    if nbytes <= 0 or size < offset + nbytes * 1.3:
        return
    here = f.tell()
    try:
        f.seek(offset)
        chunk = f.read(65536)
    finally:
        f.seek(here)
    if chunk.count(b"\xef\xbf\xbd") * 3 > 0.25 * max(len(chunk), 1):
        raise ValueError(TEXT_MANGLED)


def _decode(raw: bytes, lay: Layout, shape) -> np.ndarray:
    data = np.frombuffer(raw, dtype=_BITPIX_DTYPE[lay.bitpix]).reshape(shape).astype(np.float32)
    if lay.bscale != 1.0:
        data *= np.float32(lay.bscale)
    if lay.bzero != 0.0:
        data += np.float32(lay.bzero)
    return data


def read_rows(f, lay: Layout, r0: int, n: int) -> np.ndarray:
    """Read n consecutive rows (file order) of a 2-D image from an open file."""
    f.seek(lay.offset + r0 * lay.row_bytes)
    raw = f.read(n * lay.row_bytes)
    if len(raw) < n * lay.row_bytes:
        raise ValueError("File is truncated (image data incomplete)")
    return _decode(raw, lay, (n, lay.shape[-1]))


def is_xisf(path) -> bool:
    return Path(path).suffix.lower() == ".xisf"


def read_xisf_frame(path, header_only: bool = False):
    """An XISF frame the way read_fits returns a FITS one: (data, cards, look). XISF properties the
    FITS keywords lack (camera, filter, exposure, frame type, Bayer pattern, size) are added as cards,
    and floating-point data saved on PixInsight's 0..1 scale is put back in camera units."""
    import xisf
    _r, _img, cards, look = xisf.read_header(path)
    have = {c[0] for c in cards}
    for k in ("NAXIS1", "NAXIS2", "IMAGETYP", "INSTRUME", "FILTER", "EXPTIME", "BAYERPAT"):
        if k not in have and look.get(k) is not None:
            cards.append((k, look[k], "from the XISF properties"))
    if header_only:
        return None, cards, look
    data, _c, _l = xisf.read_xisf(path)
    fmt = str(look.get("_XISF_FORMAT", ""))
    if fmt.startswith("Float"):
        b = str(look.get("_XISF_BOUNDS", "") or "")
        peak = float(np.nanmax(data[..., ::7, ::7]))
        if b.startswith("0:1") or peak <= 1.5:
            data = data * np.float32(65535.0)
    return data, cards, look


def read_fits(path, header_only: bool = False):
    """
    Read the first image in a FITS file.

    Returns (data, cards, lookup):
      data   - float32 array (None if header_only), shape (ny, nx) or (nz, ny, nx)
      cards  - list of (keyword, value, comment) from primary + image extension
      lookup - dict keyword -> value
    The file is fully read and closed before returning (important on Windows,
    where an open file cannot be moved).
    """
    if is_xisf(path):
        return read_xisf_frame(path, header_only)
    with open(Path(path), "rb") as f:
        lay = read_layout(f)
        if header_only:
            return None, lay.cards, lay.look
        count = int(np.prod(lay.shape))
        raw = f.read(count * lay.bytes_per_pixel)
        if len(raw) < count * lay.bytes_per_pixel:
            raise ValueError("File is truncated (image data incomplete)")
    return _decode(raw, lay, lay.shape), lay.cards, lay.look


def _ascii(x) -> str:
    t = str(x).replace("β", "b").replace("″", '"').replace("×", "x").replace("–", "-").replace("—", "-")
    return t.encode("ascii", "replace").decode("ascii")


def write_fits(path, data: np.ndarray, header=None, bitpix: int = -32):
    """
    Write a single-image FITS file. header is a dict {key: value} or a list of
    (key, value, comment) cards; HISTORY / COMMENT cards take their text from
    the value. Strings are made ASCII-safe.
    """
    data = np.asarray(data)
    if bitpix == 16:
        # work in float: uint16 input would wrap around in "data - 32768"
        arr = np.clip(np.round(np.asarray(data, np.float64) - 32768), -32768, 32767).astype(">i2")
        extra = [("BZERO", 32768, "offset for unsigned 16-bit data"), ("BSCALE", 1, "")]
    elif bitpix == -32:
        arr = data.astype(">f4")
        extra = []
    elif bitpix == 8:
        arr = np.clip(data, 0, 255).astype(">u1")
        extra = []
    else:
        raise ValueError("bitpix must be 8, 16 or -32")

    def card(key, value, comment=""):
        if isinstance(value, bool):
            v = f"{'T' if value else 'F':>20}"
        elif isinstance(value, (int, np.integer)):
            v = f"{int(value):>20}"
        elif isinstance(value, (float, np.floating)):
            s = f"{float(value):.10G}"
            if not any(ch in s for ch in ".EN"):
                s += "."
            v = f"{s:>20}"
        else:
            s = _ascii(value).replace("'", "''")[:68]
            v = f"'{s:<8}'"
        line = f"{key[:8]:<8}= {v}"
        if comment:
            line += f" / {_ascii(comment)}"
        return f"{line:<80}"[:80]

    cards = [card("SIMPLE", True, "conforms to FITS standard"),
             card("BITPIX", bitpix), card("NAXIS", arr.ndim)]
    for i, n in enumerate(reversed(arr.shape), start=1):
        cards.append(card(f"NAXIS{i}", n))
    for k, v, c in extra:
        cards.append(card(k, v, c))
    reserved = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BZERO", "BSCALE", "END", "EXTEND"}
    items = header.items() if isinstance(header, dict) else [(k, v) if len(c) == 0 else (k, (v, c))
                                                              for k, v, c in (header or [])]
    for k, v in items:
        comment = ""
        if isinstance(v, tuple):
            v, comment = v
        k = k.upper()
        if k in ("HISTORY", "COMMENT"):
            text = _ascii(v if v is not None else comment)
            for i in range(0, max(len(text), 1), 72):
                cards.append(f"{k:<8}{text[i:i + 72]}".ljust(80)[:80])
            continue
        if k in reserved or v is None:
            continue
        cards.append(card(k, v, comment))
    cards.append(f"{'END':<80}")
    head = "".join(cards)
    head += " " * (_padded(len(head)) - len(head))
    body = arr.tobytes()
    body += b"\0" * (_padded(len(body)) - len(body))
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(body)


# --------------------------------------------------------------------------
# Image preparation and auto-stretch
# --------------------------------------------------------------------------

def is_bayer(look: dict) -> bool:
    """Raw one-shot-colour data. A 3-plane image (e.g. a Seestar stack) is already
    debayered even if it still carries BAYERPAT."""
    try:
        if int(look.get("NAXIS3") or 1) > 1 or int(look.get("_XISF_CHANNELS") or 1) > 1:
            return False
    except (TypeError, ValueError):
        pass
    pat = str(look.get("BAYERPAT", "") or "").strip().upper()
    return pat in {"RGGB", "BGGR", "GRBG", "GBRG"}


def precalibrated(cards, look: dict, name: str = "") -> str | None:
    """Why these light frames look already calibrated (ready-made data from Telescope Live and the
    like, or lights calibrated by other software), or None. A short reason for the user."""
    texts = " ".join(str(c[2] if c[1] is None else c[1]) for c in (cards or [])
                     if c and c[0] in ("HISTORY", "COMMENT")).lower()
    calstat = str(look.get("CALSTAT", "") or "").strip().upper()
    if calstat and any(ch in calstat for ch in "BDF"):
        return f"MaxIm DL: CALSTAT = '{calstat}'"
    proc = str(look.get("PROC", "") or "").strip().lower()
    stem = Path(name).stem.lower() if name else ""
    cal_name = bool(re.search(r"(^|[_\-. ])(cal|calibrated)$", stem) or re.match(r"(pp|cal)_", stem)
                    or re.search(r"_c(_cc)?(_d)?$", stem))          # PixInsight WBPP: name_c, name_c_cc_d
    if proc in ("true", "t", "1", "yes"):
        tl = "processed" in texts or cal_name
        return ("Telescope Live: " if tl else "") + "PROC = True"
    if "imagecalibration" in texts:
        return "calibrated in PixInsight (ImageCalibration)"
    if re.search(r"\bcalibrat(ed|ion)\b|dark[- ]?subtract|flat[- ]?field(ed|ing)?\b", texts):
        return "calibration noted in the header (HISTORY)"
    if cal_name:
        return "the file name marks it as calibrated"
    return None


def smart_scope(look: dict) -> str | None:
    """Model name of a smart telescope (Seestar, DWARF, Vespera, Stellina, eVscope), else None."""
    text = " ".join(str(look.get(k) or "") for k in ("INSTRUME", "TELESCOP", "CREATOR", "ORIGIN", "PRODUCER",
                                                    "CAMERA")).lower()
    comments = " ".join(str(c) for c in (look.get("COMMENT") or [])).lower() \
        if isinstance(look.get("COMMENT"), list) else str(look.get("COMMENT") or "").lower()
    m = re.search(r"seestar\s*(s\d+)(\s*pro)?", text)
    if m:
        return f"Seestar {m.group(1).upper()}" + (" Pro" if m.group(2) else "")
    if "seestar" in text:
        return "Seestar"
    m = re.search(r"dwarf\s*(mini|ii|iii|2|3)?", text)
    if m:
        v = (m.group(1) or "").lower()
        return {"mini": "DWARF mini", "ii": "DWARF II", "2": "DWARF II", "iii": "DWARF 3", "3": "DWARF 3"}.get(v,
                                                                                                         "DWARF")
    if "vespera" in text or "vaonis" in text or "vaonis" in comments:
        if "vesperapro" in text.replace(" ", "") or "vespera pro" in comments:
            return "Vespera Pro"
        if re.search(r"vespera\s*(ii|2)", text):
            return "Vespera II"
        return "Vespera"
    if "stellina" in text:
        return "Stellina"
    if "evscope" in text or "unistellar" in text:
        return "eVscope"
    return None


def to_display_plane(data: np.ndarray, look: dict) -> np.ndarray:
    """
    Turn raw FITS data into a 2-D greyscale plane, oriented for display.
      * colour cubes (3 planes) -> average
      * Bayer (OSC) raw frames -> 2x2 superpixel average (removes the checkerboard)
      * flipped so row 0 is the top, unless ROWORDER says it already is
    """
    if data.ndim == 3:
        data = data.mean(axis=0) if data.shape[0] == 3 else data[0]
    elif data.ndim != 2:
        raise ValueError(f"Unsupported image dimensions: {data.shape}")

    if is_bayer(look):
        h, w = data.shape
        h2, w2 = h // 2 * 2, w // 2 * 2
        data = data[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))

    roworder = str(look.get("ROWORDER", "BOTTOM-UP") or "BOTTOM-UP").strip().upper()
    if roworder != "TOP-DOWN" and "_XISF_FORMAT" not in look:      # XISF: top row first
        data = data[::-1]
    return np.ascontiguousarray(data, dtype=np.float32)


def stretch_params(sample: np.ndarray, target_bg: float = 0.25, shadows_clip: float = -2.8):
    """Screen-transfer-style auto stretch parameters (black point, white point, midtone)."""
    s = sample[np.isfinite(sample)].ravel()
    if s.size == 0:
        return 0.0, 1.0, 0.5
    if s.size > 400_000:
        s = s[:: s.size // 400_000]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    lo = max(float(s.min()), med + shadows_clip * mad)
    hi = float(s.max())
    if hi <= lo:
        hi = lo + 1.0
    n = (med - lo) / (hi - lo)
    n = min(max(n, 1e-6), 0.999)
    # midtone balance m such that MTF(n; m) == target_bg
    m = n * (1 - target_bg) / (n - 2 * n * target_bg + target_bg)
    m = min(max(m, 1e-6), 0.999999)
    return lo, hi, m


def apply_stretch(data: np.ndarray, lo: float, hi: float, m: float) -> np.ndarray:
    """Apply the stretch and return uint8 pixels."""
    x = (data - np.float32(lo)) * np.float32(1.0 / (hi - lo))
    np.clip(x, 0.0, 1.0, out=x)
    m = np.float32(m)
    y = (m - 1) * x / ((2 * m - 1) * x - m)
    np.nan_to_num(y, copy=False)
    return np.ascontiguousarray(np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8))


def make_thumbnail(plane: np.ndarray, long_side: int = 320):
    """Return (thumb_uint8, stretch_params) - params are reused for the full view."""
    h, w = plane.shape
    step = max(1, int(np.ceil(max(h, w) / long_side)))
    small = plane[::step, ::step]
    params = stretch_params(small)
    return apply_stretch(small, *params), params


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------

_FILTER_MAP = {
    "HA": "Ha", "H": "Ha", "HALPHA": "Ha", "HALFA": "Ha",
    "OIII": "OIII", "O3": "OIII",
    "SII": "SII", "S2": "SII",
    "HB": "Hβ", "HBETA": "Hβ",
    "L": "L", "LUM": "L", "LUMINANCE": "L",
    "R": "R", "RED": "R",
    "G": "G", "GREEN": "G",
    "B": "B", "BLUE": "B",
    # dual-band / multi-band filters for colour cameras (names as people type them)
    "LULTIMATE": "L-Ultimate", "LULT": "L-Ultimate", "OPTOLONGLULTIMATE": "L-Ultimate",
    "LEXTREME": "L-eXtreme", "LEXT": "L-eXtreme", "OPTOLONGLEXTREME": "L-eXtreme",
    "LENHANCE": "L-eNhance", "LENH": "L-eNhance", "OPTOLONGLENHANCE": "L-eNhance",
    "LPRO": "L-Pro", "OPTOLONGLPRO": "L-Pro",
    "LSYNERGY": "L-Synergy", "LSYN": "L-Synergy", "OPTOLONGLSYNERGY": "L-Synergy",
    "ALPT": "ALP-T", "ANTLIAALPT": "ALP-T",
    "NBZ": "NBZ", "IDASNBZ": "NBZ", "NBX": "NBX",
    "DUALBAND": "Dual-band", "DUOBAND": "Duo-band",
    # Askar Colour Magic duo-narrowband: D1 = Ha + OIII, D2 = SII + OIII (usually shot as a pair)
    **{k + "D1": "Askar D1" for k in ("AS", "ASKAR", "COLORMAGIC", "COLOURMAGIC", "ASKARCOLORMAGIC",
                                      "ASKARCOLOURMAGIC", "CM", "SUPER", "ASKARSUPER", "")},
    **{k + "D2": "Askar D2" for k in ("AS", "ASKAR", "COLORMAGIC", "COLOURMAGIC", "ASKARCOLORMAGIC",
                                      "ASKARCOLOURMAGIC", "CM", "SUPER", "ASKARSUPER", "")},
}

DUAL_BAND = {"L-Ultimate", "L-eXtreme", "L-eNhance", "ALP-T", "NBZ", "NBX", "Dual-band", "Duo-band", "L-Synergy",
             "Askar D1", "Askar D2"}
SII_OIII_BAND = {"L-Synergy", "Askar D2"}   # pass SII + OIII (the others pass Ha + OIII)


def is_dual_band(filter_name) -> bool:
    return normalize_filter(filter_name) in DUAL_BAND


_NAME_FILTER = re.compile(r"(?i)(?<![a-z])(l[-_ ]?ult(?:imate)?|l[-_ ]?ext(?:reme)?|l[-_ ]?enh(?:ance)?|"
                          r"l[-_ ]?syn(?:ergy)?|alp[-_ ]?t|nbz|nbx|l[-_ ]?pro|as[-_ ]?d[12]|askar[-_ ]?d[12]|"
                          r"colou?r[-_ ]?magic[-_ ]?d[12])(?![a-z0-9])")


def filter_from_name(text) -> str | None:
    """A filter named in a file or folder name ("..._SQA55Lult_0107.fit" -> L-Ultimate), for cameras
    whose software doesn't record the filter."""
    m = _NAME_FILTER.search(str(text))
    if not m:
        return None
    f = normalize_filter(re.sub(r"[-_ ]", "", m.group(1)))
    return f if f in DUAL_BAND or f == "L-Pro" else None


def band_lines(filter_name):
    """The two emission lines a dual-band filter passes: (red-pixel line, 'OIII')."""
    return ("SII", "OIII") if normalize_filter(filter_name) in SII_OIII_BAND else ("Ha", "OIII")


_BRANDS = ("ANTLIA", "ASTRODON", "CHROMA", "BAADER", "ZWO", "OPTOLONG", "ASTRONOMIK", "SVBONY", "PLAYERONE",
           "QHY", "ATIK", "IDAS", "STARIZONA", "MOTHERSHIP", "PRO", "ULTRA", "V", "3NM", "PREMIUM")
_LETTER = {"O": "OIII", "S": "SII"}                # single letters on many filter wheels ("H" is in the map)


def normalize_filter(raw) -> str:
    """One name per filter, however the capture software spells it:
    'Ha 7nm', 'H-alpha', 'Hα', 'Antlia Ha 3nm Pro', 'OIII 6.5nm', 'O', 'S' -> Ha / OIII / SII."""
    if raw is None:
        return "No filter"
    s = str(raw).strip()
    if not s:
        return "No filter"
    up = s.upper().replace("Α", "A").replace("α".upper(), "A")
    key = re.sub(r"[\s_\-.,()]", "", up)
    if key in _FILTER_MAP:
        return _FILTER_MAP[key]
    if key in _LETTER:
        return _LETTER[key]
    # drop bandwidths (7nm, 6.5nm, 3NM) and brand / series words, then try again
    k2 = re.sub(r"\d+(?:[.,]\d+)?NM", "", re.sub(r"[\s_\-,()]", "", up))
    k2 = k2.replace(".", "")
    for b_ in _BRANDS:
        k2 = k2.replace(b_, "") if len(k2) > len(b_) + 1 or k2 != b_ else k2
    if k2 in _FILTER_MAP and _FILTER_MAP[k2] in ("Ha", "OIII", "SII", "Hβ"):
        return _FILTER_MAP[k2]
    if k2 in _LETTER:
        return _LETTER[k2]
    for token, name in (("HALPHA", "Ha"), ("HALFA", "Ha"), ("OIII", "OIII"), ("O3", "OIII"), ("SII", "SII"),
                        ("S2", "SII"), ("HBETA", "Hβ")):
        if token in k2:
            return name
    return _FILTER_MAP.get(k2, s)


def frame_info(look: dict) -> dict:
    """Pick out the header values the app uses."""
    def get(*keys):
        for k in keys:
            v = look.get(k)
            if v not in (None, ""):
                return v
        return None

    focal = get("FOCALLEN", "FOCAL")
    pix = get("XPIXSZ", "PIXSIZE1", "PIXSZ")
    scale = None
    try:
        if focal and pix and float(focal) > 0:
            scale = 206.265 * float(pix) / float(focal)      # arcsec per pixel
    except (TypeError, ValueError):
        pass
    smart = smart_scope(look)
    camera = get("INSTRUME", "CAMERA")
    telescope = get("TELESCOP")
    if smart:                       # serial numbers make every session look like a new rig
        camera = smart
        telescope = smart
    return {
        "smart": smart,
        "stackcnt": get("STACKCNT"),
        "totalexp": get("TOTALEXP"),
        "object": get("OBJECT", "OBJNAME", "TARGET"),
        "filter": normalize_filter(get("FILTER", "FILTER1", "FILTNAME")),
        "exptime": get("EXPTIME", "EXPOSURE"),
        "date_obs": get("DATE-OBS", "DATE-LOC", "DATE"),
        "date_loc": get("DATE-LOC"),
        "longitude": get("SITELONG", "OBSGEO-L", "LONG-OBS"),
        "camera": camera,
        "telescope": telescope,
        "software": get("SWCREATE", "CREATOR", "PROGRAM"),
        "gain": get("GAIN"),
        "offset": get("OFFSET"),
        "temp": get("CCD-TEMP", "SET-TEMP", "DET-TEMP", "TEMP"),
        "bayer": str(look.get("BAYERPAT", "")).strip() if is_bayer(look) else None,
        "imagetyp": get("IMAGETYP", "FRAME"),
        "scale": scale,
        "rotator": get("ROTATOR", "ROTATANG", "OBJCTROT"),
    }


def _parse_time(text):
    from datetime import datetime
    if not text:
        return None
    t = str(text).strip().replace("Z", "")
    if "." in t:                       # trim to microseconds
        head, frac = t.split(".", 1)
        t = head + "." + frac[:6]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    return None


def night_of(info: dict) -> str | None:
    """
    The calendar date of the evening a frame belongs to ("night of"), so that
    frames taken after midnight count with the evening they started.
    Uses local time if recorded, else UTC shifted by the site longitude,
    else UTC minus 12 hours (right for the Americas and Europe).
    """
    from datetime import timedelta
    loc = _parse_time(info.get("date_loc"))
    if loc is not None:
        return (loc - timedelta(hours=12)).date().isoformat()
    utc = _parse_time(info.get("date_obs"))
    if utc is None:
        return None
    lon = info.get("longitude")
    try:
        if lon is not None:
            return (utc + timedelta(hours=float(lon) / 15.0 - 12)).date().isoformat()
    except (TypeError, ValueError):
        pass
    return (utc - timedelta(hours=12)).date().isoformat()


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


# --------------------------------------------------------------------------
# Frames, folder scanning, reject / restore
# --------------------------------------------------------------------------

@dataclass
class Frame:
    uid: int
    path: Path
    folder: Path            # the folder the frame belongs in (parent of REJECTED/)
    root: Path              # folder the user opened
    rejected: bool = False
    size: int = 0
    mtime_ns: int = 0
    loaded: bool = False
    error: str | None = None
    thumb: object = None                # QImage in the GUI
    thumb_arr: np.ndarray | None = None
    stretch: tuple | None = None
    display_size: tuple | None = None   # (w, h) of the display plane
    cards: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    flagged: bool = False
    flag_note: str = ""                 # why an auto-cull preset flagged it
    filter_override: str | None = None  # filter name set by the user (camera didn't record it)
    badge: str = "none"                 # green / amber / red / grey / none
    reasons: list = field(default_factory=list)
    score: float | None = None
    night: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def rel_folder(self) -> str:
        try:
            r = self.folder.relative_to(self.root)
            return "" if str(r) == "." else str(r)
        except ValueError:
            return str(self.folder)

    @property
    def group(self) -> str:
        """Frames are compared within a camera + filter group."""
        return f"{self.info.get('camera') or '?'}|{self.info.get('filter') or 'No filter'}"

    @property
    def cache_key(self) -> str:
        return f"{self.path.name}|{self.size}|{self.mtime_ns}"


def _fits_entries(d: Path):
    """(path, size, mtime_ns) for FITS files in d. os.scandir gets size and time
    from the directory listing itself, which is much faster on network drives."""
    out = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if os.path.splitext(e.name)[1].lower() in LIGHT_EXTS and not is_hidden_file(e.name) \
                        and e.is_file():
                    st = e.stat()
                    out.append((Path(e.path), st.st_size, st.st_mtime_ns))
    except OSError:
        pass
    out.sort(key=lambda t: natural_key(t[0].name))
    return out


def scan_folder(root, recursive: bool = False, skip_prefixes=None):
    """
    Find FITS files. Files already in a REJECTED/ subfolder are returned as
    rejected frames so earlier rejections can still be undone.
    Returns a list of Frame objects.
    """
    root = Path(root)
    skip = [s.upper() for s in (DEFAULT_SKIP_PREFIXES if skip_prefixes is None else skip_prefixes)]
    frames: list[Frame] = []

    def add_dir(d: Path):
        for p, size, mt in _fits_entries(d):
            frames.append(Frame(len(frames), p, d, root, size=size, mtime_ns=mt))
        rej = d / REJECTED_DIR
        if rej.is_dir():
            for p, size, mt in _fits_entries(rej):
                frames.append(Frame(len(frames), p, d, root, rejected=True, size=size, mtime_ns=mt))

    if recursive:
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = sorted(
                (n for n in dirnames
                 if n.upper() != REJECTED_DIR and not n.startswith(".")
                 and not any(n.upper().startswith(s) for s in skip)),
                key=natural_key)
            add_dir(Path(dirpath))
    else:
        add_dir(root)
    return frames


def frames_from_files(files):
    """Frames for individually chosen/dropped files."""
    frames = []
    for f in sorted((Path(x) for x in files), key=lambda p: natural_key(str(p))):
        if f.suffix.lower() not in LIGHT_EXTS or is_hidden_file(f.name):
            continue
        try:
            st = f.stat()
            size, mt = st.st_size, st.st_mtime_ns
        except OSError:
            size, mt = 0, 0
        if f.parent.name.upper() == REJECTED_DIR:
            frames.append(Frame(len(frames), f, f.parent.parent, f.parent.parent, rejected=True,
                                size=size, mtime_ns=mt))
        else:
            frames.append(Frame(len(frames), f, f.parent, f.parent, size=size, mtime_ns=mt))
    return frames


def _date_obs(path) -> str | None:
    """DATE-OBS from the first header block (cheap: one small read)."""
    if is_xisf(path):
        try:
            import xisf
            v = xisf.read_header(path)[3].get("DATE-OBS")
            return str(v) if v else None
        except Exception:  # noqa: BLE001
            return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(2880 * 3)
        for i in range(0, len(head) - 79, 80):
            card = head[i:i + 80]
            if card.startswith(b"DATE-OBS="):
                return card[10:80].split(b"/")[0].strip().strip(b"'").strip().decode("ascii", "replace")
            if card.startswith(b"END "):
                break
    except OSError:
        pass
    return None


def scan_sources(sources, duplicates: list | None = None):
    """
    Combine several folders and/or file lists into one session.
    sources: list of ("folder", path, recursive) or ("files", [paths]).
    Returns (frames, root): frames with unique uids (duplicates dropped) and a
    root folder used to show short folder names in the sidebar.
    Duplicates - the same file added twice, or the same frame copied into two folders
    (same name, size and capture time) - are skipped and listed in `duplicates`.
    """
    frames: list[Frame] = []
    seen = set()
    by_name = {}                       # (name, size) -> frame, to spot copies in other folders
    folders = []
    for src in sources:
        if src[0] == "folder":
            found = scan_folder(src[1], recursive=src[2])
            folders.append(Path(src[1]))
        else:
            found = frames_from_files(src[1])
            folders += sorted({f.folder for f in found})
        for f in found:
            key = os.path.normcase(os.path.abspath(str(f.path)))
            if key in seen:
                if duplicates is not None:
                    duplicates.append(f.path)
                continue
            twin = by_name.get((f.path.name.lower(), f.size))
            if twin is not None and f.size and _date_obs(f.path) == _date_obs(twin.path) is not None:
                if duplicates is not None:
                    duplicates.append(f.path)
                continue
            seen.add(key)
            by_name.setdefault((f.path.name.lower(), f.size), f)
            f.uid = len(frames)
            frames.append(f)
    root = common_root(folders)
    for f in frames:
        f.root = root
    return frames, root


def common_root(folders) -> Path:
    """Deepest folder containing all of them (or the first one if they are on
    different drives)."""
    folders = [Path(f) for f in folders]
    if not folders:
        return Path.home()
    try:
        return Path(os.path.commonpath([str(f) for f in folders]))
    except ValueError:
        return folders[0]


def _unique(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def set_rejected(frame: Frame, reject: bool):
    """Move a frame into or out of its REJECTED/ subfolder. Raises OSError on failure."""
    if frame.rejected == reject:
        return
    if reject:
        dest_dir = frame.folder / REJECTED_DIR
        dest_dir.mkdir(exist_ok=True)
    else:
        dest_dir = frame.folder
    old_parent = frame.path.parent
    dest = _unique(dest_dir / frame.path.name)
    shutil.move(str(frame.path), str(dest))
    frame.path = dest
    frame.rejected = reject
    if not reject and old_parent.name.upper() == REJECTED_DIR:
        try:
            old_parent.rmdir()          # only succeeds if now empty
        except OSError:
            pass


# --------------------------------------------------------------------------
# One-pass analysis: thumbnail + star metrics
# --------------------------------------------------------------------------

THUMB_LONG = 320
# Fast mode measures stars in three horizontal bands (top, middle, bottom),
# each 1/12 of the height: about a quarter of the file is read.
BANDS = (0.2, 0.5, 0.8)
BAND_FRACTION = 1 / 12


def _saturation_level(lay: Layout) -> float | None:
    dm = lay.look.get("DATAMAX")
    if isinstance(dm, (int, float)) and dm > 0:
        return float(dm)
    if lay.bitpix == 16:
        return 65535.0 if lay.bzero == 32768 else 32767.0
    if lay.bitpix == 8:
        return 255.0
    return None


def _orient(plane: np.ndarray, look: dict) -> np.ndarray:
    roworder = str(look.get("ROWORDER", "BOTTOM-UP") or "BOTTOM-UP").strip().upper()
    return plane if (roworder == "TOP-DOWN" or "_XISF_FORMAT" in look) else plane[::-1]


def analyze_frame(path, mode: str = "fast") -> dict:
    """
    Read a frame once and return everything the sidebar, inspector and chart need.
    mode: "fast" (sampled read, best for network drives), "full" (whole frame),
          or "off" (thumbnail only, no star metrics).
    """
    import metrics as M
    path = Path(path)
    if is_xisf(path):
        return _analyze_xisf(path, mode)
    with open(path, "rb") as f:
        lay = read_layout(f)
        look = lay.look
        bayer = is_bayer(look)
        ny, nx = lay.shape[-2], lay.shape[-1]
        sat = _saturation_level(lay)
        result = {"cards": lay.cards, "info": frame_info(look), "mode": mode}

        if len(lay.shape) != 2 or mode == "full":
            count = int(np.prod(lay.shape))
            raw = f.read(count * lay.bytes_per_pixel)
            if len(raw) < count * lay.bytes_per_pixel:
                raise ValueError("File is truncated (image data incomplete)")
            data = _decode(raw, lay, lay.shape)
            plane = to_display_plane(data, look)
            thumb, params = make_thumbnail(plane)
            result.update(thumb=thumb, stretch=params, display_size=(plane.shape[1], plane.shape[0]))
            if mode != "off":
                img = data if data.ndim == 2 else (data.mean(axis=0) if data.shape[0] == 3 else data[0])
                result["metrics"] = M.measure(img, bayer=bayer and data.ndim == 2, saturation=sat)
            return result

        # --- sampled read ---------------------------------------------------
        k = 2 if bayer else 1
        dny, dnx = ny // k, nx // k                       # display-plane size
        step = max(1, int(np.ceil(max(dny, dnx) / THUMB_LONG)))
        rows = []
        for r in range(0, dny, step):
            block = read_rows(f, lay, r * k, k)          # 1 row (mono) or a Bayer row pair
            if bayer:
                w2 = (nx // 2) * 2
                block = block[:, :w2].reshape(2, nx // 2, 2).mean(axis=(0, 2))
            else:
                block = block[0]
            rows.append(block[::step])
        small = _orient(np.asarray(rows, dtype=np.float32), look)
        params = stretch_params(small)
        result.update(thumb=apply_stretch(small, *params), stretch=params, display_size=(dnx, dny))

        if mode == "off":
            return result

        band_h = max(64, int(ny * BAND_FRACTION)) // 2 * 2
        parts = []
        for c in BANDS:
            r0 = int(ny * c - band_h / 2) // 2 * 2
            r0 = min(max(r0, 0), max(ny - band_h, 0))
            parts.append(M.measure(read_rows(f, lay, r0, min(band_h, ny)), bayer=bayer,
                                   saturation=sat, area_fraction=1.0))
    result["metrics"] = _combine_bands(parts, (band_h * len(BANDS)) / ny)
    return result


def _analyze_xisf(path, mode: str) -> dict:
    """analyze_frame for an XISF file: read once into memory, then measured exactly like a FITS
    frame (the same thumbnail and the same three bands in 'fast' mode)."""
    import metrics as M
    data, cards, look = read_xisf_frame(path)
    bayer = is_bayer(look)
    fmt = str(look.get("_XISF_FORMAT", ""))
    sat = {"UInt8": 255.0, "UInt16": 65535.0}.get(fmt)
    dm = look.get("DATAMAX")
    if isinstance(dm, (int, float)) and dm > 0:
        sat = float(dm)
    if sat is None and fmt.startswith("Float"):
        sat = 65535.0
    result = {"cards": cards, "info": frame_info(look), "mode": mode}
    plane = to_display_plane(data, look)
    thumb, params = make_thumbnail(plane)
    result.update(thumb=thumb, stretch=params, display_size=(plane.shape[1], plane.shape[0]))
    if mode == "off":
        return result
    if data.ndim != 2 or mode == "full":
        img = data if data.ndim == 2 else (data.mean(axis=0) if data.shape[0] == 3 else data[0])
        result["metrics"] = M.measure(img, bayer=bayer and data.ndim == 2, saturation=sat)
        return result
    ny = data.shape[0]
    band_h = max(64, int(ny * BAND_FRACTION)) // 2 * 2
    parts = []
    for c in BANDS:
        r0 = int(ny * c - band_h / 2) // 2 * 2
        r0 = min(max(r0, 0), max(ny - band_h, 0))
        parts.append(M.measure(data[r0:r0 + min(band_h, ny)], bayer=bayer, saturation=sat, area_fraction=1.0))
    result["metrics"] = _combine_bands(parts, (band_h * len(BANDS)) / ny)
    return result


def _combine_bands(parts, fraction):
    """Merge per-band measurements into one set of frame metrics."""
    def wmed(key):
        vals = [(p[key], p["fitted"]) for p in parts if p.get(key) is not None and p["fitted"] > 0]
        if not vals:
            return None
        v = np.array([a for a, _ in vals])
        w = np.array([b for _, b in vals], dtype=float)
        return float(np.sum(v * w) / np.sum(w))

    snrs = [(p["snr"], p.get("snr_n", 0)) for p in parts if p.get("snr") is not None]
    snr = (float(sum(a * b for a, b in snrs) / max(sum(b for _, b in snrs), 1)) if snrs else None)
    return {
        "fwhm": wmed("fwhm"),
        "eccentricity": wmed("eccentricity"),
        "snr": snr,
        "stars": int(round(sum(p["stars"] for p in parts) / max(fraction, 1e-3))),
        "background": float(np.median([p["background"] for p in parts])),
        "noise": float(np.median([p["noise"] for p in parts])),
        "fitted": int(sum(p["fitted"] for p in parts)),
    }


def load_frame_display(path, params=None, max_pixels: float | None = 25e6):
    """
    Read a frame and return (uint8 image, factor). Large mono frames are reduced
    2x2 (factor 2) unless max_pixels is None; the viewer loads full resolution
    only when you zoom in.
    """
    data, _cards, look = read_fits(path)
    plane = to_display_plane(data, look)
    factor = 1
    if max_pixels is not None and plane.size > max_pixels:
        h, w = plane.shape[0] // 2 * 2, plane.shape[1] // 2 * 2
        plane = plane[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        factor = 2
    if params is None:
        h, w = plane.shape
        step = max(1, int(np.ceil(max(h, w) / 1000)))
        params = stretch_params(plane[::step, ::step])
    return apply_stretch(plane, *params), factor


# --------------------------------------------------------------------------
# Quality: per-group medians, badges and score
# --------------------------------------------------------------------------

METRIC_NAMES = {
    "score": "Quality score",
    "fwhm": "FWHM (px)",
    "eccentricity": "Eccentricity",
    "snr": "SNR",
    "stars": "Star count",
    "background": "Background (ADU)",
}
# True = bigger is better
METRIC_HIGHER_BETTER = {"score": True, "fwhm": False, "eccentricity": False, "snr": True,
                        "stars": True, "background": False}


def frame_value(f: Frame, key: str):
    if key == "score":
        return f.score
    return f.metrics.get(key) if f.metrics else None


def group_medians(frames) -> dict:
    """{group: {metric: median}} over frames that are not rejected."""
    groups: dict[str, dict[str, list]] = {}
    for f in frames:
        if not f.metrics or f.rejected:
            continue
        g = groups.setdefault(f.group, {k: [] for k in ("fwhm", "eccentricity", "snr", "stars", "background")})
        for k in g:
            v = f.metrics.get(k)
            if v is not None:
                g[k].append(v)
    return {g: {k: (float(np.median(v)) if v else None) for k, v in d.items()} for g, d in groups.items()}


def assess(frames, med: dict | None = None):
    """Set badge, reasons and score on every frame, relative to its group median."""
    med = group_medians(frames) if med is None else med
    for f in frames:
        f.reasons = []
        m = f.metrics
        gm = med.get(f.group)
        if not m:
            f.badge, f.score = ("grey" if f.loaded and not f.error else "none"), None
            continue
        if m.get("fwhm") is None or gm is None or gm.get("fwhm") is None:
            f.badge, f.score = "grey", None
            f.reasons = ["No measurable stars"]
            continue
        rf = m["fwhm"] / gm["fwhm"]
        rs = (m["stars"] / gm["stars"]) if gm.get("stars") else 1.0
        rn = (m["snr"] / gm["snr"]) if gm.get("snr") and m.get("snr") else 1.0
        e = m.get("eccentricity") or 0.0
        em = gm.get("eccentricity") or 0.0
        rb = (m["background"] / gm["background"]) if gm.get("background") else 1.0

        red = amber = False
        if e > 0.8 or e > max(0.6, em + 0.15):
            red = True; f.reasons.append(f"Elongated / trailed stars (e = {e:.2f})")
        elif e > max(0.5, em + 0.08):
            amber = True; f.reasons.append(f"Slightly elongated stars (e = {e:.2f})")
        if rf > 1.30:
            red = True; f.reasons.append(f"Soft focus / poor seeing (FWHM {100*(rf-1):.0f}% above median)")
        elif rf > 1.15:
            amber = True; f.reasons.append(f"FWHM {100*(rf-1):.0f}% above median")
        if rs < 0.5:
            red = True; f.reasons.append(f"Few stars ({100*rs:.0f}% of median) - clouds?")
        elif rs < 0.75:
            amber = True; f.reasons.append(f"Fewer stars ({100*rs:.0f}% of median)")
        if rn < 0.5:
            red = True; f.reasons.append(f"Low SNR ({100*rn:.0f}% of median) - clouds, haze or dew?")
        elif rn < 0.75:
            amber = True; f.reasons.append(f"Lower SNR ({100*rn:.0f}% of median)")
        if rb > 1.5:
            amber = True; f.reasons.append(f"Bright background ({100*(rb-1):.0f}% above median)")
        f.badge = "red" if red else ("amber" if amber else "green")

        # 100 = a typical frame for its group; higher is better
        score = 100.0 * (1.0 / rf) ** 1.5 * min(rs, 1.25) ** 0.5 * min(rn, 1.5) ** 0.3
        score *= max(0.0, 1.0 - max(0.0, e - max(em, 0.3)) * 1.5)
        f.score = float(score)
    return med


# --------------------------------------------------------------------------
# Auto-cull presets
# --------------------------------------------------------------------------

# Limits relative to each camera + filter median.
#   fwhm:  flag if FWHM is more than this many times the median
#   stars: flag if the star count is below this fraction of the median
#   ecc:   flag if eccentricity is above max(abs, median + rel)
#   snr:   flag if SNR is below this fraction of the median (thin cloud or haze
#          can dim every star without making any disappear)
#   combo: also flag when the "how close to the limit" scores add up to this
#          much, i.e. a frame that is somewhat worse in two ways at once
CULL_PRESETS = {
    "aggressive": {"label": "Aggressive", "fwhm": 1.10, "stars": 0.80, "snr": 0.75,
                   "ecc_abs": 0.55, "ecc_rel": 0.08, "combo": 1.2},
    "semi": {"label": "Semi-aggressive", "fwhm": 1.20, "stars": 0.65, "snr": 0.60,
             "ecc_abs": 0.65, "ecc_rel": 0.12, "combo": 1.5},
    "less": {"label": "Less aggressive", "fwhm": 1.35, "stars": 0.50, "snr": 0.45,
             "ecc_abs": 0.75, "ecc_rel": 0.20, "combo": 1.8},
}


def cull_reasons(f: Frame, gm: dict, p: dict) -> list[str]:
    """Why a frame would be culled under preset p (empty list = keep)."""
    m = f.metrics
    if not m or not gm or m.get("fwhm") is None or not gm.get("fwhm"):
        return []
    out = []
    rf = m["fwhm"] / gm["fwhm"]
    rs = (m.get("stars", 0) / gm["stars"]) if gm.get("stars") else 1.0
    rn = (m["snr"] / gm["snr"]) if m.get("snr") is not None and gm.get("snr") else 1.0
    e = m.get("eccentricity") or 0.0
    em = gm.get("eccentricity") or 0.0
    e_lim = max(p["ecc_abs"], em + p["ecc_rel"])

    # 0 = at the median, 1 = at the limit
    s_f = max(0.0, (rf - 1) / (p["fwhm"] - 1))
    s_s = max(0.0, (1 - rs) / (1 - p["stars"]))
    s_e = max(0.0, (e - em) / max(e_lim - em, 1e-6)) if e > em else 0.0
    s_n = max(0.0, (1 - rn) / (1 - p["snr"]))
    if s_f >= 1:
        out.append(f"FWHM {100 * (rf - 1):.0f}% above median")
    if s_s >= 1:
        out.append(f"star count {100 * rs:.0f}% of median")
    if e > e_lim:
        out.append(f"eccentricity {e:.2f} (limit {e_lim:.2f})")
    if s_n >= 1:
        out.append(f"SNR {100 * rn:.0f}% of median (cloud or haze?)")
    if not out and s_f + s_s + s_e + s_n >= p["combo"]:
        out.append("worse than usual in several ways at once")
    return out


def cull_preset(frames, medians: dict, level: str) -> list[int]:
    """uids of not-yet-rejected frames that preset `level` would flag."""
    p = CULL_PRESETS[level]
    return [f.uid for f in frames
            if not f.rejected and cull_reasons(f, medians.get(f.group, {}), p)]


# --------------------------------------------------------------------------
# Local cache of results (so reopening a session is fast)
# --------------------------------------------------------------------------

CACHE_VERSION = 2


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    # storage keeps its original name so saved measurements survive the rename to SubZeroAstro
    if base:
        d = Path(base) / "FITSBlasterWin" / "cache"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Caches" / "SubZeroAstro"
    else:
        d = Path.home() / ".cache" / "fitsblasterwin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_file(folder: Path) -> Path:
    import hashlib
    key = os.path.normcase(os.path.abspath(str(folder)))
    return cache_dir() / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".cache")


def load_cache(folder) -> dict:
    """{cache_key: entry} for one folder; empty dict if missing or unreadable."""
    import pickle
    import zlib
    p = _cache_file(Path(folder))
    try:
        obj = pickle.loads(zlib.decompress(p.read_bytes()))
        if obj.get("version") == CACHE_VERSION:
            return obj["entries"]
    except Exception:  # noqa: BLE001
        pass
    return {}


def save_cache(folder, entries: dict):
    import pickle
    import zlib
    p = _cache_file(Path(folder))
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(zlib.compress(pickle.dumps({"version": CACHE_VERSION, "folder": str(folder),
                                                "entries": entries}, protocol=4), 1))
    os.replace(tmp, p)


def cache_entry(result: dict) -> dict:
    return {k: result.get(k) for k in ("thumb", "stretch", "display_size", "cards", "info", "metrics", "mode")}


def is_network_path(path) -> bool:
    """True for UNC paths (\\\\server\\share) and mapped network drives on Windows."""
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    if os.name == "nt" and len(p) >= 2 and p[1] == ":":
        try:
            import ctypes
            return ctypes.windll.kernel32.GetDriveTypeW(p[:2] + "\\") == 4   # DRIVE_REMOTE
        except Exception:  # noqa: BLE001
            return False
    return False
