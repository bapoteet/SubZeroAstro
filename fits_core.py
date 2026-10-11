"""
fits_core.py - the non-GUI parts of FITS Blaster for Windows.

Contains:
  * a small, dependency-free FITS reader/writer (NumPy only)
  * the auto-stretch used for thumbnails and the viewer
  * folder scanning and the reject / restore file moves

Nothing in here imports Qt, so it can be tested on its own.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FITS_EXTS = {".fits", ".fit", ".fts"}
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
    path = Path(path)
    with open(path, "rb") as f:
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
            # Image is probably in an extension
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
                    nbytes = enbytes
                    break
                if xt == "BINTABLE" and elook.get("ZIMAGE"):
                    raise ValueError("Compressed FITS (fpack/.fz) is not supported")
                f.seek(_padded(enbytes), os.SEEK_CUR)

        if header_only:
            return None, cards, look

        bitpix = int(look["BITPIX"])
        if bitpix not in _BITPIX_DTYPE:
            raise ValueError(f"Unsupported BITPIX {bitpix}")
        naxis = int(look["NAXIS"])
        shape = tuple(int(look[f"NAXIS{i}"]) for i in range(naxis, 0, -1))
        count = int(np.prod(shape))
        raw = f.read(count * abs(bitpix) // 8)
        if len(raw) < count * abs(bitpix) // 8:
            raise ValueError("File is truncated (image data incomplete)")

    data = np.frombuffer(raw, dtype=_BITPIX_DTYPE[bitpix]).reshape(shape).astype(np.float32)
    bscale = float(look.get("BSCALE", 1.0) or 1.0)
    bzero = float(look.get("BZERO", 0.0) or 0.0)
    if bscale != 1.0:
        data *= np.float32(bscale)
    if bzero != 0.0:
        data += np.float32(bzero)
    return data, cards, look


def write_fits(path, data: np.ndarray, header: dict | None = None, bitpix: int = -32):
    """Write a simple single-image FITS file (used by tests and, later, stacking)."""
    data = np.asarray(data)
    if bitpix == 16:
        arr = np.clip(np.round(data - 32768), -32768, 32767).astype(">i2")
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
            s = str(value).replace("'", "''")
            v = f"'{s:<8}'"
        line = f"{key[:8]:<8}= {v}"
        if comment:
            line += f" / {comment}"
        return f"{line:<80}"[:80]

    cards = [card("SIMPLE", True, "conforms to FITS standard"),
             card("BITPIX", bitpix), card("NAXIS", arr.ndim)]
    for i, n in enumerate(reversed(arr.shape), start=1):
        cards.append(card(f"NAXIS{i}", n))
    for k, v, c in extra:
        cards.append(card(k, v, c))
    reserved = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BZERO", "BSCALE", "END", "EXTEND"}
    for k, v in (header or {}).items():
        if k.upper() in reserved or v is None:
            continue
        cards.append(card(k.upper(), v))
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
    pat = str(look.get("BAYERPAT", "") or "").strip().upper()
    return pat in {"RGGB", "BGGR", "GRBG", "GBRG"}


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
    if roworder != "TOP-DOWN":
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
}


def normalize_filter(raw) -> str:
    if raw is None:
        return "No filter"
    s = str(raw).strip()
    if not s:
        return "No filter"
    key = re.sub(r"[\s_\-]", "", s.upper())
    return _FILTER_MAP.get(key, s)


def frame_info(look: dict) -> dict:
    """Pick out the header values the app shows."""
    def get(*keys):
        for k in keys:
            v = look.get(k)
            if v not in (None, ""):
                return v
        return None

    return {
        "object": get("OBJECT", "OBJNAME", "TARGET"),
        "filter": normalize_filter(get("FILTER", "FILTER1", "FILTNAME")),
        "exptime": get("EXPTIME", "EXPOSURE"),
        "date_obs": get("DATE-OBS", "DATE-LOC", "DATE"),
        "camera": get("INSTRUME", "CAMERA"),
        "telescope": get("TELESCOP"),
        "gain": get("GAIN"),
        "temp": get("CCD-TEMP", "SET-TEMP"),
        "bayer": str(look.get("BAYERPAT", "")).strip() if is_bayer(look) else None,
        "imagetyp": get("IMAGETYP", "FRAME"),
    }


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
    loaded: bool = False
    error: str | None = None
    thumb: np.ndarray | None = None
    stretch: tuple | None = None
    display_size: tuple | None = None   # (w, h) of the display plane
    cards: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

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


def _is_fits(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in FITS_EXTS


def scan_folder(root, recursive: bool = False, skip_prefixes=None):
    """
    Find FITS files. Files already in a REJECTED/ subfolder are returned as
    rejected frames so earlier rejections can still be undone.
    Returns a list of Frame objects.
    """
    root = Path(root)
    skip = [s.upper() for s in (DEFAULT_SKIP_PREFIXES if skip_prefixes is None else skip_prefixes)]
    frames: list[Frame] = []
    uid = 0

    def add_dir(d: Path):
        nonlocal uid
        try:
            entries = sorted(d.iterdir(), key=lambda p: natural_key(p.name))
        except OSError:
            return
        for p in entries:
            if _is_fits(p):
                frames.append(Frame(uid, p, d, root))
                uid += 1
        rej = d / REJECTED_DIR
        if rej.is_dir():
            for p in sorted(rej.iterdir(), key=lambda p: natural_key(p.name)):
                if _is_fits(p):
                    frames.append(Frame(uid, p, d, root, rejected=True))
                    uid += 1

    if recursive:
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = sorted(
                (n for n in dirnames
                 if n.upper() != REJECTED_DIR and not any(n.upper().startswith(s) for s in skip)),
                key=natural_key)
            add_dir(Path(dirpath))
    else:
        add_dir(root)
    return frames


def frames_from_files(files):
    """Frames for individually chosen/dropped files."""
    frames = []
    for i, f in enumerate(sorted((Path(x) for x in files), key=lambda p: natural_key(str(p)))):
        if f.suffix.lower() not in FITS_EXTS:
            continue
        if f.parent.name.upper() == REJECTED_DIR:
            frames.append(Frame(i, f, f.parent.parent, f.parent.parent, rejected=True))
        else:
            frames.append(Frame(i, f, f.parent, f.parent))
    return frames


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


def load_frame_preview(path):
    """Worker helper: read a file and build everything the sidebar needs."""
    data, cards, look = read_fits(path)
    plane = to_display_plane(data, look)
    thumb, params = make_thumbnail(plane)
    h, w = plane.shape
    return {
        "thumb": thumb,
        "stretch": params,
        "display_size": (w, h),
        "cards": cards,
        "info": frame_info(look),
    }


def load_frame_display(path, params=None):
    """Worker helper: read a file and return the stretched full-size uint8 image."""
    data, _cards, look = read_fits(path)
    plane = to_display_plane(data, look)
    if params is None:
        h, w = plane.shape
        step = max(1, int(np.ceil(max(h, w) / 1000)))
        params = stretch_params(plane[::step, ::step])
    return apply_stretch(plane, *params)
