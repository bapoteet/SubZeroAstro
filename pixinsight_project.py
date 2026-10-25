"""Open a PixInsight project (.pxiproject) in SubZeroAstro.

A PixInsight project is a folder (a bundle on macOS) holding project.xosm - an XML list of the open image
windows - and project.data/, where images changed in PixInsight are kept in PixInsight's own compressed raw
format. An image that was not changed is only referred to by the path of its file.

Each image window becomes one image: from project.data when PixInsight stored it there, else from its file
(looked for where PixInsight had it, then next to the project). Masks and other 8-bit helper images are left
out. The images are written as XISF files that SubZeroAstro opens like any master.
"""
from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NS = "{http://www.pixinsight.com/xosm}"
RAW_MAGIC = b"egamiwar"           # 'rawimage', as PixInsight writes it
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass
class PIImage:
    view_id: str
    file_path: str | None = None          # the file PixInsight opened it from
    blob: Path | None = None              # its pixels inside project.data (when changed in PixInsight)
    history: list = field(default_factory=list)   # PixInsight processes applied to it (names)
    note: str = ""

    @property
    def name(self) -> str:
        return self.view_id or (Path(self.file_path).stem if self.file_path else "image")


def find_xosm(path) -> Path | None:
    """project.xosm for a .pxiproject folder, the .xosm itself, or a folder holding one project."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".xosm":
        return p
    if p.is_dir():
        if (p / "project.xosm").is_file():
            return p / "project.xosm"
        found = sorted(p.glob("*.pxiproject/project.xosm"))
        if len(found) == 1:
            return found[0]
    return None


def read_project(path) -> list[PIImage]:
    """The image windows of a PixInsight project."""
    xosm = find_xosm(path)
    if xosm is None:
        raise ValueError("no PixInsight project there (a .pxiproject folder holds a project.xosm file)")
    root = ET.parse(xosm).getroot()
    data_dir = xosm.parent / "project.data"
    out = []
    for win in root.iter(f"{NS}ImageWindow"):
        mv = win.find(f"{NS}MainView")
        if mv is None:
            continue
        img = PIImage(mv.get("id") or "", win.get("filePath"))
        src = mv.find(f"{NS}image")
        if src is not None and src.get("src"):
            name = src.get("src").replace("@project_data_dir/", "")
            if (data_dir / name).is_file():
                img.blob = data_dir / name
        for inst in mv.iter(f"{NS}instance"):
            c = inst.get("class")
            if c:
                img.history.append(c)
        out.append(img)
    return out


def _unshuffle(b: bytes, item: int) -> bytes:
    if item <= 1:
        return b
    n = len(b) // item
    a = np.frombuffer(b[:n * item], np.uint8).reshape(item, n).T.reshape(-1)
    return a.tobytes() + b[n * item:]


def _zstd_frames(data: bytes) -> list[bytes]:
    try:
        import zstandard as zstd
    except ImportError:                                   # pragma: no cover - bundled with the app
        zstd = None
    out, rest = [], data
    while rest[:4] == ZSTD_MAGIC:
        if zstd is not None:
            d = zstd.ZstdDecompressor().decompressobj()
            out.append(d.decompress(rest))
            rest = d.unused_data
        else:
            try:
                from compression import zstd as std_zstd     # Python 3.14+
            except ImportError:
                raise ValueError("PixInsight stores project images with Zstandard compression and the zstandard "
                                 "module is missing")
            d = std_zstd.ZstdDecompressor()
            out.append(d.decompress(rest))
            rest = d.unused_data
    return out


def _roughness(a: np.ndarray) -> float:
    s = a[..., ::7, ::7].astype(np.float64)
    return float(np.mean(np.abs(np.diff(s, axis=-1)))) / (float(np.std(s)) + 1e-12)


def decode_rawimage(blob: bytes) -> np.ndarray:
    """PixInsight's project image format -> (channels, h, w) float32 in 0..1."""
    if blob[:8] != RAW_MAGIC:
        raise ValueError("not a PixInsight project image")
    w, h, nc = struct.unpack_from("<III", blob, 28)
    bits = struct.unpack_from("<I", blob, 108)[0]
    start = blob.find(ZSTD_MAGIC, 32)
    if not (0 < w < 200000 and 0 < h < 200000 and 0 < nc <= 4) or start < 0:
        raise ValueError("unrecognised PixInsight project image")
    frames = _zstd_frames(blob[start:])
    raw_len = sum(len(f) for f in frames)
    item = raw_len // max(w * h * nc, 1)
    if item not in (1, 2, 4, 8) or item * w * h * nc != raw_len:
        raise ValueError(f"unexpected PixInsight image size ({raw_len} bytes for {w} x {h} x {nc})")
    dtype = {1: np.uint8, 2: np.uint16, 4: np.float32, 8: np.float64}[item]
    cands = []
    # byte shuffling ('+sh') per compressed block, or over the whole image - whichever gives a real picture
    for joined in (b"".join(_unshuffle(f, item) for f in frames), _unshuffle(b"".join(frames), item)):
        a = np.frombuffer(joined, dtype).reshape(nc, h, w)
        if item == 4 and (not np.isfinite(a[:, ::97, ::97]).all() or np.abs(a[:, ::97, ::97]).max() > 1e6):
            a = np.frombuffer(joined, np.uint32).reshape(nc, h, w)       # 32-bit integers, not floats
        cands.append(a)
    a = min(cands, key=_roughness) if item > 1 else cands[0]
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return np.clip(a.astype(np.float32), 0, None)


def locate_file(img: PIImage, project_path) -> Path | None:
    """The image's file: where PixInsight had it, else (moved computer / folder) the same name near the project."""
    if not img.file_path:
        return None
    p = Path(img.file_path)
    if p.is_file():
        return p
    name = Path(img.file_path.replace("\\", "/")).name
    proj = Path(project_path)
    for d in [proj, proj.parent, proj.parent.parent] + [c for c in proj.parent.glob("*") if c.is_dir()]:
        q = d / name
        if q.is_file():
            return q
    return None


def is_helper(img: PIImage, data: np.ndarray | None) -> bool:
    """Masks and other helper images (not pictures to process)."""
    if re.search(r"mask|_stf|_lut|psf", img.view_id or "", re.I):
        return True
    return data is not None and data.shape[0] == 1 and len(np.unique(data[:, ::53, ::53])) <= 2


def export_images(path, out_dir, progress=None) -> tuple[list[Path], list[str]]:
    """Every picture of the project as an XISF file in out_dir. Returns (files, notes)."""
    import fits_core as core
    import xisf
    imgs = read_project(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, notes = [], []
    proj = find_xosm(path).parent
    for k, img in enumerate(imgs):
        if progress:
            progress(f"Reading {img.name} ({k + 1}/{len(imgs)})")
        cards, data = [], None
        src = locate_file(img, proj)
        try:
            if img.blob is not None:
                data = decode_rawimage(img.blob.read_bytes())
                if src is not None:                      # keep the original file's header (target, filter, ...)
                    try:
                        cards = [(c[0], c[1], c[2]) for c in _cards_of(src)]
                    except Exception:  # noqa: BLE001
                        cards = []
            elif src is not None:
                d, cards = _read_file(src)
                data = d
            else:
                notes.append(f"{img.name}: its file ({img.file_path}) was not found - copy it next to the project")
                continue
        except ValueError as e:
            notes.append(f"{img.name}: could not be read ({e})")
            continue
        if is_helper(img, data):
            notes.append(f"{img.name}: a mask / helper image - left out")
            continue
        keep = [(k_, v, c) for k_, v, c in cards if k_ not in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                                                                "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END")]
        keep.append(("HISTORY", None, f"Opened from PixInsight project {proj.name}, view {img.view_id}"
                     + (f" ({', '.join(dict.fromkeys(img.history))})" if img.history else "")))
        if not any(k_ == "IMAGETYP" for k_, _v, _c in keep):
            keep.append(("IMAGETYP", "Master Light", ""))
        out = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', img.name)}.xisf"
        xisf.write_xisf(out, data[0] if data.shape[0] == 1 else data, keep)
        files.append(out)
    if not files and not notes:
        notes.append("the project has no image windows")
    return files, notes


def _cards_of(path):
    import stacking as st
    return st.ImageSource(path).cards


def _read_file(path):
    import stacking as st
    src = st.ImageSource(path)
    d = src.read().astype(np.float32)
    if d.ndim == 2:
        d = d[None]
    hi = float(np.percentile(d[:, ::17, ::17], 99.999)) if d.size else 1.0
    if hi > 1.5:                                          # integer data: to 0..1
        d = d / (65535.0 if hi <= 65535 else hi)
    return d, src.cards
