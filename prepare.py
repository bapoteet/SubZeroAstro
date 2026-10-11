"""
prepare.py - Stage 4: get master lights ready for PixInsight.

Linear pipeline (each step optional):
    crop -> plate solve (ASTAP) -> gradient removal (GraXpert or built-in)
    -> BlurXTerminator -> NoiseXTerminator -> StarXTerminator -> save
plus a quick-look colour image (auto-stretch, RGB / LRGB / SHO / HOO …) and a
step-by-step PixInsight plan tailored to what was done.

Pixel data is kept in camera units (ADU) internally and written in
PixInsight's 0..1 convention (see stacking.save_master_file).
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import fits_core as core
import stacking as st
import tools

STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END",
              "PIXRANGE", "_XISF_CHANNELS", "_XISF_BOUNDS", "_XISF_FORMAT"}
WCS_PREFIXES = ("CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_", "CDELT", "CROTA", "CUNIT", "PC1_", "PC2_", "A_", "B_",
                "AP_", "BP_", "PLTSOLVD", "LONPOLE", "LATPOLE")


class Cancelled(Exception):
    pass


# ==========================================================================
# Masters
# ==========================================================================

@dataclass
class Master:
    path: Path
    data: np.ndarray                      # (channels, h, w) float32, ADU
    cards: list                           # (key, value, comment) to carry forward
    look: dict
    target: str
    filter: str
    alignref: str | None
    autocrop: tuple | None
    steps: list = field(default_factory=list)      # what was done, for the log and the plan
    outputs: list = field(default_factory=list)
    solved: bool = False

    @property
    def is_color(self):
        return self.data.shape[0] == 3

    @property
    def name(self):
        return self.path.stem

    @property
    def shape(self):
        return self.data.shape[1:]

    def set_card(self, key, value, comment=""):
        self.cards = [c for c in self.cards if c[0] != key] + [(key, value, comment)]

    def get(self, key, default=None):
        for k, v, _c in self.cards:
            if k == key:
                return v
        return default


def is_master_light(look: dict, name: str) -> bool:
    t = str(look.get("IMAGETYP", "")).lower()
    return "master" in t and "light" in t or ("_master" in name.lower() and "flat" not in t and "dark" not in t
                                              and "bias" not in t)


def find_masters(paths) -> list:
    """Master light files in the given files / folders (one per base name; FITS preferred)."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in p.iterdir() if f.suffix.lower() in st.IMAGE_EXTS]
        elif p.suffix.lower() in st.IMAGE_EXTS:
            files.append(p)
    by_base = {}
    for f in sorted(files, key=lambda x: core.natural_key(str(x))):
        b = f.with_suffix("")
        if b not in by_base or f.suffix.lower() != ".xisf":
            by_base[b] = f
    out = []
    for f in by_base.values():
        try:
            src = st.ImageSource(f)
        except Exception:  # noqa: BLE001
            continue
        if is_master_light(src.look, f.name) and not f.stem.endswith("_prepared"):
            out.append(f)
    return out


def load_master(path) -> Master:
    src = st.ImageSource(path)
    d = src.read().astype(np.float32)
    if d.ndim == 2:
        d = d[None]
    cards = [(k, v, c) for k, v, c in src.cards if k not in STRUCTURAL and not k.startswith("_") and v is not None]
    hist = [(k, v, c) for k, v, c in src.cards if k in ("HISTORY", "COMMENT")]
    look = src.look
    ac = None
    try:
        ac = tuple(int(x) for x in str(look.get("AUTOCROP", "")).split()) or None
        if ac and len(ac) != 4:
            ac = None
    except ValueError:
        ac = None
    return Master(Path(path), d, cards + [(k, c, "") for k, _v, c in hist], look,
                  st.target_name(look.get("OBJECT")) if look.get("OBJECT") else "Unknown",
                  core.normalize_filter(look.get("FILTER")) if look.get("FILTER") else
                  ("RGB" if d.shape[0] == 3 else "No filter"),
                  look.get("ALIGNREF"), ac, solved=bool(look.get("PLTSOLVD")) or bool(look.get("CTYPE1")))


def target_sets(masters: list) -> list:
    """Masters that belong together (same target and pixel grid)."""
    sets = {}
    for m in masters:
        key = (m.target, m.alignref or f"{m.path}", m.shape)
        sets.setdefault(key, []).append(m)
    return list(sets.values())


# ==========================================================================
# Crop
# ==========================================================================

def common_autocrop(masters: list):
    rects = [m.autocrop for m in masters if m.autocrop]
    if not rects:
        return None
    x0 = max(r[0] for r in rects)
    y0 = max(r[1] for r in rects)
    x1 = min(r[2] for r in rects)
    y1 = min(r[3] for r in rects)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def crop(m: Master, rect):
    x0, y0, x1, y1 = (int(v) for v in rect)
    h, w = m.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return
    m.data = np.ascontiguousarray(m.data[:, y0:y1, x0:x1])
    for k, d in (("CRPIX1", x0), ("CRPIX2", y0)):
        v = m.get(k)
        if isinstance(v, (int, float)):
            m.set_card(k, float(v) - d, "adjusted for crop")
    m.cards = [c for c in m.cards if c[0] != "AUTOCROP"]
    m.autocrop = None
    m.steps.append(("crop", f"Cropped to x {x0}-{x1}, y {y0}-{y1} ({x1 - x0} x {y1 - y0})"))


# ==========================================================================
# Temporary files for external tools (0..1 float FITS)
# ==========================================================================

def _write_temp(m: Master, path: Path, data=None, mono=False):
    d = m.data if data is None else data
    if mono and d.shape[0] == 3:
        d = d.mean(axis=0, keepdims=True)
    arr = np.clip(d / 65535.0, 0, 1).astype(np.float32)
    arr = arr[0] if arr.shape[0] == 1 else arr
    hdr = [(k, v, c) for k, v, c in m.cards if k not in ("HISTORY", "COMMENT") and k not in STRUCTURAL]
    core.write_fits(path, arr, hdr, bitpix=-32)


def _read_result(path, like: np.ndarray) -> np.ndarray:
    d = st.ImageSource(path)
    arr = d.read() if d.scale != 1.0 else d.read()
    arr = np.asarray(arr, np.float32)
    if float(np.nanmax(arr)) <= 1.5:          # tools write 0..1
        arr = arr * np.float32(65535.0)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape != like.shape:
        if arr.shape[0] == 1 and like.shape[0] == 3 and arr.shape[1:] == like.shape[1:]:
            arr = np.repeat(arr, 3, axis=0)
        else:
            raise tools.ToolError(f"tool returned an image of a different size {arr.shape} vs {like.shape}")
    return np.nan_to_num(arr).astype(np.float32)


# ==========================================================================
# Plate solving hints
# ==========================================================================

def _sexa(v, hours=False):
    if isinstance(v, (int, float)):
        return float(v)
    if not v:
        return None
    parts = re.split(r"[\s:hdms°'\"]+", str(v).strip())
    parts = [p for p in parts if p]
    try:
        sign = -1.0 if parts[0].startswith("-") else 1.0
        nums = [abs(float(parts[0]))] + [float(x) for x in parts[1:3]]
    except (ValueError, IndexError):
        return None
    val = nums[0] + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
    return sign * val * (15.0 if hours else 1.0)


def solve_hints(m: Master):
    L = m.look
    ra = L.get("CRVAL1") if L.get("CTYPE1") else None
    dec = L.get("CRVAL2") if L.get("CTYPE1") else None
    if ra is None:
        ra = _num(L.get("RA")) if _num(L.get("RA")) is not None else _sexa(L.get("OBJCTRA"), hours=True)
        dec = _num(L.get("DEC")) if _num(L.get("DEC")) is not None else _sexa(L.get("OBJCTDEC"))
    fov = None
    pix, foc = _num(L.get("XPIXSZ")), _num(L.get("FOCALLEN"))
    if pix and foc:
        fov = m.shape[0] * 206.265 * pix / foc / 3600.0
    return ra, dec, fov


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def apply_wcs(m: Master, cards):
    m.cards = [c for c in m.cards if not c[0].startswith(WCS_PREFIXES)]
    m.cards += [(k, v, c) for k, v, c in cards if k != "SIMPLE"]
    m.solved = True


# ==========================================================================
# Built-in gradient removal
# ==========================================================================

def builtin_gradient(plane: np.ndarray, degree=3, grid=(24, 16)):
    """Fit a smooth 2-D polynomial to the sky (stars and bright objects ignored)
    and subtract it, keeping the overall sky level."""
    h, w = plane.shape
    gx, gy = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    pts, vals = [], []
    for i in range(gy):
        for j in range(gx):
            box = plane[ys[i]:ys[i + 1]:2, xs[j]:xs[j + 1]:2]
            med = np.median(box)
            mad = np.median(np.abs(box - med)) * 1.4826 + 1e-6
            good = box[np.abs(box - med) < 2.5 * mad]
            if good.size:
                pts.append(((xs[j] + xs[j + 1]) / 2 / w - 0.5, (ys[i] + ys[i + 1]) / 2 / h - 0.5))
                vals.append(np.median(good))
    pts = np.array(pts)
    vals = np.array(vals)
    # drop samples sitting on bright nebulae / galaxies
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    keep = vals < med + 3 * mad
    pts, vals = pts[keep], vals[keep]

    def terms(x, y):
        return np.stack([x ** i * y ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)
    coef, *_ = np.linalg.lstsq(terms(pts[:, 0], pts[:, 1]), vals, rcond=None)
    Y, X = np.mgrid[0:h:8, 0:w:8]
    model_small = terms(X / w - 0.5, Y / h - 0.5) @ coef
    # upsample the smooth model
    ry = np.minimum(np.arange(h) // 8, model_small.shape[0] - 1)
    rx = np.minimum(np.arange(w) // 8, model_small.shape[1] - 1)
    model = model_small[ry][:, rx]
    return (plane - model + np.median(model)).astype(np.float32)


# ==========================================================================
# Quick-look image
# ==========================================================================

def stretch(x: np.ndarray, target=0.22, black_sigma=-2.8):
    s = x[::4, ::4]
    s = s[np.isfinite(s)]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826 + 1e-6
    lo = med + black_sigma * mad
    hi = float(np.percentile(s, 99.98))
    hi = max(hi, lo + 1e-6)
    n = np.clip((x - lo) / (hi - lo), 0, 1)
    nm = min(max((med - lo) / (hi - lo), 1e-6), 0.999)
    mbal = nm * (1 - target) / (nm - 2 * nm * target + target)
    return ((mbal - 1) * n / ((2 * mbal - 1) * n - mbal)).astype(np.float32)


def palettes_for(masters: list) -> list:
    f = {m.filter for m in masters}
    color = [m for m in masters if m.is_color]
    out = []
    if color:
        out.append("RGB (colour camera)")
    if {"R", "G", "B"} <= f:
        out += ["LRGB"] if "L" in f else []
        out += ["RGB"]
    if {"Ha", "OIII", "SII"} <= f:
        out += ["SHO (Hubble)", "HOO"]
    elif {"Ha", "OIII"} <= f:
        out += ["HOO"]
    out += [f"{m.filter} (mono)" for m in masters if not m.is_color]
    return out


def quick_look(masters: list, palette: str) -> np.ndarray:
    """(h, w, 3) float image in 0..1 for display/export."""
    by = {m.filter: m for m in masters if not m.is_color}

    def ch(f):
        return stretch(by[f].data[0])

    if palette.startswith("RGB (colour"):
        m = next(m for m in masters if m.is_color)
        rgb = np.stack([stretch(m.data[i]) for i in range(3)], -1)
    elif palette == "RGB":
        rgb = np.stack([ch("R"), ch("G"), ch("B")], -1)
    elif palette == "LRGB":
        rgb = np.stack([ch("R"), ch("G"), ch("B")], -1)
        lum = ch("L")
        cur = rgb.mean(-1, keepdims=True) + 1e-6
        rgb = np.clip(rgb * (lum[..., None] / cur), 0, 1)
    elif palette.startswith("SHO"):
        rgb = np.stack([ch("SII"), ch("Ha"), ch("OIII")], -1)
    elif palette == "HOO":
        o = ch("OIII")
        rgb = np.stack([ch("Ha"), o, o], -1)
    else:
        f = palette.split(" ")[0]
        g = ch(f)
        rgb = np.stack([g, g, g], -1)
    return np.clip(rgb, 0, 1)


def write_tiff16(path, rgb: np.ndarray):
    """Uncompressed 16-bit RGB TIFF (baseline), no extra libraries needed."""
    import struct
    h, w, _ = rgb.shape
    data = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype("<u2").tobytes()
    entries = []

    def ent(tag, typ, count, value):
        entries.append((tag, typ, count, value))
    n_entries = 11
    ifd_offset = 8
    ifd_size = 2 + n_entries * 12 + 4
    bps_offset = ifd_offset + ifd_size
    data_offset = bps_offset + 6
    ent(256, 4, 1, w)                       # ImageWidth
    ent(257, 4, 1, h)                       # ImageLength
    ent(258, 3, 3, bps_offset)              # BitsPerSample -> 16,16,16
    ent(259, 3, 1, 1)                       # no compression
    ent(262, 3, 1, 2)                       # RGB
    ent(273, 4, 1, data_offset)             # StripOffsets
    ent(277, 3, 1, 3)                       # SamplesPerPixel
    ent(278, 4, 1, h)                       # RowsPerStrip
    ent(279, 4, 1, len(data))               # StripByteCounts
    ent(284, 3, 1, 1)                       # chunky
    ent(339, 3, 3, 0)                       # SampleFormat (placeholder -> uint, below)
    entries[-1] = (339, 3, 1, 1)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_offset))
        f.write(struct.pack("<H", len(entries)))
        for tag, typ, count, value in entries:
            if typ == 3 and count == 1:
                f.write(struct.pack("<HHIHH", tag, typ, count, value, 0))
            else:
                f.write(struct.pack("<HHII", tag, typ, count, value))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<HHH", 16, 16, 16))
        f.write(data)


# ==========================================================================
# The PixInsight plan
# ==========================================================================

def pixinsight_plan(masters: list, palette_hint: str | None = None) -> list:
    """Ordered list of dicts: {title, where, why, done(bool), note}."""
    done = {k for m in masters for k, _t in m.steps}
    solved = all(m.solved for m in masters)
    color = any(m.is_color for m in masters)
    filters = {m.filter for m in masters}
    narrow = bool(filters & {"Ha", "OIII", "SII"}) and not color
    starsplit = "sxt" in done
    multi = len(masters) > 1
    aligned = multi and all(m.alignref for m in masters)
    steps = []

    def add(title, where, why, is_done=False, note=""):
        steps.append({"title": title, "where": where, "why": why, "done": is_done, "note": note})

    add("Open the prepared masters", "File › Open (the *_prepared.xisf files)",
        "These are linear 32-bit images — use the STF auto-stretch (Ctrl+A) only to look at them.")
    if multi and not aligned:
        add("StarAlignment", "Process › ImageRegistration › StarAlignment",
            "These masters do not share a pixel grid yet — align them to one reference before combining.")
    add("Crop the stacking edges", "Process › Geometry › DynamicCrop",
        "Ragged edges from dithering confuse gradient tools and colour calibration.", "crop" in done)
    add("Plate solve", "Script › Image Analysis › ImageSolver",
        "Needed for SPCC and for annotation.", solved,
        "Solved by ASTAP — the solution is in the file header." if solved else "")
    add("Remove gradients", "GraXpert, or Process › BackgroundModelization › DynamicBackgroundExtraction",
        "Light pollution and moon gradients must go before colour calibration.", "gradient" in done)
    if multi and not color:
        if narrow:
            add("Combine the narrowband channels", "Process › PixelMath (or the NarrowbandNormalization script)",
                "e.g. SHO: R = SII, G = Ha, B = OIII; HOO: R = Ha, G = B = OIII. "
                "Linear-fit the weaker channels to Ha first (Process › ColorCalibration › LinearFit).")
        else:
            add("Combine the colour channels", "Process › ChannelManagement › ChannelCombination",
                "Combine R, G and B into one colour image (keep L separate for LRGB).")
    add("Sharpen with BlurXTerminator", "Process › RC-Astro › BlurXTerminator",
        "Deconvolution works best on linear data, before colour calibration and noise reduction.",
        "bxt" in done)
    if color or (multi and not narrow):
        add("Colour calibration", "Process › ColorCalibration › SpectrophotometricColorCalibration (SPCC)",
            "Uses Gaia star colours — requires the plate solution. Run BackgroundNeutralization first if "
            "SPCC asks for it.", False,
            "" if solved else "Plate solve first (step above).")
    elif narrow:
        add("Balance the narrowband colours", "Process › ColorCalibration › BackgroundNeutralization",
            "SPCC is not meant for narrowband palettes; neutralise the background instead "
            "(SPCC's narrowband mode is an option for Ha/OIII/SII filters).")
    add("Reduce noise (linear, light)", "Process › RC-Astro › NoiseXTerminator",
        "A light pass now (denoise ~0.5–0.7) keeps the stretch clean; a final pass can follow.", "nxt" in done)
    add("Separate the stars", "Process › RC-Astro › StarXTerminator (Generate star image)",
        "Lets you stretch the nebula/galaxy hard without bloating the stars.", starsplit,
        "Starless and stars-only images were already saved." if starsplit else "Optional.")
    add("Stretch to non-linear", "Script › Utilities › GeneralizedHyperbolicStretch  (or HistogramTransformation "
        "from the STF)", "Stretch the starless image (and the stars image gently) — this is where the image "
        "comes to life.")
    if not color and "L" in filters:
        add("Add the luminance", "Process › ColorSpaces › LRGBCombination",
            "After both L and RGB are stretched to similar brightness.")
    add("Fine-tune contrast and colour", "Process › IntensityTransformations › CurvesTransformation, "
        "LocalHistogramEqualization", "Saturation, contrast and local detail on the stretched image.")
    if starsplit:
        add("Put the stars back", "Process › PixelMath:  ~((~starless)*(~stars))",
            "Screen-blend the stretched stars onto the processed starless image.")
    add("Final noise pass and export", "NoiseXTerminator (light) › File › Save As (TIFF/JPG)",
        "Keep a 16-bit TIFF and an XISF project for later.")
    return steps


def plan_html(target: str, masters: list, steps: list, files: list) -> str:
    rows = []
    for i, s in enumerate(steps, 1):
        mark = "✅" if s["done"] else "➡️"
        state = "<span class='done'>done here</span>" if s["done"] else ""
        rows.append(f"<tr class='{'d' if s['done'] else ''}'><td>{i}</td><td>{mark}</td>"
                    f"<td><b>{s['title']}</b> {state}<br><span class='where'>{s['where']}</span><br>"
                    f"{s['why']}" + (f"<br><i>{s['note']}</i>" if s['note'] else "") + "</td></tr>")
    flist = "".join(f"<li>{Path(f).name}</li>" for f in files)
    order, who = [], {}
    for m in masters:
        for _k, t in m.steps:
            if t not in who:
                order.append(t)
                who[t] = []
            who[t].append(m.filter)
    all_f = [m.filter for m in masters]
    did = "".join(f"<li>{t}" + ("" if len(masters) == 1 or who[t] == all_f else
                                f" <span class='where'>({', '.join(who[t])})</span>") + "</li>" for t in order)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{target} – next steps in PixInsight</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#1e1e22;color:#e4e4ea;max-width:900px;margin:24px auto;padding:0 16px}}
h1{{font-size:22px}} table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #3a3a42;padding:8px;vertical-align:top}}
tr.d td{{opacity:.6}} .where{{color:#8fb4ff}} .done{{color:#5fd27a;font-size:12px}} li{{margin:2px 0}}</style></head><body>
<h1>{target}: your images are ready for final processing</h1>
<p>SubZeroAstro prepared these linear masters. Open them in PixInsight and continue with the steps below —
steps marked ✅ were already done here.</p>
<h3>Prepared files</h3><ul>{flist}</ul>
<h3>What was done</h3><ul>{did or '<li>(nothing — masters copied as they were)</li>'}</ul>
<h3>Recommended PixInsight steps, in order</h3><table>{''.join(rows)}</table>
</body></html>"""


def plan_text(target: str, steps: list) -> str:
    lines = [f"{target}: next steps in PixInsight", ""]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i:2d}. [{'done' if s['done'] else '    '}] {s['title']}  —  {s['where']}")
        lines.append(f"      {s['why']}" + (f"  ({s['note']})" if s['note'] else ""))
    return "\n".join(lines) + "\n"


# ==========================================================================
# Running everything
# ==========================================================================

@dataclass
class PrepareOptions:
    crop: bool = True
    crop_rect: tuple | None = None          # None = automatic
    solve: bool = True
    astap: str | None = None
    solve_radius: float = 30.0
    gradient: bool = True
    gradient_tool: str = "graxpert"         # "graxpert" or "builtin"
    graxpert: str | None = None
    gx_smoothing: float = 0.1
    gx_correction: str = "Subtraction"
    rcastro: str | None = None
    xt: dict = field(default_factory=dict)  # key -> {"enabled", "values", "mode"}
    device: str = "auto"
    formats: tuple = ("xisf",)
    quicklook: bool = True
    palette: str | None = None
    out_dir: str | None = None              # default: <master folder>/Prepared
    work_dir: str | None = None


def run_prepare(masters: list, opts: PrepareOptions, progress=None, log=None, cancel=None,
                save_image=None) -> dict:
    """Prepare one set of masters (same target / grid). save_image(path, rgb8) writes
    JPG/PNG (the GUI supplies it). Returns {"files", "plan", "report"}."""
    t0 = time.time()
    lg = log or (lambda s: None)

    def pr(text, frac):
        if progress:
            progress(text, frac)
        if cancel and cancel():
            raise Cancelled()

    def can():
        return bool(cancel and cancel())

    target = masters[0].target
    out_dir = Path(opts.out_dir) if opts.out_dir else masters[0].path.parent / "Prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fbprep_", dir=opts.work_dir or None))
    n = len(masters)
    files = []

    # 1. crop (same rectangle for every master so they stay aligned)
    if opts.crop:
        rect = opts.crop_rect or common_autocrop(masters)
        if rect:
            for m in masters:
                crop(m, rect)
            lg(f"Cropped {n} master(s) to {rect}")
        else:
            lg("Crop: no auto-crop information in these masters — skipped (draw a crop rectangle to crop)")

    # 2. plate solve
    if opts.solve and opts.astap:
        astap = tools.ASTAP(opts.astap)
        for i, m in enumerate(masters):
            pr(f"Plate solving {m.name}", 0.05 + 0.1 * i / n)
            ra, dec, fov = solve_hints(m)
            tmp = work / f"solve_{i}.fits"
            _write_temp(m, tmp, mono=True)
            try:
                cards = astap.solve(tmp, ra, dec, fov, radius=opts.solve_radius if ra is not None else 180.0,
                                    cancel=can)
                apply_wcs(m, cards)
                c = {k: v for k, v, _ in cards}
                m.steps.append(("solve", f"Plate solved with ASTAP (RA {c.get('CRVAL1', 0):.4f}°, "
                                         f"Dec {c.get('CRVAL2', 0):.4f}°)"))
                lg(f"Solved {m.name}: RA {c.get('CRVAL1', 0):.4f}, Dec {c.get('CRVAL2', 0):.4f}")
            except tools.ToolCancelled:
                raise Cancelled()
            except tools.ToolError as e:
                lg(f"Plate solve failed for {m.name}: {e}")
    elif opts.solve:
        lg("Plate solve skipped — set the ASTAP folder to enable it")

    # 3. gradient removal
    if opts.gradient:
        for i, m in enumerate(masters):
            pr(f"Removing gradients: {m.name}", 0.15 + 0.15 * i / n)
            if opts.gradient_tool == "graxpert" and opts.graxpert:
                tmp = work / f"bge_{i}.fits"
                _write_temp(m, tmp)
                try:
                    res = tools.GraXpert(opts.graxpert).background_extraction(
                        tmp, work / f"bge_{i}_out", smoothing=opts.gx_smoothing, correction=opts.gx_correction,
                        cancel=can)
                    m.data = _read_result(res, m.data)
                    m.steps.append(("gradient", f"Gradients removed with GraXpert "
                                                f"({opts.gx_correction.lower()}, smoothing {opts.gx_smoothing:g})"))
                except tools.ToolCancelled:
                    raise Cancelled()
                except tools.ToolError as e:
                    lg(f"GraXpert failed for {m.name}: {e} — using the built-in method instead")
                    m.data = np.stack([builtin_gradient(p) for p in m.data])
                    m.steps.append(("gradient", "Gradients removed (built-in polynomial model)"))
            else:
                m.data = np.stack([builtin_gradient(p) for p in m.data])
                m.steps.append(("gradient", "Gradients removed (built-in polynomial model)"))

    # 4-6. RC-Astro tools, in the recommended order
    rc = tools.RCAstro(opts.rcastro) if opts.rcastro else None
    stars_images = {}
    for k_i, key in enumerate(("bxt", "nxt", "sxt")):
        cfg = opts.xt.get(key) or {}
        if not cfg.get("enabled"):
            continue
        if rc is None:
            lg(f"{key.upper()} skipped — set the RC-Astro folder to enable it")
            continue
        spec = cfg.get("spec") or rc.spec(key)
        args = spec.args(cfg.get("values", {}), cfg.get("mode"))
        for i, m in enumerate(masters):
            base = 0.3 + 0.15 * k_i + 0.15 * i / n
            pr(f"{spec.name}: {m.name}", base)
            tmp_in = work / f"{key}_{i}.fits"
            tmp_out = work / f"{key}_{i}_out.fits"
            _write_temp(m, tmp_in)
            try:
                written = rc.run(key, tmp_in, tmp_out, args, device=opts.device,
                                 progress=lambda f, t, b=base: pr(f"{spec.name}: {m.name} {t}", b + 0.15 / n * f),
                                 log=lg, cancel=can)
            except tools.ToolCancelled:
                raise Cancelled()
            main = next((w for w in written if "_stars" not in Path(w).stem), str(tmp_out))
            m.data = _read_result(main, m.data)
            extra = [w for w in written if "_stars" in Path(w).stem]
            if extra:
                stars_images[id(m)] = _read_result(extra[0], m.data)
            desc = " ".join(args) or "default settings"
            m.steps.append((key, f"{spec.name} ({desc})"))
            lg(f"{spec.name} done: {m.name}  [{desc}]")

    # 7. save the prepared masters
    for i, m in enumerate(masters):
        pr(f"Saving {m.name}", 0.8 + 0.1 * i / n)
        hdr = [(k, v, c) for k, v, c in m.cards if k not in ("HISTORY",)]
        hdr += [("HISTORY", f"SubZeroAstro prepare: {t}", "") for _k, t in m.steps]
        hdr.append(("PREPARED", True, "prepared for PixInsight by SubZeroAstro"))
        base = out_dir / f"{m.name}_prepared"
        data = m.data[0] if m.data.shape[0] == 1 else m.data
        written = st.save_master_file(base, data, hdr, opts.formats, "MasterLight")
        m.outputs += written
        files += written
        if id(m) in stars_images:
            sd = stars_images[id(m)]
            sw = st.save_master_file(out_dir / f"{m.name}_prepared_stars", sd[0] if sd.shape[0] == 1 else sd,
                                     hdr + [("STARONLY", True, "stars-only image from StarXTerminator")],
                                     opts.formats, "MasterLight")
            files += sw

    # 8. quick look
    if opts.quicklook:
        pal = opts.palette or (palettes_for(masters) or [None])[0]
        if pal:
            pr(f"Quick-look image ({pal})", 0.92)
            rgb = quick_look(masters, pal)
            stem = f"{target}_{re.sub(r'[^A-Za-z0-9]+', '', pal.split(' ')[0])}_quicklook"
            tif = out_dir / f"{stem}.tif"
            write_tiff16(tif, rgb)
            files.append(tif)
            if save_image:
                rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
                for ext in ("jpg", "png"):
                    p = out_dir / f"{stem}.{ext}"
                    if save_image(p, rgb8):
                        files.append(p)

    # 9. the PixInsight plan
    steps = pixinsight_plan(masters)
    report = out_dir / f"{target}_PixInsight_next_steps.html"
    report.write_text(plan_html(target, masters, steps, files), encoding="utf-8")
    (out_dir / f"{target}_PixInsight_next_steps.txt").write_text(plan_text(target, steps), encoding="utf-8")
    files.append(report)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    lg(f"Prepared {n} master(s) for {target} in {time.time() - t0:.0f} s → {out_dir}")
    pr("Done", 1.0)
    return {"files": files, "plan": steps, "report": report, "out_dir": out_dir}
