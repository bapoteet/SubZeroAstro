"""
selftest.py - run the real app window on synthetic frames and check it works.

    python subzeroastro.py --selftest [output_folder]
    "SubZeroAstro.exe" --selftest [output_folder]

Creates a small two-filter session with a few deliberately bad frames, opens it
in the real window, waits for measuring to finish, exercises flag / reject /
undo / Auto-Flag, saves screenshots and writes selftest_result.txt.
Exit code 0 = all checks passed. Used by the automatic GitHub build.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

import fits_core as core


def _frame(rng, stars, fwhm=3.0, ecc=0.1, transp=1.0, bg=1000.0, noise=14.0, h=900, w=1350):
    img = np.full((h, w), bg, np.float32)
    alpha = fwhm / 0.8698
    q = np.sqrt(1 - ecc ** 2)
    amaj, amin = alpha / np.sqrt(q), alpha * np.sqrt(q)
    yy, xx = np.mgrid[-14:15, -14:15].astype(np.float32)
    for y, x, a in stars:
        iy, ix = int(y), int(x)
        dx, dy = xx - (x - ix), yy - (y - iy)
        u, v = (dx * 0.8 + dy * 0.6) / amaj, (-dx * 0.6 + dy * 0.8) / amin
        img[iy - 14:iy + 15, ix - 14:ix + 15] += transp * a * (1 + u * u + v * v) ** -4
    # a faint "galaxy" so the picture is not only stars
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    img += 180 * np.exp(-(((gx - w * 0.55) / 160) ** 2 + ((gy - h * 0.45) / 70) ** 2))
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 65535)


def make_session(folder: Path):
    rng = np.random.default_rng(11)
    stars = [(rng.uniform(30, 870), rng.uniform(30, 1320), rng.uniform(400, 12000)) for _ in range(220)]
    bad = {}
    for i in range(16):
        filt = "Ha" if i % 2 else "OIII"
        kw, tag = {}, ""
        if i == 5:
            kw, tag = dict(ecc=0.85), "trailed"
        if i == 8:
            kw, tag = dict(transp=0.25, bg=1500), "cloud"
        if i == 12:
            kw, tag = dict(fwhm=4.8), "focus"
        name = f"Light_{filt}_{i:03d}.fits"
        hdr = {"OBJECT": "M 33", "FILTER": filt, "EXPTIME": 300.0, "INSTRUME": "Test Camera",
               "IMAGETYP": "LIGHT", "DATE-OBS": f"2026-10-11T0{3 + i // 8}:{(i % 8) * 7:02d}:00",
               "SITELONG": -95.65, "FOCALLEN": 1140.0, "XPIXSZ": 3.76, "ROWORDER": "TOP-DOWN"}
        core.write_fits(folder / name, _frame(rng, stars, **kw), hdr, bitpix=16)
        if tag:
            bad[name] = tag
    return bad


def run(app, out_dir: Path) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QLabel
    import subzeroastro as fb

    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    failures: list[str] = []

    def log(msg):
        log_lines.append(msg)
        try:
            print(msg, flush=True)
        except Exception:  # noqa: BLE001 - windowed exe has no console
            pass

    def check(cond, what):
        log(("PASS  " if cond else "FAIL  ") + what)
        if not cond:
            failures.append(what)

    def fits_small(win, name, w_max=1280, h_max=680):
        """A page must fit a small laptop screen (1366x768 or 1920x1080 at 150 %, minus the taskbar)."""
        hint = win.minimumSizeHint()
        mw, mh = max(hint.width(), win.minimumWidth()), max(hint.height(), win.minimumHeight())
        check(mw <= w_max and mh <= h_max, f"{name} fits a small laptop screen (needs at least {mw}x{mh}, "
                                           f"limit {w_max}x{h_max})")

    def finish(code):
        log(f"\nRESULT: {'ALL CHECKS PASSED' if code == 0 else f'{len(failures)} CHECK(S) FAILED'}")
        (out_dir / "selftest_result.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        app.exit(code)

    def excepthook(etype, value, tb):
        log("EXCEPTION:\n" + "".join(traceback.format_exception(etype, value, tb)))
        failures.append(f"exception: {value}")
        finish(1)

    sys.excepthook = excepthook

    session = Path(tempfile.mkdtemp(prefix="sza_selftest_")) / "M33"
    session.mkdir()
    t0 = time.time()
    bad = make_session(session)
    log(f"Created 16 synthetic frames in {time.time() - t0:.1f}s at {session}")

    # --- sky flat: no flats for a filter -> a flat made from the light frames ---------------
    import stacking as _stk
    _sfd = session.parent / "skyflat"
    _sfd.mkdir(exist_ok=True)
    _rs = np.random.default_rng(4)
    _sh, _sw = 300, 400
    _yy, _xx = np.mgrid[:_sh, :_sw].astype(np.float32)
    _true = 1 - 0.3 * (np.hypot(_xx - 210, _yy - 145) / 250) ** 2
    _true *= 1 - 0.06 * np.clip((1 - np.hypot(_xx - 100, _yy - 90) / 22) / 0.2, 0, 1)
    _true = (_true / np.median(_true)).astype(np.float32)
    _lfs = []
    for _i in range(16):
        _sky = 800 * _rs.uniform(0.8, 1.2)
        _ang, _gr = _rs.uniform(0, 6.28), _rs.uniform(0, 0.08)
        _sc = _sky * (1 + _gr * 2 * ((_xx / _sw - 0.5) * np.cos(_ang) + (_yy / _sh - 0.5) * np.sin(_ang)))
        for _k in range(60):
            _sx, _sy = int(_rs.uniform(3, _sw - 3)), int(_rs.uniform(3, _sh - 3))
            _sc[_sy - 1:_sy + 2, _sx - 1:_sx + 2] += _rs.uniform(300, 9000)
        _raw = _sc * _true + 300 + _rs.normal(0, 30, (_sh, _sw))
        _fp = _sfd / f"L_{_i:02d}.fits"
        core.write_fits(_fp, np.clip(_raw, 0, 65535).astype(np.uint16),
                        {"IMAGETYP": "Light Frame", "FILTER": "L", "EXPTIME": 60.0, "OBJECT": f"Field{_i % 4}",
                         "INSTRUME": "Test Camera"}, bitpix=16)
        _lfs.append(_stk.LightFrame(_stk.read_info(_fp)) if hasattr(_stk.LightFrame, "__dataclass_fields__")
                    else type("LF", (), {"info": _stk.read_info(_fp)})())
    _grp = _stk.LightGroup("Field0", "L", "Test Camera", "TestRig", _lfs)
    _pool = []
    for _lf in _lfs:
        _lf.info._rig = "TestRig"
        _pool.append(_lf.info)
    _pl = _stk.plan_group(_grp, [], sky_pool=_pool)
    check(_pl.flat is not None and _pl.flat.sky and not any(k == "flat" for k, _ in _pl.warnings)
          and "SkyFlat (Mote Buster)" in " ".join(_pl.notes) and "SkyFlat (Mote Buster)" in _pl.flat.label(),
          "no flats for a filter: page 2 plans a sky flat from the light frames")
    _built = _stk.build_sky_flat(_pl.flat, np.full((_sh, _sw), 300.0, np.float32))
    _vt = _true[10:30, 10:30].mean() / _true[135:155, 200:220].mean()
    _vb = _built[10:30, 10:30].mean() / _built[135:155, 200:220].mean()
    _din = np.hypot(_xx - 100, _yy - 90) < 18
    _dout = (np.hypot(_xx - 100, _yy - 90) > 30) & (np.hypot(_xx - 100, _yy - 90) < 38)
    _dt = 1 - _true[_din].mean() / _true[_dout].mean()
    _db = 1 - _built[_din].mean() / _built[_dout].mean()
    check(abs(_vb - _vt) < 0.04 and _db > 0.6 * _dt,
          f"sky flat captures the vignetting (corner {_vt:.3f} -> {_vb:.3f}) and the dust donut "
          f"({_dt:.3f} -> {_db:.3f}) with the stars and light-pollution tilt taken out")
    try:
        _stk.build_sky_flat(_pl.flat, np.full((_sh, _sw), 1e6, np.float32))
        _ate = ""
    except ValueError as _e:
        _ate = str(_e)
    check("as bright as the sky" in _ate, "a dark as bright as the sky stops the sky flat with a clear message")
    # a SkyFlat that looks poor (or can't be made) asks: proceed without flats / use anyway / cancel
    import stack_window as _sw
    _good = _stk.build_master(_pl.flat)
    _probs_good = _stk.sky_flat_problems(_pl.flat)
    _keep_stats = dict(_pl.flat.sky_stats)
    _pl.flat.sky_stats.update(grain=0.05, high=2.5)
    _probs_bad = _stk.sky_flat_problems(_pl.flat)
    _pl.flat.sky_stats = _keep_stats
    _wk = _sw.StackWorker([], [], _stk.StackOptions(), {})
    _said = []
    _wk.log.connect(_said.append)
    _real_bm = _sw.st.build_master

    def _boom(*_a, **_k):
        raise ValueError("SkyFlat (Mote Buster) can't be made: the master dark or bias is as bright as the sky")
    _sw.st.build_master = _boom
    try:
        _wk.sky_answer = "without"
        _without = _wk._sky_flat(_stk.CalItem("flat", _pl.flat.infos, False, sky=True), None, None, lambda t, f: None)
    finally:
        _sw.st.build_master = _real_bm
    _it2 = _stk.CalItem("flat", _pl.flat.infos, False, sky=True, data=_good, sky_stats=dict(_keep_stats, grain=0.05))
    _wk.sky_answer = "use"
    _use = _wk._sky_flat(_it2, None, None, lambda t, f: None)
    check(not _probs_good and any("grainy" in x for x in _probs_bad) and any("unusual" in x for x in _probs_bad)
          and _without is False and _use is True and any("may not be good enough" in x for x in _said)
          and hasattr(_sw.StackWindow, "ask_sky_flat"),
          f"a poor SkyFlat is caught ({len(_probs_bad)} reasons) and the user chooses: without flats or use anyway")
    check(_stk.plan_group(_grp, [], sky_pool=_pool[:5]).flat is None,
          "too few light frames: no sky flat, the usual 'no flats' warning")

    # --- XISF light frames on page 1, and every XISF compression PixInsight writes ----------------
    import xisf as _xs
    _xd = session.parent / "xisf_lights"
    _xd.mkdir(exist_ok=True)
    _xr = np.random.default_rng(3)
    _xH, _xW = 300, 400
    _xyy, _xxx = np.mgrid[:_xH, :_xW]
    _ximg = 800 + _xr.normal(0, 10, (_xH, _xW))
    for _k in range(80):
        _sx, _sy, _sa = _xr.uniform(10, _xW - 10), _xr.uniform(10, _xH - 10), _xr.uniform(500, 20000)
        _ximg += _sa * np.exp(-((_xxx - _sx) ** 2 + (_xyy - _sy) ** 2) / (2 * 1.8 ** 2))
    _ximg[:20, :20] = 30000
    _xhd = {"IMAGETYP": "Light Frame", "EXPTIME": 120.0, "FILTER": "Ha", "OBJECT": "M42", "INSTRUME": "TestCam",
            "DATE-OBS": "2026-10-17T03:00:00", "ROWORDER": "TOP-DOWN"}
    _xu = np.clip(_ximg, 0, 65535).astype(np.uint16)
    core.write_fits(_xd / "a.fits", _xu, _xhd, bitpix=16)
    _xs.write_xisf(_xd / "b.xisf", _xu.astype(np.float32), [(k, v, "") for k, v in _xhd.items()],
                   image_type="Light", sample_format="UInt16", compression="zlib+sh")
    _xs.write_xisf(_xd / "c.xisf", (_xu / 65535.0).astype(np.float32), [(k, v, "") for k, v in _xhd.items()],
                   image_type="Light")
    (_xd / "._a.fits").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ")   # macOS AppleDouble
    (_xd / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    _xfr = core.scan_folder(_xd)
    _xres = {f.path.name: core.analyze_frame(f.path) for f in _xfr}
    _xm = {n: (round(r["metrics"]["fwhm"], 3), r["metrics"]["stars"], round(r["metrics"]["background"]))
           for n, r in _xres.items()}
    _xtop = {n: bool(r["thumb"][:2, :2].mean() > r["thumb"].mean()) for n, r in _xres.items()}
    import stacking as _stk2
    check(not any(f.path.name.startswith(".") for f in _xfr) and len(_xfr) == 3
          and not any(i.path.name.startswith(".") for i in _stk2.scan_paths([_xd])[0]),
          "macOS '._' copies and .DS_Store on a NAS / external drive are ignored, not shown as unreadable")
    check(sorted(_xm) == ["a.fits", "b.xisf", "c.xisf"] and len(set(_xm.values())) == 1 and all(_xtop.values()),
          f"page 1 loads XISF light frames (16-bit and 0..1 float) and measures them exactly like FITS {_xm}")
    _xdata = _xu.astype(np.float32)
    _xok = {}
    for _codec in ("zlib", "lz4", "lz4hc+sh", "zstd+sh"):
        try:
            _xs.write_xisf(_xd / "cmp.xisf", _xdata, {"IMAGETYP": "Light"}, sample_format="UInt16", compression=_codec)
            _xok[_codec] = bool(np.array_equal(_xs.read_xisf(_xd / "cmp.xisf")[0], _xdata))
        except ValueError as _e:
            _xok[_codec] = str(_e)
    _xpure = None
    if _xs._lz4block is not None:
        _src = _xu.tobytes()
        _xpure = _xs.lz4_block_decompress(_xs._lz4block.compress(_src, store_size=False), len(_src)) == _src
    check(all(v is True for v in _xok.values()) and _xpure is not False,
          f"XISF with zlib, LZ4, LZ4-HC and Zstandard compression opens ({_xok}; built-in LZ4 decoder agrees: {_xpure})")

    # --- plate solving retries when the header's field size is wrong (binning, reducer) -----------
    import prepare as _pp
    import tools as _tl

    class _StubAstap:
        exe = None

        def __init__(self):
            self.calls = []

        def solve(self, path, ra, dec, fov, radius=30.0, cancel=None):
            self.calls.append(fov)
            if fov and abs(fov - 0.355) < 0.03:
                return [("CRVAL1", 190.0, ""), ("CRVAL2", -11.6, "")]
            raise _tl.ToolError("ASTAP could not solve the image: No solution found!")
    _sa = _StubAstap()
    _cards, _note = _pp.solve_with_retries(_sa, "x.fits", 190.0, -11.6, 0.71, 30.0)
    try:
        _pp.solve_with_retries(_StubAstap(), "x.fits", 190.0, -11.6, 3.0, 30.0)
        _fail_msg = ""
    except _tl.ToolError as _e:
        _fail_msg = str(_e)
    check(_cards and "binning" in _note and len(_sa.calls) == 2 and "were tried" in _fail_msg,
          f"plate solving retries other field sizes when the header's is wrong ({_note}); a final failure "
          "says what was tried")

    # --- Build Calibration Masters: one mixed folder of subframes -> masters in the library ----------
    import cal_builder as _cb
    _cr = np.random.default_rng(1)
    _cdir = session.parent / "ScopeXCalis"
    (_cdir / "Darks").mkdir(parents=True, exist_ok=True)
    _cH, _cW = 120, 160
    _cy, _cx = np.mgrid[:_cH, :_cW]
    _cvig = 1 - 1.2 * (((_cx - _cW / 2) / _cW) ** 2 + ((_cy - _cH / 2) / _cH) ** 2)

    def _cput(name, data, typ, exp, filt=None, folder=_cdir):
        hd = {"EXPTIME": exp, "GAIN": 100, "OFFSET": 40, "CCD-TEMP": -10.0, "INSTRUME": "ZWO ASI2600MM Pro"}
        if typ:
            hd["IMAGETYP"] = typ
        if filt:
            hd["FILTER"] = filt
        core.write_fits(folder / name, np.clip(data, 0, 65535).astype(np.uint16), hd, bitpix=16)
    for _i in range(5):
        _cput(f"b_{_i}.fits", 500 + _cr.normal(0, 4, (_cH, _cW)), "Bias Frame", 0.0)
        _cput(f"d300_{_i}.fits", 510 + _cr.normal(0, 5, (_cH, _cW)), "Dark Frame", 300.0)
        _cput(f"leak_{_i}.fits", 500 + 160 * _cvig + _cr.normal(0, 5, (_cH, _cW)), "Dark Frame", 180.0)
        _cput(f"x_{_i}.fits", 501 + _cr.normal(0, 4, (_cH, _cW)), None, 2.0, folder=_cdir / "Darks")
        _cput(f"fH_{_i}.fits", 500 + 25000 * _cvig + _cr.normal(0, 60, (_cH, _cW)), "Flat Frame", 2.0, "H")
        _cput(f"fO_{_i}.fits", 500 + 26000 * _cvig + _cr.normal(0, 60, (_cH, _cW)), "Flat Frame", 2.0, "O")
    _cinf, _ = _stk.scan_paths([_cdir])
    _cjobs, _ = _cb.plan(_cinf)
    _crig = _cb.suggest_rig([_cdir], _cinf)
    _clib = session.parent / "CalLibrary"
    _cb.build(_cjobs, _clib, _crig, log=lambda *_a: None)
    _ckinds = sorted(j.item.kind + (j.item.filter if j.item.kind == "flat" else f"{j.item.exptime:g}")
                     for j in _cjobs if j.files)
    _cleak = [j for j in _cjobs if j.item.kind == "dark" and j.item.exptime == 180]
    _cclean = [j for j in _cjobs if not (j.item.kind == "dark" and j.item.exptime == 180)]
    _cflat = [j for j in _cjobs if j.item.kind == "flat"]
    _clibitems = _stk.scan_library(_clib)
    check(_crig == "ScopeX" and _ckinds == ["bias0", "dark180", "dark2", "dark300", "flatHa", "flatOIII"]
          and _cleak and any("leaked" in w for w in _cleak[0].warnings) and not any(j.warnings for j in _cclean)
          and all(j.helper_dark is not None for j in _cflat) and len(_clibitems) == 6
          and all(c.rig == "ScopeX" for c in _clibitems),
          f"Build Calibration Masters sorts one mixed folder into {_ckinds} for rig '{_crig}', saves them to "
          "the library and flags the leaky dark")

    # --- calibration check: a dark on the wrong scale is caught, a float master is read right ----
    _lt = session.parent / "calcheck_light.fits"
    _lraw = (510 + _rs.normal(0, 8, (120, 160)) + 6).astype(np.float32)
    core.write_fits(_lt, np.clip(_lraw, 0, 65535).astype(np.uint16), {"IMAGETYP": "Light Frame", "EXPTIME": 180.0},
                    bitpix=16)
    _li = _stk.read_info(_lt)
    _good = type("Cal", (), {"sub": np.full((120, 160), 510.0, np.float32), "hot": ("y", "x")})()
    _bad = type("Cal", (), {"sub": np.full((120, 160), 510.0 / 65535, np.float32), "hot": ("y", "x")})()
    _g, _b = _stk.calibration_check(_li, _good), _stk.calibration_check(_li, _bad)
    _yl, _xl = np.mgrid[:120, :160].astype(np.float32)
    _leak = type("Cal", (), {"sub": (510 + 150 * np.exp(-((_xl - 80) ** 2 + (_yl - 60) ** 2) / (2 * 50 ** 2))
                                      + _rs.normal(0, 3, (120, 160))).astype(np.float32), "hot": ("y", "x")})()
    _lk = _stk.calibration_check(_li, _leak)
    check(any("BRIGHTER than the lights" in x for x in _lk) and any("uneven like a picture" in x for x in _lk)
          and not any("uneven" in x for x in _g),
          "a dark that light leaked into (brighter than the lights, uneven like a picture) is reported plainly")
    check(_g and "Calibration check" in _g[0] and not any("WARNING" in x for x in _g)
          and any("WARNING" in x and "practically zero" in x for x in _b),
          f"page 2 checks the dark subtraction and warns when a master is on the wrong scale ({_b[-1][:60]}…)")
    import xisf as _xs
    _xp = session.parent / "float_dark_nobounds.xisf"
    _xs.write_xisf(_xp, np.full((40, 50), 510.0 / 65535, np.float32), image_type="Master Dark")
    _src = _stk.ImageSource(_xp)                         # its stated range is the data's own min:max
    check(abs(float(np.median(_src.read())) - 510.0) < 1.0,
          "a 0..1 floating-point master with any (or no) stated range is read in camera units")

    # --- RGB + Ha of a galaxy: only the HII knots are added to red ------------------------------
    import finish as _Fg
    _rg = np.random.default_rng(2)
    _gy, _gx = np.mgrid[:300, :400].astype(np.float32)
    _disk = 0.08 * np.exp(-np.sqrt(((_gx - 200) / 110) ** 2 + ((_gy - 150) / 50) ** 2) * 2.5)
    _kn = np.zeros((300, 400), np.float32)
    _kp = [(200 + 80 * np.cos(a), 150 + 35 * np.sin(a)) for a in np.linspace(0, 6.0, 10)]
    for _x, _y in _kp:
        _kn += 0.012 * np.exp(-((_gx - _x) ** 2 + (_gy - _y) ** 2) / (2 * 2.5 ** 2))
    _ns = lambda: _rg.normal(0, 0.0015, (300, 400))
    _chg = {"R": (0.01 + _disk + _ns()).astype(np.float32), "G": (0.01 + 0.95 * _disk + _ns()).astype(np.float32),
            "B": (0.01 + 0.8 * _disk + _ns()).astype(np.float32),
            "Ha": (0.003 + 0.3 * 1.06 * _disk + _kn + _ns()).astype(np.float32)}
    _sg = _Fg.State(channels=_chg, linear=True, kind="mono_rgb", target="M 101")
    _vg = _Fg.TOOLS["ha_glow"].variants(_sg)
    _ln = _Fg.TOOLS["ha_glow"].line(_sg, _vg[0].params)
    _km = np.zeros((300, 400), bool)
    for _x, _y in _kp:
        _km |= (_gx - _x) ** 2 + (_gy - _y) ** 2 < 9
    _dm = (np.sqrt(((_gx - 200) / 110) ** 2 + ((_gy - 150) / 50) ** 2) < 0.5)
    check(_Fg.is_galaxy_target("M 101") and not _Fg.is_galaxy_target("NGC 7000") and _vg[0].recommended
          and "HII" in _vg[0].label and _ln[_km].mean() > 15 * _ln[_dm].mean(),
          f"RGB+Ha galaxy: the HII regions are added on their own (knots {_ln[_km].mean():.4f}, "
          f"smooth disk {_ln[_dm].mean():.5f})")

    # --- smart telescopes: header recognition (synthetic files, real header layouts) ----
    import numpy as _np
    import stacking as _st
    sm = session.parent / "smart"
    sm.mkdir()
    rng = _np.random.default_rng(1)
    cfa = (rng.normal(400, 8, (64, 96))).astype(_np.uint16)
    core.write_fits(sm / "M 31_15s60_Astro_20251231-215912880_14C.fits", cfa,
                    [("DATE-OBS", "2025-12-31T21:59:12.879", ""), ("EXPTIME", 15.0, ""), ("GAIN", 60, ""),
                     ("FILTER", "Astro", ""), ("DET-TEMP", 14, ""), ("FOCALLEN", 150.0, ""), ("XPIXSZ", 2.9, ""),
                     ("OBJECT", "M 31", ""), ("BAYERPAT", "RGGB", ""), ("TELESCOP", "DWARF mini", ""),
                     ("INSTRUME", "DWARF mini", ""), ("ORIGIN", "DWARFLAB", "")], bitpix=16)
    core.write_fits(sm / "img-0001.fits", cfa,
                    [("DATE-OBS", "2026-07-01T03:02:08", ""), ("EXPOSURE", 10.0, ""), ("TEMP", 28.8, ""),
                     ("FOCAL", 250, ""), ("INSTRUME", "vesperapro-2ccf67443292", ""), ("BAYERPAT", "GBRG", ""),
                     ("GAIN", 201, ""), ("PIXSZ", 2.0, ""), ("OBJECT", "SH2-101", ""), ("FILTER", "CLS", "")],
                    bitpix=16)
    rgb = _np.stack([cfa, cfa, cfa])
    core.write_fits(sm / "Stacked_62_IC 1396_20.0s_LP_20251125-190001.fit", rgb,
                    [("CREATOR", "ZWO Seestar S30", ""), ("FOCALLEN", 150.0, ""), ("XPIXSZ", 2.9, ""),
                     ("IMAGETYP", "Light", ""), ("STACKCNT", 62, ""), ("EXPTIME", 20.0, ""),
                     ("DATE-OBS", "2025-11-26T00:59:35.8", ""), ("FILTER", "LP", ""), ("INSTRUME", "Seestar S30", ""),
                     ("BAYERPAT", "GRBG", ""), ("TELESCOP", "S30_5031f5de", ""), ("OBJECT", "IC 1396", ""),
                     ("TOTALEXP", 1240.0, "")], bitpix=16)
    i_dw = _st.read_info(sm / "M 31_15s60_Astro_20251231-215912880_14C.fits")
    i_ve = _st.read_info(sm / "img-0001.fits")
    i_ss = _st.read_info(sm / "Stacked_62_IC 1396_20.0s_LP_20251125-190001.fit")
    check(i_dw.type == "light" and i_dw.smart == "DWARF mini" and i_dw.temp == 14 and i_dw.bayer == "RGGB",
          f"DWARF mini subs recognised ({i_dw.type}, {i_dw.smart}, {i_dw.temp} C)")
    check(i_ve.type == "light" and i_ve.camera == "Vespera Pro" and i_ve.exptime == 10,
          f"Vespera Pro subs recognised ({i_ve.type}, {i_ve.camera}, {i_ve.exptime} s)")
    check(i_ss.planes == 3 and i_ss.bayer is None and i_ss.exposure_total == 1240.0 and i_ss.smart == "Seestar S30",
          f"Seestar stack recognised as stacked colour ({i_ss.planes} planes, {i_ss.exposure_total:.0f} s)")
    import prepare as _P
    check(len(_P.find_masters([sm])) == 1, "a Seestar stack can go straight into Prepare")
    import finish as _F
    check(_F.dual_band_filter("LUlt") and _F.dual_band_filter("Optolong L-Synergy") and not _F.dual_band_filter("L-Pro")
          and core.band_lines("L-Synergy")[0] == "SII" and core.band_lines("L-Ultimate")[0] == "Ha",
          "dual-band filters recognised (L-Ultimate: Ha+OIII, L-Synergy: SII+OIII)")
    check(core.filter_from_name("Light_IC 1805_180.0s_Bin1_2600MC_SQA55Lult_0107.fit") == "L-Ultimate"
          and core.filter_from_name("Light_Ha_001.fits") is None and core.filter_from_name("Lullaby.fits") is None,
          "filter found in file names ('…SQA55Lult_0107.fit' -> L-Ultimate)")
    _pal = _P.palettes_for([_P.load_master(_P.find_masters([sm])[0])])
    check(_pal[0].startswith("HOO (dual-band)"), f"Seestar LP stack gets dual-band quick-look palettes ({_pal})")
    # faint, noisy subs with hot pixels (like 10 s Seestar subs) must still align
    def _faint(dx, dy, seed):
        r = _np.random.default_rng(seed)
        h, w = 600, 800
        yy, xx = _np.mgrid[0:h, 0:w]
        img = r.normal(2300, 60, (h, w)).astype(_np.float32)
        sr = _np.random.default_rng(7)
        for x, y, a in zip(sr.uniform(20, w - 20, 120), sr.uniform(20, h - 20, 120), sr.uniform(150, 900, 120)):
            img += a * _np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.6 ** 2))
        hy, hx = _np.random.default_rng(3).integers(0, h, 400), _np.random.default_rng(4).integers(0, w, 400)
        img[hy, hx] = 30000                                  # hot pixels: same place in every frame
        return img
    f1, f2 = _faint(0, 0, 1), _faint(13.4, -7.2, 2)
    l1, fac = _st.luminance(f1, True)
    l2, _f = _st.luminance(f2, True)
    try:
        T, inl, rms, _ok, _j = _st.match_frames(_st.find_stars(l1, fac, robust=True)[0],
                                                _st.find_stars(l2, fac, robust=True)[0])
        shift_ok = abs(T[0, 2] + 13.4) < 0.6 and abs(T[1, 2] - 7.2) < 0.6
    except ValueError as e:
        T, inl, shift_ok = None, 0, False
    # alt-az field rotation: frames turning through 60 degrees (a long alt-az session)
    rot_dir = sm / "altaz"
    rot_dir.mkdir()
    rr = _np.random.default_rng(11)
    sx, sy, sa = rr.uniform(-170, 170, 140), rr.uniform(-170, 170, 140), rr.uniform(300, 3000, 140)
    yy, xx = _np.mgrid[0:400, 0:400]
    for k, ang in enumerate(_np.radians([0, 12, 24, 36, 48, 60])):
        img = _np.random.default_rng(20 + k).normal(1000, 15, (400, 400)).astype(_np.float32)
        ca, sa_ = _np.cos(ang), _np.sin(ang)
        for x, y, a in zip(sx, sy, sa):
            px, py = 200 + ca * x - sa_ * y, 200 + sa_ * x + ca * y
            img += a * _np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.5 ** 2))
        core.write_fits(rot_dir / f"rot_{k}.fits", _np.clip(img, 0, 65535).astype(_np.uint16),
                        [("IMAGETYP", "Light", ""), ("OBJECT", "AltAz test", ""), ("EXPTIME", 10.0, ""),
                         ("INSTRUME", "Test Cam", ""), ("DATE-OBS", f"2026-01-01T02:{k:02d}:00", "")], bitpix=16)
    rg = _st.group_lights([_st.LightFrame(_st.read_info(f)) for f in sorted(rot_dir.glob("*.fits"))], _st.Rigs())[0]
    rg.plan = _st.CalPlan()
    try:
        rres = _st.stack_group(rg, _st.StackOptions(masters_root=str(sm / "m"), formats=("fits",)))
        rot_ok = rres.used == 6 and not rres.skipped and rres.autocrop is not None
        detail = f"{rres.used}/6 frames, auto-crop {rres.autocrop}"
    except Exception as e:  # noqa: BLE001
        rot_ok, detail = False, str(e)
    check(rot_ok, f"alt-az field rotation through 60 degrees stacks and auto-crops ({detail})")
    check(shift_ok, f"robust star finder aligns faint subs with hot pixels ({inl} stars, "
                    f"shift {None if T is None else (round(float(T[0, 2]), 2), round(float(T[1, 2]), 2))})")

    w = fb.MainWindow()
    w.resize(1600, 950)
    import ui_header as _uh
    check(getattr(w, "app_header", None) is not None and w.app_header.page.text() == _uh.PAGE_CULL
          and w.app_header.name.text() == "SubZeroAstro" and len(w.menuBar().actions()) >= 5,
          "banner 'SubZeroAstro / Astro Image Quality and Cull' above working menus")
    _mp = _uh.manual_path()
    _help = [x.text() for x in w.menuBar().actions() if x.text().replace("&", "") == "Help"]
    _hm = next((x.menu() for x in w.menuBar().actions() if x.text().replace("&", "") == "Help"), None)
    check(_mp is not None and _mp.stat().st_size > 100000 and _hm is not None
          and any("User Manual" in x.text() for x in _hm.actions()),
          f"Help > User Manual (PDF) is there and the manual ships with the app ({_mp})")
    check(_uh.app_icon_path() is not None and not QApplication.windowIcon().isNull(),
          f"the SubZeroAstro icon ships with the app and is set on the windows ({_uh.app_icon_path()})")
    w.show()
    for k, a in w.mode_actions.items():
        a.setChecked(k == "fast")
    check(w.fills_screen() in (True, False) and w.settings.value("window/fill", True, type=bool) in (True, False),
          "the main window knows whether it fills the screen (it opens filled next time)")
    _sub_old = w.settings.value("load/subfolders")
    w.settings.remove("load/subfolders")                  # a new user
    w._restore_settings()
    check(w.chk_sub.isChecked(), "'Include subfolders' is on by default")
    if _sub_old is not None:
        w.settings.setValue("load/subfolders", _sub_old)
    w.chk_sub.setChecked(False)
    state = {"t": time.time()}

    def shot(name, widget=None):
        QApplication.processEvents()
        pm = (widget or w).grab()
        ok = pm.save(str(out_dir / name))
        log(f"screenshot {name}: {'saved' if ok else 'FAILED'} ({pm.width()}x{pm.height()})")

    def start():
        # the intro animation: frames at set moments, and it closes by itself
        import splash as _sp
        sp = _sp.Splash("SubZeroAstro", "Brian A. Poteet", "From subs to masters - zero hassle.", "0.0 test")
        sp.show()
        for t_ in (0.5, 1.0, 1.5, 1.9, 2.2, 2.6, 3.0, 3.5, 4.0):
            sp.set_time(t_)
            QApplication.processEvents()
            pm = sp.grab()
            pm.save(str(out_dir / f"0_splash_{t_:.2f}s.png"))
        sp.close()
        sp2 = _sp.Splash("SubZeroAstro", "Brian A. Poteet", "From subs to masters - zero hassle.")
        state["splash_done"] = False
        sp2.finished.connect(lambda: state.__setitem__("splash_done", True))
        sp2.start()
        sp2.app_ready()
        t_end = time.time() + 8
        while not state["splash_done"] and time.time() < t_end:
            QApplication.processEvents()
            time.sleep(0.01)
        check(state["splash_done"] and not sp2.isVisible(), "intro animation plays and closes by itself (~4.5 s)")
        w.open_folder(session, use_cache=False)
        QTimer.singleShot(250, wait_loaded)

    def wait_loaded():
        if w.loaded_count >= len(w.frames) and len(w.frames) > 0:
            log(f"Measured {len(w.frames)} frames in {time.time() - state['t']:.1f}s")
            QTimer.singleShot(400, after_load)
        elif time.time() - state["t"] > 240:
            check(False, "frames measured within 4 minutes")
            finish(1)
        else:
            QTimer.singleShot(250, wait_loaded)

    def after_load():
        w.refresh_stats()
        frames = {f.name: f for f in w.frames.values()}
        check(len(frames) == 16, "16 frames listed")
        check(not any(f.error for f in frames.values()), "no unreadable frames")
        for name, tag in bad.items():
            f = frames[name]
            check(f.badge == "red", f"{tag} frame {name} has a red badge (got {f.badge}: {'; '.join(f.reasons)})")
        goods = [f for n, f in frames.items() if n not in bad]
        n_green = sum(f.badge == "green" for f in goods)
        check(n_green >= len(goods) - 1, f"good frames green ({n_green}/{len(goods)})")
        fw = [f.metrics["fwhm"] for n, f in frames.items() if n not in bad]
        check(all(abs(x - 3.0) < 0.15 for x in fw), f"FWHM of good frames ≈ 3.0 px (got {min(fw):.2f}–{max(fw):.2f})")
        check(len(w.chart.points) == 16, "chart has 16 points")
        check(w.list.count() == 16, "sidebar lists 16 frames")
        # show the first frame with the chart on FWHM
        w.chart_metric.setCurrentIndex(w.chart_metric.findData("fwhm"))
        w.list.setCurrentRow(0)
        QTimer.singleShot(1500, shots_1)

    def shots_1():
        shot("1_main_window_fwhm.png")
        # --- optional helper programs: banner on page 1 + Your Tools window --------------
        import tool_check as _tc
        import os as _os
        _fk = Path(__file__).resolve().parent / "tests" / "fake_tools"
        _none = {t[2]: str(session.parent / "no_such_tool.exe") for t in _tc.TOOLS}
        _miss = _tc.check(_none)
        check(not any(r["ok"] for r in _miss) and all("not found" in r["detail"] for r in _miss),
              "missing helper programs are each reported with what happens without them")
        bn = w.tools_banner
        _old_off = w.settings.value(_tc.BANNER_OFF, "")
        w.settings.setValue(_tc.BANNER_OFF, "")
        bn.show_results(_miss)
        check(not bn.isHidden() and "RC-Astro" in bn.label.text() and "cull and stack" in bn.label.text(),
              "page 1 shows an amber banner when helper programs are missing")
        shot("1a_tools_banner.png")
        bn.dismiss()
        bn.show_results(_miss)
        check(bn.isHidden(), "'Don't Remind Me' hides the banner while nothing changes")
        if _fk.is_dir() and not getattr(sys, "frozen", False):
            _paths = {"prep/astap": str(_fk / "fake_astap.py"), "prep/graxpert": str(_fk / "fake_graxpert.py"),
                      "prep/rcastro": str(_fk / "fake_rcastro.py")}
            _env = _os.environ.get("FAKE_RC_UNLICENSED")
            _os.environ["FAKE_RC_UNLICENSED"] = "sxt"
            _some = _tc.check(_paths)
            _rc = next(r for r in _some if r["key"] == "rcastro")
            check(_rc["problem"] == "not licensed" and "StarXTerminator" in _rc["detail"],
                  "an installed but unlicensed RC-Astro tool is flagged on page 1")
            bn.show_results(_some)
            check(not bn.isHidden(), "a new problem brings the banner back after 'Don't Remind Me'")
            if _env is None:
                _os.environ.pop("FAKE_RC_UNLICENSED", None)
            else:
                _os.environ["FAKE_RC_UNLICENSED"] = _env
            _all = _tc.check(_paths)
            check(all(r["ok"] and not r["problem"] for r in _all), "all helper programs found and ready")
            bn.show_results(_all)
            check(bn.isHidden(), "no banner when every helper program is ready")
        _old = {k: w.settings.value(k, "") for k in _none}
        for k, v in _none.items():
            w.settings.setValue(k, v)
        dlg = _tc.ToolsDialog(w.settings, w, _miss)
        dlg.show()
        dlg.checker.wait(30000)
        QApplication.processEvents()
        check(all("not found" in dlg.rows[k][0].text() for k in dlg.rows) and not dlg.b_lic.isEnabled(),
              "Your Tools window lists each helper with its status, Browse… and Download Page")
        shot("1b_your_tools.png", dlg)
        dlg.accept()
        for k, v in _old.items():
            w.settings.setValue(k, v)
        w.settings.setValue(_tc.BANNER_OFF, _old_off)
        bn.hide()
        fits_small(w, "page 1 (Quality and Cull)")
        _g = w.geometry()
        w.setGeometry(_g.x() - 500, _g.y() - 400, 4000, 3000)
        import ui_header as _uh1
        _uh1.fit_to_screen(w)
        _av = (w.screen() or QApplication.primaryScreen()).availableGeometry()
        check(_av.contains(w.geometry()), f"a page never hangs off the screen "
                                          f"(window {w.geometry().getRect()}, screen {_av.getRect()})")
        QApplication.processEvents()
        _sb = w.statusBar()
        check(_sb.isVisible() and _av.contains(_sb.mapToGlobal(_sb.rect().bottomLeft())),
              "page 1 status line (frame counts) is on screen")
        w.setGeometry(_g)                    # back to the screenshot size
        check("good" in w.kept_label.text() and "poor" in w.kept_label.text(),
              "good / marginal / poor counts shown in the Kept frames box")
        from PySide6.QtWidgets import QMenu as _QMenu
        check(any("Your Tools" in a.text() for mm in w.findChildren(_QMenu) for a in mm.actions()),
              "Settings menu has 'Your Tools…'")
        w.chart_metric.setCurrentIndex(w.chart_metric.findData("score"))
        w.sort_combo.setCurrentText("Quality score")
        w.worst_first.setChecked(True)
        QTimer.singleShot(600, shots_2)

    def shots_2():
        top = [w.frames[u].name for u in w.order[:3]]
        check(set(top) == set(bad), f"worst-first sort puts the 3 bad frames on top ({top})")
        shot("2_sorted_worst_first_score_chart.png")
        # Auto-Flag dialog (shown non-modally for the screenshot)
        dlg = fb.AutoFlagDialog(w, list(w.frames.values()), w.medians, w.settings)
        hits = {w.frames[u].name for u in dlg.matches()}
        check(set(bad) <= hits, f"Auto-Flag defaults catch all bad frames ({sorted(hits)})")
        dlg.show()
        QTimer.singleShot(400, lambda: shots_3(dlg))

    def shots_3(dlg):
        shot("3_autoflag_dialog.png", dlg)
        uids = dlg.matches()
        dlg.close()
        w.set_flags(uids, True)
        w.show_combo.setCurrentText("Flagged")
        check(w.list.count() == len(uids), "Flagged view lists the flagged frames")
        # reject the flagged ones (without the confirmation box)
        done = w.apply_reject(uids, True)
        w.undo_stack.append(done)
        moved = sorted(p.name for p in (session / "REJECTED").iterdir())
        check(len(moved) == len(uids), f"flagged frames moved to REJECTED ({len(moved)})")
        w.show_combo.setCurrentIndex(0)
        w.chart_metric.setCurrentIndex(w.chart_metric.findData("eccentricity"))
        QTimer.singleShot(800, shots_4)

    def shots_4():
        shot("4_after_reject_eccentricity_chart.png")
        w.undo()
        check(not (session / "REJECTED").exists(), "undo restored all files and removed REJECTED")
        w.save_disk_cache()
        cache_files = list(core.cache_dir().glob("*.cache"))
        check(len(cache_files) >= 1, "results cache written")
        # zoom to 100% -> full-resolution request path
        w.view.actual_size()
        QTimer.singleShot(1200, shots_5)

    def shots_5():
        shot("5_zoom_100.png")
        # --- 0.8.3: kept-frames summary, header window, filter name
        kept_txt = w.kept_label.text()
        check("of 16 frames" in kept_txt and "FWHM" in kept_txt, f"kept-frames summary shows live averages "
                                                                 f"({kept_txt[:60]}…)")
        check(w.btn_next.isEnabled() and 'Calibrate and Stack' in w.btn_next.text(), "next step 'Calibrate and Stack ➜' in the banner")
        w.show_header_window()
        hd = getattr(w, "header_dialog", None)
        check(hd is not None and hd.table.rowCount() >= 10 and "NAXIS1" in hd.as_text(),
              f"FITS header window lists the header ({hd.table.rowCount() if hd else 0} cards)")
        if hd:
            hd.search.setText("EXPTIME")
            check(hd.table.rowCount() == 1, "header window search finds a keyword")
            hd.close()
        all_uids = list(w.frames)
        w.set_filter(all_uids, "L-Ultimate")
        check(all(f.info.get("filter") == "L-Ultimate" for f in w.frames.values()) and "L-Ultimate" in w.known_filters,
              "Set Filter Name relabels the frames")
        w.set_filter(all_uids, None)
        check(not any(f.filter_override for f in w.frames.values()), "filter name can be set back to the recorded one")
        w.act_show_image.setChecked(False)
        check(not w.view.isVisible(), "the image can be hidden")
        w.act_show_image.setChecked(True)
        w.view.fit()
        # --- auto-cull presets
        for key in ("less", "semi", "aggressive"):
            w.apply_preset(key)
            names = {f.name for f in w.frames.values() if f.flagged}
            check(set(bad) <= names, f"{key} preset flags all bad frames ({len(names)} flagged)")
            check(not (names - set(bad)) or key == "aggressive",
                  f"{key} preset flags no good frames ({sorted(names - set(bad))})")
        agg = {f.name for f in w.frames.values() if f.flagged}
        w.apply_preset("less")
        w.undo()
        check({f.name for f in w.frames.values() if f.flagged} == agg, "one Ctrl+Z undoes a whole Auto-cull preset")
        w.apply_preset("semi")
        cdlg = fb.CheckpointDialog(w, w.settings)
        check("Currently using" in cdlg.size_label.text(), "Saved Progress dialog shows the space used")
        cdlg.close()
        w.save_session()
        state["sess_flags"] = {f.name for f in w.frames.values() if f.flagged}
        QTimer.singleShot(600, shots_6)

    def shots_6():
        shot("6_semi_aggressive_flagged.png")
        # --- Load Data window with two folders
        second = session.parent / "M33_night2"
        second.mkdir()
        make_session(second)
        for fpath in list(second.glob("*.fits")):             # a different night: different file names
            fpath.rename(fpath.with_name("N2_" + fpath.name))
        state["second"] = second
        dlg = fb.LoadDataDialog(w, w.settings, False)
        dlg.add_source(("folder", session, False))
        dlg.add_source(("folder", second, False))
        dlg.add_source(("folder", second, False))           # duplicate is ignored
        check(len(dlg.sources()) == 2, "Load Data lists 2 folders (duplicate ignored)")
        _pk = fb.folder_picker(dlg, "test", session.parent)
        from PySide6.QtWidgets import QAbstractItemView as _QAIV, QListView as _QLV, QTreeView as _QTV
        _views = _pk.findChildren(_QLV) + _pk.findChildren(_QTV)
        _multi = [v for v in _views if v.selectionMode() == _QAIV.ExtendedSelection]
        check(len(_multi) >= 2, f"Add Folders… lets you pick several folders at once (Shift/Ctrl-click) "
                                f"({len(_multi)} of {len(_views)} views)")
        _pk.selectFile(f'"{session.name}" "{second.name}"')
        _pk.show()
        shot("7b_folder_picker.png", _pk)
        _got = fb.picked_folders(_pk)
        check(sorted(p.name for p in _got) == sorted([session.name, second.name]),
              f"several picked folders come back together ({[p.name for p in _got]})")
        _pk.deleteLater()
        _d2 = fb.LoadDataDialog(w, w.settings, True)
        _d2.add_folders([session, second])
        check(len(_d2.sources()) == 2 and all(s[2] for s in _d2.sources()),
              "picking two folders adds both, with their subfolders")
        _d2.deleteLater()
        dlg.filter_combo.setCurrentText("LUlt")
        check(dlg.filter_override() == "L-Ultimate", "Load Data: filter name typed as 'LUlt' becomes L-Ultimate")
        dlg.filter_combo.setCurrentIndex(0)
        check(dlg.filter_override() is None, "Load Data: '(as recorded)' leaves the files' filter")
        dlg.show()
        QTimer.singleShot(500, lambda: shots_7(dlg))

    def shots_7(dlg):
        shot("7_load_data_window.png", dlg)
        sources = dlg.sources()
        dlg.close()
        # --- Start Over, then process both folders together
        w.reset_session()
        check(not w.frames and w.list.count() == 0 and not w.chart.points, "Start Over clears the session")
        w.reopen_last_session()
        got = {f.name for f in w.frames.values() if f.flagged}
        check(got == state["sess_flags"] and len(got) > 0,
              f"Reopen Last Session restores folders and flags ({len(got)} flagged)")
        w.reset_session()
        state["t2"] = time.time()
        w.open_sources(sources + [("folder", session, False)])
        check(w.toast.isVisible() and "16 duplicate" in w.toast.text(),
              f"duplicate frames skipped with a notice ({w.toast.text()[:50]})")
        QTimer.singleShot(300, wait_two)

    def wait_two():
        if w.loaded_count >= len(w.frames) and len(w.frames) > 0:
            check(len(w.frames) == 32, f"two folders loaded together ({len(w.frames)} frames)")
            pass
            folders = {f.rel_folder for f in w.frames.values()}
            check(folders == {"M33", "M33_night2"}, f"frames keep their folder names ({sorted(folders)})")
            w.refresh_stats()
            QTimer.singleShot(800, shots_8)
        elif time.time() - state["t2"] > 240:
            check(False, "second session measured within 4 minutes")
            finish(1)
        else:
            QTimer.singleShot(250, wait_two)

    def shots_8():
        shot("8_two_folders.png")
        # --- Stage 3: stacking window --------------------------------------------
        import stack_window as sw
        cal = session.parent / "Calibration Frames"
        raw_cal = session.parent / "raw_cal"
        masters = session.parent / "Master Lights"
        rng = np.random.default_rng(3)
        for kind, n, exp in (("BIAS", 7, 0.0), ("DARK", 7, 300.0), ("FLAT", 7, 1.0)):
            d = raw_cal / kind.lower()
            d.mkdir(parents=True)
            for i in range(n):
                level = {"BIAS": 0.0, "DARK": 5.0, "FLAT": 20000.0}[kind]
                img = np.clip(level + rng.normal(0, 8, (900, 1350)), 0, 65535).astype(np.float32)
                hdr = {"IMAGETYP": kind, "EXPTIME": exp, "INSTRUME": "Test Camera", "FOCALLEN": 1140.0,
                       "DATE-OBS": f"2026-10-11T02:{i:02d}:00", "ROWORDER": "TOP-DOWN"}
                if kind == "FLAT":
                    hdr["FILTER"] = "Ha" if i % 2 else "OIII"
                core.write_fits(d / f"{kind.lower()}_{i}.fits", img, hdr, bitpix=16)
        w.settings.setValue("stack/cal_root", str(cal))
        w.settings.setValue("stack/masters_root", str(masters))
        w.settings.setValue("stack/xisf", True)
        cal.mkdir()
        w.open_stacking()
        win = w.stack_win
        state["win"] = win
        check(len(win.groups) == 2, f"stacking window groups lights by filter ({[g.label() for g in win.groups]})")
        check(all(len(g.frames) == 16 for g in win.groups), "each group has its 16 kept frames")
        check(all(any(k == "dark" for k, _ in g.plan.warnings) for g in win.groups),
              "missing darks are reported before calibration is added")
        win._add_cal_paths([raw_cal])
        kinds = sorted(c.kind for c in win.items)
        check(kinds.count("bias") == 1 and kinds.count("dark") == 1, f"calibration sets recognised ({kinds})")
        plans = [(g.filter, bool(g.plan.bias), g.plan.dark_mode, bool(g.plan.flat)) for g in win.groups]
        check(all(p[1] and p[2] == "exact" and p[3] for p in plans), f"bias, darks and flats matched ({plans})")
        win.resize(1300, 860)
        QTimer.singleShot(700, shots_9)

    def shots_9():
        win = state["win"]
        shot("9_stacking_window.png", win)
        state["t3"] = time.time()
        win.start(skip_preflight=True)
        QTimer.singleShot(500, wait_stack)

    def wait_stack():
        win = state["win"]
        if win.worker is not None and win.worker.isFinished():
            QTimer.singleShot(500, after_stack)
        elif time.time() - state["t3"] > 300:
            check(False, "stacking finished within 5 minutes")
            finish(1)
        else:
            QTimer.singleShot(500, wait_stack)

    def after_stack():
        win = state["win"]
        res = win.worker.results
        log(f"Stacked {len(res)} groups in {time.time() - state['t3']:.1f}s")
        check(len(res) == 2 and not any(r.error for r in res), f"both groups stacked ({[r.error for r in res]})")
        masters = session.parent / "Master Lights" / "M33"
        files = sorted(p.name for p in masters.iterdir()) if masters.is_dir() else []
        log("master lights: " + ", ".join(files))
        check(any(f.endswith("_Ha_Master_2026-10-10.fits") for f in files), "Ha master saved as FITS")
        check(any(f.endswith("_OIII_Master_2026-10-10.xisf") for f in files), "OIII master saved as XISF")
        check(any(f.endswith("_stacklog.txt") for f in files), "stack log saved next to the master")
        cal = session.parent / "Calibration Frames"
        lib = sorted(str(p.relative_to(cal)) for p in cal.rglob("*.fits"))
        log("calibration library: " + "; ".join(lib))
        check(any("Masters" in x and "Bias" in x for x in lib) and any("Darks" in x for x in lib)
              and any("Flats" in x and "Ha" in x for x in lib), "calibration masters saved into the library")
        d, _c, look = core.read_fits(masters / next(f for f in files if f.endswith("_Ha_Master_2026-10-10.fits")))
        check(look.get("NCOMBINE") == 16 and look.get("IMAGETYP") == "Master Light",
              f"master header records 16 frames ({look.get('NCOMBINE')}, {look.get('IMAGETYP')})")
        check(look.get("PIXRANGE") == 65535.0 and float(d.max()) <= 1.0,
              f"master saved in PixInsight's 0-1 range (max {float(d.max()):.3f})")
        check(bool(look.get("ALIGNREF")) and look.get("FOCALLEN") == 1140.0,
              f"master records its common reference and plate-solve hints ({look.get('ALIGNREF')})")
        import astrobin as AB
        ses = AB.sessions_from_header(look)
        check(sum(x.number for x in ses) == 16 and all(x.duration == 300 for x in ses) and ses
              and ses[0].cal.get("flats", 0) > 0 and ses[0].filter == "Ha",
              f"master records each night's acquisition for AstroBin "
              f"({[(x.date, x.number, x.duration, x.fwhm) for x in ses]}, {ses[0].cal if ses else None})")
        origins = {(g.filter, k): getattr(g.plan, k).origin for g in win.groups for k in ("bias", "dark", "flat")
                   if getattr(g.plan, k) is not None}
        check(not getattr(win, "last_cal_warnings", []),
              f"well-matched calibration frames give no calibration warnings ({getattr(win, 'last_cal_warnings', [])[:1]})")
        check(all(o == "library" for o in origins.values()),
              f"after stacking, the saved library masters are used next time ({origins})")
        QTimer.singleShot(600, shots_10)

    def shots_10():
        shot("10_stacking_done.png", state["win"])
        # --- Stage 4: prepare for PixInsight ---------------------------------------
        win = state["win"]
        check(win.btn_prepare.isEnabled(), "Continue to Image Processing button enabled after stacking")
        import checkpoints as cpm
        left = [p.name for p in cpm.root().glob("stack*_*.json")]
        check(win.worker.completed and not left, f"a finished stacking run leaves no resume notes ({left})")
        _cbd = win.build_cal_masters(folder=str(session.parent / "ScopeXCalis"), modal=False)
        if _cbd.work is not None:
            _cbd.work.wait(60000)
        for _k in range(20):
            QApplication.processEvents()
        _cbn = len(_cbd.jobs)
        _cbd.reject()
        check(win.btn_build_masters.isVisible() and _cbn == 6 and _cbd.rig.currentText() == "ScopeX",
              f"page 2 has Build Calibration Masters; its window found {_cbn} masters to build in a folder")
        _cbd1 = w.open_cal_builder(folder=str(session.parent / "ScopeXCalis"), modal=False)
        if _cbd1.work is not None:
            _cbd1.work.wait(60000)
        for _k in range(20):
            QApplication.processEvents()
        _cbn1 = len(_cbd1.jobs)
        _cbd1.reject()
        check(w.act_cal_builder is not None and _cbn1 == 6,
              f"page 1 Stack menu opens Build Calibration Masters without lights ({_cbn1} masters found)")
        fake = Path(__file__).resolve().parent / "tests" / "fake_tools"
        state["fake"] = fake.is_dir() and not getattr(sys, "frozen", False)
        if state["fake"]:
            win.settings.setValue("prep/astap", str(fake / "fake_astap.py"))
            win.settings.setValue("prep/graxpert", str(fake / "fake_graxpert.py"))
            win.settings.setValue("prep/rcastro", str(fake / "fake_rcastro.py"))
        else:
            for k in ("prep/astap", "prep/graxpert", "prep/rcastro"):
                win.settings.setValue(k, str(session.parent / "no_such_tool.exe"))
        win.settings.setValue("prep/xt_sxt", '{"enabled": true, "values": {"stars": true}}')
        win.settings.setValue("prep/fits", True)
        win.settings.setValue("prep/device", "")
        win.open_prepare()
        pw = win.prep_win
        state["pw"] = pw
        check(len(pw.sets) == 1 and len(pw.sets[0]) == 2, f"prepare window loads the 2 masters as one target "
                                                           f"({[[m.filter for m in s] for s in pw.sets]})")
        if state["fake"]:
            check("1.0.3-fake" in pw.rc_status.text(), "RC-Astro CLI detected with version and licenses")
            check(pw.device.currentData() == "gpu1", f"dedicated NVIDIA GPU chosen by default "
                                                     f"({pw.device.currentData()})")
            nx = pw.panels.get("nxt")
            check(nx is not None and not nx.widgets["dn"].isHidden() and nx.widgets["di"].isHidden(),
                  "NoiseXTerminator controls generated from the CLI description")
            if nx:
                nx.widgets["csep"].setChecked(True)
                check(nx.widgets["dn"].isHidden() and not nx.widgets["di"].isHidden(),
                      "Intensity/Color Separation switches the NoiseXTerminator sliders")
                nx.widgets["csep"].setChecked(False)
            bx = pw.panels.get("bxt")
            check(bx is not None and not bx.widgets["nsd"].isEnabled(),
                  "BlurXTerminator PSF Diameter greyed out while Automatic PSF is on")
        else:
            check("not found" in pw.rc_status.text(), "missing RC-Astro CLI reported clearly")
        check(hasattr(pw, "btn_rc_lic") and pw.btn_rc_lic.isEnabled()
              and "Licences" in pw.btn_rc_lic.text(), "'RC-Astro Licences…' button on the RC-Astro CLI row")
        import tools as _tl
        check(_tl._clean_console("\x1b[1mHi\x1b[0m\r\nwait 10%\rwait 100%\n\n\n\nok") == "Hi\nwait 100%\n\nok",
              "rc-astro console messages shown without colour codes or progress redraws")
        pw.resize(1450, 900)
        QTimer.singleShot(900, rc_lic_1 if state["fake"] else shots_11)

    def _when_idle(dlg, then, t0=None):
        t0 = t0 or time.time()
        if dlg.busy() or dlg.before is None:
            if time.time() - t0 > 60:
                check(False, "RC-Astro licence window answered within a minute")
                QTimer.singleShot(10, rc_lic_end)
            else:
                QTimer.singleShot(150, lambda: _when_idle(dlg, then, t0))
        else:
            QTimer.singleShot(50, then)

    def rc_lic_1():
        pw = state["pw"]
        state["rc_env"] = {k: os.environ.get(k) for k in ("FAKE_RC_UNLICENSED", "FAKE_RC_STATE")}
        st_file = session.parent / "fake_rc_activated.txt"
        if st_file.exists():
            st_file.unlink()
        os.environ["FAKE_RC_UNLICENSED"] = "sxt"
        os.environ["FAKE_RC_STATE"] = str(st_file)
        pw.detect_tools()
        check(not pw.xt_boxes["sxt"].isEnabled(), "an unlicensed StarXTerminator is greyed out on the page")
        dlg = pw.rc_licences(show=False)
        state["dlg"] = dlg
        dlg.show()
        _when_idle(dlg, rc_lic_2)

    def rc_lic_2():
        dlg = state["dlg"]
        txt = dlg.status.text()
        check("StarXTerminator" in txt and "not licensed" in txt and "1.0.3-fake" in txt,
              "licence window lists each RC-Astro tool and its licence")
        shot("11a_rc_licences.png", dlg)
        dlg.code_edit.setText("BAD-CODE")
        dlg.activate()
        _when_idle(dlg, rc_lic_3)

    def rc_lic_3():
        dlg = state["dlg"]
        out = dlg.out.toPlainText()
        check("not valid" in out and "did not change" in out and not dlg.changed,
              "a refused activation code is reported and nothing changes")
        dlg.code_edit.setText("ABC-123")
        dlg.activate()
        _when_idle(dlg, rc_lic_4)

    def rc_lic_4():
        dlg, pw = state["dlg"], state["pw"]
        out = dlg.out.toPlainText()
        check("registered" in out and "Activated - 3 of 3" in out and dlg.changed and not dlg.code_edit.text(),
              "activation code pasted into 'rc-astro license' activates every tool")
        dlg.check_update()
        state["t_up"] = time.time()
        QTimer.singleShot(200, rc_lic_5)

    def rc_lic_5():
        dlg, pw = state["dlg"], state["pw"]
        if dlg.busy() and time.time() - state["t_up"] < 60:
            QTimer.singleShot(150, rc_lic_5)
            return
        check("up to date" in dlg.out.toPlainText(), "Check for Updates shows what rc-astro update says")
        dlg.accept()
        pw.detect_tools()
        check(pw.xt_boxes["sxt"].isEnabled() and "not licensed" not in pw.rc_status.text(),
              "after activating, the page re-reads the licences and StarXTerminator is available")
        QTimer.singleShot(10, rc_lic_end)

    def rc_lic_end():
        for k, v in state.get("rc_env", {}).items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        d = state.get("dlg")
        if d is not None and d.isVisible():
            d.reject()
        state["pw"].detect_tools()
        QTimer.singleShot(600, shots_11)

    def shots_11():
        pw = state["pw"]
        shot("11_prepare_window.png", pw)
        state["prep_out"] = session.parent / "Final Images"
        pw.out_edit.setText(str(state["prep_out"]))
        for cb, on in ((pw.fmt_xisf, True), (pw.fmt_fits, True), (pw.fmt_tif, True), (pw.fmt_jpg, True),
                       (pw.fmt_png, False)):
            cb.setChecked(on)
        state["t4"] = time.time()
        pw.start(show_dialog=False)
        QTimer.singleShot(500, wait_prepare)

    def wait_prepare():
        pw = state["pw"]
        if pw.worker is not None and pw.worker.isFinished():
            QTimer.singleShot(400, after_prepare)
        elif time.time() - state["t4"] > 300:
            check(False, "prepare finished within 5 minutes")
            finish(1)
        else:
            QTimer.singleShot(400, wait_prepare)

    def after_prepare():
        pw = state["pw"]
        res = getattr(pw, "last_results", []) or []
        check(len(res) == 1, f"prepare produced results ({len(res)})")
        check(getattr(pw.btn_start, "_done", False) and "Run Again" in pw.btn_start.text()
              and getattr(state["win"].btn_start, "_done", False),
              "after a run the page's run button turns amber ('Run Again?') on both pages")
        check("Ready for Processing Finishing Touches" in pw.progress.format(),
              f"page 3 says 'Ready for Processing Finishing Touches' when done ({pw.progress.format()})")
        import prepare_window as _pwm
        _rd = _pwm.ReadyDialog(pw, res)
        check("complete" in _rd.windowTitle().lower() and "Finishing Touches" in _rd.findChild(QLabel).text(),
              "page 3 completion message says the page is complete and what comes next")
        _rd.deleteLater()
        from prepare import target_out_dir as _tod
        check(str(res[0]["out_dir"]) in pw.out_where.text() and "Saved in" in pw.out_where.text(),
              f"page 3 shows the folder the images were saved in ({pw.out_where.text()[:90]})")
        check(_tod("C:/out/NGC891", "IC1805").name == "IC1805"
              and _tod("C:/out", "IC1805").name == "IC1805"
              and _tod("C:/out/IC1805", "IC1805").parent.name == "out",
              "custom output folder gets a per-target subfolder")
        # masters of one target stacked in separate runs (e.g. Ha one night, SII + OIII later)
        import prepare as _P2
        _r = np.random.default_rng(5)
        _yy, _xx = np.mgrid[:300, :400]
        _st = [(_r.uniform(15, 385), _r.uniform(15, 285), _r.uniform(800, 9000)) for _ in range(160)]

        def _img(dx, dy):
            a = np.full((300, 400), 900.0) + _r.normal(0, 8, (300, 400))
            for x, y, f in _st:
                a += f * np.exp(-((_yy - y - dy) ** 2 + (_xx - x - dx) ** 2) / (2 * 1.4 ** 2))
            return a.astype(np.float32)[None]
        _ms = [_P2.Master(Path(f"{n}.fits"), d, [], {}, "NGC 7000", n, ref, None)
               for n, d, ref in (("Ha", _img(0, 0), "runA"), ("SII", _img(9.4, -6.2), "runB"),
                                 ("OIII", _img(9.4, -6.2), "runB"))]
        _sets = _P2.target_sets(_ms)
        _ov = _P2.line_up(_ms)
        check(len(_sets) == 1 and _ov is not None and _ms[1].alignref == "runA"
              and float(np.abs(_ms[1].data[0, 20:-20, 20:-20] - _ms[0].data[0, 20:-20, 20:-20]).max()) < 2500,
              f"Ha + SII/OIII stacked in separate runs become one set and are lined up ({_ov})")
        _names = {"Ha 7nm": "Ha", "Antlia Ha 3nm Pro": "Ha", "Hα": "Ha", "OIII 6.5nm": "OIII", "O": "OIII",
                  "S": "SII", "SII 3nm": "SII", "S-II": "SII", "ZWO R": "R", "Lum": "L", "LUlt": "L-Ultimate",
                  "UV/IR": "UV/IR"}
        _names.update({"ASD1": "Askar D1", "Askar Colour Magic D2": "Askar D2", "H": "Ha"})
        _bad = {k: core.normalize_filter(k) for k, v in _names.items() if core.normalize_filter(k) != v}
        check(core.band_lines("ASD2") == ("SII", "OIII") and core.band_lines("ASD1") == ("Ha", "OIII"),
              "Askar D1 (Ha+OIII) and D2 (SII+OIII) are known dual-band filters")
        check(not _bad, f"filter-wheel names like 'Ha 7nm', 'O', 'S', 'Antlia Ha 3nm Pro' are recognised ({_bad})")
        import stacking as _stk
        _rg = np.random.default_rng(3)
        _sky = np.c_[_rg.uniform(0, 4000, 900), _rg.uniform(0, 3000, 900)]
        _fl = _rg.pareto(1.6, 900) + 1
        _o = np.argsort(-_fl)[:150]
        _ref = _sky[_o] + _rg.normal(0, 0.3, (150, 2))
        _mir = np.c_[4000 - _sky[_o, 0], _sky[_o, 1]] + _rg.normal(0, 0.3, (150, 2))
        try:
            _T, _n, _rms, *_r = _stk.match_frames(_ref, _mir)
            _okm = _stk.is_mirrored(_T) and _rms < 1.0 and _n > 100
        except ValueError:
            _okm = False
        check(_okm, "a frame saved as a mirror image (other capture setting / row order) is still lined up")
        import tools as _tl
        check(_tl._json_answer('Update available\n{"products": []}') == {"products": []}
              and _tl._json_answer("usage: something") is None,
              "a tool's JSON answer is found even with notices around it; no answer is handled, not a crash")
        import prepare_window as _pwm
        check(_pwm.xt_start_values("nxt", {"dn": 0.0})["dn"] == 0.6 and _pwm.xt_start_values("nxt", {})["dn"] == 0.6
              and _pwm.xt_start_values("nxt", {"dn": 0.8})["dn"] == 0.8,
              "NoiseXTerminator never starts at 0 (typical 0.6), a chosen value is kept")
        if res:
            names = sorted(Path(f).name for f in res[0]["files"])
            log("prepared: " + ", ".join(names))
            check(sum(n.endswith("_prepared.xisf") for n in names) == 2, "two prepared XISF masters")
            check(any(n.endswith("_quicklook.jpg") for n in names) and any(n.endswith("_quicklook.tif")
                                                                           for n in names),
                  "quick-look JPG and 16-bit TIFF saved")
            check(any(n.endswith("PixInsight_next_steps.html") for n in names), "PixInsight plan saved")
            check(not any(n.endswith(".png") for n in names), "unticked PNG format not saved")
            check(all(Path(f).parent == state["prep_out"] / "M33" for f in res[0]["files"]),
                  f"final images saved to the chosen folder, in a per-target subfolder "
                  f"({state['prep_out'].name}\\M33)")
            prepared = next(Path(f) for f in res[0]["files"] if str(f).endswith("_prepared.fits"))
            _d, _c, lk = core.read_fits(prepared)
            if state["fake"]:
                check(lk.get("PLTSOLVD") is True, "prepared master is plate solved (ASTAP)")
                check(any(n.endswith("_prepared_stars.xisf") for n in names), "StarXTerminator stars image saved")
                logtxt = pw.log_view.toPlainText()
                check("returned the picture upside down — turned it back" in logtxt and lk.get("STARLESS") is True,
                      "a tool that hands the picture back upside down is turned back; unannounced stars "
                      "image picked up")
                import prepare as _P
                stars_p = prepared.with_name(prepared.stem + "_stars.fits")
                sd_, _c2, _lk2 = core.read_fits(stars_p)
                filt = lk.get("FILTER")
                orig = next(p for p in sorted((session.parent / "Master Lights" / "M33").iterdir())
                            if p.suffix in (".fits", ".xisf") and "_prepared" not in p.name
                            and _P.st.ImageSource(p).look.get("FILTER") == filt)
                od = _P.st.ImageSource(orig).read()
                pd = _d + sd_
                hh, ww = pd.shape[-2:]
                m_ = re.search(r"x (\d+)-\d+, y (\d+)-\d+", " ".join(f"{v} {c}" for _k, v, c in _c))
                x0, y0 = (int(m_.group(1)), int(m_.group(2))) if m_ else (0, 0)
                check(_P.orientation_of(pd, od[..., y0:y0 + hh, x0:x0 + ww]) == "same",
                      f"prepared master faces the same way as the stacked master (crop at {x0},{y0})")
                done = [s["title"] for s in res[0]["plan"] if s["done"]]
                check("Sharpen with BlurXTerminator" in done and "Remove gradients" in done,
                      f"plan ticks the steps done here ({done})")
            else:
                done = [s["title"] for s in res[0]["plan"] if s["done"]]
                check("Remove gradients" in done and "Plate solve" not in done,
                      f"without tools: built-in gradient removal, plate solve left for PixInsight ({done})")
            import prepare_window as pwm
            dlg = pwm.ReadyDialog(pw, res)
            dlg.show()
            state["dlg"] = dlg
        QTimer.singleShot(800, shots_12)

    def shots_12():
        shot("12_prepare_done.png", state["pw"])
        if state.get("dlg"):
            shot("13_ready_for_pixinsight.png", state["dlg"])
            state["dlg"].close()
        # 0.6: step history, go back & redo, resume
        import prepare_window as pwm
        pw = state["pw"]
        n_steps = pw.hist_combo.count() - 1
        expect = 5 if state["fake"] else 2          # crop, solve, gradient, BXT, SXT (NXT off) | crop + gradient
        check(n_steps == expect, f"step history lists each finished step ({n_steps}, expected {expect})")
        check(pw.view.locked, "after Prepare the view shows the result of the last step")
        pw.hist_combo.setCurrentIndex(2)
        check(pw.btn_redo.isEnabled() and pw.btn_before.isEnabled(), "Before and Go Back buttons available")
        shot("14_step_history.png", pw)
        orig_q = pwm.QMessageBox.question
        pwm.QMessageBox.question = staticmethod(lambda *a, **k: pwm.QMessageBox.Yes)
        try:
            pw.redo_from_step()
        finally:
            pwm.QMessageBox.question = orig_q
        check(len(pw.cps.steps) == 1, f"Go Back & Redo keeps only the earlier steps ({len(pw.cps.steps)})")
        state["t4"] = time.time()
        pw.start(show_dialog=False)
        QTimer.singleShot(500, wait_prepare2)

    def wait_prepare2():
        pw = state["pw"]
        if pw.worker is not None and pw.worker.isFinished():
            QTimer.singleShot(400, after_prepare2)
        elif time.time() - state["t4"] > 300:
            check(False, "second prepare finished within 5 minutes")
            finish(1)
        else:
            QTimer.singleShot(400, wait_prepare2)

    def after_prepare2():
        pw = state["pw"]
        res = getattr(pw, "last_results", []) or []
        check(len(res) == 1 and res[0].get("resumed") == 1,
              f"re-run resumed from saved progress ({res[0].get('resumed') if res else None} step kept)")
        expect = 5 if state["fake"] else 2
        check(pw.hist_combo.count() - 1 == expect, "history complete again after the redo")
        QTimer.singleShot(300, finish_stage)

    def finish_stage():
        # --- Stage 5: Finish the Image ------------------------------------------
        import prepare_window as pwm
        import finish_window as fwm
        import finish as F
        import tools as _tools
        _tools.reveal = lambda *a, **k: True                # don't open Explorer on the test machine
        pw = state["pw"]
        files = pwm.prepared_masters(pw.last_results)
        check(len(files) == 2, f"prepared masters found for Finish ({[f.name for f in files]})")
        fw = fwm.FinishWindow(pw, pw.settings)
        fw.synchronous = True
        fw.resize(1500, 950)
        fw.show()
        fw.load_masters(files)
        st0 = fw.state
        check(st0 is not None and st0.kind == "mono_nb", f"Finish loads the Ha/OIII masters as narrowband "
                                                          f"({None if st0 is None else st0.kind})")
        if state["fake"]:
            check(bool(st0.channel_stars), "stars separated by Prepare are picked up (per filter)")
        nxt = F.next_step(fw.state)
        check(nxt is not None and nxt.key == "combine", f"first recommendation is to combine the filters "
                                                        f"({nxt.key if nxt else None})")
        state["fw"] = fw
        fits_small(state["pw"].parent(), "page 2 (Calibrate and Stack)")
        fits_small(pw, "page 3 (Astro Image Processing)")
        fits_small(fw, "page 4 (Finishing Touches)")
        import ui_header as _uh
        check(fw.app_header.page.text() == _uh.PAGE_FINISH and pw.app_header.page.text() == _uh.PAGE_PREPARE
              and state["pw"].parent().app_header.page.text() == _uh.PAGE_STACK,
              "every window shows the banner with its page title")
        check(all(fb.APP_VERSION in x.app_header.by.text() for x in (fw, pw, state["pw"].parent(), w)),
              f"every page shows the version in small print ({w.app_header.by.text()})")
        sw_ = state["pw"].parent()
        check(fw.app_header.btn_back is not None and _uh.PAGE_PREPARE in fw.app_header.btn_back.text()
              and pw.app_header.btn_back is not None and _uh.PAGE_STACK in pw.app_header.btn_back.text()
              and sw_.app_header.btn_back is not None and _uh.PAGE_CULL in sw_.app_header.btn_back.text(),
              "each page has '◀ previous page' at the top left")
        check(pw.app_header.right.indexOf(pw.btn_finish) >= 0 and sw_.app_header.right.indexOf(sw_.btn_prepare) >= 0,
              "next-step buttons sit at the top right")
        pw.app_header.btn_back.click()
        t_end = time.time() + 1.5
        while time.time() < t_end and pw.isVisible():
            QApplication.processEvents()
            time.sleep(0.02)
        check(sw_.isVisible() and not pw.isVisible(),
              "◀ back steps this page aside and shows the previous page")
        sw_.open_prepare()
        QApplication.processEvents()
        check(sw_.prep_win is pw and pw.isVisible(),
              "Next brings the same page back as it was (input unchanged)")
        fw.raise_()
        QApplication.processEvents()
        shot("15_finish_start.png", fw)
        fw.open_tool("combine")
        check(len(fw.variants) >= 2 and len(fw.group.buttons()) == len(fw.variants),
              f"palette choices shown as pictures ({[v.label for v in fw.variants]})")
        QTimer.singleShot(500, finish_steps)

    def finish_steps():
        fw = state["fw"]
        import finish as F
        shot("16_finish_palette_choices.png", fw)
        done = []
        for _ in range(16):
            t = F.next_step(fw.state)
            if t is None:
                break
            if t.key == "stretch" and not state.get("dust_ui"):
                state["dust_ui"] = True
                fw.open_tool("dust")
                QApplication.processEvents()
                check(fw.tool is not None and fw.tool.key == "dust" and fw.view.on_click is not None
                      and "donut" in fw.choice_note.text().lower(),
                      f"Mote Buster opens with clickable circles ({fw.choice_note.text()[:70]})")
                check(fw.tool.title.startswith("Mote Buster"), "the dust donut tool is called Mote Buster")
                fw._add_spot(fw.state.shape[1] * 0.5, fw.state.shape[0] * 0.5, 40.0)
                fw.run_preview()
                check(len(fw.spots) >= 1 and len(fw.spot_items) == len(fw.spots),
                      "a circle can be added for a donut SubZeroAstro did not find")
                fw.close_tool()
            fw.open_tool(t.key)
            if t.key == "stretch":
                shot("17_finish_brighten_choices.png", fw)
                check(len(fw.variants) == 5, "Brighten offers 5 strengths to pick from")
                n0 = len(fw.project.steps)
                check("not applied" in fw.preview_hint.text() and fw.state.linear,
                      "clicking a version previews it big without applying it")
                sl, sp = fw.sliders["brightness"]
                sl.setValue(min(sl.maximum(), sl.value() + 12))
                check(fw.fine_box.isChecked() and "fine-tuned" in fw.preview_hint.text(),
                      "moving a fine-tune slider updates the preview")
                shot("19_finish_fine_tune.png", fw)
                fw.cancel_tool()
                check(len(fw.project.steps) == n0 and fw.state.linear and fw.tool is None,
                      "Cancel leaves the picture exactly as it was")
                fw.open_tool("stretch")
                sl, sp = fw.sliders["protect"]
                sl.setValue(sl.maximum() // 2)
            if t.key == "curves":
                check(fw.curve.isVisible() or fw.curve.isVisibleTo(fw), "Curves shows a curve you can drag")
                fw.curve.points[2] = [0.5, 0.58]
                fw._curve_dragged([list(q) for q in fw.curve.points])
            fw.apply_choice()
            done.append(t.key)
        st_ = fw.state
        check(F.next_step(st_) is None and not st_.linear, f"recommended workflow runs to a finished picture "
                                                           f"({done})")
        if state["fake"]:                                  # stars were separated in Prepare (fake StarX)
            check(done.index("stars_out") == done.index("combine") + 1 if "stars_out" in done and "combine" in done
                  else False, f"stars from Prepare: 'Take Stars Out' is offered as a real step, not pre-ticked "
                              f"({done})")
            check("star_color" in done and done.index("star_color") < done.index("stars_back"),
                  "narrowband: Star Colours is recommended before the stars go back")
        check(any("(fine-tuned)" in x["label"] for x in fw.project.steps), "fine-tuned values are applied")
        n_before = len(fw.project.steps)
        for key in ("core", "local", "spotfix"):            # the optional tools
            fw.open_tool(key)
            if key == "core":
                fw.chk_mask.setChecked(True)
                check(fw.chk_mask.isVisibleTo(fw) and "not applied" in fw.preview_hint.text(),
                      "Tame Bright Core previews with 'Show what will change'")
                shot("20_finish_tame_core_mask.png", fw)
            if key == "spotfix":
                fw._spot_drag(120.0, 90.0, "press")
                fw._spot_drag(120.0, 90.0, "release")
                _r0 = fw.spots[0][2]
                fw._spot_wheel(3)                               # resize the circle just placed
                _r1 = fw.spots[0][2]
                _rect = fw.spot_items[0].rect()
                fw._spot_drag(200.0, 150.0, "press")
                fw._spot_drag(200.0, 150.0, "release")
                _c1 = list(fw.spots[0][:2])
                fw._spot_drag(120.0, 90.0, "press")             # pick the first again and drag it
                fw._spot_drag(130.0, 95.0, "move")
                fw._spot_drag(130.0, 95.0, "release")
                _f = getattr(fw, "disp_factor", 1.0)
                _moved = abs(fw.spots[0][0] - _c1[0] - 10 * _f) < 1e-3 and fw.spot_active == 0
                fw._spot_hover(300.0, 200.0)
                _ring = getattr(fw, "hover_ring", None) is not None and fw.hover_ring.isVisible()
                fw._spot_click(200.0, 150.0, Qt.RightButton)    # right-click removes the circle under the mouse
                check(len(fw.spots) == 1 and _r1 > _r0 * 1.3 and abs(_rect.width() * _f / 2 - _r1) < 1e-3
                      and _moved and _ring,
                      f"Spot Fix: click places a circle, the wheel resizes it at once ({_r0:.0f} -> {_r1:.0f} px), "
                      "a circle can be picked and dragged, a dashed ring shows the size, right-click removes")
            fw.apply_choice()
        check("core" in fw.state.done and "local" in fw.state.done and "spotfix" in fw.state.done,
              "optional tools (Tame Bright Core, local contrast, Spot Fix) work")
        n1 = len(fw.project.steps)
        fw.undo_last()
        check(len(fw.project.steps) == n1 - 1 and "spotfix" not in fw.state.done, "Undo removes the last step")
        sky = [float(__import__("numpy").median(c[::3, ::3])) for c in st_.rgb]
        check(max(sky) - min(sky) < 0.03 and 0.05 < sum(sky) / 3 < 0.25,
              f"finished sky is neutral and sensibly bright ({[round(x, 3) for x in sky]})")
        check(len(fw.project.steps) == n_before + 2,           # core + local contrast (Spot Fix undone)
              f"every step saved in the project ({len(fw.project.steps)})")
        # Final Crop: a box on the picture, trimmed to what the user drags
        h0, w0 = fw.state.shape
        fw.open_tool("final_crop")
        QApplication.processEvents()
        check(fw.crop_row.isVisibleTo(fw) and not fw.view.locked and fw.view.rect_item.isVisible(),
              "Final Crop shows a draggable box on the picture")
        shot("22_finish_final_crop.png", fw)
        fw.view.set_crop((20, 10, w0 - 30, h0 - 16))
        fw.apply_choice()
        QApplication.processEvents()
        check(fw.state.shape == (h0 - 26, w0 - 50) and "final_crop" in fw.state.done,
              f"Final Crop trims the picture to the box ({fw.state.shape} from {(h0, w0)})")
        # Flip the Picture: two toggle buttons; pressing again flips back
        _orig = fw.state.rgb.copy()
        _nflip = len(fw.project.steps)
        fw.open_tool("flip")
        fw.btn_flip_h.click()
        fw.btn_flip_h.click()                    # toggled back
        fw.btn_flip_v.click()
        QApplication.processEvents()
        _fp = fw.current_params()
        fw.apply_choice()
        QApplication.processEvents()
        _flip_ok = np.array_equal(fw.state.rgb, _orig[:, ::-1, :])
        check(_fp == {"h": False, "v": True} and _flip_ok
              and "flipped vertically" in fw.project.steps[-1]["label"],
              f"Flip the Picture: Flip Horizontally / Flip Vertically toggle (pressing again flips back) - {_fp}")
        fw.undo_last()
        QApplication.processEvents()
        check(len(fw.project.steps) == _nflip and np.array_equal(fw.state.rgb, _orig), "undo un-flips the picture")
        # Sign Your Picture
        before_sig = fw.state.rgb.copy()
        fw.open_tool("signature")
        fw.sign_text.setText("B. Poteet")
        fw._sign_changed()
        QApplication.processEvents()
        check(fw.sign_row.isVisibleTo(fw) and len(fw.variants) == 3,
              f"Sign Your Picture offers 3 fonts ({[v.label for v in fw.variants]})")
        shot("21_finish_signature.png", fw)
        # the target name line, and dragging the signature somewhere else
        _r0 = fw.sign_rect
        _pw, _ph = fw.view.full
        _tl = fw.sign_target.text()
        if _r0 is not None:
            fw._sign_drag(_r0[0] + _r0[2] / 2, _r0[1] + _r0[3] / 2, "press")
            fw._sign_drag(0.1 * _pw + _r0[2] / 2, 0.1 * _ph + _r0[3] / 2, "release")
            QApplication.processEvents()
        _moved = fw.sign_pos is not None and abs(fw.sign_pos[0] - 0.1) < 0.02 and abs(fw.sign_pos[1] - 0.1) < 0.02
        _r1 = fw.sign_rect
        check(_tl.startswith("M33") and _r0 is not None and _moved and _r1 is not None and _r1[1] < 0.2 * _ph
              and fw.sign_corner.currentText() == "Where I dragged it",
              f"the signature carries the target name ('{_tl}') and can be dragged anywhere (now at {fw.sign_pos})")
        shot("21b_finish_signature_dragged.png", fw)
        fw.sign_corner.setCurrentText("Bottom right")          # back to the corner for the next check
        fw._sign_changed()
        QApplication.processEvents()
        fw.apply_choice()
        QApplication.processEvents()
        h_, w_ = fw.state.rgb.shape[1:]
        corner_ = (fw.state.rgb[:, int(h_ * 0.75):, int(w_ * 0.5):] - before_sig[:, int(h_ * 0.75):, int(w_ * 0.5):])
        top_ = (fw.state.rgb[:, :int(h_ * 0.5), :] - before_sig[:, :int(h_ * 0.5), :])
        check("signature" in fw.state.done and float(abs(corner_).max()) > 0.2 and float(abs(top_).max()) == 0.0,
              "the signature is drawn in the bottom-right corner only")
        fw.stop_here_pixinsight()
        pf = getattr(fw, "last_pi_files", [])
        check(len(pf) >= 2 and pf[0].suffix == ".txt" and all(p.exists() for p in pf),
              f"Stop Here - Continue in PixInsight saves XISF + a note ({[p.name for p in pf]})")
        fw.share_astrobin(auto_accept=True)
        af = [Path(p) for p in getattr(fw, "last_astrobin_files", [])]
        rows = []
        if af and af[-1].suffix == ".csv":
            import csv as _csv
            rows = list(_csv.reader(open(af[-1], encoding="utf-8")))
        check(any(p.suffix == ".jpg" for p in af) and len(rows) >= 3 and rows[0][:4] == ["date", "filter", "number",
                                                                                            "duration"]
              and sum(int(r[2]) for r in rows[1:]) == 32,
              f"Share to AstroBin saves the picture + an acquisition CSV ({[p.name for p in af]}, {rows[:3]})")
        # go back two steps, then reopen: the project resumes
        fw.hist.setCurrentIndex(2)                           # just after the first two steps
        fw.go_back()
        check(len(fw.project.steps) == 3 == fw.hist.count() and F.next_step(fw.state) is not None,
              "Go Back to Here undoes the later steps")
        shot("18_finish_done.png", fw)
        sources = list(fw.state.sources)
        n_steps = len(fw.project.steps)
        fw.close()
        import finish_window as fwm
        fw2 = fwm.FinishWindow(state["pw"], state["pw"].settings)
        fw2.synchronous = True
        fw2.load_masters(sources)
        check(fw2.project is not None and len(fw2.project.steps) == n_steps,
              f"reopening the same masters resumes the project ({n_steps} steps)")
        fw2.back_to_start()
        check(len(fw2.project.steps) == 1 and fw2.state.linear and not [k for k in fw2.state.done
                                                                        if k != "stars_out"],
              "Back to Start returns to the original masters")
        import finish as F
        # HOO for Ha + OIII + SII sets, Flip the Picture, the two-line signature block, the description
        _nb = {k: np.random.default_rng(1).random((40, 50)).astype(np.float32) for k in ("Ha", "OIII", "SII")}
        _labels = [v.label for v in F.TOOLS["combine"].variants(F.State(channels=_nb, kind="mono_nb"))]
        _fs = F.State(rgb=np.random.default_rng(2).random((3, 30, 40)).astype(np.float32), linear=False)
        _lr = F.TOOLS["flip"].apply(_fs, {"h": True, "v": False}).rgb
        _ud = F.TOOLS["flip"].apply(_fs, {"h": False, "v": True}).rgb
        _bt = F.TOOLS["flip"].apply(_fs, {"h": True, "v": True}).rgb
        _flip_ok = (np.array_equal(_lr, _fs.rgb[:, :, ::-1]) and np.array_equal(_ud, _fs.rgb[:, ::-1, :])
                    and np.array_equal(_bt, _fs.rgb[:, ::-1, ::-1]))
        _wf = F.WORKFLOWS["mono_nb"]
        check("HOO" in _labels and _flip_ok and _wf.index("flip") < _wf.index("final_crop") < _wf.index("signature"),
              f"palettes for Ha+OIII+SII include HOO ({_labels}); Flip the Picture flips horizontally and / or "
              "vertically, before the signature")
        _sp = {"text": "B. Poteet", "target_text": 'NGC 281 "Pacman Nebula"', "font": "Georgia", "size": 3.0}
        _b1 = F.SignPicture.block((600, 900), _sp)
        _b2 = F.SignPicture.block((600, 900), {**_sp, "target_text": ""})
        _b3 = F.SignPicture.block((600, 900), {**_sp, "pos": [0.1, 0.05]})
        check(_b1 is not None and _b2 is not None and _b1[0].shape[0] > _b2[0].shape[0] * 1.4
              and _b1[1] + _b1[0].shape[1] <= 900 and _b1[2] > 450 and _b3[1] == 90 and _b3[2] == 30,
              f"the target name sits above the signature (block {_b1[0].shape} vs {_b2[0].shape}) and a dragged "
              f"position is kept ({_b3[1]}, {_b3[2]})")
        import target_info as _ti
        import targets as _tg2
        _fake = {"query": {"pages": {"1": {"title": "Pacman Nebula", "index": 1,
                                           "extract": "NGC 281 is an H II region in Cassiopeia.\nIt is known as "
                                                      "the Pacman Nebula.\nThird paragraph."}}}}
        _asked = []
        _res = _ti.lookup("NGC281", getter=lambda p: (_asked.append(p), _fake)[1])
        _note = _ti.note_text("NGC281", *_res) if _res else ""
        _sx = ('<?xml version="1.0"?><Sesame><Target option="S"><name>M 42</name><Resolver name="S=Simbad">'
               '<otype>HII</otype><jradeg>83.82208</jradeg><jdedeg>-5.39111</jdedeg><mag band="V"><v>4.0</v>'
               '</mag><oname>M  42</oname></Resolver></Target></Sesame>')
        _sf = _ti.lookup_simbad("M42", getter=lambda q: _sx)
        _sn = _ti.note_text("Sh2155", facts=_sf)
        check(_sf is not None and _sf["type_code"] == "HII" and "H II region" in _sn and "RA 05h 35m" in _sn
              and "SIMBAD" in _sn and _ti.simbad_query("Sh2155") == "Sh 2-155"
              and _ti.lookup_simbad("x", getter=lambda q: "<Sesame><Target><name>x</name></Target></Sesame>") is None,
              "SIMBAD facts: type, position and brightness, credited (and nothing when SIMBAD doesn't know it)")
        _ctx = _ti.ssl_context()
        _ncert = _ctx.cert_store_stats().get("x509_ca", 0)
        try:
            _live = _ti.lookup("M42")
            _live_msg = f"live lookup: {_live[0] if _live else 'nothing found'}"
            _live_ok = _live is not None
        except Exception as _e:  # noqa: BLE001
            _live_msg = f"live lookup failed: {_e}"
            _live_ok = "CERTIFICATE" not in str(_e).upper()          # only a missing-certificate failure is ours
        check(_ncert > 50 and _live_ok, f"secure connections work in this build ({_ncert} certificates; {_live_msg})")
        try:
            _lw, _lf, _lp = _ti.lookup_all("NGC281")
            _lmsg = f"{(_lw or ['-'])[0]} / {_lf}" + (f" / {_lp}" if _lp else "")
            if _lf is None:
                try:
                    _raw = _ti._sesame_get(_ti.simbad_query("NGC281"))
                except Exception as _e2:  # noqa: BLE001
                    _raw = f"(fetch failed: {_e2})"
                log("SESAME RAW: " + _raw[:1500].replace("\n", " | "))
            _lok = _lf is not None and _lf.get("type_code") == "HII" or any("CERTIFICATE" not in x.upper() for x in _lp)
        except Exception as _e:  # noqa: BLE001
            _lok, _lmsg = False, f"failed: {_e}"
        check(_lok, f"live Wikipedia + SIMBAD lookup of NGC 281: {_lmsg}")
        check(_res is not None and _asked[0]["titles"] == "Pacman Nebula" and "Third" not in _res[1]
              and "CC BY-SA" in _note and _note.startswith('NGC 281 "Pacman Nebula"')
              and _tg2.display_name("ElephantsTrunkNebula") == "Elephants Trunk Nebula",
              "the target description takes the first two paragraphs from Wikipedia, with the source credited")

        # Darken the Sky Background: sky darker, nebula / faint dust / stars keep their brightness above it
        _r = np.random.default_rng(7)
        _H, _W = 600, 900
        _yy, _xx = np.mgrid[:_H, :_W].astype(np.float32)
        _glow = 0.17 + 0.05 * ((_xx / _W) ** 2)                                  # sky with a gradient
        _neb = 0.05 * np.exp(-((_xx - 330) ** 2 + (_yy - 300) ** 2) / (2 * 110 ** 2))   # broad faint nebula
        _img = (_glow + _neb + _r.normal(0, 0.015, (_H, _W))).astype(np.float32)
        for _k in range(120):
            _x0, _y0 = int(_r.uniform(10, _W - 10)), int(_r.uniform(10, _H - 10))
            _img[_y0 - 1:_y0 + 2, _x0 - 1:_x0 + 2] += 0.5
        _st = F.State(rgb=np.stack([_img] * 3), linear=False)
        _out = F.TOOLS["sky"].apply(_st, {"amount": 0.5}).rgb[1]
        _sky = (slice(20, 120), slice(650, 880))
        _nb = (slice(280, 320), slice(310, 350))
        _loc = (slice(5, 45), slice(310, 350))                 # the sky right above the nebula
        _sb, _sa = float(np.median(_img[_sky])), float(np.median(_out[_sky]))
        _nbb = float(np.median(_img[_nb])) - float(np.median(_img[_loc]))
        _nba = float(np.median(_out[_nb])) - float(np.median(_out[_loc]))
        check(_sa < 0.65 * _sb and abs(_nba - _nbb) < 0.25 * _nbb and "sky" in F.WORKFLOWS["mono_nb"]
              and F.WORKFLOWS["osc"].index("sky") < F.WORKFLOWS["osc"].index("final_crop"),
              f"Darken the Sky: sky {_sb:.3f} -> {_sa:.3f}, faint nebula above the sky {_nbb:.3f} -> {_nba:.3f}; "
              "offered before Final Crop")
        # Mote Buster (Fix Dust Donuts): found in the sky, divided out with a synthetic flat
        _r2 = np.random.default_rng(9)
        _H2, _W2 = 700, 1000
        _y2, _x2 = np.mgrid[:_H2, :_W2].astype(np.float32)
        _clean = 0.02 + 0.004 * _x2 / _W2
        for _k in range(250):
            _sx, _sy = int(_r2.uniform(5, _W2 - 5)), int(_r2.uniform(5, _H2 - 5))
            _clean[_sy - 1:_sy + 2, _sx - 1:_sx + 2] += _r2.uniform(0.02, 0.3)
        _rr = np.hypot(_x2 - 700, _y2 - 350) / 60
        _dimg = (_clean * (1 - 0.06 * np.clip((1 - _rr) / 0.15, 0, 1)) + _r2.normal(0, 0.0012, (_H2, _W2))).astype(np.float32)
        _dst = F.State(rgb=np.stack([_dimg] * 3), linear=True)
        _found = F.TOOLS["dust"].found_spots(_dst)
        _fixed = F.TOOLS["dust"].apply(_dst, {"spots": _found}).rgb[0] if _found else _dimg
        _m = _rr < 0.9
        _eb = float(np.median((_dimg[_m] - _clean[_m]) / _clean[_m]))
        _ea = float(np.median((_fixed[_m] - _clean[_m]) / _clean[_m]))
        _ok_pos = bool(_found) and abs(_found[0][0] - 700) < 15 and abs(_found[0][1] - 350) < 15
        check(len(_found) == 1 and _ok_pos and abs(_ea) < 0.01 and "dust" in F.WORKFLOWS["osc"]
              and F.WORKFLOWS["osc"].index("dust") < F.WORKFLOWS["osc"].index("stretch"),
              f"Mote Buster finds the donut ({[[round(v) for v in d] for d in _found]}) and divides it out "
              f"(shadow {_eb:+.3f} -> {_ea:+.3f}); offered while the picture is unstretched")
        # Final Noise Reduction (NoiseXTerminator) - through the stand-in rc-astro when it is there
        _dn = F.TOOLS["denoise"]
        if state["fake"]:
            _small = F.State(rgb=np.stack([_img[:300, :300]] * 3).copy(), linear=False)
            _ok, _why = _dn.applicable(_small)
            _res = _dn.apply(_small, {"amount": 0.5}) if _ok else None
            _diff = float(np.abs(_res.rgb - _small.rgb).max()) if _res is not None else 9
            check(_ok and _res is not None and _diff < 0.01 and "denoise" in F.WORKFLOWS["osc"]
                  and F.WORKFLOWS["osc"].index("denoise") == F.WORKFLOWS["osc"].index("sky") + 1,
                  f"Final Noise Reduction runs NoiseXTerminator and keeps the picture the right way up "
                  f"(largest change {_diff:.4f}); offered after Darken the Sky ({_why})")
        else:
            check(_dn.runner is None or _dn.applicable(fw2.state)[0], "Final Noise Reduction says what it needs "
                                                                     "when NoiseXTerminator is missing")
        # Save Project As twice into the same folder (Mac report: the second save deleted the project)
        _pd = session.parent / "Final Astro Pics"
        _pd.mkdir(exist_ok=True)
        (_pd / "other_picture.jpg").write_bytes(b"x")         # a folder with other things in it
        _n = len(fw2.project.steps)
        _p1 = fw2.project.save_as(_pd)
        _p2 = _p1.save_as(_pd)
        _p3 = _p2.save_as(_pd / _p2.dir.name)
        check(_p1.dir.name.endswith(".szproj") and _p2.dir == _p1.dir == _p3.dir and _p3.dir.is_dir()
              and len(F.Project(_p3.dir).steps) == _n and (_pd / "other_picture.jpg").exists(),
              f"saving a project again to the same folder keeps it ({_n} steps in {_p3.dir.name})")
        _p4 = _p3.save_as(session.parent / "Another Place")
        check(_p4.dir.is_dir() and _p3.dir.is_dir() and len(_p4.steps) == _n
              and not list(_pd.glob("*.saving")), "saving a project to a new folder copies it completely")
        # Zip My Logs: every open page's log plus a note about this computer, in one zip
        import zip_logs as _zl
        import zipfile as _zf
        _zp = _zl.zip_my_logs(None, folder=session.parent)
        _names = _zf.ZipFile(_zp).namelist() if _zp else []
        _about = _zf.ZipFile(_zp).read("00_about_this_computer.txt").decode() if _zp else ""
        check(_zp is not None and "00_about_this_computer.txt" in _names and fb.APP_VERSION in _about
              and any("Calibrate and Stack_log" in n for n in _names)
              and any("Astro Image Processing_log" in n for n in _names)
              and any(n.endswith("_stacklog.txt") for n in _names)
              and getattr(fw2.app_header, "btn_zip", None) is not None,
              f"Zip My Logs packs every page's log, the stack logs and a system note ({len(_names)} files)")
        # Target types: recognised from the name and used only to tailor what is recommended
        import targets as _tg
        _names_t = {"M31": "galaxy", "NGC 7000": "nebula", "M57": "planetary nebula", "M45": "star cluster",
                    "161P/Hartley-IRAS": "comet", "C/2023 A3": "comet", "Elephant's Trunk Nebula": "nebula",
                    "Whirlpool Galaxy": "galaxy", "Sh2-155": "nebula", "my target": ""}
        _bad_t = {k: _tg.classify(k) for k, v in _names_t.items() if _tg.classify(k) != v}
        _wf_neb = F.workflow(F.State(rgb=np.zeros((3, 4, 4), np.float32), kind="osc", target="NGC 7000"))
        _wf_cl = F.workflow(F.State(rgb=np.zeros((3, 4, 4), np.float32), kind="osc", target="M45"))
        check(not _bad_t and [t.key for t, _ in _wf_neb] == [t.key for t, _ in _wf_cl]
              and F.target_type(F.State(target="M45")) == "star cluster"
              and "stars_out" in F.TYPE_TUNING["star cluster"]["optional"]
              and "sky" in F.TYPE_TUNING["comet"]["recommend"],
              f"target types recognised ({_bad_t or 'all right'}); a star cluster keeps every step, with star "
              "removal made optional")
        # Comet: a second stack follows the comet; the main master joins sharp stars and a sharp comet
        import stacking as st
        _cr = np.random.default_rng(8)
        _cd = session.parent / "comet_frames"
        _cd.mkdir(exist_ok=True)
        _CH, _CW = 400, 560
        _cyy, _cxx = np.mgrid[:_CH, :_CW].astype(np.float32)
        _cst = [(_cr.uniform(10, _CW - 10), _cr.uniform(10, _CH - 10), _cr.uniform(300, 6000)) for _ in range(150)]
        for _i in range(10):
            _dx, _dy = _cr.uniform(-5, 5), _cr.uniform(-5, 5)
            _ci = 1000 + _cr.normal(0, 12, (_CH, _CW))
            for _x, _y, _a in _cst:
                _x, _y = _x + _dx, _y + _dy
                if 5 < _x < _CW - 5 and 5 < _y < _CH - 5:
                    _sl = (slice(int(_y) - 5, int(_y) + 6), slice(int(_x) - 5, int(_x) + 6))
                    _ci[_sl] += _a * np.exp(-((_cxx[_sl] - _x) ** 2 + (_cyy[_sl] - _y) ** 2) / (2 * 1.4 ** 2))
            _hx, _hy = 230 + 50 * _i / 9 + _dx, 200 + 12 * _i / 9 + _dy
            _r2c = (_cxx - _hx) ** 2 + (_cyy - _hy) ** 2
            _ci += 900 * np.exp(-_r2c / (2 * 3.0 ** 2)) + 300 * np.exp(-np.sqrt(_r2c) / 12)
            core.write_fits(_cd / f"Comet_{_i:02d}.fits", np.clip(_ci, 0, 65535).astype(np.uint16),
                            {"IMAGETYP": "Light Frame", "OBJECT": "C/2026 X1", "FILTER": "L", "EXPTIME": 120.0,
                             "DATE-OBS": f"2026-10-16T02:{_i * 2:02d}:00"}, bitpix=16)
        _cinf = [st.read_info(_p) for _p in sorted(_cd.glob("*.fits"))]
        _cg = st.LightGroup(st.target_name(_cinf[0].target), "L", "Test", "rig", [st.LightFrame(_x) for _x in _cinf])
        _cg.plan = st.CalPlan()
        _co = st.StackOptions(work_dir=str(session.parent / "comet_work"),
                              masters_root=str(session.parent / "comet_masters"), formats=("fits",))
        _cres = st.stack_group(_cg, _co)
        _cm = _cres.extra.get("Comet") if _cres.extra else None
        _cfiles = st.save_master_light(_cres, _co) if _cm is not None else []
        _mv = (_cres.comet or {}).get("moved", 0) if hasattr(_cres, "comet") else 0
        check(_cm is not None and abs(_mv - 51.4) < 4 and any(str(f).endswith("_Comet.fits") for f in _cfiles)
              and all("/" not in Path(f).name for f in _cfiles)
              and all(Path(f).parent.parent == Path(_co.masters_root) for f in _cfiles),
              f"comet stack follows the comet (moved {_mv:.1f} px, truly 51.4) and saves "
              f"{[Path(f).name for f in _cfiles]}")
        # End this Project and Start New Project: every page closes, page 1 is empty and shown
        import ui_header as _uh2
        check("Start New Project" in fw2.btn_new_project.text() and
              fw2.btn_new_project.parent() is not None, "Finishing Touches has 'End this Project and Start New "
                                                        "Project' at the top")
        _before = len(_uh2.other_pages(w))
        ok_new = fw2.end_project(ask=False, load=False)
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, 0)
        left = [x.windowTitle() for x in _uh2.other_pages(w) if x.isVisible()]
        check(ok_new and _before >= 2 and not left and not w.frames and w.list.count() == 0
              and getattr(w, "stack_win", None) is None and not w.isHidden(),
              f"Start New Project closes every page ({_before} open, still showing: {left}) and empties page 1")
        QTimer.singleShot(900, lambda: (shot("19a_new_project.png"), finish(0 if not failures else 1)))

    QTimer.singleShot(300, start)
    return app.exec()
