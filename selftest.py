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
    # hot pixels are not stars: with many of them (calibrated remote-observatory frames, no dithering)
    # the star list was mostly sensor defects, fixed on the sensor - the matcher could line up the
    # sensor instead of the sky (doubled stars). Spikes are removed before stars are looked for.
    _rh = np.random.default_rng(7)
    _H = 900
    _hot = np.c_[_rh.integers(5, _H - 5, 1500), _rh.integers(5, _H - 5, 1500)]
    _hv = _rh.uniform(2e4, 6e4, 1500)
    _st_xy = np.c_[_rh.uniform(60, _H - 60, 60), _rh.uniform(60, _H - 60, 60)]
    _st_f = _rh.uniform(1e4, 6e4, 60)
    _yy2, _xx2 = np.mgrid[:_H, :_H]

    def _hp_frame(dx, dy, seed):
        r = np.random.default_rng(seed)
        img = 100 + r.normal(0, 5, (_H, _H))
        sg = 1.9 / 2.355
        for (x, y), f in zip(_st_xy, _st_f):
            x, y = x + dx, y + dy
            sl = (slice(int(y) - 5, int(y) + 6), slice(int(x) - 5, int(x) + 6))
            img[sl] += f * np.exp(-((_xx2[sl] - x) ** 2 + (_yy2[sl] - y) ** 2) / (2 * sg * sg))
        img[_hot[:, 1], _hot[:, 0]] += _hv * r.uniform(0.97, 1.03, len(_hv))
        return img.astype(np.float32)
    _hA, _hB = _hp_frame(0, 0, 1), _hp_frame(-40, 3, 2)
    _nspk = _stk.despike(_hA)[1]
    _pa = _stk.find_stars(*_stk.luminance(_hA, False))[0]
    _pb = _stk.find_stars(*_stk.luminance(_hB, False))[0]
    _real = float(np.mean(np.hypot(_pa[:, None, 0] - _st_xy[None, :, 0], _pa[:, None, 1] - _st_xy[None, :, 1]).min(1) < 1.5))
    _Th, _nh, _rmsh, *_r = _stk.match_frames(_pa, _pb)
    check(_nspk >= 1400 and _real > 0.75 and abs(_Th[0, 2] - 40) < 0.3 and abs(_Th[1, 2] + 3) < 0.3 and _rmsh < 0.3,
          f"hot pixels are not taken for stars ({_nspk} removed, {_real:.0%} of the star list real): frames "
          f"with many hot pixels line up on the sky ({_nh} stars, {_rmsh:.2f} px)")
    # plate solutions line frames up when the star pattern can't (few stars in common, many strays),
    # and a star match that disagrees with them is caught
    import math as _m

    def _wlook(cx, cy, rot, s_=1.7228e-4):
        r_ = _m.radians(rot)
        return {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 109.63, "CRVAL2": -13.23, "CRPIX1": cx,
                "CRPIX2": cy, "CD1_1": -s_ * _m.cos(r_), "CD1_2": s_ * _m.sin(r_), "CD2_1": -s_ * _m.sin(r_),
                "CD2_2": -s_ * _m.cos(r_)}
    _wa, _wb = _wlook(1528, 1528, 0.0), _wlook(1578, 1508, 1.0)
    _Tw = _stk.wcs_transform(_wa, _wb, (3056, 3056))
    _rsw = np.random.default_rng(3)
    _refp = _rsw.uniform(50, 3000, (200, 2))
    _Tiw = np.linalg.inv(np.vstack([_Tw, [0, 0, 1]]))[:2]
    _tgtp = np.vstack([_stk._apply(_Tiw, _refp[:50]), _rsw.uniform(50, 3000, (150, 2))])
    _Tg, _ng, _rg, _o, _j, _how = _stk.match_frames_wcs(_refp, _tgtp, _wa, _wb, (3056, 3056))
    _gap_id = _stk._corner_gap(np.array([[1.0, 0, 0], [0, 1.0, 0]]), _Tw, (3056, 3056))
    _Toff = _Tw + np.array([[0, 0, 9.0], [0, 0, -6.0]])        # a plate solution that is 11 px out
    _Tg2, _ng2, *_r2 = _stk.match_guided(_refp, _tgtp, _Toff)
    check(_ng >= 45 and float(np.abs(_Tg - _Tw).max()) < 0.01 and _ng2 >= 45
          and float(np.abs(_Tg2 - _Tw).max()) < 0.01 and _gap_id > 40
          and _stk.wcs_of({}) is None,
          f"plate solutions guide the star match ({_ng2} stars confirm it even from 11 px out), "
          f"and a zero-shift match {_gap_id:.0f} px away from the plate solutions would be caught")
    # cosmetic correction: hot pixels in each calibrated frame are replaced by their neighbours (they
    # survive stacking when frames aren't dithered); stars are left alone
    _cimg = (100 + np.random.default_rng(1).normal(0, 3, (120, 120))).astype(np.float32)
    _yyc, _xxc = np.mgrid[:120, :120]
    _cimg += 4000 * np.exp(-((_xxc - 60.3) ** 2 + (_yyc - 40.6) ** 2) / (2 * 0.9 ** 2))   # a sharp star
    _cimg[10, 10] = _cimg[90, 30] = 6000.0                                                 # hot pixels
    _cfix, _cn = _stk.cosmetic(_cimg)
    check(_cn == 2 and _cfix[10, 10] < 120 and _cfix[90, 30] < 120 and abs(_cfix[41, 60] - _cimg[41, 60]) < 1e-3
          and _stk.StackOptions().cosmetic,
          "hot pixels are removed from each frame before stacking (cosmetic correction); stars are untouched")
    # Telescope Live writes the filter into the target name (OBJECT = 'M78_B'): every filter must still be one target
    _tn = [_stk.target_name(t, f) for t, f in (("M78_B", "Blue"), ("M78_L", "Luminance"), ("M78_Ha", "H-alpha"),
                                                ("NGC 7000 - Red", "R"), ("Sh2-155", "SII"), ("IC 1396", "Ha"),
                                                ("M78_B", "R"))]
    check(_tn == ["M78", "M78", "M78", "NGC7000", "Sh2-155", "IC1396", "M78_B"],
          f"a filter tacked onto the target name (Telescope Live: M78_B, M78_G …) is dropped, so the colours "
          f"are combined as one target ({_tn})")
    # pictures on another brightness scale (a 0..1 master, or GraXpert 'Division' pushing values past 65535)
    # must reach BlurXTerminator & co. as a normal 0..1 picture - not near-black, not clipped flat
    import prepare as _prp
    import fits_core as _fcs
    _sct = []
    for _sd in (_cimg[None] * 20, _cimg[None] * 20 / 65535, _cimg[None] * 20 * 1.8):
        _sm = type("M", (), {})()
        _sm.data, _sm.cards = _sd.astype(np.float32), [("IMAGETYP", "Master Light", "")]
        _ssc = _prp._write_temp(_sm, _sfd / "scale_in.fits")
        _ssent = _fcs.read_fits(_sfd / "scale_in.fits")[0]
        _fcs.write_fits(_sfd / "scale_out.fits", _ssent, [("IMAGETYP", "Master Light", "")], bitpix=-32)
        _sback = _prp._read_result(_sfd / "scale_out.fits", _sm.data, _ssc)
        _sct.append((float(np.median(_ssent)) > 0.005 and float(_ssent.max()) <= 1.0
                     and bool(np.allclose(_sback, np.clip(_sm.data, 0, _ssc), rtol=1e-3))))
    check(all(_sct), f"the RC-Astro / GraXpert hand-off works on any brightness scale: normal, 0..1 and past 65535 "
                     f"masters all reach the tool as a normal picture and come back unchanged ({_sct})")
    # GraXpert correction 'Automatic': Subtraction for flat-calibrated and ready-made data, Division without flats
    _gcm = []
    for _gl, _glog in (({"PRECAL": "Telescope Live: PROC = True", "CALFLAT": "none"}, None), ({"CALFLAT": "24 flat subframes"}, None),
                       ({}, None), ({"CALFLAT": "none"}, "Note: These lights are already calibrated (x)"),
                       ({"CALFLAT": "none"}, None)):
        _gm = type("M", (), {})()
        _gm.look, _gm.path = _gl, _sfd / f"gc_{len(_gcm)}.fits"
        if _glog:
            (_sfd / f"gc_{len(_gcm)}_stacklog.txt").write_text(_glog, encoding="utf-8")
        _gcm.append(_prp.gradient_correction_for(_gm)[0])
    check(_gcm == ["Subtraction"] * 4 + ["Division"],
          f"gradient removal 'Automatic': Subtraction for Telescope Live / flat-calibrated / unknown masters, Division "
          f"only for a master stacked without flats ({_gcm})")
    # Tame Bright Core: a big burnt area (a nebula's heart) gets 'Strong' by default, a small nucleus 'Medium'
    import finish as _fnc
    _tcp = []
    for _tr in (15, 60):
        _ts = _fnc.State.__new__(_fnc.State)
        _tyy, _txx = np.mgrid[:1200, :1200]
        _ts.rgb = np.full((3, 1200, 1200), 0.1, np.float32)
        _ts.rgb[:, (_tyy - 600) ** 2 + (_txx - 600) ** 2 < _tr ** 2] = 0.97
        _ts.linear, _ts.sat_core = False, False
        _ts.target = "NGC 2068"
        try:
            _tcp.append(next(v.label for v in _fnc.TOOLS["core"].variants(_ts) if v.recommended))
        except Exception as _te:  # noqa: BLE001
            _tcp.append(f"error: {_te}")
    check(_tcp == ["Medium", "Strong"], f"Tame Bright Core suggests 'Strong' for a big burnt core, 'Medium' for a "
                                        f"small nucleus ({_tcp})")
    # BlurXTerminator: ready-made data gets firmer star sharpening; backyard data and the user's own values stay
    _bm_pro = type("M", (), {"look": {"PRECAL": "Telescope Live: PROC = True"}, "path": _sfd / "x.fits"})()
    _bm_own = type("M", (), {"look": {"CALFLAT": "20 flat subframes"}, "path": _sfd / "y.fits"})()
    _bv = [_prp.bxt_values_for([_bm_pro], {"ss": 0.40, "ash": 0.40, "sn": 0.03})[0],
           _prp.bxt_values_for([_bm_own], {"ss": 0.40, "ash": 0.40, "sn": 0.03})[0],
           _prp.bxt_values_for([_bm_pro], {"ss": 0.30, "ash": 0.40, "sn": 0.03})[0]]
    check([(b["ss"], b["ash"]) for b in _bv] == [(0.55, 0.50), (0.40, 0.40), (0.30, 0.40)]
          and _stk.dither_report([np.eye(2, 3)] * 3, [2.8] * 3)["small_stars"] is False,
          f"ready-made data: BlurXTerminator Sharpen Stars 0.55 / Star Halos 0.50; backyard data and your own values "
          f"unchanged; drizzle is only suggested for stars under 2.2 px ({_bv})")
    # session log on disk: page log lines and memory notes land in a file; one without 'closed normally'
    # (a crash, a frozen computer) is flagged when the logs are zipped
    import session_log as _slg
    _sl_dir = _sfd / "session_logs"
    _sl_old = _slg.logs_dir
    _slg.logs_dir = lambda: _sl_dir
    try:
        _sl_dir.mkdir(exist_ok=True)
        (_sl_dir / "session_2000-01-01_000000.log").write_text("00:00:00 [page 3] BlurXTerminator ...\n")
        _slp = _slg.start("selftest")
        _slg.write("page 3", "a line from page 3")
        _sl_ok = (_slp is not None and "a line from page 3" in _slp.read_text(encoding="utf-8")
                  and "memory" in _slp.read_text(encoding="utf-8")
                  and [p.name for p in _slg.previous()] == ["session_2000-01-01_000000.log"]
                  and _slg.ended_badly(_sl_dir / "session_2000-01-01_000000.log"))
        _slg.stop()
        _sl_ok = _sl_ok and not _slg.ended_badly(_slp)
    finally:
        _slg.logs_dir = _sl_old
        _slg._path = None
    _sl_mem = _slg.memory()
    check("app" in _sl_mem and "free" in _sl_mem, f"the memory note has the app's own use and what is free ({_sl_mem})")
    check(_sl_ok, f"every session writes a log on disk (page lines, memory); a session that ended badly is "
                  f"flagged for Zip My Logs ({_slg.memory_text()})")
    # Star Colours on a big overexposed star: its core turns white, never a guessed pure blue (Shark's eye)
    _syy, _sxx = np.mgrid[:300, :300]
    _sst = np.zeros((3, 300, 300), np.float32)
    _sr = np.hypot(_syy - 150, _sxx - 150)
    _spf = np.clip(1.6 * np.exp(-(_sr / 30) ** 2), 0, 1)
    _sst[0], _sst[1], _sst[2] = _spf, _spf * 0.8, _spf * 0.6
    _sr2 = np.hypot(_syy - 60, _sxx - 60)
    _sp2 = 0.5 * np.exp(-(_sr2 / 2) ** 2)
    _sst[0] += _sp2 * 0.6
    _sst[1] += _sp2 * 0.8
    _sst[2] += _sp2
    _sx, _ = _fnc.star_colour_map(_sst)
    _scol = _fnc._star_ramp(_sx)
    _scol = _scol / np.max(_scol, axis=0, keepdims=True)
    _score, _sblue = _scol[:, 150, 150], _scol[:, 60, 60]
    check(float(_score.min()) > 0.85 and float(_sblue[2] - _sblue[0]) > 0.2,
          f"Star Colours: a big overexposed star's core stays white (no guessed blue disc), small stars keep their "
          f"colour (core {np.round(_score, 2).tolist()}, blue star {np.round(_sblue, 2).tolist()})")
    # a meridian flip (half a turn) is not alt-az field rotation; real field rotation still counts
    def _rT(d):
        _rr = np.radians(d)
        return np.array([[np.cos(_rr), -np.sin(_rr), 0], [np.sin(_rr), np.cos(_rr), 0]])
    _rflip = _stk._rotation_spread([_rT(0), _rT(0.1), _rT(179.8), _rT(180.1)])
    _raz = _stk._rotation_spread([_rT(0), _rT(5), _rT(12)])
    check(_rflip < 0.5 and abs(_raz - 12) < 0.01, f"a meridian flip is not taken for alt-az field rotation "
                                                   f"({_rflip:.2f}° vs {_raz:.1f}°)")
    # StarXTerminator on a half-size copy for big (drizzled) stars: no star left in the starless picture, its fine
    # detail kept, and starless + stars is exactly the original (the Shark's squares)
    _hr = np.random.default_rng(0)
    _hH, _hW = 400, 500
    _hy, _hx = np.mgrid[:_hH, :_hW].astype(np.float32)
    _hneb = 2000 + 800 * np.exp(-((_hx - 250) ** 2 + (_hy - 200) ** 2) / (2 * 100 ** 2)) + 60 * np.sin(_hx / 7) * np.sin(_hy / 9)
    _hst = np.zeros((_hH, _hW), np.float32)
    for _ in range(60):
        _a, _b, _f = _hr.uniform(15, _hW - 15), _hr.uniform(15, _hH - 15), _hr.uniform(500, 30000)
        _hst += _f * np.exp(-((_hx - _a) ** 2 + (_hy - _b) ** 2) / (2 * 2.6 ** 2))
    _hno = _hr.normal(0, 15, (_hH, _hW))
    _hfull = (_hneb + _hst + _hno)[None].astype(np.float32)
    _hsl, _hstars = _prp._starless_full(_hfull, _prp._half2(_hfull), _prp._half2((_hneb + _hno)[None].astype(np.float32)),
                                        _prp._half2(_hst[None]))
    _herr = _hsl[0] - (_hneb + _hno)
    _hfw = _prp.star_size(_hfull)
    check(_hfw is not None and _hfw >= _prp.SXT_HALF_FWHM and float(_herr.max()) < 0.01 * float(_hst.max())
          and float(_herr.std()) < 10 and bool(np.allclose(_hsl + _hstars, _hfull, atol=0.05)),
          f"big stars (FWHM {(_hfw or 0):.1f} px): StarXTerminator on a half-size copy leaves no squares or stars "
          f"behind ({float(_herr.max()):.0f} of {float(_hst.max()):.0f}), keeps the fine detail, and starless + "
          f"stars = the original")
    # a big picture (drizzled master) also goes half size for StarXTerminator, a normal one with small stars does not
    _bhalf_keep = _prp.SXT_HALF_PIXELS
    try:
        _bsmall = _prp.sxt_half(_hfull * 0 + _hneb[None].astype(np.float32) + _prp._half2(_hfull)[:, :1, :1].mean())[0]
        _prp.SXT_HALF_PIXELS = _hH * _hW - 1
        _bbig = _prp.sxt_half(_hfull)[0]
    finally:
        _prp.SXT_HALF_PIXELS = _bhalf_keep
    check(_bbig and not _bsmall, f"StarXTerminator goes half size for big (drizzled) pictures, not for a normal one "
                                 f"({_bbig}, {_bsmall})")
    # Darken the Sky Background: 'Much darker' by default, 'Darker' when that would take the sky below about 3%
    _dk = []
    for _dsky, _dtg in ((0.15, "IC1396"), (0.07, "IC1396"), (0.15, "M28"), (0.15, "M27")):
        _ds = _fnc.State.__new__(_fnc.State)
        _ds.kind, _ds.target = "mono_lrgb", _dtg
        _drng = np.random.default_rng(3)
        _ds.rgb = np.clip(_dsky + _drng.normal(0, 0.005, (3, 300, 300)), 0, 1).astype(np.float32)
        _ds.linear = False
        try:
            _dk.append(next(v.label for v in _fnc.TOOLS["sky"].variants(_ds) if v.recommended))
        except Exception as _de:  # noqa: BLE001
            _dk.append(f"error: {_de}")
    _ck = []
    for _ctg in ("M27", "M51"):
        _cs4 = _fnc.State.__new__(_fnc.State)
        _cs4.kind, _cs4.target, _cs4.linear = "mono_nb", _ctg, False
        _ck.append(next(v.label for v in _fnc.TOOLS["curves"].variants(_cs4) if v.recommended))
    check(_dk == ["Much darker", "Darker", "Darkest", "Darker"] and _ck == ["Lift faint detail", "More contrast"],
          f"Darken the Sky Background suggests 'Much darker', 'Darker' when the sky is already dark or for a planetary "
          f"(its faint halo), 'Darkest' for a star cluster ({_dk}); Curves lifts faint detail for a planetary ({_ck})")
    # Clean Bright-Star Leftovers (from the real Shark data): StarXTerminator paints a smooth, grain-free box over a big
    # star, with a bright bar along its edge and a leftover dot. The box is found by its missing grain and patched with
    # the surroundings' texture; a painted box inside bright structured nebula is left alone; a picture without
    # painted boxes is left alone
    _gr = np.random.default_rng(5)
    _gH, _gW = 700, 1000
    _gy, _gx = np.mgrid[:_gH, :_gW].astype(np.float32)
    _gneb = 0.05 + 0.02 * np.exp(-((_gx - 500) ** 2 + (_gy - 350) ** 2) / (2 * 220 ** 2))
    _gneb = _gneb + 0.03 * np.exp(-((_gx - 760) ** 2 + (_gy - 330) ** 2) / (2 * 70 ** 2)) * (1 + np.sin(_gx / 6) * np.sin(_gy / 7))
    _gclean = (_gneb + _gr.normal(0, 0.002, (_gH, _gW))).astype(np.float32)
    _gst = np.zeros((_gH, _gW), np.float32)
    for _x, _y, _f in ((250, 250, 1.0), (760, 330, 0.9)):
        _gst += _f * np.exp(-((_gx - _x) ** 2 + (_gy - _y) ** 2) / (2 * 3 ** 2))
    for _ in range(150):
        _x, _y, _fl = _gr.uniform(10, _gW - 10), _gr.uniform(10, _gH - 10), _gr.uniform(0.01, 0.08)
        _gst += _fl * np.exp(-((_gx - _x) ** 2 + (_gy - _y) ** 2) / (2 * 1.3 ** 2))
    _gpaint = _gclean.copy()
    for (_x, _y), _hs in (((250, 250), 45), ((760, 330), 22)):
        _m = (np.abs(_gx - _x) < _hs) & (np.abs(_gy - _y) < _hs)
        _gpaint[_m] = float(np.median(_gneb[_m])) - 0.004                   # smooth, no grain, a little dark
    _gpaint[(np.abs(_gx - 250) < 45) & (np.abs(_gy - 205) < 4)] += 0.01     # bright bar along the top edge
    _gpaint[(_gx - 262) ** 2 + (_gy - 250) ** 2 < 5 ** 2] += 0.05           # leftover dot
    _gboxes, _gbusy = _fnc.find_star_leftovers(_gpaint[None], _gst[None])
    _gnone, _ = _fnc.find_star_leftovers(_gclean[None], _gst[None])
    _gs = _fnc.State.__new__(_fnc.State)
    _gs.rgb, _gs.stars, _gs.stars_out, _gs.stars_hidden, _gs.done, _gs.linear = _gpaint[None].copy(), _gst[None], True, False, [], True
    _gs.copy = lambda: _gs
    _gv = next(v for v in _fnc.TOOLS["leftovers"].variants(_gs) if v.recommended)
    _gout = _fnc.TOOLS["leftovers"].apply(_gs, _gv.params).rgb[0]
    _gin = (np.abs(_gx - 250) < 40) & (np.abs(_gy - 250) < 40)
    _glevel = float(np.mean((_gout - _gclean)[_gin]))
    _ggrain = _fnc.SpotFix._grain(_gout[200:300, 200:300], np.ones((100, 100), bool)) / \
        _fnc.SpotFix._grain(_gclean[200:300, 200:300], np.ones((100, 100), bool))
    _grefl = float(np.abs(_gout - _gpaint)[300:360, 730:790].max())
    check(len(_gboxes) == 1 and _gbusy >= 1 and not _gnone and abs(_glevel) < 0.002 and 0.6 < _ggrain < 1.5
          and _grefl == 0.0,
          f"Clean Bright-Star Leftovers: a painted grain-free box (with its bar and dot) is found and patched (level "
          f"off by {_glevel:.4f}, grain {_ggrain:.2f}x the sky's); a box inside bright nebula is left alone "
          f"({_gbusy}); a picture without painted boxes is left alone ({len(_gnone)})")
    # ... and the square glow star removal leaves around the box (a plate with straight edges) is made round
    _gsq = _gpaint + 0.008 * ((np.abs(_gx - 250) < 80) & (np.abs(_gy - 250) < 80))
    _gsb, _ = _fnc.find_star_leftovers(_gsq[None], _gst[None])
    _gso = _gsq[None]
    for _i, (_a, _b, _c, _d) in enumerate(_gsb):
        _gso = _fnc.heal_box(_gso, int(_a * _gW), int(_b * _gH), int(np.ceil(_c * _gW)), int(np.ceil(_d * _gH)), seed=_i)
    _gso = _gso[0]
    _grr = np.hypot(_gx - 250, _gy - 250)
    _gang = np.abs(np.sin(2 * np.arctan2(_gy - 250, _gx - 250)))         # 1 on the diagonals, 0 on the axes
    _gband = (_grr > 88) & (_grr < 104)
    def _gsquare(a):
        return float(np.mean(a[_gband & (_gang > 0.9)]) - np.mean(a[_gband & (_gang < 0.3)]))
    _gq0, _gq1 = _gsquare(_gsq), _gsquare(_gso)
    check(len(_gsb) == 1 and abs(_gq1) < 0.25 * abs(_gq0),
          f"Clean Bright-Star Leftovers: the square glow around a painted box is taken away (corners vs sides "
          f"{_gq0:.4f} -> {_gq1:.4f})")
    # ... and a big leftover core in one colour (the blue dot in the Shark's eye) is removed, not spread into a disc
    _cr = np.random.default_rng(1)
    _cy, _cx = np.mgrid[:800, :800].astype(np.float32)
    _cbase = (0.05 + _cr.normal(0, 0.002, (3, 800, 800))).astype(np.float32)
    _cimg2 = _cbase.copy()
    _cimg2[:, (np.abs(_cx - 400) < 60) & (np.abs(_cy - 400) < 60)] = 0.05
    _cimg2[2][(_cx - 415) ** 2 + (_cy - 400) ** 2 < 14 ** 2] += 0.08
    _cout = _fnc.heal_box(_cimg2, 340, 340, 460, 460, seed=0)
    _cex = [float(_fnc._blur(_cout[k] - _cbase[k], 3)[300:500, 300:500].max()) for k in range(3)]
    check(max(_cex) < 0.002, f"Clean Bright-Star Leftovers: a big blue leftover core in a box is removed, not spread "
                             f"into a blue disc (left {max(_cex):.4f}, was 0.08)")
    # ... and the nebula's own structure around the box is kept (no smooth 'painted' disc), with no glow left by
    # default and a quarter of it with 'Clean, faint glow'
    _ny, _nx = np.mgrid[:800, :800].astype(np.float32)
    _ntex = 0.004 * np.sin(_nx / 9) * np.sin(_ny / 11)
    _ntrue = (0.05 + _ntex + np.random.default_rng(2).normal(0, 0.001, (800, 800))).astype(np.float32)
    _nr = np.hypot(_nx - 400, _ny - 400)
    _nglow = 0.01 * np.exp(-_nr ** 2 / (2 * 75 ** 2))
    _nimg = _ntrue + _nglow + 0.006 * ((np.abs(_nx - 400) < 110) & (np.abs(_ny - 400) < 110))
    _nimg[(np.abs(_nx - 400) < 50) & (np.abs(_ny - 400) < 50)] = 0.06
    _nout0 = _fnc.heal_box(_nimg[None].astype(np.float32), 350, 350, 450, 450, seed=0, keep=0.0)[0]
    _nout1 = _fnc.heal_box(_nimg[None].astype(np.float32), 350, 350, 450, 450, seed=0, keep=0.25)[0]
    _nann = (_nr > 120) & (_nr < 280)
    _nc = float(np.corrcoef((_nout0 - _fnc._blur(_nout0, 30))[_nann], _ntex[_nann])[0, 1])
    _nres = float(np.mean((_nout0 - _ntrue)[_nann]))
    _nkept = float(np.mean((_nout1 - _nout0)[_nr < 120]))
    check(_nc > 0.8 and abs(_nres) < 0.0015 and 0.0003 < _nkept < 0.004,
          f"Clean Bright-Star Leftovers: the nebula's own detail around the box is kept (match {_nc:.2f}), the "
          f"glow is taken away (left {_nres:.4f} of 0.01), 'faint glow' keeps a little ({_nkept:.4f})")
    # ... and a small leftover star core in one colour inside bright nebula (the red dot on M27) is removed, while
    # nebula structure without a dot is left alone
    _dr2 = np.random.default_rng(3)
    _dy2, _dx2 = np.mgrid[:600, :800].astype(np.float32)
    _dneb = 0.1 + 0.05 * np.exp(-((_dx2 - 400) ** 2 + (_dy2 - 300) ** 2) / (2 * 150 ** 2)) * (
        1 + 0.5 * np.sin(_dx2 / 7) * np.sin(_dy2 / 9))
    _dclean = np.stack([_dneb + _dr2.normal(0, 0.002, (600, 800)) for _ in range(3)]).astype(np.float32)
    _dst2 = np.zeros((3, 600, 800), np.float32)
    for _x, _y, _f in [(420, 310, 1.0), (200, 150, 0.8), (650, 450, 0.6)] + [
            (_dr2.uniform(20, 780), _dr2.uniform(20, 580), _dr2.uniform(0.05, 0.3)) for _ in range(60)]:
        _dst2 += _f * np.exp(-((_dx2 - _x) ** 2 + (_dy2 - _y) ** 2) / (2 * 2.0 ** 2))
    _ddirty = _dclean.copy()
    _ddirty[0] += 0.04 * np.exp(-((_dx2 - 420) ** 2 + (_dy2 - 310) ** 2) / (2 * 1.5 ** 2))
    _dfound = _fnc.find_leftover_cores(_ddirty, _dst2)
    _dnone = _fnc.find_leftover_cores(_dclean, _dst2)
    _ds2 = _fnc.State.__new__(_fnc.State)
    _ds2.rgb, _ds2.stars, _ds2.stars_out, _ds2.stars_hidden, _ds2.done, _ds2.linear = _ddirty.copy(), _dst2, True, False, [], True
    _ds2.copy = lambda: _ds2
    _dv2 = next(v for v in _fnc.TOOLS["leftovers"].variants(_ds2) if v.recommended)
    _dout = _fnc.TOOLS["leftovers"].apply(_ds2, _dv2.params).rgb
    _dleft = float(_fnc._box_blur((_dout - _dclean)[0], 1)[305:316, 415:426].max())
    check(len(_dfound) == 1 and not _dnone and _dleft < 0.006 and float(np.abs(_dout - _dclean)[:, :, :300].max()) == 0,
          f"Clean Bright-Star Leftovers: a small leftover star core in one colour inside nebula (the red dot on "
          f"M27) is found and removed (left {_dleft:.4f} of 0.04); nebula without a dot is left alone")
    # ... and only at stars' places: a whole star left in the starless picture, a small galaxy and a small
    # one-colour emission knot (an HII region in a galaxy) are kept
    _o = _dclean.copy()
    _o[0] += 0.05 * np.exp(-((_dx2 - 500) ** 2 + (_dy2 - 280) ** 2) / (2 * 1.8 ** 2))
    _o += 0.05 * np.exp(-((_dx2 - 200) ** 2 + (_dy2 - 400) ** 2) / (2 * 1.8 ** 2))
    _o += 0.03 * np.exp(-(((_dx2 - 650) / 9) ** 2 + ((_dy2 - 150) / 3.5) ** 2) / 2)
    _of = _fnc.find_leftover_cores(_o, _dst2)
    check(not _of, f"Clean Bright-Star Leftovers keeps a whole star left in the starless picture, a small galaxy and "
                   f"an emission knot ({len(_of)} taken)")
    # Colour Intensity: narrowband (or already very colourful) data gets 'Slight', ordinary colour data 'Rich'
    _cy3, _cx3 = np.mgrid[:200, :200].astype(np.float32)
    _cneb = 0.1 + 0.5 * np.exp(-((_cx3 - 100) ** 2 + (_cy3 - 100) ** 2) / (2 * 40 ** 2))
    _sat = []
    for _kind, _tint, _tgt in (("mono_nb", (1.0, 0.45, 0.15), "NGC 7000"), ("osc", (1.0, 0.9, 0.8), "M31"),
                               ("osc", (1.0, 0.35, 0.1), "M42"), ("mono_lrgb", (1.0, 0.9, 0.8), "M27"),
                               ("osc", (1.0, 0.9, 0.8), "M45")):
        _cs = _fnc.State.__new__(_fnc.State)
        _cs.kind, _cs.linear, _cs.target = _kind, False, _tgt
        _cs.rgb = np.stack([_cneb * t for t in _tint]).astype(np.float32)
        _sat.append(next(v.label for v in _fnc.TOOLS["saturation"].variants(_cs) if v.recommended))
    check(_sat == ["Slight", "Rich", "Slight", "Slight", "Calmer"],
          f"Colour Intensity's suggestion follows the target and the data: narrowband, already strong colours, "
          f"planetaries 'Slight'; a galaxy in colour 'Rich'; a star cluster 'Calmer' ({_sat})")
    # Remove Green Cast on balanced colour data: nothing suggested (SCNR on noise turns a picture pink); a real
    # green cast still gets removed
    _gr3 = np.random.default_rng(8)
    _gbal = np.clip(0.15 + _gr3.normal(0, 0.02, (3, 300, 300)), 0, 1).astype(np.float32)
    _ggrn = _gbal.copy()
    _ggrn[1] += 0.02
    _gsc = []
    for _im in (_gbal, _ggrn):
        _gs3 = _fnc.State.__new__(_fnc.State)
        _gs3.kind, _gs3.palette, _gs3.rgb, _gs3.linear = "osc", None, _im, False
        _gsc.append((next(v.label for v in _fnc.TOOLS["scnr"].variants(_gs3) if v.recommended),
                     _fnc.TOOLS["scnr"].is_optional(_gs3)))
    check(_gsc == [("None needed", True), ("75%", False)],
          f"Remove Green Cast suggests nothing on balanced colour data (no pink cast) and removes a real green "
          f"cast ({_gsc})")
    # a globular cluster's core is overlapping stars: Tame Bright Core is only offered (it punched dark holes in
    # M28), a galaxy still gets it recommended
    _wf = {}
    for _tg in ("M28", "M51"):
        _ws = _fnc.State(rgb=np.clip(np.random.default_rng(0).normal(0.2, 0.02, (3, 64, 64)), 0, 1).astype(np.float32),
                         kind="mono_lrgb", target=_tg, done=["combine", "balance", "scnr", "stretch"], linear=False)
        _wf[_tg] = dict((t.key, st) for t, st in _fnc.workflow(_ws)).get("core")
    check(_wf == {"M28": "optional", "M51": "next"},
          f"Tame Bright Core is only offered for a star cluster (dark holes between the core's stars), "
          f"recommended for a galaxy ({_wf})")
    # narrowband palettes: a planetary gets 'HOO + SII' (red and teal, like most pictures of them), other targets
    # the Hubble palette; HSO's green is SII signal (removing it turned M27 purple), HOO's green is the OIII teal
    _pch = {k: (np.random.default_rng(1).random((64, 64)) * 0.1 + 0.05).astype(np.float32) for k in ("Ha", "OIII", "SII")}
    _psug = [next(v.label for v in _fnc.TOOLS["combine"].variants(_fnc.State(channels=dict(_pch), kind="mono_nb", target=_t))
                  if v.recommended) for _t in ("M27", "NGC 7000")]
    _pgr = [next(v.label for v in _fnc.TOOLS["scnr"].variants(_fnc.State(rgb=np.full((3, 32, 32), 0.3, np.float32),
                                                                       kind="mono_nb", palette=_p, linear=False))
                 if v.recommended) for _p in ("HSO", "HOO+S", "SHO")]
    check(_psug == ["HOO + SII", "Hubble (SHO)"] and _pgr == ["None needed", "25%", "Gold 100%"],
          f"narrowband: a planetary nebula gets 'HOO + SII', other targets the Hubble palette ({_psug}); Remove Green "
          f"Cast leaves HSO's SII and HOO's teal alone, turns SHO gold ({_pgr})")
    # the page-4 alignment check reads star centres in the stars-only images: two filters with the same stars but a
    # different nebula and faint stars (OIII) match (M27 was reported 1.9 px off while it matched to 0.1 px), and a
    # real 1.5 px shift still shows
    _ar = np.random.default_rng(9)
    _ay, _ax = np.mgrid[:900, :1200].astype(np.float32)
    _apts = [(_ar.uniform(30, 1170), _ar.uniform(30, 870), _ar.uniform(0.05, 1.0)) for _ in range(400)]

    def _astars(dx, k):
        a = np.zeros((900, 1200), np.float32)
        for _x, _y, _f in _apts:
            _y0, _x0 = int(_y) - 8, int(_x) - 8
            gy, gx = _ay[_y0:_y0 + 17, _x0:_x0 + 17], _ax[_y0:_y0 + 17, _x0:_x0 + 17]
            a[_y0:_y0 + 17, _x0:_x0 + 17] += k * _f * np.exp(-((gx - _x - dx) ** 2 + (gy - _y) ** 2) / (2 * 1.8 ** 2))
        return a + _ar.normal(0, 0.002, a.shape).astype(np.float32)
    _aref = _astars(0, 1.0)
    _a0 = _fnc._edge_offset_stars(_aref, _astars(0, 0.3), (900, 1200))
    _a1 = _fnc._edge_offset_stars(_aref, _astars(1.5, 0.3), (900, 1200))
    check(_a0 is not None and _a0[0] < 0.3 and _a1 is not None and 1.2 < _a1[0] < 1.8,
          f"page 4's alignment check reads star centres: matching faint stars are found to match ({_a0 and round(_a0[0], 2)} px), "
          f"a real 1.5 px shift shows ({_a1 and round(_a1[0], 2)} px)")
    # HOO + SII: an Ha-rich area fades to pink and white, not a flat red (M27 vs the Telescope Live picture); a narrowband
    # planetary gets 'Calmer' colour
    _hp = _fnc.build_palette("HOO+S", {"Ha": np.float32([0.8]), "OIII": np.float32([0.3]), "SII": np.float32([0.2])})
    _hs = _fnc.State.__new__(_fnc.State)
    _hs.kind, _hs.linear, _hs.target = "mono_nb", False, "M27"
    _hs.rgb = np.stack([_cneb * t for t in (1.0, 0.8, 0.8)]).astype(np.float32)
    _hsat = next(v.label for v in _fnc.TOOLS["saturation"].variants(_hs) if v.recommended)
    check(float(_hp[1, 0]) > 0.5 * float(_hp[0, 0]) and _hsat == "Calmer",
          f"HOO + SII: an Ha-rich area is pink, not flat red (R {float(_hp[0, 0]):.2f} G {float(_hp[1, 0]):.2f} B "
          f"{float(_hp[2, 0]):.2f}); a narrowband planetary gets '{_hsat}' colour")
    # a planetary with a dim interior (the Helix) does not get Tame Bright Core recommended; one whose ring burns out
    # (M27) does
    _pc = []
    for _lvl in (0.45, 1.0):
        _pyy, _pxx = np.mgrid[:300, :300].astype(np.float32)
        _prr = np.hypot(_pxx - 150, _pyy - 150)
        _pim = 0.08 + (_lvl - 0.08) * np.exp(-((_prr - 60) / 18) ** 2)
        _ps = _fnc.State(rgb=np.stack([_pim, _pim * 0.98, _pim * 0.97]).astype(np.float32), kind="mono_nb",
                         target="NGC 7293", done=["combine", "stars_out", "stretch", "scnr", "curves"], linear=False,
                         palette="HOO+S")
        _pc.append(dict((t.key, st) for t, st in _fnc.workflow(_ps)).get("core"))
    check(_pc[0] == "optional" and _pc[1] == "next", f"Tame Bright Core for a planetary only when its ring burns out "
                                                     f"(dim interior: {_pc[0]}, white ring: {_pc[1]})")
    # a star cluster is stretched one step gentler and gets 'Darker background' curves (its centre's stars stay apart)
    _cl = []
    for _tg in ("M28", "M51"):
        _cls = _fnc.State(rgb=np.clip(np.random.default_rng(0).normal(0.05, 0.01, (3, 64, 64)), 0, 1).astype(np.float32),
                          kind="mono_lrgb", target=_tg, linear=True)
        _st1 = next(v.label for v in _fnc.TOOLS["stretch"].variants(_cls) if v.recommended)
        _cls.linear = False
        _cl.append((_st1, next(v.label for v in _fnc.TOOLS["curves"].variants(_cls) if v.recommended)))
    check(_cl[0][1] == "Darker background" and _cl[0][0] != _cl[1][0] and _cl[1][1] == "More contrast",
          f"a star cluster gets a gentler stretch and 'Darker background' curves, a galaxy 'Medium' and 'More contrast' ({_cl})")
    # a star cluster's glow is balanced to a warm white (M28's centre came out blue-violet)
    _gy4, _gx4 = np.mgrid[:400, :400].astype(np.float32)
    _gg = 0.3 * np.exp(-((_gx4 - 200) ** 2 + (_gy4 - 200) ** 2) / (2 * 40 ** 2))
    _gimg = np.stack([0.05 + 0.78 * _gg, 0.05 + 0.69 * _gg, 0.05 + 1.0 * _gg]).astype(np.float32)
    _gs5 = _fnc.State(rgb=_gimg, kind="mono_rgb", target="M28", linear=True)
    _gv5 = next(v for v in _fnc.TOOLS["balance"].variants(_gs5) if v.recommended)
    _go5 = _fnc.TOOLS["balance"].apply(_gs5, _gv5.params).rgb
    _gc5 = np.array(_fnc.cluster_glow_colour(_go5, [float(np.median(c[:40, :40])) for c in _go5]))
    _gc5 = _gc5 / _gc5.max()
    check(_gv5.label == "Cluster glow warm white" and _gc5[2] < _gc5[0] and _gc5[1] > 0.9 * _gc5[0],
          f"a star cluster's glow is balanced to a warm white (was R 0.78 G 0.69 B 1.00, now "
          f"R {_gc5[0]:.2f} G {_gc5[1]:.2f} B {_gc5[2]:.2f})")
    # a globular's core: the blended glow between its stars is lowered, the stars kept (M28's core was a bright haze)
    _qy, _qx = np.mgrid[:600, :600].astype(np.float32)
    _qglow = 0.6 * np.exp(-((_qx - 300) ** 2 + (_qy - 300) ** 2) / (2 * 60 ** 2))
    _qst = np.zeros((600, 600), np.float32)
    _qr = np.random.default_rng(12)
    for _ in range(500):
        _x, _y = _qr.uniform(10, 590), _qr.uniform(10, 590)
        _y0, _x0 = int(_y) - 6, int(_x) - 6
        _qst[_y0:_y0 + 13, _x0:_x0 + 13] += 0.3 * np.exp(-((_qx[_y0:_y0 + 13, _x0:_x0 + 13] - _x) ** 2 +
                                                           (_qy[_y0:_y0 + 13, _x0:_x0 + 13] - _y) ** 2) / (2 * 1.5 ** 2))
    _qimg = np.clip(0.05 + _qglow + _qst, 0, 1)
    _qs = _fnc.State(rgb=np.stack([_qimg] * 3).astype(np.float32), kind="mono_rgb", target="M28", linear=False,
                     done=["combine", "balance", "stretch", "curves"])
    _qv = next(v for v in _fnc.TOOLS["cluster_core"].variants(_qs) if v.recommended)
    _qo = _fnc.TOOLS["cluster_core"].apply(_qs, _qv.params).rgb[0]
    _qc = (np.abs(_qx - 300) < 25) & (np.abs(_qy - 300) < 25)
    _qglow_after = float(np.median(_qo[_qc]))
    _qstar = float(np.mean((_qo - _fnc.TOOLS["cluster_core"].glow(_qo[None])[0])[_qc & (_qst > 0.2)]))
    _qnext = dict((t.key, st) for t, st in _fnc.workflow(_qs)).get("cluster_core")
    check(_qglow_after < 0.45 and _qstar > 0.2 and _qnext == "next",
          f"Resolve the Cluster Core lowers a globular's blended glow (core {float(np.median(_qimg[_qc])):.2f} -> "
          f"{_qglow_after:.2f}) and keeps its stars ({_qstar:.2f} above the glow); recommended for a star cluster")
    # the colour-preserving stretch keeps a bright red star red to its centre (its red was cut first and the centre
    # turned yellow-white - M28)
    _ry, _rx = np.mgrid[:200, :200].astype(np.float32)
    _rst = np.exp(-((_rx - 100) ** 2 + (_ry - 100) ** 2) / (2 * 2.0 ** 2))
    _rimg = (np.stack([0.02 + 3.0 * _rst, 0.02 + 1.6 * _rst, 0.02 + 1.0 * _rst])
             + np.random.default_rng(0).normal(0, 0.002, (3, 200, 200))).astype(np.float32)
    _rs6 = _fnc.State(rgb=_rimg, kind="osc", target="M28", linear=True)
    _rv6 = next(v for v in _fnc.TOOLS["stretch"].variants(_rs6) if v.recommended)
    _ro6 = _fnc.TOOLS["stretch"].apply(_rs6, _rv6.params).rgb
    _rc6, _rr6 = _ro6[:, 100, 100], _ro6[:, 100, 103]
    check(abs(float(_rc6[1] / _rc6[0]) - float(_rr6[1] / _rr6[0])) < 0.08 and float(_rc6[2]) < 0.6,
          f"a bright red star stays red to its centre when brightened (centre {np.round(_rc6, 2).tolist()}, "
          f"edge {np.round(_rr6, 2).tolist()})")
    # stars kept aside by page 3 but not taken out (a star cluster): brightened colour-preserving, with its stars
    _hs7 = _fnc.State(rgb=np.full((3, 64, 64), 0.05, np.float32), stars=np.zeros((3, 64, 64), np.float32), kind="mono_rgb",
                      target="M28", linear=True, stars_out=True, stars_hidden=True)
    _hm7 = _fnc.TOOLS["stretch"].recommended_method(_hs7)
    _hs7.stars_hidden = False
    _hm8 = _fnc.TOOLS["stretch"].recommended_method(_hs7)
    check(_hm7 == "colour" and _hm8 == "ghs", f"stars from page 3 still in the picture are brightened colour-preserving "
                                              f"({_hm7}); a starless picture with the stars out gets GHS ({_hm8})")
    _hs7.stars_hidden = True
    _hs7.stars = np.zeros((3, 64, 64), np.float32); _hs7.stars[:, 30:33, 30:33] = 0.6; _hs7.stars_linear = True
    _hv7 = [v.params for v in _fnc.TOOLS["stretch"].variants(_hs7)]
    _hv8 = [v.params for v in _fnc.TOOLS["stretch"].variants(_fnc.merge_hidden_stars(_hs7))]
    check(repr(_hv7) == repr(_hv8), "Brighten's versions are measured on the picture with page 3's stars in, "
                                    "as it will be brightened")
    # small background galaxies around a planetary go to the stars image (the nebula's colours turned the Helix's
    # teal); the nebula and a faint, flat knot of its shell stay
    _rg = np.random.default_rng(7)
    _gy, _gx = np.mgrid[:600, :900].astype(np.float32)
    _gneb = 0.6 * np.exp(-((_gx - 450) ** 2 + (_gy - 300) ** 2) / (2 * 110 ** 2))
    _ggal = 0.25 * np.exp(-(((_gx - 120) / 2.5) ** 2 + ((_gy - 100) / 1.5) ** 2) / 2)
    _gknot = 0.012 * np.exp(-((_gx - 780) ** 2 + (_gy - 480) ** 2) / (2 * 4 ** 2))
    _gimg = np.stack([(0.02 + _gneb * t_ + _ggal * g_ + _gknot) for t_, g_ in ((0.3, 1.0), (1.0, 0.8), (0.8, 0.6))])
    _gimg = (_gimg + _rg.normal(0, 0.001, _gimg.shape)).astype(np.float32)
    _gr, _gs, _gn = _fnc.small_objects_to_stars(_gimg, np.zeros_like(_gimg))
    check(_gn == 1 and float(_gs[:, 90:111, 105:136].sum()) > 0.8 * float(_ggal[90:111, 105:136].sum() * 2.4)
          and abs(float((_gr - _gimg)[:, 200:400, 350:550].mean())) < 1e-4
          and abs(float((_gr - _gimg)[:, 470:491, 770:791].mean())) < 1e-4,
          f"a small galaxy beside a planetary is moved to the stars image ({_gn}); the nebula and a faint knot stay")
    # star colour balance on linear data with page 3's stars added back (stars far above 1): still measured
    _sr = np.random.default_rng(5)
    _sy, _sx = np.mgrid[:300, :300].astype(np.float32)
    _simg = np.full((3, 300, 300), 0.02, np.float32) + _sr.normal(0, 0.001, (3, 300, 300)).astype(np.float32)
    for _k in range(60):
        _cx, _cy = _sr.uniform(15, 285, 2)
        _g = np.exp(-((_sx - _cx) ** 2 + (_sy - _cy) ** 2) / (2 * 1.5 ** 2)) * _sr.uniform(2, 40)
        _simg += np.stack([_g * 1.3, _g, _g * 0.8])
    _sf = _fnc._star_colour_factors(_simg)
    check(_sf is not None and abs(_sf[0] - 1.3) < 0.1 and abs(_sf[1] - 0.8) < 0.1,
          f"Balance Colours measures the stars on linear data whose stars run far above 1 ({_sf})")
    # no OBJECT in the header (some ASIAIR files): the target comes from the file name or the ASIAIR folder
    _tfn = [_stk.target_from_name(n) for n in (
        "/x/Light_M51_180.0s_Bin1_6200MC_gain0_20240525-212957_-6.6C_C11_0001.fit",
        "/x/Light_NGC 7000_300.0s_Bin1_2600MM_Ha_gain100_20240901-010101_-10.0C_0002.fit",
        "/a/Autorun/Light/IC1396/Light_120.0s_Bin1_0001.fit", "/x/M51_2024-05-25_L_300s_0001.fits",
        "/x/Flat_Ha_1.0s_Bin1_0001.fit", "/x/Light_180.0s_Bin1_0001.fit")]
    check(_tfn == ["M51", "NGC 7000", "IC1396", "M51", None, None],
          f"no target in the header: it is read from the file name (ASIAIR, Seestar, N.I.N.A.) or the ASIAIR folder ({_tfn})")
    # a placeholder OBJECT ('Snapshot' - Telescope Live comet frames) gives way to a folder named after the target
    _tsd = _sfd / "C_2025_R2_SWAN"
    _tsd.mkdir(exist_ok=True)
    core.write_fits(_tsd / "2026-04-16_05-31-24_0006_Snapshot_c_d_r.fits", np.zeros((40, 40), np.uint16),
                    {"IMAGETYP": "LIGHT", "OBJECT": "Snapshot", "EXPTIME": 15.0}, bitpix=16)
    _tsi = _stk.read_info(_tsd / "2026-04-16_05-31-24_0006_Snapshot_c_d_r.fits")
    import targets as _tgm
    check(_tsi.target == "C_2025_R2_SWAN" and _tgm.is_comet(_tsi.target),
          f"a placeholder OBJECT ('Snapshot') gives way to the folder's target name ({_tsi.target}, a comet)")
    # a FITS file whose binary data was turned into text (each byte > 127 -> U+FFFD) is caught plainly
    _okf = _sfd / "intact.fits"
    core.write_fits(_okf, (_rs.random((200, 200)) * 60000).astype(np.uint16), {"IMAGETYP": "Light Frame"}, bitpix=16)
    _raw_b = _okf.read_bytes()
    _badf = _sfd / "mangled.fits"
    _badf.write_bytes(_raw_b[:2880] + _raw_b[2880:].decode("utf-8", "replace").encode("utf-8"))
    try:
        with open(_badf, "rb") as _fh:
            core.read_layout(_fh)
        _mang = ""
    except ValueError as _e:
        _mang = str(_e)
    with open(_okf, "rb") as _fh:
        _okl = core.read_layout(_fh)
    check("converted to text" in _mang and _okl.shape == (200, 200),
          "a FITS file damaged by a text-mode download is reported as damaged (intact files read normally)")
    # ready-made data (Telescope Live and the like) is already calibrated: nothing is applied again
    _tl = []
    for _i in range(14):
        _fp = _sfd / f"CHI-1-CCD_2019-12-02T05-36-{_i:02d}_NGC_2359_Halpha_300s_{_i}_cal.fits"
        core.write_fits(_fp, np.full((_sh, _sw), 1000, np.uint16),
                        {"IMAGETYP": "Light Frame", "FILTER": "L", "EXPTIME": 60.0, "OBJECT": "Field0",
                         "INSTRUME": "Test Camera", "PROC": "True", "COMMENT": "Processed 0.5 on 2020-07-10"},
                        bitpix=16)
        _tl.append(_stk.LightFrame(_stk.read_info(_fp)))
    _tlg = _stk.LightGroup("Field0", "L", "Test Camera", "TestRig", _tl)
    _tlp = _stk.plan_group(_tlg, [], sky_pool=_pool + [f.info for f in _tl])
    _rawp = _stk.plan_group(_grp, [], sky_pool=[f.info for f in _tl])
    check(_tl[0].info.precal and "Telescope Live" in _tl[0].info.precal and _tlp.precal and not _tlp.warnings
          and _tlp.flat is None and _tlp.dark is None and _tlp.bias is None
          and _stk.sky_flat_frames(_grp, [f.info for f in _tl]) == [] and _lfs[0].info.precal is None
          and (_rawp.flat is None or not _rawp.flat.sky),
          f"ready-made data is recognised as already calibrated ({_tl[0].info.precal}): no calibration, "
          "no SkyFlat, no missing-calibration warnings - and it is never used to make a SkyFlat")
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

    # --- flats taken every session: each night's lights are calibrated with that night's flat -------------
    _nr = np.random.default_rng(8)
    _NH, _NW = 300, 400
    _nyy, _nxx = np.mgrid[:_NH, :_NW].astype(np.float32)
    _nvig = 1 - 0.25 * (((_nxx - 200) / 200) ** 2 + ((_nyy - 150) / 150) ** 2)
    def _dust(cx, cy):
        return 1 - 0.3 * np.exp(-(((_nxx - cx) ** 2 + (_nyy - cy) ** 2) / (2 * 12.0 ** 2)))
    _nflat = {"2026-10-17": _nvig * _dust(120, 100), "2026-10-18": _nvig * _dust(280, 200)}   # the dust moved
    _nst = [(_nr.uniform(10, 390), _nr.uniform(10, 290), _nr.uniform(2000, 20000)) for _ in range(150)]
    _nd = session / "nightflats"
    _nd.mkdir(exist_ok=True)
    for _night, _day in (("2026-10-17", "2026-10-18"), ("2026-10-18", "2026-10-19")):
        for _i in range(6):
            _im = np.full((_NH, _NW), 1500.0, np.float32)
            for _x, _y, _f in _nst:
                _x += _i * 0.7
                _im += _f * np.exp(-((_nxx - _x) ** 2 + (_nyy - _y) ** 2) / (2 * 1.6 ** 2))
            _im = 300 + _im * _nflat[_night] + _nr.normal(0, 5, _im.shape)
            core.write_fits(_nd / f"L_{_night}_{_i}.fits", np.clip(_im, 0, 65535).astype(np.uint16),
                            {"IMAGETYP": "Light Frame", "OBJECT": "NF", "FILTER": "L", "EXPTIME": 60.0, "GAIN": 100,
                             "INSTRUME": "Cam", "DATE-OBS": f"{_day}T02:{_i:02d}:00"}, bitpix=16)
        for _i in range(5):                                      # flats after the session, before sunrise
            _fl = 300 + 30000 * _nflat[_night] + _nr.normal(0, 30, (_NH, _NW))
            core.write_fits(_nd / f"F_{_night}_{_i}.fits", np.clip(_fl, 0, 65535).astype(np.uint16),
                            {"IMAGETYP": "Flat Frame", "FILTER": "L", "EXPTIME": 1.0, "GAIN": 100, "INSTRUME": "Cam",
                             "DATE-OBS": f"{_day}T05:{_i:02d}:00"}, bitpix=16)
        for _i in range(5):
            core.write_fits(_nd / f"B_{_night}_{_i}.fits", np.clip(300 + _nr.normal(0, 5, (_NH, _NW)), 0, 65535).astype(np.uint16),
                            {"IMAGETYP": "Bias Frame", "EXPTIME": 0.0, "GAIN": 100, "INSTRUME": "Cam",
                             "DATE-OBS": f"{_day}T06:{_i:02d}:00"}, bitpix=16)
    _ninfos = [_stk.read_info(_p) for _p in sorted(_nd.glob("*.fits"))]
    _nlights = [_stk.LightFrame(i) for i in _ninfos if i.type == "light"]
    _nitems = _stk.group_calibration([i for i in _ninfos if i.type != "light"])
    _ng = _stk.group_lights(_nlights, _stk.Rigs())[0]
    _ng.plan = _stk.plan_group(_ng, _nitems)
    for _it, _hb, _hd in [(_ng.plan.bias, None, None), (_ng.plan.dark, None, None)] + _ng.plan.flat_sets():
        if _it is not None:
            _stk.build_master(_it, _hb, _hd, 3.0)
    _nres = _stk.stack_group(_ng, _stk.StackOptions(work_dir=str(session / "nf_work"), formats=("fits",)))
    _nm = _nres.data
    _nbg = _nm - _stk._smooth(_nm, 25.0)
    _spots = [float(np.median(_nm[(np.hypot(_nxx - cx, _nyy - cy) < 8)]) / np.median(_nm)) for cx, cy in ((120, 100), (280, 200))]
    check(len(_ng.plan.night_flats) == 2 and any("Flats per night" in l for l in _nres.log)
          and all(abs(v - 1) < 0.03 for v in _spots),
          f"flats taken every session: each night's lights use that night's flat (nights "
          f"{sorted(_ng.plan.night_flats)}), so dust that moved between nights leaves no shadow "
          f"(dust spots at {[round(v, 3) for v in _spots]} of the background)")

    # --- Even out the sky between frames (local normalization): frames whose gradients change through the night
    # (moon, haze, light pollution) are matched to the reference frame
    _lr = np.random.default_rng(21)
    _LH, _LW = 320, 420
    _lyy, _lxx = np.mgrid[:_LH, :_LW].astype(np.float32)
    _lst = [(_lr.uniform(10, 410), _lr.uniform(10, 310), _lr.uniform(1500, 15000)) for _ in range(160)]
    _lneb = 300 * np.exp(-(((_lxx - 210) / 90) ** 2 + ((_lyy - 160) / 60) ** 2))
    _ld = session / "localnorm"
    _ld.mkdir(exist_ok=True)
    _ltd = _lyy - (0.5 * _lxx + 40)
    for _i in range(12):
        _ang = _lr.uniform(0, 2 * np.pi)
        _grad = 300 * _lr.uniform(0.3, 1.0) * ((np.cos(_ang) * (_lxx - 210) / 210 + np.sin(_ang) * (_lyy - 160) / 160) ** 2)   # changes every frame
        _im = 1000 + _lneb + _grad + _lr.normal(0, 12, _lneb.shape)
        for _x, _y, _f in _lst:
            _im += _f * np.exp(-((_lxx - _x - _i * 0.4) ** 2 + (_lyy - _y) ** 2) / (2 * 1.6 ** 2))
        if _i == 5:                                                  # a satellite trail in one frame
            _im += 60 * np.exp(-(_ltd ** 2) / 2.0)
        core.write_fits(_ld / f"L_{_i:02d}.fits", np.clip(_im, 0, 65535).astype(np.uint16),
                        {"IMAGETYP": "Light Frame", "OBJECT": "LN", "FILTER": "L", "EXPTIME": 60.0,
                         "DATE-OBS": f"2026-10-18T02:{_i:02d}:00"}, bitpix=16)
    _linfo = [_stk.read_info(_p) for _p in sorted(_ld.glob("*.fits"))]

    def _lstack(on):
        _g = _stk.LightGroup("LN", "L", "Cam", "rig", [_stk.LightFrame(_x) for _x in _linfo])
        _g.plan = _stk.CalPlan()
        _r = _stk.stack_group(_g, _stk.StackOptions(work_dir=str(session / "ln_work"), formats=("fits",), local_norm=on))
        _ref = core.read_fits(_r.ref_path)[0].astype(np.float32) if hasattr(core, "read_fits") else None
        _diff = _stk.background_map((_r.data - _ref)[10:-10, 10:-10], 40)
        return float(np.std(_diff)), _r.log
    _t0, _ = _lstack(False)
    _t1, _llog = _lstack(True)
    check(_t1 < 0.25 * _t0 and any("Even out the sky" in l for l in _llog),
          f"Even out the sky between frames: frames whose gradients change through the night are matched to the reference frame "
          f"everywhere (background mismatch {_t0:.1f} -> {_t1:.1f} ADU)")
    # several frames calibrated and lined up at once (fast computers): the master is exactly the same
    _fw0 = _stk.frame_workers

    def _pstack(nw):
        _stk.frame_workers = lambda *a, **k: nw
        try:
            _g = _stk.LightGroup("LN", "L", "Cam", "rig", [_stk.LightFrame(_x) for _x in _linfo])
            _g.plan = _stk.CalPlan()
            return _stk.stack_group(_g, _stk.StackOptions(work_dir=str(session / f"par_work{nw}"), formats=("fits",)))
        finally:
            _stk.frame_workers = _fw0
    _pa, _pb = _pstack(1), _pstack(3)
    check(_pa.used == _pb.used == len(_linfo) and np.array_equal(_pa.data, _pb.data, equal_nan=True)
          and _fw0(100, 100, 1) >= 1,
          f"stacking several frames at once gives exactly the same master ({_pa.used} and {_pb.used} frames; "
          f"{_fw0(6388, 9576, 1)} at a time for a 61-megapixel frame here)")

    # ... and a bright target (a planetary nebula) is left out of the sky estimate, so it leaves no block-shaped
    # border around itself in the stack (M27 from Telescope Live)
    _oy, _ox = np.mgrid[:600, :800].astype(np.float32)
    _orr = np.hypot(_ox - 400, _oy - 300)
    _oim = (500 + 400 * np.clip(1 - (_orr / 130) ** 2, 0, 1) ** 0.5
            + np.random.default_rng(4).normal(0, 8, (600, 800))).astype(np.float32)
    _om = _stk.object_mask(_oim, 64)
    _ob0 = float(_stk.background_map(_oim, 64)[300, 400])
    _ob1 = float(_stk.background_map(_oim, 64, _om)[300, 400])
    check(0.03 < _om.mean() < 0.4 and abs(_ob1 - 500) < 15 and _ob0 > 600,
          f"Even out the sky: a bright target is left out of the sky estimate (sky under it {_ob0:.0f} -> {_ob1:.0f}, "
          f"true 500; {100 * _om.mean():.0f}% of the picture left out)")
    # ... also in a dense star field: the stars are not taken for the target (M27: the mask covered over half the
    # picture in Ha and the whole of it - so none - in SII)
    _sfr = np.random.default_rng(6)
    _sy, _sx = np.mgrid[:1200, :1600].astype(np.float32)
    _srr = np.hypot(_sx - 800, _sy - 600)
    _sim = 500 + 300 * np.exp(-(_srr / 110) ** 2) + _sfr.normal(0, 8, (1200, 1600))
    for _ in range(1500):
        _x, _y, _f = _sfr.uniform(0, 1600), _sfr.uniform(0, 1200), _sfr.uniform(30, 3000)
        _x0, _y0 = int(_x), int(_y)
        _sl = (slice(max(0, _y0 - 6), _y0 + 7), slice(max(0, _x0 - 6), _x0 + 7))
        _sim[_sl] += _f * np.exp(-((_sx[_sl] - _x) ** 2 + (_sy[_sl] - _y) ** 2) / (2 * 1.8 ** 2))
    _smk = _stk.object_mask(_sim.astype(np.float32), 100)
    _sbg = float(_stk.background_map(_sim.astype(np.float32), 100, _smk)[600, 800])
    check(0.01 < _smk.mean() < 0.25 and abs(_sbg - 500) < 25,
          f"Even out the sky in a dense star field: the target is left out, the stars are not "
          f"({100 * _smk.mean():.0f}% left out; sky under the target {_sbg:.0f}, true 500)")
    # --- 2x drizzle: undersampled (blocky) stars from dithered frames come out finer than in the regular stack
    _dr = np.random.default_rng(31)
    _DH, _DW = 200, 260
    _dst = [(_dr.uniform(12, 248), _dr.uniform(12, 188), _dr.uniform(3000, 20000)) for _ in range(90)]
    _dd = session / "drizzle"
    _dd.mkdir(exist_ok=True)
    _sub = 8                                               # render each pixel from 8x8 sub-samples: real pixels
    _fy, _fx = (np.mgrid[:_DH * _sub, :_DW * _sub].astype(np.float32) + 0.5) / _sub - 0.5
    for _i in range(16):
        _ox, _oy = _dr.uniform(-3, 3), _dr.uniform(-3, 3)     # dither
        _hi = np.zeros(_fy.shape, np.float32)
        for _x, _y, _f in _dst:
            _x += _ox; _y += _oy
            _sl = (slice(max(0, int((_y - 4) * _sub)), int((_y + 5) * _sub)), slice(max(0, int((_x - 4) * _sub)), int((_x + 5) * _sub)))
            _hi[_sl] += _f * np.exp(-((_fx[_sl] - _x) ** 2 + (_fy[_sl] - _y) ** 2) / (2 * 0.45 ** 2))
        _im = 800 + _hi.reshape(_DH, _sub, _DW, _sub).mean(axis=(1, 3)) + _dr.normal(0, 6, (_DH, _DW))
        core.write_fits(_dd / f"D_{_i:02d}.fits", np.clip(_im, 0, 65535).astype(np.uint16),
                        {"IMAGETYP": "Light Frame", "OBJECT": "DZ", "FILTER": "L", "EXPTIME": 60.0,
                         "DATE-OBS": f"2026-10-18T02:{_i:02d}:00"}, bitpix=16)
    _dinfo = [_stk.read_info(_p) for _p in sorted(_dd.glob("*.fits"))]

    def _dstack(on):
        _g = _stk.LightGroup("DZ", "L", "Cam", "rig", [_stk.LightFrame(_x) for _x in _dinfo])
        _g.plan = _stk.CalPlan()
        _r = _stk.stack_group(_g, _stk.StackOptions(work_dir=str(session / "dz_work"), formats=("fits",), drizzle=on))
        return _r

    def _fwhm(m, scale):
        pts, pk = _stk.find_stars(m.astype(np.float32), 1, 60)
        out = []
        for x, y in pts[:40]:
            x, y = int(round(x)), int(round(y))
            b = m[y - 6 * scale:y + 6 * scale + 1, x - 6 * scale:x + 6 * scale + 1].astype(np.float64)
            if b.shape != (12 * scale + 1, 12 * scale + 1):
                continue
            b = b - np.median(m)
            out.append(2 * np.sqrt(np.log(2) * b.sum() / (np.pi * b.max())) / scale)   # in reference pixels
        return float(np.median(out))
    _r0 = _dstack(False)
    _r1 = _dstack(True)
    _f0, _f1 = _fwhm(_r0.data, 1), _fwhm(_r1.data, 2)
    check(_r1.data.shape == (2 * _r0.data.shape[0], 2 * _r0.data.shape[1]) and _f1 < 0.85 * _f0
          and any("Drizzle 2x" in l for l in _r1.log),
          f"2x drizzle: dithered, undersampled frames give a double-size master with finer stars "
          f"(star size {_f0:.2f} -> {_f1:.2f} px of the original pixels)")
    _drift = [np.array([[1.0, 0, 0.4 * k], [0, 1.0, 0.0]]) for k in range(12)]   # slow drift, no dithering
    _dr_no = _stk.dither_report(_drift, [3.5] * 12)
    check(any(l.startswith("TIP: drizzle") and "dithered" in l for l in _r0.log) and not _dr_no["dithered"]
          and not any(l.startswith("TIP: drizzle") for l in _r1.log),
          f"dithered frames with small stars get the tip to try drizzle (and its longer stacking); a slow drift "
          f"is not taken for dithering ({_r0.dither})")

    # --- alt-az field rotation without darks: the camera's fixed pattern is taken out (no swirls) ---
    _ar = np.random.default_rng(11)
    _AH, _AW = 400, 520
    _ayy, _axx = np.mgrid[:_AH, :_AW].astype(np.float32)
    _ast = [(_ar.uniform(-300, 300), _ar.uniform(-300, 300), _ar.uniform(400, 6000)) for _ in range(500)]
    _apat = _ar.normal(0, 25, (_AH, _AW)).astype(np.float32)          # the camera's own fixed pattern
    _ad = session / "altaz_frames"
    _ad.mkdir(exist_ok=True)
    for _i in range(16):
        _a = np.radians(0.6 * _i)                                       # ~9 degrees of field rotation
        _c, _s = np.cos(_a), np.sin(_a)
        _im = 600 + _apat + _ar.normal(0, 8, (_AH, _AW))
        for _x, _y, _f in _ast:
            X, Y = _c * _x - _s * _y + _AW / 2, _s * _x + _c * _y + _AH / 2
            if 6 < X < _AW - 6 and 6 < Y < _AH - 6:
                _sl = (slice(int(Y) - 5, int(Y) + 6), slice(int(X) - 5, int(X) + 6))
                _im[_sl] += _f * np.exp(-((_axx[_sl] - X) ** 2 + (_ayy[_sl] - Y) ** 2) / (2 * 1.6 ** 2))
        core.write_fits(_ad / f"A_{_i:02d}.fits", np.clip(_im, 0, 65535).astype(np.uint16),
                        {"IMAGETYP": "Light Frame", "OBJECT": "AltAz", "FILTER": "L", "EXPTIME": 10.0,
                         "DATE-OBS": f"2026-10-18T02:{_i:02d}:00"}, bitpix=16)
    _ainfo = [_stk.read_info(_p) for _p in sorted(_ad.glob("*.fits"))]

    def _astack(on):
        _g = _stk.LightGroup("AltAz", "L", "Cam", "rig", [_stk.LightFrame(_x) for _x in _ainfo])
        _g.plan = _stk.CalPlan()
        _r = _stk.stack_group(_g, _stk.StackOptions(work_dir=str(session / "altaz_work"), formats=("fits",), pattern=on))
        _m = _r.data
        _bgm = (np.hypot(_axx - _AW / 2, _ayy - _AH / 2) > 80) & (np.hypot(_axx - _AW / 2, _ayy - _AH / 2) < 180)
        _hp = _m - _stk._smooth(_m, 3.0)
        _bg = _bgm & (np.abs(_hp) < 5 * np.median(np.abs(_hp[_bgm])))
        return float(np.std(_hp[_bg])), _r.log
    _off, _ = _astack(False)
    _on, _alog = _astack(True)
    check(_on < 0.75 * _off and any("Field rotation" in l for l in _alog),
          f"alt-az frames without darks: the camera's fixed pattern is measured and taken out, so it does not stack "
          f"into swirls (fine-scale background {_off:.1f} -> {_on:.1f} ADU)")

    # --- a PixInsight project opens in SubZeroAstro: images PixInsight stored inside the project are decoded,
    # unchanged ones are found by their file (also after the project moved to another computer); masks left out
    import base64 as _b64
    import pixinsight_project as _pp
    _pid = session / "pi" / "Test.pxiproject"
    (_pid / "project.data").mkdir(parents=True, exist_ok=True)
    (_pid / "project.data" / "TST-000001").write_bytes(_b64.b64decode(
        "ZWdhbWl3YXIAAAEAAQAACAEAAAAAAAAAAAAAAGgYAABQEAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAtgAAAAAAAAAI4hgAAAAAAEbr"
        "MvJlin8sAAAAAAAAAAAotS/9oAjiGADUAAAwAAD/AP8ABADGtmt4chCGoJ8xBP0lwx5gAUwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQ"
        "AkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwA"
        "AAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAk0AAAgAAQAEYh0IAa0AAAAAAAAACOIYAAAAAAAcGf9Ydq4I1QAAAAAA"
        "AAAAKLUv/aAI4hgAjAAAIAD/AP8CAJXna3hyEYY9GAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA"
        "/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85"
        "EAJMAAAIAAEA/P85EAJNAAAIAAEABGIdCAGtAAAAAAAAAAjiGAAAAAAAHBn/WHauCNUAAAAAAAAAACi1L/2gCOIYAIwAACAA/wD/"
        "AgCV52t4chGGPRgCTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAA"
        "CAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTQAACAAB"
        "AARiHQgBrQAAAAAAAAAI4hgAAAAAABwZ/1h2rgjVAAAAAAAAAAAotS/9oAjiGACMAAAgAP8A/wIAledreHIRhj0YAkwAAAgAAQD8"
        "/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQ"
        "AkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAk0AAAgAAQAEYh0IAa0AAAAAAAAACOIYAAAA"
        "AAAcGf9Ydq4I1QAAAAAAAAAAKLUv/aAI4hgAjAAAIAD/AP8CAJXna3hyEYY9GAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAI"
        "AAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA"
        "/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJNAAAIAAEABGIdCAGtAAAAAAAAAAjiGAAAAAAAHBn/WHauCNUAAAAAAAAAACi1"
        "L/2gCOIYAIwAACAA/wD/AgCV52t4chGGPRgCTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORAC"
        "TAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAA"
        "CAABAPz/ORACTQAACAABAARiHQgBrQAAAAAAAAAI4hgAAAAAABwZ/1h2rgjVAAAAAAAAAAAotS/9oAjiGACMAAAgAP8A/wIAledr"
        "eHIRhj0YAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8"
        "/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAk0AAAgAAQAEYh0I"
        "Aa0AAAAAAAAACOIYAAAAAAAcGf9Ydq4I1QAAAAAAAAAAKLUv/aAI4hgAjAAAIAD/AP8CAJXna3hyEYY9GAJMAAAIAAEA/P85EAJM"
        "AAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAI"
        "AAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJNAAAIAAEABGIdCAGtAAAAAAAAAAjiGAAAAAAAHBn/"
        "WHauCNUAAAAAAAAAACi1L/2gCOIYAIwAACAA/wD/AgCV52t4chGGPRgCTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/"
        "ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORAC"
        "TAAACAABAPz/ORACTAAACAABAPz/ORACTQAACAABAARiHQgBrQAAAAAAAAAI4hgAAAAAABwZ/1h2rgjVAAAAAAAAAAAotS/9oAji"
        "GACMAAAgAP8A/wIAledreHIRhj0YAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgA"
        "AQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8"
        "/zkQAk0AAAgAAQAEYh0IAa0AAAAAAAAACOIYAAAAAAAcGf9Ydq4I1QAAAAAAAAAAKLUv/aAI4hgAjAAAIAD/AP8CAJXna3hyEYY9"
        "GAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJM"
        "AAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJNAAAIAAEABGIdCAGtAAAA"
        "AAAAAAjiGAAAAAAAHBn/WHauCNUAAAAAAAAAACi1L/2gCOIYAIwAACAA/wD/AgCV52t4chGGPRgCTAAACAABAPz/ORACTAAACAAB"
        "APz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/"
        "ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTQAACAABAARiHQgBrQAAAAAAAAAI4hgAAAAAABwZ/1h2rgjV"
        "AAAAAAAAAAAotS/9oAjiGACMAAAgAP8A/wIAledreHIRhj0YAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwA"
        "AAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgA"
        "AQD8/zkQAkwAAAgAAQD8/zkQAk0AAAgAAQAEYh0IAa0AAAAAAAAACOIYAAAAAAAcGf9Ydq4I1QAAAAAAAAAAKLUv/aAI4hgAjAAA"
        "IAD/AP8CAJXna3hyEYY9GAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85"
        "EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJMAAAIAAEA/P85EAJN"
        "AAAIAAEABGIdCAGtAAAAAAAAAAjiGAAAAAAAHBn/WHauCNUAAAAAAAAAACi1L/2gCOIYAIwAACAA/wD/AgCV52t4chGGPRgCTAAA"
        "CAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAAB"
        "APz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTAAACAABAPz/ORACTQAACAABAARiHQgBsAAAAAAAAAAI"
        "4hgAAAAAAJK/Q+tm7rW1AAAAAAAAAAAotS/9oAjiGACMAAAgAP8A/wIAledreHIRhj0YAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQ"
        "AkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwA"
        "AAgAAQD8/zkQAkwAAAgAAQD8/zkQAkwAAAgAAQD8/zkQAmUAAAgAAgBjCHrwTOrgCg=="))
    _pim = (np.random.default_rng(3).random((3, 60, 80)) * 0.5).astype(np.float32)
    import xisf as _xs_pi
    _xs_pi.write_xisf(session / "pi" / "master_M42.xisf", _pim, [("OBJECT", "M42", ""), ("FILTER", "L-Pro", "")])
    (_pid / "project.xosm").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><xosm version="1.2" xmlns="http://www.pixinsight.com/xosm">'
        '<ImageWindow format="XISF" filePath="/Users/someone/elsewhere/master_M42.xisf">'
        '<MainView id="master_M42"><initialProcessing><instance class="GradientCorrection"/></initialProcessing>'
        '</MainView></ImageWindow>'
        '<ImageWindow format="XISF" filePath="/Users/someone/elsewhere/master_M42.xisf">'
        '<MainView id="MasterLight_crop_mask"><image src="@project_data_dir/TST-000001"/></MainView></ImageWindow>'
        '</xosm>', encoding="utf-8")
    try:
        _pmask = _pp.decode_rawimage((_pid / "project.data" / "TST-000001").read_bytes())
        _pfiles, _pnotes = _pp.export_images(_pid, session / "pi" / "out")
        import prepare as _prp_pi
        _pm = _prp_pi.load_master(_pfiles[0]) if _pfiles else None
        _pi_ok = (_pmask.shape == (1, 4176, 6248) and bool(_pmask[0, 1, 1] > 0.5) and bool(_pmask[0, 100, 100] == 0)
                  and len(_pfiles) == 1 and _pm is not None and _pm.target == "M42"
                  and np.allclose(_pm.data / max(float(_pm.data.max()), 1e-9), _pim / _pim.max(), atol=1e-3)
                  and any("mask" in n_ for n_ in _pnotes))
        _pi_detail = f"{len(_pfiles)} picture, notes: {_pnotes}"
    except Exception as _pe:  # noqa: BLE001
        _pi_ok, _pi_detail = False, repr(_pe)
    check(_pi_ok, f"a PixInsight project opens: stored images decoded, moved files found, masks left out ({_pi_detail})")

    # --- meridian flip: the optics' distortion turns with the camera; the stack corrects it --------
    _fr = np.random.default_rng(3)
    _FH = 600
    _fc = _FH / 2
    _fst = [(_fr.uniform(-330, 330), _fr.uniform(-330, 330), _fr.uniform(500, 9000)) for _ in range(380)]

    def _fcam(x, y, flipped):
        if flipped:
            x, y = -x, -y
        ox, oy = x - 140, y + 90                      # optical axis off the sensor centre
        q = 1 + 4.5e-8 * (ox * ox + oy * oy)
        return x * q + _fc, y * q + _fc
    _fd = session.parent / "flip_frames"
    _fd.mkdir(exist_ok=True)
    _fyy, _fxx = np.mgrid[:_FH, :_FH].astype(np.float32)
    for _i in range(8):
        _dx, _dy = _fr.uniform(-3, 3), _fr.uniform(-3, 3)
        _im = 800 + _fr.normal(0, 10, (_FH, _FH))
        for _x, _y, _a in _fst:
            X, Y = _fcam(_x + _dx, _y + _dy, _i >= 4)
            if 5 < X < _FH - 5 and 5 < Y < _FH - 5:
                _sl = (slice(int(Y) - 5, int(Y) + 6), slice(int(X) - 5, int(X) + 6))
                _im[_sl] += _a * np.exp(-((_fxx[_sl] - X) ** 2 + (_fyy[_sl] - Y) ** 2) / (2 * 1.5 ** 2))
        core.write_fits(_fd / f"L_{_i:02d}.fits", np.clip(_im, 0, 65535).astype(np.uint16),
                        {"IMAGETYP": "Light Frame", "OBJECT": "Flip", "FILTER": "L", "EXPTIME": 60.0,
                         "DATE-OBS": f"2026-10-18T02:{_i:02d}:00"}, bitpix=16)
    _finfo = [_stk.read_info(_p) for _p in sorted(_fd.glob("*.fits"))]

    def _fstack():
        _g = _stk.LightGroup("Flip", "L", "Cam", "rig", [_stk.LightFrame(_x) for _x in _finfo])
        _g.plan = _stk.CalPlan()
        _r = _stk.stack_group(_g, _stk.StackOptions(work_dir=str(session.parent / "flip_work"), formats=("fits",)))
        _m = _r.data[0] if _r.data.ndim == 3 else _r.data
        _pts, _pk = _stk.find_stars(_m, 1, 300)
        _rad = np.hypot(_pts[:, 0] - _fc, _pts[:, 1] - _fc)

        def _sharp(p):
            x, y = int(round(p[0])), int(round(p[1]))
            b = _m[y - 3:y + 4, x - 3:x + 4] - np.median(_m)
            return float(b.max() / max(b.sum(), 1e-6)) if b.shape == (7, 7) else np.nan
        _sh = np.array([_sharp(p) for p in _pts])
        return float(np.nanmedian(_sh[_rad < 120])), float(np.nanmedian(_sh[_rad > 220])), _r.log
    _keep_dm = _stk.distortion_model
    _stk.distortion_model = lambda *a, **k: None
    try:
        _c0, _e0, _ = _fstack()
    finally:
        _stk.distortion_model = _keep_dm
    _c1, _e1, _flog = _fstack()
    check(_e1 > 0.97 * _c1 and _e1 > _e0 + 0.003 and any("distortion corrected" in l for l in _flog),
          f"after a meridian flip the stack corrects the field distortion: edge stars as sharp as the centre "
          f"({_e1:.3f} vs {_c1:.3f}; without it {_e0:.3f})")

    # --- plate solving retries when the header's field size is wrong (binning, reducer) -----------
    import prepare as _pp
    import tools as _tl

    class _StubAstap:
        exe = None

        def __init__(self):
            self.calls = []

        def solve(self, path, ra, dec, fov, radius=30.0, cancel=None):
            self.calls.append(fov)
            if fov and abs(fov - 0.355) < 0.03 or fov == 0.35:
                return [("CRVAL1", 190.0, ""), ("CRVAL2", -11.6, "")]
            raise _tl.ToolError("ASTAP could not solve the image: No solution found!")
    _sa = _StubAstap()
    _cards, _note = _pp.solve_with_retries(_sa, "x.fits", 190.0, -11.6, 0.71, 30.0)
    try:
        _pp.solve_with_retries(_StubAstap(), "x.fits", 190.0, -11.6, 3.0, 30.0)
        _fail_msg = ""
    except _tl.ToolError as _e:
        _fail_msg = str(_e)
    _memo = {}
    _s3 = _StubAstap()
    _c3, _n3 = _pp.solve_with_retries(_s3, "x.fits", 190.0, -11.6, None, 30.0, memo=_memo)   # no scale in the header
    _s4 = _StubAstap()
    _pp.solve_with_retries(_s4, "y.fits", 190.0, -11.6, None, 30.0, memo=_memo)              # the next master
    _said = []
    _s5 = _StubAstap()
    _s5.solve = lambda *a, **k: (_ for _ in ()).throw(_tl.ToolError("No solution found!")) if k.get("radius") \
        and a[3] != 0.35 else [("CRVAL1", 1.0, "")]
    _pp.solve_with_retries(_s5, "z.fits", 190.0, -11.6, None, 30.0, progress=_said.append,
                           memo={"fov": 0.9})                                                # retries with no field
    check(_c3 and "no pixel scale" in _n3 and _s4.calls[0] == 0.35 and len(_s4.calls) == 1,
          f"no focal length / pixel size in the header: typical field sizes are tried ({_n3}) and the one that "
          "worked is used first for the other masters")
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
        # Start Fresh: forgets measurements and progress, moves old masters aside - never deletes images
        import tempfile as _tf
        _fd = Path(_tf.mkdtemp())
        (_fd / "c").mkdir()
        (_fd / "c" / "x.cache").write_bytes(b"0" * 64)
        (_fd / "m" / "NGC2359").mkdir(parents=True)
        (_fd / "m" / "NGC2359" / "master.fits").write_text("m")
        _res = __import__("checkpoints").fresh_start(_fd / "c", progress=False, masters_root=_fd / "m")
        _dlg = __import__("subzeroastro").StartFreshDialog(w, w.settings) if hasattr(
            __import__("subzeroastro"), "StartFreshDialog") else None
        _ok_dlg = _dlg is not None and _dlg.chk_measure.isChecked() and _dlg.chk_progress.isChecked() \
            and not _dlg.chk_masters.isChecked()
        if _dlg is not None:
            _dlg.close()
        check(_res["measurements"] == 64 and not list((_fd / "c").glob("*.cache")) and _res["moved"] == 1
              and (Path(_res["moved_to"]) / "NGC2359" / "master.fits").exists() and _ok_dlg
              and w.act_fresh in w.menuBar().actions()[0].menu().actions(),
              "File › Clear Everything – Start From Scratch clears saved measurements and progress and moves old masters aside "
              "(nothing deleted)")
        from PySide6.QtCore import QSettings as _QS
        _qs = _QS(str(_fd / "s.ini"), _QS.IniFormat)
        for _k, _v in (("stack/kappa", 2.0), ("stack/masters_root", "/x/masters"), ("prep/rcastro", "/rc"),
                       ("ui/splash", False), ("stack/rigs", "[]"), ("finish/signature", "B. Poteet")):
            _qs.setValue(_k, _v)
        __import__("subzeroastro").reset_settings(_qs)
        check(sorted(_qs.allKeys()) == ["finish/signature", "prep/rcastro", "stack/masters_root", "stack/rigs"],
              "reset all settings to factory defaults keeps folders, tool locations, rigs and signature")
        # the intro animation: frames at set moments, and it closes by itself
        import splash as _sp
        sp = _sp.Splash("SubZeroAstro", "Brian A. Poteet, DVM", "From subs to masters - zero hassle.", "0.0 test")
        sp.show()
        check(sp.xray_img is not None and sp.xray_img.width() > 100,
              "intro animation finds its radiograph picture (assets/splash_radiograph.png)")
        _xi = sp.xray_img
        _alphas = [_xi.pixelColor(int(_xi.width() * fx), int(_xi.height() * fy)).alpha()
                   for fx in (0.3, 0.4, 0.5, 0.6, 0.7) for fy in (0.3, 0.4, 0.5, 0.6, 0.7)] if _xi is not None else [0]
        check(_xi is not None and _xi.pixelColor(0, 0).alpha() == 0 and max(_alphas) > 200,
              "the radiograph's edges and its black film fade away softly (no hard rectangle); the X-ray shows")
        for t_ in (0.5, 1.0, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 4.0, 4.6, 5.2):
            sp.set_time(t_)
            QApplication.processEvents()
            pm = sp.grab()
            pm.save(str(out_dir / f"0_splash_{t_:.2f}s.png"))
        sp.close()
        sp2 = _sp.Splash("SubZeroAstro", "Brian A. Poteet, DVM", "From subs to masters - zero hassle.")
        state["splash_done"] = False
        sp2.finished.connect(lambda: state.__setitem__("splash_done", True))
        sp2.start()
        sp2.app_ready()
        t_end = time.time() + 8
        while not state["splash_done"] and time.time() < t_end:
            QApplication.processEvents()
            time.sleep(0.01)
        check(state["splash_done"] and not sp2.isVisible(), "intro animation plays and closes by itself (~5.7 s)")
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
            import shutil as _tsh
            _nd = session.parent / "astap_no_database"
            _nd.mkdir(parents=True, exist_ok=True)
            _tsh.copy2(_fk / "fake_astap.py", _nd / "fake_astap.py")
            _ast = next(r for r in _tc.check(dict(_paths, **{"prep/astap": str(_nd / "fake_astap.py")}))
                        if r["key"] == "astap")
            _ast_ok = next(r for r in _all if r["key"] == "astap")
            check(_ast["problem"] == "no star database" and "D50" in _ast["detail"] and "D50" in _ast_ok["detail"],
                  f"ASTAP without a star database is flagged (plate solving needs one); with one it says which "
                  f"({_ast_ok['detail']})")
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
        import stacking as _stb
        _ord = sorted(["SII", "B", "Ha", "L", "OIII", "G", "R", "L-Ultimate"], key=_stb._filter_order)
        _bands = {_stb.filter_band(g.filter) for g in win.groups}
        _b0 = next(iter(_bands))
        win.set_band_enabled(_b0, False)
        _off = all(not g.enabled for g in win.groups if _stb.filter_band(g.filter) == _b0)
        win.set_band_enabled(_b0, True)
        check(_ord == ["L", "R", "G", "B", "Ha", "OIII", "SII", "L-Ultimate"] and _off
              and all(g.enabled for g in win.groups) and hasattr(win, "chk_broad") and hasattr(win, "chk_narrow"),
              f"stacking rows: broadband first (L R G B), then narrowband; one tick box per kind ({_ord})")
        check(all(any(k == "dark" for k, _ in g.plan.warnings) for g in win.groups),
              "missing darks are reported before calibration is added")
        _g_off = win.groups[-1].label()
        win.groups[-1].enabled = False                      # the user unticks a row, then adds calibration
        win._add_cal_paths([raw_cal])
        _still_off = [g.enabled for g in win.groups if g.label() == _g_off] == [False]
        for g in win.groups:
            g.enabled = True
        win.fill_table()
        check(_still_off, "a row the user unticked stays unticked after calibration frames are added")
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
        import prepare_window as _pwx
        check(_pwx.XT_START["bxt"]["values"].get("ss") == 0.50 and _pwx.XT_START["bxt"]["values"].get("ash") == 0.40,
              "BlurXTerminator starts at Sharpen Stars 0.50 and Adjust Star Halos 0.40 for a new user")
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
        _seen = getattr(pw, "step_previews_seen", [])
        check(len(_seen) >= 1, f"page 3 shows the picture after each step while the run goes on ({_seen})")
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
        for _x in (w, sw_, pw, fw):
            _x.app_header._target_timer.timeout.emit()
        _tg = [x.app_header.target.text() for x in (sw_, pw, fw)]
        check(all(t_.startswith("Target") for t_ in _tg) and fw.state.target in _tg[2],
              f"every page shows the target being worked on at the top ({_tg})")
        check(_uh.targets_text(["M27", "M27", "M28"]) == "Targets: M27, M28" and _uh.targets_text([]) == ""
              and _uh.targets_text(list("ABCDE")) == "Targets: A, B + 3 more", "the target line names one, a few or many")
        _nums = [x.app_header.num.text() for x in (w, sw_, pw, fw)]
        check(_nums == ["Page 1 of 4", "Page 2 of 4", "Page 3 of 4", "Page 4 of 4"],
              f"every page shows its page number ({_nums})")
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
        _ui = {}
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
                n0 = len(fw.project.steps)
                fw.apply_choice()
                QApplication.processEvents()
                r1 = (fw.tool is not None and fw.tool.key == "dust" and fw.btn_done_tool.isVisibleTo(fw)
                      and not fw.spots and "new picture" in fw.choice_note.text())
                fw._add_spot(fw.state.shape[1] * 0.3, fw.state.shape[0] * 0.3, 30.0)
                fw.apply_choice()
                QApplication.processEvents()
                r2 = fw.tool is not None and fw.tool.key == "dust" and len(fw.project.steps) == n0 + 2
                fw.done_tool()
                _nx = F.next_step(fw.state)
                # ✓ Applied – Next Step opens whatever is recommended next (Brighten, or Clean Bright-Star Leftovers
                # when star removal left something in the test picture)
                _ok_next = _nx is not None and _nx.key in ("stretch", "leftovers")
                check(r1 and r2 and (fw.tool is None or (_nx is not None and fw.tool.key == _nx.key))
                      and "dust" in fw.state.done and _ok_next,
                      "Mote Buster can be run again on the fixed picture; ✓ Applied – Next Step opens the next step"
                      + ("" if (r1 and r2 and _ok_next) else
                         f" (r1 {r1}, r2 {r2}, tool {fw.tool.key if fw.tool else None}, next {_nx.key if _nx else None}, "
                         f"done {fw.state.done})"))
                again_ok = fw.btn_again.isVisibleTo(fw) and "Mote Buster" in fw.btn_again.toolTip()
                fw.do_again()
                QApplication.processEvents()
                again_ok = again_ok and fw.tool is not None and fw.tool.key == "dust"
                fw.cancel_tool()
                check(again_ok and F.TOOLS["curves"].repeatable and not F.TOOLS["stretch"].repeatable,
                      "↻ Do It Again reopens the last step (for steps that can be repeated)")
            fw.open_tool(t.key)
            if t.key in ("core", "scnr", "stretch") and fw.sliders and not state.get("live_" + t.key):
                # the real app previews in a background thread after a short pause: test that path too
                state["live_" + t.key] = True

                def _sig():
                    QApplication.processEvents()
                    im = fw.view.pix.pixmap().toImage()
                    return (im.width(), im.height(), hash(bytes(im.constBits())[: im.sizeInBytes()]))
                before = _sig()
                fw.synchronous = False
                try:
                    name_, (sl_, sp_) = next(iter(fw.sliders.items()))
                    target_ = sl_.minimum() if sl_.value() > (sl_.minimum() + sl_.maximum()) / 2 else sl_.maximum()
                    for v_ in range(sl_.value(), target_, 1 if target_ > sl_.value() else -1):
                        sl_.setValue(v_)                      # like dragging: many small moves
                        QApplication.processEvents()
                    sl_.setValue(target_)
                    t_end = time.time() + 30
                    while time.time() < t_end:
                        QApplication.processEvents()
                        busy = (fw.preview_timer.isActive() or fw.preview_pending
                                or (fw.preview_job is not None and fw.preview_job.isRunning()))
                        if not busy:
                            break
                        time.sleep(0.02)
                    for _ in range(20):
                        QApplication.processEvents()
                        time.sleep(0.01)
                    after = _sig()
                finally:
                    fw.synchronous = True
                log(f"live preview {t.key}: slider {name_} -> {target_}, before {before[:2]}, after {after[:2]}, "
                    f"hint: {fw.preview_hint.text()[:80]}")
                check(after != before and "fine-tuned" in fw.preview_hint.text(),
                      f"{t.title}: moving a fine-tune slider changes the big picture (live, as in the app)")
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
                _np_ = [i_ for i_, v_ in enumerate(fw.variants) if not v_.recommended]
                if _np_:
                    fw.group.buttons()[_np_[-1]].click()          # the user picks another version than the ★
                    QApplication.processEvents()
                _ui["picked"] = fw.variants[_np_[-1]].label if _np_ else None
                check(fw.curve.isVisible() or fw.curve.isVisibleTo(fw), "Curves shows a curve you can drag")
                fw.curve.points[2] = [0.5, 0.58]
                fw._curve_dragged([list(q) for q in fw.curve.points])
            if t.key == "shrink":
                _ui["star_box"] = fw.chk_star_bg.isVisibleTo(fw)
                _ui["star_only"] = (fw.state.stars_out and fw.state.stars is not None and not np.array_equal(
                    F.state_display(fw.state.small(300), with_stars="only"),
                    F.state_display(fw.state.small(300), with_stars=True)))
            fw.apply_choice()
            QApplication.processEvents()
            if t.key == "stretch":
                _ui["one_way"] = (fw.applied_label.isVisibleTo(fw) and not fw.btn_apply.isVisibleTo(fw)
                                  and fw.btn_done_tool.isVisibleTo(fw) and "Next Step" in fw.btn_done_tool.text())
            if t.key == "curves":
                _ui["picked_kept"] = (_ui.get("picked") is not None and fw.group.checkedId() >= 0
                                      and fw.variants[fw.group.checkedId()].label == _ui["picked"])
                _ui["repeat"] = (fw.tool is not None and fw.tool.key == "curves" and fw.btn_apply.text() == "Apply Again"
                                 and fw.btn_done_tool.isVisibleTo(fw) and fw.applied_label.isVisibleTo(fw))
            if t.spatial:
                _side = sys.modules[type(fw).__module__].PREVIEW_SIDE
                _ui.setdefault("full_view", []).append(
                    fw.view.pix.pixmap().width() == fw.state.small(_side).shape[1])
            done.append(t.key)
        check(_ui.get("one_way", True) and _ui.get("repeat", True),
              f"after Apply the panel says it is applied: one-way steps show only '✓ Applied – Next Step', repeatable "
              f"ones 'Apply Again' too ({_ui})")
        check(all(_ui.get("full_view", [True])), "after a step the whole picture is shown again (not the close-up preview)")
        if "picked_kept" in _ui:
            check(_ui["picked_kept"], f"after Apply the version the user picked stays marked, not the suggested one "
                                      f"(picked {_ui.get('picked')})")
        if "star_box" in _ui:
            check(_ui["star_box"] and _ui["star_only"], "the star steps show the stars on their own, with a tick box "
                                                        "for the nebula behind them")
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
                _fs = getattr(fw, "disp_factor", 1.0)
                _start_seen = fw.spot_r / _fs
                fw._spot_drag(120.0, 90.0, "press")
                fw._spot_drag(120.0, 90.0, "release")
                _r0 = fw.spots[0][2]
                fw._spot_wheel(3)                               # resize the circle just placed
                _r1 = fw.spots[0][2]
                _rect = fw.spot_items[0].rect()
                fw._spot_wheel(40)                              # far past the old 80 px limit
                _big = fw.spots[0][2]
                fw._spot_wheel(-40)
                _pen_w = fw.spot_items[0].pen().width()
                check(_big > 300 and _start_seen >= 9.9 - 1e-6 and _pen_w >= 2,
                      f"Spot Fix circles are easy to see (start {_start_seen:.0f} screen px, 2-px line) and "
                      f"grow as big as Mote Buster's ({_big:.0f} px)")
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
                # Mac-friendly ways too: Delete removes the picked circle; Control-click acts as a right-click
                _nb = len(fw.spots)
                fw._spot_drag(260.0, 180.0, "press")
                fw._spot_drag(260.0, 180.0, "release")
                fw._spot_delete_key()
                _del_ok = len(fw.spots) == _nb
                fw._spot_drag(260.0, 180.0, "press")
                fw._spot_drag(260.0, 180.0, "release")
                from PySide6.QtGui import QMouseEvent as _QME
                from PySide6.QtCore import QEvent as _QEv, QPointF as _QPF
                _vp = fw.view.mapFromScene(_QPF(260.0, 180.0))
                _ev = _QME(_QEv.MouseButtonPress, _QPF(_vp), _QPF(fw.view.mapToGlobal(_vp)), Qt.LeftButton,
                           Qt.LeftButton, Qt.MetaModifier)
                fw.view.mousePressEvent(_ev)
                check(_del_ok and len(fw.spots) == _nb,
                      "a circle can also be removed with Delete (the picked one) or Control-click on a Mac")
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
        check(_fp == {"h": False, "v": True, "rot": 0} and _flip_ok
              and "flipped vertically" in fw.project.steps[-1]["label"],
              f"Flip the Picture: Flip Horizontally / Flip Vertically toggle (pressing again flips back) - {_fp}")
        # Rotate 90°: each press turns a quarter turn; clockwise then counter-clockwise twice = one turn back
        _orig2 = fw.state.rgb.copy()
        fw.open_tool("flip")
        fw.btn_rot_cw.click()
        fw.btn_rot_ccw.click()
        fw.btn_rot_ccw.click()
        QApplication.processEvents()
        _fp2 = fw.current_params()
        fw.apply_choice()
        QApplication.processEvents()
        check(_fp2 == {"h": False, "v": False, "rot": 3}
              and np.array_equal(fw.state.rgb, np.rot90(_orig2, 1, axes=(1, 2)))
              and "turned 90° counter-clockwise" in fw.project.steps[-1]["label"],
              f"Rotate 90° Clockwise / Counter-clockwise turn the picture a quarter turn per press - {_fp2}")
        fw.undo_last()                           # the turn
        QApplication.processEvents()
        fw.undo_last()                           # the flip
        QApplication.processEvents()
        check(len(fw.project.steps) == _nflip and np.array_equal(fw.state.rgb, _orig), "undo un-flips the picture")
        # Spot Fix heals big circles smoothly: the slope of the sky carries across, grain is put back
        _yy, _xx = np.mgrid[:900, :900]
        _sky = (0.2 + 0.1 * _xx / 900 + 0.05 * _yy / 900).astype(np.float32)
        _sky = _sky + np.random.default_rng(3).normal(0, 0.01, _sky.shape).astype(np.float32)
        _blob = (_yy - 450) ** 2 + (_xx - 450) ** 2 < 150 ** 2
        _im = np.stack([_sky] * 3)
        _im[:, _blob] = 0.9
        _hl = F.SpotFix.heal(_im, 450, 450, 160)
        _res = _hl[0][_blob] - (0.2 + 0.1 * _xx / 900 + 0.05 * _yy / 900)[_blob]
        check(abs(float(_res.mean())) < 0.003 and 0.004 < float(_res.std()) < 0.014
              and F.SpotFix.spot_range[1] >= 600,
              f"Spot Fix heals a big (160 px) circle smoothly (off by {abs(float(_res.mean())):.4f}, grain "
              f"{float(_res.std()):.4f} vs sky 0.0100)")

        # Spot Fix keeps the stars inside a big circle (put back as bright as they really are), but
        # still removes the thing clicked at the middle of a small one
        _rg = np.random.default_rng(5)
        _yy, _xx = np.mgrid[:700, :700]
        _sk = 0.15 + 0.05 * _xx / 700

        def _st(cx, cy, a, sg=2.0):
            return a * np.exp(-((_xx - cx) ** 2 + (_yy - cy) ** 2) / (2 * sg * sg))
        _d0 = np.hypot(_xx - 350, _yy - 350)
        _dn = 1 - 0.25 * np.exp(-((_d0 - 100) / 35) ** 2)
        _tr = _sk + _st(250, 330, 0.5) + _st(350, 220, 0.6, 2.5)
        _im = np.stack([_tr * _dn + _rg.normal(0, 0.004, _tr.shape)] * 3).astype(np.float32)
        _o = F.SpotFix.heal(_im, 350, 350, 160)[0]
        _pk = [float(_o[y, x] - _sk[y, x]) for x, y in ((250, 330), (350, 220))]
        _o2 = F.SpotFix.heal(_im, 350, 350, 160, keep_stars=False)[0]
        _im2 = np.stack([_sk + _st(200, 200, 0.6) + _st(212, 200, 0.4)] * 3).astype(np.float32)
        _o3 = F.SpotFix.heal(_im2, 200, 200, 9)[0]
        check(abs(_pk[0] - 0.5) < 0.05 and abs(_pk[1] - 0.6) < 0.06 and float(_o2[330, 250] - _sk[330, 250]) < 0.05
              and float(_o3[200, 200] - _sk[200, 200]) < 0.03 and float(_o3[200, 212] - _sk[200, 212]) > 0.3,
              f"Spot Fix keeps the stars inside a big circle at their true brightness ({_pk[0]:.2f}/0.50, "
              f"{_pk[1]:.2f}/0.60) and still removes what is clicked in a small one")

        # colour-camera masters: every master of the filter is used (deepest as reference, noise-weighted),
        # and unfiltered ones are set aside when there is dual-band data
        from types import SimpleNamespace as _NS
        _rg = np.random.default_rng(7)
        _yy, _xx = np.mgrid[:400, :500]
        _sk = np.zeros((400, 500))
        for _ in range(80):
            _sk += _rg.uniform(2000, 30000) * np.exp(-((_xx - _rg.uniform(15, 485)) ** 2 + (_yy - _rg.uniform(15, 385)) ** 2)
                                                    / (2 * 1.8 ** 2))

        def _cm(name, filt, exp, noise):
            d = np.stack([1000 + _sk * k + _rg.normal(0, noise, _sk.shape) for k in (1, .5, .5)]).astype(np.float32)
            return _NS(path=Path(name + ".xisf"), data=d, cards=[("EXPTOTAL", exp, ""), ("ALIGNREF", "r", "")],
                       look={}, filter=filt, is_color=True)
        _deep, _short, _broad = (_cm("deep", "L-Ultimate", 110880, 40), _cm("short", "L-Ultimate", 540, 400),
                                 _cm("broad", "No filter", 9000, 120))
        _used, _left = F.pick_colour_masters([_short, _broad, _deep], lambda m: F.dual_band_filter(m.filter, m.look))
        _rgbs, _srgb, _cn = F.assemble_colour(_used)
        _noise = F._sky_noise(_rgbs["L-Ultimate"][0] * 65535)[1]
        check([m.path.stem for m in _left] == ["broad"] and _noise < 60 and "2 L-Ultimate masters combined" in _cn,
              f"colour masters: all of a filter are combined around the deepest (noise {_noise:.0f} vs deep 40, short "
              f"400) and unfiltered ones are left out of dual-band pictures")

        # Tame Bright Core on a burnt-out core: the white goes and the structure comes back from the linear data,
        # nothing outside the core changes
        _yy, _xx = np.mgrid[:600, :800]
        _r = np.hypot(_xx - 400, _yy - 300)
        _lin = (0.1 + 0.35 * np.exp(-(_r / 220) ** 2) + 0.75 * np.exp(-(_r / 60) ** 2)
                + 0.08 * np.sin(_xx / 6.0) * np.sin(_yy / 7.0) * np.exp(-(_r / 60) ** 2))
        _y = np.clip(_lin, 0, 1)
        _cs = F.State(rgb=np.stack([_y] * 3).astype(np.float32), linear=False,
                      lin_lum=(_lin / _lin.max() * 0.2).astype(np.float32))
        _co = F.TOOLS["core"].apply(_cs, {"amount": 0.7, "size": 0.78, "detail": 30.0}).rgb[0]
        _ci = _r < 40
        _w0, _w1 = float((_y[_ci] > 0.98).mean()), float((_co[_ci] > 0.98).mean())
        _d1 = float((_co - F._blur(_co, 8))[_ci].std())
        check(_w0 > 0.5 and _w1 < 0.05 and _d1 > 0.018 and float(np.abs(_co - _y)[_r > 250].max()) < 1e-4,
              f"Tame Bright Core (Medium) clears a burnt-out core ({_w0:.0%} -> {_w1:.0%} white) and brings its "
              f"structure back (detail {_d1:.3f}, real 0.027) without touching the rest")

        # SkyFlat across a meridian flip: the nebula (turned 180° on the sensor) must not get into the flat
        import stacking as _stk
        import warnings as _wn
        _rg = np.random.default_rng(1)
        _yy, _xx = np.mgrid[:200, :300]
        _Ft = (1 - 0.35 * (((_xx - 150) / 150) ** 2 + ((_yy - 100) / 100) ** 2))
        _nb = np.zeros((200, 300))
        for _cx, _cy, _a in ((60, 50, 0.6), (110, 160, 0.5), (250, 80, 0.4)):
            _nb += _a * np.exp(-((_xx - _cx) ** 2 / (2 * 22 ** 2) + (_yy - _cy) ** 2 / (2 * 8 ** 2)))
        _fr = np.stack([(_Ft * (1 + (_nb if i < 30 else _nb[::-1, ::-1])) + _rg.normal(0, 0.01, _nb.shape))
                        for i in range(40)]).astype(np.float32)
        _g = _stk._orientation_groups(_fr)
        with _wn.catch_warnings():
            _wn.simplefilter("ignore")

            def _lh(a):
                m_ = np.nanmedian(a, 0)
                return np.nanmean(np.where(a <= m_[None], a, np.nan), 0)
            _fl = np.fmin(_lh(_fr[_g]), _lh(_fr[~_g])) if _g is not None else _lh(_fr)
        _leak = float((_fl / np.median(_fl) / (_Ft / np.median(_Ft)) - 1)[_nb > 0.2].mean())
        _g0 = _stk._orientation_groups(np.stack([(_Ft * (1 + _nb) + _rg.normal(0, 0.01, _nb.shape))
                                                 for i in range(40)]).astype(np.float32))
        check(_g is not None and sorted([int(_g.sum()), int((~_g).sum())]) == [10, 30] and abs(_leak) < 0.02
              and _g0 is None,
              f"SkyFlat across a meridian flip: the two sides are found (30 + 10 frames) and the nebula stays out of "
              f"the flat (leak {_leak:.3f}); frames all one way up are left as they are")

        # a small galaxy nucleus burnt out even in the masters: no hard-edged white dot is left after taming
        _yy, _xx = np.mgrid[:400, :600]
        _rr = np.hypot(_xx - 300, _yy - 200)
        _L = np.minimum(0.02 + 0.5 * np.exp(-(np.hypot(_xx - 300, (_yy - 200) / 0.6) / 2.0) ** 0.5), 0.30).astype(np.float32)
        _ys = np.clip(np.arcsinh(_L * 60) / np.arcsinh(0.30 * 60) * 1.15, 0, 1).astype(np.float32)
        _no = F.TOOLS["core"].apply(F.State(rgb=np.stack([_ys] * 3), linear=False, lin_lum=_L),
                                    F.TOOLS["core"].variants(F.State(rgb=np.stack([_ys] * 3), linear=False))[1].params
                                    ).rgb[0]
        _pf = np.array([_no[(_rr >= k) & (_rr < k + 1)].mean() for k in range(0, 30)])
        _ar = float(np.median(_no[(_rr > 10) & (_rr < 14)]))
        check(_pf[0] < 0.86 and float(np.max(-np.diff(_pf))) < 0.065,
              f"Tame Bright Core leaves no hard white dot at a burnt-out nucleus (centre {_pf[0]:.2f} vs {_ar:.2f} "
              f"around, edge {float(np.max(-np.diff(_pf))):.3f}/px)")

        # dual-band colour masters (red = Ha, blue = OIII look different), one of them shifted: every colour is
        # lined up with the same colour of the deepest master - no colour may go missing (Mac report: crash)
        _yy, _xx = np.mgrid[:360, :480]
        _sf = [(_rg.uniform(12, 468), _rg.uniform(12, 348), _rg.uniform(1500, 25000)) for _ in range(110)]

        def _fld(dx, dy):
            a_ = np.zeros((360, 480))
            for _x, _y, _f in _sf:
                a_ += _f * np.exp(-((_xx - _x - dx) ** 2 + (_yy - _y - dy) ** 2) / (2 * 1.8 ** 2))
            return a_
        _ha = 4000 * np.exp(-((_xx - 200) ** 2 / (2 * 70 ** 2) + (_yy - 180) ** 2 / (2 * 30 ** 2)))
        _o3 = 3000 * np.exp(-((_xx - 320) ** 2 / (2 * 40 ** 2) + (_yy - 140) ** 2 / (2 * 50 ** 2)))

        def _dbm(name, exp, dx, dy, ref):
            sf_ = _fld(dx, dy)
            sh_ = lambda a_: np.roll(np.roll(a_, int(round(dy)), 0), int(round(dx)), 1)  # noqa: E731
            d_ = np.stack([1000 + sf_ + sh_(_ha), 1000 + 0.35 * sf_ + sh_(_o3), 1000 + 0.3 * sf_ + 1.1 * sh_(_o3)])
            d_ = (d_ + _rg.normal(0, 40, d_.shape)).astype(np.float32)
            return _NS(path=Path(name + ".xisf"), data=d_, cards=[("EXPTOTAL", exp, ""), ("ALIGNREF", ref, "")],
                       look={}, filter="L-eNhance", is_color=True)
        _db, _dn = F.assemble_colour([_dbm("short", 19000, 3.4, -2.2, "b"), _dbm("deep", 54000, 0, 0, "a")])[:2][0], None
        check("L-eNhance" in _db and _db["L-eNhance"].shape[0] == 3,
              f"dual-band colour masters from different runs are combined in every colour ({list(_db)})")

        # a planetary nebula: Tame Bright Core works as "Tame the Ring" - the nebula is brightened again gently
        # from the linear data: a ring AND interior burnt out white come back well below white, the ring
        # brighter than the inside, the faint halo untouched; the mask tint shows in green
        _yy, _xx = np.mgrid[:1000, :1500]
        _r = np.hypot(_xx - 750, (_yy - 500) / 1.25)
        _lin = (0.01 + 0.25 * np.exp(-((_r - 38) / 9) ** 2) + 0.16 * np.exp(-(_r / 32) ** 2)
                + 0.015 * np.exp(-((_r - 75) / 25) ** 2)).astype(np.float32)
        _yr = np.clip(np.arcsinh(_lin * 80) / np.arcsinh(0.25 * 80) * 1.25, 0, 1).astype(np.float32)
        _ps = F.State(rgb=np.stack([_yr] * 3), linear=False, lin_lum=_lin, target="M57")
        _pv = F.TOOLS["core"].variants(_ps)
        _po = F.TOOLS["core"].apply(_ps, _pv[1].params).rgb[0]
        _rgm, _cnm, _hlm = (_r > 32) & (_r < 44), _r < 12, (_r > 65) & (_r < 85)
        _prev = F.TOOLS["core"].preview(_ps.small(600), dict(_pv[1].params, show_mask=True)).astype(np.float32)
        _pnm = F.TOOLS["core"].preview(_ps.small(600), dict(_pv[1].params)).astype(np.float32)
        _green = (float((_prev[..., 1] - _pnm[..., 1]).mean()) > float((_prev[..., 0] - _pnm[..., 0]).mean()) + 0.2
                  and bool(((_prev[..., 0] > 200) & (_prev[..., 1] > 180) & (_prev[..., 2] < 100)).any()))
        check(all(v.params.get("ring") for v in _pv) and float(_yr[_rgm].mean()) > 0.97 and float(_yr[_cnm].mean()) > 0.97
              and float(_po[_rgm].mean()) < 0.82 and float(_po[_cnm].mean()) < float(_po[_rgm].mean()) - 0.04
              and abs(float(_po[_hlm].mean()) - float(_yr[_hlm].mean())) < 0.03 and _green,
              f"planetary nebula (Medium): a white ring and interior come back below white - ring "
              f"{_yr[_rgm].mean():.2f} -> {_po[_rgm].mean():.2f}, inside {_yr[_cnm].mean():.2f} -> {_po[_cnm].mean():.2f}, "
              "halo unchanged; the area shows as a light green wash with a yellow edge")

        # a planetary nebula that star removal put (partly) into the stars image is moved back to the nebula,
        # bright stars stay; and its ring is tamed after Curves
        _H, _W = 1200, 1800
        _sl = np.zeros((_H, _W), np.float32)
        _yy, _xx = np.mgrid[:_H, :_W]
        _pts = [(_rg.uniform(20, _W - 20), _rg.uniform(20, _H - 20), _rg.uniform(0.002, 0.1), 1.0) for _ in range(300)]
        _pts += [(1000, 700, 0.9, 4.0), (800, 520, 0.9, 4.0)]
        for _x, _y, _a, _sg in _pts:
            _y0, _y1, _x0, _x1 = int(_y) - 20, int(_y) + 21, int(_x) - 20, int(_x) + 21
            _gy, _gx = np.mgrid[_y0:_y1, _x0:_x1]
            _sl[_y0:_y1, _x0:_x1] += (_a * np.exp(-((_gx - _x) ** 2 + (_gy - _y) ** 2) / (2 * _sg * _sg))).astype(np.float32)
        _rr = np.hypot(_xx - 900, (_yy - 600) / 1.25)
        _sl += (0.06 * np.exp(-((_rr - 42) / 10) ** 2)).astype(np.float32)
        _stk = np.stack([_sl, _sl * 0.9, _sl * 0.8])
        _rgb2, _st2, _sh = F.rescue_object(np.full_like(_stk, 0.01), _stk)
        _rgm = (_rr > 34) & (_rr < 50)
        _wf_p = [t.key for t, _ in F.workflow(F.State(rgb=np.full((3, 40, 40), 0.3, np.float32), kind="osc_dual",
                                                      target="M57", linear=False))]
        check(float(_st2[0][_rgm].mean()) < 0.1 * float(_stk[0][_rgm].mean()) and abs(float(_st2[0][700, 1000]) - 0.9) < 0.02
              and float(_rgb2[0][_rgm].mean()) > 0.04 and _wf_p.index("curves") < _wf_p.index("core") < _wf_p.index("saturation"),
              f"planetary nebula: the part of it star removal took for a star goes back to the nebula (stars image "
              f"{_stk[0][_rgm].mean():.3f} -> {_st2[0][_rgm].mean():.3f}), bright stars stay; the ring is tamed after Curves")

        # ... and the part moved back is round and follows the object's light - no square (NGC 7662 report)
        _qy, _qx = np.mgrid[:508, :664].astype(np.float32)
        _qr = np.hypot(_qx - 332, _qy - 254)
        _qcore = 0.8 * np.exp(-_qr ** 2 / (2 * 14.0 ** 2)) + 0.004 * np.exp(-_qr / 25.0)   # bright body + faint wings
        _qst = np.stack([0.3 * _qcore, _qcore, _qcore]).astype(np.float32) + 0.001
        _qo, _qs, _qsh = F.rescue_object(np.full_like(_qst, 0.02), _qst)
        _qring = [float(np.ptp(_qs.mean(0)[(_qr > _k - 1) & (_qr < _k + 1)])) for _k in range(8, 60, 4)]
        _qwing = float(_qs.mean(0)[(_qr > 40) & (_qr < 60)].mean())
        # the faintest outer glow goes too (no ring where the moved part ended), and only light above the sky
        # moves (no disc darker than the sky)
        _qst2 = _qst + (0.0006 * np.exp(-_qr / 50.0)).astype(np.float32)
        _qo2, _qs2, _ = F.rescue_object(np.full_like(_qst2, 0.02), _qst2)
        _qprof = [float(_qs2.mean(0)[(_qr > _k - 4) & (_qr < _k + 4)].mean()) - 0.001 for _k in range(20, 230, 10)]
        check(_qsh > 0.5 and max(_qring) < 0.01 and _qwing < 0.0015 and max(_qprof) < 0.00012 and min(_qprof) > -0.00005,
              f"planetary core moved back from the stars image evenly in every direction - no square "
              f"(moved {_qsh:.0%}, largest unevenness around a circle {max(_qring):.4f}); its faint wings go too, so no dark "
              f"rim ({_qwing:.4f} left), no ring of outer glow left ({max(_qprof):.5f}) and no disc below the sky "
              f"({min(_qprof):.5f})")

        # the spikes star removal leaves around a bright core leave the stars image too (they showed as spokes
        # after Brighten Stars Separately); a star beside the core stays
        _sp = _qcore.copy()
        _bar = ((np.abs(_qy - 254) < 1.6) | (np.abs(_qx - 332) < 1.6)) & (_qr > 30) & (_qr < 60)
        _sp = _sp + 0.01 * _bar + 0.05 * np.exp(-((_qx - 412) ** 2 + (_qy - 214) ** 2) / (2 * 1.5 ** 2))
        _spst = np.stack([_sp, _sp, _sp]).astype(np.float32) + 0.001
        _spo, _sps, _ = F.rescue_object(np.full_like(_spst, 0.02), _spst)
        _spk = float((_sps[1][_bar] - 0.001).mean()) / float((_spst[1][_bar] - 0.001).mean())
        _spstar = float(_sps[1][214, 412] - 0.001) / float(_spst[1][214, 412] - 0.001)
        _spd = np.hypot(_qx - 412, _qy - 214)
        _spring = float((_spo[1] - 0.02)[(_spd > 2) & (_spd < 9)].max() - (_spo[1] - 0.02)[(_spd > 12) & (_spd < 16)].mean())
        check(_spk < 0.2 and _spstar > 0.8 and _spring < 0.0002,
              f"spikes around a planetary core go back to the nebula ({_spk:.0%} left in the stars image), "
              f"a star beside it stays whole ({_spstar:.0%}, no ring of it in the nebula image: {_spring:.4f})")

        # a real star sitting on a big planetary's body stays in the stars image (moved into the nebula image it came
        # out as a coloured dot - the red dot on M27's edge); the body itself still moves
        _by, _bx = np.mgrid[:900, :1100].astype(np.float32)
        _bbody = 0.05 * np.exp(-(np.hypot(_bx - 550, _by - 450) / 150) ** 4)
        _bstar = 0.6 * np.exp(-((_bx - 650) ** 2 + (_by - 480) ** 2) / (2 * 2.2 ** 2))
        _bst = np.stack([_bbody + _bstar, _bbody * 0.8 + _bstar * 0.5, _bbody * 0.9 + _bstar * 0.4]).astype(np.float32) + 0.001
        _bo, _bs, _bsh = F.rescue_object(np.full_like(_bst, 0.02), _bst)
        _bd = np.hypot(_bx - 650, _by - 480)
        _bkept = float((_bs[0] - 0.001)[_bd < 10].sum() / _bstar[_bd < 10].sum())
        check(_bkept > 0.95 and _bsh > 0.5, f"a star on a planetary nebula's body stays a star ({_bkept:.0%} kept in the "
                                           f"stars image), the body still goes back to the nebula ({_bsh:.0%} moved)")
        # ... also a bright, wide star (a compact core with a wide halo that survives the small opening - M27's red dot, real data)
        _wr2 = (_bx - 650) ** 2 + (_by - 480) ** 2
        _wstar = 0.9 * np.exp(-_wr2 / (2 * 2.2 ** 2)) + 0.08 * np.exp(-_wr2 / (2 * 9.0 ** 2))   # core + wide halo
        _wst = np.stack([_bbody + _wstar, _bbody * 0.8 + _wstar * 0.5, _bbody * 0.9 + _wstar * 0.4]).astype(np.float32) + 0.001
        _wo, _ws, _ = F.rescue_object(np.full_like(_wst, 0.02), _wst)
        _wdot = float((_wo[0] - 0.02)[_bd < 4].mean() - (_wo[0] - 0.02)[(_bd > 25) & (_bd < 30)].mean())
        check(_wdot < 0.008, f"a bright, wide star on a planetary nebula's body leaves no dot in the nebula image (before the fix: 0.036) "
                             f"({_wdot:.4f}; the star's peak is 0.98)")

        # a big planetary (the Helix) that star removal did not take for a star: bright stars with spikes near the
        # middle are not moved into the nebula image (they came out as teal blobs)
        _hy, _hx = np.mgrid[:1200, :1600].astype(np.float32)
        _hst = np.zeros((1200, 1600), np.float32) + 0.001
        _hr = np.random.default_rng(4)
        for _ in range(14):
            _x, _y = _hr.uniform(100, 1500), _hr.uniform(100, 1100)
            _r2 = (_hx - _x) ** 2 + (_hy - _y) ** 2
            _hst += 0.9 * np.exp(-_r2 / (2 * 3.0 ** 2)) + 0.05 * np.exp(-_r2 / (2 * 14 ** 2))
            _hst += 0.03 * (((np.abs(_hy - _y) < 1.5) & (np.abs(_hx - _x) < 60)) | ((np.abs(_hx - _x) < 1.5) & (np.abs(_hy - _y) < 60)))
        _, _, _hsh = F.rescue_object(np.full((3, 1200, 1600), 0.02, np.float32), np.stack([_hst, _hst * 0.9, _hst * 0.8]))
        check(_hsh < 0.001, f"bright stars with spikes are not taken for a planetary nebula (moved {_hsh:.3f}; "
                            f"the Helix's stars turned into blobs)")

        # an L master much noisier than R, G and B (one L frame against hours of colour - M101) is not suggested:
        # RGB is; a good L still makes LRGB the suggestion
        _lr = np.random.default_rng(1)
        _lyy, _lxx = np.mgrid[:300, :400].astype(np.float32)
        _lg = 0.3 * np.exp(-((_lxx - 200) ** 2 + (_lyy - 150) ** 2) / (2 * 40 ** 2))
        _lch = lambda n: (0.05 + _lg + _lr.normal(0, n, _lg.shape)).astype(np.float32)  # noqa: E731
        _lrec = {}
        for _ln in (0.004, 0.03):
            _ls = F.State(channels={"R": _lch(0.006), "G": _lch(0.006), "B": _lch(0.006), "L": _lch(_ln)},
                          kind="mono_lrgb")
            _lrec[_ln] = [v.label for v in F.TOOLS["combine"].variants(_ls) if v.recommended]
        check(_lrec[0.004] == ["LRGB"] and _lrec[0.03] == ["RGB"],
              f"LRGB suggested with a good L, RGB when the L is much noisier than the colour filters ({_lrec})")

        # HOO + faint OIII: an OIII-only glow too faint to see (the Squid in the Flying Bat) is lifted, the
        # Ha-following OIII and the sky noise are left about as they were
        _sr = np.random.default_rng(5)
        _sy, _sx = np.mgrid[:400, :500].astype(np.float32)
        _bat = 0.02 * np.exp(-((_sx - 120) / 25) ** 2) + 0.015 * np.exp(-((_sy - 320) / 20) ** 2)
        _sqd = 0.0012 * np.exp(-(((_sx - 320) / 50) ** 2 + ((_sy - 150) / 20) ** 2))
        _sha = (0.01 + _bat + _sr.normal(0, 0.002, _sx.shape)).astype(np.float32)
        _so3 = (0.01 + 0.3 * _bat + _sqd + _sr.normal(0, 0.002, _sx.shape)).astype(np.float32)
        _ss = F.State(channels={"Ha": _sha, "OIII": _so3}, kind="mono_nb", linear=True)
        _ct = F.TOOLS["combine"]
        _cv = {v.label: v for v in _ct.variants(_ss)}
        _sqm = np.hypot((_sx - 320) / 50, (_sy - 150) / 20) < 0.7
        _bgm = (_sx > 250) & (_sy > 250) & (_sy < 300)

        def _sq_snr(lbl):
            _b = _ct.apply(_ss, _cv[lbl].params).rgb[2]
            _sm = F._blur(_b, 2.0)
            return float((_sm[_sqm].mean() - _sm[_bgm].mean()) / _sm[_bgm].std()), float(_b[_bgm].std())
        _s0, _n0 = _sq_snr("HOO")
        _s1, _n1 = _sq_snr("HOO + faint OIII")
        check("HOO + faint OIII" in _cv and _s1 > 2.0 * _s0 and _n1 < 1.05 * _n0,
              f"HOO + faint OIII lifts a faint OIII-only glow like the Squid (stands out {_s0:.1f} -> {_s1:.1f} "
              f"times the sky's ripple; sky noise {_n1 / _n0:.2f}x)")

        # a planetary nebula is brightened colour-preserving: its colours survive (a per-colour stretch turned M57 grey)
        _yy, _xx = np.mgrid[:600, :900]
        _r = np.hypot(_xx - 450, (_yy - 300) / 1.25)
        _rg_, _in_ = np.exp(-((_r - 40) / 8) ** 2), np.exp(-(_r / 34) ** 2)
        _pl = np.stack([0.02 + 0.30 * _rg_ + 0.12 * _in_, 0.02 + 0.15 * _rg_ + 0.17 * _in_,
                        0.02 + 0.13 * _rg_ + 0.19 * _in_]).astype(np.float32)
        _pst = F.State(rgb=_pl, kind="osc_dual", target="M57", linear=True, stars_out=True)
        _bt = F.TOOLS["stretch"]
        _meth = _bt.recommended_method(_pst)
        _s1 = _bt.apply(_pst, [x for x in _bt.variants(_pst, _meth) if x.label == "Gentle"][0].params)
        _ci = _s1.rgb[:, _r < 15].mean(1)
        check(_meth == "colour" and float(_ci[2] - _ci[0]) > 0.2,
              f"a planetary nebula is brightened colour-preserving and keeps its colours (inside R {_ci[0]:.2f} "
              f"G {_ci[1]:.2f} B {_ci[2]:.2f})")

        # Adjust Colours smooths abrupt colour changes (colour only - the brightness, and so the detail, stays)
        _yy, _xx = np.mgrid[:400, :600]
        _band = (_xx // 30) % 2 == 0
        _cb = np.stack([np.where(_band, 0.7, 0.2), np.full((400, 600), 0.4), np.where(_band, 0.2, 0.7)]).astype(np.float32)
        _cs = F.State(rgb=_cb, linear=False, kind="osc", target="x")
        _ct = F.TOOLS["colours"]
        _cp = dict([x for x in _ct.variants(_cs) if x.label == "Blues and reds"][0].params)
        _c0 = _ct.apply(_cs, dict(_cp, smooth=0.0)).rgb
        _c1 = _ct.apply(_cs, dict(_cp, smooth=0.6)).rgb
        _step = lambda im: float(np.abs(np.diff(im[0, 200, :] / np.maximum(im[:, 200, :].mean(0), 1e-6))).max())  # noqa: E731
        check(_step(_c1) < 0.6 * _step(_c0) and float(np.abs(_c1.mean(0) - _c0.mean(0)).max()) < 0.06
              and "smooth" in _cp,
              f"Adjust Colours can smooth abrupt colour changes ({_step(_c0):.2f} -> {_step(_c1):.2f} per pixel) "
              "keeping the brightness")

        # Askar D1 + D2 (or L-Ultimate + L-Synergy): OIII from both, with the Ha / SII leakage measured and taken out
        import fits_core as _fc
        _rg2 = np.random.default_rng(0)
        _yy, _xx = np.mgrid[:600, :900]
        _Ha = 0.3 * np.exp(-((_xx - 380) ** 2 / (2 * 110 ** 2) + (_yy - 300) ** 2 / (2 * 70 ** 2)))
        _S2 = 0.12 * np.exp(-((_xx - 490) ** 2 / (2 * 90 ** 2) + (_yy - 260) ** 2 / (2 * 60 ** 2)))
        _O3 = 0.15 * np.exp(-((_xx - 320) ** 2 / (2 * 70 ** 2) + (_yy - 340) ** 2 / (2 * 45 ** 2)))
        _o1 = _O3 + 0.08 * _Ha + _rg2.normal(0, 0.004, _Ha.shape)
        _o2 = 0.6 * _O3 + 0.07 * _S2 + _rg2.normal(0, 0.008, _Ha.shape)
        _oc, _on = F.combine_oiii([_o1, _o2], [_Ha + _rg2.normal(0, 0.004, _Ha.shape), _S2 + _rg2.normal(0, 0.008, _Ha.shape)])
        _hm = _Ha > 0.1
        # IC1396-like: the OIII follows the Ha closely over half the cloud - the leak must not swallow real OIII
        _O3c = 0.4 * _Ha * (_xx > 380) + 0.3 * _O3
        _occ, _onc = F.combine_oiii([_O3c + 0.08 * _Ha + _rg2.normal(0, 0.004, _Ha.shape),
                                     0.6 * _O3c + 0.08 * _S2 + _rg2.normal(0, 0.006, _Ha.shape)],
                                    [_Ha + _rg2.normal(0, 0.004, _Ha.shape), _S2 + _rg2.normal(0, 0.006, _Ha.shape)])
        check(abs(float((_occ - _O3c)[_hm].mean())) < 0.006 and re.search(r"\((7|8|9|10)% of the Ha", _onc) is not None,
              f"OIII that follows the Ha is kept, only the leak (8%, at most 10%) taken out ({_onc}; "
              f"error {float((_occ - _O3c)[_hm].mean()):+.4f})")
        import prepare as _pp
        _cn1 = _pp.crop_outside_note((1923, 0, 4157, 1312), (1914, 649, 5049, 2981))
        _o2e = _o2.copy()
        _o2e[:, :120] *= 0.3                                  # an edge only some filter-2 frames reached
        _one = F.combine_oiii([_o1, _o2e], [_Ha, _S2])[1]
        _lk = lambda t: re.search(r"out \((\d+)% of", t)  # noqa: E731
        check(_lk(_one) is not None and _lk(_on) is not None and abs(int(_lk(_one).group(1)) - int(_lk(_on).group(1))) <= 1,
              f"the leak is measured away from poorly covered edges (clean: {_on[33:75]}; with a bad edge: {_one[33:75]})")
        _ow, _onw = F.combine_oiii([_o1, _o2], [_Ha, _S2], [25380, 1])
        _sh = re.search(r"by exposure \((\d+)% / (\d+)%\)", _onw)
        check(_sh is not None and _sh.group(1) == "80",
              f"OIII weighted by each filter's exposure, never more than 80/20 ({_onw})")
        _o0, _on0 = F.combine_oiii([_o1[:20, :20], _o2[:20, :20]], [_Ha[:20, :20], _S2[:20, :20]])
        check("top edge" in _cn1 and _pp.crop_outside_note((2000, 700, 4000, 2900), (1914, 649, 5049, 2981)) == ""
              and "averaged" in _on0,
              f"a hand crop past the well-covered area is flagged ({_cn1[:60]}...); an OIII fallback says why ({_on0[:60]}...)")
        # SHO: Remove Green Cast turns the Ha green gold (keeps its brightness) instead of deleting it
        _rgt = F.TOOLS["scnr"]
        _sg = F.State()
        _sg.rgb = np.ones((3, 8, 8), np.float32) * np.array([0.1, 0.8, 0.1], np.float32)[:, None, None]
        _sg.kind, _sg.palette = "osc_sho", "SHO"
        _gv = [v for v in _rgt.variants(_sg) if v.recommended][0]
        _gp = _rgt.apply(_sg, _gv.params).rgb[:, 0, 0]
        _sg.palette = "HOO"
        check(_gv.label == "Gold 100%" and _gp[0] > _gp[1] > 0.4 and _gp[2] < 0.15
              and all("Gold" not in v.label for v in _rgt.variants(_sg)),
              f"SHO green (Ha) turned gold, not deleted ({_gv.label}: {np.round(_gp, 2).tolist()}); HOO keeps plain removal")
        _names = {n: _fc.normalize_filter(n) for n in ("D1", "Askar D2", "D1 Ha/OIII", "SII-OIII", "Ha/OIII")}
        check(abs(float((_oc - _O3)[_hm].mean())) < 0.004 and "leakage" in _on
              and _names == {"D1": "Askar D1", "Askar D2": "Askar D2", "D1 Ha/OIII": "Askar D1",
                             "SII-OIII": "Dual-band SII/OIII", "Ha/OIII": "Dual-band"}
              and _fc.band_lines("SII-OIII")[0] == "SII",
              f"two dual-band filters: the Ha/SII leakage is taken out of the OIII (Ha left {float((_oc - _O3)[_hm].mean()):+.4f}; "
              f"{_on}); D1/D2-style names recognised")

        # the "Processed with SubZeroAstro" credit: small, bottom-left (bottom-right when the signature is there)
        _dark = np.full((3, 600, 900), 0.05, np.float32)
        _cr = F.add_credit(_dark, "bottom left")
        _bl = float(_cr[:, -40:, :300].max())
        _rest = float(np.abs(_cr[:, :500, :] - _dark[:, :500, :]).max())
        _cr2 = F.add_credit(_dark, "bottom right")
        check(_bl > 0.2 and _rest == 0.0 and float(_cr2[:, -40:, -300:].max()) > 0.2
              and F.credit_corner([]) == "bottom left"
              and F.credit_corner([{"tool": "signature", "params": {"corner": "bottom left"}}]) == "bottom right"
              and F.credit_corner([{"tool": "signature", "params": {"pos": [0.05, 0.9]}}]) == "bottom right"
              and F.credit_corner([{"tool": "signature", "params": {"corner": "bottom right"}}]) == "bottom left",
              "the small 'Processed with SubZeroAstro' credit goes bottom-left (bottom-right when the signature is "
              "there) and touches nothing else")

        # Adjust Colours: one colour at a time; grey stays grey
        _cs_ = F.State(rgb=np.full((3, 20, 30), 0.3, np.float32), linear=False, kind="mono_nb")
        _cs_.rgb[:, 5, 5] = [0.2, 0.3, 0.7]                       # a blue pixel
        _cs_.rgb[:, 6, 6] = [0.7, 0.2, 0.2]                       # a red one
        _cv = [v for v in F.TOOLS["colours"].variants(_cs_) if v.label == "Boost blues"][0]
        _co = F.TOOLS["colours"].apply(_cs_, _cv.params).rgb
        _col_ok = (_co[2, 5, 5] > 0.75 and np.allclose(_co[:, 6, 6], _cs_.rgb[:, 6, 6], atol=0.02)
                   and np.allclose(_co[:, 0, 0], 0.3, atol=1e-4) and "colours" in F.WORKFLOWS["mono_nb"]
                   and F.TOOLS["colours"].repeatable)
        # Full Window: the picture gets the whole window, everything comes back; zoom draws full detail
        _vis0 = fw._left_panel.isVisible()
        fw.btn_full.setChecked(True)
        QApplication.processEvents()
        _hid = not fw._left_panel.isVisible() and not fw.log_view.isVisible() and not fw.card.isVisible()
        fw.full_esc.activated.emit()
        QApplication.processEvents()
        _back = fw._left_panel.isVisible() == _vis0 and fw.log_view.isVisible() and not fw.btn_full.isChecked()
        fw.show_current()
        _w0 = fw.view.pix.pixmap().width()
        fw.view._zoomed = True
        fw._zoom_detail()
        QApplication.processEvents()
        _w1 = fw.view.pix.pixmap().width()
        fw.view._fit()
        fw.view._zoomed = False
        check(_col_ok and _hid and _back and _w1 >= _w0 and _w1 == fw.state.shape[1],
              f"Adjust Colours boosts one colour and leaves the rest; Full Window hides the panels and Esc brings "
              f"them back; zooming in draws the picture at full resolution ({_w0} -> {_w1} px wide)")

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
        _sig_params = dict(fw.current_params())
        _sig_pre = fw.state.rgb.copy()
        fw.apply_choice()
        QApplication.processEvents()
        h_, w_ = fw.state.rgb.shape[1:]
        corner_ = (fw.state.rgb[:, int(h_ * 0.75):, int(w_ * 0.5):] - before_sig[:, int(h_ * 0.75):, int(w_ * 0.5):])
        top_ = (fw.state.rgb[:, :int(h_ * 0.5), :] - before_sig[:, :int(h_ * 0.5), :])
        _tz = np.argwhere(np.abs(top_).max(0) > 0)
        if len(_tz):
            try:
                from PySide6.QtGui import QImage
                _dimg = np.clip(np.abs(fw.state.rgb - before_sig).max(0) * 20, 0, 1)
                _qi = QImage((_dimg * 255).astype(np.uint8).copy().data, _dimg.shape[1], _dimg.shape[0],
                             _dimg.shape[1], QImage.Format_Grayscale8)
                _qi.save(str(out_dir / "21c_signature_diff.png"))
                log(f"signature diagnostics: params {_sig_params}; change before the drag/apply "
                    f"{float(np.abs(_sig_pre - before_sig).max()):.4f}")
            except Exception as _e:  # noqa: BLE001
                log(f"signature diagnostics failed: {_e}")
        check("signature" in fw.state.done and float(abs(corner_).max()) > 0.2 and float(abs(top_).max()) == 0.0,
              f"the signature is drawn in the bottom-right corner only (corner change {float(abs(corner_).max()):.3f}, "
              f"top change {float(abs(top_).max()):.4f}"
              + (f" at rows {_tz[:, 0].min()}-{_tz[:, 0].max()}, cols {_tz[:, 1].min()}-{_tz[:, 1].max()} of {h_}x{w_}"
                 if len(_tz) else "") + f"; steps end {fw.state.done[-3:]})")
        # flipping / turning a signed picture keeps the signature the right way round, in its place
        _nsig = len(fw.project.steps)
        _sigp = dict(fw.project.steps[-1]["params"])
        fw.open_tool("flip")
        _note = fw.choice_note.text()
        fw.btn_flip_h.click()
        fw.btn_rot_cw.click()
        QApplication.processEvents()
        _fpp = fw.current_params()
        fw.apply_choice()
        QApplication.processEvents()
        _want = F.TOOLS["signature"].apply(F.TOOLS["flip"].apply(
            F.State(rgb=before_sig.copy(), linear=False), _fpp), _sigp).rgb
        _tools = [x["tool"] for x in fw.project.steps]
        check(len(fw.project.steps) == _nsig + 1 and _tools[-2:] == ["flip", "signature"]
              and fw.state.rgb.shape == _want.shape and float(np.abs(fw.state.rgb - _want).max()) < 2e-3
              and "already signed" in _note,
              "flipping or turning a signed picture puts the signature back on top the right way round "
              f"(steps now end {_tools[-2:]})")
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
        # masters from different setups: every master used, and lined up WITH the field distortion
        import types as _types
        import stacking as _stg
        _rg = np.random.default_rng(5)
        _RH = 600
        _rstars = [(_rg.uniform(5, _RH - 5), _rg.uniform(5, _RH - 5), _rg.uniform(300, 8000)) for _ in range(350)]

        def _rend(fn, gain=1.0, off=500.0, noise=8.0):
            _yy, _xx = np.mgrid[:_RH, :_RH].astype(np.float32)
            im = np.full((_RH, _RH), off, np.float32) + _rg.normal(0, noise, (_RH, _RH)).astype(np.float32)
            for _x, _y, _a in _rstars:
                X, Y = fn(_x, _y)
                if 4 < X < _RH - 4 and 4 < Y < _RH - 4:
                    sl = (slice(int(Y) - 5, int(Y) + 6), slice(int(X) - 5, int(X) + 6))
                    im[sl] += gain * _a * np.exp(-((_xx[sl] - X) ** 2 + (_yy[sl] - Y) ** 2) / (2 * 1.6 ** 2))
            return im

        def _dist(x, y, k=6e-7, rot=0.003, dx=1.3, dy=-0.7):
            c = _RH / 2
            u, v = x - c, y - c
            q = 1 + k * (u * u + v * v)
            u, v = u * q, v * q
            return u * np.cos(rot) - v * np.sin(rot) + c + dx, u * np.sin(rot) + v * np.cos(rot) + c + dy
        _ms = [_types.SimpleNamespace(filter=f, data=d[None], path=Path(n)) for f, d, n in (
            ("Ha", _rend(lambda x, y: (x, y)), "ha"), ("OIII", _rend(_dist, 0.6, 300.0), "o3a"),
            ("OIII", _rend(lambda x, y: _dist(x, y, 4e-7, -0.002, -2, 1), noise=12.0), "o3b"),
            ("SII", _rend(lambda x, y: (x, y), 0.4), "s2"))]
        _rstars_bak = _rstars
        _rstars = [(_rg.uniform(5, _RH - 5), _rg.uniform(5, _RH - 5), _rg.uniform(300, 8000)) for _ in range(350)]
        _ms.append(_types.SimpleNamespace(filter="OIII", data=_rend(lambda x, y: (x, y))[None], path=Path("o3_other_field")))
        _rstars = _rstars_bak
        _ch, _cs, _notes = F.assemble_mono(_ms)
        _rp = _stg.find_stars(_ch["Ha"] * 65535, 1, 250)[0]
        _op = _stg.find_stars(_ch["OIII"] * 65535, 1, 250)[0]
        _dd = np.sqrt(((_rp[:, None] - _op[None]) ** 2).sum(-1)).min(1)
        _far = np.hypot(_rp[:, 0] - _RH / 2, _rp[:, 1] - _RH / 2) > 220
        _corner = float(np.median(_dd[_far & (_dd < 5)]))
        check(set(_ch) == {"Ha", "OIII", "SII"} and any("2 OIII masters combined" in x for x in _notes)
              and any("o3_other_field" in x and "left out" in x for x in _notes)
              and any("distortion" in x for x in _notes) and _corner < 0.3,
              f"masters from other setups are all used and lined up with their field distortion (corner stars "
              f"{_corner:.2f} px apart; {'; '.join(_notes)})")

        # after lining up, every master's corner stars are checked against the reference; one still
        # off (a master the distortion fit could not bend into place) is left out when its filter has others
        _real_eo = F._edge_offset
        _calls = []

        def _fake_eo(ref_pts, img, shape):
            _calls.append(1)
            return (3.0, 0.2) if len(_calls) == 2 else _real_eo(ref_pts, img, shape)
        F._edge_offset = _fake_eo
        try:
            _ch3, _cs3, _notes3 = F.assemble_mono(_ms[:4])
        finally:
            F._edge_offset = _real_eo
        check(any("o3b" in x and "3.0 px off" in x and "left out" in x for x in _notes3)
              and any("alignment check" in x for x in _notes) and "OIII" in _ch3,
              f"page 4 checks every master's corner stars after lining up and leaves out one still off "
              f"({'; '.join(_notes3)})")

        # starless masters (StarXTerminator on page 3): lined up by starless + stars-only together - the
        # starless image alone has no stars; masters page 3 already put on one grid are used as they are
        _sky = lambda: (500 + _rg.normal(0, 8, (_RH, _RH))).astype(np.float32)
        _sl = [_types.SimpleNamespace(filter="Ha", data=_sky()[None], path=Path("ha_sl"),
                                      cards=[("EXPTOTAL", 9000.0, "")]),
               _types.SimpleNamespace(filter="OIII", data=_sky()[None], path=Path("o3_sl"),
                                      cards=[("EXPTOTAL", 3000.0, "")])]
        _stars_img = {"ha_sl": (_rend(lambda x, y: (x, y), off=0.0, noise=0.5) / 65535.0)[None],
                      "o3_sl": (_rend(_dist, 0.6, off=0.0, noise=0.5) / 65535.0)[None]}
        _chs, _css, _notess = F.assemble_mono(_sl, lambda m: _stars_img[m.path.stem])
        _okd = any("corrected for a different field distortion" in x for x in _notess)
        _rp2 = _stg.find_stars(_css["Ha"] * 65535, 1, 250)[0]
        _op2 = _stg.find_stars(_css["OIII"] * 65535, 1, 250)[0]
        _dd2 = np.sqrt(((_rp2[:, None] - _op2[None]) ** 2).sum(-1)).min(1)
        _far2 = np.hypot(_rp2[:, 0] - _RH / 2, _rp2[:, 1] - _RH / 2) > 220
        _corner2 = float(np.median(_dd2[_far2 & (_dd2 < 5)]))
        _sg = [_types.SimpleNamespace(filter="Ha", data=_rend(lambda x, y: (x, y))[None], path=Path("ha_g"),
                                      cards=[("ALIGNREF", "grid.fits", ""), ("EXPTOTAL", 9000.0, "")]),
               _types.SimpleNamespace(filter="OIII", data=_rend(lambda x, y: (x, y), 0.6)[None], path=Path("o3_g"),
                                      cards=[("ALIGNREF", "grid.fits", ""), ("EXPTOTAL", 3000.0, "")])]
        _chg, _csg, _notesg = F.assemble_mono(_sg)
        check(_okd and _corner2 < 0.3 and any("already lined up on page 3" in x for x in _notesg),
              f"starless masters are lined up by their stars-only images (corner stars {_corner2:.2f} px apart); "
              "masters page 3 already lined up are used as they are")

        # a master page 3 already put on the grid is not moved further away by a wrong re-fit (Sh2-129: the SII
        # was shifted 65 px and the picture cropped narrower)
        _keep_rd, _keep_eo = _stg.register_distortion, F._edge_offset
        _sg3 = [_types.SimpleNamespace(filter=f_, data=_rend(lambda x, y: (x, y), k_)[None], path=Path(f_ + "_g"),
                                       cards=[("ALIGNREF", "grid.fits", ""), ("EXPTOTAL", e_, "")])
                for f_, k_, e_ in (("Ha", 1.0, 9000.0), ("OIII", 0.6, 3000.0), ("SII", 0.4, 2000.0))]

        def _bad_fit(ref_full, full, shape, **kw):
            _m, _i = _keep_rd(ref_full, full, shape, **kw)
            _m = dict(_m)
            _m["coef"] = np.array(_m["coef"], float).copy()
            _m["coef"][0, 0] += 40.0                              # a wrong fit: 40 px sideways
            return _m, dict(_i, shift=40.0, edge_affine=40.0)

        def _eo(ref_img, img, shape):
            r_ = _keep_eo(ref_img, img, shape)
            return None if r_ is None else (max(r_[0], 0.9), r_[1])   # the quick check is fooled by the nebula
        try:
            _stg.register_distortion, F._edge_offset = _bad_fit, _eo
            _ch3, _cs3, _n3 = F.assemble_mono(_sg3)
        except Exception as _e3:  # noqa: BLE001
            _ch3, _n3 = None, [str(_e3)]
        finally:
            _stg.register_distortion, F._edge_offset = _keep_rd, _keep_eo
        _sh3 = None if _ch3 is None else F._edge_offset(_ch3["Ha"], _ch3["SII"], _ch3["Ha"].shape)
        _ok3 = (_sh3 is not None and _sh3[0] < 1.0 and _ch3["SII"].shape == _sg3[0].data[0].shape
                and not any("cropped to the area" in x for x in _n3))
        check(_ok3, f"masters page 3 already lined up stay where page 3 put them when a new fit would move them "
                    f"further off - no shift ({'-' if _sh3 is None else round(_sh3[0], 2)} px), no narrower crop")

        # the deepest master is the reference - not simply the first Ha (a short, poor Ha master
        # used to drag every other master onto its grid)
        _md = [_types.SimpleNamespace(filter="Ha", data=_rend(lambda x, y: (x + 7, y - 4), 0.5)[None],
                                      path=Path("ha_short"), cards=[("EXPTOTAL", 1800.0, "")]),
               _types.SimpleNamespace(filter="OIII", data=_rend(lambda x, y: (x, y))[None], path=Path("o3_deep"),
                                      cards=[("EXPTOTAL", 17000.0, "")])]
        _chd, _csd, _notesd = F.assemble_mono(_md)
        check(any("o3_deep" in x and "deepest" in x for x in _notesd),
              f"page 4 lines the masters up to the deepest master ({'; '.join(_notesd)})")

        # HOO for Ha + OIII + SII sets, Flip the Picture, the two-line signature block, the description
        _nb = {k: np.random.default_rng(1).random((40, 50)).astype(np.float32) for k in ("Ha", "OIII", "SII")}
        _labels = [v.label for v in F.TOOLS["combine"].variants(F.State(channels=_nb, kind="mono_nb"))]
        _fs = F.State(rgb=np.random.default_rng(2).random((3, 30, 40)).astype(np.float32), linear=False)
        _lr = F.TOOLS["flip"].apply(_fs, {"h": True, "v": False}).rgb
        _ud = F.TOOLS["flip"].apply(_fs, {"h": False, "v": True}).rgb
        _bt = F.TOOLS["flip"].apply(_fs, {"h": True, "v": True}).rgb
        _flip_ok = (np.array_equal(_lr, _fs.rgb[:, :, ::-1]) and np.array_equal(_ud, _fs.rgb[:, ::-1, :])
                    and np.array_equal(_bt, _fs.rgb[:, ::-1, ::-1]))
        _cw = F.TOOLS["flip"].apply(_fs, {"rot": 1}).rgb
        _ccw = F.TOOLS["flip"].apply(_fs, {"rot": -1}).rgb
        _flip_ok = (_flip_ok and _cw.shape == (3, 40, 30) and _cw[0, 0, -1] == _fs.rgb[0, 0, 0]       # top-left -> top-right
                    and _ccw[0, -1, 0] == _fs.rgb[0, 0, 0]                                                 # top-left -> bottom-left
                    and F.FlipPicture.describe({"v": True, "rot": 3}) == "flipped vertically and turned 90° counter-clockwise")
        _wf = F.WORKFLOWS["mono_nb"]
        check("HOO" in _labels and _flip_ok and _wf.index("flip") < _wf.index("final_crop") < _wf.index("signature"),
              f"palettes for Ha+OIII+SII include HOO ({_labels}); Flip or Rotate flips horizontally and / or vertically and turns 90° either way, "
              "before the signature")
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
        # Darken the Sky also calms colour blotches in the darkest sky (narrowband: purple / blue), nebula untouched
        _cr = np.random.default_rng(4)
        _cyy, _cxx = np.mgrid[:300, :400].astype(np.float32)
        _cneb = 0.5 * np.exp(-((_cxx - 280) ** 2 + (_cyy - 150) ** 2) / (2 * 50.0 ** 2))
        _cb = lambda: F._blur(_cr.normal(0, 1, (300, 400)).astype(np.float32), 8) * 0.03  # noqa: E731
        _crgb = np.stack([0.08 + _cneb + _cb(), 0.08 + 0.6 * _cneb + _cb(), 0.08 + 0.2 * _cneb + _cb()])
        _cs = F.State(rgb=np.clip(_crgb, 0, 1).astype(np.float32), linear=False, kind="mono_nb")
        _cv = [v for v in F.TOOLS["sky"].variants(_cs) if v.recommended][0]
        _c0 = F.TOOLS["sky"].apply(_cs, {**_cv.params, "calm": 0.0}).rgb
        _c1 = F.TOOLS["sky"].apply(_cs, _cv.params).rgb
        _chr = lambda o, sl: float((o.max(0) - o.min(0))[sl].mean())  # noqa: E731
        _cbg, _cnb = (slice(20, 120), slice(10, 110)), (slice(130, 170), slice(260, 300))
        check(_cv.params.get("calm", 0) > 0 and _chr(_c1, _cbg) < 0.75 * _chr(_c0, _cbg)
              and abs(_chr(_c1, _cnb) - _chr(_c0, _cnb)) < 0.01,
              f"Darken the Sky calms colour blotches in the dark sky ({_chr(_c0, _cbg):.4f} -> {_chr(_c1, _cbg):.4f}) "
              f"and keeps the nebula's colour ({_chr(_c0, _cnb):.3f} -> {_chr(_c1, _cnb):.3f})")
        # ... and red blotches spread over ALL of a galaxy picture's sky (not only its darkest part) go too, the
        # galaxy's own colour kept (most galaxy pictures showed a blotchy red sky Calm sky colours didn't reach)
        _gr = np.random.default_rng(2)
        _gyy, _gxx = np.mgrid[:500, :700].astype(np.float32)
        _gbl = F._blur(_gr.normal(0, 1, (500, 700)).astype(np.float32), 10)
        _gbl /= _gbl.std()
        _ggal = 0.5 * np.exp(-(((_gxx - 350) * 0.9 + (_gyy - 250) * 0.4) ** 2 / (2 * 200 ** 2)
                               + ((_gxx - 350) * -0.4 + (_gyy - 250) * 0.9) ** 2 / (2 * 25 ** 2)))
        _gbase = 0.08 + _ggal
        _grgb = np.clip(np.stack([_gbase + 0.012 * _gbl + _gr.normal(0, 0.01, _gbase.shape),
                                  _gbase + _gr.normal(0, 0.01, _gbase.shape),
                                  _gbase * np.where(_ggal > 0.05, 1.4, 1.0) + _gr.normal(0, 0.01, _gbase.shape)]),
                        0, 1).astype(np.float32)
        _gs = F.State(rgb=_grgb, linear=False, kind="osc")
        _gv = [v for v in F.TOOLS["sky"].variants(_gs) if v.recommended][0]
        _go = F.TOOLS["sky"].apply(_gs, {**_gv.params, "amount": 0.0}).rgb
        _gsky, _ggm = _ggal < 0.01, _ggal > 0.2
        _gb0 = float(F._blur(_grgb[0] - _grgb[1], 4)[_gsky].std())
        _gb1 = float(F._blur(_go[0] - _go[1], 4)[_gsky].std())
        _gc0, _gc1 = float((_grgb[2] - _grgb[1])[_ggm].mean()), float((_go[2] - _go[1])[_ggm].mean())
        check(_gb1 < 0.5 * _gb0 and abs(_gc1 - _gc0) < 0.01,
              f"Calm sky colours clears red blotches across the whole sky of a galaxy picture ({_gb0:.4f} -> "
              f"{_gb1:.4f}) and keeps the galaxy's colour ({_gc0:.3f} -> {_gc1:.3f})")
        # Adjust Colours › Calm blues and purples works in the sky too (protection is for boosts only)
        _pyy, _pxx = np.mgrid[:200, :300].astype(np.float32)
        _pneb = 0.5 * np.exp(-((_pxx - 200) ** 2 + (_pyy - 100) ** 2) / (2 * 40.0 ** 2))
        _prgb = np.clip(np.stack([0.10 + _pneb, 0.06 + 0.6 * _pneb, 0.14 + 0.2 * _pneb]), 0, 1).astype(np.float32)
        _ps = F.State(rgb=_prgb, linear=False, kind="mono_nb")
        _pv = [v for v in F.TOOLS["colours"].variants(_ps) if v.label == "Calm blues and purples"]
        _po = F.TOOLS["colours"].apply(_ps, _pv[0].params).rgb if _pv else _prgb
        _pch = lambda a, sl: float((a.max(0) - a.min(0))[sl].mean())  # noqa: E731
        _pbg, _pnb = (slice(10, 60), slice(10, 60)), (slice(90, 110), slice(190, 210))
        check(bool(_pv) and _pch(_po, _pbg) < 0.6 * _pch(_prgb, _pbg) and abs(_pch(_po, _pnb) - _pch(_prgb, _pnb)) < 0.01,
              f"Adjust Colours › Calm blues and purples calms a purple sky ({_pch(_prgb, _pbg):.3f} -> "
              f"{_pch(_po, _pbg):.3f}) and keeps the gold ({_pch(_prgb, _pnb):.3f} -> {_pch(_po, _pnb):.3f})")
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
        # a galaxy with L, R, G, B, Ha AND OIII, SII masters is processed as LRGB (+ Ha regions), not narrowband
        import stacking as _stg2
        _gd = session.parent / "ic342_mix"
        _gd.mkdir(exist_ok=True)
        _gr = np.random.default_rng(2)
        _gyy, _gxx = np.mgrid[:160, :200]
        _gbase = np.exp(-((_gxx - 100) ** 2 + (_gyy - 80) ** 2) / 600.0)
        _gst = np.zeros((160, 200), np.float32)
        for _k in range(50):
            _gst[_gr.integers(5, 155), _gr.integers(5, 195)] = 1
        _gst = F._blur(_gst, 1.2) * 30
        _gfiles = []
        for _flt in ("L", "R", "G", "B", "Ha", "OIII", "SII"):
            _gi = (1000 + 3000 * _gbase + 20000 * _gst + _gr.normal(0, 20, (160, 200))).astype(np.float32)
            _go = _stg2.save_master_file(_gd / f"IC342_{_flt}_Master", _gi,
                                         [("IMAGETYP", "Master Light", ""), ("FILTER", _flt, ""), ("OBJECT", "IC342", ""),
                                          ("EXPTOTAL", 3600.0, ""), ("INSTRUME", "ZWO ASI2600MM Pro", "")], formats=("fits",))
            _gfiles.append(_go[0] if isinstance(_go, (list, tuple)) else _go)
        _gs = F.load_state(_gfiles)
        _gflow = [t.key for t, _st in F.workflow(_gs)]
        _grec = [v.label for v in F.TOOLS["combine"].variants(_gs) if v.recommended]
        check(_gs.kind == "mono_lrgb" and _gflow[:2] == ["ha_glow", "combine"] and _grec == ["LRGB"],
              f"a galaxy with LRGB + Ha + OIII + SII gets the LRGB workflow with Add Hydrogen Glow first "
              f"({_gs.kind}; {_gflow[:2]}; suggested {_grec})")
        # combining an RGB+Ha set without the hydrogen glow keeps "Add Hydrogen Glow" in the list (skipped),
        # with a clear way back - it must not just vanish (IC342 report)
        _hr = np.random.default_rng(5)
        _hch = {k: (0.1 + 0.01 * _hr.random((60, 80))).astype(np.float32) for k in ("R", "G", "B", "Ha")}
        _hs = F.State(channels=dict(_hch), kind="mono_rgb", target="IC342")
        _hs2 = F.TOOLS["combine"].apply(_hs, {"palette": "RGB"})
        _hok, _hwhy = F.TOOLS["ha_glow"].applicable(_hs2)
        _hstat = dict((t.key, st_) for t, st_ in F.workflow(_hs2)) if hasattr(F, "workflow") else {}
        check("ha_glow" in _hs2.skipped and not _hok and "Back to Start" in _hwhy
              and (not _hstat or _hstat.get("ha_glow") == "skipped"),
              f"Add Hydrogen Glow stays visible (skipped) after combining, and says how to go back ({_hwhy[:60]}...)")
        # Bring Out Hidden Structure leaves the empty sky alone (boosting it shows leftover blotches - "moire")
        _ry = np.random.default_rng(3)
        _yy4, _xx4 = np.mgrid[:300, :400].astype(np.float32)
        _blot = F._blur(_ry.normal(0, 1, (300, 400)).astype(np.float32), 12.0)
        _blot = 0.004 * _blot / float(_blot.std())
        _gal = 0.6 * np.exp(-((_xx4 - 280) ** 2 + (_yy4 - 150) ** 2) / (2 * 40.0 ** 2))
        _sky4 = np.clip(0.08 + _blot + _gal, 0, 1).astype(np.float32)
        _ls = F.State(rgb=np.stack([_sky4] * 3), linear=False)
        _lt = F.TOOLS["local"]
        _lv = [v for v in _lt.variants(_ls) if v.recommended][0]
        _sk = (slice(20, 120), slice(10, 110))
        _s0 = float(_ls.rgb[1][_sk].std())
        _s1 = float(_lt.apply(_ls, {**_lv.params, "protect": 0.0}).rgb[1][_sk].std())
        _s2 = float(_lt.apply(_ls, _lv.params).rgb[1][_sk].std())
        check(_s2 < 1.15 * _s0 < _s1, f"Bring Out Hidden Structure protects the sky (blotches {_s0:.4f}: "
                                       f"unprotected {_s1:.4f}, protected {_s2:.4f})")
        # ... and without a dark square or moat around a bright star's glow (M33 report)
        _c5 = 300
        _yy5, _xx5 = np.mgrid[:601, :601].astype(np.float32)
        _r5 = np.hypot(_xx5 - _c5, _yy5 - _c5)
        _g5 = np.clip(np.clip(np.exp(-_r5 ** 2 / 18.0) * 3, 0, 1) + 0.01 * np.exp(-_r5 / 12), 0, 1).astype(np.float32)
        _s5 = F.State(rgb=np.stack([_g5] * 3), stars=np.stack([_g5] * 3), stars_out=True, linear=False)
        _bt5 = F.TOOLS["star_stretch"]
        _o5 = _bt5.apply(_s5, [v for v in _bt5.variants(_s5) if v.recommended][0].params).stars[0]
        _ri5 = _r5.astype(int)
        _pr5 = np.bincount(_ri5.ravel(), _o5.ravel()) / np.bincount(_ri5.ravel())
        _rise5 = float(np.diff(_pr5[12:200]).max())
        _round5 = max(float(np.ptp(_o5[_ri5 == k])) for k in range(14, 120, 4))
        check(_rise5 < 0.005 and _round5 < 0.02,
              f"a bright star's glow fades smoothly and round - no dark square or ring (largest rise {_rise5:.4f}, "
              f"unevenness around a circle {_round5:.3f})")
        # a folder entry the drive answers with 'Invalid argument' (Mac + network drive report) is skipped
        import stacking as _stk
        from unittest import mock as _mock
        _libroot = session.parent / "cal_einval"
        (_libroot / "BadRig").mkdir(parents=True, exist_ok=True)
        _orig_isdir = Path.is_dir

        def _bad_isdir(self_):
            if self_.name == "BadRig":
                raise OSError(22, "Invalid argument", str(self_))
            return _orig_isdir(self_)
        with _mock.patch.object(Path, "is_dir", _bad_isdir):
            try:
                _lib = _stk.scan_library(_libroot)
                _einval_ok = _lib == []
            except OSError:
                _einval_ok = False
        check(_einval_ok, "a calibration folder entry the drive can't read (Errno 22) is skipped, not an error")
        # Mote Buster also flattens a BRIGHT over-corrected ring (dust that moved between flats and lights),
        # in the one colour that has it, and leaves the other colours alone
        _rb = np.hypot(_x2 - 300, _y2 - 300) / 70
        _ringb = 0.05 * np.exp(-((_rb - 0.7) / 0.18) ** 2) * (_rb < 1.05)
        _bimg = np.stack([_clean, _clean, _clean * (1 + _ringb)]) + _r2.normal(0, 0.0012, (3, _H2, _W2))
        _bst = F.State(rgb=_bimg.astype(np.float32), linear=True)
        _bfx = F.TOOLS["dust"].apply(_bst, {"spots": [[300, 300, 75]]}).rgb
        _ann = (_rb > 0.55) & (_rb < 0.85)
        _rex = lambda a: float(np.median(((a - _clean) / _clean)[_ann]))  # noqa: E731
        _bfound = F.find_donuts(_clean * (1 + 2.5 * _ringb) + _r2.normal(0, 0.0012, (_H2, _W2)), bright=True)
        check(_rex(_bimg[2]) > 0.03 and abs(_rex(_bfx[2])) < 0.006 and abs(_rex(_bfx[0]) - _rex(_bimg[0])) < 0.003
              and any(abs(x - 300) < 20 and abs(y - 300) < 20 for x, y, _r in _bfound),
              f"Mote Buster: a bright over-corrected ring is found and flattened in its own colour "
              f"({_rex(_bimg[2]):+.3f} -> {_rex(_bfx[2]):+.3f}), the others untouched")
        # Brighten Stars Separately keeps bright stars their real size (no big soft discs); faint ones as before
        _sy3, _sx3 = np.mgrid[:200, :400].astype(np.float32)
        _bstar = np.exp(-((_sx3 - 100) ** 2 + (_sy3 - 100) ** 2) / (2 * 2.0 ** 2)) \
            + 0.004 * np.exp(-np.hypot(_sx3 - 100, _sy3 - 100) / 8.0)               # a bright star and its glow
        _fstar = 0.03 * np.exp(-((_sx3 - 300) ** 2 + (_sy3 - 100) ** 2) / (2 * 2.0 ** 2))   # a faint star
        _sl = np.clip(_bstar + _fstar, 0, 1).astype(np.float32)
        _ss = F.State(rgb=np.stack([_sl] * 3), stars=np.stack([_sl] * 3), stars_out=True, linear=False)
        _bt = F.TOOLS["star_stretch"]
        _bv = [v for v in _bt.variants(_ss) if v.recommended][0]
        _big = _bt.apply(_ss, {**_bv.params, "small": 0.0}).stars[0]
        _sm3 = _bt.apply(_ss, _bv.params).stars[0]
        _rad = lambda im: int(np.sum(im[100, 100:200] > 0.25))  # noqa: E731
        check(_rad(_sm3) < 0.6 * _rad(_big) and abs(float(_sm3[100, 300]) - float(_big[100, 300])) < 0.02,
              f"Brighten Stars keeps a bright star small ({_rad(_big)} -> {_rad(_sm3)} px) and a faint star as bright "
              f"({float(_big[100, 300]):.2f} -> {float(_sm3[100, 300]):.2f})")
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
        # Save the Picture: the folder picked for one target is where the next new target starts
        import finish_window as _fwm
        _ed = _fwm.ExportDialog(fw2, fw2.settings, session.parent / "masters_A" / "Finished", "TargetA")
        _ed.dir.setText(str(session.parent / "My Final Pictures"))
        _ed.accept()
        _ed2 = _fwm.ExportDialog(fw2, fw2.settings, session.parent / "masters_B" / "Finished", "TargetB")
        _ed3 = _fwm.ExportDialog(fw2, fw2.settings, session.parent / "masters_A" / "Finished", "TargetA")
        check(_ed2.dir.text() == str(session.parent / "My Final Pictures") == _ed3.dir.text(),
              f"Save the Picture: a new target starts in the folder used last time ({Path(_ed2.dir.text()).name})")
        _ed2.reject()
        _ed3.reject()
        _pfn = F.picture_folder_name(fw2.state.target, fw2.state.sources)
        check(_pfn.startswith(fw2.state.target.split()[0]) and "_" in _pfn and "/" not in _pfn,
              f"Save the Picture puts every file of the picture in one folder named target_scope ({_pfn})")
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
        # Open a Saved Project from page 1: the Load window's button, and the folder it was saved into
        _ld = fb.LoadDataDialog(w, w.settings)
        _pb = [b for b in _ld.buttons.buttons() if b.text() == "Open a Saved Project…"]
        if _pb:
            _pb[0].click()
        _asked = bool(_pb) and _ld.wants_project
        _ld.deleteLater()
        _fwp = w.open_saved_project(str(_pd))                 # the folder it was saved into (<target>.szproj inside)
        QApplication.processEvents()
        check(_asked and F.find_project(_pd) == _p3.dir and _fwp is not None and _fwp.project is not None
              and _fwp.project.dir == _p3.dir and len(_fwp.project.steps) == _n and _fwp.state is not None
              and "act_open_project" in vars(w),
              f"page 1 opens a saved project straight onto Finishing Touches ({_n} steps from {_p3.dir.name})")
        if _fwp is not None:
            _fwp.close()
        # Process Existing Masters from page 1: a button and a menu item; from a folder only the newest restack
        _ld2 = fb.LoadDataDialog(w, w.settings)
        _mb = [b for b in _ld2.buttons.buttons() if b.text() == "Process Existing Masters…"]
        if _mb:
            _mb[0].click()
        _askedm = bool(_mb) and _ld2.wants_masters
        _ld2.deleteLater()
        import prepare as _prp
        _vs = [Path(f"/x/T_Ha_Master_2026-06-29_to_2026-07-03{v}.fits") for v in ("", "_4", "_8")] + [Path("/x/T_OIII_Master.fits")]
        _nv = sorted(f.name for f in _prp.newest_versions(_vs))
        check(_askedm and "act_open_masters" in vars(w) and _nv == ["T_Ha_Master_2026-06-29_to_2026-07-03_8.fits", "T_OIII_Master.fits"],
              f"page 1 can go straight to existing master lights; a folder gives the newest restack only ({_nv})")
        # menu items pass Qt's 'checked' flag: Open a Saved Project from the menu must not crash on it
        import subzeroastro as _sza
        _keep_ged = _sza.QFileDialog.getExistingDirectory
        _sza.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
        try:
            w.open_saved_project(False)                  # what the menu's triggered(checked) signal calls
            _menu_ok = True
        except Exception as _me:  # noqa: BLE001
            _menu_ok = str(_me)
        finally:
            _sza.QFileDialog.getExistingDirectory = _keep_ged
        check(_menu_ok is True, f"File › Open a Saved Project from the menu opens the folder picker ({_menu_ok})")
        # in-app updates: the newest GitHub release, the blue banner, Update Now's helper (no network needed)
        import updater as _up
        import zipfile as _uzf
        _an = _up.asset_name()
        _rels = [{"tag_name": "v9.0.200-beta", "draft": True, "assets": [{"name": _an}]},
                 {"tag_name": "v9.0.150-beta", "assets": [{"name": "SomethingElse.zip"}]},
                 {"tag_name": "v9.0.130-beta", "name": "SubZeroAstro 9.0.130 beta", "body": "New: updates.",
                  "html_url": "https://github.com/bapoteet/SubZeroAstro/releases/tag/v9.0.130-beta",
                  "assets": [{"name": _an, "browser_download_url": "https://example.invalid/a.zip", "size": 150e6}]},
                 {"tag_name": "v9.0.125-beta", "assets": [{"name": _an}]},
                 {"tag_name": "v0.9.100-beta", "assets": [{"name": _an}]}]
        _nr = _up.newest_release(_rels, "0.9.120 beta")
        check(_nr is not None and _nr["tag"] == "v9.0.130-beta" and _nr["version"] == "9.0.130"
              and _up.newest_release(_rels, "9.0.130 beta") is None and _up.is_newer("v10.0", "9.0.130 beta")
              and not _up.is_newer("v0.9.119-beta", "0.9.119 beta") and _up.is_newer("v0.9.120-beta", fb.APP_VERSION) is False
              and _up.asset_name("win32") == "SubZeroAstro-Windows.zip"
              and _up.asset_name("darwin") == "SubZeroAstro-macOS-AppleSilicon.zip",
              f"updates: the newest published release for this computer is found ({_nr and _nr['tag']}); "
              f"drafts, older versions and other platforms are ignored")
        _ub = w.update_banner
        _ub.settings.remove(_up.SKIP_KEY)
        _ub.start_check(manual=True, fetch=lambda: _rels)
        _ub.checker.wait(10000)
        QApplication.processEvents()
        _shown = _ub.isVisibleTo(w) and "9.0.130" in _ub.label.text() and fb.APP_VERSION in _ub.label.text()
        _btn = _ub.b_update.text()
        _ub.skip()
        _ub.show_release(_nr)
        _skipped = not _ub.isVisibleTo(w) and _ub.settings.value(_up.SKIP_KEY) == "v9.0.130-beta"
        _ub.show_release(_nr, force=True)
        _forced = _ub.isVisibleTo(w)
        _ub.hide()
        _ub.settings.remove(_up.SKIP_KEY)
        _said = []
        _keep_info = _ub._info
        _ub._info = lambda t, m: _said.append(m)
        _ub.start_check(manual=True, fetch=lambda: _rels[-1:])
        _ub.checker.wait(10000)
        QApplication.processEvents()
        _ub._info = _keep_info
        check(_shown and _btn in ("Update Now", "Download…") and _skipped and _forced and not _ub.isVisibleTo(w)
              and any("newest version" in m for m in _said)
              and "act_check_updates" in vars(w) and w.act_auto_update.isCheckable(),
              f"updates: a newer version shows the blue banner with '{_btn}'; Skip This Version hides it; "
              f"Help › Check for Updates says when you're up to date ({_said[:1]})")
        _uz = session.parent / "update_test"
        _uz.mkdir(exist_ok=True)
        _zip = _uz / "SubZeroAstro-Windows-src.zip"
        with _uzf.ZipFile(_zip, "w") as _z:
            _z.writestr("SubZeroAstro/SubZeroAstro.exe", b"MZ fake")
            _z.writestr("SubZeroAstro/_internal/base_library.zip", b"x")
        _stage = _uz / "stage"
        _stage.mkdir(exist_ok=True)
        _got = _up.download(_zip.as_uri(), _stage / "SubZeroAstro-Windows.zip")
        _cmd = _up.prepare_handover(_got, _stage, platform="win32", pid=4242,
                                    target=_uz / "Installed" / "SubZeroAstro")
        _bat = (_stage / "update.bat").read_text(encoding="utf-8")
        _new = _stage / "new" / "SubZeroAstro"
        _mac = _up.mac_script(4242, _got, Path("/Applications/SubZeroAstro.app"), _uz / "mstage")
        _mac2 = _up.mac_script(4242, _got, Path("/Users/me/Downloads/SubZeroAstro 9120.app"), _uz / "mstage")
        check(str(_uz / "mstage" / "x" / "SubZeroAstro.app" / "Contents") in _mac2
              and str(Path("/Users/me/Downloads/SubZeroAstro 9120.app") / "Contents.new") in _mac2,
              "updates: a renamed copy (e.g. 'SubZeroAstro 9120.app') is updated in place from the zip's SubZeroAstro.app")
        _tl = _up._untranslocated(Path("/Applications/SubZeroAstro.app"))
        check(_tl == Path("/Applications/SubZeroAstro.app") and isinstance(_up.why_not(), str)
              and "Update Now:" in __import__("zip_logs").system_note()
              and "Contents.new" in _mac and "mv " in _mac,
              f"updates: on the Mac the new version is copied inside SubZeroAstro.app and swapped in only when "
              f"complete; the logs say whether Update Now can install here ({_up.why_not() or 'yes'})")
        check(_cmd[:2] == ["cmd", "/c"] and (_new / "SubZeroAstro.exe").is_file()
              and (_new / "_internal" / "base_library.zip").is_file()
              and "PID eq 4242" in _bat and "robocopy" in _bat and "SubZeroAstro.exe" in _bat
              and "ditto -x -k" in _mac and f'open "{Path("/Applications/SubZeroAstro.app")}"' in _mac
              and _up.install_target("darwin", "/Applications/SubZeroAstro.app/Contents/MacOS/SubZeroAstro", True)
              == Path("/Applications/SubZeroAstro.app")
              and _up.install_target("win32", "C:/Apps/SubZeroAstro/SubZeroAstro.exe", True)
              == Path("C:/Apps/SubZeroAstro") and _up.install_target(frozen=False) is None,
              "updates: Update Now downloads and unpacks the new version and writes the helper that swaps it in "
              "after SubZeroAstro closes, then restarts it (Windows and Mac)")
        # Update Now from start to the hand-over: the download thread has ended before SubZeroAstro quits (a
        # thread still running at the quit made macOS report 'SubZeroAstro quit unexpectedly')
        _qs = {}
        _keep = (_up.prepare_handover, _up.launch, _up.staging_dir, _up.why_not, _up.QApplication)
        try:
            _ust = _uz / "ustage"
            _ust.mkdir(exist_ok=True)
            _up.staging_dir = lambda: _ust
            _up.why_not = lambda platform=None: ""
            _up.prepare_handover = lambda zp, st: ["true"]
            _up.launch = lambda cmd: None
            class _FakeApp:
                closeAllWindows = staticmethod(lambda: None)
                quit = staticmethod(lambda: _qs.setdefault(
                    "finished", _ub.downloader is not None and _ub.downloader.isFinished()))
            _up.QApplication = _FakeApp
            _ub.release = {"url": _zip.as_uri(), "version": "9.9.9", "tag": "v9.9.9-beta"}
            _ub.update_now(ask=False)
            _t0 = time.time()
            while "finished" not in _qs and time.time() - _t0 < 20:
                QApplication.processEvents()
                time.sleep(0.02)
        finally:
            _up.prepare_handover, _up.launch, _up.staging_dir, _up.why_not, _up.QApplication = _keep
        check(_qs.get("finished") is True, f"updates: Update Now waits for the download to end before closing "
                                            f"SubZeroAstro (no 'quit unexpectedly' on the Mac) ({_qs})")
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
                    "Whirlpool Galaxy": "galaxy", "Sh2-155": "nebula", "my target": "", "NGC 1097": "galaxy",
                    "M83": "galaxy", "NGC 281": "nebula", "C_2025_Panstarrs": "comet", "C2025 PANSTARRS": "comet"}
        _bad_t = {k: _tg.classify(k) for k, v in _names_t.items() if _tg.classify(k) != v}
        _wf_neb = F.workflow(F.State(rgb=np.zeros((3, 4, 4), np.float32), kind="osc", target="NGC 7000"))
        _wf_cl = F.workflow(F.State(rgb=np.zeros((3, 4, 4), np.float32), kind="osc", target="M45"))
        check(not _bad_t and [t.key for t, _ in _wf_neb] == [t.key for t, _ in _wf_cl if t.key != "cluster_core"]
              and "cluster_core" in [t.key for t, _ in _wf_cl]
              and F.target_type(F.State(target="M45")) == "star cluster"
              and "stars_out" in F.TYPE_TUNING["star cluster"]["optional"]
              and "sky" in F.TYPE_TUNING["comet"]["recommend"],
              f"target types recognised ({_bad_t or 'all right'}); a star cluster keeps every step (plus Resolve the "
              "Cluster Core), with star removal made optional")
        # a galaxy gets Tame Bright Core as a recommended step (right after brightening); other targets
        # only when a burnt-out core is found
        _st_g = F.State(rgb=np.full((3, 40, 40), 0.3, np.float32), kind="osc", target="M31", linear=False,
                        done=["balance", "scnr", "stars_out", "stretch"])
        _st_n = F.State(rgb=np.full((3, 40, 40), 0.3, np.float32), kind="osc", target="NGC 7000", linear=False,
                        done=["balance", "scnr", "stars_out", "stretch"])
        _cg = dict((t.key, stt) for t, stt in F.workflow(_st_g))
        _cn = dict((t.key, stt) for t, stt in F.workflow(_st_n))
        check(_cg.get("core") == "next" and _cn.get("core") == "optional"
              and "galaxy" in F.why_next(_st_g, F.TOOLS["core"]).lower(),
              f"Tame Bright Core is the next step for a galaxy ({_cg.get('core')}) and optional for a nebula "
              f"({_cn.get('core')})")
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
        # comet processing on page 4: nucleus, Reveal Comet Jets, Bring Out the Tail, filters lined up on the comet
        import finish as _cf
        _ch, _cw = 600, 800
        _gy, _gx = np.mgrid[0:_ch, 0:_cw].astype(np.float32)
        _ncx, _ncy = 470.0, 250.0
        _rr = np.hypot(_gx - _ncx, _gy - _ncy)
        _ua = np.deg2rad(200)
        _al = (_gx - _ncx) * np.cos(_ua) + (_gy - _ncy) * np.sin(_ua)
        _pp = -(_gx - _ncx) * np.sin(_ua) + (_gy - _ncy) * np.cos(_ua)
        _ja = np.deg2rad(-60)
        _jl = (_gx - _ncx) * np.cos(_ja) + (_gy - _ncy) * np.sin(_ja)
        _jp = -(_gx - _ncx) * np.sin(_ja) + (_gy - _ncy) * np.cos(_ja)
        _cimg = (0.08 + 0.5 * np.exp(-_rr / 20) + 0.15 * np.exp(-_rr / 90)
                 + np.where(_al > 0, 0.12 * np.exp(-_al / 350) * np.exp(-(_pp / (15 + 0.15 * _al)) ** 2), 0)
                 + np.where((_jl > 5) & (_jl < 70), 0.06 * np.exp(-(_jp / 2.5) ** 2) * np.exp(-_jl / 50), 0)
                 + np.random.default_rng(5).normal(0, 0.004, (_ch, _cw))).astype(np.float32)
        _cs = _cf.State(rgb=np.clip(np.stack([_cimg * 0.9, _cimg, _cimg * 0.95]), 0, 1), linear=False,
                        kind="osc", target="C/2026 X1 (Test)")
        _nu = _cf.comet_nucleus(_cf._lum(_cs.rgb))
        _wfk = [t.key for t, _st in _cf.workflow(_cs)]
        _wfo = [t.key for t, _st in _cf.workflow(_cf.State(rgb=_cs.rgb, linear=False, kind="osc", target="M 42"))]
        _jt, _tt = _cf.TOOLS["comet_jets"], _cf.TOOLS["comet_tail"]
        _jo = _jt.apply(_cs, _jt.variants(_cs)[1].params)
        _jm = (np.abs(_jp) < 2) & (_jl > 15) & (_jl < 45)
        _jb = (np.abs(_jp) > 7) & (np.abs(_jp) < 12) & (_jl > 15) & (_jl < 45)
        _jc0 = float(_cf._lum(_cs.rgb)[_jm].mean() - _cf._lum(_cs.rgb)[_jb].mean())
        _jc1 = float(_cf._lum(_jo.rgb)[_jm].mean() - _cf._lum(_jo.rgb)[_jb].mean())
        _to = _tt.apply(_cs, _tt.variants(_cs)[1].params)

        def _tail_ratio(img):
            yl = _cf._lum(img)
            return float(yl[(_al > 120) & (_al < 200) & (np.abs(_pp) < 8)].mean()
                         / yl[(_rr > 120) & (_rr < 200) & (_al < -40)].mean())
        check(_nu is not None and abs(_nu[0] * _cw - 0.5 - _ncx) < 2.5 and abs(_nu[1] * _ch - 0.5 - _ncy) < 2.5
              and "comet_jets" in _wfk and "comet_tail" in _wfk and "comet_jets" not in _wfo
              and _jc1 > 1.5 * _jc0 and _tail_ratio(_to.rgb) > 1.15 * _tail_ratio(_cs.rgb),
              f"comets on page 4: the head is found ({_nu and round(_nu[0] * _cw - 0.5, 1)}, "
              f"{_nu and round(_nu[1] * _ch - 0.5, 1)} - truly {_ncx:g}, {_ncy:g}); Reveal Comet Jets brings out a "
              f"jet ({_jc0:.3f} -> {_jc1:.3f}) and Bring Out the Tail lifts the tail over the round coma "
              f"({_tail_ratio(_cs.rgb):.2f} -> {_tail_ratio(_to.rgb):.2f}); offered only for comets")

        def _cchan(dx, dy, seed):
            rr = np.hypot(_gx - (_ncx + dx), _gy - (_ncy + dy))
            c = 0.05 + 0.4 * np.exp(-rr / 20) + 0.1 * np.exp(-rr / 80)
            c = c + np.random.default_rng(seed).normal(0, 0.002, (_ch, _cw))
            for i in range(40):
                g = np.random.default_rng(300 + i)
                sx, sy = g.uniform(0, _cw), g.uniform(0, _ch)
                c = c + 0.4 * np.exp(-((_gx - sx) ** 2 + (_gy - sy) ** 2) / (2 * 1.3 ** 2))
            return c.astype(np.float32)
        _chs = {"L": _cchan(0, 0, 1), "R": _cchan(6, -3, 2), "G": _cchan(3, -1.5, 3), "B": _cchan(-4, 2, 4)}
        _rb = _chs["R"].copy()
        _an = _cf.align_comet_channels(_chs)
        _pos = {k: _cf.comet_nucleus(v) for k, v in _chs.items()}
        _spread = max(np.hypot((p[0] - _pos["L"][0]) * _cw, (p[1] - _pos["L"][1]) * _ch) for p in _pos.values())
        _far = _rr > 250
        check(_an and "lined up" in _an[0] and _spread < 0.6 and float(np.abs(_chs["R"][_far] - _rb[_far]).max()) < 0.01,
              f"comet imaged filter by filter (NINA Orbitals: R, G, B, L): the comet is lined up across the filters "
              f"(heads now within {_spread:.2f} px) while the stars stay put ({_an[:1]})")
        # Save Project As can keep the masters with the project (one folder per target)
        import shutil as _ksh
        _kd = session.parent / "keep_masters_test"
        _ksh.rmtree(_kd, ignore_errors=True)
        (_kd / "masters").mkdir(parents=True)
        _km1 = _kd / "masters" / "M101_L_Master_2026-10-15.fits"
        _km1.write_bytes(b"x" * 1000)
        (_kd / "masters" / "M101_L_Master_2026-10-15_stars.fits").write_bytes(b"y" * 500)
        (_kd / "masters" / "M101_L_Master_2026-10-15_stacklog.txt").write_text("log")
        _km2 = _kd / "masters" / "M101_R_Master.fits"
        _km2.write_bytes(b"z" * 800)
        (_kd / "masters" / "Other_target.fits").write_bytes(b"o")
        _kp = _cf.Project(_kd / "work")
        _kst = _cf.State(rgb=np.zeros((3, 40, 50), np.float32), target="M101", sources=[str(_km1), str(_km2)])
        _kp.save(_kst, "load", "Masters loaded")
        _kp.save(_kst, "stretch", "Brighten")
        _kp2 = _kp.save_as(_kd / "M101 Pinwheel", keep_masters=True)
        _knames = sorted(x.name for x in (_kp2.dir / "Masters").iterdir())
        _ksh.move(str(_kd / "M101 Pinwheel"), str(_kd / "moved"))
        _kp3 = _cf.Project(_kd / "moved")
        _ksrc = _kp3.load(1).sources
        check(_knames == ["M101_L_Master_2026-10-15.fits", "M101_L_Master_2026-10-15_stacklog.txt",
                          "M101_L_Master_2026-10-15_stars.fits", "M101_R_Master.fits"]
              and _km1.exists() and all(Path(x).exists() and "moved" in x for x in _ksrc)
              and _kp3.keep_masters() == (0, 0),
              f"Save Project As keeps the masters with the project ({len(_knames)} files: masters, star images, "
              "stack log - not other targets'), the originals stay, and the moved project still finds them")
        # a page-3 (prepared) master brings its stacked master along; projects copied between Mac and PC find theirs
        (_kd / "pm" / "Prepared").mkdir(parents=True)
        (_kd / "pm" / "M51_L_Master.fits").write_bytes(b"s" * 300)
        (_kd / "pm" / "M51_L_Master_stacklog.txt").write_text("log")
        _kpp = _kd / "pm" / "Prepared" / "M51_L_Master_prepared.xisf"
        _kpp.write_bytes(b"p" * 300)
        _kq = _cf.Project(_kd / "w2")
        _kq.save(_cf.State(rgb=np.zeros((3, 20, 20), np.float32), target="M51", sources=[str(_kpp)]), "load", "x")
        _kq2 = _kq.save_as(_kd / "M51", keep_masters=True)
        _kn2 = sorted(x.name for x in (_kq2.dir / "Masters").iterdir())
        _kq2.manifest["sources"] = [r"C:\Users\me\Astro\Prepared\M51_L_Master_prepared.xisf"]   # written on a PC
        check(_kn2 == ["M51_L_Master.fits", "M51_L_Master_prepared.xisf", "M51_L_Master_stacklog.txt"]
              and Path(_kq2.resolve_sources(_kq2.manifest["sources"])[0]).exists(),
              f"keeping the masters also keeps the stacked master behind a page-3 one ({_kn2}), and a project "
              "saved on a PC finds its masters on a Mac (and the other way round)")
        # Organize My Astro Files: a messy folder sorted by header, duplicates spotted, moved, then undone
        import organizer as _og
        _od = session.parent / "organize_test"
        _ksh.rmtree(_od, ignore_errors=True)
        (_od / "dump" / "sub").mkdir(parents=True)
        _orng = np.random.default_rng(7)

        def _ofits(name, typ, obj=None, flt=None, exp=60.0, t="2026-10-02T03:10:00", **extra):
            hd = {"IMAGETYP": typ, "EXPTIME": exp, "DATE-OBS": t, "INSTRUME": "ZWO ASI2600MM Pro",
                  "GAIN": 100, "OFFSET": 50, "CCD-TEMP": -10.0, "SITELONG": -84.5,
                  "TELESCOP": "SQA60", "FOCALLEN": 360}
            if obj:
                hd["OBJECT"] = obj
            if flt:
                hd["FILTER"] = flt
            hd.update(extra)
            core.write_fits(_od / "dump" / name, (_orng.random((24, 32)) * 4000).astype(np.uint16), hd, bitpix=16)
        _ofits("L1.fits", "Light Frame", "M 31", "Ha", 180.0, "2026-10-02T03:10:00")
        _ofits("L2.fits", "Light Frame", "M 31", "Ha", 180.0, "2026-10-02T03:14:00")
        _ksh.copy2(_od / "dump" / "L2.fits", _od / "dump" / "sub" / "L2 copy.fits")          # a true duplicate
        _ofits("sub/L1 (2).fits", "Light Frame", "M 31", "Ha", 180.0, "2026-10-02T03:10:00")  # same capture, other bytes
        _ofits("L3.fits", "Light Frame", "M 31", "L", 120.0, "2026-10-09T03:10:00", TELESCOP="RASA8", FOCALLEN=400)
        _ofits("MD.fits", "Dark Frame", None, None, 180.0, "2026-09-01T12:00:00", NCOMBINE=30)
        (_od / "dump" / "MD.fits").rename(_od / "dump" / "MasterDark_180s.fits")
        _ofits("M31_final.fits", "Light Frame", "M 31", "Ha", 180.0, "2026-10-02T03:10:00", NCOMBINE=40)
        _ofits("F1.fits", "Flat Field", None, "Ha", 1.5, "2026-10-02T10:00:00")
        _ofits("D1.fits", "Dark Frame", None, None, 180.0, "2026-09-01T12:00:00")
        _ofits("B1.fits", "Bias Frame", None, None, 0.001, "2026-09-01T12:00:00")
        (_od / "dump" / "notes.txt").write_text("not a picture")
        _oitems = _og.plan(_og.classify(_og.scan(_od / "dump"), layout="target"), _od / "dump")
        _odst = {i.src.name: str(i.dst.relative_to(_og.org_base(_od / "dump"))).replace("\\", "/") for i in _oitems
                 if i.status == "move"}
        _odup = [i.src.name for i in _oitems if i.status == "duplicate"]
        _night = _odst.get("L1.fits", "").split("/")[2] if _odst.get("L1.fits", "").count("/") >= 3 else "?"
        check(_odst.get("L1.fits", "").startswith("Targets/M31/") and "/LIGHT/Ha/" in _odst["L1.fits"]
              and _odst.get("F1.fits", "").startswith(f"Targets/M31/{_night}/FLAT/Ha")
              and _odst.get("D1.fits", "").startswith("Darks library/") and "/180s/" in _odst["D1.fits"]
              and _odst.get("B1.fits", "").startswith("Bias library/SQA60/ASI2600MM_G100_O50_-10C/")
              and _odst.get("L3.fits", "").startswith("Targets/M31/2026-10-08_RASA8/LIGHT/L")
              and _odst.get("M31_final.fits", "") == "Targets/M31/Masters/M31_final.fits"
              and _odst.get("MasterDark_180s.fits", "").startswith("Calibration masters/SQA60/ASI2600MM_G100_O50_-10C/") and sorted(_odup) == ["L1 (2).fits", "L2 copy.fits"],
              f"Organize My Astro Files sorts by header - lights and that night's flats under the target "
              f"({_night}), darks, bias and calibration masters by rig and camera settings, the rig added when a target was shot with two - and spots the duplicate ({_odup})")
        # a folder of calibration frames only, sorted by scope (Flats/<filter>, Darks/<exposure>, Bias), some
        # without IMAGETYP: flats find the target already organized from that night (right scope), others are
        # kept by scope and night; darks and bias go to the libraries
        _oc = session.parent / "organize_cal_test"
        _ksh.rmtree(_oc, ignore_errors=True)
        (_og.org_base(_oc) / "Targets" / "M31" / "2026-10-01_SQA60" / "LIGHT" / "Ha").mkdir(parents=True)
        (_og.org_base(_oc) / "Targets" / "M31" / "2026-10-08_RASA8" / "LIGHT" / "L").mkdir(parents=True)

        def _cfits(rel, hd):
            _p = _oc / "Calibration" / rel
            _p.parent.mkdir(parents=True, exist_ok=True)
            _h = {"INSTRUME": "ZWO ASI2600MM Pro", "GAIN": 100, "OFFSET": 50, "CCD-TEMP": -10.0, "SITELONG": -84.5,
                  "EXPTIME": 1.0}
            _h.update(hd)
            core.write_fits(_p, (_orng.random((10, 12)) * 1000).astype(np.uint16), _h, bitpix=16)
        _cfits("SQA60/Flats/Ha/f1.fits", {"IMAGETYP": "Flat", "FILTER": "Ha", "TELESCOP": "SQA60",
                                          "DATE-OBS": "2026-10-02T10:00:00"})
        _cfits("SQA60/Flats/OIII/f2.fits", {"FILTER": "OIII", "TELESCOP": "SQA60", "DATE-OBS": "2026-10-02T10:05:00"})
        _cfits("RASA8/Flats/L/f3.fits", {"IMAGETYP": "Flat", "FILTER": "L", "TELESCOP": "RASA8",
                                         "DATE-OBS": "2026-10-06T10:00:00"})
        _cfits("RASA8/Darks/300s/d1.fits", {"IMAGETYP": "Dark", "TELESCOP": "RASA8", "EXPTIME": 300.0,
                                            "DATE-OBS": "2026-09-02T10:00:00"})
        _cfits("RASA8/Bias/b1.fits", {"IMAGETYP": "Bias", "TELESCOP": "RASA8", "EXPTIME": 0.0,
                                      "DATE-OBS": "2026-09-02T10:00:00"})
        _cfits("SQA60/Darks/300s/d2.fits", {"IMAGETYP": "Dark", "TELESCOP": "SQA60", "EXPTIME": 300.0,
                                            "DATE-OBS": "2026-09-02T10:00:00"})
        _cfits("Esprit100/Darks/300s/d3.fits", {"IMAGETYP": "Dark", "EXPTIME": 300.0, "DATE-OBS": "2026-09-02T10:00:00"})
        _cdst = {i.src.name: str(i.dst.relative_to(_og.org_base(_oc))).replace("\\", "/")
                 for i in _og.plan(_og.classify(_og.scan(_oc / "Calibration"), library=_oc, layout="target"), _oc) if i.dst}
        check(_cdst.get("f1.fits") == "Targets/M31/2026-10-01_SQA60/FLAT/Ha/f1.fits"
              and _cdst.get("f2.fits") == "Targets/M31/2026-10-01_SQA60/FLAT/OIII/f2.fits"
              and _cdst.get("f3.fits") == "Flats/RASA8/2026-10-05/L/f3.fits"
              and _cdst.get("d1.fits", "").startswith("Darks library/RASA8/ASI2600MM_G100_O50_-10C/300s/")
              and _cdst.get("b1.fits", "").startswith("Bias library/RASA8/ASI2600MM_G100_O50_-10C/")
              and _cdst.get("d2.fits", "").startswith("Darks library/SQA60/ASI2600MM_G100_O50_-10C/300s/")
              and _cdst.get("d3.fits", "").startswith("Darks library/Esprit100/ASI2600MM_G100_O50_-10C/300s/"),
              f"Organize a calibration-only folder (by scope, some files without IMAGETYP): flats join the target "
              f"shot that night with that scope, the rest are kept by scope and night ({_cdst.get('f3.fits')}); "
              "darks and bias go to each rig's own library - three ASI2600MMs on three rigs, shot at the same "
              "second, stay three cameras")
        # the rig-first layout (the default): Lights / rig / target / night / filter and Calibration / rig / …
        _rdst = {i.src.name: str(i.dst.relative_to(_og.org_base(_oc))).replace("\\", "/")
                 for i in _og.plan(_og.classify(_og.scan(_oc / "Calibration"), library=_oc), _oc) if i.dst}
        _rl = {i.src.name: str(i.dst.relative_to(_og.org_base(_od / "dump"))).replace("\\", "/")
               for i in _og.plan(_og.classify(_og.scan(_od / "dump")), _od / "dump") if i.status == "move"}
        check(_rl.get("L1.fits", "").startswith("Light Frames/SQA60/M31/2026-10-01/Ha/")
              and _rl.get("L3.fits", "").startswith("Light Frames/RASA8/M31/2026-10-08/L/")
              and _rl.get("F1.fits", "").startswith("Calibration Frames/SQA60/Flats/ASI2600MM/2026-10-01/Ha/")
              and _rl.get("D1.fits", "").startswith("Calibration Frames/SQA60/Darks/ASI2600MM_G100_O50_-10C/180s/")
              and _rl.get("B1.fits", "").startswith("Calibration Frames/SQA60/Bias/ASI2600MM_G100_O50_-10C/")
              and _rl.get("M31_final.fits", "") == "Light Frames/SQA60/M31/Masters/M31_final.fits"
              and _rl.get("MasterDark_180s.fits", "").startswith("Calibration Frames/SQA60/Masters/ASI2600MM_G100_O50_-10C/")
              and _rdst.get("d3.fits", "").startswith("Calibration Frames/Esprit100/Darks/")
              and _rdst.get("f2.fits", "").startswith("Calibration Frames/SQA60/Flats/ASI2600MM/2026-10-01/OIII/"),
              f"Organize, by rig (the default): Light Frames/rig/target/night/filter and Calibration Frames/rig/Flats, Darks, "
              f"Bias, Masters ({_rl.get('L1.fits')}; {_rl.get('D1.fits')})")
        # no telescope in the headers (only a focal length): the rig name comes from your own folders -
        # SQA60Pro/Veil/2026-09-01/… is filed under SQA60Pro, not "360mm", whether the drive or that folder is picked
        _gr = session.parent / "organize_rigname_test" / "IncomingAstroData"
        _ksh.rmtree(_gr.parent, ignore_errors=True)

        def _gfits(rel, hd):
            _p = _gr / rel
            _p.parent.mkdir(parents=True, exist_ok=True)
            _h = {"INSTRUME": "ZWO ASI2600MC Pro", "GAIN": 100, "OFFSET": 50, "CCD-TEMP": -10.0, "SITELONG": -84.5,
                  "EXPTIME": 180.0, "FOCALLEN": 360}
            _h.update(hd)
            core.write_fits(_p, (_orng.random((10, 12)) * 1000).astype(np.uint16), _h, bitpix=16)
        _gfits("SQA60Pro/Veil/2026-09-01/Light_001.fits", {"IMAGETYP": "Light", "OBJECT": "Veil Nebula",
                                                           "DATE-OBS": "2026-09-02T03:00:00"})
        _gfits("SQA60Pro/Veil/2026-09-01/FLAT/Flat_001.fits", {"IMAGETYP": "Flat", "DATE-OBS": "2026-09-02T10:00:00",
                                                               "EXPTIME": 1.0})
        _gfits("RedCat/M31/2026-09-03/Light_001.fits", {"IMAGETYP": "Light", "OBJECT": "M31", "FOCALLEN": 250,
                                                        "DATE-OBS": "2026-09-04T03:00:00"})
        _gd = {}
        for _groot in (_gr, _gr / "SQA60Pro"):
            for _gi in _og.plan(_og.classify(_og.scan(_groot), library=_groot, root=_groot), _groot):
                _gd[(_groot.name, _gi.src.name, _gi.src.parent.name)] = str(_gi.dst.relative_to(_og.org_base(_groot))).replace("\\", "/")
        check(_gd.get(("IncomingAstroData", "Light_001.fits", "2026-09-01"), "").startswith("Light Frames/SQA60Pro/VeilNebula/")
              and _gd.get(("IncomingAstroData", "Light_001.fits", "2026-09-03"), "").startswith("Light Frames/RedCat/M31/")
              and _gd.get(("IncomingAstroData", "Flat_001.fits", "FLAT"), "").startswith("Calibration Frames/SQA60Pro/Flats/")
              and _gd.get(("SQA60Pro", "Light_001.fits", "2026-09-01"), "").startswith("Light Frames/SQA60Pro/"),
              f"Organize: with no telescope in the headers, the rig is named from your folders "
              f"({_gd.get(('IncomingAstroData', 'Light_001.fits', '2026-09-01'))}), not the focal length")
        check(any(a is w.act_organize for mm in w.menuBar().actions() if mm.menu() for a in mm.menu().actions()),
              "File › Organize My Astro Files… is in the File menu")
        w.settings.setValue("organize/intro_ok", True)       # the first-time note is a click-through dialog
        # Your Rigs: name <-> focal length(s) within ±5 %, a second focal length for a reducer; nothing guessed -
        # a focal length that fits no rig, or two, waits for you
        _yl = [{"name": "SQA60", "focals": [360, 290]}, {"name": "RASA8", "focals": [400], "camera": "ASI2600MM"},
               {"name": "Redcat", "focals": [410], "camera": "ASI2600MC"}]

        class _Fi:
            def __init__(self, f, c="ZWO ASI2600MM Pro"):
                self.focal, self.camera = f, c
        _ym = [_og.match_rig(_Fi(f, c), "/x/y.fits", _yl, 0.05)[0] or _og.match_rig(_Fi(f, c), "/x/y.fits", _yl, 0.05)[1]
               for f, c in ((352, None), (371, None), (290, None), (400, "ZWO ASI2600MM Pro"),
                            (405, "ZWO ASI2600MC Pro"), (530, None), (None, None))]
        _yl2 = [dict(r) for r in _yl]
        _yl2[2] = {"name": "Redcat", "focals": [410]}
        _yamb = _og.match_rig(_Fi(405), "/data/x/y.fits", _yl2, 0.05)
        _yl2[2]["folders"] = ["/data/Redcat"]
        _yfold = _og.match_rig(_Fi(405), "/data/Redcat/M31/y.fits", _yl2, 0.05)
        check(_ym == ["SQA60", "SQA60", "SQA60", "RASA8", "Redcat", "nomatch", "nofocal"]
              and _yamb == (None, "ambiguous: RASA8, Redcat") and _yfold == ("Redcat", "folder"),
              f"Your Rigs: 352 and 371 mm are the SQA60 (360 ±5 %), 290 mm its reducer, the camera splits two rigs "
              f"of about the same focal length, 530 mm matches none, and a folder you assigned settles a tie ({_ym})")
        _cp = [{"name": "SQA105", "focals": [506]}, {"name": "Pleiades111", "focals": [530]}]
        _cpm = [_og.match_rig(_Fi(f), "/x", _cp, 0.03)[0] for f in (495, 512, 518, 525, 545)]
        check(_cpm == ["SQA105", "SQA105", None, "Pleiades111", "Pleiades111"],
              f"Your Rigs, two scopes close in focal length (506 and 530 mm, ±3 %): the clearly nearer one wins, "
              f"a reading right between them is asked about ({_cpm})")
        _keep_rl = (w.settings.value("organize/riglist", "[]"), w.settings.value("organize/rig_tol", 5),
                    w.settings.value("stack/rigs", "[]"))
        # a drive that answers "Invalid argument" (Errno 22) when asked about an entry (seen on a Mac with an
        # external volume): that folder or file is skipped, the scan carries on
        _e22 = session.parent / "organize_errno22"
        _ksh.rmtree(_e22, ignore_errors=True)
        (_e22 / "FL132").mkdir(parents=True)
        core.write_fits(_e22 / "FL132" / "a.fits", np.zeros((8, 8), np.uint16), {"IMAGETYP": "Light"}, bitpix=16)
        (_e22 / "FL132" / "manifest.json").write_text("{}")
        import pathlib as _pl
        _keep_isfile, _keep_exists = _pl.Path.is_file, _pl.Path.exists

        def _bad_isfile(self):
            if self.name == "manifest.json":
                raise OSError(22, "Invalid argument", str(self))
            return _keep_isfile(self)
        _pl.Path.is_file = _bad_isfile
        try:
            _e22s = _og.scan(_e22)
            _e22ok = True
        except OSError as _e:
            _e22s, _e22ok = [], str(_e)
        finally:
            _pl.Path.is_file = _keep_isfile
        check(_e22ok is True and [p.name for p in _e22s] == ["a.fits"],
              f"Organize: a drive entry that answers 'Invalid argument' (Errno 22) is skipped, the look-through "
              f"carries on ({_e22ok})")
        # PixInsight masters often carry no gain or temperature: the folder is named from what is known -
        # never with a '?' (Windows refuses it: WinError 123) - and the exposure comes from the file name
        _pm = session.parent / "organize_pi_masters"
        _ksh.rmtree(_pm, ignore_errors=True)
        (_pm / "SQA106FlatsMay28").mkdir(parents=True)
        for _n in ("masterBias_BIN-1_6384x4258.fits", "masterDark_BIN-1_6384x4258_EXPOSURE-120.00s.fits",
                   "Dark_EXPOSURE-300.00s.fits"):
            core.write_fits(_pm / "SQA106FlatsMay28" / _n, np.zeros((6, 6), np.uint16),
                            {"IMAGETYP": "Master Dark" if "ark" in _n else "Master Bias",
                             "INSTRUME": "ZWO ASI6200MC Pro", "FOCALLEN": 530}, bitpix=16)
        _pmi = _og.plan(_og.classify(_og.scan(_pm), library=_pm, root=_pm,
                                     riglist=[{"name": "SQA106", "focals": [530]}]), _pm)
        _pmd = sorted(str(i.dst.relative_to(_og.org_base(_pm))).replace("\\", "/") for i in _pmi)
        check(len(_pmd) == 3 and not any(c in x for x in _pmd for c in '?*:<>|"')
              and all(x.startswith("Calibration Frames/SQA106/Masters/ASI6200MC/") for x in _pmd)
              and _og._exp_folder(type("I", (), {"exptime": None})(), "masterDark_EXPOSURE-120.00s.xisf") == "120s",
              f"Organize: masters without gain or temperature in the header get a folder Windows accepts "
              f"(no '?'), exposure read from the name ({_pmd[0] if _pmd else '-'})")
        _owin = w.open_organizer()
        _owin.edit_rigs(rows=[{"name": "SQA60", "focals": [360]}, {"name": "RASA8", "focals": [400]}], tol_pct=5)
        _owin.layout_box.setCurrentIndex(_owin.layout_box.findData("rig"))
        _owin.same.setChecked(True)
        _owin.set_sources([_od / "dump" / "sub", _od / "dump"])        # several folders at once
        _owin.move.setChecked(True)
        _owin.scan(wait=True)
        _on = len([i for i in _owin.items if i.status == "move"])
        _odd = _owin.open_duplicates()
        _odn = _odd.table.rowCount() if _odd else 0
        _odd.act(True, ask=False, wait=True)
        check(_odn == 2 and not (_od / "dump" / "sub" / "L2 copy.fits").exists()
              and (_od / "dump" / "sub" / "L1 (2).fits").exists() and (_od / "dump" / "L2.fits").exists(),
              f"Organize › Duplicates… lists {_odn} duplicates and deletes only the exact copy - the same capture "
              f"with different bytes is kept for you to check ({_owin.status.text()[:90]})")
        _owin.go(ask=False, wait=True)
        _oleft = (_owin.table.rowCount(), "left where they are" in _owin.status.text())
        _owin.safe_to_delete(wait=True)
        _osr = getattr(_owin, "safe_result", None) or {}
        _osafe = ([p.name for p, _ in _osr.get("keep", [])], [p.name for p, _ in _osr.get("other", [])],
                  "Not yet" in _owin.status.text())
        check(_osafe == (["L1 (2).fits"], ["notes.txt"], True),
              f"Safe to Delete?: after a Move, the one astro file that exists only in the old folder is flagged "
              f"(don't delete), the notes file is listed for you to check, and the verdict says 'Not yet' ({_osafe})")
        _omoved = sorted(str(p.relative_to(_od / "dump")).replace("\\", "/") for p in (_od / "dump").rglob("*.fits"))
        _ogone = not (_od / "dump" / "L1.fits").exists() and (_od / "dump" / "sub" / "L1 (2).fits").exists()
        _osteps = (_owin.b_go.text().startswith("✓"), _owin.b_rigs.text().startswith("✓"))
        _owin.undo_last(ask=False, wait=True)
        _oback = sorted(p.name for p in (_od / "dump").glob("*.fits"))
        check(_on == 8 and all(_osteps) and _oleft == (1, True) and _ogone and len(_omoved) == 9 and _oback == ["B1.fits", "D1.fits", "F1.fits", "L1.fits", "L2.fits", "L3.fits", "M31_final.fits", "MasterDark_180s.fits"]
              and not (_og.org_base(_od / "dump") / "Light Frames").exists(),
              f"(steps ticked off: {_osteps}; left behind and why: {_oleft}) "
              f"File › Organize My Astro Files… moves {_on} files into place (the duplicate stays put) and Undo "
              f"Last Run puts every one back ({_oback})")
        _owin.src.setText(str(_gr))
        _owin.move.setChecked(False)
        _owin.scan(wait=True)
        _y1 = sorted({i.why.split(" - ")[0] for i in _owin.items if i.status == "norig"})
        _owin.edit_rigs(rows=[{"name": "SQA60Pro", "focals": [361]}, {"name": "RedCat", "focals": [250]}], tol_pct=5)
        _y2 = {i.src.parent.name: str(i.dst.relative_to(_og.org_base(_gr))).replace("\\", "/") for i in _owin.items
               if i.status == "move"}
        check(any("250 mm" in x for x in _y1) and _y2.get("2026-09-03", "").startswith("Light Frames/RedCat/M31/")
              and _y2.get("2026-09-01", "").startswith("Light Frames/SQA60Pro/VeilNebula/")
              and not [i for i in _owin.items if i.status == "norig"],
              f"Organize: a rig not in Your Rigs is held back ({_y1[:1]}); after Add New Rig the plan updates at "
              f"once, without reading the files again ({_y2.get('2026-09-03')})")
        for _k, _v in zip(("organize/riglist", "organize/rig_tol", "stack/rigs"), _keep_rl):
            w.settings.setValue(_k, _v)
        _owin.close()
        # Tonight's Target Picker: the sky worked out here (no internet), N.I.N.A.'s objects, a list N.I.N.A. imports
        import datetime as _pdt
        import target_picker as _tp
        _pcat, _psrc = _tp.load_catalog()
        _pm31 = next((t for t in _pcat if t.name == "M 31"), None)
        _pnt, _ppl = _tp.plan_night([_pm31] if _pm31 else [], 32.78, -96.8, _pdt.date(2026, 10, 15),
                                    _tp.Horizon(), 30.0)
        _pp = _ppl[0] if _ppl else None
        _psec = _tp.parse_view("open sky from NNW to SSW")
        check(len(_pcat) > 3000 and _pm31 is not None and _pm31.common == "Andromeda Galaxy" and _pp is not None
              and 7 < _pp.hours < 10.5 and 80 < _pp.peak_alt < 83 and 8.5 < _pnt.dark_hours < 10.5
              and _psec[12] == 0 and _psec[4] == 90,
              f"Tonight's Target Picker: {len(_pcat)} objects ({_psrc}); from Dallas on 15 Oct 2026 M 31 is up "
              f"{_pp.hours if _pp else 0:.1f} h of the {_pnt.dark_hours:.1f} dark hours, peaking at "
              f"{_pp.peak_alt if _pp else 0:.0f}°; 'open sky from NNW to SSW' opens the west and blocks the east")
        _pwin = w.open_target_picker()
        _pwin.lat.setValue(32.78)
        _pwin.lon.setValue(-96.8)
        from PySide6.QtCore import QDate as _QDate
        _pwin.date.setDate(_QDate(2026, 10, 15))
        _pwin.view.setText("open sky from NNW to SSW")
        _pwin._apply_view()
        _pwin.focal.setValue(360)
        _pwin.pixel.setValue(3.76)
        _pwin.wpx.setValue(6248)
        _pwin.hpx.setValue(4176)
        _pwin.compute()
        _pshown = _pwin.visible_plans()
        _pwin.group.setCurrentIndex(1)
        _pgroups = _pwin.tree.topLevelItemCount()
        _pcsv = session.parent / "picker_test.csv"
        _pwin.tree.setCurrentItem(_pwin._first_plan_item())
        _pwin.save_csv(str(_pcsv))
        _plines = _pcsv.read_text(encoding="utf-8").splitlines() if _pcsv.exists() else []
        check(len(_pshown) > 10 and _pgroups >= 2 and len(_plines) == 2
              and _plines[0].startswith("Familiar Name,Catalogue Entry,Right Ascension (J2000),Declination (J2000)"),
              f"File › Tonight's Target Picker… lists {len(_pshown)} targets for a western view, groups them by "
              f"type ({_pgroups} groups) and saves a target list N.I.N.A. imports ({_plines[1:2]})")
        # straight into N.I.N.A.'s target list: plain, then with one of those as the template
        import json as _pj
        _pnf = session.parent / "nina_targets_test"
        _ksh.rmtree(_pnf, ignore_errors=True)
        _pwin.settings.setValue("picker/nina_template", "")
        _pw1 = _pwin.send_to_nina(folder=_pnf, ask=False) or []
        _pdoc = _pj.loads(_pw1[0].read_text(encoding="utf-8")) if _pw1 else {}
        _pdoc["Items"]["$values"].append({"$id": "99", "$type": "NINA.Sequencer.SequenceItem.Imaging.TakeExposure, "
                                          "NINA.Sequencer", "ExposureTime": 300.0, "Coordinates": dict(
                                              _pdoc["Target"]["InputCoordinates"], **{"$id": "98"})})
        _ptpl = _pnf / "template.json"
        _ptpl.write_text(_pj.dumps(_pdoc), encoding="utf-8")
        _pwin.choose_template(str(_ptpl))
        _pm = next((it for it in _pwin.plans if it.target.name == "M 31"), None)
        _pwin.tree.clearSelection()
        _pwin.find.setText("M 31")
        _pwin.tree.setCurrentItem(_pwin._first_plan_item())
        _pw2 = _pwin.send_to_nina(folder=_pnf, ask=False) or []
        _pd2 = _pj.loads(_pw2[0].read_text(encoding="utf-8")) if _pw2 else {}
        _pc = _pd2.get("Items", {}).get("$values", [{}])[-1].get("Coordinates", {})
        check(len(_pw1) == 1 and _pdoc.get("$type", "").startswith("NINA.Sequencer.Container.DeepSkyObjectContainer")
              and _pd2.get("Target", {}).get("TargetName") == "Andromeda Galaxy" and _pc.get("RAHours") == 0
              and _pc.get("RAMinutes") == 42 and _pc.get("DecDegrees") == 41
              and _pd2["Items"]["$values"][-1].get("ExposureTime") == 300.0,
              f"Send to N.I.N.A. writes targets into N.I.N.A.'s target list ({_pw1[0].name if _pw1 else '-'}), and "
              f"with one of your own as the template keeps its steps and moves every coordinate to the new target "
              f"({_pw2[0].name if _pw2 else '-'})")
        _pwin.find.setText("")
        _pwin.close()
        # a calibration subframe that vanished after it was added is left out (not a failed stack); mount
        # drivers in TELESCOP don't become the telescope's name
        import stacking as _sk
        _mb = session.parent / "missing_bias_test"
        _mb.mkdir(parents=True, exist_ok=True)
        for _i in range(6):
            core.write_fits(_mb / f"bias_{_i}.fits", (np.random.default_rng(_i).random((20, 30)) * 50 + 500).astype(np.uint16),
                            {"IMAGETYP": "Bias Frame", "GAIN": 125}, bitpix=16)
        _mi = _sk.CalItem("bias", [_sk.read_info(p) for p in sorted(_mb.glob("*.fits"))], False)
        (_mb / "bias_2.fits").unlink()
        _mm = _sk.build_master(_mi)

        class _Ti:
            telescope, focal = "AvalonStarGo.NET", 400.0
        _rg = _sk.Rigs.suggestion(_Ti())
        check(_mm is not None and _mm.shape == (20, 30) and _mi.count == 5 and _mi.skipped
              and _mi.skipped[0][0] == "bias_2.fits" and _rg == "400mm scope",
              f"a calibration subframe deleted after it was added is left out and the master still builds "
              f"({_mi.count} of 6); a mount driver in TELESCOP gives '{_rg}', not the mount's name")
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
        QTimer.singleShot(900, lambda: (shot("19a_new_project.png"), fast_track_test()))

    def fast_track_test():
        # ⚡ Fast Track from page 1: open the frames again, then one press runs measuring, stacking, page 3 and
        # every page-4 step on the suggested settings, with a film of the steps; the pages are all there afterwards
        import fasttrack_window as _ftw
        w.open_folder(session, use_cache=True)
        w._fast_auto = True
        check(hasattr(w, "btn_fast") and "Fast Track" in w.btn_fast.text(), "page 1 has the ⚡ Fast Track button")
        state["ft_open"] = time.time()

        def _go():
            if not w.frames and time.time() - state["ft_open"] < 60:
                QTimer.singleShot(300, _go)
                return
            state["ft"] = _ftw.start(w, "frames")
            state["ft_t"] = time.time()
            QTimer.singleShot(800, fast_track_wait)
        QTimer.singleShot(300, _go)

    def fast_track_wait():
        import fasttrack_window as _ftw
        ft = state["ft"]
        if ft is not None and not ft.finished_ok and not ft.cancelled and time.time() - state["ft_t"] < 600:
            QTimer.singleShot(500, fast_track_wait)
            return
        shot("19b_fast_track.png", ft) if ft is not None else None
        fw_ = getattr(ft, "fw", None) if ft is not None else None
        steps = [s_["label"] for s_ in fw_.project.steps] if fw_ is not None and fw_.project else []
        log(f"Fast Track: {time.time() - state['ft_t']:.0f} s, {len(ft.frames_shot) if ft else 0} film frames, "
            f"steps: {steps}")
        check(ft is not None and ft.finished_ok and ft.target.replace(" ", "") == "M33",
              f"Fast Track runs from page 1's frames to a finished picture "
              f"({'finished' if ft and ft.finished_ok else 'not finished'}; "
              f"{[ft.steps.item(i).text() for i in range(ft.steps.count())][-4:] if ft else None})")
        check(ft is not None and len(ft.frames_shot) >= 5 and ft.cine._new is not None and ft.strip.count() == len(ft.frames_shot),
              f"Fast Track shows each step as a frame of its film ({len(ft.frames_shot) if ft else 0} frames)")
        check(fw_ is not None and fw_.state is not None and "sky" in fw_.state.done
              and ("denoise" in fw_.state.done or __import__("finish").TOOLS["denoise"].runner is None),
              f"Fast Track also darkens the sky and runs the final NoiseXTerminator pass when it can "
              f"({fw_.state.done if fw_ is not None and fw_.state is not None else None})")
        check(fw_ is not None and len(steps) >= 4 and fw_.state is not None and "stretch" in fw_.state.done
              and ft.btn_save.isVisibleTo(ft) and ft.btn_redo.isVisibleTo(ft),
              f"after Fast Track page 4 holds every step (Go Back to Here works) and the picture can be saved or "
              f"reprocessed step by step ({len(steps)} steps)")
        if fw_ is not None and ft is not None:
            # ✈ Fly Into It: the finished picture, big, and a slow flight towards its target
            try:
                _fimg, _ffocus = ft.flight_frame()
                ft.cine.start_flight(_ftw.rgb8_to_qimage(_fimg, _ftw.FLY_SIDE), _ffocus, "test flight")
                ft.cine.flight.setCurrentTime(_ftw.FLY_MS // 2)
                _fgrab = ft.cine.grab()
                _fok = ft.cine.flying() and not _fgrab.isNull() and 0.2 <= _ffocus[0] <= 0.8 and 0.2 <= _ffocus[1] <= 0.8
                shot("19c_fast_track_flight.png", ft)
            except Exception as _fe:  # noqa: BLE001
                _fok, _ffocus = False, repr(_fe)
            check(_fok and ft.btn_fly.isVisibleTo(ft), f"✈ Fly Into It flies slowly into the finished picture "
                                                        f"(target at {_ffocus})")
            ft.cine.stop_flight()
            ft.redo_by_hand()
            QApplication.processEvents()
            check(fw_.project is not None and len(fw_.project.steps) == 1 and fw_.state is not None
                  and not fw_.state.done, "Reprocess Step by Step opens page 4 back at the masters as loaded")
        finish(0 if not failures else 1)

    QTimer.singleShot(300, start)
    return app.exec()
