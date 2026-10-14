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
    from PySide6.QtWidgets import QApplication
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
        pw.resize(1450, 900)
        QTimer.singleShot(900, shots_11)

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
        from prepare import target_out_dir as _tod
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
        import ui_header as _uh
        check(fw.app_header.page.text() == _uh.PAGE_FINISH and pw.app_header.page.text() == _uh.PAGE_PREPARE
              and state["pw"].parent().app_header.page.text() == _uh.PAGE_STACK,
              "every window shows the banner with its page title")
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
                fw._spot_click(120.0, 90.0, Qt.LeftButton)
                fw._spot_click(200.0, 150.0, Qt.LeftButton)
                fw._spot_click(0, 0, Qt.RightButton)
                check(len(fw.spots) == 1, "Spot Fix: click marks a spot, right-click removes the last")
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
        # Sign Your Picture
        before_sig = fw.state.rgb.copy()
        fw.open_tool("signature")
        fw.sign_text.setText("B. Poteet")
        fw._sign_changed()
        QApplication.processEvents()
        check(fw.sign_row.isVisibleTo(fw) and len(fw.variants) == 3,
              f"Sign Your Picture offers 3 fonts ({[v.label for v in fw.variants]})")
        shot("21_finish_signature.png", fw)
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
        fw2.close()
        state["pw"].close()
        state["win"].close()
        finish(0 if not failures else 1)

    QTimer.singleShot(300, start)
    return app.exec()
