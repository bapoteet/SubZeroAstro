"""
selftest.py - run the real app window on synthetic frames and check it works.

    python fits_blaster.py --selftest [output_folder]
    "FITS Blaster.exe" --selftest [output_folder]

Creates a small two-filter session with a few deliberately bad frames, opens it
in the real window, waits for measuring to finish, exercises flag / reject /
undo / Auto-Flag, saves screenshots and writes selftest_result.txt.
Exit code 0 = all checks passed. Used by the automatic GitHub build.
"""

from __future__ import annotations

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
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    import fits_blaster as fb

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

    session = Path(tempfile.mkdtemp(prefix="fb_selftest_")) / "M33"
    session.mkdir()
    t0 = time.time()
    bad = make_session(session)
    log(f"Created 16 synthetic frames in {time.time() - t0:.1f}s at {session}")

    w = fb.MainWindow()
    w.resize(1600, 950)
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
        w.view.fit()
        # --- auto-cull presets
        for key in ("less", "semi", "aggressive"):
            w.apply_preset(key)
            names = {f.name for f in w.frames.values() if f.flagged}
            check(set(bad) <= names, f"{key} preset flags all bad frames ({len(names)} flagged)")
            check(not (names - set(bad)) or key == "aggressive",
                  f"{key} preset flags no good frames ({sorted(names - set(bad))})")
        w.apply_preset("semi")
        QTimer.singleShot(600, shots_6)

    def shots_6():
        shot("6_semi_aggressive_flagged.png")
        # --- Load Data window with two folders
        second = session.parent / "M33_night2"
        second.mkdir()
        make_session(second)
        state["second"] = second
        dlg = fb.LoadDataDialog(w, w.settings, False)
        dlg.add_source(("folder", session, False))
        dlg.add_source(("folder", second, False))
        dlg.add_source(("folder", second, False))           # duplicate is ignored
        check(len(dlg.sources()) == 2, "Load Data lists 2 folders (duplicate ignored)")
        dlg.show()
        QTimer.singleShot(500, lambda: shots_7(dlg))

    def shots_7(dlg):
        shot("7_load_data_window.png", dlg)
        sources = dlg.sources()
        dlg.close()
        # --- Start Over, then process both folders together
        w.reset_session()
        check(not w.frames and w.list.count() == 0 and not w.chart.points, "Start Over clears the session")
        state["t2"] = time.time()
        w.open_sources(sources)
        QTimer.singleShot(300, wait_two)

    def wait_two():
        if w.loaded_count >= len(w.frames) and len(w.frames) > 0:
            check(len(w.frames) == 32, f"two folders loaded together ({len(w.frames)} frames)")
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
        origins = {(g.filter, k): getattr(g.plan, k).origin for g in win.groups for k in ("bias", "dark", "flat")
                   if getattr(g.plan, k) is not None}
        check(all(o == "library" for o in origins.values()),
              f"after stacking, the saved library masters are used next time ({origins})")
        QTimer.singleShot(600, shots_10)

    def shots_10():
        shot("10_stacking_done.png", state["win"])
        # --- Stage 4: prepare for PixInsight ---------------------------------------
        win = state["win"]
        check(win.btn_prepare.isEnabled(), "Prepare for PixInsight button enabled after stacking")
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
        if res:
            names = sorted(Path(f).name for f in res[0]["files"])
            log("prepared: " + ", ".join(names))
            check(sum(n.endswith("_prepared.xisf") for n in names) == 2, "two prepared XISF masters")
            check(any(n.endswith("_quicklook.jpg") for n in names) and any(n.endswith("_quicklook.tif")
                                                                           for n in names),
                  "quick-look JPG and 16-bit TIFF saved")
            check(any(n.endswith("PixInsight_next_steps.html") for n in names), "PixInsight plan saved")
            prepared = next(Path(f) for f in res[0]["files"] if str(f).endswith("_prepared.fits"))
            _d, _c, lk = core.read_fits(prepared)
            if state["fake"]:
                check(lk.get("PLTSOLVD") is True, "prepared master is plate solved (ASTAP)")
                check(any(n.endswith("_prepared_stars.xisf") for n in names), "StarXTerminator stars image saved")
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
        state["pw"].close()
        state["win"].close()
        finish(0 if not failures else 1)

    QTimer.singleShot(300, start)
    return app.exec()
