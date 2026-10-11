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
        finish(0 if not failures else 1)

    QTimer.singleShot(300, start)
    return app.exec()
