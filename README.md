# FITS Blaster for Windows — Stage 1 (culling)

A Windows app for quickly reviewing a night's FITS frames and setting aside the
bad ones. Inspired by Tom van Peer's FITS Blaster for macOS
(https://astrophoto-app.com/), written independently for Windows.

## Install and run (Windows 11)

1. Install **Python 3.11 or newer** from https://www.python.org/downloads/
   During setup, tick **"Add python.exe to PATH"**.
2. Unzip this folder somewhere permanent, e.g. `C:\Astro\FITS Blaster`.
3. Double-click **`run.bat`**.
   The first run sets things up (2–5 minutes, needs internet). After that it opens immediately.

Optional: double-click **`build_exe.bat`** to make a stand-alone
`dist\FITS Blaster\FITS Blaster.exe` you can pin to the taskbar.

If Windows SmartScreen warns about an unrecognized app, click **More info → Run anyway**
(the scripts are unsigned because they were made for you, not bought from a store).

## Using it

1. Click **Open Folder** (or press Ctrl+O, or drag a folder onto the window).
   Tick **Include subfolders** for multi-night sessions. Folders whose names start with
   FLAT, DARK, BIAS, CALIB or OFFSET are skipped.
2. Frames appear in the left sidebar as they load — you can start right away.
3. Step through with the arrow keys. Press **X** (or Delete) to reject a frame; it jumps to the next one.
4. To reject a run of frames: click the first, Shift+click the last, press **X**.
5. Made a mistake? **Ctrl+Z**, or select the frame and press X again to bring it back.

Rejected frames are **moved** into a `REJECTED` subfolder next to them. Nothing is ever
deleted and your images are never modified. When you're done, point your stacking program
at the same folder — it will only see the frames you kept.

Use the dropdowns at the top of the sidebar to show just one filter (Ha, OIII, L…),
just kept or just rejected frames, or to sort by capture time.

## Keyboard shortcuts

| Key | Action |
|---|---|
| ↑ ↓ ← → | Previous / next frame |
| Shift + arrows, Shift/Ctrl + click | Select several frames |
| X or Delete | Reject / restore |
| Ctrl+Z | Undo |
| F / 1 / double-click | Fit to window / 100% / toggle |
| Mouse wheel, drag | Zoom, pan |
| Ctrl+O, F5 | Open folder, reload |
| Ctrl+E | Save a text list of kept frames |

## What it supports

- `.fits`, `.fit`, `.fts`; 8, 16 and 32-bit integer and 32/64-bit float data
- Mono and one-shot-colour (Bayer) cameras — colour frames are shown in greyscale for now
- Header info from N.I.N.A., SharpCap, ASIAIR, APT and similar (OBJECT, FILTER, EXPTIME, DATE-OBS…)
- Not supported: compressed `.fz` files

## Coming next

- **Stage 2:** star quality metrics (FWHM, eccentricity, SNR, star count), red/amber/green badges,
  session chart with drag-to-select, Auto-Flag
- **Stage 3:** optional stacking app — aligns and combines the kept frames into
  `<Target>_Master_<date>.fits`

## Files

- `fits_blaster.py` — the app window
- `fits_core.py` — FITS reading, stretching, scanning and the reject/restore file moves
- `run.bat`, `build_exe.bat`, `requirements.txt` — setup and launch
