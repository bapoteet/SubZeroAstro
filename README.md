# FITS Blaster for Windows — Stage 2 (culling with quality metrics)

A Windows app for reviewing a night's (or several nights') FITS frames, measuring
star quality on every frame, and setting aside the bad ones. Inspired by Tom van
Peer's FITS Blaster for macOS (https://astrophoto-app.com/), written independently
for Windows.

## Download (easiest — no Python needed)

Go to the repository's **Releases** page (right-hand side on GitHub), download
**FITS-Blaster-Windows.zip** from the latest release, unzip it anywhere and run
**`FITS Blaster.exe`** inside the folder. If Windows SmartScreen warns about an
unrecognized app, click **More info → Run anyway**.

Every change pushed to GitHub is built and tested automatically on Windows
(see the **Actions** tab), including a self-test that opens the real window.
You can run the same self-test yourself: `python fits_blaster.py --selftest`
(or `"FITS Blaster.exe" --selftest`) — it writes `selftest_output\selftest_result.txt`
and screenshots.

## Updating from Stage 1

Unzip over your existing folder (or into a new one) and double-click **`run.bat`**
as before. Nothing new needs installing. If you built an `.exe`, run
`build_exe.bat` again.

## Install and run (first time)

1. Install **Python 3.11 or newer** from https://www.python.org/downloads/ and tick
   **"Add python.exe to PATH"** during setup.
2. Unzip this folder somewhere permanent, e.g. `C:\Astro\FITS Blaster`.
3. Double-click **`run.bat`**. The first run sets things up (2–5 minutes, needs internet).

Optional: **`build_exe.bat`** makes a stand-alone `dist\FITS Blaster\FITS Blaster.exe`.

## What's new in Stage 2

**Star quality on every frame** — FWHM (pixels and arcseconds), eccentricity, SNR,
star count and sky background, shown in the sidebar and the inspector, with a
green / amber / red badge and a score (100 = a typical frame for that filter).
Frames are only ever compared with frames from the **same camera and filter**.
Hover over a frame in the sidebar to see why it got its badge.

**Session chart** (below the image) — one dot per frame in capture order:
- choose what to plot: score, FWHM, eccentricity, SNR, star count or background
- dot colour = filter; dashed line = that filter's median
- rejected frames are faded with a red ✕, flagged frames have a yellow ring,
  the current frame is the big white dot
- dotted lines separate nights
- **click** a dot to jump to that frame; **drag** across the chart to select a range,
  then press **X** to reject them or **F** to flag them

**Flag, review, reject** — flag suspects with **F** (or drag on the chart, or
**Auto-Flag**), switch the sidebar to **Flagged** to step through them, then press X
on the ones to drop, or **Shift+X** to reject every flagged frame.

**Auto-Flag** (Ctrl+F) — flags frames worse than limits you choose (FWHM, eccentricity,
star count, SNR, background — all relative to the filter's median) and shows how many
frames match before you commit.

**Sort by quality** — sort the sidebar by score or any metric; tick **Worst first** to
put the frames to cull at the top.

**Session warnings** — if most frames share a problem (for example elongated stars
all night) a warning appears above the chart, since that points at the rig rather
than the sky.

## Speed (and network drives / NAS)

- **Fast star measurement** (default): reads about a quarter of each file — the
  thumbnail rows plus three bands across the frame — which gives nearly the same
  numbers as measuring every pixel and is 3–4× faster over a network.
  Change it in **Settings → Star measurement** (Fast / Full frame / Off).
- **Results are saved on your PC** (`%LOCALAPPDATA%\FITSBlasterWin\cache`).
  Reopening a session is nearly instant; when you add a new night, only the new
  frames are measured. Rejected frames keep their results. **File → Re-measure All**
  ignores the saved results.
- On network drives the app reads 3 files at a time (more just slows a NAS down);
  on local disks it uses all your processor cores.
- Big frames are shown at half resolution when fitted to the window; full resolution
  loads automatically when you zoom in past 50%.
- The progress bar shows an estimate of the time left.

## Keyboard shortcuts

| Key | Action |
|---|---|
| ↑ ↓ ← → | Previous / next frame |
| Shift + arrows, Shift/Ctrl + click | Select several frames |
| X or Delete | Reject / restore |
| F | Flag / unflag |
| Shift+X | Reject all flagged frames |
| D | Clear all flags |
| Ctrl+F | Auto-Flag… |
| Ctrl+Z | Undo reject / restore |
| Ctrl+0 / 1 / double-click | Fit / 100% / toggle |
| Mouse wheel, drag | Zoom, pan |
| Ctrl+O, F5 | Open folder, reload |
| Ctrl+E | Save a text list of kept frames |

Note: **F** now flags (it was "fit" in Stage 1) — fit is **Ctrl+0**.

## How the metrics are measured

- **FWHM**: up to 150 bright, unsaturated stars per measured area are fitted with an
  elliptical Moffat profile (β = 4, PixInsight SubframeSelector's default); the median
  is reported. Arcseconds use FOCALLEN and XPIXSZ from the header.
- **Eccentricity**: from the same fits (0 = round). Above ~0.5 is visible elongation.
- **SNR**: how far the brighter stars (90th percentile) rise above the sky noise.
- **Stars**: stars detected more than 5σ above the sky.
- **Background**: median sky level in ADU.
- Colour (OSC) frames are measured on the raw mosaic after balancing the four colour
  channels, so no debayering is needed and FWHM is in true sensor pixels.

## Rejected frames

Rejected frames are **moved** into a `REJECTED` subfolder next to them. Nothing is ever
deleted and images are never modified. Point your stacking program at the same folder
and it only sees the frames you kept.

## Files

- `fits_blaster.py` — the app window
- `chart.py` — the session chart
- `metrics.py` — star detection and fitting
- `fits_core.py` — FITS reading, fast sampled reads, stretching, scanning, cache,
  badges, and the reject/restore file moves
- `run.bat`, `build_exe.bat`, `requirements.txt` — setup and launch

## Coming next

**Stage 3:** the stacking app — calibration masters (bias, darks, flats; reads your
PixInsight .xisf masters too), per-camera and per-filter masters, star alignment,
outlier rejection, and `M33_Ha_Master_2026-10-09_to_2026-10-11.fits` naming.
