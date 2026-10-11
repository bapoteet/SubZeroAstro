# SubZeroAstro

***From subs to masters - zero hassle.***

A Windows app that takes a night's — or many nights' — subframes all the way to
PixInsight-ready masters:

1. **Cull** — star quality on every frame (FWHM, eccentricity, SNR, stars, background),
   a session chart, and one-click Aggressive / Semi-aggressive / Less aggressive culling.
2. **Stack** — calibration with your bias / darks / flats (and a calibration library that
   remembers your masters), star alignment and outlier rejection.
3. **Prepare for PixInsight** — crop, plate solve (ASTAP), gradient removal (GraXpert),
   BlurXTerminator / NoiseXTerminator / StarXTerminator, a quick-look image, and a
   step-by-step plan for finishing in PixInsight.

SubZeroAstro doesn't replace PixInsight — it does the routine work so your PixInsight
time goes into the creative part. The culling view was inspired by Tom van Peer's
FITS Blaster for macOS (https://astrophoto-app.com/); SubZeroAstro is written
independently for Windows.

## Download (easiest — no Python needed)

Go to the repository's **Releases** page (right-hand side on GitHub), download
**SubZeroAstro-Windows.zip** from the latest release, unzip it anywhere and run
**`SubZeroAstro.exe`** inside the folder. If Windows SmartScreen warns about an
unrecognized app, click **More info → Run anyway**.

Every change pushed to GitHub is built and tested automatically on Windows
(see the **Actions** tab), including a self-test that opens the real window.
You can run the same self-test yourself: `python subzeroastro.py --selftest`
(or `"SubZeroAstro.exe" --selftest`) — it writes `selftest_output\selftest_result.txt`
and screenshots.

## Updating

Download the newest **SubZeroAstro-Windows.zip** from Releases, delete the old app
folder (or unzip to a new one) and run **SubZeroAstro.exe**. Your settings, tool paths,
rig names and saved measurements are stored separately and carry over — including
from versions released under the earlier name "FITS Blaster for Windows".

## Install and run (first time)

1. Install **Python 3.11 or newer** from https://www.python.org/downloads/ and tick
   **"Add python.exe to PATH"** during setup.
2. Unzip this folder somewhere permanent, e.g. `C:\Astro\SubZeroAstro`.
3. Double-click **`run.bat`**. The first run sets things up (2–5 minutes, needs internet).

Optional: **`build_exe.bat`** makes a stand-alone `dist\SubZeroAstro\SubZeroAstro.exe`.

## Prepare for PixInsight (new in 0.5.0)

After stacking, click **Prepare for PixInsight…** in the Stacking window (or
**Stack › Prepare Masters for PixInsight…**). The masters you just made are loaded,
grouped by target. SubZeroAstro then takes them as far as it sensibly can on linear
data and hands over to PixInsight with a step-by-step plan.

**Your tools** (top of the window — auto-detected, or Browse… to point at them):
- **ASTAP** — plate solving (astap.exe or astap_cli.exe, with a star database installed)
- **GraXpert** — AI gradient removal (GraXpert-win64.exe)
- **RC-Astro CLI** — the stand-alone BlurXTerminator / NoiseXTerminator / StarXTerminator
  (normally `C:\Program Files\RC-Astro\CLI\rc-astro.exe`). Your existing licenses work;
  the status line shows each product's license.

**Steps** (each optional, run in this order on every master of the target):
1. **Crop** — the yellow box on the preview. *Auto-crop* trims the ragged stacking edges
   (the area covered by ~all frames); drag the box or its edges to adjust. The same crop is
   applied to every filter so they stay aligned.
2. **Plate solve** with ASTAP (uses the position and pixel scale recorded in the masters).
3. **Remove gradients** — GraXpert (AI), or a built-in polynomial model if GraXpert isn't set.
4. **BlurXTerminator** (on by default) → **NoiseXTerminator** (light, optional) →
   **StarXTerminator** (optional; saves a starless and a stars-only image).
   Their controls are read from your installed RC-Astro CLI, so they always match your
   version. **Device**: a dedicated NVIDIA/AMD GPU is chosen by default — RC-Astro's own
   "auto" picks the first GPU, which on many laptops is the slower integrated one.
   (Running `rc-astro --benchmark-all` once also fixes RC-Astro's own choice.)
5. **Quick-look image** — auto-stretched colour preview in RGB, LRGB, SHO or HOO.

Under **Save final images** pick the folder and tick the formats you want: linear masters as
**XISF** and/or **FITS** (32-bit, 0–1, plate solved — these are what you process in PixInsight),
and the stretched preview as **TIFF (16-bit)**, **JPG** and/or **PNG**. Leave the folder blank to
use a **Prepared** folder next to the masters. Originals are never changed. At the end SubZeroAstro says your images are **ready for
final processing** and shows (and saves as `<target>_PixInsight_next_steps.html`) the
recommended PixInsight processes and scripts in order — e.g. SPCC, stretching with GHS, star
recombination with PixelMath — with the steps already done here ticked off.

## Stacking (new in 0.4.0)

When you've culled your frames, click **Stack Kept Frames…** (Ctrl+K) — or the
**Open Stacking Window** button in the blue "ready to stack?" bar that appears after
**Reject Flagged**. The Stacking window opens with your kept frames already loaded,
grouped by **target + filter + camera + telescope**.

**Save locations** (top of the window, remembered, change them any time):
- **Calibration masters folder** — your "calibration frames" folder with a subfolder
  per telescope rig. New masters are saved as
  `<folder>\<rig>\Masters\<camera>\Bias | Darks | Flats\<filter>\…` and are found
  there automatically next time.
- **Master lights folder** — masters go to `<folder>\<target>\`, e.g.
  `M33\M33_Ha_Master_2026-10-09_to_2026-10-11.fits` (one date for a single night;
  the camera is added to the name when one target was shot with two cameras).

**Telescope (rig) names** — pick the rig folder for each camera + focal length once
(e.g. ASI6200MC at 506 mm → *SQA106*). This matters for ASIAIR, which writes the
mount name, not the telescope.

**Calibration** — add a folder (or files) of bias, dark and flat subframes or existing
masters (FITS or PixInsight XISF). Frame types are read from the headers. For each
group the app picks:
- **Bias** matching camera, gain and offset
- **Darks** matching exposure, gain, offset and temperature (±3 °C); darks of another
  exposure are scaled using the bias
- **Flats** matching camera, filter and rig, nearest in date; flat subframes are
  calibrated with dark-flats or a bias at the *flat's* gain
- Lights shot on part of the sensor (ASIAIR ROI) use the matching part of
  full-size calibration frames automatically

Before stacking, anything missing is listed ("No darks for 180 s, gain 100, −2 °C…")
with **Stack Anyway** or **Add Calibration Frames First…**. Notes also appear when
flats are old or were taken at a different camera angle. *Search Calibration Folder
for Subframes* optionally looks through your rig folders for raw calibration frames.

**How lights are stacked** — calibrate ((light − dark) ÷ flat, hot pixels from the
master dark repaired) → debayer colour frames (full or half resolution) → align every
frame to the best-scoring one by matching star patterns (handles meridian flips) →
normalise sky level and brightness (thin cloud) → weight by noise → sigma-clipped
mean, which removes satellite trails, planes and remaining hot pixels.

**Output** — 32-bit floating-point **FITS** and/or **XISF**, in PixInsight's 0–1
convention (1.0 = 65535 ADU; the `PIXRANGE` keyword records this), so calibration
masters load straight into PixInsight / WBPP. Master lights carry the reference frame's
RA/Dec, pixel size and WCS (when the subframes had one) to help plate solving.
With **Line up all filters of a target** (on by default) every master of the same
target, camera and telescope shares one pixel grid, ready to combine (LRGB, SHO…)
without re-aligning. Each master records in its
header which frames, calibration masters and settings were used, and a
`…_stacklog.txt` next to it lists every frame — so the record survives if you delete
the subframes afterwards. Colour masters are not colour-calibrated (they look green,
as usual for raw OSC data) — do colour calibration in PixInsight, Siril etc.

Aligned frames are cached temporarily on your local disk (≈120 MB per 61 MP mono frame).
If there isn't room, a slower two-pass method is used instead.

## What's new in 0.3.0

**Load Data window** (Ctrl+O, or the **Load Data…** button) — build a list of folders
and/or individual files from as many places as you like (several nights, different
NAS shares…). Nothing is read until you press **Start Processing**. You can also drag
several folders onto the app. *Open One Folder…* (Ctrl+Shift+O) is still there for a
quick single folder.

**Auto-cull buttons** — **Aggressive**, **Semi-aggressive** and **Less aggressive**
flag frames for rejection using FWHM, eccentricity and star count (plus SNR, which
catches thin cloud and haze), each compared with the median of the frame's own
filter. Frames that are somewhat worse in several ways at once are flagged too.
Hover over a button to see how many frames it would flag. The picks are only
*flagged*: review them in the Flagged view (the inspector says why each was picked),
then press **Reject Flagged** (Shift+X).

| Preset | FWHM above | Stars below | SNR below | Eccentricity above |
|---|---|---|---|---|
| Aggressive | 1.10 × median | 80% of median | 75% of median | 0.55 (or median + 0.08) |
| Semi-aggressive | 1.20 × median | 65% of median | 60% of median | 0.65 (or median + 0.12) |
| Less aggressive | 1.35 × median | 50% of median | 45% of median | 0.75 (or median + 0.20) |

**Start Over – Load New Data** (Ctrl+N) — clears the session and opens the Load Data
window. Rejected frames stay in their REJECTED folders and measurements stay saved.

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
| Ctrl+O | Load Data (one or more folders) |
| Ctrl+Shift+O | Open one folder |
| Ctrl+N | Start Over – load new data |
| Ctrl+K | Stack kept frames… |
| F5 | Reload |
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

- `subzeroastro.py` — the main (culling) window
- `chart.py` — the session chart
- `metrics.py` — star detection and fitting
- `fits_core.py` — FITS reading, fast sampled reads, stretching, scanning, cache,
  badges, and the reject/restore file moves
- `stacking.py` — calibration masters, star alignment and stacking
- `stack_window.py` — the Stacking window
- `xisf.py` — PixInsight XISF reading and writing
- `prepare.py` — Stage 4 pipeline: crop, solve, gradients, XTerminators, quick-look, PixInsight plan
- `prepare_window.py` — the Prepare for PixInsight window
- `tools.py` — running ASTAP, GraXpert and the RC-Astro CLI
- `tests/fake_tools/` — stand-ins for those tools used by the automatic tests
- `selftest.py` — the automatic real-window self-test
- `run.bat`, `build_exe.bat`, `requirements.txt` — setup and launch

