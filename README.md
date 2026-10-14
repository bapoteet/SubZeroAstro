# SubZeroAstro

***From subs to masters - zero hassle.***

> **Abstract.** SubZeroAstro is a free Windows app for deep-sky astrophotographers that
> automates the routine work between capture and creative processing. It grades every
> subframe (FWHM, eccentricity, SNR, star count, background) and culls the bad ones with one
> click; calibrates and stacks the keepers with your bias, darks and flats into per-filter master
> lights, keeping a reusable calibration library; and prepares those masters for PixInsight by
> cropping, plate solving with ASTAP, removing gradients with GraXpert, and running RC-Astro's
> BlurXTerminator, NoiseXTerminator and StarXTerminator. It works with mono and one-shot-colour
> cameras and with NINA and ASIAIR files, and it ends with a quick-look colour preview and a
> step-by-step PixInsight plan for finishing the image.

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
unrecognized app, click **More info → Run anyway**. Releases are being set up for
free code signing by SignPath — see the [Code signing policy](#code-signing-policy).

**Mac (Apple silicon, M1 or newer):** download **SubZeroAstro-macOS-AppleSilicon.zip**
from the same release, double-click it, and drag **SubZeroAstro.app** to your Applications folder.
The app is not notarized by Apple yet, so the first time **right-click (or Control-click) the app ›
Open › Open**. If macOS still says it "is damaged" or "can't be opened", run this once in Terminal:
`xattr -dr com.apple.quarantine /Applications/SubZeroAstro.app`. On a Mac, point the Tools fields to
`ASTAP.app`, `GraXpert.app` and the RC-Astro CLI wherever you installed them.

Every change pushed to GitHub is built and tested automatically on Windows and macOS
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

## Finishing Touches (formerly "Finish the Image", new in 0.7)

After **Prepare**, click **Finish the Image…** (or *Stack › Finish the Image…*) to carry on in
SubZeroAstro instead of PixInsight - as far as you like:

- A **Recommended next step** card always says what to do next and why, based on your data
  (colour camera, mono RGB / LRGB, narrowband, smart telescope) and what is already done.
  **Do It** opens the step, **Skip** leaves it out. The workflow list on the left shows ✓ done,
  ➜ next, ○ later, – not needed.
- **Pick the one you like.** Each step shows 4-5 small versions of your image - gentle to
  strong, or different colour mixes - instead of sliders. ★ marks the suggested one. Tools whose
  effect depends on pixel size (Shrink Stars) show a full-size piece so you see the real effect.
- **Clicking a version only previews it** (big, in the main view) - nothing changes until you press
  **Apply**. **Cancel** closes the step and leaves the picture exactly as it was.
- **Fine-tune (optional):** after picking, one to four plain-language sliders (Brightness, Protect bright
  areas, Warmth, Tint, Contrast, Shadows, Glow strength, ...) start exactly at the version you picked;
  the big preview follows as you move them, **Reset** returns to the picked version. Curves also has a
  curve you can drag.
- Every step is saved: **Undo** (Ctrl+Z), **Back to Start** (the original masters), **Show / Hold for
  Before / Go Back to Here**, resume after a crash, and **Save Project As… / Open Project…**.

| Step | What it does (PixInsight equivalent) |
|---|---|
| Choose a Colour Palette | For colour cameras behind a **dual-band filter** (L-Ultimate, L-eXtreme, L-eNhance, ALP-T, NBZ, Seestar LP, **Askar D1**: Ha + OIII; **L-Synergy**, **Askar D2**: SII + OIII) - the two lines are separated and offered as As captured, HOO / SOO, HOO vivid, Gold & blue and Foraxx. An **L-Ultimate + L-Synergy** or **Askar D1 + D2** pair of the same target becomes a full SHO set (Hubble, Foraxx, HSO, HOO + SII) |
| Combine Colour Channels | Builds colour from your filters: RGB, LRGB, Hubble SHO, HSO, HOO, HOO vivid, RGB + Ha glow (ChannelCombination / LRGBCombination / PixelMath) |
| Balance Colours | Neutral grey sky and white average star, warmer / cooler options (BackgroundNeutralization + colour calibration; SPCC stays in PixInsight) |
| Remove Green Cast | Removes the green tint, or turns the Hubble palette gold and blue (SCNR) |
| Take Stars Out | Separates stars with StarXTerminator (or uses the stars Prepare already separated) |
| Brighten the Image | The stretch: Automatic (STF-like), Colour-preserving (arcsinh) or Hyperbolic (GHS); 5 strengths that mean the same in every method; colours linked or not |
| Brighten Stars Separately | Gentle star stretch so stars stay small and colourful |
| Shrink Stars | Star reduction, 4 strengths |
| Put Stars Back | Screen blend at the strength you pick |
| Tame Bright Core | Brings back detail in a burnt-out galaxy core or bright nebula centre: an automatic soft-edged mask (tick *Show what will change* to see it) and the core re-made from the unstretched data. Suggested automatically when a white core is detected; tells you if the core was already saturated in the masters (HDRMultiscaleTransform through a mask) |
| Adjust Brightness & Contrast (Curves) | Presets (More contrast, Lift faint detail, Darker background, ...), sliders, drag-the-curve; colours kept (CurvesTransformation) |
| Colour Intensity (Saturation) | Richer or calmer colour; the dark sky is protected from colour noise |
| Bring Out Hidden Structure ◇ | Local contrast for cores and dust lanes (LocalHistogramEqualization) |
| Add Hydrogen Glow | Ha into red for RGB + Ha sets, with the starlight removed from Ha first (continuum subtraction) |
| Spot Fix ◇ | Click dust donuts / satellite remnants / hot pixels to patch them (CloneStamp) |

◇ = optional: listed in the workflow but not pushed by "Recommended next".

**Save the Picture…** writes TIFF (16-bit), JPG, PNG, XISF or FITS.

### Stop Here – Continue in PixInsight
The **Stop Here – Continue in PixInsight** button (Stacking, Prepare and Finish windows) saves the
image exactly as it is at that moment as PixInsight-ready XISF (linear or stretched, starless + stars
if they are separated, or each filter if not combined yet) with a short note listing the PixInsight
processes that carry on from there - and opens the folder with the files selected.

### Share to AstroBin (new in 0.8.9)

**Share to AstroBin…** (top of the Finish window) saves the picture as a JPG (and optionally a
16-bit TIFF) plus `<target>_astrobin_acquisition.csv`, opens the folder and AstroBin's upload page.
Upload the picture, then on AstroBin's *Acquisition* step click **Import CSV** and pick the file:
every night's subs, exposure, gain, sensor temperature, binning, mean FWHM (arcsec) and your
calibration frame counts are filled in. Optional: your AstroBin filter ID numbers (remembered per
filter), Bortle class and SQM.

AstroBin's public API is read-only, so the upload itself happens in your browser. The night-by-night
details come from `ACQnn` / `ACQCAL` keywords SubZeroAstro now writes into every master it stacks;
masters stacked elsewhere get one row each from their standard headers.

## Smart telescopes (new in 0.6)

SubZeroAstro reads the files from **ZWO Seestar**, **DWARF** (mini / II / 3), **Vaonis Vespera**
(incl. Pro), **Stellina** and **eVscope**, alongside traditional NINA / ASIAIR rigs:

- **Raw subs** (DWARF, Vespera, Seestar with "save each frame" on) are recognised as lights even
  without an IMAGETYP keyword. Culling, Auto-cull and stacking work as usual; star alignment
  handles the field rotation of alt-az mounts and the auto-crop trims the rotated edges.
  Calibration frames are optional — no "missing darks/flats" warnings for these scopes.
  Short smart-scope subs (10–20 s at high gain, uncalibrated hot pixels) use a **robust star
  finder** (hot-pixel filter + matched filter), so even faint frames align — tested on raw
  Seestar S50 Pro subs.
- **Seestar stacks** (`Stacked_62_IC 1396_….fit`, already debayered colour) can be combined
  across nights in the Stacking window: each stack is aligned, brought to the same brightness
  (the Seestar app scales every stack differently) and weighted by its noise. The master's
  EXPTOTAL counts the real integration of every sub inside the stacks.
- A single Seestar stack can go straight into **Prepare for PixInsight** — it is already plate
  solved, so Prepare crops, removes gradients and runs the XTerminators.

## Saving progress and going back (new in 0.6)

SubZeroAstro now saves your progress as it works, so a crash, a frozen laptop or a
power cut doesn't cost you the evening — and you can step back when you don't like a result.

- **Culling** — the folders you loaded and the frames you flagged are saved as you go. If
  SubZeroAstro didn't close normally it offers to reopen the session at the next start
  (measurements come from the cache, so it's quick). *File › Reopen Last Session* does it any time.
  **Ctrl+Z** now also undoes a whole Auto-cull preset, Auto-Flag, Clear All Flags or
  Reject Flagged (files *and* flags come back) in one step.
- **Stacking** — each master is noted as soon as it's saved. If a run is interrupted, the next
  run offers **Continue With the Rest**: finished masters are kept and calibration masters it
  already built are reused instead of rebuilt. Stacking again never overwrites a master — new
  versions are saved as `_2`, `_3`, …
- **Prepare for PixInsight** — the image is saved after every step (crop, plate solve, gradient,
  BXT, NXT, SXT). The **Step history** under the preview lets you look at the result of any step,
  hold **Before** to compare with the step before it, and **Go Back & Redo This Step**: change its
  settings and press Prepare — only that step and the ones after it run again. Changing a setting
  is noticed automatically too (e.g. a stronger BXT sharpen re-runs BXT, NXT and SXT, but not
  ASTAP or GraXpert). After a crash, Prepare simply continues from the last finished step.

*Settings › Saved Progress* shows how much space this uses, lets you move it to another drive,
and sets how long unused progress is kept (30 days by default; Prepare copies are roughly
100–400 MB per filter per step). Saved progress never touches your subframes or masters.

## Astro Image Processing (formerly "Prepare for PixInsight", new in 0.5.0)

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
- `checkpoints.py` — saved progress (crash recovery, step history)
- `finish.py` / `finish_window.py` — Stage 5: Finish the Image
- `prepare.py` — Stage 4 pipeline: crop, solve, gradients, XTerminators, quick-look, PixInsight plan
- `prepare_window.py` — the Prepare for PixInsight window
- `tools.py` — running ASTAP, GraXpert and the RC-Astro CLI
- `tests/fake_tools/` — stand-ins for those tools used by the automatic tests
- `selftest.py` — the automatic real-window self-test
- `run.bat`, `build_exe.bat`, `requirements.txt` — setup and launch

## Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by
[SignPath Foundation](https://signpath.org).

- Only `SubZeroAstro.exe` is signed: it is built automatically from the source code in
  this repository by GitHub Actions (see `.github/workflows/build-windows.yml`). The bundled
  Python, Qt and numpy files belong to their own projects and are not signed by us.
- Every signing request is approved by hand before a release is published.

Team roles:

- Committers and reviewers: [Brian A. Poteet (@bapoteet)](https://github.com/bapoteet)
- Approvers: [Brian A. Poteet (@bapoteet)](https://github.com/bapoteet)

## Privacy policy

This program will not transfer any information to other networked systems unless
specifically requested by the user or the person installing or operating it.

- SubZeroAstro reads and writes image files only where you point it, and keeps its
  settings and saved progress on your own computer.
- **Share to AstroBin…** only opens AstroBin's upload page in your web browser when you
  click it; nothing is uploaded by SubZeroAstro itself.
- Outside tools you choose to use (ASTAP, GraXpert, the RC-Astro CLI) are separate programs
  with their own privacy terms — for example, the RC-Astro CLI may check for its own updates.

## Credits

Created by Brian A. Poteet ([@bapoteet](https://github.com/bapoteet)), built with Claude (Anthropic).

## License

SubZeroAstro is free software under the [MIT License](LICENSE): you may use, copy, modify and share
it, as long as the copyright notice and license text stay with it. It comes with no warranty.
