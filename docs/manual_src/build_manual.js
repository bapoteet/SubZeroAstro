// SubZeroAstro User Manual - builds SubZeroAstro_User_Manual.docx
// node build_manual.js [pages.json]   (pages.json: chapter -> page number, from a first PDF render)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Header, Footer, PageNumber, Bookmark, Tab, LeaderType,
  InternalHyperlink, ExternalHyperlink, PositionalTab, PositionalTabAlignment, PositionalTabLeader,
  PositionalTabRelativeTo, TabStopType,
} = require("docx");

const IMG = path.join(__dirname, "img");
const PAGES = fs.existsSync(process.argv[2] || "") ? JSON.parse(fs.readFileSync(process.argv[2])) : {};
// the version is read from the app itself, so the manual always matches the release it ships with
const VERSION = (fs.readFileSync(path.join(__dirname, "..", "..", "subzeroastro.py"), "utf8")
  .match(/^APP_VERSION = "([^"]+)"/m) || [null, "?"])[1];
const BLUE = "1F4E79", ACCENT = "2F6FD6", GREY = "595959";

// ------------------------------------------------------------------ content model
const body = [];          // docx children (after the cover and contents)
const toc = [];           // {level, num, title, anchor}
const index = {};         // term -> Set of section numbers
let listInstance = 0;

const anchorOf = (num) => "s_" + String(num).replace(/\./g, "_");

function runs(text, base = {}) {
  // **bold**, *italic*, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

let first = false;
function ch(num, title) {
  const anchor = anchorOf(num);
  toc.push({ level: 1, num, title, anchor });
  body.push(new Paragraph({
    heading: HeadingLevel.HEADING_1, pageBreakBefore: !first,
    children: [new Bookmark({ id: anchor, children: [new TextRun(`${num}   ${title}`)] })],
  }));
  first = false;
}
function sec(num, title, terms = []) {
  const anchor = anchorOf(num);
  toc.push({ level: 2, num, title, anchor });
  body.push(new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new Bookmark({ id: anchor, children: [new TextRun(`${num}   ${title}`)] })],
  }));
  for (const t of terms) (index[t] = index[t] || new Set()).add(num);
}
function sub(title) { body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(title)] })); }
function p(text) { body.push(new Paragraph({ children: runs(text), spacing: { after: 120 } })); }
function bullets(items) {
  for (const it of items) body.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(it),
    spacing: { after: 60 } }));
}
function steps(items) {
  listInstance += 1;
  for (const it of items) body.push(new Paragraph({ numbering: { reference: "steps", level: 0, instance: listInstance },
    children: runs(it), spacing: { after: 60 } }));
}
function tip(text, label = "Tip") {
  body.push(new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "EAF1FB", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
    indent: { left: 200, right: 200 }, spacing: { before: 120, after: 160 },
    children: [new TextRun({ text: label + ":  ", bold: true, color: ACCENT }), ...runs(text)],
  }));
}
function warn(text) {
  body.push(new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "FDF3E3", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "D98B00", space: 8 } },
    indent: { left: 200, right: 200 }, spacing: { before: 120, after: 160 },
    children: [new TextRun({ text: "Important:  ", bold: true, color: "B36B00" }), ...runs(text)],
  }));
}
function img(file, caption, maxW = 600) {
  const f = path.join(IMG, file);
  const buf = fs.readFileSync(f);
  // PNG size from the IHDR chunk
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const width = Math.min(maxW, w), height = Math.round(h * width / w);
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
      altText: { title: caption, description: caption, name: file } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: GREY })] }));
}
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(headers, rows, widths, out = body, size = undefined) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, head, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    shading: head ? { type: ShadingType.CLEAR, fill: "DCE6F2", color: "auto" } : undefined,
    margins: size ? { top: 30, bottom: 30, left: 90, right: 90 } : { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(text, { ...(head ? { bold: true } : {}), ...(size ? { size } : {}) }) })],
  });
  out.push(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, true, widths[i])) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, false, widths[i])) }))],
  }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

// ================================================================== THE MANUAL
ch(1, "Welcome to SubZeroAstro");
sec("1.1", "What SubZeroAstro does", ["Four pages", "Workflow overview"]);
p("SubZeroAstro takes you **from a folder of raw sub-frames to a finished astrophoto** - with as little hassle as possible. It walks you through four pages, one after the other, like turning the pages of a book (each shows **Page 1 of 4**, **Page 2 of 4** … next to its title at the top, the same numbers this manual uses):");
table(["Page", "What happens there"], [
  ["1. Astro Image Quality and Cull", "Every sub-frame is measured (star size, roundness, signal, cloud). You keep the good ones and set aside the bad ones."],
  ["2. Calibrate and Stack", "Your kept frames are calibrated with darks, flats and bias and combined into one clean **master** per filter."],
  ["3. Astro Image Processing", "The masters are cropped, plate solved, cleared of gradients and (optionally) sharpened, de-noised and separated from their stars with the RC-Astro XTerminators."],
  ["4. Finishing Touches", "A guided, pick-the-one-you-like workflow turns the masters into a finished colour picture you can save, sign and share."],
], [2700, 6660]);
p("At every page you can also **stop and continue in PixInsight** - SubZeroAstro saves exactly what you have so far, plus a note of what to do next there.");
img("0_splash_4.00s.png", "The SubZeroAstro start-up screen", 520);
sec("1.2", "Who this manual is for");
p("This manual is written for **newcomers** to astrophotography processing. No knowledge of PixInsight or other processing software is needed. Experienced imagers will find the PixInsight equivalent of each step mentioned along the way.");
sec("1.3", "Words you will meet", ["Glossary", "Sub-frame", "Master", "Linear", "Stretched", "OSC", "Mono", "Dual-band filter", "Narrowband", "Plate solving", "FWHM", "Eccentricity", "SNR"]);
table(["Word", "What it means"], [
  ["Sub-frame (sub, light)", "One single exposure of your target, e.g. 180 seconds."],
  ["Calibration frames", "**Darks** (lens cap on, same exposure), **flats** (evenly lit, shows dust and vignetting), **bias** (shortest exposure) and **dark-flats**. They remove the camera's and telescope's own fingerprints."],
  ["Master", "Many frames combined into one: a master light (your target) or a master dark / flat / bias."],
  ["Linear / stretched", "Straight from the camera a picture is **linear** and looks almost black. **Stretching** (Brighten the Image) makes the faint detail visible."],
  ["OSC (colour camera)", "One-shot-colour camera: records red, green and blue at once (e.g. ZWO ASI2600MC)."],
  ["Mono camera", "Records one filter at a time (L, R, G, B, Ha, OIII, SII); the colours are combined later."],
  ["Narrowband / dual-band", "Filters that pass only the glow of hydrogen (Ha), oxygen (OIII) or sulphur (SII). A **dual-band** filter (e.g. Optolong L-Ultimate, Askar D1) passes two of them on a colour camera."],
  ["Plate solving", "Working out exactly where in the sky the picture points, by matching its stars to a star catalogue."],
  ["FWHM", "Star size (full width at half maximum). Smaller = sharper."],
  ["Eccentricity", "How round stars are: 0 = round, 1 = a line. Above about 0.5 stars look elongated."],
  ["SNR", "Signal-to-noise ratio: how clearly the stars rise above the sky. Drops with cloud, haze or moonlight."],
  ["Starless / stars-only", "The picture split in two by StarXTerminator, so the nebula and the stars can be processed separately."],
], [2600, 6760]);

sec("1.4", "Your data, improved - never painted", ["Honest processing", "AI", "Painting", "Real data"]);
p("SubZeroAstro brings out what your camera **recorded** - it does not invent detail or paint the target. Every step works from your own data:");
bullets([
  "**Brightness and colour:** brightening, Curves, Saturation and the colour tools re-map the brightness and colour your camera measured. The colour-preserving stretch keeps each pixel's measured colour mix; palettes such as HOO and Foraxx assign your filters' real signal to colours.",
  "**Bright cores and rings:** Tame Bright Core and Tame the Ring re-make bright areas from your own unstretched (linear) data - detail they show was in your frames, hidden by the brightening.",
  "**AI, used the accepted way:** BlurXTerminator sharpens by deconvolution using the blur measured from your own stars; NoiseXTerminator removes noise; GraXpert models the light-pollution gradient so it can be subtracted. None of them adds structure that is not in the data.",
  "**Smoothing** (colour transitions, noise reduction) only averages what is there.",
]);
p("**Where pixels are replaced - always to remove a flaw, never to add to the target:**");
bullets([
  "**Spot Fix** fills a blemish you circle (a hot pixel, a satellite remnant, a dust speck) from the sky around it - like PixInsight's CloneStamp, and only where you ask.",
  "**Star removal** (StarXTerminator) has to fill in the background behind each star it lifts out; *Put Stars Back* covers those places again with your real stars.",
  "**Mote Buster** and flats correct dust shadows and vignetting - artefacts of the optics, not of the sky.",
]);

ch(2, "What you need");
sec("2.1", "Computer requirements", ["Hardware requirements", "System requirements", "Memory (RAM)", "Disk space", "Graphics card (GPU)"]);
p("SubZeroAstro runs on **Windows** and on **Macs with Apple silicon**. Astro images are big (a 26-megapixel camera makes ~50 MB files; a night can easily be 20-50 GB), so more memory and a fast disk make everything quicker.");
table(["", "Minimum", "Recommended"], [
  ["Operating system", "Windows 10 (64-bit) or a Mac with Apple silicon (M1 or newer) and a recent macOS", "Windows 11, or macOS 14 (Sonoma) or newer"],
  ["Processor", "4 cores", "8 cores or more"],
  ["Memory (RAM)", "16 GB", "32 GB or more (64 GB for 60-megapixel cameras or very large stacks)"],
  ["Free disk space", "20 GB plus room for your images", "An SSD with 100 GB or more free - stacking keeps temporary aligned frames on the local disk"],
  ["Graphics card", "Not required", "An NVIDIA RTX card (Windows) or Apple silicon GPU makes the RC-Astro XTerminators many times faster"],
  ["Screen", "1600 × 900", "1920 × 1080 or larger"],
], [2100, 3630, 3630]);
tip("Images on a network drive (NAS) work fine. SubZeroAstro reads them carefully (3 files at a time) and remembers the measurements, so re-opening a session is quick.");
sec("2.2", "Companion programs (optional but recommended)", ["ASTAP", "GraXpert", "RC-Astro CLI", "BlurXTerminator", "NoiseXTerminator", "StarXTerminator", "PixInsight"]);
p("SubZeroAstro works on its own, but it can call these free or commercial programs for the best results. Install them **before** the first setup (Chapter 4); SubZeroAstro finds them automatically in their usual places. Each time it starts, page 1 checks for them quietly in the background and shows an amber note if one is missing or not licensed (4.1).");
table(["Program", "Used for", "Where to get it", "Without it"], [
  ["**ASTAP** + a star database (D50 suits most setups)", "Plate solving (page 3)", "hnsky.org/astap.htm (free)", "Plate solving is skipped (PixInsight can do it later)"],
  ["**GraXpert**", "AI gradient removal (page 3)", "graxpert.com (free)", "A built-in gradient remover is used"],
  ["**RC-Astro CLI** with your BlurXTerminator, NoiseXTerminator and StarXTerminator licences", "Sharpening, noise reduction and star removal (pages 3 and 4)", "Your RC-Astro account (rc-astro.com)", "These steps are skipped; Take Stars Out is not available"],
  ["**PixInsight** (optional)", "Further processing after any page", "pixinsight.com", "Not needed - SubZeroAstro can finish the picture"],
], [2500, 2200, 2360, 2300]);
sec("2.3", "Your images", ["FITS", "XISF", "Folder organisation", "File formats"]);
bullets([
  "SubZeroAstro reads **FITS** (`.fit`, `.fits`, `.fts`) and PixInsight **XISF** (`.xisf`) files on every page - light frames, calibration frames and masters - from capture programs such as N.I.N.A., ASIAIR, SharpCap and smart telescopes (ZWO Seestar, DWARF, Vaonis Vespera, Stellina, eVscope). XISF files can be uncompressed or compressed with any of PixInsight's options (zlib, LZ4, LZ4-HC, Zstandard, with or without byte shuffling).",
  "Keep each night's lights in their own folder, and your calibration frames in a **calibration library** folder with one subfolder per telescope (see 4.3).",
  "Your original files are **never changed or deleted**. Rejected frames are only moved into a `REJECTED` subfolder.",
]);

ch(3, "Installing SubZeroAstro");
sec("3.1", "Windows", ["Install (Windows)", "SmartScreen"]);
steps([
  "Open the SubZeroAstro releases page: **github.com/bapoteet/SubZeroAstro/releases**.",
  "Under the newest release, click **SubZeroAstro-Windows.zip** to download it.",
  "Right-click the downloaded zip › **Extract All…** and choose a permanent place, e.g. `C:\\Astro\\SubZeroAstro`.",
  "Open the extracted folder and double-click **SubZeroAstro.exe**. (Right-click › *Pin to Start* or *Send to › Desktop* gives you a shortcut.)",
  "If Windows shows **\"Windows protected your PC\"**, click **More info › Run anyway**. This appears because the app is new and not yet code-signed; it is safe to continue.",
]);
sec("3.2", "Mac (Apple silicon)", ["Install (Mac)", "Gatekeeper", "Quarantine"]);
steps([
  "Open **github.com/bapoteet/SubZeroAstro/releases** and download **SubZeroAstro-macOS-AppleSilicon.zip** from the newest release.",
  "Double-click the zip in your Downloads folder; **SubZeroAstro.app** appears.",
  "Drag **SubZeroAstro.app** into your **Applications** folder.",
  "The first time only: **right-click** (or Control-click) the app › **Open**, then **Open** again in the message. After that it opens normally.",
  "If macOS says the app \"is damaged\" or \"can't be opened\", open **Terminal** (Applications › Utilities) and type: `xattr -dr com.apple.quarantine /Applications/SubZeroAstro.app` then press Return and open the app again.",
]);
tip("Both messages appear because SubZeroAstro is not yet signed by Microsoft or notarized by Apple. Signing is being set up for future versions.", "Why the warnings?");
tip("SubZeroAstro opens filling the screen, with the menu bar and Dock still showing. Avoid the green button's full-screen mode on a Mac: it hides the menu bar (where **Stack › Build Calibration Masters…** and **Help** live). To fill the screen yourself, **Option-click** the green button, or double-click the window's title bar. If you make the window smaller, it opens at that size next time.", "Mac: window size");
sec("3.3", "Updating to a new version", ["Updating"]);
p("Download the newest zip from the releases page and replace the old app folder (Windows) or the app in Applications (Mac). **Your settings, tool locations, rig names, saved measurements and saved progress are kept** - they are stored separately from the app.");
sec("3.4", "Removing SubZeroAstro", ["Uninstall"]);
p("Delete the app folder (Windows) or drag SubZeroAstro from Applications to the Bin (Mac). Saved progress can be cleared first under **Settings › Saved Progress**. Your images are never touched.");

ch(4, "First-time setup");
sec("4.1", "Tell SubZeroAstro where your tools are", ["Tool locations", "ASTAP", "GraXpert", "RC-Astro CLI", "Device (GPU)"]);
p("When SubZeroAstro starts, it checks for ASTAP, GraXpert and the RC-Astro CLI in the background. If one is missing - or RC-Astro is installed but a tool is not licensed, or ASTAP has no star database next to it (ASTAP can't plate solve without one; D50 is a separate download on the same page) - a slim **amber banner** appears at the top of page 1, saying what you would miss (for example *plate solving is skipped*). Nothing is wrong: you can still cull and stack. The banner has two buttons:");
bullets([
  "**Set Up…** - opens the **Your Tools** window (also in **Settings › Your Tools (ASTAP, GraXpert, RC-Astro)…**).",
  "**Don't Remind Me** - hides the banner until something else changes. Use it if you have chosen not to use one of the programs.",
]);
img("1a_tools_banner.png", "The amber banner on page 1 when helper programs are missing (example)");
p("The **Your Tools** window lists each program with its status, a **Download Page** button for anything missing, and **Browse…** to point at a program installed somewhere unusual. Click **Check Again** after installing something.");
img("1b_your_tools.png", "Settings › Your Tools: status, Browse… and Download Page for each helper program");
p("The same programs are also shown in the **Your tools** box at the top of page 3 (**Astro Image Processing**), with a green **✓ found** when detected. If one is missing, click **Browse…** next to it and pick the program:");
table(["Tool", "Typical location on Windows", "Typical location on a Mac"], [
  ["ASTAP", "`C:\\Program Files\\astap\\astap.exe`", "`/Applications/ASTAP.app` (or `astap_cli`)"],
  ["GraXpert", "`C:\\Users\\<you>\\AppData\\Local\\Programs\\GraXpert\\GraXpert.exe`", "`/Applications/GraXpert.app`"],
  ["RC-Astro CLI", "`C:\\Program Files\\RC-Astro\\CLI\\rc-astro.exe`", "where the RC-Astro installer put `rc-astro`"],
], [1700, 4200, 3460]);
p("**Activating RC-Astro licences.** You do not need a terminal: click **RC-Astro Licences…** (next to the RC-Astro CLI row on page 3, or in the Your Tools window). The window shows which tools are licensed on this computer. To activate:");
steps([
  "Click **Open My RC-Astro Account** and sign in on rc-astro.com.",
  "Click **Get Activation Code** on the account page, then click it again to copy the code.",
  "Paste the code into the box and click **Activate**. **One code activates every RC-Astro tool you own.**",
]);
p("The tools turn green and page 3 is updated straight away. If activation does not work, **Do It in a Terminal Window** runs RC-Astro's own `rc-astro license` command, where you paste the code and press Enter. **Check for Updates** and **Download & Install Update** keep the RC-Astro CLI itself up to date.");
img("11a_rc_licences.png", "The RC-Astro Licences window (example: StarXTerminator not yet activated)");
p("For the RC-Astro CLI the status line lists **BlurXTerminator ✓ · StarXTerminator ✓ · NoiseXTerminator ✓** for every product licensed on this computer. Below the XTerminator settings, **RC-Astro device** chooses the processor: pick your dedicated graphics card (e.g. *NVIDIA RTX…*) - it is much faster than integrated graphics.");
img("11_prepare_window.png", "Page 3 with the tools box at the top (example data)");
sec("4.2", "Choose where results are saved", ["Save locations", "Calibration masters folder", "Master lights folder"]);
p("On page 2 (**Calibrate and Stack**) set two folders under **Save locations** once; they are remembered:");
bullets([
  "**Calibration masters** - your calibration library (4.3). New master darks, flats and bias are saved there and found again automatically next time.",
  "**Master lights** - where your stacked masters go, one subfolder per target, e.g. `…\\Master Lights\\IC1805\\IC1805_L-Ultimate_Master_2026-10-09_to_2026-10-11.fits`.",
]);
sec("4.3", "The calibration library", ["Calibration library", "Rig names", "Flats", "Darks", "Bias", "Dark-flats"]);
p("SubZeroAstro keeps calibration masters in a tidy structure inside the calibration folder:");
p("`<calibration folder>\\<telescope rig>\\Masters\\<camera>\\Bias | Darks | Flats\\<filter>\\…`");
bullets([
  "**Rig names**: the first time, pick a rig name for each camera + focal length (e.g. *265mm scope* or *SQA55*). This matters for ASIAIR, which records the mount, not the telescope.",
    "**Even out the sky between frames** (on by default; PixInsight calls it LocalNormalization): light pollution, haze and moonlight change through the night, so each frame's sky is brighter in a different corner. Before combining, every frame's sky is matched to the best frame's, so satellites, planes and noise are rejected properly and the background comes out even.",
  "**Double the resolution (drizzle 2x)** (off by default; PixInsight: DrizzleIntegration): if you dither (the scope moves a few pixels between frames) and your stars are small and blocky, the frames' slightly different positions are combined on a grid twice as fine - a master twice the size with finer stars. It uses the regular stack to leave out satellites and planes. Slower, the masters are 4x larger, and without dithering it adds nothing. After stacking, SubZeroAstro tells you when your frames are dithered and the stars small enough (under about 2.2 px) for drizzle to help (and when it was on but the frames were hardly dithered).",
"**Darks** are matched by exposure, gain, offset and temperature (±3 °C). **Flats** are matched by camera, filter and rig, nearest in date. If you take flats every session (e.g. NINA's flat-panel sequence after imaging), a group of lights from several nights uses **each night's own flat** for that night's lights - dust that moved between nights leaves no shadows (the Flats cell and the stack log list them). **Bias** or **dark-flats** are needed to calibrate flats properly.",
  "You can add a folder of raw calibration sub-frames; SubZeroAstro builds the masters and saves them into the library.",
]);
sec("4.4", "Settings worth knowing", ["Settings", "Intro animation", "Saved progress", "Star measurement"]);
bullets([
  "**Settings › Star measurement** - *Fast* (default, about 3-4× quicker on a network drive), *Full frame* or *Off*.",
  "**Settings › Your Tools (ASTAP, GraXpert, RC-Astro)…** - see, locate and download the helper programs, and activate RC-Astro licences (4.1).",
  "**Settings › Saved Progress…** - where crash-recovery copies are kept, how much space they use and how long they are kept (30 days by default).",
  "**Settings › Show Intro Animation at Startup** - turn the start-up animation on or off. *Help › Play the Intro Animation* plays it any time.",
]);

ch(5, "Finding your way around");
sec("5.1", "The banner and page buttons", ["Banner", "Continue button", "Back button", "Page navigation"]);
p("Every page has the same banner at the top: **SubZeroAstro**, the author and the **version** in small print (e.g. *v0.9.10 beta*), and the **page title**. On the right is a blue **Continue to …  ➜** button that turns to the next page; on the left a grey **◀ previous page** button goes back. Going back keeps your work: press Continue again and the page you left comes back exactly as it was.");
sec("5.2", "The log and the status bar", ["Log", "Status bar", "Progress bar"]);
p("The text area at the bottom of each page is the **log**: it tells you what was done, what was found and any warnings, in plain language. The orange bar shows progress and an estimate of the time left.");
p("When a page has finished its job, its green run button turns **amber** - **✓ Stacked – Run Again?** on page 2, **✓ Done – Run Again?** on page 3 - so you can see at a glance that the page is done. Pressing it again asks first, so nothing is redone by accident. Add new images and it turns green again.");
sec("5.3", "Saved progress and undo", ["Undo", "Crash recovery", "Resume"]);
bullets([
  "**Page 1**: your folders and flags are saved as you go. **Ctrl+Z** undoes rejects, flags and auto-cull.",
  "**Pick up where you left off**: when you start SubZeroAstro again - after quitting normally or after a crash - it offers **Pick Up Where I Left Off**: your frames and flags come back and it goes straight to the page you were on (Calibrate and Stack, Image Processing with the same masters, or Finishing Touches, where your steps resume). **Start Fresh** begins a new project; after *End this Project and Start New Project* nothing is offered.",
  "**Page 2**: each master is noted as soon as it is saved; an interrupted run offers **Continue With the Rest**.",
  "**Page 3**: the image is saved after every step; **Go Back & Redo This Step** and **Start Fresh** are under the preview.",
  "**Page 4**: every step is saved: **Undo** (Ctrl+Z), **Go Back to Here**, **Back to Start**, and reopening the same masters later resumes the project.",
  "**Clear everything and start from scratch**: **File › Clear Everything – Start From Scratch…** forgets the saved quality measurements (page 1 measures every frame again), deletes all saved progress (stacking resume points, page 3 and page 4 step histories, the last session) can move your existing master lights aside into an *Old masters <date>* folder, so a new stack starts truly from nothing, and can reset all settings to factory defaults (your folders, tool locations, rig names and signature are kept) - like PixInsight's *Reset Engine*. Your images, calibration library and masters are never deleted. Then load your light frames again.",
]);
sec("5.4", "Stop Here – Continue in PixInsight", ["Stop Here – Continue in PixInsight", "PixInsight hand-over"]);
p("The purple **Stop Here – Continue in PixInsight** button (pages 2, 3 and 4) saves the image exactly as it is right now as PixInsight-ready XISF, writes a short note listing the PixInsight processes to use next, and opens the folder with the files selected.");

ch(6, "Page 1 – Astro Image Quality and Cull");
img("1_main_window_fwhm.png", "Page 1: frame list (left), the selected frame, the session chart and the measurements (example data)");
sec("6.1", "Loading your sub-frames", ["Load Light Frame Data", "Add Folders", "Several folders at once", "Include subfolders", "Filter used", "Network drive (NAS)"]);
steps([
  "Click **Load Light Frame Data** (Ctrl+O), then **Add Folders…**. You can pick **several folders at once**: click one, then **Shift**-click to add a range, or **Ctrl**-click (**Cmd**-click on a Mac) to add others one by one. Click **Add Folders…** again for folders in other places (other nights, other drives), or **Add Files…** for individual files.",
  "**Include subfolders** is ticked by default, so nights kept in subfolders are found too (FLAT, DARK, BIAS, CALIB and OFFSET folders are skipped). Untick it to load only the files directly in each folder; your choice is remembered.",
  "If your capture software does not record the filter (e.g. ASIAIR with a manual filter drawer), choose it under **Filter used**.",
  "Press **Start Processing**. Frames appear at once and are measured in the background.",
]);
img("7_load_data_window.png", "The Load Light Frame Data window", 480);
tip("Load Light Frame Data is for **light frames only**. To turn bias, darks and flats into masters, use **Stack › Build Calibration Masters…** instead (7.6) - no light frames needed.", "Calibration frames?");
tip("If the file names mention the filter (e.g. `…_SQA55Lult_0001.fit`), SubZeroAstro notices after the first few frames and offers to use it for the whole set. Duplicate files are skipped and reported.");
sec("6.2", "Reading the measurements", ["Score", "FWHM", "Eccentricity", "SNR", "Star count", "Background", "Badges", "Session chart"]);
p("Each frame gets a coloured **badge** (green = good, amber = check it, red = probably reject) and a **score** where 100 = a typical frame for that filter. Frames are only ever compared with frames from the **same camera and filter**.");
table(["Measurement", "What it tells you", "Bad sign"], [
  ["FWHM", "Star size in pixels (and arcseconds)", "Much larger than the median: poor focus or seeing"],
  ["Eccentricity", "Star roundness (0 = round)", "Above ~0.5: tracking, guiding or wind"],
  ["SNR", "How clearly stars stand out", "A drop: cloud, haze, dew, moonlight"],
  ["Stars", "Number of stars detected", "A sudden drop: cloud or trees"],
  ["Background", "Sky brightness", "A rise: moon, twilight, light pollution"],
], [1800, 3800, 3760]);
p("The **session chart** below the image shows one dot per frame in capture order. Choose what to plot, **click** a dot to jump to that frame, or **drag** across the chart to select a range. The **Kept frames** box on the right shows the averages of what you are keeping.");
sec("6.3", "Choosing which frames to keep", ["Auto-cull", "Aggressive", "Semi-aggressive", "Less aggressive", "Flag", "Reject", "REJECTED folder", "Auto-Flag"]);
steps([
  "Click an **Auto-cull** button - **Aggressive**, **Semi-aggressive** or **Less aggressive**. It only **flags** frames (yellow ring); hover over a button first to see how many it would flag.",
  "Review the flagged frames (switch the list to *Flagged*); the measurements panel says why each was picked. Press **F** to un-flag one you want to keep.",
  "Press **Reject Flagged** (Shift+X). Rejected frames are moved into a `REJECTED` subfolder - nothing is deleted, and **Ctrl+Z** brings them back.",
]);
img("6_semi_aggressive_flagged.png", "After Semi-aggressive: suspect frames are flagged for review (example data)");
p("**Auto-Flag** (Ctrl+F) lets you set your own limits. **X** rejects or restores the selected frame(s), **F** flags them, **D** clears all flags.");
sec("6.4", "Filters and the FITS header", ["Set Filter Name", "FITS header", "Hide image"]);
bullets([
  "**Frame › Set Filter Name…** changes the filter for the selected frames (your files are not changed). Use it when the filter is missing or wrong - flats are matched by filter.",
  "**Ctrl+H** opens the full **FITS header** of the selected file, with search and Copy All.",
  "**Ctrl+I** hides or shows the image - useful on a slow network drive.",
]);
sec("6.5", "Tonight's Target Picker", ["Tonight's Target Picker", "Target picker", "What to image tonight", "Horizon", "N.I.N.A. horizon file", "Import Targets", "Telescopius"]);
p("**File › Tonight's Target Picker…** (Ctrl+Shift+T) answers *what can I shoot tonight from here?* Everything is worked out on your computer - no account, no internet, no AI - from the position of the Sun, Moon and each object through the night.");
bullets([
  "**Where and when**: your rig's latitude and longitude (west is negative). Pick one of your **N.I.N.A. profiles** - it also brings your focal length, pixel size and horizon file - or **Fill In From One of My Pictures…** to read them from a FITS/XISF header. Choose the night (tonight by default).",
  "**Your sky**: click the compass to mark what you can't see - each click cycles *open → low trees (20°) → tall trees (35°) → blocked*. Or type it: *open sky from NNW to SSW* (the shorter way round), *trees to the east up to 30 degrees*, *house blocks N*, then **Use This Description**. A N.I.N.A. horizon file loads with **Load a N.I.N.A. Horizon File…**. **Lowest altitude** (30° by default) keeps you out of the thick, light-polluted air near the horizon.",
  "**Your rig**: focal length, pixel size and sensor size give the *In your frame* column - how much of your picture each target fills, or whether it needs a mosaic.",
  "**Work It Out** lists every target that is in the dark and above your horizon for at least the hours you ask: its window, how high it climbs and when, how far it is from the Moon, and a **Best** score (more hours, higher up, Moon out of the way - a bright Moon hurts galaxies far more than emission nebulae shot with a narrowband filter - and a good fit in your frame). Tick the types you want, filter by size, **group** by type, size or time of night, or type a name in **Find**. The chart below shows the selected target's altitude through the night with your horizon and the Moon.",
  "**Into N.I.N.A.**: select targets and **Send to N.I.N.A.** - they appear straight away in N.I.N.A.'s target list (Advanced Sequencer › Targets › *SubZeroAstro*), ready to drag into your sequence. Pick **N.I.N.A. Template…** once - one of your own saved targets - and every target you send gets its exposures, autofocus, dithering and checks, with only the name and coordinates changed. (Running N.I.N.A. on another computer? Copy the files from *Documents/N.I.N.A/Targets/SubZeroAstro* to the same folder there.) **Save List for N.I.N.A.…** writes a target list for the simple Sequencer's **Import Targets** (the CSV layout Telescopius exports), and **Copy Coordinates** puts RA/Dec (J2000) on the clipboard for the Framing Assistant. **About This Target** shows the Wikipedia and SIMBAD notes; **Telescopius** copies the name and opens Telescopius.",
  "The objects are N.I.N.A.'s deep-sky database - read from N.I.N.A. itself when it is installed on this computer, otherwise SubZeroAstro's built-in copy of its 4 000+ imaging-worthy objects. Times are in your computer's time zone; positions are good to a fraction of a degree - plenty for planning.",
]);
sec("6.6", "On to stacking", ["Continue to Calibrate and Stack"]);
p("When you are happy, click **Calibrate and Stack ➜** at the top right. Only your kept frames go to page 2.");

ch(7, "Page 2 – Calibrate and Stack");
img("10_stacking_done.png", "Page 2 after stacking: one row per target and filter, with the calibration found for each (example data)");
sec("7.1", "The light frame groups", ["Light groups", "Filter column", "Telescope (rig)", "Broadband and narrowband rows"]);
p("Each target's rows come **broadband first** (L, R, G, B - tinted blue), then **narrowband** (Ha, OIII, SII and dual-band filters - tinted red). The **Stack: Broadband / Narrowband** tick boxes above the table include or leave out a whole kind at once - handy when one target has both, and you want, say, only the LRGB + Ha of a galaxy (untick a single row with its own box).");
p("Your kept frames are grouped by **target + filter + camera + telescope** - one row each, with the number of frames, total exposure and nights. The **Filter** of a row can be changed from its drop-down (for example to match your flats).");
sec("7.2", "Calibration: what the ✓ ⚠ ✗ mean", ["Calibration", "Old flats", "Missing calibration", "Flat warnings", "Calibration check", "Light leak", "SkyFlats (Mote Buster)", "Sky flat", "No flats"]);
table(["Mark", "Meaning"], [
  ["✓ green", "A matching calibration master was found (or will be built)."],
  ["⚠ orange", "Usable, with a note - e.g. darks scaled from another exposure, or flats taken long ago (*\"dust may have moved\"*). It **will** be used."],
  ["✗ red", "Nothing suitable - e.g. *No flats for filter OIII*. You can still stack, without that correction."],
], [1600, 7760]);
p("Click a row to read the details below the table. Add calibration with **Add Calibration Folder…**, **Add Calibration Files…** or **Search Calibration Folder for Subframes**.");
p("**No flats at all? SkyFlats (Mote Buster).** When a filter has no flats, SubZeroAstro can make a flat from your **light frames** (option *SkyFlats (Mote Buster) when flats are missing*, on by default; at least 12 frames). It uses every light with that filter, camera and rig - from all your targets and nights, because the stars land somewhere different in each. Each frame is scaled to the same brightness, its stars are hidden, and its own light-pollution tilt is taken out (that tilt changes from frame to frame, while the vignetting stays put). Of every pixel's values only the lower half is used - a nebula can only add light - and what remains is what your optics and dust do to an even sky: a flat. The Flats cell then shows ⚠ *SkyFlat (Mote Buster) (made from N light frames)*. SkyFlats are never saved to your library. (For donuts that remain, *Mote Buster – Fix Dust Donuts* on Finishing Touches works on one picture.)");
p("**Ready-made data is recognised.** Lights bought already calibrated - from Telescope Live and similar services, or calibrated earlier in other software - must not be calibrated again (the dark would be subtracted twice and the flat divided in twice). SubZeroAstro recognises them from their headers and names: Telescope Live's *PROC = True*, MaxIm DL's *CALSTAT*, PixInsight's and Siril's calibration notes, and names such as *_cal.fits* or PixInsight's *_c.xisf*. Such a set shows ✓ *not needed - the lights are already calibrated (…)* in the Bias, Darks and Flats cells, gets no SkyFlat, raises no missing-calibration warnings - and its frames are never used to build a SkyFlat for your own lights.");
p("**Every SkyFlat is checked before it is used.** If it may not be good enough - too few usable frames, grainy, a brightness range no vignetting has (nebulosity or a light-pollution gradient got in), large areas filled in - or it cannot be made at all because the master dark or bias is as bright as the sky (a light leak, or the wrong exposure or gain), SubZeroAstro stops and asks: **Proceed Without Flats** (the default), **Use the SkyFlat Anyway**, or **Cancel Stacking**. The reasons are listed in the box and in the log. A poor flat adds blotches or rings instead of removing them; without flats you can still fix dust donuts afterwards with *Mote Buster – Fix Dust Donuts* on Finishing Touches.");
p("**Meridian flips are handled.** When the camera turned 180° with the telescope halfway through the night, the nebula sits on different sensor pixels before and after the flip. SubZeroAstro finds the two sides from the frames themselves and takes each pixel of the SkyFlat from the side that shows plain sky there - otherwise a faint dark copy of the nebula, turned upside down, would be printed into the picture. The stacking log says when this was done.");
tip("A SkyFlat is best from several targets. From a single target whose nebula fills most of the picture, part of the nebula could be mistaken for vignetting - the Flats note warns you. Real flats are still the best: take some when you can.", "SkyFlats");
p("**Calibration check.** Every stack checks its dark (or bias) against the lights and writes a *Calibration check* line in the log: light background - dark = sky left. If a master does not fit - **brighter than the lights**, **uneven like a picture** (light leaked in while the darks were taken: lens cap, filter wheel, focuser or room light) or **practically zero** (saved on another scale) - a window says so when stacking finishes. Such a frame makes bright bands, borders or blotches that no later step can fix: remove it from the *Calibration frames* list, or take new ones (scope fully covered, same gain, offset, temperature and exposure as the lights).");
warn("If the Flats cell says the flats have **no matching bias or dark-flats**, the correction will be too strong: bright corners and dark rings. Add a bias or dark-flats for that camera and gain.");
sec("7.3", "Stacking options", ["Reject outliers", "Weight frames by noise", "Colour cameras", "Line up all filters", "Save as FITS/XISF"]);
bullets([
  "**Reject outliers beyond 3.0 σ** - removes satellite trails, planes and hot pixels. The default suits almost everything.",
  "**Weight frames by noise** - better frames count more.",
  "**Colour cameras: Full resolution** (default) or half resolution (faster, less noise per pixel).",
  "**Save as** FITS and/or XISF; **Save new calibration masters** into your library.",
  "**Line up all filters of a target** (on) - all masters of a target share one pixel grid, ready to combine (LRGB, SHO).",
]);
p("**Sharp stars to the corners.** Every telescope bends the field a little, mostly at the edges. After a **meridian flip** (or between nights with another setup) that bending is turned round with the camera, so a plain shift-and-rotate alignment lines up the centre but leaves stars a pixel or two apart near the edges - doubled or smeared corner stars. SubZeroAstro measures each frame's stars against the reference and, when the edges are off, lines that frame up with a smooth distortion model instead. The stack log says how many frames were corrected.");
sec("7.4", "Running the stack", ["Check & Stack", "Continue With the Rest"]);
p("Press **▶ Check & Stack**. Anything missing is listed first, with **Stack Anyway** or **Add Calibration Frames First…**. The log shows each step. If a run is interrupted, the next run offers **Continue With the Rest**. When it finishes, the button turns amber: **✓ Stacked – Run Again?**. Stacking again never overwrites a master - new versions are saved as `_2`, `_3`…");
sec("7.5", "Results", ["Open Master Lights Folder", "Stack log"]);
p("Masters are saved in your master lights folder, one subfolder per target, with a `_stacklog.txt` listing every frame used. **Open Master Lights Folder** shows them. Click **Continue to Image Processing ➜** to carry on.");

sec("7.6", "Build Calibration Masters (no lights needed)", ["Build Calibration Masters", "Master bias", "Master dark", "Master flat", "Dark-flats", "Calibration library"]);
p("Took a fresh set of darks, flats or bias? Turn them into masters right away - **no light frames needed**: on page 1 choose **Stack › Build Calibration Masters…** (or click **Build Calibration Masters…** in the *Calibration frames* box on page 2). Choose the folder with the subframes - bias, darks, flats and dark-flats can all be mixed together in one folder (for example *RedCat51Calis*), or sit in subfolders. SubZeroAstro reads each file's header (and, if the header doesn't say, the file or folder name: *Darks*, *Flats*, *Bias*…) and sorts them:");
bullets(["**Bias** - one master per gain and offset.",
  "**Darks** - one master per exposure, gain, offset and temperature (180 s and 300 s darks make two master darks).",
  "**Flats** - one master per filter (and night). Each flat is calibrated with dark-flats or short darks of the same exposure, or else a bias - from the same folder or already in your library.",
  "**Dark-flats** - saved as masters too, for next time."]);
p("The telescope (rig) name is taken from the folder name (*RedCat51Calis* → *RedCat51*) or the headers - check it, because flats are only used for lights from the same rig. Untick any set you don't want, then click **Build Masters**. Each master is saved to your calibration library (*<library>/<rig>/Masters/<camera>/…*) with its settings in the name, and is used automatically whenever you stack lights with matching settings.");
p("Every master is checked as it is built. You are warned when a dark or bias is **uneven like a picture** (light leaked in), a dark is **far brighter than the bias** (light, or the camera not cooled), a bias is **practically zero** (saved on a 0..1 scale or with a pedestal taken off), or flats are **over-exposed** (clipped) or **very dim**. A warned master is still saved, so you can compare - but better to retake that set.");
tip("Darks: cap the telescope (or take the camera off) and cover it, at night or in a dark room, with the filter wheel on a filter - not an empty slot. Match the lights' exposure, gain, offset and temperature.", "Taking good darks");

ch(8, "Page 3 – Astro Image Processing");
img("12_prepare_done.png", "Page 3 after processing: masters (left), preview with crop box, step history, and the step settings (right)");
sec("8.1", "Masters and target sets", ["Target sets", "Will be lined up", "Add Masters"]);
p("The masters from page 2 are loaded automatically (or use **Add Masters…** / **Add Folder…**; with **Add Folder…** only the newest restack of each master is taken). Masters of the **same target** form one row - e.g. *NGC281 [Ha, OIII, SII]*. If they were stacked in separate runs the row says **will be lined up**: they are matched star-to-star onto one grid before anything else, and edges not covered by every filter are trimmed.");
sec("8.2", "Crop", ["Crop", "Auto-crop"]);
p("The yellow box on the preview is the crop. **Auto-crop** trims the ragged stacking edges; drag the box or its sides to adjust, or **Full image** to keep everything. The same box is used for **every filter**, so they stay aligned.");
sec("8.3", "Plate solve", ["Plate solve", "Blind solve", "Search radius"]);
p("ASTAP finds where the picture points in the sky, using the position and pixel scale recorded in the masters. If it fails, tick **Blind solve** (slower, searches the whole sky).");
sec("8.4", "Remove gradients", ["Gradient removal", "GraXpert", "Light pollution"]);
p("Removes light-pollution glow and uneven backgrounds. **GraXpert (AI)** is recommended; **Smoothing** 0.1 is a good default. **Correction: Automatic** picks for each master: *Subtraction* when the lights were flat-calibrated (your flats, a SkyFlat, or ready-made data such as Telescope Live) - light pollution and moon glow add to the picture - and *Division* only for a master stacked without any flats, where vignetting is still in it. The log says which was used. Without GraXpert a built-in model is used.");
sec("8.5", "The XTerminators", ["BlurXTerminator", "NoiseXTerminator", "StarXTerminator", "Unscreen Stars"]);
table(["Tool", "What it does", "Good starting settings"], [
  ["BlurXTerminator", "Sharpens stars and detail", "Sharpen Stars 0.50, Adjust Star Halos 0.40, Sharpen Nonstellar 0.50, Automatic PSF on (ready-made data such as Telescope Live: Sharpen Stars 0.55, Adjust Star Halos 0.50 - its clean signal takes firmer sharpening; the log says so)"],
  ["NoiseXTerminator", "Reduces noise (a light pass on unstretched data)", "Denoise 0.6"],
  ["StarXTerminator", "Separates a **starless** and a **stars-only** image", "Generate Star Image on; **Unscreen Stars off**. When the picture is big (over about 40 megapixels, e.g. after drizzle) or the stars are large (about 5 px or more) it works on a half-size copy with the most tile overlap, so the glow of a bright star fits its tiles and leaves no squares; the starless picture keeps its full detail (the log says so)"],
], [2000, 3900, 3460]);
p("Hover over any setting for a plain-language explanation. **Unscreen Stars** should stay **off** here: the stars image is then simply *original minus starless* and is added back later (what Finishing Touches expects). Turn it on only if you will recombine stretched images with a screen blend in PixInsight.");
sec("8.6", "Saving the results", ["Save final images", "Output folder", "Quick-look image", "Preview palette"]);
bullets([
  "The first time you run, SubZeroAstro asks **where the processed images should go**: *Next to the Masters* (a `Prepared` folder), *Folder Used Last Time*, or *Choose a Folder…*. Each target gets its own subfolder.",
  "**Linear masters** as XISF and/or FITS (these are what you process further), and a **stretched preview** as TIFF, JPG and/or PNG in the chosen **Preview palette**.",
]);
sec("8.7", "Running and the step history", ["Run Selected Steps", "Step history", "Hold for Before", "Go Back & Redo This Step", "Start Fresh"]);
p("Tick the steps you want and press **▶ Run Selected Steps**. As each step finishes, the preview shows its result. Under **Step history**:");
bullets([
  "**Show** - look at the result after any step.",
  "**Hold for Before** - press and hold to compare with the step before.",
  "**Go Back & Redo This Step** - change that step's settings and run again; only that step and the ones after it are redone.",
  "**Start Fresh** - forget the saved steps and start from the original masters.",
]);
img("14_step_history.png", "Step history under the preview (example data)");
p("When it finishes, the bar at the bottom reads **✓ Ready for Processing Finishing Touches**, the run button turns amber (**✓ Done – Run Again?**) and the **Astro Image Processing – Complete** window opens. It says what comes next and, for PixInsight users, lists a recommended order of PixInsight processes (also saved next to your files).");
img("13_ready_for_pixinsight.png", "The Astro Image Processing – Complete window");
p("Click **Continue to Finishing Touches ➜**, or **Stop Here – Continue in PixInsight**.");

ch(9, "Page 4 – Finishing Touches");
img("15_finish_start.png", "Finishing Touches: the workflow (left), the recommended next step and the picture (example data)");
sec("9.1", "How it works", ["Recommended next", "Do It", "Do It Again", "Full Window", "Zoom", "Skip", "Pick the one you like", "Fine-tune", "Apply", "Cancel", "Workflow list", "Several masters per filter", "Field distortion", "Double stars"]);
steps([
  "The **workflow list** on the left shows every step for your kind of data: ✓ done, ➜ next, ○ later, – not needed, ◇ optional.",
  "The card at the top says what is **Recommended next** and why. Click **Do It** (or click any step in the list).",
  "Pick a version: each step shows 3-5 small pictures of **your** image - gentle to strong, or different colour mixes. ★ marks the suggestion. **Clicking only previews it big** - nothing changes yet.",
  "Optional: tick **Fine-tune** and move the plain-language sliders; the preview follows. **Reset** returns to the version you picked. Under the sliders the app says how much of the picture the setting visibly changes - if it says *This barely changes your picture*, there is little for that step to do on this target (no green cast to remove, no burnt-out core to tame), so the sliders have almost nothing to work on.",
  "Press the green **Apply** to keep it - or **Cancel** to leave the picture exactly as it was. When the step is done the panel says so in green (**✓ … applied**), Apply disappears and an amber **✓ Applied – Next Step ➜** button takes you straight to the next recommended step - so after a break you can always tell whether a step was applied. Steps that can be repeated also offer **Apply Again**. The picture goes back to the whole view after every step.",
  "Not quite enough? Many steps can be run again on their own result: after one of them the card shows **↻ Do It Again** (and a ✓ step in the list can be clicked to run it again). That works for Balance Colours, Remove Green Cast, Tame Bright Core, Curves, Saturation, Bring Out Hidden Structure, Shrink Stars, Star Colours, Mote Buster, Darken the Sky Background, Final Noise Reduction, Flip, Final Crop and Spot Fix. One-way steps - combining, brightening, taking the stars out or putting them back, signing - are not repeated; use **Go Back to Here** to redo those.",
]);
img("17_finish_brighten_choices.png", "Picking a version: Brighten the Image (Stretch) offers five strengths (example data)");
p("**Several masters per filter, or masters from different setups.** When you load more than one master of the same filter (other nights, other cameras or telescopes), all of them are used: they are brought to the same brightness and averaged, the less noisy ones counting more. This works the same for colour cameras (each colour separately). With a colour camera and a **dual-band filter** (L-Ultimate, L-eXtreme …), masters taken **without** the filter are left out of the picture - broadband starlight would bloat the stars and wash out the Ha / OIII colours - and the log says so. Every master is first matched star by star to one reference master. Masters from another camera or telescope are bent slightly differently by their optics - most at the edges, which would give coloured double stars in the corners - so SubZeroAstro measures that distortion from hundreds of stars and corrects it. When the masters carry a plate solution (data from remote observatories usually does, and page 3 adds one with ASTAP), it guides the match: the masters are lined up from where they point on the sky and the stars fine-tune it, so even masters with few stars in common line up. The deepest master (the most hours) is the reference everything is lined up to. Masters that page 3 already put on one grid are used as they are, and starless masters (StarXTerminator on page 3) are measured together with their stars-only images, so the stars are always there to line up by. After lining up, every master is checked: its four corners are compared with the reference's, and one still more than 1.5 px off is left out (when its filter has other masters) - the note says *alignment check: … within X px in every corner*. A master whose stars cannot be matched to the others is left out (when its filter has other masters), so it can't cause double stars, and the picture is cropped to the area all masters cover. The note under the title says how many masters were combined, corrected or left out.");
p("**Looking closely.** Scroll on the picture to zoom in (past the preview's own pixels the full-resolution picture is drawn), drag to move around, double-click to fit it again. **⛶ Full Window** (next to *Show:*) hides the step list, the panels and the log so the picture fills the window - press it again or **Esc** to go back. Both work at every step, also while looking back through earlier steps.");
p("**Bringing out one colour.** *Adjust Colours* strengthens or calms one colour range at a time: pick a version (Boost blues, Boost reds, Blues and reds, Boost magentas, Calm the greens), then fine-tune with one slider per colour (1 = unchanged), *Brighten boosted colours* and *Protect the sky*. It is best done after *Colour Intensity* and before *Final Noise Reduction* - boosting a colour also boosts its colour noise, which the noise step then cleans - but it can be opened at any time later too (and run again).");
sec("9.2", "Going back", ["Undo", "Go Back to Here", "Back to Start", "Hold for Before (Finishing Touches)"]);
bullets([
  "**Undo** (Ctrl+Z) removes the last step.",
  "**Show:** pick any earlier step to look at it; **Hold for Before** compares; **Go Back to Here** continues from that step.",
  "**Back to Start** returns to the original masters.",
  "Every step is saved. Opening the same masters again offers to **carry on where you left off**.",
]);
sec("9.3", "The steps", ["Combine Colour Channels", "Choose a Colour Palette", "Balance Colours", "Remove Green Cast", "Mote Buster", "Fix Dust Donuts", "Take Stars Out", "Brighten the Image (Stretch)", "Tame Bright Core", "Curves", "Saturation", "Adjust Colours", "Bring Out Hidden Structure", "Brighten Stars Separately", "Shrink Stars", "Star Colours", "Put Stars Back", "Darken the Sky Background", "Final Noise Reduction", "Flip or Rotate the Picture", "Final Crop", "Spot Fix", "Sign Your Picture", "Add Hydrogen Glow"]);
p("Which steps appear depends on your data. In typical order:");
table(["Step", "What it does", "PixInsight equivalent"], [
  ["Add Hydrogen Glow", "Adds Ha to the red of an RGB set, without its stars. For a galaxy (M31, M33, M101, NGC 891…) it adds only the compact HII regions - the red knots along the arms", "Continuum subtraction + PixelMath / NBRGBCombination"],
  ["Combine Colour Channels", "Builds colour from separate filters: RGB, LRGB, Hubble (SHO), HSO, HOO, Foraxx… and **HOO + faint OIII**, which lifts faint OIII-only glow (the Squid in the Flying Bat) while OIII that follows the Ha is left alone (slider: Faint OIII boost)", "ChannelCombination / LRGBCombination; OIII minus scaled Ha in PixelMath"],
  ["Choose a Colour Palette", "Dual-band colour cameras: separates Ha and OIII (or SII and OIII) and offers HOO, Gold & blue, Foraxx…", "PixelMath palettes"],
  ["Balance Colours", "Neutral sky and natural star colour", "BackgroundNeutralization / colour calibration"],
  ["Remove Green Cast", "Removes the green tint (or turns Hubble colours gold and blue)", "SCNR"],
  ["Mote Buster ◇", "Fix Dust Donuts: finds the round shadows of dust and divides them out with a synthetic flat - the detail underneath comes back", "A new flat, or DBE with division"],
  ["Take Stars Out", "Hides the stars so the nebula can be brightened strongly (instant if page 3 already separated them)", "StarXTerminator"],
  ["Clean Bright-Star Leftovers", "Recommended only when something is found. StarXTerminator paints a smooth, grain-free square over each big star; once the nebula and the stars are brightened differently those squares show, often with bright bars along their edges and a leftover dot where a close double star was. This finds the painted squares by their missing grain and patches each with the surroundings' own texture. The square glow and the bars around it (what you see once the picture is brightened) are taken away while the nebula's own detail is kept. Four versions are offered: Clean, no glow (suggested - the star's own glow comes back with the stars), Clean, faint glow, Clean, soft glow, and Don't clean. Squares inside bright, structured nebula (a reflection nebula hugging the star) are left alone - use Spot Fix on those if they show", "CloneStamp"],
  ["Brighten the Image (Stretch)", "The stretch - five strengths, three methods", "STF / HistogramTransformation / GHS"],
  ["Bring Out the Tail ◇ (comets)", "Takes away part of the round glow of the comet's head (the coma), measured all the way round the nucleus, so the tail and ion streamers stand out", "1/rho or azimuthal-median coma model + PixelMath"],
  ["Reveal Comet Jets ◇ (comets)", "Brings out jets and shells near the nucleus by comparing the picture with copies of itself turned a little around the nucleus (Larson-Sekanina); *Jets map only* gives the grey study picture. Best with the stars out", "LarsonSekanina"],
  ["Adjust Colours ◇", "Bring out (or calm) one colour at a time - reds, yellows, greens, cyans, blues, magentas; grey and white stay as they are", "ColorSaturation"],
  ["Tame Bright Core ◇", "Recovers detail in a burnt-out galaxy core or bright nebula: lowers the core's glow towards its surroundings and brings its structure back from the linear data; a nucleus burnt out even in the masters is rounded off so no hard white dot is left, and the stronger versions take in more of the area around the core (works best right after Brighten the Image (Stretch))", "HDRMultiscaleTransform with a mask"],
  ["Adjust Brightness & Contrast", "Curves presets, sliders and a curve you can drag", "CurvesTransformation"],
  ["Colour Intensity", "Richer or calmer colour; dark sky protected", "Saturation curve"],
  ["Bring Out Hidden Structure ◇", "Local contrast for dust lanes and cores; the empty sky is protected (fine-tune *Protect the sky*)", "LocalHistogramEqualization"],
  ["Brighten Stars Separately", "Gentle star stretch - small, colourful stars. The star steps show the stars on their own; tick *Show the nebula behind the stars* to see them over the nebula", "Star stretch"],
  ["Shrink Stars", "Makes stars smaller", "Star reduction / MorphologicalTransformation"],
  ["Star Colours", "Natural star colours (estimated from your data) or clean white stars - recommended for narrowband", "Star colour correction"],
  ["Put Stars Back", "Blends the stars back at the strength you pick", "PixelMath screen blend"],
  ["Darken the Sky Background ◇", "Darker (or lighter) sky without dimming the nebula, faint dust or stars; also evens out a brighter corner or side", "ABE/DBE model + PixelMath"],
  ["Final Noise Reduction ◇", "A last, light NoiseXTerminator pass on the finished picture (needs the RC-Astro CLI)", "NoiseXTerminator"],
  ["Flip or Rotate the Picture ◇", "Flip Horizontally and/or Vertically (each press toggles), Rotate 90° Clockwise or Counter-clockwise (each press turns once more) - the saved pictures are turned too", "FastRotation"],
  ["Final Crop ◇", "A last clean-up crop; finds ragged or off-colour edges for you", "DynamicCrop"],
  ["Spot Fix ◇", "Click dust donuts, satellite remnants or hot pixels to patch them", "CloneStamp"],
  ["Sign Your Picture ◇", "Your signature - and the target name above it - in a corner or wherever you drag it", "Annotation"],
], [2500, 4460, 2400]);
p("◇ = optional: offered in the list but not pushed by *Recommended next*.");
sec("9.4", "Mote Buster: dust donuts", ["Mote Buster", "Fix Dust Donuts", "Dust donuts", "Dust motes", "Synthetic flat"]);
p("Dust on a filter or on the camera window casts round shadows - **dust donuts** (rings with a reflector, discs with a refractor). A good flat removes them; when the flat is missing or no longer matches, **Mote Buster – Fix Dust Donuts** helps. A dust shadow does not wipe anything out: it lets through the same fraction of light everywhere inside it, nebula included. So SubZeroAstro measures each donut's shadow ring by ring - in 16 directions against the sky just outside it, levelled for any gradient - builds a **synthetic flat** just for that donut and divides it out, the way a uniformity (flood-field) correction works. The detail underneath comes back instead of being painted over.");
bullets([
  "Open **Mote Buster** while the picture is still unstretched (it comes before *Take Stars Out*). The donuts it finds in the sky are already circled in yellow.",
  "Donuts on bright nebulosity are not searched automatically (a gap between filaments could look like one) - **click** those yourself. Click to place a circle, then **turn the mouse wheel to size that circle** while you watch (just cover the donut). Click a circle to pick it again (it turns blue) and drag to move it; to remove it, **right-click** it, **Control-click** it on a Mac (Option- or Command-click also work), or pick it and press **Delete**. A dashed ring follows the mouse to show the size of the next circle. Clicks need not be exact: each circle is centred on its donut automatically.",
  "**Bright rings count too.** When the dust moved between the flats and the lights (or a filter was cleaned), the flat over-corrects: you get a *brighter* ring, often with a dark centre, and with mono filters it is usually in one filter only - a coloured (e.g. purple) halo. Mote Buster finds and fixes these as well, dimming the bright part and lifting the dark part. Every colour is centred and measured on its own (the same speck sits a little differently in each filter), and lopsided rings are followed too.",
  "Check the preview, then press the fix button. Mote Buster stays open on the **fixed picture**, so you can circle any donuts that are left and fix again - as many rounds as you like (each round is its own step; *Undo Last Step* takes back the last one). Press **✓ Applied – Next Step ➜** when you are happy. Spot Fix works the same way. For small specks (hot pixels, satellite remnants) use **Spot Fix** instead.",
]);
sec("9.5", "Stars: take out, brighten, colour, put back", ["Stars workflow", "Starless"]);
p("Stars and nebula look best when processed separately. **Take Stars Out** hides the stars (the picture shows the nebula alone); the nebula steps follow; then **Brighten Stars Separately**, **Shrink Stars** and **Star Colours** work on the stars (their previews show the stars); **Put Stars Back** recombines them. If you **Skip** Take Stars Out, the stars simply stay in the picture.");
sec("9.6", "Darker sky, Final Crop, Spot Fix and your signature", ["Darken the Sky Background", "Grey sky", "Final Crop", "Spot Fix", "Sign Your Picture", "Signature fonts"]);
bullets([
  "**Darken the Sky Background** is for a sky that looks grey or washed out. Pick *Slightly darker*, *Darker*, *Much darker* or *Lighter* (each shows the sky's brightness before and after), then fine-tune with **Sky darkness**. It works like PixInsight's background extraction: SubZeroAstro maps the sky glow across the picture (a nebula cannot push that map up) and takes part of it away. Everything that shines above the sky - nebula, faint dust, stars - keeps exactly the brightness it had **above the sky**, so faint filaments stay visible and stars are not dimmed. A brighter corner or side loses more, so the sky also evens out. Darker skies show grain more: a lighter setting often looks more natural.",
  "**Final Noise Reduction** runs NoiseXTerminator once more on the finished picture - most useful right after a darker sky, which shows grain more. The choices (*Light*, *Medium*, *Strong*) are shown at full size on the busiest part of the picture so you can judge the grain; **Strength** fine-tunes it. (BlurXTerminator is not offered here on purpose: it is made for unstretched data and already runs on page 3. For more crispness at the end, use **Bring Out Hidden Structure**.)",
  "**Final Crop** shows a yellow box on the whole picture, already placed where it found ragged or off-colour edges. Drag it or its sides (**Found Edges** / **Whole Picture** reset it) and press **Crop to the Box**.",
  "**Spot Fix**: click the middle of a blemish - a circle appears - then **turn the mouse wheel to make that circle bigger or smaller** until it just covers the blemish. Click a circle to pick it again (it turns blue) and drag to move it; to remove it, **right-click** it, **Control-click** it on a Mac (Option- or Command-click also work), or pick it and press **Delete**. A dashed ring follows the mouse to show the size of the next circle. Circles go from a hot pixel to a large satellite blob or halo (up to 700 pixels in radius, like Mote Buster's), and the patch blends in smoothly with the sky's own slope and grain even when it is big. **Stars inside the circle are kept**: they are lifted off, the sky under them is patched, and they are put back on top - as bright as they really are, since a dust donut dims its stars too. Only the thing at the middle of the circle (what you clicked on - a hot pixel, a satellite remnant, an unwanted star) is removed. Then **Fix the marked spots**.",
  "**Flip or Rotate the Picture** (optional): press **Flip Horizontally** (left ↔ right) and/or **Flip Vertically** (upside down) - each press flips the preview, pressing again flips it back (both together turn it 180°). **Rotate 90° Clockwise** and **Rotate 90° Counter-clockwise** turn it a quarter turn per press (a portrait picture becomes landscape). When it looks right, press **Keep This Orientation**. Useful to match how the object usually looks in charts and other pictures. If the picture is already signed, the preview turns the signature too, but when you keep it the signature is put back on top the right way round, in the same place (steps done after signing - crop, Spot Fix, colour - are done again underneath it).",
  "**Sign Your Picture**: type your name once (it is remembered) and, if you like, a **Target name** for a line above it - filled in for you (for example *NGC 281 \"Pacman Nebula\"*) and remembered for each target; empty it for none. Pick a font - *Elegant script*, *Handwritten* or *Classic* - a corner and a colour, and fine-tune size, opacity and distance from the edge. To put it anywhere else, **drag the dashed box** on the picture (or click where it should go); *Place* then says *Where I dragged it*. Do this last: the signature becomes part of the picture.",
]);
img("21_finish_signature.png", "Sign Your Picture: signature text, corner, colour and three fonts (example data)");
sec("9.7", "Saving and sharing", ["Save the Picture", "Save Project As", "Open Project", "Share to AstroBin", "AstroBin"]);
bullets([
  "**Save the Picture…** - TIFF (16-bit), JPG, PNG, XISF or FITS. All the files of one picture go into **their own folder inside the one you pick**, named after the target and telescope - for example *Final Astro Pics/M101_RedCat51/* holds M101_finished.tif, .jpg, .fits and the description. The folder you pick is remembered: each target keeps its own, and a new target starts in the folder you used last. Tick **Also save a short description of the target** to save *<target>_about.txt* next to the pictures: the opening paragraphs of its Wikipedia article plus **facts from SIMBAD** (the professional astronomy database at CDS, Strasbourg) - the kind of object, its position (RA / Dec) and brightness. Faint objects without a Wikipedia article still get the SIMBAD facts. It is shown first so you can edit it (or type your own when there is no internet), and both sources are credited. **Add a small “Processed with SubZeroAstro” credit** (ticked at first) puts that line in small, light-grey, see-through letters in the bottom-left corner of the TIFF, JPG and PNG pictures - or the bottom-right corner if your signature is bottom-left. XISF and FITS files are never marked, and untick the box to leave the pictures clean (the choice is remembered; pictures shared to AstroBin follow it too).",
  "**Save Project As… / Open Project…** - keep every step in a folder of your choice and carry on later. A saved project can also be opened **right from the start**: the **Open a Saved Project…** button in the Load Light Frame Data window (or File › Open a Saved Project, Ctrl+Shift+O, on page 1) goes straight to Finishing Touches on its last step. Pick the project's folder or the folder you saved it into. **Keep the masters with the project?** (asked when you save; Yes by default) copies the masters the project uses - with their star images and stack logs - into a *Masters* folder inside the project, so one folder holds everything for the target and you can move or back it up as a whole; redoing page 3, adding nights and Share to AstroBin keep working. It also keeps the stacked master behind each page 3 (*_prepared*) master, so you can redo page 3 from scratch. Your original masters are copied, never moved or deleted - delete them yourself once the project reopens fine. A project folder with its masters inside can be copied between a Mac and a PC and still finds them. For a project saved before this option existed: open it, choose Save Project As… and pick the same folder.",
  "**Already stacked, no saved project?** Use **Process Existing Masters…** in the same window (or File › Process Existing Master Lights…, Ctrl+Shift+M): pick your master lights - e.g. the D1 and D2 masters of one target - and you go straight to Astro Image Processing (page 2) without restacking. Masters are saved next to your lights' folder, named like *IC1396_Askar D1_Master_2026-06-29_to_2026-07-03.fits*; each restack adds _2, _3, … - from a folder only the newest is used. WBPP masters from PixInsight open the same way.",
  "**A PixInsight project?** File › **Open a PixInsight Project…**: pick the project's *.pxiproject* folder (or the folder it is in). Each picture in it - as PixInsight left it, including changes kept inside the project - is saved as an XISF file in *<project>_SubZeroAstro* next to the project and opened on Astro Image Processing; masks are left out. If the project was moved, put the pictures' original files next to it so they can be found.",
  "**Share to AstroBin…** - saves an upload-ready JPG (and optional TIFF) plus an **acquisition CSV** with every night's subs, exposure, gain, temperature and FWHM, then opens AstroBin's upload page. Upload the picture, and on the *Acquisition* step choose **Import CSV**.",
  "**Stop Here – Continue in PixInsight** - see 5.4.",
]);
sec("9.8", "Starting your next target", ["End this Project and Start New Project", "New project", "Next target"]);
p("When a picture is finished, click the teal **↺ End this Project and Start New Project** button at the top right of Finishing Touches. SubZeroAstro asks first - and if you have not saved the picture yet, offers **Save the Picture First…**. Then every page and image closes, the helper programs are checked again, and you are back on **page 1 (Quality and Cull)** with the **Load Light Frame Data** window open, ready for the next target's light frames: cull, stack, process and finish, from the beginning.");
tip("Nothing on your disk is deleted. Your masters, finished pictures and every Finishing Touches step stay where they are: open the same masters on Finishing Touches later and the project carries on where you stopped.", "Your work is safe");

ch(10, "Recipes for common setups");
sec("10.1", "Colour camera (OSC), broadband", ["OSC broadband recipe"]);
p("Cull › Stack (flats + darks or bias) › page 3: crop, plate solve, GraXpert, BlurX, StarX › page 4: Balance Colours, Remove Green Cast, Take Stars Out, Brighten, Curves, Colour Intensity, star steps, Put Stars Back, Final Crop, Sign.");
sec("10.2", "Colour camera with a dual-band filter", ["Dual-band recipe", "L-Ultimate", "L-eXtreme", "Askar D1", "Askar D2", "L-Synergy"]);
p("Make sure the filter is set (page 1, *Filter used* or *Set Filter Name*) so the matching flats are found. On page 4 the first step is **Choose a Colour Palette** (HOO, Gold & blue, Foraxx…), and **Star Colours** is recommended so the stars are not all teal. Two dual-band filters of the same target - **L-Ultimate + L-Synergy** or **Askar D1 + D2** - are combined into a full **SHO** set automatically: the red pixels of the first give **Ha**, those of the second **SII**, and the green and blue pixels of both give **OIII**. Those pixels also catch a little of the red line, so each OIII estimate carries some Ha or SII; because both filters see the same OIII, SubZeroAstro measures that leakage from your data, for each filter (real OIII is never negative, so where the Ha or SII is brightest the lowest OIII-to-line ratio is the leak - typically 5-9 %, never more than 10 %), takes it out, and combines the two OIII estimates weighted by each filter's exposure time (never less than 20% for either, since both hold real OIII) (the page 4 log shows the numbers). *Combine Colour Channels* then offers the narrowband palettes - **Hubble (SHO)**, HSO, HOO + SII … - exactly as for a mono SHO set. Filter names such as *D1*, *Askar D2*, *Ha/OIII* or *SII-OIII* are all recognised.");
sec("10.3", "Mono camera: LRGB", ["LRGB recipe", "RGB + Ha galaxies", "HII regions"]);
p("**RGB (or LRGB) + Ha of a galaxy**: SubZeroAstro recognises galaxies by name (M31, M33, M51, M81, M101, NGC 891, IC 342… or names like *Andromeda* or *Pinwheel*). **Add Hydrogen Glow** then recommends **HII regions**: the starlight - and the galaxy's own smooth light - is subtracted from the Ha, and only the compact hydrogen knots that clearly stand out are added to red. They stand out crisply against the blue arms without tinting the whole galaxy pink. Tick **Show the hydrogen regions** to see exactly what is added; *All hydrogen glow* adds everything, for galaxies set inside nebulosity.");
p("Each filter is stacked separately on page 2 (one row each; keep *Line up all filters* on). Load all masters together on page 3 - they must appear as **one row**. Page 4 starts with **Combine Colour Channels** (RGB or LRGB).");
sec("10.4", "Mono camera: narrowband (SHO)", ["SHO recipe", "Narrowband recipe", "Hubble palette"]);
p("As 10.3 with Ha, OIII and SII. Filter names from filter wheels such as *Ha 7nm*, *H-alpha*, *O*, *S* or *Antlia Ha 3nm Pro* are recognised. Page 4: **Combine Colour Channels** › Hubble (SHO), Foraxx or HOO; then Remove Green Cast - for Hubble (SHO) and Gold & blue it offers **Gold 50 / 75 / 100%**, which turn the green (your Ha, the strongest signal) gold instead of deleting it, so the bright structure keeps its brightness; *Remove only* is plain SCNR (darker, bluer, more magenta) - and **Star Colours** before Put Stars Back.");
sec("10.5", "Smart telescopes (Seestar, DWARF, Vespera)", ["Smart telescopes", "Seestar", "DWARF", "Vespera", "Alt-az"]);
p("Raw subs are recognised as lights automatically, with no calibration needed (the telescope calibrates on board). Alt-az field rotation is handled during alignment and the rotated edges are cropped. Because the frames turn against each other, the camera's own fixed pattern (warm pixels and colour blotches that stay put on the sensor) would stack into circular swirls; without darks the stacker measures that pattern from the lights themselves and takes it out of every frame (it says so in the stack log). Dithering, if your telescope offers it, helps even more. Seestar *Stacked_…* files from several nights can be combined on page 2, or a single stack can go straight to page 3.");
sec("10.6", "Comets", ["Comets", "Comet stack", "_Comet master", "_Stars master"]);
p("A comet moves against the stars while you image it, so a normal stack (lined up on the stars) smears it, and a stack lined up on the comet trails the stars. SubZeroAstro does both and joins them. It recognises a comet by its name (*C/2023 A3*, *12P/Pons-Brooks*, *161P Hartley-IRAS*, or any name containing *comet*) - page 2's plan shows ☄ for it. While stacking it finds the comet's head in every frame, works out its motion from the frames' times, makes a **second stack that follows the comet** (the stars, which move through it, are rejected), and saves three masters: the main one with **sharp stars and a sharp comet**, plus **_Comet** (comet only) and **_Stars** (star-aligned) next to it. The log says how far the comet moved.");
p("When the comet barely moved during the session (under 3 pixels - common with short sessions), or it is too faint or too uncertain to follow, only the ordinary stack is saved and the log says why. Long sessions of several frames per minute work best. On page 4 the main master goes through the usual steps; for a comet **Darken the Sky** is recommended, while star steps (*Take Stars Out*, *Shrink Stars*) are optional.");
sec("10.7", "Target types", ["Target types", "Galaxy", "Nebula", "Star cluster", "Planetary nebula"]);
p("From the target's name SubZeroAstro works out what kind of object it is - **galaxy, nebula, planetary nebula, star cluster or comet** - and shows it in the title of page 4. The type only fine-tunes the recommendations; **no step is ever taken away**. Galaxies get *Tame Bright Core* (right after brightening - skip it if the core already shows its detail) and *Local Contrast* recommended, and HII regions in *Add Hydrogen Glow*; other targets get Tame Bright Core recommended only when a burnt-out core is found; star clusters make star removal and shrinking optional (the stars are the subject); comets get a comet stack on page 2. When the name gives nothing away, the usual workflow for your data applies. Tips for each kind of target are in chapter 15.");

sec("10.8", "Step-by-step recipes: how to use them", ["Step-by-step recipes", "Recipes", "Workflow by data type"]);
p("The next sections walk through each kind of data from start to finish, with the choice to pick at every step. They are a **starting point**, not rules: every step shows its versions side by side, so trust your eyes when another one looks better. Pick the section for your data (what you captured with), then check **10.15** for what changes with your kind of target (galaxy, nebula, planetary nebula, cluster).");
sub("Pages 1-3: the same for every recipe");
steps([
  "**Page 1 - Quality and Cull:** load the lights, press *Auto-Flag › Semi-aggressive* (or *Less aggressive* for a short session), look through the flagged frames and reject them. Set the filter name if the files don't carry it.",
  "**Page 2 - Calibrate and Stack:** check every row has ✓ for darks (or bias) and flats - add them with *Add Calibration Folder…* or set your calibration masters folder. Keep **Line up all filters of a target** on, so every filter of one target lands on the same pixels. Press *Check & Stack*.",
  "**Page 3 - Astro Image Processing:** all masters of one target must show as **one row**. Crop inside the area every frame covers (*Auto-crop* does it; a note in the log warns when a hand-drawn crop goes past it), keep plate solving and gradient removal on, **BlurXTerminator** *Sharpen Stars* 0.50 / *Adjust Star Halos* 0.40, **NoiseXTerminator** about 0.6, and **StarXTerminator** on (it makes *Take Stars Out* instant). Press *Run Selected Steps*, then *Continue to Finishing Touches*.",
]);
p("On page 4 the steps below are listed in order in *Your workflow*; ◇ marks optional ones. Each recipe names the version to pick - the ★ the app suggests is usually the same.");

sec("10.9", "Recipe: colour camera, broadband (RGB, no filter or UV/IR cut)", ["OSC recipe", "Broadband recipe", "RGB colour camera"]);
p("Best for galaxies, star clusters and reflection nebulae - all the starlight colours are kept.");
steps([
  "**Balance Colours › Stars white** (neutral). *Stars white, warmer / cooler* only if you prefer that look.",
  "**Remove Green Cast › 25-50 %** - a colour camera leaves a little green; real green almost never exists in space.",
  "**Mote Buster** ◇ if dust donuts show (check the circles, right-click or Control-click any that are not donuts).",
  "**Take Stars Out**, then **Brighten the Image (Stretch) › Medium** (method *Automatic*; *Colour-preserving* keeps star and core colours better).",
  "**Tame Bright Core** - galaxies: *Medium*; other targets only if the centre is burnt out.",
  "**Curves › More contrast** or **Brighter highlights**. Use *Lift faint detail* only for deep data.",
  "**Colour Intensity › Slight** or **Rich** (the sky is protected).",
  "**Adjust Colours** ◇ - a gentle *Boost blues* for galaxy arms or reflection nebulae; *Smooth colour transitions* 0.2-0.3 if the colour looks blotchy.",
  "**Bring Out Hidden Structure › Subtle** or **Moderate** (keep *Protect the sky* about 0.85).",
  "**Brighten Stars Separately › Medium** (*Keep bright stars small* stays on), **Shrink Stars › Moderate**, **Star Colours › Natural**, **Put Stars Back › 80-100 %**.",
  "**Darken the Sky Background › Slightly darker** or **Darker**, then **Final Noise Reduction › Medium**.",
  "**Final Crop**, **Spot Fix** for any specks, **Sign Your Picture**, *Save the Picture…*",
]);

sec("10.10", "Recipe: colour camera with one dual-band filter (HOO)", ["Dual-band step by step", "HOO recipe", "L-eXtreme recipe", "L-Ultimate recipe"]);
p("For emission nebulae with L-eXtreme, L-Ultimate, L-eNhance, Askar D1/ColourMagic and similar: red pixels record Ha, green and blue record OIII.");
steps([
  "**Choose a Colour Palette › HOO** (natural red and teal). Alternatives: *HOO (vivid)*, *Gold & blue* (Hubble-like), *Foraxx (dynamic)*.",
  "**Mote Buster** ◇ if needed, then **Take Stars Out**.",
  "**Brighten the Image (Stretch) › Medium** (stronger is fine with the stars out).",
  "**Remove Green Cast** - *HOO* rarely needs much; *Gold & blue* offers **Gold 100 %**, which turns the green gold instead of deleting it.",
  "**Tame Bright Core** ◇ for a bright centre (Bubble, Orion, Lagoon).",
  "**Curves › More contrast** or **Lift faint detail**.",
  "**Colour Intensity › Rich**.",
  "**Adjust Colours › Blues and reds** separates the teal OIII from the red Ha; *Calm the greens* for leftover green; *Smooth colour transitions* 0.3 if the colour is grainy.",
  "**Bring Out Hidden Structure › Moderate**.",
  "Stars: **Brighten Stars Separately › Medium**, **Shrink Stars › Moderate**, **Star Colours › Natural** (a dual-band filter makes all stars one colour - this gives them back their blue-to-orange range), **Put Stars Back › 80-100 %**.",
  "**Darken the Sky**, **Final Noise Reduction**, crop, sign, save.",
]);
tip("A galaxy through a dual-band filter: pick **As captured** (or *HOO*), keep *Colour Intensity* at *Slight*, and expect strong red HII knots but little blue in the arms - the filter blocks most of the starlight that carries a galaxy's colour. Long integration helps; a few hours of broadband data give the full colour.");

sec("10.11", "Recipe: colour camera with two dual-band filters (SHO)", ["D1 D2 step by step", "Askar D1 D2 recipe", "L-Ultimate L-Synergy recipe"]);
p("Askar D1 + D2, or L-Ultimate + L-Synergy, of the same target: SubZeroAstro builds Ha, OIII and SII and the page 4 log shows the line leakage it took out and how the two OIII estimates were combined.");
steps([
  "Stack each filter on page 2 (one row each); load both masters together on page 3 - **one row**.",
  "**Combine Colour Channels › Hubble (SHO)** (or *Foraxx*, *HOO + SII*).",
  "**Mote Buster** ◇, **Take Stars Out**, **Brighten the Image (Stretch) › Medium**.",
  "**Remove Green Cast › Gold 100 %** - the green is your Ha; *Gold* turns it gold instead of deleting it, which gives the classic gold and blue.",
  "**Tame Bright Core** ◇, **Curves › More contrast**.",
  "**Colour Intensity › Slight** - the SII and OIII from a colour camera are weak, and stronger saturation turns their noise into speckle.",
  "**Adjust Colours**: Blues about 1.2-1.3, *Smooth colour transitions* **0.4-0.6** (this is what keeps the background calm).",
  "**Bring Out Hidden Structure › Subtle**, then the star steps (**Star Colours › Natural**), **Darken the Sky**, **Final Noise Reduction › Medium or Strong**.",
]);

sec("10.12", "Recipe: mono camera, RGB or LRGB", ["Mono LRGB step by step", "LRGB recipe step by step", "Mono RGB recipe"]);
p("Galaxies, star clusters and reflection nebulae with separate R, G, B (and L) filters. L carries the detail, RGB the colour - so RGB can be shorter or binned.");
steps([
  "**Combine Colour Channels › LRGB** (or *RGB* without L).",
  "**Balance Colours › Stars white**, **Remove Green Cast › 25-50 %**.",
  "**Mote Buster** ◇, **Take Stars Out**, **Brighten the Image (Stretch) › Medium** (*Colour-preserving* keeps a galaxy's core colour).",
  "**Tame Bright Core** - galaxies *Medium*; an L filter often needs it.",
  "**Curves › Brighter highlights** or **More contrast**.",
  "**Colour Intensity › Slight** or **Rich**.",
  "**Adjust Colours** ◇ - *Boost blues* 1.2-1.4 for spiral arms, *Smooth colour transitions* about 0.3 (RGB is usually noisier than L).",
  "**Bring Out Hidden Structure › Subtle** or **Moderate** (sky protected).",
  "Star steps (**Star Colours › Natural** or *As they are*), **Darken the Sky › Slightly darker**, **Final Noise Reduction**, crop, sign.",
]);

sec("10.13", "Recipe: mono camera, LRGB + Ha (galaxies)", ["LRGB Ha step by step", "Galaxy with Ha", "HII regions recipe"]);
p("The classic galaxy set: broadband colour with the red hydrogen (HII) knots along the arms added from Ha.");
steps([
  "**Add Hydrogen Glow › HII regions - natural** - the starlight is taken out of the Ha first, so only the compact pink knots are added to red, not a pink wash over the disc. *Strong* if the knots are the show-piece; tick *Show the hydrogen regions* to see what is added.",
  "**Combine Colour Channels › LRGB**.",
  "**Balance Colours › Stars white** (not warmer - neutral keeps the blue arms), **Remove Green Cast › 25-50 %**.",
  "**Take Stars Out**, **Brighten the Image (Stretch) › Medium** or *Gentle*, method *Colour-preserving*.",
  "**Tame Bright Core › Medium**.",
  "**Curves › Brighter highlights** (avoid *Darker background* - it brings out faint background blotches).",
  "**Colour Intensity › Slight** or **Rich**; **Adjust Colours › Boost blues** 1.2-1.4 with *Smooth colour transitions* 0.3.",
  "**Bring Out Hidden Structure › Subtle** or **Moderate** (sky protected).",
  "Star steps, **Darken the Sky › Slightly darker** (keep the faint outer arms), **Final Noise Reduction**, crop, sign.",
]);
tip("**Add Hydrogen Glow comes before Combine Colour Channels** - it works on the separate R and Ha images. Combined without it? The step stays in the list marked ⤼ (passed by): use **Back to Start** (or pick *Start (masters loaded)* under *Show* and press *Go Back to Here*), then Add Hydrogen Glow, then Combine. Also loaded OIII and SII masters of the galaxy? It is still processed as LRGB + Ha; the SHO and HOO palettes stay available under Combine Colour Channels.", "Order matters");
tip("Look for: a yellow-white core, blue arms and pink-red knots. A pink galaxy means the Ha is too strong - go back to *HII regions - natural*. Typical time split: L 40-50 %, R, G and B equal, Ha a few hours.");

sec("10.14", "Recipe: mono camera, narrowband (SHO or HOO)", ["Mono SHO step by step", "Mono HOO step by step", "Narrowband step by step"]);
steps([
  "**Combine Colour Channels › Hubble (SHO)** - or *HOO* with only Ha and OIII, *Foraxx (dynamic)*, *HOO + SII*, *HSO*.",
  "**Mote Buster** ◇ - with filters, each has its own dust: bright rings or coloured halos are fixed per colour.",
  "**Take Stars Out**, **Brighten the Image (Stretch) › Medium**.",
  "**Remove Green Cast › Gold 100 %** for SHO (the classic gold and blue); plain removal for HOO.",
  "**Tame Bright Core** ◇ for a bright centre, **Curves › More contrast**.",
  "**Colour Intensity › Rich**; **Adjust Colours › Blues and reds** to separate the gold and blue; *Smooth colour transitions* 0.2-0.3.",
  "**Bring Out Hidden Structure › Moderate**.",
  "Star steps - **Star Colours › Natural** is recommended (narrowband makes stars magenta or one colour), **Put Stars Back › 80-100 %**.",
  "**Darken the Sky**, **Final Noise Reduction**, crop, sign.",
]);

sec("10.15", "Adjusting any recipe for your target", ["Recipe by target type", "Galaxy recipe", "Nebula recipe", "Planetary recipe", "Cluster recipe"]);
table(["Target", "What changes"], [
  ["Galaxy", "*Tame Bright Core* (Medium) after brightening; *Colour-preserving* brightening; *Curves › Brighter highlights*; *Darken the Sky › Slightly darker* only (keep the halo and outer arms); a little *Boost blues*. See 15.2."],
  ["Emission nebula", "Stars out before brightening, then brighten strongly; *Bring Out Hidden Structure › Moderate*; *Adjust Colours › Blues and reds*; *Tame Bright Core* only for a bright centre. See 15.3."],
  ["Planetary nebula", "Small and bright: *HOO* (or SHO), *Brighten › Gentle* with *Colour-preserving*; *Tame Bright Core* comes after *Curves* and tames the bright ring to show the inner structure; skip *Remove Green Cast* if the OIII looks green-teal; *Smooth colour transitions* is on; crop tightly. See 15.4."],
  ["Dark nebula / faint dust", "*Curves › Lift faint detail*, a light *Bring Out Hidden Structure (Subtle)*, no strong sky darkening - the dust *is* the faint part. See 15.5."],
  ["Star cluster / reflection nebula", "Stars are the subject: *Take Stars Out* and *Shrink Stars* are optional, *Star Colours › Natural, vivid*, *Put Stars Back 100 %*; reflection nebulae: *Boost blues*. See 15.6."],
], [2200, 7160]);

ch(11, "Troubleshooting and questions");
sec("11.1", "Common questions", ["Troubleshooting", "FAQ"]);
table(["Problem", "What to do"], [
  ["Windows: \"Windows protected your PC\"", "Click **More info › Run anyway** (3.1)."],
  ["Mac: \"can't be opened\" or \"is damaged\"", "Right-click › Open, or the Terminal command in 3.2."],
  ["Frames show **No filter**", "Set the filter on page 1 (*Filter used* or *Frame › Set Filter Name…*). Needed to match flats and to offer the right palettes."],
  ["Flats column is ✗ red", "No flats for that filter/rig. Change the row's Filter to match your flats, or add the flats folder."],
  ["Bright corners and dark rings", "Flats calibrated without bias/dark-flats, or flats that don't match. Check the ⚠ note on page 2 and add a bias or dark-flats (7.2)."],
  ["Filters of one target in separate rows on page 3", "Check that the target names match (OBJECT). Masters stacked in separate runs are lined up automatically."],
  ["No *Combine Colour Channels* on page 4", "The masters were not loaded together, or a filter name was not recognised - the log lists the filters it found."],
  ["Plate solve failed", "SubZeroAstro already retries with other field sizes (wrong binning or a reducer in the header are the usual cause; when the header has no focal length at all - Telescope Live files say 0 - typical fields from 0.25° to 4.5° are tried, and the one that works is reused for the other masters). If it still fails, read the log line: for small fields (under about 0.5°, e.g. galaxies with long focal lengths) ASTAP needs the **D50** or **D80** star database; with no pointing in the header tick **Search the whole sky**. A failed solve only means the picture has no sky coordinates - every other step works as usual."],
  ["Tools show *not found*", "Click **Browse…** and point at the program, or use **Settings › Your Tools…** (4.1)."],
  ["Amber banner on page 1", "A helper program is missing or not licensed. **Set Up…** fixes it; **Don't Remind Me** if you don't use it (4.1)."],
  ["An XTerminator says *not licensed*", "Click **RC-Astro Licences…** on page 3 and activate with the code from your rc-astro.com account (4.1)."],
  ["Frames skipped: *could not match the star field*", "Usually clouds, trailing or a badly out-of-focus frame. Frames saved mirror-image (some capture programs do this) are recognised and lined up automatically."],
  ["XTerminators very slow", "Choose your graphics card under **RC-Astro device** (4.1)."],
  ["Slow on a network drive", "Use *Star measurement: Fast*; hide the image (Ctrl+I); results are cached for next time."],
  ["Something went wrong mid-run", "Restart: SubZeroAstro offers to resume from its saved progress."],
  ["Coloured specks (single red, green or blue pixels) all over the picture", "Hot pixels that survived stacking - they do when the frames were not dithered (common with remote-observatory data), because they sit on the same pixel in every frame. Since 0.9.54 **Remove hot pixels** (page 2, on by default) replaces them in every frame before stacking; re-stack such data."],
  ["Doubled stars with frames from a remote observatory (or any camera with many hot pixels)", "Fixed in 0.9.51. Calibrated frames that were not dithered can keep hundreds of hot pixels, and those sit in the same place on the sensor in every frame. Before 0.9.51 most of the \"stars\" found could be hot pixels, so frames from different nights could be lined up on the sensor instead of the sky - giving doubled stars. Now single-pixel spikes are ignored when stars are matched (a real star always spills light into its neighbours). Re-stack such data with 0.9.51 or later."],
  ["A frame shows *This file is damaged: its picture data was converted to text*", "The file was downloaded or copied in a way that treated it as text: every byte the text conversion didn't understand was replaced, so the file is about twice its proper size (a 3056×3056 16-bit frame should be about 18.7 MB) and the picture is noise - while the header still reads fine. Such frames can never be aligned or stacked. Download them again (another browser, or as a zip) and check the file size."],
  ["Half the frames show *unreadable*, names starting with `._`", "Those are hidden macOS copies (AppleDouble files) that a Mac writes next to every file on a NAS or external drive - not pictures. Version 0.9.31 and later ignore them; with an older version, delete the `._` files or turn off *Include subfolders*."],
], [3300, 6060]);
sec("11.2", "Getting help and reporting problems", ["Support", "Reporting problems", "Zip My Logs", "Logs", "Self-test", "Updates", "Check for Updates", "Update Now"]);
p("**Zip My Logs**: the button at the top left of every page (also **Help › Zip My Logs**) puts the log of every open page, the stack logs of your masters, page 3's processing report and a short note about your computer (SubZeroAstro version, system, screen, which helper programs were found) into one zip on your Desktop and shows it - text only, no pictures. Attach that zip to your message: it is the quickest way to get help.");
p("**After a crash or a frozen computer**: every session also writes its log to disk as it goes (Windows: *%LOCALAPPDATA%\\SubZeroAstro\\Logs*, Mac: *~/Library/Logs/SubZeroAstro*), with a note every 30 seconds of how much memory the app uses and how much the computer has left. Start SubZeroAstro again and press **Zip My Logs**: the zip includes the last sessions, and one that ended badly is marked *ENDED_BADLY*.");
p("**Updates**: when a newer SubZeroAstro is published, a slim blue bar at the top of page 1 says so (checked quietly a few seconds after start-up). **What's New** shows what changed, **Update Now** downloads the new version, closes SubZeroAstro, puts the new files in place and starts it again - your settings and projects are kept - and **Skip This Version** hides the bar until the next one. Let any stacking or processing finish first. **Help › Check for Updates…** checks right away; untick **Help › Check for Updates at Startup** to stop the automatic check. If SubZeroAstro is in a folder it can't write to (or runs from Python), the button says **Download…** and opens the download page instead; hover over it to see why. On a Mac, keep SubZeroAstro.app in **Applications** (or another folder of yours) - an app opened straight from the Downloads folder runs from a hidden read-only copy and can't update itself.");
p("Note the version (in small print under the name at the top of every page, or **Help › About**) and take a screenshot of the page and its log. Problems and ideas can be reported on the GitHub page (**github.com/bapoteet/SubZeroAstro**, *Issues*). Advanced: `SubZeroAstro --selftest` runs the built-in self-test.");

ch(12, "Keyboard shortcuts");
sec("12.1", "Page 1 – Quality and Cull", ["Keyboard shortcuts", "Shortcuts"]);
table(["Keys", "Action"], [
  ["↑ ↓ ← →", "Previous / next frame"],
  ["Shift + arrows, Shift / Ctrl + click", "Select several frames"],
  ["X or Delete", "Reject / restore the selected frame(s)"],
  ["F", "Flag / unflag"],
  ["Shift+X", "Reject all flagged frames"],
  ["D", "Clear all flags"],
  ["Ctrl+F", "Auto-Flag…"],
  ["Ctrl+Z", "Undo"],
  ["Ctrl+0  /  1  /  double-click", "Fit to window / 100% / toggle"],
  ["Mouse wheel, drag", "Zoom, pan"],
  ["Ctrl+O", "Load Light Frame Data"],
  ["Ctrl+Shift+O", "Open one folder"],
  ["Ctrl+N", "Start over – load new data"],
  ["Ctrl+H", "FITS header window"],
  ["Ctrl+I", "Show / hide the image"],
  ["Ctrl+E", "Save a text list of kept frames"],
  ["F5", "Reload folder"],
], [3600, 5760]);
sec("12.2", "Page 4 – Finishing Touches");
table(["Keys", "Action"], [["Ctrl+Z", "Undo the last step"], ["Mouse wheel (Spot Fix)", "Spot size"], ["Right-click (Spot Fix)", "Remove the last spot"]], [3600, 5760]);

ch(13, "Files and folders SubZeroAstro creates");
sec("13.1", "Next to your images", ["REJECTED folder", "Prepared folder", "Finished folder", "Output files"]);
table(["Where", "What"], [
  ["`<lights folder>\\REJECTED\\`", "Rejected sub-frames (moved, never deleted)"],
  ["`<calibration folder>\\<rig>\\Masters\\…`", "Calibration masters (bias, darks, flats per filter)"],
  ["`<master lights folder>\\<target>\\`", "Master lights (`…_Master_<dates>.fits/.xisf`) and their `_stacklog.txt`"],
  ["`Prepared\\` (or your chosen folder)`\\<target>\\`", "Page 3 results: `…_prepared.xisf`, `…_prepared_stars.xisf`, quick-look JPG/TIFF, PixInsight next steps"],
  ["`Finished\\`, `PixInsight\\`, `AstroBin\\`", "Page 4 saves, PixInsight hand-overs and AstroBin uploads"],
], [4000, 5360]);
sec("13.2", "On your computer", ["Settings location", "Saved progress location", "Cache"]);
table(["What", "Windows", "Mac"], [
  ["Saved progress", "`%LOCALAPPDATA%\\SubZeroAstro\\checkpoints`", "`~/Library/Application Support/SubZeroAstro`"],
  ["Measurement cache", "`%LOCALAPPDATA%\\FITSBlasterWin\\cache`", "`~/Library/Caches/SubZeroAstro`"],
  ["Settings", "Windows registry (SubZeroAstro settings)", "`~/Library/Preferences`"],
], [2200, 3600, 3560]);

sec("13.3", "Organize My Astro Files", ["Organize My Astro Files", "Organize", "Library layout", "Darks library", "NAS", "Undo Last Run", "Duplicates"]);
p("**The first time** you open it, a note asks you to try it on one or two folders with **Copy** first, look at the result, then **Undo Last Run**, switch to **Move** and run it again - click it through (tick *Don't show this again* once you know the routine). The first Move asks once more.");
p("**File › Organize My Astro Files…** (page 1) sorts a folder full of FITS and XISF files - subs and masters, lights, flats, dark flats, darks and bias, from any number of nights and targets - into one tidy library. It reads each file's own header (IMAGETYP, OBJECT, FILTER, DATE-OBS, camera, gain, temperature), so the file names and the current mess don't matter. Pick the **folder to organize** (a whole drive or a **NAS** share works too - on Windows a mapped drive or `\\\\nas\\astro`, on a Mac the share as it appears under /Volumes) and where the organized files go (the same folder by default).");
p("Everything the organizer makes goes into one folder, **Organized Astro Files**, inside the folder you choose under *Put the organized files in* - so you can always see where your sorted files are. Use the same one every time: organize one rig folder now and the whole drive later, and the files already sorted show as *already in the right place*. The undo logs are kept in it too.");
p("**Folders / files to organize**: Browse… picks one or several folders at once (Shift-click a run, Ctrl/⌘-click separate ones), Add Files… picks single files. The buttons show where you are: a step you have done gets a ✓ and turns grey, the next one is highlighted ▶, and the window says what you did last time. Pick the **Layout** in the window (remembered). **By rig** (the default) starts with two big folders - **Light Frames** and **Calibration Frames** - and under each, one folder per rig (a telescope and the camera that lives on it). Calibration then goes by frame type, then camera:");
table(["Files", "Go to (By rig)"], [
  ["Lights", "`Light Frames/<rig>/<target>/<night>/<filter>/` (rejected ones keep their `REJECTED` folder)"],
  ["Master lights, finished pictures", "`Light Frames/<rig>/<target>/Masters/` with their star images and stack logs"],
  ["Flats / dark flats", "`Calibration Frames/<rig>/Flats/<camera>/<night>/<filter>/` and `Calibration Frames/<rig>/Dark flats/<camera>/<night>/<filter>/`"],
  ["Darks", "`Calibration Frames/<rig>/Darks/<camera>_G<gain>_O<offset>_<temp>C/<exposure>s/`"],
  ["Bias", "`Calibration Frames/<rig>/Bias/<camera>_G<gain>_O<offset>_<temp>C/`"],
  ["Calibration masters", "`Calibration Frames/<rig>/Masters/<camera settings>/` (master flats in `Flats <night>/<filter>`)"],
  ["No header to go by", "`Unsorted/`"],
], [2600, 6760]);
img("organize_routing.png", "Where each file goes (By rig): lights by rig, target and night (purple); each rig's own calibration (teal)", 480);
img("organize_tree.png", "An organized library, By rig (example: two telescopes with the same camera model)", 480);
p("**By target** puts all of one object's data together instead: `Targets/<target>/<night>_<rig>/LIGHT/<filter>/` and `…/FLAT/<filter>/`, `Targets/<target>/Masters/`, with darks and bias in `Darks library/<rig>/…` and `Bias library/<rig>/…`. Flats join the target shot that night with the same telescope - including targets organized in an earlier run, so a folder of calibration frames only can be sorted on its own; otherwise they go to `Flats/<scope>/<night>/<filter>/`.");
bullets([
  "**Your Rigs… comes first** (once): one row per rig - its short name (*SQA60*, *RASA8* …), the focal length it writes in your FITS headers, and the camera only if two rigs have about the same focal length. Files are matched to a rig by focal length within ±3 % (adjustable in Your Rigs), because the header value drifts a little from night to night; when a reading is within reach of two rigs (a 506 mm and a 530 mm scope), the clearly nearer one wins and a reading right between them is asked about. A scope used with and without a reducer is one rig with two focal lengths (*360, 290*). **Find My Rigs in My Files** reads a sample of headers and adds a row for each focal length none of your rigs matches, with a name taken from your folders; **Add New Rig** when you get another one. Remembered, and Calibrate and Stack uses the same names.",
  "**Safe to Delete?…** answers the question before you clear out the old folders: it checks every file still in the folders you organized - is there a byte-for-byte identical copy in Organized Astro Files? The verdict is **✓ Safe to delete** or **✗ Not yet** (with the files that exist only there), plus PixInsight / Siril work copies, files in saved projects and non-astro files (JPG, notes, logs) listed for you to look at. It only reads; it never deletes anything.",
  "**What stays behind, and why**: after Step 3 the list shows every file left where it was, with the reason - it needs a rig (Your Rigs… / Which Rig?…), it is a duplicate of a file that was organized (Step 2 deletes it), it could not be read, or a file of that name with different contents is already in the organized folder. Folders not looked into on purpose (saved projects, PixInsight / Siril work folders such as *calibrated* or *registered*, hidden folders) are named too. Run Step 1 again after fixing the reason and those files go too.",
  "**Nothing is filed under a guess.** A file whose focal length matches none of your rigs stays where it is and the plan says so (add the rig in Your Rigs… and the plan updates at once). When a focal length fits two rigs - or the header has none - **Which Rig?…** asks you, folder by folder, and remembers your answer for that folder.",
  "**The night** is the evening date at your site (from the header's longitude), so a session that runs past midnight stays together.",
  "**Step 1 – Look Through the Files** only reads - it shows how many files go where, and a list of each file, where it goes and why. Nothing changes until you press **Step 3 – Organize** (greyed out until Step 1 has run). Do **Step 2 – Duplicates…** in between when Step 1 finds any.",
  "**Copy** (the default) leaves the originals; **Move** moves them and removes the empty folders left behind. A copy is checked before a move removes the original.",
  "**Duplicates** (the same capture twice, e.g. *L_001 copy.fits*) are left where they are and flagged. Press **Step 2 – Duplicates…** (before Step 3 – Organize) to see each one next to the copy that is kept, tick the ones to deal with, then **Keep Them**, **Move to a Duplicates Folder** (set aside; Undo Last Run puts them back) or **Delete Ticked Duplicates…**. Delete compares every byte first and only deletes an exact copy - a file with the same capture time but different contents is kept for you to look at. Files that can't be read are left where they are and listed.",
  "**Saved projects** (page 4), files already in the right place, and the in-between folders other programs make (*calibrated*, *registered*, *debayered*, *process* … from PixInsight WBPP, Siril or APP) are left alone, so running it again on the same library only tidies what is new.",
  "**Move on the same drive is instant** - the files only get a new folder, nothing is copied - so organizing a big drive in place takes about as long as reading the headers. Copy, or moving to another drive, copies every byte.",
  "**Undo Last Run** puts every moved file back (or removes the copies). Each run's log is kept in `_SubZeroAstro_organize` inside the library.",
  "A big NAS takes a while - reading headers over the network is the slow part. Stop is safe: what was done is in the log and can be undone.",
]);

ch(14, "Privacy, licence and credits");
sec("14.1", "Privacy", ["Privacy"]);
p("SubZeroAstro does not send any information anywhere unless you ask it to. It reads and writes images only where you point it. **Share to AstroBin…** only opens AstroBin's page in your browser. Companion programs (ASTAP, GraXpert, RC-Astro) are separate programs with their own terms.");
sec("14.2", "Licence and credits", ["Licence", "MIT License", "Credits"]);
p("SubZeroAstro is free software under the **MIT License**. Created by **Brian A. Poteet, DVM**, built with Claude (Anthropic). The culling view was inspired by FITS Blaster for macOS by Tom van Peer. PixInsight, BlurXTerminator, NoiseXTerminator, StarXTerminator, GraXpert, ASTAP, AstroBin and the telescope and filter names mentioned are trademarks of their respective owners. The deep-sky object list of Tonight's Target Picker comes from N.I.N.A. - Nighttime Imaging 'N' Astronomy (github.com/isbeorn/nina), used under the Mozilla Public License 2.0.");

// ------------------------------------------------------------------ index (own two-column section)
ch(15, "Processing Techniques and Tips");
p("Every target type has its own small tricks. These are the ones that work best with SubZeroAstro - short points to try, gathered from real data (yours included). The names in *italics* are the versions and sliders on page 4. Nothing here is a rule: if a picture looks right to you, it is right.");

sec("15.1", "Tips for every target", ["Processing tips", "Headroom", "Colour headroom"]);
bullets([
  "**Follow the recommended order.** The card at the top of page 4 knows what your data needs; skipping ahead (Curves before Tame Bright Core, say) makes later steps work harder.",
  "**Colour needs room below white.** Anything brightened close to white cannot show colour - Saturation and Adjust Colours only tint its edges. Lower the brightness first (*Protect bright areas* in Brighten the Image (Stretch), or Tame Bright Core); the colour then comes back on its own.",
  "**Small steps, then Do It Again.** A gentle version run twice usually looks more natural than the strongest version once - and each round can be undone on its own.",
  "**Judge at full size.** *Full Window*, the mouse wheel to zoom, and *Hold for Before* show what a step really does. The little versions are for choosing a direction, not for judging detail.",
  "**Good data beats any processing.** Flats every night (a flat panel at the end of the session is ideal), a dark and bias library at one fixed camera temperature (0 °C works well for modern CMOS), and as many hours as you can: noise halves for every four times the exposure.",
]);

sec("15.2", "Galaxies", ["Galaxy processing", "Galaxies"]);
bullets([
  "**Tame Bright Core** is recommended for every galaxy, right after Brighten the Image (Stretch). Start with *Medium*; use *Strong* or *Maximum* for a very bright nucleus. Drag *Core size* to the left when the bright area is larger than the tint of *Show what will change*.",
  "**Bring Out Hidden Structure** (*Moderate* or *Strong*) after Curves shows the dust lanes and spiral arms.",
  "**HII regions:** with a mono RGB set plus Ha, *Add Hydrogen Glow* puts the red knots along the arms; with a colour camera, a few hours through a dual-band filter can be added the same way.",
  "**Filters:** broadband (or a mild light-pollution filter) gives the brightest arms and most natural colours. A dual-band filter (L-eNhance, L-Ultimate) lets through only a little of a galaxy's starlight, but with long integration it works well and makes the red HII regions stand out - M31 with a 55 mm refractor, an ASI2600MC and an L-Ultimate over about 20 hours shows the disc, the dust lanes and the HII regions clearly.",
  "**Small galaxies:** crop tightly at the end (*Final Crop*), keep the stars in check (*Shrink Stars Moderate*, *Put Stars Back 80-100%*) so the galaxy stays the subject.",
  "**Don't darken the sky too much** - *Slightly darker* keeps the faint outer arms and halo that make a galaxy look real.",
  "**Warm core, blue arms:** a spiral's bulge is old, yellowish starlight and its arms young, blue stars. If the disc looks grey, a gentle *Adjust Colours › Boost reds* (with *Protect the sky* up) warms the core, and *Boost blues* the arms.",
  "**The faint outer disc:** a gentle *Curves › Lift faint detail* before *Darken the Sky Background* shows more of the outer arms and halo, when the data is deep enough.",
]);

sec("15.3", "Emission nebulae", ["Emission nebula processing", "Nebulae", "HOO", "Foraxx"]);
bullets([
  "**Palettes for a colour camera with a dual-band filter:** *HOO* (Ha red, OIII teal - natural), *HOO (vivid)*, *Gold & blue*, or *Foraxx (dynamic)*. With a mono camera, *Hubble (SHO)* or *HOO + SII*.",
  "**Take the stars out before brightening** - the nebula can then be brightened strongly without bloating the stars, and they go back at the end.",
  "**Separate Ha and OIII:** *Adjust Colours › Blues and reds* makes the blue OIII and red Ha stand apart; *Calm the greens* for leftover green.",
  "**Bright centres** (Bubble, Orion, Lagoon): *Tame Bright Core* right after Brighten the Image (Stretch).",
  "**Targets that fill the frame** (Veil, North America): a SkyFlat made from that target's own lights can leave faint dark copies of the nebula - use real flats, or untick *SkyFlats (Mote Buster)* and let gradient removal and Mote Buster deal with vignetting and dust.",
  "**Stars:** *Star Colours › Natural* (narrowband paints stars one colour) and *Put Stars Back 80-100%* keep the nebula the star of the picture.",
]);

sec("15.4", "Planetary nebulae", ["Planetary nebula processing", "Planetary nebulae", "M57", "Ring Nebula", "M27"]);
bullets([
  "**They are tiny and very bright.** Brighten the Image (Stretch) sets the brightness for the faint background, which pushes the bright ring or shell to nearly white - that is why it looks \"hot\". Use *Gentle*, and keep the *Method* on **Colour-preserving** (the default for planetary nebulae): the *Hyperbolic (GHS)* method brightens each colour separately and turns a bright planetary grey - no colour step can bring that back.",
  "**Tame the ring, not the centre.** A planetary's bright light is in its ring or shell; the middle is faint and see-through. For a planetary nebula *Tame Bright Core* (recommended, right **after Curves** - the workflow puts it there, because Curves' contrast would make the ring white again) works as *Tame the Ring*: inside the mask the nebula is brightened again, gently, from the linear data - the brightest part of the ring goes to the *Ring brightness* you pick (Gentle 85% … Maximum 55%), everything else follows the nebula's real brightness, so ring and inside separate again and both leave white behind. The outer halo is left as it is. Tick *Show what will change* to see the area - a light green wash with a yellow line at its edge, which should lie outside the ring; drag *Area* to the left if it doesn't. The result shows through the wash; untick the box for a clean look. *Medium* or *Strong* is a good start. Don't brighten or add contrast after it.",
  "**Star removal and small nebulae:** StarXTerminator can take a small, bright planetary for a star and put part of it in the stars image - where no nebula step reaches it, and *Put Stars Back* makes it white again. For a planetary nebula SubZeroAstro moves anything in the stars image that is much wider than a star, near the middle of the picture, back to the nebula when the stars are taken out.",
  "**Palette HOO:** Ha gives the red outer ring, OIII the teal-blue inside - the classic Ring Nebula look. *As captured* tends to look pale.",
  "**Colour:** *Saturation Vivid* or *Rich*, then *Adjust Colours › Blues and reds* (lower *Reds* a little for a bluer centre). If the colours change in hard bands around the ring, raise *Smooth colour transitions* - it blends the colours without softening the detail.",
  "**Faint outer halo:** after taming, *Curves › Lift faint detail* can show the dim shell around the bright ring (M57 has one).",
  "**Crop tightly:** at 900 mm M57 is only about 100 pixels across - *Final Crop* makes it fill the picture.",
  "**At the telescope:** shorter subs (60-120 s) keep the brightest part from burning out in the masters; collect plenty of them.",
]);
tip("A recipe that works for M57 (colour camera + L-Ultimate, 3.6 h): *HOO* palette · *Take Stars Out* · *Brighten the Image (Stretch): Gentle* (Colour-preserving) · skip *Remove Green Cast* · *Curves: Lift faint detail* · *Tame Bright Core: Medium* · *Saturation: Vivid* · *Adjust Colours: Blues and reds* with Blues about 1.3, Reds 0.85 and *Smooth colour transitions* 0.6-1.0 · the star steps · *Put Stars Back* · a tight *Final Crop*. The result: a blue centre, a gold-to-red ring and the faint outer halo.", "Worked example");

sec("15.5", "Dark nebulae and faint dust", ["Dark nebula processing", "Dark nebulae", "IFN", "Barnard"]);
bullets([
  "**The darkness is the subject.** Leave *Darken the Sky Background* off (or *Slightly darker*): darkening the sky erases the faint glow the dark clouds stand out against.",
  "**Broadband, dark skies and many hours** - dust shows up in broadband light only; dual-band filters hide it.",
  "**Gradients matter most here:** check the GraXpert result on page 3; a raised *GraXpert smoothing* keeps it from mistaking large faint dust for a gradient.",
  "**Lift, don't stretch:** *Curves › Lift faint detail* and a light *Bring Out Hidden Structure* (*Subtle*).",
  "**Tame the star field:** *Shrink Stars Strong* and *Put Stars Back 60-80%* so the dust shapes are not lost among thousands of stars.",
]);

sec("15.6", "Reflection nebulae and star clusters", ["Reflection nebula processing", "Star cluster processing", "Globular clusters", "M45", "M13"]);
bullets([
  "**Reflection nebulae** (Pleiades, Iris, M78) shine by reflected starlight: broadband only, no dual-band filter. *Balance Colours › Neutral sky only* keeps their blue.",
  "**Bright stars inside a nebula:** BlurXTerminator's *Adjust Star Halos* on page 3, and *Shrink Stars* with *Halo reduction*, calm the glow around them.",
  "**Star clusters:** the stars *are* the subject - taking them out is optional. *Star Colours › Natural, vivid* shows the mix of blue and orange stars.",
  "**Globular clusters** (M13, M92): *Brighten Gentle* keeps the core resolved; *Tame Bright Core* if the centre turns into a white disc.",
]);

sec("15.7", "Comets", ["Comet processing", "Orbitals", "Comet LRGB", "Larson-Sekanina", "Comet tail"]);
p("SubZeroAstro recognises a comet by its name (*C/2023 A3*, *12P/Pons-Brooks*, or any name with *comet* in it) and handles it from stacking to the finished picture:");
bullets([
  "**Capturing:** many short subs - the comet moves. With N.I.N.A.'s **Orbitals** plugin the mount follows the comet; a typical plan cycles the filters, e.g. 6 x 60 s each of R, G and B, then 6 x 30 s of L, again and again. Keep the target name the comet's name so SubZeroAstro knows it is a comet. Take flats for every filter as usual.",
  "**Page 2 (stacking):** each filter is stacked on the stars and then again following the comet (10.6). The main master joins the sharp stars and the sharp comet; *_Comet* and *_Stars* masters are saved next to it.",
  "**Page 4 (combining):** when the filters were shot one after another, the comet sits in a slightly different place in each filter's master. SubZeroAstro moves just the comet (head and tail) in each filter onto the comet in the L (or first) master, so the comet's colours line up while the stars stay where they are. The load note says how far each filter's comet was moved.",
  "**Page 4 (steps):** after **Brighten the Image (Stretch)**, two comet steps are offered: **Bring Out the Tail** (take away part of the round coma so the tail and streamers show) and **Reveal Comet Jets** (Larson-Sekanina: jets and shells near the nucleus; *Jets map only* makes the grey study picture to save separately). **Darken the Sky Background** is recommended so the faint tail stands out; star removal and shrinking are optional.",
  "**Tips:** take the stars out before Reveal Comet Jets (stars turned around the nucleus leave little arcs). Use Go Back to Here to compare versions. A comet that barely moves during the session is simply stacked on the stars.",
]);

sec("15.8", "Moon and planets (coming)", ["Planetary imaging", "Moon processing", "Lucky imaging"]);
p("A page for video of the Moon and planets (lucky imaging: grade thousands of frames, stack the best, sharpen) is on the way. For now: video (SER/AVI) with very short exposures for planets and Moon close-ups; single frames, stacked, for a full Moon; the best 10-25% of the frames usually give the sharpest result.");

sec("15.9", "Questions and fixes", ["FAQ", "Bloated stars", "Fuzzy stars", "Washed-out colours", "Burnt-out core"]);
table(["Problem", "What to try"], [
  ["Bright stars turn into big soft discs", "*Brighten Stars Separately* keeps bright stars their real size (fine-tune *Keep bright stars small*, about 0.85): one brightening curve for everything would lift a bright star's faint outer glow as much as a faint star's core. Turn it down toward 0 only if you want the big-glow look."],
  ["My stars look bloated and have fuzzy edges", "Page 3: BlurXTerminator *Sharpen Stars* about 0.5 and *Adjust Star Halos* 0.3-0.5. Page 4: *Brighten Stars Separately › Subtle*, *Put Stars Back 80-100%* (not 120%), *Shrink Stars* with *Halo reduction* - and *Do It Again* if needed. If the stars are large on page 1 too (FWHM), it is focus or seeing."],
  ["The core of my galaxy or nebula is a white blob", "*Tame Bright Core* right after *Brighten the Image (Stretch)* - *Strong* or *Maximum*, *Core size* to the left. Raise *Protect bright areas* in Brighten the Image (Stretch)."],
  ["My colours look washed out, or the object stays white", "Colour needs room below white: lower the brightness first (above), then *Saturation Vivid* and *Adjust Colours*. Saturation on a white area does nothing."],
  ["The background is green or magenta", "*Remove Green Cast*; *Balance Colours › Neutral sky only*; *Adjust Colours › Calm the greens*."],
  ["The background is blotchy or noisy", "More hours of data helps most. Then *Final Noise Reduction* (*Medium*), a gentler *Brighten*, and *Darken the Sky Background*."],
  ["Dark rings (dust donuts)", "Flats taken the same night fix them at the source. Afterwards: *Mote Buster* (9.4) for donuts, *Spot Fix* for single blemishes."],
  ["A faint dark copy of the nebula, turned or shifted", "A SkyFlat made from a target that fills the frame. Use real flats, or stack again with *SkyFlats (Mote Buster)* unticked (15.3)."],
  ["Double stars, or coloured edges on stars near the corners", "Masters from different nights or setups that did not line up - the page 4 log says which master was left out. Stack those nights together on page 2 if you can."],
  ["The faint outer parts disappeared", "The sky was darkened too much, or noise reduction was too strong. Use *Slightly darker*, *Curves › Lift faint detail*, a lighter *Final Noise Reduction*."],
  ["Red, purple, blue or green blotches in the sky", "Colour noise from the weaker colours or filters (OIII, SII; a short RGB set) showing in the sky. *Darken the Sky Background* has **Calm sky colours** (fine-tune; on by default, 0.6 for narrowband, 0.5 otherwise): across the whole sky - everything near the sky's own level - it smooths the colour much wider than the blotches and pulls it toward grey, without touching the brightness or the galaxy's / nebula's colours. Raise it toward 1.0 for a neutral grey sky. When the faint areas are faint nebula rather than empty sky (IC 1396, the Veil), use **Adjust Colours › Calm blues and purples** - it calms those hues everywhere, the gold and red untouched. Also: *Adjust Colours › Smooth colour transitions* 0.4-0.6, *Colour Intensity › Slight*, and more time on OIII and SII."],
  ["The background shows blotches or a moire-like pattern", "Faint leftover noise at a large scale (often well under 1 % of the sky brightness), made visible by a strong stretch. *Bring Out Hidden Structure* protects the sky (keep *Protect the sky* about 0.85); also try *Curves* without *Darker background*, a stronger *Final Noise Reduction*, and dithering every 2-3 frames when you capture."],
  ["The picture looks over-sharpened or crunchy", "Use a lower *Bring Out Hidden Structure* version (or skip it), and *Sharpen Nonstellar* about 0.3-0.5 on page 3."],
  ["Satellite trails", "Many frames and sigma clipping on page 2 remove most of them; *Spot Fix* (with its stars kept) cleans up what is left."],
  ["Dew on the camera window", "Put the imaging train together indoors, add a dew strip around the filter drawer or spacer, and keep a desiccant pack in the train; a fixed 0 °C camera temperature also helps."],
], [3300, 6060]);

ch(16, "Index");
const indexBody = [];
const IDX_TAB = 4200;
const SPECIAL = { "Stop Here – Continue in PixInsight": 1 };
const terms = Object.keys(index).sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }));
let letter = "";
for (const t of terms) {
  const L = t[0].toUpperCase();
  if (L !== letter) {
    letter = L;
    indexBody.push(new Paragraph({ spacing: { before: 160, after: 40 }, keepNext: true,
      children: [new TextRun({ text: L, bold: true, color: ACCENT, size: 24 })] }));
  }
  const nums = [...index[t]].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const kids = [new TextRun({ text: t, size: 19 }), new TextRun({ children: [new Tab()], size: 19 })];
  nums.forEach((n, i) => {
    if (i) kids.push(new TextRun({ text: ", " }));
    kids.push(new InternalHyperlink({ anchor: anchorOf(n), children: [new TextRun({ text: n, style: "Hyperlink", size: 19 })] }));
  });
  indexBody.push(new Paragraph({ spacing: { after: 10 }, children: kids,
    tabStops: [{ type: TabStopType.RIGHT, position: IDX_TAB, leader: LeaderType.DOT }] }));
}

// ------------------------------------------------------------------ cover + contents
const cover = [
  new Paragraph({ spacing: { before: 400, after: 160 }, alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(IMG, "logo_badge.png")),
      transformation: { width: 210, height: 210 }, altText: { title: "SubZeroAstro logo", description: "The SubZeroAstro badge: a telescope under a bright star, a dog and a cat beside it, US and Texas flags", name: "logo" } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SubZeroAstro", bold: true, size: 72, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "From subs to masters – zero hassle.", italics: true, size: 30, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 120 },
    border: { top: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 12 } },
    children: [new TextRun({ text: "User Manual", bold: true, size: 44 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A beginner's guide to culling, stacking, processing and finishing your astrophotos", size: 24, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(IMG, "0_splash_4.00s.png")),
      transformation: { width: 390, height: 220 }, altText: { title: "SubZeroAstro", description: "Start-up screen", name: "cover" } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360 },
    children: [new TextRun({ text: `Version ${VERSION}  ·  Windows and Mac (Apple silicon)`, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "by Brian A. Poteet, DVM  ·  October 2026", size: 22, color: GREY })] }),
];
// ------------------------------------------------------------------ highlights (after the cover, before the contents)
const HIGHLIGHTS = [
  ["Across the whole app", [
    ["Ready-made data recognised", "Telescope Live (PROC = True), MaxIm DL (CALSTAT), PixInsight and Siril calibration notes, and _cal / _c file names mark lights as already calibrated: no double calibration, no SkyFlat, no false warnings", "0.9.49"],
    ["Damaged-file detection", "Spots FITS files whose picture data was turned into text by a bad download or copy, and says so instead of stacking noise", "0.9.50"],
    ["Clear Everything – Start From Scratch", "One menu item, like PixInsight's Reset Engine: forgets measurements and progress, moves old masters aside, optionally resets settings (keeps folders and tools)", "0.9.52"],
    ["Crash recovery", "Every page saves as it goes; after a crash or restart the app offers to bring you back to the same page", "0.6"],
    ["Stop Here – Continue in PixInsight", "From pages 2, 3 or 4, hands the image over as PixInsight-ready XISF with a note of the processes to use next", "0.7"],
    ["Smart telescopes", "Seestar, DWARF, Vespera, Stellina and eVscope raw subs and stacks are recognised and handled without calibration frames", "0.6"],
    ["XISF everywhere", "Reads PixInsight XISF light frames, including every compression (zlib, LZ4, LZ4-HC, Zstandard)", "0.9.28"],
    ["Windows and Apple-silicon Mac", "Both builds run a full self-test on real hardware before every release", "0.9.0"],
  ]],
  ["Page 1 – Quality & Cull", [
    ["Tonight's Target Picker", "What to shoot tonight from your site and patch of sky - hours in the dark, peak altitude, Moon, size in your frame; sends targets straight into N.I.N.A., built from your own template", "0.9.131"],
    ["Organize My Astro Files", "Sorts a folder, drive or NAS of FITS/XISF subs and masters by rig (Light Frames / Calibration Frames) or by target - preview first, copy or move, undo; duplicates flagged and kept, set aside or deleted", "0.9.129 / 0.9.130"],
    ["Star quality on every frame", "FWHM, eccentricity, SNR, star count and background, with a badge and a score against frames of the same camera and filter", "Stage 2"],
    ["Session chart", "One dot per frame in capture order, nights separated; click to jump, drag a range to flag or reject", "Stage 2"],
    ["Auto-cull presets", "Three strengths flag frames from FWHM, stars, SNR (catches thin cloud) and eccentricity; you review before rejecting, and it can be undone", "0.3.0"],
    ["Fast measuring over a NAS", "Reads about a quarter of each file, 3–4× faster on network drives; results are saved, so reopening is nearly instant", "Stage 2"],
  ]],
  ["Page 2 – Calibrate & Stack", [
    ["Automatic calibration matching", "Bias, darks and flats picked by camera, gain, offset, temperature, exposure, filter, rig and nearest date, with ✓ ⚠ ✗ and a plain reason", "0.4.0"],
    ["SkyFlats (Mote Buster)", "No flats? A flat is built from your own light frames; a poor one stops and asks before it is used", "0.9.22"],
    ["Build Calibration Masters", "One folder of mixed bias, dark, flat and dark-flat subs is sorted, checked and turned into masters, no lights needed", "0.9.24"],
    ["Field-distortion correction", "Meridian flips and other setups get a distortion model per frame, so corner stars stay sharp", "0.9.43"],
    ["Hot pixels removed", "Cosmetic correction in every frame: hot pixels and cosmic-ray hits are replaced by their neighbours, so undithered data has no coloured specks", "0.9.54"],
    ["Hot-pixel-proof star matching", "Single-pixel spikes are ignored, so frames with hundreds of hot pixels line up on the sky, not the sensor", "0.9.51"],
    ["Plate-solution guided alignment", "Frames and masters that carry a plate solution are lined up from where they point on the sky, with the stars fine-tuning it - rescues matches that fail and catches ones that are wrong", "0.9.53"],
    ["Comet stacking", "A second stack follows the comet; stars and comet are both sharp in the final master", "0.9.22"],
    ["Comet LRGB", "Filters shot one after another (NINA Orbitals): the comet is lined up across the filters on page 4, the stars stay put", "0.9.126"],
  ]],
  ["Page 3 – Astro Image Processing", [
    ["One run for every filter", "Auto-crop, plate solve, gradient removal and the RC-Astro tools on every master of the target, with the same crop", "0.5.0"],
    ["RC-Astro without a terminal", "BlurXTerminator, NoiseXTerminator and StarXTerminator through the RC-Astro CLI, on the fast GPU, licences activated with a pasted code", "0.5.0"],
    ["Persistent plate solving", "ASTAP retries other field sizes and works when the header has no focal length", "0.9.37"],
    ["PixInsight next steps", "Linear XISF/FITS plus a plan of the PixInsight processes still to run, with done steps ticked", "0.5.0"],
  ]],
  ["Page 4 – Finishing Touches", [
    ["Masters kept with the project", "Save Project As can copy the masters into the project folder - one folder per target", "0.9.127"],
    ["Comet steps", "Bring Out the Tail and Reveal Comet Jets (Larson-Sekanina), offered for comets after the stretch", "0.9.126"],
    ["Recommended next step", "Tailored to your data (colour, mono, narrowband, smart scope) and the target (galaxy, nebula, cluster, comet)", "0.7"],
    ["Pick from your own image, then fine-tune", "Small versions of your picture to choose from; clicking previews big without applying; plain-language sliders fine-tune it, and the app says how much of the picture they change", "0.7 / 0.9.48"],
    ["Do It Again", "Repeatable steps (curves, saturation, noise, Mote Buster …) can be run once more on their own result", "0.9.48"],
    ["Adjust Colours, Full Window, zoom", "Bring out one colour at a time; see the picture full-window and zoom in to full resolution at any step", "0.9.58"],
    ["“Processed with SubZeroAstro” credit", "A small, optional credit line in the corner of finished pictures - moves aside for your signature; never on XISF / FITS data", "0.9.59"],
    ["Planetary nebulae recognised", "From the name (M57, M27 …) the whole process is tailored: colour-preserving brightening, the nebula moved back from the stars image, Tame the Ring keeps the centre and halo", "0.9.71-0.9.76"],
    ["Mote Buster – Fix Dust Donuts", "Measures each dust shadow ring by ring and divides it out with a synthetic flat, so the detail underneath comes back; run it in rounds", "0.9.22 / 0.9.47"],
    ["Every master used", "Several masters per filter averaged (less noisy ones count more), other setups bent onto one grid, unmatched ones left out", "0.9.41"],
    ["Target description", "A short write-up from Wikipedia plus SIMBAD facts, credited, saved next to your picture", "0.9.34"],
  ]],
];
const highlights = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
    children: [new TextRun("Highlights – what makes SubZeroAstro special")] }),
  new Paragraph({ spacing: { after: 160 }, children: runs(
    "The features that set SubZeroAstro apart, grouped by page, with the version each arrived in. Each one is explained in its chapter.") }),
];
for (const [group, rows] of HIGHLIGHTS) {
  highlights.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(group)] }));
  table(["Feature", "What it does", "Since"], rows, [2500, 5960, 900], highlights, 18);
}

const contents = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Contents")] }),
];
for (const e of toc) {
  const isCh = e.level === 1;
  const kids = [new InternalHyperlink({ anchor: e.anchor, children: [
    new TextRun({ text: `${e.num}   ${e.title}`, bold: isCh, color: isCh ? BLUE : undefined })] })];
  if (isCh && PAGES[String(e.num)]) {
    kids.push(new TextRun({ children: [new Tab()] }));
    kids.push(new TextRun({ text: String(PAGES[String(e.num)]), bold: true }));
  }
  contents.push(new Paragraph({ indent: { left: isCh ? 0 : 480 }, spacing: { before: isCh ? 140 : 0, after: 20 },
    tabStops: [{ type: TabStopType.RIGHT, position: 9360, leader: LeaderType.DOT }], children: kids }));
}

const doc = new Document({
  creator: "Brian A. Poteet, DVM", title: "SubZeroAstro User Manual", description: "User manual for SubZeroAstro " + VERSION,
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } },
        titlePage: true },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "SubZeroAstro User Manual", size: 16, color: GREY })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GREY })] })] }) },
      children: [...cover, ...highlights, ...contents, ...body] },
    { properties: { type: "continuous", page: { size: { width: 12240, height: 15840 },
        margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } }, column: { count: 2, space: 720 } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "SubZeroAstro User Manual", size: 16, color: GREY })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GREY })] })] }) },
      children: indexBody },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "SubZeroAstro_User_Manual.docx"), buf);
  console.log("written", buf.length, "bytes;", toc.length, "headings;", terms.length, "index terms");
});
