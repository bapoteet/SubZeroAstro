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
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, head, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    shading: head ? { type: ShadingType.CLEAR, fill: "DCE6F2", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(text, head ? { bold: true } : {}) })],
  });
  body.push(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, true, widths[i])) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, false, widths[i])) }))],
  }));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

// ================================================================== THE MANUAL
ch(1, "Welcome to SubZeroAstro");
sec("1.1", "What SubZeroAstro does", ["Four pages", "Workflow overview"]);
p("SubZeroAstro takes you **from a folder of raw sub-frames to a finished astrophoto** - with as little hassle as possible. It walks you through four pages, one after the other, like turning the pages of a book:");
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
  "SubZeroAstro reads **FITS** (`.fit`, `.fits`, `.fts`) and PixInsight **XISF** files, from capture programs such as N.I.N.A., ASIAIR, SharpCap and smart telescopes (ZWO Seestar, DWARF, Vaonis Vespera, Stellina, eVscope).",
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
sec("3.3", "Updating to a new version", ["Updating"]);
p("Download the newest zip from the releases page and replace the old app folder (Windows) or the app in Applications (Mac). **Your settings, tool locations, rig names, saved measurements and saved progress are kept** - they are stored separately from the app.");
sec("3.4", "Removing SubZeroAstro", ["Uninstall"]);
p("Delete the app folder (Windows) or drag SubZeroAstro from Applications to the Bin (Mac). Saved progress can be cleared first under **Settings › Saved Progress**. Your images are never touched.");

ch(4, "First-time setup");
sec("4.1", "Tell SubZeroAstro where your tools are", ["Tool locations", "ASTAP", "GraXpert", "RC-Astro CLI", "Device (GPU)"]);
p("When SubZeroAstro starts, it checks for ASTAP, GraXpert and the RC-Astro CLI in the background. If one is missing - or RC-Astro is installed but a tool is not licensed - a slim **amber banner** appears at the top of page 1, saying what you would miss (for example *plate solving is skipped*). Nothing is wrong: you can still cull and stack. The banner has two buttons:");
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
  "**Darks** are matched by exposure, gain, offset and temperature (±3 °C). **Flats** are matched by camera, filter and rig, nearest in date. **Bias** or **dark-flats** are needed to calibrate flats properly.",
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
  "**Page 1**: your folders and flags are saved as you go. After a crash SubZeroAstro offers to reopen the session. **Ctrl+Z** undoes rejects, flags and auto-cull.",
  "**Page 2**: each master is noted as soon as it is saved; an interrupted run offers **Continue With the Rest**.",
  "**Page 3**: the image is saved after every step; **Go Back & Redo This Step** and **Start Fresh** are under the preview.",
  "**Page 4**: every step is saved: **Undo** (Ctrl+Z), **Go Back to Here**, **Back to Start**, and reopening the same masters later resumes the project.",
]);
sec("5.4", "Stop Here – Continue in PixInsight", ["Stop Here – Continue in PixInsight", "PixInsight hand-over"]);
p("The purple **Stop Here – Continue in PixInsight** button (pages 2, 3 and 4) saves the image exactly as it is right now as PixInsight-ready XISF, writes a short note listing the PixInsight processes to use next, and opens the folder with the files selected.");

ch(6, "Page 1 – Astro Image Quality and Cull");
img("1_main_window_fwhm.png", "Page 1: frame list (left), the selected frame, the session chart and the measurements (example data)");
sec("6.1", "Loading your sub-frames", ["Load Data", "Include subfolders", "Filter used", "Network drive (NAS)"]);
steps([
  "Click **Load Data** (Ctrl+O). Add one or more folders (several nights, several drives) or individual files. Tick **Include subfolders** if your nights are in subfolders.",
  "If your capture software does not record the filter (e.g. ASIAIR with a manual filter drawer), choose it under **Filter used**.",
  "Press **Start Processing**. Frames appear at once and are measured in the background.",
]);
img("7_load_data_window.png", "The Load Data window", 480);
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
sec("6.5", "On to stacking", ["Continue to Calibrate and Stack"]);
p("When you are happy, click **Calibrate and Stack ➜** at the top right. Only your kept frames go to page 2.");

ch(7, "Page 2 – Calibrate and Stack");
img("10_stacking_done.png", "Page 2 after stacking: one row per target and filter, with the calibration found for each (example data)");
sec("7.1", "The light frame groups", ["Light groups", "Filter column", "Telescope (rig)"]);
p("Your kept frames are grouped by **target + filter + camera + telescope** - one row each, with the number of frames, total exposure and nights. The **Filter** of a row can be changed from its drop-down (for example to match your flats).");
sec("7.2", "Calibration: what the ✓ ⚠ ✗ mean", ["Calibration", "Old flats", "Missing calibration", "Flat warnings"]);
table(["Mark", "Meaning"], [
  ["✓ green", "A matching calibration master was found (or will be built)."],
  ["⚠ orange", "Usable, with a note - e.g. darks scaled from another exposure, or flats taken long ago (*\"dust may have moved\"*). It **will** be used."],
  ["✗ red", "Nothing suitable - e.g. *No flats for filter OIII*. You can still stack, without that correction."],
], [1600, 7760]);
p("Click a row to read the details below the table. Add calibration with **Add Calibration Folder…**, **Add Calibration Files…** or **Search Calibration Folder for Subframes**.");
warn("If the Flats cell says the flats have **no matching bias or dark-flats**, the correction will be too strong: bright corners and dark rings. Add a bias or dark-flats for that camera and gain.");
sec("7.3", "Stacking options", ["Reject outliers", "Weight frames by noise", "Colour cameras", "Line up all filters", "Save as FITS/XISF"]);
bullets([
  "**Reject outliers beyond 3.0 σ** - removes satellite trails, planes and hot pixels. The default suits almost everything.",
  "**Weight frames by noise** - better frames count more.",
  "**Colour cameras: Full resolution** (default) or half resolution (faster, less noise per pixel).",
  "**Save as** FITS and/or XISF; **Save new calibration masters** into your library.",
  "**Line up all filters of a target** (on) - all masters of a target share one pixel grid, ready to combine (LRGB, SHO).",
]);
sec("7.4", "Running the stack", ["Check & Stack", "Continue With the Rest"]);
p("Press **▶ Check & Stack**. Anything missing is listed first, with **Stack Anyway** or **Add Calibration Frames First…**. The log shows each step. If a run is interrupted, the next run offers **Continue With the Rest**. When it finishes, the button turns amber: **✓ Stacked – Run Again?**. Stacking again never overwrites a master - new versions are saved as `_2`, `_3`…");
sec("7.5", "Results", ["Open Master Lights Folder", "Stack log"]);
p("Masters are saved in your master lights folder, one subfolder per target, with a `_stacklog.txt` listing every frame used. **Open Master Lights Folder** shows them. Click **Continue to Image Processing ➜** to carry on.");

ch(8, "Page 3 – Astro Image Processing");
img("12_prepare_done.png", "Page 3 after processing: masters (left), preview with crop box, step history, and the step settings (right)");
sec("8.1", "Masters and target sets", ["Target sets", "Will be lined up", "Add Masters"]);
p("The masters from page 2 are loaded automatically (or use **Add Masters…** / **Add Folder…**). Masters of the **same target** form one row - e.g. *NGC281 [Ha, OIII, SII]*. If they were stacked in separate runs the row says **will be lined up**: they are matched star-to-star onto one grid before anything else, and edges not covered by every filter are trimmed.");
sec("8.2", "Crop", ["Crop", "Auto-crop"]);
p("The yellow box on the preview is the crop. **Auto-crop** trims the ragged stacking edges; drag the box or its sides to adjust, or **Full image** to keep everything. The same box is used for **every filter**, so they stay aligned.");
sec("8.3", "Plate solve", ["Plate solve", "Blind solve", "Search radius"]);
p("ASTAP finds where the picture points in the sky, using the position and pixel scale recorded in the masters. If it fails, tick **Blind solve** (slower, searches the whole sky).");
sec("8.4", "Remove gradients", ["Gradient removal", "GraXpert", "Light pollution"]);
p("Removes light-pollution glow and uneven backgrounds. **GraXpert (AI)** is recommended; **Smoothing** 0.1 and **Correction: Subtraction** are good defaults. Without GraXpert a built-in model is used.");
sec("8.5", "The XTerminators", ["BlurXTerminator", "NoiseXTerminator", "StarXTerminator", "Unscreen Stars"]);
table(["Tool", "What it does", "Good starting settings"], [
  ["BlurXTerminator", "Sharpens stars and detail", "Sharpen Stars 0.25, Sharpen Nonstellar 0.50, Automatic PSF on"],
  ["NoiseXTerminator", "Reduces noise (a light pass on unstretched data)", "Denoise 0.6"],
  ["StarXTerminator", "Separates a **starless** and a **stars-only** image", "Generate Star Image on; **Unscreen Stars off**"],
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
sec("9.1", "How it works", ["Recommended next", "Do It", "Skip", "Pick the one you like", "Fine-tune", "Apply", "Cancel", "Workflow list"]);
steps([
  "The **workflow list** on the left shows every step for your kind of data: ✓ done, ➜ next, ○ later, – not needed, ◇ optional.",
  "The card at the top says what is **Recommended next** and why. Click **Do It** (or click any step in the list).",
  "Pick a version: each step shows 3-5 small pictures of **your** image - gentle to strong, or different colour mixes. ★ marks the suggestion. **Clicking only previews it big** - nothing changes yet.",
  "Optional: tick **Fine-tune** and move the plain-language sliders; the preview follows. **Reset** returns to the version you picked.",
  "Press **Apply** to keep it - or **Cancel** to leave the picture exactly as it was.",
]);
img("17_finish_brighten_choices.png", "Picking a version: Brighten the Image offers five strengths (example data)");
sec("9.2", "Going back", ["Undo", "Go Back to Here", "Back to Start", "Hold for Before (Finishing Touches)"]);
bullets([
  "**Undo** (Ctrl+Z) removes the last step.",
  "**Show:** pick any earlier step to look at it; **Hold for Before** compares; **Go Back to Here** continues from that step.",
  "**Back to Start** returns to the original masters.",
  "Every step is saved. Opening the same masters again offers to **carry on where you left off**.",
]);
sec("9.3", "The steps", ["Combine Colour Channels", "Choose a Colour Palette", "Balance Colours", "Remove Green Cast", "Take Stars Out", "Brighten the Image", "Tame Bright Core", "Curves", "Saturation", "Bring Out Hidden Structure", "Brighten Stars Separately", "Shrink Stars", "Star Colours", "Put Stars Back", "Final Crop", "Spot Fix", "Sign Your Picture", "Add Hydrogen Glow"]);
p("Which steps appear depends on your data. In typical order:");
table(["Step", "What it does", "PixInsight equivalent"], [
  ["Add Hydrogen Glow", "Adds Ha to the red of an RGB set, without its stars", "Continuum subtraction + PixelMath"],
  ["Combine Colour Channels", "Builds colour from separate filters: RGB, LRGB, Hubble (SHO), HSO, HOO, Foraxx…", "ChannelCombination / LRGBCombination"],
  ["Choose a Colour Palette", "Dual-band colour cameras: separates Ha and OIII (or SII and OIII) and offers HOO, Gold & blue, Foraxx…", "PixelMath palettes"],
  ["Balance Colours", "Neutral sky and natural star colour", "BackgroundNeutralization / colour calibration"],
  ["Remove Green Cast", "Removes the green tint (or turns Hubble colours gold and blue)", "SCNR"],
  ["Take Stars Out", "Hides the stars so the nebula can be brightened strongly (instant if page 3 already separated them)", "StarXTerminator"],
  ["Brighten the Image", "The stretch - five strengths, three methods", "STF / HistogramTransformation / GHS"],
  ["Tame Bright Core ◇", "Recovers detail in a burnt-out galaxy core or bright nebula", "HDRMultiscaleTransform with a mask"],
  ["Adjust Brightness & Contrast", "Curves presets, sliders and a curve you can drag", "CurvesTransformation"],
  ["Colour Intensity", "Richer or calmer colour; dark sky protected", "Saturation curve"],
  ["Bring Out Hidden Structure ◇", "Local contrast for dust lanes and cores", "LocalHistogramEqualization"],
  ["Brighten Stars Separately", "Gentle star stretch - small, colourful stars", "Star stretch"],
  ["Shrink Stars", "Makes stars smaller", "Star reduction / MorphologicalTransformation"],
  ["Star Colours", "Natural star colours (estimated from your data) or clean white stars - recommended for narrowband", "Star colour correction"],
  ["Put Stars Back", "Blends the stars back at the strength you pick", "PixelMath screen blend"],
  ["Final Crop ◇", "A last clean-up crop; finds ragged or off-colour edges for you", "DynamicCrop"],
  ["Spot Fix ◇", "Click dust donuts, satellite remnants or hot pixels to patch them", "CloneStamp"],
  ["Sign Your Picture ◇", "Your signature in a corner, in one of three fonts", "Annotation"],
], [2500, 4460, 2400]);
p("◇ = optional: offered in the list but not pushed by *Recommended next*.");
sec("9.4", "Stars: take out, brighten, colour, put back", ["Stars workflow", "Starless"]);
p("Stars and nebula look best when processed separately. **Take Stars Out** hides the stars (the picture shows the nebula alone); the nebula steps follow; then **Brighten Stars Separately**, **Shrink Stars** and **Star Colours** work on the stars (their previews show the stars); **Put Stars Back** recombines them. If you **Skip** Take Stars Out, the stars simply stay in the picture.");
sec("9.5", "Final Crop, Spot Fix and your signature", ["Final Crop", "Spot Fix", "Sign Your Picture", "Signature fonts"]);
bullets([
  "**Final Crop** shows a yellow box on the whole picture, already placed where it found ragged or off-colour edges. Drag it or its sides (**Found Edges** / **Whole Picture** reset it) and press **Crop to the Box**.",
  "**Spot Fix**: click each blemish (the mouse wheel sets the size, right-click removes the last mark), then **Fix the marked spots**.",
  "**Sign Your Picture**: type your name once (it is remembered), pick a font - *Elegant script*, *Handwritten* or *Classic* - the corner and colour, and fine-tune size, opacity and distance from the edge. Do this last: the signature becomes part of the picture.",
]);
img("21_finish_signature.png", "Sign Your Picture: signature text, corner, colour and three fonts (example data)");
sec("9.6", "Saving and sharing", ["Save the Picture", "Save Project As", "Open Project", "Share to AstroBin", "AstroBin"]);
bullets([
  "**Save the Picture…** - TIFF (16-bit), JPG, PNG, XISF or FITS. Each target remembers its own folder.",
  "**Save Project As… / Open Project…** - keep every step in a folder of your choice and carry on later.",
  "**Share to AstroBin…** - saves an upload-ready JPG (and optional TIFF) plus an **acquisition CSV** with every night's subs, exposure, gain, temperature and FWHM, then opens AstroBin's upload page. Upload the picture, and on the *Acquisition* step choose **Import CSV**.",
  "**Stop Here – Continue in PixInsight** - see 5.4.",
]);
sec("9.7", "Starting your next target", ["End this Project and Start New Project", "New project", "Next target"]);
p("When a picture is finished, click the teal **↺ End this Project and Start New Project** button at the top right of Finishing Touches. SubZeroAstro asks first - and if you have not saved the picture yet, offers **Save the Picture First…**. Then every page and image closes, the helper programs are checked again, and you are back on **page 1 (Quality and Cull)** with the **Load Data** window open, ready for the next target's light frames: cull, stack, process and finish, from the beginning.");
tip("Nothing on your disk is deleted. Your masters, finished pictures and every Finishing Touches step stay where they are: open the same masters on Finishing Touches later and the project carries on where you stopped.", "Your work is safe");

ch(10, "Recipes for common setups");
sec("10.1", "Colour camera (OSC), broadband", ["OSC broadband recipe"]);
p("Cull › Stack (flats + darks or bias) › page 3: crop, plate solve, GraXpert, BlurX, StarX › page 4: Balance Colours, Remove Green Cast, Take Stars Out, Brighten, Curves, Colour Intensity, star steps, Put Stars Back, Final Crop, Sign.");
sec("10.2", "Colour camera with a dual-band filter", ["Dual-band recipe", "L-Ultimate", "L-eXtreme", "Askar D1", "Askar D2", "L-Synergy"]);
p("Make sure the filter is set (page 1, *Filter used* or *Set Filter Name*) so the matching flats are found. On page 4 the first step is **Choose a Colour Palette** (HOO, Gold & blue, Foraxx…), and **Star Colours** is recommended so the stars are not all teal. Two dual-band filters of the same target - **L-Ultimate + L-Synergy** or **Askar D1 + D2** - are combined into a full **SHO** set automatically.");
sec("10.3", "Mono camera: LRGB", ["LRGB recipe"]);
p("Each filter is stacked separately on page 2 (one row each; keep *Line up all filters* on). Load all masters together on page 3 - they must appear as **one row**. Page 4 starts with **Combine Colour Channels** (RGB or LRGB).");
sec("10.4", "Mono camera: narrowband (SHO)", ["SHO recipe", "Narrowband recipe", "Hubble palette"]);
p("As 10.3 with Ha, OIII and SII. Filter names from filter wheels such as *Ha 7nm*, *H-alpha*, *O*, *S* or *Antlia Ha 3nm Pro* are recognised. Page 4: **Combine Colour Channels** › Hubble (SHO), Foraxx or HOO; then Remove Green Cast (turns the green gold), and **Star Colours** before Put Stars Back.");
sec("10.5", "Smart telescopes (Seestar, DWARF, Vespera)", ["Smart telescopes", "Seestar", "DWARF", "Vespera", "Alt-az"]);
p("Raw subs are recognised as lights automatically, with no calibration needed (the telescope calibrates on board). Alt-az field rotation is handled during alignment and the rotated edges are cropped. Seestar *Stacked_…* files from several nights can be combined on page 2, or a single stack can go straight to page 3.");

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
  ["Plate solve failed", "Tick **Blind solve**; check that ASTAP has a star database installed."],
  ["Tools show *not found*", "Click **Browse…** and point at the program, or use **Settings › Your Tools…** (4.1)."],
  ["Amber banner on page 1", "A helper program is missing or not licensed. **Set Up…** fixes it; **Don't Remind Me** if you don't use it (4.1)."],
  ["An XTerminator says *not licensed*", "Click **RC-Astro Licences…** on page 3 and activate with the code from your rc-astro.com account (4.1)."],
  ["Frames skipped: *could not match the star field*", "Usually clouds, trailing or a badly out-of-focus frame. Frames saved mirror-image (some capture programs do this) are recognised and lined up automatically."],
  ["XTerminators very slow", "Choose your graphics card under **RC-Astro device** (4.1)."],
  ["Slow on a network drive", "Use *Star measurement: Fast*; hide the image (Ctrl+I); results are cached for next time."],
  ["Something went wrong mid-run", "Restart: SubZeroAstro offers to resume from its saved progress."],
], [3300, 6060]);
sec("11.2", "Getting help and reporting problems", ["Support", "Reporting problems", "Self-test"]);
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
  ["Ctrl+O", "Load Data"],
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

ch(14, "Privacy, licence and credits");
sec("14.1", "Privacy", ["Privacy"]);
p("SubZeroAstro does not send any information anywhere unless you ask it to. It reads and writes images only where you point it. **Share to AstroBin…** only opens AstroBin's page in your browser. Companion programs (ASTAP, GraXpert, RC-Astro) are separate programs with their own terms.");
sec("14.2", "Licence and credits", ["Licence", "MIT License", "Credits"]);
p("SubZeroAstro is free software under the **MIT License**. Created by **Brian A. Poteet**, built with Claude (Anthropic). The culling view was inspired by FITS Blaster for macOS by Tom van Peer. PixInsight, BlurXTerminator, NoiseXTerminator, StarXTerminator, GraXpert, ASTAP, AstroBin and the telescope and filter names mentioned are trademarks of their respective owners.");

// ------------------------------------------------------------------ index (own two-column section)
ch(15, "Index");
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
  new Paragraph({ spacing: { before: 2600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SubZeroAstro", bold: true, size: 72, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "From subs to masters – zero hassle.", italics: true, size: 30, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
    border: { top: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 12 } },
    children: [new TextRun({ text: "User Manual", bold: true, size: 44 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A beginner's guide to culling, stacking, processing and finishing your astrophotos", size: 24, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(IMG, "0_splash_4.00s.png")),
      transformation: { width: 468, height: 264 }, altText: { title: "SubZeroAstro", description: "Start-up screen", name: "cover" } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 },
    children: [new TextRun({ text: `Version ${VERSION}  ·  Windows and Mac (Apple silicon)`, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "by Brian A. Poteet  ·  October 2026", size: 22, color: GREY })] }),
];
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
  creator: "Brian A. Poteet", title: "SubZeroAstro User Manual", description: "User manual for SubZeroAstro " + VERSION,
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
      children: [...cover, ...contents, ...body] },
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
