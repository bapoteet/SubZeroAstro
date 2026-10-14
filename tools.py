"""
tools.py - running external programs: RC-Astro CLI (BlurXTerminator,
StarXTerminator, NoiseXTerminator), GraXpert and ASTAP.

RC-Astro integration follows RC-Astro's README-DEVS.txt (schemaVersion 4):
  * `rc-astro --json`            products + license status
  * `rc-astro <key> --json`      self-describing parameters (built into a UI)
  * `rc-astro <key> in -o out --json`  NDJSON event stream (progress, errors)
  * exit code 77 = not licensed on this machine
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 4
EX_NOPERM = 77
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class ToolError(Exception):
    pass


class NotLicensed(ToolError):
    pass


class ToolCancelled(ToolError):
    pass


def _command(exe) -> list:
    """How to start a tool. A .py path runs with this Python (used by the tests)."""
    exe = str(exe)
    if exe.lower().endswith(".py"):
        return [sys.executable, exe]
    return [exe]


def _json_answer(text):
    """The JSON document in a tool's output, even with notices printed before or after it."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except ValueError:
            pass
    return None


def _run_capture(cmd, timeout=120):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


RC_ACCOUNT_URL = "https://www.rc-astro.com/account/"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_console(text):
    """Console output without colour codes, carriage-return redraws or empty runs of lines."""
    lines = []
    for ln in _ANSI.sub("", text or "").replace("\r\n", "\n").split("\n"):
        ln = ln.split("\r")[-1].rstrip()
        if ln or (lines and lines[-1]):
            lines.append(ln)
    return "\n".join(lines).strip()


def open_in_terminal(cmd):
    """Run a command in a visible terminal window the user can type into
    (Command Prompt on Windows, Terminal on macOS). Returns True when one was opened."""
    cmd = [str(c) for c in cmd]
    try:
        if os.name == "nt":
            subprocess.Popen(["cmd", "/k"] + cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10))
            return True
        import shlex
        line = " ".join(shlex.quote(c) for c in cmd)
        if sys.platform == "darwin":
            line = line.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{line}"',
                              "-e", 'tell application "Terminal" to activate'])
            return True
        for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
            if shutil.which(term):
                subprocess.Popen([term, "-e", "sh", "-c", line + "; exec sh"])
                return True
    except OSError:
        pass
    return False


def _stream(cmd, on_line, cancel=None, cwd=None):
    """Run a command, calling on_line(text) for every stdout line. Returns exit code."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                         creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace", cwd=cwd)
    err_lines = []
    try:
        for line in p.stdout:
            if cancel and cancel():
                p.kill()
                raise ToolCancelled()
            on_line(line.rstrip("\r\n"))
        p.wait()
        err_lines = (p.stderr.read() or "").splitlines()
    finally:
        if p.poll() is None:
            p.kill()
    return p.returncode, err_lines


# ==========================================================================
# visibleIf / disabledIf expressions:  identifiers, ! && || ( )
# ==========================================================================

_TOKEN = re.compile(r"\s*(\(|\)|!|&&|\|\||[A-Za-z_][A-Za-z0-9_]*)")


def eval_expr(expr: str, values: dict) -> bool:
    """Evaluate a README-DEVS boolean expression. Unknown names are false;
    non-bool values are true when non-zero."""
    tokens, pos = [], 0
    expr = expr or ""
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            if expr[pos:].strip() == "":
                break
            raise ValueError(f"bad expression: {expr!r}")
        tokens.append(m.group(1))
        pos = m.end()
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def take():
        nonlocal i
        i += 1
        return tokens[i - 1]

    def atom():
        t = take()
        if t == "!":
            return not atom()
        if t == "(":
            v = orx()
            take()  # ')'
            return v
        v = values.get(t, False)
        return bool(v)

    def andx():
        v = atom()
        while peek() == "&&":
            take()
            r = atom()
            v = v and r
        return v

    def orx():
        v = andx()
        while peek() == "||":
            take()
            r = andx()
            v = v or r
        return v

    return orx() if tokens else False


# ==========================================================================
# RC-Astro CLI
# ==========================================================================

MAC = sys.platform == "darwin"


def app_exe(p) -> Path:
    """A macOS .app bundle -> the program inside it (Contents/MacOS/...)."""
    p = Path(p)
    if p.suffix.lower() == ".app" and p.is_dir():
        inner = p / "Contents" / "MacOS"
        named = [inner / p.stem, inner / p.stem.lower()]
        for c in named + sorted(inner.glob("*")) if inner.is_dir() else []:
            if c.is_file():
                return c
    return p


def _mac_cands(*paths):
    return [app_exe(Path(x).expanduser()) for x in paths] if MAC else []


def find_rcastro(configured: str | None = None) -> str | None:
    cands = []
    if configured:
        p = app_exe(Path(configured))
        cands += [p] if p.suffix or p.is_file() else [p / "rc-astro.exe", p / "rc-astro"]
    cands += _mac_cands("/Applications/RC-Astro/CLI/rc-astro", "/Applications/RC-Astro CLI/rc-astro",
                        "/usr/local/bin/rc-astro", "/opt/homebrew/bin/rc-astro", "~/Applications/RC-Astro/rc-astro")
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files"):
        cands.append(Path(base) / "RC-Astro" / "CLI" / "rc-astro.exe")
    for c in cands:
        if c.is_file():
            return str(c)
    w = shutil.which("rc-astro")
    return w


@dataclass
class ProductSpec:
    doc: dict

    @property
    def key(self):
        return self.doc.get("key")

    @property
    def name(self):
        return self.doc.get("displayName") or self.doc.get("name") or self.key

    @property
    def parameters(self) -> list:
        return self.doc.get("parameters", [])

    @property
    def modes(self) -> list:
        return self.doc.get("modes", [])

    @property
    def groups(self) -> list:
        g = self.doc.get("groups")
        if g:
            return g
        return [{"name": "parametersGroup", "label": "Parameters",
                 "params": [p["name"] for p in self.parameters]}]

    def param(self, name):
        return next((p for p in self.parameters if p.get("name") == name), None)

    @property
    def licensed(self) -> bool:
        return bool((self.doc.get("license") or {}).get("valid", False))

    def defaults(self) -> dict:
        return {p["name"]: p.get("default") for p in self.parameters}

    def effective(self, values: dict, mode: str | None = None) -> dict:
        """Values after applying an engaged mode's pins and disabledIf rules."""
        v = dict(self.defaults())
        v.update({k: x for k, x in values.items() if self.param(k)})
        if mode:
            m = next((m for m in self.modes if m.get("name") == mode), None)
            if m:
                v.update(m.get("pins", {}))
        for p in self.parameters:
            if p.get("disabledIf") and eval_expr(p["disabledIf"], v):
                v[p["name"]] = p.get("disabledValue", p.get("default"))
        return v

    def is_visible(self, p: dict, values: dict) -> bool:
        return not p.get("visibleIf") or eval_expr(p["visibleIf"], values)

    def is_disabled(self, p: dict, values: dict, mode: str | None = None) -> bool:
        if mode:
            m = next((m for m in self.modes if m.get("name") == mode), None)
            if m and p["name"] in m.get("pins", {}):
                return True
        return bool(p.get("disabledIf")) and eval_expr(p["disabledIf"], values)

    def args(self, values: dict, mode: str | None = None) -> list:
        """CLI options for these settings (README-DEVS section 8)."""
        out = []
        pinned = set()
        if mode:
            m = next((m for m in self.modes if m.get("name") == mode), None)
            if m:
                out.append(m["flag"])
                pinned = set(m.get("pins", {}))
        v = self.effective(values, mode)
        for p in self.parameters:
            flag = p.get("flag")
            if not flag or p.get("guiOnly") or p["name"] in pinned:
                continue
            if not self.is_visible(p, v):
                continue
            if p.get("disabledIf") and eval_expr(p["disabledIf"], v):
                continue                    # forced by the CLI itself
            val = v.get(p["name"])
            if val == p.get("default"):
                continue
            t = p.get("type")
            if t == "bool":
                out.append(flag if val else f"{flag}=false")
            elif t == "float":
                out += [flag, f"{float(val):.{int(p.get('precision', 2))}f}"]
            elif t == "int":
                out += [flag, str(int(val))]
            else:
                out += [flag, str(val)]
        return out


def preferred_device(devices: list) -> str:
    """Pick a dedicated GPU (NVIDIA / AMD) when the machine also has integrated
    graphics - RC-Astro's 'auto' simply takes the first GPU, which on many laptops
    is the slower Intel one."""
    gpus = [d for d in devices or [] if d.get("kind") == "gpu"]
    for d in gpus:
        name = (d.get("name") or "").lower()
        if any(k in name for k in ("nvidia", "geforce", "rtx", "quadro", "radeon rx", "radeon pro", "amd radeon")):
            return d["id"]
    return "auto"


class RCAstro:
    def __init__(self, exe: str):
        self.exe = exe
        self.cli_version = None
        self.schema = None

    def products(self) -> dict:
        try:
            code, out, err = _run_capture(_command(self.exe) + ["--json"])
        except (OSError, subprocess.SubprocessError) as e:
            raise ToolError(f"could not start {Path(self.exe).name}: {e}")
        if code != 0:
            raise ToolError((err or out or f"rc-astro exited with {code}").strip()[:400])
        doc = _json_answer(out)
        if not isinstance(doc, dict) or "products" not in doc:
            said = " ".join((out or err or "nothing").split())[:160]
            raise ToolError(f"{Path(self.exe).name} did not answer like the RC-Astro CLI (it said: {said}). "
                            "Use Browse… to pick the rc-astro program from RC-Astro's CLI download.")
        self.cli_version = doc.get("cliVersion")
        self.schema = doc.get("schemaVersion")
        return doc

    def licences(self) -> list:
        """[(key, display name, valid, message)] for every product the CLI knows."""
        doc = self.products()
        return [(p["key"], p.get("displayName") or p.get("name") or p["key"],
                 bool((p.get("license") or {}).get("valid")), (p.get("license") or {}).get("message", ""))
                for p in doc.get("products", [])]

    def activate(self, code: str, timeout=120):
        """Run `rc-astro license` and paste the activation code from the user's rc-astro.com
        account into it, as RC-Astro's instructions say to do in a terminal. One code activates
        every product the account owns. Returns (exit code, everything the CLI printed)."""
        code = (code or "").strip()
        if not code:
            raise ToolError("paste the activation code from your RC-Astro account first")
        try:
            p = subprocess.run(_command(self.exe) + ["license"], input=code + "\n", capture_output=True,
                               text=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise ToolError("rc-astro license did not finish (no answer from RC-Astro's server?). "
                            "Check the internet connection, or use 'Do It in a Terminal Window'.")
        except (OSError, subprocess.SubprocessError) as e:
            raise ToolError(f"could not start {Path(self.exe).name}: {e}")
        return p.returncode, _clean_console((p.stdout or "") + ("\n" + p.stderr if p.stderr else ""))

    def check_update(self, timeout=60):
        """`rc-astro update`: an online check; returns (exit code, what it said)."""
        try:
            code, out, err = _run_capture(_command(self.exe) + ["update"], timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ToolError("the update check did not finish - is this computer online?")
        except (OSError, subprocess.SubprocessError) as e:
            raise ToolError(f"could not start {Path(self.exe).name}: {e}")
        return code, _clean_console(out + ("\n" + err if err else ""))

    def install_update(self):
        """`rc-astro update --install`: downloads the new installer and launches it (Windows, macOS)."""
        return open_in_terminal(_command(self.exe) + ["update", "--install"])

    def spec(self, key: str) -> ProductSpec:
        code, out, err = _run_capture(_command(self.exe) + [key, "--json"])
        if code == EX_NOPERM and not out.strip():
            raise NotLicensed(f"{key} is not licensed on this computer")
        doc = _json_answer(out)
        if not isinstance(doc, dict):
            raise ToolError((err or out or f"rc-astro {key} --json failed ({code})").strip()[:400])
        return ProductSpec(doc)

    def run(self, key, inp, out, args, device="auto", progress=None, log=None, cancel=None):
        """Process one image. progress(fraction, text); returns list of files written."""
        cmd = _command(self.exe) + [key, str(inp), "-o", str(out), "--overwrite", "--json"]
        if device and device != "auto":
            cmd += ["--device", device]
        cmd += list(args)
        written, errors = [], []

        def on_line(line):
            line = line.strip()
            if not line:
                return
            try:
                ev = json.loads(line)
            except ValueError:
                if log:
                    log(line)
                return
            kind = ev.get("event")
            if kind == "progress" and progress:
                eta = ev.get("eta")
                progress(float(ev.get("done", 0)) / 100.0,
                         f"{ev.get('done', 0):.0f}%" + (f", {eta:.0f} s left" if eta else ""))
            elif kind == "status":
                if ev.get("phase") == "complete" and ev.get("output"):
                    written.append(ev["output"])
                elif ev.get("phase") == "initializing" and progress:
                    progress(0.0, "loading AI model…")
            elif kind == "info":
                topic = ev.get("topic")
                if topic == "device" and log:
                    log(f"    running on {ev.get('device', '?').upper()} {ev.get('name') or ''}".rstrip())
                elif topic == "modelDownload" and log:
                    log(f"    {ev.get('status', '')} model {ev.get('file', '')}")
                elif topic == "license" and log:
                    log(f"    {ev.get('message', '')}")
                elif topic == "update" and log:
                    log(f"    RC-Astro update available: {ev.get('new', '')}")
            elif kind == "warning" and log:
                log(f"    warning: {ev.get('message', '')}")
            elif kind == "error":
                errors.append(ev.get("message", "error"))
        t0 = time.time()
        code, err_lines = _stream(cmd, on_line, cancel)
        if code == 0:
            # rc-astro may not announce every file it writes (e.g. the stars-only image):
            # pick up any new image next to the output
            outp, seen = Path(out), {str(Path(w)) for w in written}
            try:
                for p in sorted(outp.parent.iterdir()):
                    if (p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff")
                            and str(p) not in seen and p.resolve() != Path(inp).resolve()
                            and (p.stem.startswith(outp.stem) or "star" in p.stem.lower())
                            and p.stat().st_mtime >= t0 - 0.5):
                        written.append(str(p))
            except OSError:
                pass
        if code == EX_NOPERM:
            raise NotLicensed(f"{key} is not licensed on this computer — click 'RC-Astro Licences…' next to "
                              f"the RC-Astro CLI row (Astro Image Processing page) to activate it")
        if code != 0:
            raise ToolError("; ".join(errors + err_lines)[:600] or f"rc-astro {key} failed (exit {code})")
        return written or [str(out)]


# ==========================================================================
# GraXpert
# ==========================================================================

def find_graxpert(configured: str | None = None) -> str | None:
    cands = []
    if configured:
        p = app_exe(Path(configured))
        cands += [p] if p.suffix or p.is_file() else [p / "GraXpert-win64.exe", p / "GraXpert.exe"]
    cands += _mac_cands("/Applications/GraXpert.app", "~/Applications/GraXpert.app")
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
        if base:
            cands += [Path(base) / "GraXpert" / "GraXpert-win64.exe", Path(base) / "GraXpert" / "GraXpert.exe",
                      Path(base) / "Programs" / "GraXpert" / "GraXpert-win64.exe"]
    for c in cands:
        if c.is_file():
            return str(c)
    return shutil.which("GraXpert-win64") or shutil.which("GraXpert")


class GraXpert:
    def __init__(self, exe):
        self.exe = exe

    def _run(self, inp, out_base, extra, log=None, cancel=None, progress=None):
        cmd = _command(self.exe) + [str(inp), "-cli", "-output", str(out_base)] + extra
        tail = []

        def on_line(line):
            tail.append(line)
            del tail[:-20]
            if progress and "%" in line:
                m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                if m:
                    progress(float(m.group(1)) / 100, f"{m.group(1)}%")
        code, err_lines = _stream(cmd, on_line, cancel)
        found = sorted(Path(out_base).parent.glob(Path(out_base).name + ".*"))
        found = [f for f in found if f.suffix.lower() in (".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff")]
        if code != 0 or not found:
            msg = " | ".join((err_lines + tail)[-6:]).strip()
            raise ToolError(f"GraXpert failed (exit {code}): {msg[:500]}")
        return str(found[0])

    def background_extraction(self, inp, out_base, smoothing=0.1, correction="Subtraction", gpu=True, **kw):
        return self._run(inp, out_base, ["-cmd", "background-extraction", "-correction", correction,
                                         "-smoothing", f"{smoothing:g}", "-gpu", "true" if gpu else "false"], **kw)

    def denoise(self, inp, out_base, strength=0.5, gpu=True, **kw):
        return self._run(inp, out_base, ["-cmd", "denoising", "-strength", f"{strength:g}",
                                         "-gpu", "true" if gpu else "false"], **kw)


# ==========================================================================
# ASTAP
# ==========================================================================

def find_astap(configured: str | None = None) -> str | None:
    cands = []
    if configured:
        p = app_exe(Path(configured))
        cands += [p] if p.suffix or p.is_file() else [p / "astap_cli.exe", p / "astap.exe", p / "astap_cli",
                                                      p / "astap"]
    cands += _mac_cands("/usr/local/bin/astap_cli", "/opt/homebrew/bin/astap_cli", "/Applications/astap_cli",
                        "/Applications/ASTAP.app/Contents/MacOS/astap_cli", "/Applications/ASTAP.app")
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files"):
        cands += [Path(base) / "astap" / "astap_cli.exe", Path(base) / "astap" / "astap.exe"]
    for c in cands:
        if c.is_file():
            return str(c)
    return shutil.which("astap_cli") or shutil.which("astap")


def parse_header_text(text: str) -> list:
    """FITS header cards from a .wcs file (fixed 80-character cards, with or
    without line breaks)."""
    import fits_core as core
    if "\n" in text.strip():
        raw = [ln.rstrip("\r") for ln in text.splitlines()]
    else:
        raw = [text[i:i + 80] for i in range(0, len(text), 80)]
    cards = []
    for c in raw:
        if not c.strip() or c[:8].strip() == "END":
            continue
        cards.append(core.parse_card(c.ljust(80)[:80]))
    return cards


class ASTAP:
    def __init__(self, exe):
        self.exe = exe

    def solve(self, fits_path, ra_deg=None, dec_deg=None, fov_height_deg=None, radius=30.0, downsample=0,
              log=None, cancel=None):
        """Plate solve a FITS file. Returns a list of WCS cards (keyword, value, comment)."""
        fits_path = Path(fits_path)
        out_base = fits_path.with_suffix("")
        cmd = _command(self.exe) + ["-f", str(fits_path), "-r", f"{radius:g}", "-z", str(int(downsample)),
                                    "-o", str(out_base)]
        if fov_height_deg:
            cmd += ["-fov", f"{fov_height_deg:.4f}"]
        if ra_deg is not None and dec_deg is not None:
            cmd += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}"]
        lines = []
        code, err = _stream(cmd, lines.append, cancel)
        wcs = out_base.with_suffix(".wcs")
        ini = out_base.with_suffix(".ini")
        info = {}
        if ini.is_file():
            for ln in ini.read_text(errors="replace").splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    info[k.strip().upper()] = v.strip()
        solved = info.get("PLTSOLVD", "").upper().startswith("T")
        if not wcs.is_file() or (info and not solved):
            reason = info.get("ERROR") or info.get("WARNING") or " ".join((lines + err)[-3:]) or f"exit {code}"
            raise ToolError(f"ASTAP could not solve the image: {reason}")
        cards = parse_header_text(wcs.read_text(errors="replace"))
        keep = ("CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_", "CDELT", "CROTA", "CUNIT", "EQUINOX", "RADESYS",
                "PLTSOLVD", "A_", "B_", "AP_", "BP_", "LONPOLE", "LATPOLE")
        return [c for c in cards if c[0] and c[1] is not None and c[0].startswith(keep)]


def reveal(paths):
    """Open Explorer at the files (selected). Works for one file or a folder."""
    import subprocess
    import sys as _sys
    paths = [Path(x) for x in (paths if isinstance(paths, (list, tuple)) else [paths]) if x]
    if not paths:
        return False
    p = paths[0]
    if _sys.platform == "darwin":                     # Finder, with the file selected
        try:
            subprocess.Popen(["open", str(p)] if p.is_dir() else ["open", "-R", str(p)])  # noqa: S603,S607
            return True
        except OSError:
            return False
    if not _sys.platform.startswith("win"):
        try:
            subprocess.Popen(["xdg-open", str(p if p.is_dir() else p.parent)])  # noqa: S603,S607
            return True
        except OSError:
            return False
    try:
        if p.is_dir():
            os.startfile(str(p))  # noqa: S606
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])  # noqa: S603,S607
        return True
    except OSError:
        return False
