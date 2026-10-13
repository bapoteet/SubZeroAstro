"""Stand-in for the RC-Astro CLI used by the tests. Follows README-DEVS.txt
(schemaVersion 4): products document, per-product parameter documents and
the NDJSON event stream. 'Processing' copies the image (BXT sharpens nothing,
NXT smooths slightly, SXT writes a starless and a stars image).
Set FAKE_RC_UNLICENSED=sxt,nxt to make products unlicensed (exit 77)."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VERSION = "1.0.3-fake"
UNLICENSED = set(filter(None, os.environ.get("FAKE_RC_UNLICENSED", "").split(",")))
# like the real rc-astro 1.0.1 on ROWORDER=TOP-DOWN files: images come back upside down,
# and the stars-only image is written without being announced
FLIP = os.environ.get("FAKE_RC_FLIP", "1") == "1"


def lic(key):
    if key in UNLICENSED:
        return {"status": "required", "valid": False, "message": "License required"}
    return {"status": "permanent", "valid": True, "message": "Permanently licensed."}


SPEC_DIR = Path(__file__).resolve().parent / "specs"


def _load_specs():
    """The real parameter documents captured from rc-astro 1.0.1 (tests/fake_tools/specs)."""
    specs = {}
    for key in ("bxt", "sxt", "nxt"):
        d = json.loads((SPEC_DIR / f"{key}.json").read_text())
        specs[key] = {k: v for k, v in d.items() if k not in ("schemaVersion", "cliVersion", "key", "license",
                                                              "devices")}
    return specs


SPECS = _load_specs()
DEVICES = json.loads((SPEC_DIR / "products.json").read_text())["devices"]


def out(o):
    print(json.dumps(o), flush=True)


def main(argv):
    if argv == ["--json"]:
        out({"schemaVersion": 4, "cliVersion": VERSION, "devices": DEVICES,
             "products": [{"key": k, "name": v["name"], "displayName": v["displayName"],
                           "mlVersions": [v["mlVersion"]], "license": lic(k)} for k, v in SPECS.items()]})
        return 0
    if not argv or argv[0] not in SPECS:
        print("Error: unknown product", file=sys.stderr)
        return 106
    key, rest = argv[0], argv[1:]
    if rest == ["--json"]:
        doc = {"schemaVersion": 4, "cliVersion": VERSION, "key": key, "license": lic(key), "devices": DEVICES}
        doc.update(SPECS[key])
        out(doc)
        return 0
    # processing run
    files = [a for a in rest if not a.startswith("-") and Path(a).suffix]
    outp = rest[rest.index("-o") + 1] if "-o" in rest else None
    inp = next(f for f in files if f != outp)
    out({"event": "info", "topic": "version", "cliVersion": VERSION, "schemaVersion": 4})
    if key in UNLICENSED:
        out({"event": "error", "message": f"{key} is not licensed on this machine."})
        return 77
    known = {p["flag"] for p in SPECS[key]["parameters"] if p.get("flag")} | \
            {m["flag"] for m in SPECS[key].get("modes", [])}
    for i, a in enumerate(rest):           # values must respect min/max
        p = next((p for p in SPECS[key]["parameters"] if p.get("flag") == a), None)
        if p and p["type"] in ("float", "int") and i + 1 < len(rest):
            v = float(rest[i + 1])
            if not p["min"] <= v <= p["max"]:
                print(f"Error: {a} out of range", file=sys.stderr)
                return 105
    with open(Path(outp).with_suffix(".args.json"), "w") as f:
        json.dump(rest, f)
    for a in rest:
        if a.startswith("--") and a.split("=")[0] not in known | {"--overwrite", "--json", "--device"}:
            print(f"Error: unknown option {a}", file=sys.stderr)
            return 109
    out({"event": "status", "phase": "initializing", "message": "Initializing"})
    out({"event": "info", "topic": "device", "device": "cpu", "name": "", "provider": "CPU", "runtime": ""})
    import numpy as np
    import stacking as st
    import fits_core as core
    src = st.ImageSource(inp)
    data = src.read() / 65535.0
    for d in (25.0, 50.0, 75.0, 100.0):
        out({"event": "progress", "done": d, "mpPerSec": 9.9, "eta": (100 - d) / 50})
        time.sleep(0.02)
    outputs = []
    if key == "nxt":
        data = data * 0.999 + np.float32(0.0)
    if key == "sxt":
        stars = np.clip(data - np.median(data), 0, None) * (data > np.percentile(data, 99.5))
        starless = data - stars
        outputs.append((outp, starless))
        if "--stars" in rest:
            sp = Path(outp)
            outputs.append((str(sp.with_name(sp.stem + "_stars" + sp.suffix)), stars))
    else:
        outputs.append((outp, data))
    for j, (path, arr) in enumerate(outputs):
        if FLIP:
            arr = arr[..., ::-1, :]
        core.write_fits(path, np.ascontiguousarray(arr).astype(np.float32), {"SWCREATE": "fake rc-astro"},
                        bitpix=-32)
        if j == 0 or not FLIP:
            out({"event": "status", "phase": "saving", "message": "Saving", "output": path})
            out({"event": "status", "phase": "complete", "message": "Done", "output": path})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
