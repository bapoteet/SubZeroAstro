"""Stand-in for GraXpert's command line used by the tests: subtracts a fitted
plane (background extraction) or smooths slightly (denoising) and writes
<output>.fits, like GraXpert's -cli mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv):
    import numpy as np
    import fits_core as core
    if "-cli" not in argv:
        print("GUI would start", file=sys.stderr)
        return 2
    inp = argv[0]
    outb = argv[argv.index("-output") + 1]
    cmd = argv[argv.index("-cmd") + 1] if "-cmd" in argv else "background-extraction"
    d, _c, look = core.read_fits(inp)
    print("Progress: 50%", flush=True)
    planes = d if d.ndim == 3 else d[None]
    res = []
    for p in planes:
        if cmd == "background-extraction":
            h, w = p.shape
            yy, xx = np.mgrid[0:h:8, 0:w:8]
            s = p[::8, ::8]
            A = np.c_[np.ones(s.size), xx.ravel(), yy.ravel()]
            coef, *_ = np.linalg.lstsq(A, s.ravel(), rcond=None)
            Y, X = np.mgrid[0:h, 0:w]
            bg = coef[0] + coef[1] * X + coef[2] * Y
            res.append(p - bg + np.median(bg))
        else:
            res.append(p * 0.999)
    out = np.stack(res) if d.ndim == 3 else res[0]
    core.write_fits(outb + ".fits", out.astype(np.float32), {"SWCREATE": "fake GraXpert"}, bitpix=-32)
    print("Progress: 100%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
