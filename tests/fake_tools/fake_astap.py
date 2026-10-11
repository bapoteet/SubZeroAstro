"""Stand-in for ASTAP's command line used by the tests: 'solves' using the
-ra/-spd hint (or a fixed position) and writes <o>.wcs (80-character FITS
cards, no line breaks) and <o>.ini with PLTSOLVD=T."""
import sys
from pathlib import Path


def card(k, v, c=""):
    if isinstance(v, str):
        v = f"'{v:<8}'"
    elif isinstance(v, bool):
        v = "T" if v else "F"
    else:
        v = f"{v:.12E}" if isinstance(v, float) else str(v)
    s = f"{k:<8}= {v:>20}" + (f" / {c}" if c else "")
    return f"{s:<80}"[:80]


def main(argv):
    f = argv[argv.index("-f") + 1]
    o = argv[argv.index("-o") + 1] if "-o" in argv else str(Path(f).with_suffix(""))
    ra = float(argv[argv.index("-ra") + 1]) * 15 if "-ra" in argv else 23.46
    dec = float(argv[argv.index("-spd") + 1]) - 90 if "-spd" in argv else 30.66
    if "FAIL" in Path(f).name.upper():
        Path(o + ".ini").write_text("PLTSOLVD=F\nERROR=No solution found!\n")
        print("No solution found!  :(")
        return 1
    cards = [card("SIMPLE", True), card("CTYPE1", "RA---TAN"), card("CTYPE2", "DEC--TAN"),
             card("CRPIX1", 100.5), card("CRPIX2", 80.5), card("CRVAL1", ra, "RA of reference pixel (deg)"),
             card("CRVAL2", dec), card("CD1_1", -1.9e-4), card("CD1_2", 1.0e-6), card("CD2_1", 1.0e-6),
             card("CD2_2", 1.9e-4), card("PLTSOLVD", True, "Astrometric solved by fake ASTAP"), "END".ljust(80)]
    Path(o + ".wcs").write_text("".join(cards))
    Path(o + ".ini").write_text(f"PLTSOLVD=T\nCRVAL1={ra}\nCRVAL2={dec}\n")
    print(f"Solution found: {ra / 15:.4f} {dec:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
