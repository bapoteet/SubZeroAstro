"""
targets.py - what kind of object a target is: galaxy, nebula, planetary nebula, star cluster or
comet - from its name (catalogue number or common name), as written by the capture software.

Used to tailor the recommendations (never to take steps away): comets get a comet-aligned stack
on Calibrate and Stack, galaxies their HII regions and structure, clusters keep their stars …
"""

from __future__ import annotations

import re

GALAXY, NEBULA, PLANETARY, CLUSTER, COMET, UNKNOWN = ("galaxy", "nebula", "planetary nebula", "star cluster",
                                                     "comet", "")

# Messier objects by type (S = supernova remnant, counted as nebula)
_M_GAL = {31, 32, 33, 49, 51, 58, 59, 60, 61, 63, 64, 65, 66, 74, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
          94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 108, 109, 110}
_M_NEB = {1, 8, 16, 17, 20, 42, 43, 78}
_M_PN = {27, 57, 76, 97}
# everything else in Messier's list is a star cluster (or a star cloud / asterism)

_NGC_GAL = {55, 247, 253, 300, 891, 925, 1023, 1300, 1365, 2403, 2683, 2841, 2903, 3077, 3115, 3184, 3344, 3521,
            3628, 3938, 4038, 4039, 4216, 4236, 4244, 4449, 4490, 4559, 4565, 4605, 4631, 4656, 5005, 5033, 5128,
            5395, 5907, 6744, 6822, 6946, 7331, 7640, 7479}
_NGC_NEB = {281, 896, 1333, 1491, 1499, 1579, 1931, 2023, 2070, 2174, 2237, 2238, 2239, 2244, 2246, 2264, 2359,
            3372, 6188, 6334, 6357, 6559, 6820, 6888, 6914, 6960, 6979, 6992, 6995, 7000, 7023, 7129, 7380, 7635,
            7822, 1977, 1973, 1975, 2149, 6302}
_NGC_PN = {40, 246, 1501, 1514, 2022, 2346, 2371, 2392, 2438, 2440, 3132, 3242, 6210, 6302, 6543, 6720, 6781,
           6826, 6853, 7009, 7027, 7293, 7662, 650, 651}
_NGC_CL = {869, 884, 457, 663, 752, 1502, 2158, 2169, 2362, 2451, 2477, 3532, 4755, 5139, 6231, 6397, 6752, 6791,
           6819, 6940, 7789, 104, 362}
_IC_GAL = {10, 342, 1613, 2574, 5332}
_IC_NEB = {59, 63, 348, 405, 410, 417, 434, 443, 1318, 1396, 1795, 1805, 1848, 2118, 2177, 4592, 4628, 5067,
           5070, 5146, 1284, 2944, 4604, 4685, 4703}

_WORDS = [
    (COMET, ("comet",)),
    (PLANETARY, ("planetary", "ring nebula", "dumbbell", "owl nebula", "cat's eye", "cats eye", "helix",
                 "saturn nebula", "eskimo", "little dumbbell", "blinking", "ghost of jupiter")),
    (GALAXY, ("galaxy", "galaxies", "andromeda", "triangulum", "whirlpool", "pinwheel", "bode", "cigar",
              "sombrero", "leo triplet", "needle", "fireworks", "sculptor", "black eye", "blackeye", "sunflower",
              "hamburger", "silver dollar", "cartwheel", "antennae", "stephan", "markarian", "owl galaxy")),
    (CLUSTER, ("cluster", "pleiades", "beehive", "praesepe", "hyades", "omega centauri", "jewel box",
               "wild duck", "owl cluster", "hercules")),
    (NEBULA, ("nebula", "trunk", "veil", "rosette", "heart", "soul", "horsehead", "pelican", "north america",
              "eagle", "lagoon", "trifid", "orion", "running man", "crescent", "bubble", "pacman", "pac-man",
              "california", "cone", "christmas tree", "wizard", "cocoon", "witch head", "jellyfish",
              "flaming star", "sadr", "spaghetti", "simeis", "squid", "flying bat", "iris", "seagull",
              "monkey head", "tadpoles", "fish head", "medulla", "soap bubble", "cave", "ghost", "elephant",
              "sh2", "sh 2", "sh-2", "ldn", "lbn", "vdb", "barnard", "ced", "gum", "rcw")),
]

_COMET_RE = re.compile(r"^\s*(\d{1,4}\s*P\b|\d{1,4}P(?=[A-Za-z/_\- ])|\d{1,4}P$|[CPDXIA]\s*/\s*\d{4}\b|"
                       r"[CPDXI]_?\d{4}_?[A-Z]{1,2}\d{0,3}\b)", re.I)


def _cat(name: str):
    """('M'|'NGC'|'IC'|'SH2', number) from a target name, or None."""
    key = re.sub(r"[^a-z0-9]", "", (name or "").lower()).upper()
    key = re.sub(r"^MESSIER", "M", key)
    key = re.sub(r"^SHARPLESS", "SH", key)
    m = re.match(r"^(M|NGC|IC|SH2|SH)0*(\d+)", key)
    if not m:
        return None
    cat = "SH2" if m.group(1) in ("SH", "SH2") else m.group(1)
    num = m.group(2)
    if m.group(1) == "SH" and num.startswith("2") and len(num) > 1:   # "Sh2155" -> Sh2-155
        num = num[1:]
    return cat, int(num)


def is_comet(name: str) -> bool:
    n = (name or "").strip()
    return bool(_COMET_RE.match(n)) or "comet" in n.lower()


def classify(name: str) -> str:
    """The kind of object, or '' when the name doesn't tell."""
    n = (name or "").strip()
    if not n:
        return UNKNOWN
    if is_comet(n):
        return COMET
    c = _cat(n)
    if c:
        cat, num = c
        if cat == "M":
            return (GALAXY if num in _M_GAL else NEBULA if num in _M_NEB else PLANETARY if num in _M_PN
                    else CLUSTER if 1 <= num <= 110 else UNKNOWN)
        if cat == "NGC":
            for typ, ids in ((GALAXY, _NGC_GAL), (PLANETARY, _NGC_PN), (NEBULA, _NGC_NEB), (CLUSTER, _NGC_CL)):
                if num in ids:
                    return typ
        if cat == "IC":
            for typ, ids in ((GALAXY, _IC_GAL), (NEBULA, _IC_NEB)):
                if num in ids:
                    return typ
        if cat == "SH2":
            return NEBULA
    low = n.lower()
    for typ, words in _WORDS:
        if any(w in low for w in words):
            return typ
    return UNKNOWN


def is_galaxy(name: str) -> bool:
    return classify(name) == GALAXY


DESCRIBE = {GALAXY: "a galaxy", NEBULA: "a nebula", PLANETARY: "a planetary nebula", CLUSTER: "a star cluster",
            COMET: "a comet", UNKNOWN: ""}
