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
            5395, 5907, 6744, 6822, 6946, 7331, 7640, 7479,
            # more popular galaxies (southern and northern)
            134, 157, 613, 772, 1068, 1097, 1232, 1291, 1316, 1399, 1512, 1532, 1566, 1792, 2207, 2280, 2336, 2442,
            2525, 2775, 2976, 3166, 3169, 3190, 3198, 3310, 3359, 3486, 3556, 3587, 3623, 3627, 3718, 3726, 3953,
            3992, 4027, 4088, 4111, 4145, 4214, 4258, 4274, 4302, 4321, 4382, 4395, 4414, 4438, 4472, 4486, 4494,
            4535, 4569, 4594, 4618, 4625, 4627, 4725, 4736, 4826, 4945, 5055, 5068, 5078, 5194, 5195, 5236, 5248,
            5457, 5474, 5529, 5866, 6503, 6907, 7090, 7217, 7424, 7457, 7479, 7741, 7814}
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
                       r"[CPDXI]_?\d{4}_?[A-Z]{1,2}\d{0,3}\b|"
                       # a comet's year without the slash (a file or folder name can't hold one): C 2025 PANSTARRS,
                       # C_2025_Panstarrs, C2025-Lemmon
                       r"[CPDXI][ _\-]?(?:19|20)\d{2}(?:[ _\-]|$))", re.I)


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
    if OVERRIDES.get(_okey(n)) is not None:              # picked in Fast Track ("What is it?"), e.g. 'Snapshot'
        return OVERRIDES[_okey(n)] == COMET
    return bool(_COMET_RE.match(n)) or "comet" in n.lower()


# the kind of object the user picked for a target (Fast Track's "What is it?"), by name - wins over the catalogue
OVERRIDES: dict = {}

TYPE_CHOICES = [(GALAXY, "Galaxy"), (NEBULA, "Nebula (emission, reflection or dark)"),
                (PLANETARY, "Planetary nebula"), (CLUSTER, "Star cluster"), (COMET, "Comet"),
                (UNKNOWN, "Something else")]


def _okey(name) -> str:
    return re.sub(r"[\s_\-]+", "", str(name or "")).lower()


def set_override(name, kind):
    if kind is None:
        OVERRIDES.pop(_okey(name), None)
    else:
        OVERRIDES[_okey(name)] = kind


def classify(name: str) -> str:
    """The kind of object, or '' when the name doesn't tell."""
    n = (name or "").strip()
    if not n:
        return UNKNOWN
    if _okey(n) in OVERRIDES:
        return OVERRIDES[_okey(n)]
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


# Common names of popular targets, for the target line of the signature and the description lookup
COMMON_NAMES = {
    ("M", 1): "Crab Nebula", ("M", 8): "Lagoon Nebula", ("M", 13): "Hercules Cluster", ("M", 16): "Eagle Nebula",
    ("M", 17): "Omega Nebula", ("M", 20): "Trifid Nebula", ("M", 27): "Dumbbell Nebula", ("M", 31): "Andromeda Galaxy",
    ("M", 33): "Triangulum Galaxy", ("M", 42): "Orion Nebula", ("M", 44): "Beehive Cluster", ("M", 45): "Pleiades",
    ("M", 51): "Whirlpool Galaxy", ("M", 57): "Ring Nebula", ("M", 63): "Sunflower Galaxy", ("M", 64): "Black Eye Galaxy",
    ("M", 76): "Little Dumbbell Nebula", ("M", 78): "M78 Nebula", ("M", 81): "Bode's Galaxy", ("M", 82): "Cigar Galaxy",
    ("M", 83): "Southern Pinwheel Galaxy", ("M", 97): "Owl Nebula", ("M", 101): "Pinwheel Galaxy",
    ("M", 104): "Sombrero Galaxy", ("M", 106): "M106", ("NGC", 281): "Pacman Nebula", ("NGC", 891): "Silver Sliver Galaxy",
    ("NGC", 1499): "California Nebula", ("NGC", 2237): "Rosette Nebula", ("NGC", 2244): "Rosette Cluster",
    ("NGC", 2264): "Cone Nebula", ("NGC", 2359): "Thor's Helmet", ("NGC", 2392): "Eskimo Nebula",
    ("NGC", 3372): "Carina Nebula", ("NGC", 6543): "Cat's Eye Nebula", ("NGC", 6888): "Crescent Nebula",
    ("NGC", 6960): "Western Veil Nebula", ("NGC", 6992): "Eastern Veil Nebula", ("NGC", 6995): "Eastern Veil Nebula",
    ("NGC", 7000): "North America Nebula", ("NGC", 7293): "Helix Nebula", ("NGC", 7380): "Wizard Nebula",
    ("NGC", 7635): "Bubble Nebula", ("NGC", 6946): "Fireworks Galaxy", ("NGC", 4565): "Needle Galaxy",
    ("NGC", 253): "Sculptor Galaxy", ("NGC", 5128): "Centaurus A", ("NGC", 1977): "Running Man Nebula",
    ("NGC", 2024): "Flame Nebula", ("NGC", 6302): "Butterfly Nebula", ("NGC", 7023): "Iris Nebula",
    ("IC", 434): "Horsehead Nebula", ("IC", 1396): "Elephant's Trunk Nebula", ("IC", 1805): "Heart Nebula",
    ("IC", 1848): "Soul Nebula", ("IC", 5070): "Pelican Nebula", ("IC", 5146): "Cocoon Nebula",
    ("IC", 2118): "Witch Head Nebula", ("IC", 443): "Jellyfish Nebula", ("IC", 405): "Flaming Star Nebula",
    ("IC", 410): "Tadpoles Nebula", ("IC", 1318): "Butterfly Nebula (Sadr region)", ("IC", 63): "Ghost of Cassiopeia",
    ("SH2", 129): "Flying Bat Nebula", ("SH2", 240): "Spaghetti Nebula", ("SH2", 155): "Cave Nebula",
    ("SH2", 101): "Tulip Nebula",
}


def catalogue_name(name: str) -> str | None:
    """'NGC281' -> 'NGC 281', 'M42' -> 'M42', 'Sh2155' -> 'Sh2-155', or None for other names."""
    c = _cat(name)
    if not c:
        return None
    cat, num = c
    return {"M": f"M{num}", "NGC": f"NGC {num}", "IC": f"IC {num}", "SH2": f"Sh2-{num}"}[cat]


def common_name(name: str) -> str | None:
    c = _cat(name)
    return COMMON_NAMES.get(c) if c else None


def display_name(name: str) -> str:
    """A readable target name for the picture: 'NGC 281 "Pacman Nebula"', 'Elephant's Trunk Nebula'…"""
    n = (name or "").strip()
    if not n:
        return ""
    cat = catalogue_name(n)
    if cat:
        com = common_name(n)
        return f'{cat} "{com}"' if com and com != cat else cat
    # capture programs often squeeze names together: ElephantsTrunkNebula -> Elephants Trunk Nebula
    if " " not in n and re.search(r"[a-z][A-Z]", n):
        n = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n)
    return n.replace("_", " ")


def wiki_query(name: str) -> str:
    """What to look up on Wikipedia for a target."""
    n = (name or "").strip()
    c = _cat(n)
    if c:
        cat, num = c
        if (cat, num) in COMMON_NAMES and cat != "M":
            return COMMON_NAMES[(cat, num)]
        return {"M": f"Messier {num}", "NGC": f"NGC {num}", "IC": f"IC {num}", "SH2": f"Sh2-{num}"}[cat]
    return display_name(n)
