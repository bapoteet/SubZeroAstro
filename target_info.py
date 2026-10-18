"""
target_info.py - a short description of the target (the opening paragraphs of its Wikipedia
article), saved as a text file next to the finished pictures. Needs an internet connection; the
text is shown first so it can be edited, and the source is credited (Wikipedia text is CC BY-SA).
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

try:                         # the packaged app carries its own certificates (a Mac app can't use the system's)
    import certifi as _certifi
except ImportError:          # pragma: no cover
    _certifi = None

import targets

API = "https://en.wikipedia.org/w/api.php"
UA = "SubZeroAstro (https://github.com/bapoteet/SubZeroAstro) astrophotography app"
LICENCE = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"


def ssl_context() -> ssl.SSLContext:
    """Secure connections that work in the packaged app too: Python inside a Mac app finds none of the
    system's certificates, so the certifi bundle that ships with the app is used (plus the system's)."""
    ctx = ssl.create_default_context()
    if _certifi is not None:
        try:
            ctx.load_verify_locations(cafile=_certifi.where())
        except (OSError, ssl.SSLError):
            pass
    return ctx


def _get(params: dict, timeout=15) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
        return json.loads(r.read().decode("utf-8"))


def explain(err) -> str:
    """A plain-words reason a lookup failed."""
    t = str(err)
    if "CERTIFICATE_VERIFY_FAILED" in t or "certificate" in t.lower():
        return "the secure connection could not be checked (missing certificates)"
    if "timed out" in t.lower():
        return "the server did not answer in time"
    if "Name or service not known" in t or "nodename nor servname" in t or "getaddrinfo" in t:
        return "no internet connection (the server's address could not be found)"
    return t


def parse_result(data: dict, paragraphs: int = 2):
    """(title, text, url) from a MediaWiki extracts reply, or None."""
    pages = (data.get("query") or {}).get("pages") or {}
    if isinstance(pages, dict):
        pages = list(pages.values())
    pages = [p for p in pages if p.get("extract")]
    if not pages:
        return None
    p = sorted(pages, key=lambda q: q.get("index", 0))[0]
    title = p.get("title", "")
    paras = [x.strip() for x in re.split(r"\n+", p["extract"]) if x.strip()]
    text = "\n\n".join(paras[:paragraphs])
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return title, text, url


def lookup(target: str, paragraphs: int = 2, getter=_get):
    """(title, text, url) for a target name, or None when nothing is found. Raises OSError when
    Wikipedia can't be reached."""
    query = targets.wiki_query(target)
    if not query:
        return None
    base = {"action": "query", "format": "json", "prop": "extracts", "exintro": 1, "explaintext": 1,
            "redirects": 1}
    # the exact page first (M42 -> Orion Nebula via its redirect), then a search
    res = parse_result(getter({**base, "titles": query}), paragraphs)
    if res is None:
        res = parse_result(getter({**base, "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                                   "gsrnamespace": 0}), paragraphs)
    return res


# ==========================================================================
# SIMBAD facts (CDS, Strasbourg): object type, position, brightness - via the Sesame name resolver
# ==========================================================================

SESAME = "https://cds.unistra.fr/cgi-bin/nph-sesame/-ox/S?"
SIMBAD_CREDIT = "This research has made use of the SIMBAD database, operated at CDS, Strasbourg, France."

OTYPES = {
    "HII": "H II region (glowing hydrogen)", "PN": "planetary nebula", "SNR": "supernova remnant",
    "RNe": "reflection nebula", "DNe": "dark nebula", "EmO": "emission object", "ISM": "interstellar matter",
    "Cld": "cloud", "MoC": "molecular cloud", "GNe": "nebula", "BNe": "bright nebula", "Neb": "nebula",
    "OpC": "open star cluster", "GlC": "globular star cluster", "Cl*": "star cluster", "As*": "stellar association",
    "G": "galaxy", "GiG": "galaxy in a group", "GiC": "galaxy in a cluster", "IG": "interacting galaxies",
    "PaG": "pair of galaxies", "GrG": "group of galaxies", "ClG": "cluster of galaxies", "SBG": "starburst galaxy",
    "EmG": "emission-line galaxy", "AGN": "active galaxy", "Sy1": "Seyfert galaxy", "Sy2": "Seyfert galaxy",
    "SyG": "Seyfert galaxy", "LIN": "LINER galaxy", "rG": "radio galaxy", "BiC": "brightest galaxy in a cluster",
    "QSO": "quasar", "*": "star", "**": "double star", "WR*": "Wolf-Rayet star", "Be*": "Be star",
    "Em*": "emission-line star", "C*": "carbon star", "V*": "variable star", "Ce*": "Cepheid variable star",
    "Mi*": "Mira variable star", "SN*": "supernova", "No*": "nova", "Comet": "comet",
}


def _sesame_get(name: str, timeout=15) -> str:
    url = SESAME + urllib.parse.quote(name)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
        return r.read().decode("utf-8", "replace")


def simbad_query(target: str) -> str:
    """How SIMBAD knows the target: 'M 42', 'NGC 7000', 'Sh 2-155' …"""
    c = targets._cat(target)
    if c:
        cat, num = c
        return {"M": f"M {num}", "NGC": f"NGC {num}", "IC": f"IC {num}", "SH2": f"Sh 2-{num}"}[cat]
    return targets.display_name(target)


def _hms(deg: float, hours=True) -> str:
    v = deg / 15.0 if hours else abs(deg)
    a = int(v)
    b = int((v - a) * 60)
    c = ((v - a) * 60 - b) * 60
    if hours:
        return f"{a:02d}h {b:02d}m {c:04.1f}s"
    return f"{'-' if deg < 0 else '+'}{a:02d}° {b:02d}′ {c:02.0f}″"


def parse_sesame(xml: str):
    """dict(name, type, type_code, ra, dec, vmag) from a Sesame XML reply, or None when SIMBAD
    does not know the object."""
    import xml.etree.ElementTree as ET
    xml = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", xml or "")        # namespaces would hide the tags
    xml = re.sub(r"<(/?)\w+:", r"<\1", xml)
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except ET.ParseError:
        return None
    res = root.find(".//Resolver")
    if res is None or res.find("jradeg") is None:
        return None

    def t(tag):
        e = res.find(tag)
        return e.text.strip() if e is not None and e.text else ""
    out = {"name": " ".join(t("oname").split()), "type_code": t("otype")}
    out["type"] = OTYPES.get(out["type_code"], out["type_code"])
    try:
        out["ra"], out["dec"] = float(t("jradeg")), float(t("jdedeg"))
    except ValueError:
        return None
    out["vmag"] = None
    for m in res.findall("mag"):
        if (m.get("band") or "").upper() == "V":
            v = m.find("v")
            try:
                out["vmag"] = float(v.text) if v is not None and v.text else None
            except ValueError:
                pass
    return out


def lookup_simbad(target: str, getter=_sesame_get):
    q = simbad_query(target)
    if not q:
        return None
    return parse_sesame(getter(q))


def facts_text(f: dict) -> str:
    bits = []
    if f.get("type"):
        bits.append(f"Type: {f['type']}")
    bits.append(f"Position (J2000): RA {_hms(f['ra'])}, Dec {_hms(f['dec'], hours=False)}")
    if f.get("vmag") is not None:
        bits.append(f"Brightness: magnitude {f['vmag']:.1f} (V)")
    name = f" ({f['name']})" if f.get("name") else ""
    return f"Facts from SIMBAD{name}:\n" + "\n".join(f"  • {b}" for b in bits)


def note_text(target: str, title: str = "", text: str = "", url: str = "", facts: dict | None = None) -> str:
    name = targets.display_name(target) or title or (facts or {}).get("name", "")
    out = f"{name}\n{'=' * len(name)}\n\n"
    if text:
        out += text.strip() + "\n\n"
    if facts:
        out += facts_text(facts) + "\n\n"
    src = []
    if text:
        src.append(f"Source: Wikipedia, \"{title}\" - {url}\nText licensed under {LICENCE}.")
    if facts:
        src.append("Facts: SIMBAD, https://simbad.cds.unistra.fr - " + SIMBAD_CREDIT)
    return out + "\n".join(src) + "\n"


def lookup_all(target: str):
    """(wiki result or None, simbad facts or None, [problems]) - each source on its own; one failing
    does not stop the other."""
    wiki = facts = None
    problems = []
    try:
        wiki = lookup(target)
    except Exception as e:  # noqa: BLE001
        problems.append(f"Wikipedia: {explain(e)}")
    try:
        facts = lookup_simbad(target)
    except Exception as e:  # noqa: BLE001
        problems.append(f"SIMBAD: {explain(e)}")
    return wiki, facts, problems


def save(folder, target: str, content: str) -> Path:
    import checkpoints as cp
    p = Path(folder) / f"{cp.safe_name(target) or 'target'}_about.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
