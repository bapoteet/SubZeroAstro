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
        return "Wikipedia did not answer in time"
    if "Name or service not known" in t or "nodename nor servname" in t or "getaddrinfo" in t:
        return "no internet connection (Wikipedia's address could not be found)"
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


def note_text(target: str, title: str, text: str, url: str) -> str:
    name = targets.display_name(target) or title
    return (f"{name}\n{'=' * len(name)}\n\n{text.strip()}\n\n"
            f"Source: Wikipedia, \"{title}\" - {url}\nText licensed under {LICENCE}.\n")


def save(folder, target: str, content: str) -> Path:
    import checkpoints as cp
    p = Path(folder) / f"{cp.safe_name(target) or 'target'}_about.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
