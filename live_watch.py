"""
live_watch.py - grading frames while the night is still going (File › Watch for New Frames).

The folder N.I.N.A. saves into (or the folder they are copied to) is checked every little while; each new frame is
measured as soon as it has finished writing and compared with the night's frames before it. Two poor frames in a
row raise an alert - clouds, dew, focus, guiding, trailing - and two good frames in a row say it has cleared.
This file holds the plain logic; the window part is in subzeroastro.py.
"""
from __future__ import annotations

import os
from pathlib import Path

POLL_S = 20             # how often the folder is looked at
SETTLE_S = 4            # a file is used once its size hasn't changed and it is at least this old
STREAK = 2              # poor (or good) frames in a row before saying so
MIN_REF = 4             # frames of the same camera + filter needed before comparing

KIND_TITLE = {"clouds": "Clouds?", "dew": "Dew or haze?", "bright": "Bright sky", "focus": "Soft stars",
              "trailing": "Trailed stars", "guiding": "Guiding trouble"}


def frame_key(path, folder=None) -> tuple:
    """Same frame whether it's in its folder or moved to REJECTED/ under it."""
    p = Path(path)
    return (os.path.normcase(os.path.abspath(str(folder if folder is not None else p.parent))), p.name.lower())


def settled(path, size, mtime, previous: dict, now: float) -> bool:
    """A file N.I.N.A. has finished writing: same size as last look and not touched for a few seconds."""
    prev = previous.get(str(path))
    previous[str(path)] = size
    return prev == size and size > 0 and now - mtime >= SETTLE_S


def problem(m: dict, gm: dict):
    """(kind, text) for a frame that is clearly worse than the night's median for its filter, else None."""
    if not m or not gm:
        return None
    if m.get("fwhm") is None or not gm.get("stars"):
        return "clouds", "No stars measured - clouds, or the dome/roof closed?"
    rs = m.get("stars", 0) / gm["stars"]
    rn = (m["snr"] / gm["snr"]) if m.get("snr") is not None and gm.get("snr") else 1.0
    rf = m["fwhm"] / gm["fwhm"] if gm.get("fwhm") else 1.0
    rb = (m["background"] / gm["background"]) if m.get("background") and gm.get("background") else 1.0
    e, em = m.get("eccentricity") or 0.0, gm.get("eccentricity") or 0.0
    g, gg = m.get("guide_rms"), gm.get("guide_rms")
    if rs < 0.5 or rn < 0.5:
        return "clouds", f"Stars {100 * rs:.0f}% / signal {100 * rn:.0f}% of the night's median - clouds?"
    if rf > 1.35 and rn < 0.8:
        return "dew", f"Soft, dim stars (FWHM +{100 * (rf - 1):.0f}%, signal {100 * rn:.0f}%) - dew or haze?"
    if rb > 1.6:
        return "bright", f"Background {100 * (rb - 1):.0f}% brighter - moon, dawn, a light, or cloud?"
    if g and gg and g > max(2.25 * gg, gg + 0.6):
        return "guiding", f"Guiding RMS {g:.2f}″ (night's median {gg:.2f}″) - wind, or lost the guide star?"
    if e > max(0.7, em + 0.2):
        return "trailing", f"Trailed stars (eccentricity {e:.2f}) - wind, cable snag or guiding?"
    if rf > 1.35:
        return "focus", f"Soft stars (FWHM +{100 * (rf - 1):.0f}%) - focus drift or seeing?"
    return None


class AlertState:
    """Feeds on each new frame's verdict; says when trouble starts and when it clears."""

    def __init__(self, streak=STREAK):
        self.streak = streak
        self.bad = []           # current run of poor frames: (name, kind, text)
        self.good = 0
        self.alerting = None    # kind while an alert is up
        self.log = []           # (name, kind or "", text) for every graded frame

    def feed(self, name, verdict):
        """-> ("alert", kind, text) / ("clear", "", text) / None."""
        self.log.append((name, verdict[0] if verdict else "", verdict[1] if verdict else "fine"))
        if verdict:
            self.good = 0
            self.bad.append((name, verdict[0], verdict[1]))
            if self.alerting is None and len(self.bad) >= self.streak:
                kinds = [k for _, k, _ in self.bad[-self.streak:]]
                kind = max(set(kinds), key=kinds.count)
                self.alerting = kind
                text = next(t for _, k, t in reversed(self.bad) if k == kind)
                return "alert", kind, f"{len(self.bad)} poor frames in a row (latest {name}): {text}"
            return None
        self.bad = []
        self.good += 1
        if self.alerting is not None and self.good >= self.streak:
            self.alerting = None
            return "clear", "", f"Looking good again - {self.good} good frames in a row (latest {name})."
        return None

    def summary(self) -> str:
        n = len(self.log)
        poor = sum(1 for _, k, _ in self.log if k)
        return f"{n} new frame(s) graded, {poor} poor" if n else "waiting for new frames"
