"""
cal_builder.py - "Build Calibration Masters": a folder of bias, dark, flat and dark-flat subframes
(all mixed together, or in subfolders) is sorted by frame type, camera, exposure, gain, offset,
temperature and filter; each set becomes a master, is checked (light leaks, a bias at zero, clipped
or dim flats) and is saved into the calibration library, where stacking finds it automatically.
No light frames are needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import stacking as st

ORDER = {"bias": 0, "dark": 1, "darkflat": 2, "flat": 3}


@dataclass
class Job:
    item: st.CalItem
    helper_bias: st.CalItem | None = None
    helper_dark: st.CalItem | None = None
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    files: list = field(default_factory=list)
    error: str = ""

    def label(self) -> str:
        it = self.item
        bits = [st.KIND_LABEL[it.kind]]
        if it.kind == "flat":
            bits.append(f"filter {it.filter}")
        if it.kind in ("dark", "darkflat") and it.exptime is not None:
            bits.append(f"{it.exptime:g} s")
        if it.gain is not None:
            bits.append(f"gain {it.gain:g}")
        if it.offset is not None and it.kind != "flat":
            bits.append(f"offset {it.offset:g}")
        if it.temp is not None and it.kind != "flat":
            bits.append(f"{it.temp:g} °C")
        if it.camera:
            bits.append(it.camera)
        return " · ".join(bits) + f"  ({len(it.infos)} subframes)"


def suggest_rig(folders, infos, rigs: st.Rigs | None = None) -> str:
    """A rig (telescope) name: from a folder called '<telescope>Calis' / '<telescope> Calibration',
    else from the frames' headers."""
    for f in folders:
        n = Path(f).name
        m = re.sub(r"[\s_\-]*(cali(bration)?s?|cal(s)?|calibration[\s_\-]*frames)$", "", n, flags=re.I).strip(" _-")
        if m and m.lower() != n.lower():
            return st.safe_name(m)
    if infos:
        return (rigs or st.Rigs()).name_for(infos[0])
    return "My scope"


def plan(infos, library=()) -> tuple[list, list]:
    """(jobs, skipped) from the infos of a calibration folder. Lights and masters are skipped (masters
    can still help calibrate the flats); the library's masters are used for the flats when the folder
    has no matching bias or dark-flats."""
    skipped = [i for i in infos if i.type not in st.KIND_FOLDER]
    items = st.group_calibration([i for i in infos if i.type in st.KIND_FOLDER])
    helpers = list(items) + list(library)
    jobs = []
    for it in items:
        if it.master:
            continue
        job = Job(it)
        if it.kind == "flat":
            fl = it.ref
            fd = [c for c in helpers if c.kind in ("darkflat", "dark") and st._same_cam(c.camera, fl.camera)
                  and c.exptime is not None and fl.exptime is not None
                  and abs(c.exptime - fl.exptime) <= 0.05 * max(fl.exptime, 0.01) + 0.002
                  and st._eq(c.gain, fl.gain) and st.geometry_ok(c, fl)]
            fb = [c for c in helpers if c.kind == "bias" and st._same_cam(c.camera, fl.camera)
                  and st._eq(c.gain, fl.gain) and st._eq(c.offset, fl.offset) and st.geometry_ok(c, fl)]
            pick = lambda cs: sorted(cs, key=lambda c: (c.master, -c.count))[0] if cs else None  # noqa: E731
            job.helper_dark = pick(fd)
            job.helper_bias = None if job.helper_dark else pick(fb)
            h = job.helper_dark or job.helper_bias
            if h is not None:
                where = "your library" if h.origin == "library" else "this folder"
                job.notes.append(f"calibrated with the {st.KIND_LABEL[h.kind].lower()}"
                                 f"{' master' if h.master else 's'} from {where}")
            else:
                job.warnings.append("no bias or dark-flats with the same gain - the camera offset stays in this "
                                    "flat (stacking will then over-correct the corners). Add a bias or dark-flats.")
        if len(it.infos) < 5:
            job.notes.append(f"only {len(it.infos)} subframes - 20 or more give a smoother master")
        jobs.append(job)
    jobs.sort(key=lambda j: (ORDER[j.item.kind], j.item.filter or "", j.item.exptime or 0))
    return jobs, skipped


def check_master(job: Job, jobs=(), library=()) -> list:
    """Warnings about a finished master: light leaks, a bias at zero, a dark brighter than the camera
    can make, clipped or dim flats."""
    it, m = job.item, job.item.data
    out = []
    if m is None:
        return out
    s = m[::4, ::4].astype(np.float32)
    med = float(np.median(s))
    if it.kind in ("bias", "dark", "darkflat"):
        spread, noise = st.master_unevenness(m)
        if spread > max(30.0, 6 * noise):
            out.append(f"it is uneven like a picture (varies by {spread:.0f} ADU across the frame; a real "
                       f"{st.KIND_LABEL[it.kind].lower()} is flat to within a few ADU) - light probably leaked in "
                       "(lens cap, filter wheel on an empty slot, focuser, room light or daylight). Retake with the "
                       "scope fully covered.")
        if it.kind == "bias" and med < 10 and (it.offset or 0) != 0:
            out.append(f"it is practically zero ({med:.2f} ADU) although the camera offset is {it.offset:g} - it was "
                       "probably saved on a 0..1 scale or with a pedestal taken off; stacking would leave the offset "
                       "in your pictures.")
        if it.kind in ("dark", "darkflat"):
            bias = [j.item for j in jobs if j.item.kind == "bias" and j.item.data is not None
                    and st._eq(j.item.gain, it.gain) and st._eq(j.item.offset, it.offset)
                    and st._same_cam(j.item.camera, it.camera)]
            if bias:
                b = float(np.median(bias[0].data[::4, ::4]))
                if b >= 10:
                    if med < b - max(5.0, 3 * noise):
                        out.append(f"it is darker than the bias ({med:.0f} vs {b:.0f} ADU) - the gain or offset "
                                   "does not match.")
                    elif med > b + max(50.0, 20 * noise):
                        out.append(f"it is {med - b:.0f} ADU brighter than the bias after {it.exptime or 0:g} s - far "
                                   "more than a cooled camera makes on its own. Light may have got in (or the camera "
                                   "was not cooled).")
    if it.kind == "flat":
        raw = st.ImageSource(it.ref.path).read()
        raw = (raw[0] if raw.ndim == 3 else raw)[::4, ::4].astype(np.float32)
        top = float(raw.max())
        clipped = float(np.mean(raw >= top)) if top > 0 else 0.0     # saturated pixels pile up at one value
        level = float(np.median(raw))
        if top >= 1000 and clipped > 0.01:
            out.append(f"{100 * clipped:.0f}% of a flat subframe is at its brightest value - the flats are "
                       "over-exposed (aim for about half of full brightness).")
        elif level < 500 and top > 1.5:
            out.append(f"the flat subframes are very dim ({level:.0f} ADU) - they are under-exposed and will "
                       "add noise (aim for about half of full brightness).")
    return out


def build(jobs, cal_root, rig, formats=("fits",), kappa=3.0, progress=None, cancel=None, log=print,
          library=()):
    """Build, check and save every job. Returns the jobs (with .files, .warnings, .error)."""
    n = len(jobs)
    keep = {id(j.helper_bias) for j in jobs} | {id(j.helper_dark) for j in jobs}
    for k, job in enumerate(jobs):
        it = job.item

        def prog(t, f, k=k):
            if progress:
                progress(t, (k + max(0.0, min(f, 1.0))) / max(n, 1))
            if cancel and cancel():
                raise st.Cancelled()
        log(f"\n— {job.label()}")
        try:
            st.build_master(it, job.helper_bias, job.helper_dark, kappa, progress=prog, cancel=cancel)
            job.warnings += check_master(job, jobs, library)
            job.files = st.save_calibration_master(it, cal_root, rig, formats)
            for f in job.files:
                log(f"  saved {f}")
            for nte in job.notes:
                log(f"  Note: {nte}")
            for w in job.warnings:
                log(f"  WARNING: {w}")
        except st.Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - one bad set never stops the others
            job.error = str(e)
            log(f"  FAILED: {e}")
        if id(it) not in keep and it.kind != "bias":
            it.data = None                  # masters are big: keep only those still needed
    if progress:
        progress("Done", 1.0)
    return jobs
