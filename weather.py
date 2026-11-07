"""
weather.py - My Weather Forecast: the next few nights at your site, hour by hour through the dark, and a bottom
line for each night.

Sources (free, no account): Open-Meteo (open-meteo.com, CC BY 4.0) for cloud at three heights, rain, dew point,
wind and temperature; 7Timer! ASTRO (7timer.info) for astronomical seeing and transparency. The dark hours come from
the same Sun and Moon positions Tonight's Target Picker uses.
"""
from __future__ import annotations

import datetime as _dt
import json
import urllib.request

OPEN_METEO = ("https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
              "&hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,precipitation_probability,"
              "relative_humidity_2m,dew_point_2m,temperature_2m,wind_speed_10m,wind_gusts_10m"
              "&timezone=UTC&forecast_days={days}")
SEVEN_TIMER = ("https://www.7timer.info/bin/api.pl?lon={lon:.3f}&lat={lat:.3f}&product=astro&output=json",
               "http://www.7timer.info/bin/api.pl?lon={lon:.3f}&lat={lat:.3f}&product=astro&output=json")
UA = {"User-Agent": "SubZeroAstro"}

# 7Timer's scales: 1 = best .. 8 = worst
SEEING_ARCSEC = {1: "<0.5″", 2: "0.5–0.75″", 3: "0.75–1″", 4: "1–1.25″", 5: "1.25–1.5″", 6: "1.5–2″",
                 7: "2–2.5″", 8: ">2.5″"}
TRANSP_WORD = {1: "superb", 2: "excellent", 3: "very good", 4: "good", 5: "average", 6: "below average",
               7: "poor", 8: "very poor"}

GREAT, GOOD, POOR = "great", "good", "poor"
HEADLINE = {GREAT: "Looks like a great night for imaging!",
            GOOD: "Looks like a good but not perfect night - expect some clouds or so-so conditions.",
            POOR: "Not a good night for imaging - but a great night for processing with SubZeroAstro."}


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8", "replace"))


def _utc(s) -> _dt.datetime:
    t = _dt.datetime.fromisoformat(str(s).replace("Z", ""))
    return t.replace(tzinfo=_dt.timezone.utc) if t.tzinfo is None else t.astimezone(_dt.timezone.utc)


def parse_open_meteo(d) -> dict:
    """{UTC hour: {cloud, low, mid, high, rain, rh, dew, temp, wind, gust}} from Open-Meteo's hourly answer."""
    h = (d or {}).get("hourly") or {}
    keys = {"cloud": "cloud_cover", "low": "cloud_cover_low", "mid": "cloud_cover_mid", "high": "cloud_cover_high",
            "rain": "precipitation_probability", "rh": "relative_humidity_2m", "dew": "dew_point_2m",
            "temp": "temperature_2m", "wind": "wind_speed_10m", "gust": "wind_gusts_10m"}
    out = {}
    for i, t in enumerate(h.get("time") or []):
        row = {}
        for k, src in keys.items():
            v = (h.get(src) or [None] * (i + 1))
            v = v[i] if i < len(v) else None
            row[k] = float(v) if isinstance(v, (int, float)) else None
        out[_utc(t)] = row
    return out


def parse_7timer(d) -> dict:
    """{UTC time: {seeing 1-8, transparency 1-8}} from 7Timer's ASTRO answer (3-hourly)."""
    out = {}
    try:
        init = _dt.datetime.strptime(str(d["init"]), "%Y%m%d%H").replace(tzinfo=_dt.timezone.utc)
    except (KeyError, TypeError, ValueError):
        return out
    for r in d.get("dataseries") or []:
        try:
            t = init + _dt.timedelta(hours=int(r["timepoint"]))
        except (KeyError, TypeError, ValueError):
            continue
        s, tr = r.get("seeing"), r.get("transparency")
        out[t] = {"seeing": int(s) if isinstance(s, int) and 1 <= s <= 8 else None,
                  "transp": int(tr) if isinstance(tr, int) and 1 <= tr <= 8 else None}
    return out


def merge(hours: dict, astro: dict) -> dict:
    """Each hour gets the seeing / transparency of the nearest 7Timer time (within 2 hours)."""
    if not astro:
        return hours
    ts = sorted(astro)
    for t, row in hours.items():
        near = min(ts, key=lambda x: abs((x - t).total_seconds()))
        if abs((near - t).total_seconds()) <= 2 * 3600:
            row.update(astro[near])
    return hours


def fetch(lat, lon, days=4) -> tuple:
    """(hours, notes): the merged hourly forecast and what couldn't be fetched."""
    notes = []
    try:
        hours = parse_open_meteo(_get(OPEN_METEO.format(lat=lat, lon=lon, days=days)))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"couldn't get the forecast from Open-Meteo ({e}) - is this computer online?") from e
    astro = {}
    for url in SEVEN_TIMER:
        try:
            astro = parse_7timer(_get(url.format(lat=lat, lon=lon)))
            if astro:
                break
        except Exception as e:  # noqa: BLE001
            notes.append(f"seeing and transparency unavailable (7Timer: {e})")
    if astro and notes:
        notes = []
    return merge(hours, astro), notes


def night_hours(hours: dict, lat, lon, date: _dt.date) -> list:
    """The forecast hours that fall in the dark of the night starting on the evening of `date`:
    [(UTC time, row)], plus the night's dark start / end and Moon."""
    import target_picker as tp
    nt = tp.night(lat, lon, date)
    ds, de = nt.dark_start, nt.dark_end
    rows = []
    if ds is not None and de is not None:
        rows = [(t, r) for t, r in sorted(hours.items()) if ds - _dt.timedelta(minutes=30) <= t <= de]
    return rows, nt


def _runs(flags):
    best = cur = 0
    start = best_start = None
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def verdict(rows) -> dict:
    """The bottom line for one night: {level, headline, reasons, clear_hours, window: (start, end) or None,
    cloud, seeing, dew}."""
    if not rows:
        return {"level": POOR, "headline": "No forecast for the dark hours of this night.", "reasons": [],
                "clear_hours": 0, "window": None, "cloud": None, "seeing": None, "dew": False}
    clouds = [r.get("cloud") if r.get("cloud") is not None else 100.0 for _, r in rows]
    clear = [c <= 25 for c in clouds]
    n = len(rows)
    mean_cloud = sum(clouds) / n
    run, start = _runs(clear)
    window = (rows[start][0], rows[start + run - 1][0] + _dt.timedelta(hours=1)) if run else None
    rain = max((r.get("rain") or 0) for _, r in rows)
    gust = max((r.get("gust") or 0) for _, r in rows)
    spreads = [r["temp"] - r["dew"] for _, r in rows if r.get("temp") is not None and r.get("dew") is not None]
    dew = bool(spreads) and sum(1 for s in spreads if s < 2.0) >= max(1, len(spreads) // 3)
    see = [r["seeing"] for _, r in rows if r.get("seeing")]
    tra = [r["transp"] for _, r in rows if r.get("transp")]
    seeing = sum(see) / len(see) if see else None
    transp = sum(tra) / len(tra) if tra else None
    reasons = []
    frac = sum(clear) / n
    if frac >= 0.75:
        reasons.append(f"clear most of the night ({sum(clear)} of {n} dark hours, cloud {mean_cloud:.0f}% on average)")
    elif run:
        reasons.append(f"clear for {run} hour(s) in a row, cloudy otherwise (cloud {mean_cloud:.0f}% on average)")
    else:
        reasons.append(f"cloudy all night (cloud {mean_cloud:.0f}% on average)")
    if rain >= 40:
        reasons.append(f"rain likely ({rain:.0f}% chance)")
    if gust >= 35:
        reasons.append(f"gusty wind (up to {gust:.0f} km/h) - expect wobbly stars")
    if dew:
        reasons.append("damp air (temperature close to the dew point) - run your dew heaters")
    if seeing is not None and seeing >= 5.5:
        reasons.append(f"poor seeing (about {SEEING_ARCSEC[round(seeing)]}) - soft stars at long focal lengths")
    if transp is not None and transp >= 6:
        reasons.append(f"hazy sky (transparency {TRANSP_WORD[round(transp)]})")

    if rain >= 60 or frac < 0.25 and run < 3:
        level = POOR
    elif frac >= 0.75 and rain < 30 and gust < 35 and (seeing is None or seeing < 5) and (transp is None or transp < 5):
        level = GREAT
    elif frac >= 0.4 or run >= 3:
        level = GOOD
    else:
        level = POOR
    return {"level": level, "headline": HEADLINE[level], "reasons": reasons, "clear_hours": sum(clear),
            "window": window if run >= 2 else None, "cloud": mean_cloud, "seeing": seeing, "transp": transp,
            "dew": dew}


def nights(hours: dict, lat, lon, first: _dt.date, count=3) -> list:
    """[(date, rows, night, verdict)] for `count` nights starting on the evening of `first`."""
    out = []
    for k in range(count):
        d = first + _dt.timedelta(days=k)
        rows, nt = night_hours(hours, lat, lon, d)
        if not rows and k > 0:
            break
        out.append((d, rows, nt, verdict(rows)))
    return out
