"""
weather_window.py - My Weather Forecast: a bottom line for each of the next nights, and the dark hours hour by hour
(cloud at three heights, rain, seeing, transparency, wind, dew).
"""
from __future__ import annotations

import datetime as _dt

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

import weather as W

LEVEL_COLOR = {W.GREAT: "#2f9e57", W.GOOD: "#c9921e", W.POOR: "#b8423f"}
ROWS = [("cloud", "Cloud (all)", "%"), ("low", "  low", "%"), ("mid", "  middle", "%"), ("high", "  high", "%"),
        ("rain", "Rain chance", "%"), ("seeing", "Seeing", ""), ("transp", "Transparency", ""),
        ("wind", "Wind (gusts)", "km/h"), ("temp", "Temperature", "°C"), ("spread", "Above dew point", "°C")]


def _shade(good_0_to_1):
    """Clear-sky-chart colours: deep blue = good, white = bad."""
    g = max(0.0, min(1.0, good_0_to_1))
    return QColor(int(255 - 225 * g), int(255 - 175 * g), int(255 - 55 * g))


def cell(key, r):
    """(text, colour or None) for one hour."""
    v = r.get(key) if key not in ("spread", "wind") else None
    if key in ("cloud", "low", "mid", "high"):
        return ("" if r.get(key) is None else f"{r[key]:.0f}"), (None if r.get(key) is None
                                                                 else _shade(1 - r[key] / 100))
    if key == "rain":
        return ("" if v is None else f"{v:.0f}"), (None if v is None else _shade(1 - v / 100))
    if key == "seeing":
        return (W.SEEING_ARCSEC.get(v, "") if v else ""), (_shade((8 - v) / 7) if v else None)
    if key == "transp":
        return (W.TRANSP_WORD.get(v, "") if v else ""), (_shade((8 - v) / 7) if v else None)
    if key == "wind":
        w, g = r.get("wind"), r.get("gust")
        if w is None:
            return "", None
        return f"{w:.0f} ({g:.0f})" if g is not None else f"{w:.0f}", _shade(1 - min((g or w) / 50, 1))
    if key == "temp":
        return ("" if v is None else f"{v:.0f}"), None
    if key == "spread":
        t, d = r.get("temp"), r.get("dew")
        if t is None or d is None:
            return "", None
        s = t - d
        return f"{s:.0f}", (QColor("#e7b3b3") if s < 2 else None)
    return "", None


class _Fetch(QThread):
    done = Signal(object)

    def __init__(self, lat, lon, parent=None):
        super().__init__(parent)
        self.lat, self.lon = lat, lon

    def run(self):
        try:
            self.done.emit(W.fetch(self.lat, self.lon))
        except Exception as e:  # noqa: BLE001
            self.done.emit(e)


class WeatherDialog(QDialog):
    def __init__(self, owner, settings, auto=True):
        super().__init__(owner)
        self.owner, self.settings = owner, settings
        self.setWindowTitle("My Weather Forecast")
        lay = QVBoxLayout(self)
        h = QHBoxLayout()
        h.addWidget(QLabel("Latitude"))
        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90, 90)
        self.lat.setDecimals(4)
        self.lat.setValue(float(settings.value("picker/lat", 0.0) or 0.0))
        h.addWidget(self.lat)
        h.addWidget(QLabel("Longitude"))
        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180, 180)
        self.lon.setDecimals(4)
        self.lon.setValue(float(settings.value("picker/lon", 0.0) or 0.0))
        h.addWidget(self.lon)
        self.b_here = QPushButton("Use My Location")
        self.b_here.clicked.connect(lambda: self.use_my_location())
        h.addWidget(self.b_here)
        self.b_go = QPushButton("Get Forecast")
        self.b_go.clicked.connect(lambda: self.refresh())
        h.addWidget(self.b_go)
        h.addStretch(1)
        lay.addLayout(h)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        credit = QLabel("Forecast: <a href='https://open-meteo.com/'>Open-Meteo</a> (cloud, rain, wind, dew; CC BY "
                        "4.0) and <a href='https://www.7timer.info/'>7Timer!</a> (seeing, transparency). Only the dark "
                        "hours are shown (sun 18° below the horizon). Forecasts are most reliable for the next two or three nights.")
        credit.setOpenExternalLinks(True)
        credit.setWordWrap(True)
        credit.setStyleSheet("color:#8a8a96;")
        lay.addWidget(credit)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(1100, 640)
        self.result = None
        if auto and (self.lat.value() or self.lon.value()):
            self.refresh()
        elif auto:
            self.status.setText("Where are you? Press <b>Use My Location</b>, or type your latitude and longitude.")

    # ------------------------------------------------------------------ location / fetch
    def use_my_location(self, answer=None):
        import target_picker as tp
        if answer is None:
            class _Find(QThread):
                found = Signal(object)

                def run(self):
                    ll, _why = tp.locate_windows()
                    if ll:
                        self.found.emit((ll[0], ll[1]))
                        return
                    got, _why2 = tp.locate_ip()
                    self.found.emit((got[0], got[1]) if got else (None, None))
            self.status.setText("Finding where you are…")
            self._find = _Find(self)
            self._find.found.connect(lambda a: self.use_my_location(a))
            self._find.start()
            return None
        lat, lon = answer
        if lat is None:
            self.status.setText("Couldn't find your location - type your latitude and longitude.")
            return False
        self.lat.setValue(round(float(lat), 4))
        self.lon.setValue(round(float(lon), 4))
        self.refresh()
        return True

    def refresh(self, data=None):
        """Fetch the forecast (in the background), or show `data` = (hours, notes) straight away (self-test)."""
        lat, lon = self.lat.value(), self.lon.value()
        if self.settings is not None:
            self.settings.setValue("picker/lat", lat)
            self.settings.setValue("picker/lon", lon)
        if data is not None:
            return self.show_forecast(data)
        self.status.setText("Getting the forecast…")
        self.b_go.setEnabled(False)
        self._fetch = _Fetch(lat, lon, self)
        self._fetch.done.connect(self.show_forecast)
        self._fetch.start()
        return None

    # ------------------------------------------------------------------ show
    def show_forecast(self, data):
        self.b_go.setEnabled(True)
        if isinstance(data, Exception):
            self.status.setText(f"<span style='color:#eba932'>{data}</span>")
            return None
        hours, notes = data
        today = _dt.datetime.now().date()
        if _dt.datetime.now().hour < 9:                 # after midnight: "tonight" is the night already going
            today -= _dt.timedelta(days=1)
        self.result = W.nights(hours, self.lat.value(), self.lon.value(), today, 7)
        self.tabs.clear()
        self.tabs.addTab(self._week_tab(self.result), "The Week Ahead")
        for i, (d, rows, nt, v) in enumerate(self.result):
            self.tabs.addTab(self._night_tab(d, rows, nt, v, i), f"{self._label(i, d)}  ●")
            self.tabs.tabBar().setTabTextColor(i + 1, QColor(LEVEL_COLOR[v["level"]]))
        best = self.result[0][3] if self.result else None
        self.status.setText(("<b>Tonight:</b> " + best["headline"] if best else "No forecast.")
                            + ("<br><span style='color:#8a8a96'>" + "; ".join(notes) + "</span>" if notes else ""))
        return self.result

    @staticmethod
    def _label(i, d):
        return "Tonight" if i == 0 else ("Tomorrow" if i == 1 else d.strftime("%a %b %d"))

    def _week_tab(self, result):
        """One line per night: the bottom line, clear dark hours, best window, Moon, how sure the forecast is."""
        w = QWidget()
        lay = QVBoxLayout(w)
        good = [self._label(i, d) for i, (d, _r, _n, v) in enumerate(result) if v["level"] == W.GREAT]
        lead = QLabel("<b>Best nights for imaging this week:</b> " + (", ".join(good) if good else
                      "none look great yet - forecasts more than a few days out change a lot, so check again.")
                      if result else "No forecast.")
        lead.setWordWrap(True)
        lay.addWidget(lead)
        t = QTableWidget(len(result), 6)
        t.setHorizontalHeaderLabels(["Night", "Bottom line", "Clear dark hours", "Best window", "Moon",
                                     "Forecast confidence"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        loc = (lambda x: x.astimezone().strftime("%H:%M") if x else "")
        for i, (d, rows, nt, v) in enumerate(result):
            vals = [f"{self._label(i, d)} ({d.strftime('%b %d')})", v["headline"],
                    f"{v['clear_hours']} of {len(rows)}",
                    f"{loc(v['window'][0])} – {loc(v['window'][1])}" if v.get("window") else "",
                    f"{nt.moon_lit * 100:.0f}% lit", W.confidence(i)]
            for c, text in enumerate(vals):
                it = QTableWidgetItem(text)
                if c == 1:
                    it.setBackground(QColor(LEVEL_COLOR[v["level"]]))
                    it.setForeground(QColor("white"))
                t.setItem(i, c, it)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        t.cellDoubleClicked.connect(lambda r, _c: self.tabs.setCurrentIndex(r + 1))
        lay.addWidget(t, 1)
        hint = QLabel("Double-click a night for its hours. Seeing and transparency reach about three days ahead.")
        hint.setStyleSheet("color:#8a8a96;")
        lay.addWidget(hint)
        return w

    def _night_tab(self, d, rows, nt, v, k=0):
        w = QWidget()
        lay = QVBoxLayout(w)
        col = LEVEL_COLOR[v["level"]]
        head = QLabel(f"<span style='font-size:16pt; font-weight:bold; color:white'>{v['headline']}</span>")
        head.setStyleSheet(f"QLabel {{ background:{col}; border-radius:6px; padding:10px; }}")
        head.setWordWrap(True)
        lay.addWidget(head)
        loc = (lambda t: t.astimezone().strftime("%H:%M") if t else "–")
        info = [f"Dark from {loc(nt.dark_start)} to {loc(nt.dark_end)} ({nt.dark_hours:.1f} h)",
                f"Moon {nt.moon_lit * 100:.0f}% lit", f"forecast confidence: {W.confidence(k)}"]
        if v.get("window"):
            info.append(f"<b>Best window: {loc(v['window'][0])} – {loc(v['window'][1])}</b>")
        details = QLabel(" · ".join(info) + "<br>" + "<br>".join("• " + r for r in v["reasons"]))
        details.setWordWrap(True)
        lay.addWidget(details)
        t = QTableWidget(len(ROWS), len(rows))
        t.setVerticalHeaderLabels([f"{lab} {unit}".rstrip() for _, lab, unit in ROWS])
        t.setHorizontalHeaderLabels([tm.astimezone().strftime("%H") for tm, _ in rows])
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        for c, (_tm, r) in enumerate(rows):
            for ri, (key, _lab, _u) in enumerate(ROWS):
                text, colour = cell(key, r)
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignCenter)
                if colour is not None:
                    it.setBackground(colour)
                    it.setForeground(QColor("white") if colour.lightness() < 140 else QColor("black"))
                t.setItem(ri, c, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay.addWidget(t, 1)
        return w
