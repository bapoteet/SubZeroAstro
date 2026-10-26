"""File › Tonight's Target Picker… - what to shoot tonight from where the rig is, with your trees and house
taken into account. All computed here (target_picker.py); nothing is sent anywhere."""
from __future__ import annotations

import datetime as _dt
import math
import webbrowser
from pathlib import Path

from PySide6.QtCore import QDate, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit, QDialog,
                               QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

import target_picker as tp

SECTOR_STEPS = (0.0, 20.0, 35.0, 90.0)      # a click cycles: open, low trees, tall trees, blocked


def _local(t):
    return t.astimezone() if t else None


def _hm(t):
    return _local(t).strftime("%H:%M") if t else ""


class CompassWidget(QWidget):
    """Your sky from above: 16 compass sectors, coloured by how high they are blocked. Click to cycle
    open → low trees (20°) → tall trees (35°) → blocked."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sectors = [0.0] * 16
        self.min_alt = 30.0
        self.setMinimumSize(190, 190)
        self.setToolTip("Click a direction to cycle: open → low trees (20°) → tall trees (35°) → blocked")

    def set_sectors(self, s):
        self.sectors = [float(x) for x in s]
        self.update()
        self.changed.emit()

    def _geom(self):
        side = min(self.width(), self.height()) - 26
        c = QPointF(self.width() / 2, self.height() / 2)
        return c, side / 2

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c, r = self._geom()
        pal = self.palette()
        for i, h in enumerate(self.sectors):
            a0 = 90 - (i * 22.5 + 11.25)              # Qt angles: 0 = east, counter-clockwise
            if h >= 89:
                col = QColor(90, 90, 90)
            elif h > self.min_alt:
                col = QColor(120, 150, 90)
            elif h > 0:
                col = QColor(70, 140, 70)
            else:
                col = QColor(40, 70, 140)
            p.setBrush(col)
            p.setPen(QPen(pal.window().color(), 1.5))
            p.drawPie(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), int(a0 * 16), int(22.5 * 16))
            if 0 < h < 89:
                ang = math.radians(i * 22.5)
                tx, ty = c.x() + 0.62 * r * math.sin(ang), c.y() - 0.62 * r * math.cos(ang)
                p.setPen(Qt.white)
                f = p.font()
                f.setPointSizeF(max(6.5, f.pointSizeF() - 2))
                p.setFont(f)
                p.drawText(QRectF(tx - 14, ty - 8, 28, 16), Qt.AlignCenter, f"{h:.0f}°")
        p.setFont(self.font())
        p.setPen(pal.windowText().color())
        for i, lab in ((0, "N"), (4, "E"), (8, "S"), (12, "W")):
            ang = math.radians(i * 22.5)
            tx, ty = c.x() + (r + 9) * math.sin(ang), c.y() - (r + 9) * math.cos(ang)
            p.drawText(QRectF(tx - 10, ty - 8, 20, 16), Qt.AlignCenter, lab)

    def mousePressEvent(self, e):
        c, r = self._geom()
        dx, dy = e.position().x() - c.x(), e.position().y() - c.y()
        if math.hypot(dx, dy) > r:
            return
        az = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
        i = int(((az + 11.25) % 360) // 22.5)
        cur = self.sectors[i]
        nxt = next((s for s in SECTOR_STEPS if s > cur + 0.5), SECTOR_STEPS[0])
        self.sectors[i] = nxt
        self.update()
        self.changed.emit()


class AltitudeChart(QWidget):
    """Altitude through the night for the chosen target: dark time shaded, your horizon in that direction,
    the Moon dashed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nt = None
        self.plan = None
        self.horizon = None
        self.min_alt = 30.0
        self.setMinimumHeight(150)

    def show_plan(self, nt, plan, horizon, min_alt):
        self.nt, self.plan, self.horizon, self.min_alt = nt, plan, horizon, min_alt
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        if not self.nt:
            return
        L, R, T, B = 38, 10, 10, 22
        w, h = self.width() - L - R, self.height() - T - B
        nt = self.nt
        n = len(nt.times)
        # show from 1 h before dusk to 1 h after dawn
        di = [i for i in range(n) if nt.sun_alt[i] < -6] or list(range(n))
        i0, i1 = max(0, di[0] - 12), min(n - 1, di[-1] + 12)
        span = max(1, i1 - i0)
        X = lambda i: L + (i - i0) / span * w  # noqa: E731
        Y = lambda a: T + (1 - max(0.0, min(90.0, a)) / 90) * h  # noqa: E731
        for i in range(i0, i1):
            if nt.dark[i]:
                p.fillRect(QRectF(X(i), T, X(i + 1) - X(i) + 0.5, h), QColor(20, 30, 60, 90))
        p.setPen(QPen(pal.mid().color(), 1))
        for a in (0, 30, 60, 90):
            p.drawLine(QPointF(L, Y(a)), QPointF(L + w, Y(a)))
            p.drawText(QRectF(0, Y(a) - 8, L - 4, 16), Qt.AlignRight | Qt.AlignVCenter, f"{a}°")
        hrs = sorted({i for i in range(i0, i1 + 1) if _local(nt.times[i]).minute < tp.STEP_MIN})
        for i in hrs[::2]:
            p.drawText(QRectF(X(i) - 20, T + h + 3, 40, 16), Qt.AlignCenter, _local(nt.times[i]).strftime("%H"))
        p.setPen(QPen(QColor(200, 200, 120), 1.2, Qt.DashLine))
        path = QPainterPath()
        for k, i in enumerate(range(i0, i1 + 1)):
            pt = QPointF(X(i), Y(nt.moon_alt[i]))
            path.moveTo(pt) if k == 0 else path.lineTo(pt)
        p.drawPath(path)
        if self.plan is None or self.plan.alt is None:
            return
        alt, az = self.plan.alt, self.plan.az
        lim = [max(self.min_alt, float(self.horizon.alt[int(az[i]) % 360]) if self.horizon else 0) for i in range(n)]
        p.setPen(QPen(QColor(200, 90, 70), 1.2))
        path = QPainterPath()
        for k, i in enumerate(range(i0, i1 + 1)):
            pt = QPointF(X(i), Y(lim[i]))
            path.moveTo(pt) if k == 0 else path.lineTo(pt)
        p.drawPath(path)
        p.setPen(QPen(QColor(90, 170, 255), 2.2))
        path = QPainterPath()
        for k, i in enumerate(range(i0, i1 + 1)):
            pt = QPointF(X(i), Y(alt[i]))
            path.moveTo(pt) if k == 0 else path.lineTo(pt)
        p.drawPath(path)
        p.setPen(pal.windowText().color())
        p.drawText(QRectF(L + 6, T + 2, w - 12, 16), Qt.AlignLeft,
                   f"{self.plan.target.name}: blue = its altitude, red = your horizon / minimum, dashed = Moon, "
                   "shaded = dark")


class _Item(QTreeWidgetItem):
    """Sorts numbers as numbers (the sort key is kept in UserRole)."""

    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a, b = self.data(col, Qt.UserRole), other.data(col, Qt.UserRole)
        if a is not None and b is not None:
            try:
                return a < b
            except TypeError:
                pass
        return self.text(col) < other.text(col)


COLS = ["Target", "Name", "Type", "Size", "In your frame", "Mag", "Hours", "Window", "Highest", "Moon", "Best"]


class TargetPickerDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Tonight's Target Picker")
        self.catalog, self.source = tp.load_catalog()
        self.plans, self.nt = [], None
        self.horizon = tp.Horizon()
        self.profiles = tp.nina_profiles()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        lay.addLayout(top)

        # ---- where and when
        g = QGroupBox("Where and when")
        gl = QGridLayout(g)
        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90, 90)
        self.lat.setDecimals(4)
        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180, 180)
        self.lon.setDecimals(4)
        self.lon.setToolTip("East is positive, west negative (e.g. Dallas -96.8)")
        self.lat.setValue(float(settings.value("picker/lat", 0.0)))
        self.lon.setValue(float(settings.value("picker/lon", 0.0)))
        gl.addWidget(QLabel("Latitude"), 0, 0)
        gl.addWidget(self.lat, 0, 1)
        gl.addWidget(QLabel("Longitude"), 1, 0)
        gl.addWidget(self.lon, 1, 1)
        self.date = QDateEdit(QDate.currentDate())
        self.date.setCalendarPopup(True)
        self.date.setToolTip("The evening the night starts")
        gl.addWidget(QLabel("Night of"), 2, 0)
        gl.addWidget(self.date, 2, 1)
        self.prof = QComboBox()
        self.prof.addItem("N.I.N.A. profile…" if self.profiles else "(no N.I.N.A. profile on this computer)")
        for d in self.profiles:
            self.prof.addItem(d["name"])
        self.prof.setEnabled(bool(self.profiles))
        self.prof.setToolTip("Take the site (and telescope focal length, pixel size and horizon file) from one of "
                             "your N.I.N.A. profiles")
        self.prof.activated.connect(self._use_profile)
        gl.addWidget(self.prof, 3, 0, 1, 2)
        b = QPushButton("Fill In From One of My Pictures…")
        b.setToolTip("Reads the site (SITELAT / SITELONG) and your focal length, pixel size and sensor size from a "
                     "FITS or XISF file you shot with this rig")
        b.clicked.connect(lambda: self.from_picture())
        gl.addWidget(b, 4, 0, 1, 2)
        top.addWidget(g)

        # ---- your sky
        g = QGroupBox("Your sky")
        gl = QGridLayout(g)
        self.compass = CompassWidget()
        sec = settings.value("picker/sectors", "")
        if sec:
            try:
                self.compass.sectors = [float(x) for x in str(sec).split(",")][:16]
            except ValueError:
                pass
        self.compass.changed.connect(self._sky_changed)
        gl.addWidget(self.compass, 0, 0, 4, 1)
        self.view = QLineEdit(settings.value("picker/view", "") or "")
        self.view.setPlaceholderText("Describe your view, e.g. open sky from NNW to SSW; trees to the east up to 30°")
        self.view.returnPressed.connect(self._apply_view)
        gl.addWidget(self.view, 0, 1, 1, 2)
        b = QPushButton("Use This Description")
        b.clicked.connect(self._apply_view)
        gl.addWidget(b, 1, 1)
        b = QPushButton("Load a N.I.N.A. Horizon File…")
        b.clicked.connect(lambda: self.load_horizon())
        gl.addWidget(b, 1, 2)
        gl.addWidget(QLabel("Lowest altitude to image at"), 2, 1)
        self.min_alt = QSpinBox()
        self.min_alt.setRange(0, 80)
        self.min_alt.setSuffix("°")
        self.min_alt.setValue(int(settings.value("picker/min_alt", 30)))
        self.min_alt.setToolTip("Below about 30° you look through a lot of air - and your neighbours' lights")
        self.min_alt.valueChanged.connect(self._sky_changed)
        gl.addWidget(self.min_alt, 2, 2)
        self.horizon_note = QLabel("")
        self.horizon_note.setWordWrap(True)
        gl.addWidget(self.horizon_note, 3, 1, 1, 2)
        top.addWidget(g, 1)

        # ---- your rig
        g = QGroupBox("Your rig (for 'In your frame')")
        gl = QGridLayout(g)
        self.focal = QDoubleSpinBox()
        self.focal.setRange(0, 10000)
        self.focal.setSuffix(" mm")
        self.pixel = QDoubleSpinBox()
        self.pixel.setRange(0, 30)
        self.pixel.setDecimals(2)
        self.pixel.setSuffix(" µm")
        self.wpx = QSpinBox()
        self.wpx.setRange(0, 20000)
        self.hpx = QSpinBox()
        self.hpx.setRange(0, 20000)
        for w, key in ((self.focal, "focal"), (self.pixel, "pixel"), (self.wpx, "wpx"), (self.hpx, "hpx")):
            try:
                w.setValue(type(w.value())(float(settings.value(f"picker/{key}", 0) or 0)))
            except (TypeError, ValueError):
                pass
        for r, (lab, w) in enumerate((("Focal length", self.focal), ("Pixel size", self.pixel),
                                      ("Sensor width (pixels)", self.wpx), ("Sensor height (pixels)", self.hpx))):
            gl.addWidget(QLabel(lab), r, 0)
            gl.addWidget(w, r, 1)
        self.fov_note = QLabel("")
        gl.addWidget(self.fov_note, 4, 0, 1, 2)
        top.addWidget(g)

        # ---- filters
        f = QHBoxLayout()
        self.kind_boxes = {}
        for k in tp.TYPES:
            cb = QCheckBox(k)
            cb.setChecked(settings.value(f"picker/type/{k}", k not in (tp.DARK, tp.GCLUSTER), type=bool))
            cb.toggled.connect(self.refresh)
            self.kind_boxes[k] = cb
            f.addWidget(cb)
        f.addStretch(1)
        lay.addLayout(f)
        f = QHBoxLayout()
        f.addWidget(QLabel("At least"))
        self.min_hours = QDoubleSpinBox()
        self.min_hours.setRange(0, 14)
        self.min_hours.setSingleStep(0.5)
        self.min_hours.setSuffix(" h")
        self.min_hours.setValue(float(settings.value("picker/min_hours", 2.0)))
        self.min_hours.valueChanged.connect(self.refresh)
        f.addWidget(self.min_hours)
        f.addWidget(QLabel("up in the dark.   Size:"))
        self.size = QComboBox()
        self.size.addItems(["Any size", "Fits my frame well (10–100 %)", "Small in my frame (< 10 %)",
                            "Needs a mosaic (> 100 %)"])
        self.size.currentIndexChanged.connect(self.refresh)
        f.addWidget(self.size)
        f.addWidget(QLabel("Group by:"))
        self.group = QComboBox()
        self.group.addItems(["Nothing - best first", "Type", "Size in my frame", "Best time of night"])
        self.group.currentIndexChanged.connect(self.refresh)
        f.addWidget(self.group)
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find (M 31, Veil, Heart …)")
        self.find.textChanged.connect(self.refresh)
        f.addWidget(self.find, 1)
        b = QPushButton("Work It Out")
        b.setDefault(True)
        b.clicked.connect(lambda: self.compute())
        f.addWidget(b)
        lay.addLayout(f)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

        split = QSplitter(Qt.Vertical)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLS))
        self.tree.setHeaderLabels(COLS)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(True)
        hh = self.tree.header()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.currentItemChanged.connect(self._chart)
        self.tree.itemDoubleClicked.connect(lambda *_: self.about())
        split.addWidget(self.tree)
        self.chart = AltitudeChart()
        split.addWidget(self.chart)
        split.setSizes([420, 170])
        lay.addWidget(split, 1)

        h = QHBoxLayout()
        for text, slot, tip in (
                ("Copy Coordinates", self.copy_coords, "RA / Dec (J2000) of the selected targets, ready to paste "
                                                       "into N.I.N.A.'s Framing or a Deep Sky Object container"),
                ("Send to N.I.N.A.", self.send_to_nina, "Puts the selected targets in N.I.N.A.'s target list "
                                                         "(Advanced Sequencer › Targets › SubZeroAstro) - built "
                                                         "from your own saved target as a template, if you pick one"),
                ("N.I.N.A. Template…", self.choose_template, "Pick one of your saved N.I.N.A. targets: its "
                                                            "exposures, triggers and checks are copied to every "
                                                            "target you send"),
                ("Save List for N.I.N.A.…", self.save_csv, "A target list N.I.N.A. imports: Sequencer (simple) › "
                                                            "Import Targets"),
                ("About This Target", self.about, "Wikipedia and SIMBAD notes"),
                ("Telescopius", self.telescopius, "Copies the name and opens Telescopius - paste it into its "
                                                  "search box")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, s=slot: s())
            h.addWidget(b)
        h.addStretch(1)
        src = QLabel(f"Objects: {self.source}")
        src.setStyleSheet("color: gray")
        h.addWidget(src)
        b = QPushButton("Close")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.resize(1250, 860)

        # first time: take the most recently used N.I.N.A. profile
        if self.lat.value() == 0 and self.lon.value() == 0 and self.profiles:
            self._use_profile(1)
        self._sky_changed()
        self._fov_note()
        for w in (self.focal, self.pixel, self.wpx, self.hpx):
            w.valueChanged.connect(lambda *_: self._fov_note())

    # ---------------------------------------------------------------- inputs
    def _use_profile(self, idx):
        if idx <= 0 or idx > len(self.profiles):
            return
        d = self.profiles[idx - 1]
        self.lat.setValue(d["lat"])
        self.lon.setValue(d["lon"])
        if d.get("focal"):
            self.focal.setValue(d["focal"])
        if d.get("pixel"):
            self.pixel.setValue(d["pixel"])
        if d.get("horizon") and Path(d["horizon"]).is_file():
            self.load_horizon(d["horizon"])
        self.prof.setCurrentIndex(idx)

    def from_picture(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "A picture shot with this rig",
                                                  str(self.settings.value("picker/pic_dir", Path.home())),
                                                  "FITS / XISF (*.fits *.fit *.fts *.xisf)")
            if not path:
                return
        self.settings.setValue("picker/pic_dir", str(Path(path).parent))
        try:
            lat, lon, focal, pixel, w, h = tp.site_from_fits(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Tonight's Target Picker", f"Couldn't read that file: {e}")
            return
        got = []
        if lat is not None and lon is not None:
            self.lat.setValue(lat)
            self.lon.setValue(lon)
            got.append("site")
        for widget, v, name in ((self.focal, focal, "focal length"), (self.pixel, pixel, "pixel size"),
                                (self.wpx, w, "sensor width"), (self.hpx, h, "sensor height")):
            if v:
                widget.setValue(type(widget.value())(v))
                got.append(name)
        self.summary.setText("From the picture: " + (", ".join(got) if got else "nothing useful (no site or "
                                                                                 "optics in its header)") + ".")

    def load_horizon(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "N.I.N.A. horizon file", str(Path.home()),
                                                  "Horizon files (*.hrz *.hzn *.txt *.csv);;All files (*)")
            if not path:
                return
        try:
            hz = tp.Horizon.from_file(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Tonight's Target Picker", f"Couldn't read that horizon file: {e}")
            return
        self.horizon = hz
        self.compass.sectors = hz.sectors()
        self.compass.update()
        self.horizon_note.setText(f"Horizon from {Path(path).name}")
        self._sky_changed(from_file=True)

    def _apply_view(self):
        text = self.view.text().strip()
        if not text:
            return
        self.compass.set_sectors(tp.parse_view(text))

    def _sky_changed(self, *_a, from_file=False):
        self.compass.min_alt = self.min_alt.value()
        self.compass.update()
        if not from_file:
            self.horizon = tp.Horizon.from_sectors(self.compass.sectors)
            if self.horizon_note.text().startswith("Horizon from"):
                self.horizon_note.setText("")
        open_dirs = [tp.COMPASS[i] for i, h in enumerate(self.compass.sectors) if h <= self.min_alt.value()]
        if not from_file:
            self.horizon_note.setText(f"Open down to {self.min_alt.value()}°: "
                                      + (", ".join(open_dirs) if open_dirs else "nowhere - click the compass"))

    def fov(self):
        return tp.fov_arcmin(self.focal.value(), self.pixel.value(), self.wpx.value(), self.hpx.value())

    def _fov_note(self):
        f = self.fov()
        self.fov_note.setText(f"Field of view {f[0] / 60:.2f}° × {f[1] / 60:.2f}°" if f else
                              "Fill these in to see how big each target is in your frame")

    # ---------------------------------------------------------------- compute and show
    def compute(self):
        if self.lat.value() == 0 and self.lon.value() == 0:
            QMessageBox.information(self, "Tonight's Target Picker", "Where is your rig? Enter its latitude and "
                                    "longitude, pick a N.I.N.A. profile, or fill them in from one of your pictures.")
            return
        for k, v in (("lat", self.lat.value()), ("lon", self.lon.value()), ("min_alt", self.min_alt.value()),
                     ("sectors", ",".join(f"{x:g}" for x in self.compass.sectors)), ("view", self.view.text()),
                     ("focal", self.focal.value()), ("pixel", self.pixel.value()), ("wpx", self.wpx.value()),
                     ("hpx", self.hpx.value()), ("min_hours", self.min_hours.value())):
            self.settings.setValue(f"picker/{k}", v)
        d = self.date.date()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.nt, self.plans = tp.plan_night(self.catalog, self.lat.value(), self.lon.value(),
                                                _dt.date(d.year(), d.month(), d.day()), self.horizon,
                                                float(self.min_alt.value()), self.fov(), keep_curves=True)
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh()

    def visible_plans(self):
        if not self.plans:
            return []
        kinds = {k for k, cb in self.kind_boxes.items() if cb.isChecked()}
        q = self.find.text().strip().lower().replace(" ", "")
        mh = self.min_hours.value()
        si = self.size.currentIndex()
        out = []
        for p in self.plans:
            t = p.target
            if q:
                hay = " ".join([t.name, t.common] + list(t.aka)).lower().replace(" ", "")
                if q not in hay:
                    continue
            else:
                if t.kind not in kinds or p.hours < max(mh, 0.01):
                    continue
                if si and p.fill is not None:
                    if si == 1 and not 0.1 <= p.fill <= 1.0:
                        continue
                    if si == 2 and p.fill >= 0.1:
                        continue
                    if si == 3 and p.fill <= 1.0:
                        continue
            out.append(p)
        out.sort(key=lambda p: -p.score)
        return out

    def _group_of(self, p):
        g = self.group.currentIndex()
        if g == 1:
            return p.target.kind
        if g == 2:
            if p.fill is None:
                return "Size unknown (fill in your rig)"
            return ("Needs a mosaic" if p.fill > 1.05 else "Fits well" if p.fill >= 0.1 else "Small in the frame")
        if g == 3:
            if p.start is None or not self.nt.dark_start:
                return "Not up tonight"
            third = (self.nt.dark_end - self.nt.dark_start) / 3
            mid = p.start + (p.end - p.start) / 2
            return ("1 Evening" if mid < self.nt.dark_start + third else
                    "2 Middle of the night" if mid < self.nt.dark_start + 2 * third else "3 Morning")
        return None

    def refresh(self):
        for k, cb in self.kind_boxes.items():
            self.settings.setValue(f"picker/type/{k}", cb.isChecked())
        self.tree.clear()
        nt = self.nt
        if nt is None:
            self.summary.setText("Set where your rig is and your sky, then press <b>Work It Out</b>.")
            return
        if nt.twilight == "none":
            self.summary.setText("It doesn't get dark enough tonight at this latitude (the Sun stays above -12°).")
            return
        moon_up = [i for i in range(len(nt.times)) if nt.dark[i] and nt.moon_alt[i] > 0]
        moon_txt = (f"Moon {nt.moon_lit * 100:.0f}% lit" +
                    (f", up {_hm(nt.times[moon_up[0]])}–{_hm(nt.times[moon_up[-1]])} in the dark" if moon_up else
                     ", down all night"))
        tips = ""
        if nt.moon_lit > 0.5 and moon_up:
            tips = " - a bright Moon: emission nebulae with a narrowband filter are the best bet tonight."
        elif nt.moon_lit < 0.25 or not moon_up:
            tips = " - a dark night: good for galaxies and faint broadband targets."
        tw = "" if nt.twilight == "astronomical" else " (no full darkness tonight - using nautical twilight)"
        shown = self.visible_plans()
        self.summary.setText(f"<b>Dark from {_hm(nt.dark_start)} to {_hm(nt.dark_end)}</b> ({nt.dark_hours:.1f} h"
                             f"{tw}), times in this computer's time zone. {moon_txt}{tips}<br>"
                             f"{len(shown)} target(s) shown - best first; click a column to sort, double-click "
                             "for more about it.")
        self.tree.setSortingEnabled(False)
        groups = {}
        for rank, p in enumerate(shown[:600]):
            parent = self.tree
            gname = self._group_of(p)
            if gname is not None:
                if gname not in groups:
                    gi = _Item(self.tree, [gname.lstrip("123 ")])
                    gi.setData(0, Qt.UserRole, gname)
                    order = 10000 - int(gname[0]) if gname[0].isdigit() else -rank
                    for c in range(len(COLS)):
                        gi.setData(c, Qt.UserRole, order if c == len(COLS) - 1 else gname)
                    gi.setFirstColumnSpanned(True)
                    f = gi.font(0)
                    f.setBold(True)
                    gi.setFont(0, f)
                    groups[gname] = gi
                parent = groups[gname]
            t = p.target
            win = f"{_hm(p.start)}–{_hm(p.end)}" if p.start else ""
            vals = [(t.name, t.name), (t.common, t.common), (t.kind, t.kind), (t.size_text, t.size),
                    (tp.fit_text(p.fill), p.fill if p.fill is not None else -1),
                    ("" if t.mag is None else f"{t.mag:.1f}", 99 if t.mag is None else t.mag),
                    (f"{p.hours:.1f}", p.hours), (win, p.start.timestamp() if p.start else 0),
                    (f"{p.peak_alt:.0f}° at {_hm(p.peak_time)}", p.peak_alt),
                    (f"{p.moon_sep:.0f}° away" if p.moon_up > 0 else "down", p.moon_sep if p.moon_up > 0 else 999),
                    (f"{p.score:.1f}", -rank)]
            it = _Item(parent, [v[0] for v in vals])
            for c, (_txt, key) in enumerate(vals):
                it.setData(c, Qt.UserRole, key)
            it.setData(0, Qt.UserRole + 1, p)
            aka = ", ".join(t.aka[:6])
            it.setToolTip(0, f"{t.name}" + (f" - also {aka}" if aka else "") + f"\n{tp.nina_coords(t)}  "
                          f"({t.constellation})")
        self.tree.setSortingEnabled(True)
        self.tree.sortItems(len(COLS) - 1, Qt.DescendingOrder)
        if groups:
            self.tree.sortItems(len(COLS) - 1, Qt.DescendingOrder)
            self.tree.expandAll()
        first = self._first_plan_item()
        if first:
            self.tree.setCurrentItem(first)

    def _first_plan_item(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top.data(0, Qt.UserRole + 1):
                return top
            if top.childCount():
                return top.child(0)
        return None

    def selected_plans(self):
        out = [it.data(0, Qt.UserRole + 1) for it in self.tree.selectedItems()]
        out = [p for p in out if p]
        if not out and self.tree.currentItem() is not None and self.tree.currentItem().data(0, Qt.UserRole + 1):
            out = [self.tree.currentItem().data(0, Qt.UserRole + 1)]
        return out

    def _chart(self, cur, _prev=None):
        p = cur.data(0, Qt.UserRole + 1) if cur is not None else None
        self.chart.show_plan(self.nt, p, self.horizon, float(self.min_alt.value()))

    # ---------------------------------------------------------------- actions
    def copy_coords(self):
        ps = self.selected_plans()
        if not ps:
            return
        text = "\n".join(f"{p.target.name}{' (' + p.target.common + ')' if p.target.common else ''}  "
                         f"{tp.nina_coords(p.target)}" for p in ps)
        QGuiApplication.clipboard().setText(text)
        self.summary.setText(f"Copied (J2000): {text.splitlines()[0]}" + (" …" if len(ps) > 1 else ""))

    def save_csv(self, path=None):
        ps = self.selected_plans() or self.visible_plans()[:20]
        if not ps:
            return None
        if path is None:
            default = Path(self.settings.value("picker/csv_dir", str(Path.home()))) / \
                f"Targets_{self.date.date().toString('yyyy-MM-dd')}.csv"
            path, _ = QFileDialog.getSaveFileName(self, "Save targets for N.I.N.A.", str(default), "CSV (*.csv)")
            if not path:
                return None
        self.settings.setValue("picker/csv_dir", str(Path(path).parent))
        tp.nina_csv([p.target for p in ps], path)
        self.summary.setText(f"Saved {len(ps)} target(s) to {path}. In N.I.N.A.: Sequencer › the simple sequencer › "
                             "Import Targets (the Telescopius button) and pick this file - each becomes a target "
                             "with its coordinates; add your exposures and go.")
        return path

    def about(self):
        ps = self.selected_plans()
        if not ps:
            return
        t = ps[0].target
        try:
            import target_info
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                wiki, facts, problems = target_info.lookup_all(t.name)
            finally:
                QApplication.restoreOverrideCursor()
            title, body, url = wiki if wiki else ("", "", "")
            text = (body.strip() + (f"\n\nWikipedia: {url}" if url else "")) if body else ""
            if facts:
                text += "\n\n" + target_info.facts_text(facts)
            if not text:
                text = "Nothing found online" + (f" ({'; '.join(problems)})" if problems else "") + "."
        except Exception as e:  # noqa: BLE001
            text = f"Couldn't look it up ({e})."
        QMessageBox.information(self, t.name, f"{t.name}{' - ' + t.common if t.common else ''}\n{t.kind}, "
                                f"{t.size_text}, {t.constellation}\n{tp.nina_coords(t)} (J2000)\n\n{text[:1800]}")

    def telescopius(self):
        ps = self.selected_plans()
        if not ps:
            return
        QGuiApplication.clipboard().setText(ps[0].target.name)
        webbrowser.open("https://telescopius.com/")
        self.summary.setText(f"'{ps[0].target.name}' is copied - paste it into Telescopius's search box.")

    # ---------------------------------------------------------------- N.I.N.A. advanced sequencer
    def _profile(self):
        i = self.prof.currentIndex()
        return self.profiles[i - 1] if 0 < i <= len(self.profiles) else (self.profiles[0] if self.profiles else None)

    def choose_template(self, path=None):
        if path is None:
            start = tp.nina_targets_folder(self._profile())
            path, _ = QFileDialog.getOpenFileName(self, "One of your saved N.I.N.A. targets (the template)",
                                                  str(start if start.is_dir() else Path.home()),
                                                  "N.I.N.A. target (*.json)")
            if not path:
                return None
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
            import json
            doc = json.loads(text)
            if "DeepSkyObjectContainer" not in str(doc.get("$type", "")):
                raise ValueError("it isn't a N.I.N.A. target (Deep Sky Object container)")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "N.I.N.A. Template", f"Can't use that file as a template: {e}")
            return None
        self.settings.setValue("picker/nina_template", str(path))
        self.summary.setText(f"N.I.N.A. template: {Path(path).name} - every target you send gets its exposures, "
                             "triggers and checks, with the new target's name and coordinates.")
        return path

    def send_to_nina(self, folder=None, ask=True):
        ps = self.selected_plans()
        if not ps:
            QMessageBox.information(self, "Send to N.I.N.A.", "Select one or more targets first.")
            return None
        tpl_path = self.settings.value("picker/nina_template", "") or ""
        if tpl_path and not Path(tpl_path).is_file():
            tpl_path = ""
        if ask and not tpl_path and not self.settings.value("picker/nina_plain_ok", False, type=bool):
            r = QMessageBox.question(
                self, "Send to N.I.N.A.",
                "Use one of your saved N.I.N.A. targets as a template? Its exposures, autofocus, dithering and "
                "checks are copied to each new target - only the name and coordinates change.\n\n"
                "Yes - pick a template.   No - send plain targets (just the name and coordinates).",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if r == QMessageBox.Cancel:
                return None
            if r == QMessageBox.Yes:
                tpl_path = self.choose_template() or ""
                if not tpl_path:
                    return None
            else:
                self.settings.setValue("picker/nina_plain_ok", True)
        template = Path(tpl_path).read_text(encoding="utf-8-sig") if tpl_path else None
        folder = Path(folder) if folder else tp.nina_targets_folder(self._profile())
        label = self.date.date().toString("MMM d")
        try:
            paths = tp.save_nina_targets([p.target for p in ps], folder, label, template)
        except OSError as e:
            QMessageBox.warning(self, "Send to N.I.N.A.", f"Couldn't write to {folder}: {e}")
            return None
        self.summary.setText(f"Sent {len(paths)} target(s) to N.I.N.A. ({folder / 'SubZeroAstro'})"
                             + (f", built from {Path(tpl_path).name}" if tpl_path else "")
                             + ". In N.I.N.A.: Sequencer › Advanced › the Targets list on the right › SubZeroAstro - "
                               "drag one into your sequence. (N.I.N.A. on another computer: copy the files into "
                               "its Documents\\N.I.N.A\\Targets folder.)")
        return paths
