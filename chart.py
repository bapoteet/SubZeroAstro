"""
chart.py - the session chart: one dot per frame in capture order.

  * dot colour = filter, dashed line = median for that filter (camera + filter group)
  * rejected frames are faded with a red cross, flagged frames get a yellow ring,
    the current frame is a larger white dot
  * vertical dotted lines separate nights
  * click a dot to jump to that frame; drag across the chart to select a range
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

FILTER_COLORS = {
    "L": "#a8a8b0", "R": "#e05252", "G": "#4cbf5a", "B": "#4a86e8",
    "Ha": "#c2304a", "OIII": "#26b0a0", "SII": "#e0a030", "Hβ": "#6a78d0",
    "No filter": "#8c8c96",
}
_EXTRA = ["#c070d0", "#d4c040", "#40c0e0", "#e07a40", "#90d060", "#e060a0", "#7090ff"]


def filter_color(name: str) -> QColor:
    if name in FILTER_COLORS:
        return QColor(FILTER_COLORS[name])
    h = sum(ord(c) * (i + 1) for i, c in enumerate(name or ""))
    return QColor(_EXTRA[h % len(_EXTRA)])


class ChartPoint:
    __slots__ = ("uid", "value", "color", "rejected", "flagged", "selected", "night", "label", "group")

    def __init__(self, uid, value, color, rejected, flagged, selected, night, label, group):
        self.uid, self.value, self.color = uid, value, color
        self.rejected, self.flagged, self.selected = rejected, flagged, selected
        self.night, self.label, self.group = night, label, group


class SessionChart(QWidget):
    frameClicked = Signal(int)          # uid
    rangeSelected = Signal(list)        # list of uids

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.points: list[ChartPoint] = []
        self.medians: list[tuple[QColor, float]] = []
        self.current_uid = None
        self.metric_label = ""
        self.higher_better = True
        self.message = "The chart fills in as frames are measured"
        self._drag_from = None
        self._drag_to = None
        self._xs = []
        self._ys = []
        self._ymin = 0.0
        self._ymax = 1.0
        self.ML, self.MR, self.MT, self.MB = 56, 14, 10, 22

    # ------------------------------------------------------------------ data
    def set_data(self, points, medians, current_uid, metric_label, higher_better):
        self.points = points
        self.medians = medians
        self.current_uid = current_uid
        self.metric_label = metric_label
        self.higher_better = higher_better
        self._compute_range()
        self.update()

    def set_current(self, uid):
        if uid != self.current_uid:
            self.current_uid = uid
            self.update()

    def _compute_range(self):
        vals = sorted(p.value for p in self.points if p.value is not None)
        if not vals:
            self._ymin, self._ymax = 0.0, 1.0
            return
        n = len(vals)
        lo = vals[int(0.01 * (n - 1))]
        hi = vals[int(math.ceil(0.99 * (n - 1)))]
        for _c, m in self.medians:
            lo, hi = min(lo, m), max(hi, m)
        if hi - lo < 1e-9:
            pad = abs(hi) * 0.1 + 1e-3
        else:
            pad = (hi - lo) * 0.12
        self._ymin, self._ymax = lo - pad, hi + pad

    # --------------------------------------------------------------- geometry
    def _plot_rect(self) -> QRectF:
        return QRectF(self.ML, self.MT, max(10, self.width() - self.ML - self.MR),
                      max(10, self.height() - self.MT - self.MB))

    def _x(self, i, r: QRectF):
        n = len(self.points)
        if n <= 1:
            return r.center().x()
        return r.left() + 6 + (r.width() - 12) * i / (n - 1)

    def _y(self, v, r: QRectF):
        t = (v - self._ymin) / (self._ymax - self._ymin)
        return r.bottom() - t * r.height()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        r = self._plot_rect()
        small = QFont(self.font())
        small.setPointSize(8)
        p.setFont(small)

        if not any(pt.value is not None for pt in self.points):
            p.setPen(QColor(130, 130, 140))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            p.end()
            return

        # grid + y labels
        p.setPen(QPen(QColor(55, 55, 62), 1))
        for k in range(5):
            v = self._ymin + (self._ymax - self._ymin) * k / 4
            y = self._y(v, r)
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QColor(150, 150, 160))
        for k in range(5):
            v = self._ymin + (self._ymax - self._ymin) * k / 4
            y = self._y(v, r)
            txt = f"{v:.0f}" if abs(self._ymax - self._ymin) > 20 else f"{v:.2f}"
            p.drawText(QRectF(0, y - 8, self.ML - 6, 16), Qt.AlignRight | Qt.AlignVCenter, txt)
        p.save()
        p.translate(10, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-r.height() / 2, -8, r.height(), 14), Qt.AlignCenter,
                   self.metric_label + ("  ↑ better" if self.higher_better else "  ↓ better"))
        p.restore()

        n = len(self.points)
        self._xs = [self._x(i, r) for i in range(n)]

        # night separators + labels
        p.setPen(QPen(QColor(90, 90, 100), 1, Qt.DotLine))
        last = None
        starts = []
        for i, pt in enumerate(self.points):
            if pt.night != last:
                starts.append((i, pt.night))
                if i > 0:
                    x = (self._xs[i - 1] + self._xs[i]) / 2
                    p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                last = pt.night
        p.setPen(QColor(150, 150, 160))
        for j, (i, night) in enumerate(starts):
            x0 = self._xs[i]
            x1 = self._xs[starts[j + 1][0] - 1] if j + 1 < len(starts) else self._xs[-1]
            label = f"Night of {night}" if night else ""
            p.drawText(QRectF(x0, r.bottom() + 3, max(x1 - x0, 80), 16), Qt.AlignLeft | Qt.AlignTop, label)

        # medians
        for color, m in self.medians:
            pen = QPen(color, 1, Qt.DashLine)
            p.setPen(pen)
            y = self._y(m, r)
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        # drag rectangle
        if self._drag_from is not None and self._drag_to is not None:
            x0, x1 = sorted((self._drag_from, self._drag_to))
            p.fillRect(QRectF(x0, r.top(), x1 - x0, r.height()), QColor(196, 120, 40, 60))

        # dots
        rad = max(1.6, min(4.0, (r.width() / max(n, 1)) * 0.45))
        self._ys = []
        cur = None
        for i, pt in enumerate(self.points):
            if pt.value is None:
                self._ys.append(None)
                continue
            v = min(max(pt.value, self._ymin), self._ymax)
            x, y = self._xs[i], self._y(v, r)
            self._ys.append(y)
            if pt.uid == self.current_uid:
                cur = (x, y)
                continue
            col = QColor(pt.color)
            if pt.rejected:
                col.setAlpha(55)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            off_scale = pt.value != v
            if off_scale:      # outside the plotted range: draw a small triangle at the edge
                d = rad + 1
                tri = QPolygonF([QPointF(x - d, y + (d if v == self._ymax else -d)),
                                 QPointF(x + d, y + (d if v == self._ymax else -d)),
                                 QPointF(x, y)])
                p.drawPolygon(tri)
            else:
                p.drawEllipse(QPointF(x, y), rad, rad)
            if pt.rejected:
                p.setPen(QPen(QColor(230, 70, 70, 170), 1.2))
                d = rad + 0.5
                p.drawLine(QPointF(x - d, y - d), QPointF(x + d, y + d))
                p.drawLine(QPointF(x - d, y + d), QPointF(x + d, y - d))
            if pt.flagged:
                p.setPen(QPen(QColor(255, 220, 60), 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x, y), rad + 2.2, rad + 2.2)
            if pt.selected:
                p.setPen(QPen(QColor(230, 140, 50), 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x - rad - 2, y - rad - 2, 2 * rad + 4, 2 * rad + 4))
        if cur:
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPointF(*cur), rad + 2, rad + 2)
        p.end()

    # ------------------------------------------------------------------ mouse
    def _nearest(self, pos, max_dist=12):
        best, bd = None, max_dist ** 2
        for i, x in enumerate(self._xs):
            if i >= len(self._ys) or self._ys[i] is None:
                continue
            d = (x - pos.x()) ** 2 + (self._ys[i] - pos.y()) ** 2
            if d < bd:
                best, bd = i, d
        if best is None:      # fall back to nearest in x only
            for i, x in enumerate(self._xs):
                if i < len(self._ys) and self._ys[i] is not None and abs(x - pos.x()) < 6:
                    return i
        return best

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_from = ev.position().x()
            self._drag_to = None

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag_from is not None and ev.buttons() & Qt.LeftButton:
            if abs(pos.x() - self._drag_from) > 4:
                self._drag_to = pos.x()
                self.update()
            return
        i = self._nearest(pos)
        if i is not None:
            pt = self.points[i]
            val = "–" if pt.value is None else (f"{pt.value:.2f}" if abs(pt.value) < 100 else f"{pt.value:.0f}")
            extra = "  (rejected)" if pt.rejected else ("  (flagged)" if pt.flagged else "")
            QToolTip.showText(ev.globalPosition().toPoint(),
                              f"{pt.label}\n{self.metric_label}: {val}{extra}", self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._drag_from is None:
            return
        if self._drag_to is None:
            i = self._nearest(ev.position(), 16)
            if i is not None:
                self.frameClicked.emit(self.points[i].uid)
        else:
            x0, x1 = sorted((self._drag_from, self._drag_to))
            uids = [self.points[i].uid for i, x in enumerate(self._xs) if x0 <= x <= x1]
            if uids:
                self.rangeSelected.emit(uids)
        self._drag_from = self._drag_to = None
        self.update()
