"""
splash.py - the animated intro shown while SubZeroAstro starts.

About three seconds, click (or any key) to skip:
  0.0 - 1.4 s  subframes fly in from the dark and stack into one bright master
  1.2 - 2.0 s  a flash, then "SubZeroAstro" settles out of the starfield
  1.9 - 2.6 s  author and slogan fade in; an ice-blue glint sweeps the title
  ...          fades away when the main window is ready

Drawn with QPainter only (no image files). Turn it off in Settings.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)
from PySide6.QtWidgets import QApplication, QWidget

W, H = 780, 440
DURATION = 3.2          # seconds before the splash may close (it waits for the app if it is slower)
FADE = 0.45


def _ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def _ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _seg(t, a, b):
    """0..1 progress of t through the interval [a, b]."""
    return min(max((t - a) / (b - a), 0.0), 1.0) if b > a else float(t >= a)


class Splash(QWidget):
    finished = Signal()

    def __init__(self, title="SubZeroAstro", author="Brian A. Poteet",
                 slogan="From subs to masters - zero hassle.", version="", parent=None):
        super().__init__(parent, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(W, H)
        self.title, self.author, self.slogan, self.version = title, author, slogan.replace(" - ", " — "), version
        rnd = random.Random(1805)                       # IC 1805: the same sky every launch
        self.stars = [(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(0.25, 1.0) ** 2,
                       rnd.uniform(0, 6.283), rnd.choice((0, 0, 0, 1, 2))) for _ in range(260)]
        # subframes: start position (off-centre), rotation, delay
        self.subs = []
        for i in range(14):
            ang = rnd.uniform(0, 6.283)
            dist = rnd.uniform(330, 520)
            self.subs.append((math.cos(ang) * dist, math.sin(ang) * dist * 0.7, rnd.uniform(-28, 28),
                              0.08 * i + rnd.uniform(0, 0.05)))
        self.clock = QElapsedTimer()
        self.fixed_t = None                             # set_time() for screenshots / tests
        self.ready = False                              # main window built
        self.closing_at = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        screen = QApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.move(g.center().x() - W // 2, g.center().y() - H // 2)

    # ------------------------------------------------------------ control
    def start(self):
        self.clock.start()
        self.show()
        self.raise_()
        self.timer.start(16)

    def set_time(self, t):
        """Freeze the animation at t seconds (tests and screenshots)."""
        self.fixed_t = t
        self.update()

    def t(self):
        return self.fixed_t if self.fixed_t is not None else self.clock.elapsed() / 1000.0

    def app_ready(self):
        self.ready = True

    def skip(self):
        if self.closing_at is None:
            self.closing_at = self.t()

    def mousePressEvent(self, _e):
        self.skip()

    def keyPressEvent(self, _e):
        self.skip()

    def _tick(self):
        t = self.t()
        if self.closing_at is None and self.ready and t >= DURATION:
            self.closing_at = t
        if self.closing_at is not None:
            k = _seg(t, self.closing_at, self.closing_at + FADE)
            self.setWindowOpacity(1.0 - _ease_in_out(k))
            if k >= 1.0:
                self.timer.stop()
                self.close()
                self.finished.emit()
                return
        self.update()

    # ------------------------------------------------------------ drawing
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        self.draw(p, self.t())
        p.end()

    def draw(self, p: QPainter, t: float):
        rect = QRectF(0.5, 0.5, W - 1, H - 1)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 18, 18)
        p.setClipPath(clip)
        cx, cy = W / 2, H / 2 - 18

        # deep sky
        bg = QRadialGradient(QPointF(cx, cy), W * 0.75)
        bg.setColorAt(0.0, QColor(16, 30, 58))
        bg.setColorAt(0.55, QColor(8, 13, 28))
        bg.setColorAt(1.0, QColor(3, 4, 10))
        p.fillRect(rect, QBrush(bg))

        # nebula clouds: Ha red, OIII teal, a touch of gold - they bloom as the master forms
        bloom = 0.35 + 0.65 * _ease_out(_seg(t, 0.9, 2.2))
        breathe = 1.0 + 0.04 * math.sin(t * 1.7)
        for (dx, dy, r, col, a) in ((-150, 30, 230, (190, 40, 70), 70), (160, -20, 250, (20, 140, 150), 60),
                                    (40, 70, 170, (230, 150, 60), 32), (-40, -60, 200, (90, 60, 170), 40)):
            drift = 8 * math.sin(t * 0.6 + dx)
            g = QRadialGradient(QPointF(cx + dx + drift, cy + dy), r * breathe)
            c = QColor(*col, int(a * bloom))
            g.setColorAt(0.0, c)
            g.setColorAt(1.0, QColor(*col, 0))
            p.fillRect(rect, QBrush(g))

        # stars: a quick warp at the start that settles into a slow drift, twinkling
        warp = 1.0 - _ease_out(_seg(t, 0.0, 1.1))
        p.setPen(Qt.NoPen)
        for (sx, sy, mag, ph, tint) in self.stars:
            z = 1.0 + 0.03 * t
            x, y = cx + sx * W * 0.62 * z, cy + sy * H * 0.75 * z
            tw = 0.75 + 0.25 * math.sin(t * (2 + mag * 3) + ph)
            alpha = int(255 * mag * tw * _ease_out(_seg(t, 0.0, 0.5)))
            col = (QColor(200, 220, 255, alpha), QColor(255, 225, 200, alpha), QColor(170, 240, 255, alpha))[tint]
            if warp > 0.02:                                  # streaks toward the centre
                L = 60 * warp * mag
                vx, vy = x - cx, y - cy
                n = math.hypot(vx, vy) or 1.0
                pen = QPen(col, 0.6 + 1.6 * mag)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(x, y), QPointF(x + vx / n * L, y + vy / n * L))
                p.setPen(Qt.NoPen)
            else:
                p.setBrush(col)
                rr = 0.5 + 1.7 * mag
                p.drawEllipse(QPointF(x, y), rr, rr)
                if mag > 0.85:                                # a few bright stars get spikes
                    pen = QPen(QColor(col.red(), col.green(), col.blue(), alpha // 3), 0.8)
                    p.setPen(pen)
                    s = 7 * mag * tw
                    p.drawLine(QPointF(x - s, y), QPointF(x + s, y))
                    p.drawLine(QPointF(x, y - s), QPointF(x, y + s))
                    p.setPen(Qt.NoPen)

        # subframes flying in and stacking into one master
        fw, fh = 150, 100
        for (x0, y0, rot, delay) in self.subs:
            k = _seg(t, 0.05 + delay, 0.85 + delay)
            if k <= 0 or k >= 1:
                continue
            e = _ease_in_out(k)
            x, y = cx + x0 * (1 - e), cy + y0 * (1 - e)
            p.save()
            p.translate(x, y)
            p.rotate(rot * (1 - e))
            a = int(150 * math.sin(math.pi * k))
            fr = QRectF(-fw / 2, -fh / 2, fw, fh)
            g = QLinearGradient(fr.topLeft(), fr.bottomRight())
            g.setColorAt(0, QColor(120, 170, 230, a // 3))
            g.setColorAt(1, QColor(40, 70, 120, a // 4))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(170, 210, 255, a), 1.2))
            p.drawRoundedRect(fr, 4, 4)
            p.restore()

        # the master: a glowing frame that brightens as subs land, then flashes
        stack = _ease_out(_seg(t, 0.4, 1.5))
        flash = math.exp(-((t - 1.45) / 0.16) ** 2)
        if stack > 0:
            mframe = QRectF(cx - fw / 2, cy - fh / 2, fw, fh)
            fade_master = 1.0 - _ease_in_out(_seg(t, 1.5, 2.0))
            glow = QRadialGradient(QPointF(cx, cy), 160)
            glow.setColorAt(0, QColor(160, 210, 255, int(120 * stack * fade_master + 200 * flash)))
            glow.setColorAt(1, QColor(160, 210, 255, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(cx, cy), 160, 160)
            p.setBrush(QColor(200, 230, 255, int(60 * stack * fade_master)))
            p.setPen(QPen(QColor(210, 235, 255, int(230 * stack * fade_master)), 1.6))
            p.drawRoundedRect(mframe, 5, 5)
        if flash > 0.01:
            p.fillRect(rect, QColor(220, 240, 255, int(140 * flash)))

        # title: letters settle in with widening-to-normal spacing
        k = _ease_out(_seg(t, 1.35, 2.05))
        if k > 0:
            f = QFont("Segoe UI", 44, QFont.Bold)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 2 + 18 * (1 - k))
            p.setFont(f)
            fm = QFontMetricsF(f)
            parts = [("SubZero", QColor(150, 210, 255)), ("Astro", QColor(240, 246, 255))]
            total = sum(fm.horizontalAdvance(s) for s, _c in parts)
            x = cx - total / 2
            base = cy + 22 + 14 * (1 - k)
            for s, col in parts:
                path = QPainterPath()
                path.addText(QPointF(x, base), f, s)
                for spread, a in ((6, 30), (3, 55)):          # soft icy glow
                    p.setPen(QPen(QColor(120, 190, 255, int(a * k)), spread * 2, Qt.SolidLine, Qt.RoundCap,
                                  Qt.RoundJoin))
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(path)
                c = QColor(col)
                c.setAlpha(int(255 * k))
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawPath(path)
                x += fm.horizontalAdvance(s)
            # an ice glint sweeping across the title
            g = _seg(t, 2.05, 2.75)
            if 0 < g < 1:
                gx = cx - total / 2 - 60 + (total + 120) * _ease_in_out(g)
                sweep = QLinearGradient(QPointF(gx - 40, 0), QPointF(gx + 40, 0))
                sweep.setColorAt(0, QColor(255, 255, 255, 0))
                sweep.setColorAt(0.5, QColor(255, 255, 255, 170))
                sweep.setColorAt(1, QColor(255, 255, 255, 0))
                tpath = QPainterPath()
                xx = cx - total / 2
                for s, _c in parts:
                    tpath.addText(QPointF(xx, base), f, s)
                    xx += fm.horizontalAdvance(s)
                p.setBrush(QBrush(sweep))
                p.drawPath(tpath)
                # a six-pointed ice crystal riding the glint
                self._crystal(p, QPointF(gx, base - fm.ascent() * 0.62), 9 * math.sin(math.pi * g),
                              int(230 * math.sin(math.pi * g)))

        # author and slogan
        a1 = _ease_out(_seg(t, 1.9, 2.4))
        if a1 > 0:
            f = QFont("Segoe UI", 10)
            f.setItalic(True)
            p.setFont(f)
            p.setPen(QColor(150, 162, 186, int(255 * a1)))
            p.drawText(QRectF(0, cy + 34, W, 22), Qt.AlignHCenter | Qt.AlignVCenter, f"by {self.author}")
        a2 = _ease_out(_seg(t, 2.15, 2.7))
        if a2 > 0:
            f = QFont("Segoe UI", 13, QFont.DemiBold)
            p.setFont(f)
            p.setPen(QColor(205, 225, 250, int(255 * a2)))
            p.drawText(QRectF(0, cy + 64 + 8 * (1 - a2), W, 26), Qt.AlignHCenter | Qt.AlignVCenter, self.slogan)

        # small print
        a3 = _ease_out(_seg(t, 0.6, 1.4))
        f = QFont("Segoe UI", 8)
        p.setFont(f)
        p.setPen(QColor(110, 122, 146, int(200 * a3)))
        if self.version:
            p.drawText(QRectF(0, H - 30, W - 18, 20), Qt.AlignRight | Qt.AlignVCenter, f"v{self.version}")
        p.drawText(QRectF(18, H - 30, 300, 20), Qt.AlignLeft | Qt.AlignVCenter, "click to skip")

        # frame edge
        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(70, 110, 170, 160), 1.2))
        p.drawRoundedRect(rect, 18, 18)

    @staticmethod
    def _crystal(p, c: QPointF, r, alpha):
        if r <= 0.5 or alpha <= 0:
            return
        pen = QPen(QColor(220, 240, 255, alpha), 1.3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for i in range(6):
            a = math.pi / 3 * i
            ex, ey = c.x() + r * math.cos(a), c.y() + r * math.sin(a)
            p.drawLine(c, QPointF(ex, ey))
            for side in (-1, 1):                              # little branches
                bx, by = c.x() + 0.55 * r * math.cos(a), c.y() + 0.55 * r * math.sin(a)
                ba = a + side * math.pi / 4
                p.drawLine(QPointF(bx, by), QPointF(bx + 0.3 * r * math.cos(ba), by + 0.3 * r * math.sin(ba)))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, alpha))
        p.drawEllipse(c, 1.6, 1.6)
