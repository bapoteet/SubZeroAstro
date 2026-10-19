"""
splash.py - the animated intro shown while SubZeroAstro starts.

"The Great Dog" - where veterinary radiology meets the night sky (about 4 s, click to skip):
  0.1 - 1.4 s  subframes slide onto a lightbox like films; with every one stacked, a lateral
               radiograph of a dog rises out of the noise (stacking lowers noise - in both worlds)
  1.6 - 2.7 s  points light up on the anatomy (skull, jaw, shoulder, elbow, spine), the film fades
               and they glide onto the real stars of Canis Major, the Great Dog; the constellation
               lines draw in and Sirius - the Dog Star - flares
  2.3 - 3.6 s  "SubZeroAstro", author and slogan; an ice-blue glint sweeps the title
  ...          fades away when the main window is ready

The radiograph is a real one (assets/splash_radiograph.png); everything else is drawn with
QPainter. Turn it off in Settings.
"""

from __future__ import annotations

import math
import random

from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QRadialGradient)
from PySide6.QtWidgets import QApplication, QWidget

W, H = 780, 440
DURATION = 4.1          # seconds before the splash may close (it waits for the app if that is slower)
FADE = 0.45

# --------------------------------------------------------------------------
# Canis Major: RA (hours), Dec (degrees) - J2000
# --------------------------------------------------------------------------
STARS = {
    "Sirius": (6.7525, -16.716, -1.46), "Mirzam": (6.3783, -17.956, 1.98), "Muliphein": (7.0627, -15.633, 4.1),
    "theta": (6.9033, -12.039, 4.07), "iota": (6.9356, -17.054, 4.37), "nu2": (6.6114, -19.256, 3.95),
    "omicron2": (7.0502, -23.833, 3.02), "Wezen": (7.1399, -26.393, 1.83), "sigma": (7.0287, -27.934, 3.47),
    "Adhara": (6.9771, -28.972, 1.50), "Aludra": (7.4016, -29.303, 2.45), "Furud": (6.3385, -30.063, 3.02),
    "kappa": (6.8303, -32.509, 3.96),
}
LINES = [("theta", "iota"), ("Muliphein", "iota"), ("iota", "Sirius"), ("Sirius", "nu2"), ("nu2", "Mirzam"),
         ("Sirius", "omicron2"), ("omicron2", "Wezen"), ("Wezen", "Aludra"), ("Wezen", "sigma"),
         ("sigma", "Adhara"), ("Adhara", "kappa"), ("Adhara", "Furud")]

# where each star of the Great Dog sits on the radiograph (fractions of the picture: x to the right,
# y down; the dog lies with its head to the left). The hind quarters are off the film, so those
# stars simply light up in the sky.
LANDMARKS = {
    "theta": (0.060, 0.040),        # top of the skull
    "Muliphein": (0.162, 0.220),    # occiput
    "iota": (0.090, 0.520),         # angle of the jaw
    "Sirius": (0.327, 0.570),       # point of the shoulder - the Dog Star at the dog's chest
    "nu2": (0.296, 0.910),          # elbow
    "Mirzam": (0.183, 0.990),       # fore paw (leaving the film)
    "omicron2": (0.625, 0.360),     # mid back
    "Wezen": (0.985, 0.270),        # lumbar spine
}
RADIOGRAPH = "splash_radiograph.png"


def _ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def _ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _seg(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0) if b > a else float(t >= a)


def _radiograph_path():
    import sys
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for p in (here / "assets" / RADIOGRAPH, Path(__file__).resolve().parent / "assets" / RADIOGRAPH):
        if p.is_file():
            return p
    return None


def load_radiograph():
    """The radiograph, toned a cool lightbox blue, or None when the file is missing."""
    p = _radiograph_path()
    if p is None:
        return None
    img = QImage(str(p))
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    q = QPainter(img)
    q.setCompositionMode(QPainter.CompositionMode_Multiply)
    q.fillRect(img.rect(), QColor(206, 226, 255))
    q.end()
    return img


# --------------------------------------------------------------------------
# The splash window
# --------------------------------------------------------------------------

class Splash(QWidget):
    finished = Signal()

    def __init__(self, title="SubZeroAstro", author="Brian A. Poteet",
                 slogan="From subs to masters - zero hassle.", version="", parent=None):
        super().__init__(parent, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(W, H)
        self.title, self.author, self.slogan, self.version = title, author, slogan.replace(" - ", " — "), version
        rnd = random.Random(1805)
        self.stars = [(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(0.25, 1.0) ** 2,
                       rnd.uniform(0, 6.283), rnd.choice((0, 0, 0, 1, 2))) for _ in range(230)]
        self.xray_img = load_radiograph()
        self._layout_constellation()
        self.films = []
        for i in range(9):
            ang = rnd.uniform(math.radians(100), math.radians(260)) if i % 2 else rnd.uniform(-1.2, 1.2)
            self.films.append((math.cos(ang) * rnd.uniform(380, 520), math.sin(ang) * rnd.uniform(260, 380),
                               rnd.uniform(-22, 22), 0.12 + 0.11 * i))
        self.grain = self._make_grain(rnd)
        self.clock = QElapsedTimer()
        self.fixed_t = None
        self.ready = False
        self.closing_at = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        screen = QApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.move(g.center().x() - W // 2, g.center().y() - H // 2)

    # ------------------------------------------------------------ geometry
    def _layout_constellation(self):
        """Star positions on the splash (left side) and the transform that lays the dog's
        skeleton over them (least-squares similarity fit of joints -> stars)."""
        dec0 = -22.0
        raw = {k: (-(ra * 15.0) * math.cos(math.radians(dec0)), dec) for k, (ra, dec, _m) in STARS.items()}
        xs = [v[0] for v in raw.values()]
        ys = [v[1] for v in raw.values()]
        box = QRectF(62, 72, 245, 288)
        s = min(box.width() / (max(xs) - min(xs)), box.height() / (max(ys) - min(ys)))
        cxr, cyr = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        self.star_xy = {k: QPointF(box.center().x() + (x - cxr) * s, box.center().y() - (y - cyr) * s)
                        for k, (x, y) in raw.items()}
        self.star_mag = {k: m for k, (_r, _d, m) in STARS.items()}
        # the lightbox with the radiograph, on the left; its anatomy lines up under the constellation
        img_aspect = (self.xray_img.width() / self.xray_img.height()) if self.xray_img is not None else 1.48
        pw = 350.0
        self.panel = QRectF(20, (H - pw / img_aspect) / 2 - 6, pw, pw / img_aspect)
        self.joint_xy = {k: QPointF(self.panel.left() + fx * self.panel.width(),
                                    self.panel.top() + fy * self.panel.height())
                         for k, (fx, fy) in LANDMARKS.items()}

    def _make_grain(self, rnd):
        gw, gh = 150, 170
        img = QImage(gw, gh, QImage.Format_ARGB32)
        for y in range(gh):
            for x in range(gw):
                v = int(min(255, max(0, rnd.gauss(128, 70))))
                img.setPixelColor(x, y, QColor(v, v, v, 200))
        return img

    # ------------------------------------------------------------ control
    def start(self):
        self.clock.start()
        self.show()
        self.raise_()
        self.timer.start(16)

    def set_time(self, t):
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
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.draw(p, self.t())
        p.end()

    def draw(self, p: QPainter, t: float):
        rect = QRectF(0.5, 0.5, W - 1, H - 1)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 18, 18)
        p.setClipPath(clip)
        self._sky(p, rect, t)
        self._radiograph(p, t)
        self._constellation(p, t)
        self._titles(p, t)
        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(70, 110, 170, 160), 1.2))
        p.drawRoundedRect(rect, 18, 18)

    def _sky(self, p, rect, t):
        bg = QRadialGradient(QPointF(W * 0.55, H * 0.45), W * 0.8)
        bg.setColorAt(0.0, QColor(16, 28, 54))
        bg.setColorAt(0.55, QColor(8, 12, 26))
        bg.setColorAt(1.0, QColor(3, 4, 10))
        p.fillRect(rect, QBrush(bg))
        bloom = 0.3 + 0.7 * _ease_out(_seg(t, 1.6, 3.0))
        breathe = 1.0 + 0.04 * math.sin(t * 1.7)
        for (x, y, r, col, a) in ((560, 230, 240, (190, 40, 70), 55), (650, 150, 230, (20, 140, 150), 50),
                                  (470, 300, 170, (230, 150, 60), 26), (200, 200, 230, (60, 80, 170), 40)):
            g = QRadialGradient(QPointF(x + 8 * math.sin(t * 0.6 + x), y), r * breathe)
            g.setColorAt(0.0, QColor(*col, int(a * bloom)))
            g.setColorAt(1.0, QColor(*col, 0))
            p.fillRect(rect, QBrush(g))
        p.setPen(Qt.NoPen)
        fade_in = _ease_out(_seg(t, 0.0, 0.6))
        for (sx, sy, mag, ph, tint) in self.stars:
            x, y = W / 2 + sx * W * 0.52, H / 2 + sy * H * 0.52
            if self.panel.adjusted(-6, -6, 6, 6).contains(QPointF(x, y)) and t < 2.2:
                mag *= 0.25 + 0.75 * _seg(t, 1.6, 2.2)              # keep the lightbox area clean at first
            tw = 0.75 + 0.25 * math.sin(t * (2 + mag * 3) + ph)
            alpha = int(255 * mag * tw * fade_in)
            col = (QColor(200, 220, 255, alpha), QColor(255, 225, 200, alpha), QColor(170, 240, 255, alpha))[tint]
            p.setBrush(col)
            rr = 0.45 + 1.4 * mag
            p.drawEllipse(QPointF(x, y), rr, rr)

    def _radiograph(self, p, t):
        panel = self.panel
        land = [_seg(t, d, d + 0.55) for (_x, _y, _r, d) in self.films]
        stacked = sum(1 for k in land if k >= 1.0) / len(land)
        xray = _ease_out(stacked) * (1.0 - _ease_in_out(_seg(t, 1.65, 2.35)))
        box_on = _ease_out(_seg(t, 0.1, 0.5)) * (1.0 - _ease_in_out(_seg(t, 1.7, 2.3)))
        if box_on > 0:                                             # the lightbox / film
            g = QLinearGradient(panel.topLeft(), panel.bottomRight())
            g.setColorAt(0, QColor(22, 30, 42, int(235 * box_on)))
            g.setColorAt(1, QColor(10, 14, 22, int(235 * box_on)))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(120, 160, 210, int(90 * box_on)), 1.0))
            p.drawRoundedRect(panel, 6, 6)
        if box_on > 0 or xray > 0.005:
            p.save()
            clip = QPainterPath()
            clip.addRoundedRect(panel, 6, 6)
            p.setClipPath(clip, Qt.IntersectClip)
            if xray > 0.005 and self.xray_img is not None:        # the real radiograph
                p.setOpacity(xray)
                p.drawImage(panel, self.xray_img)
            p.setOpacity(box_on * (0.08 + 0.55 * (1 - stacked)))   # noise falls as frames stack
            p.drawImage(panel, self.grain)
            p.restore()
        for (x0, y0, rot, d), k in zip(self.films, land):        # films sliding onto the box
            if 0 < k < 1:
                e = _ease_in_out(k)
                c = panel.center()
                p.save()
                p.translate(c.x() + x0 * (1 - e), c.y() + y0 * (1 - e))
                p.rotate(rot * (1 - e))
                a = int(160 * math.sin(math.pi * min(1.0, k * 1.15)))
                fr = QRectF(-panel.width() / 2, -panel.height() / 2, panel.width(), panel.height())
                g = QLinearGradient(fr.topLeft(), fr.bottomRight())
                g.setColorAt(0, QColor(120, 170, 230, a // 4))
                g.setColorAt(1, QColor(40, 70, 120, a // 5))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(170, 210, 255, a), 1.2))
                p.drawRoundedRect(fr, 6, 6)
                p.restore()

    def _constellation(self, p, t):
        move = _ease_in_out(_seg(t, 1.7, 2.55))
        show = _ease_out(_seg(t, 1.55, 2.0))
        if show <= 0:
            return
        glide = self.xray_img is not None
        pos, vis = {}, {}
        for k, s in self.star_xy.items():
            j = self.joint_xy.get(k) if glide else None
            if j is None:                                          # off the film: lights up in place
                pos[k], vis[k] = s, show * (move if glide else 1.0)
            else:
                pos[k] = QPointF(j.x() + (s.x() - j.x()) * move, j.y() + (s.y() - j.y()) * move)
                vis[k] = show
        draw = _seg(t, 2.25, 2.95)                                 # lines draw in one after another
        n = len(LINES)
        for i, (a, b) in enumerate(LINES):
            k = _seg(draw * n, i, i + 1)
            if k <= 0:
                continue
            pa, pb = pos[a], pos[b]
            pe = QPointF(pa.x() + (pb.x() - pa.x()) * k, pa.y() + (pb.y() - pa.y()) * k)
            p.setPen(QPen(QColor(140, 190, 255, 120), 1.3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pa, pe)
        for k, q in pos.items():
            v = vis[k]
            if v <= 0:
                continue
            mag = self.star_mag[k]
            size = max(1.4, 4.6 - 0.95 * mag)
            tw = 0.85 + 0.15 * math.sin(t * 3 + q.x())
            glow = QRadialGradient(q, size * 4.5)
            glow.setColorAt(0, QColor(170, 210, 255, int(150 * v * tw)))
            glow.setColorAt(1, QColor(170, 210, 255, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(q, size * 4.5, size * 4.5)
            p.setBrush(QColor(240, 248, 255, int(255 * v)))
            p.drawEllipse(q, size * 0.6, size * 0.6)
        # Sirius flares: the Dog Star
        fl = math.exp(-((t - 2.65) / 0.22) ** 2) + 0.35 * _seg(t, 2.6, 2.9)
        if fl > 0.02:
            q = pos["Sirius"]
            for ang, L in ((0, 34), (90, 34), (45, 16), (135, 16)):
                g = QLinearGradient(QPointF(q.x() - L * fl * math.cos(math.radians(ang)),
                                            q.y() - L * fl * math.sin(math.radians(ang))),
                                    QPointF(q.x() + L * fl * math.cos(math.radians(ang)),
                                            q.y() + L * fl * math.sin(math.radians(ang))))
                g.setColorAt(0, QColor(200, 230, 255, 0))
                g.setColorAt(0.5, QColor(225, 240, 255, int(220 * min(1, fl))))
                g.setColorAt(1, QColor(200, 230, 255, 0))
                p.setPen(QPen(QBrush(g), 1.6, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(q.x() - L * fl * math.cos(math.radians(ang)),
                                   q.y() - L * fl * math.sin(math.radians(ang))),
                           QPointF(q.x() + L * fl * math.cos(math.radians(ang)),
                                   q.y() + L * fl * math.sin(math.radians(ang))))
        lab = _ease_out(_seg(t, 2.8, 3.3))
        if lab > 0:
            f = QFont("Segoe UI", 8)
            f.setItalic(True)
            p.setFont(f)
            p.setPen(QColor(140, 160, 190, int(220 * lab)))
            q = pos["Sirius"]
            p.drawText(QRectF(q.x() + 12, q.y() - 22, 120, 16), Qt.AlignLeft | Qt.AlignVCenter, "Sirius")
            p.drawText(QRectF(self.panel.left(), H - 52, self.panel.width(), 16), Qt.AlignHCenter,
                       "Canis Major — the Great Dog")

    def _titles(self, p, t):
        cx, cy = 560, H / 2 - 22
        k = _ease_out(_seg(t, 2.3, 3.0))
        if k > 0:
            f = QFont("Segoe UI", 38, QFont.Bold)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 1 + 16 * (1 - k))
            fm = QFontMetricsF(f)
            parts = [("SubZero", QColor(150, 210, 255)), ("Astro", QColor(240, 246, 255))]
            total = sum(fm.horizontalAdvance(s) for s, _c in parts)
            base = cy + 18 + 12 * (1 - k)
            x = cx - total / 2
            tpath = QPainterPath()
            for s, col in parts:
                path = QPainterPath()
                path.addText(QPointF(x, base), f, s)
                tpath.addText(QPointF(x, base), f, s)
                for spread, a in ((6, 28), (3, 52)):
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
            g = _seg(t, 3.05, 3.75)
            if 0 < g < 1:
                gx = cx - total / 2 - 60 + (total + 120) * _ease_in_out(g)
                sweep = QLinearGradient(QPointF(gx - 40, 0), QPointF(gx + 40, 0))
                sweep.setColorAt(0, QColor(255, 255, 255, 0))
                sweep.setColorAt(0.5, QColor(255, 255, 255, 170))
                sweep.setColorAt(1, QColor(255, 255, 255, 0))
                p.setBrush(QBrush(sweep))
                p.drawPath(tpath)
                self._crystal(p, QPointF(gx, base - fm.ascent() * 0.62), 9 * math.sin(math.pi * g),
                              int(230 * math.sin(math.pi * g)))
        a1 = _ease_out(_seg(t, 2.8, 3.3))
        if a1 > 0:
            f = QFont("Segoe UI", 10)
            f.setItalic(True)
            p.setFont(f)
            p.setPen(QColor(150, 162, 186, int(255 * a1)))
            p.drawText(QRectF(cx - 200, cy + 30, 400, 22), Qt.AlignHCenter | Qt.AlignVCenter, f"by {self.author}")
        a2 = _ease_out(_seg(t, 3.05, 3.6))
        if a2 > 0:
            f = QFont("Segoe UI", 12, QFont.DemiBold)
            p.setFont(f)
            p.setPen(QColor(205, 225, 250, int(255 * a2)))
            p.drawText(QRectF(cx - 210, cy + 60 + 8 * (1 - a2), 420, 26), Qt.AlignHCenter | Qt.AlignVCenter,
                       self.slogan)
        a3 = _ease_out(_seg(t, 0.6, 1.4))
        f = QFont("Segoe UI", 8)
        p.setFont(f)
        p.setPen(QColor(110, 122, 146, int(200 * a3)))
        if self.version:
            p.drawText(QRectF(0, H - 30, W - 18, 20), Qt.AlignRight | Qt.AlignVCenter, f"v{self.version}")
        p.drawText(QRectF(W - 300, 12, 282, 18), Qt.AlignRight | Qt.AlignVCenter, "click to skip")

    @staticmethod
    def _crystal(p, c: QPointF, r, alpha):
        if r <= 0.5 or alpha <= 0:
            return
        pen = QPen(QColor(220, 240, 255, alpha), 1.3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for i in range(6):
            a = math.pi / 3 * i
            p.drawLine(c, QPointF(c.x() + r * math.cos(a), c.y() + r * math.sin(a)))
            for side in (-1, 1):
                bx, by = c.x() + 0.55 * r * math.cos(a), c.y() + 0.55 * r * math.sin(a)
                ba = a + side * math.pi / 4
                p.drawLine(QPointF(bx, by), QPointF(bx + 0.3 * r * math.cos(ba), by + 0.3 * r * math.sin(ba)))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, alpha))
        p.drawEllipse(c, 1.6, 1.6)
