"""
splash.py - the animated intro shown while SubZeroAstro starts.

"The Great Dog" - where veterinary radiology meets the night sky (about 4 s, click to skip):
  0.1 - 1.4 s  subframes slide onto a lightbox like films; with every one stacked, a lateral
               radiograph of a dog rises out of the noise (stacking lowers noise - in both worlds)
  1.6 - 2.7 s  the bones fade, and the joints glide onto the real stars of Canis Major, the
               Great Dog; the constellation lines draw in and Sirius - the Dog Star - flares
  2.3 - 3.6 s  "SubZeroAstro", author and slogan; an ice-blue glint sweeps the title
  ...          fades away when the main window is ready

Everything is drawn with QPainter (no image files). Turn it off in Settings.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPolygonF, QRadialGradient, QTransform)
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

# which joint of the skeleton becomes which star (dog-local coordinates: x towards the nose, y up)
JOINTS = {
    "theta": (0.835, 0.703),        # top of the skull
    "Muliphein": (0.772, 0.655),    # occiput
    "iota": (0.800, 0.572),         # angle of the jaw
    "Sirius": (0.660, 0.428),       # point of the shoulder - the Dog Star at the dog's chest
    "nu2": (0.615, 0.300),          # elbow
    "Mirzam": (0.705, 0.004),       # front paw
    "omicron2": (0.450, 0.545),     # mid back
    "Wezen": (0.160, 0.530),        # lumbosacral junction
    "sigma": (0.125, 0.470),        # hip
    "Adhara": (0.175, 0.280),       # stifle
    "kappa": (0.085, 0.132),        # hock
    "Furud": (0.145, 0.004),        # hind paw
    "Aludra": None,                 # tail tip (computed)
}


def _ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def _ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _seg(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0) if b > a else float(t >= a)


# --------------------------------------------------------------------------
# A lateral radiograph of a dog, as bone shapes in dog-local units (body length ~1)
# --------------------------------------------------------------------------

def _lerp(a, b, k):
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def _spine():
    """Centres of the vertebral bodies: cervical, thoracic, lumbar, sacrum, tail."""
    pts, kinds = [], []
    for i in range(7):                                           # C1-C7
        k = i / 6
        pts.append((0.775 - 0.135 * k, 0.615 - 0.11 * k + 0.02 * math.sin(math.pi * k)))
        kinds.append("C")
    for i in range(13):                                          # T1-T13
        k = i / 12
        pts.append((0.625 - 0.29 * k, 0.502 + 0.04 * math.sin(math.pi * (0.2 + 0.7 * k))))
        kinds.append("T")
    for i in range(7):                                           # L1-L7
        k = i / 6
        pts.append((0.322 - 0.157 * k, 0.540 - 0.008 * k))
        kinds.append("L")
    pts += [(0.140, 0.526), (0.112, 0.519)]
    kinds += ["S", "S"]
    x, y, ang, step = 0.090, 0.515, math.radians(192), 0.024     # tail
    for i in range(16):
        pts.append((x, y))
        kinds.append("Cd")
        x += step * math.cos(ang)
        y += step * math.sin(ang)
        ang += math.radians(3.4)
        step *= 0.955
    return pts, kinds


SPINE, SPINE_KIND = _spine()
JOINTS["Aludra"] = SPINE[-1]


def _bone_paths():
    """(path, width, alpha, filled) in dog-local units."""
    out = []

    def line(*pts, w=0.02, a=1.0):
        p = QPainterPath(QPointF(*pts[0]))
        for q in pts[1:]:
            p.lineTo(QPointF(*q))
        out.append((p, w, a, False))

    def poly(*pts, a=1.0, w=0.006):
        p = QPainterPath()
        p.addPolygon(QPolygonF([QPointF(*q) for q in pts]))
        p.closeSubpath()
        out.append((p, w, a, True))

    def ell(c, rx, ry, a=1.0, w=0.006, filled=True):
        p = QPainterPath()
        p.addEllipse(QPointF(*c), rx, ry)
        out.append((p, w, a, filled))

    # skull and mandible
    poly((0.768, 0.618), (0.763, 0.652), (0.785, 0.690), (0.825, 0.706), (0.862, 0.700), (0.888, 0.680),
         (0.935, 0.656), (0.990, 0.633), (1.018, 0.612), (1.012, 0.596), (0.960, 0.592), (0.890, 0.590),
         (0.845, 0.600), (0.800, 0.606), a=0.95)
    ell((0.862, 0.648), 0.020, 0.017, a=0.0, w=0.005, filled=False)          # orbit (drawn as a dark ring)
    poly((0.788, 0.608), (0.792, 0.574), (0.812, 0.562), (0.870, 0.566), (0.995, 0.582), (0.998, 0.590),
         (0.870, 0.580), (0.820, 0.584), (0.806, 0.610), a=0.9)
    for i in range(9):                                                        # teeth
        x = 0.885 + i * 0.013
        line((x, 0.592), (x + 0.002, 0.582), w=0.006, a=0.9)
    line((0.982, 0.594), (0.986, 0.574), w=0.007, a=1.0)                      # canine
    # vertebrae
    for i, ((x, y), kind) in enumerate(zip(SPINE, SPINE_KIND)):
        nx, ny = SPINE[min(i + 1, len(SPINE) - 1)]
        px, py = SPINE[max(i - 1, 0)]
        ang = math.atan2(ny - py, nx - px)
        ln = (0.017 if kind in ("C", "T", "L") else 0.019 if kind == "S" else 0.016 * (0.96 ** (i - 29)))
        ht = {"C": 0.024, "T": 0.022, "L": 0.026, "S": 0.028}.get(kind, 0.014 * (0.95 ** (i - 29)))
        dx, dy = math.cos(ang) * ln / 2, math.sin(ang) * ln / 2
        ox, oy = -math.sin(ang) * ht / 2, math.cos(ang) * ht / 2
        poly((x - dx - ox, y - dy - oy), (x + dx - ox, y + dy - oy), (x + dx + ox, y + dy + oy),
             (x - dx + ox, y - dy + oy), a=0.95)
        if kind == "T":                                                       # spinous processes
            j = i - 7
            L = 0.075 - 0.0042 * j
            line((x, y + 0.01), (x - 0.35 * L, y + 0.01 + L), w=0.006, a=0.85)
        elif kind == "L":
            line((x, y + 0.01), (x + 0.008, y + 0.04), w=0.007, a=0.8)
        elif kind == "C" and i == 1:
            line((x - 0.01, y + 0.018), (x - 0.04, y + 0.03), w=0.008, a=0.85)
    # ribs and sternum
    for j in range(13):
        x, y = SPINE[7 + j]
        end_y = 0.300 if j < 9 else 0.33 + 0.035 * (j - 9)
        ex = x - 0.055 - 0.004 * j
        p = QPainterPath(QPointF(x, y - 0.012))
        p.quadTo(QPointF(x + 0.03, (y + end_y) / 2), QPointF(ex, end_y))
        out.append((p, 0.0075, 0.75, False))
        if j < 9:
            line((ex, end_y), (ex + 0.025, 0.292 - 0.006 * j), w=0.005, a=0.35)  # costal cartilage
    for i in range(8):
        a_ = _lerp((0.640, 0.330), (0.420, 0.282), i / 7)
        ell(a_, 0.012, 0.007, a=0.85)
    # fore limbs (far one first, fainter)
    for off, a in ((0.035, 0.45), (0.0, 1.0)):
        poly((0.565 + off, 0.588), (0.628 + off, 0.598), (0.668 + off, 0.432), (0.645 + off, 0.424), a=0.8 * a)
        line((0.656 + off, 0.428), (0.616 + off, 0.300), w=0.022, a=a)        # humerus
        ell((0.655 + off, 0.430), 0.016, 0.016, a=a)
        line((0.616 + off, 0.300), (0.633 + off, 0.085), w=0.016, a=a)        # radius / ulna
        line((0.600 + off, 0.318), (0.618 + off, 0.270), w=0.012, a=a)        # olecranon
        ell((0.636 + off, 0.075), 0.012, 0.010, a=a)                          # carpus
        line((0.638 + off, 0.066), (0.662 + off, 0.024), (0.692 + off, 0.008), (0.705 + off, 0.004),
             w=0.010, a=a)
    # pelvis and hind limbs
    for off, a in ((0.03, 0.45), (0.0, 1.0)):
        line((0.192 + off, 0.588), (0.126 + off, 0.472), w=0.030, a=0.9 * a)  # ilium
        line((0.126 + off, 0.472), (0.052 + off, 0.458), w=0.020, a=0.9 * a)  # ischium
        ell((0.093 + off, 0.452), 0.020, 0.010, a=0.0, w=0.004, filled=False)
        ell((0.125 + off, 0.470), 0.015, 0.015, a=a)
        line((0.125 + off, 0.470), (0.176 + off, 0.282), w=0.022, a=a)        # femur
        ell((0.190 + off, 0.292), 0.007, 0.012, a=a)                          # patella
        line((0.176 + off, 0.282), (0.086 + off, 0.135), w=0.018, a=a)        # tibia
        line((0.086 + off, 0.135), (0.058 + off, 0.158), w=0.012, a=a)        # calcaneus
        ell((0.088 + off, 0.124), 0.012, 0.010, a=a)                          # tarsus
        line((0.090 + off, 0.114), (0.105 + off, 0.030), (0.130 + off, 0.008), (0.145 + off, 0.004),
             w=0.010, a=a)
    return out


def _soft_tissue():
    pts = [(1.030, 0.604), (0.995, 0.556), (0.800, 0.520), (0.700, 0.430), (0.690, 0.200), (0.725, 0.000),
           (0.600, 0.000), (0.590, 0.250), (0.420, 0.262), (0.250, 0.330), (0.215, 0.270), (0.160, 0.000),
           (0.050, 0.000), (0.035, 0.170), (0.025, 0.470), (0.060, 0.560), (0.350, 0.598), (0.580, 0.660),
           (0.745, 0.690), (0.810, 0.735), (0.900, 0.705)]
    p = QPainterPath()
    p.addPolygon(QPolygonF([QPointF(*q) for q in pts]))
    p.closeSubpath()
    return p


BONES = _bone_paths()
SOFT = _soft_tissue()


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
        # similarity fit in complex numbers: w = a z + b  (z: dog-local, y up; w: screen with y up)
        zs = [complex(*JOINTS[k]) for k in JOINTS]
        ws = [complex(self.star_xy[k].x(), -self.star_xy[k].y()) for k in JOINTS]
        zm, wm = sum(zs) / len(zs), sum(ws) / len(ws)
        num = sum((w - wm) * (z - zm).conjugate() for z, w in zip(zs, ws))
        den = sum(abs(z - zm) ** 2 for z in zs)
        a = num / den
        b = wm - a * zm
        self.dog_a, self.dog_b = a, b
        self.dog_tf = QTransform(a.real, -a.imag, -a.imag, -a.real, b.real, -b.imag)
        self.dog_scale = abs(a)
        self.joint_xy = {k: self._to_screen(JOINTS[k]) for k in JOINTS}
        pts = [self._to_screen(p) for p in ((0.0, 0.0), (1.05, 0.0), (1.05, 0.75), (0.0, 0.75),
                                            (-0.25, 0.3))] + list(self.star_xy.values())
        x0 = min(p.x() for p in pts) - 18
        x1 = max(p.x() for p in pts) + 18
        y0 = min(p.y() for p in pts) - 18
        y1 = max(p.y() for p in pts) + 18
        self.panel = QRectF(max(14, x0), max(14, y0), min(W * 0.47, x1) - max(14, x0), min(H - 40, y1) - max(14, y0))

    def _to_screen(self, p):
        w = self.dog_a * complex(*p) + self.dog_b
        return QPointF(w.real, -w.imag)

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
            p.setOpacity(box_on * (0.10 + 0.55 * (1 - stacked)))   # noise falls as frames stack
            p.drawImage(panel, self.grain)
            p.setOpacity(1.0)
            # the lead side marker every radiograph carries
            mk = QRectF(panel.left() + 10, panel.top() + 10, 20, 22)
            p.setBrush(QColor(235, 240, 250, int(200 * box_on * _seg(t, 0.4, 0.8))))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(mk, 3, 3)
            f = QFont("Segoe UI", 10, QFont.Bold)
            p.setFont(f)
            p.setPen(QColor(15, 20, 30, int(255 * box_on * _seg(t, 0.4, 0.8))))
            p.drawText(mk, Qt.AlignCenter, "R")
        if xray > 0.005:
            p.save()
            p.setTransform(self.dog_tf, True)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(150, 170, 190, int(42 * xray)))      # soft tissue
            p.drawPath(SOFT)
            for path, w, a, filled in BONES:
                if a == 0.0:                                       # holes (orbit, obturator foramen)
                    p.setPen(QPen(QColor(20, 28, 40, int(200 * xray)), w))
                    p.setBrush(QColor(20, 28, 40, int(160 * xray)))
                    p.drawPath(path)
                    continue
                glow = QPen(QColor(150, 200, 255, int(45 * a * xray)), w * 2.6, Qt.SolidLine, Qt.RoundCap,
                            Qt.RoundJoin)
                p.setPen(glow)
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                core = QPen(QColor(228, 238, 250, int(225 * a * xray)), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                p.setPen(core)
                p.setBrush(QColor(205, 222, 242, int(150 * a * xray)) if filled else Qt.NoBrush)
                p.drawPath(path)
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
        pos = {k: QPointF(self.joint_xy[k].x() + (self.star_xy[k].x() - self.joint_xy[k].x()) * move,
                          self.joint_xy[k].y() + (self.star_xy[k].y() - self.joint_xy[k].y()) * move)
               for k in self.star_xy}
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
            mag = self.star_mag[k]
            size = max(1.4, 4.6 - 0.95 * mag)
            tw = 0.85 + 0.15 * math.sin(t * 3 + q.x())
            glow = QRadialGradient(q, size * 4.5)
            glow.setColorAt(0, QColor(170, 210, 255, int(150 * show * tw)))
            glow.setColorAt(1, QColor(170, 210, 255, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(q, size * 4.5, size * 4.5)
            p.setBrush(QColor(240, 248, 255, int(255 * show)))
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
