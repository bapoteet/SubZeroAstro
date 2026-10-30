"""
fasttrack_window.py - Fast Track: from the loaded frames (or the masters) to a finished picture on the
suggested settings, with every step shown as it happens, like a short film.

The real pages do the work, out of sight, with the same code and defaults as when you go step by step:
page 1's frames are checked (the least aggressive cull - nothing is moved), page 2 calibrates and stacks,
page 3 runs its steps, page 4 takes each recommended step with its suggested version. Afterwards every page
is there with ◀ Back, so anything can be changed by hand.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEasingCurve, QRectF, QSize, Qt, QThread, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

import ui_header

FRAME_SIDE = 1100            # the film's frames are kept at most this big (memory: ~2-3 MB each)
FADE_MS = 900                # cross-fade between two frames
SHOW_MS = 1700               # how long each frame stays when the film is played again
FLY_MS = 18000               # Fly Into It: the slow flight in (and the same back out)
FLY_SIDE = 2600              # the finished picture is kept this big for the flight, so the close-up stays sharp

FAST_STYLE = ("QPushButton { background:#b8860b; color:white; font-weight:700; font-size:10.5pt; "
              "padding:7px 14px; border-radius:6px; border:1px solid #e0b040; }"
              "QPushButton:hover { background:#cf9a14; }"
              "QPushButton:disabled { background:#2a2f3a; color:#6c7486; border:1px solid #363c48; }")


def fast_button(text="⚡  Fast Track"):
    b = QPushButton(text)
    b.setStyleSheet(FAST_STYLE)
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip("Fast Track: SubZeroAstro does every step for you on its suggested settings and shows each one "
                 "as it happens. Afterwards you can save the picture, or change any step by hand.")
    return b


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def rgb8_to_qimage(a: np.ndarray, side: int = FRAME_SIDE) -> QImage:
    """(h, w, 3) or (h, w) uint8 -> QImage, reduced to at most `side` (box average, no jaggies)."""
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    h, w = a.shape[:2]
    f = max(1, int(np.ceil(max(h, w) / side)))
    if f > 1:
        hh, ww = h // f * f, w // f * f
        a = a[:hh, :ww].reshape(hh // f, f, ww // f, f, 3).mean(axis=(1, 3)).astype(np.uint8)
    a = np.ascontiguousarray(a[..., :3], dtype=np.uint8)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Job(QThread):
    """One function off the window's thread (reading a frame for the film must not stutter it)."""
    ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def _header_info(f, cache):
    """The header of a frame page 1 has not measured yet (read once per file, cached)."""
    import stacking as st
    k = str(f.path)
    if k not in cache:
        try:
            cache[k] = st.read_info(f.path)
        except Exception:  # noqa: BLE001
            cache[k] = None
    return cache[k]


def frame_target(f, cache=None) -> str:
    """The target a page-1 frame belongs to: OBJECT in its header, else its file or folder name."""
    import stacking as st
    cache = cache if cache is not None else {}
    n = str((f.info or {}).get("object") or "").strip()
    if not n and getattr(f, "cards", None):
        import fits_core as core
        look = core.cards_to_dict(f.cards)
        n = str(look.get("OBJECT") or look.get("OBJNAME") or "").strip()
    elif not n:
        inf = _header_info(f, cache)
        n = str(getattr(inf, "target", "") or "").strip() if inf is not None else ""
    if n and not st._GENERIC_TARGET.match(n):
        return n
    d = ("folder", str(getattr(f, "folder", "")))
    if d not in cache:
        cache[d] = st.target_from_name(f.path) or ""
    return cache[d] or n


def is_calibration(f, cache=None) -> bool:
    """A bias, dark or flat frame among the files page 1 opened."""
    import fits_core as core
    import stacking as st
    try:
        if getattr(f, "cards", None):
            typ, _master = st.frame_type(core.cards_to_dict(f.cards), Path(f.path).name)
        else:
            inf = _header_info(f, cache if cache is not None else {})
            typ = getattr(inf, "type", None) if inf is not None else None
    except Exception:  # noqa: BLE001
        return False
    return typ not in (None, "light")


# --------------------------------------------------------------------------
# the film: a big picture that cross-fades from step to step and slowly drifts in
# --------------------------------------------------------------------------

class CineView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self._old: QPixmap | None = None
        self._new: QPixmap | None = None
        self._fade = 1.0
        self._drift = 0.0
        self.caption = ""
        self.sub = ""
        self.badge = ""
        self.fade = QVariantAnimation(self)
        self.fade.setDuration(FADE_MS)
        self.fade.setStartValue(0.0)
        self.fade.setEndValue(1.0)
        self.fade.setEasingCurve(QEasingCurve.InOutQuad)
        self.fade.valueChanged.connect(self._set_fade)
        self.drift = QVariantAnimation(self)
        self.drift.setDuration(7000)
        self.drift.setStartValue(0.0)
        self.drift.setEndValue(1.0)
        self.drift.setEasingCurve(QEasingCurve.OutCubic)
        self.drift.valueChanged.connect(self._set_drift)
        self.flight = None               # Fly Into It: a looping animation over the finished picture
        self._fly_pm: QPixmap | None = None
        self._fly_t = 0.0
        self._focus = (0.5, 0.5)

    # ------------------------------------------------------------ Fly Into It
    def start_flight(self, img: QImage, focus, caption: str):
        """A slow flight into the picture: zoom towards the target while turning gently, then back out, again
        and again - like gliding in on a slow plane."""
        self.stop_flight()
        self._fly_pm = QPixmap.fromImage(img)
        self._focus = (min(max(float(focus[0]), 0.2), 0.8), min(max(float(focus[1]), 0.2), 0.8))
        self.caption, self.sub, self.badge = caption, "", ""
        a = QVariantAnimation(self)
        a.setDuration(2 * FLY_MS)
        a.setStartValue(0.0)
        a.setEndValue(2.0)
        a.setLoopCount(-1)
        a.valueChanged.connect(self._set_fly)
        self.flight = a
        self._fly_t = 0.0
        a.start()
        self.update()

    def stop_flight(self):
        if self.flight is not None:
            self.flight.stop()
            self.flight = None
        self._fly_pm = None
        self.update()

    def flying(self) -> bool:
        return self.flight is not None

    def _set_fly(self, v):
        self._fly_t = float(v)
        self.update()

    def _paint_flight(self, p):
        import math
        pm = self._fly_pm
        W, H = self.width(), self.height()
        v = self._fly_t
        ph = v if v <= 1.0 else 2.0 - v                      # in, then back out
        e = 0.5 - 0.5 * math.cos(math.pi * ph)                # smooth start and stop
        fit = min(W / pm.width(), H / pm.height())
        zoom = fit * (1.04 + 1.7 * e)
        fx, fy = self._focus
        cx = pm.width() * (0.5 + (fx - 0.5) * e)
        cy = pm.height() * (0.5 + (fy - 0.5) * e)
        ang = 14.0 * e + 2.0 * math.sin(2 * math.pi * v)      # a slow turn, with a little sway
        p.save()
        p.translate(W / 2, H / 2)
        p.rotate(ang)
        p.scale(zoom, zoom)
        p.translate(-cx, -cy)
        p.drawPixmap(0, 0, pm)
        p.restore()

    def _set_fade(self, v):
        self._fade = float(v)
        self.update()

    def _set_drift(self, v):
        self._drift = float(v)
        self.update()

    def show_frame(self, img: QImage, caption: str, sub: str = "", badge: str = "", animate=True):
        if self.flight is not None:
            self.stop_flight()
            animate = False
        pm = QPixmap.fromImage(img)
        self._old = self._new if animate else None
        self._new = pm
        self.caption, self.sub, self.badge = caption, sub, badge
        self.fade.stop()
        self.drift.stop()
        if animate and self._old is not None:
            self._fade = 0.0
            self.fade.start()
        else:
            self._fade = 1.0
        self._drift = 0.0
        self.drift.start()
        self.update()

    def set_sub(self, sub: str):
        if sub != self.sub:
            self.sub = sub
            self.update()

    def _rect(self, pm: QPixmap, zoom: float) -> QRectF:
        W, H = self.width(), self.height() - 4
        s = min(W / pm.width(), H / pm.height()) * zoom
        w, h = pm.width() * s, pm.height() * s
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#05070b"))
        if self._new is None:
            p.setPen(QColor("#8a93a6"))
            p.drawText(self.rect(), Qt.AlignCenter, "The picture appears here as SubZeroAstro works")
            return
        zoom = 1.0 + 0.035 * self._drift            # a slow drift in, so a still frame feels alive
        if self._fly_pm is not None:
            self._paint_flight(p)
        elif self._old is not None and self._fade < 1.0:
            p.setOpacity(1.0)
            p.drawPixmap(self._rect(self._old, 1.035), self._old, QRectF(self._old.rect()))
            p.setOpacity(self._fade)
        if self._fly_pm is None:
            p.drawPixmap(self._rect(self._new, zoom), self._new, QRectF(self._new.rect()))
        p.setOpacity(1.0)
        # caption band at the bottom
        if self.caption:
            band_h = 54 if self.sub else 36
            r = QRectF(0, self.height() - band_h, self.width(), band_h)
            p.fillRect(r, QColor(5, 8, 14, 185))
            f = QFont(self.font())
            f.setPointSizeF(ui_header._PT(12.5))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#f2f5fb"))
            p.drawText(r.adjusted(16, 4, -16, -(22 if self.sub else 4)), Qt.AlignLeft | Qt.AlignVCenter, self.caption)
            if self.sub:
                f.setBold(False)
                f.setPointSizeF(ui_header._PT(9.5))
                p.setFont(f)
                p.setPen(QColor("#aeb8cc"))
                p.drawText(r.adjusted(16, 28, -16, -4), Qt.AlignLeft | Qt.AlignVCenter, self.sub)
        if self.badge:
            f = QFont(self.font())
            f.setPointSizeF(ui_header._PT(9.5))
            f.setBold(True)
            p.setFont(f)
            tw = p.fontMetrics().horizontalAdvance(self.badge) + 20
            r = QRectF(self.width() - tw - 12, 12, tw, 26)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(184, 134, 11, 220))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignCenter, self.badge)
        p.end()


# --------------------------------------------------------------------------
# "What is it?" - the one question Fast Track asks
# --------------------------------------------------------------------------

class StartDialog(QDialog):
    def __init__(self, parent, targets_found: list, counts: dict | None = None, note: str = ""):
        super().__init__(parent)
        import targets as tg
        self.setWindowTitle("Fast Track")
        lay = QVBoxLayout(self)
        intro = QLabel("<b>⚡ Fast Track</b> takes your data all the way to a finished picture on SubZeroAstro's "
                       "suggested settings for your kind of target, and shows every step as it happens.<br>"
                       "Afterwards you can save the picture, or change any step by hand.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.target = QComboBox()
        for t in targets_found:
            n = (counts or {}).get(t)
            self.target.addItem(f"{t}  ({n} frames)" if n else t, t)
        form.addRow("Target:", self.target)
        self.kind = QComboBox()
        for key, label in tg.TYPE_CHOICES:
            self.kind.addItem(label, key)
        form.addRow("What is it?", self.kind)
        lay.addLayout(form)
        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setStyleSheet("color:#8a93a6;")
        lay.addWidget(self.why)
        if note:
            n_ = QLabel(note)
            n_.setWordWrap(True)
            n_.setStyleSheet("color:#d9b45a;")
            lay.addWidget(n_)
        bb = QDialogButtonBox()
        self.btn_go = bb.addButton("⚡  Start Fast Track", QDialogButtonBox.AcceptRole)
        self.btn_go.setStyleSheet(FAST_STYLE)
        bb.addButton(QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.target.currentIndexChanged.connect(self._guess)
        self._guess()
        self.resize(520, 260)

    def _guess(self):
        import targets as tg
        name = self.target.currentData() or ""
        auto = tg.classify(name) if name else tg.UNKNOWN
        i = self.kind.findData(auto)
        self.kind.setCurrentIndex(max(i, 0))
        self.why.setText(f"SubZeroAstro recognised <b>{name}</b> as: "
                         f"<b>{dict(tg.TYPE_CHOICES).get(auto, 'something else').lower()}</b>. "
                         "Change it if that is wrong - the steps and their settings follow the kind of object."
                         if auto else
                         f"SubZeroAstro does not know <b>{name or 'this target'}</b> by name - please pick what it "
                         "is, so the steps suit it.")

    def choice(self):
        return self.target.currentData(), self.kind.currentData()


# --------------------------------------------------------------------------
# the window and the run
# --------------------------------------------------------------------------

STAGES = [("cull", "1 · Check the frames"), ("stack", "2 · Calibrate and stack"),
          ("prepare", "3 · Image processing"), ("finish", "4 · Finishing touches")]


class FastTrackWindow(QMainWindow):
    def __init__(self, owner, settings):
        super().__init__(None)
        self.owner = owner
        self.settings = settings
        self.setWindowTitle("SubZeroAstro — Fast Track")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.frames_shot: list = []          # [(QImage, caption, sub)]
        self.stage = None
        self.cancelled = False
        self.finished_ok = False
        self.t0 = time.time()
        self.main = self.sw = self.pw = self.fw = None
        self.target = ""
        self.kind = ""
        self.jobs = []
        self._replay = None
        self._build_ui()
        self.resize(1360, 900)

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background:#10131a; color:#d8dee9;")
        v = QVBoxLayout(root)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        top = QHBoxLayout()
        self.title = QLabel("<b>⚡ Fast Track</b>")
        self.title.setStyleSheet(f"font-size:{ui_header._PT(15)}pt; color:#ffe7a8;")
        top.addWidget(self.title)
        top.addStretch(1)
        self.stage_labels = {}
        for key, text in STAGES:
            lb = QLabel(text)
            lb.setStyleSheet(self._chip("todo"))
            self.stage_labels[key] = lb
            top.addWidget(lb)
        v.addLayout(top)
        mid = QHBoxLayout()
        self.cine = CineView()
        mid.addWidget(self.cine, 1)
        side = QVBoxLayout()
        lab = QLabel("<b>Steps</b>")
        side.addWidget(lab)
        self.steps = QListWidget()
        self.steps.setStyleSheet("QListWidget { background:#151a23; border:1px solid #2a3140; }")
        self.steps.setMinimumWidth(320)
        self.steps.setWordWrap(True)                      # long lines wrap instead of running off the side
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side.addWidget(self.steps, 1)
        mid.addLayout(side)
        v.addLayout(mid, 1)
        self.strip = QListWidget()
        self.strip.setViewMode(QListWidget.IconMode)
        self.strip.setFlow(QListWidget.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListWidget.Static)
        self.strip.setIconSize(QSize(132, 88))
        self.strip.setFixedHeight(128)
        self.strip.setSpacing(4)
        self.strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.strip.setStyleSheet("QListWidget { background:#0b0e14; border:1px solid #2a3140; }")
        self.strip.itemClicked.connect(self._strip_clicked)
        v.addWidget(self.strip)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        v.addWidget(self.progress)
        row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setStyleSheet("color:#aeb8cc;")
        row.addWidget(self.status, 1)
        self.btn_play = QPushButton("▶  Play the Film Again")
        self.btn_play.clicked.connect(self.replay)
        self.btn_fly = QPushButton("✈  Fly Into It")
        self.btn_fly.setToolTip("A slow flight into your finished picture: zooming towards the target and turning "
                                "gently, then back out - again and again until you stop it")
        self.btn_fly.clicked.connect(self.toggle_flight)
        self.btn_save = QPushButton("💾  Save the Picture…")
        self.btn_save.clicked.connect(lambda: self.fw.export_dialog() if self.fw else None)
        self.btn_proj = QPushButton("📁  Save Project…")
        self.btn_proj.clicked.connect(lambda: self.fw.save_project_dialog() if self.fw else None)
        self.btn_astro = QPushButton("🔭  Share to AstroBin…")
        self.btn_astro.setToolTip("Upload the finished picture to AstroBin with its acquisition details "
                                  "(the same as Share to AstroBin on page 4)")
        self.btn_astro.clicked.connect(lambda: self.fw.share_astrobin() if self.fw else None)
        self.btn_tune = QPushButton("🔧  Fine-Tune on Page 4")
        self.btn_tune.setToolTip("Open Finishing Touches with every Fast Track step in its list: click any step and "
                                 "use Go Back to Here to try other versions")
        self.btn_tune.clicked.connect(self.fine_tune)
        self.btn_redo = QPushButton("↺  Reprocess Step by Step")
        self.btn_redo.setToolTip("Open Finishing Touches at the start (your masters as loaded) and go through the "
                                 "steps yourself, picking each version")
        self.btn_redo.clicked.connect(self.redo_by_hand)
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        for b in (self.btn_play, self.btn_fly, self.btn_save, self.btn_proj, self.btn_astro, self.btn_tune,
                  self.btn_redo):
            b.setVisible(False)
        for b in (self.btn_fly, self.btn_play, self.btn_save, self.btn_astro, self.btn_proj, self.btn_tune, self.btn_redo, self.btn_cancel,
                  self.btn_close):
            b.setCursor(Qt.PointingHandCursor)
            row.addWidget(b)
        self.btn_redo.setStyleSheet(ui_header.NEXT_STYLE)
        v.addLayout(row)
        self.setCentralWidget(root)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self.clock.start(1000)

    @staticmethod
    def _chip(state):
        col = {"todo": ("#1a1f29", "#6c7486", "#2a3140"), "now": ("#3a3020", "#ffe7a8", "#b8860b"),
               "done": ("#16301f", "#9fe0b0", "#2e8b57"), "skip": ("#1a1f29", "#8a93a6", "#2a3140")}[state]
        return (f"background:{col[0]}; color:{col[1]}; border:1px solid {col[2]}; border-radius:8px; "
                f"padding:3px 9px; font-weight:600;")

    def _stage(self, key, state="now"):
        if key in self.stage_labels:
            self.stage_labels[key].setStyleSheet(self._chip(state))
        if state == "now":
            self.stage = key

    def _tick(self):
        if self.finished_ok or self.cancelled:
            return
        el = int(time.time() - self.t0)
        self.title.setText(f"<b>⚡ Fast Track</b> — {self.target or ''}"
                           + (f"  <span style='color:#8a93a6'>({self.kind_label()})</span>" if self.target else "")
                           + f"  <span style='color:#8a93a6; font-size:10pt'>{el // 60}:{el % 60:02d}</span>")

    def kind_label(self):
        import targets as tg
        return dict(tg.TYPE_CHOICES).get(self.kind, "something else").lower()

    def current_target(self):
        return [self.target] if self.target else []

    # ------------------------------------------------------------ the film
    def add_frame(self, rgb8, caption, sub=""):
        try:
            img = rgb8_to_qimage(rgb8)
        except Exception:  # noqa: BLE001 - the film never stops the run
            return
        self.frames_shot.append((img, caption, sub))
        n = len(self.frames_shot)
        self.cine.show_frame(img, caption, sub, badge=f"{n}")
        it = QListWidgetItem(QIcon(QPixmap.fromImage(img.scaled(264, 176, Qt.KeepAspectRatio,
                                                                Qt.SmoothTransformation))), "")
        it.setToolTip(f"{n}. {caption}" + (f"\n{sub}" if sub else ""))
        it.setData(Qt.UserRole, n - 1)
        self.strip.addItem(it)
        self.strip.scrollToItem(it)

    def _strip_clicked(self, it):
        self.stop_replay()
        self.btn_fly.setText("✈  Fly Into It")
        i = it.data(Qt.UserRole)
        if i is not None and 0 <= i < len(self.frames_shot):
            img, cap, sub = self.frames_shot[i]
            self.cine.show_frame(img, cap, sub, badge=f"{i + 1} / {len(self.frames_shot)}")

    def replay(self):
        if not self.frames_shot:
            return
        self.stop_replay()
        self.btn_fly.setText("✈  Fly Into It")
        self._replay_i = 0
        self._replay = QTimer(self)
        self._replay.timeout.connect(self._replay_next)
        self._replay.start(SHOW_MS)
        self._replay_next()

    def _replay_next(self):
        i = self._replay_i
        if i >= len(self.frames_shot):
            self.stop_replay()
            return
        img, cap, sub = self.frames_shot[i]
        self.cine.show_frame(img, cap, sub, badge=f"{i + 1} / {len(self.frames_shot)}")
        self.strip.setCurrentRow(i)
        self._replay_i += 1

    # ------------------------------------------------------------ Fly Into It
    def flight_frame(self):
        """The finished picture, big, and where its target is (the brightest broad area near the middle)."""
        import finish as F
        img = F.state_display(self.fw.state.small(FLY_SIDE), with_stars=True)
        L = img.astype(np.float32).mean(axis=2)
        k = max(1, L.shape[1] // 240)
        sm = F._blur(L[::k, ::k], 6.0)
        h, w = sm.shape
        y0, y1, x0, x1 = int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8)
        part = sm[y0:y1, x0:x1]
        # the centre of the bright mass, not its single brightest spot: a planetary's ring or a galaxy's arms
        # pull it to the middle of the object
        wgt = np.clip(part - np.percentile(part, 90), 0, None)
        if float(wgt.sum()) <= 0:
            return img, (0.5, 0.5)
        yy, xx = np.mgrid[:part.shape[0], :part.shape[1]]
        cy, cx = float((yy * wgt).sum() / wgt.sum()), float((xx * wgt).sum() / wgt.sum())
        return img, ((x0 + cx + 0.5) / w, (y0 + cy + 0.5) / h)

    def fly(self):
        try:
            if self.fw is None or not self.isVisible():
                return
        except RuntimeError:
            return
        if self.fw is None or self.fw.state is None or self.fw.state.rgb is None:
            return
        self.stop_replay()
        if getattr(self, "_flight", None) is not None:
            img, focus = self._flight
            self.cine.start_flight(img, focus, f"{self.target} — flying in")
            self.btn_fly.setText("■  Stop Flying")
            return

        def ok(r):
            try:
                img, focus = r
                self._flight = (rgb8_to_qimage(img, FLY_SIDE), focus)
                if self.isVisible():
                    self.fly()
            except RuntimeError:          # the window was closed while the picture was being made
                pass
        self.run_job(self.flight_frame, ok)

    def toggle_flight(self):
        if self.cine.flying():
            self.cine.stop_flight()
            self.btn_fly.setText("✈  Fly Into It")
            if self.frames_shot:
                img, cap, sub = self.frames_shot[-1]
                self.cine.show_frame(img, cap, sub, animate=False)
        else:
            self.fly()

    def stop_replay(self):
        if self._replay is not None:
            self._replay.stop()
            self._replay = None

    def note_step(self, text, state="done"):
        mark = {"done": "✓", "now": "▶", "skip": "–", "warn": "⚠"}.get(state, "·")
        it = QListWidgetItem(f"{mark}  {text}")
        if state == "warn":
            it.setForeground(QColor("#e0b040"))
        elif state == "skip":
            it.setForeground(QColor("#8a93a6"))
        self.steps.addItem(it)
        self.steps.scrollToBottom()
        return it

    def say(self, text):
        self.status.setText(text)
        self.cine.set_sub(text)

    def run_job(self, fn, ok, failed=None):
        j = Job(fn)
        self.jobs.append(j)
        j.ok.connect(ok)
        j.failed.connect(failed or (lambda m: None))
        j.finished.connect(lambda j=j: self.jobs.remove(j) if j in self.jobs else None)
        j.start()

    # ------------------------------------------------------------ starting points
    def ask(self, names, counts=None, note=""):
        """The one question: which target and what kind of object. False when cancelled."""
        import targets as tg
        dlg = StartDialog(self.owner, names, counts, note)
        if getattr(self.owner, "_fast_auto", False):         # self-test: take the suggestion
            ok = True
        else:
            ok = dlg.exec() == QDialog.Accepted
        if not ok:
            return False
        self.target, self.kind = dlg.choice()
        if self.kind != tg.classify(self.target) or tg._okey(self.target) in tg.OVERRIDES:
            tg.set_override(self.target, self.kind)
        self._tick()
        return True

    def start_from_frames(self, main):
        """Page 1: the frames you opened (lights, and calibration frames if they are among them)."""
        self.main = main
        cache = self._hdr = {}
        lights = [f for f in main.frames.values() if not f.rejected and not f.error and not is_calibration(f, cache)]
        names, counts = [], {}
        for f in lights:
            t = frame_target(f, cache)
            if t:
                counts[t] = counts.get(t, 0) + 1
                if t not in names:
                    names.append(t)
        if not names:
            QMessageBox.information(main, "Fast Track", "Open a folder of light frames first (File › Open Folder) - "
                                    "Fast Track starts from the frames on this page.")
            return False
        note = ""
        if not any(is_calibration(f, cache) for f in main.frames.values()):
            note = ("No calibration frames (darks, flats, bias) are among the files on this page: the masters in your "
                    "calibration library are used if they match, otherwise the lights are stacked as they are "
                    "(fine for data that is already calibrated, like Telescope Live's).")
        if not self.ask(names, counts, note):
            return False
        self._lights = [f for f in lights if frame_target(f, cache) == self.target]
        self.show_me()
        QTimer.singleShot(200, self.stage_cull)
        return True

    def start_from_prepare(self, pw):
        """Page 3: the masters loaded there."""
        if not pw.sets:
            QMessageBox.information(pw, "Fast Track", "Add the master lights of one target first.")
            return False
        self.pw = pw
        names = [s[0].target for s in pw.sets if s]
        if not self.ask(names[:1]):
            return False
        for c in ("cull", "stack"):
            self._stage(c, "skip")
        self.show_me()
        QTimer.singleShot(200, self.stage_prepare)
        return True

    def start_from_finish(self, fw):
        """Page 4: carry on from the picture as it is now."""
        if fw.state is None:
            QMessageBox.information(fw, "Fast Track", "Add the prepared masters of one target first.")
            return False
        self.fw = fw
        if not self.ask([fw.state.target]):
            return False
        for c in ("cull", "stack", "prepare"):
            self._stage(c, "skip")
        self.show_me()
        fw.after_change()                       # the type may have changed: the step list follows it
        QTimer.singleShot(200, self.finish_loop)
        return True

    def show_me(self):
        ui_header.present(self, self.owner)
        self.raise_()

    # ------------------------------------------------------------ stage 1: frames
    def stage_cull(self):
        if self.cancelled:
            return
        self._stage("cull")
        self.say("Waiting for page 1 to finish measuring the frames…")
        self._wait_t = time.time()
        self._last_peek = 0.0
        self._wait_measured()

    def _wait_measured(self):
        if self.cancelled:
            return
        lights = self._lights
        done = [f for f in lights if f.loaded or f.error]
        self.progress.setValue(int(1000 * len(done) / max(len(lights), 1)) // 4)
        self.progress.setFormat(f"Measuring frames: {len(done)} / {len(lights)}")
        if time.time() - self._last_peek > 5 and done:
            self._last_peek = time.time()
            f = done[-1]
            self._peek_frame(f, f"Measuring frame {len(done)} of {len(lights)}", Path(f.path).name, keep=False)
        if len(done) < len(lights):
            QTimer.singleShot(500, self._wait_measured)
            return
        self._cull()

    def _peek_frame(self, f, caption, sub="", keep=True):
        import fits_core as core

        def work():
            img, _factor = core.load_frame_display(f.path)
            return img
        def ok(img):
            if keep:
                self.add_frame(img, caption, sub)
            else:
                try:
                    self.cine.show_frame(rgb8_to_qimage(img), caption, sub)
                except Exception:  # noqa: BLE001
                    pass
        self.run_job(work, ok)

    def _cull(self):
        import fits_core as core
        # measured now: sort lights from calibration frames by their headers, as page 2 will
        frames = [f for f in self.main.frames.values() if not f.rejected and not f.error]
        self._cal = [f for f in frames if is_calibration(f, self._hdr)]
        cal_ids = {id(f) for f in self._cal}
        lights = [f for f in frames if id(f) not in cal_ids and frame_target(f, self._hdr) == self.target]
        measured = [f for f in lights if f.metrics]
        left_out = set()
        if len(measured) >= 5:
            try:
                left_out = set(core.cull_preset(lights, self.main.medians, "less"))
            except Exception as e:  # noqa: BLE001
                self.note_step(f"Could not judge the frames ({e}) - all are used", "warn")
        self._kept = [f for f in lights if f.uid not in left_out]
        hours = sum(float((f.info or {}).get("exptime") or 0) for f in self._kept) / 3600
        self.note_step(f"Frames checked: {len(self._kept)} kept ({hours:.1f} h)"
                       + (f", {len(left_out)} left out (soft, trailed or cloudy)" if left_out else ""))
        best = max(self._kept, key=lambda f: (f.score is not None, f.score or 0), default=None)
        self._stage("cull", "done")
        if best is not None:
            self._peek_frame(best, "Your best frame", f"{len(self._kept)} frames go on to stacking"
                             + (f" · {len(left_out)} left out (nothing is moved or deleted)" if left_out else ""))
        QTimer.singleShot(1200, self.stage_stack)

    # ------------------------------------------------------------ stage 2: stacking
    def stage_stack(self):
        if self.cancelled:
            return
        import stack_window
        self._stage("stack")
        self.say("Calibrating and stacking - the longest part; each master appears here when it is done")
        main = self.main
        kept = self._kept
        try:
            sw = ui_header.make_page(main, "stack_win", tuple(sorted(str(f.path) for f in kept)),
                                     lambda: stack_window.StackWindow(main, kept, main.settings))
        except Exception as e:  # noqa: BLE001
            return self.fail(f"The stacking page could not start: {e}")
        self.sw = sw
        if self._cal:
            sw._add_cal_paths([str(f.path) for f in self._cal])
        warn = sorted({w for g in sw.groups if g.enabled for _k, w in g.plan.warnings})
        for w in warn[:4]:
            self.note_step(w, "warn")
        sw.start(auto=True)
        if sw.worker is None:
            return self.fail("Stacking could not start (no light frames to stack?)")
        self._n_filters = sum(1 for g in sw.groups if g.enabled)
        self._stack_item = self.note_step(f"Stacking {self._n_filters} filter(s)…", "now")
        sw.worker.progress.connect(self._stack_progress)
        sw.worker.group_done.connect(self._master_done)
        sw.worker.finished.connect(lambda: QTimer.singleShot(400, self._after_stack))

    def _stack_progress(self, text, frac):
        self.progress.setValue(250 + int(250 * max(0.0, min(frac, 1.0))))
        self.progress.setFormat(text)

    def _master_done(self, res):
        if getattr(res, "error", None):
            self.note_step(f"{res.group.label()}: failed - {res.error}", "warn")
            return
        files = [f for f in (res.files or []) if Path(f).suffix.lower() in (".fits", ".xisf")
                 and not Path(f).stem.endswith(("_Comet", "_Stars"))]
        label = res.group.label()
        self.note_step(f"Master {label}: {getattr(res, 'used', 0) or len(res.group.frames)} frames stacked")
        if not files:
            return

        def work():
            import finish as F
            import prepare as P
            m = P.load_master(files[0])
            d = m.data
            k = max(1, int(np.ceil(max(d.shape[1:]) / FRAME_SIDE)))
            return F.display_rgb8(d[:, ::k, ::k], True, linked=False)
        self.run_job(work, lambda img: self.add_frame(img, f"Master: {label}", "calibrated, aligned and stacked"))

    def _after_stack(self):
        if self.cancelled:
            return self.stopped()
        files = self.sw.master_files()
        it = getattr(self, "_stack_item", None)
        if it is not None and self.steps.row(it) >= 0:
            it.setText(f"✓  Stacked {self._n_filters} filter(s)" if files else f"⚠  Stacking {self._n_filters} filter(s)")
        if not files:
            return self.fail("Stacking made no masters - the log on page 2 says why")
        self._stage("stack", "done")
        import prepare_window
        sw = self.sw
        try:
            self.pw = ui_header.make_page(sw, "prep_win", tuple(str(f) for f in files),
                                          lambda: prepare_window.PrepareWindow(sw, sw.settings, files))
        except Exception as e:  # noqa: BLE001
            return self.fail(f"Image processing could not start: {e}")
        QTimer.singleShot(300, self.stage_prepare)

    # ------------------------------------------------------------ stage 3: page 3's steps
    def stage_prepare(self):
        if self.cancelled:
            return self.stopped()
        pw = self.pw
        self._stage("prepare")
        self.say("Image processing: crop, gradients, sharpening and star separation, each shown when done")
        pw.start(show_dialog=False)
        if pw.worker is None:
            return self.fail("Image processing could not start - see page 3")
        pw.worker.progress.connect(lambda t, f: (self.progress.setValue(500 + int(250 * max(0.0, min(f, 1.0)))),
                                                 self.progress.setFormat(t)))
        pw.worker.stepPreview.connect(self._prep_step)
        pw.worker.finished.connect(lambda: QTimer.singleShot(400, self._after_prepare))

    def _prep_step(self, title, rgb8):
        self.note_step(f"Page 3: {title}")
        self.add_frame(rgb8, title, "image processing (page 3)")

    def _after_prepare(self):
        if self.cancelled:
            return self.stopped()
        import finish_window
        import prepare_window
        res = getattr(self.pw, "last_results", None) or []
        files = prepare_window.prepared_masters(res[:1]) or prepare_window.prepared_masters(res)
        if not files:
            return self.fail("Image processing made no prepared masters - the log on page 3 says why")
        self._stage("prepare", "done")
        pw = self.pw
        try:
            fw = ui_header.make_page(pw, "finish_win", tuple(str(f) for f in files),
                                     lambda: finish_window.FinishWindow(pw, pw.settings))
        except Exception as e:  # noqa: BLE001
            return self.fail(f"Finishing Touches could not start: {e}")
        self.fw = fw
        fw.fresh_load = True
        fw.load_masters(files)
        self._load_t = time.time()
        QTimer.singleShot(300, self._wait_loaded)

    def _wait_loaded(self):
        fw = self.fw
        if fw.state is None or fw.busy():
            if time.time() - self._load_t > 600:
                return self.fail("The masters did not load on page 4")
            QTimer.singleShot(300, self._wait_loaded)
            return
        fw.fresh_load = False
        self.add_frame(self._display(), "Masters loaded on page 4", "linear (not brightened yet)")
        QTimer.singleShot(600, self.finish_loop)

    # ------------------------------------------------------------ stage 4: page 4, step by step
    def _display(self, final=False):
        import finish as F
        import finish_window as fwm
        small = self.fw.state.small(fwm.PREVIEW_SIDE)
        return F.state_display(small, with_stars=True if final else None)

    def finish_loop(self):
        if self.cancelled:
            return self.stopped()
        self._stage("finish")
        self._n_steps = getattr(self, "_n_steps", 0)
        self._loop_next()

    def _loop_next(self):
        if self.cancelled:
            return self.stopped()
        fw = self.fw
        t = fw.auto_next()
        if t is None or self._n_steps >= 40:
            return self.done()
        self._n_steps += 1
        self._cur = self.note_step(f"{t.title}…", "now")
        self.say(f"Step {self._n_steps}: {t.title}…")
        self.progress.setValue(750 + min(240, 12 * self._n_steps))
        self.progress.setFormat(f"Finishing Touches: {t.title}")
        fw.auto_step(self._step_done)

    def _step_done(self, t, label):
        if t is None:
            return self.done()
        row = self.steps.row(self._cur)
        if row >= 0:
            self.steps.takeItem(row)
        if label is None:
            self.note_step(f"{t.title} - skipped", "skip")
        else:
            self.note_step(f"{t.title}: {label}")
            try:
                self.add_frame(self._display(), t.title, label)
            except Exception:  # noqa: BLE001
                pass
        QTimer.singleShot(FADE_MS + 250, self._loop_next)

    # ------------------------------------------------------------ the end
    def done(self):
        self._stage("finish", "done")
        self.finished_ok = True
        el = int(time.time() - self.t0)
        try:
            self.add_frame(self._display(final=True), f"{self.target} — finished",
                           f"Fast Track in {el // 60} min {el % 60:02d} s · save it, or change any step by hand")
        except Exception:  # noqa: BLE001
            pass
        self.note_step(f"Finished in {el // 60}:{el % 60:02d}")
        self.progress.setValue(1000)
        self.progress.setFormat("Finished")
        self.status.setText("Finished. Save the picture, play the film again, or change any step by hand.")
        if self.fw is not None:
            self.fw.log(f"Fast Track finished {self.target} in {el // 60}:{el % 60:02d} "
                        f"({self._n_steps} step(s) on page 4)")
        self._end_buttons(True)
        QTimer.singleShot(3000, self.fly)              # the finale: a slow flight into the finished picture

    def _end_buttons(self, finished):
        self.btn_cancel.setVisible(False)
        self.btn_play.setVisible(bool(self.frames_shot))
        self.btn_fly.setVisible(finished and self.fw is not None and self.fw.state is not None)
        has_fw = self.fw is not None and self.fw.state is not None
        self.btn_save.setVisible(finished and has_fw)
        self.btn_proj.setVisible(finished and has_fw)
        self.btn_astro.setVisible(finished and has_fw)
        self.btn_tune.setVisible(has_fw)
        self.btn_redo.setVisible(has_fw)

    def fail(self, msg):
        self.note_step(msg, "warn")
        self.status.setText(msg)
        self.progress.setFormat("Stopped")
        self.cancelled = True
        self._end_buttons(False)

    def cancel(self):
        self.cancelled = True
        self.say("Stopping after the current step…")
        try:
            if self.sw is not None and self.sw.worker is not None and self.sw.worker.isRunning():
                self.sw.worker.cancel()
            if self.pw is not None and self.pw.worker is not None and self.pw.worker.isRunning():
                self.pw.cancel()
        except RuntimeError:
            pass
        QTimer.singleShot(500, self.stopped)

    def stopped(self):
        self.note_step("Stopped", "warn")
        self.progress.setFormat("Stopped")
        self.status.setText("Fast Track stopped. Everything done so far is kept on its page.")
        self._end_buttons(False)

    def fine_tune(self):
        if self.fw is None:
            return
        self.stop_replay()
        ui_header.present(self.fw, self)
        self.close()

    def redo_by_hand(self):
        fw = self.fw
        if fw is None or fw.project is None:
            return
        self.stop_replay()
        if fw.project.steps and len(fw.project.steps) > 1:
            fw.project.truncate(1)
            fw.state = fw.project.load(0)
            fw.log("Back to the masters as loaded - go through the steps yourself (the Fast Track film is gone, "
                   "but its settings were the suggested ones: each step's card marks them)")
            fw.after_change()
        ui_header.present(fw, self)
        self.close()

    def closeEvent(self, ev):
        self.stop_replay()
        self.cine.stop_flight()
        if not self.finished_ok and not self.cancelled and self.stage is not None:
            if getattr(self.owner, "_fast_auto", False) or QMessageBox.question(
                    self, "Fast Track", "Fast Track is still working. Stop it and close?") == QMessageBox.Yes:
                self.cancel()
            else:
                ev.ignore()
                return
        for j in list(self.jobs):
            j.wait(5000)
        super().closeEvent(ev)


def start(owner, where: str):
    """Open Fast Track from a page: where = 'frames' (page 1), 'prepare' (page 3) or 'finish' (page 4)."""
    w = FastTrackWindow(owner, getattr(owner, "settings", None))
    owner._fast_track = w
    ok = {"frames": w.start_from_frames, "prepare": w.start_from_prepare,
          "finish": w.start_from_finish}[where](owner)
    if not ok:
        w.deleteLater()
        owner._fast_track = None
        return None
    return w
