import math
from PIL import Image, ImageDraw, ImageFilter, ImageChops

S = 1024 * 2                     # draw big, scale down (antialiasing)
def C(x): return int(x * S / 1024)

def rounded_mask(size, r):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    return m

def radial(size, cx, cy, rad, color, alpha):
    g = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(g)
    steps = 60
    for i in range(steps, 0, -1):
        rr = rad * i / steps
        a = int(alpha * (1 - i / steps) ** 1.6)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=a)
    layer = Image.new("RGBA", (size, size), color + (0,))
    layer.putalpha(g)
    return layer

def make():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # background: deep sky, vertical gradient
    bg = Image.new("RGBA", (S, S))
    px = bg.load()
    for y in range(S):
        t = y / S
        top, bot = (18, 34, 70), (4, 6, 14)
        c = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(S):
            px[x, y] = c + (255,)
    # nebula glows
    for (cx, cy, r, col, a) in ((330, 640, 430, (190, 40, 75), 150), (700, 380, 420, (20, 150, 165), 140),
                                (560, 560, 300, (235, 160, 70), 60)):
        bg = Image.alpha_composite(bg, radial(S, C(cx), C(cy), C(r), col, a))
    # faint background stars
    import random
    rnd = random.Random(1805)
    d = ImageDraw.Draw(bg)
    for _ in range(90):
        x, y, r = rnd.uniform(0, S), rnd.uniform(0, S), rnd.choice((1.2, 1.6, 2.2, 3.0))
        a = rnd.randint(90, 220)
        d.ellipse([x - C(r), y - C(r), x + C(r), y + C(r)], fill=(220, 232, 255, a))
    # three stacked sub-frames, converging toward the centre (subs -> master)
    frames = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frames)
    for i, (dx, dy, a) in enumerate(((-150, -120, 70), (-75, -60, 110), (0, 0, 170))):
        x0, y0 = C(250 + dx), C(300 + dy)
        x1, y1 = C(774 + dx), C(724 + dy)
        fd.rounded_rectangle([x0, y0, x1, y1], radius=C(34), fill=(120, 170, 235, a // 4),
                             outline=(175, 215, 255, a), width=C(9))
    bg = Image.alpha_composite(bg, frames)
    # the ice-crystal star: glow, six arms with side branches, bright core
    cx, cy = C(512), C(512)
    bg = Image.alpha_composite(bg, radial(S, cx, cy, C(330), (170, 215, 255), 200))
    star = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(star)
    L, w = C(300), C(30)
    for k in range(6):
        ang = math.pi / 3 * k + math.pi / 2
        ex, ey = cx + L * math.cos(ang), cy - L * math.sin(ang)
        sd.line([cx, cy, ex, ey], fill=(235, 245, 255, 255), width=w)
        sd.ellipse([ex - w / 2, ey - w / 2, ex + w / 2, ey + w / 2], fill=(235, 245, 255, 255))
        for frac in (0.55,):
            bx, by = cx + L * frac * math.cos(ang), cy - L * frac * math.sin(ang)
            for side in (-1, 1):
                ba = ang + side * math.pi / 4
                b2x, b2y = bx + L * 0.28 * math.cos(ba), by - L * 0.28 * math.sin(ba)
                sd.line([bx, by, b2x, b2y], fill=(235, 245, 255, 255), width=int(w * 0.8))
                sd.ellipse([b2x - w * 0.4, b2y - w * 0.4, b2x + w * 0.4, b2y + w * 0.4], fill=(235, 245, 255, 255))
    glow = star.filter(ImageFilter.GaussianBlur(C(18)))
    bg = Image.alpha_composite(bg, glow)
    bg = Image.alpha_composite(bg, star)
    core = radial(S, cx, cy, C(120), (255, 255, 255), 255)
    bg = Image.alpha_composite(bg, core)
    d = ImageDraw.Draw(bg)
    d.ellipse([cx - C(58), cy - C(58), cx + C(58), cy + C(58)], fill=(255, 255, 255, 255))
    # rounded tile (macOS-style grid: 824/1024 content with margin), thin rim
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    m = C(100)
    inner = S - 2 * m
    t = bg.resize((inner, inner), Image.LANCZOS)
    mask = rounded_mask(inner, int(inner * 0.225))
    tile.paste(t, (m, m), mask)
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(rim).rounded_rectangle([m, m, m + inner - 1, m + inner - 1], radius=int(inner * 0.225),
                                          outline=(90, 140, 210, 200), width=C(6))
    tile = Image.alpha_composite(tile, rim)
    # soft drop shadow
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([m, m + C(14), m + inner, m + inner + C(14)], radius=int(inner * 0.225),
                                         fill=(0, 0, 0, 110))
    sh = sh.filter(ImageFilter.GaussianBlur(C(20)))
    out = Image.alpha_composite(sh, tile)
    return out.resize((1024, 1024), Image.LANCZOS)

icon = make()
icon.save("icon_1024.png")
# Windows .ico: the full-bleed tile looks best on Windows, so crop the macOS margin away
win = icon.crop((88, 88, 936, 936)).resize((256, 256), Image.LANCZOS)
win.save("SubZeroAstro.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
icon.save("SubZeroAstro.icns")
win.save("icon_256.png")
# preview: sizes side by side on light and dark
prev = Image.new("RGBA", (1100, 300), (245, 245, 245, 255))
x = 10
for s in (256, 128, 64, 48, 32, 16):
    im = win.resize((s, s), Image.LANCZOS)
    prev.alpha_composite(im, (x, 20 + (256 - s)))
    x += s + 20
dark = Image.new("RGBA", (1100, 300), (32, 32, 36, 255))
x = 10
for s in (256, 128, 64, 48, 32, 16):
    im = icon.resize((s, s), Image.LANCZOS)
    dark.alpha_composite(im, (x, 20 + (256 - s)))
    x += s + 20
both = Image.new("RGBA", (1100, 600)); both.paste(prev, (0, 0)); both.paste(dark, (0, 300))
both.convert("RGB").save("preview.png")
print("ok")
