"""make_icon.py - builds the app icons from the logo artwork.

The logo (the Night Patch badge and its simple app icon) lives in assets/logo/ as SVG, with
the text already turned into outlines; the PNGs next to them are rendered from those SVGs.
This script turns assets/logo/subzeroastro-icon-1024.png into:
  SubZeroAstro.ico   Windows (16-256 px, the full rounded tile)
  SubZeroAstro.icns  macOS (on Apple's icon grid: 824 px tile with margin and a soft shadow)
  icon_256.png       the window/taskbar icon the app loads at run time
  icon_1024.png      the macOS-style master
Run it from the assets folder:  python make_icon.py
"""
from PIL import Image, ImageDraw, ImageFilter

src = Image.open("logo/subzeroastro-icon-1024.png").convert("RGBA")

# Windows: the full tile
win = src.resize((256, 256), Image.LANCZOS)
win.save("icon_256.png")
win.save("SubZeroAstro.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

# macOS: 824 px tile centred on a 1024 canvas, with a soft drop shadow
S, inner = 1024, 824
m = (S - inner) // 2
tile = src.resize((inner, inner), Image.LANCZOS)
shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(shadow).rounded_rectangle([m, m + 12, m + inner, m + inner + 12], radius=int(inner * 0.222),
                                         fill=(0, 0, 0, 100))
shadow = shadow.filter(ImageFilter.GaussianBlur(18))
mac = shadow.copy()
mac.alpha_composite(tile, (m, m))
mac.save("icon_1024.png")
mac.save("SubZeroAstro.icns")
print("ok")
