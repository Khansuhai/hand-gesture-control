"""Draws assets/icon.ico (a white hand on a blue circle) for the desktop shortcut.

Run: .venv\\Scripts\\python.exe tools\\make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

S = 1024  # draw big, then scale down for smooth edges
OUT = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"


def rounded_bar(draw, x, y_top, y_bottom, width, fill):
    draw.rounded_rectangle([x - width / 2, y_top, x + width / 2, y_bottom], radius=width / 2, fill=fill)


def main():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Blue circle background with a lighter ring.
    d.ellipse([20, 20, S - 20, S - 20], fill=(28, 92, 230, 255))
    d.ellipse([20, 20, S - 20, S - 20], outline=(120, 170, 255, 255), width=28)

    white = (255, 255, 255, 255)
    # Fingers (index, middle, ring, pinky) as rounded bars.
    fw = 118
    for x, top in [(355, 215), (480, 170), (605, 200), (722, 290)]:
        rounded_bar(d, x, top, 640, fw, white)
    # Palm.
    d.rounded_rectangle([290, 480, 785, 850], radius=170, fill=white)
    # Thumb: a thick round-ended line from the palm out to the upper left.
    base, tip = (380, 760), (215, 545)
    d.line([base, tip], fill=white, width=fw)
    for cx, cy in (base, tip):
        d.ellipse([cx - fw / 2, cy - fw / 2, cx + fw / 2, cy + fw / 2], fill=white)

    # Small yellow "motion" arcs left of the index finger, hinting at gestures.
    for r in (80, 130):
        d.arc([300 - r, 270 - r, 300 + r, 270 + r], start=200, end=250, fill=(255, 210, 60, 255), width=24)

    OUT.parent.mkdir(exist_ok=True)
    img.save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.resize((256, 256), Image.LANCZOS).save(OUT.with_suffix(".png"))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
