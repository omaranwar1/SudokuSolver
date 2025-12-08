"""
Utility script to generate digit template images (1-9) in multiple fonts.

Outputs 50x50 PNGs into a `digit_templates/` folder alongside the repo root.
Update `FONT_CANDIDATES` with additional font paths on your system if needed.
"""

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("digit_templates")
IMAGE_SIZE = (50, 50)
FONT_SIZE = 32
DIGITS = [str(d) for d in range(1, 10)]

# Add or remove fonts based on availability on your system.
# Preferred: bold weights from bundled fonts (32 pt): DejaVuSans-Bold, Albert Sans (wght),
# Noto Sans JP (wght), and Lato Bold if available locally.
FONT_CANDIDATES: List[Path] = [
    Path("fonts/DejaVuSans-Bold.ttf"),
    Path("fonts/AlbertSans-Bold.ttf"),
    Path("fonts/NotoSansJP-Bold.ttf"),
    Path("fonts/Lato-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/albertsans/AlbertSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansJP-Bold.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansJP-Bold.otf"),
    Path("/usr/share/fonts/google-noto/NotoSansJP-Bold.otf"),
    Path("/usr/share/fonts/truetype/lato/Lato-Bold.ttf"),
    Path("/usr/share/fonts/opentype/lato/Lato-Bold.ttf"),
    Path("/Library/Fonts/AlbertSans-Bold.ttf"),
    Path("/Library/Fonts/NotoSansJP-Bold.ttf"),
    Path("/Library/Fonts/Lato-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/AlbertSans-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/NotoSansJP-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Lato-Bold.ttf"),
]


def existing_fonts(font_paths: Iterable[Path]) -> List[Path]:
    """Return only the font paths that exist on the current machine."""
    allowed_suffixes = {".ttf", ".otf", ".ttc"}
    return [font for font in font_paths if font.exists() and font.suffix.lower() in allowed_suffixes]


def generate_templates(fonts: Iterable[Path]) -> None:
    """Generate digit template PNGs for each provided font."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    for font_path in fonts:
        try:
            font = ImageFont.truetype(str(font_path), FONT_SIZE)
        except OSError as exc:
            print(f"Skipping {font_path} (unloadable): {exc}")
            continue

        stem_lower = font_path.stem.lower()
        stroke_width = 2 if ("notosansjp" in stem_lower or "albertsans" in stem_lower) else 0

        for digit in DIGITS:
            # Grayscale 50x50 to mirror cell crops: black background, white digit.
            img = Image.new("L", IMAGE_SIZE, color=0)
            draw = ImageDraw.Draw(img)

            # Anchor at image center so font metrics/baseline don't skew alignment.
            draw.text(
                (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2),
                digit,
                font=font,
                fill=255,
                anchor="mm",
                stroke_width=stroke_width,
                stroke_fill=255,
            )

            # Fine-tune: recenter based on drawn content's bounding box center.
            bbox = img.getbbox()
            if bbox:
                target = ((IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0)
                # Center using center-of-mass of non-zero pixels for uniform alignment.
                coords = [divmod(i, IMAGE_SIZE[0]) for i, v in enumerate(img.getdata()) if v > 0]
                if coords:
                    ys, xs = zip(*coords)
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    dx = int(round(target[0] - cx))
                    dy = int(round(target[1] - cy))

                    if dx or dy:
                        recentered = Image.new("L", IMAGE_SIZE, color=0)
                        recentered.paste(img, (dx, dy))
                        img = recentered

            font_name = font_path.stem
            img.save(OUTPUT_DIR / f"{digit}_{font_name}.png")

    print(f"Template images for digits 1 to 9 generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    fonts_available = existing_fonts(FONT_CANDIDATES)
    if not fonts_available:
        print("No fonts found. Please update FONT_CANDIDATES with valid paths.")
    else:
        generate_templates(fonts_available)
