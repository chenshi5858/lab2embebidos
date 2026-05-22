#!/usr/bin/env python3
"""Convert an image into main/static_image_data.cc for ESP inference."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SIZE = 96


def format_array(values: bytes) -> str:
    lines = []
    for i in range(0, len(values), 16):
        chunk = ", ".join(f"{value:3d}" for value in values[i : i + 16])
        lines.append(f"    {chunk},")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("main/static_image_data.cc"),
    )
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Falta Pillow. Instala con: pip install Pillow") from exc

    with Image.open(args.image) as img:
        img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        values = img.tobytes()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "#include \"static_image_data.h\"\n\n"
        "const uint8_t g_static_image_data[kImageElementCount] = {\n"
        f"{format_array(values)}\n"
        "};\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} from {args.image}")


if __name__ == "__main__":
    main()
