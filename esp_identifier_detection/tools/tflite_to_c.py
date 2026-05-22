#!/usr/bin/env python3
"""Convert a .tflite file into the C array used by the ESP-IDF app."""

from __future__ import annotations

import argparse
from pathlib import Path


def format_bytes(data: bytes) -> str:
    lines = []
    for index in range(0, len(data), 12):
        chunk = ", ".join(f"0x{value:02x}" for value in data[index : index + 12])
        lines.append(f"    {chunk},")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("main/identifier_model_data.cc"),
    )
    args = parser.parse_args()

    data = args.model.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "#include \"identifier_model_data.h\"\n\n"
        "alignas(16) extern const unsigned char g_identifier_model_data[] = {\n"
        f"{format_bytes(data)}\n"
        "};\n"
        f"extern const unsigned int g_identifier_model_data_len = {len(data)};\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
