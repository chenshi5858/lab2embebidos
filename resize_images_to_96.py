import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reescala imagenes al tamano de entrada de los modelos (por defecto 96x96)."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Carpeta con imagenes originales.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Carpeta de salida.")
    parser.add_argument("--size", type=int, default=96, help="Tamano objetivo (size x size).")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca imagenes recursivamente dentro de input_dir.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convierte imagenes a monocromatico (modo L).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe archivos si ya existen en output_dir.",
    )
    return parser.parse_args()


def choose_resample(width: int, height: int, target_size: int) -> Image.Resampling:
    if width > target_size or height > target_size:
        return Image.Resampling.LANCZOS
    if width < target_size or height < target_size:
        return Image.Resampling.BICUBIC
    return Image.Resampling.NEAREST


def collect_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        candidates: Iterable[Path] = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")

    images = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(images)


def process_image(
    src_path: Path,
    dst_path: Path,
    size: int,
    grayscale: bool,
    overwrite: bool,
) -> str:
    if dst_path.exists() and not overwrite:
        return "skip"

    try:
        with Image.open(src_path) as img:
            if grayscale and img.mode != "L":
                img = img.convert("L")

            resample = choose_resample(img.width, img.height, size)
            if img.size != (size, size):
                img = img.resize((size, size), resample=resample)

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {}
            if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs = {"quality": 95, "optimize": True}

            img.save(dst_path, **save_kwargs)
            return "ok"
    except (UnidentifiedImageError, OSError):
        return "error"


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir no existe o no es carpeta: {args.input_dir}")

    if args.size <= 0:
        raise ValueError("size debe ser mayor a 0")

    images = collect_images(args.input_dir, args.recursive)
    if not images:
        print("No se encontraron imagenes para procesar.")
        return

    ok_count = 0
    skip_count = 0
    error_count = 0

    for src_path in images:
        relative = src_path.relative_to(args.input_dir)
        dst_path = args.output_dir / relative

        status = process_image(
            src_path=src_path,
            dst_path=dst_path,
            size=args.size,
            grayscale=args.grayscale,
            overwrite=args.overwrite,
        )

        if status == "ok":
            ok_count += 1
        elif status == "skip":
            skip_count += 1
        else:
            error_count += 1

    print("Proceso completado.")
    print(f"Imagenes detectadas: {len(images)}")
    print(f"Procesadas: {ok_count}")
    print(f"Omitidas (ya existian): {skip_count}")
    print(f"Errores: {error_count}")
    print(f"Salida: {args.output_dir}")


if __name__ == "__main__":
    main()
