from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import ensure_dir, iter_images, safe_stem


def center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a flat resized image subset.")
    parser.add_argument("--input", required=True, help="Root folder with images.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of images.")
    parser.add_argument("--size", type=int, default=256, help="Output square size.")
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Resize without center-cropping. Default is center crop to square.",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = ensure_dir(args.output)
    rows = []

    for index, path in enumerate(iter_images(input_root)):
        if index >= args.limit:
            break
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if not args.no_crop:
                    img = center_crop_square(img)
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
                out_name = f"{index:06d}_{safe_stem(path, input_root)}.png"
                out_path = output_root / out_name
                img.save(out_path)
                rows.append(out_name)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    print(f"Wrote {len(rows)} images to {output_root}")


if __name__ == "__main__":
    main()

