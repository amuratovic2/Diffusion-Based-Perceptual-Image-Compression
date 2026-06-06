from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

from common import ensure_dir, file_bpp, iter_images, write_rows_csv


def save_codec(img: Image.Image, path: Path, codec: str, quality: int) -> None:
    if codec == "jpeg":
        img.save(path, format="JPEG", quality=quality, optimize=True)
    elif codec == "webp":
        img.save(path, format="WEBP", quality=quality, method=6)
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pillow JPEG/WebP baselines.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec", choices=["jpeg", "webp"], default="jpeg")
    parser.add_argument("--quality", type=int, default=35)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = ensure_dir(args.output)
    image_root = ensure_dir(output_root / "images")
    rows = []

    extension = ".jpg" if args.codec == "jpeg" else ".webp"

    for src in iter_images(input_root):
        with Image.open(src) as img:
            img = img.convert("RGB")
            width, height = img.size
            encoded_name = src.with_suffix(extension).name
            encoded_path = image_root / encoded_name
            save_codec(img, encoded_path, args.codec, args.quality)

            # Decode back to PNG so all metrics compare RGB images consistently.
            recon_name = src.with_suffix(".png").name
            recon_path = image_root / recon_name
            with Image.open(encoded_path) as decoded:
                decoded.convert("RGB").save(recon_path)

            rows.append(
                {
                    "filename": recon_name,
                    "codec": args.codec,
                    "quality": args.quality,
                    "encoded_file": str(encoded_path),
                    "bpp": f"{file_bpp(encoded_path, width, height):.8f}",
                }
            )

    write_rows_csv(output_root / "bitrates.csv", rows)
    print(f"Wrote {len(rows)} reconstructions to {image_root}")


if __name__ == "__main__":
    main()

