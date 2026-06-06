from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from common import ensure_dir, iter_images, read_bitrates, write_rows_csv


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def psnr(ref: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((ref - recon) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def mae(ref: np.ndarray, recon: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - recon)))


def ssim_global(ref: np.ndarray, recon: np.ndarray) -> float:
    # Compact SSIM implementation over RGB channels. It is not windowed MS-SSIM,
    # but is deterministic and dependency-light for baseline reporting.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for channel in range(ref.shape[2]):
        x = ref[:, :, channel]
        y = recon[:, :, channel]
        mux = float(np.mean(x))
        muy = float(np.mean(y))
        varx = float(np.var(x))
        vary = float(np.var(y))
        cov = float(np.mean((x - mux) * (y - muy)))
        numerator = (2 * mux * muy + c1) * (2 * cov + c2)
        denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
        scores.append(numerator / denominator)
    return float(np.mean(scores))


def optional_torch_metrics(ref_paths: list[Path], recon_paths: list[Path]) -> dict[str, float | None]:
    out = {"lpips": None, "ms_ssim": None}
    try:
        import torch
        import lpips
        from pytorch_msssim import ms_ssim
    except Exception:
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    lpips_values = []
    ms_values = []

    for ref_path, recon_path in zip(ref_paths, recon_paths):
        ref = torch.from_numpy(load_rgb(ref_path)).permute(2, 0, 1).unsqueeze(0).to(device)
        recon = torch.from_numpy(load_rgb(recon_path)).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_values.append(float(lpips_model(ref * 2 - 1, recon * 2 - 1).item()))
            if min(ref.shape[-2:]) >= 160:
                ms_values.append(float(ms_ssim(ref, recon, data_range=1.0).item()))

    out["lpips"] = float(np.mean(lpips_values)) if lpips_values else None
    out["ms_ssim"] = float(np.mean(ms_values)) if ms_values else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed images.")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--recon", required=True)
    parser.add_argument("--bitrate-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--optional-torch", action="store_true")
    args = parser.parse_args()

    ref_root = Path(args.ref)
    recon_root = Path(args.recon)
    bitrates = read_bitrates(args.bitrate_csv)

    rows = []
    ref_paths_for_optional: list[Path] = []
    recon_paths_for_optional: list[Path] = []

    for ref_path in iter_images(ref_root):
        recon_path = recon_root / ref_path.name
        if not recon_path.exists():
            print(f"Missing reconstruction for {ref_path.name}")
            continue
        ref = load_rgb(ref_path)
        recon = load_rgb(recon_path)
        if ref.shape != recon.shape:
            print(f"Skipping shape mismatch {ref_path.name}: {ref.shape} vs {recon.shape}")
            continue
        row = {
            "filename": ref_path.name,
            "bpp": bitrates.get(ref_path.name, ""),
            "psnr": f"{psnr(ref, recon):.6f}",
            "ssim": f"{ssim_global(ref, recon):.6f}",
            "mae": f"{mae(ref, recon):.8f}",
        }
        rows.append(row)
        ref_paths_for_optional.append(ref_path)
        recon_paths_for_optional.append(recon_path)

    if args.optional_torch and rows:
        optional = optional_torch_metrics(ref_paths_for_optional, recon_paths_for_optional)
        for row in rows:
            row["dataset_lpips_mean"] = "" if optional["lpips"] is None else f"{optional['lpips']:.6f}"
            row["dataset_ms_ssim_mean"] = "" if optional["ms_ssim"] is None else f"{optional['ms_ssim']:.6f}"

    write_rows_csv(args.output, rows)

    if rows:
        mean_bpp = np.mean([float(r["bpp"]) for r in rows if r["bpp"] != ""]) if any(r["bpp"] != "" for r in rows) else float("nan")
        mean_psnr = np.mean([float(r["psnr"]) for r in rows])
        mean_ssim = np.mean([float(r["ssim"]) for r in rows])
        mean_mae = np.mean([float(r["mae"]) for r in rows])
        print(f"Images: {len(rows)}")
        print(f"Mean bpp: {mean_bpp:.6f}")
        print(f"Mean PSNR: {mean_psnr:.4f}")
        print(f"Mean SSIM: {mean_ssim:.6f}")
        print(f"Mean MAE: {mean_mae:.8f}")
    else:
        print("No matching image pairs found.")


if __name__ == "__main__":
    main()

