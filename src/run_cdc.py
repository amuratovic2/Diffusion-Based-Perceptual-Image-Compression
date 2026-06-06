from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CDC_ROOT = PROJECT_ROOT / "external" / "CDC_compression"


def add_variant_to_path(variant: str) -> Path:
    variant_root = CDC_ROOT / variant
    if not variant_root.exists():
        raise FileNotFoundError(f"Missing CDC variant folder: {variant_root}")
    sys.path.insert(0, str(variant_root))
    return variant_root


def choose_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_xparam(lpips_weight: float):
    import torch
    from ema_pytorch import EMA
    from modules.compress_modules import ResnetCompressor
    from modules.denoising_diffusion import GaussianDiffusion
    from modules.unet import Unet

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1, 2, 3, 4, 5, 6],
        context_dim_mults=[1, 2, 3, 4],
        embd_type="01",
    )
    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2, 1],
        hyper_dims_mults=[4, 4, 4],
        channels=3,
        out_channels=64,
    )
    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    return diffusion, ema


def build_epsilon(lpips_weight: float):
    from modules.compress_modules import BigCompressor
    from modules.denoising_diffusion import GaussianDiffusion
    from modules.unet import Unet

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )
    context_model = BigCompressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )
    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        num_timesteps=20000,
        loss_type="l1",
        clip_noise="none",
        vbr=False,
        lagrangian=0.9,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=lpips_weight,
        aux_loss_type="lpips",
    )
    return diffusion, None


def load_model(variant: str, ckpt: Path, lpips_weight: float, device):
    import torch

    add_variant_to_path(variant)
    if variant == "xparam":
        diffusion, ema = build_xparam(lpips_weight)
        loaded = torch.load(ckpt, map_location="cpu")
        ema.load_state_dict(loaded["ema"])
        model = ema.ema_model
    elif variant == "epsilonparam":
        diffusion, _ = build_epsilon(lpips_weight)
        loaded = torch.load(ckpt, map_location="cpu")
        diffusion.load_state_dict(loaded["model"])
        model = diffusion
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CDC inference on a folder of images.")
    parser.add_argument("--variant", choices=["xparam", "epsilonparam"], default="xparam")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lpips-weight", type=float, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    import torchvision
    from PIL import Image

    from common import ensure_dir, iter_images, write_rows_csv

    device = choose_device(args.device)
    steps = args.steps if args.steps is not None else (65 if args.variant == "xparam" else 200)
    model = load_model(args.variant, Path(args.ckpt), args.lpips_weight, device)

    input_root = Path(args.input)
    output_root = ensure_dir(args.output)
    image_root = ensure_dir(output_root / "images")
    rows = []

    for img_path in iter_images(input_root):
        tensor = torchvision.io.read_image(str(img_path)).unsqueeze(0).float().to(device) / 255.0
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        tensor = tensor[:, :3, :, :]
        with torch.no_grad():
            if args.variant == "xparam":
                recon, bpp = model.compress(
                    tensor * 2.0 - 1.0,
                    sample_steps=steps,
                    bpp_return_mean=True,
                    init=torch.randn_like(tensor) * args.gamma,
                )
            else:
                recon, bpp = model.compress(
                    tensor * 2.0 - 1.0,
                    sample_steps=steps,
                    sample_mode="ddim",
                    bpp_return_mean=False,
                    init=torch.randn_like(tensor) * args.gamma,
                )
        recon = recon.clamp(-1, 1) / 2.0 + 0.5
        out_path = image_root / img_path.with_suffix(".png").name
        torchvision.utils.save_image(recon.cpu(), str(out_path))
        bpp_value = float(bpp.mean().detach().cpu().item()) if hasattr(bpp, "detach") else float(bpp)
        rows.append({"filename": out_path.name, "codec": f"cdc_{args.variant}", "bpp": f"{bpp_value:.8f}"})
        print(f"{img_path.name}: bpp={bpp_value:.6f}")

    write_rows_csv(output_root / "bitrates.csv", rows)
    print(f"Wrote {len(rows)} CDC reconstructions to {image_root}")


if __name__ == "__main__":
    main()
