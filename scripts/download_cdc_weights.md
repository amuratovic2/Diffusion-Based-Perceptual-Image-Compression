# CDC Weights

Download the pretrained CDC checkpoints manually from:

https://huggingface.co/rhyang/CDC_params

Place them under:

```text
checkpoints/
```

The smoke-tested checkpoint in this workspace is:

```text
checkpoints/cdc_xparam_b00032_aux0.pt
```

It comes from:

```text
https://huggingface.co/rhyang/CDC_params/resolve/main/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt
```

The wrapper expects a checkpoint path passed with `--ckpt`, for example:

```powershell
.\.venv\Scripts\python.exe src\run_cdc.py `
  --variant xparam `
  --ckpt checkpoints\xparam_checkpoint.pt `
  --input data\imagenet_subset_256 `
  --output results\cdc_xparam `
  --lpips-weight 0.0
```

Use the `--lpips-weight` value that matches the checkpoint. The CDC test script
comments indicate either `0.0` or `0.9`.
