import torch
from typing import Dict

# cores
from cores.types import IO
from cores.model_management_utils import (
    get_torch_device,
    module_size,
    free_memory,
    OOM_EXCEPTION,
)
from cores.utils import get_tiled_scale_steps, ProgressBar, tiled_scale, common_upscale

# tunnels
from tunnels.samplers import ksampler


def upscale_by_model(upscale_model: IO.UPSCALE_MODEL, image: IO.IMAGE) -> IO.IMAGE:
    """
    Upscales an input image using the provided upscaling model, automatically tiling to avoid out-of-memory (OOM) errors.

    Args:
        upscale_model (IO.UPSCALE_MODEL): Pre-loaded upscaling model (e.g., Real-ESRGAN, SwinIR).
        image (IO.IMAGE): Input image as a IO.IMAGE (B, H, W, C), float32, [0, 1].

    Returns:
        IO.IMAGE: Upscaled image as a IO.IMAGE (B, new_H, new_W, C), float32, [0, 1].

    Raises:
        OOM_EXCEPTION: If the image cannot be upscaled even with the smallest allowed tile size.

    Notes:
        - The function estimates required memory and frees device memory if necessary.
        - The image is processed in overlapping tiles to handle very large images or limited GPU memory.
        - Tile size is dynamically reduced in case of OOM errors, down to a minimum of 128x128.
        - The output image is clamped to [0, 1] and has the same channel count as the input.
    """
    device = get_torch_device()
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
    memory_required += image.nelement() * image.element_size()
    free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = ProgressBar(steps)
            s = tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False
        except OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.to("cpu")
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    return s


def upscale_image(
    image: IO.IMAGE,
    width: int,
    height: int,
    upscale_method: str = "lanczos",
    crop: str = "disabled",
) -> IO.IMAGE:
    """
    Upscales an image to the target width and height using the specified interpolation and crop mode.

    Args:
        image (IO.IMAGE): Input image tensor (B, H, W, C) or (H, W, C), float32, [0,1].
        width (int): Target width. If 0, calculated to preserve aspect ratio.
        height (int): Target height. If 0, calculated to preserve aspect ratio.
        upscale_method (str): Interpolation method. One of ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"].
        crop (str): Crop mode. "disabled" (fit to size), "center" (center crop to fit).

    Returns:
        IO.IMAGE: Upscaled image tensor (same batch size, new H, new W, C).
    """
    if width == 0 and height == 0:
        return image
    samples = image.movedim(-1, 1)
    if width == 0:
        width = max(1, round(samples.shape[3] * height / samples.shape[2]))
    elif height == 0:
        height = max(1, round(samples.shape[2] * width / samples.shape[3]))
    s = common_upscale(samples, width, height, upscale_method, crop)
    return s.movedim(1, -1)


def upscale_image_by_scale(
    image: IO.IMAGE, scale_by: float = 1.0, upscale_method: str = "lanczos"
) -> IO.IMAGE:
    """
    Upscales an image by a given scale factor using the specified interpolation method.

    Args:
        image (IO.IMAGE): Input image tensor (B, H, W, C) or (H, W, C), float32, [0,1].
        scale_by (float): Scale factor (e.g., 2.0 doubles the size).
        upscale_method (str): Interpolation method. One of ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"].

    Returns:
        IO.IMAGE: Upscaled image tensor (same batch size, new H, new W, C).
    """
    samples = image.movedim(-1, 1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = common_upscale(samples, width, height, upscale_method, "disabled")
    return s.movedim(1, -1)


def upscale_latent(
    samples: Dict[str, IO.LATENT],
    upscale_method: str = "nearest-exact",
    width: int = 512,
    height: int = 512,
    crop: str = "disabled",
) -> Dict[str, IO.LATENT]:
    """
    Upscales the given latent dictionary using the specified interpolation method and target resolution.

    Args:
        samples (Dict[str, IO.LATENT]): Latent dictionary containing at least the key "samples" (tensor).
        upscale_method (str, optional): Interpolation method to use.
            One of ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]. Default is "nearest-exact".
        width (int, optional): Target width in pixels. If 0, inferred from height. Default is 512.
        height (int, optional): Target height in pixels. If 0, inferred from width. Default is 512.
        crop (str, optional): Cropping method to use after upscaling.
            One of ["disabled", "center"]. Default is "disabled".

    Returns:
        Dict[str, IO.LATENT]: The upscaled latent dictionary.

    Notes:
        - If both width and height are 0, the input is returned unchanged.
        - The upscaling is performed in latent space (resolution divided by 8).
        - If only one of width or height is 0, it is automatically inferred to maintain aspect ratio.
    """
    if width == 0 and height == 0:
        s = samples
    else:
        s = samples.copy()

        if width == 0:
            height = max(64, height)
            width = max(
                64,
                round(
                    samples["samples"].shape[-1] * height / samples["samples"].shape[-2]
                ),
            )
        elif height == 0:
            width = max(64, width)
            height = max(
                64,
                round(
                    samples["samples"].shape[-2] * width / samples["samples"].shape[-1]
                ),
            )
        else:
            width = max(64, width)
            height = max(64, height)
        s["samples"] = common_upscale(
            samples["samples"], width // 8, height // 8, upscale_method, crop
        )
    return s


def hires_fix(
    model: IO.MODEL,
    pos: IO.CONDITIONING,
    neg: IO.CONDITIONING,
    latent: Dict[str, IO.LATENT],
    upscale_configs: Dict[str, str],
    sampler_configs: Dict[str, str],
) -> Dict[str, IO.LATENT]:
    latent = upscale_latent(samples=latent, **upscale_configs)
    out = ksampler(
        model=model,
        latent=latent,
        poitive_condition=pos,
        negative_condition=neg,
        **sampler_configs
    )
    return out
