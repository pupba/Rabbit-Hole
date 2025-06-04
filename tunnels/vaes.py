from typing import Dict
import torch
import math

# cores
from cores.types import IO


def vae_decode(vae: IO.VAE, samples: Dict[str, IO.LATENT]) -> IO.IMAGE:
    """
    Decodes a latent tensor into an RGB image using the provided VAE.

    Args:
        vae (IO.VAE): A VAE object with a .decode() method.
        samples (Dict[str, IO.LATENT]): Dictionary with a "samples" key containing the latent tensor.

    Returns:
        IO.IMAGE: Decoded image tensor of shape (B, H, W, 3), float32 in [0.0, 1.0].
    """
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5:  # Combine batches
        images = images.reshape(
            -1, images.shape[-3], images.shape[-2], images.shape[-1]
        )
    return images


def vae_encode(vae: IO.VAE, pixel_images: IO.IMAGE) -> Dict[str, IO.LATENT]:
    """
    Encodes RGB image tensors into latent representations using the provided VAE.

    Args:
        vae (IO.VAE): A VAE object with an .encode() method.
        pixel_images (IO.IMAGE): Image tensor of shape (B, H, W, 3), float32 in [0.0, 1.0].

    Returns:
        Dict[str, IO.LATENT]: Dictionary with a "samples" key containing the encoded latent tensor.
    """
    latent = vae.encode(pixel_images[:, :, :, :3])
    return {"samples": latent}


def vae_encode_inpaint(
    vae: IO.VAE, pixel_images: IO.IMAGE, mask: IO.MASK, grow_mask_by: IO.INT = 6
) -> Dict[str, IO.LATENT]:
    """
    Encodes an image for inpainting, with mask dilation and zero-centering of masked regions.

    Args:
        vae (IO.VAE): A VAE object with .encode() and .downscale_ratio.
        pixel_images (IO.IMAGE): Input image tensor (B, H, W, 3), float32 in [0.0, 1.0].
        mask (IO.MASK): Inpainting mask tensor (B, H, W) or (B, 1, H, W), 1=masked.
        grow_mask_by (IO.INT): Dilation kernel size for mask growing (default: 6).

    Returns:
        Dict[str, IO.LATENT]: Dictionary containing:
            - "samples": latent tensor after encoding
            - "noise_mask": grown mask tensor (rounded, dtype float32), shape matches latent spatial dims.
    """
    # 1. resize/crop to downscale multiple
    x = (pixel_images.shape[1] // vae.downscale_ratio) * vae.downscale_ratio
    y = (pixel_images.shape[2] // vae.downscale_ratio) * vae.downscale_ratio
    mask = torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(pixel_images.shape[1], pixel_images.shape[2]),
        mode="bilinear",
    )

    clone_pixel_images = pixel_images.clone()

    if clone_pixel_images.shape[1] != x or clone_pixel_images.shape[2] != y:
        x_offset = (clone_pixel_images.shape[1] % vae.downscale_ratio) // 2
        y_offset = (clone_pixel_images.shape[2] % vae.downscale_ratio) // 2
        pixels = clone_pixel_images[
            :, x_offset : x + x_offset, y_offset : y + y_offset, :
        ]
        mask = mask[:, :, x_offset : x + x_offset, y_offset : y + y_offset]

    # 2. Grow (dilate) mask for seamless blending in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding),
            0,
            1,
        )
    # 3. Zero-center masked pixels for inpainting
    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5

    # 4. VAE Encode
    latent = vae.encode(pixels)

    return {"samples": latent, "noise_mask": (mask_erosion[:, :, :x, :y].round())}
