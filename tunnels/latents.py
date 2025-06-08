from typing import Dict
import torch

# cores
from cores.model_management_utils import intermediate_device
from cores.types import IO


def empty_latent(
    width: int = 512, height: int = 512, batch_size: int = 1
) -> Dict[str, IO.LATENT]:
    """
    Create an empty latent tensor for diffusion models.

    Args:
        width (int): Output image width (pixels).
        height (int): Output image height (pixels).
        batch_size (int): Number of latent samples in the batch.

    Returns:
        Dict[str, IO.LATENT]: Dictionary with a key "samples" and a value tensor of shape
            (batch_size, 4, height//8, width//8), filled with zeros,
            allocated on the device selected by `intermediate_device()`.

    Example:
        latent = empty_latent(512, 512, 2)
        >>> latent["samples"].shape == (2, 4, 64, 64)
    """
    device = intermediate_device()
    latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
    return {"samples": latent}


def empty_rgb(
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    red: int = 0,
    green: int = 0,
    blue: int = 0,
) -> IO.IMAGE:
    """
    Create a batch of blank RGB images as float32 torch tensors in (B, H, W, 3) format.

    Args:
        width (int): Image width (pixels).
        height (int): Image height (pixels).
        batch_size (int): Number of images in the batch.
        red (int): Red channel value (0-255).
        green (int): Green channel value (0-255).
        blue (int): Blue channel value (0-255).

    Returns:
        IO.IMAGE: Tensor of shape (batch_size, height, width, 3),
            values in [0.0, 1.0], where [:, :, :, 0] = R, [:, :, :, 1] = G, [:, :, :, 2] = B.

    Example:
        rgb = empty_rgb(256, 256, 4, red=255)
        >>> rgb.shape == (4, 256, 256, 3)
    """
    r_norm = torch.full([batch_size, height, width, 1], red / 255.0)
    g_norm = torch.full([batch_size, height, width, 1], green / 255.0)
    b_norm = torch.full([batch_size, height, width, 1], blue / 255.0)
    rgb_image = torch.cat((r_norm, g_norm, b_norm), dim=-1)
    return rgb_image


def empty_sd3_latent_image(
    width: int = 1024, height: int = 1024, batch_size: int = 1
) -> Dict[str, IO.LATENT]:
    """
    Create an empty latent tensor for diffusion models.

    Args:
        width (int): Output image width (pixels).
        height (int): Output image height (pixels).
        batch_size (int): Number of latent samples in the batch.

    Returns:
        Dict[str, IO.LATENT]: Dictionary with a key "samples" and a value tensor of shape
            (batch_size, 16, height//8, width//8), filled with zeros,
            allocated on the device selected by `intermediate_device()`.

    Example:
        latent = empty_latent(1024, 1024, 2)
        >>> latent["samples"].shape == (2, 16, 128, 128)
    """
    device = intermediate_device()
    latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=device)
    return {"samples": latent}
