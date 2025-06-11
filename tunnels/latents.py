from typing import Dict, List, Tuple
import torch

# cores
from cores.model_management_utils import intermediate_device
from cores.types import IO
from cores.node_helpers import conditioning_set_values


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

def inpaintmodelconditioning(
        positive,
        negative,
        vae,
        pixels,
        mask,
        noise_mask=True
    ) -> Tuple[IO.CONDITIONING, IO.CONDITIONING, Dict[str, IO.LATENT]]:

    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:,:,:,i] -= 0.5
        pixels[:,:,:,i] *= m
        pixels[:,:,:,i] += 0.5
    concat_latent = vae.encode(pixels)
    orig_latent = vae.encode(orig_pixels)

    out_latent = {}

    out_latent["samples"] = orig_latent
    if noise_mask:
        out_latent["noise_mask"] = mask

    out = []
    for conditioning in [positive, negative]:
        c = conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                "concat_mask": mask})
        out.append(c)
    return (out[0], out[1], out_latent)