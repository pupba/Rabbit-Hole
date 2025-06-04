from typing import Dict, Optional
import torch

# tools
from tools.preview_tools import prepare_callback

# cores
from cores.types import IO
from cores.sample import fix_empty_latent_channels, prepare_noise, sample


def ksampler(
    model: IO.MODEL,
    latent: Dict[str, IO.LATENT],
    seed: int = 0,
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    poitive_condition: Optional[IO.CONDITIONING] = None,
    negative_condition: Optional[IO.CONDITIONING] = None,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step: int = None,
    last_step: int = None,
    force_full_denoise: bool = False,
    disable_pbar: bool = False,
) -> Dict[str, IO.LATENT]:
    """
    Denoises a latent tensor using the specified diffusion model and sampling configuration.

    Args:
        model (IO.MODEL): The diffusion model used for denoising.
        latent (Dict[str, IO.LATENT]): Dictionary containing the input latent tensor under "samples".
        seed (int): Seed for random noise generation (default: 0).
        steps (int): Number of denoising steps (default: 20).
        cfg (float): Classifier-Free Guidance scale (default: 7.0).
        sampler_name (str): Name of the sampler algorithm to use (e.g., "euler").
        scheduler (str): Scheduler type for sampling (e.g., "normal").
        poitive_condition (Optional[IO.CONDITIONING]): Positive conditioning (prompt embedding).
        negative_condition (Optional[IO.CONDITIONING]): Negative conditioning (negative prompt embedding).
        denoise (float): Amount of denoising to apply, from 0.0 (none) to 1.0 (full noise) (default: 1.0).
        disable_noise (bool): If True, disables noise and uses zeros (default: False).
        start_step (int, optional): Start step for sampling (advanced).
        last_step (int, optional): Last step for sampling (advanced).
        force_full_denoise (bool): If True, forces full denoising even with low denoise values.
        disable_pbar (bool): If True, disables the progress bar display (default: False).

    Returns:
        Dict[str, IO.LATENT]: Dictionary with the denoised latent tensor under "samples", preserving other input keys.

    Notes:
        - This function prepares and applies random noise, supports noise masking, and uses a callback for progress updates.
        - The output dictionary has the same structure as the input latent, but with updated denoised samples.
    """
    # 1. Latent Channel match
    latent_image = fix_empty_latent_channels(model, latent["samples"])

    # 2. Create Noise
    if disable_noise:
        noise = torch.zeros_like(latent_image, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = prepare_noise(latent_image, seed, batch_inds)

    # 3. Noise_mask support
    noise_mask = latent.get("noise_mask", None)
    # 4. Callback
    callback = prepare_callback(model, steps)  # "x0_output_dict" future update...maybe?
    # 5. Sampler Callback(Here)
    samples = sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=poitive_condition,
        negative=negative_condition,
        latent_image=latent_image,
        denoise=denoise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    # 6. result
    out = latent.copy()
    out["samples"] = samples
    return out
