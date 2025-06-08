from typing import Dict, Tuple, Optional
import torch

# tools
from tools.preview_tools import prepare_callback
from tools.sampler_tools import (
    Guider_Basic,
    sampler_object,
    calculate_sigmas,
    Noise_RandomNoise,
    Noise_EmptyNoise,
)

# cores
from cores.types import IO
from cores.utils import PROGRESS_BAR_ENABLED
from cores.sample import (
    fix_empty_latent_channels,
    prepare_noise,
    sample,
    fix_empty_latent_channels,
)
from cores.model_management_utils import intermediate_device


def ksampler(
    model: IO.MODEL,
    latent: Dict[str, IO.LATENT],
    poitive_condition: IO.CONDITIONING,
    negative_condition: IO.CONDITIONING,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
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


def basic_guider(model: IO.MODEL, conditioning: IO.CONDITIONING) -> IO.GUIDER:
    """
    Create a basic guider object for the given model and apply the provided conditioning.

    Args:
        model (IO.MODEL): The neural network model to use for guidance.
        conditioning (IO.CONDITIONING): The conditioning information (e.g., prompt embeddings) to guide the model.

    Returns:
        IO.GUIDER: A guider object set up with the provided model and conditioning.

    Raises:
        RuntimeError: If the model or conditioning is invalid.
    """
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    return guider


def ksampler_select(sampler_name: str) -> IO.SAMPLER:
    """
    Select a sampler object by its name.

    Args:
        sampler_name (str): The name of the sampler algorithm to retrieve.

    Returns:
        IO.SAMPLER: The sampler object corresponding to the specified name.

    Raises:
        ValueError: If the sampler name is invalid or not supported.
    """

    sampler = sampler_object(sampler_name)
    return sampler


def basic_scheduler(
    model: IO.MODEL,
    scheduler: str = "normal",
    steps: IO.INT = 20,
    denoise: IO.FLOAT = 1.0,
) -> IO.SIGMAS:
    """
    Generate a sigma schedule for diffusion sampling.

    Args:
        model (IO.MODEL): The neural network model for which to calculate sigmas.
        scheduler (str, optional): Scheduler algorithm to use (e.g., 'normal', 'karras'). Defaults to 'normal'.
        steps (IO.INT, optional): Number of denoising steps. Defaults to 20.
        denoise (IO.FLOAT, optional): Denoising factor between 0.0 and 1.0. Defaults to 1.0.

    Returns:
        IO.SIGMAS: The computed sigma schedule for diffusion steps.

    Notes:
        - If denoise < 1.0, the number of steps is increased accordingly.
        - If denoise <= 0.0, returns an empty sigma tensor.
    """
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps / denoise)

    sigmas = calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    sigmas = sigmas[-(steps + 1) :]
    return sigmas


def split_sigmas(sigmas: IO.SIGMAS, step: IO.INT = 4) -> Tuple[IO.SIGMAS, IO.SIGMAS]:
    """
    Split the sigma schedule at a specified step index.

    Args:
        sigmas (IO.SIGMAS): The full sigma schedule tensor.
        step (IO.INT, optional): The split index. Defaults to 0.

    Returns:
        Tuple[IO.SIGMAS, IO.SIGMAS]: (sigmas1, sigmas2)
            - sigmas1: Sigmas up to and including the split step.
            - sigmas2: Sigmas from the split step onward.

    Raises:
        IndexError: If the split index is out of bounds.
    """
    sigmas1 = sigmas[: step + 1]
    sigmas2 = sigmas[step:]
    return (sigmas1, sigmas2)


def random_noise(noise_seed: Optional[IO.INT] = 0) -> IO.NOISE:
    """
    Generate random noise based on a seed value.

    Args:
        noise_seed (IO.INT, optional): Seed value for noise generation. If None, returns empty noise. Defaults to 0.
    Returns:
        IO.NOISE: The generated noise object.

    Notes:
        - If noise_seed is None, returns an empty noise object.
    """
    if noise_seed is None:
        return Noise_EmptyNoise()
    return Noise_RandomNoise(noise_seed)


def sampler_custom_advanced(
    noise: IO.NOISE,
    guider: IO.GUIDER,
    sampler: IO.SAMPLER,
    sigmas: IO.SIGMAS,
    latent_image: IO.LATENT,
) -> Tuple[IO.LATENT, IO.LATENT]:
    """
    Perform advanced custom sampling using the provided noise, guider, sampler, and sigma schedule.

    Args:
        noise (IO.NOISE): The noise object for the initial latent.
        guider (IO.GUIDER): The guider object containing the model and conditioning.
        sampler (IO.SAMPLER): The sampler algorithm to use.
        sigmas (IO.SIGMAS): The sigma schedule for diffusion.
        latent_image (IO.LATENT): The initial latent image.

    Returns:
        Tuple[IO.LATENT, IO.LATENT]: (out, out_denoised)
            - out: The final latent after sampling.
            - out_denoised: The denoised latent (if available), otherwise the same as out.

    Raises:
        RuntimeError: If sampling fails or an invalid configuration is detected.

    Notes:
        - Supports custom callbacks and progress bar.
        - Handles empty latent channels and noise masks.
    """
    latent = latent_image
    latent_image = latent["samples"]
    latent = latent.copy()
    latent_image = fix_empty_latent_channels(guider.model_patcher, latent_image)
    latent["samples"] = latent_image

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    callback = prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    print(sigmas.shape, "gg")
    disable_pbar = not PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(latent),
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    samples = samples.to(intermediate_device())

    out = latent.copy()
    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = guider.model_patcher.model.process_latent_out(
            x0_output["x0"].cpu()
        )
    else:
        out_denoised = out
    return (out, out_denoised)
