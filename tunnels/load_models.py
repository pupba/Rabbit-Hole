import torch

from typing import Tuple
from spandrel import ModelLoader, ImageModelDescriptor
from cores.types import IO
from cores.path_utils import get_full_path_or_raise, get_folder_paths
from cores.sd_model import (
    load_checkpoint_guess_config,
    load_lora_for_models,
    load_diffusion_model,
)
from cores.sd_model_class import VAE
from cores.utils import load_torch_file, state_dict_prefix_replace
from cores.controlnet import load_controlnet


def load_checkpoint(ckpt_name: str) -> Tuple[IO.MODEL, IO.CLIP, IO.VAE]:
    """
    Loads a model checkpoint and returns the core model components.

    Args:
        ckpt_name (str): The filename of the checkpoint to load.
            The function will search for this file in the "checkpoints" directory.

    Returns:
        Tuple[IO.MODEL, IO.CLIP, IO.VAE]:
            - MODEL: The main Stable Diffusion (or similar) model object.
            - CLIP:  The text encoder (CLIP) component used for text conditioning.
            - VAE:   The Variational Autoencoder component for latent-image conversion.

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
        RuntimeError: If checkpoint configuration or model type cannot be detected.

    Example:
        >>> model, clip, vae = load_checkpoint("your_checkpoint.safetensors")
    """
    ckpt_path = get_full_path_or_raise("checkpoints", ckpt_name)
    model, clip, vae, _ = load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=get_folder_paths("embeddings"),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, clip, vae


def load_vae(vae_name: str) -> IO.VAE:
    """
    Loads a VAE (Variational Autoencoder) checkpoint and returns the initialized VAE object.

    Args:
        vae_name (str): The filename of the VAE checkpoint to load.
            The function looks for this file in the "vae" directory.

    Returns:
        IO.VAE: An initialized VAE object loaded from the specified checkpoint.

    Raises:
        FileNotFoundError: If the specified VAE checkpoint file does not exist.
        RuntimeError: If the loaded checkpoint is invalid.

    Example:
        >>> vae = load_vae("your_vae.safetensors")
    """
    vae_path = get_full_path_or_raise("vae", vae_name)
    sd = load_torch_file(vae_path, safe_load=True)
    vae = VAE(sd=sd)
    vae.throw_exception_if_invalid()
    return vae


def load_lora(
    model: IO.MODEL,
    clip: IO.CLIP,
    lora_name: str,
    strength_model: float = 1.0,
    strength_clip: float = 1.0,
) -> Tuple[IO.MODEL, IO.CLIP]:
    """
    Loads a LoRA (Low-Rank Adaptation) checkpoint and applies it to the model and CLIP encoder.

    Args:
        model (IO.MODEL): The base model object to which the LoRA will be applied.
        clip (IO.CLIP): The CLIP text encoder object to which the LoRA will be applied.
        lora_name (str): The filename of the LoRA checkpoint to load (from the "loras" directory).
        strength_model (float, optional): Scaling strength for applying LoRA to the main model. Defaults to 1.0.
        strength_clip (float, optional): Scaling strength for applying LoRA to the CLIP encoder. Defaults to 1.0.

    Returns:
        Tuple[IO.MODEL, IO.CLIP]:
            - model_lora: The model object with LoRA applied.
            - clip_lora:  The CLIP encoder with LoRA applied.

    Notes:
        - If both `strength_model` and `strength_clip` are 0, returns the original model and CLIP objects unchanged.
        - Internally caches the last loaded LoRA to avoid redundant file reads.

    Raises:
        FileNotFoundError: If the specified LoRA checkpoint does not exist.
        RuntimeError: If the checkpoint fails to load or apply.

    Example:
        >>> model_lora, clip_lora = load_lora(model, clip, "MyLora.safetensors", 0.8, 0.6)
    """
    loaded_lora = None
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)

    lora_path = get_full_path_or_raise("loras", lora_name)
    lora = None
    if loaded_lora is not None:
        if loaded_lora[0] == lora_path:
            lora = loaded_lora[1]
        else:
            loaded_lora = None

    if lora is None:
        lora = load_torch_file(lora_path, safe_load=True)
        loaded_lora = (lora_path, lora)

    model_lora, clip_lora = load_lora_for_models(
        model, clip, lora, strength_model, strength_clip
    )
    return (model_lora, clip_lora)


def load_control_net(control_net_name: str) -> IO.CONTROL_NET:
    """
    Loads a ControlNet checkpoint and returns the initialized ControlNet model.

    Args:
        control_net_name (str): The filename of the ControlNet checkpoint to load.
            The function searches for this file in the "controlnet" directory.

    Returns:
        IO.CONTROL_NET: The loaded ControlNet model object.

    Raises:
        FileNotFoundError: If the specified ControlNet checkpoint file does not exist.
        RuntimeError: If the loaded checkpoint is invalid or does not contain a valid ControlNet model.

    Example:
        >>> controlnet = load_control_net("control_v11p_sd15_canny.pth")
    """
    controlnet_path = get_full_path_or_raise("controlnet", control_net_name)
    controlnet = load_controlnet(controlnet_path)
    if controlnet is None:
        raise RuntimeError(
            "ERROR: controlnet file is invalid and does not contain a valid controlnet model."
        )
    return controlnet


def load_upscale_model(model_name: str) -> IO.UPSCALE_MODEL:
    """
    Loads an upscale model from the 'upscale_models' directory and returns a single-image model instance.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        IO.UPSCALE_MODEL: Loaded upscale model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If the loaded model is not a single-image model.
    """
    upscale_model_path = get_full_path_or_raise("upscale_models", model_name)
    sd = load_torch_file(upscale_model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(out, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")

    return out


def load_unet(model_name: str, weight_dtype: str = "default") -> IO.MODEL:
    """
    Load a UNet diffusion model with optional FP8/FP16/FP32 dtype configuration.

    Args:
        model_name (str): The name (filename or identifier) of the diffusion model to load.
        weight_dtype (str, optional): Desired weight dtype.
            - "default": Use the model's default precision (usually fp16 or fp32).
            - "fp8_e4m3fn": Use FP8 E4M3FN quantization.
            - "fp8_e4m3fn_fast": Use FP8 E4M3FN with optimizations for speed.
            - "fp8_e5m2": Use FP8 E5M2 quantization.
            (default: "default")

    Returns:
        IO.MODEL: The loaded diffusion UNet model, ready for inference.

    Raises:
        FileNotFoundError: If the specified model file cannot be found.

    Example:
        >>> model = load_unet("sdxl_unet.safetensors", weight_dtype="fp8_e4m3fn")
    """

    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    unet_path = get_full_path_or_raise("diffusion_models", model_name)
    model = load_diffusion_model(unet_path, model_options=model_options)
    return model
