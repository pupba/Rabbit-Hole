import torch
from typing import Optional

import cores.sd_model
from cores.sd_model import load_clip, CLIPType
from cores.types import IO
from cores.path_utils import get_full_path_or_raise, get_folder_paths
import cores.sd_model_class


def dual_clip_loader(
    clip_name1: str,
    clip_name2: str,
    type: str = "sdxl",
    device: Optional[str] = "default",
) -> IO.CLIP:
    clip_type = getattr(
        cores.sd_model_class.CLIPType,
        type.upper(),
        cores.sd_model_class.CLIPType.STABLE_DIFFUSION,
    )

    clip_path1 = get_full_path_or_raise("text_encoders", clip_name1)
    clip_path2 = get_full_path_or_raise("text_encoders", clip_name2)

    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device(
            "cpu"
        )

    clip = cores.sd_model.load_clip(
        ckpt_paths=[clip_path1, clip_path2],
        embedding_directory=get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )

    return clip


def load_text_encoder(
    model_name: str, type: str = "stable_diffusion", device: str = "default"
) -> IO.CLIP:
    """
    Loads a text encoder (CLIP) model.

    Args:
        model_name (str): Name of the text encoder model file to load.
        type (str, optional): Type of text encoder to use.
            Supported: "stable_diffusion", "sd", "open_clip", "deepfloyd", "flux", etc.
            Defaults to "stable_diffusion".
        device (str, optional): Device for model loading. "cpu" or "default" (GPU preferred if available).
            Defaults to "default".

    Returns:
        IO.CLIP: Loaded text encoder model instance.

    Raises:
        FileNotFoundError: If the specified model file cannot be found.
    """
    clip_type = getattr(CLIPType, type.upper(), CLIPType.STABLE_DIFFUSION)
    clip_path = get_full_path_or_raise("text_encoders", model_name)
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device(
            "cpu"
        )
    clip = load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )
    return clip
