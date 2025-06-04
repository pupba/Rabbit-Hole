import logging
from typing import Tuple, Optional, Any
import torch
import math
import os
import random

from PIL import Image

import cores.path_utils

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

# cores
import cores.model_base
import cores.utils
from cores.types import IO
from cores.clip_vision import load as load_clip_vision
from cores.sd_model import load_lora_for_models
from cores.path_utils import get_full_path

# tools
from tools.ipadapter_plus.utils import (
    get_ipadapter_file,
    get_clipvision_file,
    ipadapter_model_loader,
    get_lora_file,
    insightface_loader,
    contrast_adaptive_sharpening,
)
from tools.ipadapter_plus.ipadapter_plus_tools import (
    ipadapter_execute,
    encode_image_masked,
)
from tools.ipadapter_plus.node_helpers import conditioning_set_values

# IO.INSIGHTFACE?


def ipadapter_unified_loader(
    model: IO.MODEL,
    preset: str = "LIGHT",
    lora_strength: float = 0.0,
    provider: str = "CPU",
    ipadapter: Optional[Any] = None,  # "IPADAPTER"
) -> Tuple[IO.MODEL, Any]:  # "IPADAPTER"
    """
    "LIGHT - SD1.5 only (low strength)",
    "STANDARD (medium strength)",
    "VIT-G (medium strength)",
    "PLUS (high strength)",
    "PLUS FACE (portraits)",
    "FULL FACE - SD1.5 only (portraits stronger)",
    """
    init_lora = None
    init_clipvision = {"file": None, "model": None}
    init_ipadapter = {"file": None, "model": None}
    init_insightface = {"provider": None, "model": None}

    pipeline = {
        "clipvision": {"file": None, "model": None},
        "ipadapter": {"file": None, "model": None},
        "insightface": {"provider": None, "model": None},
    }

    if ipadapter is not None:
        pipeline = ipadapter

    if "insightface" not in pipeline:
        pipeline["insightface"] = {"provider": None, "model": None}

    if "ipadapter" not in pipeline:
        pipeline["ipadapter"] = {"file": None, "model": None}

    if "clipvision" not in pipeline:
        pipeline["clipvision"] = {"file": None, "model": None}

    # 1. Load the clipvision model
    clipvision_file = get_clipvision_file(preset)

    if clipvision_file is None:
        raise Exception("ClipVision model not found")

    if clipvision_file != init_clipvision["file"]:
        if clipvision_file != pipeline["clipvision"]["file"]:
            init_clipvision["file"] = clipvision_file
            init_clipvision["model"] = load_clip_vision(clipvision_file)
            logging.info(f"Clip Vision model loaded from {clipvision_file}")
        else:
            init_clipvision = pipeline["clipvision"]

    # 2. Load the ipadapter model
    is_sdxl = isinstance(
        model.model,
        (
            cores.model_base.SDXL,
            cores.model_base.SDXLRefiner,
            cores.model_base.SDXL_instructpix2pix,
        ),
    )
    ipadapter_file, is_insightface, lora_pattern = get_ipadapter_file(
        preset=preset, is_sdxl=is_sdxl
    )
    if ipadapter_file is None:
        raise Exception("IPAdapter model not found.")

    if ipadapter_file != init_ipadapter["file"]:
        if pipeline["ipadapter"]["file"] != ipadapter_file:
            init_ipadapter["file"] = ipadapter_file
            init_ipadapter["model"] = ipadapter_model_loader(ipadapter_file)
            logging.info(f"IPAdapter model loaded from {ipadapter_file}")
        else:
            init_ipadapter = pipeline["ipadapter"]

    # 3. Load the lora model if needed
    if lora_pattern is not None:
        lora_file = get_lora_file(lora_pattern)
        lora_model = None
        if lora_file is None:
            raise Exception("LoRA model not found.")

        if init_lora is not None:
            if lora_file == init_lora["file"]:
                lora_model = init_lora["model"]
            else:
                init_lora = None
                torch.cuda.empty_cache()

        if lora_model is None:
            lora_model = cores.utils.load_torch_file(lora_file, safe_load=True)
            init_lora = {"file": lora_file, "model": lora_model}
            logging.info(f"LoRA model loaded from {lora_file}")

        if lora_strength > 0:
            model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

    # 4. Load the insightface model if needed
    if is_insightface:
        if provider != init_insightface["provider"]:
            if pipeline["insightface"]["provider"] != provider:
                init_insightface["provider"] = provider
                init_insightface["model"] = insightface_loader(provider)
                logging.info(f"InsightFace model loaded with {provider} provider")
            else:
                init_insightface = pipeline["insightface"]

    return model, {
        "clipvision": init_clipvision,
        "ipadapter": init_ipadapter,
        "insightface": init_insightface,
    }


def ipadapter_unified_loader_faceid(
    model: IO.MODEL,
    preset: str = "FACEID",
    lora_strength: float = 0.6,
    provider: str = "CPU",
    ipadapter: Optional[Any] = None,  # "IPADAPTER"
) -> Tuple[IO.MODEL, Any]:  # "IPADAPTER"
    """
    Preset
    "FACEID",
    "FACEID PLUS - SD1.5 only",
    "FACEID PLUS V2",
    "FACEID PORTRAIT (style transfer)",
    "FACEID PORTRAIT UNNORM - SDXL only (strong)",

    providers: ["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"]
    """
    return ipadapter_unified_loader(model, preset, lora_strength, provider, ipadapter)


def ipadapter_unified_loader_community(
    model: IO.MODEL,
    preset: str = "LIGHT",
    lora_strength: float = 0.6,
    provider: str = "Composition",
    ipadapter: Optional[Any] = None,  # "IPADAPTER"
) -> Tuple[IO.MODEL, Any]:  # "IPADAPTER"
    """providers ["Composition", "Kolors"]"""
    return ipadapter_unified_loader(model, preset, lora_strength, provider, ipadapter)


# def ipadapter_model_loader(model_name: str) -> Any:  # "IPADAPTER"
#     ipadapter_file = get_full_path("ipadapter", model_name)
#     return ipadapter_model_loader(ipadapter_file)


def ipadapter_insightface_loader(
    provider: str = "CPU", model_name: str = "buffalo_l"
) -> Any:  # IO.INSIGHTFACE
    """
    provider : 'CPU','CUDA','ROCM'
    model_name : 'buffalo_l','antelopev2'
    """
    return insightface_loader(provider, model_name)


def ipadapter_simple(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    weight_type: str = "standard",
    attn_mask: Optional[IO.MASK] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type :'standard', 'prompt is more important', 'style transfer'
    """

    if weight_type.startswith("style"):
        weight_type = "style transfer"
    elif weight_type == "prompt is more important":
        weight_type = "ease out"
    else:
        weight_type = "linear"

    ipa_args = {
        "image": image,
        "weight": weight,
        "start_at": start_at,
        "end_at": end_at,
        "attn_mask": attn_mask,
        "weight_type": weight_type,
        "insightface": (
            ipadapter["insightface"]["model"] if "insightface" in ipadapter else None
        ),
    }
    if "ipadapter" not in ipadapter:
        raise Exception(
            "IPAdapter model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node."
        )
    if "clipvision" not in ipadapter:
        raise Exception(
            "CLIPVision model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node."
        )

    return ipadapter_execute(
        model.clone(),
        ipadapter["ipadapter"]["model"],
        ipadapter["clipvision"]["model"],
        **ipa_args,
    )


def ipadapter_advanced(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    weight_composition: IO.FLOAT = 1.0,  #
    expand_style: IO.BOOLEAN = False,  #
    weight_style: IO.FLOAT = 1.0,  #
    weight_faceidv2=None,  #
    image_style=None,  #
    image_composition=None,  #
    insightface=None,  #
    layer_weights=None,  #
    ipadapter_params=None,  #
    encode_batch_size: IO.INT = 0,  #
    style_boost=None,  #
    composition_boost=None,  #
    enhance_tiles: IO.INT = 1,  #
    enhance_ratio: IO.FLOAT = 1.0,  #
    weight_kolors: IO.FLOAT = 1.0,  #
    unfold_batch: IO.BOOLEAN = False,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",
    combine_embeds : "concat", "add", "subtract", "average", "norm average"
    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    is_sdxl = isinstance(
        model.model,
        (
            cores.model_base.SDXL,
            cores.model_base.SDXLRefiner,
            cores.model_base.SDXL_instructpix2pix,
        ),
    )

    if "ipadapter" in ipadapter:
        ipadapter_model = ipadapter["ipadapter"]["model"]
        clip_vision = (
            clip_vision if clip_vision is not None else ipadapter["clipvision"]["model"]
        )
    else:
        ipadapter_model = ipadapter

    if clip_vision is None:
        raise Exception("Missing CLIPVision model.")

    if image_style is not None:  # we are doing style + composition transfer
        if not is_sdxl:
            raise Exception(
                "Style + Composition transfer is only available for SDXL models at the moment."
            )  # TODO: check feasibility for SD1.5 models

        image = image_style
        weight = weight_style
        if image_composition is None:
            image_composition = image_style

        weight_type = (
            "strong style and composition" if expand_style else "style and composition"
        )
    if ipadapter_params is not None:  # we are doing batch processing
        image = ipadapter_params["image"]
        attn_mask = ipadapter_params["attn_mask"]
        weight = ipadapter_params["weight"]
        weight_type = ipadapter_params["weight_type"]
        start_at = ipadapter_params["start_at"]
        end_at = ipadapter_params["end_at"]
    else:
        # at this point weight can be a list from the batch-weight or a single float
        weight = [weight]

    image = image if isinstance(image, list) else [image]

    work_model = model.clone()

    for i in range(len(image)):
        if image[i] is None:
            continue

        ipa_args = {
            "image": image[i],
            "image_composition": image_composition,
            "image_negative": image_negative,
            "weight": weight[i],
            "weight_composition": weight_composition,
            "weight_faceidv2": weight_faceidv2,
            "weight_type": (
                weight_type if not isinstance(weight_type, list) else weight_type[i]
            ),
            "combine_embeds": combine_embeds,
            "start_at": start_at if not isinstance(start_at, list) else start_at[i],
            "end_at": end_at if not isinstance(end_at, list) else end_at[i],
            "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
            "unfold_batch": unfold_batch,
            "embeds_scaling": embeds_scaling,
            "insightface": (
                insightface
                if insightface is not None
                else (
                    ipadapter["insightface"]["model"]
                    if "insightface" in ipadapter
                    else None
                )
            ),
            "layer_weights": layer_weights,
            "encode_batch_size": encode_batch_size,
            "style_boost": style_boost,
            "composition_boost": composition_boost,
            "enhance_tiles": enhance_tiles,
            "enhance_ratio": enhance_ratio,
            "weight_kolors": weight_kolors,
        }

        work_model, face_image = ipadapter_execute(
            work_model, ipadapter_model, clip_vision, **ipa_args
        )

    del ipadapter
    return work_model, face_image


def ipadapter_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    encode_batch_size: IO.INT = 0,
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """

    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        encode_batch_size=encode_batch_size,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        unfold_batch=unfold_batch,
    )


def ipadapter_style_composition(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image_style: IO.IMAGE,
    image_composition: IO.IMAGE,
    weight_style: IO.FLOAT = 1.0,
    weight_composition: IO.FLOAT = 1.0,
    expand_style: IO.BOOLEAN = False,
    combine_embeds: str = "average",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds :"concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image_style=image_style,
        image_composition=image_composition,
        weight_style=weight_style,
        weight_composition=weight_composition,
        expand_style=expand_style,
        combine_embeds=combine_embeds,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
    )


def ipadapter_style_composition_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image_style: IO.IMAGE,
    image_composition: IO.IMAGE,
    weight_style: IO.FLOAT = 1.0,
    weight_composition: IO.FLOAT = 1.0,
    expand_style: IO.BOOLEAN = False,
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image_style=image_style,
        image_composition=image_composition,
        weight_style=weight_style,
        weight_composition=weight_composition,
        expand_style=expand_style,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        unfold_batch=unfold_batch,
    )


def ipadapter_faceid(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_faceidv2: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    insightface: Optional[Any] = None,  # IO.INSIGHTFACE
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds :"concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        weight_faceidv2=weight_faceidv2,
        weight_type=weight_type,
        combine_embeds=combine_embeds,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        insightface=insightface,
    )


def ipadapter_faceid_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_faceidv2: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    insightface: Optional[Any] = None,  # IO.INSIGHTFACE
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds :"concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_faceid(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        weight_faceidv2=weight_faceidv2,
        weight_type=weight_type,
        combine_embeds=combine_embeds,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        insightface=insightface,
        unfold_batch=unfold_batch,
    )


def ipadapter_faceid_kolors(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_faceidv2: IO.FLOAT = 1.0,
    weight_kolors: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    insightface: Optional[Any] = None,  # IO.INSIGHTFACE
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds :"concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        weight_faceidv2=weight_faceidv2,
        weight_kolors=weight_kolors,
        weight_type=weight_type,
        combine_embeds=combine_embeds,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        insightface=insightface,
    )


def ipadapter_tiled(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    sharpening: IO.FLOAT = 0.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    encode_batch_size: IO.INT = 0,
    unfold_batch: IO.BOOLEAN = False,
) -> Tuple[IO.MODEL, IO.IMAGE, IO.MASK]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds :"concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    # 1. Select the model
    if "ipadapter" in ipadapter:
        ipadapter_model = ipadapter["ipadapter"]["model"]
        clip_vision = (
            clip_vision if clip_vision is not None else ipadapter["clipvision"]["model"]
        )
    else:
        ipadapter_model = ipadapter
        clip_vision = clip_vision

    if clip_vision is None:
        raise Exception("Missing CLIPVision model.")

    del ipadapter

    # 2. Extract the tiles
    tile_size = 256  # I'm using 256 instead of 224 as it is more likely divisible by the latent size, it will be downscaled to 224 by the clip vision encoder
    _, oh, ow, _ = image.shape
    if attn_mask is None:
        attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)

    image = image.permute([0, 3, 1, 2])
    attn_mask = attn_mask.unsqueeze(1)
    # the mask should have the same proportions as the reference image and the latent
    attn_mask = T.Resize(
        (oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )(attn_mask)

    # if the image is almost a square, we crop it to a square
    if oh / ow > 0.75 and oh / ow < 1.33:
        # crop the image to a square
        image = T.CenterCrop(min(oh, ow))(image)
        resize = (tile_size * 2, tile_size * 2)

        attn_mask = T.CenterCrop(min(oh, ow))(attn_mask)
    # otherwise resize the smallest side and the other proportionally
    else:
        resize = (
            (int(tile_size * ow / oh), tile_size)
            if oh < ow
            else (tile_size, int(tile_size * oh / ow))
        )

        # using PIL for better results
    imgs = []
    for img in image:
        img = T.ToPILImage()(img)
        img = img.resize(resize, resample=Image.Resampling["LANCZOS"])
        imgs.append(T.ToTensor()(img))
    image = torch.stack(imgs)
    del imgs, img

    # we don't need a high quality resize for the mask
    attn_mask = T.Resize(
        resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )(attn_mask)

    # we allow a maximum of 4 tiles
    if oh / ow > 4 or oh / ow < 0.25:
        crop = (tile_size, tile_size * 4) if oh < ow else (tile_size * 4, tile_size)
        image = T.CenterCrop(crop)(image)
        attn_mask = T.CenterCrop(crop)(attn_mask)

    attn_mask = attn_mask.squeeze(1)

    if sharpening > 0:
        image = contrast_adaptive_sharpening(image, sharpening)

    image = image.permute([0, 2, 3, 1])

    _, oh, ow, _ = image.shape

    # find the number of tiles for each side
    tiles_x = math.ceil(ow / tile_size)
    tiles_y = math.ceil(oh / tile_size)
    overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
    overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))

    base_mask = torch.zeros(
        [attn_mask.shape[0], oh, ow], dtype=image.dtype, device=image.device
    )

    # extract all the tiles from the image and create the masks
    tiles = []
    masks = []
    for y in range(tiles_y):
        for x in range(tiles_x):
            start_x = int(x * (tile_size - overlap_x))
            start_y = int(y * (tile_size - overlap_y))
            tiles.append(
                image[
                    :, start_y : start_y + tile_size, start_x : start_x + tile_size, :
                ]
            )
            mask = base_mask.clone()
            mask[:, start_y : start_y + tile_size, start_x : start_x + tile_size] = (
                attn_mask[
                    :, start_y : start_y + tile_size, start_x : start_x + tile_size
                ]
            )
            masks.append(mask)
    del mask

    # 3. Apply the ipadapter to each group of tiles
    model = model.clone()
    for i in range(len(tiles)):
        ipa_args = {
            "image": tiles[i],
            "image_negative": image_negative,
            "weight": weight,
            "weight_type": weight_type,
            "combine_embeds": combine_embeds,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": masks[i],
            "unfold_batch": unfold_batch,
            "embeds_scaling": embeds_scaling,
            "encode_batch_size": encode_batch_size,
        }
        # apply the ipadapter to the model without cloning it
        model, _ = ipadapter_execute(model, ipadapter_model, clip_vision, **ipa_args)

    return model, torch.cat(tiles), torch.cat(masks)


def ipadapter_tiled_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    sharpening: IO.FLOAT = 0.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    encode_batch_size: IO.INT = 0,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE, IO.MASK]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_tiled(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
        sharpening=sharpening,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        encode_batch_size=encode_batch_size,
        unfold_batch=unfold_batch,
    )


# IO.EMBEDS?
def ipadapter_embeds(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    pos_embed: Any,  # IO.EMBEDS
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    neg_embed: Optional[Any] = None,  # IO.EMBEDS
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = False,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    ipa_args = {
        "pos_embed": pos_embed,
        "neg_embed": neg_embed,
        "weight": weight,
        "weight_type": weight_type,
        "start_at": start_at,
        "end_at": end_at,
        "attn_mask": attn_mask,
        "embeds_scaling": embeds_scaling,
        "unfold_batch": unfold_batch,
    }

    if "ipadapter" in ipadapter:
        ipadapter_model = ipadapter["ipadapter"]["model"]
        clip_vision = (
            clip_vision if clip_vision is not None else ipadapter["clipvision"]["model"]
        )
    else:
        ipadapter_model = ipadapter
        clip_vision = clip_vision

    if clip_vision is None and neg_embed is None:
        raise Exception("Missing CLIPVision model.")

    del ipadapter

    return ipadapter_execute(model.clone(), ipadapter_model, clip_vision, **ipa_args)


def ipadapter_embeds_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    pos_embed: Any,  # IO.EMBEDS
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    neg_embed: Optional[Any] = None,  # IO.EMBEDS
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_embeds(
        mode=model,
        ipadapter=ipadapter,
        pos_embed=pos_embed,
        weight=weight,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        neg_embed=neg_embed,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        unfold_batch=unfold_batch,
    )


def ipadapter_ms(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    pos_embed: Any,  # IO.EMBEDS
    weight: IO.FLOAT = 1.0,
    weight_faceidv2: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    layer_weights: str = "",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    insightface: Optional[Any] = None,  # IO.INSIGHTFACE
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds : "concat", "add", "subtract", "average", "norm average"
    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        pos_embed=pos_embed,
        weight=weight,
        weight_faceidv2=weight_faceidv2,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        layer_weights=layer_weights,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        insightface=insightface,
    )


def ipadapter_clip_vision_enhancer(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    enhance_tiles: IO.INT = 2,
    enhance_ratio: IO.FLOAT = 1.0,
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",

    combine_embeds : "concat", "add", "subtract", "average", "norm average"
    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        combine_embeds=combine_embeds,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
        embeds_scaling=embeds_scaling,
        enhance_tiles=enhance_tiles,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        enhance_ratio=enhance_ratio,
    )


# Any_PARAMS
def ipadapter_from_params(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    ipadapter_params: Any,  # Any_PARAMS
    combine_embeds: str = "concat",
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds : "concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        ipadapter_params=ipadapter_params,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        clip_vision=clip_vision,
    )


def ipadapter_precise_style_transfer(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    style_boost: IO.FLOAT = 1.0,
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds : "concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        style_boost=style_boost,
        start_at=start_at,
        end_at=end_at,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
    )


def ipadapter_precise_style_transfer_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    style_boost: IO.FLOAT = 1.0,
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds : "concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_precise_style_transfer(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        style_boost=style_boost,
        start_at=start_at,
        end_at=end_at,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        unfold_batch=unfold_batch,
    )


def ipadapter_precise_composition(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    composition_boost: IO.FLOAT = 0.0,
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds : "concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_advanced(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        composition_boost=composition_boost,
        start_at=start_at,
        end_at=end_at,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
    )


def ipadapter_precise_composition_batch(
    model: IO.MODEL,
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    composition_boost: IO.FLOAT = 0.0,
    combine_embeds: str = "concat",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    embeds_scaling: str = "V only",
    image_negative: Optional[IO.IMAGE] = None,
    attn_mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
    unfold_batch: IO.BOOLEAN = True,
) -> Tuple[IO.MODEL, IO.IMAGE]:
    """
    combine_embeds : "concat", "add", "subtract", "average", "norm average"

    embeds_scaling : 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
    """
    return ipadapter_precise_composition(
        model=model,
        ipadapter=ipadapter,
        image=image,
        weight=weight,
        composition_boost=composition_boost,
        start_at=start_at,
        end_at=end_at,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling,
        image_negative=image_negative,
        attn_mask=attn_mask,
        clip_vision=clip_vision,
        unfold_batch=unfold_batch,
    )


def ipadapter_encoder(
    ipadapter: Any,  # "IPADAPTER"
    image: IO.IMAGE,
    weight: IO.FLOAT = 1.0,
    mask: Optional[IO.MASK] = None,
    clip_vision: Optional[IO.CLIP_VISION] = None,
) -> Tuple[Any, Any]:  # IO.ENBEDS
    if "ipadapter" in ipadapter:
        ipadapter_model = ipadapter["ipadapter"]["model"]
        clip_vision = (
            clip_vision if clip_vision is not None else ipadapter["clipvision"]["model"]
        )
    else:
        ipadapter_model = ipadapter
        clip_vision = clip_vision

    if clip_vision is None:
        raise Exception("Missing CLIPVision model.")

    is_plus = (
        "proj.3.weight" in ipadapter_model["image_proj"]
        or "latents" in ipadapter_model["image_proj"]
        or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]
    )
    is_kwai_kolors = (
        is_plus
        and "layers.0.0.to_out.weight" in ipadapter_model["image_proj"]
        and ipadapter_model["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048
    )

    clipvision_size = 224 if not is_kwai_kolors else 336

    # resize and crop the mask to 224x224
    if mask is not None and mask.shape[1:3] != torch.Size(
        [clipvision_size, clipvision_size]
    ):
        mask = mask.unsqueeze(1)
        transforms = T.Compose(
            [
                T.CenterCrop(min(mask.shape[2], mask.shape[3])),
                T.Resize(
                    (clipvision_size, clipvision_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
        mask = transforms(mask).squeeze(1)
        # mask = T.Resize((image.shape[1], image.shape[2]), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(mask.unsqueeze(1)).squeeze(1)

    img_cond_embeds = encode_image_masked(
        clip_vision, image, mask, clipvision_size=clipvision_size
    )

    if is_plus:
        img_cond_embeds = img_cond_embeds.penultimate_hidden_states
        img_uncond_embeds = encode_image_masked(
            clip_vision,
            torch.zeros([1, clipvision_size, clipvision_size, 3]),
            clipvision_size=clipvision_size,
        ).penultimate_hidden_states
    else:
        img_cond_embeds = img_cond_embeds.image_embeds
        img_uncond_embeds = torch.zeros_like(img_cond_embeds)

    if weight != 1:
        img_cond_embeds = img_cond_embeds * weight

    return img_cond_embeds, img_uncond_embeds


# IO.EMBEDS
def ipadapter_combine_embeds(
    embed1: Any,
    method: str = "concat",
    embed2: Optional[Any] = None,
    embed3: Optional[Any] = None,
    embed4: Optional[Any] = None,
    embed5: Optional[Any] = None,
) -> Any:
    """
    method: "concat", "add", "subtract", "average", "norm average", "max", "min"
    """
    if (
        method == "concat"
        and embed2 is None
        and embed3 is None
        and embed4 is None
        and embed5 is None
    ):
        return (embed1,)

    embeds = [embed1, embed2, embed3, embed4, embed5]
    embeds = [embed for embed in embeds if embed is not None]
    embeds = torch.cat(embeds, dim=0)

    if method == "add":
        embeds = torch.sum(embeds, dim=0).unsqueeze(0)
    elif method == "subtract":
        embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
        embeds = embeds.unsqueeze(0)
    elif method == "average":
        embeds = torch.mean(embeds, dim=0).unsqueeze(0)
    elif method == "norm average":
        embeds = torch.mean(
            embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0
        ).unsqueeze(0)
    elif method == "max":
        embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
    elif method == "min":
        embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

    return embeds


def ipadapter_noise(
    type: str = "fade",
    strength: IO.FLOAT = 1.0,
    blur: IO.INT = 0,
    image_optional: Optional[IO.IMAGE] = None,
) -> IO.IMAGE:
    """
    type: "fade", "dissolve", "gaussian", "shuffle"
    """
    if image_optional is None:
        image = torch.zeros([1, 224, 224, 3])
    else:
        transforms = T.Compose(
            [
                T.CenterCrop(min(image_optional.shape[1], image_optional.shape[2])),
                T.Resize(
                    (224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
        image = transforms(image_optional.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])

    seed = (
        int(torch.sum(image).item()) % 1000000007
    )  # hash the image to get a seed, grants predictability
    torch.manual_seed(seed)

    if type == "fade":
        noise = torch.rand_like(image)
        noise = image * (1 - strength) + noise * strength
    elif type == "dissolve":
        mask = (torch.rand_like(image) < strength).float()
        noise = torch.rand_like(image)
        noise = image * (1 - mask) + noise * mask
    elif type == "gaussian":
        noise = torch.randn_like(image) * strength
        noise = image + noise
    elif type == "shuffle":
        transforms = T.Compose(
            [
                T.ElasticTransform(alpha=75.0, sigma=(1 - strength) * 3.5),
                T.RandomVerticalFlip(p=1.0),
                T.RandomHorizontalFlip(p=1.0),
            ]
        )
        image = transforms(image.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        noise = torch.randn_like(image) * (strength * 0.75)
        noise = image * (1 - noise) + noise

    del image
    noise = torch.clamp(noise, 0, 1)

    if blur > 0:
        if blur % 2 == 0:
            blur += 1
        noise = T.functional.gaussian_blur(noise.permute([0, 3, 1, 2]), blur).permute(
            [0, 2, 3, 1]
        )
    return noise


def prep_image_for_clip_vision(
    image: IO.IMAGE,
    interpolation: str = "LANCZOS",
    crop_position: str = "center",
    sharpening: IO.FLOAT = 0.0,
) -> IO.IMAGE:
    """
    interpolation: 'LANCZOS','BICUBIC','HAMMING','BILINEAR','BOX','NEAREST'
    crop_position: 'top','bottom','left','right','center','pad'
    """
    size = (224, 224)
    _, oh, ow, _ = image.shape
    output = image.permute([0, 3, 1, 2])

    if crop_position == "pad":
        if oh != ow:
            if oh > ow:
                pad = (oh - ow) // 2
                pad = (pad, 0, pad, 0)
            elif ow > oh:
                pad = (ow - oh) // 2
                pad = (0, pad, 0, pad)
            output = T.functional.pad(output, pad, fill=0)
    else:
        crop_size = min(oh, ow)
        x = (ow - crop_size) // 2
        y = (oh - crop_size) // 2
        if "top" in crop_position:
            y = 0
        elif "bottom" in crop_position:
            y = oh - crop_size
        elif "left" in crop_position:
            x = 0
        elif "right" in crop_position:
            x = ow - crop_size

        x2 = x + crop_size
        y2 = y + crop_size

        output = output[:, :, y:y2, x:x2]

    imgs = []
    for img in output:
        img = T.ToPILImage()(img)  # using PIL for better results
        img = img.resize(size, resample=Image.Resampling[interpolation])
        imgs.append(T.ToTensor()(img))
    output = torch.stack(imgs, dim=0)
    del imgs, img

    if sharpening > 0:
        output = contrast_adaptive_sharpening(output, sharpening)

    output = output.permute([0, 2, 3, 1])
    return output


# IO.EMBEDS
def ipadapter_save_embeds(
    embeds: Any, filename_prefix: str = "IP_embeds", output_dir: str = os.getcwd()
) -> None:
    full_output_folder, filename, counter, subfolder, filename_prefix = (
        cores.path_utils.get_save_image_path(
            filename_prefix=filename_prefix, output_dir=output_dir
        )
    )
    file = f"{filename}_{counter:05}.ipadpt"
    file = os.path.join(full_output_folder, file)
    torch.save(embeds, file)


# IO.EMBEDS
def ipadapter_load_embeds(input_dir: str = "./") -> Any:
    files = [
        os.path.relpath(os.path.join(root, file), input_dir)
        for root, dirs, files in os.walk(input_dir)
        for file in files
        if file.endswith(".ipadpt")
    ]
    embeds = sorted(files)

    path = cores.path_utils.get_annotated_filepath(embeds)
    return torch.load(path).cpu()


# IO.WEIGHT_STRATEGY ?
def ipadapter_weights(
    weights: str = "1.0,0.0",
    timing: str = "custom",
    frames: IO.INT = 0,
    start_frame: IO.INT = 0,
    end_frame: IO.INT = 9999,
    add_starting_frames: IO.INT = 0,
    add_ending_frames: IO.INT = 0,
    method: str = "full batch",
    weights_strategy: Optional[Any] = None,  # IO.WEIGHT_STRATEGY
    image: Optional[IO.IMAGE] = None,
) -> Tuple[IO.FLOAT, IO.FLOAT, IO.INT, IO.IMAGE, IO.IMAGE, Any]:  # IO.WEIGHT_STRATEGY
    """
    timing : "custom", "linear", "ease_in_out", "ease_in", "ease_out", "random"
    method : "full batch", "shift batches", "alternate batches"
    """
    frame_count = image.shape[0] if image is not None else 0
    if weights_strategy is not None:
        weights = weights_strategy["weights"]
        timing = weights_strategy["timing"]
        frames = weights_strategy["frames"]
        start_frame = weights_strategy["start_frame"]
        end_frame = weights_strategy["end_frame"]
        add_starting_frames = weights_strategy["add_starting_frames"]
        add_ending_frames = weights_strategy["add_ending_frames"]
        method = weights_strategy["method"]
        frame_count = weights_strategy["frame_count"]
    else:
        weights_strategy = {
            "weights": weights,
            "timing": timing,
            "frames": frames,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "add_starting_frames": add_starting_frames,
            "add_ending_frames": add_ending_frames,
            "method": method,
            "frame_count": frame_count,
        }

    # convert the string to a list of floats separated by commas or newlines
    weights = weights.replace("\n", ",")
    weights = [float(weight) for weight in weights.split(",") if weight.strip() != ""]

    if timing != "custom":
        frames = max(frames, 2)
        start = 0.0
        end = 1.0

        if len(weights) > 0:
            start = weights[0]
            end = weights[-1]

        weights = []

        end_frame = min(end_frame, frames)
        duration = end_frame - start_frame
        if start_frame > 0:
            weights.extend([start] * start_frame)

        for i in range(duration):
            n = duration - 1
            if timing == "linear":
                weights.append(start + (end - start) * i / n)
            elif timing == "ease_in_out":
                weights.append(
                    start + (end - start) * (1 - math.cos(i / n * math.pi)) / 2
                )
            elif timing == "ease_in":
                weights.append(start + (end - start) * math.sin(i / n * math.pi / 2))
            elif timing == "ease_out":
                weights.append(
                    start + (end - start) * (1 - math.cos(i / n * math.pi / 2))
                )
            elif timing == "random":
                weights.append(random.uniform(start, end))

        weights[-1] = end if timing != "random" else weights[-1]
        if end_frame < frames:
            weights.extend([end] * (frames - end_frame))

    if len(weights) == 0:
        weights = [0.0]

    frames = len(weights)

    # repeat the images for cross fade
    image_1 = None
    image_2 = None

    # Calculate the min and max of the weights
    min_weight = min(weights)
    max_weight = max(weights)

    if image is not None:

        if "shift" in method:
            image_1 = image[:-1]
            image_2 = image[1:]

            weights = weights * image_1.shape[0]
            image_1 = image_1.repeat_interleave(frames, 0)
            image_2 = image_2.repeat_interleave(frames, 0)
        elif "alternate" in method:
            image_1 = image[::2].repeat_interleave(2, 0)
            image_1 = image_1[1:]
            image_2 = image[1::2].repeat_interleave(2, 0)

            # Invert the weights relative to their own range
            mew_weights = weights + [max_weight - (w - min_weight) for w in weights]

            mew_weights = mew_weights * (image_1.shape[0] // 2)
            if image.shape[0] % 2:
                image_1 = image_1[:-1]
            else:
                image_2 = image_2[:-1]
                mew_weights = mew_weights + weights

            weights = mew_weights
            image_1 = image_1.repeat_interleave(frames, 0)
            image_2 = image_2.repeat_interleave(frames, 0)
        else:
            weights = weights * image.shape[0]
            image_1 = image.repeat_interleave(frames, 0)

        # add starting and ending frames
        if add_starting_frames > 0:
            weights = [weights[0]] * add_starting_frames + weights
            image_1 = torch.cat(
                [image[:1].repeat(add_starting_frames, 1, 1, 1), image_1], dim=0
            )
            if image_2 is not None:
                image_2 = torch.cat(
                    [image[:1].repeat(add_starting_frames, 1, 1, 1), image_2], dim=0
                )
        if add_ending_frames > 0:
            weights = weights + [weights[-1]] * add_ending_frames
            image_1 = torch.cat(
                [image_1, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0
            )
            if image_2 is not None:
                image_2 = torch.cat(
                    [image_2, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0
                )

    # reverse the weights array
    weights_invert = weights[::-1]

    frame_count = len(weights)

    return weights, weights_invert, frame_count, image_1, image_2, weights_strategy


def ipadapter_weights_from_strategy(
    weights_strategy: Any,  # IO.WEIGHT_STRATEGY
    weights: str = "1.0,0.0",
    timing: str = "custom",
    frames: IO.INT = 0,
    start_frame: IO.INT = 0,
    end_frame: IO.INT = 9999,
    add_starting_frames: IO.INT = 0,
    add_ending_frames: IO.INT = 0,
    method: str = "full batch",
    image: Optional[IO.IMAGE] = None,
) -> Tuple[IO.FLOAT, IO.FLOAT, IO.INT, IO.IMAGE, IO.IMAGE, Any]:  # IO.WEIGHT_STRATEGY
    """
    timing : "custom", "linear", "ease_in_out", "ease_in", "ease_out", "random"
    method : "full batch", "shift batches", "alternate batches"
    """
    return ipadapter_weights(
        weights=weights,
        timing=timing,
        frames=frames,
        start_frame=start_frame,
        end_frame=end_frame,
        add_starting_frames=add_starting_frames,
        add_ending_frames=add_ending_frames,
        method=method,
        image=image,
        weights_strategy=weights_strategy,
    )


def ipadapter_prompt_schedule_from_weights_strategy(
    weights_strategy: Any, prompt: str = ""  # IO.WEIGHT_STRATEGY
) -> str:
    frames = weights_strategy["frames"]
    add_starting_frames = weights_strategy["add_starting_frames"]
    add_ending_frames = weights_strategy["add_ending_frames"]
    frame_count = weights_strategy["frame_count"]

    out = ""

    prompt = [p for p in prompt.split("\n") if p.strip() != ""]

    if len(prompt) > 0 and frame_count > 0:
        # prompt_pos must be the same size as the image batch
        if len(prompt) > frame_count:
            prompt = prompt[:frame_count]
        elif len(prompt) < frame_count:
            prompt += [prompt[-1]] * (frame_count - len(prompt))

        if add_starting_frames > 0:
            out += f'"0": "{prompt[0]}",\n'
        for i in range(frame_count):
            out += f'"{i * frames + add_starting_frames}": "{prompt[i]}",\n'
        if add_ending_frames > 0:
            out += f'"{frame_count * frames + add_starting_frames}": "{prompt[-1]}",\n'

    return out


def ipadapter_combine_weights(
    weights_1: IO.FLOAT = 0.0,
    weights_2: IO.FLOAT = 0.0,
) -> Tuple[IO.FLOAT, IO.INT]:
    if not isinstance(weights_1, list):
        weights_1 = [weights_1]
    if not isinstance(weights_2, list):
        weights_2 = [weights_2]
    weights = weights_1 + weights_2

    return weights, len(weights)


# Any_PARAMS?
def ipadapter_regional_conditioning(
    image: IO.IMAGE,
    image_weight: IO.FLOAT = 1.0,
    prompt_weight: IO.FLOAT = 1.0,
    weight_type: str = "linear",
    start_at: IO.FLOAT = 0.0,
    end_at: IO.FLOAT = 1.0,
    mask: Optional[IO.MASK] = None,
    positive: Optional[IO.CONDITIONING] = None,
    negative: Optional[IO.CONDITIONING] = None,
) -> Tuple[Any, IO.CONDITIONING, IO.CONDITIONING]:  # Any_PARAMS
    """
    weight_type:
    "linear",
    "ease in",
    "ease out",
    "ease in-out",
    "reverse in-out",
    "weak input",
    "weak output",
    "weak middle",
    "strong middle",
    "style transfer",
    "composition",
    "strong style transfer",
    "style and composition",
    "style transfer precise",
    "composition precise",
    """
    set_area_to_bounds = False  # if set_cond_area == "default" else True

    if mask is not None:
        if positive is not None:
            positive = conditioning_set_values(
                positive,
                {
                    "mask": mask,
                    "set_area_to_bounds": set_area_to_bounds,
                    "mask_strength": prompt_weight,
                },
            )
        if negative is not None:
            negative = conditioning_set_values(
                negative,
                {
                    "mask": mask,
                    "set_area_to_bounds": set_area_to_bounds,
                    "mask_strength": prompt_weight,
                },
            )

    ipadapter_params = {
        "image": [image],
        "attn_mask": [mask],
        "weight": [image_weight],
        "weight_type": [weight_type],
        "start_at": [start_at],
        "end_at": [end_at],
    }

    return ipadapter_params, positive, negative


# Any_PARAMS
def ipadapter_combine_params(
    params_1: Any,
    params_2: Any,
    params_3: Optional[Any] = None,
    params_4: Optional[Any] = None,
    params_5: Optional[Any] = None,
) -> Any:
    ipadapter_params = {
        "image": params_1["image"] + params_2["image"],
        "attn_mask": params_1["attn_mask"] + params_2["attn_mask"],
        "weight": params_1["weight"] + params_2["weight"],
        "weight_type": params_1["weight_type"] + params_2["weight_type"],
        "start_at": params_1["start_at"] + params_2["start_at"],
        "end_at": params_1["end_at"] + params_2["end_at"],
    }

    if params_3 is not None:
        ipadapter_params["image"] += params_3["image"]
        ipadapter_params["attn_mask"] += params_3["attn_mask"]
        ipadapter_params["weight"] += params_3["weight"]
        ipadapter_params["weight_type"] += params_3["weight_type"]
        ipadapter_params["start_at"] += params_3["start_at"]
        ipadapter_params["end_at"] += params_3["end_at"]
    if params_4 is not None:
        ipadapter_params["image"] += params_4["image"]
        ipadapter_params["attn_mask"] += params_4["attn_mask"]
        ipadapter_params["weight"] += params_4["weight"]
        ipadapter_params["weight_type"] += params_4["weight_type"]
        ipadapter_params["start_at"] += params_4["start_at"]
        ipadapter_params["end_at"] += params_4["end_at"]
    if params_5 is not None:
        ipadapter_params["image"] += params_5["image"]
        ipadapter_params["attn_mask"] += params_5["attn_mask"]
        ipadapter_params["weight"] += params_5["weight"]
        ipadapter_params["weight_type"] += params_5["weight_type"]
        ipadapter_params["start_at"] += params_5["start_at"]
        ipadapter_params["end_at"] += params_5["end_at"]

    return ipadapter_params
