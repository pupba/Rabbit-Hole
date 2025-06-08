from __future__ import annotations
import logging
import torch
from typing import Any, Optional, Tuple

# cores
import cores.utils
import cores.lora_convert
from cores import model_detection
from cores import model_management_utils
from cores.sd_model_class import VAE, CLIP, CLIPType, TEModel
from cores import clip_vision
from cores.utils import load_torch_file

# text encder
from cores.text_encoder_utils import sd1_clip
from cores.text_encoder_utils import sdxl_clip
import cores.text_encoder_utils.sd2_clip
import cores.text_encoder_utils.sd3_clip
import cores.text_encoder_utils.sa_t5
import cores.text_encoder_utils.aura_t5
import cores.text_encoder_utils.pixart_t5
import cores.text_encoder_utils.hydit
import cores.text_encoder_utils.flux
import cores.text_encoder_utils.long_clipl
import cores.text_encoder_utils.genmo
import cores.text_encoder_utils.lt
import cores.text_encoder_utils.hunyuan_video
import cores.text_encoder_utils.cosmos
import cores.text_encoder_utils.lumina2
import cores.text_encoder_utils.wan
import cores.text_encoder_utils.hidream
import cores.text_encoder_utils.ace


def load_checkpoint_guess_config(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    output_clipvision: bool = False,
    embedding_directory: Optional[str] = None,
    output_model: bool = True,
    device: Optional[torch.device] = None,
    model_options: Optional[dict] = None,
    te_model_options: Optional[dict] = None,
) -> Any:
    sd, metadata = load_torch_file(ckpt_path, return_metadata=True)

    out = load_state_dict_guess_config(
        sd,
        output_vae,
        output_clip,
        output_clipvision,
        embedding_directory,
        output_model,
        model_options,
        te_model_options=te_model_options,
        metadata=metadata,
    )
    if out is None:
        raise RuntimeError(f"ERROR: Could not detect model type of: {ckpt_path}")
    return out


def load_diffusion_model_state_dict(
    sd: dict, model_options: dict = {}
):  # load unet in diffusers or regular format
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = cores.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = cores.utils.calculate_parameters(sd)
    weight_dtype = cores.utils.weight_dtype(sd)

    load_device = model_management_utils.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = cores.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management_utils.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management_utils.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management_utils.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get(
        "custom_operations", model_config.custom_operations
    )
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return cores.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )


def load_diffusion_model(unet_path: str, model_options={}):
    sd = load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(unet_path)
        )
    return model


def load_state_dict_guess_config(
    sd: dict,
    output_vae: bool = True,
    output_clip: bool = True,
    output_clipvision: bool = False,
    embedding_directory: Optional[str] = None,
    output_model: bool = True,
    model_options: Optional[dict] = None,
    te_model_options: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Optional[Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]]:
    """
    sd : state_dict loaded from checkpoint
    Returns: (model_patcher, clip, vae, clipvision) or None if load fails
    """
    clip: Optional[Any] = None
    clipvision: Optional[Any] = None
    vae: Optional[Any] = None
    model: Optional[Any] = None
    model_patcher: Optional[Any] = None

    if model_options is None:
        model_options = {}
    if te_model_options is None:
        te_model_options = {}

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = cores.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = cores.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management_utils.get_torch_device()

    model_config = model_detection.model_config_from_unet(
        sd, diffusion_model_prefix, metadata=metadata
    )
    if model_config is None:
        logging.warning(
            "Warning, This is not a checkpoint file, trying to load it as a diffusion model only."
        )
        diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
        if diffusion_model is None:
            return None
        return (
            diffusion_model,
            None,
            VAE(sd={}),
            None,
        )  # Placeholder VAE for exception

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))
    if unet_dtype is None:
        unet_dtype = model_management_utils.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )

    manual_cast_dtype = model_management_utils.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(
                sd, model_config.clip_vision_prefix, True
            )

    if output_model:
        inital_load_device = model_management_utils.unet_inital_load_device(
            parameters, unet_dtype
        )
        model = model_config.get_model(
            sd, diffusion_model_prefix, device=inital_load_device
        )
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = cores.utils.state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
        )
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd, metadata=metadata)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            # None
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = cores.utils.calculate_parameters(clip_sd)
                # clip,tokenizer, params
                clip = CLIP(
                    clip_target,
                    embedding_directory=embedding_directory,
                    tokenizer_data=clip_sd,
                    parameters=parameters,
                    model_options=te_model_options,
                )
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(
                        filter(
                            lambda a: ".logit_scale" not in a
                            and ".transformer.text_projection.weight" not in a,
                            m,
                        )
                    )
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))
                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = cores.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=model_management_utils.unet_offload_device(),
        )
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management_utils.load_models_gpu(
                [model_patcher], force_full_load=True
            )

    return (model_patcher, clip, vae, clipvision)


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    key_map = {}
    if model is not None:
        key_map = cores.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = cores.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    lora = cores.lora_convert.convert_lora(lora)
    loaded = cores.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher, new_clip)


def t5xxl_detect(clip_data):
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"

    for sd in clip_data:
        if weight_name in sd or weight_name_old in sd:
            return cores.text_encoder_utils.sd3_clip.t5_xxl_detect(sd)

    return {}


def llama_detect(clip_data):
    weight_name = "model.layers.0.self_attn.k_proj.weight"

    for sd in clip_data:
        if weight_name in sd:
            return cores.text_encoder_utils.hunyuan_video.llama_detect(sd)

    return {}


def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
        elif weight.shape[-1] == 2048:
            return TEModel.T5_XL
    if "encoder.block.23.layer.1.DenseReluDense.wi.weight" in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        return TEModel.T5_BASE
    if "model.layers.0.post_feedforward_layernorm.weight" in sd:
        return TEModel.GEMMA_2_2B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TEModel.LLAMA3_8
    return None


def load_clip(
    ckpt_paths,
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(
        clip_data,
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        model_options=model_options,
    )


def load_text_encoder_state_dicts(
    state_dicts=[],
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = cores.utils.clip_text_transformers_convert(
                clip_data[i], "", ""
            )
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i][
                    "text_projection"
                ].transpose(
                    0, 1
                )  # old models saved with the CLIPSave node

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = cores.text_encoder_utils.sd3_clip.sd3_clip(
                    clip_l=False, clip_g=True, t5=False
                )
                clip_target.tokenizer = cores.text_encoder_utils.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
                    clip_l=False,
                    clip_g=True,
                    t5=False,
                    llama=False,
                    dtype_t5=None,
                    dtype_llama=None,
                    t5xxl_scaled_fp8=None,
                    llama_scaled_fp8=None,
                )
                clip_target.tokenizer = (
                    cores.text_encoder_utils.hidream.HiDreamTokenizer
                )
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = cores.text_encoder_utils.sd2_clip.SD2ClipModel
            clip_target.tokenizer = cores.text_encoder_utils.sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = cores.text_encoder_utils.sd3_clip.sd3_clip(
                    clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data)
                )
                clip_target.tokenizer = cores.text_encoder_utils.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = cores.text_encoder_utils.lt.ltxv_te(
                    **t5xxl_detect(clip_data)
                )
                clip_target.tokenizer = cores.text_encoder_utils.lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART or clip_type == CLIPType.CHROMA:
                clip_target.clip = cores.text_encoder_utils.pixart_t5.pixart_te(
                    **t5xxl_detect(clip_data)
                )
                clip_target.tokenizer = (
                    cores.text_encoder_utils.pixart_t5.PixArtTokenizer
                )
            elif clip_type == CLIPType.WAN:
                clip_target.clip = cores.text_encoder_utils.wan.te(
                    **t5xxl_detect(clip_data)
                )
                clip_target.tokenizer = cores.text_encoder_utils.wan.WanT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
                    **t5xxl_detect(clip_data),
                    clip_l=False,
                    clip_g=False,
                    t5=True,
                    llama=False,
                    dtype_llama=None,
                    llama_scaled_fp8=None,
                )
                clip_target.tokenizer = (
                    cores.text_encoder_utils.hidream.HiDreamTokenizer
                )
            else:  # CLIPType.MOCHI
                clip_target.clip = cores.text_encoder_utils.genmo.mochi_te(
                    **t5xxl_detect(clip_data)
                )
                clip_target.tokenizer = cores.text_encoder_utils.genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = cores.text_encoder_utils.cosmos.te(
                **t5xxl_detect(clip_data)
            )
            clip_target.tokenizer = cores.text_encoder_utils.cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = cores.text_encoder_utils.aura_t5.AuraT5Model
            clip_target.tokenizer = cores.text_encoder_utils.aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            if clip_type == CLIPType.ACE or "spiece_model" in clip_data[0]:
                clip_target.clip = cores.text_encoder_utils.ace.AceT5Model
                clip_target.tokenizer = cores.text_encoder_utils.ace.AceT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            else:
                clip_target.clip = cores.text_encoder_utils.sa_t5.SAT5Model
                clip_target.tokenizer = cores.text_encoder_utils.sa_t5.SAT5Tokenizer
        elif te_model == TEModel.GEMMA_2_2B:
            clip_target.clip = cores.text_encoder_utils.lumina2.te(
                **llama_detect(clip_data)
            )
            clip_target.tokenizer = cores.text_encoder_utils.lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.LLAMA3_8:
            clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
                **llama_detect(clip_data),
                clip_l=False,
                clip_g=False,
                t5=False,
                llama=True,
                dtype_t5=None,
                t5xxl_scaled_fp8=None,
            )
            clip_target.tokenizer = cores.text_encoder_utils.hidream.HiDreamTokenizer
        else:
            # clip_l
            if clip_type == CLIPType.SD3:
                clip_target.clip = cores.text_encoder_utils.sd3_clip.sd3_clip(
                    clip_l=True, clip_g=False, t5=False
                )
                clip_target.tokenizer = cores.text_encoder_utils.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
                    clip_l=True,
                    clip_g=False,
                    t5=False,
                    llama=False,
                    dtype_t5=None,
                    dtype_llama=None,
                    t5xxl_scaled_fp8=None,
                    llama_scaled_fp8=None,
                )
                clip_target.tokenizer = (
                    cores.text_encoder_utils.hidream.HiDreamTokenizer
                )
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = cores.text_encoder_utils.sd3_clip.sd3_clip(
                clip_l=TEModel.CLIP_L in te_models,
                clip_g=TEModel.CLIP_G in te_models,
                t5=TEModel.T5_XXL in te_models,
                **t5xxl_detect(clip_data),
            )
            clip_target.tokenizer = cores.text_encoder_utils.sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = cores.text_encoder_utils.hydit.HyditModel
            clip_target.tokenizer = cores.text_encoder_utils.hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            clip_target.clip = cores.text_encoder_utils.flux.flux_clip(
                **t5xxl_detect(clip_data)
            )
            clip_target.tokenizer = cores.text_encoder_utils.flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = (
                cores.text_encoder_utils.hunyuan_video.hunyuan_video_clip(
                    **llama_detect(clip_data)
                )
            )
            clip_target.tokenizer = (
                cores.text_encoder_utils.hunyuan_video.HunyuanVideoTokenizer
            )
        elif clip_type == CLIPType.HIDREAM:
            # Detect
            hidream_dualclip_classes = []
            for hidream_te in clip_data:
                te_model = detect_te_model(hidream_te)
                hidream_dualclip_classes.append(te_model)

            clip_l = TEModel.CLIP_L in hidream_dualclip_classes
            clip_g = TEModel.CLIP_G in hidream_dualclip_classes
            t5 = TEModel.T5_XXL in hidream_dualclip_classes
            llama = TEModel.LLAMA3_8 in hidream_dualclip_classes

            # Initialize t5xxl_detect and llama_detect kwargs if needed
            t5_kwargs = t5xxl_detect(clip_data) if t5 else {}
            llama_kwargs = llama_detect(clip_data) if llama else {}

            clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
                clip_l=clip_l,
                clip_g=clip_g,
                t5=t5,
                llama=llama,
                **t5_kwargs,
                **llama_kwargs,
            )
            clip_target.tokenizer = cores.text_encoder_utils.hidream.HiDreamTokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = cores.text_encoder_utils.sd3_clip.sd3_clip(
            **t5xxl_detect(clip_data)
        )
        clip_target.tokenizer = cores.text_encoder_utils.sd3_clip.SD3Tokenizer
    elif len(clip_data) == 4:
        clip_target.clip = cores.text_encoder_utils.hidream.hidream_clip(
            **t5xxl_detect(clip_data), **llama_detect(clip_data)
        )
        clip_target.tokenizer = cores.text_encoder_utils.hidream.HiDreamTokenizer

    parameters = 0
    for c in clip_data:
        parameters += cores.utils.calculate_parameters(c)
        tokenizer_data, model_options = (
            cores.text_encoder_utils.long_clipl.model_options_long_clip(
                c, tokenizer_data, model_options
            )
        )
    clip = CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        tokenizer_data=tokenizer_data,
        model_options=model_options,
    )
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip
