import torch
import logging
import math
import json

from enum import Enum

from cores import model_management_utils
from cores.utils import ProgressBar
from cores import diffusers_convert
from cores.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
from cores.ldm.cascade.stage_a import StageA
from cores.ldm.cascade.stage_c_coder import StageC_coder
from cores.ldm.audio.autoencoder import AudioOobleckVAE


import cores.model_patcher
import cores.hooks


class TEModel(Enum):
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16


class CLIP:
    def __init__(
        self,
        target=None,
        embedding_directory=None,
        no_init=False,
        tokenizer_data={},
        parameters=0,
        model_options={},
    ):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip

        tokenizer = target.tokenizer

        load_device = model_options.get(
            "load_device", model_management_utils.text_encoder_device()
        )
        offload_device = model_options.get(
            "offload_device", model_management_utils.text_encoder_offload_device()
        )
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management_utils.text_encoder_dtype(load_device)

        params["dtype"] = dtype
        params["device"] = model_options.get(
            "initial_device",
            model_management_utils.text_encoder_initial_device(
                load_device,
                offload_device,
                parameters * model_management_utils.dtype_size(dtype),
            ),
        )
        params["model_options"] = model_options
        self.cond_stage_model = clip(**(params))

        for dt in self.cond_stage_model.dtypes:
            if not model_management_utils.supports_cast(load_device, dt):
                load_device = offload_device
                if params["device"] != offload_device:
                    self.cond_stage_model.to(offload_device)
                    logging.warning("Had to shift TE back.")

        self.tokenizer = tokenizer(
            embedding_directory=embedding_directory, tokenizer_data=tokenizer_data
        )
        self.patcher = cores.model_patcher.ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.patcher.hook_mode = cores.hooks.EnumHookMode.MinVram
        self.patcher.is_clip = True
        self.apply_hooks_to_conds = None

        if params["device"] == load_device:
            model_management_utils.load_models_gpu([self.patcher], force_full_load=True)
        self.layer_idx = None
        self.use_clip_schedule = False
        logging.info(
            "CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(
                load_device, offload_device, params["device"], dtype
            )
        )
        self.tokenizer_options = {}

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def set_tokenizer_option(self, option_name, value):
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
        if len(tokenizer_options) > 0:
            kwargs["tokenizer_options"] = tokenizer_options
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def add_hooks_to_dict(self, pooled_dict: dict[str]):
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds
        return pooled_dict

    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict: dict[str] = {}, show_pbar=True
    ):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(
                tokens, return_pooled=return_pooled, return_dict=True
            )
            cond = pooled_dict.pop("cond")
            # add/update any keys with the provided add_dict
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model()
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            for scheduled_opts in scheduled_keyframes:
                t_range = scheduled_opts[0]
                # don't bother encoding any conds outside of start_percent and end_percent bounds
                if "start_percent" in add_dict:
                    if t_range[1] < add_dict["start_percent"]:
                        continue
                if "end_percent" in add_dict:
                    if t_range[0] > add_dict["end_percent"]:
                        continue
                hooks_keyframes = scheduled_opts[1]
                for hook, keyframe in hooks_keyframes:
                    hook.hook_keyframe._current_keyframe = keyframe
                # apply appropriate hooks with values that match new hook_keyframe
                self.patcher.patch_hooks(all_hooks)
                # perform encoding as normal
                o = self.cond_stage_model.encode_token_weights(tokens)
                cond, pooled = o[:2]
                pooled_dict = {"pooled_output": pooled}
                # add clip_start_percent and clip_end_percent in pooled
                pooled_dict["clip_start_percent"] = t_range[0]
                pooled_dict["clip_end_percent"] = t_range[1]
                # add/update any keys with the provided add_dict
                pooled_dict.update(add_dict)
                # add hooks stored on clip
                self.add_hooks_to_dict(pooled_dict)
                all_cond_pooled.append([cond, pooled_dict])
                if show_pbar:
                    pbar.update(1)
                model_management_utils.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self):
        model_management_utils.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        if (
            "decoder.up_blocks.0.resnets.0.norm1.weight" in sd.keys()
        ):  # diffusers format
            sd = diffusers_convert.convert_vae_state_dict(sd)

        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * model_management_utils.dtype_size(
            dtype
        )  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * model_management_utils.dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False

        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None

        if config is None:
            if "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(
                    regularizer_config={
                        "target": "cores.ldm.models.autoencoder.DiagonalGaussianRegularizer"
                    },
                    encoder_config={
                        "target": "cores.ldm.modules.diffusionmodules.model.Encoder",
                        "params": encoder_config,
                    },
                    decoder_config={
                        "target": "cores.ldm.modules.temporal_ae.VideoDecoder",
                        "params": decoder_config,
                    },
                )
            elif "taesd_decoder.1.weight" in sd:
                self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
                self.first_stage_model = cores.taesd.taesd.TAESD(
                    latent_channels=self.latent_channels
                )
            elif "vquantizer.codebook.weight" in sd:  # VQGan: stage a of stable cascade
                self.first_stage_model = StageA()
                self.downscale_ratio = 4
                self.upscale_ratio = 4
                # TODO
                # self.memory_used_encode
                # self.memory_used_decode
                self.process_input = lambda image: image
                self.process_output = lambda image: image
            elif (
                "backbone.1.0.block.0.1.num_batches_tracked" in sd
            ):  # effnet: encoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["encoder.{}".format(k)] = sd[k]
                sd = new_sd
            elif (
                "blocks.11.num_batches_tracked" in sd
            ):  # previewer: decoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["previewer.{}".format(k)] = sd[k]
                sd = new_sd
            elif (
                "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd
            ):  # combined effnet and previewer for stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
            elif "decoder.conv_in.weight" in sd:
                # default SD1.x/SD2.x VAE parameters
                ddconfig = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }

                if (
                    "encoder.down.2.downsample.conv.weight" not in sd
                    and "decoder.up.3.upsample.conv.weight" not in sd
                ):  # Stable diffusion x4 upscaler VAE
                    ddconfig["ch_mult"] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig["z_channels"] = sd[
                    "decoder.conv_in.weight"
                ].shape[1]
                if "post_quant_conv.weight" in sd:
                    self.first_stage_model = AutoencoderKL(
                        ddconfig=ddconfig,
                        embed_dim=sd["post_quant_conv.weight"].shape[1],
                    )
                else:
                    self.first_stage_model = AutoencodingEngine(
                        regularizer_config={
                            "target": "cores.ldm.models.autoencoder.DiagonalGaussianRegularizer"
                        },
                        encoder_config={
                            "target": "cores.ldm.modules.diffusionmodules.model.Encoder",
                            "params": ddconfig,
                        },
                        decoder_config={
                            "target": "cores.ldm.modules.diffusionmodules.model.Decoder",
                            "params": ddconfig,
                        },
                    )
            elif "decoder.layers.1.layers.0.beta" in sd:
                self.first_stage_model = AudioOobleckVAE()
                self.memory_used_encode = lambda shape, dtype: (
                    1000 * shape[2]
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (
                    1000 * shape[2] * 2048
                ) * model_management_utils.dtype_size(dtype)
                self.latent_channels = 64
                self.output_channels = 2
                self.upscale_ratio = 2048
                self.downscale_ratio = 2048
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.disable_offload = True
            elif (
                "blocks.2.blocks.3.stack.5.weight" in sd
                or "decoder.blocks.2.blocks.3.stack.5.weight" in sd
                or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd
                or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd
            ):  # genmo mochi vae
                if "blocks.2.blocks.3.stack.5.weight" in sd:
                    sd = cores.utils.state_dict_prefix_replace(sd, {"": "decoder."})
                if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                    sd = cores.utils.state_dict_prefix_replace(sd, {"": "encoder."})
                self.first_stage_model = cores.ldm.genmo.vae.model.VideoVAE()
                self.latent_channels = 12
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (
                    1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (
                    1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)
                ) * model_management_utils.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
                self.upscale_index_formula = (6, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
                self.downscale_index_formula = (6, 8, 8)
                self.working_dtypes = [torch.float16, torch.float32]
            elif (
                "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd
            ):  # lightricks ltxv
                tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
                version = 0
                if tensor_conv1.shape[0] == 512:
                    version = 0
                elif tensor_conv1.shape[0] == 1024:
                    version = 1
                    if "encoder.down_blocks.1.conv.conv.bias" in sd:
                        version = 2
                vae_config = None
                if metadata is not None and "config" in metadata:
                    vae_config = json.loads(metadata["config"]).get("vae", None)
                self.first_stage_model = (
                    cores.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(
                        version=version, config=vae_config
                    )
                )
                self.latent_channels = 128
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (
                    900 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (
                    70 * max(shape[2], 7) * shape[3] * shape[4]
                ) * model_management_utils.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                self.upscale_index_formula = (8, 32, 32)
                self.downscale_ratio = (
                    lambda a: max(0, math.floor((a + 7) / 8)),
                    32,
                    32,
                )
                self.downscale_index_formula = (8, 32, 32)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.conv_in.conv.weight" in sd:
                ddconfig = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }
                ddconfig["conv3d"] = True
                ddconfig["time_compress"] = 4
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = ddconfig["z_channels"] = sd[
                    "decoder.conv_in.conv.weight"
                ].shape[1]
                self.first_stage_model = AutoencoderKL(
                    ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1]
                )
                self.memory_used_decode = lambda shape, dtype: (
                    1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (
                    900 * max(shape[2], 2) * shape[3] * shape[4]
                ) * model_management_utils.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            elif "decoder.unpatcher3d.wavelets" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
                self.upscale_index_formula = (8, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
                self.downscale_index_formula = (8, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {
                    "z_channels": 16,
                    "latent_channels": self.latent_channels,
                    "z_factor": 1,
                    "resolution": 1024,
                    "in_channels": 3,
                    "out_channels": 3,
                    "channels": 128,
                    "channels_mult": [2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [32],
                    "dropout": 0.0,
                    "patch_size": 4,
                    "num_groups": 1,
                    "temporal_compression": 8,
                    "spacial_compression": 8,
                }
                self.first_stage_model = (
                    cores.ldm.cosmos.vae.CausalContinuousVideoTokenizer(**ddconfig)
                )
                # TODO: these values are a bit off because this is not a standard VAE
                self.memory_used_decode = lambda shape, dtype: (
                    50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (
                    50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]
                ) * model_management_utils.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.middle.0.residual.0.gamma" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {
                    "dim": 96,
                    "z_dim": self.latent_channels,
                    "dim_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_scales": [],
                    "temperal_downsample": [False, True, True],
                    "dropout": 0.0,
                }
                self.first_stage_model = cores.ldm.wan.vae.WanVAE(**ddconfig)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.memory_used_encode = (
                    lambda shape, dtype: 6000
                    * shape[3]
                    * shape[4]
                    * model_management_utils.dtype_size(dtype)
                )
                self.memory_used_decode = (
                    lambda shape, dtype: 7000
                    * shape[3]
                    * shape[4]
                    * (8 * 8)
                    * model_management_utils.dtype_size(dtype)
                )
            elif "geo_decoder.cross_attn_decoder.ln_1.bias" in sd:
                self.latent_dim = 1
                ln_post = "geo_decoder.ln_post.weight" in sd
                inner_size = sd["geo_decoder.output_proj.weight"].shape[1]
                downsample_ratio = sd["post_kl.weight"].shape[0] // inner_size
                mlp_expand = (
                    sd["geo_decoder.cross_attn_decoder.mlp.c_fc.weight"].shape[0]
                    // inner_size
                )
                self.memory_used_encode = lambda shape, dtype: (
                    1000 * shape[2]
                ) * model_management_utils.dtype_size(
                    dtype
                )  # TODO
                self.memory_used_decode = lambda shape, dtype: (
                    1024 * 1024 * 1024 * 2.0
                ) * model_management_utils.dtype_size(
                    dtype
                )  # TODO
                ddconfig = {
                    "embed_dim": 64,
                    "num_freqs": 8,
                    "include_pi": False,
                    "heads": 16,
                    "width": 1024,
                    "num_decoder_layers": 16,
                    "qkv_bias": False,
                    "qk_norm": True,
                    "geo_decoder_mlp_expand_ratio": mlp_expand,
                    "geo_decoder_downsample_ratio": downsample_ratio,
                    "geo_decoder_ln_post": ln_post,
                }
                self.first_stage_model = cores.ldm.hunyuan3d.vae.ShapeVAE(**ddconfig)
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            elif "vocoder.backbone.channel_layers.0.0.bias" in sd:  # Ace Step Audio
                self.first_stage_model = (
                    cores.ldm.ace.vae.music_dcae_pipeline.MusicDCAE(
                        source_sample_rate=44100
                    )
                )
                self.memory_used_encode = lambda shape, dtype: (
                    shape[2] * 330
                ) * model_management_utils.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (
                    shape[2] * shape[3] * 87000
                ) * model_management_utils.dtype_size(dtype)
                self.latent_channels = 8
                self.output_channels = 2
                self.upscale_ratio = 4096
                self.downscale_ratio = 4096
                self.latent_dim = 2
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.disable_offload = True
                self.extra_1d_channel = 16
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config["params"]))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = model_management_utils.vae_device()
        self.device = device
        offload_device = model_management_utils.vae_offload_device()
        if dtype is None:
            dtype = model_management_utils.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management_utils.intermediate_device()

        self.patcher = cores.model_patcher.ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )
        logging.info(
            "VAE load device: {}, offload device: {}, dtype: {}".format(
                self.device, offload_device, self.vae_dtype
            )
        )

    def throw_exception_if_invalid(self):
        if self.first_stage_model is None:
            raise RuntimeError(
                "ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your checkpoint does not contain a valid VAE."
            )

    def vae_encode_crop_pixels(self, pixels):
        downscale_ratio = self.spacial_compression_encode()

        dims = pixels.shape[1:-1]
        for d in range(len(dims)):
            x = (dims[d] // downscale_ratio) * downscale_ratio
            x_offset = (dims[d] % downscale_ratio) // 2
            if x != dims[d]:
                pixels = pixels.narrow(d + 1, x_offset, x)
        return pixels

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * cores.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x, tile_y, overlap
        )
        steps += samples.shape[0] * cores.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap
        )
        steps += samples.shape[0] * cores.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap
        )
        pbar = cores.utils.ProgressBar(steps)

        decode_fn = lambda a: self.first_stage_model.decode(
            a.to(self.vae_dtype).to(self.device)
        ).float()
        output = self.process_output(
            (
                cores.utils.tiled_scale(
                    samples,
                    decode_fn,
                    tile_x // 2,
                    tile_y * 2,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
                + cores.utils.tiled_scale(
                    samples,
                    decode_fn,
                    tile_x * 2,
                    tile_y // 2,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
                + cores.utils.tiled_scale(
                    samples,
                    decode_fn,
                    tile_x,
                    tile_y,
                    overlap,
                    upscale_amount=self.upscale_ratio,
                    output_device=self.output_device,
                    pbar=pbar,
                )
            )
            / 3.0
        )
        return output

    def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
        if samples.ndim == 3:
            decode_fn = lambda a: self.first_stage_model.decode(
                a.to(self.vae_dtype).to(self.device)
            ).float()
        else:
            og_shape = samples.shape
            samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
            decode_fn = lambda a: self.first_stage_model.decode(
                a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1]))
                .to(self.vae_dtype)
                .to(self.device)
            ).float()

        return self.process_output(
            cores.utils.tiled_scale_multidim(
                samples,
                decode_fn,
                tile=(tile_x,),
                overlap=overlap,
                upscale_amount=self.upscale_ratio,
                out_channels=self.output_channels,
                output_device=self.output_device,
            )
        )

    def decode_tiled_3d(
        self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)
    ):
        decode_fn = lambda a: self.first_stage_model.decode(
            a.to(self.vae_dtype).to(self.device)
        ).float()
        return self.process_output(
            cores.utils.tiled_scale_multidim(
                samples,
                decode_fn,
                tile=(tile_t, tile_x, tile_y),
                overlap=overlap,
                upscale_amount=self.upscale_ratio,
                out_channels=self.output_channels,
                index_formulas=self.upscale_index_formula,
                output_device=self.output_device,
            )
        )

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * cores.utils.get_tiled_scale_steps(
            pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap
        )
        steps += pixel_samples.shape[0] * cores.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x // 2,
            tile_y * 2,
            overlap,
        )
        steps += pixel_samples.shape[0] * cores.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x * 2,
            tile_y // 2,
            overlap,
        )
        pbar = cores.utils.ProgressBar(steps)

        encode_fn = lambda a: self.first_stage_model.encode(
            (self.process_input(a)).to(self.vae_dtype).to(self.device)
        ).float()
        samples = cores.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x,
            tile_y,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += cores.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x * 2,
            tile_y // 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += cores.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x // 2,
            tile_y * 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples /= 3.0
        return samples

    def encode_tiled_1d(self, samples, tile_x=256 * 2048, overlap=64 * 2048):
        if self.latent_dim == 1:
            encode_fn = lambda a: self.first_stage_model.encode(
                (self.process_input(a)).to(self.vae_dtype).to(self.device)
            ).float()
            out_channels = self.latent_channels
            upscale_amount = 1 / self.downscale_ratio
        else:
            extra_channel_size = self.extra_1d_channel
            out_channels = self.latent_channels * extra_channel_size
            tile_x = tile_x // extra_channel_size
            overlap = overlap // extra_channel_size
            upscale_amount = 1 / self.downscale_ratio
            encode_fn = (
                lambda a: self.first_stage_model.encode(
                    (self.process_input(a)).to(self.vae_dtype).to(self.device)
                )
                .reshape(1, out_channels, -1)
                .float()
            )

        out = cores.utils.tiled_scale_multidim(
            samples,
            encode_fn,
            tile=(tile_x,),
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=out_channels,
            output_device=self.output_device,
        )
        if self.latent_dim == 1:
            return out
        else:
            return out.reshape(
                samples.shape[0], self.latent_channels, extra_channel_size, -1
            )

    def encode_tiled_3d(
        self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)
    ):
        encode_fn = lambda a: self.first_stage_model.encode(
            (self.process_input(a)).to(self.vae_dtype).to(self.device)
        ).float()
        return cores.utils.tiled_scale_multidim(
            samples,
            encode_fn,
            tile=(tile_t, tile_x, tile_y),
            overlap=overlap,
            upscale_amount=self.downscale_ratio,
            out_channels=self.latent_channels,
            downscale=True,
            index_formulas=self.downscale_index_formula,
            output_device=self.output_device,
        )

    def decode(self, samples_in, vae_options={}):
        self.throw_exception_if_invalid()
        pixel_samples = None
        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            model_management_utils.load_models_gpu(
                [self.patcher],
                memory_required=memory_used,
                force_full_load=self.disable_offload,
            )
            free_memory = model_management_utils.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            for x in range(0, samples_in.shape[0], batch_number):
                samples = (
                    samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
                )
                out = self.process_output(
                    self.first_stage_model.decode(samples, **vae_options)
                    .to(self.output_device)
                    .float()
                )
                if pixel_samples is None:
                    pixel_samples = torch.empty(
                        (samples_in.shape[0],) + tuple(out.shape[1:]),
                        device=self.output_device,
                    )
                pixel_samples[x : x + batch_number] = out
        except model_management_utils.OOM_EXCEPTION:
            logging.warning(
                "Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding."
            )
            dims = samples_in.ndim - 2
            if dims == 1 or self.extra_1d_channel is not None:
                pixel_samples = self.decode_tiled_1d(samples_in)
            elif dims == 2:
                pixel_samples = self.decode_tiled_(samples_in)
            elif dims == 3:
                tile = 256 // self.spacial_compression_decode()
                overlap = tile // 4
                pixel_samples = self.decode_tiled_3d(
                    samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap)
                )

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode_tiled(
        self,
        samples,
        tile_x=None,
        tile_y=None,
        overlap=None,
        tile_t=None,
        overlap_t=None,
    ):
        self.throw_exception_if_invalid()
        memory_used = self.memory_used_decode(
            samples.shape, self.vae_dtype
        )  # TODO: calculate mem required for tile
        model_management_utils.load_models_gpu(
            [self.patcher],
            memory_required=memory_used,
            force_full_load=self.disable_offload,
        )
        dims = samples.ndim - 2
        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        if dims == 1:
            args.pop("tile_y")
            output = self.decode_tiled_1d(samples, **args)
        elif dims == 2:
            output = self.decode_tiled_(samples, **args)
        elif dims == 3:
            if overlap_t is None:
                args["overlap"] = (1, overlap, overlap)
            else:
                args["overlap"] = (max(1, overlap_t), overlap, overlap)
            if tile_t is not None:
                args["tile_t"] = max(2, tile_t)

            output = self.decode_tiled_3d(samples, **args)
        return output.movedim(1, -1)

    def encode(self, pixel_samples):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            model_management_utils.load_models_gpu(
                [self.patcher],
                memory_required=memory_used,
                force_full_load=self.disable_offload,
            )
            free_memory = model_management_utils.get_free_memory(self.device)
            batch_number = int(free_memory / max(1, memory_used))
            batch_number = max(1, batch_number)
            samples = None
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (
                    self.process_input(pixel_samples[x : x + batch_number])
                    .to(self.vae_dtype)
                    .to(self.device)
                )
                out = (
                    self.first_stage_model.encode(pixels_in)
                    .to(self.output_device)
                    .float()
                )
                if samples is None:
                    samples = torch.empty(
                        (pixel_samples.shape[0],) + tuple(out.shape[1:]),
                        device=self.output_device,
                    )
                samples[x : x + batch_number] = out

        except model_management_utils.OOM_EXCEPTION:
            logging.warning(
                "Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding."
            )
            if self.latent_dim == 3:
                tile = 256
                overlap = tile // 4
                samples = self.encode_tiled_3d(
                    pixel_samples,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=(1, overlap, overlap),
                )
            elif self.latent_dim == 1 or self.extra_1d_channel is not None:
                samples = self.encode_tiled_1d(pixel_samples)
            else:
                samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode_tiled(
        self,
        pixel_samples,
        tile_x=None,
        tile_y=None,
        overlap=None,
        tile_t=None,
        overlap_t=None,
    ):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        dims = self.latent_dim
        pixel_samples = pixel_samples.movedim(-1, 1)
        if dims == 3:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        memory_used = self.memory_used_encode(
            pixel_samples.shape, self.vae_dtype
        )  # TODO: calculate mem required for tile
        model_management_utils.load_models_gpu(
            [self.patcher],
            memory_required=memory_used,
            force_full_load=self.disable_offload,
        )

        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        if dims == 1:
            args.pop("tile_y")
            samples = self.encode_tiled_1d(pixel_samples, **args)
        elif dims == 2:
            samples = self.encode_tiled_(pixel_samples, **args)
        elif dims == 3:
            if tile_t is not None:
                tile_t_latent = max(2, self.downscale_ratio[0](tile_t))
            else:
                tile_t_latent = 9999
            args["tile_t"] = self.upscale_ratio[0](tile_t_latent)

            if overlap_t is None:
                args["overlap"] = (1, overlap, overlap)
            else:
                args["overlap"] = (
                    self.upscale_ratio[0](
                        max(
                            1,
                            min(tile_t_latent // 2, self.downscale_ratio[0](overlap_t)),
                        )
                    ),
                    overlap,
                    overlap,
                )
            maximum = pixel_samples.shape[2]
            maximum = self.upscale_ratio[0](self.downscale_ratio[0](maximum))

            samples = self.encode_tiled_3d(pixel_samples[:, :, :maximum], **args)

        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()

    def spacial_compression_decode(self):
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio

    def spacial_compression_encode(self):
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio

    def temporal_compression_decode(self):
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None
