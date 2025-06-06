import torch
import os
import cores.path_utils
import copy

import cores.model_management_utils as model_management_utils

import torch.nn as nn

from tools.ipadapter_plus.image_proj_models import (
    MLPProjModel,
    MLPProjModelFaceId,
    ProjModelFaceIdPlus,
    Resampler,
    ImageProjModel,
)
from tools.ipadapter_plus.CrossAttentionPatch import Attn2Replace, ipadapter_attention
from tools.ipadapter_plus.utils import (
    encode_image_masked,
    tensor_to_size,
    tensor_to_image,
    image_to_tensor,
)

# set the models directory
if "ipadapter" not in cores.path_utils.folder_names_and_paths:
    current_paths = [os.path.join(cores.path_utils.models_dir, "ipadapter")]
else:
    current_paths, _ = cores.path_utils.folder_names_and_paths["ipadapter"]
cores.path_utils.folder_names_and_paths["ipadapter"] = (
    current_paths,
    cores.path_utils.supported_pt_extensions,
)

WEIGHT_TYPES = [
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
]

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main IPAdapter Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class IPAdapter(nn.Module):
    def __init__(
        self,
        ipadapter_model,
        cross_attention_dim=1024,
        output_cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
        is_sdxl=False,
        is_plus=False,
        is_full=False,
        is_faceid=False,
        is_portrait_unnorm=False,
        is_kwai_kolors=False,
        encoder_hid_proj=None,
        weight_kolors=1.0,
    ):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_portrait_unnorm = is_portrait_unnorm
        self.is_kwai_kolors = is_kwai_kolors

        if is_faceid and not is_portrait_unnorm:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus or is_portrait_unnorm:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(
            ipadapter_model["ip_adapter"],
            encoder_hid_proj=encoder_hid_proj,
            weight_kolors=weight_kolors,
        )

        self.multigpu_clones = {}

    def create_multigpu_clone(self, device):
        if device not in self.multigpu_clones:
            orig_multigpu_clones = self.multigpu_clones
            try:
                self.multigpu_clones = {}
                new_clone = copy.deepcopy(self)
                new_clone = new_clone.to(device)
                orig_multigpu_clones[device] = new_clone
            finally:
                self.multigpu_clones = orig_multigpu_clones

    def get_multigpu_clone(self, device):
        return self.multigpu_clones.get(device, self)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
        )
        return image_proj_model

    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl and not self.is_kwai_kolors else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=self.clip_extra_context_tokens,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed, batch_size):
        torch_device = model_management_utils.get_torch_device()
        intermediate_device = model_management_utils.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        clip_embed = torch.split(clip_embed, batch_size, dim=0)
        clip_embed_zeroed = torch.split(clip_embed_zeroed, batch_size, dim=0)

        image_prompt_embeds = []
        uncond_image_prompt_embeds = []

        for ce, cez in zip(clip_embed, clip_embed_zeroed):
            image_prompt_embeds.append(
                self.image_proj_model(ce.to(torch_device)).to(intermediate_device)
            )
            uncond_image_prompt_embeds.append(
                self.image_proj_model(cez.to(torch_device)).to(intermediate_device)
            )

        del clip_embed, clip_embed_zeroed

        image_prompt_embeds = torch.cat(image_prompt_embeds, dim=0)
        uncond_image_prompt_embeds = torch.cat(uncond_image_prompt_embeds, dim=0)

        torch.cuda.empty_cache()

        # image_prompt_embeds = self.image_proj_model(clip_embed)
        # uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(
        self, face_embed, clip_embed, s_scale, shortcut, batch_size
    ):
        torch_device = model_management_utils.get_torch_device()
        intermediate_device = model_management_utils.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        face_embed_batch = torch.split(face_embed, batch_size, dim=0)
        clip_embed_batch = torch.split(clip_embed, batch_size, dim=0)

        embeds = []
        for face_embed, clip_embed in zip(face_embed_batch, clip_embed_batch):
            embeds.append(
                self.image_proj_model(
                    face_embed.to(torch_device),
                    clip_embed.to(torch_device),
                    scale=s_scale,
                    shortcut=shortcut,
                ).to(intermediate_device)
            )

        embeds = torch.cat(embeds, dim=0)
        del face_embed_batch, clip_embed_batch
        torch.cuda.empty_cache()
        # embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds


class To_KV(nn.Module):
    def __init__(self, state_dict, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        if encoder_hid_proj is not None:
            hid_proj = nn.Linear(
                encoder_hid_proj["weight"].shape[1],
                encoder_hid_proj["weight"].shape[0],
                bias=True,
            )
            hid_proj.weight.data = encoder_hid_proj["weight"] * weight_kolors
            hid_proj.bias.data = encoder_hid_proj["bias"] * weight_kolors

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            if encoder_hid_proj is not None:
                linear_proj = nn.Linear(value.shape[1], value.shape[0], bias=False)
                linear_proj.weight.data = value
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = (
                    nn.Sequential(hid_proj, linear_proj)
                )
            else:
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(
                    value.shape[1], value.shape[0], bias=False
                )
                self.to_kvs[
                    key.replace(".weight", "").replace(".", "_")
                ].weight.data = value


def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(
            ipadapter_attention, **patch_kwargs
        )
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(ipadapter_attention, **patch_kwargs)


def ipadapter_execute(
    model,
    ipadapter,
    clipvision,
    insightface=None,
    image=None,
    image_composition=None,
    image_negative=None,
    weight=1.0,
    weight_composition=1.0,
    weight_faceidv2=None,
    weight_kolors=1.0,
    weight_type="linear",
    combine_embeds="concat",
    start_at=0.0,
    end_at=1.0,
    attn_mask=None,
    pos_embed=None,
    neg_embed=None,
    unfold_batch=False,
    embeds_scaling="V only",
    layer_weights=None,
    encode_batch_size=0,
    style_boost=None,
    composition_boost=None,
    enhance_tiles=1,
    enhance_ratio=1.0,
):
    device = model_management_utils.get_torch_device()
    dtype = model_management_utils.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = (
            torch.float16 if model_management_utils.should_use_fp16() else torch.float32
        )

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait_unnorm = "portraitunnorm" in ipadapter
    is_plus = (
        is_full
        or "latents" in ipadapter["image_proj"]
        or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]
    ) and not is_portrait_unnorm
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048
    is_kwai_kolors_faceid = (
        "perceiver_resampler.layers.0.0.to_out.weight" in ipadapter["image_proj"]
        and ipadapter["image_proj"][
            "perceiver_resampler.layers.0.0.to_out.weight"
        ].shape[0]
        == 4096
    )
    is_faceidv2 = "faceidplusv2" in ipadapter or is_kwai_kolors_faceid
    is_kwai_kolors = (
        is_sdxl
        and "layers.0.0.to_out.weight" in ipadapter["image_proj"]
        and ipadapter["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048
    ) or is_kwai_kolors_faceid
    is_portrait = (
        "proj.2.weight" in ipadapter["image_proj"]
        and not "proj.3.weight" in ipadapter["image_proj"]
        and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
        and not is_kwai_kolors_faceid
    )
    is_faceid = (
        is_portrait
        or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
        or is_portrait_unnorm
        or is_kwai_kolors_faceid
    )

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight * 2

    if is_kwai_kolors_faceid:
        cross_attention_dim = 4096
    elif is_kwai_kolors:
        cross_attention_dim = 2048
    elif (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm:
        cross_attention_dim = 1280
    else:
        cross_attention_dim = output_cross_attention_dim

    if is_kwai_kolors_faceid:
        clip_extra_context_tokens = 6
    elif (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm:
        clip_extra_context_tokens = 16
    else:
        clip_extra_context_tokens = 4

    if image is not None and image.shape[1] != image.shape[2]:
        print(
            "\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m"
        )

    if isinstance(weight, list):
        weight = (
            torch.tensor(weight).unsqueeze(-1).unsqueeze(-1).to(device, dtype=dtype)
            if unfold_batch
            else weight[0]
        )

    if style_boost is not None:
        weight_type = "style transfer precise"
    elif composition_boost is not None:
        weight_type = "composition precise"

    # special weight types
    if layer_weights is not None and layer_weights != "":
        weight = {
            int(k): float(v) * weight
            for k, v in [x.split(":") for x in layer_weights.split(",")]
        }
        weight_type = (
            weight_type
            if weight_type == "style transfer precise"
            or weight_type == "composition precise"
            else "linear"
        )
    elif weight_type == "style transfer":
        weight = (
            {6: weight}
            if is_sdxl
            else {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
        )
    elif weight_type == "composition":
        weight = {3: weight} if is_sdxl else {4: weight * 0.25, 5: weight}
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                4: weight,
                5: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
            }
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = {3: weight_composition, 6: weight}
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition * 0.25,
                5: weight_composition,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight_composition,
                4: weight,
                5: weight,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
            }
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition,
                5: weight_composition,
                6: weight,
                7: weight,
                8: weight,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "style transfer precise":
        weight_composition = style_boost if style_boost is not None else weight
        if is_sdxl:
            weight = {3: weight_composition, 6: weight}
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition * 0.25,
                5: weight_composition,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }
    elif weight_type == "composition precise":
        weight_composition = weight
        weight = composition_boost if composition_boost is not None else weight
        if is_sdxl:
            weight = {
                0: weight * 0.1,
                1: weight * 0.1,
                2: weight * 0.1,
                3: weight_composition,
                4: weight * 0.1,
                5: weight * 0.1,
                6: weight,
                7: weight * 0.1,
                8: weight * 0.1,
                9: weight * 0.1,
                10: weight * 0.1,
            }
        else:
            weight = {
                0: weight,
                1: weight,
                2: weight,
                3: weight,
                4: weight_composition * 0.25,
                5: weight_composition,
                6: weight * 0.1,
                7: weight * 0.1,
                8: weight * 0.1,
                9: weight,
                10: weight,
                11: weight,
                12: weight,
                13: weight,
                14: weight,
                15: weight,
            }

    clipvision_size = 224 if not is_kwai_kolors else 336

    img_comp_cond_embeds = None
    face_cond_embeds = None
    if is_faceid:
        if insightface is None:
            raise Exception("Insightface model is required for FaceID models")

        from insightface.utils import face_align

        insightface.det_model.input_size = (640, 640)  # reset the detection size
        image_iface = tensor_to_image(image)
        face_cond_embeds = []
        image = []

        for i in range(image_iface.shape[0]):
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = (
                    size  # TODO: hacky but seems to be working
                )
                face = insightface.get(image_iface[i])
                if face:
                    if not is_portrait_unnorm:
                        face_cond_embeds.append(
                            torch.from_numpy(face[0].normed_embedding).unsqueeze(0)
                        )
                    else:
                        face_cond_embeds.append(
                            torch.from_numpy(face[0].embedding).unsqueeze(0)
                        )
                    image.append(
                        image_to_tensor(
                            face_align.norm_crop(
                                image_iface[i],
                                landmark=face[0].kps,
                                image_size=(
                                    336
                                    if is_kwai_kolors_faceid
                                    else 256 if is_sdxl else 224
                                ),
                            )
                        )
                    )

                    if 640 not in size:
                        print(
                            f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m"
                        )
                    break
            else:
                raise Exception("InsightFace: No face detected.")
        face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
        image = torch.stack(image)
        del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(
            clipvision,
            image,
            batch_size=encode_batch_size,
            tiles=enhance_tiles,
            ratio=enhance_ratio,
            clipvision_size=clipvision_size,
        )
        if image_composition is not None:
            img_comp_cond_embeds = encode_image_masked(
                clipvision,
                image_composition,
                batch_size=encode_batch_size,
                tiles=enhance_tiles,
                ratio=enhance_ratio,
                clipvision_size=clipvision_size,
            )

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = (
                image_negative
                if image_negative is not None
                else torch.zeros([1, clipvision_size, clipvision_size, 3])
            )
            img_uncond_embeds = encode_image_masked(
                clipvision,
                image_negative,
                batch_size=encode_batch_size,
                clipvision_size=clipvision_size,
            ).penultimate_hidden_states
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        else:
            img_cond_embeds = (
                img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            )
            if image_negative is not None and not is_faceid:
                img_uncond_embeds = encode_image_masked(
                    clipvision,
                    image_negative,
                    batch_size=encode_batch_size,
                    clipvision_size=clipvision_size,
                ).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.image_embeds
        del image_negative, image_composition

        image = (
            None if not is_faceid else image
        )  # if it's face_id we need the cropped face for later
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(
                    clipvision,
                    torch.zeros([1, clipvision_size, clipvision_size, 3]),
                    clipvision_size=clipvision_size,
                ).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        del pos_embed, neg_embed
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
    if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.sum(img_comp_cond_embeds, dim=0).unsqueeze(
                    0
                )
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(
                img_cond_embeds[1:], dim=0
            )
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(
                    face_cond_embeds[1:], dim=0
                )
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = img_comp_cond_embeds[0] - torch.mean(
                    img_comp_cond_embeds[1:], dim=0
                )
                img_comp_cond_embeds = img_comp_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(
                    img_comp_cond_embeds, dim=0
                ).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(
                img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True),
                dim=0,
            ).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(
                    face_cond_embeds
                    / torch.norm(face_cond_embeds, dim=0, keepdim=True),
                    dim=0,
                ).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(
                    img_comp_cond_embeds
                    / torch.norm(img_comp_cond_embeds, dim=0, keepdim=True),
                    dim=0,
                ).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(
            0
        )  # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)

    encoder_hid_proj = None

    if (
        is_kwai_kolors_faceid
        and hasattr(model.model, "diffusion_model")
        and hasattr(model.model.diffusion_model, "encoder_hid_proj")
    ):
        encoder_hid_proj = model.model.diffusion_model.encoder_hid_proj.state_dict()

    ipa = IPAdapter(
        ipadapter,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=img_cond_embeds.shape[-1],
        clip_extra_context_tokens=clip_extra_context_tokens,
        is_sdxl=is_sdxl,
        is_plus=is_plus,
        is_full=is_full,
        is_faceid=is_faceid,
        is_portrait_unnorm=is_portrait_unnorm,
        is_kwai_kolors=is_kwai_kolors,
        encoder_hid_proj=encoder_hid_proj,
        weight_kolors=weight_kolors,
    ).to(device, dtype=dtype)

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(
            face_cond_embeds,
            img_cond_embeds,
            weight_faceidv2,
            is_faceidv2,
            encode_batch_size,
        )
        # TODO: check if noise helps with the uncond face embeds
        uncond = ipa.get_image_embeds_faceid_plus(
            torch.zeros_like(face_cond_embeds),
            img_uncond_embeds,
            weight_faceidv2,
            is_faceidv2,
            encode_batch_size,
        )
    else:
        cond, uncond = ipa.get_image_embeds(
            img_cond_embeds, img_uncond_embeds, encode_batch_size
        )
        if img_comp_cond_embeds is not None:
            cond_comp = ipa.get_image_embeds(
                img_comp_cond_embeds, img_uncond_embeds, encode_batch_size
            )[0]

    cond = cond.to(device, dtype=dtype)
    uncond = uncond.to(device, dtype=dtype)

    cond_alt = None
    if img_comp_cond_embeds is not None:
        cond_alt = {3: cond_comp.to(device, dtype=dtype)}

    del img_cond_embeds, img_uncond_embeds, img_comp_cond_embeds, face_cond_embeds

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "weight": weight,
        "cond": cond,
        "cond_alt": cond_alt,
        "uncond": uncond,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    number = 0
    if not is_sdxl:
        for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            number += 1
        for id in [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]:  # id of output_blocks that have cross attention
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            number += 1
        patch_kwargs["module_key"] = str(number * 2 + 1)
        set_model_patch_replace(model, patch_kwargs, ("middle", 1))
    else:
        for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6):  # id of output_blocks that have cross attention
            block_indices = (
                range(2) if id in [3, 4, 5] else range(10)
            )  # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 1, index))
            number += 1

    return (model, image)
