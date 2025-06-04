"""
This file is part of ComfyUI.
Copyright (C) 2024 Stability AI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import enum
import os
import cores.options


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--cuda-device",
    type=int,
    default=None,
    metavar="DEVICE_ID",
    help="Set the id of the cuda device this instance will use.",
)
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument(
    "--cuda-malloc",
    action="store_true",
    help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).",
)
cm_group.add_argument(
    "--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync."
)


fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument(
    "--force-fp32",
    action="store_true",
    help="Force fp32 (If this makes your GPU work better please report it).",
)
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument(
    "--fp32-unet", action="store_true", help="Run the diffusion model in fp32."
)
fpunet_group.add_argument(
    "--fp64-unet", action="store_true", help="Run the diffusion model in fp64."
)
fpunet_group.add_argument(
    "--bf16-unet", action="store_true", help="Run the diffusion model in bf16."
)
fpunet_group.add_argument(
    "--fp16-unet", action="store_true", help="Run the diffusion model in fp16"
)
fpunet_group.add_argument(
    "--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn."
)
fpunet_group.add_argument(
    "--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2."
)
fpunet_group.add_argument(
    "--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu."
)

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument(
    "--fp16-vae",
    action="store_true",
    help="Run the VAE in fp16, might cause black images.",
)
fpvae_group.add_argument(
    "--fp32-vae", action="store_true", help="Run the VAE in full precision fp32."
)
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument(
    "--fp8_e4m3fn-text-enc",
    action="store_true",
    help="Store text encoder weights in fp8 (e4m3fn variant).",
)
fpte_group.add_argument(
    "--fp8_e5m2-text-enc",
    action="store_true",
    help="Store text encoder weights in fp8 (e5m2 variant).",
)
fpte_group.add_argument(
    "--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16."
)
fpte_group.add_argument(
    "--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32."
)
fpte_group.add_argument(
    "--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16."
)

parser.add_argument(
    "--force-channels-last",
    action="store_true",
    help="Force channels last format when inferencing the models.",
)

parser.add_argument(
    "--directml",
    type=int,
    nargs="?",
    metavar="DIRECTML_DEVICE",
    const=-1,
    help="Use torch-directml.",
)

parser.add_argument(
    "--oneapi-device-selector",
    type=str,
    default=None,
    metavar="SELECTOR_STRING",
    help="Sets the oneAPI device(s) this instance will use.",
)
parser.add_argument(
    "--disable-ipex-optimize",
    action="store_true",
    help="Disables ipex.optimize default when loading models with Intel's Extension for Pytorch.",
)


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


parser.add_argument(
    "--preview-method",
    type=LatentPreviewMethod,
    default=LatentPreviewMethod.NoPreviews,
    help="Default preview method for sampler nodes.",
    action=EnumAction,
)

parser.add_argument(
    "--preview-size",
    type=int,
    default=512,
    help="Sets the maximum preview size for sampler nodes.",
)

"""
config_path
"""
parser.add_argument(
    "--config-path", type=str, default="configs/test.yaml", help="configs file path"
)

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument(
    "--cache-classic",
    action="store_true",
    help="Use the old style (aggressive) caching.",
)
cache_group.add_argument(
    "--cache-lru",
    type=int,
    default=0,
    help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.",
)
cache_group.add_argument(
    "--cache-none",
    action="store_true",
    help="Reduced RAM/VRAM usage at the expense of executing every node for each run.",
)

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument(
    "--use-split-cross-attention",
    action="store_true",
    help="Use the split cross attention optimization. Ignored when xformers is used.",
)
attn_group.add_argument(
    "--use-quad-cross-attention",
    action="store_true",
    help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.",
)
attn_group.add_argument(
    "--use-pytorch-cross-attention",
    action="store_true",
    help="Use the new pytorch 2.0 cross attention function.",
)
attn_group.add_argument(
    "--use-sage-attention", action="store_true", help="Use sage attention."
)
attn_group.add_argument(
    "--use-flash-attention", action="store_true", help="Use FlashAttention."
)

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument(
    "--force-upcast-attention",
    action="store_true",
    help="Force enable attention upcasting, please report if it fixes black images.",
)
upcast.add_argument(
    "--dont-upcast-attention",
    action="store_true",
    help="Disable all upcasting of attention. Should be unnecessary except for debugging.",
)


vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument(
    "--gpu-only",
    action="store_true",
    help="Store and run everything (text encoders/CLIP models, etc... on the GPU).",
)
vram_group.add_argument(
    "--highvram",
    action="store_true",
    help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.",
)
vram_group.add_argument(
    "--normalvram",
    action="store_true",
    help="Used to force normal vram use if lowvram gets automatically enabled.",
)
vram_group.add_argument(
    "--lowvram", action="store_true", help="Split the unet in parts to use less vram."
)
vram_group.add_argument(
    "--novram", action="store_true", help="When lowvram isn't enough."
)
vram_group.add_argument(
    "--cpu", action="store_true", help="To use the CPU for everything (slow)."
)

parser.add_argument(
    "--reserve-vram",
    type=float,
    default=None,
    help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.",
)

parser.add_argument(
    "--async-offload", action="store_true", help="Use async weight offloading."
)

parser.add_argument(
    "--default-hashing-function",
    type=str,
    choices=["md5", "sha1", "sha256", "sha512"],
    default="sha256",
    help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.",
)

parser.add_argument(
    "--disable-smart-memory",
    action="store_true",
    help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.",
)


class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"


parser.add_argument(
    "--fast",
    nargs="*",
    type=PerformanceFeature,
    help="Enable some untested and potentially quality deteriorating optimizations. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: fp16_accumulation fp8_matrix_mult cublas_ops",
)

parser.add_argument(
    "--mmap-torch-files",
    action="store_true",
    help="Use mmap when loading ckpt/pt files.",
)


parser.add_argument(
    "--verbose",
    default="INFO",
    const="DEBUG",
    nargs="?",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level",
)
parser.add_argument(
    "--log-stdout",
    action="store_true",
    help="Send normal process output to stdout instead of stderr (default).",
)


if cores.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.force_fp16:
    args.fp16_unet = True


# '--fast' is not provided, use an empty set
if args.fast is None:
    args.fast = set()
# '--fast' is provided with an empty list, enable all optimizations
elif args.fast == []:
    args.fast = set(PerformanceFeature)
# '--fast' is provided with a list of performance features, use that list
else:
    args.fast = set(args.fast)
