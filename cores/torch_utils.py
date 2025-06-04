"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

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

import torch
import safetensors
import logging
from typing import Any, Optional, Tuple, Dict

from cores import checkpoint_pickle

MMAP_TORCH_FILES = False
ALWAYS_SAFE_LOAD = False


# from ComfyUI/comfy/utils.py https://github.com/comfyanonymous/ComfyUI
def load_torch_file(
    ckpt: str,
    safe_load: bool = False,
    device: Optional[torch.device] = None,
    return_metadata: bool = False,
) -> Tuple[dict, Optional[dict]] | dict:
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith((".safetensors", ".sft")):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError(
                        f"{message}\n\nFile path: {ckpt}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype."
                    )
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError(
                        f"{message}\n\nFile path: {ckpt}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly."
                    )
            raise e
    else:
        torch_args = {}
        if MMAP_TORCH_FILES:
            torch_args["mmap"] = True
        if safe_load or ALWAYS_SAFE_LOAD:
            pl_sd = torch.load(
                ckpt, map_location=device, weights_only=True, **torch_args
            )
        else:
            pl_sd = torch.load(
                ckpt, map_location=device, pickle_module=checkpoint_pickle
            )
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd
