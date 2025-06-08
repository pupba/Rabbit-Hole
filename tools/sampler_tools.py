from typing import List
import torch

# cores
import cores.samplers
from cores.sample import prepare_noise
from cores.samplers import (
    KSAMPLER_NAMES,
    SCHEDULER_NAMES,
    sampler_object,
    calculate_sigmas,
)


def get_ksampler_namses() -> List[str]:
    return KSAMPLER_NAMES


def get_scheduler_names() -> List[str]:
    return SCHEDULER_NAMES


class Guider_Basic(cores.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return prepare_noise(latent_image, self.seed, batch_inds)
