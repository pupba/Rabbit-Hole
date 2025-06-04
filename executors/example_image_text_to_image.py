import time
import logging
from PIL import Image

# cores
from cores.cli_args_utils import args

# tunnels
from tunnels.load_models import (
    load_checkpoint,
    load_vae,
    load_lora,
    load_control_net,
    load_upscale_model,
)
from tunnels.encoders import encode_SDXL, clip_skip
from tunnels.latents import empty_rgb
from tunnels.samplers import ksampler
from tunnels.vaes import vae_decode, vae_encode
from tunnels.images import (
    save_images,
    load_image_from_path,
    enhance_hint_image,
    get_image_sizes,
)
from tunnels.processors import Processor
from tunnels.controlnets import apply_controlnet
from tunnels.upscales import upscale_by_model

# utils
from executors.utils import load_yaml_config


class IT2IExecutor:
    def __init__(self):
        # model_load
        configs = load_yaml_config(args.config_path)
        MODEL, CLIP, _ = load_checkpoint(configs["models"]["checkpoint"])
        CLIP = clip_skip(CLIP, -2)  # skip 2
        self.MODEL, self.CLIP = load_lora(
            MODEL, CLIP, configs["models"]["lora"], 1, 0.8
        )
        self.VAE = load_vae(configs["models"]["vae"])
        self.CONTROLNET = load_control_net(configs["models"]["controlnet"])
        self.UPSCALE = load_upscale_model(configs["models"]["upscale"])
        self.CANNY = Processor().canny

    def __call__(
        self,
        image_path: str,
        pos_text: str,
        neg_text: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
    ):
        start_time = time.time()

        image = load_image_from_path(image_path)

        h, w = get_image_sizes(image)[0]

        image = enhance_hint_image(image, 1024, 1024, "Resize and Fill")
        canny = self.CANNY(image)

        pos = encode_SDXL(self.CLIP, text_g=pos_text, text_l=pos_text)
        neg = encode_SDXL(self.CLIP, text_g=neg_text, text_l=neg_text)

        pos_c, neg_c = apply_controlnet(pos, neg, self.CONTROLNET, canny)

        rgb = empty_rgb(width, height, 1, 255, 255, 255)
        latent = vae_encode(self.VAE, rgb)

        samples = ksampler(
            self.MODEL,
            latent,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            pos_c,
            neg_c,
            denoise,
        )

        out = vae_decode(self.VAE, samples)
        out = enhance_hint_image(out, w, h, "Crop and Resize")
        out = upscale_by_model(self.UPSCALE, out)
        save_images(out)
        elapsed = time.time() - start_time
        logging.info(f"\n\nInference Time: {elapsed:.2f}sec.\n\n")
