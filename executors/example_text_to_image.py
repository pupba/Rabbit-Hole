import time
import logging

# cores
from cores.cli_args_utils import args

# tunnels
from tunnels.load_models import load_checkpoint, load_vae, load_lora, load_text_encoder
from tunnels.encoders import encode_SDXL, clip_skip
from tunnels.latents import empty_latent
from tunnels.samplers import ksampler
from tunnels.vaes import vae_decode
from tunnels.upscales import hires_fix
from tunnels.images import save_images

# utils
from executors.utils import load_yaml_config


class T2IExecutor:
    def __init__(self):
        # model_load
        configs = load_yaml_config(args.config_path)
        self.MODEL, CLIP, _ = load_checkpoint(configs["models"]["checkpoint"])
        self.CLIP = clip_skip(CLIP, -2)  # skip 2
        self.VAE = load_vae(configs["models"]["vae"])

    def __call__(
        self,
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
        pos = encode_SDXL(self.CLIP, text_g=pos_text, text_l=pos_text)
        neg = encode_SDXL(self.CLIP, text_g=neg_text, text_l=neg_text)
        latent = empty_latent(width=width, height=height)
        samples = ksampler(
            self.MODEL,
            latent,
            pos,
            neg,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        )
        out = vae_decode(self.VAE, samples)
        save_images(out)
        elapsed = time.time() - start_time
        logging.info(f"\n\nInference Time: {elapsed:.2f}sec.\n\n")


class T2ILoRAExecutor:
    def __init__(self):
        # model_load
        configs = load_yaml_config(args.config_path)
        MODEL, CLIP, _ = load_checkpoint(configs["models"]["checkpoint"])
        CLIP = clip_skip(CLIP, -2)  # skip 2
        self.MODEL, self.CLIP = load_lora(
            MODEL, CLIP, configs["models"]["lora"], 1, 0.8
        )
        self.VAE = load_vae(configs["models"]["vae"])

    def __call__(
        self,
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
        pos = encode_SDXL(self.CLIP, text_g=pos_text, text_l=pos_text)
        neg = encode_SDXL(self.CLIP, text_g=neg_text, text_l=neg_text)
        latent = empty_latent(width=width, height=height)
        samples = ksampler(
            self.MODEL,
            latent,
            pos,
            neg,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        )
        out = vae_decode(self.VAE, samples)
        save_images(out)
        elapsed = time.time() - start_time
        logging.info(f"\n\nInference Time: {elapsed:.2f}sec.\n\n")


class T2IHireFixExecutor:
    def __init__(self):
        # model_load
        configs = load_yaml_config(args.config_path)
        self.MODEL, CLIP, _ = load_checkpoint(configs["models"]["checkpoint"])
        self.CLIP = clip_skip(CLIP, -2)  # skip 2
        self.VAE = load_vae(configs["models"]["vae"])

    def __call__(
        self,
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
        pos = encode_SDXL(self.CLIP, text_g=pos_text, text_l=pos_text)
        neg = encode_SDXL(self.CLIP, text_g=neg_text, text_l=neg_text)
        latent = empty_latent(width=width, height=height)
        samples = ksampler(
            self.MODEL,
            latent,
            seed,
            pos,
            neg,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        )
        # hirefix
        upscale_configs = {"width": 1024, "height": 1500}
        sampler_configs = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
        }
        logging.info(f"HireFix: {upscale_configs}")
        samples = hires_fix(
            self.MODEL, pos, neg, samples, upscale_configs, sampler_configs
        )
        out = vae_decode(self.VAE, samples)
        save_images(out)
        elapsed = time.time() - start_time
        logging.info(f"\n\nInference Time: {elapsed:.2f}sec.\n\n")
