import torch
import gc

from cores.cli_args_utils import args

# tunnels
from tunnels.load_models import load_checkpoint, load_vae
from tunnels.encoders import encode_SDXL, clip_skip
from tunnels.vaes import vae_decode
from tunnels.images import load_image_from_path, save_images, enhance_hint_image
from tunnels.ipadapter_plus import ipadapter_unified_loader, ipadapter_simple
from tunnels.latents import empty_latent
from tunnels.samplers import ksampler


# utils
from executors.utils import load_yaml_config


class IPAdapterIT2I:
    def __init__(self):
        configs = load_yaml_config(args.config_path)
        self.MODEL, CLIP, _ = load_checkpoint(configs["models"]["checkpoint"])
        self.CLIP = clip_skip(CLIP, -2)
        self.VAE = load_vae(configs["models"]["vae"])
        self.MODEL, self.IPADAPTER = ipadapter_unified_loader(
            self.MODEL, preset="STANDARD"
        )

    def __call__(
        self,
        image_path: str,
        ppromt: str,
        nprompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
    ):
        try:
            image = load_image_from_path(image_path)
            image = enhance_hint_image(image, 1024, 1024, resize_mode="Resize and Fill")
            MODEL, _ = ipadapter_simple(self.MODEL, self.IPADAPTER, image, 0.8)

            pos = encode_SDXL(self.CLIP, text_g=ppromt, text_l=ppromt)
            neg = encode_SDXL(self.CLIP, text_g=nprompt, text_l=nprompt)

            latent = empty_latent(width=width, height=height)
            print()
            print("Start...")
            samples = ksampler(
                model=MODEL,
                latent=latent,
                seed=seed,
                cfg=cfg,
                steps=steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                poitive_condition=pos,
                negative_condition=neg,
                denoise=denoise,
            )

            out = vae_decode(self.VAE, samples)
            save_images(out, save_names=["ipadapter_test.webp"])
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            torch.cuda.empty_cache()
            gc.collect()
