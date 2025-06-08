from tunnels.load_models import load_checkpoint, load_vae, load_unet
from tunnels.load_clip import dual_clip_loader
from tunnels.encoders import encode
from tunnels.flux_tunnels.utils import flux_guidance
from tunnels.latents import empty_sd3_latent_image, empty_latent
from tunnels.samplers import (
    ksampler,
    basic_guider,
    basic_scheduler,
    random_noise,
    split_sigmas,
    sampler_custom_advanced,
    ksampler_select,
)
from tunnels.vaes import vae_decode
from tunnels.images import save_images


class FluxSimpleExectuor:
    def __init__(self):
        self.MODEL, self.CLIP, self.VAE = load_checkpoint(
            "flux1-dev-scaled_fp8.safetensors"
        )

    def __call__(self):
        pos = encode(
            self.CLIP,
            "cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open placing a fancy black forest cake with candles on top of a dinner table of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere there are paintings on the walls",
        )
        neg = encode(
            self.CLIP,
            "",
        )
        pos = flux_guidance(pos, 3.5)
        latent = empty_sd3_latent_image(1024, 1024)
        samples = ksampler(
            poitive_condition=pos,
            negative_condition=neg,
            model=self.MODEL,
            latent=latent,
            seed=12412521,
            cfg=1.0,
            steps=20,
        )

        out = vae_decode(vae=self.VAE, samples=samples)
        save_images(out)


class FluxUnderVRAM12GB:
    def __init__(self):
        self.MODEL = load_unet(
            "flux1-dev-scaled_fp8.safetensors", weight_dtype="fp8_e4m3fn"
        )
        self.CLIP = dual_clip_loader(
            "t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "flux"
        )
        self.VAE = load_vae("ae.safetensors")

    def __call__(self):
        cond = encode(
            self.CLIP,
            "a portrait of a duc who has a sign around its nec which says 'Flux under 12GB VRAM' in the style of vincent van gogh, heavy brushwork, impasto, painted with a palate knife",
        )
        guider = basic_guider(self.MODEL, cond)

        sampler = ksampler_select("euler")

        noise = random_noise(12421412312)

        sigmas = basic_scheduler(self.MODEL)
        sigmas, _ = split_sigmas(sigmas=sigmas, step=20)

        latent = empty_latent(1024, 576)

        out, _ = sampler_custom_advanced(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )

        out = vae_decode(self.VAE, out)

        save_images(out)
