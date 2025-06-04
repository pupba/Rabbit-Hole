# tunnels

## Overview

The **tunnels** directory is the most important component of the rabbit_hole project.

A tunnel is a modular building block for your pipelines—each tunnel implements a single functional unit that can be used as a step in the Executor’s Flow.  
Think of each tunnel as a distinct path within a rabbit hole: you connect them together to define complete inference flows.

All tunnels are designed to be composable and reusable, enabling highly flexible and customizable workflows.

---

## File Overview

### controlnets.py
- `apply_controlnet`:  
  Applies a ControlNet model for conditional image guidance.  
  Updates both positive and negative conditionings with ControlNet hints and strength parameters:contentReference[oaicite:0]{index=0}.

### encoders.py
- `encode`, `encode_SDXL`:  
  Text prompt encoding using CLIP, supporting both standard and SDXL modes.  
- `clip_skip`:  
  Allows skipping layers in the CLIP encoder for fine-tuning:contentReference[oaicite:1]{index=1}.

### images.py
- `convert_pil`, `convert_cv2`:  
  Converts image tensors to PIL or OpenCV (NumPy) images.
- `save_images`, `load_image_from_path`, `load_image_with_mask_from_path`:  
  I/O utilities for saving and loading images and masks.
- `enhance_hint_image`:  
  Preprocessing and resizing hint images (for ControlNet, etc.) with various modes:contentReference[oaicite:2]{index=2}.

### latents.py
- `empty_latent`:  
  Creates an empty latent tensor for initializing diffusion model pipelines.
- `empty_rgb`:  
  Generates a batch of blank RGB images as torch tensors (with optional color):contentReference[oaicite:3]{index=3}.

### load_models.py
- `load_checkpoint`, `load_vae`, `load_lora`, `load_control_net`, `load_text_encoder`, `load_upscale_model`:  
  Utilities for loading checkpoints and various model components (main models, VAE, LoRA, ControlNet, text encoder, upscaling model):contentReference[oaicite:4]{index=4}.

### processors.py
- `get_processors`:  
  Lists all available image processors (by reflection).
- `processor`:  
  Dynamically calls a processor function by name, e.g. for Canny edge detection, etc.:contentReference[oaicite:5]{index=5}.

### samplers.py
- `ksampler`:  
  Diffusion sampling utility for denoising latent tensors, supporting various samplers and progress reporting:contentReference[oaicite:6]{index=6}.

### upscales.py
- `upscale_by_model`:  
  Model-based image upscaling (e.g., Real-ESRGAN), with tiling for memory efficiency.
- `upscale_image`, `upscale_image_by_scale`:  
  General image upscaling using classic interpolation.
- `upscale_latent`, `hires_fix`:  
  Latent space upscaling and high-resolution fix pipeline for diffusion models:contentReference[oaicite:7]{index=7}.

### vaes.py
- `vae_decode`, `vae_encode`:  
  Decodes latent representations to images and encodes images to latent space using a VAE.
- `vae_encode_inpaint`:  
  Mask-aware encoding for inpainting workflows, including mask dilation and blending in latent space:contentReference[oaicite:8]{index=8}.

---

## How to Use

- Import and chain tunnel functions to compose your custom inference flow inside an Executor.
- Each tunnel is designed to handle one specific function—mix and match as needed!

**Example:**
```python
from tunnels.encoders import encode
from tunnels.images import load_image_from_path
from tunnels.samplers import ksampler

# Compose your flow
prompt_cond = encode(clip, "A cat")
image = load_image_from_path("cat.jpg")
result = ksampler(model, image, poitive_condition=prompt_cond, ...)
```

## Contribution
- If you implement a new reusable unit, please add it as a new tunnel file or function!

- Keep each tunnel function focused, well-documented, and modular.