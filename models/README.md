# models

## Overview

This directory contains all models used for inference in the rabbit_hole project.  
You can store pretrained models, model components, and any additional weights required for your pipelines.

The default structure is provided below, but you are free to add new folders as needed (e.g., for LLMs or other types of models in the future).

---

## Directory Structure
```
models/
├── checkpoints/
├── clip/
├── clip_vision/
├── controlnet/
├── diffusers/
├── diffusion_models/
├── embeddings/
├── gligen/
├── hypernetworks/
├── loras/
├── photomaker/
├── style_models/
├── text_encoders/
├── unet/
├── upscale_models/
├── vae/
└── vae_approx/
```

- Each folder groups related model files or components.
- For example, place your diffusion model checkpoints in `diffusion_models/`, CLIP models in `clip/`, VAE weights in `vae/`, etc.

---

## Adding New Models

If you need to use other models (e.g., LLMs), simply create a new folder in `models/` and organize the files as required.

**Example:**
```
models/
└── llm/
├── my_llm.bin
└── config.json
```
---

## Best Practices

- Use clear, descriptive folder names for any new models or architectures you add.
- Keep each model type separated for clarity and easier management.
- Document any non-standard model files or dependencies in the corresponding folder or main project documentation.

---

