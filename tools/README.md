# tools

## Overview

This directory contains utility modules and helper functions for building tunnels in the rabbit_hole project.
If you need a specific tool, feel free to implement and use it here!  
Contributions are always welcome.

The base tools provided are the ones currently uploaded in this directory.

---

## File Overview

### image_tools.py
- **Image file validation** (`is_allowed_image`)
- **Image mode conversion** (`convert_image_mode_keep`)
- **Image loading as tensor** (`load_image`)
- **Tensor/image conversion utilities** (`tensor2image`, `image2tensor`)
- **High quality image resizing** (`high_quality_resize`)
- **Aspect ratio preserving resize and fit** (`execute_outer_fit`, `execute_inner_fit`, `execute_resize`)

### preview_tools.py
- **Latent preview utilities** for visualization and debugging
- **Latent to image conversion** (e.g., TAESD-based preview, Latent2RGB)
- **Progress callback generator** (`prepare_callback`)

### processor_tools.py
- **Basic image processing functions** for use in tunnels
    - Example: `Processor.canny()` for Canny edge detection (returns 3-channel edge map as tensor)

### sampler_tools.py
- **Sampler and scheduler name retrieval**
    - `get_ksampler_namses()` returns available ksampler names
    - `get_scheduler_names()` returns available scheduler names

---

## Adding New Tools

- Please add new utilities or helper functions as separate modules or as additions to the relevant file.
- Keep each module focused and well-documented for easy maintenance.

---

## Contribution

- Contributions, improvements, and new utility implementations are welcome!
- Please add docstrings and keep functions modular.

---