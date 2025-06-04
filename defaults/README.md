# default

## Overview

This directory contains the default configuration and environment setup files for the rabbit_hole project.  
Files in this folder are responsible for initializing logging, CUDA settings, and other basic runtime options.

Use this directory to manage environment settings and startup routines for your pipelines.

---

## File Overview

### bootstrap.py
- **Purpose:**  
  Main environment bootstrapper.  
  - Enables CLI argument parsing.
  - Sets up the logger and logging level.
  - Configures CUDA malloc options and warns about unsupported GPUs.
  - Suppresses known non-critical tokenizer warnings for diffusion models:contentReference[oaicite:0]{index=0}.

### cuda_malloc.py
- **Purpose:**  
  Detects and configures CUDA malloc behavior based on GPU compatibility.  
  - Detects GPU model.
  - Enables or disables `cudaMallocAsync` as appropriate for your hardware.
  - Maintains a blacklist for incompatible GPUs.
  - Can be extended for platform-specific GPU detection:contentReference[oaicite:1]{index=1}.

### logger.py
- **Purpose:**  
  Sets up the application's logging system.
  - Intercepts and buffers log output for real-time and historical inspection.
  - Supports switching between stdout and stderr.
  - Allows log flushing, startup warnings, and log formatting customization:contentReference[oaicite:2]{index=2}.

---

## Best Practices

- Keep only basic, reusable, environment-wide configuration logic in this folder.
- Modify these files only if you need to change global settings for the project or adapt to a new environment.

---

## Contribution

- If you add more environment or startup configuration scripts, place them here.
- Please keep the code well-documented and avoid introducing pipeline-specific logic.

---