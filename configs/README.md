# configs

## Overview

This directory contains various YAML configuration files for the rabbit_hole project.  
These files are intended to be used for setting up static elements, such as model loading parameters and executor options, typically specified in constructor methods.

By separating configuration from code, it becomes easier to manage, modify, and share environment or model settings for different runs and deployments.

---

## How to Use

To use a specific YAML configuration, pass the path to your configuration file with the `--config-path` argument when running your executor or main script.

**Example:**

```bash
python inference.py --config-path configs/config.yaml
