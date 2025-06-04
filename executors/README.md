# executors

## Overview

This directory contains custom **executors** for the rabbit_hole project.

An **executor** is a customizable inference unit built by composing one or more tunnels.  
It defines a fixed inference flow, which you can freely design according to your needs by combining different tunnels and logic.

Each executor represents a single, well-defined inference flow.

---

## Basic Template

A typical executor follows this structure:

```python
class <YourExecutorsName>:
    def __init__(self):
        # Initialization logic, including loading models, tunnels, etc.
        pass

    def __call__(self, *args, **kwargs):
        # Inference flow logic
        pass
