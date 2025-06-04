from typing import List

# cores
from cores.samplers import KSAMPLER_NAMES, SCHEDULER_NAMES


def get_ksampler_namses() -> List[str]:
    return KSAMPLER_NAMES


def get_scheduler_names() -> List[str]:
    return SCHEDULER_NAMES
