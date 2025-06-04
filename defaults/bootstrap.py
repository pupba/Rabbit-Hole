import cores.options

cores.options.enable_args_parsing()
from cores.cli_args_utils import args

# logger
from defaults.logger import setup_logger

"""
Suppress the warning message:
"Token indices sequence length is longer than the specified maximum sequence length for this model (88 > 77).
Running this sequence through the model will result in indexing errors"

This warning can safely be ignored in SDXL/ComfyUI pipelines:
The entire prompt is passed to the tokenizer at once (triggering the warning),
but is then split internally, so you will not get actual indexing errors.
Additional words at the end of the prompt will still affect the output as expected.
"""
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

import defaults.cuda_malloc as cuda_malloc  # cuda malloc
import cores.model_management_utils  # setting


def cuda_malloc_warning():
    device = cores.model_management_utils.get_torch_device()
    device_name = cores.model_management_utils.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning(
                '\nWARNING: this card most likely does not support cuda-malloc, if you get "CUDA error" please run ComfyUI with: --disable-cuda-malloc\n'
            )


cuda_malloc_warning()
