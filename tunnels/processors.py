import inspect
from typing import List

# tools
from tools.processor_tools import Processor

# cores
from cores.types import IO


def get_processors() -> List[str]:
    """
    Get a list of all public method names in the Processor class.

    Returns:
        List[str]: List of processor method names.
    """
    return [
        name
        for name, _ in inspect.getmembers(Processor, predicate=inspect.isfunction)
        if not name.startswith("__")
    ]


def processor(image: IO.IMAGE, processor_name: str, **kwargs) -> IO.IMAGE:
    """
    Calls a Processor class method by name, passing the image and any extra arguments.

    Args:
        image (IO.IMAGE): The input image.
        processor_name (str): The method name to call on Processor (e.g., 'canny').
        kwargs: Additional keyword arguments for the processor method.

    Returns:
        IO.IMAGE: The processed image.
    """
    p = Processor()
    method = getattr(p, processor_name, None)
    if not callable(method):
        raise ValueError(f"Processor has no method '{processor_name}'.")
    return method(image, **kwargs)
