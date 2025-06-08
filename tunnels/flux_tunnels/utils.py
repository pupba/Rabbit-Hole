from cores.types import IO
from cores.node_helpers import conditioning_set_values


def flux_guidance(
    conditioning: IO.CONDITIONING, guidance: IO.FLOAT = 3.5
) -> IO.CONDITIONING:
    """
    Apply a guidance scale value to the given conditioning object.

    Args:
        conditioning (IO.CONDITIONING): The input conditioning object to which guidance will be applied.
        guidance (IO.FLOAT, optional): The guidance scale value to set. Default is 3.5.

    Returns:
        IO.CONDITIONING: The conditioning object with the updated guidance value.
    """
    return conditioning_set_values(
        conditioning=conditioning, values={"guidance": guidance}
    )


def flux_disable_guidance(conditioning: IO.CONDITIONING) -> IO.CONDITIONING:
    """
    Disable (remove) the guidance scale from the provided conditioning object.

    Args:
        conditioning (IO.CONDITIONING): The conditioning object to update.

    Returns:
        IO.CONDITIONING: The conditioning object with the guidance scale disabled (set to None).
    """
    return conditioning_set_values(conditioning, {"guidance": None})


def flux_encode(
    clip: IO.CLIP, clip_l: str = "", t5xxl: str = "", guidance: IO.FLOAT = 3.5
) -> IO.CONDITIONING:
    """
    Encode the input prompts using both CLIP and T5-XXL tokenizers and encoders, and return a combined conditioning object with guidance.

    Args:
        clip (IO.CLIP): The CLIP model instance used for tokenization and encoding.
        clip_l (str, optional): The prompt string for the CLIP encoder. Default is an empty string.
        t5xxl (str, optional): The prompt string for the T5-XXL encoder. Default is an empty string.
        guidance (IO.FLOAT, optional): The guidance scale to attach to the conditioning object. Default is 3.5.

    Returns:
        IO.CONDITIONING: The encoded conditioning object, including both CLIP and T5-XXL tokens, with the specified guidance value.
    """
    tokens = clip.tokenize(clip_l)
    tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

    return clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
