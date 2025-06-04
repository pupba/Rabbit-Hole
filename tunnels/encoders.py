from cores.types import IO


def encode(clip: IO.CLIP, text: str = "") -> IO.CONDITIONING:
    """
    Encodes a text prompt using the provided CLIP text encoder and returns a conditioning tensor.

    Args:
        clip (IO.CLIP): The CLIP text encoder object to use for encoding.
        text (str, optional): The input prompt to encode. Can be a positive or negative prompt. Defaults to "" (empty string).

    Returns:
        IO.CONDITIONING: The conditioning tensor generated from the encoded text prompt.

    Raises:
        RuntimeError: If the provided CLIP object is None or invalid.

    Notes:
        - The `text` argument may be used for either positive or negative prompts.
        - If `clip` is None, this usually means your checkpoint does not contain a valid CLIP/text encoder model.

    Example:
        >>> cond = encode(clip, "A photo of a cat")         # Positive prompt
        >>> cond = encode(clip, "blurry, low quality")      # Negative prompt
    """
    if clip is None:
        raise RuntimeError(
            "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
        )
    tokens = clip.tokenize(
        text,
        tokenizer_options={
            "max_length": 77,
            "truncation": True,
            "padding": "max_length",
        },
    )
    return clip.encode_from_tokens_scheduled(tokens)


def encode_SDXL(
    clip: IO.CLIP,
    width: int = 1024,
    height: int = 1024,
    crop_w: int = 0,
    crop_h: int = 0,
    target_width: int = 1024,
    target_height: int = 1024,
    text_g: str = "",
    text_l: str = "",
) -> IO.CONDITIONING:
    """
    Encodes text prompts using the provided CLIP text encoder with SDXL-specific options.

    Args:
        clip (CLIP): The CLIP text encoder object.
        width (int): Image width to encode with conditioning (default: 1024).
        height (int): Image height to encode with conditioning (default: 1024).
        crop_w (int): Crop width (default: 0).
        crop_h (int): Crop height (default: 0).
        target_width (int): Target width after resizing (default: 1024).
        target_height (int): Target height after resizing (default: 1024).
        text_g (str): Global prompt (can be positive or negative).
        text_l (str): Local prompt (can be positive or negative).

    Returns:
        IO.CONDITIONING: Conditioning tensor for SDXL.

    Notes:
        - Pads or truncates tokens so that "g" and "l" lists are of the same length.

    Example:
        # Positive prompt encoding
        >>> positive = encode_clip_text_sdxl(
        ...     clip, 1024, 1024, 0, 0, 1024, 1024,
        ...     "a photo of a dog", "bright, sunny"
        ... )

        # Negative prompt encoding
        >>> negative = encode_clip_text_sdxl(
        ...     clip, 1024, 1024, 0, 0, 1024, 1024,
        ...     "blurry, low quality", "distorted, low-res"
        ... )
    """
    tokens = clip.tokenize(text_g)
    tokens["l"] = clip.tokenize(text_l)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    conditioning = clip.encode_from_tokens_scheduled(
        tokens,
        add_dict={
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height,
        },
    )
    return conditioning


def clip_skip(clip: IO.CLIP, skip: int = -1) -> IO.CLIP:
    """
    Skips a specified number of layers in the given CLIP encoder.

    This function clones the provided CLIP encoder object and applies the `clip_layer` method
    to skip the desired number of layers.

    Args:
        clip (IO.CLIP): The input CLIP encoder object.
        skip (int, optional): The number of layers to skip. Defaults to -1 (no skip or last layer).

    Returns:
        IO.CLIP: A new CLIP encoder object with the specified number of layers skipped.
    """
    clip = clip.clone()
    clip.clip_layer(skip)
    return clip
