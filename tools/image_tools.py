import numpy as np
import torch
import os
import cv2

from typing import Tuple, List, Optional
from PIL import Image, ImageOps

# cores
from cores.types import IO

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def is_allowed_image(filename: str) -> bool:
    """
    Checks if a given filename has an allowed image file extension.

    Args:
        filename (str): The name or path of the file.

    Returns:
        bool: True if the file extension is in the allowed set (".png", ".jpg", ".jpeg", ".webp"), otherwise False.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def convert_image_mode_keep(
    image: Image.Image | np.ndarray, mode: str
) -> Image.Image | np.ndarray:
    """
    Convert image to the specified mode (RGB or RGBA) while preserving input type (PIL or OpenCV).

    Args:
        image: PIL.Image.Image or numpy.ndarray (OpenCV image)
        mode: Target mode ("RGB" or "RGBA")

    Returns:
        Converted image in the same type as input

    Raises:
        TypeError: If input is not PIL.Image or numpy.ndarray
        ValueError: If mode is not "RGB" or "RGBA"
    """
    if mode not in {"RGB", "RGBA"}:
        raise ValueError('Only "RGB" and "RGBA" modes are supported.')

    # Determine input type
    if isinstance(image, Image.Image):
        input_type = "pil"
        image_pil = image
    elif isinstance(image, np.ndarray):
        input_type = "cv2"

        # Convert OpenCV BGR/BGRA to RGB/RGBA for PIL
        if image.ndim == 3 and image.shape[2] == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif image.ndim == 3 and image.shape[2] == 4:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            raise ValueError("Unsupported OpenCV image format.")
    else:
        raise TypeError("Input must be a PIL.Image or numpy.ndarray (OpenCV image).")

    # Convert to target mode if necessary
    if image_pil.mode != mode:
        image_pil = image_pil.convert(mode)

    # Return result in original format
    if input_type == "pil":
        return image_pil
    else:
        # Convert back from PIL to OpenCV BGR/BGRA
        np_img = np.array(image_pil)
        if mode == "RGB":
            return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        elif mode == "RGBA":
            return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)


def load_image(image_path: str) -> Tuple[IO.IMAGE, Tuple[int, int]]:
    """
    Loads an image from the given file path, converts it to RGB, and returns it as a normalized float tensor.

    Args:
        image_path (str): The file path to the image.

    Returns:
        Tuple[IO.IMAGE, Tuple[int, int]]:
            - IO.IMAGE: The loaded image as a 4D torch tensor (1, H, W, 3), normalized to [0, 1], on CPU.
            - Tuple[int, int]: The original image (width, height) as a tuple.

    Raises:
        FileNotFoundError: If the image file does not exist or cannot be opened.
        OSError: If the file is not a valid image.

    Notes:
        - The image is loaded as RGB and auto-rotated according to EXIF orientation.
        - The returned tensor has shape (1, H, W, 3) and dtype float32.

    Example:
        >>> tensor_img, (width, height) = load_image("input.jpg")
    """
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    np_img = np.array(img).astype(np.float32) / 255.0  # (H,W,3), float32
    tensor_img = torch.from_numpy(np_img)[None, ...].to("cpu")  # (1,H,W,3)

    return (tensor_img, (img.width, img.height))


def tensor2image(
    tensor_images: IO.IMAGE, img_type: str = "pil"
) -> List[Image.Image | np.ndarray]:
    """
    Converts a batch of torch image tensors to a list of PIL Images or NumPy arrays.

    Args:
        images (torch.Tensor): (B, H, W, 3) or (B, C, H, W) float32, [0,1]
        output_format (str): 'pil' for PIL.Image.Image, 'cv' for OpenCV ndarray (BGR)

    Returns:
        List[Image.Image | np.ndarray]: List of converted images.
    """
    results = []
    if tensor_images.ndim == 4 and tensor_images.shape[-1] == 3:  # (B, H, W, 3)
        batch_iter = tensor_images
    elif tensor_images.ndim == 4 and tensor_images.shape[1] == 3:  # (B, 3, H, W)
        batch_iter = tensor_images.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported image tensor shape: {tensor_images.shape}")

    for image in batch_iter:
        # image : (H,W,3), float32, [0,1]
        arr = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if img_type == "pil":
            img_obj = Image.fromarray(arr, mode="RGB")
        elif img_type == "cv":
            img_obj = arr[..., ::-1]  # RGB -> BGR
        else:
            raise ValueError("Unknown output_type,support 'cv' or 'pil'")
        results.append(img_obj)
    return results


def image2tensor(
    image: Image.Image | np.ndarray, mask: Optional[Image.Image | np.ndarray] = None
) -> Tuple[IO.IMAGE, Optional[IO.MASK]]:
    """
    Converts a PIL.Image or numpy.ndarray to a torch.Tensor (float32, [0,1], shape (1, H, W, 3)),
    and (optionally) a mask to a torch.Tensor (float32, [0,1], shape (1, H, W)).

    Args:
        image: Input image, RGB/RGBA/grayscale, PIL.Image.Image or np.ndarray.
        mask:  Optional. Separate mask image, single channel, PIL.Image.Image or np.ndarray.

    Returns:
        Tuple:
            - image_tensor: (1, H, W, 3) float32, [0,1]
            - mask_tensor:  (1, H, W) float32, [0,1] or None
    """
    # 1. PIL -> RGB, exif transpose
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB" and image.mode != "RGBA":
            image = image.convert("RGB")
        arr = np.array(image).astype(np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32) / 255.0 if image.dtype != np.float32 else image
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
    else:
        raise TypeError("Input image must be PIL.Image or np.ndarray")

    # 2. (H,W,3) -> (1,H,W,3)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    img_tensor = torch.from_numpy(arr).float()

    # 3. Mask Processing
    mask_tensor = None
    if mask is not None:
        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask)
        elif isinstance(mask, np.ndarray):
            mask_arr = mask
        else:
            raise TypeError("Mask must be PIL.Image or np.ndarray")

        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]  # use first channel
        mask_arr = (
            mask_arr.astype(np.float32) / 255.0
            if mask_arr.dtype != np.float32
            else mask_arr
        )
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()
    return img_tensor, mask_tensor


# Resize
def high_quality_resize(x: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resizes an image (optionally with alpha channel) to the target size, using high-quality interpolation.

    Handles special cases such as binary images and inpaint masks.
    If the image includes an alpha channel, it is resized and re-attached.

    Args:
        x (np.ndarray): Input image array of shape (H, W, 3) or (H, W, 4), dtype uint8.
        size (Tuple[int, int]): Target size as (width, height).

    Returns:
        np.ndarray: Resized image array of shape (target_height, target_width, C), dtype uint8.
    """
    inpaint_mask = None
    if x.ndim == 3 and x.shape[2] == 4:
        inpaint_mask = x[:, :, 3]
        x = x[:, :, :3]

    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
        new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
        unique_color_count = len(np.unique(x.reshape(-1, x.shape[2]), axis=0))
        is_one_pixel_edge = False
        is_binary = False
        if unique_color_count == 2:
            is_binary = np.min(x) < 16 and np.max(x) > 240
            # (이하 binary 처리 생략/필요시 추가)
        if 2 < unique_color_count < 200:
            interpolation = cv2.INTER_NEAREST
        elif new_size_is_smaller:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        y = cv2.resize(x, size, interpolation=interpolation)
        if inpaint_mask is not None:
            inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)
        # (이하 binary/mask logic 생략, 필요시 추가)
    else:
        y = x
    if inpaint_mask is not None:
        inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
        inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
        y = np.concatenate([y, inpaint_mask], axis=2)
    return y


# Helper: outer fit
def execute_outer_fit(detected_map: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Resizes and fits the input image to the target size by preserving aspect ratio and padding the borders.
    The empty space is filled with the median border color for seamless composition.

    Args:
        detected_map (np.ndarray): Input image array of shape (H, W, C).
        w (int): Target width.
        h (int): Target height.

    Returns:
        np.ndarray: Output image array of shape (h, w, C).
    """
    old_h, old_w, _ = detected_map.shape
    k = min(float(h) / old_h, float(w) / old_w)
    high_quality_border_color = np.median(
        np.concatenate(
            [
                detected_map[0, :, :],
                detected_map[-1, :, :],
                detected_map[:, 0, :],
                detected_map[:, -1, :],
            ],
            axis=0,
        ),
        axis=0,
    ).astype(detected_map.dtype)
    high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
    resized = high_quality_resize(
        detected_map, (int(round(old_w * k)), int(round(old_h * k)))
    )
    new_h, new_w, _ = resized.shape
    pad_h = max(0, (h - new_h) // 2)
    pad_w = max(0, (w - new_w) // 2)
    high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return high_quality_background


# Helper: inner fit
def execute_inner_fit(detected_map: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Resizes and fits the input image to the target size by preserving aspect ratio and center-cropping any overflow.

    Args:
        detected_map (np.ndarray): Input image array of shape (H, W, C).
        w (int): Target width.
        h (int): Target height.

    Returns:
        np.ndarray: Output image array of shape (h, w, C).
    """
    old_h, old_w, _ = detected_map.shape
    k = max(float(h) / old_h, float(w) / old_w)
    resized = high_quality_resize(
        detected_map, (int(round(old_w * k)), int(round(old_h * k)))
    )
    new_h, new_w, _ = resized.shape
    pad_h = max(0, (new_h - h) // 2)
    pad_w = max(0, (new_w - w) // 2)
    return resized[pad_h : pad_h + h, pad_w : pad_w + w]


# Helper: resize (just scale to target size)
def execute_resize(detected_map: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Directly resizes the input image to the specified width and height using high-quality interpolation.

    Args:
        detected_map (np.ndarray): Input image array of shape (H, W, C).
        w (int): Target width.
        h (int): Target height.

    Returns:
        np.ndarray: Output image array of shape (h, w, C).
    """
    return high_quality_resize(detected_map, (w, h))
