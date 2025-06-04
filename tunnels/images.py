import os
import cv2
import numpy as np
import logging
import torch

from uuid import uuid4
from typing import Optional, List, Tuple
from PIL import Image

from tools.image_tools import (
    tensor2image,
    image2tensor,
    is_allowed_image,
    convert_image_mode_keep,
    execute_resize,
    execute_outer_fit,
    execute_inner_fit,
)
from cores.types import IO


# Convert


def convert_pil(images: IO.IMAGE) -> List[Image.Image]:
    """
    Converts a batch of tensor images to a list of PIL.Image objects.

    Args:
        images (IO.IMAGE): Batch of images as a tensor.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects, one for each input image.
    """
    return tensor2image(tensor_images=images, img_type="pil")


def convert_cv2(images: IO.IMAGE) -> List[np.ndarray]:
    """
    Converts a batch of tensor images to a list of NumPy arrays in OpenCV (BGR) format.

    Args:
        images (IO.IMAGE): Batch of images as a tensor.

    Returns:
        List[np.ndarray]: List of images as NumPy arrays in BGR channel order.
    """
    return tensor2image(tensor_images=images, img_type="cv2")


# Save


def save_images(
    images: IO.IMAGE,
    image_type: str = "pil",
    save_dir: Optional[str] = None,
    save_names: Optional[List[str]] = None,
) -> None:
    """
    Saves a batch of images to disk, supporting PNG, JPG, JPEG, and WEBP formats.

    Args:
        images (IO.IMAGE): Batch of images as a tensor (B, H, W, 3) or (B, H, W, 4).
        image_type (str): Type to convert tensor images to before saving ("pil" or "cv2"). Default is "pil".
        save_dir (Optional[str]): Directory to save images. If None, uses the current working directory.
        save_names (Optional[List[str]]): List of filenames to use. If None or shorter than batch size, generates random UUID names.

    Notes:
        - Only files with .png, .jpg, .jpeg, .webp extensions will be saved.
        - If save_names is shorter or longer than the batch, names will be auto-adjusted.
        - PNGs are saved with RGBA, JPG/JPEG with RGB, WEBP with mode-appropriate settings.
    """
    images = tensor2image(tensor_images=images, img_type=image_type)
    # Name
    if save_names is None:  # if no names
        save_names = [f"{str(uuid4())}.png" for _ in range(len(images))]
    else:
        diff = len(images) - len(save_names)
        if diff > 0:  # length : save_name < images
            save_names += [str(uuid4()) for _ in range(diff)]
        elif diff < 0:  # length : save_name > images
            save_names = save_names[: len(images)]
    # zip names and images
    for image, save_name in zip(images, save_names):
        ext = os.path.splitext(save_name)[1].lower()

        if not is_allowed_image(save_name):
            logging.error(
                f"{ext} is not allowed. Please this extensions.[.png .jpg .jpeg .webp]"
            )
            continue
        save_path = os.path.join(save_dir or os.getcwd(), save_name)
        if ext == ".png":  # RGBA
            image = convert_image_mode_keep(image=image, mode="RGBA")
            if isinstance(image, Image.Image):
                image.save(save_path, format="PNG")
            else:
                cv2.imwrite(save_path, image, [cv2.IMWRITE_WEBP_QUALITY, 100])
        elif ext in {".jpg", ".jpeg"}:  # RGB
            image = convert_image_mode_keep(image=image, mode="RGB")
            if isinstance(image, Image.Image):
                image.save(save_path, format="JPG")
            else:
                cv2.imwrite(save_path, image)
        elif ext == ".webp":  # .webp is all support
            if isinstance(image, Image.Image):
                image.save(
                    save_path,
                    format="WEBP",
                    lossless=False if image.mode == "RGB" else True,
                )
            else:
                cv2.imwrite(save_path, image, [cv2.IMWRITE_WEBP_QUALITY, 100])


# Load
def load_image_from_path(image_path: str) -> IO.IMAGE:
    """
    Loads an image from the given file path and converts it to a normalized torch tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        IO.IMAGE: Image tensor of shape (1, H, W, 3), float32, values in [0, 1].

    Raises:
        FileNotFoundError: If the image file does not exist.
        OSError: If the image file cannot be opened or read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        t_img, _ = image2tensor(image)
        return t_img
    except OSError as e:
        raise OSError(f"Failed to load image : {image_path}") from e


def load_image_with_mask_from_path(
    image_path: str, mask_path: str
) -> Tuple[IO.IMAGE, IO.MASK]:
    """
    Loads an image and its mask from the given file paths and converts them to normalized torch tensors.

    Args:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file (should be single channel or convertible to grayscale).

    Returns:
        Tuple[IO.IMAGE, IO.MASK]:
            - IO.IMAGE: Image tensor of shape (1, H, W, 3), float32, values in [0, 1].
            - IO.MASK: Mask tensor of shape (1, H, W), float32, values in [0, 1].

    Raises:
        FileNotFoundError: If the image or mask file does not exist.
        OSError: If the image or mask file cannot be opened or read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask
        t_img, t_mask = image2tensor(image, mask)
        return t_img, t_mask
    except OSError as e:
        raise OSError(f"Failed to load image or mask: {image_path}, {mask_path}") from e


def get_image_sizes(images: IO.IMAGE) -> List[Tuple[int, int]]:
    """
    Returns the (height, width) of each image in a batch of images.

    Args:
        images (IO.IMAGE): Batch of images as a torch.Tensor (B, H, W, C) or iterable of images.

    Returns:
        List[Tuple[int, int]]: List of (height, width) tuples for each image in the batch.
    """
    return [(image.shape[0], image.shape[1]) for image in images]


# Resize
def enhance_hint_image(
    image: IO.IMAGE,
    width: int = 512,
    height: int = 512,
    resize_mode: str = "JUST RESIZE",
) -> IO.IMAGE:
    """
    Enhances and resizes a batch of hint images for use in ControlNet or similar preprocessors.

    Args:
        image (IO.IMAGE): input image as a torch.Tensor (1, H, W, C), float32, [0, 1].
        width (int): Target width of the output images.
        height (int): Target height of the output images.
        resize_mode (str): One of "Just Resize" (simple resize to target size),
            "Resize and Fill" (outer fit with border filling),
            or "Crop and Resize" (inner fit with center crop).

    Returns:
        IO.IMAGE: image as torch.Tensor (1, H, W, C), float32, [0, 1].

    Raises:
        ValueError: If an invalid resize_mode is provided.

    Notes:
        - Each image is first converted to uint8, then resized according to the selected mode,
          and finally normalized back to float32 in [0, 1].
        - The function expects images in (B, H, W, C) format and returns the same.
    """
    outs = []
    hint_images = [img.detach().cpu().numpy() for img in image]
    for np_hint_image in hint_images:
        np_hint_image = (
            (np_hint_image * 255.0).astype(np.uint8)
            if np_hint_image.max() <= 1.0
            else np_hint_image.astype(np.uint8)
        )
        if resize_mode.upper() == "JUST RESIZE":
            np_hint_image = execute_resize(np_hint_image, width, height)
        elif resize_mode.upper() == "RESIZE AND FILL":
            np_hint_image = execute_outer_fit(np_hint_image, width, height)
        elif resize_mode.upper() == "CROP AND RESIZE":
            np_hint_image = execute_inner_fit(np_hint_image, width, height)
        else:
            raise ValueError(
                "resize_mode must be one of: 'JUST RESIZE', 'RESIZE AND FILL', 'CROP AND RESIZE'"
            )
        outs.append(torch.from_numpy(np_hint_image.astype(np.float32) / 255.0))
    return torch.stack(outs, dim=0)
