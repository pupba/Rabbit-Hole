# Future Update
import cv2
import numpy as np

# tools
from tools.image_tools import image2tensor

# cores
from cores.types import IO


class Processor:
    # Canny
    def canny(
        self, images: IO, low_threshold: int = 100, high_threshold: int = 200
    ) -> IO.IMAGE:
        """
        Applies Canny edge detection to a single IO.IMAGE tensor and returns a 3-channel edge map as IO.IMAGE.

        Args:
            image (torch.Tensor or np.ndarray): Input image tensor of shape (1, H, W, 3) or (H, W, 3), float32, [0,1].
            low_threshold (int): Lower threshold for Canny edge detector.
            high_threshold (int): Higher threshold for Canny edge detector.

        Returns:
            torch.Tensor: Edge image tensor of shape (1, H, W, 3), float32, values in [0,1].
        """
        img = images.detach().cpu().numpy()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]  # (H, W, 3)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, low_threshold, high_threshold)
        t_image, _ = image2tensor(edge)
        return t_image
