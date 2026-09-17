import cv2
import numpy as np

class ImagePyramid:
    """A lightweight image pyramid for multi-scale processing."""
    def __init__(self, base_image: np.ndarray, levels: list[int] = [0, 1, 2]):
        self.levels = sorted(levels)
        self.images = {}
        self._build_pyramid(base_image)
        
    def _build_pyramid(self, base_image: np.ndarray):
        """Builds the downsampled versions of the image."""
        self.images[0] = base_image.copy()
        
        current_image = base_image
        for level in range(1, max(self.levels) + 1):
            # Downsample by 2x
            # Gaussian blur first to avoid aliasing
            blurred = cv2.GaussianBlur(current_image, (5, 5), 0)
            downsampled = cv2.resize(blurred, (current_image.shape[1] // 2, current_image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            
            if level in self.levels:
                self.images[level] = downsampled
                
            current_image = downsampled

    def get_level(self, level: int) -> np.ndarray:
        if level not in self.images:
            raise ValueError(f"Level {level} not found in pyramid.")
        return self.images[level]
        
    def get_scale_factor_for_level(self, level: int) -> float:
        """Returns the scale factor compared to level 0 (e.g. level 1 -> 0.5)"""
        return 1.0 / (2 ** level)
