import numpy as np
from typing import Protocol, Tuple

class Matcher(Protocol):
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Finds point correspondences between two images.
        
        Args:
            image_a: First image (e.g. input)
            image_b: Second image (e.g. reference)
            
        Returns:
            Tuple containing:
            - points_a: (N, 2) array of matched points in image A
            - points_b: (N, 2) array of matched points in image B
            - scores: (N,) array of match scores/confidences
            - matcher_name: string identifying the matcher used
        """
        ...
