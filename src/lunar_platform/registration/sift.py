import cv2
import numpy as np
from typing import Tuple
from .base import Matcher

class SiftMatcher(Matcher):
    def __init__(self, nfeatures: int = 5000, ratio_threshold: float = 0.75):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.ratio_threshold = ratio_threshold
        
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if len(image_a.shape) > 2:
            image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        if len(image_b.shape) > 2:
            image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
            
        kp_a, desc_a = self.sift.detectAndCompute(image_a, None)
        kp_b, desc_b = self.sift.detectAndCompute(image_b, None)
        
        if desc_a is None or desc_b is None:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), "sift"
            
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(desc_a, desc_b, k=2)
        except Exception:
            # Fallback for very small number of descriptors
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc_a, desc_b, k=2)
            
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
                
        if not good_matches:
             return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), "sift"
             
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches])
        
        # SIFT doesn't natively give a "score" per match that is a probability,
        # but we can use 1 / (1 + distance) as a pseudo-score
        scores = np.float32([1.0 / (1.0 + m.distance) for m in good_matches])
        
        return pts_a, pts_b, scores, "sift"
