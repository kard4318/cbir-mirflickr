"""Histogram of Oriented Gradients (Dalal & Triggs).

The grey image is resized to 64 x 128 (width x height, the classic detection
window, ignoring the aspect ratio so every image yields the same layout).
Gradients are computed, and each 8 x 8 pixel cell accumulates a 9-bin histogram
of unsigned gradient orientations (0-180 degrees) weighted by gradient magnitude.
Cells are grouped into overlapping 2 x 2 blocks (stride one cell) and every
block vector is normalized with L2-Hys (L2 norm, clip at 0.2, renormalize),
which gives robustness to illumination and contrast changes.
Layout: 8 x 16 cells -> 7 x 15 blocks x (2 x 2 x 9) = 3780 values.
The descriptor captures the global shape/edge layout of the scene.
"""
import cv2
import numpy as np
from skimage.feature import hog


def hog_descriptor(img_bgr, size=(64, 128), orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys"):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, tuple(size), interpolation=cv2.INTER_AREA)
    vec = hog(gray, orientations=orientations, pixels_per_cell=tuple(pixels_per_cell),
              cells_per_block=tuple(cells_per_block), block_norm=block_norm,
              feature_vector=True)
    return vec.astype(np.float32)
