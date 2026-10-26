"""Local Binary Patterns (uniform, rotation-invariant), multi-scale.

For every pixel, P neighbours are sampled on a circle of radius R and thresholded
against the centre pixel, giving a P-bit binary code. We use the "riu2" variant
(skimage method="uniform"): codes with at most two 0/1 transitions around the
circle are "uniform" and are labelled by their number of 1s (rotation
invariant, P + 1 labels); all non-uniform codes share one extra label. The
descriptor is therefore a histogram of P + 2 bins per (P, R) scale.

Default scales are (8, 1) and (16, 2) -> 10 + 18 = 28 bins. Each per-scale
histogram is L1-normalized, then the concatenation is divided by the number of
scales so the whole vector sums to 1 (histogram intersection remains valid).
The grey image is first rescaled to a fixed longest side so the radii R always
correspond to the same relative scale.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def lbp_histogram(img_bgr, scales=((8, 1), (16, 2))):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    parts = []
    for p, r in scales:
        codes = local_binary_pattern(gray, P=p, R=r, method="uniform")
        hist, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2))
        hist = hist.astype(np.float32)
        parts.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(parts) / len(scales)
