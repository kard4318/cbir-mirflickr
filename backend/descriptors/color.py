"""HSV colour histogram.

The image is converted from BGR to HSV and the three channels are jointly
quantized into a 3-D histogram of H x S x V bins (default 8 x 4 x 4 = 128 bins).
Hue receives the most bins because it carries the chromatic identity of a colour,
while saturation and value mainly encode purity and brightness. OpenCV stores hue
in [0, 180) and S, V in [0, 256), so the bins are uniform over those ranges.

The histogram is L1-normalized (it sums to 1), which makes it independent of the
image size and allows histogram intersection: for two normalized histograms
1 - sum(min(a, b)) = 0.5 * L1(a, b), so intersection is also a true metric.
"""
import cv2
import numpy as np


def color_histogram(img_bgr, bins=(8, 4, 4)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist
