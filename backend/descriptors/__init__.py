"""Descriptor computation shared by the indexer (offline) and the API (online)."""
from .color import color_histogram
from .hog import hog_descriptor
from .lbp import lbp_histogram
from .preprocess import load_image_file, resize_longest


def compute_descriptors(img_bgr, params):
    """Compute every pixel-based descriptor of one BGR image with the given params.

    Both build_index.py and the API call this function with the parameters
    stored in the artifacts manifest, so offline and online vectors are identical.
    """
    small = resize_longest(img_bgr, params["resize_longest"])
    return {
        "color": color_histogram(small, bins=params["color"]["bins"]),
        "lbp": lbp_histogram(small, scales=params["lbp"]["scales"]),
        # HOG resizes to its own fixed window, from the original image.
        "hog": hog_descriptor(img_bgr, **params["hog"]),
    }


def compute_file(job):
    """Multiprocessing worker: job = (path, params) -> (descriptors, (h, w)) or None."""
    path, params = job
    img = load_image_file(path)
    if img is None:
        return None
    return compute_descriptors(img, params), img.shape[:2]
