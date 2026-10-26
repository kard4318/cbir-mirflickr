"""Image loading and the preprocessing shared by the offline and online pipelines."""
import cv2
import numpy as np


def load_image_file(path):
    """Read an image from disk as a BGR uint8 array (None if unreadable)."""
    # np.fromfile + imdecode handles non-ASCII Windows paths, unlike cv2.imread.
    buf = np.fromfile(str(path), dtype=np.uint8)
    return decode_image_bytes(buf)


def decode_image_bytes(data):
    """Decode encoded image bytes (JPEG, PNG...) into a BGR uint8 array."""
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def resize_longest(img, longest):
    """Rescale an image so its longest side equals `longest` pixels (aspect ratio kept)."""
    h, w = img.shape[:2]
    scale = longest / max(h, w)
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, new_size, interpolation=interp)
