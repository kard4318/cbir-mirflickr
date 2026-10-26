"""Parsers for the MPEG-7 descriptors provided with MIRFLICKR-25k.

Each image id N has a file N.txt with one float per line and no header.

Edge Histogram Descriptor (EHD), 150 values. The image is split into 4 x 4
sub-images, and each sub-image's small blocks are classified into 5 edge types
(vertical, horizontal, 45 deg, 135 deg, non-directional). This gives 80 local
bins. 5 global bins and 65 semi-global bins (rows, columns and 2 x 2 groupings
of sub-images) follow, for 80 + 5 + 65 = 150. The global bins are visibly
weighted (larger values), as in the usual EHD matching scheme. The values are
used as provided and compared with the L1 distance, the standard EHD matching
measure.

Homogeneous Texture Descriptor (HTD), 43 values. The standard MPEG-7 HTD has 62
values (mean, standard deviation, 30 Gabor-channel energies and 30 energy
deviations). These files contain 43, and the exact layout is not documented with
the dataset, so the vector is treated as an opaque 43-dimensional texture
signature. Because its dimensions have different ranges, each dimension is
z-score normalized with the database mean and standard deviation (computed at
indexing time) before L1/L2 comparison.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MPEG7_FOLDERS = {"ehd": ("edgehistogram", 150), "htd": ("homogeneoustexture", 43)}


def _read_vector(path):
    with open(path, "r") as fh:
        return np.array(fh.read().split(), dtype=np.float32)


def load_mpeg7(mpeg7_dir, name, image_ids, workers=16):
    """Return an (len(image_ids), dim) float32 matrix for descriptor `name`."""
    folder, dim = MPEG7_FOLDERS[name]
    paths = [mpeg7_dir / folder / f"{i}.txt" for i in image_ids]
    # Many small files: threads overlap the I/O latency.
    with ThreadPoolExecutor(workers) as pool:
        vectors = list(pool.map(_read_vector, paths, chunksize=64))
    out = np.zeros((len(paths), dim), dtype=np.float32)
    for row, vec in enumerate(vectors):
        if vec.size != dim:
            raise ValueError(f"{paths[row]}: expected {dim} values, got {vec.size}")
        out[row] = vec
    return out
