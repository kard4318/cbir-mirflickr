"""Central configuration.

Every path can be overridden with an environment variable, so the same code can
index another image base (e.g. the course "Objects" base) without edits.
Descriptor parameters live here too: the offline indexer writes them to the
artifacts manifest and the online server reads them back from that manifest, which
guarantees identical parameters for indexing and querying.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("CBIR_DATA_DIR", ROOT / "data"))
IMAGES_DIR = Path(os.environ.get("CBIR_IMAGES_DIR", DATA_DIR / "mirflickr"))
META_DIR = Path(os.environ.get("CBIR_META_DIR", IMAGES_DIR / "meta"))
ANNOTATIONS_DIR = Path(os.environ.get("CBIR_ANNOTATIONS_DIR", DATA_DIR / "annotations"))
MPEG7_DIR = Path(os.environ.get("CBIR_MPEG7_DIR", DATA_DIR / "mpeg7"))
ARTIFACTS_DIR = Path(os.environ.get("CBIR_ARTIFACTS_DIR", ROOT / "artifacts"))
FRONTEND_DIR = ROOT / "frontend"
REPORT_DIR = ROOT / "report_assets"

# Default descriptor parameters (copied into the manifest at build time).
DESCRIPTOR_PARAMS = {
    # Every image is rescaled so its longest side is this many pixels before the
    # colour and LBP descriptors are computed (removes the dependence on image size).
    "resize_longest": 256,
    "color": {"bins": [8, 4, 4]},  # H x S x V bins -> 128-dim histogram
    "lbp": {"scales": [[8, 1], [16, 2]]},  # (P, R) pairs -> 10 + 18 = 28 bins
    "hog": {
        "size": [64, 128],  # width, height
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
    },
    "vgg16": {"input_size": 224},  # extracted by scripts/extract_vgg16.py
}

# Metrics offered per descriptor; the first one is the default.
DESCRIPTOR_METRICS = {
    "color": ["intersection", "l1", "l2"],
    "lbp": ["l1", "intersection", "l2"],
    "hog": ["l2", "l1"],
    "ehd": ["l1", "l2"],
    "htd": ["l1", "l2"],
    "vgg16": ["l2"],
}

# Keyword typo tolerance: at fuzziness 100% a tag matches a keyword when their
# rapidfuzz ratio (0-100) is >= this value. At 0% only exact tags match.
# The cutoff is interpolated linearly in between.
FUZZY_MIN_TAG_SIMILARITY = 70
FUZZY_MAX_EXPANSIONS = 10  # tags kept per keyword (best similarity first)

# Descriptors that can be computed from raw pixels (and therefore for uploads).
COMPUTED_DESCRIPTORS = ["color", "lbp", "hog"]
# Descriptors read from the provided MPEG-7 files (database images only).
MPEG7_DESCRIPTORS = ["ehd", "htd"]

# Leaf bucket size: larger leaves trade a few more distance computations for fewer
# Python-level node visits; 512 was fastest on all descriptors at N = 25k.
VPTREE_LEAF_SIZE = 512
