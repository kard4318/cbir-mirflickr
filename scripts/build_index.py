"""OFFLINE phase: compute/load descriptors, fill SQLite, build the VP-Trees, save to disk.

Usage (from the project root, venv active):
    python -m scripts.build_index --limit 1000 --out artifacts_test   # quick test
    python -m scripts.build_index                                       # full 25k

Outputs in --out (default artifacts/):
    manifest.json              descriptor parameters, dims, metrics, dataset paths
    ids.npy                    image id of each descriptor row
    features_<name>.npy        float32 descriptor matrix (one row per image)
    vptree_<name>_<metric>.joblib
    stats.json                 HTD z-score parameters, distance scales, build times
    cbir.sqlite                images, tags, metadata, labels
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from backend import config
from backend.db import database, metadata
from backend.descriptors import compute_file
from backend.descriptors.mpeg7 import load_mpeg7
from backend.index import VPTree, distances

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir, limit=None):
    """Return [(image_id, path)] sorted by id.

    MIRFLICKR files are named imN.jpg -> id N. For another base without a number
    in the file name, ids are assigned sequentially in alphabetical order.
    """
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    numbered = [(int(m.group(1)), p) for p in files if (m := re.search(r"(\d+)$", p.stem))]
    if len(numbered) == len(files) and len({i for i, _ in numbered}) == len(files):
        items = sorted(numbered)
    else:
        items = list(enumerate(files, start=1))
    return items[:limit] if limit else items


def _init_worker():
    # One OpenCV thread per process: parallelism comes from the process pool, and
    # per-process thread pools otherwise multiply memory use (OOM with 16 cores).
    cv2.setNumThreads(1)


def extract_pixel_descriptors(items, params, workers):
    jobs = [(str(p), params) for _, p in items]
    results = []
    with mp.Pool(workers, initializer=_init_worker, maxtasksperchild=2000) as pool:
        for res in tqdm(pool.imap(compute_file, jobs, chunksize=16), total=len(jobs),
                        desc=f"descriptors ({workers} processes)"):
            results.append(res)
    return results


def read_metadata(meta_dir, ids, workers=16):
    def one(i):
        return (metadata.read_tags(meta_dir, i), metadata.read_exif(meta_dir, i),
                metadata.read_license(meta_dir, i))
    with ThreadPoolExecutor(workers) as pool:
        return list(tqdm(pool.map(one, ids), total=len(ids), desc="metadata"))


def distance_scale(X, metric, rng, n_pairs=2000):
    """Median distance between random pairs: used to normalize distances before combining."""
    a = rng.integers(0, len(X), n_pairs)
    b = rng.integers(0, len(X), n_pairs)
    d = np.array([distances(X[i], X[j][None, :], metric)[0] for i, j in zip(a, b) if i != j])
    return float(np.median(d)) or 1.0


def build_trees(features, out, stats):
    """Build and save one VP-Tree per (descriptor, metric); record build times and distance scales."""
    rng = np.random.default_rng(0)
    for name, X in features.items():
        for metric in config.DESCRIPTOR_METRICS[name]:
            t0 = time.perf_counter()
            tree = VPTree(X, metric, leaf_size=config.VPTREE_LEAF_SIZE).build()
            build = time.perf_counter() - t0
            tree.save(out / f"vptree_{name}_{metric}.joblib")
            stats["build_seconds"][f"{name}/{metric}"] = build
            stats["scales"][f"{name}/{metric}"] = distance_scale(X, metric, rng)
            print(f"  VP-Tree {name}/{metric}: {build:.2f}s, {len(tree.vp)} nodes")


def rebuild_trees_only(out):
    """Rebuild the VP-Trees from the saved descriptor files (no image re-extraction)."""
    manifest = json.loads((out / "manifest.json").read_text())
    stats = json.loads((out / "stats.json").read_text())
    features = {name: np.load(out / info.get("file", f"features_{name}.npy"))
                for name, info in manifest["descriptors"].items()}
    build_trees(features, out, stats)
    stats["vptree_leaf_size"] = config.VPTREE_LEAF_SIZE
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"VP-Trees rebuilt in {out} (leaf size {config.VPTREE_LEAF_SIZE})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--meta-dir", type=Path, default=config.META_DIR)
    ap.add_argument("--annotations-dir", type=Path, default=config.ANNOTATIONS_DIR)
    ap.add_argument("--mpeg7-dir", type=Path, default=config.MPEG7_DIR)
    ap.add_argument("--no-mpeg7", action="store_true", help="skip EHD/HTD (e.g. for another image base)")
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only index the first N images")
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--trees-only", action="store_true",
                    help="only rebuild the VP-Trees from existing descriptor files in --out")
    args = ap.parse_args()
    if args.trees_only:
        return rebuild_trees_only(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    params = config.DESCRIPTOR_PARAMS
    t_start = time.perf_counter()

    # 1. Images and pixel-based descriptors (multiprocessing).
    items = list_images(args.images_dir, args.limit)
    print(f"{len(items)} images in {args.images_dir}")
    results = extract_pixel_descriptors(items, params, args.workers)
    ok = [(item, res) for item, res in zip(items, results) if res is not None]
    failed = [str(p) for (_, p), res in zip(items, results) if res is None]
    if failed:
        print(f"WARNING: {len(failed)} unreadable images skipped, e.g. {failed[:3]}")
    ids = np.array([i for (i, _), _ in ok], dtype=np.int64)
    features = {name: np.stack([res[0][name] for _, res in ok]).astype(np.float32)
                for name in config.COMPUTED_DESCRIPTORS}
    t_extract = time.perf_counter() - t_start

    # 2. Provided MPEG-7 descriptors.
    stats = {"build_seconds": {}, "scales": {}, "vptree_leaf_size": config.VPTREE_LEAF_SIZE}
    use_mpeg7 = not args.no_mpeg7 and args.mpeg7_dir.exists()
    if use_mpeg7:
        t0 = time.perf_counter()
        for name in config.MPEG7_DESCRIPTORS:
            features[name] = load_mpeg7(args.mpeg7_dir, name, ids.tolist())
        mean, std = features["htd"].mean(axis=0), features["htd"].std(axis=0)
        std[std == 0] = 1.0
        features["htd"] = ((features["htd"] - mean) / std).astype(np.float32)
        stats["htd_zscore"] = {"mean": mean.tolist(), "std": std.tolist()}
        print(f"MPEG-7 descriptors loaded in {time.perf_counter() - t0:.1f}s")

    np.save(args.out / "ids.npy", ids)
    for name, X in features.items():
        np.save(args.out / f"features_{name}.npy", X)
        print(f"  {name:6s} {X.shape[1]:5d} dims")

    # 3. SQLite: images, tags, metadata, labels.
    t0 = time.perf_counter()
    has_meta = args.meta_dir.exists()
    meta = read_metadata(args.meta_dir, ids.tolist()) if has_meta else [([], {}, {})] * len(ids)
    conn = database.create(args.out / "cbir.sqlite")
    records = []
    for ((image_id, path), (_, (h, w))), (tags, exif, lic) in zip(ok, meta):
        rel = path.relative_to(args.images_dir).as_posix()
        records.append({"id": int(image_id), "filename": rel, "url": f"/images/{rel}",
                        "width": int(w), "height": int(h), "tags": tags, "exif": exif, "license": lic})
    database.insert_images(conn, records)
    if args.annotations_dir.exists():
        database.insert_labels(conn, metadata.read_annotations(args.annotations_dir), set(ids.tolist()))
    n_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    n_labels = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    conn.close()
    print(f"SQLite: {len(records)} images, {n_tags} tag rows, {n_labels} label rows "
          f"({time.perf_counter() - t0:.1f}s)")

    # 4. VP-Trees (one per descriptor and metric) + distance scales.
    build_trees(features, args.out, stats)

    manifest = {
        "images_dir": str(args.images_dir.resolve()),
        "n_images": int(len(ids)),
        "params": params,
        "descriptors": {name: {"dim": int(X.shape[1]), "metrics": config.DESCRIPTOR_METRICS[name],
                               "computable": name in config.COMPUTED_DESCRIPTORS}
                        for name, X in features.items()},
        "extraction_seconds": t_extract,
        "total_seconds": time.perf_counter() - t_start,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Done in {manifest['total_seconds']:.1f}s (pixel descriptors {t_extract:.1f}s) -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
