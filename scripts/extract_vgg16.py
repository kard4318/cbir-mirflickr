"""OFFLINE: extract VGG16 deep features for every indexed image, then register the descriptor.

Usage (from the project root, venv active, after scripts.build_index):
    python -m scripts.extract_vgg16 --limit 200          # speed test, nothing registered
    python -m scripts.extract_vgg16                       # all images (resumable)

The features are written into artifacts/vgg16.npy (memory-mapped, in the row
order of ids.npy). Progress is checkpointed in artifacts/vgg16_progress.json every
--save-every batches. If the script is interrupted (Ctrl+C), running the same
command again resumes from the last checkpoint. When every row is done, the
descriptor is registered: its VP-Tree is built and it is added to manifest.json
and stats.json, so the server picks it up at the next start.
Re-running scripts.build_index rewrites the manifest: run this script again
afterwards. If the image list is unchanged it only re-registers, without
re-extracting.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backend import config
from backend.db import database
from backend.descriptors.preprocess import load_image_file
from backend.descriptors.vgg16 import VGG16Extractor, preprocess
from backend.index import VPTree
from scripts.build_index import distance_scale


def _write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)  # atomic: a checkpoint is never half-written


def load_batch(paths, size):
    """Read and preprocess one batch (runs in a background thread)."""
    out, bad = [], []
    for i, p in enumerate(paths):
        img = load_image_file(p)
        if img is None:
            bad.append(i)
            img = np.zeros((size, size, 3), dtype=np.uint8)
        out.append(preprocess(img, size))
    return np.stack(out), bad


def extract(paths, out_path, progress_path, ids_hash, size, batch_size, save_every, threads):
    n = len(paths)
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    resume = out_path.exists() and state.get("n") == n and state.get("ids_sha1") == ids_hash
    if resume:
        feats = np.lib.format.open_memmap(out_path, mode="r+")
        done = state["done"]
        print(f"Resuming at {done}/{n}")
    else:
        feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 512))
        done = 0
    if done >= n:
        return feats, None

    extractor = VGG16Extractor(size, threads=threads)
    starts = list(range(done, n, batch_size))
    bad_total, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(1) as loader, tqdm(total=n, initial=done, unit="img", desc="VGG16") as bar:
        future = loader.submit(load_batch, paths[starts[0]:starts[0] + batch_size], size)
        for b, start in enumerate(starts):
            batch, bad = future.result()
            if b + 1 < len(starts):  # prefetch the next batch while the CPU runs the network
                nxt = starts[b + 1]
                future = loader.submit(load_batch, paths[nxt:nxt + batch_size], size)
            feats[start:start + len(batch)] = extractor.extract_preprocessed(batch)
            bad_total += [paths[start + i] for i in bad]
            bar.update(len(batch))
            if (b + 1) % save_every == 0 or b + 1 == len(starts):
                feats.flush()
                _write_json(progress_path, {"n": n, "ids_sha1": ids_hash, "done": start + len(batch)})
    speed = (n - done) / (time.perf_counter() - t0)
    if bad_total:
        print(f"WARNING: {len(bad_total)} unreadable images got a zero vector, e.g. {bad_total[:3]}")
    return feats, speed


def register(art, feats, size):
    """Build the VP-Tree and add the descriptor to the manifest and stats."""
    manifest = json.loads((art / "manifest.json").read_text())
    stats = json.loads((art / "stats.json").read_text())
    X = np.asarray(feats)
    metric = config.DESCRIPTOR_METRICS["vgg16"][0]
    t0 = time.perf_counter()
    tree = VPTree(X, metric, leaf_size=config.VPTREE_LEAF_SIZE).build()
    stats["build_seconds"][f"vgg16/{metric}"] = time.perf_counter() - t0
    tree.save(art / f"vptree_vgg16_{metric}.joblib")
    stats["scales"][f"vgg16/{metric}"] = distance_scale(X, metric, np.random.default_rng(0))
    manifest["params"]["vgg16"] = {"input_size": size, "weights": "torchvision IMAGENET1K_V1",
                                   "layer": "features (conv blocks 1-5, incl. pool5)",
                                   "pooling": "global average", "normalization": "L2"}
    manifest["descriptors"]["vgg16"] = {"dim": int(X.shape[1]), "metrics": config.DESCRIPTOR_METRICS["vgg16"],
                                        "computable": True, "file": "vgg16.npy"}
    _write_json(art / "manifest.json", manifest)
    _write_json(art / "stats.json", stats)
    print(f"Registered vgg16: VP-Tree built in {stats['build_seconds'][f'vgg16/{metric}']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=10, help="checkpoint every N batches")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all cores)")
    ap.add_argument("--limit", type=int, default=None,
                    help="speed test on the first N images (separate file, not registered)")
    args = ap.parse_args()
    art = args.artifacts
    size = config.DESCRIPTOR_PARAMS["vgg16"]["input_size"]

    manifest = json.loads((art / "manifest.json").read_text())
    ids = np.load(art / "ids.npy")
    with closing(database.connect(art / "cbir.sqlite")) as conn:
        files = dict(conn.execute("SELECT id, filename FROM images").fetchall())
    images_dir = Path(manifest["images_dir"])
    paths = [str(images_dir / files[int(i)]) for i in ids]

    if args.limit:
        paths = paths[:args.limit]
        out, prog = art / f"vgg16_test{args.limit}.npy", art / f"vgg16_test{args.limit}_progress.json"
        for p in (out, prog):
            p.unlink(missing_ok=True)  # a speed test always starts from scratch
    else:
        out, prog = art / "vgg16.npy", art / "vgg16_progress.json"
    ids_hash = hashlib.sha1(ids[:len(paths)].tobytes()).hexdigest()

    feats, speed = extract(paths, out, prog, ids_hash, size, args.batch_size, args.save_every, args.threads)
    if speed:
        print(f"Speed: {speed:.1f} images/s -> {len(ids) / speed / 60:.1f} min for all {len(ids)} images")
    norms = np.linalg.norm(np.asarray(feats), axis=1)
    print(f"Vector norms: min {norms.min():.3f} max {norms.max():.3f} (should be 1)")
    if not args.limit:
        register(art, feats, size)


if __name__ == "__main__":
    main()
