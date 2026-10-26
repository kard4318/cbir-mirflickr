"""ONLINE phase: the search engine loaded once at server start-up.

Distance combination: when several descriptors are selected, each distance is
first divided by that descriptor's scale (the median distance between random
pairs of database images, computed offline). The normalized distances are then
averaged with the user weights:
    D(q, x) = sum_i w_i * d_i(q, x) / s_i  /  sum_i w_i
A weighted sum of metrics is still a metric. Brute-force search over D is exact.
In VP-Tree mode, each selected descriptor's tree returns its own nearest
candidates (max(10k, 200) per tree); the union is then re-ranked with the exact
combined distance D. This is fast but approximate, and the evaluation measures
how close it is to brute force.
"""
import json
import time
from pathlib import Path

import numpy as np

from .db import database
from .descriptors import compute_descriptors
from .descriptors.preprocess import decode_image_bytes
from .index import BruteForce, VPTree, distances


class SearchError(ValueError):
    pass


class SearchEngine:
    def __init__(self, artifacts_dir):
        self.dir = Path(artifacts_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.stats = json.loads((self.dir / "stats.json").read_text())
        self.params = self.manifest["params"]
        self.images_dir = Path(self.manifest["images_dir"])
        self.db_path = self.dir / "cbir.sqlite"
        self.ids = np.load(self.dir / "ids.npy")
        self.row_of = {int(i): r for r, i in enumerate(self.ids)}
        self.features, self.trees, self.brute = {}, {}, {}
        for name, info in self.manifest["descriptors"].items():
            X = np.load(self.dir / info.get("file", f"features_{name}.npy"))
            self.features[name] = X
            for metric in info["metrics"]:
                self.trees[(name, metric)] = VPTree.load(self.dir / f"vptree_{name}_{metric}.joblib", X)
                self.brute[(name, metric)] = BruteForce(X, metric)
        # The CNN is loaded once at start-up (only if VGG16 was extracted), so
        # uploaded images can be described with the same network as offline.
        self.vgg = None
        if "vgg16" in self.descriptors:
            from .descriptors.vgg16 import VGG16Extractor
            self.vgg = VGG16Extractor(self.params["vgg16"]["input_size"])

    # ---------------------------------------------------------------- helpers
    def connect(self):
        return database.connect(self.db_path)

    @property
    def descriptors(self):
        return self.manifest["descriptors"]

    def default_metric(self, name):
        return self.descriptors[name]["metrics"][0]

    def scale(self, name, metric):
        return self.stats["scales"][f"{name}/{metric}"]

    def vectors_for_image_id(self, image_id):
        row = self.row_of.get(int(image_id))
        if row is None:
            raise SearchError(f"unknown image id {image_id}")
        return {name: X[row] for name, X in self.features.items()}, row

    @property
    def upload_descriptors(self):
        """Descriptors that can be computed from the pixels of an uploaded image."""
        return [name for name, info in self.descriptors.items() if info["computable"]]

    def vectors_for_upload(self, data, names):
        img = decode_image_bytes(data)
        if img is None:
            raise SearchError("the uploaded file is not a readable image")
        # Same functions and parameters as the offline indexer.
        vecs = compute_descriptors(img, self.params)
        if "vgg16" in names:
            vecs["vgg16"] = self.vgg.extract([img])[0]
        return vecs

    def normalize_specs(self, specs, available):
        """specs: [(name, metric or None, weight)] -> validated list."""
        out = []
        for name, metric, weight in specs:
            if name not in self.descriptors:
                raise SearchError(f"unknown descriptor {name!r}")
            if name not in available:
                raise SearchError(f"descriptor {name!r} is not available for uploaded images "
                                  "(MPEG-7 EHD/HTD only exist for database images)")
            metric = metric or self.default_metric(name)
            if metric not in self.descriptors[name]["metrics"]:
                raise SearchError(f"metric {metric!r} not supported for {name!r}")
            if weight is None or weight <= 0:
                continue
            out.append((name, metric, float(weight)))
        if not out:
            raise SearchError("select at least one descriptor with a positive weight")
        return out

    # ----------------------------------------------------------------- search
    def search(self, qvecs, specs, k, index="vptree", rows=None, exclude_row=None):
        """Return (rows, distances, info). `rows` restricts the search to a subset."""
        t0 = time.perf_counter()
        kk = k + (1 if exclude_row is not None else 0)
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
            res_rows, res_d, n_dist = self._brute_combined(qvecs, specs, kk, rows)
            index_used = f"brute force on keyword subset ({len(rows)} images)"
        elif len(specs) == 1:
            name, metric, _ = specs[0]
            engine = self.trees if index == "vptree" else self.brute
            res_rows, res_d, n_dist = engine[(name, metric)].search(qvecs[name], kk)
            index_used = "VP-Tree" if index == "vptree" else "brute force"
        elif index == "vptree":
            res_rows, res_d, n_dist = self._vptree_combined(qvecs, specs, kk)
            index_used = "VP-Tree (per-descriptor candidates + exact re-ranking)"
        else:
            res_rows, res_d, n_dist = self._brute_combined(qvecs, specs, kk, None)
            index_used = "brute force"
        if exclude_row is not None:
            keep = res_rows != exclude_row
            res_rows, res_d = res_rows[keep][:k], res_d[keep][:k]
        elapsed = time.perf_counter() - t0
        return res_rows, res_d, {"time_ms": elapsed * 1000, "index_used": index_used,
                                 "distance_computations": int(n_dist)}

    def _combined_on(self, qvecs, specs, rows):
        total_w = sum(w for _, _, w in specs)
        acc = np.zeros(len(rows), dtype=np.float32)
        for name, metric, w in specs:
            d = distances(qvecs[name], self.features[name][rows], metric)
            acc += (w / total_w) * d / self.scale(name, metric)
        return acc

    def _brute_combined(self, qvecs, specs, k, rows):
        if len(specs) == 1 and rows is None:
            name, metric, _ = specs[0]
            return self.brute[(name, metric)].search(qvecs[name], k)
        if rows is None:
            rows = np.arange(len(self.ids))
        if len(specs) == 1:
            name, metric, _ = specs[0]
            return self.brute[(name, metric)].search(qvecs[name], k, rows=rows)
        d = self._combined_on(qvecs, specs, rows)
        k = min(k, len(d))
        top = np.argpartition(d, k - 1)[:k] if k else np.empty(0, dtype=np.int64)
        top = top[np.argsort(d[top], kind="stable")]
        return rows[top], d[top], len(rows) * len(specs)

    def _vptree_combined(self, qvecs, specs, k):
        n_cand = max(10 * k, 200)
        cand, n_dist = [], 0
        for name, metric, _ in specs:
            r, _, nd = self.trees[(name, metric)].search(qvecs[name], n_cand)
            cand.append(r)
            n_dist += nd
        cand = np.unique(np.concatenate(cand))
        d = self._combined_on(qvecs, specs, cand)
        n_dist += len(cand) * len(specs)
        top = np.argsort(d, kind="stable")[:k]
        return cand[top], d[top], n_dist

    @staticmethod
    def apply_fuzziness(dists, fuzziness):
        """Keep the results whose normalized distance <= fuzziness * normalized distance of the K-th result.

        The combined distance is min-max normalized to [0, 1] over the whole
        database: n(d) = (d - d_min) / (d_max - d_min). Here d_min is the
        distance of the best result, since the search returns the database
        minimum first. The rule n(d_i) <= f * n(d_K) is then equivalent to
            r_i = (d_i - d_1) / (d_K - d_1) <= f,
        because d_max cancels out. So the rule is applied exactly without a full
        scan to find d_max, and in VP-Tree mode as well. r_i is returned as the
        "relative distance" of each result (0 = best match, 1 = K-th match).
        f = 1 keeps the plain top-K. Small f keeps only results close to the
        best match, possibly fewer than K.
        """
        dists = np.asarray(dists, dtype=np.float64)
        if len(dists) == 0:
            return np.zeros(0, dtype=bool), dists
        span = dists[-1] - dists[0]
        rel = (dists - dists[0]) / span if span > 0 else np.zeros_like(dists)
        return rel <= fuzziness + 1e-9, rel

    def per_descriptor_distances(self, qvecs, specs, rows):
        """Raw distance of each result for each selected descriptor (for display)."""
        return {f"{name}/{metric}": distances(qvecs[name], self.features[name][rows], metric)
                for name, metric, _ in specs}
