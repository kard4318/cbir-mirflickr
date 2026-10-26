"""Check that the VP-Tree returns exactly the brute-force neighbours.

Run: python -m tests.test_vptree
"""
import numpy as np

from backend.index import BruteForce, VPTree


def check(dim, metric, n=3000, k=20, n_queries=50, histogram=False):
    rng = np.random.default_rng(1)
    data = rng.random((n, dim), dtype=np.float32)
    if histogram:
        data /= data.sum(axis=1, keepdims=True)
    tree = VPTree(data, metric, leaf_size=16).build()
    bf = BruteForce(data, metric)
    visited = []
    for qi in rng.choice(n, n_queries, replace=False):
        q = data[qi] + rng.normal(0, 0.01, dim).astype(np.float32)
        ti, td, nd = tree.search(q, k)
        bi, bd, _ = bf.search(q, k)
        assert np.allclose(td, bd, atol=1e-5), (metric, dim, td[:5], bd[:5])
        visited.append(nd / n)
    print(f"ok  dim={dim:4d} metric={metric:12s} avg distance computations={np.mean(visited):6.1%} of brute force")


if __name__ == "__main__":
    for dim in (4, 16, 128):
        for metric in ("l1", "l2"):
            check(dim, metric)
    check(128, "intersection", histogram=True)
