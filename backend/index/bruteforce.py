"""Exhaustive (brute-force) k-NN search: the reference for exactness and timing."""
import numpy as np

from .distances import distances


class BruteForce:
    def __init__(self, data, metric):
        self.data = data
        self.metric = metric

    def search(self, q, k, rows=None):
        """Return (row indices, distances, number of distance computations).

        `rows` optionally restricts the search to a subset of rows (keyword filter).
        """
        q = np.asarray(q, dtype=np.float32)
        X = self.data if rows is None else self.data[rows]
        d = distances(q, X, self.metric)
        k = min(k, len(d))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), len(d)
        top = np.argpartition(d, k - 1)[:k]
        top = top[np.argsort(d[top], kind="stable")]
        ids = top if rows is None else np.asarray(rows)[top]
        return ids.astype(np.int64), d[top], len(d)
