"""Distance functions between one query vector and a matrix of vectors.

- l1: sum |x - q| (Manhattan)
- l2: sqrt(sum (x - q)^2) (Euclidean)
- intersection: 1 - sum min(x, q), for L1-normalized histograms. It equals
  0.5 * L1 on such histograms, so it satisfies the triangle inequality and can be
  used inside the VP-Tree.
All three are metrics, which the VP-Tree pruning rule requires.
"""
import numpy as np

METRICS = ("l1", "l2", "intersection")

# Rows processed per chunk, to avoid huge temporaries on large descriptors (HOG).
_CHUNK = 2048


def _block(q, X, metric):
    if metric == "l1":
        return np.abs(X - q).sum(axis=1)
    if metric == "l2":
        diff = X - q
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))
    if metric == "intersection":
        return 1.0 - np.minimum(X, q).sum(axis=1)
    raise ValueError(f"unknown metric {metric!r}")


def distances(q, X, metric):
    """Distances from q (d,) to every row of X (n, d); returns float32 (n,)."""
    n = X.shape[0]
    if n <= _CHUNK:
        return _block(q, X, metric)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CHUNK):
        out[start:start + _CHUNK] = _block(q, X[start:start + _CHUNK], metric)
    return out


def distance(a, b, metric):
    """Distance between two single vectors."""
    return float(_block(a, b[None, :], metric)[0])
