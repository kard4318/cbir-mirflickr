"""Vantage-Point Tree (Yianilos, 1993) for exact k-nearest-neighbour search in a metric space.

Construction (recursive):
  1. Pick a vantage point vp among the current points. We sample a few
     candidates and keep the one whose distances to a random sample have the
     largest spread, which gives more discriminating splits.
  2. Compute d(vp, x) for every other point x. Split at the median mu: the inner
     child receives points with d <= mu and the outer child points with d >= mu.
     The two children therefore have equal sizes and the tree stays balanced.
  3. For each child, store the [min, max] range of d(vp, x) over its points.
  4. Stop when a node holds at most `leaf_size` points. A leaf keeps its points
     in a bucket that is scanned with one vectorized distance computation.

k-NN search: keep the k best distances found so far, with tau the current k-th
best (initially +inf). At an internal node compute dq = d(q, vp). By the
triangle inequality, every point x of a child with d(vp, x) in [lo, hi]
satisfies d(q, x) >= max(lo - dq, dq - hi). The child can therefore contain a
better point only if dq + tau >= lo and dq - tau <= hi; otherwise it is pruned.
The child on the query's side of mu is visited first so tau shrinks early.
The search is exact: it returns the same neighbours as brute force.

The tree only stores integer row indices and float bounds in flat numpy arrays.
The descriptor matrix is attached at load time, so it is never duplicated on
disk.
"""
import joblib
import numpy as np

from .distances import distances


class VPTree:
    def __init__(self, data, metric, leaf_size=32, seed=0):
        self.data = data
        self.metric = metric
        self.leaf_size = leaf_size
        self.seed = seed

    # ------------------------------------------------------------------ build
    def build(self):
        rng = np.random.default_rng(self.seed)
        self._vp, self._children, self._bounds, self._leaf_range = [], [], [], []
        self._leaf_items = []
        self._leaf_count = 0
        self._build_node(np.arange(len(self.data)), rng)
        # Freeze the node lists into compact arrays.
        self.vp = np.array(self._vp, dtype=np.int64)  # -1 for leaves
        self.children = np.array(self._children, dtype=np.int64).reshape(-1, 2)
        self.bounds = np.array(self._bounds, dtype=np.float32).reshape(-1, 4)
        self.leaf_range = np.array(self._leaf_range, dtype=np.int64).reshape(-1, 2)
        self.leaf_items = (np.concatenate(self._leaf_items) if self._leaf_items
                           else np.empty(0, dtype=np.int64))
        del self._vp, self._children, self._bounds, self._leaf_range, self._leaf_items
        return self

    def _new_node(self):
        self._vp.append(-1)
        self._children.append((-1, -1))
        self._bounds.append((0.0, 0.0, 0.0, 0.0))
        self._leaf_range.append((0, 0))
        return len(self._vp) - 1

    def _select_vp(self, idx, rng, n_candidates=5, n_sample=100):
        if len(idx) <= n_candidates:
            return int(idx[0])
        candidates = rng.choice(idx, n_candidates, replace=False)
        sample = self.data[rng.choice(idx, min(n_sample, len(idx)), replace=False)]
        spreads = [distances(self.data[c], sample, self.metric).std() for c in candidates]
        return int(candidates[int(np.argmax(spreads))])

    def _build_node(self, idx, rng):
        node = self._new_node()
        if len(idx) <= self.leaf_size:
            start = self._leaf_count
            self._leaf_items.append(idx.astype(np.int64))
            self._leaf_count += len(idx)
            self._leaf_range[node] = (start, self._leaf_count)
            return node
        vp = self._select_vp(idx, rng)
        rest = idx[idx != vp]
        d = distances(self.data[vp], self.data[rest], self.metric)
        m = len(rest) // 2
        part = np.argpartition(d, m)
        inner, outer = part[:m], part[m:]
        self._vp[node] = vp
        self._bounds[node] = (d[inner].min(), d[inner].max(), d[outer].min(), d[outer].max())
        left = self._build_node(rest[inner], rng)
        right = self._build_node(rest[outer], rng)
        self._children[node] = (left, right)
        return node

    # ----------------------------------------------------------------- search
    def search(self, q, k):
        """Return (row indices, distances, number of distance computations), sorted by distance."""
        q = np.asarray(q, dtype=np.float32)
        best_i = np.empty(0, dtype=np.int64)
        best_d = np.empty(0, dtype=np.float32)
        tau = np.inf
        n_dist = 0

        def offer(ids, ds):
            nonlocal best_i, best_d, tau
            best_i = np.concatenate([best_i, ids])
            best_d = np.concatenate([best_d, ds])
            if len(best_d) > k:
                keep = np.argpartition(best_d, k - 1)[:k]
                best_i, best_d = best_i[keep], best_d[keep]
            if len(best_d) == k:
                tau = float(best_d.max())

        # Each stack entry carries a lower bound on d(q, x) for any x in that subtree.
        stack = [(0, 0.0)]
        while stack:
            node, lower = stack.pop()
            if lower > tau:  # tau may have shrunk since the node was pushed
                continue
            vp = self.vp[node]
            if vp < 0:  # leaf: scan its bucket
                s, e = self.leaf_range[node]
                if e > s:
                    items = self.leaf_items[s:e]
                    offer(items, distances(q, self.data[items], self.metric))
                    n_dist += e - s
                continue
            dq = float(distances(q, self.data[vp][None, :], self.metric)[0])
            n_dist += 1
            if dq < tau or len(best_d) < k:
                offer(np.array([vp]), np.array([dq], dtype=np.float32))
            in_lo, in_hi, out_lo, out_hi = self.bounds[node]
            left, right = self.children[node]
            # Push the far child first so the near child is processed first (LIFO).
            order = [(right, out_lo, out_hi), (left, in_lo, in_hi)]
            if dq > in_hi:  # query lies on the outer side
                order.reverse()
            for child, lo, hi in order:
                bound = max(lo - dq, dq - hi, 0.0)  # triangle-inequality lower bound
                if bound <= tau:
                    stack.append((child, bound))
        order = np.argsort(best_d, kind="stable")
        return best_i[order], best_d[order], n_dist

    # ------------------------------------------------------------ persistence
    def save(self, path):
        joblib.dump({"metric": self.metric, "leaf_size": self.leaf_size, "seed": self.seed,
                     "vp": self.vp, "children": self.children, "bounds": self.bounds,
                     "leaf_range": self.leaf_range, "leaf_items": self.leaf_items}, path)

    @classmethod
    def load(cls, path, data):
        state = joblib.load(path)
        tree = cls(data, state["metric"], state["leaf_size"], state["seed"])
        for key in ("vp", "children", "bounds", "leaf_range", "leaf_items"):
            setattr(tree, key, state[key])
        return tree
