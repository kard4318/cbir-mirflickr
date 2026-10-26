"""Evaluation: retrieval quality per descriptor/combination and VP-Tree vs brute-force benchmark.

Usage (from the project root, venv active):
    python -m scripts.run_evaluation                          # artifacts/, 100 queries
    python -m scripts.run_evaluation --artifacts artifacts_test --n-queries 30

Ground truth: the 24 MIRFLICKR "potential" concept labels (data/annotations/*.txt,
without the _r1 files). A database image is RELEVANT to a query if it shares at
least one concept label with it. The query itself is excluded from its results,
and queries are drawn only among images having at least one label.

Metrics at K (K = 10, 20), averaged over the queries:
  P@K  = relevant results in the top K / K
  R@K  = relevant results in the top K / all relevant images in the database
  AP@K = (1 / min(K, R)) * sum_{i<=K} P@i * rel_i   (R = number of relevant images)
  mAP@K is the mean of AP@K over the queries.

Outputs (report_assets/): CSV files, PNG figures and evaluation_summary.md.
"""
import argparse
import time
from contextlib import closing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend import config
from backend.db import database
from backend.engine import SearchEngine
from backend.index import BruteForce, VPTree

KS = (10, 20)
COMBINATIONS = [
    ("color", "lbp"),
    ("color", "hog"),
    ("ehd", "htd"),
    ("color", "lbp", "hog"),
    ("color", "ehd", "htd"),
    ("color", "lbp", "hog", "ehd", "htd"),
    ("color", "lbp", "vgg16"),  # skipped automatically if VGG16 was not extracted
]

# Reference data-viz palette (light mode): two categorical slots + neutral inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ------------------------------------------------------------------ ground truth
def label_matrix(engine):
    with closing(engine.connect()) as conn:
        labels = database.get_labels(conn, "potential")
    names = sorted({lab for labs in labels.values() for lab in labs})
    col = {n: j for j, n in enumerate(names)}
    L = np.zeros((len(engine.ids), len(names)), dtype=np.float32)
    for image_id, labs in labels.items():
        row = engine.row_of.get(image_id)
        if row is not None:
            for lab in labs:
                L[row, col[lab]] = 1
    return L, names


def relevance(L, q_row):
    rel = (L @ L[q_row]) > 0
    rel[q_row] = False
    return rel


def metrics_at_k(result_rows, rel, n_relevant):
    hits = rel[result_rows].astype(np.float32)
    out = {}
    for k in KS:
        h = hits[:k]
        precisions = np.cumsum(h) / np.arange(1, len(h) + 1)
        out[f"P@{k}"] = h.sum() / k
        out[f"R@{k}"] = h.sum() / n_relevant if n_relevant else 0.0
        out[f"AP@{k}"] = (precisions * h).sum() / min(k, n_relevant) if n_relevant else 0.0
    return out


# ------------------------------------------------------------------ evaluation
def evaluate_retrieval(engine, L, queries):
    configs = []
    for name, info in engine.descriptors.items():
        for metric in info["metrics"]:
            configs.append((f"{name} [{metric}]", [(name, metric, 1.0)], "bruteforce"))
    for combo in COMBINATIONS:
        if all(n in engine.descriptors for n in combo):
            specs = [(n, engine.default_metric(n), 1.0) for n in combo]
            label = " + ".join(combo)
            configs.append((label, specs, "bruteforce"))
            configs.append((label, specs, "vptree"))

    kmax = max(KS)
    rows = []
    for label, specs, index in configs:
        per_query, agreement = [], []
        for q in queries:
            qvecs = {n: X[q] for n, X in engine.features.items()}
            res, _, _ = engine.search(qvecs, specs, kmax, index, exclude_row=q)
            rel = relevance(L, q)
            per_query.append(metrics_at_k(res, rel, int(rel.sum())))
            if index == "vptree":
                exact, _, _ = engine.search(qvecs, specs, kmax, "bruteforce", exclude_row=q)
                agreement.append(len(set(res.tolist()) & set(exact.tolist())) / kmax)
        row = {"config": label, "n_descriptors": len(specs), "index": index,
               **pd.DataFrame(per_query).mean().to_dict()}
        row["overlap_with_bruteforce@20"] = np.mean(agreement) if agreement else 1.0
        rows.append(row)
        print(f"  {label:38s} {index:10s} P@10={row['P@10']:.3f} mAP@20={row['AP@20']:.3f}")

    # Random-ranking baseline: expected precision = fraction of relevant images.
    base = np.mean([relevance(L, q).sum() / (len(L) - 1) for q in queries])
    rows.append({"config": "random baseline", "n_descriptors": 0, "index": "-",
                 "P@10": base, "P@20": base, "R@10": 10 / (len(L) - 1), "R@20": 20 / (len(L) - 1),
                 "AP@10": np.nan, "AP@20": np.nan, "overlap_with_bruteforce@20": np.nan})
    return pd.DataFrame(rows)


def benchmark(engine, queries, k=20):
    """Build time and query time of VP-Tree vs brute force, per descriptor (default metric)."""
    rows = []
    for name, X in engine.features.items():
        metric = engine.default_metric(name)
        t0 = time.perf_counter()
        tree = VPTree(X, metric, leaf_size=config.VPTREE_LEAF_SIZE).build()
        build = time.perf_counter() - t0
        bf = BruteForce(X, metric)
        tree.search(X[queries[0]], k), bf.search(X[queries[0]], k)  # warm-up
        t_vp, t_bf, nd, exact = [], [], [], 0
        for q in queries:
            t0 = time.perf_counter(); r_vp, d_vp, n = tree.search(X[q], k); t_vp.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); r_bf, d_bf, _ = bf.search(X[q], k); t_bf.append(time.perf_counter() - t0)
            nd.append(n)
            exact += np.allclose(d_vp, d_bf, atol=1e-5)
        rows.append({"descriptor": name, "metric": metric, "dim": X.shape[1], "n_images": len(X),
                     "vptree_build_s": build, "vptree_query_ms": 1000 * np.mean(t_vp),
                     "bruteforce_query_ms": 1000 * np.mean(t_bf),
                     "speedup": np.mean(t_bf) / np.mean(t_vp),
                     "vptree_distance_computations_pct": 100 * np.mean(nd) / len(X),
                     "vptree_exact_pct": 100 * exact / len(queries)})
        r = rows[-1]
        print(f"  {name:6s} build {build:6.2f}s  VP {r['vptree_query_ms']:7.2f} ms  "
              f"BF {r['bruteforce_query_ms']:7.2f} ms  dist. comp. {r['vptree_distance_computations_pct']:5.1f}%")
    return pd.DataFrame(rows)


def scalability(engine, rng, sizes, n_queries=50, k=20):
    """Average query time vs database size N, for each descriptor (default metric)."""
    rows = []
    for name, X_all in engine.features.items():
        metric = engine.default_metric(name)
        for n in sizes:
            if n > len(X_all):
                continue
            X = X_all[:n]
            t0 = time.perf_counter()
            tree = VPTree(X, metric, leaf_size=config.VPTREE_LEAF_SIZE).build()
            build = time.perf_counter() - t0
            bf = BruteForce(X, metric)
            qs = rng.choice(n, min(n_queries, n), replace=False)
            t_vp, t_bf, nd = [], [], []
            for q in qs:
                t0 = time.perf_counter(); _, _, c = tree.search(X[q], k); t_vp.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); bf.search(X[q], k); t_bf.append(time.perf_counter() - t0)
                nd.append(c)
            rows.append({"descriptor": name, "metric": metric, "n_images": n, "vptree_build_s": build,
                         "vptree_query_ms": 1000 * np.mean(t_vp), "bruteforce_query_ms": 1000 * np.mean(t_bf),
                         "vptree_distance_computations_pct": 100 * np.mean(nd) / n})
        print(f"  {name}: done")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _style(ax, xgrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def fig_quality(df, out):
    """Horizontal grouped bars: P@10 and P@20 per configuration (brute force, exact).

    Precision is plotted (not mAP) so the random-ranking baseline, which is a
    precision, can be drawn on the same axis.
    """
    d = df[(df["index"] == "bruteforce")].sort_values("P@20")
    base = df.loc[df["config"] == "random baseline", "P@20"].iloc[0]
    y = np.arange(len(d))
    h = 0.36
    fig, ax = _new_fig(8, 0.34 * len(d) + 1.4)
    _style(ax)
    ax.barh(y + h / 2 + 0.01, d["P@10"], height=h, color=BLUE, label="P@10")
    ax.barh(y - h / 2 - 0.01, d["P@20"], height=h, color=ORANGE, label="P@20")
    for yi, v in zip(y, d["P@10"]):
        ax.text(v + 0.004, yi + h / 2, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.axvline(base, color=INK_2, linewidth=1, linestyle="--")
    ax.text(base, len(d) - 0.4, f" random baseline ({base:.2f})", fontsize=8, color=INK_2, va="bottom")
    ax.set_yticks(y, d["config"], color=INK)
    ax.set_xlim(0, min(1.0, d[["P@10", "P@20"]].max().max() + 0.08))  # room for the value labels
    ax.set_xlabel("precision (higher is better)", color=INK_2)
    ax.set_title("Retrieval quality per descriptor and combination", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_query_time(bench, out):
    """Grouped bars: average query time VP-Tree vs brute force per descriptor (log scale)."""
    x = np.arange(len(bench))
    w = 0.36
    fig, ax = _new_fig(7, 3.8)
    _style(ax, xgrid=False)
    ax.bar(x - w / 2 - 0.01, bench["vptree_query_ms"], width=w, color=BLUE, label="VP-Tree")
    ax.bar(x + w / 2 + 0.01, bench["bruteforce_query_ms"], width=w, color=ORANGE, label="Brute force")
    for xi, v in zip(x, bench["vptree_query_ms"]):
        ax.text(xi - w / 2, v * 1.08, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    for xi, v in zip(x, bench["bruteforce_query_ms"]):
        ax.text(xi + w / 2, v * 1.08, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{d}\n({m}, {dim}-d)" for d, m, dim in
                      zip(bench["descriptor"], bench["metric"], bench["dim"])], color=INK)
    ax.set_ylabel("average query time, ms (log scale)", color=INK_2)
    ax.set_title(f"k-NN query time (k = 20, N = {bench['n_images'].iloc[0]:,})", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scalability(scal, out):
    """Small multiples: query time vs N, one panel per descriptor, same y scale."""
    names = list(dict.fromkeys(scal["descriptor"]))
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        d = scal[scal["descriptor"] == name]
        _style(ax, xgrid=False)
        ax.plot(d["n_images"], d["vptree_query_ms"], color=BLUE, linewidth=2, marker="o", markersize=4, label="VP-Tree")
        ax.plot(d["n_images"], d["bruteforce_query_ms"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Brute force")
        ax.set_yscale("log")
        ax.set_title(f"{name} [{d['metric'].iloc[0]}]", color=INK, fontsize=10)
        ax.set_xlabel("database size N", color=INK_2, fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].set_ylabel("query time, ms (log)", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Query time vs database size", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_distance_computations(bench, out):
    """Share of the database the VP-Tree compares against (100% = no pruning)."""
    d = bench.sort_values("vptree_distance_computations_pct")
    fig, ax = _new_fig(6.5, 2.8)
    _style(ax)
    y = np.arange(len(d))
    ax.barh(y, d["vptree_distance_computations_pct"], height=0.55, color=BLUE)
    for yi, v in zip(y, d["vptree_distance_computations_pct"]):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [f"{n} ({dim}-d)" for n, dim in zip(d["descriptor"], d["dim"])], color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel("distance computations per query, % of brute force", color=INK_2)
    ax.set_title("VP-Tree pruning efficiency (k = 20)", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--out", type=Path, default=config.REPORT_DIR)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    engine = SearchEngine(args.artifacts)
    L, names = label_matrix(engine)
    labeled = np.flatnonzero(L.sum(axis=1) > 0)
    rng = np.random.default_rng(args.seed)
    queries = rng.choice(labeled, min(args.n_queries, len(labeled)), replace=False)
    print(f"{len(engine.ids)} images, {len(names)} concept labels, {len(labeled)} labelled images, "
          f"{len(queries)} queries")

    print("Retrieval quality:")
    quality = evaluate_retrieval(engine, L, queries)
    quality.to_csv(args.out / "retrieval_metrics.csv", index=False, float_format="%.4f")
    print("VP-Tree vs brute force:")
    bench = benchmark(engine, queries)
    bench.to_csv(args.out / "benchmark_vptree_vs_bruteforce.csv", index=False, float_format="%.4f")
    print("Scalability:")
    scal = scalability(engine, rng, args.sizes)
    scal.to_csv(args.out / "scalability.csv", index=False, float_format="%.4f")

    fig_quality(quality, args.out / "fig_retrieval_quality.png")
    fig_query_time(bench, args.out / "fig_query_time.png")
    fig_scalability(scal, args.out / "fig_scalability.png")
    fig_distance_computations(bench, args.out / "fig_distance_computations.png")

    with open(args.out / "evaluation_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Evaluation summary\n\n{len(engine.ids)} images, {len(queries)} random labelled queries "
                 f"(seed {args.seed}). Relevant = shares at least one of the {len(names)} MIRFLICKR concept "
                 "labels with the query (query excluded).\n\n## Retrieval quality\n\n")
        fh.write(quality.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\n## VP-Tree vs brute force\n\n")
        fh.write(bench.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\n## Scalability\n\n")
        fh.write(scal.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n")
    print(f"Saved CSV, figures and evaluation_summary.md to {args.out}")


if __name__ == "__main__":
    main()
