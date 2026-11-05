"""Figure: per-query work of PCA-only vs hybrid, as a fraction of a
brute-force scan, on a tie dataset (SIFT1M) and an add dataset (512
clusters). Linear axis. This is the figure that separates the two
cascades, which the survivor-count plot cannot: on clustered data the
hybrid reaches the same survivors as PCA-only with far less work.

SIFT numbers are Table 3 of the paper (bench_faiss_5rep.py); the
clustered numbers are the Mterms/Mbound columns that `hybrid_search.py
demo` prints for the 512-cluster regime (hybrid_demo.txt).
Outputs pvldb/fig_work.pdf."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#4C72B0", "#DD8452"

# work per query, in millions; brute = n*d terms
DATA = {
    "SIFT1M\n(tie)": dict(
        brute=128.0, pca_terms=10.34, hyb_terms=6.29, hyb_bound=1.00),
    "512 clusters\n(add)": dict(
        brute=12.794, pca_terms=1.633, hyb_terms=0.069, hyb_bound=0.200),
}


def pct(x, brute):
    return 100.0 * x / brute


def main():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    xs = range(len(DATA))
    w = 0.34
    for i, (name, d) in enumerate(DATA.items()):
        pca = pct(d["pca_terms"], d["brute"])
        h_t = pct(d["hyb_terms"], d["brute"])
        h_b = pct(d["hyb_bound"], d["brute"])
        ax.bar(i - w / 2, pca, w, color=BLUE,
               label="PCA-only (terms)" if i == 0 else None)
        ax.bar(i + w / 2, h_t, w, color=ORANGE,
               label="hybrid (terms)" if i == 0 else None)
        ax.bar(i + w / 2, h_b, w, bottom=h_t, color=ORANGE, alpha=0.45,
               hatch="///", edgecolor="white", linewidth=0,
               label="hybrid (ball bound-ops)" if i == 0 else None)
        ax.text(i - w / 2, pca + 0.4, f"{pca:.1f}", ha="center",
                va="bottom", fontsize=7, color=BLUE)
        ax.text(i + w / 2, h_t + h_b + 0.4, f"{h_t + h_b:.1f}", ha="center",
                va="bottom", fontsize=7, color=ORANGE)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(DATA.keys(), fontsize=8)
    ax.set_ylabel("work per query\n(% of brute-force scan)", fontsize=8)
    ax.set_ylim(0, 15)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              borderaxespad=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("pvldb/fig_work.pdf")
    print("saved pvldb/fig_work.pdf")


if __name__ == "__main__":
    main()
