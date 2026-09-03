"""Figure: cumulative explained variance per dataset, with the funnel's
auto-selected cascade widths marked. The curve is the expected fractional
tightness of the width-k bound (paper Eq. in SS4), so this one plot is each
dataset's prunability signature. Outputs pvldb/fig_eigen.pdf (+ a spectra
cache so re-styling doesn't recompute four PCAs)."""

import os

import numpy as np

from diagnose import LEVEL_TARGETS, load_vectors, pca_fit_transform, pick_levels

DATASETS = [
    ("SIFT1M", "sift/sift_base.fvecs", "#4C72B0"),
    ("GIST1M", "gist/gist_base.fvecs", "#55A868"),
    ("DBpedia-OpenAI", "dbpedia_openai_1M.npy", "#C44E52"),
    ("DEEP10M", "deep10M.npy", "#DD8452"),
]
CACHE = "eigen_spectra.npz"


def spectra():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    out = {}
    rng = np.random.default_rng(0)
    for name, path, _ in DATASETS:
        X = load_vectors(path)
        X = X[rng.choice(len(X), 100_000, replace=False)]
        _, ev = pca_fit_transform(X.astype(np.float64))
        out[name] = np.cumsum(ev) / max(ev.sum(), 1e-30)
        print(f"{name}: d={len(out[name])} done")
    np.savez(CACHE, **out)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cums = spectra()
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for t in LEVEL_TARGETS:
        ax.axhline(t * 100, ls=":", c="gray", lw=0.7, zorder=0)
    for name, _, color in DATASETS:
        cum = cums[name]
        d = len(cum)
        ks = np.arange(1, d + 1)
        ax.plot(ks, cum * 100, c=color, lw=1.6, label=f"{name} (d={d})")
        lv = pick_levels(cum)
        ax.plot(lv, [cum[k - 1] * 100 for k in lv], "o", c=color, ms=4.5,
                mec="white", mew=0.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(1, 2048)
    ax.set_ylim(0, 100)
    ax.set_xlabel("PCA components $k$ (log scale)")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.text(1.15, 87, "85%", fontsize=7, c="gray", va="bottom")
    ax.text(1.15, 31, "30%", fontsize=7, c="gray", va="bottom")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False,
              borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # corner inset: raw per-component variance fractions (log-log) --
    # recovered exactly as the difference of the cumulative curve
    ins = ax.inset_axes([0.56, 0.10, 0.42, 0.40])
    for name, _, color in DATASETS:
        frac = np.diff(cums[name], prepend=0.0)
        ins.plot(np.arange(1, len(frac) + 1), frac, c=color, lw=1.0)
    ins.set_xscale("log")
    ins.set_yscale("log")
    ins.set_xlim(1, 2048)
    ins.set_title(r"raw spectra: $\lambda_k / \sum_i \lambda_i$",
                  fontsize=6.5, pad=2)
    ins.tick_params(labelsize=5.5, length=2, pad=1)
    ins.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("pvldb/fig_eigen.pdf")
    print("saved pvldb/fig_eigen.pdf")


if __name__ == "__main__":
    main()
