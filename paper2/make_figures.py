"""Generate the three figures for Paper II (vector PDFs for LaTeX).

Fig 1  fig_bounds.pdf   the two bound geometries (schematic)
Fig 2  fig_cascade.pdf  SIFT1M survivor cascade, log scale (parity data)
Fig 3  fig_law.pdf      ball-value law: gain vs ball-stage survivor rate

Colors: PCA/control #4C72B0, ball/hybrid #DD8452 (CVD-validated pair),
neutral gray for references. All data from the parity-corrected runs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY = "#4C72B0", "#DD8452", "#8a8a8a"
INK, MUT = "#333333", "#777777"
plt.rcParams.update({"font.size": 8, "text.color": INK,
                     "axes.edgecolor": MUT, "xtick.color": MUT,
                     "ytick.color": MUT, "axes.labelcolor": INK})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.3, color="#dddddd", zorder=0)


# --- Fig 1: the two bound geometries ----------------------------------------
rng = np.random.default_rng(3)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6))

# (a) PCA truncation bound: elongated cloud, distance along PC1 is a bound
th = np.deg2rad(28)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
cloud = (rng.normal(size=(320, 2)) * [2.6, 0.5]) @ R.T
a.scatter(*cloud.T, s=4, color=BLUE, alpha=0.35, lw=0, zorder=2)
a.plot([-3.6 * np.cos(th), 3.6 * np.cos(th)],
       [-3.6 * np.sin(th), 3.6 * np.sin(th)],
       color=MUT, lw=0.8, ls=":", zorder=1)
a.annotate("PC 1", (3.0 * np.cos(th), 3.0 * np.sin(th)),
           textcoords="offset points", xytext=(4, -8), color=MUT)
q = np.array([-1.4, 1.9])
p = (np.array([2.2, 0.6]) @ R.T)
u = np.array([np.cos(th), np.sin(th)])           # PC1 direction
qp, pp = (q @ u) * u, (p @ u) * u                # projections onto PC1
a.plot(*zip(q, p), color=GRAY, lw=1.0, zorder=3)
a.plot(*zip(qp, pp), color=BLUE, lw=2.2, zorder=4)
for x, xp in ((q, qp), (p, pp)):
    a.plot(*zip(x, xp), color=MUT, lw=0.6, ls="--", zorder=1)
a.scatter(*q, marker="*", s=90, color=ORANGE, zorder=5)
a.scatter(*p, s=22, color=INK, zorder=5)
a.annotate("$q$", q, textcoords="offset points", xytext=(-10, 2))
a.annotate("$p$", p, textcoords="offset points", xytext=(5, -2))
a.annotate(r"$\|q-p\|$", (q + p) / 2, textcoords="offset points",
           xytext=(4, 6), color=GRAY)
a.annotate("bound: distance\nalong first $k$ PCs", (qp + pp) / 2,
           textcoords="offset points", xytext=(-4, -26), color=BLUE)
a.set_title("PCA truncation bound\n(tight: variance concentrates)",
            fontsize=8.5)
a.set_aspect("equal"); a.axis("off")

# (b) ball bound: clusters, centroid, d_p, ||q-c||
centers = np.array([[0, 0], [2.6, 1.8], [2.4, -1.4]])
for i, c in enumerate(centers):
    pts = c + 0.42 * rng.normal(size=(70, 2))
    b.scatter(*pts.T, s=4, color=ORANGE, alpha=0.35, lw=0, zorder=2)
c = centers[0]
p2 = c + np.array([0.30, -0.38])
q2 = np.array([-2.6, 1.7])
circ = plt.Circle(c, 0.75, fill=False, color=MUT, lw=0.7, ls=":")
b.add_patch(circ)
b.plot(*zip(q2, c), color=ORANGE, lw=2.0, zorder=3)
b.plot(*zip(c, p2), color=INK, lw=1.2, zorder=3)
b.scatter(*q2, marker="*", s=90, color=ORANGE, zorder=5)
b.scatter(*c, marker="X", s=40, color=INK, zorder=5)
b.scatter(*p2, s=22, color=INK, zorder=5)
b.annotate("$q$", q2, textcoords="offset points", xytext=(-10, 2))
b.annotate("$c$", c, textcoords="offset points", xytext=(-2, 7))
b.annotate("$p$", p2, textcoords="offset points", xytext=(5, -6))
b.annotate(r"$\|q-c\|$", (q2 + c) / 2, textcoords="offset points",
           xytext=(-6, 7), color=ORANGE)
b.annotate("$d_p$", (c + p2) / 2, textcoords="offset points",
           xytext=(6, -2))
b.annotate(r"bound: $\|q-c\| - d_p$", (-2.4, -1.6), color=ORANGE)
b.set_title("Triangle-inequality ball bound\n(tight: points huddle at "
            "centroids)", fontsize=8.5)
b.set_aspect("equal"); b.axis("off")
fig.tight_layout()
fig.savefig("fig_bounds.pdf")
plt.close(fig)

# --- Fig 2: SIFT1M survivor cascade (parity data) ----------------------------
stages = ["all", "ball", "8D", "16D", "32D", "64D"]
x = np.arange(len(stages))
pca = [1e6, np.nan, 147000, 52100, 7900, 600]
hyb = [1e6, 506000, 145600, 50800, 7400, 500]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(x[[0, 2, 3, 4, 5]], [pca[i] for i in (0, 2, 3, 4, 5)], "-o",
        color=BLUE, lw=2, ms=4, zorder=3, label="PCA-only")
ax.plot(x, hyb, "-o", color=ORANGE, lw=2, ms=4, zorder=3, label="hybrid")
ax.set_yscale("log")
ax.set_xticks(x, stages)
ax.set_ylabel("candidates remaining")
ax.annotate("1,000,000", (0, 1e6), textcoords="offset points",
            xytext=(6, 4), fontsize=7, color=MUT)
ax.annotate("600", (5, 600), textcoords="offset points",
            xytext=(-2, 8), fontsize=7, color=BLUE)
ax.annotate("500", (5, 500), textcoords="offset points",
            xytext=(-4, -12), fontsize=7, color=ORANGE)
ax.legend(frameon=False, fontsize=7.5)
style(ax)
fig.tight_layout()
fig.savefig("fig_cascade.pdf")
plt.close(fig)

# --- Fig 3: the ball-value law -----------------------------------------------
pts = [
    (0.24, 6.6, "512 clusters"),
    (2.98, 4.2, "512 clusters (60k)"),
    (14.8, 2.65, "latent-8"),
    (22.8, 1.9, "latent-8 (60k)"),
    (50.6, 0.98, "SIFT1M"),
    (59.4, 1.0, "SIFT (100k)"),
    (99.9, 0.96, "i.i.d."),
    (100.0, 0.8, "latent-48"),
]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
ax.axhline(1.0, color=GRAY, lw=1.0, ls="--", zorder=1)
ax.annotate("no effect", (0.3, 1.0), textcoords="offset points",
            xytext=(0, 4), color=GRAY, fontsize=7)
ax.plot(xs, ys, color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
ax.scatter(xs, ys, s=26, color=ORANGE, zorder=3)
offs = [(5, 3), (5, 3), (5, 3), (5, 3), (-12, 8), (5, 5), (-24, 8), (-30, -11)]
for (px, py, lab), off in zip(pts, offs):
    ax.annotate(lab, (px, py), textcoords="offset points", xytext=off,
                fontsize=6.5, color=INK)
ax.set_xscale("log")
ax.set_xlabel("ball-stage survivors (%)")
ax.set_ylabel("ball-level gain over PCA-only")
ax.set_xlim(0.15, 220)
style(ax)
fig.tight_layout()
fig.savefig("fig_law.pdf")
plt.close(fig)

print("wrote fig_bounds.pdf, fig_cascade.pdf, fig_law.pdf")
