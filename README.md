# Hierarchical PCA Pruning for Exact Nearest Neighbor Search

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21387359.svg)](https://doi.org/10.5281/zenodo.21387359)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper:  
**"A Hierarchical Pruning Algorithm for Fast, Exact Nearest Neighbor Search in High-Dimensional Spaces"** (VLDB '26 / Zenodo 2026).

## Overview

Finding the exact nearest neighbor in high-dimensional space is bottlenecked by the "Curse of Dimensionality," particularly the I/O costs of loading full vectors (e.g., 128-D) from disk. 

This repository implements a **Dynamic Top-K Branch and Bound algorithm** using PCA and the L2 Euclidean norm. By evaluating candidates in progressively higher dimensions (8D $\rightarrow$ 16D $\rightarrow$ 32D $\rightarrow$ 64D), the algorithm mathematically guarantees 100% exactness while safely pruning **over 99.9% of the candidate space** before performing a full-dimensional distance calculation.

### Results on SIFT1M
* **4.3x reduction** in query latency compared to a fully optimized, vectorized CPU SIMD baseline.
* **Reduces disk I/O exponentially:** Out of 1,000,000 vectors, an average of only 564 vectors require full 128-D evaluation.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/hierarchical-pca-search.git
cd hierarchical-pca-search
pip install -r requirements.txt
```

## Quick Start (Benchmarking)

To run the SIFT1M benchmark and reproduce the results from the paper:

```bash
python benchmark.py
```

**What this script does automatically:**
1. Downloads and extracts the [SIFT1M dataset](http://corpus-texmex.irisa.fr/) (if not present).
2. Fits a PCA model on the 1,000,000 base vectors and saves it (offline indexing).
3. Runs 100 randomly selected queries using both the **Vectorized Brute-Force Baseline** and the **Hierarchical PCA Search**.
4. Verifies 100% exactness (zero false dismissals).
5. Generates a logarithmic bar chart (`figures/pruning_effectiveness.png`) showing the candidate drop-off at each dimensional cascade.

## Will it help *your* data? (prunability diagnostic)

The speedup depends entirely on how **correlated** your dimensions are: correlated data
concentrates variance in a few PCA components (fast eigenvalue decay), which makes the
low-dimensional lower bounds tight and the pruning aggressive. Uncorrelated data (flat
eigenvalue spectrum) gives loose bounds and no benefit.

`diagnose.py` measures this on a sample of *your* vectors in seconds, before you
integrate anything:

```bash
python diagnose.py your_vectors.npy        # or .fvecs
python diagnose.py --demo                  # synthetic tour: watch the verdict flip
                                           # as correlation weakens
```

It reports the eigenvalue decay, runs the actual pruning cascade on the sample
(verifying exactness against brute force), and prints measured survivor rates, the
fraction of full vectors touched (your out-of-core I/O cost), a wall-clock speedup,
and a plain verdict:

```
VERDICT: STRONG   -- highly correlated data; pruning will pay off.
VERDICT: POOR     -- dimensions too uncorrelated (flat eigenvalue decay);
                     bounds are loose, use brute force / ANN instead.
```

Rule of thumb: if 95% of the variance fits in a small fraction of the dimensions,
this method will prune hard; if it needs nearly all dimensions, it won't. Only
numpy is required (matplotlib optional, for `--plot decay.png`).

### Validation: the diagnostic reproduces the paper's SIFT1M results

Running `diagnose.py` on the full SIFT1M base set recovers the paper's measured
cascade stage-by-stage, from a single command:

| stage | paper (Fig. 1) | `diagnose.py` (1M) |
|---|---|---|
| after 8D | 146,955 (14.7%) | 135,356 (13.5%) |
| after 16D | 52,107 (5.2%) | 42,322 (4.2%) |
| after 32D | 7,896 (0.8%) | 5,953 (0.6%) |
| after 64D | 564 (0.056%) | 366 (0.04%) |
| speedup | 4.3x | 5.0x |
| exactness | 100% | 100/100 |

(Small deltas are expected: the diagnostic uses held-out base vectors as queries,
so it works on any `.npy`/`.fvecs` file without a separate query set.)

![SIFT1M prunability](figures/sift_prunability.png)

The left panel is the cause (SIFT's first 8 PCA dims hold 58.5% of the variance —
strongly correlated dimensions); the right panel is the effect (candidates drop
four orders of magnitude through the cascade).

## Understanding the Code

The core of the logic is inside `hierarchical_pca_1nn_l2` in `benchmark.py`. 
It utilizes a **Top-50 Probe**: at the lowest dimensional representation (8D), it identifies the 50 closest candidates and immediately computes their full 128D distance. The absolute minimum of these 50 becomes a mathematically guaranteed, tight upper-bound radius for pruning the remaining 999,950 vectors.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chiang2025hierarchical,
  title={A Hierarchical Pruning Algorithm for Fast, Exact Nearest Neighbor Search in High-Dimensional Spaces},
  author={Chiang, Tai-Wu},
  publisher={Zenodo},
  year={2026},
  doi={10.5281/zenodo.21387359},
  url={https://doi.org/10.5281/zenodo.21387359}
}
```

## License
MIT License
