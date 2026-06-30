# Hierarchical PCA Pruning for Exact Nearest Neighbor Search

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper:  
**"A Hierarchical Pruning Algorithm for Fast, Exact Nearest Neighbor Search in High-Dimensional Spaces"** (VLDB '25 / arXiv 2025).

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

## Understanding the Code

The core of the logic is inside `hierarchical_pca_1nn_l2` in `benchmark.py`. 
It utilizes a **Top-50 Probe**: at the lowest dimensional representation (8D), it identifies the 50 closest candidates and immediately computes their full 128D distance. The absolute minimum of these 50 becomes a mathematically guaranteed, tight upper-bound radius for pruning the remaining 999,950 vectors.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chiang2025hierarchical,
  title={A Hierarchical Pruning Algorithm for Fast, Exact Nearest Neighbor Search in High-Dimensional Spaces},
  author={Chiang, Tai-Wu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License
MIT License