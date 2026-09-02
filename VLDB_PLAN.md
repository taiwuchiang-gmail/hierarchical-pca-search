# PVLDB Volume 20 submission plan

**Decision (2026-08-31):** target PVLDB Vol. 20 (VLDB 2027), *regular
research track*, with a single 12-page paper merging Paper I (hierarchical
PCA pruning) and Paper II (composable bounds + self-configuring funnel).
JOSS submission dropped. The Zenodo records (Paper I concept
10.5281/zenodo.21387358; Paper II concept 10.5281/zenodo.22199955,
v2 published 2026-09-02 as 10.5281/zenodo.22258679) stay up as
preprints — PVLDB review is single-blind, preprints are fine and will
be disclosed.

## Logistics

- Rolling deadlines: 1st of each month 17:00 PT, **mandatory abstract by
  the 25th of the previous month**; final cycle March 1, 2027.
- Target: **December 1, 2026** (abstract Nov 25), fallback Jan/Feb/Mar.
- 12 pages excluding references, camera-ready PVLDB template.
- Supplemental materials must live in a public archival repo — this one.
- A PVLDB rejection bars resubmission of the same work for one year, so
  we submit once, when ready.

## Paper thesis

Composable exact lower bounds (PCA truncation + triangle-inequality ball)
give exact k-NN at ANN-class RAM residency (8–40 B/vector), with a
self-configuring diagnostic ("statistics nominate, measurement decides")
and hardware-independent work accounting. The headline application is
**out-of-core exact search** — the regime where FAISS-flat's in-RAM
wall-clock advantage is irrelevant and work/byte counts govern.

## Gap → task matrix (what reviewers will demand vs what exists)

| # | Gap | Task | Status |
|---|-----|------|--------|
| 1 | 1-NN only | k-NN support (k-th-best radius) + exactness tests | **DONE 2026-08-31**; measured (sift/gist_knn_results.txt): k≤10 wins, k=100 loses in-RAM wall-clock on both datasets — paper leads k≤10 + out-of-core fetch% |
| 2 | One dataset (SIFT1M) | GIST1M (960-D), DEEP1M/10M (96-D), one modern embedding set (768–1536-D); funnel + full benchmark on each | **GIST1M + DBpedia-OpenAI-1M DONE 2026-08-31** (auto levels): GIST 6.2×/20.8× work, DBpedia (1536-D ada-002) 6.0×/12.8× work, both exact at k=1 — **embedding risk resolved POSITIVE**; wall-clock gain GROWS with d. Required eigen-curve level selection (pick_levels, a797b44). DEEP1M/10M todo |
| 3 | Out-of-core claim unmeasured | Two-tier on-disk layout (resident metadata + cluster-major blocks), cold-cache protocol (drop_caches, needs sudo terminal), measured latency vs mmap flat scan | **DONE 2026-09-01** (`ooc_bench.py`, O_DIRECT so no sudo needed; results `ooc_run_*_k1.txt`, device `ooc_disk.txt`: 2.04 GiB/s seq, 80 us random 4K). All exact. Best-arm speedup vs cold scan: SIFT 7x (61 vs 402 ms, 2.3 vs 488 MB/q), GIST 7x, **DBpedia-1536D 17x (8 vs 5,860 MB/q = 730x fewer bytes)** — headline; DEEP10M 3x (weak case, 0.5% fetch set). Point-fetch beats block-fetch except DEEP10M. **Finding:** auto levels optimize in-RAM FLOPs, not fetches — SIFT auto [8,16,40] fetched 3,381 pts/q vs 588 at [8,16,32,64] (91→61 ms); OOC wants a deeper final level, diagnostic should cost-model the fetch |
| 4 | Weak baselines | ADSampling (SIGMOD'23) + successors (DADE/PDX-class); FAISS-flat already measured | **MEASURED 2026-09-01** (`adsampling_bench.py` — faithful NumPy port of the reference DCO from gaoj0017/ADSampling, dims-touched = same terms currency as count_ops; results `adsampling_results.txt`). k=1 Mterms, ADS eps0=2.1 (recall) vs our pca-only (exact): SIFT 33.0 3.9x (1.00) vs 12.6 10.2x; GIST 44.5 21.6x (1.00) vs 46.1 20.8x; DBpedia 49.8 30.8x (1.00) vs 119.6 12.8x; DEEP10M 325.8 2.9x (1.00) vs 244.0 3.9x. Honest read: ADS wins in-RAM work at high d and k=10, we win SIFT/DEEP10M; GIST tie. **Our case**: (a) deterministic exactness — but state it honestly: at default eps0 ADS recall was empirically perfect at k=1 on all four datasets (an earlier "2/100 GIST misses" was a float32 checker artifact — catastrophic cancellation on near-duplicates; fixed with f64 rescoring); the one real miss is GIST k=10 (recall 0.9990, 1/100 queries), so the argument is guarantee-vs-empirical, not observed failure; (b) structural floor — ADS pays >=delta_d=32 dims for EVERY candidate (SIFT 33.0M ~ floor 32M, DEEP10M 325.8M ~ floor 320M), i.e. it touches every vector, so out-of-core it costs full-scan bytes (row-major; dimension-major = the PDX successor) while our cascade fetches 0.06–1% of vectors — ADS cannot serve the OOC headline regime; (c) composability — their DCO could run on our survivor sets. Optional remaining: native C++ wall-clock (Eigen), DADE/PDX discussion is related-work not baseline |
| 5 | k choice unjustified | Re-add the measured k-sweep (data exists, was reverted; see §6.1 note) | **RE-MEASURED 2026-09-01** (`ksweep.py` → `ksweep_sift.txt`; old data was in a lost /tmp file, never committed): ball survivors 81/66/57/49/43% for k=8/64/256/1024/4096; k=8 hurts (73 ms vs 55 pca-only), k=4096 buys 1.08x query for 3.1x fit time — sqrt(N)=1024 sits at the knee. Still needs the §6.1 paragraph when writing |
| 6 | Scale | SIFT100M / DEEP10M+ out-of-core run if disk allows | stretch |

Scientific risk to test EARLY (task 2): modern embedding spectra may not
prune well. If the funnel says POOR on embeddings, that becomes an honest
negative result + the diagnostic's value proposition, but we need to know
in September, not November.

## Timeline

- **Sept 2026**: k-NN experiments on SIFT1M; download + funnel GIST1M,
  DEEP1M, embeddings (kill/confirm the embedding risk).
- **Oct 2026**: out-of-core layout + cold-cache measurements; ADSampling
  baseline.
- **Nov 2026**: merge papers into PVLDB template, related-work expansion,
  internal review; abstract due Nov 25.
- **Dec 1, 2026**: submit (slip to Jan–Mar cycles if experiments demand).
