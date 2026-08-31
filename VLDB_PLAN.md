# PVLDB Volume 20 submission plan

**Decision (2026-08-31):** target PVLDB Vol. 20 (VLDB 2027), *regular
research track*, with a single 12-page paper merging Paper I (hierarchical
PCA pruning) and Paper II (composable bounds + self-configuring funnel).
JOSS submission dropped. The Zenodo records (10.5281/zenodo.21387358,
10.5281/zenodo.22199956) stay up as preprints — PVLDB review is
single-blind, preprints are fine and will be disclosed.

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
| 1 | 1-NN only | k-NN support (k-th-best radius) + exactness tests | **DONE 2026-08-31** |
| 2 | One dataset (SIFT1M) | GIST1M (960-D), DEEP1M/10M (96-D), one modern embedding set (768–1536-D); funnel + full benchmark on each | todo |
| 3 | Out-of-core claim unmeasured | Two-tier on-disk layout (resident metadata + cluster-major blocks), cold-cache protocol (drop_caches, needs sudo terminal), measured latency vs mmap flat scan | todo — the paper's headline |
| 4 | Weak baselines | ADSampling (SIGMOD'23) + successors (DADE/PDX-class); FAISS-flat already measured | todo |
| 5 | k choice unjustified | Re-add the measured k-sweep (data exists, was reverted; see §6.1 note) | todo, 1 edit |
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
