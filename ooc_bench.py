"""Out-of-core cold-read benchmark: two-tier layout vs full disk scan.

Three arms, identical exact k-NN results, different disk traffic:

  scan   -- stream the whole rotated file, brute force (the out-of-core
            baseline: sequential bandwidth, reads 100% of the bytes)
  point  -- cascade on the resident sketch, then fetch each needed
            vector from the ORIGINAL-order file (scattered small reads)
  block  -- same cascade, but fetch whole clusters from the
            CLUSTER-MAJOR file (few large reads)

All reads use O_DIRECT: the page cache is bypassed, so every query is
cold-cache by construction -- no drop_caches, no sudo, repeatable.
Resident state = sketch (first max(levels) PCA dims) + centroids/bounds/
offsets; full-D vectors live on disk only.

Usage:
  python ooc_bench.py build <dataset> [levels]   # fit + write layout
  python ooc_bench.py run   <dataset> [k]        # timed three-arm bench
  python ooc_bench.py disk  <dataset>            # device characterization
"""

import mmap
import os
import sys
import time

import numpy as np

from hybrid_search import HybridIndex, fvecs_read

PAGE = 4096
CHUNK = 8 << 20                     # scan read size
N_QUERIES = 100
N_SCAN_QUERIES = 10                 # full scans are slow and deterministic


def _paths(name):
    tag = os.path.basename(name).replace(".npy", "")
    return (f"ooc_{tag}_orig.bin", f"ooc_{tag}_cmaj.bin",
            f"ooc_{tag}_meta.npz")


def _load(name):
    if name.endswith(".npy"):
        X = np.load(name, mmap_mode="r")
        rng = np.random.default_rng(0)
        q_idx = rng.choice(len(X), N_QUERIES, replace=False)
        mask = np.ones(len(X), bool)
        mask[q_idx] = False
        return np.asarray(X[mask]), np.asarray(X[q_idx])
    base = fvecs_read(f"{name}/{name}_base.fvecs")
    return base, fvecs_read(f"{name}/{name}_query.fvecs")[:N_QUERIES]


def build(name, levels="auto"):
    base, queries = _load(name)
    n, d = base.shape
    t0 = time.perf_counter()
    idx = HybridIndex(levels=levels, n_clusters=1024).fit(base)
    print(f"fit {time.perf_counter() - t0:.0f}s; levels {idx.levels}")

    f_orig, f_cmaj, f_meta = _paths(name)
    vec = d * 4

    with open(f_orig, "wb") as f:
        os.posix_fallocate(f.fileno(), 0, n * vec)
        idx.X.tofile(f)

    # cluster-major: rows grouped by cluster, each block padded to a page
    counts = np.bincount(idx.assign, minlength=len(idx.C))
    order = np.argsort(idx.assign, kind="stable")
    rank = np.empty(n, np.int64)                    # id -> row within block
    rank[order] = np.arange(n) - np.repeat(
        np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    blk_bytes = (counts * vec + PAGE - 1) // PAGE * PAGE
    offsets = np.concatenate([[0], np.cumsum(blk_bytes)[:-1]])
    with open(f_cmaj, "wb") as f:
        os.posix_fallocate(f.fileno(), 0, int(blk_bytes.sum()))
        for c in range(len(counts)):
            rows = idx.X[order[np.searchsorted(idx.assign[order], [c, c + 1])[0]:
                               np.searchsorted(idx.assign[order], [c, c + 1])[1]]]
            f.write(rows.tobytes())
            f.write(b"\0" * int(blk_bytes[c] - counts[c] * vec))

    lmax = max(idx.levels)
    np.savez(f_meta, mean=idx.mean, rot=idx.rot, levels=idx.levels,
             C=idx.C, assign=idx.assign, d_p=idx.d_p, rank=rank,
             offsets=offsets, counts=counts, blk_bytes=blk_bytes,
             sketch=idx.X[:, :lmax], queries=queries, n=n, d=d)
    print(f"layout: {f_orig} {n * vec / 2**30:.2f} GiB, "
          f"{f_cmaj} {blk_bytes.sum() / 2**30:.2f} GiB "
          f"(padding {blk_bytes.sum() / (n * vec) - 1:+.2%}), "
          f"resident sketch {n * lmax * 4 / 2**20:.0f} MiB "
          f"({d / lmax:.1f}x residency reduction)")


def _pread(fd, offset, length, counter):
    a0 = offset & ~(PAGE - 1)
    a1 = (offset + length + PAGE - 1) & ~(PAGE - 1)
    buf = mmap.mmap(-1, a1 - a0)
    done = 0
    while done < a1 - a0:                # preadv may return partial reads
        got = os.preadv(fd, [memoryview(buf)[done:]], a0 + done)
        if got == 0:                     # EOF (file ends mid-page)
            break
        done += got
    assert done >= offset - a0 + length, "short O_DIRECT read"
    counter[0] += a1 - a0
    counter[1] += 1
    return np.frombuffer(buf, np.uint8)[offset - a0:offset - a0 + length]


class OOCIndex(HybridIndex):
    """Cascade on the resident sketch; full-D distances hit the disk."""

    def __init__(self, meta, mode, fd_orig, fd_cmaj):
        self.levels = [int(x) for x in meta["levels"]]
        self.mean, self.rot = meta["mean"], meta["rot"]
        self.C, self.assign, self.d_p = meta["C"], meta["assign"], meta["d_p"]
        self.X = meta["sketch"]                     # (n, lmax) -- NOT full-D
        self.rank, self.offsets = meta["rank"], meta["offsets"]
        self.counts = meta["counts"]
        self.d = int(meta["d"])
        self.mode, self.fd_orig, self.fd_cmaj = mode, fd_orig, fd_cmaj
        self.io = [0, 0]                            # bytes, read calls

    def _transform(self, q):                        # q stays full-D
        return (q.astype(np.float32) - self.mean) @ self.rot

    def _exact(self, ids, q):
        vec = self.d * 4
        out = np.empty(len(ids), np.float32)
        if self.mode == "point":
            for j, i in enumerate(ids):
                raw = _pread(self.fd_orig, int(i) * vec, vec, self.io)
                v = np.frombuffer(raw.tobytes(), np.float32)
                out[j] = ((v - q) ** 2).sum()
            return out
        for c in np.unique(self.assign[ids]):       # block mode
            raw = _pread(self.fd_cmaj, int(self.offsets[c]),
                         int(self.counts[c]) * vec, self.io)
            blk = np.frombuffer(raw.tobytes(), np.float32).reshape(-1, self.d)
            m = self.assign[ids] == c
            out[m] = ((blk[self.rank[ids[m]]] - q) ** 2).sum(1)
        return out


def _scan_knn(fd, n, d, q_rot, k, counter):
    """Streaming brute-force k-NN over the orig-order file (O_DIRECT)."""
    vec = d * 4
    rows_per = CHUNK // vec
    best_d = np.full(k, np.inf, np.float32)
    best_i = np.full(k, -1, np.int64)
    pos = 0
    while pos < n:
        m = min(rows_per, n - pos)
        raw = _pread(fd, pos * vec, m * vec, counter)
        blk = np.frombuffer(raw.tobytes(), np.float32).reshape(m, d)
        d2 = ((blk - q_rot) ** 2).sum(1)
        alld = np.concatenate([best_d, d2])
        alli = np.concatenate([best_i, np.arange(pos, pos + m)])
        sel = np.argpartition(alld, k - 1)[:k]
        best_d, best_i = alld[sel], alli[sel]
        pos += m
    o = np.argsort(best_d, kind="stable")
    return best_i[o], best_d[o]


def run(name, k=1):
    f_orig, f_cmaj, f_meta = _paths(name)
    meta = np.load(f_meta)
    queries = meta["queries"]
    n, d = int(meta["n"]), int(meta["d"])
    fd_o = os.open(f_orig, os.O_RDONLY | os.O_DIRECT)
    fd_c = os.open(f_cmaj, os.O_RDONLY | os.O_DIRECT)
    print(f"{name}: {n:,} x {d}, k={k}, levels "
          f"{[int(x) for x in meta['levels']]}, O_DIRECT cold reads")

    # scan baseline (exact reference) on a query subset
    scan_res, scan_t, scan_io = [], [], [0, 0]
    idx0 = OOCIndex(meta, "block", fd_o, fd_c)      # for _transform only
    for q in queries[:N_SCAN_QUERIES]:
        t0 = time.perf_counter()
        scan_res.append(_scan_knn(fd_o, n, d, idx0._transform(q), k, scan_io))
        scan_t.append(time.perf_counter() - t0)

    results = {}
    for mode in ("point", "block"):
        idx = OOCIndex(meta, mode, fd_o, fd_c)
        times, per_q = [], []
        for qi, q in enumerate(queries):
            io0 = list(idx.io)
            t0 = time.perf_counter()
            r = idx.query(q, k=k)
            times.append(time.perf_counter() - t0)
            per_q.append((idx.io[0] - io0[0], idx.io[1] - io0[1]))
            if qi < N_SCAN_QUERIES:                 # exactness vs scan arm
                sd = scan_res[qi][1]
                rd = np.atleast_1d(np.asarray(r[1], np.float32))
                assert np.allclose(np.sort(rd), np.sort(sd)), \
                    f"{mode} mismatch vs scan on q{qi}"
        results[mode] = (times, per_q)

    smb = scan_io[0] / len(scan_t) / 2**20
    sms = np.median(scan_t) * 1000
    print(f"\n  {'arm':>6} {'ms/query':>9} {'speedup':>8} {'MB/query':>9} "
          f"{'reads':>6}   (exact on all checked queries)")
    print(f"  {'scan':>6} {sms:>9.0f} {'1.0x':>8} {smb:>9.0f} "
          f"{scan_io[1] // len(scan_t):>6}")
    for mode, (times, per_q) in results.items():
        ms = np.median(times) * 1000
        mb = np.mean([b for b, _ in per_q]) / 2**20
        rd = np.mean([r for _, r in per_q])
        print(f"  {mode:>6} {ms:>9.1f} {sms / ms:>7.0f}x {mb:>9.2f} "
              f"{rd:>6.0f}")
    os.close(fd_o), os.close(fd_c)


def disk(name):
    f_orig, _, f_meta = _paths(name)
    size = os.path.getsize(f_orig)
    fd = os.open(f_orig, os.O_RDONLY | os.O_DIRECT)
    io = [0, 0]
    t0 = time.perf_counter()
    pos, budget = 0, min(size, 1 << 30)
    while pos < budget:
        _pread(fd, pos, min(CHUNK, size - pos), io)
        pos += CHUNK
    seq = budget / (time.perf_counter() - t0) / 2**30
    rng = np.random.default_rng(0)
    offs = rng.integers(0, size - PAGE, 500) & ~(PAGE - 1)
    t0 = time.perf_counter()
    for o in offs:
        _pread(fd, int(o), PAGE, io)
    lat = (time.perf_counter() - t0) / len(offs) * 1e6
    print(f"{name} device: sequential {seq:.2f} GiB/s (8 MiB O_DIRECT), "
          f"random 4 KiB {lat:.0f} us")
    os.close(fd)


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "build":
        lv = sys.argv[3] if len(sys.argv) > 3 else "auto"
        if lv != "auto":
            lv = tuple(int(x) for x in lv.split(","))
        build(name, lv)
    elif cmd == "run":
        run(name, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif cmd == "disk":
        disk(name)
