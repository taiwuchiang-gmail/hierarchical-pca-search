"""File loading: .fvecs (TEXMEX format) and .npy round-trips."""

import numpy as np
import pytest

from diagnose import fvecs_read, ivecs_read, load_vectors


def write_fvecs(path, X):
    n, d = X.shape
    rec = np.zeros((n, d + 1), dtype=np.float32)
    rec[:, 1:] = X
    rec.view(np.int32)[:, 0] = d
    rec.tofile(path)


def test_fvecs_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(100, 16)).astype(np.float32)
    path = tmp_path / "vectors.fvecs"
    write_fvecs(path, X)
    assert np.array_equal(fvecs_read(str(path)), X)
    assert np.array_equal(load_vectors(str(path)), X)


def test_ivecs_roundtrip(tmp_path):
    G = np.random.default_rng(1).integers(0, 1000, size=(50, 10),
                                          dtype=np.int32)
    rec = np.zeros((50, 11), dtype=np.int32)
    rec[:, 0] = 10
    rec[:, 1:] = G
    path = tmp_path / "gt.ivecs"
    rec.tofile(path)
    assert np.array_equal(ivecs_read(str(path)), G)


def test_npy_loading(tmp_path):
    X = np.random.default_rng(2).normal(size=(30, 8)).astype(np.float32)
    path = tmp_path / "vectors.npy"
    np.save(path, X)
    assert np.array_equal(load_vectors(str(path)), X)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "vectors.csv"
    path.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_vectors(str(path))
