# Contributing

Thanks for your interest! Bug reports, questions, and pull requests are all
welcome via the [GitHub issue tracker](https://github.com/taiwuchiang-gmail/hierarchical-pca-search/issues).

## Reporting problems / getting support

Open an issue with:

- what you ran (command line or minimal code snippet),
- what you expected and what happened instead,
- your data's shape/dtype (or a synthetic generator that reproduces it),
- Python / NumPy / scikit-learn versions.

If `diagnose.py` gave you a verdict that measurement later contradicted,
that is a bug in the funnel — please report it with the diagnostic output.

## Development setup

```bash
git clone https://github.com/taiwuchiang-gmail/hierarchical-pca-search.git
cd hierarchical-pca-search
pip install -e ".[test]"
pytest
```

The suite runs in a few seconds on synthetic data; no dataset download is
needed. CI also runs it without scikit-learn to exercise the pure-NumPy
k-means fallback.

## The one non-negotiable invariant

**Every query path must return the exact brute-force nearest neighbor, on
every data regime, always.** Bound quality may affect speed, never
correctness. Any change to `hybrid_search.py` or `diagnose.py` must keep
`tests/test_exactness.py` green, and performance changes should be justified
in hardware-independent work counts (`count_ops`) rather than wall-clock
alone — see Paper II (`paper2/`) for the accounting rules.

## Style

Plain NumPy, no hard dependencies beyond it. Keep the code small and
readable — this repository is meant to be an engineering artifact people can
read end-to-end.
