# FlexiWalker Test Suite

The test suite is split into two layers, each with its own pytest marker.

## Layers

- **`tests/compiler/` (marker: `compiler`)** — golden-file tests of the
  compiler pipeline output, plus regression tests for the `eRVS_only`
  fallback detection. No GPU required. Fast (single pipeline run per test
  module, cached as a fixture).
- **`tests/e2e/` (marker: `e2e`)** — end-to-end smoke tests that build
  the `flowwalker` binary and run it on `wiki-Vote` for each of the 5
  shipped walkers. GPU required; the whole layer is skipped if
  `nvidia-smi` is unavailable.

## Running the tests

All commands assume the repository root as the working directory and are run
inside the toolchain container (see [Toolchain container](../README.md#toolchain-container-recommended)
in the top-level README) — or directly on the host if you have an equivalent
toolchain.

```bash
# All tests (compiler + e2e if GPU is available)
python3 -m pytest tests/

# Compiler tests only (fast, GPU-free)
python3 -m pytest tests/compiler/ -v

# E2E smoke tests only
python3 -m pytest tests/e2e/ -v

# Run a single test case
python3 -m pytest \
    'tests/compiler/test_golden.py::test_generated_file_matches_golden[walker_traits.cuh-Node2vec]' -v
```

## Regenerating golden snapshots

Golden snapshots live in `tests/compiler/golden/<walker>/`. They are
**intentionally** committed and should only change when the compiler's
output legitimately changes (e.g., a Phase-2 AST-rewrite refactor alters
intermediate JSON).

To update them after verifying the change is intentional:

```bash
python3 -m pytest tests/compiler/test_golden.py --update-golden -v
git diff tests/compiler/golden/   # REVIEW the diff carefully before committing
git add tests/compiler/golden/
git commit -m "test: regenerate goldens for <reason>"
```

`--update-golden` writes one snapshot per `(walker, filename)` pair and
passes the assertion unconditionally. Never commit a regenerated golden
without reviewing the diff.

## Adding a new walker

If you add a new walker to `include/app.cuh`:

1. Add its name to `SHIPPED_WALKERS` in `tests/compiler/conftest.py`.
2. Add an entry for it to `WALKERS` in `tests/e2e/test_walkers.py`
   (choose `"fixed"` or `"probabilistic"` depending on its stop behavior).
3. Run `pytest tests/compiler/test_golden.py --update-golden` to create
   the initial snapshot and commit it.

## Adding a new eRVS_only test case

1. Append the synthetic walker class to
   `tests/compiler/fixtures/ervs_test_walkers.cuh`.
2. Add an `(ClassName, expected_ervs_only)` tuple to the `ERVS_CASES`
   list in `tests/compiler/test_ervs_fallback.py`.
3. Run `pytest tests/compiler/test_ervs_fallback.py -v` to verify.

## Test-specific wiki-Vote config

The default `config/graphs/wiki-Vote.config` assumes graph files live in
the shared `/data` mount. The e2e suite uses
`tests/e2e/fixtures/wiki-Vote-test.config` instead, which points at the
repo's own `data/wiki-Vote_*.bin` files — so the suite is self-contained
and does not depend on external dataset mounts.

## CI expectations

The test suite is expected to finish green on every commit to `main`.
`tests/compiler/` must pass without a GPU; `tests/e2e/` may be gated on
GPU availability.
