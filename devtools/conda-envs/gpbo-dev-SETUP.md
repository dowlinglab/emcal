# `gpbo-dev` environment — working setup (Phase 0)

Reproducible dev environment for the package refactor, validated on **macOS arm64
(Apple Silicon), Python 3.10**. This is the env that runs the *current* (pre-refactor)
code end-to-end and anchors the golden regression baseline.

## Why these pins
The README pins `gpflow==2.9.1` but **not** TensorFlow. gpflow 2.9.1 predates TF 2.16's
switch to Keras 3, so a naive `pip install gpflow==2.9.1` pulls TF 2.16 / TFP 0.25 / Keras 3,
which **fails to import** (`TFP requires TensorFlow >= 2.18`). The working triple is:

| package | version | note |
|---|---|---|
| python | 3.10 | |
| gpflow | 2.9.1 | as pinned in README |
| tensorflow-macos | 2.15.0 | arm64 mac; on Linux use `tensorflow==2.15.*` |
| tensorflow-probability | 0.23.0 | matches TF 2.15 |
| keras | 2.15.0 | TF 2.15 bundles Keras 2 (NOT Keras 3) |
| numpy | 1.26.4 | `<2` required by TF 2.15 |
| Tasmanian | 8.2 | README pinned 7.7.1; 8.2 is current on PyPI, `makeGlobalGrid` API stable |
| setuptools | <81 | gpflow 2.9.1 uses `pkg_resources` (removed in setuptools 81) |
| scipy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.9, pyomo 6.10.1, pygad 3.7.0 | | |

Full lock: `gpbo-dev-pip-freeze.txt`.

## Reproduce
```bash
conda create -y -n gpbo-dev python=3.10 pip
conda activate gpbo-dev
pip install "numpy<2" "gpflow==2.9.1"
# fix the TF/TFP/Keras combo (pip over-resolves TF):
pip install "tensorflow-macos==2.15.0" "tensorflow-probability==0.23.0" "keras<2.16"
pip uninstall -y tensorflow                       # remove the shim that pulls Keras 3
pip install --force-reinstall --no-deps "tensorflow-macos==2.15.0"   # restore clean tensorflow/ tree
pip install "setuptools<81" pandas scikit-learn matplotlib pyomo pygad pytest pytest-cov
# Tasmanian (no arm64 wheel / not on conda-forge for osx-arm64 -> build from source):
pip install "cmake<4" scikit-build ninja wheel    # CMake 4 drops policies Tasmanian 7/8 need
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install --no-build-isolation Tasmanian
```

### Tasmanian dylib path workaround (macOS)
The Tasmanian wheel hardcodes `~/.local/lib/...` into `site-packages/TasmanianConfig.py`,
but the dylibs install into `$CONDA_PREFIX/lib`. Symlink them where the wrapper looks:
```bash
mkdir -p ~/.local/lib ~/.local/share/Tasmanian
for l in libtasmaniansparsegrid libtasmaniandream libtasmaniancaddons; do
  ln -sf "$CONDA_PREFIX/lib/$l.dylib" ~/.local/lib/"$l.dylib"
done
```

## Validation done in Phase 0
- Library imports fully (`bo_methods_lib.bo_methods_lib.GPBO_Classes_New` + `GPBO_Class_fxns`).
- CS1 (Simple Linear) runs end-to-end via `GPBO_Driver.run_bo_restarts(job=None)` — **no signac**:
  - Method 1 (Conventional/A1): converges (`why_term=obj`), recovers θ ≈ [1.0, −1.0] (true = [1, −1]).
  - Method 5 (Emulator/Sparse Grid/C2): runs through Tasmanian, recovers θ ≈ [0.998, −0.997].

## Known issue found in Phase 0 (not a blocker)
`bo_iter_tot == 1` raises `IndexError` at `GPBO_Classes_New.py:6804`
(`self.gpbo_res_simple[i] = ...` on a list only populated when ≥2 BO iterations run).
Harmless for paper runs (50–75 iters); use `bo_iter_tot >= 2` for regression tests.
Candidate fix during refactor: pre-size or `append` the result lists in `run_bo_restarts`.

## Notes for packaging (Phase 1/2)
- On Linux/CI use `tensorflow==2.15.*` instead of `tensorflow-macos`.
- The `tensorflow` extra in `pyproject.toml` should pin `tensorflow>=2.12,<2.16`,
  `tensorflow-probability>=0.20,<0.24`, `keras<2.16`, `setuptools<81`.
- Tasmanian belongs in a `sparsegrid` extra; document the source-build + dylib workaround.
