# Phase 8.1 — Quality-Hardening Audit (read-only)

**Status:** AUDIT ONLY. No code was changed to produce this report. Written 2026-07-04,
against the post-7C tree (golden 9/9, net 13/13, pytest 79/79 green).

This feeds two follow-up sessions: sections 1+3 become the **8.3 cleanup** prompts, section 4
becomes the **8.2 test-build** prompt. Section 2 (naming) is left for a separate adjudication
pass — it's listed as candidates only, no fixes proposed.

---

## 1. Duplication

### 1.1 `emulators.py` — `ObjectiveGP` vs `EmulatorGP` (the primary de-bloat target)

`emulators.py` is the largest module (1941 lines). Both classes inherit `GPEmulator` but
re-implement 9 method pairs. Classification below; line numbers are current (this session's
tree).

| # | Pair | Classification | Divergence |
|---|---|---|---|
| 1 | `predict_sse` (998–1029 / 1565–1614) | NEAR-IDENTICAL | Shared target-dispatch skeleton; `EmulatorGP` adds `method`/`exp_data` params + extra asserts (inherent — Type-2 needs them to derive SSE) |
| 2 | `__eval_gp_sse_var` (962–996 / 1481–1563) | **GENUINELY DIVERGENT** | ObjectiveGP: trivial passthrough (`sse = prediction.mean`). EmulatorGP: reshape into blocks, sum-of-squares vs `exp_data.y_vals`, quadratic-form variance propagation (`2·tr(Σ²) + 4rᵀΣr`). Inherent to Type-1 (GP fits SSE directly) vs Type-2 (GP fits model output, SSE derived) |
| 3 | `featurize_data` (830–858 / 1336–1367) | NEAR-IDENTICAL wrapper / 1-line divergent core | ObjectiveGP: `data.theta_vals`. EmulatorGP: `concat(theta_vals, x_vals)`. This one line *is* the Type-1/Type-2 distinction |
| 4 | `expected_improvement` (1110–1149 / 1763–1825) | NEAR-IDENTICAL | Shared target-dispatch + 4-assertion block byte-identical; EmulatorGP adds `method`/`sg_mc_samples` + a sparse-grid/MC multi-theta guard |
| 5 | `__eval_gp_ei` (1066–1108 / 1693–1761) | NEAR-IDENTICAL | Both reduce to: predict-if-none → build `ExpectedImprovement(...)` → `.compute()`. EmulatorGP passes 2 extra trailing args + an upfront method-range assert |
| 6 | `calc_best_error` (1031–1064 / 1616–1691) | **GENUINELY DIVERGENT** | ObjectiveGP: `np.min(train_data.y_vals)` (4 lines). EmulatorGP: block-reshape + sum-of-squares + argmin, returns an extra `best_sq_error` element. Same root cause as #2 |
| 7 | `append_training_point` (1151–1190 / 1827–1870) | NEAR-IDENTICAL | Identical `theta_vals`/`y_vals` vstack/concat; EmulatorGP has one extra `x_vals` vstack line (Type-2 carries per-point x) |
| 8 | `split_train_test` (860–960 / 1369–1479) | NEAR-IDENTICAL wrapper / divergent core | Precondition block (6 asserts) and trailing feature-setting block are byte-identical. Divergent core: ObjectiveGP indexes theta directly (one row per theta); EmulatorGP must expand theta-level indices to full `(theta,x)`-row boolean masks via `get_unique_theta()` + `np.isin` |
| 9 | `get_dim_gp_data` (812–828 / 1305–1334) | GENUINELY DIVERGENT (1-line) | `theta_dim` vs `theta_dim + x_dim` — same fact as #3, viewed from dimensionality instead of construction |

**Proposed shared-base design** (Template Method — hooks on `GPEmulator`, overridden by each subclass):

```python
class GPEmulator:
    def _resolve_target(self, target):
        """Shared target -> {"test","val","cand"} -> self.test_data/gp_val_data/cand_data
        dispatch, used by predict_sse/expected_improvement. Replaces 4 byte-identical
        if/elif blocks (1017-1024, 1131-1138, 1585-1592, 1792-1799)."""

    # ---- abstract hooks; subclasses override ----
    def _gp_input_dim(self): raise NotImplementedError            # #9
    def featurize_data(self, data): raise NotImplementedError      # #3 (no shared body -- keep as override)
    def _sse_from_prediction(self, data, prediction, covar=False, **kw): raise NotImplementedError  # #2
    def _best_error_from_train_data(self, **kw): raise NotImplementedError  # #6
    def _select_train_test_rows(self, train_idx, test_idx): raise NotImplementedError  # #8 core
    def _extend_train_data_arrays(self, new_data): raise NotImplementedError  # #7 core
    def _compute_ei(self, data, exp_data, ep_bias, best_error_metrics, gp_prediction, **kw): raise NotImplementedError  # #5

    # ---- templated methods, now shared verbatim (moves ~150-200 currently-duplicated lines up) ----
    def get_dim_gp_data(self): ...        # calls self._gp_input_dim()
    def predict_sse(self, target=None, data=None, covar=False, prediction=None, **kw): ...
    def calc_best_error(self, **kw): ...
    def expected_improvement(self, target=None, data=None, exp_data=None, ep_bias=None,
                              best_error_metrics=None, gp_prediction=None, **kw): ...
    def split_train_test(self, sep_fact, shuffle_seed=None): ...
    def append_training_point(self, new_point_data): ...
```

Each subclass shrinks to exactly the 7 hooks above — the genuinely-divergent logic, nothing
else. Safe to extract incrementally (one hook at a time); each public method already has a
docstring stating pre/post-conditions to pin against.

**Extra duplication noticed incidentally** (not in the original 9-pair list, worth folding into
the same cleanup):
- The blockwise-SSE reshape-and-sum-of-squares pattern (`len_theta`/`len_x`/`indices`/`n_blocks`
  → reshape → subtract `exp_data.y_vals` → square → sum) is **independently reimplemented 3
  times**: `EmulatorGP.__eval_gp_sse_var` (emulators.py:1519–1534), `EmulatorGP.calc_best_error`
  (emulators.py:1665–1678), and `Simulator.to_sse_data` (simulator.py:736–770). Propose a shared
  `blockwise_sse(y_vals, exp_y_vals, n_blocks, len_x)` helper — the most valuable "extra" fix
  given how error-prone reshape math is to keep in sync by hand.

### 1.2 `__vector_to_1D_array` — `data.py` vs `simulator.py`

Byte-for-byte identical (including docstring) in two unrelated classes that don't share
inheritance:
- `Data.__vector_to_1D_array` — `data.py:158-175`
- `Simulator.__vector_to_1D_array` — `simulator.py:305-322`

**Proposal**: new `src/emcal/_utils.py` with a module-level `vector_to_1D_array(array)`
function; both files import and call it, deleting their private methods. (Not a staticmethod
on `Data` — `Simulator` importing `Data` just to reach a generic reshape utility would be a
needless coupling.) Call sites to update: `data.py:298` (1 site) and `simulator.py:437, 441,
490, 599, 603, 631, 688` (7 sites).

---

## 2. Naming (candidates only — no fixes proposed; for adjudication)

### Cross-cutting (same concept, multiple spellings — flagged once, propagates from `config.py`'s `BOConfig` fields)
- **`retrain_GP`** — uppercase `GP` in snake_case. Origin `config.py:37,77,106-107,143-144`;
  propagates to `driver.py:190,1377`, `analysis.py:946`, `emulators.py` (16 sites).
- **`DateTime`** — PascalCase field among snake_case. Origin `config.py:49,83,146,156-157`;
  `driver.py:1366`, `analysis.py:952,973`.
- **`get_y_sse`** — boolean phrased as a verb, no `is_`/`has_` prefix. `config.py:57,87`;
  `driver.py:890,992,1228,1245,1248,1388`; `analysis.py:1112,1140,1176`.
- **`w_noise`** — cryptic abbreviation, boolean with no prefix. `config.py:60,88`;
  `simulator.py:519,542`; `driver.py:353,752,787,890,894,914`.
- **lengthscale/outputscale family** — ≥4 spellings for the same concepts: `lenscl` (singular)
  vs `lenscls` (plural, `gp_backend/base.py:28-29,33`) vs `lengthscale(s)` (spelled out,
  `base.py:34,43`; `emulators.py:442,525`) vs `outputscl` vs `tau` (cryptic 1-letter alias,
  `base.py:28,33`; `emulators.py` 9 sites) vs `outputscale`/`outputscl_final`.
- **idx/idc abbreviation** — `train_idx`/`test_idx` vs `train_idc`/`org_train_idcs` within the
  same methods (`data.py:338` vs `349-350`; `emulators.py:911,930` vs `1041,1061-1062,1064`
  vs `1420,1426,1432-1433`); plus a genuine **misspelling** `indeces_to_consider` (not just
  abbreviation) at `case_studies.py:533`, `analysis.py:1046,1048`, coexisting with the correct
  `indices_to_consider` elsewhere.
- **"best error" family** — `best_error`, `be_theta`/`be_list`, `best_err_data`,
  `best_error_metrics`, `best_errors_x` — 4+ styles for one concept (`acquisition.py:103-104`
  has `self.be_theta` directly beside `self.best_error_x` for adjacent tuple elements).
- **seed family** — `set_seed` (param) → `self.seed` (attr) vs `shuffle_seed`/`rng_seed`/
  `base_seed`/`sim_x_seed`/`val_seed`/`start_seed` for related-but-distinct seed concepts
  scattered across `data.py`, `diagnostics.py`, `simulator.py`, `emulators.py`.

### By file (high-confidence, non-cross-cutting)
- **`results.py`**: `simulator_class`(23,36-37,52,64), `exp_data_class`(24,38,53,65),
  `list_gp_emulator_class`(25,40,54,56-57,69) — all hold **instances**, not classes; misleading
  `_class` suffix. `why_term`(27,46,60,68) — awkward/cryptic.
- **`data.py`**: `as_covar`(29,32-33) — boolean without `is_`/`has_` prefix. Line 115's assert
  message references a nonexistent `ei` identifier (stale text from before `acq` rename).
- **`diagnostics.py`**: `hyperparameters`/`hyper`/`trained_hyperparams` — 3 names, one concept,
  ~150 lines apart. `ep_inc` (32) vs docstring's `e_inc` (51) — doc/signature drift.
- **`exploration.py`**: clean; the `ep_*` family used consistently — good reference point.
- **`simulator.py`**: `x_data`(599,603,607-608,639) inconsistent with pervasive `x_vals`;
  `sim_theta_vals`(631) one-off vs plain `theta_vals` elsewhere; `lower_bound`/`upper_bound`
  (220-221) inconsistent with the `_l`/`_u` suffix convention used everywhere else; docstring
  at 402 claims `gen_meth_x: bool` but it's a `GenMethod` enum.
- **`acquisition.py`**: `ei_temp` (5 sibling methods) vs `ei_mean` (renamed within
  `__calc_ei_mc` for the same role); `ci_l`/`ci_u`(844-845) vs DataFrame columns
  `ci_lower`/`ci_upper`(808-809) one line away; `ns`(865) cryptic 2-letter param; docstring
  references a nonexistent `set_seed` param.
- **`driver.py`**: `__make_BO_results_temp`(150) — sole uppercase `BO` (siblings use lowercase
  `bo`). `MSE_acq_obj_act`/`MSE_obj_act`/`MSE_obj_gp`(987,994,1003) — uppercase locals vs
  lowercase DataFrame columns for the same quantities. `fileObj1`/`fileObj2`(1352,1356,1458,1464)
  — camelCase. `improvement` used as **bool** in one method vs **numeric magnitude** in
  another (950-954 vs 1119-1137) — same name, different type across methods. Class docstring
  (44-46) references a `gp_model` param that doesn't exist in the actual signatures.
- **`analysis.py`**: `General_Analysis`(117) — embedded-underscore class name (mixed
  Pascal/snake). `get_ei`(1112,1140,1176) — verb-phrased boolean unlike `is_job_like`/
  `data_needs_ei` elsewhere in the same file. `cand_pred`(1433-1434) abbreviates "prediction"
  while sibling vars (`test_prediction`, `hm_prediction`, `sse_prediction`) spell it out.
- **`emulators.py`**: `self.scalerX`/`self.scalerY`(153-154), `org_scalerX`(254) — camelCase.
  `covar`(bool param, 18 sites) reads as a noun, no boolean prefix despite being asserted
  `isinstance(covar, bool)` everywhere. Assert message text `"...Exploration_bias"` (1144,1805)
  misnames the actual class `ExplorationBias`.
- **`gp_backend/gpflow_backend.py`**: `gpKernel`(32,36,38,40) — the single clearest camelCase
  violation in the package, directly beside correctly-cased `gp_model`.
- **`case_studies.py`**: `xP`(1118,1125,1154-1155), `lngC`/`lngR`(1176,1179) — camelCase-flavored
  in an otherwise snake_case file. `l`(1150) — classic ambiguous single-letter (confusable with
  `1`/`I`). `CSMuller`(class) breaks the uniform `CS`+number pattern.
- **`plotting.py`**: `title_fntsz`/`other_fntsz`(74-75) vs spelled-out `x_size`/`y_size`(887-888)
  for an analogous concept. Docstring says `save_path_to`(41) but the actual param is
  `save_path`(843) — 4 call sites use the stale docstring name. String literal
  `"Aquisition Function"`(393,812) — misspelled **user-facing plot label**, not just a comment.

---

## 3. Dead code / complexity / bloat

### 3.1 Confirmed dead write
`driver.py:597` — `self.__min_obj_class.acq = obj`, inside `__scipy_fxn`. Confirmed via
package-wide grep: every reader of this object's `.acq` was moved to `self.__min_obj_val` in
the 7C refactor (set alongside `__min_obj_class` at lines 539/544/549/560/588/596, read at
414/542/547/553/565). No code path anywhere reads `.acq` off this specific object (other
`.acq` sites in `data.py:133`, `analysis.py`, `simulator.py:753` are unrelated objects). Safe
to delete along with the explanatory comment block at 527-532.

### 3.2 Unused imports
- **`GPBO_Classes_New.py`** (pure re-export shim) — lines 2,3,5,8,9,10,11,12,13,14,15,16,17,
  18,19,20,21,22,23,24,25,26: `random`, `default_rng`, `datetime`, `math`, `norm`,
  `multivariate_normal`, `integrate`, `optimize`, `distance`, `os`, `time`, `StandardScaler`,
  `PowerTransformer`, `RobustScaler`, `qmc`, `pandas`, `Enum`, `dataclass`, `pickle`, `gzip`,
  `itertools`, `combinations`, `copy`, `scipy`, `plt` — only `np`/`warnings` actually used.
- `driver.py:6,11,15` — `scipy` (bare), `os`, `Enum`; `driver.py:17` — `Kernel` (from the
  `enums` import).
- `emulators.py:7,9` — `copy`; `StandardScaler`, `PowerTransformer`.
- `acquisition.py:7` — `multivariate_normal` (code uses `rng.multivariate_normal`, unrelated).
- `analysis.py:6,8,9,10` — `literal_eval`, `MinMaxScaler`, `pdist, squareform`, `string`.
- `case_studies.py:2,3,4,5` — `qmc`, `pandas`, `math`, `field`.
- `plotting.py:1,12,13,15` — the shadowed `from matplotlib import pyplot as plt` (line 10
  re-imports it unaliased-equivalent), `matplotlib.ticker as ticker` (unaliased import at
  line 6 is what's used), `Data`/`MethodName`, `JobContext` (only in a string literal).
- `exploration.py:4,7` — `numpy` (zero `np.` usages), `EpSchedule` (only in a string literal).
- `simulator.py:12,13` — `Data` (docstring-only), `GPBOMethod` (docstring-only).
- `gp_backend/gpflow_backend.py:10` — `tensorflow_probability as tfp` (only `bijectors as tfb`
  is used).
- Clean: `__init__.py`, `config.py`, `data.py`, `diagnostics.py`, `enums.py`, `methods.py`,
  `results.py`, `gp_backend/base.py`.

### 3.3 Unreachable branches
Three identical chained-comparison bugs, each making a `raise ValueError` unreachable
(`Enum in range(1,4) == False` parses as `(Enum in range(1,4)) and (range(1,4) == False)`,
always `False` — verified at runtime):
- `enums.py:42-43` (`Kernel`)
- `enums.py:61-62` (`GenMethod`)
- `enums.py:81-82` (`EpSchedule`)

No `if False:`, no dead code after unconditional return/raise, no except-duplicates-try
anti-pattern found anywhere in the package (verified via full AST walk).

### 3.4 Unused parameters
Genuine bloat:
- `emulators.py:39,51-54` — `GPEmulator.__init__`'s `__feature_train_data`/`_test_data`/
  `_val_data`/`_cand_data` params are accepted then unconditionally overwritten with `None`
  (155-158) regardless of what's passed.
- `emulators.py:1616` — `EmulatorGP.calc_best_error`'s `method` param used only in an
  `isinstance` assert, never in the computation.
- `emulators.py:1481` — `EmulatorGP.__eval_gp_sse_var`'s `method` param appears only in the
  docstring, never referenced in the body.
- `acquisition.py:778` — `__calc_ei_mc`'s `gp_mean` param unused (method uses `self.gp_mean`).
- `acquisition.py:730` — `__get_sparse_grids`'s `alpha` param documented but unused.

Likely intentional (interface parity — not flagged as bugs): `plotting.py:308`
`custom_format(self, x, pos)`'s `pos` (required by `matplotlib.ticker.FuncFormatter`'s calling
convention); `case_studies.py`'s `calc_cs*_...(..., args=None)` family (8 sites) — fixed 3-arg
callback signature shared polymorphically across all case-study model functions.

### 3.5 Unused instance attributes
- `data.py:129,132` — `Data.gp_covar`/`sse_covar` initialized `None`, written by `analysis.py`
  in the `gp_covar` case, never read back as an attribute anywhere; `sse_covar` never
  read/written at all outside `__init__`.
- `data.py:330` — `Data.seed` (set in `train_test_idx_split`) never read back.
- `acquisition.py:103` — `ExpectedImprovement.be_theta` set, never read.
- `emulators.py:155-158` — the 4 mangled `__feature_*` attributes (pairs with 3.4's dead
  params) — the real working attributes are the separate unmangled `feature_train_data` etc.
  set later via `split_train_test`.
- `analysis.py:188` — `General_Analysis.study_results_dir` set, never read.
- `plotting.py:73,76-84,94-102` — `Plotters.zbins`, `.colors`, `.gpbo_meth_dict` — all set,
  never read.
- `results.py:67` — `BOResults.max_ei_details_df` — stored, never read back off the object.
- Lower-confidence: `case_studies.py` — `self.param_name_str` set in 9 case-study classes but
  only read back by `CSMuller`'s own copy — possibly intentional per-case-study metadata.

### 3.6 Commented-out code (3+ consecutive lines)
`emulators.py:493-496` (debug print), `emulators.py:644-646` (`grad_mean` unscale),
`analysis.py:256-260` (alternate `result_dir`), `analysis.py:267-269` (regex alternative),
`analysis.py:563-565`, `analysis.py:567-572`, `analysis.py:581-585` (three blocks, all
signac-era leftovers), `analysis.py:1483-1489` (debug print), `plotting.py:387-389`,
`plotting.py:1072-1074`.

### 3.7 Misc bugs noticed in passing (not dead code, but worth flagging alongside)
- `analysis.py:227-229` — `sorted_dict` computed in `make_dir_name_from_criteria` but the loop
  below iterates the original unsorted dict instead — dead/misleading local.
- `analysis.py:406` — `dirname` assigned, never used (the analogous line in `save_data` does
  use it).
- `analysis.py:559` — calls `warnings.warn(...)` but `warnings` is never imported in
  `analysis.py` — **latent `NameError`** if that branch ever executes.
- `acquisition.py:175` — `eigvecs` unpacked from `np.linalg.eigh(covar)`, never used.

### 3.8 The `GPBODriver` god-class (1466 lines) — decomposition proposal

**Responsibility inventory** (methods grouped by cluster, current line numbers):

| Cluster | Methods | Span |
|---|---|---|
| A. Acquisition/scipy optimization | `__make_starting_opt_pts`(235), `__gen_start_pts_mc_sparse`(258), `__gen_start_pts_not_mc_sparse`(327), `__opt_with_scipy`(343), `__scipy_fxn`(426) | ~365 lines |
| B. Best-error tracking | `__get_best_error`(197) | ~37 lines |
| C. Heat-map diagnostics | `create_heat_map_param_data`(601, public) | ~128 lines |
| D. Per-iter results + train augmentation | `__augment_train_data`(730), `create_data_instance_from_theta`(742, public), `__run_bo_iter`(810) | ~297 lines |
| E. Workflow orchestration | `__make_BO_results_temp`(150), `__run_bo_to_term`(1028), `__run_bo_workflow`(1269), `run`(1328, public), `save_results_run`(1446, public) | ~460 lines |
| F. Emulator factory delegation | `__gen_emulator`(171) — thin wrapper around the already-extracted `build_gp_emulator` | ~25 lines |
| G. RNG/seed management | `reset_rng`(1319, public) | ~9 lines (footprint much larger — read by A/D, serialized by E) |
| `__init__` | — | ~97 lines |

`__init__` sets injected collaborators, tuning constants, and the `__min_obj_*` scratch
triplet (all `None`) — but notably does *not* initialize `rng_set`, `opt_start_pts`, or
`gpbo_res_simple`/`gpbo_res_GP`, which only come into existence later via other methods. That
implicit-initialization pattern is itself a minor smell worth fixing during the split.

**Proposed decomposition** (mirrors the `build_gp_emulator` extraction precedent):

- **`AcquisitionOptimizer`** (new `src/emcal/acquisition_optimizer.py`, kept separate from the
  pure-math `acquisition.py` for the same reason `emulators.py` stayed separate from it) — owns
  `min_obj_class`/`min_obj_val`/`min_obj_prediction`/`opt_start_pts`. Does **not** cache
  `gp_emulator` or `rng_set` (see risks). Methods: `make_starting_points(...)`,
  `optimize(opt_obj, get_y=False, w_noise=False, rng=None)`, private `_objective(...)` callback.
- `create_data_instance_from_theta`/`__get_best_error` → free functions (same pattern as
  `build_gp_emulator`), injected into `AcquisitionOptimizer` as a `create_data_fn` dependency
  rather than hardcoded.
- `create_heat_map_param_data` → free function in `emulators.py` or a new `diagnostics.py`
  companion.
- **`ResultsRowBuilder`** (new, in `results.py`) — `build_iteration_row(...)` absorbs the
  dataframe-assembly tail of `__run_bo_iter` (970-1026); fixes the `column_names` list
  duplicated verbatim between `__run_bo_iter`(935-947ish) and `__run_bo_to_term`(1056-1069) as
  a side effect.
- **What stays on `GPBODriver`**: cluster E (orchestration). Post-split size: ~460 (cluster E)
  + `__init__` + thin delegation calls — a reasonable orchestrator, down from 1466.

High-level shape after the split:
```python
def __init__(self, ...):
    self.rng_set = None
    self.acq_optimizer = AcquisitionOptimizer(cs_params, method, simulator, exp_data,
                                               ep_bias, sse_penalty, sg_mc_samples)
    self.gpbo_res_simple, self.gpbo_res_GP = [], []

def __run_bo_workflow(self, run_num, job=None):
    ...
    self.acq_optimizer.gp_emulator = self.gp_emulator   # explicit re-sync, never cached

def __run_bo_iter(self, iteration):
    best_err_data, best_error_metrics = self.acq_optimizer.get_best_error()
    self.acq_optimizer.opt_start_pts = self.acq_optimizer.make_starting_points(best_error_metrics, iter_seed)
    min_sse, min_theta_data, min_sse_prediction = self.acq_optimizer.optimize("sse", rng=self.rng_set, ...)
    opt_acq, acq_theta_data, best_prediction = self.acq_optimizer.optimize("neg_ei", rng=self.rng_set, ...)
    iter_df = build_iteration_row(...)
```

**Riskiest seams** (ranked by how silently they'd fail):

1. **`self.gp_emulator`** — fully *replaced* in `__run_bo_workflow` (not just mutated), then
   mutated in place every iteration (`fit()`, `append_training_point()`). If a collaborator
   caches it at construction, it holds a stale/`None` reference after the real emulator is
   built — best case a loud `AttributeError`, worst case (2nd+ restart) *silently* optimizing
   against the previous restart's stale GP. Fix: never cache; re-sync explicitly after both
   replacement sites, guarded by an identity-assertion test.
2. **`self.rng_set`** — created late (only via `reset_rng()`/checkpoint-restore), read by
   clusters A/D, serialized/restored by cluster E. If the optimizer captures its own reference
   instead of the live one, RNG streams desync from pickled checkpoints — silently
   non-reproducible, not a crash. Fix: never let the optimizer own it; pass the `Generator`
   explicitly into every call that needs it.
3. **`__min_obj_class`/`__min_obj_val`/`__min_obj_prediction`** — reset at the top of every
   `__opt_with_scipy` call, mutated only in `__scipy_fxn`, read only at that same call's end —
   already scoped to one call, the safest of the three to move verbatim. Risk: `__run_bo_iter`
   calls `optimize` twice back-to-back per iteration (lines 890/894) — dropping the
   reset-then-accumulate-then-read pattern would leak stale state between those two calls.
4. **`self.opt_start_pts`** — set once by cluster D, deliberately *reused* across both
   `optimize` calls in the same iteration. Splitting `make_starting_points`/`optimize` onto
   different objects risks a call site silently reusing stale points from a prior iteration —
   no crash, just quietly degraded optimization. Fix: keep both on the same
   `AcquisitionOptimizer`; don't "fix" the reuse without a pinning test first.
5. **`self.gpbo_res_simple`/`self.gpbo_res_GP`** — the checkpoint/resume backbone, written
   every iteration (not just per-restart) and read by the resume branch. Moving this to a
   separate checkpoint manager risks losing per-iteration save granularity needed for
   crash-resume correctness — a durability bug, harder to catch without an explicit
   interruption test. Recommendation: leave ownership on `GPBODriver` in the first pass.

---

## 4. Test-gap & coverage plan (baseline 32% → target ~80%)

### 4.1 Baseline (measured this session, `pytest -m "not slow" --cov=emcal`)

```
Name                                     Stmts   Miss  Cover
----------------------------------------------------------------------
GPBO_Classes_New.py                         36      0   100%
__init__.py                                  8      0   100%
gp_backend/base.py                          17      0   100%
methods.py                                  19      0   100%
data.py                                    104      3    97%
results.py                                  24      1    96%
config.py                                   59      3    95%
case_studies.py                            394     26    93%
enums.py                                    27      3    89%
simulator.py                               202     52    74%
exploration.py                              79     21    73%
diagnostics.py                             142     67    53%
gp_backend/__init__.py                       9      5    44%
acquisition.py                              248    223    10%
emulators.py                               505    457    10%
driver.py                                  463    423     9%
analysis.py                                562    562     0%
plotting.py                                381    381     0%
gp_backend/gpflow_backend.py                46     46     0%
----------------------------------------------------------------------
TOTAL                                     3325   2273    32%
```

Two structurally different kinds of gap: (a) modules with **zero pytest coverage but real
devtools-net coverage** (`analysis.py`, `plotting.py` — every statement is already exercised
by `devtools/verify_analysis.py`'s 13 checks, just not inside `pytest`); and (b) modules whose
**orchestration logic is entangled with the real gpflow backend** (`emulators.py`,
`driver.py`, and transitively `acquisition.py`) so only the `@pytest.mark.slow` end-to-end
tests reach them today. `gp_backend/gpflow_backend.py` is genuinely integration-only — it
should **stay** slow/golden-covered, not faked (faking the backend under test defeats the
point of testing it).

### 4.2 `FakeGPBackend` — deterministic backend for fast unit tests

`src/emcal/gp_backend/base.py`'s `GPBackend` ABC has exactly 7 abstract methods:

```python
configure(self)
make_bounded_parameter(self, low, high, initial_value)
build_model(self, data, kernel_value, lenscls, tau, white_var, fix_lengthscale, fix_outputscale, noise_variance=1e-5)
train(self, model)                          # -> (success: bool, training_loss: float)
get_hyperparameters(self, model)             # -> [lengthscale: np.ndarray, noise: float, outputscale: float]
make_posterior(self, model)
predict_f(self, posterior, eval_points, full_cov=True)   # -> (mean, covar), covar's leading singleton dim squeezed
```

All model/posterior objects are opaque to `GPEmulator` — never introspected, only ever passed
back to the same backend instance. This means a `FakeGPBackend` can use **trivial opaque
objects** (plain dicts or `SimpleNamespace`) as "model"/"posterior" and return **canned,
deterministic** mean/covariance arrays from `predict_f`, with zero gpflow/TF dependency:

```python
class FakeGPBackend(GPBackend):
    name = "fake"
    def __init__(self, mean_fn=None, covar_fn=None):
        # mean_fn(eval_points) -> np.ndarray, covar_fn(eval_points) -> np.ndarray (SPD);
        # sane defaults (e.g. mean_fn = lambda x: x.sum(axis=1), covar_fn = lambda x: np.eye(len(x)))
        # let most tests not even need to pass these.
    def configure(self): pass
    def make_bounded_parameter(self, low, high, initial_value): return initial_value
    def build_model(self, data, *a, **kw): return {"data": data}         # opaque dict
    def train(self, model): return True, 0.0                              # always "succeeds"
    def get_hyperparameters(self, model): return [np.array([1.0]), 1e-4, 1.0]
    def make_posterior(self, model): return model                        # pass-through
    def predict_f(self, posterior, eval_points, full_cov=True):
        mean = self._mean_fn(eval_points)
        covar = self._covar_fn(eval_points)
        return mean, covar
```

Registering it (test-only, doesn't touch `gp_backend/__init__.py`'s real registry): construct
`ObjectiveGP`/`EmulatorGP` normally, then monkeypatch/inject `emulator._backend =
FakeGPBackend(...)` before calling `fit()` — check how `_backend` is currently wired in
`GPEmulator.__init__`/`set_gp_model` to confirm the cleanest injection point (likely a
constructor kwarg `backend=` added to `GPEmulator.__init__`, defaulting to
`get_backend("gpflow")` as today, so no production code path changes).

**What this unlocks**: `ObjectiveGP`/`EmulatorGP`'s `predict`/`predict_sse`/
`expected_improvement`/`split_train_test`/`calc_best_error`/`append_training_point`/
`featurize_data` — i.e. essentially all of `emulators.py`'s orchestration — become testable in
milliseconds with known, hand-computable expected outputs. Layered one level up,
`GPBODriver`'s `__scipy_fxn`/`__opt_with_scipy`/`__run_bo_iter` become testable too (a full BO
iteration against a `FakeGPBackend` with a trivial mean function, asserting the results
DataFrame shape/columns and that training-data augmentation grew by 1 row) — this is the
"emulator + driver ORCHESTRATION... without gpflow" the task calls for.

### 4.3 `analysis.py`/`plotting.py` — bring the existing devtools-net coverage into pytest

Every method in the current (post-7B-trim) `analysis.py`/`plotting.py` is already exercised
by `devtools/verify_analysis.py`'s `build_fixture()`/`run_analysis()` pattern — a `JobContext`
wrapping a small real (or `FakeGPBackend`-produced) `GPBODriver.run(job=None)` result. That
pattern should become a pytest **fixture**, not stay devtools-only:

```python
@pytest.fixture
def cs1_method7_job(tmp_path):
    """Mirrors devtools/verify_analysis.py's build_fixture(): a small deterministic
    CS1/method-7 run saved to a JobContext workspace."""
    ...
    return JobContext(ws, statepoint, job_id="test")
```

Target methods (all currently 0% in pytest, exercised only via the devtools net):
`General_Analysis.get_run_dataframe`, `.best_error`, `.objective_trajectories`,
`.parameter_trajectories`, `.hyperparameter_trajectories`, `.gp_parity_data`,
`.gp_heat_map_data` (all branches — the method-1/method-6 fixtures from 7C-a's guard step are
the template for exercising the ObjectiveGP/SG-MC branches specifically); `Plotters.
plot_hyperparameters`, `.plot_parameters`, `.plot_gp_fit` (via the same `_capture_shown_figure`
technique already in `devtools/verify_analysis.py`, moved into a pytest helper/fixture).
This alone could bring `analysis.py`+`plotting.py` from 0%/0% to an estimated 60-80% (some
branches — the `self.save_csv`-triggered disk-cache paths, the sparse-grid `Tasmanian` path —
are legitimately harder to reach and lower priority).

### 4.4 `acquisition.py` — EI math is pure, unit-testable on synthetic arrays today

`ExpectedImprovement.__init__` takes only plain arrays (`gp_mean`, `gp_covar`), an
`ExplorationBias`, a `Data` (for `exp_data.y_vals`), a `best_error_metrics` tuple, a seed, and
an optional `GPBOMethod` — **no GP, no gpflow, no backend dependency at all**. `compute()`
dispatches to `__compute_standard()` (method=None — a closed-form z-score/norm.cdf/norm.pdf
calculation on `gp_mean`/`gp_var`/`best_error`, ~15 lines, hand-verifiable) or
`__compute_emulator()` (method given — dispatches further to `__calc_ei_emulator`/
`__calc_ei_log_emulator`/`__calc_ei_sparse`/`__calc_ei_mc` per method type). All of
`ExplorationBias`/`Data`/`GPBOMethod` already have simple, fast-tested constructors
(`test_boconfig.py`, `test_data.py`, `test_enums_and_method.py`). Priority order: (1)
`__compute_standard` on hand-picked `gp_mean`/`gp_var` with a known best_error → assert EI
against the closed-form formula; (2) `__calc_ei_emulator`/`__calc_ei_log_emulator` (methods
3/4, Gaussian-quadrature-free); (3) `__calc_ei_sparse`/`__calc_ei_mc` (methods 5/6 — these use
`Tasmanian`/Monte-Carlo sampling, so pin with a fixed seed and a generous tolerance rather than
exact equality). This is the single highest-value, lowest-risk coverage target in the whole
package — pure math, currently at 10%, realistically reachable to 70-80%+.

### 4.5 `exploration.py` / `simulator.py` / `diagnostics.py` — smaller, targeted gaps

- **`exploration.py`** (73% → ~95%+): missed lines 210, 233-251, 266-278 are
  `__set_ep_boyle`/`__set_ep_jasrasaria` — pure-float heuristics, no dependencies, same pattern
  as the existing `test_decay_schedule_moves_from_ep0_toward_ep_f` test. Cheap, high-confidence
  wins.
- **`diagnostics.py`** (53% → ~90%+): missed lines 77-157 are `GPDiagnostics.
  _calibration_verdict`/`.summary`/`.__str__`/`.parity_plot`/`.calibration_plot`/
  `.residual_plot`/`.plot_all`. These operate on a `GPDiagnostics` built directly via
  `_diagnostics_from_arrays(actual, predicted, std, n_train, label, hyper)` on synthetic
  numpy arrays — exactly the pattern the *existing* passing tests already use for the metrics
  (`test_metrics_on_perfect_predictions` etc.) — just extend it to also call the plotting
  methods under the Agg backend (same `_capture_shown_figure`-style technique as 4.3). Missed
  lines 241-322 (`fit_gp`/`evaluate_gp`/`cross_validate_gp`) genuinely need a real/fake GP —
  low priority, `FakeGPBackend`-testable later if desired, otherwise leave golden-covered.
- **`simulator.py`** (74% → ~85%): the missed lines are scattered edge-case branches inside
  otherwise-tested methods (`evaluate_model`, `generate_simulation_data`, `to_sse_data`) —
  noise-handling branches and specific `GenMethod` combinations not hit by the current fast
  tests. Needs a closer per-branch look at test-build time; not as high-value as 4.2-4.4.

### 4.6 Prioritized target list (to reach ~80% without faking genuinely-integration paths)

Excluding `gp_backend/gpflow_backend.py` (46 stmts, stays slow/golden-covered by design), the
fast-suite denominator is 3279 statements; 80% of that is ~2623 covered / ~656 missed
allowed, vs. today's 2227 missed (excluding the backend file) — i.e. **~1571 more statements**
need coverage. Suggested order (highest value-per-effort first):

1. **`acquisition.py`** (223 missed) — pure math, zero new fixtures needed. *Do first.*
2. **`analysis.py` + `plotting.py`** (562 + 381 = 943 missed) — one `JobContext` pytest
   fixture (ported from `devtools/verify_analysis.py`) unlocks both files at once.
3. **`emulators.py`** (457 missed) — needs `FakeGPBackend` built first (4.2); then covers
   `ObjectiveGP`/`EmulatorGP` orchestration end-to-end.
4. **`driver.py`** (423 missed) — layers on top of `FakeGPBackend` + the emulator tests; covers
   `__scipy_fxn`/`__opt_with_scipy`/`__run_bo_iter` orchestration.
5. **`diagnostics.py`** (67 missed) + **`exploration.py`** (21 missed) — cheap, no new
   infrastructure, can slot in anywhere (even before #1 if a quick early win is wanted).
6. **`simulator.py`** (52 missed) — lowest priority of the substantial gaps; scattered branches,
   needs individual attention rather than one unlocking pattern.

Items 1+2+5 alone (223+943+67+21 = 1254 missed statements addressed) are achievable with **no
new test infrastructure** — just new test files/fixtures using patterns that already exist in
the codebase (`devtools/verify_analysis.py`'s `JobContext`/`_capture_shown_figure`, the
existing `test_diagnostics.py`'s synthetic-array construction). Items 3+4 (880 missed) require
building `FakeGPBackend` first but are the biggest single remaining lever and directly answer
the task's "emulator + driver orchestration without gpflow" ask.
