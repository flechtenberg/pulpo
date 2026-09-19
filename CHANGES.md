# Changelog

All notable changes to this project will be documented in this file.

## [1.8.0] - 2026-09-19

### Added
* **Exact chance-constrained optimization via a second-order cone** —
  `PulpoOptimizerUnc.create_SOC_formulation()` / `solve_SOC_problem()`. The
  existing `solve_CC_problem` bounds the impact's standard deviation by an `L1`
  sum over processes, which over-estimates it and treats the per-process
  contributions as independent although one characterization factor multiplies
  them all. The new formulation represents the standard deviation exactly,
  shared characterization factors included, so at the same reliability level it
  returns a front at or below the `L1` one. `method='cutting_plane'` (the
  default) solves a sequence of the ordinary LPs PULPO already builds and is
  what works at ecoinvent scale; `method='direct'` hands the cone to Gurobi in a
  single solve and suits small or well-conditioned systems. The cutting-plane
  solve reports a certified optimality gap per iteration.
  `restore_deterministic_objective()` returns the instance to the plain impact
  objective.
* **Closed-form moments** — `create_SOC_formulation(moments='closed_form')`.
  Mean and variance are computed analytically per distribution family instead of
  by refitting every parameter to a Normal through Monte Carlo: faster, and free
  of sampling noise in the coefficients.
* **Joint chance constraints** — `cc.bonferroni_budget()` and
  `apply_CC_formulation(risk_budget=...)`. Imposing each row at `lambda`
  individually controls no joint probability: with `K` constrained rows, the
  chance that at least one of them fails reaches `K(1 - lambda)`. A risk budget
  divides the failure probability among the rows so they hold *together* at
  `lambda`. It adds no variables and no constraints — only each right-hand side
  moves — and it needs no correlation estimates. Opt-in; the default behaviour
  is unchanged.
* **Exact quantiles for uncertain bounds** —
  `apply_CC_formulation(bound_quantile='exact')`. A chance constraint on a bare
  bound is exactly the declared distribution's own quantile and needs no
  Gaussian approximation. This matters at high reliability, where a
  moment-matched normal can demand a bound below the distribution's support — a
  negative capacity, in practice. Opt-in, and requires a risk budget.
* **Optional LP equilibration** — `instantiate(scale=True)`, off by default. An
  ecoinvent technosphere spans 1e-13 to 2e+11, because infrastructure processes
  have a functional unit of one whole facility. Solvers apply their feasibility
  tolerance per row relative to that row's largest coefficient, so such a
  facility can be under-supplied by more than its own activity level and still
  count as feasible — worth roughly 1 % of the optimum on an unaggregated
  system, with different solvers landing on different answers. No solver option
  repairs it. Equilibration rescales rows and columns by powers of two, which is
  exact in floating point, and the solution is unscaled after the solve, so
  `scaling_vector`, `impacts` and everything `extract_results()` returns stay in
  original units. It covers the uncertainty formulations too, so
  `solve_CC_problem` and `solve_SOC_problem` return the same fronts either way.
  Recommended for unaggregated ecoinvent backgrounds.
* `import_and_filter_uncertainty_data(scaling_vector_strategy='none')` — retain
  every declared uncertain parameter. The contribution filter is a device for
  large systems; on a small one it strips the uncertainty from alternatives that
  happen to be inactive at the vector it filters on, which biases the subsequent
  risk-averse choice towards exactly those alternatives.
* `run_gsa(seed=...)` — the SALib sampler's seed is now controllable from the
  façade instead of fixed (still 161 by default, so previously reported indices
  are unchanged).
* `pulpo/datasets/soc_demo_database.py` — an open six-activity demonstration
  system covering every supported uncertainty family, usable without an
  ecoinvent licence.

### Changed
* `solve_gurobi` no longer recommends `ScaleFlag=2` / `NumericFocus=3`. On an
  ecoinvent-scale model those made the optimum worse rather than better, because
  the feasibility tolerance is still applied relative to the 1e11 facility
  coefficients. On a scaled model it now applies `ScaleFlag=0`,
  `FeasibilityTol=OptimalityTol=1e-9` and `NumericFocus=1` unless the caller
  overrides them (`scaling.GUROBI_OPTIONS_SCALED`); add `Method=1` for
  bit-identical repeated solves.
* Under `scale=True`, process bounds inherited from `default_limits` become
  infinite, with a warning saying how many. A finite `upper_bound=1e9` never
  binds anyway, but on a facility column it scales to 1e20 and degrades the
  solve. Explicit `lower_limit` / `upper_limit` values and choice capacities are
  kept.

### Fixed
* **An infinite variable bound was registered as an uncertain parameter.** Every
  choice alternative and every explicit limit was recorded without testing its
  value, so an alternative declared `float('inf')` — the default already used to
  mean "unlimited" — became an uncertain bound with an `inf` amount, a `nan`
  variance and a `nan` bound handed to the solver. Infinite bounds are now
  skipped, so an unbounded alternative no longer has to be written as a large
  sentinel capacity.
* **A deterministic characterization factor turned every sampled impact into
  `NaN`** in the global sensitivity analysis. A flow whose factor declares no
  distribution received an all-`NaN` column, and one such column makes every
  sample impact `NaN`. Such a factor is constant rather than missing, and is now
  filled with its value from the characterization matrix. Previously the only
  way to keep the flow was to gap-fill its factor with an invented spread.
* **`seed=` did not make a Monte Carlo reproducible unless every parameter was
  Normal.** Lognormal, triangular and uniform parameters drew from NumPy's
  global stream and ignored the seed, so any run containing one was
  irreproducible, `run_mc_from_uncertainty(seed=...)` included.

### Known issues
* `run_mc_from_uncertainty` draws its per-draw seeds from `[0, 10**6)`, so about
  164 of 20 000 draws are duplicates and the effective sample is ~0.8 % smaller
  than requested. Widening the range would shift published draw sequences, so it
  is deferred rather than bundled into this release.

## [1.7.0] - 2026-08-11
* Add a goal-programming objective (`objective='goal'`): minimize the average transgression of user-defined soft impact limits (`imp_goals`), e.g. for Planetary-Boundary-style budgets. Unlike `upper_imp_limit`, goals can be exceeded — the solver stays feasible and reports the transgression level per category instead. Available on both `PulpoOptimizer` and the time-extended `PulpoOptimizerTime` (goals apply to impacts aggregated across the whole time horizon), and carried through Monte Carlo re-instantiation.
* Report per-category goal results (impact, goal, transgression level) via `extract_results()["Transgressions"]`, `summarize_results()`, and the Excel export.
* Fix several `default_limits` / goal-programming interactions found while hardening the new objective:
  * `default_limits` no longer silently hard-caps categories that carry a goal (was causing infeasibility or artificially suppressed transgression, #33).
  * `default_limits` now persists across Monte Carlo re-instantiation instead of reverting to the built-in infinite defaults on every sample.
  * Goal limits accept numpy scalar types (e.g. from a DataFrame/array) and correctly reject `bool`.
  * `extract_transgressions()`'s empty-goal-case result now shares the same `Method`-indexed schema as the populated case.
  * `objective='goal'` with an empty goal set now raises a clear `ValueError` instead of a bare `ZeroDivisionError`.
* Fix Monte Carlo re-instantiation to forward every `instantiate()` argument instead of a hardcoded, drifting kwarg list — time-indexed workers were silently losing `time_steps`/`storage` on every MC sample and falling back to the static formulation.
* Fix bw25 uncertainty: stabilize `construct_scaling_vector_from_choices()` (NaN scaling vector from unmapped node ids, non-deterministic cutoff filtering from unstable dict ordering) so MC/CC/GSA results are deterministic and match the bw2 stack.
* Fix GSA on the bw25 stack under SALib 1.5 / numpy 2 (bump the `uncertainty` extra to SALib 1.5.1).
* Fix `import_data` to activate its `project` argument instead of silently using whatever bw2data project happened to be active.
* Dev: migrate the test suite to `pytest` with in-memory bw2data databases (faster, order-independent); see `tests/README.md`.

## [1.6.1] - 2026-07-13
* Fix a solver hang on Windows with pyomo >= 6.6: the huge finite default limits (±1e20 / ±1e24) made HiGHS log "treated as ±Infinity" warnings during model construction, which deadlocked pyomo's appsi output capture (solve stuck forever at zero CPU). Unspecified limits and supply-slack bounds are now truly infinite (`float('inf')`), which HiGHS, Gurobi, and GAMS all handle natively — the resulting LP is unchanged. The choices documentation now recommends `float('inf')` instead of `1e20` for unconstrained capacities.
* Modernize dependencies to enable Python 3.13: unpin numpy 2 (the `<2` cap now lives only in the legacy `bw2` extra, where bw2data 3.x needs it), relax pyomo to `>=6.8.0,<7` (6.7.3 still touches the removed `np.float_` under numpy 2), and replace the phantom `bw2data<=3.9.9` pin with `bw2data<4.0.0`.
* Skip `pypardiso` on macOS via an environment marker (no wheels there; scipy's solver is used instead).
* Use a stable sort for result ordering in the saver so extracted results are deterministic across numpy versions.
* Dev: configure pytest discovery and ignore project virtualenvs.

## [1.6.0] - 2026-07-10
* Add a time-dependent extension as a first-class feature via the new `pulpo.pulpo_time` module (`PulpoOptimizerTime`):
  * Time-indexed LP formulation with per-timestep demands, limits, and impacts, plus aggregated impact bounds across timesteps.
  * Storage / carry-over between timesteps via a product-by-product `K` matrix (`storage` argument), supporting the 4-activity CHARGE / HOLD / HOLD t-1 / DISCHARGE battery pattern.
  * Time-indexed result extraction and saving; toy battery examples (daily, two-week, and hourly intra-day scenarios) in `notebooks/`.
* Major model-construction speedup on ecoinvent-scale databases (several times faster instantiation):
  * Build Pyomo models directly as `ConcreteModel` instead of `AbstractModel.create_instance()`.
  * Embed the technology, intervention, and environmental cost matrices as plain float coefficients in `LinearExpression` constraint rows instead of per-entry mutable Params (near-zero coefficients no longer reach the solver).
  * Apply production-capacity, intervention-flow, and impact limits as variable bounds instead of explicit constraints, and create supply slack variables only for products with a specified supply.
  * The chance-constrained formulation updates environmental costs through the new `optimizer.update_env_cost()`, which rebuilds the impact constraints in place.
* Faster and more robust Brightway data import:
  * Build LCI matrices once per database rather than once per method, and push `retrieve_processes` filtering down to SQL.
  * Load all databases in a single LCA so that the database order does not matter.
  * Decouple Monte Carlo RNG seeds per matrix in the bw2 path and skip unused uncertainty parameters during MC sampling.
* Fixes and maintenance: numpy 2.0 compatibility, `highspy` pinned to 1.15.1, `pypardiso` added as dependency, `None` defaults instead of mutable dict arguments in `instantiate()`, uncertainty characterization-factor index rename fix.

## [1.5.2] - 2026-06-02
* Fix bw25 uncertainty parameter extraction: dynamically inspect datapackage resources by metadata instead of relying on hard-coded indices. This resolves failures when extracting uncertainty information from brightway databases that lack uncertainty distributions.
* Add comprehensive uncertainty handling to the rice-husk example database, matching the uncertainty specification pattern used in the sample database.
* Fix multi-database parameter accumulation in bw25 path: ensure both foreground and background database flows are included in the combined parameter array (previously only the last database's bio_params were retained).
* Refactor `bw_parser.py` for improved maintainability: move the rarely-used bw25 uncertainty handling utilities (`build_bw25_params()`, `BW25_PARAM_DTYPE`, `BW25_DISTRIBUTION_FIELDS`) to `pulpo.utils.utils`, keeping the primary import orchestration logic lean and focused.
* Add comprehensive test coverage for bw25 uncertainty extraction with and without complete uncertainty information in both sample and rice-husk databases.

## [1.5.1] - 2026-04-24
* Fix chance-constrained environmental-cost mean computation by replacing fragile pandas sparse updates with direct sparse matrix updates and multiplication in `pulpo.utils.uncertainty.cc`.
* Update and re-run the Section 10 uncertainty showcase notebook to include the deterministic reference result used by `run_gsa` and validate the hotfix workflow end-to-end.
* Move the sample and rice-husk database helpers from `tests/` into a new `pulpo.datasets` subpackage so that `pulpo.install_sample_db()` / `pulpo.install_rice_husk_db()` work from an installed wheel (previously they imported from `tests/`, which is not shipped).
* Fix wheel packaging: exclude `tests/` from the distribution and drop the redundant `utils*` include in `[tool.setuptools.packages.find]`.
* Declare the mutual exclusivity of the `bw2` and `bw25` extras via `[tool.uv] conflicts` so that `uv` can resolve the environment without picking one over the other.

## [1.5.0] - 2026-04-23
* Integrate full uncertainty analysis pipeline as a first-class feature via the new `pulpo.pulpo_unc` module (`PulpoOptimizerUnc`):
  * Import and filter uncertain LCI parameters from Brightway databases (`import_and_filter_uncertainty_data()`)
  * Apply uncertainty strategies to fill missing distributions (`apply_uncertainty_strategies()`)
  * Monte Carlo sampling from prepared uncertainty distributions (`run_mc_from_uncertainty()`)
  * Chance-Constrained optimization with Pareto front solving (`create_CC_formulation()`, `solve_CC_problem()`)
  * Global Sensitivity Analysis via Sobol indices (`run_gsa()`)
* Section 10 of the main showcase notebook (`notebooks/pulpo_showcase.ipynb`) demonstrates the complete uncertainty workflow on the methanol system.
* Enhanced Pareto front visualization (base case overlay, choice highlighting, grouping by process/product/location)
* L1-norm variance used in CC formulation
* Minor bugfixes and code cleanup

## [1.4.2] - 2025-06-06
* Enable the use of Gurobi solver
* Pass email as argument to NEOS solver
* Enable choices to be specified as dict of lists rather than dict of dicts

## [0.1.5] - 2025-01-26
* Convenience changes to optimization problem formulation
* Enable users to specify separate fore- and background databases to retrieve data from simultaneously.
* Several bugfixes
  * Instances can now be solved more than once
  * Enable negative slack values
  * Downgrade pyomo in requirements to facilitate highspy solution
* Switch from `setup.py` to `pyproject.toml` for package management

## [0.1.4] - 2025-01-05
* Resolve dependency issues revolving around scipy

## [0.1.3] - 2024-12-01
* Start tracing changes in changelog.
* Enable PULPO to use both bw2 and bw25 projects. Different install options are available via `pip install pulpo-dev[bw2]` or `pip install pulpo-dev[bw25]`.

## [0.0.1] - 2023-10-10
* Initial version .
