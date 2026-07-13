# Changelog

All notable changes to this project will be documented in this file.

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
