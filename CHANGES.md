# Changelog

All notable changes to this project will be documented in this file.

## [1.5.1] - 2026-04-24
* Fix chance-constrained environmental-cost mean computation by replacing fragile pandas sparse updates with direct sparse matrix updates and multiplication in `pulpo.utils.uncertainty.cc`.
* Update and re-run the Section 10 uncertainty showcase notebook to include the deterministic reference result used by `run_gsa` and validate the hotfix workflow end-to-end.

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
