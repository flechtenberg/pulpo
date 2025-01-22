# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] - 2025-01-??
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
