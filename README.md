<div align="center">

<img src="https://github.com/flechtenberg/flechtenberg_images/blob/main/Pulpo-Logo_INKSCAPE.png?raw=true" width="300" />

<h3>Python-based User-defined Lifecycle Production Optimization</h3>

<!-- Development Tools -->
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![Markdown](https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white)](https://www.markdownguide.org/)

<!-- Project Metadata -->
[![License](https://img.shields.io/github/license/flechtenberg/pulpo?style=flat&color=5D6D7E)](https://github.com/flechtenberg/pulpo/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/flechtenberg/pulpo?style=flat&color=5D6D7E)](https://github.com/flechtenberg/pulpo/commits/main)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/flechtenberg/pulpo?style=flat&color=5D6D7E)](https://github.com/flechtenberg/pulpo/pulse)

<!-- Additional -->
[![PyPI - Version](https://img.shields.io/pypi/v/pulpo-dev?color=%2300549f)](https://pypi.org/project/pulpo-dev/)
[![GitHub Stars](https://img.shields.io/github/stars/flechtenberg/pulpo?style=flat&color=FFD700)](https://github.com/flechtenberg/pulpo/stargazers)
[![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/v2/projects/fabian/pulpo-test/sessions/01JRM54S4NKMS84Y6BAYT832WH/start)

</div>

---

## 📍 Overview

**PULPO** is a Python package for **[Life Cycle Optimization (LCO)](https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.13561)** based on life cycle inventories. It is designed to serve as a platform for optimization tasks of varying complexity.

The package builds on top of the **[Brightway LCA framework](https://docs.brightway.dev/en/latest)** and the **[Pyomo optimization modeling framework](https://www.pyomo.org/)**.

---

## ✨ Capabilities

Applying optimization is recommended when the system of study has (1) many degrees of freedom that would otherwise prompt the manual assessment of a large number of scenarios, or (2) any of the following capabilities is relevant to the goal and scope of the study:

- **Specify technology and regional choices** throughout the entire supply chain (fore- and background), such as the production technology of electricity or the origin of metal resources. Consistently accounting for background changes in large-scale decisions [can be significant](https://www.sciencedirect.com/science/article/pii/S2352550924002422).
- **Specify constraints** on any activity in the life cycle inventories, interpreted as tangible limitations such as raw material availability, production capacity, or environmental regulations.
- **Optimize for or constrain any impact category** for which characterization factors are available.
- **Specify supply values** instead of final demands, which is relevant when only production volumes are known (e.g. [here](https://www.pnas.org/doi/10.1073/pnas.1821029116)).
- **Optimize under uncertainty** via a dedicated pipeline: import and filter uncertain LCI parameters, apply uncertainty strategies, run Global Sensitivity Analysis (Sobol), perform Monte Carlo sampling, or solve Chance-Constrained programs to obtain Pareto-optimal solutions at user-defined probability levels.

**Features recently completed:**

> - [X] `ℹ️  Optimization under uncertainty [chance-constraints, Monte Carlo, global sensitivity analysis]`
> - [X] `ℹ️  Development of a GUI for simple optimization tasks` [Link](https://github.com/flechtenberg/pulpo-gui)
> - [X] `ℹ️  Enable PULPO to work on both bw2 and bw25 projects`
> - [X] `ℹ️  Thorough documentation hosted on flechtenberg.github.io/pulpo/`

**Features currently under development:**

> - [ ] `ℹ️  Multi-objective optimization [bi-objective epsilon constrained, goal programming ...]`
> - [ ] `ℹ️  Integration of economic and social indicators in the optimization problem formulation`

Feature requests are more than welcome!

---

### 🔧 Installation
PULPO is available on PyPI. Depending on the version of Brightway you want to work with, install either the `bw2` or `bw25` variant:

```sh
pip install "pulpo-dev[bw2]"
```
or
```sh
pip install "pulpo-dev[bw25]"
```

### 🤖 Running PULPO

Use this link to start a cloud session and test PULPO right away:

[![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/v2/projects/fabian/pulpo-test/sessions/01JRM54S4NKMS84Y6BAYT832WH/start)

The main reference is the [PULPO showcase notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb), which revolves around methanol production and covers both the core optimization features (Sections 1–9) and the full workflow of the `pulpo_unc` module (Section 10): uncertainty data import and filtering, gap-filling strategies, Monte Carlo from prepared distributions, Chance-Constrained optimization, and Global Sensitivity Analysis.

Additional example notebooks are available for a [hydrogen case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/hydrogen_showcase.ipynb), an [electricity case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/electricity_showcase.ipynb), and a [plastic case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/plastic_showcase.ipynb).

There is also a workshop repository ([here](https://github.com/flechtenberg/pulpo_workshop)) created for the Brightcon 2024 conference, with guided notebooks and exercises.

### 🧪 Tests

Run from the package folder:

```sh
python -m unittest discover -s tests
```

---
## What's new in 1.5.1?
- Hotfix for chance-constrained uncertainty workflow: corrected environmental-cost mean computation in `pulpo.utils.uncertainty.cc` by replacing fragile pandas sparse updates with direct sparse-matrix updates and multiplication.
- Updated Section 10 of the main [showcase](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb) to include the deterministic reference result used by `run_gsa`, and re-ran the workflow end-to-end.
- Moved the sample and rice-husk database helpers into a new `pulpo.datasets` subpackage so that `pulpo.install_sample_db()` and `pulpo.install_rice_husk_db()` work from an installed wheel.
- Packaging fixes: `tests/` is no longer included in the wheel, and the `bw2` / `bw25` extras are now declared as mutually exclusive in `[tool.uv] conflicts` so that `uv` can resolve the lockfile cleanly.

## What's new in 1.5.0?

This release integrates the full **uncertainty analysis** pipeline into PULPO via the new `pulpo.pulpo_unc` module, turning the long-running development effort into a first-class feature:

- **`PulpoOptimizerUnc`** — A subclass of `PulpoOptimizer` that exposes the entire uncertainty workflow through a single worker object.
- **Uncertainty data import and filtering** — Import uncertain parameters directly from Brightway databases via `import_and_filter_uncertainty_data()`, with configurable cutoff-based filtering.
- **Uncertainty strategies** — Fill missing or incomplete uncertainty specifications using `apply_uncertainty_strategies()`, with built-in strategies (e.g. triangular bound interpolation) and support for custom expert-knowledge distributions.
- **Monte Carlo from prepared distributions** — Run MC on the curated uncertainty data (without re-sampling the full Brightway matrices) via `run_mc_from_uncertainty()`.
- **Chance-Constrained (CC) optimization** — Formulate and solve chance-constrained programs with `create_CC_formulation()` and `solve_CC_problem()`, yielding a Pareto front of optimal solutions at varying probability (risk) levels.
- **Global Sensitivity Analysis (GSA)** — Identify the uncertain parameters that drive optimization outcomes using Sobol sensitivity indices via `run_gsa()`.
- **End-to-end showcase** — Section 10 of the [PULPO showcase notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb) walks through the complete `pulpo_unc` workflow on the methanol system.
- Note: The uncertainty features in this release have been implemented and tested with Brightway `bw2` only; users working with `bw25` should validate workflows and may need to adapt configuration.
- Minor bugfixes and code cleanup.

## What's new in 1.4.3?
- Allow users to pass lower inventory flow and lower impact limits via `lower_inv_limit` and `lower_imp_limit` dicts.
- Provide new [showcase](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb) notebook.
- Enable users to pass custom default upper limits on elements, given that Gurobi identified `1e20` (and `1e24`) as infinite in some cases. See section 8 of the [showcase](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb) for usage. Setting them lower may also improve convergence speed.
- Enable dependent constraint definition. See section 9 of the [showcase](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb) for usage.

## What's new in 1.4.2?
- Enable the use of Gurobi solver.

## What's new in 1.4.0?
- Enable the use of NEOS solver (commercial solvers without a local license).
- Enable Monte Carlo sampling feature.
- Retrieve uncertainty information to `lci_data` for future use.

## What's new in 1.3.0?
- Switch packaging logic from `setup.py` to `pyproject.toml` and align PyPI with GitHub versioning.

---

## 🤝 Contributing
Contributions are very welcome. To request a feature or report a bug, please [open an Issue](https://github.com/flechtenberg/pulpo/issues). If you are confident in your coding skills, feel free to implement your suggestions and [send a Pull Request](https://github.com/flechtenberg/pulpo/pulls).

---

## 📄 License

This project is licensed under the `ℹ️  BSD 3-Clause` License. See the [LICENSE](LICENSE) file for additional info.  
Copyright (c) 2026, Fabian Lechtenberg. All rights reserved.


---

## 👏 Acknowledgments

We would like to express our gratitude to the authors and contributors of the following packages that **PULPO** builds upon:

- [**pyomo**](https://github.com/Pyomo/pyomo)
- [**brightway2**](https://github.com/brightway-lca/brightway2)

We also acknowledge the pioneering ideas and contributions from the following works:

- **[Computational Structure of LCA](http://link.springer.com/10.1007/978-94-015-9900-9)**
- **[Technology Choice Model](https://pubs.acs.org/doi/10.1021/acs.est.6b04270)**
- **[Modular LCA](http://link.springer.com/10.1007/s11367-015-1015-3)**

The development of PULPO culminated in the following publication, which details the approach and outlines its implementation:

> **Fabian Lechtenberg, Robert Istrate, Victor Tulus, Antonio Espuña, Moisès Graells, and Gonzalo Guillén‐Gosálbez.**  
> “PULPO: A Framework for Efficient Integration of Life Cycle Inventory Models into Life Cycle Product Optimization.”  
> *Journal of Industrial Ecology*, October 10, 2024.  
> [https://doi.org/10.1111/jiec.13561](https://doi.org/10.1111/jiec.13561)


Please cite this article if PULPO is used to produce results for a publication or project.

---

## Authors
- [@flechtenberg](https://www.github.com/flechtenberg)
- [@robyistrate](https://www.github.com/robyistrate)
- [@vtulus](https://www.github.com/vtulus)
- Bartolomeus Haeussling Loewgren
---
[↑ Return](#Top)