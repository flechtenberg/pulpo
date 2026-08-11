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
> - [X] `ℹ️  Time-dependent optimization [time-indexed formulation with inter-timestep storage/carry-over]`
> - [X] `ℹ️  Development of a GUI for simple optimization tasks` [Link](https://github.com/flechtenberg/pulpo-gui)
> - [X] `ℹ️  Enable PULPO to work on both bw2 and bw25 projects`
> - [X] `ℹ️  Thorough documentation hosted on flechtenberg.github.io/pulpo/`
> - [X] `ℹ️  Goal-programming objective (average transgression of soft impact limits)`

**Features currently under development:**

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

Add the `uncertainty` extra (SALib, stats_arrays, seaborn) if you plan to use the `pulpo_unc` module for Monte Carlo, Chance-Constrained optimization, or Global Sensitivity Analysis:

```sh
pip install "pulpo-dev[bw25,uncertainty]"
```

### 🤖 Running PULPO

PULPO is organized into three optimizer classes, one per module, each covering a different use case with its own reference notebook:

- **`pulpo.pulpo.PulpoOptimizer`** — the core LCO framework: technology/region choices, constraints, single- and multi-objective optimization (including goal programming), and supply-driven optimization. Start with the [PULPO showcase notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb), a complete walkthrough built around a methanol production case.
- **`pulpo.pulpo_time.PulpoOptimizerTime`** — the time-indexed extension: per-timestep demands and limits, impact budgets aggregated across the horizon, and inter-timestep storage/carry-over. See the [time-dependent toy notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/elec_time_toy.ipynb) for hourly and daily battery-dispatch examples.
- **`pulpo.pulpo_unc.PulpoOptimizerUnc`** — the uncertainty extension: import and filter uncertain LCI parameters, apply gap-filling strategies, run Monte Carlo sampling, Chance-Constrained optimization, and Global Sensitivity Analysis. See the [uncertainty toy notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/uncertainty_toy.ipynb).

Additional example notebooks are available for a [hydrogen case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/hydrogen_showcase.ipynb), an [electricity case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/electricity_showcase.ipynb), and a [plastic case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/plastic_showcase.ipynb).

There is also a workshop repository ([here](https://github.com/flechtenberg/pulpo_workshop)) created for the Brightcon 2024 conference, with guided notebooks and exercises.

### 🧪 Tests

The test suite runs with `pytest` against dedicated virtual environments for the modern (`bw25`) and legacy (`bw2`) Brightway stacks. See the [testing README](https://github.com/flechtenberg/pulpo/blob/master/tests/README.md) for setup instructions and the exact commands.

---
## What's new in 1.7.0?
- **Goal-programming objective** — `objective='goal'` minimizes the average transgression of user-defined soft impact limits (`imp_goals`), suited e.g. to Planetary-Boundary-based budgets. Unlike a hard `upper_imp_limit`, a goal can be exceeded — the solver stays feasible and reports the transgression level per category via `extract_results()["Transgressions"]`. Works in both `PulpoOptimizer` and the time-extended `PulpoOptimizerTime` (goals apply to impacts aggregated over the whole horizon).
- **Hardening** — several `default_limits` interactions with goal categories, Monte Carlo re-instantiation (now forwards the full `instantiate()` signature, including `time_steps`/`storage`), and bw25 uncertainty (GSA under SALib 1.5/numpy 2, deterministic scaling-vector construction) were fixed.
- **From 1.6.1: Windows solve-hang fix & Python 3.13 support** — unspecified limits are now truly infinite (no more HiGHS/pyomo deadlock), and numpy 2 / pyomo `>=6.8` are supported.

See the [changelog](https://github.com/flechtenberg/pulpo/blob/master/CHANGES.md) for the full details and earlier releases.

---

## 🤝 Contributing
Contributions are very welcome. To request a feature or report a bug, please [open an Issue](https://github.com/flechtenberg/pulpo/issues). If you are confident in your coding skills, feel free to implement your suggestions and [send a Pull Request](https://github.com/flechtenberg/pulpo/pulls).

---

## 📄 License

This project is licensed under the `ℹ️  BSD 3-Clause` License. See the [LICENSE](https://github.com/flechtenberg/pulpo/blob/master/LICENSE) file for additional info.  
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