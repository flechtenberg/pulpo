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
- **Optimize under uncertainty**: import the declared distributions of the inventory and characterization factors, compute the impact's mean and variance in closed form, and solve chance-constrained programs (jointly over the impact and uncertain capacities) to obtain the Pareto front over reliability levels. Decompose the variance exactly, screen the parameters that declare no uncertainty, and validate every solution out of sample. *As of now, uncertainty in the LCA data is considered only in the biosphere flows and the characterization factors; technosphere exchanges are treated as deterministic.*
- **Solve in reduced space** with `solve(method='reduced')`: the same LP over the choice alternatives only, exact and much smaller on large databases.

**Features recently completed:**

> - [X] `ℹ️  Optimization under uncertainty [chance-constraints, Monte Carlo]`
> - [X] `ℹ️  Time-dependent optimization [time-indexed formulation with inter-timestep storage/carry-over]`
> - [X] `ℹ️  Development of a GUI for simple optimization tasks` [Link](https://github.com/flechtenberg/pulpo-gui)
> - [X] `ℹ️  Enable PULPO to work on both bw2 and bw25 projects`
> - [X] `ℹ️  Thorough documentation hosted on flechtenberg.github.io/pulpo/`
> - [X] `ℹ️  Goal-programming objective (average transgression of soft impact limits)`
> - [X] `ℹ️  Exact chance-constrained optimization (second-order cone, joint risk budgets, exact bound quantiles)`
> - [X] `ℹ️  Numerical scaling of the LP for unaggregated ecoinvent backgrounds`
> - [X] `ℹ️  Reduced-space solves, and an exact, validated uncertainty method`

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

The uncertainty features need no extra packages; the `uncertainty` extra is kept as an empty alias, so `pip install "pulpo-dev[bw25,uncertainty]"` still works.

On macOS and on Linux for ARM, the PARDISO solver is not available. Install `scikit-umfpack` from conda-forge (`conda install -c conda-forge scikit-umfpack`) for fast reduced solves; see the [installation guide](https://flechtenberg.github.io/pulpo/content/installation.html).

### 🤖 Running PULPO

PULPO is organized into three optimizer classes, one per module, each covering a different use case with its own reference notebook:

- **`pulpo.pulpo.PulpoOptimizer`** — the core LCO framework: technology/region choices, constraints, single- and multi-objective optimization (including goal programming), and supply-driven optimization. Start with the [PULPO showcase notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/pulpo_showcase.ipynb), a complete walkthrough built around a methanol production case.
- **`pulpo.pulpo_time.PulpoOptimizerTime`** — the time-indexed extension: per-timestep demands and limits, impact budgets aggregated across the horizon, and inter-timestep storage/carry-over. See the [time-dependent toy notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/elec_time_toy.ipynb) for hourly and daily battery-dispatch examples.
- **`pulpo.pulpo_unc.PulpoOptimizerUnc`** — the uncertainty extension (a thin layer over `pulpo.utils.uncertainty`): import declared distributions, add expert knowledge, and solve chance-constrained programs in reduced space. See the [uncertainty toy notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/uncertainty_toy.ipynb).

Additional example notebooks are available for a [hydrogen case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/hydrogen_showcase.ipynb), an [electricity case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/electricity_showcase.ipynb), and a [plastic case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/showcases/plastic_showcase.ipynb).

There is also a workshop repository ([here](https://github.com/flechtenberg/pulpo_workshop)) created for the Brightcon 2024 conference, with guided notebooks and exercises.

### ⚡ Reduced-space solves

`solve(method='reduced')` solves the same LP over the choice alternatives only. With the technosphere matrix square and invertible, every scaling vector that meets the balances is `s = s0 + S v`, with one variable `v_k` per alternative, so the problem over `v` is the problem over `s`: exact, for every static constraint type, and small however large the database (one column per alternative, one row per constraint that is not a balance). The solution is written back onto the instance, so `extract_results()` and the rest read it as before. The time-dependent model supports `method='full'` only.

The chance-constrained problems of `pulpo.utils.uncertainty` are always solved this way. Solvers:

| Problem | Default | Alternative |
|---|---|---|
| deterministic LP (`full` or `reduced`) | HiGHS | Gurobi (and GAMS/NEOS for `full`) |
| chance-constrained cone | Clarabel | Gurobi |

HiGHS and Clarabel are installed with PULPO and need no licence. Gurobi is used when `gurobipy` is installed; the size-limited licence that ships with `pip install gurobipy` is for non-production use (see Gurobi's licence terms) and covers problems of up to 2,000 variables and 2,000 linear constraints, or 200 variables once quadratic terms are present. That fits most reduced problems, since they have one column per alternative.

### 🧪 Tests

The test suite runs with `pytest` against dedicated virtual environments for the modern (`bw25`) and legacy (`bw2`) Brightway stacks. See the [testing README](https://github.com/flechtenberg/pulpo/blob/master/tests/README.md) for setup instructions and the exact commands.

---
## What's new in 2.0.0?
- **Reduced-space solves** — `solve(method='reduced')`, see above.
- **One uncertainty method** — `pulpo.utils.uncertainty` replaces the 1.x generations (uncertainty in the biosphere flows and characterization factors only, as before; technosphere exchanges are deterministic): declared distributions only (undeclared parameters stay deterministic and are screened, not gap-filled), closed-form moments, a joint chance-constrained front over the impact and uncertain capacities solved in reduced space with Clarabel, an exact variance decomposition, and out-of-sample validation. The [uncertainty notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/uncertainty_toy.ipynb) runs it on the bundled demo database without ecoinvent or a commercial solver. The 1.x uncertainty API is removed (see the changelog).
- **Defaults unchanged** — `method='full'` and `instantiate(scale=False)`. Current development considers `method='reduced'` and `scale=True` (from 1.8.0) superior; a future release may switch the defaults.

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