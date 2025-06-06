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

This is a python package for **[Life Cycle Optimization (LCO)](https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.13561)** based on life cycle inventories. `pulpo` is intended to serve as a platform for optimization tasks of varying complexity.   

The package builds on top of the **[Brightway LCA framework](https://docs.brightway.dev/en/latest)** as well as the **[optimization modeling framework Pyomo](https://www.pyomo.org/)**.

---

## ✨ Capabilities

Applying optimization is recommended when the system of study has (1) many degrees of freedoms which would prompt the manual assessment of a manifold of scenarios, although only the "optimal" one is of interest and/or (2) any of the following capabilities makes sense within the goal and scope of the study:

- **Specify technology and regional choices** throughout the entire supply chain (i.e. fore- and background), such as choices for the production technology of electricity or origin of metal resources. Consistently accounting for changes in the background in "large scale" decisions [can be significant](https://www.sciencedirect.com/science/article/pii/S2352550924002422). 
- **Specify constraints** on any activity in the life cycle inventories, which can be interpreted as tangible limitations such as raw material availability, production capacity, or environmental regulations.
- **Optimize for or constrain any impact category** for which the **characterization factors** are available.
- **Specify supply values** instead of final demands, which can become relevant if only production values are available (e.g. [here](https://www.pnas.org/doi/10.1073/pnas.1821029116)).

The following features are currently under development:

> - [ ] `ℹ️  Optimization under uncertainty [chance-constraints, stochastic optimization ...]`
> - [ ] `ℹ️  Multi-objective optimization [bi-objective epsilon constrained, goal programming ...]`
> - [ ] `ℹ️  Integration of economic and social indicators in the optimization problem formulation`

> - [X] `ℹ️  Development of a GUI for simple optimization tasks` [Link](https://github.com/flechtenberg/pulpo-gui)
> - [X] `ℹ️  Enable PULPO to work on both bw2 and bw25 projects`
> - [X] `ℹ️  Thorough documentation hosted on flechtenberg.github.io/pulpo/`

Feature requests are more than welcome!

---

### 🔧 Installation
PULPO has been deployed to the pypi index. Depending on the version of brightway projects you want to work on, install either the bw2 or bw25 version via:
```sh
pip install pulpo-dev[bw2]
```
or
```sh
pip install pulpo-dev[bw25]
```

### 🤖 Running pulpo

Use this link to start a session and test PULPO
[![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/v2/projects/fabian/pulpo-test/sessions/01JRM54S4NKMS84Y6BAYT832WH/start)

Find further example notebooks for a [hydrogen case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/hydrogen_showcase.ipynb), an [electricity case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/electricity_showcase.ipynb), and a [plastic case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/plastic_showcase.ipynb) here.

There is also a workshop repository ([here](https://github.com/flechtenberg/pulpo_workshop)), which has been created for the Brightcon 2024 conference. It contains several notebooks that guide you through the PULPO package and its functionalities, as well as an exercise.


### 🧪 Tests

Calling from the package folder: 

```sh
python -m unittest discover -s tests
```

---
## What's new in 1.4.2?
- Enable the use of gurobi solver

## What's new in 1.4.0?
- Enable the use of NEOS solver (commercial solvers without license)
- Enable Monte-Carlo sampling feature 
- Retrieve uncertainty information to `lci_data` for future use

## What's new in 1.3.0?
- Switch packaging logic from setup.py to pyproject.toml and align pypi with Github versioning number

---

## 🤝 Contributing
Contributions are very welcome. If you would like to request a feature or report a bug please [open an Issue](https://github.com/flechtenberg/pulpo/issues). If you are confident in your coding skills don't hesitate to implement your suggestions and [send a Pull Request](https://github.com/flechtenberg/pulpo/pulls).

---

## 📄 License

This project is licensed under the `ℹ️  BSD 3-Clause` License. See the [LICENSE](LICENSE) file for additional info.
Copyright (c) 2025, Fabian Lechtenberg. All rights reserved.


---

## 👏 Acknowledgments

We would like to express our gratitude to the authors and contributors of the following main packages that **PULPO** is based on:

- [**pyomo**](https://github.com/Pyomo/pyomo)
- [**brightway2**](https://github.com/brightway-lca/brightway2)

In addition, we acknowledge the pioneering ideas and contributions from the following works:

- **[Computational Structure of LCA](http://link.springer.com/10.1007/978-94-015-9900-9)**
- **[Technology Choice Model](https://pubs.acs.org/doi/10.1021/acs.est.6b04270)**
- **[Modular LCA](http://link.springer.com/10.1007/s11367-015-1015-3)**

Follow-up work, incorporating features such as top-down matrix construction for the use of entire life cycle inventory databases and supply specification, was implemented in **PULPO** and culminated in the following publication, which details the approach and outlines its implementation:

> **Fabian Lechtenberg, Robert Istrate, Victor Tulus, Antonio Espuña, Moisès Graells, and Gonzalo Guillén‐Gosálbez.**  
> “PULPO: A Framework for Efficient Integration of Life Cycle Inventory Models into Life Cycle Product Optimization.”  
> *Journal of Industrial Ecology*, October 10, 2024.  
> [https://doi.org/10.1111/jiec.13561](https://doi.org/10.1111/jiec.13561)


This article is to be cited / referred to if PULPO is used to derive results of a publication or project.

---
## Authors
- [@flechtenberg](https://www.github.com/flechtenberg)
- [@robyistrate](https://www.github.com/robyistrate)
- [@vtulus](https://www.github.com/vtulus)
---
[↑ Return](#Top)