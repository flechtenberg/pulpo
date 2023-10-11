<div align="center">
<h1 align="center">
<img src="https://github.com/flechtenberg/flechtenberg_images/blob/main/Pulpo-Logo_INKSCAPE.png?raw=true" width="300" />
<h3>◦ Python-based User-defined Lifecycle Production Optimization!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
</p>
<img src="https://img.shields.io/github/license/flechtenberg/pulpo?style=flat&color=5D6D7E" alt="GitHub license" />
<img src="https://img.shields.io/github/last-commit/flechtenberg/pulpo?style=flat&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/flechtenberg/pulpo?style=flat&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/flechtenberg/pulpo?style=flat&color=5D6D7E" alt="GitHub top language" />
</div>

---

## 📖 Table of Contents
- [📍 Overview](#-overview)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running pulpo](#-running-pulpo)
    - [🧪 Tests](#-tests)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

The project, "pulpo," is a life cycle assessment (LCA) optimization tool that aims to import, optimize, solve, retrieve, save, and summarize LCA results. It provides functionalities for data import from the Ecoinvent database, conversion of inputs into a dictionary for the optimization model, definition and solving of the optimization model using the pyomo package, and saving and summarizing of the model results. The tool's value lies in streamlining LCA processes, enabling optimization and analysis of environmental impacts, and facilitating the decision-making process for sustainable solutions.


---


## 📂 Repository Structure

```sh
└── pulpo/
    ├── .gitattributes
    ├── .gitconfig
    ├── .gitignore
    ├── desktop.ini
    ├── LICENSE
    ├── notebooks/
    │   └── hydrogen_showcase.ipynb
    ├── pulpo/
    │   ├── __init__.py
    │   ├── pulpo.py
    │   └── utils/
    ├── README.md
    ├── setup.cfg
    └── tests/
        ├── __init__.py
        └── test_functions.py
```


---

## ⚙️ Modules

<details closed><summary>Root</summary>

| File                                                                                                         | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---                                                                                                          | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [.gitconfig](https://github.com/flechtenberg/pulpo/blob/main/.gitconfig)                                     | This code fragment configures a git filter to clean Jupyter Notebook files in the.gitconfig file. It uses the Jupyter nbconvert command to remove the output cells and smudge to display the file's contents.                                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [pulpo.py](https://github.com/flechtenberg/pulpo/blob/main/pulpo\pulpo.py)                                   | The code implements a PulpoOptimizer class that provides functionalities for data import, optimization, solving, retrieval, saving and summarizing results related to life cycle assessments. It uses modules like optimizer, bw_parser, converter, and saver for different operations.                                                                                                                                                                                                                                                                    |
| [bw_parser.py](https://github.com/flechtenberg/pulpo/blob/main/pulpo\utils\bw_parser.py)                     | The code in `bw_parser.py` provides functions for importing, saving, and retrieving life cycle inventory (LCI) data from the Ecoinvent database using the Brightway2 library. It includes functions for importing data, checking if data needs to be reloaded, performing LCA calculations, saving LCI data to files, and reading LCI data from files. Additionally, it provides functions for retrieving activities and environmental flows from the database based on specified criteria. The `main` function demonstrates the usage of these functions. |
| [converter.py](https://github.com/flechtenberg/pulpo/blob/main/pulpo\utils\converter.py)                     | The code in pulpo\utils\converter.py combines various inputs into a dictionary for an optimization model. It converts sparse matrices to dictionaries, modifies the technosphere matrix, creates sets, specifies demand, limits, and supply, assigns weights, and assembles the final data dictionary for the model. This function serves as a crucial step in preparing the inputs for the optimization process.                                                                                                                                          |
| [optimizer.py](https://github.com/flechtenberg/pulpo/blob/main/pulpo\utils\optimizer.py)                     | The code defines an optimization model using the pyomo package. It includes sets, parameters, variables, constraints, and an objective function. The model is created and solved using different solvers.                                                                                                                                                                                                                                                                                                                                                  |
| [saver.py](https://github.com/flechtenberg/pulpo/blob/main/pulpo\utils\saver.py)                             | The code provides two main functionalities:1. save_results: Saves the results of a Pyomo model to an Excel file, including raw results, metadata, and constraints.2. summarize_results: Prints a summary of the model results, including demand, impacts, choices, and constraints.                                                                                                                                                                                                                                                                        |

</details>

---

## 🚀 Getting Started

### 🔧 Installation
PULPO has been deployed to the pypi index and can now be installed via:
```sh
pip install pulpo-dev
```


### 🤖 Running pulpo
See [pypi](https://pypi.org/project/pulpo-dev/) for a description of how to use PULPO via pip.

Find example notebooks for a [hydrogen case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/hydrogen_showcase.ipynb) and an [electricity case](https://github.com/flechtenberg/pulpo/blob/master/notebooks/electricity_showcase.ipynb) here.

### 🧪 Tests

Calling from the package folder: 

```sh
python -m unittest discover -s tests
```

---


## 🛣 Roadmap

> - [ ] `ℹ️  Task 1: Implement integer cuts to allow a fast calculation of a ranked list of best options`
> - [ ] `ℹ️  Task 2: Implement functionality to treat uncertainty in the optimization problem (robust)`
> - [ ] `ℹ️ ... Requests are welcome.`


---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `ℹ️  BSD 3-Clause` License. See the [LICENSE](LICENSE) file for additional info.
Copyright (c) 2023, Fabian Lechtenberg. All rights reserved.

---

## 👏 Acknowledgments

We would like to acknowledge the authors and contributors of these main packages that pulpo is based on:
 - [pyomo](https://github.com/Pyomo/pyomo)
 - [brightway2](https://github.com/brightway-lca/brightway2)
---
## Authors
- [@flechtenberg](https://www.github.com/flechtenberg)
- [@robyistrate](https://www.github.com/robyistrate)
- [@vtulus](https://www.github.com/vtulus)
---
[↑ Return](#Top)