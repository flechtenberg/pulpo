# Setup and Objective

To optimize a system, it is necessary to define a quantitative measure of the system's performance, known as the **objective function**. The objective function is a mathematical representation of the system's "goodness" and depends on the **decision variables** — the degrees of freedom that can be adjusted to optimize the system. Depending on the problem, the objective function can be maximized or minimized.

In the context of Life Cycle Optimization (LCO), the objective function can represent any impact category, quantified using the characterization factors of a life cycle impact assessment (LCIA) method (e.g., global warming potential, acidification potential). The objective function can also combine multiple impact categories to address more complex optimization goals.

```{warning}
The functionality for specifying objectives is being expanded to facilitate the calculation and visualization of trade-offs between multiple objectives. Additionally, ongoing work aims to integrate economic and social indicators, advancing from LCO to Life Cycle Sustainability Optimization (LCSO), encompassing all three pillars of sustainability.
```

### Setup

The objective is specified when creating a `pulpo` object, often referred to as `pulpo_worker`. Below is an example of how to create and fully specify a `pulpo_worker`.

```python
import os
from pulpo import pulpo

# Define the working directory
notebook_dir = os.path.dirname(os.getcwd())
directory = os.path.join(notebook_dir, 'data')

# (Optional) Define the path to GAMS
GAMS_PATH = r"C:\APPS\GAMS\win64\40.1\gams.exe"

# Specify the project and database name
project = "pulpo_bw25"
database = "ecoinvent-3.8-cutoff"
```

These are the general specification, to indicate the project and database to be used. It is also possible to specify multiple databases (e.g. fore- and background):

```python
databases = ["ecoinvent-3.8-cutoff",
            "foreground_inventories"]
```

#### Objective
The next step is to define the method/objective function.

```python
# Specify the method/objective
methods = "('ecoinvent-3.8', 'IPCC 2013', 'climate change', 'GWP 100a')"
```

In this example, the `pulpo_worker` is created with the objective of minimizing the global warming potential, specified as `"('ecoinvent-3.8', 'IPCC 2013', 'climate change', 'GWP 100a')"`. Once the pointers are set, the `pulpo_worker` can be created, and the lci data loaded:

```python
# Create the pulpo_worker object
pulpo_worker = pulpo.PulpoOptimizer(project, database, methods, directory)

# Retrieve the LCI data
pulpo_worker.get_lci_data()
```

#### Multiple Objectives

It is also possible to specify multiple objectives as a weighted sum. This is achieved by using a dictionary to define the methods and their respective weights. Setting the weights of other objectives to 0 enables single-objective optimization while calculating the other indicators for reference:

```python
methods = {
    "('IPCC 2013', 'climate change', 'GWP 100a')": 1,
    "('ReCiPe Endpoint (E,A)', 'resources', 'total')": 0,
    "('ReCiPe Endpoint (E,A)', 'human health', 'total')": 0,
    "('ReCiPe Endpoint (E,A)', 'ecosystem quality', 'total')": 0
}
```

#### Goal Programming: Average Transgression Level

Instead of minimizing a weighted sum of impacts, PULPO can minimize the **average transgression level** of user-defined soft limits — a goal programming formulation suited e.g. to Planetary-Boundary–based assessments with the EF v3.1 methods. For each impact category $h$ with a soft limit (goal) $L_h$, the transgression level is $TL_h = \text{Impact}_h / L_h$. Categories within their limit contribute nothing to the objective; transgressed categories contribute their relative exceedance:

$$\min \frac{1}{K} \sum_{h=1}^{K} \max\left(0,\; TL_h - 1\right)$$

where $K$ is the number of categories with a goal. The goals are passed to `instantiate` alongside the objective mode:

```python
imp_goals = {
    "('EF v3.1', 'climate change', 'global warming potential (GWP100)')": 197.0,
    "('EF v3.1', 'acidification', 'accumulated exceedance (AE)')": 29.0,
}

pulpo_worker.instantiate(choices=choices, demand=demand,
                         imp_goals=imp_goals, objective='goal')
```

Notes:
- Unlike the hard `upper_imp_limit` (see [constraints](constraints.md)), goals **can** be exceeded — the solver stays feasible and reports the transgression instead.
- The method weights are ignored with `objective='goal'`; categories with a goal are included in the model even if their weight is 0.
- Per-category results (impact, goal, transgression level) are available via `pulpo_worker.extract_results()["Transgressions"]` and shown by `summarize_results()`.
- In the time-extended model (`PulpoOptimizerTime` with `time_steps`), the goals apply to the impacts **aggregated over all timesteps** — i.e. each `imp_goals` entry is a total (e.g. yearly) budget for that category across the whole horizon.

With the `pulpo_worker` created, the next step is to define the **functional unit**, which will be covered in the following section.
