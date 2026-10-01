# Optimize and Interpret Results

### Instantiate
To optimize, the `pulpo_worker` must first instantiate the problem. This step combines the provided inputs and passes them to the abstract optimization model, creating a concrete optimization model.

#### Simple Unconstrained Instantiation
For an unconstrained problem, the instantiation can be as simple as:

```python
pulpo_worker.instantiate(demand=demand, choices=choices)
```

#### Adding Constraints
Additional constraints can be specified for different aspects of the system:
- **`upper_limit`**: Constraints on scaling vectors (e.g., capacity or availability).
- **`upper_elem_limit`**: Constraints on environmental flows (e.g., emissions or resource usage).
- **`upper_imp_limit`**: Constraints on impact indicators (e.g., global warming potential).

```python
pulpo_worker.instantiate(
    demand=demand, 
    choices=choices, 
    upper_limit=upper_limit, 
    upper_elem_limit=upper_elem_limit, 
    upper_imp_limit=upper_imp_limit
)
```

---

### Optimize / Solve
After instantiating the problem, you can solve it using the `solve()` method:

```python
pulpo_worker.solve()
# Optionally, specify the GAMS solver path:
# pulpo_worker.solve(GAMS_PATH="path/to/gams")
```

If no GAMS path is specified, the open-source solver `highspy` will be used. While `highspy` is slower than commercial solvers like CPLEX and may occasionally struggle with complex problems, it typically performs well for most scenarios.

#### Numerical scaling
An ecoinvent technosphere contains coefficients from 1e-13 to 2e+11, because infrastructure processes have a functional unit of one whole facility. Solvers apply their feasibility tolerance per row relative to its largest coefficient, so such facilities can be under-supplied "for free" — which moves the optimum by around 1% on unaggregated systems, with different solvers landing on different points.

`instantiate(scale=True)` equilibrates the LP to prevent this. It is off by default and recommended for unaggregated ecoinvent backgrounds. The scaling is exact (powers of two) and invisible in the results: `scaling_vector`, `impacts` and everything `extract_results()` returns are in original units. Only the constraint rows of `pulpo_worker.instance` are scaled; the factors are on `instance._row_scale` and `instance._col_scale`. With Gurobi, a scaled model is solved with Gurobi's own scaling disabled and tightened tolerances; pass `options={"Method": 1}` for bit-identical repeated solves.

---

### Summarize / Interpret Results
To make the results accessible and interpretable, use the following methods:

- **`summarize_results()`**: Provides a concise overview of the key decisions, objective values, and constraints.
- **`save_results()`**: Saves detailed results to an Excel file for further analysis.

#### Example Usage:
```python
pulpo_worker.summarize_results(choices=choices, demand=demand, constraints=upper_limit)
pulpo_worker.save_results(choices=choices, demand=demand, name='path/to/save/results.xlsx')
```

---

### Visualization and Iteration
The interpretation and visualization of results are left to the user. Depending on the insights gained, re-iterations to refine inputs or constraints may be necessary, just as in a traditional LCA workflow.
