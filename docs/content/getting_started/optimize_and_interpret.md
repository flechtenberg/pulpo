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
