# Installation

`pulpo` is available as Python software package installable via [`pip`](https://pypi.org/project/pip/).

```{note}
`pulpo` supports both Brightway2 (`bw2`) and Brightway25 (`bw25`). However, these dependencies must be installed in separate environments to avoid conflicts. You can use either `conda` or `venv` to manage your environments.
```

::::{tab-set}

:::{tab-item} Windows or Linux (x86-64)

1. **Create a new environment**:
   - Using `conda`:
     ```bash
     conda create -n pulpo_env python=3.10
     conda activate pulpo_env
     ```
   - Using `venv`:
     ```bash
     python -m venv pulpo_env
     source pulpo_env/bin/activate  # On Windows: pulpo_env\Scripts\activate
     ```

2. **Install `pulpo` with the appropriate dependencies**:
   - For Brightway2-compatible environments:
     ```bash
     pip install pulpo-dev[bw2]
     ```
   - For Brightway25-compatible environments:
     ```bash
     pip install pulpo-dev[bw25]
     ```
   - To run the example notebooks, add the `notebooks` extra (JupyterLab, matplotlib, seaborn), e.g. `pip install "pulpo-dev[bw25,notebooks]"`.

3. **Verify installation**:
   Ensure that `pulpo` and its dependencies are correctly installed by running:
   ```bash
   pip list
   ```

:::

:::{tab-item} macOS, or Linux on ARM

The PARDISO solver that PULPO uses on Windows and Linux (x86-64) is not available here. PULPO then uses UMFPACK from `scikit-umfpack`, which has ready-made packages on conda-forge only, so create the environment with `conda`:

```bash
conda create -n pulpo_env -c conda-forge python=3.12 scikit-umfpack
conda activate pulpo_env
pip install "pulpo-dev[bw25]"
```

For Brightway2, use `"pulpo-dev[bw2]"` instead (Python 3.12 at most). Add the `notebooks` extra to run the example notebooks, e.g. `"pulpo-dev[bw25,notebooks]"`. Without `scikit-umfpack`, PULPO falls back to SciPy's solver, which gives the same results but is much slower on large databases. Brightway's own LCA calculations use `scikit-umfpack` as well.

:::

::::

## Updating `pulpo`

`pulpo` is actively developed, with frequent new releases. To update `pulpo`, follow these steps:

1. Activate your environment:
   ```bash
   conda activate pulpo_env  # For conda
   source pulpo_env/bin/activate  # For venv
   ```

2. Update `pulpo` with the appropriate dependencies:
   - For Brightway2-compatible environments:
     ```bash
     pip install --upgrade pulpo-dev[bw2]
     ```
   - For Brightway25-compatible environments:
     ```bash
     pip install --upgrade pulpo-dev[bw25]
     ```

```{warning}
Newer versions of `pulpo` can introduce breaking changes. We recommend creating a new environment for each project and only updating `pulpo` when you are ready to update your project.
```
