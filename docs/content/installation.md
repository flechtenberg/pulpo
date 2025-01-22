# Installation

`pulpo` is a Python software package available via [`pip`](https://pypi.org/project/pip/).

```{note}
`pulpo` supports both Brightway2 (`bw2`) and Brightway25 (`bw25`). However, these dependencies must be installed in separate environments to avoid conflicts. You can use either `conda` or `venv` to manage your environments.
```

::::{tab-set}

:::{tab-item} Linux, Windows, or macOS (x64)

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

3. **Verify installation**:
   Ensure that `pulpo` and its dependencies are correctly installed by running:
   ```bash
   pip list
   ```

:::

:::{tab-item} macOS (Apple Silicon/ARM)

```{note}
Currently we can't guarantee that `pulpo` will work on Apple Silicon/ARM. You may try to follow the steps outlined in [here](https://docs.brightway.dev/en/latest/content/installation/) to work with `brightway25_nosolver` instead of `brightway25`.
```

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
