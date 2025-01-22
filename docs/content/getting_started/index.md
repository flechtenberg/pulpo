# Getting Started

This section will guide you through the workflow of `pulpo`, starting from the very basics. The illustration below shows the required user inputs to define the optimization problem, as well as how `pulpo` converts this information to be used in interconnected packages to resolve the optimization problem.

```{image} ../data/PULPO_infogram_new_light.svg
:class: only-light
:height: 450px
:align: center
```

```{image} ../data/PULPO_infogram_new_dark.svg
:class: only-dark
:height: 450px
:align: center
```
<br />

Steps 1-4 are the user inputs, constituting the "Goal and Scope" of the LCO. Step 5 is the optimization and interpretation of the results. The following sections will guide you through each step. The theory behind what's happening under the hood in `parser` and `optimizer` is explained in the [Theory](../theory.md) section.

The case study to be evaluated in the following sections performs an LCO on the german electricity market, optimizing the share of the most prominent electricity production technologies, aiming to minimize the global warming potential (GWP) of the electricity mix.

```{image} ../data/Electricity_Market_Case_Study_Light.svg
:class: only-light
:height: 300px
:align: center
```

```{image} ../data/Electricity_Market_Case_Study_Dark.svg
:class: only-dark
:height: 300px
:align: center
```

```{admonition} You want more interaction?
:class: admonition-launch

[Open the interactive examples in Renku](https://renkulab.io/v2/projects/fabian/pulpo-test) In this interactive environment, you can directly run the pulpo code yourself.

In order to access the code you must login to Renku and start the "Python/Jupyter" session.
```

```{toctree}
---
hidden:
maxdepth: 1
---
Step 1 - Objective <objective>
Step 2 - Functional Unit <functional_unit>
Step 3 - Choices <choices>
Step 4 - Constraints <constraints>
Step 5 - Optimize and Interpret <optimize_and_interpret>
```
