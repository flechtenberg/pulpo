# Constraints

Constraints represent a concept that is not inherently compatible with traditional matrix-based LCA. They define situations where certain processes within the system are restricted from scaling beyond specific limits. These constraints can include:

- **Technical limits**, such as capacity or resource availability.
- **Policy constraints**, such as environmental flow restrictions or impact thresholds.

With `pulpo`, all these constraints can be seamlessly implemented. The optimizer prioritizes the best available options up to their specified limits and then selects the next best alternatives, ensuring an efficient allocation of resources within the defined boundaries.

::::{tab-set}

:::{tab-item} Generic Example

```{image} ../data/constrained_light.svg
:class: only-light
:width: 100%
:align: center
```

```{image} ../data/constrained_dark.svg
:class: only-dark
:width: 100%
:align: center
```
The illustration above shows the "full optimization" case from the previous generic example and highlights areas where constraints can be applied. Foreground limits are often more tangible, as they fall under the direct control of the user. However, assessing scenarios for the background can also be valuable, especially when considering broader system constraints such as global land availability or other resource limitations.

:::

::::


## Specification in PULPO

In `pulpo`, constraints are specified in a manner similar to choices. As mentioned earlier, choices always include an upper bound, which can be set to a very high value to represent an "unconstrained" scenario. 

To implement constraints, identify the processes that need to be limited and assign the desired upper bound in a dictionary:

```python
activities = ["market for nuclear fuel element, for pressure water reactor, UO2 4.0% & MOX"]
reference_products = ["nuclear fuel element, for pressure water reactor, UO2 4.0% & MOX"]
locations = ["GLO"]

nuclear_fuel = pulpo_worker.retrieve_activities(activities=activities, reference_products=reference_products, locations=locations)

upper_limit = {nuclear_fuel[0]: 100000}
```

For environmental flow and impact constraints, the methodology is similar. Here we put a constraint on the environmental emission of Radon-222:

```python
elem = pulpo_worker.retrieve_envflows(activities='Radon-222')
elem = [flow for flow in elem if 'long-term' in str(flow)]
upper_elem_limit = {elem[0]: 5e12}
```

And for the impact constraints it works as follows. Here we put a constraint on the impact of ionising radiation on human health:

```python
upper_imp_limit = {"('EF v3.0', 'ionising radiation: human health', 'human exposure efficiency relative to u235')": 1e+10}
```

With all of these dictionaries defined, the optimization problem can be instantiated, solved and interpreted. The next section will guide you through this process.
