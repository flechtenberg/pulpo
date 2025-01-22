# Choices

In a traditional LCA, the technosphere represents a specific configuration of interconnected processes. Changing connection between activities can be interpreted as assessing a **scenario** different from the reference. When only a few deviations from the reference are to be considered, the approach of manually defining the scenarios makes sense. However, when the number of scenarios increases, the manual approach becomes tedious and inefficient.

`pulpo` offers the possibility to specify **choices** for the technosphere, which are then used to define a continuous range of scenarios (also called **superstructure**). The choices can be specified for any activity in the life cycle inventories, such as the production technology of electricity or the origin of metal resources.

::::{tab-set}

:::{tab-item} Foreground

```{image} ../data/foreground_choices_light.svg
:class: only-light
:width: 100%
:align: center
```

```{image} ../data/foreground_choices_dark.svg
:class: only-dark
:width: 100%
:align: center
```
This illustration depicts an exemplary generic foreground system (right) and background system (left). The foreground activities are modeled by the LCA practitioner (user), while the background activities are retrieved from the LCI database. The functional unit is represented by one of the user's activities, highlighted in the <span style="color:green;">green box</span>.

In this scenario, the problem focuses on selecting between two activities within the foreground, highlighted through the <span style="color:orange;">yellow box</span>. For example, this could involve choosing between two different production processes for manufacturing a machine part or selecting an alternative coating. The inputs required for these user-modeled activities are sourced from the background, as shown by the vectors connecting the background activities to the foreground.
:::

:::{tab-item} Background

```{image} ../data/background_choices_light.svg
:class: only-light
:width: 100%
:align: center
```

```{image} ../data/background_choices_dark.svg
:class: only-dark
:width: 100%
:align: center
```
In the illustration above, the choices shift from the foreground to the background, representing a hypothetical scenario assessment. This approach explores the optimal boundary conditions (context) for the functional unit. For example, this could involve determining the composition of an electricity market (as shown in this demonstration) or identifying the origin of metal resources, among other possibilities.

:::

:::{tab-item} Fore- and Background

```{image} ../data/fore_background_choices_light.svg
:class: only-light
:width: 100%
:align: center
```

```{image} ../data/fore_background_choices_dark.svg
:class: only-dark
:width: 100%
:align: center
```
This scenario illustrates how choices can span both foreground and background systems simultaneously. Such an approach is particularly valuable for assessing the optimal integration of an activity within a broader context, especially when dealing with large-scale decisions. 

For instance, consider the selection of a plastic recycling process. While the user models this process in the foreground, it can also influence upstream activities that utilize plastic recycling. By enabling these choices across systems, this integrated approach accounts for feedback loops, which can have significant impacts—particularly in the case of large-scale substitutions.

This approach has been further explored and exemplified in [this publication](https://www.sciencedirect.com/science/article/pii/S2352550924002422), covering scenario assessment and not optimization.


:::

:::{tab-item} Full Optimization

```{image} ../data/full_optimization_light.svg
:class: only-light
:width: 100%
:align: center
```

```{image} ../data/full_optimization_dark.svg
:class: only-dark
:width: 100%
:align: center
```

This illustration represents a "full optimization" case, simultaneously considering foreground, background, and integrated fore- and background choices. Evaluating all possible combinations from the given degrees of freedom would require assessing 3 × 2 × 2 = 12 scenarios. 

However, this calculation assumes only binary choices. When choices become continuous—as in mixed scenarios—this results in a continuous space of options, effectively leading to an infinite number of scenarios. Even with purely binary choices, the number of possible scenarios grows exponentially as more degrees of freedom are introduced.

This exponential growth highlights the importance of optimization as an efficient approach to identify the optimal configuration among the vast number of alternatives.

:::

::::

### Choices in PULPO

The specification of choices in `pulpo` is performed similarly to the specification of the functional unit. First, the activities that can be chosen between must be retrieved from the LCI database. Below is the example for the choices in the electricity market:

```python
activities = ["electricity production, lignite", 
             "electricity production, hard coal",
             "electricity production, nuclear, pressure water reactor",
             "electricity production, wind, 1-3MW turbine, onshore"]
reference_products = ["electricity, high voltage"]
locations = ["DE"]

electricity_activities = pulpo_worker.retrieve_activities(activities=activities, reference_products=reference_products, locations=locations)

choices = {'electricity': {electricity_activities[0]: 1e20,
           electricity_activities[1]: 1e20,
           electricity_activities[2]: 1e20,
           electricity_activities[3]: 1e20}}

```

Note that the choices are defined as a nested dictionary, where the outer dictionary indexes the choice sets with a label (e.g., "electricity"). The inner dictionaries specify the activities that can be selected and their respective upper bounds. 

In an unconstrained scenario, the upper bound should be set to a very large value, such as `1e20`, as demonstrated above.

Technically, this example represents a **"fore- and background" choice**, as there is no distinct foreground system. The assessment relies entirely on the background database, but the deviation in the market could also be interpreted as a foreground choice, depending on how the boundaries are defined.



