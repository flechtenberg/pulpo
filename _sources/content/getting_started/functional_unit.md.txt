# Functional Unit

The **functional unit** is the same as in any regular LCA study. It is the quantified functional output of the system under study, which is used as a reference to normalize the environmental impacts. The functional unit is defined by the user and can be any quantitative measure of the system's performance. For example, the functional unit can be the amount of product produced, the distance traveled, or the energy consumed.

### Specification in PULPO

In order to specify the functional unit, one must retrieve the corresponding activities from the LCI data. This can be conveniently performed using the `retrieve_activities` (or `retrieve_processes`) functions. 

The `retrieve_activities` function enables users to filter and extract specific activities or processes from a database based on multiple criteria, such as activity names, reference products, or locations.
Here's an example of its use:

```python
activities = ["market for electricity, high voltage"]
reference_products = ["electricity, high voltage"]
locations = ["DE"]

electricity_market = pulpo_worker.retrieve_activities(activities=activities, reference_products=reference_products, locations=locations)
```

The demand is then set by writing it to a dictionary. Here, we specify the total demand of electricity in the [German electricity market](https://www.destatis.de/EN/Themes/Society-Environment/Environment/Material-Energy-Flows/Tables/electricity-consumption-households.html):

```python
demand = {electricity_market[0]: 1.28819e+11}
```

In the case of multiple functional units, the demand dictionary can be extended accordingly.

