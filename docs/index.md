# Life Cycle Optimization (LCO) with `PULPO`

`PULPO` is a python package for **[Life Cycle Optimization (LCO)](https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.13561)** based on life cycle inventories. It is intended to serve as a platform for optimization tasks of varying complexity.   

The package builds on top of the **[Brightway LCA framework](https://docs.brightway.dev/en/latest)** as well as the **[optimization modeling framework Pyomo](https://www.pyomo.org/)**.

## ✨ Capabilities

Applying optimization is recommended when the system of study has (1) many degrees of freedoms which would prompt the manual assessment of a manifold of scenarios, although only the "optimal" one is of interest and/or (2) any of the following capabilities makes sense within the goal and scope of the study:

- **Specify technology and regional choices** throughout the entire supply chain (i.e. fore- and background), such as choices for the production technology of electricity or origin of metal resources. Consistently accounting for changes in the background in "large scale" decisions [can be significant](https://www.sciencedirect.com/science/article/pii/S2352550924002422). 
- **Specify constraints** on any activity in the life cycle inventories, which can be interpreted as tangible limitations such as raw material availability, production capacity, or environmental regulations.
- **Optimize for or constrain any impact category** for which the **characterization factors** are available.
- **Specify supply values** instead of final demands, which can become relevant if only production values are available (e.g. [here](https://www.pnas.org/doi/10.1073/pnas.1821029116)).


## 💬 Support
If you have any questions or need help, do not hesitate to contact us:
- Fabian Lechtenberg ([fabian.lechtenberg@chem.ethz.ch](mailto:fabian.lechtenberg@chem.ethz.ch))


```{toctree}
---
hidden:
maxdepth: 1
---
Installation <content/installation>
Getting Started <content/getting_started/index>
Theory <content/theory>
Examples <content/examples/index>
API <content/api/index>
Contributing <content/contributing>
Code of Conduct <content/codeofconduct>
License <content/license>
Changelog <content/changelog>
```
