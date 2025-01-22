# Theory

## Optimization

`pulpo` integrates life cycle thinking and data with optimization techniques. Its vision is to provide a versatile platform for LCA practitioners, enabling them to tackle a wide range of optimization tasks. By simplifying complex analyses and uncovering previously inaccessible insights, `pulpo` empowers users to explore new solutions and make data-driven decisions with greater ease.

### Linear Programming
**[Optimization problems](https://neos-guide.org/guide/types/)** can be defined in various ways, depending on the types of equations, constraints, variables, and parameters. Among these, problems involving linear equations and constraints with continuous variables are the simplest to solve and are known as **[linear programming](https://www.geeksforgeeks.org/linear-programming/)**.

A general formulation of a linear program (**LP**) is as follows:

$$
\begin{aligned}
& \min_{x} & c^T x \\
& \text{s.t.} & Ax \leq b \\
&            & x \geq 0
\end{aligned}
$$

Here:
- $c^T x$: The objective function, where $c$ is a vector of coefficients and $x$ is the vector of decision variables.
- $x$: The vector of continuous decision variables.
- $Ax \leq b$: The system of linear constraints.
- $x \geq 0$: Non-negativity constraints ensuring all variables are non-negative.

For simple problems, such as the example shown later in this section, graphical methods can be used to find the solution. However, for more complex problems involving thousands or even millions of variables and constraints—common in many life cycle optimization problems—efficient numerical methods like the simplex algorithm are required.

### LCA to Optimization

The [computational structure of LCA](https://link.springer.com/book/10.1007/978-94-015-9900-9) can be converted to a linear program, as shown by the **[Technology Choice Model (TCM)](https://pubs.acs.org/doi/10.1021/acs.est.6b04270)** approach. Leveraging the system's linearity, the resulting optimization problem is formulated as:

$$
\begin{aligned}
& \min_{s} & QBs \\
& \text{s.t.} & As = f \\
&            & s \geq 0
\end{aligned}
$$

Here:
- $QBs$: The objective function, where $Q$ is a vector of characterization factors, $B$ is the inventory matrix, and $s$ is the activity scaling vector.
- $As = f$: System constraints, where $A$ is the technology matrix, $f$ is the demand vector, and $s$ represents activity scaling.
- $s \geq 0$: Non-negativity constraint ensuring scaling variables remain realistic.

The matrices $A$, $B$, and $Q$ are derived from the LCA database, while the user specifies the demand vector $y$. The optimization identifies the scaling vector $s$ that minimizes the objective function while satisfying the constraints. 

When $A$ is square and invertible, the system constraints yield a unique solution, equivalent to a standard LCA with no degrees of freedom. Optimization is applied when $A$ is not square—i.e., there are more activities than products—introducing degrees of freedom. The user defines these degrees of freedom by converting the original square $A$ matrix into a rectangular matrix $A^*$, which is created in PULPO using a top-down approach, as shown below.


#### Rectangular Matrix Construction
Starting with a square technosphere matrix, choices can be introduced to the system by summing rows of functionally equivalent products. For instance, electricity used by a subsequent process may not need to be differentiated by its production process—whether it originates from coal or wind power. In a traditional LCA, however, these products would be treated as distinct. 

By summing the rows corresponding to functionally equivalent products, all activities that previously relied on individual products must now choose between or use a mix of the available options. This approach enables a system-wide integration of choices. 

This concept is similar to the **[modular LCA](http://link.springer.com/10.1007/s11367-015-1015-3)**. The illustration below contrasts this **top-down row elimination** approach with the traditional **bottom-up** method of constructing the rectangular technosphere matrix.

::::{tab-set}

:::{tab-item} Bottom-Up

```{image} data/bottom_up_light.svg
:class: only-light
:width: 80%
:align: center
```

```{image} data/bottom_up_dark.svg
:class: only-dark
:width: 80%
:align: center
```
:::

:::{tab-item} Top-Down

```{image} data/top_down_light.svg
:class: only-light
:width: 80%
:align: center
```

```{image} data/top_down_dark.svg
:class: only-dark
:width: 80%
:align: center
```
:::

::::


### Full Nomenclature

The currently implemented optimization problem formulation in `pulpo` is as follows:

$$
\begin{align}
\text{min} & \quad \sum_h w_h \cdot z_h & \label{eq:VII_OP1} \\
\text{s.t.} & \quad \sum_j (a_{ij} \cdot s_j) = f_i + \text{slack}_i \quad & \forall i \label{eq:VII_constraint1} \\
& \quad s_j^{\text{low}} \leq s_j \leq s_j^{\text{high}} \quad & \forall j \label{eq:VII_constraint2} \\
& \quad 0 \leq \text{slack}_i \leq \text{slack}_i^{\text{high}} \quad & \forall i \label{eq:VII_constraint3} \\
& \quad z_h = \sum_e \sum_j (q_{he} \cdot b_{ej} \cdot s_j) \quad & \forall h \label{eq:VII_constraint4} \\
& \quad z_h \leq z_h^{\text{high}} \quad & \forall h \label{eq:VII_constraint5} \\
& \quad \sum_j (b_{ej} \cdot s_j) \leq b_e^{\text{high}} \quad & \forall e \label{eq:VII_constraint6} 
\end{align}
$$

The calculation of the indicators $z_h$ has been shifted from the objective function to an equality constraint. The current objective function uses a set of weighting parameters $w_h$, allowing users to emphasize different indicators—individually or simultaneously—depending on the analysis goals.

An important addition to the base TCM formulation is the inclusion of slack variables. These variables relax the demand constraint, which is particularly useful when supply is specified instead of demand. This would have saved the solution of an auxiliar problem in [this study](https://www.science.org/doi/10.1126/science.abg9853), and has been used in the [paper introducing PULPO](https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.13561).

Additional additions include the specification of various constraints:
- $z_h^{\text{high}}$: Upper bounds for impact indicators.
- $b_e^{\text{high}}$: Upper bounds for environmental flows.
- $s_j^{\text{low}}$ and $s_j^{\text{high}}$: Lower and upper bounds for technosphere / activity scaling variables.

```{note}
This nomenclature will be expanded in the future to support additional functionalities, such as the integration of chance constraints, multi-objective optimization, and more.
```


## Simple Example
In this section we will walk through a minimal LCO example which can be solved by hand and illustrates the steps involved in getting from the departure data all the way to interpreting the results.

The matrix $ \mathbf{A} $ below represents a minimal system producing electricity via coal and wind power, with a market offering electricity in equal shares. Each process (column) corresponds to a unique product (row), as indicated by the ones on the diagonal:

$$
A =
\begin{array}{l c c c}
    & \text{electricity market} & \text{coal} & \text{wind} \\[1ex]
    \text{electricity (from market) [kWh]} & 1 & -0.1 & -0.05 \\
    \text{electricity (from coal) [kWh]} & -0.5 & 1 & 0 \\
    \text{electricity (from wind) [kWh]} & -0.5 & 0 & 1 \\
\end{array}
$$

For a standard LCA, supplying 1 kWh of electricity from the market involves solving $ \mathbf{A} \cdot \mathbf{s} = \mathbf{f} $, where $ \mathbf{f} $ is the demand vector. The solution is obtained by inverting $ \mathbf{A} $ and multiplying by $ \mathbf{f} $:

$$
\mathbf{s} = \mathbf{A}^{-1} \cdot \mathbf{f} =
\begin{pmatrix}
1 & -0.1 & -0.05 \\
-0.5 & 1 & 0 \\
-0.5 & 0 & 1 
\end{pmatrix}^{-1} 
\cdot 
\begin{pmatrix}
1 \\
0 \\
0 
\end{pmatrix} 
=
\begin{pmatrix}
1.08108 \\
0.54054 \\
0.54054 
\end{pmatrix}
\begin{array}{c}
\text{electricity market} \\
\text{coal} \\
\text{wind}
\end{array}
$$

To provide 1 kWh of electricity from the market, the electricity market process supplies 1.08 kWh, accounting for the self-consumption in coal and wind power plants. Impacts are obtained by multiplying the scaling vector $ \mathbf{s} $ with the biosphere matrix $ \mathbf{B} $ and the characterization factors $ \mathbf{Q} $.

Since the system is well-posed and square, inversion of $ \mathbf{A} $ is always possible. However, if the goal is to determine "which mix of technologies minimizes the GWP of 1 kWh of electricity output from the market," optimization is required.

Following the top-down approach, this requires summing the rows "electricity (from coal)" and "electricity (from wind)" to form the new row "electricity$^*$":

$$
A^* =
\begin{array}{l c c c}
    & \text{electricity market} & \text{coal} & \text{wind} \\[1ex]
    \text{electricity (from market) [kWh]} & 1 & -0.1 & -0.05 \\
    \text{electricity}^* \, [\text{kWh}] & -1 & 1 & 1 \\
\end{array}
$$


The resulting system of equations is now underdetermined, meaning the solution to $A^* \cdot s = f$ is no longer unique. By plugging this rectangularized matrix into the PULPO optimization problem and solving it, the optimal values for $s$ can be determined. These values represent the appropriate share of coal and wind power plants supplying the market, based on the specifications (objective, constraints, etc.) defined by the user.

A minimal biosphere matrix for this system could look as follows:

$$
\mathbf{B} = 
\begin{pmatrix}
0 & 1 & 0.01
\end{pmatrix}
\quad \text{kg CO}_2 \text{, in air}
$$

The characterization factor $q$ for this system is 1, representing $\frac{\text{kg CO}_2\text{eq}}{\text{kg CO}_2\text{, in air}}$. To focus solely on GWP, the weight $w$ is set to 1. Assuming a final demand of 1 kWh from the electricity market, with no constraints on production capacities or impacts, the optimization problem simplifies as follows:

$$
\begin{aligned}
\text{min} & \quad s_2 + 0.01 \cdot s_3 \\
\text{s.t.} & \quad 0.9s_2 + 0.95s_3 = 1 \\
& \quad 0 \leq s_2, s_3 \leq 1000
\end{aligned}
$$

The graphical solution to this problem is illustrated below:

::::{tab-set}

:::{tab-item} Simple Example Resolution

```{image} data/simple_example_light.svg
:class: only-light
:width: 50%
:align: center
```

```{image} data/simple_example_dark.svg
:class: only-dark
:width: 50%
:align: center
```
:::

::::

The unconstrained optimum fulfills the demand entirely with wind power, yielding a minimal GWP of 0.0105 kg CO$_2$eq. However, introducing a capacity constraint of 0.5 kWh for wind power shifts the solution to a mix of coal and wind. The new optimum includes 0.5 kWh from wind and 0.583 kWh from coal, with a total GWP of 0.5883 kg CO$_2$eq.

Switching from demand-based to supply-based optimization modifies the equations:

$$
\begin{aligned}
\text{min} & \quad 1 \cdot z_{\text{GWP}} \\
\text{s.t.} & \quad 1 - 0.1s_2 - 0.05s_3 = \text{slack}_1 \\
& \quad -1 + s_2 + s_3 = 0 \\
& \quad 0 \leq s_2, s_3 \leq 1000 \\
& \quad z_{\text{GWP}} = 1 \cdot s_2 + 0.01 \cdot s_3
\end{aligned}
$$

Here, $s_2 + s_3 = 1$ ensures the total production meets supply. The unconstrained solution again favors wind power, with a GWP of 0.05 kg CO$_2$eq and a slack of 0.95 kWh due to internal electricity consumption.
