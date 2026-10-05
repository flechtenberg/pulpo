# Uncertainty

PULPO optimizes one impact under the uncertainty its data declare. The method
needs no sampling to solve: the impact's mean and variance follow in closed
form, the chance constraints become a second-order cone, and the cone is solved
in reduced space, over the choice alternatives only. Sampling enters afterwards,
to validate a solution out of sample. The [uncertainty notebook](https://github.com/flechtenberg/pulpo/blob/master/notebooks/uncertainty_toy.ipynb)
runs every step on the bundled demo database with open-source solvers.

```{note}
As of now, uncertainty in the LCA data is considered only in the biosphere
(intervention) flows and the characterization factors. Technosphere exchanges
are deterministic, even where the database declares a distribution for them
(assumption A4 below). Uncertain capacities or availabilities enter as
uncertain process bounds.
```

```python
from pulpo.utils import uncertainty as unc

data  = unc.import_declared(worker)                     # parameters of the impact
unc.override(data, 'If', 'foreground', expert_specs)    # elicited distributions
mom   = unc.compute_moments(data, worker)               # closed-form mean and variance
ccp   = unc.ChanceConstrained(worker, mom, upper_bounds={activity: capacity_spec})
front = ccp.solve([0.5, 0.9, 0.99])                     # reliability sweep
dec   = unc.decompose(front[0.9], mom)                  # exact Sobol' indices
scr   = unc.screen_undeclared(front[0.9], data, worker, exact_cfs=unc.co2_flows(worker))
val   = unc.validate(front, ccp, data, n=200_000, seed=1)
```

`pulpo.pulpo_unc.PulpoOptimizerUnc` offers the same steps as methods of a worker.

## Data and assumptions

- $A$ is the technosphere matrix (square, invertible, deterministic), $B$ the
  biosphere matrix with entries $b_{ej}$, $q$ the characterization factors (CFs)
  of one LCIA method, $s$ the scaling vector. The impact is
  $X(s) = \sum_e q_e \sum_j b_{ej} s_j$.
- The uncertain inputs are the entries of $B$ and the CFs that *declare* a
  distribution (in the database, in the method, or by expert override), and the
  bounds on processes that the problem declares uncertain (e.g. an availability
  or a capacity).
- Supported families: lognormal (also for negative amounts), normal, uniform,
  triangular, and exact values (`uncertainty_type` 1). Any other family raises.

The method rests on five assumptions, each stated where it is used:

| | Assumption | Where |
|---|---|---|
| A1 | all uncertain inputs are mutually independent | moments, decomposition |
| A2 | $X(s)$ is approximately normal (the only normality assumption) | the impact's chance constraint; tested by the validation |
| A3 | undeclared parameters are deterministic | everywhere; tested by the screening |
| A4 | $A$ is deterministic | makes the reduced space exact |
| A5 | the failure budget is split by Boole's inequality | joint chance constraints; conservative under any dependence |

`import_declared` takes every nonzero entry of $B$ on a characterized flow and
every CF on a flow that occurs in $B$. Parameters without a distribution are kept
as *undeclared*: their mean is their amount and their variance zero. No
contribution filter is applied and nothing is gap-filled.

## Moments

$$
\begin{aligned}
\mathrm{E}[X] &= \mu^\top s, & \mu_j &= \textstyle\sum_e \mathrm{E}[q_e]\,\mathrm{E}[b_{ej}] \\
\mathrm{Var}\,X &= \textstyle\sum_j d_j s_j^2 + \sum_e w_e y_e^2, &
d_j &= \textstyle\sum_e (\mathrm{E}[q_e]^2 + \mathrm{Var}\,q_e)\,\mathrm{Var}\,b_{ej},\quad
w_e = \mathrm{Var}\,q_e
\end{aligned}
$$

with $y_e = \sum_j \mathrm{E}[b_{ej}] s_j$ the mean flow $e$. The second sum is the
covariance that every process emitting flow $e$ shares through its one factor
$q_e$; `Moments.std_independent` drops it for comparison.

## Chance constraints

At a joint reliability level $\lambda$ over $K$ events (the impact row and one
per uncertain bound), each event may fail with probability
$\varepsilon_k = w_k(1-\lambda)$, equal weights by default. Boole's inequality then
bounds the probability that any event fails by $1-\lambda$:

$$
\begin{aligned}
\text{impact row:}\quad & P(X \le z) \ge \lambda_z = 1-\varepsilon_0
&&\Rightarrow\quad \mu^\top s + \kappa\,\sigma(s) \le z,\quad \kappa = \Phi^{-1}(\lambda_z) \\
\text{bound } j:\quad & P(s_j \le U_j) \ge 1-\varepsilon_k
&&\Rightarrow\quad s_j \le F_{U_j}^{-1}(\varepsilon_k)
\end{aligned}
$$

and minimizing $z$ gives

$$
\min_s\ \mu^\top s + \kappa\,\sigma(s)\quad \text{s.t. every constraint of the instance, with the uncertain bounds at their quantiles.}
$$

The bounds enter at the exact quantile of their declared family; only the
impact row uses the normal approximation (A2). $\kappa \ge 0$ ($\lambda_z \ge 1/2$)
keeps the problem convex; lower levels are refused. `allocation='individual'`
imposes every event at $\lambda$ on its own, for comparison: it controls no joint
probability.

## Reduced space

With $A$ square and invertible, every scaling vector that meets the balances is

$$
s = s_0 + S v,\qquad s_0 = A^{-1}\tilde f,\qquad S = A^{-1}E,
$$

with one free variable $v_k$ per alternative (the net output on its product row)
and $\tilde f$ the demand on all other rows. Every static PULPO constraint is
linear in $s$, so it becomes one row in $v$, and the problem over $v$ is the
problem over $s$: the reduction is exact. $S$ is never formed; each row
$m^\top S$ costs one adjoint solve with one factorization of $A$ (PARDISO,
UMFPACK or SciPy's SuperLU). Every finite process bound is a row $S[j, :]$, so the problem
stays small as long as only the processes with a real limit are bounded: finite
`default_limits` on every process would put the whole of $S$ into it.

The deterministic LP is solved this way by `solve(method='reduced')`. For the
chance-constrained problem $\sigma(s) = \lVert R\,[1; v]\rVert$ with
$R^\top R = G^\top G$, $G = Q^{1/2}[s_0, S]$ and
$Q = \mathrm{diag}(d) + B_u^\top \mathrm{diag}(w) B_u$: a second-order cone with one
column per alternative, solved by Clarabel or Gurobi.

## Exact variance decomposition

$X$ is a sum of products of independent inputs, so its Sobol' decomposition stops
at second order. At a fixed $s$, with $V = \mathrm{Var}\,X$:

$$
\begin{aligned}
S_1(q_e) &= w_e y_e^2 / V, &
S_1(b_{ej}) &= \mathrm{E}[q_e]^2\,\mathrm{Var}(b_{ej})\,s_j^2 / V, &
S_2(q_e, b_{ej}) &= w_e\,\mathrm{Var}(b_{ej})\,s_j^2 / V, \\
S_T(q_e) &= w_e \big(y_e^2 + \textstyle\sum_j \mathrm{Var}(b_{ej})\,s_j^2\big) / V, &
S_T(b_{ej}) &= (\mathrm{E}[q_e]^2 + w_e)\,\mathrm{Var}(b_{ej})\,s_j^2 / V
\end{aligned}
$$

and $\sum S_1 + \sum S_2 = 1$ exactly. `decompose` reports them per parameter and
per family (e.g. per database with `families_by_database`, and the CFs). B entries
never interact with each other, nor CFs with each other, so a family's indices
are the sums over its members. The uncertain bounds are not inputs of $X$ at a
fixed $s$; they shape $s$.

## Undeclared parameters

`screen_undeclared` tests assumption A3. Every undeclared parameter gets the same
coefficient of variation $r$ with its mean unchanged ($\mathrm{Var}\,b = r^2 b_0^2$,
$\mathrm{Var}\,q = r^2\,\mathrm{E}[q]^2$) and the indices above are evaluated at a fixed
decision. The CFs passed as `exact_cfs` never receive a width: for a global
warming potential these are the CO₂ flows, whose factor is exact by definition
because CO₂ is the reference gas (`co2_flows` finds them by name in an
ecoinvent biosphere). Every undeclared B entry is widened: none is exact by
definition, holding one exact could only lower the figures, and only the few at
the top of the ranking need an expert's judgement. The parameters are ranked by their
deterministic contribution $\mathrm{E}[q_e]\,b_{0,ej}\,s_j$, which fixes the order
of their total-order indices up to the CF factor $1 + w_e/\mathrm{E}[q_e]^2$ and does
not depend on $r$; the indices at each $r$ are reported beside it.

`width_sensitivity` evaluates $\sigma(s^*)$ with separate widths per database and
for the CFs. The mean does not change, so $\kappa\,\Delta\sigma$ is an upper bound on
the rise of the optimal adjusted impact (re-optimizing can only do better).
`widen(data, r, exact_cfs=co2_flows(worker))` returns the widened data, with a
lognormal of the same mean and coefficient of variation (mirrored for negative
amounts), so the same data give a re-solved front and validation draws. Widths
are given per database name and `'Cf'`; a name that is not a subgroup of the data
raises.

## Out-of-sample validation

`validate` holds each solution $s^*$ and its target $z^*$ fixed and draws every
declared input from its own family, together with the uncertain bounds, from one
recorded seed. Every decision is priced on the same draws, so comparisons between
decisions are paired. Per point it reports:

- the coverage of the impact row against $\lambda_z$, of each bound against
  $1-\varepsilon_k$, and of all events together against $\lambda$, with Wilson
  intervals;
- the normal-approximation error (the empirical $\lambda_z$ quantile minus $z^*$),
  the skewness and excess kurtosis of $X$;
- the empirical CVaR at $\lambda_z$ against the Gaussian
  $\mu + \varphi(\Phi^{-1}(\lambda_z))/(1-\lambda_z)\,\sigma$;
- the regret against the lowest level of the front and against the best front
  point in each draw (mean and 95th percentile).

Further decisions, such as the deterministic optimum, are priced against the same
targets with `designs=`. Each parameter is drawn from its own stream, keyed by the
seed and its position, so its draws depend on the seed alone; decisions validated
in separate calls with the same seed are paired as well. Only B entries that carry
variance at the decisions are drawn; the smallest ones, below `tol` of each
decision's variance together, are held at their means, and the share drawn is
reported.

## Diagnostics

`diagnostics(front, moments)` reports at every point: $\sigma$ without the shared-CF
covariance and the ratio $\sigma/\sigma_{\text{indep}}$; the CFs' share of the
variance; the bound imposed on each uncertain process; the size of the reduced
problem and the solve times.

## Building blocks

The reduced system and the sampler are public, so a study can formulate its own
problem over $v$, such as a scenario-based CVaR front:

- `reduced.build(worker)` returns the `ReducedModel` of an instance: its
  `ReducedSystem` maps $v$ to $s$ (`base`, `recover`) and projects any linear
  functional of $s$ (`project`, `rows`, `project_vectors`); `linear_program()`
  assembles the constraint rows in $v$.
- `ChanceConstrained.projections()` returns $s_0$, $\mathrm{E}[X] = m_0 + m^\top v$,
  the rows $S_J$ of $S$ on the processes that carry variance and $\mathrm{E}[B_u]S$.
- `draw_parameters` returns draws of the declared parameters as arrays, and
  `sample_specs` draws any list of specs.

## Solvers and licences

| Problem | Solvers | Licence |
|---|---|---|
| deterministic LP, `method='full'` | HiGHS (default), Gurobi, GAMS/NEOS | HiGHS: none |
| deterministic LP, `method='reduced'` | HiGHS (default), Gurobi | HiGHS: none |
| chance-constrained cone | Clarabel (default), Gurobi | Clarabel: none |

HiGHS and Clarabel are installed with PULPO and need no licence. Gurobi is used
when `gurobipy` is installed; the size-limited licence that comes with
`pip install gurobipy` is for non-production use and covers reduced problems of up
to 2,000 variables and 2,000 linear constraints, or 200 variables once quadratic
terms (the cone) are present (see Gurobi's licence terms). A reduced problem has
one column per alternative, so most case studies fit.

The defaults stay `method='full'` and `instantiate(scale=False)`. Current
development considers `method='reduced'` and `scale=True` superior, and a future
release may switch the defaults.
