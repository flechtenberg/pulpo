"""Tests for PULPO's uncertainty features.

Systematic version of ``notebooks/uncertainty_toy.ipynb``, exercising the
curated ``pulpo_unc`` pipeline on the methanol + ozone sample system:

1. deterministic reference solve,
2. ``import_and_filter_uncertainty_data`` (impact cutoff filtering),
3. ``apply_uncertainty_strategies`` (triangular gap filling),
4. ``run_mc_from_uncertainty`` (Monte Carlo on the curated distributions),
5. ``create_CC_formulation`` + ``solve_CC_problem`` (chance constraints),
6. ``run_gsa`` (Sobol global sensitivity analysis).

The module also contains the bw25-only checks of the uncertainty-parameter
extraction in ``bw_parser.import_data`` (formerly test_bw25_uncertainty.py).

Requires the ``uncertainty`` extra (``pip install "pulpo-dev[uncertainty]"``);
the workflow classes skip themselves when those packages are missing.
"""

import unittest
import warnings

import numpy as np
import bw2data as bd

from pulpo.utils import bw_parser
from pulpo.utils.utils import is_bw25
from pulpo.datasets.sample_database import setup_sample_db

try:
    import pandas as pd
    import scipy.sparse
    import scipy.stats
    import stats_arrays
    from pulpo import pulpo_unc
    from pulpo.utils.uncertainty import cc, gsa, processor
    from pulpo.utils.uncertainty.processor import TriangluarBaseStrategy, DeterministicGapFillStrategy
    from pulpo.datasets.soc_demo_database import setup_soc_demo_db
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
    UNCERTAINTY_DEPS = True
    UNCERTAINTY_SKIP_REASON = ""
except ImportError as _err:  # pragma: no cover - depends on installed extras
    UNCERTAINTY_DEPS = False
    UNCERTAINTY_SKIP_REASON = (
        f"uncertainty extras not installed ({_err}); "
        'install with pip install "pulpo-dev[uncertainty]"'
    )

setup_sample_db()
if UNCERTAINTY_DEPS:
    setup_soc_demo_db()

PROJECT = "sample_project_bw25" if is_bw25() else "sample_project"
DATABASES = ["background_db", "foreground_db"]
CLIMATE_KEY = "('my project', 'climate change')"
METHODS = {CLIMATE_KEY: 1}

# Optimal climate-change impact of the deterministic methanol + ozone system;
# identical on both Brightway stacks.
DETERMINISTIC_IMPACT = 1.760427


def build_solved_worker():
    """Set up and solve the methanol + ozone system from the toy notebook."""
    worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, METHODS, "")
    worker.get_lci_data()

    methanol = worker.retrieve_processes(reference_products="methanol")
    ozone = worker.retrieve_processes(reference_products="ozone")
    electricity = worker.retrieve_processes(
        processes=["wind electricity", "natural gas electricity"])
    hydrogen = worker.retrieve_processes(
        processes=["hydrogen SMR", "hydrogen electrolysis"])
    oxygen = worker.retrieve_processes(processes=["O2-market", "O2 ASU"])
    oxygen_byproduct = worker.retrieve_processes(processes=["O2-byproduct"])

    demand = {methanol[0]: 1, ozone[0]: 2}
    choices = {
        "Electricity": {electricity[0]: 1e10, electricity[1]: 1e10},
        "Hydrogen": {hydrogen[0]: 1e10, hydrogen[1]: 1e10},
        "Oxygen": {oxygen[0]: 1e10, oxygen[1]: 1e10},
    }
    lower_limit = {oxygen_byproduct[0]: 0}

    worker.instantiate(choices=choices, demand=demand, lower_limit=lower_limit)
    worker.solve()
    return worker


def default_strategies(worker):
    """Explicit gap-filling strategies as used in the toy notebook:
    +-10% triangular for intervention flows, +-5% for characterization factors.
    """
    db_names = worker.database if isinstance(worker.database, list) else [worker.database]
    method_name = next(iter(worker.method))
    strategies = []
    for db in db_names:
        if db in worker.uncertainty_data.get("If", {}):
            strategies.append(TriangluarBaseStrategy(
                uncertain_param_type="If",
                uncertain_param_subgroup=db,
                upper_scaling_factor=0.1,
                lower_scaling_factor=0.1,
                noise_interval={"min": 0.1, "max": 0.1},
            ))
    if method_name in worker.uncertainty_data.get("Cf", {}):
        strategies.append(TriangluarBaseStrategy(
            uncertain_param_type="Cf",
            uncertain_param_subgroup=method_name,
            upper_scaling_factor=0.05,
            lower_scaling_factor=0.05,
            noise_interval={"min": 0.1, "max": 0.1},
        ))
    return strategies


def prepare_uncertainty(worker):
    """Run the import + strategy steps of the pipeline on a solved worker."""
    worker.import_and_filter_uncertainty_data(
        cutoff=0.001, scaling_vector_strategy="constructed_demand")
    worker.apply_uncertainty_strategies(
        strategies=default_strategies(worker), drop_undefined=True)


##################################################
#### bw25 uncertainty-parameter extraction    ####
##################################################

def setup_uncertainty_free_project():
    """Build a throwaway project with a single-process database and one CF,
    neither carrying any uncertainty. Used to exercise the ``None`` + warning
    branch of ``bw_parser.import_data`` without rebuilding a full example
    database (the realistic content is irrelevant to that code path)."""
    project = "sample_project_no_uncertainty"
    bd.projects.set_current(project)
    for db_name in ("no_uncertainty_db", "biosphere3"):
        if db_name in bd.databases:
            del bd.databases[db_name]

    co2 = ("biosphere3", "CO2")
    bd.Database("biosphere3").write({
        co2: {"name": "Carbon dioxide, fossil",
              "categories": ("climate change",),
              "type": "emission", "unit": "kg"},
    })
    bd.Database("no_uncertainty_db").write({
        ("no_uncertainty_db", "process"): {
            "name": "process", "unit": "kg", "location": "GLO",
            "reference product": "widget",
            "exchanges": [
                {"input": ("no_uncertainty_db", "process"),
                 "amount": 1.0, "type": "production"},
                # biosphere exchange without an 'uncertainty type' field
                {"input": co2, "amount": 2.0, "type": "biosphere"},
            ],
        },
    })
    for method in list(bd.methods):
        bd.Method(method).deregister()
    method = bd.Method(("my project", "climate change"))
    method.register(unit="kg CO2eq")
    method.write([(co2, 1.0)])  # bare CF value, no uncertainty
    return project


@unittest.skipUnless(is_bw25(), "bw25-only: structured uncertainty-parameter "
                                "arrays require bw2data >= 4")
class TestUncertaintyParamArrays(unittest.TestCase):
    """``bw_parser.import_data`` must expose combined structured arrays for
    the uncertainty preparer (or ``None`` + a warning when the databases carry
    no uncertainty)."""

    REQUIRED_FIELDS = (
        "row", "col", "amount", "uncertainty_type",
        "loc", "scale", "shape", "minimum", "maximum", "negative",
    )

    def test_with_uncertainty(self):
        lci_data = bw_parser.import_data(
            project=PROJECT,
            databases=DATABASES,
            method=CLIMATE_KEY,
            intervention_matrix_name="biosphere3",
            seed=42,
        )
        method_key = next(iter(lci_data["matrices"]))

        int_params = lci_data["intervention_params"]
        cf_params = lci_data["characterization_params"][method_key]
        self.assertIsNotNone(int_params)
        self.assertIsNotNone(cf_params)

        # Structured arrays with the fields the preparer relies on.
        for name, arr in (("intervention_params", int_params),
                          ("characterization_params", cf_params)):
            self.assertIsNotNone(arr.dtype.names, f"{name} must be a structured array")
            for field in self.REQUIRED_FIELDS:
                self.assertIn(field, arr.dtype.names, f"{name} missing field '{field}'")

        # Uncertainty actually populated (sample db uses NormalUncertainty).
        self.assertTrue((int_params["uncertainty_type"] > 0).any())
        self.assertTrue((cf_params["uncertainty_type"] > 0).any())

        # Row/col must be mapped to matrix positions (small, contiguous
        # integers), not raw brightway ids (huge 64-bit numbers).
        self.assertLess(int_params["row"].max(), 10_000)
        self.assertLess(int_params["col"].max(), 10_000)
        self.assertLess(cf_params["row"].max(), 10_000)

        # The preparer indexes intervention params on (row, col); those must
        # be unique for set_index to behave.
        pairs = list(zip(int_params["row"].tolist(), int_params["col"].tolist()))
        self.assertEqual(len(pairs), len(set(pairs)))

        # Params must be accumulated across BOTH databases, i.e. cover more
        # than a single database's process columns.
        self.assertGreater(len(set(int_params["col"].tolist())), 1)

    def test_without_uncertainty_stores_none_and_warns(self):
        # A minimal uncertainty-free database exercises the None + warning
        # branch far more cheaply than rebuilding a full example database.
        project = setup_uncertainty_free_project()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lci_data = bw_parser.import_data(
                project=project,
                databases=["no_uncertainty_db"],
                method=CLIMATE_KEY,
                intervention_matrix_name="biosphere3",
                seed=42,
            )
        method_key = next(iter(lci_data["matrices"]))

        self.assertIsNone(lci_data["intervention_params"])
        self.assertIsNone(lci_data["characterization_params"][method_key])

        messages = [str(w.message) for w in caught
                    if issubclass(w.category, UserWarning)]
        self.assertTrue(any("intervention" in m for m in messages),
                        "expected a warning about intervention params")
        self.assertTrue(any("characterization" in m for m in messages),
                        "expected a warning about characterization params")


##################################################
#### Curated uncertainty pipeline (pulpo_unc) ####
##################################################

@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestUncertaintyPipeline(unittest.TestCase):
    """Deterministic solve + uncertainty import/filter + gap-filling."""

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_worker()
        cls.deterministic_results = cls.worker.extract_results()
        cls.worker.import_and_filter_uncertainty_data(
            cutoff=0.001, scaling_vector_strategy="constructed_demand")
        # Snapshot the imported structure before the strategies run.
        cls.imported_counts = {
            section: {
                subgroup: (len(entries["defined"]), len(entries["undefined"]))
                for subgroup, entries in cls.worker.uncertainty_data[section].items()
            }
            for section in cls.worker.uncertainty_data
        }
        cls.worker.apply_uncertainty_strategies(
            strategies=default_strategies(cls.worker), drop_undefined=True)

    def test_deterministic_reference_impact(self):
        impact = self.deterministic_results["Impacts"].loc[CLIMATE_KEY, "Value"]
        self.assertAlmostEqual(impact, DETERMINISTIC_IMPACT, places=6)

    def test_imported_uncertainty_structure(self):
        self.assertEqual(sorted(self.imported_counts.keys()),
                         ["Cf", "If", "Var_bounds"])
        # The sample databases carry NormalUncertainty on every exchange, so
        # everything surviving the cutoff is 'defined'. Both stacks keep the
        # same intervention flows: the constructed-demand scaling vector is now
        # aligned identically on bw2 and bw25.
        expected_if = {"background_db": (4, 0), "foreground_db": (1, 0)}
        self.assertEqual(self.imported_counts["If"], expected_if)
        self.assertEqual(self.imported_counts["Cf"], {CLIMATE_KEY: (2, 0)})
        # Variable bounds arrive without distributions: the six choice
        # alternatives (upper) and the O2-byproduct lower limit.
        self.assertEqual(self.imported_counts["Var_bounds"]["upper_limit"], (0, 6))
        self.assertEqual(self.imported_counts["Var_bounds"]["lower_limit"], (0, 1))

    def test_strategies_leave_no_undefined_parameters(self):
        self.assertFalse(
            processor.check_missing_uncertainty_data(self.worker.uncertainty_data))

    def test_import_rejects_multiple_methods(self):
        two_methods = dict(METHODS)
        two_methods["('my project', 'air quality')"] = 1
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, two_methods, "")
        with self.assertRaises(Exception) as context:
            worker.import_and_filter_uncertainty_data()
        self.assertIn("single LCIA method", str(context.exception))

    def test_strategies_require_imported_data(self):
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, METHODS, "")
        with self.assertRaises(Exception) as context:
            worker.apply_uncertainty_strategies()
        self.assertIn("import_and_filter_uncertainty_data", str(context.exception))

    def test_mc_requires_uncertainty_data(self):
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, METHODS, "")
        with self.assertRaises(Exception) as context:
            worker.run_mc_from_uncertainty(n_samples=2)
        self.assertIn("No uncertainty data", str(context.exception))

    def test_naive_strategy_requires_result_data(self):
        worker = build_solved_worker()
        with self.assertRaises(Exception) as context:
            worker.import_and_filter_uncertainty_data(
                cutoff=0.001, scaling_vector_strategy="naive")
        self.assertIn("result_data", str(context.exception))

        # With the deterministic results passed in, the naive strategy works.
        worker.import_and_filter_uncertainty_data(
            cutoff=0.001,
            scaling_vector_strategy="naive",
            result_data=worker.extract_results(),
        )
        self.assertIn("If", worker.uncertainty_data)
        self.assertIn("Cf", worker.uncertainty_data)
        total_defined = sum(
            len(entries["defined"])
            for entries in worker.uncertainty_data["If"].values()
        )
        self.assertGreater(total_defined, 0)

    def test_none_strategy_retains_every_exchange(self):
        """'none' disables the filter: one parameter per nonzero intervention entry.

        The contribution filter selects parameters at a single scaling vector, so
        exchanges belonging to processes that are inactive there are dropped and
        enter a chance-constrained problem carrying no uncertainty. 'none' is the
        opt-out for systems small enough not to need the filter.
        """
        worker = build_solved_worker()
        worker.import_and_filter_uncertainty_data(scaling_vector_strategy="none")

        n_exchanges = worker.lci_data["intervention_matrix"].nnz
        n_imported = sum(
            len(entries["defined"]) + len(entries["undefined"])
            for entries in worker.uncertainty_data["If"].values()
        )
        self.assertEqual(n_imported, n_exchanges)

        # And it is a strict superset of what the naive filter keeps.
        filtered = build_solved_worker()
        filtered.import_and_filter_uncertainty_data(
            cutoff=0.0,
            scaling_vector_strategy="naive",
            result_data=filtered.extract_results(),
        )
        n_filtered = sum(
            len(entries["defined"]) + len(entries["undefined"])
            for entries in filtered.uncertainty_data["If"].values()
        )
        self.assertGreaterEqual(n_imported, n_filtered)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestMonteCarloFromUncertainty(unittest.TestCase):
    """``run_mc_from_uncertainty``: MC on the curated distributions."""

    N_SAMPLES = 20

    @classmethod
    def setUpClass(cls):
        worker = build_solved_worker()
        prepare_uncertainty(worker)
        # n_jobs=1 solves sequentially in-process: spawning a joblib worker
        # pool costs far more than these 20 tiny LPs.
        cls.mc_results = worker.run_mc_from_uncertainty(
            n_samples=cls.N_SAMPLES, seed=42, n_jobs=1)
        cls.samples = np.array([
            cls.mc_results[i]["Impacts"].loc[CLIMATE_KEY, "Value"]
            for i in cls.mc_results
            if "error" not in cls.mc_results[i]
        ])

    def test_result_structure(self):
        self.assertIsInstance(self.mc_results, dict)
        self.assertEqual(len(self.mc_results), self.N_SAMPLES)
        self.assertEqual(len(self.samples), self.N_SAMPLES,
                         "no MC iteration should have errored")

    def test_seeded_samples_match_reference(self):
        # Both stacks now filter to the same parameter set and pair it with the
        # same seeds, so the seeded draws agree across bw2 and bw25.
        self.assertAlmostEqual(self.samples.mean(), 1.754014, places=5)
        self.assertAlmostEqual(self.samples.std(), 0.199952, places=5)

    def test_samples_scatter_around_deterministic_optimum(self):
        self.assertTrue(np.isfinite(self.samples).all())
        self.assertGreater(self.samples.std(), 0)
        self.assertLess(abs(self.samples.mean() - DETERMINISTIC_IMPACT),
                        0.2 * DETERMINISTIC_IMPACT)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestChanceConstrained(unittest.TestCase):
    """``create_CC_formulation`` + ``solve_CC_problem`` Pareto trace."""

    LAMBDAS = [0.50, 0.75, 0.90, 0.95]

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_worker()
        prepare_uncertainty(cls.worker)
        # The normal transformation samples the triangular distributions;
        # stats_arrays draws from numpy's global RNG, so seed it for
        # reproducible CC statistics.
        np.random.seed(42)
        cls.env_meta, cls.var_bounds_meta = cls.worker.create_CC_formulation(
            CC_env_cost=True,
            CC_var_bounds=[],
            normal_transformation_sample_size=1000,
        )
        cls.results = cls.worker.solve_CC_problem(
            lambda_level=cls.LAMBDAS,
            normal_metadata_env_cost=cls.env_meta,
            normal_metadata_var_bounds=cls.var_bounds_meta,
            plot_results=False,
        )
        cls.impacts = {lam: cls.results[lam]["Impacts"].loc[CLIMATE_KEY, "Value"]
                       for lam in cls.LAMBDAS}

    def test_env_cost_metadata_structure(self):
        self.assertGreater(len(self.env_meta), 0)
        method_name = next(iter(self.worker.method))
        for (process_id, method), spec in self.env_meta.items():
            self.assertEqual(method, method_name)
            self.assertIn("loc", spec)
            self.assertIn("scale", spec)
        self.assertEqual(self.var_bounds_meta, {})

    def test_lambda_50_reproduces_deterministic_optimum(self):
        # At lambda = 0.5 the safety margin (norm ppf) is zero, so the CC
        # problem collapses onto the deterministic one.
        self.assertAlmostEqual(self.impacts[0.50], DETERMINISTIC_IMPACT, places=6)

    def test_impact_increases_with_confidence_level(self):
        trace = [self.impacts[lam] for lam in self.LAMBDAS]
        for lower, upper in zip(trace, trace[1:]):
            self.assertGreater(upper, lower)

    def test_seeded_pareto_trace_matches_reference(self):
        expected = {0.75: 1.884391, 0.90: 1.995962, 0.95: 2.062733}
        for lam, value in expected.items():
            self.assertAlmostEqual(self.impacts[lam], value, places=5)

    def test_formulation_requires_a_target(self):
        with self.assertRaises(Exception) as context:
            self.worker.create_CC_formulation(CC_env_cost=False, CC_var_bounds=[])
        self.assertIn("No CC formulation specified", str(context.exception))


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestExactSOC(unittest.TestCase):
    """``create_SOC_formulation`` + ``solve_SOC_problem`` exact-variance Pareto trace."""

    LAMBDAS = [0.50, 0.75, 0.90, 0.95]

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_worker()
        prepare_uncertainty(cls.worker)
        np.random.seed(42)
        cls.coeffs = cls.worker.create_SOC_formulation(
            normal_transformation_sample_size=1000,
        )
        cls.results = cls.worker.solve_SOC_problem(
            lambda_level=cls.LAMBDAS,
            coeffs=cls.coeffs,
            plot_results=False,
        )
        cls.impacts = {lam: cls.results[lam]["Impacts"].loc[CLIMATE_KEY, "Value"]
                       for lam in cls.LAMBDAS}

    def test_coefficients_structure(self):
        summary = self.coeffs.summary()
        self.assertGreater(summary["processes"], 0)
        self.assertEqual(self.coeffs.method, CLIMATE_KEY)

    def test_lambda_50_reproduces_deterministic_optimum(self):
        # At lambda = 0.5 the safety margin (norm ppf) is zero, so the SOC
        # problem collapses onto the deterministic one, same as the L1 path.
        self.assertAlmostEqual(self.impacts[0.50], DETERMINISTIC_IMPACT, places=5)

    def test_impact_increases_with_confidence_level(self):
        trace = [self.impacts[lam] for lam in self.LAMBDAS]
        for lower, upper in zip(trace, trace[1:]):
            self.assertGreater(upper, lower)

    def test_seeded_pareto_trace_matches_reference(self):
        expected = {0.75: 1.879708, 0.90: 1.987064, 0.95: 2.051313}
        for lam, value in expected.items():
            self.assertAlmostEqual(self.impacts[lam], value, places=5)

    def test_never_exceeds_l1_shortcut(self):
        # The L1 shortcut sums per-process sigmas with an L1 norm, which
        # over-estimates the true (Euclidean) uncertainty and ignores the
        # covariance shared processes pick up through a common
        # characterization factor - a correct exact-SOC front can therefore
        # only lie at or below the L1 front. Uses an independent worker so
        # this comparison can't be perturbed by (or perturb) `cls.worker`'s
        # instance state, which `apply_CC_formulation` mutates permanently.
        worker = build_solved_worker()
        prepare_uncertainty(worker)
        np.random.seed(42)
        env_meta, var_bounds_meta = worker.create_CC_formulation(
            CC_env_cost=True, CC_var_bounds=[], normal_transformation_sample_size=1000,
        )
        l1_results = worker.solve_CC_problem(
            lambda_level=self.LAMBDAS,
            normal_metadata_env_cost=env_meta,
            normal_metadata_var_bounds=var_bounds_meta,
            plot_results=False,
        )
        for lam in self.LAMBDAS:
            l1_impact = l1_results[lam]["Impacts"].loc[CLIMATE_KEY, "Value"]
            self.assertGreaterEqual(l1_impact, self.impacts[lam] - 1e-9)

    def test_direct_method_matches_cutting_plane(self):
        # Same coefficients, single lambda: the direct QCP solve (Gurobi) and
        # the cutting-plane LP solve (any solver) should agree on this small,
        # well-scaled toy system. Independent worker/instance, so it can run
        # in any order relative to the other tests.
        worker = build_solved_worker()
        prepare_uncertainty(worker)
        np.random.seed(42)
        coeffs = worker.create_SOC_formulation(normal_transformation_sample_size=1000)
        try:
            direct_results = worker.solve_SOC_problem(
                lambda_level=0.90, coeffs=coeffs, method='direct',
            )
        except Exception as exc:
            self.skipTest(f"'gurobi' Pyomo solver unavailable: {exc}")
        direct_impact = direct_results[0.90]["Impacts"].loc[CLIMATE_KEY, "Value"]
        self.assertAlmostEqual(direct_impact, self.impacts[0.90], places=4)

    def test_restore_deterministic_objective(self):
        self.worker.restore_deterministic_objective()
        self.worker.solve()
        result = self.worker.extract_results()
        self.assertAlmostEqual(
            result["Impacts"].loc[CLIMATE_KEY, "Value"], DETERMINISTIC_IMPACT, places=5)

    def test_formulation_requires_uncertainty_data(self):
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, METHODS, "")
        with self.assertRaises(Exception) as context:
            worker.create_SOC_formulation()
        self.assertIn("import_and_filter_uncertainty_data", str(context.exception))


class _RecordingSampler:
    """Stand-in for the SALib sampler module that keeps the design matrix.

    ``run_gsa`` takes the sampler as an object with a ``.sample`` attribute, so
    wrapping the real module is enough to observe what the seed did without
    reaching into ``GlobalSensitivityAnalysis``.
    """

    def __init__(self, module):
        self._module = module
        self.samples = None

    def sample(self, problem, N, **kwargs):
        self.samples = self._module.sample(problem, N, **kwargs)
        return self.samples


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestGlobalSensitivityAnalysis(unittest.TestCase):
    """``run_gsa``: Sobol sensitivity indices on the curated parameters."""

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_worker()
        cls.deterministic_results = cls.worker.extract_results()
        prepare_uncertainty(cls.worker)
        cls.total_order, cls.sensitivity_indices = cls.worker.run_gsa(
            result_data=cls.deterministic_results,
            sample_method=sobol_sample,
            SA_method=sobol_analyze,
            sample_size=32,  # tiny, as in the showcase notebook
            plot_gsa_results=False,
            top_sensitivity_amt=10,
        )

    def test_output_structure(self):
        self.assertEqual(list(self.total_order.columns), ["ST", "ST_conf"])
        # All filtered parameters take part: five intervention flows + two CFs
        # (identical on both stacks now that filtering is aligned).
        self.assertEqual(len(self.total_order), 7)
        for key in ("S1", "ST"):
            self.assertIn(key, self.sensitivity_indices)

    def test_parameter_names_resolve_against_metadata(self):
        # If-parameters are (intervention_idx, process_idx) tuples,
        # Cf-parameters bare intervention indices - both must resolve against
        # the lci_data metadata maps (as done for labeling in the notebook).
        process_map = self.worker.lci_data["process_map_metadata"]
        intervention_map = self.worker.lci_data["intervention_map_metadata"]
        cf_params = 0
        for param in self.total_order.index:
            if isinstance(param, tuple):
                intervention_idx, process_idx = param
                self.assertIn(intervention_idx, intervention_map)
                self.assertIn(process_idx, process_map)
            else:
                cf_params += 1
                self.assertIn(param, intervention_map)
        self.assertEqual(cf_params, 2)

    def test_sobol_indices_plausible(self):
        st = self.total_order["ST"]
        self.assertTrue(np.isfinite(st).all())
        # Some parameter must explain a substantial share of the variance;
        # on both stacks the CO2 characterization factor dominates.
        self.assertGreater(st.max(), 0.3)
        self.assertNotIsInstance(st.idxmax(), tuple,
                                 "top driver should be a characterization factor")

    def test_seed_controls_sampling(self):
        """``run_gsa(seed=...)`` reaches SALib's sampler.

        Three claims in one pass, because each run costs a full sample sweep:
        the default is still 161 (results published before the argument existed
        stay reproducible), the same seed reproduces the design matrix exactly,
        and a different seed produces a different one.
        """
        def draw(**seed_kwarg):
            recorder = _RecordingSampler(sobol_sample)
            total_order, _ = self.worker.run_gsa(
                result_data=self.deterministic_results,
                sample_method=recorder,
                SA_method=sobol_analyze,
                sample_size=16,
                plot_gsa_results=False,
                **seed_kwarg,
            )
            return recorder.samples, total_order

        default_samples, default_order = draw()
        explicit_161_samples, _ = draw(seed=161)
        other_samples, other_order = draw(seed=2024)
        other_samples_again, _ = draw(seed=2024)

        np.testing.assert_array_equal(default_samples, explicit_161_samples)
        np.testing.assert_array_equal(other_samples, other_samples_again)
        self.assertEqual(default_samples.shape, other_samples.shape)
        self.assertFalse(np.array_equal(default_samples, other_samples),
                         "a different seed must give a different design matrix")
        # The seed moves the numbers, not the story: the leading driver is a
        # property of the model and must survive the reseeding.
        self.assertEqual(default_order["ST"].idxmax(), other_order["ST"].idxmax())


##################################################
#### SOC-demo system (soc_demo_database)      ####
##################################################
# sample_database.py gives every non-production exchange a blanket
# NormalUncertainty, so it has zero native non-Normal parameters and zero
# undefined ones (see soc_demo_database's module docstring) - it cannot
# exercise DeterministicGapFillStrategy or compute_closed_form_moments
# meaningfully. soc_demo_database.py's ammonia/hydrogen-route toy system was
# purpose-built with native Normal/Lognormal/Triangular/Uniform parameters
# and several genuinely undefined ones.

SOC_DEMO_PROJECT = "soc_demo_project_bw25" if is_bw25() else "soc_demo_project"
SOC_DEMO_DATABASES = ["soc_demo_background_db", "soc_demo_foreground_db"]
SOC_DEMO_METHOD_KEY = "('soc demo', 'climate change')"
SOC_DEMO_METHODS = {SOC_DEMO_METHOD_KEY: 1}


def build_solved_soc_demo_worker():
    """Ammonia-synthesis toy system: hydrogen route choice (SMR vs.
    electrolysis). Electrolysis capacity is capped below full demand so both
    routes carry nonzero scaling at the deterministic optimum - the 'naive'
    cutoff-filter strategy below only keeps parameters on processes with
    nonzero scaling, and every native distribution family in this system
    lives on one route or the other (see soc_demo_database's module
    docstring).
    """
    worker = pulpo_unc.PulpoOptimizerUnc(SOC_DEMO_PROJECT, SOC_DEMO_DATABASES, SOC_DEMO_METHODS, "")
    worker.get_lci_data()
    ammonia = worker.retrieve_processes(reference_products="ammonia")
    hydrogen = worker.retrieve_processes(processes=["hydrogen SMR", "hydrogen electrolysis"])
    demand = {ammonia[0]: 1}
    choices = {"Hydrogen route": {hydrogen[0]: 1e10, hydrogen[1]: 0.09}}
    worker.instantiate(choices=choices, demand=demand)
    worker.solve()
    return worker


def prepare_soc_demo_uncertainty(worker, gap_fill="triangular"):
    """Import + filter + gap-fill for the SOC-demo system.

    ``gap_fill='triangular'`` uses the existing +-10% strategy (all gaps
    filled); ``gap_fill='deterministic'`` uses ``DeterministicGapFillStrategy``
    (no gaps filled - degenerate N(amount, 0) instead).
    """
    det_result = worker.extract_results()
    worker.import_and_filter_uncertainty_data(
        cutoff=0.0, scaling_vector_strategy="naive", result_data=det_result)
    method_name = next(iter(worker.method))
    if gap_fill == "triangular":
        strategies = [
            TriangluarBaseStrategy("If", "soc_demo_background_db", 0.1, 0.1,
                                    noise_interval={"min": 0.1, "max": 0.1}),
            TriangluarBaseStrategy("If", "soc_demo_foreground_db", 0.1, 0.1,
                                    noise_interval={"min": 0.1, "max": 0.1}),
            TriangluarBaseStrategy("Cf", method_name, 0.1, 0.1,
                                    noise_interval={"min": 0.1, "max": 0.1}),
        ]
    elif gap_fill == "deterministic":
        strategies = [
            DeterministicGapFillStrategy("If", "soc_demo_background_db"),
            DeterministicGapFillStrategy("If", "soc_demo_foreground_db"),
            DeterministicGapFillStrategy("Cf", method_name),
        ]
    else:
        raise ValueError(gap_fill)
    worker.apply_uncertainty_strategies(strategies=strategies, drop_undefined=True)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestDeterministicGapFillStrategy(unittest.TestCase):
    """``DeterministicGapFillStrategy``: the 'no gap filling' alternative to
    ``TriangluarBaseStrategy`` - fills undefined entries with a degenerate
    N(amount, 0) instead of spreading them out."""

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_soc_demo_worker()
        det_result = cls.worker.extract_results()
        cls.worker.import_and_filter_uncertainty_data(
            cutoff=0.0, scaling_vector_strategy="naive", result_data=det_result)
        # Snapshot the pre-fill undefined entries so the post-fill values can
        # be checked against them.
        cls.pre_fill_undefined = {
            (unc_type, subgroup): dict(entries["undefined"])
            for unc_type in ("If", "Cf")
            for subgroup, entries in cls.worker.uncertainty_data[unc_type].items()
            if entries["undefined"]
        }
        method_name = next(iter(cls.worker.method))
        cls.worker.apply_uncertainty_strategies(
            strategies=[
                DeterministicGapFillStrategy("If", "soc_demo_background_db"),
                DeterministicGapFillStrategy("If", "soc_demo_foreground_db"),
                DeterministicGapFillStrategy("Cf", method_name),
            ],
            drop_undefined=True,
        )

    def test_gaps_existed_before_filling(self):
        # Sanity check the fixture actually has something to gap-fill (unlike
        # sample_database.py, which has zero undefined 'If'/'Cf' parameters).
        self.assertGreater(sum(len(d) for d in self.pre_fill_undefined.values()), 0)

    def test_no_undefined_parameters_remain(self):
        self.assertFalse(
            processor.check_missing_uncertainty_data(self.worker.uncertainty_data))

    def test_filled_entries_are_degenerate_normals_at_amount(self):
        for (unc_type, subgroup), undefined in self.pre_fill_undefined.items():
            defined = self.worker.uncertainty_data[unc_type][subgroup]["defined"]
            for idx, spec in undefined.items():
                filled = defined[idx]
                self.assertEqual(filled["uncertainty_type"], stats_arrays.NormalUncertainty.id)
                self.assertEqual(filled["scale"], 0.0)
                self.assertEqual(filled["loc"], spec["amount"])


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestClosedFormMoments(unittest.TestCase):
    """``compute_closed_form_moments``: analytic (mean, std) per distribution
    family, against hand-computed reference values."""

    def test_matches_hand_computed_moments(self):
        uncertainty_data = {
            "If": {
                "db": {
                    "defined": {
                        "normal": {"uncertainty_type": stats_arrays.NormalUncertainty.id,
                                   "amount": 1.0, "loc": 2.0, "scale": 0.5},
                        "uniform": {"uncertainty_type": stats_arrays.UniformUncertainty.id,
                                    "amount": 1.0, "minimum": 2.0, "maximum": 6.0},
                        "triangular": {"uncertainty_type": stats_arrays.TriangularUncertainty.id,
                                       "amount": 1.0, "loc": 3.0, "minimum": 1.0, "maximum": 8.0},
                        "lognormal": {"uncertainty_type": stats_arrays.LognormalUncertainty.id,
                                      "amount": 1.0, "loc": 1.0, "scale": 0.3, "negative": False},
                    },
                    "undefined": {},
                },
            },
        }
        moments = processor.compute_closed_form_moments(uncertainty_data, unc_types=["If"])
        result = moments["If"]["db"]["defined"]

        # Normal: passthrough.
        self.assertAlmostEqual(result["normal"]["loc"], 2.0)
        self.assertAlmostEqual(result["normal"]["scale"], 0.5)

        # Uniform: mean=(min+max)/2, std=sqrt((max-min)^2/12).
        self.assertAlmostEqual(result["uniform"]["loc"], 4.0)
        self.assertAlmostEqual(result["uniform"]["scale"], np.sqrt((6.0 - 2.0) ** 2 / 12))

        # Triangular: mean=(min+max+mode)/3, variance = the standard
        # triangular-distribution formula.
        a, b, c = 1.0, 8.0, 3.0  # minimum, maximum, mode
        expected_mean = (a + b + c) / 3
        expected_var = (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18
        self.assertAlmostEqual(result["triangular"]["loc"], expected_mean)
        self.assertAlmostEqual(result["triangular"]["scale"], np.sqrt(expected_var))

        # Lognormal: mean=exp(mu+sigma^2/2), std=sqrt((exp(sigma^2)-1)*exp(2mu+sigma^2)).
        mu, sigma = 1.0, 0.3
        expected_mean = np.exp(mu + sigma ** 2 / 2)
        expected_std = np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2))
        self.assertAlmostEqual(result["lognormal"]["loc"], expected_mean)
        self.assertAlmostEqual(result["lognormal"]["scale"], expected_std)

        # Every output is tagged Normal, matching transform_to_normal's shape.
        for spec in result.values():
            self.assertEqual(spec["uncertainty_type"], stats_arrays.NormalUncertainty.id)

    def test_unsupported_uncertainty_type_raises(self):
        uncertainty_data = {
            "If": {"db": {"defined": {0: {"uncertainty_type": 1, "amount": 1.0}}, "undefined": {}}},
        }
        with self.assertRaises(NotImplementedError) as context:
            processor.compute_closed_form_moments(uncertainty_data, unc_types=["If"])
        self.assertIn("uncertainty_type=1", str(context.exception))

    def test_missing_data_guard_matches_transform_to_normal(self):
        uncertainty_data = {
            "If": {"db": {"defined": {}, "undefined": {0: {"uncertainty_type": 0, "amount": 1.0}}}},
        }
        with self.assertRaises(Exception) as context:
            processor.compute_closed_form_moments(uncertainty_data, unc_types=["If"])
        self.assertIn("undefined uncertainty data", str(context.exception))


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestSOCClosedFormMoments(unittest.TestCase):
    """``create_SOC_formulation(moments='closed_form')`` end-to-end on
    soc_demo_database - the only fixture with native non-Normal parameters
    that can exercise this meaningfully (mirrors ``TestExactSOC``)."""

    LAMBDAS = [0.50, 0.75, 0.90, 0.95]

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_soc_demo_worker()
        prepare_soc_demo_uncertainty(cls.worker, gap_fill="deterministic")
        cls.coeffs = cls.worker.create_SOC_formulation(
            normal_transformation_sample_size=1000, moments="closed_form")
        cls.results = cls.worker.solve_SOC_problem(
            lambda_level=cls.LAMBDAS, coeffs=cls.coeffs, plot_results=False)
        cls.impacts = {lam: cls.results[lam]["Impacts"].loc[SOC_DEMO_METHOD_KEY, "Value"]
                       for lam in cls.LAMBDAS}

    def test_coefficients_structure(self):
        summary = self.coeffs.summary()
        self.assertGreater(summary["processes"], 0)
        self.assertEqual(self.coeffs.method, SOC_DEMO_METHOD_KEY)

    def test_impact_increases_with_confidence_level(self):
        trace = [self.impacts[lam] for lam in self.LAMBDAS]
        for lower, upper in zip(trace, trace[1:]):
            self.assertGreater(upper, lower)

    def test_closed_form_close_to_fit(self):
        # closed_form skips resampling entirely; fit resamples and refits a
        # Normal. On the same (deterministic-filled) uncertainty data the two
        # should agree closely - a large discrepancy would mean a bug in the
        # closed-form formulas (or in reading minimum/maximum/loc off the
        # wrong fields), not sampling noise.
        worker = build_solved_soc_demo_worker()
        prepare_soc_demo_uncertainty(worker, gap_fill="deterministic")
        np.random.seed(42)
        coeffs_fit = worker.create_SOC_formulation(
            normal_transformation_sample_size=2000, moments="fit")
        results_fit = worker.solve_SOC_problem(lambda_level=self.LAMBDAS, coeffs=coeffs_fit)
        for lam in self.LAMBDAS:
            fit_impact = results_fit[lam]["Impacts"].loc[SOC_DEMO_METHOD_KEY, "Value"]
            self.assertAlmostEqual(fit_impact, self.impacts[lam],
                                   delta=0.05 * abs(self.impacts[lam]))

    def test_never_exceeds_l1_shortcut(self):
        # create_CC_formulation (L1) has no 'closed_form' option - it always
        # fits Normals via processor.transform_to_normal - so this compares
        # against a SOC(moments='fit') worker, not cls.worker's closed_form
        # coefficients: the "L1 never lies below SOC" guarantee is about the
        # L1-vs-Euclidean norm on the *variance* term (SI derivation), for a
        # shared set of per-parameter moments. Comparing fit-based L1 against
        # closed-form SOC would instead mix in the (unrelated, and at
        # lambda=0.5 undefined-sign) discrepancy between two different mean
        # estimators. Independent workers throughout: apply_CC_formulation
        # mutates its instance permanently.
        l1_worker = build_solved_soc_demo_worker()
        prepare_soc_demo_uncertainty(l1_worker, gap_fill="deterministic")
        np.random.seed(42)
        env_meta, var_bounds_meta = l1_worker.create_CC_formulation(
            CC_env_cost=True, CC_var_bounds=[], normal_transformation_sample_size=1000)
        l1_results = l1_worker.solve_CC_problem(
            lambda_level=self.LAMBDAS, normal_metadata_env_cost=env_meta,
            normal_metadata_var_bounds=var_bounds_meta, plot_results=False)

        soc_fit_worker = build_solved_soc_demo_worker()
        prepare_soc_demo_uncertainty(soc_fit_worker, gap_fill="deterministic")
        np.random.seed(42)
        coeffs_fit = soc_fit_worker.create_SOC_formulation(
            normal_transformation_sample_size=1000, moments="fit")
        soc_fit_results = soc_fit_worker.solve_SOC_problem(
            lambda_level=self.LAMBDAS, coeffs=coeffs_fit, plot_results=False)

        for lam in self.LAMBDAS:
            l1_impact = l1_results[lam]["Impacts"].loc[SOC_DEMO_METHOD_KEY, "Value"]
            soc_impact = soc_fit_results[lam]["Impacts"].loc[SOC_DEMO_METHOD_KEY, "Value"]
            self.assertGreaterEqual(l1_impact, soc_impact - 1e-9)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestDrawUncertaintySampleSeeding(unittest.TestCase):
    """``draw_uncertainty_sample(seed=...)`` must seed *every* family.

    Regression test for a defect that survived because the only other seeded
    sampling test runs on a fixture whose parameters are all Normal.  Normal
    specs are drawn from the seeded ``Generator`` directly; every other family
    is delegated to stats_arrays, which silently falls back to the legacy
    global ``np.random`` unless ``seeded_random`` is passed.  The effect was
    that lognormal, triangular and uniform parameters ignored the seed, so any
    Monte Carlo built on them was irreproducible -- and the SOC demo system's
    dominant driver is one of them.
    """

    @classmethod
    def setUpClass(cls):
        cls.worker = build_solved_soc_demo_worker()
        prepare_soc_demo_uncertainty(cls.worker, gap_fill="deterministic")
        cls.method = next(iter(cls.worker.method))

    def _draw(self, seed):
        return processor.draw_uncertainty_sample(
            self.worker.uncertainty_data, self.method, seed=seed)

    def test_fixture_actually_contains_non_normal_parameters(self):
        """Guard: without this the rest of the class could pass vacuously."""
        families = {
            spec.get("uncertainty_type")
            for group in ("If", "Cf")
            for block in self.worker.uncertainty_data.get(group, {}).values()
            for spec in block.get("defined", {}).values()
        }
        self.assertTrue(
            families - {stats_arrays.NormalUncertainty.id},
            "fixture has only Normal parameters; it cannot detect the defect")

    def test_same_seed_reproduces_despite_global_rng_use(self):
        first = self._draw(123)
        np.random.seed(999)          # disturb the legacy global stream
        np.random.random(17)
        second = self._draw(123)
        for group in ("If", "Cf"):
            self.assertEqual(first[group], second[group],
                             f"{group} draw depends on the global RNG, not the seed")

    def test_different_seeds_give_different_draws(self):
        first, other = self._draw(123), self._draw(124)
        self.assertNotEqual(first["If"], other["If"])


##################################################
#### Unbounded variable bounds                ####
##################################################

@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestUnboundedVariableBounds(unittest.TestCase):
    """An alternative declared infinite carries no chance-constrained row.

    ``converter.combine_inputs`` already treats an infinite upper bound as "no
    limit"; registering one as an uncertain parameter would give the closed-form
    triangular an ``inf`` amount and a ``nan`` variance. It also matters for a
    joint formulation: a sentinel row consumes risk budget it can never use.
    """

    def _upper_rows(self, capacity):
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, METHODS, "")
        worker.get_lci_data()
        methanol = worker.retrieve_processes(reference_products="methanol")
        hydrogen = worker.retrieve_processes(
            processes=["hydrogen SMR", "hydrogen electrolysis"])
        worker.instantiate(
            demand={methanol[0]: 1},
            choices={"Hydrogen": {hydrogen[0]: capacity, hydrogen[1]: 1e10}})
        worker.solve()
        worker.import_and_filter_uncertainty_data(
            cutoff=0.001, scaling_vector_strategy="constructed_demand")
        return worker.uncertainty_data["Var_bounds"]["upper_limit"]["undefined"]

    def test_finite_capacity_is_registered(self):
        self.assertEqual(len(self._upper_rows(1e10)), 2)

    def test_infinite_capacity_is_skipped(self):
        # Only the finite sibling survives; the unbounded one is not a bound.
        # NaN is not tested here because it cannot reach this code: Pyomo
        # rejects it first when constructing UPPER_LIMIT, whose domain is Reals.
        self.assertEqual(len(self._upper_rows(float("inf"))), 1)


##################################################
#### Joint chance constraints                 ####
##################################################

@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestRiskBudget(unittest.TestCase):
    """``bonferroni_budget`` - splitting a failure budget across K events."""

    def test_equal_split_is_the_default(self):
        budget = cc.bonferroni_budget(0.95, 4)
        self.assertEqual(budget.weights, (0.25, 0.25, 0.25, 0.25))
        self.assertAlmostEqual(budget.epsilon, 0.05)
        self.assertAlmostEqual(budget.epsilon_at(1), 0.0125)

    def test_impact_target_level_is_tightened(self):
        budget = cc.bonferroni_budget(0.95, 4)
        # 1 - w0 * eps, strictly above lambda: the target is one of the events.
        self.assertAlmostEqual(budget.lambda_impact, 0.9875)
        self.assertGreater(budget.lambda_impact, budget.lambda_level)

    def test_single_event_reproduces_the_individual_level(self):
        budget = cc.bonferroni_budget(0.95, 1)
        self.assertAlmostEqual(budget.epsilon_at(0), 0.05)
        self.assertAlmostEqual(budget.lambda_impact, 0.95)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError) as context:
            cc.bonferroni_budget(0.95, 4, weights=(0.25, 0.25, 0.25, 0.30))
        self.assertIn("sum to 1", str(context.exception))

    def test_weights_must_match_K(self):
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.95, 4, weights=(0.5, 0.5))

    def test_weights_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.95, 3, weights=(1.5, -0.25, -0.25))

    def test_lambda_must_be_a_probability(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                cc.bonferroni_budget(bad, 4)

    def test_unequal_weights_are_allowed(self):
        budget = cc.bonferroni_budget(0.95, 4, weights=(0.7, 0.1, 0.1, 0.1))
        self.assertAlmostEqual(budget.epsilon_at(0), 0.035)
        self.assertAlmostEqual(budget.lambda_impact, 0.965)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestDeclaredQuantile(unittest.TestCase):
    """``declared_quantile`` - the exact inverse CDF of each declared family."""

    TRIANGULAR = {"uncertainty_type": 5, "minimum": 0.1, "loc": 1.0,
                  "maximum": 1.1}

    def test_triangular_matches_the_closed_form(self):
        for eps in (0.001, 0.0125, 0.05, 0.125):
            expected = 0.1 + np.sqrt(eps * (1.1 - 0.1) * (1.0 - 0.1))
            self.assertAlmostEqual(
                cc.declared_quantile(self.TRIANGULAR, eps), expected, places=12)

    def test_triangular_upper_branch(self):
        # Above (b-a)/(c-a) = 0.9 the other branch applies.
        eps = 0.95
        expected = 1.1 - np.sqrt((1 - eps) * (1.1 - 0.1) * (1.1 - 1.0))
        self.assertAlmostEqual(
            cc.declared_quantile(self.TRIANGULAR, eps), expected, places=12)

    def test_triangular_never_leaves_its_support(self):
        # The property a moment-matched normal lacks: at extreme reliability the
        # fitted normal demands a negative availability, the exact one cannot
        # fall below the support floor.
        self.assertAlmostEqual(cc.declared_quantile(self.TRIANGULAR, 0.0), 0.1)
        for eps in (1e-12, 1e-9, 1e-6):
            self.assertGreaterEqual(
                cc.declared_quantile(self.TRIANGULAR, eps), 0.1)

    def test_normal_agrees_with_the_individual_formulation(self):
        spec = {"uncertainty_type": 3, "loc": 5.0, "scale": 2.0}
        for lam in (0.5, 0.9, 0.99):
            individual = spec["loc"] - spec["scale"] * scipy.stats.norm.ppf(lam)
            self.assertAlmostEqual(cc.declared_quantile(spec, 1.0 - lam),
                                   individual, places=12)

    def test_uniform_is_linear(self):
        spec = {"uncertainty_type": 4, "minimum": 2.0, "maximum": 6.0}
        self.assertAlmostEqual(cc.declared_quantile(spec, 0.25), 3.0)

    def test_lognormal_median(self):
        spec = {"uncertainty_type": 2, "loc": 0.0, "scale": 1.0}
        self.assertAlmostEqual(cc.declared_quantile(spec, 0.5), 1.0)

    def test_unsupported_family_raises(self):
        with self.assertRaises(NotImplementedError):
            cc.declared_quantile({"uncertainty_type": 0, "amount": 1.0}, 0.5)

    def test_probability_must_be_a_probability(self):
        with self.assertRaises(ValueError):
            cc.declared_quantile(self.TRIANGULAR, 1.5)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestClosedFormMomentsCarrySource(unittest.TestCase):
    """The moments keep the declared spec, so an exact quantile stays reachable."""

    def test_source_is_preserved(self):
        declared = {1: {"uncertainty_type": 5, "minimum": 0.1, "loc": 1.0,
                        "maximum": 1.1, "amount": 1.0}}
        moments = processor._closed_form_moments(declared)
        self.assertEqual(moments[1]["uncertainty_type"],
                         stats_arrays.NormalUncertainty.id)
        self.assertAlmostEqual(moments[1]["loc"], (0.1 + 1.0 + 1.1) / 3)
        self.assertEqual(moments[1]["source"], declared[1])

    def test_source_is_a_copy_not_a_reference(self):
        declared = {1: {"uncertainty_type": 5, "minimum": 0.1, "loc": 1.0,
                        "maximum": 1.1}}
        moments = processor._closed_form_moments(declared)
        declared[1]["minimum"] = 999.0
        self.assertAlmostEqual(moments[1]["source"]["minimum"], 0.1)


@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestApplyCCFormulationGuards(unittest.TestCase):
    """The argument checks that keep a budget and a model from disagreeing."""

    def test_exact_requires_a_budget(self):
        with self.assertRaises(ValueError) as context:
            cc.apply_CC_formulation(None, 0.95, {}, {}, bound_quantile="exact")
        self.assertIn("needs a risk_budget", str(context.exception))

    def test_unknown_quantile_rejected(self):
        with self.assertRaises(ValueError):
            cc.apply_CC_formulation(None, 0.95, {}, {}, bound_quantile="student")

    def test_budget_lambda_must_match(self):
        budget = cc.bonferroni_budget(0.90, 4)
        with self.assertRaises(ValueError) as context:
            cc.apply_CC_formulation(None, 0.95, {}, {}, risk_budget=budget)
        self.assertIn("was built for lambda", str(context.exception))

    def test_K_must_match_the_rows_actually_imposed(self):
        # Three bound rows plus the impact target is K = 4, not K = 9.
        bounds = {"upper_limit": {i: {"loc": 1.0, "scale": 0.1} for i in range(3)}}
        budget = cc.bonferroni_budget(0.95, 9)
        with self.assertRaises(ValueError) as context:
            cc.apply_CC_formulation(None, 0.95, {}, bounds, risk_budget=budget)
        self.assertIn("covers K=9", str(context.exception))

    def test_bound_positions_are_sorted_not_insertion_ordered(self):
        shuffled = {"upper_limit": {7: {}, 2: {}, 5: {}}}
        positions = cc._bound_positions(shuffled)
        self.assertEqual(positions,
                         {("upper_limit", 2): 1, ("upper_limit", 5): 2,
                          ("upper_limit", 7): 3})


##################################################
#### GSA with deterministic characterization  ####
##################################################

@unittest.skipUnless(UNCERTAINTY_DEPS, UNCERTAINTY_SKIP_REASON)
class TestGSADeterministicCharacterization(unittest.TestCase):
    """A flow whose factor is deterministic still reaches the decomposition.

    Reindexing the sampled factors onto the sampled flows hands an unsampled
    factor an all-NaN column, and one NaN column makes every sample impact NaN.
    The factor is not missing, it is constant, so it is filled with its amount.
    """

    def _analyser(self, cf_amounts):
        analyser = object.__new__(gsa.GlobalSensitivityAnalysis)
        analyser.method = "m"
        analyser.lci_data = {
            "matrices": {"m": scipy.sparse.diags(cf_amounts, format="csr")}}
        return analyser

    def test_deterministic_factor_is_filled_not_dropped(self):
        analyser = self._analyser([7.5, 11.0, 0.0])
        # Two sampled flows on one process; only flow 0 has a sampled factor.
        sample_if = pd.DataFrame(
            [[2.0, 3.0], [4.0, 5.0]],
            columns=pd.MultiIndex.from_tuples([(0, 10), (1, 10)]))
        sample_cf = pd.DataFrame([[7.0], [8.0]], columns=[0])
        env_cost, _ = analyser._compute_env_cost(sample_if, sample_cf)
        self.assertFalse(np.isnan(env_cost.to_numpy()).any(),
                         "an unsampled factor still poisons every impact")
        # Flow 0 uses its sampled factor, flow 1 its deterministic amount of 11.
        np.testing.assert_allclose(env_cost.to_numpy(),
                                   [[7.0 * 2.0, 11.0 * 3.0],
                                    [8.0 * 4.0, 11.0 * 5.0]])

    def test_all_factors_sampled_is_unchanged(self):
        analyser = self._analyser([7.5, 11.0, 0.0])
        sample_if = pd.DataFrame(
            [[2.0, 3.0]], columns=pd.MultiIndex.from_tuples([(0, 10), (1, 10)]))
        sample_cf = pd.DataFrame([[7.0, 9.0]], columns=[0, 1])
        env_cost, _ = analyser._compute_env_cost(sample_if, sample_cf)
        np.testing.assert_allclose(env_cost.to_numpy(), [[14.0, 27.0]])
