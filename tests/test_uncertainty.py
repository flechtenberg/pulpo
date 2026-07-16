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
from pulpo.datasets.rice_database import setup_rice_husk_db

try:
    from pulpo import pulpo_unc
    from pulpo.utils.uncertainty import processor
    from pulpo.utils.uncertainty.processor import TriangluarBaseStrategy
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
        # Force a clean rebuild of the rice example without any uncertainty.
        bd.projects.set_current("rice_husk_example")
        for db_name in ("rice_husk_example_db", "biosphere3"):
            if db_name in bd.databases:
                del bd.databases[db_name]
        setup_rice_husk_db()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lci_data = bw_parser.import_data(
                project="rice_husk_example",
                databases=["rice_husk_example_db"],
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
        # everything surviving the cutoff is 'defined'. The two stacks keep a
        # different number of intervention flows because the constructed-
        # demand scaling vectors differ between bw2 and bw25.
        expected_if = ({"background_db": (5, 0), "foreground_db": (3, 0)}
                       if is_bw25() else
                       {"background_db": (4, 0), "foreground_db": (1, 0)})
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
        # The draws differ between the stacks because the matrix row/column
        # order (and therefore the parameter <-> seed pairing) differs.
        if is_bw25():
            self.assertAlmostEqual(self.samples.mean(), 1.806788, places=5)
            self.assertAlmostEqual(self.samples.std(), 0.258532, places=5)
        else:
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
        expected = ({0.75: 1.928769, 0.90: 2.080282, 0.95: 2.170956}
                    if is_bw25() else
                    {0.75: 1.884391, 0.90: 1.995962, 0.95: 2.062733})
        for lam, value in expected.items():
            self.assertAlmostEqual(self.impacts[lam], value, places=5)

    def test_formulation_requires_a_target(self):
        with self.assertRaises(Exception) as context:
            self.worker.create_CC_formulation(CC_env_cost=False, CC_var_bounds=[])
        self.assertIn("No CC formulation specified", str(context.exception))


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
        # All filtered parameters take part: intervention flows + the two CFs.
        expected_params = 10 if is_bw25() else 7
        self.assertEqual(len(self.total_order), expected_params)
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
