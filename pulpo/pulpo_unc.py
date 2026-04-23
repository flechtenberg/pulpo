"""
pulpo_unc.py

Uncertainty-enabled façade for PULPO. Users opt into the uncertainty stack by
writing::

    from pulpo import pulpo_unc
    worker = pulpo_unc.PulpoOptimizerUnc(project, db, method, directory)

Importing this module triggers a guarded import of SALib / seaborn / joblib /
matplotlib / stats_arrays (plus the internal uncertainty sub-package). If any
of them is missing, a clear install hint is raised.

The base package (``from pulpo import pulpo``) continues to work without any
of these extras.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
_MISSING_DEPS_MSG = (
    "The uncertainty features of PULPO require additional packages that are\n"
    "not part of the base install. Install them with:\n\n"
    "    pip install \"pulpo-dev[uncertainty]\"\n\n"
    "(adds: SALib, seaborn, matplotlib, stats_arrays, typing_extensions)\n"
    "Underlying ImportError: {err}"
)

try:
    import stats_arrays  # noqa: F401
    import matplotlib  # noqa: F401
    import seaborn  # noqa: F401
    import SALib  # noqa: F401

    # Internal uncertainty sub-package (imports the above transitively)
    from pulpo.utils.uncertainty import preparer, processor, gsa, plots, cc
    from pulpo.utils.uncertainty import monte_carlo as mc_unc
    from pulpo.utils.uncertainty.preparer import UncertaintySpec
except ImportError as _err:  # pragma: no cover
    raise ImportError(_MISSING_DEPS_MSG.format(err=_err)) from _err

# ---------------------------------------------------------------------------
# Standard imports (safe – already required by base pulpo)
# ---------------------------------------------------------------------------
import array
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from pulpo.pulpo import PulpoOptimizer
from pulpo.utils import optimizer
from pulpo.utils.saver import ResultDataDict


# ---------------------------------------------------------------------------
# Subclass – adds uncertainty-driven methods on top of the core optimizer
# ---------------------------------------------------------------------------
class PulpoOptimizerUnc(PulpoOptimizer):
    """PulpoOptimizer + uncertainty/GSA/CC/MC-from-uncertainty workflows."""

    # --- MC from prepared uncertainty distributions ------------------------
    def run_mc_from_uncertainty(
        self,
        n_samples: int,
        seed: int | None = None,
        n_jobs: int = -1,
        GAMS_PATH=False,
        solver_name: str | None = None,
        options=None,
    ):
        """
        Monte-Carlo optimization driven by prepared uncertainty distributions
        (no Brightway resample).
        """
        if self.uncertainty_data is None:
            raise Exception(
                "No uncertainty data found. Run import_and_filter_uncertainty_data + "
                "apply_uncertainty_strategies first."
            )
        overlays = mc_unc.pre_sample_from_uncertainty(
            pulpo_optimizer=self, n_samples=n_samples, seed=seed
        )
        return mc_unc.solve_model_MC_pre_sampled_uncertainty(
            pulpo_optimizer=self,
            overlays=overlays,
            GAMS_PATH=GAMS_PATH,
            solver_name=solver_name,
            options=options,
            n_jobs=n_jobs,
        )

    # --- Chance-constrained optimization -----------------------------------
    def solve_CC_problem(
        self,
        lambda_level: float | List,
        normal_metadata_env_cost: Dict[Tuple[int, str], UncertaintySpec],
        normal_metadata_var_bounds: Dict[str, Dict[int, UncertaintySpec]],
        gams_path=False,
        solver_name: Optional[str] = None,
        options=None,
        neos_email=None,
        cutoff_value: float = 0.01,
        plot_results: bool = False,
        bbox_to_anchor: tuple = (0.65, -3.5),
        cmap_name: str = 'tab20',
    ) -> Dict[float, ResultDataDict]:
        """Solve one or several Pareto points at the specified lambda level(s)."""
        results: Dict[float, ResultDataDict] = {}
        if isinstance(lambda_level, float):
            cc.apply_CC_formulation(
                self.instance, lambda_level,
                normal_metadata_env_cost, normal_metadata_var_bounds,
            )
            self.solve(GAMS_PATH=gams_path, solver_name=solver_name,
                       options=options, neos_email=neos_email)
            results[lambda_level] = self.extract_results()
        elif isinstance(lambda_level, (np.ndarray, list, array.array)):
            for lambda_ in lambda_level:
                print(f'solving CC problem for lambda_QB = {lambda_}')
                cc.apply_CC_formulation(
                    self.instance, lambda_,
                    normal_metadata_env_cost, normal_metadata_var_bounds,
                )
                self.solve(GAMS_PATH=gams_path, solver_name=solver_name,
                           options=options, neos_email=neos_email)
                results[lambda_] = self.extract_results(extractparams=True)
        else:
            raise Exception('lambda_level datatype not implemented, needs to be an array or a float.')
        if plot_results:
            if self.lci_data is None:
                raise Exception('No LCI data found. Please run get_lci_data method first.')
            plots.plot_pareto_front(
                result_data_CC=results,
                cutoff_value=cutoff_value,
                method="\n".join(next(iter(self.method)).split("'")[1::2]),
                process_map_metadata=self.lci_data['process_map_metadata'],
                bbox_to_anchor=bbox_to_anchor,
                cmap_name=cmap_name,
            )
        return results

    # --- Uncertainty data import / filter ----------------------------------
    def import_and_filter_uncertainty_data(
        self,
        cutoff: float = 0,
        scaling_vector_strategy: Literal['naive', 'constructed_demand'] = 'naive',
        result_data: dict = {},
        plot_results: bool = False,
        plot_n_top_processes: int = 10,
    ):
        """Import uncertainty from the underlying databases and apply a cutoff filter."""
        if len(self.method) > 1:
            raise Exception(
                'The uncertainty data import currently only works with a single LCIA method.'
            )
        if self.lci_data is None:
            raise Exception('No LCI data found. Please run get_lci_data method first.')
        method = next(iter(self.method))
        paramfilter = preparer.ParameterFilter(
            lci_data=self.lci_data,
            choices=self.choices,
            demand=self.demand,
            method=method,
        )
        filtered_inventory_indcs, filtered_characterization_indcs = paramfilter.apply_filter(
            scaling_vector_strategy=scaling_vector_strategy,
            cutoff=cutoff,
            plot_results=plot_results,
            plot_n_top_processes=plot_n_top_processes,
            result_data=result_data,
        )
        uncertainty_importer = preparer.UncertaintyImporter(
            lci_data=self.lci_data,
            bw_databases=self.database,
            LCIA_method=method,
        )
        self.uncertainty_data = uncertainty_importer.import_uncertainty_data(
            if_indcs=filtered_inventory_indcs,
            cf_indcs=filtered_characterization_indcs,
            choices=self.choices,
            upper_limit=self.upper_limit,
            lower_limit=self.lower_limit,
            upper_elem_limit=self.upper_elem_limit,
            upper_imp_limit=self.upper_imp_limit,
        )

    # --- Strategy application ---------------------------------------------
    def apply_uncertainty_strategies(
        self,
        strategies: List['processor.UncertaintyStrategyBase'] = [],
        drop_undefined: bool = False,
        scaling_factor_if: float = 0.5,
        scaling_factor_cf: float = 0.3,
        scaling_factor_var_bounds: float = 0.2,
        **strategy_options,
    ):
        """Apply the uncertainty gap-filling / updating strategies."""
        if self.uncertainty_data is None:
            raise Exception('No uncertainty data found. Please run import_and_filter_uncertainty_data method first.')
        if strategies is None or len(strategies) == 0:
            print('Applying default uncertainty strategies.')
            strategies = processor.uncertainty_strategy_base_case(
                databases=self.database if isinstance(self.database, list) else [self.database],
                method=next(iter(self.method)),
                uncertainty_data=self.uncertainty_data,
                scaling_factor_if=scaling_factor_if,
                scaling_factor_cf=scaling_factor_cf,
                scaling_factor_var_bounds=scaling_factor_var_bounds,
            )
        processor.apply_uncertainty_strategies(self.uncertainty_data, strategies, **strategy_options)
        processor.check_missing_uncertainty_data(self.uncertainty_data)
        if drop_undefined:
            self.uncertainty_data = processor.drop_undefined_uncertainty_data(self.uncertainty_data)

    # --- Global sensitivity analysis --------------------------------------
    def run_gsa(
        self,
        result_data: dict,
        sample_method,
        SA_method,
        sample_size: int,
        plot_gsa_results: bool = False,
        top_sensitivity_amt: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run a global sensitivity analysis (SALib)."""
        if self.uncertainty_data is None:
            raise Exception('No uncertainty data found. Please run import_and_filter_uncertainty_data method first.')
        if processor.check_missing_uncertainty_data(self.uncertainty_data, unc_types=['If', 'Cf']):
            raise Exception('The uncertainty data contains undefined uncertainty types.')
        gsa_study = gsa.GlobalSensitivityAnalysis(
            result_data=result_data,
            lci_data=self.lci_data,
            uncertainty_data=self.uncertainty_data,
            sampler=sample_method,
            analyser=SA_method,
            sample_size=sample_size,
            method=next(iter(self.method)),
            plot_gsa_results=plot_gsa_results,
            top_sensitivity_amt=top_sensitivity_amt,
        )
        return gsa_study.perform_gsa()

    # --- CC formulation preparation ---------------------------------------
    def create_CC_formulation(
        self,
        CC_env_cost: bool = True,
        CC_var_bounds: List[Literal['upper_imp_limit', 'lower_limit', 'upper_elem_limit', 'upper_limit']] = [],
        plot_analysis_support_plots: bool = False,
        normal_transformation_sample_size: int = 100,
    ) -> tuple[Dict[Tuple[int, str], UncertaintySpec], Dict[str, Dict[int, UncertaintySpec]]]:
        """Prepare data (mean / std) needed to run the CC formulation."""
        if CC_env_cost is True and len(CC_var_bounds) > 0:
            unc_types = ['If', 'Cf', 'Var_bounds']
        elif CC_env_cost is True and len(CC_var_bounds) == 0:
            unc_types = ['If', 'Cf']
        elif CC_env_cost is False and len(CC_var_bounds) > 0:
            unc_types = ['Var_bounds']
        else:
            raise Exception(
                'No CC formulation specified. Set at least one of CC_env_cost / CC_var_bounds.'
            )
        if self.uncertainty_data is None or processor.check_missing_uncertainty_data(
                self.uncertainty_data, unc_types=unc_types):
            raise Exception(
                'None or incomplete uncertainty data found. Run import_and_filter_uncertainty_data '
                'and apply_uncertainty_strategies first.'
            )
        normal_uncertainty_data = processor.transform_to_normal(
            self.uncertainty_data,
            sample_size=normal_transformation_sample_size,
            plot_distribution=plot_analysis_support_plots,
            unc_types=unc_types,
        )
        if CC_env_cost:
            normal_metadata_env_cost = cc.compute_L1_env_cost_mean_var(
                normal_uncertainty_data=normal_uncertainty_data,
                lci_data=self.lci_data,
                method=next(iter(self.method)),
                plot_analysis_support_plots=plot_analysis_support_plots,
            )
        else:
            normal_metadata_env_cost = {}
        if CC_var_bounds:
            normal_metadata_var_bounds = {
                var_bound: normal_uncertainty_data['Var_bounds'][var_bound]['defined']
                for var_bound in CC_var_bounds
            }
        else:
            normal_metadata_var_bounds = {}
        return normal_metadata_env_cost, normal_metadata_var_bounds
