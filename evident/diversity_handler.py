from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import product
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from statsmodels.stats.power import tt_ind_solve_power, FTestAnovaPower

from . import exceptions as exc
from .stats import (calculate_cohens_d, calculate_cohens_f,
                    calculate_pooled_stdev)
from .utils import listify, _check_sample_overlap


@dataclass
class PowerAnalysisResults:
    alpha: float
    total_observations: int
    power: float
    effect_size: float


class BaseDiversityHandler(ABC):
    """Abstract class for handling diversity data and metadata."""
    def __init__(self, data=None, metadata: pd.DataFrame = None):
        self.data = data
        self.metadata = metadata

    @property
    def samples(self):
        """Get represented samples."""
        return self.metadata.index.to_list()

    @lru_cache()
    def calculate_effect_size(
        self,
        column: str,
        difference: float = None
    ) -> float:
        """Get effect size of diversity differences given column.

        If two categories, return Cohen's d from t-test. If more than two
        categories, return Cohen's f from ANOVA.

        :param column: Column containing categories
        :type column: str

        # When would you need to pass in the difference directly?
        # I could understand wanting to override the standard deviation, since pooling age and bmi makes no sense
        # but I can't understand wanting to override the calculated difference between groups in the dataset.
        :param difference: If provided, used as the numerator in effect size
            calculation rather than the difference in means, defaults to None
        :type difference: float

        :returns: Effect size
        :rtype: float
        """
        if self.metadata[column].dtype != np.dtype("object"):
            raise exc.NonCategoricalColumnError(self.metadata[column])

        column_choices = self.metadata[column].unique()
        num_choices = len(column_choices)

        # Maybe <= 1, in case they pass empty metadata
        if num_choices == 1:
            raise exc.OnlyOneCategoryError(self.metadata[column])
        elif num_choices == 2:
            effect_size_func = calculate_cohens_d
        else: # and >= 3
            effect_size_func = calculate_cohens_f

        # Create list of arrays for effect size calculation
        arrays = []
        for choice in column_choices:
            ids = self.metadata[self.metadata[column] == choice].index
            values = self.subset_values(ids)
            arrays.append(values)

        if difference is None:
            return effect_size_func(*arrays)
        else:
            pooled_stdev = calculate_pooled_stdev(*arrays)
            return difference / pooled_stdev

    def power_analysis(
        self,
        column: str,
        total_observations: int = None,
        difference: float = None,
        alpha: float = None,
        power: float = None
    ):
        """Perform power analysis using this diversity dataset.

        # Enforcement of the Exactly one of blah must be None is kind of annoying.
        # Some other options to try out:
            # 3 separate functions - power_analysis_alpha(), power_analysis_difference(), power_analysis_total_observations()
            # A wrapper class with three constructors: PowerAnalysisInput.from_obs(total_observations), PowerAnalysisInput.from_alpha(alpha) ...
        Exactly one of total_observations, alpha, or power must be None.

        Arguments can be either single values or sequences of values. If a
            sequence of values is passed for any parameter, power calculations
            will be done on each possible set of argument combinations in the
            Cartesian product set.

        :param column: Name of column in metadata to consider
        :type column: str

        :param difference: Difference between groups to consider, defaults to
            None. If provided, uses the pooled standard deviation as the
            denominator to calculate the effect size with the difference as the
            numerator. Can be either float or sequence of floats.
        :type difference: float or np.array[float]

        :param alpha: Significant level to use in power calculation, defaults
            to None. Can be either float or sequence of floats.
        :type alpha: float or np.array[float]

        :param power: Power level to use in power calculation, defaults to
            None. Can be either float or sequence of floats.
        :type power: float or np.array[float]
        """
        args = [alpha, power, total_observations]
        none_args = [x is None for x in args]
        if sum(none_args) != 1:  # Check to make sure exactly one arg is None
            raise exc.WrongPowerArguments(*args)

        # Pretty minor, but changing the format of the return value based on the input arguments
        # can sometimes be annoying.  If you find that you're rewriting code that calls
        # into this function when you want to switch between trying one thing and trying two things,
        # you can rewrite it so that this function always returns the bulk result and always takes bulk input
        # You can then, if needed, write a separate wrapper function (power_analysis_single) that calls the bulk version
        # and unwraps so you still have both api's, but each now has a specific return type.
        #
        # Since calculate_effect_size is your critical section for optimization, and it is only called from the _single_power_analysis,
        # wrapping it to explicitly expose the bulk version may prevent the end user from wrapping your code in a for loop that
        # loops in a bad order for your LRU cache.

        # If any of the arguments are iterable, perform power analysis on
        #     all possible argument combinations. Otherwise, perform a single
        #     power analysis to solve for the non-provided argument.
        vector_args = map(lambda x: isinstance(x, Iterable), args)
        if any(vector_args):
            power_analysis_func = self._bulk_power_analysis
        else:
            power_analysis_func = self._single_power_analysis

        result = power_analysis_func(
            column=column,
            total_observations=total_observations,
            difference=difference,
            alpha=alpha,
            power=power
        )
        return result

    def _single_power_analysis(
        self,
        column: str,
        total_observations: int = None,
        difference: float = None,
        alpha: float = None,
        power: float = None
    ) -> float:
        """Compute the power analysis for a single value.

        :param column: Name of column in metadata to consider
        :type column: str

        :param difference: Difference between groups to consider, defaults to
            None. If provided, uses the pooled standard deviation as the
            denominator to calculate the effect size with the difference as the
            numerator.
        :type difference: float

        :param alpha: Significant level to use in power calculation, defaults
            to None.
        :type alpha: float

        :param power: Power level to use in power calculation, defaults to
            None.
        :type power: float

        :returns: Collection of values from power analysis
        :rtype: PowerAnalysisResults
        """
        power_func = self._incept_power_solve_function(
            column=column,
            difference=difference,
            total_observations=total_observations
        )

        val_to_solve = power_func(power=power, alpha=alpha)

        # If calculating total_observations, check to see if doing t-test
        # If so, multiply by two as tt_ind_solve_power returns number of
        #     observations of sample 1.
        if total_observations is None:
            power_func_name = power_func.func.__qualname__
            if power_func_name == "TTestIndPower.solve_power":
                val_to_solve = np.ceil(val_to_solve) * 2

        args = [alpha, power, total_observations]
        none_args = [x is None for x in args]
        idx = none_args.index(True)

        if idx == 0:
            alpha = val_to_solve
        elif idx == 1:
            power = val_to_solve
        else:
            total_observations = val_to_solve

        results = PowerAnalysisResults(
            alpha=alpha,
            total_observations=total_observations,
            power=power,
            effect_size=power_func.keywords["effect_size"]
        )
        return results

    def _bulk_power_analysis(
        self,
        column: str,
        total_observations: int = None,
        difference: float = None,
        alpha: float = None,
        power: float = None
    ):
        """Compute the power analysis for a multiple values.

        :param column: Name of column in metadata to consider
        :type column: str

        :param difference: Differences between groups to consider, defaults to
            None. If provided, uses the pooled standard deviation as the
            denominator to calculate the effect size with the difference as the
            numerator.
        :type difference: sequence of floats

        :param alpha: Significance levels to use in power calculation, defaults
            to None.
        :type alpha: sequence of floats

        :param power: Power levels to use in power calculation, defaults to
            None.
        :type power: sequence of floats

        :returns: Collection of values from power analyses
        :rtype: list[PowerAnalysisResults]
        """
        # Convert all to list so we can use Cartesian product
        difference = listify(difference)
        total_observations = listify(total_observations)
        alpha = listify(alpha)
        power = listify(power)
        power_args = [difference, total_observations, alpha, power]

        # My feeling is that there are certain assumptions needed to make the LRU cache in calculate_effect_size useful.
        # As it depends only on column and difference (and _incept_power_solve_function on column, difference and total observations)
        # it requires a specific ordering of bulk analyses to make use of the lru cache.
        # This could be done by sorting the power_arg_products tuple, when iterating over it
        # but the single for loop over 4 element tuples could also be broken apart.
        # I think this could be written as:
        # for _diff
        #   effect_size = calc_effect_size(column, _diff)
        #   for _obs
        #       power_solve_func = _incept_power_solve_function(col, _diff, _obs)
        #       for _alpha
        #           for _power
        #               do_whatever(effect_size, power_solve_func, ...)
        #
        # With such a scheme, you wouldn't need the LRU caches, the same object is reused without lookup.

        power_arg_products = product(*power_args)
        results_list = []
        for _diff, _obs, _alpha, _power in power_arg_products:
            results_list.append(self._single_power_analysis(
                column, _obs, _diff, _alpha, _power
            ))
        return results_list

    @abstractmethod
    def subset_values(self, ids: list):
        """Get subset of data given list of indices"""

    # Love the name, but also, hate the name, and have no idea what it does
    # Maybe tells me what to do in my dreams?
    @lru_cache()
    def _incept_power_solve_function(
        self,
        column: str,
        difference: float = None,
        total_observations: int = None
    ) -> Callable:
        """Create basic function to solve for power.

        Observations arg calculated in _incept_power_solve_function and is
            included in power_func. Need to determine whether to use
            t-test or ANOVA as that determines argument to be used.

        Memoized to avoid duplicated computation in the case of multiple
            power analyses.

        :param column: Name of column in metadata to consider
        :type column: str

        :param difference: Difference between groups to consider, defaults to
            None. If provided, uses the pooled standard deviation as the
            denominator to calculate the effect size with the difference as the
            numerator.
        :type difference: float

        :param total_observations: Total number of observations for power
            calculation, defaults to None
        :type total_observations: int

        :returns: Stem of power function based on chosen column
        :rtype: partial function
        """
        if self.metadata[column].dtype != np.dtype("object"):
            raise exc.NonCategoricalColumnError(self.metadata[column])

        column_choices = self.metadata[column].unique()
        num_choices = len(column_choices)

        if num_choices == 1:
            raise exc.OnlyOneCategoryError(self.metadata[column])
        elif num_choices == 2:
            # tt_ind_solve_power uses observations per group
            if total_observations is not None:
                total_observations = total_observations / 2

            power_func = partial(
                tt_ind_solve_power,
                nobs1=total_observations,
                ratio=1.0,
            )
        else:
            # FTestAnovaPower uses *total* observations
            power_func = partial(
                FTestAnovaPower().solve_power,
                k_groups=num_choices,
                nobs=total_observations,
            )

        effect_size = self.calculate_effect_size(
            column,
            difference=difference
        )
        return partial(power_func, effect_size=effect_size)


class AlphaDiversityHandler(BaseDiversityHandler):
    """Handler for alpha diversity data."""
    def __init__(
        self,
        data: pd.Series,
        metadata: pd.DataFrame
    ):
        md_samps = set(metadata.index)
        data_samps = set(data.index)
        samps_in_common = _check_sample_overlap(md_samps, data_samps)

        super().__init__(
            data=data.loc[samps_in_common],
            metadata=metadata.loc[samps_in_common]
        )

    def subset_values(self, ids: list):
        return self.data.loc[ids]


class BetaDiversityHandler(BaseDiversityHandler):
    """Handler for beta diversity data."""
    def __init__(
        self,
        data: DistanceMatrix,
        metadata: pd.DataFrame
    ):
        md_samps = set(metadata.index)
        data_samps = set(data.ids)
        samps_in_common = _check_sample_overlap(md_samps, data_samps)

        super().__init__(
            data=data.filter(samps_in_common),
            metadata=metadata.loc[samps_in_common]
        )

    def subset_values(self, ids: list):
        return np.array(self.data.filter(ids).to_series().values)
