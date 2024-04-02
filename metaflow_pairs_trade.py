import logging
import pandas as pd
from joblib import Parallel, delayed

from metaflow import (
    FlowSpec,
    step,
    Parameter,
)

from main.utilities.paths import (
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_SECTORS_SUBSECTORS_DF,
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.utilities.constants import (
    CORES_TO_USE,
)


from main.model_building.scripts.cointegration_testing import (
    perform_multiple_cointegration_tests,
)
from main.model_building.scripts.hedge_ratio_calculations import (
    calculate_rolling_hedge_ratio_whole_set,
)
from main.model_building.scripts.hedge_ratio_calculations_kalman import (
    calculate_rolling_hedge_ratio_whole_set_kalman,
)
from main.model_building.scripts.creating_spreads import (
    create_rolling_hedge_ratio_scaled_spread_whole_set,
)
from main.model_building.scripts.adf_testing import perform_adf_whole_set
from main.model_building.scripts.hurst_exponent import hurst_exponent_whole_set
from main.model_building.scripts.half_life import half_life_ornstein_whole_set
from main.model_building.scripts.modify_results_df import (
    concatenate_sectors_in_column,
    sector_mapper,
)
from main.model_building.backtesting.backtest_execution import execute_trade
from main.model_building.backtesting_analysis.performance_measures import (
    calculate_various_performance_metrics_whole_set,
)


logging.basicConfig(level=logging.INFO)


class StatArbFlow(FlowSpec):

    """Metaflow pipeline to run all the steps for a pairs trading analysis"""

    spread_to_trigger_trade_entry = Parameter(
        name="spread_to_trigger_trade_entry",
        default=2,
        help="Spread to trigger trade entry",
    )

    spread_to_trigger_trade_exit = Parameter(
        name="spread_to_trigger_trade_exit",
        default=0.5,
        help="Spread to trigger trade exit",
    )

    spread_to_abandon_trade = Parameter(
        name="spread_to_abandon_trade",
        default=6,
        help="Spread to abandon trade, a kind of stop loss",
    )

    testing = Parameter(
        name="testing",
        default=True,
        help="Testing parameter to run the pipeline with a smaller dataset",
    )

    testing_df_length = Parameter(
        name="testing_df_length",
        default=30,
        help="Length of the testing dataset",
    )

    @step
    def start(self):

        logging.info("Starting the pairs trading analysis and defining datasets")

        self.prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)
        self.sectors_subsectors_df = pd.read_parquet(PATHWAY_TO_SECTORS_SUBSECTORS_DF)
        self.present_backtest_params = (
            str(self.spread_to_trigger_trade_entry)
            + "_"
            + str(self.spread_to_trigger_trade_exit)
            + "_"
            + str(self.spread_to_abandon_trade)
        ).replace(".", "")

        if self.testing:
            self.prices_df = self.prices_df.iloc[:, : self.testing_df_length]

        logging.info("Finished defining datasets")

        self.next(self.cointegration_testing)

    @step
    def cointegration_testing(self):

        logging.info("Running cointegration testing")

        self.results_df = perform_multiple_cointegration_tests(
            prices_df=self.prices_df,
        )

        logging.info("Finished sp_500 cointegration tests")

        self.next(self.hedge_ratio_calculations_ols)

    @step
    def hedge_ratio_calculations_ols(self):

        logging.info("Calculating rolling hedge ratios OLS for dataset")

        calculate_rolling_hedge_ratio_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=False,
        )

        calculate_rolling_hedge_ratio_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=True,
        )

        logging.info("Calculated rolling hedge ratios OLS for dataset")

        self.next(self.hedge_ratio_calculations_kalman)

    @step
    def hedge_ratio_calculations_kalman(self):

        logging.info("Calculating Kalman filter hedge ratios")

        calculate_rolling_hedge_ratio_whole_set_kalman(
            prices_df=self.prices_df,
            results_df=self.results_df,
        )

        logging.info("Kalman filter hedge ratios complete")

        self.next(self.creating_spreads)

    @step
    def creating_spreads(self):

        logging.info("Creating spreads")

        create_rolling_hedge_ratio_scaled_spread_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=False,
        )

        create_rolling_hedge_ratio_scaled_spread_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=True,
        )

        create_rolling_hedge_ratio_scaled_spread_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=False,
            kalman=True,
        )

        create_rolling_hedge_ratio_scaled_spread_whole_set(
            results_df=self.results_df,
            prices_df=self.prices_df,
            backtest_spread=True,
            kalman=True,
        )

        logging.info("Finished creating spreads")

        self.next(self.adf_testing)

    @step
    def adf_testing(self):

        logging.info("Running ADF tests")

        adf_results_list = perform_adf_whole_set(
            results_df=self.results_df,
        )
        self.results_df["adf_result"] = adf_results_list

        logging.info("ADF tests complete")

        self.next(self.calculate_hurst_exponent)

    @step
    def calculate_hurst_exponent(self):

        logging.info("Calculating Hurst Exponent")

        hurst_exponent_results = hurst_exponent_whole_set(
            results_df=self.results_df,
        )
        self.results_df["hurst_exponent_results"] = hurst_exponent_results

        logging.info("Hurst Exponent complete")

        self.next(self.calculate_half_life)

    @step
    def calculate_half_life(self):

        logging.info("Calculating half life")

        half_life_results = half_life_ornstein_whole_set(
            results_df=self.results_df,
        )
        self.results_df["half_life_results"] = half_life_results

        logging.info("Half life complete")

        self.next(self.sector_mapping)

    @step
    def sector_mapping(self):

        logging.info("Starting sector mapping")

        self.sector_df = pd.read_parquet(PATHWAY_TO_SECTORS_SUBSECTORS_DF)

        self.results_df = sector_mapper(
            sector_df=self.sector_df,
            results_df=self.results_df,
        )

        self.results_df = concatenate_sectors_in_column(
            results_df=self.results_df,
        )

        logging.info("Finished sector mapping")

        self.next(self.backtest_ols)

    @step
    def backtest_ols(self):

        logging.info("Starting backtesting ols")

        # First backtest with Kalman set to false
        Parallel(n_jobs=CORES_TO_USE)(
            delayed(execute_trade)(
                row=row,
                spread_to_trigger_trade_entry=self.spread_to_trigger_trade_entry,
                spread_to_trigger_trade_exit=self.spread_to_trigger_trade_exit,
                spread_to_abandon_trade=self.spread_to_abandon_trade,
                kalman_spread=False,
            )
            for _, row in self.results_df.iterrows()
        )

        logging.info("Backtesting complete ols")

        self.next(self.backtest_kalman)

    @step
    def backtest_kalman(self):

        logging.info("Starting backtesting kalman")

        # Second backtest with Kalman set to True
        Parallel(n_jobs=CORES_TO_USE)(
            delayed(execute_trade)(
                row=row,
                spread_to_trigger_trade_entry=self.spread_to_trigger_trade_entry,
                spread_to_trigger_trade_exit=self.spread_to_trigger_trade_exit,
                spread_to_abandon_trade=self.spread_to_abandon_trade,
                kalman_spread=True,
            )
            for _, row in self.results_df.iterrows()
        )

        logging.info("Backtesting complete kalman")

        self.next(self.calculate_performance_measures)

    @step
    def calculate_performance_measures(self):

        logging.info("Calculating performance measures")

        valuation_metrics = calculate_various_performance_metrics_whole_set(
            backtest_params=self.present_backtest_params,
            kalman=False,
        )

        self.results_df[
            [
                f"sharpe_ratio_{self.present_backtest_params}",
                f"no_profitable_trades_{self.present_backtest_params}",
                f"fraction_profitable_trades_{self.present_backtest_params}",
            ]
        ] = valuation_metrics

        logging.info("completed NON KALMAN performance measures")

        valuation_metrics_kalman = calculate_various_performance_metrics_whole_set(
            backtest_params=self.present_backtest_params,
            kalman=True,
        )

        self.results_df[
            [
                f"sharpe_ratio_{self.present_backtest_params}_kalman",
                f"no_profitable_trades_{self.present_backtest_params}_kalman",
                f"fraction_profitable_trades_{self.present_backtest_params}_kalman",
            ]
        ] = valuation_metrics_kalman

        logging.info("Performance measures complete")

        self.next(self.end)

    @step
    def end(self):

        self.results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

        logging.info(
            "Finished pairs trading analysis, saved results dataframe to parquet file"
        )


if __name__ == "__main__":
    StatArbFlow()
