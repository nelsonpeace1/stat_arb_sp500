import pandas as pd

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.utilities.config import (
    PRESENT_BACKTEST_PARAMS,
)

from main.model_building.backtesting_analysis.performance_measures import (
    _calculate_various_performance_metrics_single,
)


def test_calculate_various_performance_metrics():

    ticker_1_to_test_with = "XRXOQ"
    ticker_2_to_test_with = "PBIN"
    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    testing_row = testing_results_df[
        (testing_results_df["first_ticker"] == ticker_1_to_test_with)
        & (testing_results_df["second_ticker"] == ticker_2_to_test_with)
    ].squeeze()

    testing_object_calc_metrics = _calculate_various_performance_metrics_single(
        row=testing_row,
        backtest_params=PRESENT_BACKTEST_PARAMS,
    )

    assert all(item != 0 for item in testing_object_calc_metrics)
    assert round(testing_object_calc_metrics[0], 3) == 0.047
