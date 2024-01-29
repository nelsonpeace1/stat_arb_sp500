import pandas as pd

from main.model_building.scripts.cointegration_testing import (
    perform_multiple_cointegration_tests,
)

from main.utilities.paths import (
    PATHWAY_TO_TESTING_RESULTS_DF,
    PATHWAY_TO_TESTING_PRICES_DF,
)


testing_results_df = pd.read_parquet(PATHWAY_TO_TESTING_RESULTS_DF)
dynamic_testing_results_df_structure = pd.DataFrame(
    columns=[
        "engle_test_training",
        "pair_start_date",
        "trading_period_mid_point_date",
        "pair_finish_date",
        "length_of_trading_period_days_calendar",
    ]
)


def test_perform_multiple_cointegration_tests():

    res = perform_multiple_cointegration_tests(
        results_df=dynamic_testing_results_df_structure,
        prices_df=pd.read_parquet(PATHWAY_TO_TESTING_PRICES_DF),
    )

    assert round(res["engle_test_training"].values[0], 2) == 0.02
    assert res["first_ticker"].values[0] == "ABT"
