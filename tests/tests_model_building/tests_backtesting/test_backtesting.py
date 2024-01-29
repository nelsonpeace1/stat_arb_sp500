import pandas as pd

from main.utilities.functions import (
    generate_series_for_backtest_testing,
)

from main.utilities.paths import PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF

from main.model_building.backtesting.backtest import (
    BackTest,
)

TICKER_1_TO_TEST_WITH = "SYKN"
TICKER_2_TO_TEST_WITH = "AFLN"


def test_backtesting_one():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ].squeeze()

    test_inputs = generate_series_for_backtest_testing(
        ("2020-01-01", 0),
        ("2020-01-31", 4),
        ("2021-01-31", -4),
        ("2022-01-31", 7),
    )

    backtest_obj_one = BackTest(
        row,
        test_inputs=test_inputs,
    )
    backtest_obj_one.trade()
    assert backtest_obj_one.trade_history_frame.shape == (3, 13)
    assert backtest_obj_one.trade_history_frame.iloc[-1, -1] == True


def test_backtesting_two():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ].squeeze()

    test_inputs = generate_series_for_backtest_testing(
        ("2020-01-01", 0),
        ("2020-01-31", 3),
        ("2021-01-30", -5),
        ("2021-01-31", 7),
        ("2021-03-30", 4.5),
        ("2021-03-31", -5),
        ("2021-08-31", 10),
    )

    backtest_obj_two = BackTest(row, test_inputs=test_inputs)
    backtest_obj_two.trade()
    assert backtest_obj_two.trade_history_frame.shape == (2, 13)
    assert round(backtest_obj_two.trade_history_frame.iloc[-1, -2]) == 105210


def test_backtesting_three():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ].squeeze()

    test_inputs = generate_series_for_backtest_testing(
        ("2020-01-01", 0),
        ("2020-01-31", -3),
        ("2021-01-30", 5),
        ("2021-01-31", -5),
        ("2021-03-30", -4.5),
        ("2021-03-31", 5),
        ("2021-08-31", -10),
    )

    backtest_obj_three = BackTest(row, test_inputs=test_inputs)
    backtest_obj_three.trade()
    assert backtest_obj_three.trade_history_frame.shape == (2, 13)
    assert round(backtest_obj_three.trade_history_frame.iloc[-1, -2]) == 104077
