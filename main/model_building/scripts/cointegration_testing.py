

from itertools import combinations
import pandas as pd
from datetime import datetime, timedelta, date
from statsmodels.tsa.stattools import coint as coint_engle
import os
import logging

logging.basicConfig(level=logging.INFO)
from joblib import Parallel, delayed

from main.utilities.paths import (
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.utilities.constants import (
    TRADING_DATE_MID_POINT,
    ENGLE_COINT_P_VALUE_THRESHOLD,
    CORES_TO_USE,
)

from main.utilities.constants import (
    MIN_LENGTH_SERIES_FOR_TESTING,
)

LEVEL_TO_REJECT_TEST_DUE_SERIES_ONLY_NAN = {"engle_test_training": None}
ELEMENT_OF_ENGLE_TEST_RETURNING_PVALUE = 1
DIVISOR_OF_TRADING_PERIOD_LENGTH = 2
ROUNDING_OF_TRANSFORMED_TRAINING_PERIOD = 0
NUMBER_TICKERS_TO_COMBINE = 2
FIRST_PRODUCT_ELEMENT = 0
SECOND_PRODUCT_ELEMENT = 1
FIRST_VAL_ENGLE_DICT = 0


def _cointegration_tests(
    ticker1: str,
    ticker2: str,
    df_prices: pd.DataFrame,
    trading_period_mid_point_date: date,
) -> dict:

    """
    Calculates relevant dates, performs cointegration test (engle) and returns test result p value, the pair start date, end date, mid date,
    """

    if df_prices[ticker1].isna().all() or df_prices[ticker2].isna().all():
        return LEVEL_TO_REJECT_TEST_DUE_SERIES_ONLY_NAN

    (
        pair_start_date,
        pair_finish_date,
        length_of_trading_period_days_calendar,
    ) = _calculate_relevant_trading_dates(ticker1, ticker2, df_prices)

    ticker1_series_training = df_prices[ticker1].loc[
        pair_start_date:trading_period_mid_point_date
    ]
    ticker2_series_training = df_prices[ticker2].loc[
        pair_start_date:trading_period_mid_point_date
    ]

    if (len(ticker1_series_training) < MIN_LENGTH_SERIES_FOR_TESTING) or (
        len(ticker2_series_training) < MIN_LENGTH_SERIES_FOR_TESTING
    ):
        return None

    engle_test_training = coint_engle(ticker1_series_training, ticker2_series_training)

    return {
        "engle_test_training": [
            engle_test_training[ELEMENT_OF_ENGLE_TEST_RETURNING_PVALUE]
        ],
        "pair_start_date": [pair_start_date],
        "trading_period_mid_point_date": [trading_period_mid_point_date],
        "pair_finish_date": [pair_finish_date],
        "length_of_trading_period_days_calendar": [
            length_of_trading_period_days_calendar
        ],
    }


def _calculate_relevant_trading_dates(
    ticker1: str,
    ticker2: str,
    df_prices: pd.DataFrame,
) -> tuple:

    """
    Calculates the dates where the securities were trading at the same time (lest the cointegration test attempt to run on NaN values), as well as the length of the whole trading period, and the midpoint, so the series can be split in two for training and testing.
    """

    pair_start_date = max(
        df_prices[ticker1].first_valid_index(), df_prices[ticker2].first_valid_index()
    )
    pair_finish_date = min(
        df_prices[ticker1].last_valid_index(), df_prices[ticker2].last_valid_index()
    )
    length_of_trading_period_days_calendar = (pair_finish_date - pair_start_date).days

    return (
        pair_start_date,
        pair_finish_date,
        length_of_trading_period_days_calendar,
    )


def _process_pair(
    product: tuple,
    prices_df: pd.DataFrame,
    trading_period_mid_point_date: pd.Timestamp,
) -> pd.DataFrame | None:

    results_dictionary = _cointegration_tests(
        product[FIRST_PRODUCT_ELEMENT],
        product[SECOND_PRODUCT_ELEMENT],
        prices_df,
        trading_period_mid_point_date,
    )

    if (
        (results_dictionary is None)
        or (results_dictionary["engle_test_training"] is None)
        or (
            results_dictionary["engle_test_training"][FIRST_VAL_ENGLE_DICT]
            > ENGLE_COINT_P_VALUE_THRESHOLD
        )
    ):
        return None

    ticker_dictionary = {
        "first_ticker": product[FIRST_PRODUCT_ELEMENT],
        "second_ticker": product[SECOND_PRODUCT_ELEMENT],
    }
    final_dictionary = {**ticker_dictionary, **results_dictionary}
    df_temporary = pd.DataFrame(final_dictionary)

    return df_temporary


def perform_multiple_cointegration_tests(
    results_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    trading_period_mid_point_date: date = TRADING_DATE_MID_POINT,
) -> pd.DataFrame:

    ticker_products = list(
        combinations(
            prices_df.columns,
            NUMBER_TICKERS_TO_COMBINE,
        )
    )

    list_of_temp_dfs = Parallel(n_jobs=CORES_TO_USE)(
        delayed(_process_pair)(
            product,
            prices_df,
            trading_period_mid_point_date,
        )
        for product in ticker_products
    )

    list_of_temp_dfs = [df for df in list_of_temp_dfs if df is not None]

    results_df = pd.concat(list_of_temp_dfs).reset_index(drop=True)
    return results_df.reset_index(drop=True)


if __name__ == "__main__":

    prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)

    results_df = pd.DataFrame(
        columns=[
            "engle_test_training",
            "pair_start_date",
            "trading_period_mid_point_date",
            "pair_finish_date",
            "length_of_trading_period_days_calendar",
        ]
    )

    results = perform_multiple_cointegration_tests(results_df, prices_df)
    results.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    logging.info("Finished sp_500 cointegration tests")
