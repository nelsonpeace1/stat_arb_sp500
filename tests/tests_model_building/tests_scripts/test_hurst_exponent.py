

import pandas as pd

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)
from main.model_building.scripts.hurst_exponent import (
    hurst_exponent_whole_set,
)

TICKER_1_TO_TEST_WITH = "XRXOQ"
TICKER_2_TO_TEST_WITH = "PBIN"


def test_hurst_exponent_whole_set():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ]

    testing_obj = hurst_exponent_whole_set(row)
    assert round(testing_obj[0], 4) == 0.3995
