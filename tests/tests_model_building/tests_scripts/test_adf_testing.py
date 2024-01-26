

import pandas as pd

from main.model_building.scripts.adf_testing import perform_adf_whole_set

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

FIRST_LIST_ELEMENT_ADF_TESTING = 0
TICKER_TO_TEST_WITH = "XRXOQ"


def test_perform_adf_whole_set():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[testing_results_df["first_ticker"] == TICKER_TO_TEST_WITH]

    testing_obj = perform_adf_whole_set(row)[FIRST_LIST_ELEMENT_ADF_TESTING]
    assert round(testing_obj, 3) == 0.003
