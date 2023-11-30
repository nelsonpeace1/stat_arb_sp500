import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

import pandas as pd

from main.model_building.scripts.creating_spreads import (
    _process_row_both_spread,
)

from main.utilities.paths import (
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

TICKER_TO_TEST_WITH = "XRXOQ"


def mocking_create_db_engine_to_return_none(args):
    return None


def mocking_save_spread_series_to_database_to_return_none(*args):
    return None


def test_process_row_both_spread(mocker):

    testing_prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)
    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = (
        testing_results_df[testing_results_df["first_ticker"] == TICKER_TO_TEST_WITH]
        .iloc[0]
        .squeeze()
    )

    mocker.patch(
        "main.model_building.scripts.creating_spreads._save_spread_series_to_database",
        side_effect=mocking_save_spread_series_to_database_to_return_none,
    )

    mocker.patch(
        "main.model_building.scripts.creating_spreads.custom_create_db_engine",
        side_effect=mocking_create_db_engine_to_return_none,
    )

    testing_obj = _process_row_both_spread(row, testing_prices_df)
    testing_obj_standardised = testing_obj[1]

    assert 0.9 < testing_obj_standardised.std() < 1.1
    assert -0.01 < testing_obj_standardised.mean() < 0.01
    assert len(testing_obj_standardised) > 20
