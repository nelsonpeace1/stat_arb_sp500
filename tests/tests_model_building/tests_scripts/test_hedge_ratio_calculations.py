

import numpy as np
import pandas as pd

from main.model_building.scripts.hedge_ratio_calculations import (
    _process_row_rolling_hedge_ratio,
)

from main.utilities.paths import (
    PATHWAY_TO_TESTING_PRICES_DF,
    PATHWAY_TO_TESTING_RESULTS_DF,
)

LENGTH_OF_EXPECTED_SERIES = 1337


def mocking_function_replace_save_to_database_instead_return_object(
    hedge_ratio_series,
    table_name,
    engine,
):
    return None


def mocking_function_create_db_engine(
    args,
):
    return None


def test_calculate_rolling_hedge_ratio_whole_set(
    mocker,
):

    testing_prices_df = pd.read_parquet(PATHWAY_TO_TESTING_PRICES_DF)
    testing_results_df = pd.read_parquet(PATHWAY_TO_TESTING_RESULTS_DF)
    row = testing_results_df.iloc[0]

    mocker.patch(
        "main.model_building.scripts.hedge_ratio_calculations._save_to_database",
        side_effect=mocking_function_replace_save_to_database_instead_return_object,
    )

    mocker.patch(
        "main.model_building.scripts.hedge_ratio_calculations.custom_create_db_engine",
        side_effect=mocking_function_create_db_engine,
    )

    testing_obj = _process_row_rolling_hedge_ratio(
        row,
        testing_prices_df,
    )

    assert len(testing_obj) == LENGTH_OF_EXPECTED_SERIES
    assert testing_obj.name == "ACN"
    assert testing_obj.dtype == np.dtype("float64")
