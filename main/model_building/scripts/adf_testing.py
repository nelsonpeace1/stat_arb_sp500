import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)
from main.utilities.constants import (
    ENGLE_COINT_P_VALUE_THRESHOLD,
    CORES_TO_USE,
)

from main.utilities.functions import (
    retrieve_spread_table_from_sql_df,
)

ADF_TEST_RESULT_P_VAL_ELEMENT_NO = 1


def perform_adf_single(
    row: pd.Series,
) -> tuple:

    if row["engle_test_training"] >= ENGLE_COINT_P_VALUE_THRESHOLD:
        return np.nan

    spread_series_from_rolling = retrieve_spread_table_from_sql_df(
        row, spread_type="_regular_spread"
    )

    return adfuller(spread_series_from_rolling)[ADF_TEST_RESULT_P_VAL_ELEMENT_NO]


def perform_adf_whole_set(
    results_df: pd.DataFrame,
) -> list:

    adf_results_list = Parallel(n_jobs=CORES_TO_USE)(
        delayed(perform_adf_single)(row) for _, row in results_df.iterrows()
    )

    assert len(adf_results_list) == len(results_df)
    logging.info("adf tests successful and same length")
    return adf_results_list


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    adf_results_list = perform_adf_whole_set(results_df=results_df)
    results_df["adf_result"] = adf_results_list
    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    logging.info("adf tests complete for whole set")
