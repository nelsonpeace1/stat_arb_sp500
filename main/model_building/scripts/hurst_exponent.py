import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)
from main.utilities.constants import (
    CORES_TO_USE,
)

from main.utilities.functions import (
    retrieve_spread_table_from_sql_df,
)

MAX_LAGS_FOR_HURST_EXPONENT = 120
HURST_EXP_PARAM = 0
DEGREE_OF_POLYFIT = 1
SECOND_LAG_HE_RANGE = 2


def perform_hurst_exponent_single(
    row: pd.Series,
    max_lag: int = MAX_LAGS_FOR_HURST_EXPONENT,
) -> float:

    spread_series_from_rolling = retrieve_spread_table_from_sql_df(
        row,
    )

    lags = range(SECOND_LAG_HE_RANGE, max_lag)
    tau = [
        np.std(
            np.subtract(
                spread_series_from_rolling[lag:].values,
                spread_series_from_rolling[:-lag].values,
            )
        )
        for lag in lags
    ]
    hurst = np.polyfit(np.log(lags), np.log(tau), DEGREE_OF_POLYFIT)[HURST_EXP_PARAM]

    return hurst


def hurst_exponent_whole_set(
    results_df: pd.DataFrame,
) -> None:

    hurst_exponent_results = Parallel(n_jobs=CORES_TO_USE)(
        delayed(perform_hurst_exponent_single)(row) for _, row in results_df.iterrows()
    )

    return hurst_exponent_results


if __name__ == "__main__":
    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    hurst_exponent_results = hurst_exponent_whole_set(results_df)
    results_df["hurst_exponent_results"] = hurst_exponent_results
    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    logging.info("completed hurst exponent for whole set")
