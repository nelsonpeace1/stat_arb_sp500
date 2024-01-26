

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.utilities.constants import (
    CORES_TO_USE,
)

from main.utilities.functions import (
    retrieve_spread_table_from_sql_df,
)

import logging

logging.basicConfig(level=logging.INFO)

FIRST_OLS_PARAM = 1
LOG_ARG_HL = 2


def perform_half_life_ornstein_single(
    row: pd.Series,
) -> float:

    spread_series_from_rolling = retrieve_spread_table_from_sql_df(
        row,
    )

    spread_series_from_rolling_lagged = spread_series_from_rolling.shift(1).dropna()
    delta_spread = (
        spread_series_from_rolling - spread_series_from_rolling_lagged
    ).dropna()

    results = sm.OLS(
        delta_spread, sm.add_constant(spread_series_from_rolling_lagged)
    ).fit()
    half_life = -np.log(LOG_ARG_HL) / results.params.iloc[FIRST_OLS_PARAM]

    return half_life


def half_life_ornstein_whole_set(
    results_df: pd.DataFrame,
) -> list[float]:

    half_life_results = Parallel(n_jobs=CORES_TO_USE)(
        delayed(perform_half_life_ornstein_single)(row)
        for _, row in results_df.iterrows()
    )

    return half_life_results


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    half_life_results = half_life_ornstein_whole_set(results_df)
    results_df["half_life_results"] = half_life_results
    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    logging.info("half life ornstein process for whole set complete")
