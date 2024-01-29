import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from datetime import datetime


from main.utilities.constants import (
    CORES_TO_USE,
)

from main.utilities.functions import (
    retrieve_backtest_equity_curve_spread_table_from_sql_df,
)


def _equity_table_adder(
    row: pd.Series,
    kalman: bool = False,
) -> tuple:
    spread_table = retrieve_backtest_equity_curve_spread_table_from_sql_df(
        row=row,
        kalman=kalman,
    )

    spread_table.columns = [""]
    return (
        spread_table,
        spread_table.first_valid_index(),
        spread_table.last_valid_index(),
    )


def _concat_modify_sum_results(
    results: list[list[pd.Series], datetime, datetime],
) -> pd.Series:

    table_list, min_dates, max_dates = zip(*results)
    result = pd.concat(table_list, axis=1, join="outer")
    result = result.fillna(method="bfill", axis=0)
    result = result.fillna(method="ffill", axis=0)
    summed_result = result.sum(axis=1)

    return summed_result, min_dates, max_dates


def create_eq_curve(
    results_df: pd.DataFrame,
    first_ticker_sector: str | None = None,
    second_ticker_sector: str | None = None,
    combined_tickers_sectors: str | None = None,
    kalman: bool = False,
) -> pd.Series:

    if first_ticker_sector is not None:
        results_df = results_df[
            results_df["first_ticker_sector"] == first_ticker_sector
        ]
    if second_ticker_sector is not None:
        results_df = results_df[
            results_df["second_ticker_sector"] == second_ticker_sector
        ]
    if combined_tickers_sectors is not None:
        results_df = results_df[
            results_df["tickers_sectors_concat"] == combined_tickers_sectors
        ]

    results = Parallel(n_jobs=CORES_TO_USE)(
        delayed(_equity_table_adder)(
            row=row,
            kalman=kalman,
        )
        for _, row in results_df.iterrows()
    )

    summed_result, min_dates, max_dates = _concat_modify_sum_results(
        results=results,
    )

    assert summed_result.first_valid_index() == min(min_dates)
    assert summed_result.last_valid_index() == max(max_dates)
    assert not summed_result.isna().any()

    return summed_result
