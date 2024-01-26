

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS,
    PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST,
)

from main.utilities.constants import (
    CORES_TO_USE,
    LENGTH_OF_ROLLING_HEDGE_RATIO,
)

from main.utilities.functions import (
    custom_create_db_engine,
)

DATABASE_NAME_ROLLING_HEDGE = f"sqlite:///{PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS}"
DATABASE_NAME_ROLLING_HEDGE_BACKTEST = (
    f"sqlite:///{PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST}"
)

ADDITIONAL_DAYS_TO_MAKE_ROLLING_WINDOW = 150
TIME_WINDOW_TO_LOOK_BACK_TRAINING_PERIOD_MAKE_ROLLING_OLS_WORK = 1000


def _calculate_single_rolling_hedge_ratio_ols(
    ticker1: str,
    ticker2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prices_df: pd.DataFrame,
) -> pd.Series:

    ticker1_series_training = pd.Series(
        prices_df[ticker1].loc[
            start_date
            - timedelta(
                days=TIME_WINDOW_TO_LOOK_BACK_TRAINING_PERIOD_MAKE_ROLLING_OLS_WORK
            ) : end_date
        ],
        dtype=float,
    )
    ticker2_series_training = pd.Series(
        prices_df[ticker2].loc[
            start_date
            - timedelta(
                days=TIME_WINDOW_TO_LOOK_BACK_TRAINING_PERIOD_MAKE_ROLLING_OLS_WORK
            ) : end_date
        ],
        dtype=float,
    )

    if (
        len(ticker1_series_training) < LENGTH_OF_ROLLING_HEDGE_RATIO
    ):  # This will only return None when the training period does not have enough instances to support inclusion, and by virtue of us making this decision based on the training period, no lookahead bias will be committed.
        return None

    ols_model = RollingOLS(
        ticker1_series_training,
        ticker2_series_training,
        window=LENGTH_OF_ROLLING_HEDGE_RATIO,
    )
    ols_model = ols_model.fit()
    hedge_ratio_series = ols_model.params
    hedge_ratio_series = hedge_ratio_series.loc[start_date:]
    hedge_ratio_series.bfill(
        inplace=True
    )  # In rare instances, this will cause look ahead bias by filling in regression values based on future information

    return hedge_ratio_series


def _save_to_database(hedge_ratio_series, table_name, engine):
    hedge_ratio_series.index = hedge_ratio_series.index.astype("datetime64[ns]")
    hedge_ratio_series.name = f"{table_name}"
    hedge_ratio_series.to_sql(table_name, engine, if_exists="replace", index=True)
    logging.info(f"saved {table_name} in sql db")


def _process_row_rolling_hedge_ratio(
    row: pd.Series,
    prices_df: pd.DataFrame,
    backtest_spread: bool = False,
) -> None:

    if backtest_spread:
        engine = custom_create_db_engine(DATABASE_NAME_ROLLING_HEDGE_BACKTEST)
        start_date = row["trading_period_mid_point_date"]
        end_date = row["pair_finish_date"]
    else:
        engine = custom_create_db_engine(DATABASE_NAME_ROLLING_HEDGE)
        start_date = row["pair_start_date"]
        end_date = row["trading_period_mid_point_date"]

    if (
        (end_date - start_date).days < LENGTH_OF_ROLLING_HEDGE_RATIO
    ) and backtest_spread:
        try:
            start_date = end_date - BDay(
                LENGTH_OF_ROLLING_HEDGE_RATIO + ADDITIONAL_DAYS_TO_MAKE_ROLLING_WINDOW
            )  # This stops look ahead bias, note 1 below.
        except:
            return None

    elif (
        (end_date - start_date).days < LENGTH_OF_ROLLING_HEDGE_RATIO
    ) and not backtest_spread:
        return None
        # Trade has less than 500 days history, does not qualify. No look forward bias committed as during testing phase

    hedge_ratio_series = _calculate_single_rolling_hedge_ratio_ols(
        row["first_ticker"],
        row["second_ticker"],
        start_date,
        end_date,
        prices_df,
    )

    if hedge_ratio_series is None:
        return None

    table_name = f'{row["first_ticker"]}_{row["second_ticker"]}'
    _save_to_database(hedge_ratio_series.squeeze(), table_name, engine)
    logging.info(f"creating table for {table_name}")
    return hedge_ratio_series


def calculate_rolling_hedge_ratio_whole_set(
    results_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    backtest_spread: bool = False,
):

    Parallel(n_jobs=CORES_TO_USE)(
        delayed(_process_row_rolling_hedge_ratio)(row, prices_df, backtest_spread)
        for _, row in results_df.iterrows()
    )


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)

    calculate_rolling_hedge_ratio_whole_set(
        results_df=results_df,
        prices_df=prices_df,
        backtest_spread=False,
    )

    calculate_rolling_hedge_ratio_whole_set(
        results_df=results_df,
        prices_df=prices_df,
        backtest_spread=True,
    )

    logging.info("Calculated rolling hedge ratios for dataset")


# Note 1: checking if the series is < 500 days and in instances where its less we simply look back the full 500 + 35 days, and if a series is not that long (i.e., even going into the testing period to accommodate it) that this is possible it errors out. This averts survivorship bias as we are selecting using the testing period
