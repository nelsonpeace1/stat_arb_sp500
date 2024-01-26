

import sqlite3
import pandas as pd
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS,
    PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST,
    PATHWAY_TO_SQL_DB_SPREADS,
    PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
)

from main.utilities.constants import (
    LENGTH_OF_ROLLING_HEDGE_RATIO,
    CORES_TO_USE,
)

from main.utilities.functions import (
    custom_create_db_engine,
)

DATABASE_NAME_SPREAD = f"sqlite:///{PATHWAY_TO_SQL_DB_SPREADS}"
DATABASE_NAME_SPREAD_BACKTEST = f"sqlite:///{PATHWAY_TO_SQL_DB_SPREADS_BACKTEST}"


def _retrieve_table_from_sql_rolling_hedge_ratio_df(
    ticker1: str,
    ticker2: str,
    db_pathway: str,
    kalman: bool = False,
) -> pd.Series:

    hedge_ratio_rolling_table_name = f"{ticker1}_{ticker2}{'_kalman' if kalman else ''}"
    conn = sqlite3.connect(db_pathway)
    query = f"SELECT * FROM {hedge_ratio_rolling_table_name}"
    hedge_ratio_rolling_series = pd.read_sql_query(
        query,
        conn,
        index_col="Date",
        parse_dates=["Date"],
    )
    conn.close()
    return hedge_ratio_rolling_series


def _save_spread_series_to_database(
    table_name: str,
    spread_series: pd.Series,
    engine: sqlite3.Connection,
) -> None:
    spread_series.name = table_name
    spread_series.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=True,
    )


def _create_both_spread_single(
    ticker1: str,
    ticker2: str,
    pair_start_date: pd.Timestamp,
    pair_end_date: pd.Timestamp,
    prices_df: pd.DataFrame,
    db_pathway: str,
    kalman: bool = False,
) -> pd.Series:

    hedge_ratio_rolling_series = _retrieve_table_from_sql_rolling_hedge_ratio_df(
        ticker1=ticker1,
        ticker2=ticker2,
        db_pathway=db_pathway,
        kalman=kalman,
    )

    ticker1_series_training = prices_df[ticker1].loc[pair_start_date:pair_end_date]
    ticker2_series_training = prices_df[ticker2].loc[pair_start_date:pair_end_date]

    if len(hedge_ratio_rolling_series) != len(ticker1_series_training):
        hedge_ratio_rolling_series = hedge_ratio_rolling_series[
            -len(ticker1_series_training) :
        ]

    ticker2_series_training_copy = ticker2_series_training.copy()

    ticker2_series_training_copy *= hedge_ratio_rolling_series.squeeze().values

    spread_series_from_rolling = ticker1_series_training - ticker2_series_training_copy

    spread_series_z_standardised_from_rolling = (
        spread_series_from_rolling - spread_series_from_rolling.mean()
    ) / spread_series_from_rolling.std()

    return (
        spread_series_from_rolling.squeeze(),
        spread_series_z_standardised_from_rolling.squeeze(),
    )


def _process_row_both_spread(
    row: pd.Series,
    prices_df: pd.DataFrame,
    backtest_spread: bool = False,
    kalman: bool = False,
) -> None:

    if backtest_spread:
        engine = custom_create_db_engine(DATABASE_NAME_SPREAD_BACKTEST)
        start_date = row["trading_period_mid_point_date"]
        end_date = row["pair_finish_date"]
        db_pathway = PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST
    else:
        engine = custom_create_db_engine(DATABASE_NAME_SPREAD)
        start_date = row["pair_start_date"]
        end_date = row["trading_period_mid_point_date"]
        db_pathway = PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS

    if (
        (end_date - start_date).days < LENGTH_OF_ROLLING_HEDGE_RATIO
    ) and not backtest_spread:
        return None

    (
        spread_series_from_rolling,
        spread_series_z_standardised_from_rolling,
    ) = _create_both_spread_single(
        row["first_ticker"],
        row["second_ticker"],
        start_date,
        end_date,
        prices_df,
        db_pathway,
        kalman=kalman,
    )

    table_name_regular = f"{row['first_ticker']}_{row['second_ticker']}_regular_spread{'_kalman' if kalman else ''}"
    table_name_standardised = f"{row['first_ticker']}_{row['second_ticker']}_standardised_spread{'_kalman' if kalman else ''}"

    _save_spread_series_to_database(
        table_name_regular,
        spread_series_from_rolling,
        engine,
    )
    _save_spread_series_to_database(
        table_name_standardised,
        spread_series_z_standardised_from_rolling,
        engine,
    )

    return spread_series_from_rolling, spread_series_z_standardised_from_rolling


def create_rolling_hedge_ratio_scaled_spread_whole_set(
    results_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    backtest_spread: bool = False,
    kalman: bool = False,
) -> None:

    Parallel(n_jobs=CORES_TO_USE)(
        delayed(_process_row_both_spread)(
            row,
            prices_df,
            backtest_spread,
            kalman,
        )
        for _, row in results_df.iterrows()
    )


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF).reset_index()
    prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)

    create_rolling_hedge_ratio_scaled_spread_whole_set(
        results_df,
        prices_df,
        backtest_spread=False,
    )

    create_rolling_hedge_ratio_scaled_spread_whole_set(
        results_df,
        prices_df,
        backtest_spread=True,
    )

    create_rolling_hedge_ratio_scaled_spread_whole_set(
        results_df,
        prices_df,
        backtest_spread=True,
        kalman=True,
    )

    logging.info("Done creating both spreads for all securities")
