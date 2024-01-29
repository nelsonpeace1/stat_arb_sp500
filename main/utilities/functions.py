from main.utilities.paths import (
    PATHWAY_TO_SQL_DB_SPREADS,
    PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
    PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS,
)

from main.utilities.constants import (
    FIRST_BACKTEST_PARAMETERS,
)

import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

RANDOM_WALK_STARTING_DATE = ("1678-01-01",)
RANDOM_WALK_ENDING_DATE = ("2261-12-31",)
RANDOM_WALK_STARTING_STOCK_VALUE = 100


def retrieve_spread_table_from_sql_df(
    row: pd.Series,
    pathway: str = PATHWAY_TO_SQL_DB_SPREADS,
    spread_type: str = "_regular_spread",
) -> pd.DataFrame:

    table_name = f"{row['first_ticker']}_{row['second_ticker']}{spread_type}"
    conn = sqlite3.connect(pathway)
    query = f"SELECT * FROM {table_name}"
    spread_series = pd.read_sql_query(
        query, conn, index_col="Date", parse_dates=["Date"]
    )
    conn.close()

    return spread_series


def retrieve_backtest_equity_curve_spread_table_from_sql_df(
    row: pd.Series,
    pathway: str = PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
    backtest_params: str = FIRST_BACKTEST_PARAMETERS,
    kalman: bool = False,
) -> pd.Series:

    table_name = f"{row['first_ticker']}_{row['second_ticker']}{backtest_params}{'_kalman' if kalman else ''}"
    conn = sqlite3.connect(pathway)
    query = f"SELECT * FROM {table_name}"

    spread_series = pd.read_sql_query(
        query,
        conn,
        index_col="Date",
        parse_dates=["Date"],
    )
    conn.close()

    return spread_series["valuation"]


def custom_create_db_engine(
    pathway: str,
) -> pd.DataFrame:
    
    """This function is encapsulated like this so it can be mocked in tests"""


    return create_engine(
        pathway,
    )


def get_table_as_dataframe(
    db_path: str,
    table_name: str,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df


def get_first_entry_of_all_tables(
    db_path: str,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        tables = [table[0] for table in tables]

        first_entries = []
        for table in tables:
            entry = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
            first_entries.append(entry.iloc[0, 0] if not entry.empty else None)

    result_df = pd.DataFrame({"Table": tables, "First_Entry": first_entries})
    return result_df


def generate_series_for_backtest_testing(
    *date_value_pairs: tuple[str, float],
) -> dict:

    dates, values = zip(*sorted(date_value_pairs))
    all_dates = pd.date_range(start=dates[0], end=dates[-1])
    spread_series = pd.Series(index=all_dates, dtype=float)
    for date, value in zip(dates, values):
        spread_series[date] = value
    spread_series.interpolate(method="linear", inplace=True)
    price_series2 = pd.Series(100, index=all_dates)
    price_series1 = price_series2 + spread_series

    return {
        "ticker1_prices": price_series1,
        "ticker2_prices": price_series2,
        "standardised_spread": spread_series,
    }


def generate_random_walk_series_for_random_backtest(
    start_date=RANDOM_WALK_STARTING_DATE,
    end_date=RANDOM_WALK_ENDING_DATE,
    start_value1=RANDOM_WALK_STARTING_STOCK_VALUE,
    start_value2=RANDOM_WALK_STARTING_STOCK_VALUE,
    mean=0,
    std_dev=0.01,
) -> dict:

    dates = pd.date_range(start=start_date, end=end_date)
    steps1 = np.random.normal(mean, std_dev, len(dates))
    steps2 = np.random.normal(mean, std_dev, len(dates))
    steps1 = np.where((np.cumsum(steps1) + start_value1) < 0, -steps1, steps1)
    steps2 = np.where((np.cumsum(steps2) + start_value2) < 0, -steps2, steps2)
    price_series1 = pd.Series(np.cumsum(steps1) + start_value1, index=dates)
    price_series2 = pd.Series(np.cumsum(steps2) + start_value2, index=dates)

    spread_series = price_series1 - price_series2

    return {
        "ticker1_prices": price_series1,
        "ticker2_prices": price_series2,
        "standardised_spread": spread_series,
    }


def plot_series_with_lines(
    spread,
    level=None,
    figsize=(10, 6),
) -> None:

    plt.figure(figsize=figsize)
    spread.plot()
    plt.axhline(y=0.5, color="green", linestyle="--", alpha=0.5)
    plt.axhline(y=-0.5, color="green", linestyle="--", alpha=0.5)
    plt.axhline(y=6, color="red", linestyle="-", linewidth=2)
    plt.axhline(y=-6, color="red", linestyle="-", linewidth=2)
    if level is not None:
        plt.axhline(y=level, color="pink", linestyle="--", alpha=0.5)
        plt.axhline(y=-level, color="pink", linestyle="--", alpha=0.5)
    plt.show()


def get_db_number_tables_and_some_dimensions(
    db_path: str,
) -> tuple[int, dict]:

    """Another sanity check for sql databases after build. This test will inform the user how many tables are in the database, and retrieve the names and dimensions of 3 random tables"""

    with sqlite3.connect(db_path) as conn:
        tables_df = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        num_tables = len(tables_df)

        if num_tables == 0:
            return 0, None, None, None

        table_info = {}
        sample_tables = tables_df.sample(min(100, num_tables))
        for table_name in sample_tables["name"]:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            table_info[table_name] = df.shape

    return num_tables, table_info


def get_table_from_backtest_results_dfs(
    table_name: str,
    db_path: str = PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS,
    backtest_params: str | None = "_2_05_6",
    kalman: bool = False,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        backtest_results_df = pd.read_sql(
            f"SELECT * FROM {table_name}{backtest_params}{'_kalman' if kalman else ''}",
            conn,
        )
    return backtest_results_df
