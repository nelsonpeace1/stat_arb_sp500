

import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS,
    PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
)

from main.utilities.constants import (
    CORES_TO_USE,
    ANNUAL_RISK_FREE_RATE,
    CAPITAL_STARTING,
    NUMBER_DAYS_TRADING_YEAR,
    TRADE_STARTED_ABANDONED_STRING,
)

from main.utilities.functions import (
    retrieve_backtest_equity_curve_spread_table_from_sql_df,
    get_table_from_backtest_results_dfs,
)

INDEX_POSITION_STARTING_CAPITAL = 0


def _calculate_sharpe_ratio(
    valuation_series: pd.Series,
    risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> float:

    arith_return_series = valuation_series.diff().dropna()
    sharpe_ratio = (
        arith_return_series.mean() - (risk_free_rate / NUMBER_DAYS_TRADING_YEAR)
    ) / arith_return_series.std()

    if sharpe_ratio == -np.inf:
        return np.nan

    return sharpe_ratio


def _create_trade_pnl_list(
    row: pd.Series,
    backtest_params: str,
    kalman: bool = False,
) -> pd.Series:

    table_name = f"{row['first_ticker']}_{row['second_ticker']}"
    backtest_result_df = get_table_from_backtest_results_dfs(
        table_name=table_name,
        backtest_params=backtest_params,
        kalman=kalman,
    )

    if backtest_result_df.empty:
        return None

    elif backtest_result_df["closing_capital"].eq(TRADE_STARTED_ABANDONED_STRING).any():
        return TRADE_STARTED_ABANDONED_STRING

    series_with_start_cap = pd.Series(
        np.insert(
            backtest_result_df["closing_capital"],
            INDEX_POSITION_STARTING_CAPITAL,
            CAPITAL_STARTING,
        )
    )
    return series_with_start_cap.diff().dropna()


def _number_profitable_trades(
    trade_pnl_list: pd.Series,
) -> int:

    if trade_pnl_list is None:
        return 0
    elif (
        isinstance(trade_pnl_list, str)
        and trade_pnl_list == TRADE_STARTED_ABANDONED_STRING
    ):
        return np.nan

    return len([trade_pnl for trade_pnl in trade_pnl_list if trade_pnl > 0])


def _fraction_of_profitable_trades(
    trade_pnl_list: pd.Series,
) -> float:

    if trade_pnl_list is None:
        return 0.0
    elif (
        isinstance(trade_pnl_list, str)
        and trade_pnl_list == TRADE_STARTED_ABANDONED_STRING
    ):
        return np.nan

    try:
        fraction = len(
            [trade_pnl for trade_pnl in trade_pnl_list if trade_pnl > 0]
        ) / len(trade_pnl_list)
    except ZeroDivisionError:
        fraction = 0.0

    return fraction


def _calculate_various_performance_metrics_single(
    row: pd.Series,
    backtest_params: str,
    kalman: bool = False,
) -> list[float]:

    valuation_series = retrieve_backtest_equity_curve_spread_table_from_sql_df(
        row=row,
        pathway=PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
    )

    trade_pnl_list = _create_trade_pnl_list(
        row=row,
        backtest_params=backtest_params,
        kalman=kalman,
    )

    sharpe_ratio = _calculate_sharpe_ratio(
        valuation_series=valuation_series,
    )

    no_profitable_trades = _number_profitable_trades(
        trade_pnl_list=trade_pnl_list,
    )

    fraction_profitable_trades = _fraction_of_profitable_trades(
        trade_pnl_list=trade_pnl_list,
    )

    return [sharpe_ratio, no_profitable_trades, fraction_profitable_trades]


def calculate_various_performance_metrics_whole_set(
    backtest_params: str,
    kalman: bool = True,
) -> list:

    valuation_metrics = Parallel(n_jobs=CORES_TO_USE)(
        delayed(_calculate_various_performance_metrics_single)(
            row=row,
            backtest_params=backtest_params,
            kalman=kalman,
        )
        for _, row in results_df.iterrows()
    )

    return valuation_metrics


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    from main.utilities.config import (
        PRESENT_BACKTEST_PARAMS,
    )

    valuation_metrics_kalman = calculate_various_performance_metrics_whole_set(
        backtest_params=PRESENT_BACKTEST_PARAMS,
        kalman=False,
    )

    results_df[
        [
            f"sharpe_ratio_{PRESENT_BACKTEST_PARAMS}",
            f"no_profitable_trades_{PRESENT_BACKTEST_PARAMS}",
            f"fraction_profitable_trades_{PRESENT_BACKTEST_PARAMS}",
        ]
    ] = valuation_metrics_kalman

    logging.info("completed NON KALMAN performance metrics")

    valuation_metrics = calculate_various_performance_metrics_whole_set(
        backtest_params=PRESENT_BACKTEST_PARAMS,
        kalman=True,
    )

    results_df[
        [
            f"sharpe_ratio_{PRESENT_BACKTEST_PARAMS}_kalman",
            f"no_profitable_trades_{PRESENT_BACKTEST_PARAMS}_kalman",
            f"fraction_profitable_trades_{PRESENT_BACKTEST_PARAMS}_kalman",
        ]
    ] = valuation_metrics

    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    logging.info("completed ALL performance metrics")
