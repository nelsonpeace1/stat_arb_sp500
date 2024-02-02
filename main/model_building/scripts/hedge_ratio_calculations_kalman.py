from joblib import Parallel, delayed
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST,
)

from main.utilities.constants import (
    CORES_TO_USE,
)

from main.utilities.functions import (
    custom_create_db_engine,
)

from main.model_building.scripts.hedge_ratio_calculations import (
    _save_to_database,
)

DATABASE_NAME_ROLLING_HEDGE_BACKTEST = (
    f"sqlite:///{PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST}"
)
ADDITIONAL_DAYS_TO_MAKE_ROLLING_WINDOW = 35
FIRST_ELEMENT_PRICE_SERIES = 0


def _calculate_single_rolling_hedge_ratio_kalman(
    ticker1: str,
    ticker2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prices_df: pd.DataFrame,
    process_noise: float = 0.0001,
    measurement_noise: float = 1.99,
    error_cov: float = 1.0,
) -> pd.Series:

    ticker1_series_training = prices_df[ticker1].loc[start_date:end_date]
    ticker2_series_training = prices_df[ticker2].loc[start_date:end_date]

    hedge_ratio = (
        ticker1_series_training.iloc[FIRST_ELEMENT_PRICE_SERIES]
        / ticker2_series_training.iloc[FIRST_ELEMENT_PRICE_SERIES]
    )

    rolling_hedge_ratio = []

    observations = ticker1_series_training / ticker2_series_training

    for observation in observations:

        error_cov += process_noise
        kalman_gain = error_cov / (error_cov + measurement_noise)
        hedge_ratio += kalman_gain * (observation - hedge_ratio)
        error_cov *= 1 - kalman_gain
        rolling_hedge_ratio.append(hedge_ratio)

    assert (ticker1_series_training.index == ticker2_series_training.index).all()
    assert (
        len(rolling_hedge_ratio)
        == (len(ticker1_series_training) + len(ticker2_series_training)) / 2
    )
    return pd.Series(rolling_hedge_ratio, index=ticker1_series_training.index)


def _process_row_rolling_hedge_ratio_kalman(
    row: pd.Series,
    prices_df: pd.DataFrame,
) -> None:

    engine = custom_create_db_engine(DATABASE_NAME_ROLLING_HEDGE_BACKTEST)
    start_date = row["trading_period_mid_point_date"]
    end_date = row["pair_finish_date"]

    hedge_ratio_series_kalman = _calculate_single_rolling_hedge_ratio_kalman(
        ticker1=row["first_ticker"],
        ticker2=row["second_ticker"],
        start_date=start_date,
        end_date=end_date,
        prices_df=prices_df,
    )

    table_name = f'{row["first_ticker"]}_{row["second_ticker"]}_kalman'
    _save_to_database(
        hedge_ratio_series_kalman,
        table_name,
        engine,
    )
    logging.info(f"creating table for {table_name}")
    return hedge_ratio_series_kalman


def calculate_rolling_hedge_ratio_whole_set_kalman(
    results_df: pd.DataFrame,
    prices_df: pd.DataFrame,
):

    Parallel(n_jobs=CORES_TO_USE)(
        delayed(_process_row_rolling_hedge_ratio_kalman)(
            row,
            prices_df,
        )
        for _, row in results_df.iterrows()
    )


if __name__ == "__main__":

    prices_df = pd.read_parquet(PATHWAY_TO_PRICE_DF)
    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    calculate_rolling_hedge_ratio_whole_set_kalman(
        prices_df=prices_df,
        results_df=results_df,
    )

    logging.info("Kalman filter hedge ratios complete")
