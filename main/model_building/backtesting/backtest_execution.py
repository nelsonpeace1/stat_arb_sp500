

from joblib import Parallel, delayed
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.constants import (
    CORES_TO_USE,
)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.model_building.backtesting.backtest import (
    BackTest,
)
https://github.com/nelsonpeace1/stat_arb_sp500.git

def execute_trade(
    row,
    spread_to_trigger_trade_entry: int | float,
    spread_to_trigger_trade_exit: int | float,
    spread_to_abandon_trade: int | float,
    kalman_spread: bool,
):
    logging.info(
        f"instantiating class for {row['first_ticker']} and {row['second_ticker']} "
    )
    try:
        bt = BackTest(
            row,
            spread_to_trigger_trade_entry=spread_to_trigger_trade_entry,
            spread_to_trigger_trade_exit=spread_to_trigger_trade_exit,
            spread_to_abandon_trade=spread_to_abandon_trade,
            kalman_spread=kalman_spread,
        )
        return bt.trade()
    except:
        logging.info(f"{row['first_ticker']} and {row['second_ticker']} FAILED SOMEHOW")


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    # NOTE TO USER: these 3 numeric configs below must also be set in the config file, they must match

    # First backtest with Kalman set to false
    spread_to_trigger_trade_entry = 2
    spread_to_trigger_trade_exit = 0.5
    spread_to_abandon_trade = 6
    kalman_spread = False

    Parallel(n_jobs=CORES_TO_USE)(
        delayed(execute_trade)(
            row,
            spread_to_trigger_trade_entry,
            spread_to_trigger_trade_exit,
            spread_to_abandon_trade,
            kalman_spread=kalman_spread,
        )
        for _, row in results_df.iterrows()
    )

    # Second backtest with Kalman set to True
    spread_to_trigger_trade_entry = 2
    spread_to_trigger_trade_exit = 0.5
    spread_to_abandon_trade = 6
    kalman_spread = True

    Parallel(n_jobs=CORES_TO_USE)(
        delayed(execute_trade)(
            row,
            spread_to_trigger_trade_entry,
            spread_to_trigger_trade_exit,
            spread_to_abandon_trade,
            kalman_spread=kalman_spread,
        )
        for _, row in results_df.iterrows()
    )

    logging.info("Backtest complete")
