import pandas as pd
import sqlite3

from main.utilities.functions import (
    custom_create_db_engine,
)

from main.utilities.paths import (
    PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
    PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS,
)


class TradeHistorySaver:
    
    
    """This class is used to save the trade history dataframe to a SQL database.
    """
    
    DATABASE_NAME_BACKTEST_TRADEFRAMES = f"sqlite:///{PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS}"
    
    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        spread_to_trigger_trade_entry: int | float,
        spread_to_trigger_trade_exit: int | float,
        spread_to_abandon_trade: int | float,
        kalman_spread: bool,
        trade_history_frame: pd.DataFrame,
    ) -> None:
        
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.spread_to_trigger_trade_entry = spread_to_trigger_trade_entry
        self.spread_to_trigger_trade_exit = spread_to_trigger_trade_exit
        self.spread_to_abandon_trade = spread_to_abandon_trade
        self.kalman_spread = kalman_spread
        self.trade_history_frame = trade_history_frame


    def save_trade_history_df_to_sql(
        self,
    ) -> None:
        engine = custom_create_db_engine(self.DATABASE_NAME_BACKTEST_TRADEFRAMES)
        results_table_name = (
            f"{self.ticker1}_{self.ticker2}_{self.spread_to_trigger_trade_entry}_{self.spread_to_trigger_trade_exit}_{self.spread_to_abandon_trade}{'_kalman' if self.kalman_spread else ''}"
        ).replace(".", "")
        save_pandas_object_to_database(
            results_table_name,
            self.trade_history_frame,
            engine,
        )
       
       
class RegularSpreadSaver:      
    
    DATABASE_NAME_SPREAD_BACKTEST = f"sqlite:///{PATHWAY_TO_SQL_DB_SPREADS_BACKTEST}"
    
    
    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        spread_to_trigger_trade_entry: int | float,
        spread_to_trigger_trade_exit: int | float,
        spread_to_abandon_trade: int | float,
        kalman_spread: bool,
        regular_spread: pd.DataFrame,
        
    ) -> None:
        
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.spread_to_trigger_trade_entry = spread_to_trigger_trade_entry
        self.spread_to_trigger_trade_exit = spread_to_trigger_trade_exit
        self.spread_to_abandon_trade = spread_to_abandon_trade
        self.kalman_spread = kalman_spread
        self.regular_spread = regular_spread
    
     
    def save_regular_spread_df_to_sql(
        self,
        kalman_spread: bool,
    ) -> None:
        engine = custom_create_db_engine(self.DATABASE_NAME_SPREAD_BACKTEST)
        spread_series_name = (
            f"{self.ticker1}_{self.ticker2}_{self.spread_to_trigger_trade_entry}_{self.spread_to_trigger_trade_exit}_{self.spread_to_abandon_trade}{'_kalman' if kalman_spread else ''}"
        ).replace(".", "")
        save_pandas_object_to_database(
            spread_series_name,
            self.regular_spread,
            engine,
        )
       
       
def save_pandas_object_to_database( #TODO refactor with context manager
    table_name: str,
    trade_history_df: pd.DataFrame,
    engine: sqlite3.Connection,
) -> None:
    trade_history_df.name = table_name
    trade_history_df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=True,
    )
