import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

from datetime import datetime, timedelta
from math import floor
import logging
import sqlite3

import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO)

from main.utilities.functions import (
    retrieve_spread_table_from_sql_df,
    custom_create_db_engine,
)

from main.utilities.paths import (
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
    PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS,
)

from main.utilities.constants import (
    CAPITAL_STARTING,
    TRADE_STARTED_ABANDONED_STRING,
)

DB_NAME_BACKTEST_TRADEFRAMES = f"sqlite:///{PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS}"
DATABASE_NAME_SPREAD_BACKTEST = f"sqlite:///{PATHWAY_TO_SQL_DB_SPREADS_BACKTEST}"
STARTING_TRADE_COUNTER = 0
DAYS_IN_CALENDAR_YEAR = 365
DEFAULT_SHORTING_RATE_PER_ANNUM = 0.0025
FIRST_INDEX_TRADE_DF = 0
IBKR_COMMISSION_RATE = 0.0005
AVGE_SP500_BID_ASK_SPREAD_PERCENT = 0.03
SPREAD_SERIES_VALUATION_AND_INFO_COLS = [
    "valuation",
    "trade_open_bool",
    "trade_abandoned_bool",
]
TRADE_DF_RECORD_COLUMNS_LIST = [
    "trade_counter",
    "opening_date",
    "closing_date",
    "position_ticker1",
    "opening_price_ticker1",
    "closing_price_ticker1",
    "position_ticker2",
    "opening_price_ticker2",
    "closing_price_ticker2",
    "days_trade_open",
    "short_ticker",
    "closing_capital",
    "trade_abandoned",
]
SPREAD_TO_TRIGGER_TRADE_ENTRY = 2
SPREAD_TO_TRIGGER_TRADE_EXIT = 0.5
SPREAD_TO_ABANDON_TRADE = 6
SPREAD_HOP_TO_ABANDON_TRADE = 4

class BackTest:
    def __init__(
        self,
        row: pd.Series,
        spread_to_trigger_trade_entry: int = SPREAD_TO_TRIGGER_TRADE_ENTRY,
        spread_to_trigger_trade_exit: int = SPREAD_TO_TRIGGER_TRADE_EXIT,
        spread_to_abandon_trade: int = SPREAD_TO_ABANDON_TRADE,
        test_inputs: dict = None,
        kalman_spread: bool = False,
    ) -> None:

        self.ticker1 = row["first_ticker"]
        self.ticker2 = row["second_ticker"]
        self.ticker1_prices = pd.read_parquet(
            PATHWAY_TO_PRICE_DF, columns=[self.ticker1]
        ).squeeze()
        self.ticker2_prices = pd.read_parquet(
            PATHWAY_TO_PRICE_DF, columns=[self.ticker2]
        ).squeeze()
        if kalman_spread:
            self.kalman_spread = True
            self.regular_spread = retrieve_spread_table_from_sql_df(
                row,
                spread_type="_regular_spread_kalman",
                pathway=PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
            )
        else:
            self.kalman_spread = False
            self.regular_spread = retrieve_spread_table_from_sql_df(
                row,
                spread_type="_regular_spread",
                pathway=PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
            )
        self.standardised_spread = retrieve_spread_table_from_sql_df(
            row,
            spread_type="_standardised_spread",
            pathway=PATHWAY_TO_SQL_DB_SPREADS_BACKTEST,
        ).squeeze()
        self.spread_to_trigger_trade_entry = spread_to_trigger_trade_entry
        self.spread_to_trigger_trade_exit = spread_to_trigger_trade_exit
        self.spread_to_abandon_trade = spread_to_abandon_trade
        self.capital = CAPITAL_STARTING
        self.trade_history_frame = pd.DataFrame(columns=TRADE_DF_RECORD_COLUMNS_LIST)
        self.trade_counter = STARTING_TRADE_COUNTER
        self.trade_abandoned = False

        if test_inputs is not None:
            test_name_1 = self.ticker1_prices.name
            test_name_2 = self.ticker2_prices.name
            self.ticker1_prices = test_inputs["ticker1_prices"]
            self.ticker2_prices = test_inputs["ticker2_prices"]
            self.standardised_spread = test_inputs["standardised_spread"]
            self.ticker1_prices.name = test_name_1
            self.ticker2_prices.name = test_name_2

        # all these to be reset upon trade exit
        self.trade_status_open = False
        self.ticker1_minus_ticker2_trade_opening_spread_positive = None
        self.higher_priced_asset_ticker = None
        self.ticker1_holding = None
        self.ticker2_holding = None
        self.short_position_holding_name = None
        self.trade_opening_date = None
        self.short_capital_pool = 0
        self.ticker1_trade_opening_price = None
        self.ticker2_trade_opening_price = None

        if self.ticker1_prices.empty:
            raise ValueError(f"The price series for {self.ticker1} is empty.")
        if self.ticker2_prices.empty:
            raise ValueError(f"The price series for {self.ticker2} is empty.")

        logging.info(f"class instantiated for {self.ticker1} & {self.ticker2}")

    def trade(self, test_inputs: dict | None = None,) -> None:

        if (self.standardised_spread.iloc[0] > self.spread_to_abandon_trade) or (
            self.standardised_spread.iloc[0] < -self.spread_to_abandon_trade
        ):
            self.trade_abandoned = True
            self.trade_history_frame.loc[
                FIRST_INDEX_TRADE_DF, TRADE_DF_RECORD_COLUMNS_LIST
            ] = np.nan
            self.trade_history_frame.loc[FIRST_INDEX_TRADE_DF, "trade_abandoned"] = True
            self.trade_history_frame.loc[
                FIRST_INDEX_TRADE_DF, "closing_capital"
            ] = TRADE_STARTED_ABANDONED_STRING

        self.regular_spread[SPREAD_SERIES_VALUATION_AND_INFO_COLS] = np.nan

        for date, standardised_spread in self.standardised_spread.items():
            if (self.trade_abandoned == True) or (standardised_spread == 0):
                self.regular_spread.loc[date, SPREAD_SERIES_VALUATION_AND_INFO_COLS] = (
                    self.capital,
                    float(self.trade_status_open),
                    float(self.trade_abandoned),
                )
                continue

            if self.trade_status_open:

                self.regular_spread.loc[date, SPREAD_SERIES_VALUATION_AND_INFO_COLS] = (
                    self._perform_valuation_trade_open(date),
                    float(self.trade_status_open),
                    float(self.trade_abandoned),
                )

            else:

                self.regular_spread.loc[date, SPREAD_SERIES_VALUATION_AND_INFO_COLS] = (
                    self.capital,
                    float(self.trade_status_open),
                    float(self.trade_abandoned),
                )

            # these next two if/elif are for when the trade is to be abandoned
            if (
                (self.ticker1_minus_ticker2_trade_opening_spread_positive == True)
                and (standardised_spread >= self.spread_to_abandon_trade)
                and (self.trade_status_open)
            ):

                self._exit_when_spread_was_positive(date)
                self.trade_abandoned = True
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            elif (
                (self.ticker1_minus_ticker2_trade_opening_spread_positive == False)
                and (standardised_spread <= -self.spread_to_abandon_trade)
                and (self.trade_status_open)
            ):

                self._exit_when_spread_was_negative(date)
                self.trade_abandoned = True
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            # here we put a condition to exit the trade if it is the last day of the standardised spread, as this would be delisting
            if (
                date == self.standardised_spread.index.max()
            ) and self.trade_status_open == True:
                if standardised_spread > 0:
                    self._exit_when_spread_was_positive(date)
                    self.regular_spread.loc[date, "trade_abandoned"] = "last_listing"
                    self._record_trade(closing_date=date)
                    self._close_trade_attributes()
                elif standardised_spread < 0:
                    self._exit_when_spread_was_negative(date)
                    self.regular_spread.loc[date, "trade_abandoned"] = "last_listing"
                    self._record_trade(closing_date=date)
                    self._close_trade_attributes()

                logging.info(
                    f"{self.ticker1}_{self.ticker2} backtest incurred a last listing event on date {date}"
                )

            # these two elifs are for if the trade 'hops the spread' in one direction
            elif (
                (self.ticker1_minus_ticker2_trade_opening_spread_positive == True)
                and (standardised_spread < 0)
                and (self.trade_status_open)
                and (not self.trade_abandoned)
            ):
                logging.info(
                    f"trade ({self.ticker1}_{self.ticker2}) has 'hopped the spread' from pos to neg on date {date}"
                )

                if abs(standardised_spread) > self.spread_to_abandon_trade or (
                    abs(standardised_spread)
                    + abs(self.standardised_spread.shift().loc[date])
                    > SPREAD_HOP_TO_ABANDON_TRADE
                ):

                    self.trade_abandoned = True
                self._exit_when_spread_was_positive(date)
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            elif (
                (self.ticker1_minus_ticker2_trade_opening_spread_positive == False)
                and (standardised_spread > 0)
                and (self.trade_status_open)
                and (not self.trade_abandoned)
            ):
                logging.info(
                    f"trade ({self.ticker1}_{self.ticker2}) has 'hopped the spread' from neg to pos on date {date}"
                )
                if abs(standardised_spread) > self.spread_to_abandon_trade or (
                    (standardised_spread - self.standardised_spread.shift().loc[date])
                    > SPREAD_HOP_TO_ABANDON_TRADE
                ):

                    self.trade_abandoned = True
                self._exit_when_spread_was_negative(date)
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            # here spread is positive so we buy 2 and sell 1
            elif (
                (standardised_spread >= (self.spread_to_trigger_trade_entry))
                and (not self.trade_status_open)
                and (not self.trade_abandoned)
            ):
                self._entry_when_spread_was_positive(date)

            # here spread is negative so we buy 1 and sell 2
            elif (
                (standardised_spread <= -(self.spread_to_trigger_trade_entry))
                and (not self.trade_status_open)
                and (not self.trade_abandoned)
            ):
                self._entry_when_spread_was_negative(date)

            # here write code to exit when sold 1 and bought 2 (because the spread is/was positive)
            elif (
                (standardised_spread < (self.spread_to_trigger_trade_exit))
                and (standardised_spread > 0)
                and (self.trade_status_open)
                and (not self.trade_abandoned)
            ):

                self._exit_when_spread_was_positive(date)
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            # here write code to exit when bought 1 and sold 2 (because the spread is/was negative)
            elif (
                (standardised_spread > -(self.spread_to_trigger_trade_exit))
                and (standardised_spread < 0)
                and (self.trade_status_open)
                and (not self.trade_abandoned)
            ):

                self._exit_when_spread_was_negative(date)
                self._record_trade(closing_date=date)
                self._close_trade_attributes()

            else:
                continue

        # results df sql table creation
        if test_inputs is None:
            self._save_trade_history_df_to_sql(kalman_spread=self.kalman_spread)
            self._save_regular_spread_df_to_sql(kalman_spread=self.kalman_spread)

        logging.info(f"Completed backtest for {self.ticker1} & {self.ticker2}")
        
                
    def _trade_entry_common_logic(
        self,
        date: datetime,
        spread_positive: bool,
    ) -> None:
        
        self.trade_status_open = True
        self.ticker1_minus_ticker2_trade_opening_spread_positive = spread_positive
        self.trade_opening_date = date
        self.ticker2_trade_opening_price = self.ticker2_prices[date]
        self.ticker1_trade_opening_price = self.ticker1_prices[date]

    # entries
    def _entry_when_spread_was_negative(self, date: datetime,) -> None:
        (
            number_assets_of_higher_priced_asset,
            number_assets_of_lower_priced_asset,
        ) = self._calculate_optimum_allocation_to_each_asset(date)
        
        if self.ticker1 == self.higher_priced_asset_ticker:
            self.ticker1_holding = number_assets_of_higher_priced_asset
            self.ticker2_holding = number_assets_of_lower_priced_asset
        else:
            self.ticker2_holding = number_assets_of_higher_priced_asset
            self.ticker1_holding = number_assets_of_lower_priced_asset

        self._trade_entry_common_logic(
            date=date,
            spread_positive=False,
        )
        
        self.short_position_holding_name = self.ticker2
        self.capital -= (
            self.ticker1_holding * self.ticker1_prices[date]
        ) - self._transaction_cost_calculator(
            self.ticker1_holding,
            date,
            self.ticker1_prices[date],
        )  # reduce the capital by the amount of ticker 1 stock we bought
        self.short_capital_pool += (
            self.ticker2_holding * self.ticker2_prices[date]
        ) - self._transaction_cost_calculator(
            self.ticker2_holding,
            date,
            self.ticker2_prices[date],
            exit_and_short=True,
        )
             

    def _entry_when_spread_was_positive(self, date: datetime,) -> None:
        (
            number_assets_of_higher_priced_asset,
            number_assets_of_lower_priced_asset,
        ) = self._calculate_optimum_allocation_to_each_asset(date)

        if self.ticker1 == self.higher_priced_asset_ticker:
            self.ticker1_holding = number_assets_of_higher_priced_asset
            self.ticker2_holding = number_assets_of_lower_priced_asset
        else:
            self.ticker2_holding = number_assets_of_higher_priced_asset
            self.ticker1_holding = number_assets_of_lower_priced_asset

        self._trade_entry_common_logic(
            date=date,
            spread_positive=True,
        )
        
        self.short_position_holding_name = self.ticker1
        self.capital -= (
            self.ticker2_holding * self.ticker2_prices[date]
        ) - self._transaction_cost_calculator(
            self.ticker2_holding,
            date,
            self.ticker2_prices[date],
        )  # reduce the capital by the amount of ticker 2 stock we bought
        self.short_capital_pool += (
            self.ticker1_holding * self.ticker1_prices[date]
        ) - self._transaction_cost_calculator(
            self.ticker1_holding,
            date,
            self.ticker1_prices[date],
            exit_and_short=True,
        )

    # exits
    def _exit_when_spread_was_positive(self, date) -> None:
        revenue_from_long_position = self.ticker2_prices[date] * self.ticker2_holding
        outflow_from_short_position = self.ticker1_prices[date] * self.ticker1_holding

        self.short_capital_pool -= (
            outflow_from_short_position
            + self._transaction_cost_calculator(
                self.ticker1_holding,
                date,
                self.ticker1_prices[date],
                exit_and_short=True,
            )
        )
        self.capital += revenue_from_long_position - self._transaction_cost_calculator(
            self.ticker2_holding,
            date,
            self.ticker2_prices[date],
        )
        self.capital += self.short_capital_pool

    def _exit_when_spread_was_negative(self, date: datetime) -> None:
        revenue_from_long_position = self.ticker1_prices[date] * self.ticker1_holding
        outflow_from_short_position = self.ticker2_prices[date] * self.ticker2_holding

        self.short_capital_pool -= (
            outflow_from_short_position
            + self._transaction_cost_calculator(
                self.ticker1_holding,
                date,
                self.ticker1_prices[date],
                exit_and_short=True,
            )
        )
        self.capital += revenue_from_long_position - self._transaction_cost_calculator(
            self.ticker2_holding,
            date,
            self.ticker2_prices[date],
        )
        self.capital += self.short_capital_pool

    def _calculate_optimum_allocation_to_each_asset(
        self,
        date: datetime,
    ) -> tuple[int, int]:

        (
            higher_priced_asset_price_series,
            lower_priced_asset_price_series,
        ) = self._return_higher_priced_asset_price_series(date)

        (
            capital_allocated_to_higher_priced_asset,
            number_assets_of_higher_priced_asset,
            current_price_of_higher_priced_asset,
        ) = self._determine_capital_and_units_higher_priced_asset(
            date, higher_priced_asset_price_series
        )

        (
            number_assets_of_lower_priced_asset,
            current_price_of_lower_priced_asset,
        ) = self._determine_capital_and_units_lower_priced_asset(
            date,
            capital_allocated_to_higher_priced_asset,
            lower_priced_asset_price_series,
        )

        if (
            number_assets_of_higher_priced_asset * current_price_of_higher_priced_asset
            + number_assets_of_lower_priced_asset * current_price_of_lower_priced_asset
        ) > self.capital:
            logging.info(
                f"combined asset position is higher than capital allocated to strategy, allocation optimisation failed"
            )

        return number_assets_of_higher_priced_asset, number_assets_of_lower_priced_asset

    def _return_higher_priced_asset_price_series(
        self, date: datetime
    ) -> tuple[pd.Series, pd.Series]:
        if self.ticker1_prices[date] > self.ticker2_prices[date]:
            self.higher_priced_asset_ticker = self.ticker1_prices.name
            return self.ticker1_prices, self.ticker2_prices
        else:
            self.higher_priced_asset_ticker = self.ticker2_prices.name
            return self.ticker2_prices, self.ticker1_prices

    def _determine_capital_and_units_lower_priced_asset(
        self,
        date: datetime,
        capital_allocated_to_higher_priced_asset: float,
        lower_priced_asset_price_series: pd.Series,
    ) -> tuple[int, float]:

        current_price_of_lower_priced_asset = lower_priced_asset_price_series[date]
        if (
            round(
                capital_allocated_to_higher_priced_asset
                / current_price_of_lower_priced_asset
            )
            * current_price_of_lower_priced_asset
            + capital_allocated_to_higher_priced_asset
        ) < self.capital:
            number_assets_of_lower_priced_asset = round(
                capital_allocated_to_higher_priced_asset
                / current_price_of_lower_priced_asset
            )
        else:
            number_assets_of_lower_priced_asset = floor(
                capital_allocated_to_higher_priced_asset
                / current_price_of_lower_priced_asset
            )

        return number_assets_of_lower_priced_asset, current_price_of_lower_priced_asset

    def _determine_capital_and_units_higher_priced_asset(
        self,
        date: datetime,
        higher_priced_asset_price_series: pd.Series,
    ) -> tuple[float, int, float]:

        current_price_of_higher_priced_asset = higher_priced_asset_price_series[date]
        number_assets_of_higher_priced_asset = floor(
            (self.capital / 2) / current_price_of_higher_priced_asset
        )
        capital_allocated_to_higher_priced_asset = (
            number_assets_of_higher_priced_asset * current_price_of_higher_priced_asset
        )

        return (
            capital_allocated_to_higher_priced_asset,
            number_assets_of_higher_priced_asset,
            current_price_of_higher_priced_asset,
        )

    def _return_price_series_from_ticker(
        self,
        ticker: str,
    ) -> pd.Series:
        if ticker == self.ticker1_prices.name:
            return self.ticker1_prices
        else:
            return self.ticker2_prices

    def _transaction_cost_calculator(
        self,
        number_of_units_transacted: int,
        date: pd.Timestamp,
        cost_of_asset: float,
        exit_and_short: bool = False,
    ) -> float:

        commission_costs = (
            number_of_units_transacted * cost_of_asset
        ) * IBKR_COMMISSION_RATE
        bid_ask_spread_costs = (
            number_of_units_transacted * AVGE_SP500_BID_ASK_SPREAD_PERCENT
        )
        short_exit_costs = (
            (date - self.trade_opening_date).days
            / DAYS_IN_CALENDAR_YEAR
            * DEFAULT_SHORTING_RATE_PER_ANNUM
            * exit_and_short
        )

        return commission_costs + bid_ask_spread_costs + short_exit_costs

    def _close_trade_attributes(
        self,
    ) -> None:

        self.trade_status_open = False
        self.ticker1_minus_ticker2_trade_opening_spread_positive = None
        self.higher_priced_asset_ticker = None
        self.ticker1_holding = None
        self.ticker2_holding = None
        self.short_position_holding_name = None
        self.trade_opening_date = None
        self.short_capital_pool = 0
        self.ticker1_trade_opening_price = None
        self.ticker2_trade_opening_price = None

    def _record_trade(
        self,
        closing_date: datetime,
    ) -> None:
        trade_instance_data = {
            "trade_counter": self.trade_counter,
            "opening_date": self.trade_opening_date,
            "closing_date": closing_date,
            "position_ticker1": self.ticker1_holding,
            "opening_price_ticker1": self.ticker1_trade_opening_price,
            "closing_price_ticker1": self.ticker1_prices[closing_date],
            "position_ticker2": self.ticker2_holding,
            "opening_price_ticker2": self.ticker2_trade_opening_price,
            "closing_price_ticker2": self.ticker2_prices[closing_date],
            "days_trade_open": (closing_date - self.trade_opening_date).days,
            "short_ticker": self.short_position_holding_name,
            "closing_capital": self.capital,
            "trade_abandoned": self.trade_abandoned,
        }

        self.trade_history_frame.loc[self.trade_counter] = trade_instance_data
        self.trade_counter += 1

    def _save_trade_history_df_to_sql(
        self,
        kalman_spread: bool,
    ) -> None:
        engine = custom_create_db_engine(DB_NAME_BACKTEST_TRADEFRAMES)
        results_table_name = (
            f"{self.ticker1}_{self.ticker2}_{self.spread_to_trigger_trade_entry}_{self.spread_to_trigger_trade_exit}_{self.spread_to_abandon_trade}{'_kalman' if kalman_spread else ''}"
        ).replace(".", "")
        self._save_pandas_object_to_database(
            results_table_name,
            self.trade_history_frame,
            engine,
        )

    def _save_regular_spread_df_to_sql(
        self,
        kalman_spread: bool,
    ) -> None:
        engine = custom_create_db_engine(DATABASE_NAME_SPREAD_BACKTEST)
        spread_series_name = (
            f"{self.ticker1}_{self.ticker2}_{self.spread_to_trigger_trade_entry}_{self.spread_to_trigger_trade_exit}_{self.spread_to_abandon_trade}{'_kalman' if kalman_spread else ''}"
        ).replace(".", "")
        self._save_pandas_object_to_database(
            spread_series_name,
            self.regular_spread,
            engine,
        )

    def _save_pandas_object_to_database(
        self,
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

    def _perform_valuation_trade_open(
        self,
        date: pd.Timestamp,
    ) -> tuple:

        if self.ticker1_minus_ticker2_trade_opening_spread_positive:
            # this means we are short ticker 1
            short_position_value = (
                self.short_capital_pool
                - (self.ticker1_holding * self.ticker1_prices[date])
                - self._transaction_cost_calculator(
                    self.ticker1_holding,
                    date,
                    self.ticker1_prices[date],
                    exit_and_short=True,
                )
            )

            long_position_value = self.ticker2_holding * self.ticker2_prices[
                date
            ] - self._transaction_cost_calculator(
                self.ticker2_holding,
                date,
                self.ticker2_prices[date],
            )

        else:

            short_position_value = (
                self.short_capital_pool
                - (self.ticker2_holding * self.ticker2_prices[date])
                - self._transaction_cost_calculator(
                    self.ticker2_holding,
                    date,
                    self.ticker2_prices[date],
                    exit_and_short=True,
                )
            )

            long_position_value = self.ticker1_holding * self.ticker1_prices[
                date
            ] - self._transaction_cost_calculator(
                self.ticker1_holding,
                date,
                self.ticker1_prices[date],
            )

        return long_position_value + short_position_value + self.capital



