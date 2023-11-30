import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    URL_TO_TICKER_DATA,
    PATHWAY_TO_PRICE_DF,
)

from main.utilities.constants import (
    SP_500_CONSTITUENTS_2013,
    STRATEGY_START_DATE,
    STRATEGY_END_DATE,
)

YFINANCE_DATE_FORMAT = "%Y-%m-%d"
COLUMN_TO_RETRIEVE = "Adj Close"
MIN_LENGTH_PRICE_SERIES = 500


def retrieve_tickers_url(
    url: str = URL_TO_TICKER_DATA,
    start_date: str = "2000-01-01",
    end_date: datetime.date = datetime(year=2023, month=6, day=1),
    series_length: int | None = None,
) -> None:

    constituents_df = pd.read_csv(url)
    constituents_series = constituents_df["Symbol"][:series_length]
    series_list = []

    for ticker in constituents_series:

        raw_series = yf.download(
            ticker, start=start_date, end=end_date.strftime(YFINANCE_DATE_FORMAT)
        )[COLUMN_TO_RETRIEVE]
        raw_series.name = ticker
        series_list.append(raw_series)

    price_df = pd.concat(series_list, axis=1)
    return price_df


def retrieve_tickers_from_list(
    ticker_list: str = SP_500_CONSTITUENTS_2013,
    start_date: str = STRATEGY_START_DATE,
    end_date: str = STRATEGY_END_DATE,
) -> None:

    series_list = []

    for ticker in ticker_list:

        raw_series = yf.download(
            ticker,
            start=start_date,
            end=end_date,
        )[COLUMN_TO_RETRIEVE]

        if len(raw_series) < MIN_LENGTH_PRICE_SERIES:
            logging.info(f"length of ticker series {ticker} error, skipping")
            continue

        raw_series.name = ticker
        series_list.append(raw_series)
        logging.info(f"downloaded price series for {ticker}")

    price_df = pd.concat(series_list, axis=1)
    return price_df


if __name__ == "__main__":

    price_df = retrieve_tickers_from_list()
    price_df.to_parquet(PATHWAY_TO_PRICE_DF)
    logging.info("ticker retrieval complete")
