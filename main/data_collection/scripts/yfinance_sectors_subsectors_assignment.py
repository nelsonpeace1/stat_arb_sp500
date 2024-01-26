

import asyncio
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_PRICE_DF,
    PATHWAY_TO_SECTORS_SUBSECTORS_DF,
)


def get_stock_details_single_ticker(ticker: str) -> tuple:
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get("sector", "N/A")
    subsector = info.get("industry", "N/A")
    logging.info(f"getting data for {ticker}")
    return (ticker, sector, subsector)


async def fetch_stock_data(ticker: str, loop):
    return await loop.run_in_executor(None, get_stock_details_single_ticker, ticker)


async def fetch_data_for_all_tickers(tickers):
    loop = asyncio.get_running_loop()
    tasks = [fetch_stock_data(ticker, loop) for ticker in tickers]
    return await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":

    tickers = pd.read_parquet(PATHWAY_TO_PRICE_DF).columns
    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    stock_data = asyncio.run(fetch_data_for_all_tickers(tickers))
    sectors_subsectors_df = pd.DataFrame(
        stock_data, columns=["Ticker", "Sector", "Subsector"]
    )

    for i, row in sectors_subsectors_df.iterrows():
        if isinstance(row["Sector"], Exception):
            logging.info(
                f"Failed to retrieve data for ticker {row['Ticker']}: {row['Sector']}"
            )
            continue

    results_df = results_df.merge(
        sectors_subsectors_df, how="left", left_on="first_ticker", right_on="Ticker"
    ).rename(
        mapper={"Sector": "first_ticker_sector", "Subsector": "first_ticker_subsector"},
        axis=1,
    )

    results_df = results_df.merge(
        sectors_subsectors_df, how="left", left_on="second_ticker", right_on="Ticker"
    ).rename(
        mapper={
            "Sector": "second_ticker_sector",
            "Subsector": "second_ticker_subsector",
        },
        axis=1,
    )

    results_df.drop(["Ticker_x", "Ticker_y"], axis=1, inplace=True)

    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    sectors_subsectors_df.to_parquet(PATHWAY_TO_SECTORS_SUBSECTORS_DF)
