

from main.data_collection.scripts.yfinance_data_pull import (
    retrieve_tickers_url,
)

from datetime import datetime

TEST_DATE_START = "2020-01-01"
TEST_DATE_END_STRING = "2020-01-03"
LENGTH_TICKER_TESTING_DF = 1
SERIES_LENGTH_FOR_TESTING = 3
YFINANCE_DATE_FORMAT = "%Y-%m-%d"
TEST_DATE_END = datetime.strptime(TEST_DATE_END_STRING, YFINANCE_DATE_FORMAT)


def test_smoke_retrieve_tickers() -> None:

    tickers_data = retrieve_tickers_url(
        start_date=TEST_DATE_START,
        end_date=TEST_DATE_END,
        series_length=SERIES_LENGTH_FOR_TESTING,
    )

    assert len(tickers_data) == LENGTH_TICKER_TESTING_DF
    assert all(isinstance(value, float) for value in tickers_data.iloc[0].values)
