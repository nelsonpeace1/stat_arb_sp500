import pandas as pd
import sqlite3
from sqlalchemy.engine.base import Engine


from main.utilities.paths import (
    PATHWAY_TO_SQL_DB_SPREADS,
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)
from main.utilities.functions import (
    custom_create_db_engine,
    retrieve_spread_table_from_sql_df,
)

TICKER_1_TO_TEST_WITH = "XRXOQ"
TICKER_2_TO_TEST_WITH = "PBIN"
FIRST_VALUE_OF_XRXOQ_PBIN_REGULAR_SPREAD = -1
DATABASE_NAME_SPREAD = f"sqlite:///{PATHWAY_TO_SQL_DB_SPREADS}"


def test_retrieve_spread_table_from_sql_df():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ].squeeze()

    testing_obj = retrieve_spread_table_from_sql_df(row)
    assert type(testing_obj) == pd.DataFrame
    assert round(testing_obj.values[0][0]) == FIRST_VALUE_OF_XRXOQ_PBIN_REGULAR_SPREAD


def test_create_db_engine():

    testing_obj = custom_create_db_engine(DATABASE_NAME_SPREAD)
    assert isinstance(testing_obj, Engine)
