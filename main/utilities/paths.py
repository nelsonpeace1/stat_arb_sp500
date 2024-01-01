import os

# Place your root directory below inside the ""
ROOT_DIR = r""

PATHWAY_TO_PRICE_DF = os.path.join(
    ROOT_DIR,
    r"main\data_collection\data\processed\price_df_4.parquet"
)
PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF = os.path.join(
    ROOT_DIR, r"main\data_collection\data\processed\results_df.parquet"
)
PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS = os.path.join(
    ROOT_DIR, r"main\databases\rolling_hedge_ratio_database.db"
)
PATHWAY_TO_SQL_DB_SPREADS = os.path.join(ROOT_DIR, r"main\databases\spread_database.db")
URL_TO_TICKER_DATA = r"https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
PATHWAY_TO_TESTING_RESULTS_DF = os.path.join(
    ROOT_DIR, r"tests\static_files_for_tests\testing_results_df.parquet"
)
PATHWAY_TO_TESTING_PRICES_DF = os.path.join(
    ROOT_DIR, r"tests\static_files_for_tests\testing_prices_df.parquet"
)
PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST = os.path.join(
    ROOT_DIR, r"main\databases\rolling_hedge_ratio_database_backtest.db"
)
PATHWAY_TO_SQL_DB_SPREADS_BACKTEST = os.path.join(
    ROOT_DIR, r"main\databases\spread_database_backtest.db"
)
PATHWAY_TO_SQL_DB_OF_BACKTEST_RESULT_DFS = os.path.join(
    ROOT_DIR, r"main\databases\backtest_results_df_database.db"
)
PATHWAY_TO_SECTORS_SUBSECTORS_DF = os.path.join(
    ROOT_DIR, r"main\data_collection\data\processed\sectors_subsectors_df.parquet"
)
PATHWAY_TO_SQL_DB_OF_ROLLING_HEDGE_RATIOS_BACKTEST_KALMAN = os.path.join(
    ROOT_DIR, r"main\databases\rolling_hedge_ratio_database_backtest_kalman.db"
)
