import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

import pandas as pd

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)
from main.model_building.scripts.half_life import (
    half_life_ornstein_whole_set,
)

TICKER_1_TO_TEST_WITH = "XRXOQ"
TICKER_2_TO_TEST_WITH = "PBIN"
XRXOQ_PBIN_HALF_LIFE = 52


def test_half_life_ornstein_whole_set():

    testing_results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    row = testing_results_df[
        (testing_results_df["first_ticker"] == TICKER_1_TO_TEST_WITH)
        & (testing_results_df["second_ticker"] == TICKER_2_TO_TEST_WITH)
    ]

    testing_obj = half_life_ornstein_whole_set(row)
    assert round(testing_obj[0], 0) == XRXOQ_PBIN_HALF_LIFE
