import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
    PATHWAY_TO_SECTORS_SUBSECTORS_DF,
)


def concatenate_sectors_in_column(
    results_df: pd.DataFrame,
) -> pd.DataFrame:

    sectors_list = np.sort(
        results_df[["first_ticker_sector", "second_ticker_sector"]].values, axis=1
    )
    results_df["tickers_sectors_concat"] = [
        " - ".join(sublist) for sublist in sectors_list
    ]

    return results_df


def sector_mapper(
    sector_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    sector_map = dict(zip(sector_df["Instrument"], sector_df["sector"]))

    results_df["first_ticker_sector"] = results_df["first_ticker"].map(sector_map)
    results_df["second_ticker_sector"] = results_df["second_ticker"].map(sector_map)

    return results_df


if __name__ == "__main__":

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)
    sector_df = pd.read_parquet(PATHWAY_TO_SECTORS_SUBSECTORS_DF)

    results_df = sector_mapper(
        sector_df=sector_df,
        results_df=results_df,
    )

    results_df = concatenate_sectors_in_column(
        results_df=results_df,
    )

    results_df.to_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    logging.info("finished sector mapping")
