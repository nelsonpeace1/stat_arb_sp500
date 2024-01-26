

import pandas as pd

from main.utilities.paths import (
    PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF,
)

from main.model_building.backtesting_analysis.equity_curves import (
    create_eq_curve,
)


def test_create_equity_curve():

    results_df = pd.read_parquet(PATHWAY_TO_COINTEGRATION_AND_RESULTS_DF)

    testing_results_df = results_df.iloc[0:2, :]

    testing_object_eq_curve = create_eq_curve(
        results_df=testing_results_df,
    )

    assert isinstance(testing_object_eq_curve, pd.Series)
    assert round(testing_object_eq_curve[-1], 0) == 213057
