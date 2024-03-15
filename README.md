This herein project is an experiment in demonstrating several fundamental mechanics of pairs trading.

- A basic set of mechanics on pairs trading/stat arb tools. A user can plug in their desired universe of assets, run the strategy, and have pairs trading candidates generated. Similarly, a user can take various moving parts of this overall strategy and implement them in their own.
- This is a demonstration of moving parts, not an actual tenable strategy. Statistical arbitrage alone will not make you money, a sound underlying economic rationale for why two instruments trade in linearly consistent ratio is necessary for a successful strategy, this is the 'art' of statistical arbitrage.

Readers who want to skip ahead and into the code, can start their exploration at the back-test class: https://github.com/nelsonpeace1/stat_arb_sp500/blob/master/main/model_building/backtesting/backtest.py. 

This is presented as a highly non specific experiment under the (hopefully obvious) assumption that the reader is not expecting me to give away the nuances of the profitable trading strategies I am running.


This repo has three phases:
1. Phase 1: take a universe of assets, perform cointegration testing, adf testing, hurst exponent calculations, half life calculations, etc.
2. Phase 2: back-testing. A back-test of all possible pairs is executed whereby a user can arrive at sensible conclusions about how to run phase 3 below.
3. Phase 3: paper trading of candidate pairs. Once candidate pairs are identified, these pairs compete for capital in a fully automated paper trading environment. The building of this third phase is underway at the time of writing.


Shortfalls/constraints in this approach:
1. A more sophisticated approach may entail a different and more sophisticated apportionment of capital to each of the pairs in the back-test. At the present time, this is simply split 50-50. Other approaches may be risk weighted approaches or even principle component weighted approaches. Note, a simple 50-50 apportionment here is not as folly as it may be in other relative value strategies, as the use of a hedge ratio has to some extent determined a ratio whereby one asset is scaled to behave in pricing tandem with the second. This notwithstanding, the user may consider a more nuanced approach.
2. Constant risk free rate. The strategy assumes a constant risk free rate in the sharpe ratio. Conjecture exists as to whether a risk free rate is even appropriate in a dollar neutral trading strategy, yet if the user wishes to maintain the use of one, a more sophisticated approach would involve the use of a risk free rate benchmarked to the time of the trade.
3. Testing. The testing suite is set to the strategy I have run, the SP500 ticker list as at 2013-06-01. Tests depend on these values. A more robust implementation would have markers for testing, with a secondary testing suite which ran on saved data independent of the strategy being executed and the assets being run. This data repo uses only minimal testing, for example functions which are not crucial for the execution of the strategy (like some visualisation functions) are not tested. In the subsequent production environment for phase 3 (discussed below) a far more robust and exhaustive testing suite will be implemented, as any developer would be expected to implement for production environments.
4. At the time of writing, Metaflow is only supported for MacOs/Linux. Metaflow is my preferred ML ops pipeline and I'd have used it were I not working on Windows.
5. I write my code in a fashion whereby names explain function purposes. I therefore minimally rely on docstrings. A more robust approach suitable for production environments would have been to write docstrings for wrapper functions as a minimum.

Other considerations/disclaimers:
1. The astute programmer will note the improper use of the itterows() object in for loops in if __name__ blocks. This is chosen because it allows me to easily parallelise using joblib, and in my case, only adds a few extra seconds of overhead per script.
2. The opening and closing of database connections is deliberate, allowing me to parallelise operations.


Directions for use:
1. Update the local .env file with your pathways, and modify the paths file with your root path. Then modify the 'main\model_building\backtesting\backtest_execution.py' file with your preferred back-testing hyper parameters (entry threshold, exit threshold, abandon threshold). An example of this is below in appendix 1. Name your own constant in the constants file (mine is called FIRST_BACKTEST_PARAMS), with back-test parameters in this format (underscore/entry threshold/underscore/exit threshold with decimal point removed). The user may also wish to modify their 'CORES_TO_USE' constant in the constants file (if they have a fancier computer than mine, which they almost certainly do). The user should also note that using more than 4 cores can lead to issues in retrieving tables from the Sqlite3 implementations (the accessing of these databases is done in a parallelised fashion, and many more than 4 cores will break it)
2. You can load a list of tickers in list format into the file 'main/utilities/constants.py', of which price histories will then be retrieved using the functionality in 'main\data_collection\scripts\yfinance_data_pull.py'. Alternatively, if you wish to simply download a list of tickers from the Github listed in the paths file (not my repo), use this script and swap the function in the 'if __name__' block out for 'retrieve_tickers_url' (in the aforementioned script). At present, this function is dormant. After running this script, you will have stored a parquet file with all the prices histories of your asset. Yfinance api support is inconsistent.
3. The computations are made from within the 'run_pairs_trade.py' file. Run this file in your preferred fashion. At the time of writing, the yfinance api does not support retrieving stock sectors, so this script is removed.
4. Examine the notebook at 'main\model_building\backtesting_analysis\notebooks\backtesting-analysis.ipynb'. This reports on several initial metrics in the back-test, and the user can continue this enquiry in the same fashion for mine, or their own strategy. This notebook compares the equity curves from Phase 2 with different tools and back-test parameters (kalman filter vs ols hedge ratio, etc)
5. Before deciding on back-test parameters, a user may wish to emulate my approach in 'main\notebooks\eda\backtesting\eda-backtesting-1.0.ipynb' where I consider different thresholds. Note, I do not 'fit' the back-test to these levels, as in my opinion, doing so can (but will not necessarily) lead to back-test over fitting.
6. The user will need to upload two parquet files, one with the prices and a second with the sectors of those tickers. The ticker names must contain letters and numbers only (no special chars). The prices df should have tickers as columns and a pd.timestamp as index. The sectors parquet should contain a column called 'Instrument', with the instrument names corresponding to the columns in the prices pq file.


Appendix 1: Example of back-test params:
    spread_to_trigger_trade_entry = 2
    spread_to_trigger_trade_exit = 0.5
    spread_to_abandon_trade = 6
