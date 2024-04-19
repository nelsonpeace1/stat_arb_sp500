This herein project is an experiment in demonstrating several fundamental mechanics of pairs trading.

- A basic set of mechanics on pairs trading/stat arb tools. A user can plug in their desired universe of assets, run the strategy, and have pairs trading candidates generated. Similarly, a user can take various moving parts of this overall strategy and implement them in their own.
- This is a demonstration of moving parts, not an actual tenable strategy. Statistical arbitrage alone will not make you money, a sound underlying economic rationale for why two instruments trade in linearly consistent ratio is necessary for a successful strategy, this is the 'art' of statistical arbitrage.

This is presented as a highly non specific experiment under the (hopefully obvious) assumption that the reader is not expecting me to give away the nuances of the profitable trading strategies I am running.

For those who would rather just get right to the code, it's best to start at the metaflow pipeline in main/metaflow_pairs_trade.py. Metaflow is a great orchestrator which not only ties all your code together, but gives other devs a linear, sequential tour of your project's functionality. The user can then delve into the code from there. Readers who know metalfow well will notice that I have not used it to its full capacity. Metaflows batch processing and parallelisation capabilities would have worked well for this large compute load. However, at the time I wrote most of this code in Sept 2023, I was using SQlite3 databases, and these have significant concurrency limits. I'd do it differently now.

This repo has three phases:
1. Phase 1a: take a universe of assets, perform cointegration testing, adf testing, hurst exponent calculations, half life calculations, etc.
2. Phase 1b: back-testing. A back-test of all possible pairs is executed whereby a user can arrive at sensible conclusions about how to run phase 2 below.
3. Phase 2: paper trading of candidate pairs. Once candidate pairs are identified, these pairs compete for capital in a fully automated paper trading environment. The building of this third phase is underway at the time of writing.


Shortfalls/constraints in this approach:
1. A more sophisticated approach may entail a different and more sophisticated apportionment of capital to each of the pairs in the back-test. At present, this is simply split 50-50. Other approaches may be risk weighted approaches or even principle component weighted approaches. Note, a simple 50-50 apportionment here is not as folly as it may be in other relative value strategies, as the use of a hedge ratio has to some extent determined a ratio whereby one asset is scaled to behave in pricing tandem with the second. This notwithstanding, the user may consider a more nuanced approach.
2. Constant risk free rate. The strategy assumes a constant risk free rate in the sharpe ratio. Conjecture exists as to whether a risk free rate is even appropriate in a dollar neutral trading strategy, yet if the user wishes to maintain the use of one, a more sophisticated approach would involve the use of a risk free rate benchmarked to the time of the trade.
3. Testing. The testing suite is set to the strategy I have run, the SP500 ticker list as at 2013-06-01. Tests depend on these values. A more robust implementation would have markers for testing, with a secondary testing suite which ran on saved data independent of the strategy being executed and the assets being run. This data repo uses only minimal testing, for example functions which are not crucial for the execution of the strategy (like some visualisation functions) are not tested. In the subsequent production environment for phase 3 (discussed below) a far more robust and exhaustive testing suite will be implemented, as any developer would be expected to implement for production environments.
4. I write my code in a fashion whereby names explain function purposes. I therefore minimally rely on docstrings. A more robust approach suitable for production environments would have been to write docstrings for wrapper functions as a minimum.
5. The backtesting methodology is simple. A more robust approach would use what is now referred to as combinatorial purged cross fold validation (file:///Users/nelsonpeace/Downloads/SSRN-id4778909.pdf). My next repo will feature this.

Other considerations/disclaimers:
1. The astute programmer will note the improper use of the itterows() object in for loops in if __name__ blocks and wrapper functionality in the modules. This is chosen because it allows me to easily parallelise using joblib, and in my case, only adds a few extra seconds of overhead per script.
2. The opening and closing of database connections is deliberate, allowing me to parallelise operations.


Directions for use:
1. Modify the paths file with your root path. The user may also wish to modify their 'CORES_TO_USE' constant in the constants file (if they have a fancier computer than mine, which they almost certainly do). The user should also note that using more than 4 cores can lead to issues in retrieving tables from the Sqlite3 implementations (the accessing of these databases is done in a parallelised fashion, and many more than 4 cores will break it)
3. Everything is run from the metaflow file. You can run this file with python3 metaflow_pairs_trade.py run. Set your backtesting parameters as you wish.
4. Examine the notebook at 'main\model_building\backtesting_analysis\notebooks\backtesting-analysis.ipynb'. This reports on several initial metrics in the back-test, and the user can continue this enquiry in the same fashion for mine, or their own strategy. This notebook compares the equity curves from Phase 2 with different tools and back-test parameters (kalman filter vs ols hedge ratio, etc)
5. Before deciding on back-test parameters, a user may wish to emulate my approach in 'main\notebooks\eda\backtesting\eda-backtesting-1.0.ipynb' where I consider different thresholds. Note, I do not 'fit' the back-test to these levels, as in my opinion, doing so can (but will not necessarily) lead to back-test over fitting.
6. The user will need to upload two parquet files, one with the prices and a second with the sectors of those tickers. The ticker names must contain letters and numbers only (no special chars). The prices df should have tickers as columns and a pd.timestamp as index. The sectors parquet should contain a column called 'Instrument', with the instrument names corresponding to the columns in the prices pq file.


Appendix 1: Example of back-test params:
    spread_to_trigger_trade_entry = 2
    spread_to_trigger_trade_exit = 0.5
    spread_to_abandon_trade = 6
