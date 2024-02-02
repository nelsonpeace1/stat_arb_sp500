import sys
import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO)

from main.utilities.paths import ROOT_DIR

# List of scripts to run with their full paths

scripts = [
    f"{ROOT_DIR}main/model_building/scripts/cointegration_testing.py",
    f"{ROOT_DIR}main/model_building/scripts/hedge_ratio_calculations.py",
    f"{ROOT_DIR}main/model_building/scripts/creating_spreads.py",
    f"{ROOT_DIR}main/model_building/scripts/adf_testing.py",
    f"{ROOT_DIR}main/model_building/scripts/hurst_exponent.py",
    f"{ROOT_DIR}main/model_building/scripts/half_life.py",
    f"{ROOT_DIR}main/model_building/scripts/modify_results_df.py",
    f"{ROOT_DIR}main/model_building/backtesting/backtest_execution.py",
    f"{ROOT_DIR}main/model_building/backtesting_analysis/performance_measures.py",
]


def run_script(script_path):
    process = subprocess.Popen(
        [sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(
            f"Script {os.path.basename(script_path)} failed with error: {stderr.decode()}"
        )
    else:
        print(
            f"Script {os.path.basename(script_path)} executed successfully. Output: {stdout.decode()}"
        )


if __name__ == "__main__":
    for script in scripts:
        run_script(script)
    logging.info("Finished wrapper script")
