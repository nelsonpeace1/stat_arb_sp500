import sys
from dotenv import load_dotenv
import os
import logging

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)
logging.basicConfig(level=logging.INFO)
import subprocess

# List of scripts to run with their full paths

scripts = [
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\cointegration_testing"
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\creating_spreads.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\adf_testing.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\hurst_exponent.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\half_life.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\scripts\modify_results_df.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\backtesting\backtest_execution.py",
    r"C:\Users\Nelson\Desktop\projects\trader_experimentation\main\model_building\backtesting_analysis\performance_measures.py",
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
