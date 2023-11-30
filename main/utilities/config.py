import sys
from dotenv import load_dotenv
import os

load_dotenv()
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

from main.utilities.constants import (
    FIRST_BACKTEST_PARAMETERS,
)

# Import relevant backtest params when you run the whole strategy
PRESENT_BACKTEST_PARAMS = FIRST_BACKTEST_PARAMETERS
