# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import os
import yaml
import argparse
import logging
import pandas as pd
from datetime import datetime
from DELPHI_utils_V3_static import DELPHIBacktest

## Initializing Global Variables ##########################################################################
with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
parser = argparse.ArgumentParser()
parser.add_argument(
    '--user', '-u', type=str, required=True,
    choices=["omar", "hamza", "michael", "michael2", "ali", "mohammad", "server", "saksham"],
    help="Who is the user running? User needs to be referenced in config.yml for the filepaths (e.g. hamza, michael): "
)
parser.add_argument(
    '--prediction_date', '-pd', type=str, required=True,
    help="What prediction date would you like to backtest? Input format should be 'YYYY-MM-DD'"
)
parser.add_argument(
    '--n_days', '-n_days', type=int, required=True,
    help=(
        "How many days of prediction do you want to backtest? (e.g. if prediction date is 2020-08-01 and you want " +
        "to backtest until 2020-08-31 then input 30)"
    )
)
parser.add_argument(
    '--mse', '-mse', type=int, required=True, choices=[0, 1],
    help="Generate Mean Squared Error as well? Reply 0 or 1 (for False or True)."
)
parser.add_argument(
    '--mae', '-mae', type=int, required=True, choices=[0, 1],
    help="Generate Mean Absolute Error as well? Reply 0 or 1 (for False or True)."
)
arguments = parser.parse_args()
USER_RUNNING = arguments.user
PREDICTION_DATE = arguments.prediction_date
N_DAYS_BACKTEST = arguments.n_days
GET_MSE = bool(arguments.mse)
GET_MAE = bool(arguments.mae)
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
if not os.path.exists(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/"):
    os.mkdir(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/")

logger_filename_date = "".join(
    (str(datetime.now().date()) + f"_{datetime.now().hour}H{datetime.now().minute}M").split("-")
)
logger_filename = (
        CONFIG_FILEPATHS["logs"][USER_RUNNING] +
        f"backtest/{logger_filename_date}_delphi_backtest_prediction_date" +
        f"_{PREDICTION_DATE}_n_days_{N_DAYS_BACKTEST}.log"
)
logging.basicConfig(
    filename=logger_filename,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%m-%d-%Y %I:%M:%S %p",
)
logger = logging.getLogger("BacktestLogger")

if __name__ == "__main__":
    assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
    try:
        PREDICTION_DATE_DATETIME = pd.to_datetime(PREDICTION_DATE, format="%Y-%M-%d")
    except ValueError:
        raise ValueError(f"Wrong prediction date format, should be 'YYYY-MM-DD', got {PREDICTION_DATE}")

    logger.info(
        f"Starting backtest on {N_DAYS_BACKTEST} days on prediction date {PREDICTION_DATE}. MSE flag is " +
        f"{GET_MSE} and MAE flag is {GET_MAE}"
    )
    backtest_instance = DELPHIBacktest(
        path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
        prediction_date=PREDICTION_DATE,
        n_days_backtest=N_DAYS_BACKTEST,
        get_mae=GET_MAE,
        get_mse=GET_MSE,
        logger=logger,
    )
    df_historical = backtest_instance.get_historical_data_df()
    df_prediction = backtest_instance.get_prediction_data()
    is_backtest_feasible = backtest_instance.get_feasibility_flag(
        df_historical=df_historical,
        df_prediction=df_prediction
    )
    df_backtest = df_prediction.merge(
        df_historical,
        on=["Country", "Province", "Day"],
    )
    assert df_backtest.Day.min() == PREDICTION_DATE,\
        f"Minimum date in backtest df is {df_backtest.Day.min()} and different from prediction date {PREDICTION_DATE}"
    df_backtest["tuple_complete"] = list(zip(df_backtest.Continent, df_backtest.Country, df_backtest.Province))
    dict_df_backtest_metrics = backtest_instance.generate_empty_metrics_dict()
    for tuple_area in df_backtest.tuple_complete.unique():
        logger.info(f"Computing backtest metrics for {tuple_area}")
        dict_df_backtest_metrics = backtest_instance.get_backtest_metrics_area(
            df_backtest=df_backtest, tuple_area=tuple_area, dict_df_backtest_metrics=dict_df_backtest_metrics,
        )

    logger.info("Finished backtesting, saving file into data_sandbox/backtest_outputs")
    df_backtest_metrics = pd.DataFrame(dict_df_backtest_metrics).round(3)
    df_backtest_metrics.to_csv(
        PATH_TO_DATA_SANDBOX + f"backtest_outputs/backtest_pd_{PREDICTION_DATE}_n_days_{N_DAYS_BACKTEST}.csv", index=False
    )
