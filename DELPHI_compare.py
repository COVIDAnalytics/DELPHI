import os
import yaml
import logging
import time
import psutil
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from DELPHI_utils_V3_dynamic import DELPHIModelComparison


## Initializing Global Variables ##########################################################################
with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
time_beginning = time.time()
yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
yesterday_logs_filename = "".join(
    (str(datetime.now().date() - timedelta(days=1)) + f"_{datetime.now().hour}H{datetime.now().minute}M").split("-")
)
parser = argparse.ArgumentParser()
parser.add_argument(
    '--user', '-u', type=str, required=True,
    choices=["omar", "hamza", "michael", "michael2", "ali", "mohammad", "server", "saksham", "saksham2"],
    help="Who is the user running? User needs to be referenced in config.yml for the filepaths (e.g. hamza, michael): "
)
parser.add_argument(
    '--run_model', type=bool, required=True, choices=[True, False],
    help="Run model with Annealing and TNC or not? Choose False if model for given date has already been run and files are saved."
)
parser.add_argument(
    '--plots', '-p', type=bool, required=True, choices=[True, False],
    help="Save plots comparing predictions or not? Reply True or False"
)
parser.add_argument(
    '--verbose', '-v', type=bool, required=True, choices=[True, False],
    help="Print messages or not? Reply True or False"
)
arguments = parser.parse_args()
USER_RUNNING = arguments.user
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PLOT_OPTION = arguments.plots 
VERBOSE = arguments.verbose
TRAIN = arguments.run_model
#############################################################################################################

if __name__ == "__main__":
    assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
    if not os.path.exists(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_comparison/"):
        os.mkdir(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_comparison/")

    logger_filename = (
            CONFIG_FILEPATHS["logs"][USER_RUNNING] +
            f"model_comparison/delphi_model_V3_{yesterday_logs_filename}.log"
    )
    logging.basicConfig(
        filename=logger_filename,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S %p",
    )
    logging.info(
        f"The user is {USER_RUNNING}, Comparing Annealing and TNC performance."
    )
    today_date_str = "".join(str(datetime.now().date()).split("-"))

    ## Run annealing and tnc
    if TRAIN:
        os.system(
            f'python3 DELPHI_model_V3.py -u {USER_RUNNING} -o tnc -ci 0 -cm 0  -s100 1 -w 0'
        )
        os.system(
            f'python3 DELPHI_model_V3.py -u {USER_RUNNING} -o annealing -ci 0 -cm 0  -s100 1 -w 0'
        )

    ## Read parameter files
    global_parameters_tnc = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_V2_{today_date_str}.csv",
            index=False
    )
    global_parameters_annealing = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_V2_annealing_{today_date_str}.csv",
            index=False
    )
    global_parameters_best = pd.DataFrame(columns=global_parameters_tnc.columns)

    ## Compare metrics
    global_annealing_predictions_since_100days = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f'predicted/Global_V2_annealing_since100_{today_date_str}.csv'
    )
    total_tnc_predictions_since_100days = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f'predicted/Global_V2_since100_{today_date_str}.csv'
    )

    global_annealing_predictions_since_100days['Day'] = global_annealing_predictions_since_100days['Day'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    total_tnc_predictions_since_100days['Day'] = total_tnc_predictions_since_100days['Day'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    model_compare = DELPHIModelComparison(
        PATH_TO_FOLDER_DANGER_MAP,
        CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING],
        global_annealing_predictions_since_100days,
        total_tnc_predictions_since_100days 
    )

    comparison_results = {'region': [], 'annealing_selected': [], 'annealing_metric': [], 'tnc_metric': [], 'annealing_mape': []])
    for i in global_parameters_tnc.shape[0]:
        tnc_params = global_parameters_tnc.iloc[i]
        annealing_params = global_parameters_annealing.iloc[i]
        continent = tnc_params.Continent
        country = tnc_params.Country
        province = tnc_params.Province
        an_select, an_metric, tnc_metric, an_mape = model_compare.compare_metric((continent, country, province), plot=PLOT_OPTION, verbose=VERBOSE)
        
        comparison_results['region'].append((continent, country, province))
        comparison_results['annealing_selected'].append(an_select)
        comparison_results['annealing_metric'].append(an_metric)
        comparison_results['tnc_metric'].append(tnc_metric)
        comparison_results['annealing_mape'].append(an_mape)
        
        if not region_result[0]:
            logging.warning(f'Annealing performs worse in {country} - {province}')
            best_params = global_parameters_tnc.query('Continent == @continent').query('Country == @country').query('Province == @state')
            global_parameters_best = global_parameters_best.append(best_params.iloc[0])
        else:
            best_params = global_parameters_annealing.query('Continent == @continent').query('Country == @country').query('Province == @state')
            global_parameters_best = global_parameters_best.append(best_params.iloc[0])


    comparison_df = pd.DataFrame.from_dict(comparison_results)
    annealing_count = np.sum(comparison_df['annealing_selected'])

    model_comparison_df.to_csv(
        CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING] + f'comparison/model_comparison_{today_date_str}.csv',
        index=False
    )

    logging.info(
    f"Checked Annealing v/s TNC. Annealing performs better {annealing_count}/{model_comparison_df.shape[0]} \n"
    + f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes"
    )
    

