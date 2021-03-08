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
from DELPHI_utils_V4_dynamic import DELPHIModelComparison


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
    '--run_model', '-r', type=int, required=True, choices=[0, 1],
    help="Run model with Annealing and TNC or not? Choose 0 if model for given date has already been run and files are saved."
)
parser.add_argument(
    '--plots', '-p', type=int, required=True, choices=[0, 1],
    help="Save plots comparing predictions or not? Enter 0/1"
)
parser.add_argument(
    '--user', '-u', type=str, required=True,
    choices=["omar", "hamza", "michael", "michael2", "ali", "mohammad", "server", "saksham", "saksham2"],
    help="Who is the user running? User needs to be referenced in config.yml for the filepaths (e.g. hamza, michael): "
)

arguments = parser.parse_args()
USER_RUNNING = arguments.user
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PLOT_OPTION = arguments.plots 
RUN_MODEL = arguments.run_model

#############################################################################################################

if __name__ == "__main__":
    assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
    if not os.path.exists(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_comparison/"):
        os.mkdir(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_comparison/")
    
    if not os.path.exists(CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING] + 'comparison/'):
        os.mkdir(CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING] + 'comparison/')

    logger_filename = (
            CONFIG_FILEPATHS["logs"][USER_RUNNING] +
            f"model_comparison/delphi_model_V4_{yesterday_logs_filename}.log"
    )
    logging.basicConfig(
        filename=logger_filename,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S %p",
    )
    logger = logging.getLogger("CompareLogger")
    logger.info(
        f"The user is {USER_RUNNING}, Comparing Annealing and TNC performance."
    )

    ## Run annealing and tnc
    if RUN_MODEL:
        logger.info('Running TNC and Annealing')
        os.system(
            f'python3 DELPHI_model_V4.py -rc tnc-run-config.yml'
        )
        os.system(
            f'python3 DELPHI_model_V4.py -rc annealing-run-config.yml'
        )

    today_date_str = "".join(str(datetime.now().date()).split("-"))
    #today_date_str = '20201004'
    ## Read parameter files
    global_parameters_tnc = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_V4_{today_date_str}.csv"
    )
    global_parameters_annealing = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_V4_annealing_{today_date_str}.csv"
    )
    global_parameters_best = pd.DataFrame(columns=global_parameters_tnc.columns)

    ## Compare metrics
    global_annealing_predictions_since_100days = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f'predicted/Global_V4_annealing_since100_{today_date_str}.csv'
    )
    total_tnc_predictions_since_100days = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f'predicted/Global_V4_since100_{today_date_str}.csv'
    )

    global_annealing_predictions_since_100days['Day'] = global_annealing_predictions_since_100days['Day'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    total_tnc_predictions_since_100days['Day'] = total_tnc_predictions_since_100days['Day'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    model_compare = DELPHIModelComparison(
        PATH_TO_FOLDER_DANGER_MAP,
        CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING],
        global_annealing_predictions_since_100days,
        total_tnc_predictions_since_100days,
        logger=logger
    )

    comparison_results = {'region': [], 'annealing_selected': [], 'annealing_metric': [], 'tnc_metric': [], 'annealing_max_ape': []}
    for i in range(global_parameters_tnc.shape[0]):
        tnc_params = global_parameters_tnc.iloc[i]
        continent = tnc_params.Continent
        country = tnc_params.Country
        province = tnc_params.Province
        annealing_params = global_parameters_annealing.query('Continent == @continent').query('Country == @country').query('Province == @province')

        if annealing_params.shape[0] == 0:
            logger.warning(f'Annealing parameters not present for {country} - {province}')
        else:
            annealing_select, annnealing_metric, tnc_metric, annealing_max_ape = model_compare.compare_metric((continent, country, province), plot=PLOT_OPTION)
            comparison_results['region'].append((continent, country, province))
            comparison_results['annealing_selected'].append(annealing_select)
            comparison_results['annealing_metric'].append(annnealing_metric)
            comparison_results['tnc_metric'].append(tnc_metric)
            comparison_results['annealing_max_ape'].append(annealing_max_ape)
            if (not annealing_select) or annealing_params["Jump Time"].item()>70:
                logger.warning(f'Annealing performs worse in {country} - {province}')
                global_parameters_best = global_parameters_best.append(tnc_params, ignore_index=True)
            else:
                global_parameters_best = global_parameters_best.append(annealing_params, ignore_index=True)

    global_parameters_best.to_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_V4_{today_date_str}.csv",
            index=False,
    )

    df_comparison = pd.DataFrame.from_dict(comparison_results)
    annealing_count = np.sum(df_comparison['annealing_selected'])
    df_comparison.to_csv(
        CONFIG_FILEPATHS['data_sandbox'][USER_RUNNING] + f'comparison/model_comparison_{today_date_str}.csv',
        index=False
    )

    logger.info(
        f"Checked Annealing v/s TNC. Annealing performs better {annealing_count}/{df_comparison.shape[0]} \n"
        + f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes"
    )
    


