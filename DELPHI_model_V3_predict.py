# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import os
import yaml
import logging
import time
import psutil
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from functools import partial
from tqdm import tqdm
from scipy.optimize import dual_annealing
from DELPHI_utils_V3_static import (
    DELPHIDataCreator, DELPHIAggregations, DELPHIDataSaver, get_initial_conditions,
    get_mape_data_fitting, create_fitting_data_from_validcases, get_residuals_value
)
from DELPHI_params_V3 import (
    default_parameter_list,
    dict_default_reinit_parameters,
    dict_default_reinit_lower_bounds,
    dict_default_reinit_upper_bounds,
    validcases_threshold,
    IncubeD,
    RecoverID,
    RecoverHD,
    DetectD,
    VentilatedD,
    default_maxT,
    p_v,
    p_d,
    p_h,
    max_iter,
)

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
    '--end_date', '-d', type=str, required=True,
    help="The date for which model states should be predicted, in yyyy-mm-dd format"
)

arguments = parser.parse_args()
end_date = arguments.end_date
USER_RUNNING = arguments.user
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["website"][USER_RUNNING]
past_prediction_date = "".join(str(datetime.now().date() - timedelta(days=14)).split("-"))
#############################################################################################################

def predict_area(
        tuple_area_: tuple,
        yesterday_: str,
        past_parameters_: pd.DataFrame,
        popcountries: pd.DataFrame,
        endT: str = None, # added to change prediction date
):
    """
    Parallelizeable function to predict and save all parameters and model states
    :param tuple_area_: tuple corresponding to (continent, country, province)
    :param yesterday_: string corresponding to the date from which the model will read the previous parameters. The
    format has to be 'YYYYMMDD'
    :param past_parameters_: Parameters from yesterday_ used as a starting point for the fitting process
    :startT: date from where the model will be started (format should be 'YYYYMMDD')
    :endT: date till predictions will be calculated and saved (format should be 'YYYYMMDD')
    :return: final_model_state: dict capturing the 16 delphi model states at date endT
    """
    time_entering = time.time()
    continent, country, province = tuple_area_
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    print(f"starting to predict for {continent}, {country}, {province}")
    if os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            logging.warning(
                f"Not enough cases (less than 100) for Continent={continent}, Country={country} and Province={province}"
            )
            return None

        if past_parameters_ is not None:
            parameter_list_total = past_parameters_[
                (past_parameters_.Country == country)
                & (past_parameters_.Province == province)
            ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                parameter_list = parameter_list_line[5:]
                date_day_since100 = pd.to_datetime(parameter_list_line[3])
            else:
                # Otherwise use established lower/upper bounds
                parameter_list = default_parameter_list
                date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
        else:
            # Otherwise use established lower/upper bounds
            parameter_list = default_parameter_list
            date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])

        if date_day_since100 > pd.to_datetime(endT):
            logging.warning(
                f"End date is less than date since 100 cases for, Continent={continent}, Country={country} and Province={province} in "
                + f"{round(time.time() - time_entering, 2)} seconds"
            )
            final_state_dict = {'S':None, 'E':None, 'I':None, 'UR':None, 'DHR':None, 'DQR':None, 'UD':None, 'DHD':None, 
                'DQD':None, 'R':None, 'D':None, 'TH':None, 'DVR':None, 'DVD':None, 'DD':None, 'DT':None,
                'continent': continent, 'country':country, 'province':province}
            return (final_state_dict)

        validcases = totalcases[
            (totalcases.day_since100 >= 0)
            & (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
        ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)

        # Now we start the modeling part:
        if len(validcases) <= validcases_threshold:
            logging.warning(
                f"Not enough historical data (less than a week)"
                + f"for Continent={continent}, Country={country} and Province={province}"
            )
            return None
        else:
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
            ].pop2016.iloc[-1]
            N = PopulationT
            PopulationI = validcases.loc[0, "case_cnt"]
            PopulationR = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
            PopulationD = validcases.loc[0, "death_cnt"]
            PopulationCI = PopulationI - PopulationD - PopulationR
            if PopulationCI <= 0:
                logging.error(f"PopulationCI value is negative ({PopulationCI}), need to check why")
                raise ValueError(f"PopulationCI value is negative ({PopulationCI}), need to check why")
            """
            Fixed Parameters based on meta-analysis:
            p_h: Hospitalization Percentage
            RecoverHD: Average Days until Recovery
            VentilationD: Number of Days on Ventilation for Ventilated Patients
            maxT: Maximum # of Days Modeled
            p_d: Percentage of True Cases Detected
            p_v: Percentage of Hospitalized Patients Ventilated,
            balance: Regularization coefficient between cases and deaths
            """
            endT = default_maxT if endT is None else pd.to_datetime(endT)
            maxT = (endT - date_day_since100).days + 1
            t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
            GLOBAL_PARAMS_FIXED = (N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v)

            def model_covid(
                t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal,
            ) -> list:
                """
                SEIR based model with 16 distinct states, taking into account undetected, deaths, hospitalized and
                recovered, and using an ArcTan government response curve, corrected with a Gaussian jump in case of
                a resurgence in cases
                :param t: time step
                :param x: set of all the states in the model (here, 16 of them)
                :param alpha: Infection rate
                :param days: Median day of action (used in the arctan governmental response)
                :param r_s: Median rate of action (used in the arctan governmental response)
                :param r_dth: Rate of death
                :param p_dth: Initial mortality percentage
                :param r_dthdecay: Rate of decay of mortality percentage
                :param k1: Internal parameter 1 (used for initial conditions)
                :param k2: Internal parameter 2 (used for initial conditions)
                :param jump: Amplitude of the Gaussian jump modeling the resurgence in cases
                :param t_jump: Time where the Gaussian jump will reach its maximum value
                :param std_normal: Standard Deviation of the Gaussian jump (~ time span of the resurgence in cases)
                :return: predictions for all 16 states, which are the following
                [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 UD, 7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
                """
                r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
                r_d = np.log(2) / DetectD  # Rate of detection
                r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
                r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
                r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
                gamma_t = (
                    (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                    + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
                )
                p_dth_mod = (2 / np.pi) * (p_dth - 0.001) * (np.arctan(-t / 20 * r_dthdecay) + np.pi / 2) + 0.001
                assert (
                    len(x) == 16
                ), f"Too many input variables, got {len(x)}, expected 16"
                S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
                # Equations on main variables
                dSdt = -alpha * gamma_t * S * I / N
                dEdt = alpha * gamma_t * S * I / N - r_i * E
                dIdt = r_i * E - r_d * I
                dARdt = r_d * (1 - p_dth_mod) * (1 - p_d) * I - r_ri * AR
                dDHRdt = r_d * (1 - p_dth_mod) * p_d * p_h * I - r_rh * DHR
                dDQRdt = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * I - r_ri * DQR
                dADdt = r_d * p_dth_mod * (1 - p_d) * I - r_dth * AD
                dDHDdt = r_d * p_dth_mod * p_d * p_h * I - r_dth * DHD
                dDQDdt = r_d * p_dth_mod * p_d * (1 - p_h) * I - r_dth * DQD
                dRdt = r_ri * (AR + DQR) + r_rh * DHR
                dDdt = r_dth * (AD + DQD + DHD)
                # Helper states (usually important for some kind of output)
                dTHdt = r_d * p_d * p_h * I
                dDVRdt = r_d * (1 - p_dth_mod) * p_d * p_h * p_v * I - r_rv * DVR
                dDVDdt = r_d * p_dth_mod * p_d * p_h * p_v * I - r_dth * DVD
                dDDdt = r_dth * (DHD + DQD)
                dDTdt = r_d * p_d * I
                return [
                    dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt,
                    dDQDdt, dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt,
                ]

            t_predictions = [i for i in range(maxT)]

            def solve_best_params_and_predict(optimal_params):
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = optimal_params
                optimal_params = [
                    max(alpha, dict_default_reinit_parameters["alpha"]),
                    days,
                    max(r_s, dict_default_reinit_parameters["r_s"]),
                    max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
                    max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
                    max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
                    max(k1, dict_default_reinit_parameters["k1"]),
                    max(k2, dict_default_reinit_parameters["k2"]),
                    max(jump, dict_default_reinit_parameters["jump"]),
                    max(t_jump, dict_default_reinit_parameters["t_jump"]),
                    max(std_normal, dict_default_reinit_parameters["std_normal"]),
                ]
                x_0_cases = get_initial_conditions(
                    params_fitted=optimal_params,
                    global_params_fixed=GLOBAL_PARAMS_FIXED,
                )
                x_sol_best = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_predictions[0], t_predictions[-1]],
                    t_eval=t_predictions,
                    args=tuple(optimal_params),
                ).y
                return x_sol_best

            x_final = solve_best_params_and_predict(parameter_list)
            [S, E, I, UR, DHR, DQR, UD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT] = x_final[:, -1]
            final_state_dict = {'S':S, 'E':E, 'I':I, 'UR':UR, 'DHR':DHR, 'DQR':DQR, 'UD':UD, 'DHD':DHD, 
                'DQD':DQD, 'R':R, 'D':D, 'TH':TH, 'DVR':DVR, 'DVD':DVD, 'DD':DD, 'DT':DT,
                'continent': continent, 'country':country, 'province':province}
            
            logging.info(
                f"Finished predicting for Continent={continent}, Country={country} and Province={province} in "
                + f"{round(time.time() - time_entering, 2)} seconds"
            )
            logging.info("--------------------------------------------------------------------------------------------")
            return (final_state_dict)
    else:  # file for that tuple (continent, country, province) doesn't exist in processed files
        logging.info(
            f"Skipping Continent={continent}, Country={country} and Province={province} as no processed file available"
        )
        return None


if __name__ == "__main__":
    assert USER_RUNNING in CONFIG_FILEPATHS["delphi_repo"].keys(), f"User {USER_RUNNING} not referenced in config.yml"
    if not os.path.exists(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/"):
        os.mkdir(CONFIG_FILEPATHS["logs"][USER_RUNNING] + "model_fitting/")

    logger_filename = (
            CONFIG_FILEPATHS["logs"][USER_RUNNING] +
            f"model_fitting/delphi_model_V3_predict_{yesterday_logs_filename}.log"
    )
    logging.basicConfig(
        filename=logger_filename,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S %p",
    )
    logging.info(
        f"The user is {USER_RUNNING}"
    )
    popcountries = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
    )
    popcountries["tuple_area"] = list(zip(popcountries.Continent, popcountries.Country, popcountries.Province))
    list_tuples = popcountries.tuple_area.tolist()
    # list_tuples = [x for x in list_tuples if x[1] == "US"]
    # list_tuples = [x for x in list_tuples if x[1] in ['France', 'Germany', 'Greece', 'Poland', 
    # 'Japan', 'South Africa', 'Singapore', 'Morocco', 'Iran', 'Russia', 'Brazil'] ]
    # list_tuples = [('Oceania' , 'Papua New Guinea' , 'None'), 
    #             ('Africa' , 'Lesotho' , 'None')]

    ### Compute the state of model till a given date ###
    try:
        past_parameters = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP
            + f"predicted/Parameters_Global_V2_{yesterday}.csv"
        )
    except:
        past_parameters = None
    
    predict_area_partial = partial(
        predict_area,
        yesterday_=yesterday,
        past_parameters_=past_parameters,
        popcountries=popcountries,
        endT=end_date
    )
    n_cpu = psutil.cpu_count(logical = False) - 2
    logging.info(f"Number of CPUs found and used in this run: {n_cpu}")
    logging.info(f"Number of areas to be predicted in this run: {len(list_tuples)}")
    list_predicted_state_dicts = []
    with mp.Pool(n_cpu) as pool:
        for result_area in tqdm(
            pool.map_async(predict_area_partial, list_tuples).get(),
            total=len(list_tuples),
        ):
            if result_area is not None:
                (model_state_dict) = result_area
                # Then we add it to the list of df to be concatenated to update the tracking df
                list_predicted_state_dicts.append(model_state_dict)
            else:
                continue
        logging.info("Finished the Multiprocessing for all areas")
        pool.close()
        pool.join()
    df_predicted_states = pd.DataFrame(list_predicted_state_dicts)
    df_predicted_states.to_csv(f'data_sandbox/predicted/raw_predictions/Predicted_model_state_V3_{end_date}.csv', index=False)
