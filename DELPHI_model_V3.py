# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
from tqdm import tqdm_notebook as tqdm
from scipy.optimize import dual_annealing
from DELPHI_utils_V3 import (
    DELPHIDataCreator,
    DELPHIAggregations,
    DELPHIDataSaver,
    get_initial_conditions,
    compute_mape,
)
from DELPHI_params_V3 import (
    default_parameter_list,
    dict_default_reinit_parameters,
    dict_default_reinit_lower_bounds,
    dict_default_reinit_upper_bounds,
    default_upper_bound,
    default_lower_bound,
    percentage_drift_upper_bound,
    percentage_drift_lower_bound,
    percentage_drift_upper_bound_annealing,
    percentage_drift_lower_bound_annealing,
    default_upper_bound_annealing,
    default_lower_bound_annealing,
    default_lower_bound_jump,
    default_upper_bound_jump,
    default_lower_bound_std_normal,
    default_upper_bound_std_normal,
    default_bounds_params,
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
import os
import yaml
import logging
import time


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "hamza"
time_beginning = time.time()
yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
OPTIMIZER = input(
    "Which optimizer among 'tnc', 'trust-constr' or 'annealing' would you like to use ? "
    + "Note that 'tnc' and 'trust-constr' lead to local optima, while 'annealing' is a method for global optimization: "
)
assert OPTIMIZER in ["tnc", "trust-constr", "annealing"], "Wrong input value for optimizer"
logging.basicConfig(
    filename=f"./logs/delphi_model_V3_{yesterday}_{OPTIMIZER}.log",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%m-%d-%Y %I:%M:%S %p |",
)
logging.info(f"The chosen optimizer for this run was {OPTIMIZER}")
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["website"][USER_RUNNING]
popcountries = pd.read_csv(
    PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
)
popcountries["tuple_area"] = list(
    zip(popcountries.Continent, popcountries.Country, popcountries.Province)
)

# TODO: Docstrings for all functions
popcountries["tuple_area"] = list(zip(popcountries.Continent, popcountries.Country, popcountries.Province))
past_prediction_date = "".join(str(datetime.now().date() - timedelta(days=14)).split("-"))
def solve_and_predict_area(
    tuple_area_: tuple,
    yesterday_: str,
    pastparameters_: pd.DataFrame,
):
    """

    :param tuple_area_:
    :param yesterday_:
    :param pastparameters_:
    :return:
    """
    time_entering = time.time()
    continent, country, province = tuple_area_
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            logging.warning(
                f"Not enough cases for Continent={continent}, Country={country} and Province={province}"
            )
            return None

        if pastparameters_ is not None:
            parameter_list_total = pastparameters_[
                (pastparameters_.Country == country)
                & (pastparameters_.Province == province)
            ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                parameter_list = parameter_list_line[5:]
                if OPTIMIZER in ["tnc", "trust-constr"]:
                    # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = (
                        parameter_list
                    )
                    parameter_list = (
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
                    )
                    param_list_lower = [
                        x - max(percentage_drift_lower_bound * abs(x), default_lower_bound) for x in parameter_list
                    ]
                    (
                        alpha_lower, days_lower, r_s_lower, r_dth_lower, p_dth_lower, r_dthdecay_lower,
                        k1_lower, k2_lower, jump_lower, t_jump_lower, std_normal_lower
                    ) = param_list_lower
                    param_list_lower = [
                        max(alpha_lower, dict_default_reinit_lower_bounds["alpha"]),
                        days_lower,
                        max(r_s_lower, dict_default_reinit_lower_bounds["r_s"]),
                        max(min(r_dth_lower, 1), dict_default_reinit_lower_bounds["r_dth"]),
                        max(min(p_dth_lower, 1), dict_default_reinit_lower_bounds["p_dth"]),
                        max(r_dthdecay_lower, dict_default_reinit_lower_bounds["r_dthdecay"]),
                        max(k1_lower, dict_default_reinit_lower_bounds["k1"]),
                        max(k2_lower, dict_default_reinit_lower_bounds["k2"]),
                        max(jump_lower, dict_default_reinit_lower_bounds["jump"]),
                        max(t_jump_lower, dict_default_reinit_lower_bounds["t_jump"]),
                        max(std_normal_lower, dict_default_reinit_lower_bounds["std_normal"]),
                    ]
                    param_list_upper = [
                        x + max(percentage_drift_upper_bound * abs(x), default_upper_bound) for x in parameter_list
                    ]
                    (
                        alpha_upper, days_upper, r_s_upper, r_dth_upper, p_dth_upper, r_dthdecay_upper,
                        k1_upper, k2_upper, jump_upper, t_jump_upper, std_normal_upper
                    ) = param_list_upper
                    param_list_upper = [
                        max(alpha_upper, dict_default_reinit_upper_bounds["alpha"]),
                        days_upper,
                        max(r_s_upper, dict_default_reinit_upper_bounds["r_s"]),
                        max(min(r_dth_upper, 1), dict_default_reinit_upper_bounds["r_dth"]),
                        max(min(p_dth_upper, 1), dict_default_reinit_upper_bounds["p_dth"]),
                        max(r_dthdecay_upper, dict_default_reinit_upper_bounds["r_dthdecay"]),
                        max(k1_upper, dict_default_reinit_upper_bounds["k1"]),
                        max(k2_upper, dict_default_reinit_upper_bounds["k2"]),
                        max(jump_upper, dict_default_reinit_upper_bounds["jump"]),
                        max(t_jump_upper, dict_default_reinit_upper_bounds["t_jump"]),
                        max(std_normal_upper, dict_default_reinit_upper_bounds["std_normal"]),
                    ]
                else:  # Annealing procedure for global optimization
                    param_list_lower = [
                        x - max(percentage_drift_lower_bound_annealing * abs(x), default_lower_bound_annealing) for x in parameter_list
                    ]
                    param_list_upper = [
                        x + max(percentage_drift_upper_bound_annealing * abs(x), default_upper_bound_annealing) for x in parameter_list
                    ]
                    param_list_lower[8] = default_lower_bound_jump  # jump lower bound
                    param_list_upper[8] = default_upper_bound_jump  # jump upper bound
                    param_list_lower[10] = default_lower_bound_std_normal  # std_normal lower bound
                    param_list_upper[10] = default_upper_bound_std_normal  # std_normal upper bound

                bounds_params = [
                    (lower, upper)
                    for lower, upper in zip(param_list_lower, param_list_upper)
                ]
                date_day_since100 = pd.to_datetime(parameter_list_line[3])
                validcases = totalcases[
                    (totalcases.day_since100 >= 0)
                    & (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
                ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                bounds_params = tuple(bounds_params)
            else:
                # Otherwise use established lower/upper bounds
                parameter_list = default_parameter_list
                bounds_params = default_bounds_params
                date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
                validcases = totalcases[
                    (totalcases.day_since100 >= 0) &
                    (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
                ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        else:
            # Otherwise use established lower/upper bounds
            parameter_list = default_parameter_list
            bounds_params = default_bounds_params
            date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
            validcases = totalcases[
                (totalcases.day_since100 >= 0) &
                (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
            ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        # Now we start the modeling part:
        if len(validcases) > validcases_threshold:
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
            ].pop2016.iloc[-1]
            # We do not scale
            N = PopulationT
            PopulationI = validcases.loc[0, "case_cnt"]
            PopulationR = validcases.loc[0, "death_cnt"] * 5
            PopulationD = validcases.loc[0, "death_cnt"]
            PopulationCI = PopulationI - PopulationD - PopulationR
            """
            Fixed Parameters based on meta-analysis:
            p_h: Hospitalization Percentage
            RecoverHD: Average Days till Recovery
            VentilationD: Number of Days on Ventilation for Ventilated Patients
            maxT: Maximum # of Days Modeled
            p_d: Percentage of True Cases Detected
            p_v: Percentage of Hospitalized Patients Ventilated,
            balance: Ratio of Fitting between cases and deaths
            """
            # Currently fit on alpha, a and b, r_dth,
            # & initial condition of exposed state and infected state
            # Maximum timespan of prediction, defaulted to go to 15/06/2020
            maxT = (default_maxT - date_day_since100).days + 1
            """ Fit on Total Cases """
            t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
            validcases_nondeath = validcases["case_cnt"].tolist()
            validcases_death = validcases["death_cnt"].tolist()
            balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
            fitcasesnd = validcases_nondeath
            fitcasesd = validcases_death
            GLOBAL_PARAMS_FIXED = (N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v)

            def model_covid(
                t,
                x,
                alpha,
                days,
                r_s,
                r_dth,
                p_dth,
                r_dthdecay,
                k1,
                k2,
                jump,
                t_jump,
                std_normal,
            ):
                """

                :param t: time step
                :param x: set of all the states in the model (here, 16 of them)
                :param alpha:
                :param days:
                :param r_s:
                :param r_dth:
                :param p_dth:
                :param r_dthdecay:
                :param k1:
                :param k2:
                :param jump:
                :param t_jump:
                :param std_normal:
                :return:
                """
                """
                SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
                alpha: Infection rate
                days: Median day of action
                r_s: Median rate of action
                p_dth: Mortality rate
                k1: Internal parameter 1
                k2: Internal parameter 2
                y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
                7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
                """
                r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
                r_d = np.log(2) / DetectD  # Rate of detection
                r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
                r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
                r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
                gamma_t = (
                    (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s)
                    + 1
                    + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
                )
                p_dth_mod = (2 / np.pi) * (p_dth - 0.01) * (
                    np.arctan(-t / 20 * r_dthdecay) + np.pi / 2
                ) + 0.01
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

            def residuals_totalcases(params):
                """
                Wanted to start with solve_ivp because figures will be faster to debug
                params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
                """
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = (
                    params
                )
                # Force params values to stay in a certain range during the optimization process with re-initializations
                params = (
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
                )
                x_0_cases = get_initial_conditions(
                    params_fitted=params, global_params_fixed=GLOBAL_PARAMS_FIXED
                )
                x_sol = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_cases[0], t_cases[-1]],
                    t_eval=t_cases,
                    args=tuple(params),
                ).y
                weights = list(range(1, len(fitcasesnd) + 1))
                # weights[-15:] =[x + 50 for x in weights[-15:]]
                if OPTIMIZER in ["tnc", "trust-constr"]:
                    residuals_value = sum(
                        np.multiply((x_sol[15, :] - fitcasesnd) ** 2, weights)
                        + balance
                        * balance
                        * np.multiply((x_sol[14, :] - fitcasesd) ** 2, weights)
                    )
                else:
                    residuals_value = sum(
                        np.multiply((x_sol[15, :] - fitcasesnd) ** 2, weights)
                        + balance
                        * balance
                        * np.multiply((x_sol[14, :] - fitcasesd) ** 2, weights)
                    ) + sum(
                        np.multiply(
                            (x_sol[15, 7:] - x_sol[15, :-7] - fitcasesnd[7:] + fitcasesnd[:-7]) ** 2,
                            weights[7:],
                        )
                        + balance * balance * np.multiply(
                            (x_sol[14, 7:] - x_sol[14, :-7] - fitcasesd[7:] + fitcasesd[:-7]) ** 2,
                            weights[7:],
                        )
                    )
                return residuals_value

            if OPTIMIZER in ["tnc", "trust-constr"]:
                output = minimize(
                    residuals_totalcases,
                    parameter_list,
                    method=OPTIMIZER,  # Can't use Nelder-Mead if I want to put bounds on the params
                    bounds=bounds_params,
                    options={"maxiter": max_iter},
                )
            else:
                output = dual_annealing(
                    residuals_totalcases, x0=parameter_list, bounds=bounds_params
                )
            best_params = output.x
            t_predictions = [i for i in range(maxT)]

            def solve_best_params_and_predict(optimal_params):
                # Variables Initialization for the ODE system
                if OPTIMIZER in ["tnc", "trust-constr"]:
                    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = (
                        optimal_params
                    )
                    # TODO Change default values with dicts
                    optimal_params = [
                        max(alpha, 0),
                        days,
                        max(r_s, 0),
                        max(min(r_dth, 1), 0.02),
                        max(min(p_dth, 1), 0),
                        max(r_dthdecay, 0),
                        max(k1, 0),
                        max(k2, 0),
                        max(jump, 0),
                        max(t_jump, 0),
                        max(std_normal, 1),
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

            x_sol_final = solve_best_params_and_predict(best_params)
            data_creator = DELPHIDataCreator(
                x_sol_final=x_sol_final,
                date_day_since100=date_day_since100,
                best_params=best_params,
                continent=continent,
                country=country,
                province=province,
                testing_data_included=False,
            )
            if len(fitcasesnd) > 15:
                mape_data = (
                                    compute_mape(
                        fitcasesnd[-15:],
                        x_sol_final[15, len(fitcasesnd) - 15 : len(fitcasesnd)],
                    )
                                    + compute_mape(
                        fitcasesd[-15:],
                        x_sol_final[14, len(fitcasesnd) - 15 : len(fitcasesd)],
                    )
                ) / 2
            else:
                mape_data = (
                                    compute_mape(fitcasesnd, x_sol_final[15, : len(fitcasesnd)])
                                    + compute_mape(fitcasesd, x_sol_final[14, : len(fitcasesd)])
                ) / 2

            logging.info(f"In-Sample MAPE Last 15 Days {country, province}: {round(mape_data, 3)} %")
            logging.debug(f"Parameters for {country, province}: {best_params}")
            df_parameters_cont_country_prov = data_creator.create_dataset_parameters(mape_data)
            # Creating the datasets for predictions of this (Continent, Country, Province)
            df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                data_creator.create_datasets_predictions()
            )
            logging.info(
                f"Finished predicting for Continent={continent}, Country={country} and Province={province} in "
                + f"{round(time.time() - time_entering, 2)} seconds"
            )
            logging.info("--------------------------------------------------------------------------------------------")
            return (
                df_parameters_cont_country_prov,
                df_predictions_since_today_cont_country_prov,
                df_predictions_since_100_cont_country_prov,
                output,
            )
        else:  # len(validcases) <= 7
            logging.warning(
                f"Not enough historical data (less than a week)"
                + f"for Continent={continent}, Country={country} and Province={province}"
            )
            return None
    else:  # file for that tuple (country, province) doesn't exist in processed files
        return None


if __name__ == "__main__":
    popcountries = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
    )
    try:
        pastparameters = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP
            + f"predicted/Parameters_Global_V2_{yesterday}.csv"
        )
    except:
        pastparameters = None

    # Initalizing lists of the different dataframes that will be concatenated in the end
    list_df_global_predictions_since_today = []
    list_df_global_predictions_since_100_cases = []
    list_df_global_parameters = []
    obj_value = 0
    solve_and_predict_area_partial = partial(
        solve_and_predict_area,
        yesterday_=yesterday,
        pastparameters_=pastparameters,
    )
    n_cpu = mp.cpu_count()
    logging.info(f"Number of CPUs found and used in this run: {n_cpu}")
    popcountries["tuple_area"] = list(
        zip(popcountries.Continent, popcountries.Country, popcountries.Province)
    )
    list_tuples = popcountries.tuple_area.tolist()
    logging.info(f"Number of areas to be fitted in this run: {len(list_tuples)}")
    with mp.Pool(n_cpu) as pool:
        for result_area in tqdm(
            pool.map_async(solve_and_predict_area_partial, list_tuples).get(),
            total=len(list_tuples),
        ):
            if result_area is not None:
                (
                    df_parameters_cont_country_prov,
                    df_predictions_since_today_cont_country_prov,
                    df_predictions_since_100_cont_country_prov,
                    output,
                ) = result_area
                obj_value = obj_value + output.fun
                # Then we add it to the list of df to be concatenated to update the tracking df
                list_df_global_parameters.append(df_parameters_cont_country_prov)
                list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
                list_df_global_predictions_since_100_cases.append(df_predictions_since_100_cont_country_prov)
            else:
                continue
        logging.info("Finished the Multiprocessing for all areas")
        pool.close()
        pool.join()

    # Appending parameters, aggregations per country, per continent, and for the world
    # for predictions today & since 100
    today_date_str = "".join(str(datetime.now().date()).split("-"))
    df_global_parameters = pd.concat(list_df_global_parameters).sort_values(
        ["Country", "Province"]
    ).reset_index(drop=True)
    df_global_predictions_since_today = pd.concat(list_df_global_predictions_since_today)
    df_global_predictions_since_today = DELPHIAggregations.append_all_aggregations(
        df_global_predictions_since_today
    )
    df_global_predictions_since_100_cases = pd.concat(list_df_global_predictions_since_100_cases)
    df_global_predictions_since_100_cases = DELPHIAggregations.append_all_aggregations(
        df_global_predictions_since_100_cases
    )
    delphi_data_saver = DELPHIDataSaver(
        path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
        path_to_website_predicted=PATH_TO_WEBSITE_PREDICTED,
        df_global_parameters=df_global_parameters,
        df_global_predictions_since_today=df_global_predictions_since_today,
        df_global_predictions_since_100_cases=df_global_predictions_since_100_cases,
    )
    delphi_data_saver.save_all_datasets(optimizer=OPTIMIZER, save_since_100_cases=False, website=True)
    logging.info(
        f"Exported all 3 datasets to website & danger_map repositories, "
        + f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes"
    )
