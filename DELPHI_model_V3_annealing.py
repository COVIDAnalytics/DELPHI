# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
import multiprocessing as mp
import time
from functools import partial
from tqdm import tqdm_notebook as tqdm
from scipy.optimize import dual_annealing
from DELPHI_utils_V3_annealing import (
    DELPHIDataCreator, DELPHIAggregations, DELPHIDataSaver, get_initial_conditions, mape
)
from DELPHI_params_V3 import (
    get_default_parameter_list_and_bounds,
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
    validcases_threshold,
    IncubeD,
    RecoverID,
    RecoverHD,
    DetectD,
    VentilatedD,
    default_maxT,
    p_v,
    p_d,
    p_h
)
from DELPHI_utils_V3_dynamic import (get_bounds_params_from_pastparams)
from DELPHI_model_secondwave_with_policies import run_model_secondwave_with_policies
import os, sys
import yaml


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]

def solve_and_predict_area(
        tuple_area_: tuple, yesterday_: str, allowed_deviation_: float, pastparameters_: pd.DataFrame,
):
    time_entering = time.time()
    continent, country, province = tuple_area_
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
            return None

        print(country + ", " + province + ", " + totalcases.date.iloc[-1])
        if pastparameters_ is not None:
            parameter_list_total = pastparameters_[
                (pastparameters_.Country == country) &
                (pastparameters_.Province == province)
                ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                parameter_list = parameter_list_line[5:]
                date_day_since100 = pd.to_datetime(parameter_list_line[3])
                validcases = totalcases[
                    (totalcases.day_since100 >= 0) &
                    (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
                    ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                bounds_params = get_bounds_params_from_pastparams(
                    optimizer='annealing',
                    parameter_list=parameter_list,
                    dict_default_reinit_parameters=dict_default_reinit_parameters,
                    percentage_drift_lower_bound=percentage_drift_lower_bound,
                    default_lower_bound=default_lower_bound,
                    dict_default_reinit_lower_bounds=dict_default_reinit_lower_bounds,
                    percentage_drift_upper_bound=percentage_drift_upper_bound,
                    default_upper_bound=default_upper_bound,
                    dict_default_reinit_upper_bounds=dict_default_reinit_upper_bounds,
                    percentage_drift_lower_bound_annealing=percentage_drift_lower_bound_annealing,
                    default_lower_bound_annealing=default_lower_bound_annealing,
                    percentage_drift_upper_bound_annealing=percentage_drift_upper_bound_annealing,
                    default_upper_bound_annealing=default_upper_bound_annealing,
                    default_lower_bound_jump=default_lower_bound_jump,
                    default_upper_bound_jump=default_upper_bound_jump,
                    default_lower_bound_std_normal=default_lower_bound_std_normal,
                    default_upper_bound_std_normal=default_upper_bound_std_normal,
                )
            else:
                # Otherwise use established lower/upper bounds
                date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
                validcases = totalcases[
                    (totalcases.day_since100 >= 0) &
                    (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
                    ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                parameter_list, bounds_params = get_default_parameter_list_and_bounds(validcases)
        else:
            # Otherwise use established lower/upper bounds
            date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
            validcases = totalcases[
                (totalcases.day_since100 >= 0) &
                (totalcases.date <= str((pd.to_datetime(yesterday_) + timedelta(days=1)).date()))
                ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            parameter_list, bounds_params = get_default_parameter_list_and_bounds(validcases)
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
            GLOBAL_PARAMS_FIXED = (
                N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v
            )

            def model_covid(
                    t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal
            ):
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
                gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1 +  jump * np.exp(-(t - t_jump)**2 /(2 * std_normal ** 2))
                # gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1 + jump * (np.arctan(t - t_jump) + np.pi / 2) * min(1, 2 / np.pi * np.arctan( - (t - t_jump)/ 20 * r_decay) + 1)

                # if t < t_jump:
                #     gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                # else:
                #     gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1 + jump
                p_dth_mod = (2 / np.pi) * (p_dth - 0.01) * (np.arctan(- t / 20 * r_dthdecay) + np.pi / 2) + 0.01
                assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
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
                    dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                    dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
                ]

            def residuals_totalcases(params):
                """
                Wanted to start with solve_ivp because figures will be faster to debug
                params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
                """
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = params
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
                    params_fitted=params,
                    global_params_fixed=GLOBAL_PARAMS_FIXED
                )
                x_sol_total = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_cases[0], t_cases[-1]],
                    t_eval=t_cases,
                    args=tuple(params)
                )
                x_sol = x_sol_total.y
                # weights = list(range(1, len(fitcasesnd) + 1))
                weights = [(x/len(fitcasesnd))**2 for x in weights]
                # weights[-15:] =[x + 50 for x in weights[-15:]]
                if x_sol_total.status == 0:
                    residuals_value = sum(
                        np.multiply((x_sol[15, :] - fitcasesnd) ** 2, weights)
                        + balance * balance * np.multiply((x_sol[14, :] - fitcasesd) ** 2, weights)) + sum(
                        np.multiply((x_sol[15, 7:] - x_sol[15, :-7] - fitcasesnd[7:] + fitcasesnd[:-7]) ** 2, weights[7:])
                        + balance * balance * np.multiply((x_sol[14, 7:] - x_sol[14, :-7] - fitcasesd[7:] + fitcasesd[:-7]) ** 2, weights[7:])
                        )
                    return residuals_value
                else:
                    residuals_value = 1e12
                    return residuals_value


            # def last_point(params):
            #     alpha, days, r_s, r_dth, p_dth, k1, k2 = params
            #     params = max(alpha, 0), days, max(r_s, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0)
            #     x_0_cases = get_initial_conditions(
            #         params_fitted=params,
            #         global_params_fixed=GLOBAL_PARAMS_FIXED
            #     )
            #     x_sol = solve_ivp(
            #         fun=model_covid,
            #         y0=x_0_cases,
            #         t_span=[t_cases[0], t_cases[-1]],
            #         t_eval=t_cases,
            #         args=tuple(params),
            #     ).y
            #     return x_sol[14:16,-1]
            # nlcons = NonlinearConstraint(last_point,
            #                              [fitcasesd[-1] * (1 - allowed_deviation_), fitcasesnd[-1] * (1 - allowed_deviation_) ],
            #                              [fitcasesd[-1] * (1 + allowed_deviation_), fitcasesnd[-1] * (1 + allowed_deviation_) ])
            output = dual_annealing(residuals_totalcases, x0 = parameter_list, bounds = bounds_params)
#            output = minimize(
#                residuals_totalcases,
#                parameter_list,
#                method=dual_annealing,  # Can't use Nelder-Mead if I want to put bounds on the params
#                bounds=bounds_params,
#                options={'maxiter': max_iter}
#            )
            best_params = output.x
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
                    global_params_fixed=GLOBAL_PARAMS_FIXED
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
                x_sol_final=x_sol_final, date_day_since100=date_day_since100, best_params=best_params,
                continent=continent, country=country, province=province, testing_data_included=False
            )
            # Creating the parameters dataset for this (Continent, Country, Province)
            mape_data = (
                                mape(fitcasesnd, x_sol_final[15, :len(fitcasesnd)]) +
                                mape(fitcasesd, x_sol_final[14, :len(fitcasesd)])
                        ) / 2
            if len(fitcasesnd) > 15:
                mape_data_2 = (
                                      mape(fitcasesnd[-15:], x_sol_final[15, len(fitcasesnd) - 15:len(fitcasesnd)]) +
                                      mape(fitcasesd[-15:], x_sol_final[14, len(fitcasesnd) - 15:len(fitcasesd)])
                              ) / 2
                print(f"In-Sample MAPE Last 15 Days {country, province}: {round(mape_data_2, 3)} %")
            df_parameters_cont_country_prov = data_creator.create_dataset_parameters(mape_data)
            # Creating the datasets for predictions of this (Continent, Country, Province)
            df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                data_creator.create_datasets_predictions()
            )
#            df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
#                data_creator.create_datasets_with_confidence_intervals(fitcasesnd, fitcasesd)
#            )
            print(
                f"Finished predicting for Continent={continent}, Country={country} and Province={province} in " +
                f"{round(time.time() - time_entering, 2)} seconds"
            )
            return (
                df_parameters_cont_country_prov, df_predictions_since_today_cont_country_prov,
                df_predictions_since_100_cont_country_prov, output
            )
        else:  # len(validcases) <= 7
            print(f"Not enough historical data (less than a week)" +
                  f"for Continent={continent}, Country={country} and Province={province}")
            return None
    else:  # file for that tuple (country, province) doesn't exist in processed files
        return None

if __name__ == "__main__":
    arg = sys.argv[1:len(sys.argv)]
    RUNNING_FOR_JJ = arg[0] if len(arg) > 0 else ""
    USER = os.getenv('USER')
    USER_RUNNING = "ali" if USER == 'ali' else 'server'
    upload_to_s3 = True
    current_time = datetime.now()
    time_beginning = time.time()
    yesterday = "".join(str(current_time.date() - timedelta(days=1)).split("-"))
    PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
    PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
    popcountries = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
    )
    try:
        pastparameters = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_V2_{yesterday}.csv"
        )
        print(f"Parameters_Global_V2_{yesterday}.csv used")
    except:
        pastparameters = None
    # Initalizing lists of the different dataframes that will be concatenated in the end
    list_df_global_predictions_since_today = []
    list_df_global_predictions_since_100_cases = []
    list_df_global_parameters = []
    obj_value = 0
    allowed_deviation = 0.02
    solve_and_predict_area_partial = partial(
        solve_and_predict_area, yesterday_=yesterday, pastparameters_=pastparameters,
        allowed_deviation_=allowed_deviation
    )
    n_cpu = 90
    popcountries["tuple_area"] = list(zip(popcountries.Continent, popcountries.Country, popcountries.Province))
    list_tuples = popcountries.tuple_area.tolist()
    list_tuples = [x for x in list_tuples if x[2] == 'None' or x[1] == 'US']

    with mp.Pool(n_cpu) as pool:
        for result_area in tqdm(
                pool.map_async(
                    solve_and_predict_area_partial, list_tuples,
                ).get(), total=len(list_tuples)
        ):
            if result_area is not None:
                (
                    df_parameters_cont_country_prov, df_predictions_since_today_cont_country_prov,
                    df_predictions_since_100_cont_country_prov, output
                ) = result_area
                obj_value = obj_value + output.fun
                # Then we add it to the list of df to be concatenated to update the tracking df
                list_df_global_parameters.append(df_parameters_cont_country_prov)
                list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
                list_df_global_predictions_since_100_cases.append(df_predictions_since_100_cont_country_prov)
            else:
                continue
        print("Finished the Multiprocessing for all areas")
        pool.close()
        pool.join()

    # Appending parameters, aggregations per country, per continent, and for the world
    # for predictions today & since 100
    today_date_str = "".join(str(current_time.date()).split("-"))
    if len(list_df_global_parameters) > 0:
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
            path_to_website_predicted=PATH_TO_FOLDER_DANGER_MAP,
            df_global_parameters=df_global_parameters,
            df_global_predictions_since_today=df_global_predictions_since_today,
            df_global_predictions_since_100_cases=df_global_predictions_since_100_cases,
        )
        delphi_data_saver.save_all_datasets(save_since_100_cases=True, website=False)
        print(f"Exported all 3 datasets to website & danger_map repositories, "+
              f"total runtime was {round((time.time() - time_beginning)/60, 2)} minutes")
    else:
        print("No parameters created...")
    run_model_secondwave_with_policies(PATH_TO_FOLDER_DANGER_MAP, PATH_TO_DATA_SANDBOX,
                                       current_time, list_tuples, upload_to_s3)
