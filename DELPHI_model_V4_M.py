# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, dual_annealing, differential_evolution
from datetime import datetime, timedelta
from DELPHI_utils_M import (
    DELPHIDataCreator, DELPHIAggregations, DELPHIDataSaver, read_measures_oxford_data,
    get_initial_conditions_v3, mape, preprocess_past_parameters_and_historical_data_v3
)
import os
import time
from multiprocessing import Pool


def residuals_inner(sublist_params):
    alpha, r_dth, p_dth, k1, k2, b0, b1, b2, b3, b4, b5, b6, b7,dict_necessary_data_i,dict_fixed_parameters,measures_oxford_i, continent_k, country_k, province_k  = sublist_params
    params = (
        max(alpha, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0),
        b0, b1, b2, b3, b4, b5, b6, b7
    )
    GLOBAL_PARAMS_FIXED_k = (
        dict_necessary_data_i["N"], dict_necessary_data_i["PopulationCI"],
        dict_necessary_data_i["PopulationR"], dict_necessary_data_i["PopulationD"],
        dict_necessary_data_i["PopulationI"], dict_fixed_parameters["p_d"],
        dict_fixed_parameters["p_h"], dict_fixed_parameters["p_v"],
    )
    # Variables Initialization for the ODE system
    x_0_cases_i = get_initial_conditions_v3(
        params_fitted=params,
        global_params_fixed=GLOBAL_PARAMS_FIXED_k,
    )
    # Fitting Data and Fitting Timespan
    t_cases = (
            dict_necessary_data_i["validcases"]["day_since100"].tolist() -
            dict_necessary_data_i["validcases"].loc[0, "day_since100"]
    )
    balance_i = dict_necessary_data_i["balance"]
    fitcasesnd_i = dict_necessary_data_i["fitcasesnd"]
    fitcasesd_i = dict_necessary_data_i["fitcasesd"]
    #if len(measures_oxford_i) == 0:
    #    continue
    N = dict_necessary_data_i["N"]
    def model_covid(
            t, x, _alpha, _r_dth, _p_dth, _k1, _k2, _b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7
    ):
        """
        SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
        _alpha: Infection rate
        _p_dth: Mortality rate
        _k1: Internal parameter 1
        _k2: Internal parameter 2
        y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
        7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
        """
        r_i = dict_fixed_parameters["r_i"]  # Rate of infection leaving incubation phase
        r_d = dict_fixed_parameters["r_d"]  # Rate of detection
        # r_ri = dict_fixed_parameters["r_ri"]  # Rate of recovery not under infection
        # r_rh = dict_fixed_parameters["r_rh"]  # Rate of recovery under hospitalization
        # r_rv = dict_fixed_parameters["r_rv"]  # Rate of recovery under ventilation
        p_d = dict_fixed_parameters["p_d"]
        p_h = dict_fixed_parameters["p_h"]
        # p_v = dict_fixed_parameters["p_v"]
        # gamma_t = 1 / (1 + np.exp(
        #     -(_b0 + np.dot([_b1, _b2, _b3, _b4, _b5, _b6, _b7], measures_oxford_i.iloc[int(t), :].tolist()))
        # ))
        gamma_t = 1 / (1 + np.exp(-(
                _b0 + _b1 * measures_oxford_i["policy_1"][int(t)] + _b2 * measures_oxford_i["policy_2"][int(t)] +
                _b3 * measures_oxford_i["policy_3"][int(t)] + _b4 * measures_oxford_i["policy_4"][int(t)] +
                _b5 * measures_oxford_i["policy_5"][int(t)] + _b6 * measures_oxford_i["policy_6"][int(t)] +
                _b7 * measures_oxford_i["policy_7"][int(t)])
          ))
        assert len(x) == 7, f"Too many input variables, got {len(x)}, expected 7"
        S, E, I, DHD, DQD, DD, DT= x
        # Equations on main variables
        dSdt = -_alpha * gamma_t * S * I / N
        dEdt = _alpha * gamma_t * S * I / N - r_i * E
        dIdt = r_i * E - r_d * I
        # dARdt = r_d * (1 - _p_dth) * (1 - p_d) * I - r_ri * AR
        # dDHRdt = r_d * (1 - _p_dth) * p_d * p_h * I - r_rh * DHR
        # dDQRdt = r_d * (1 - _p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
        # dADdt = r_d * _p_dth * (1 - p_d) * I - _r_dth * AD
        dDHDdt = r_d * _p_dth * p_d * p_h * I - _r_dth * DHD
        dDQDdt = r_d * _p_dth * p_d * (1 - p_h) * I - _r_dth * DQD
        # dRdt = r_ri * (AR + DQR) + r_rh * DHR
        # dDdt = _r_dth * (AD + DQD + DHD)
        # Helper states (usually important for some kind of output)
        # dTHdt = r_d * p_d * p_h * I
        # dDVRdt = r_d * (1 - _p_dth) * p_d * p_h * p_v * I - r_rv * DVR
        # dDVDdt = r_d * _p_dth * p_d * p_h * p_v * I - _r_dth * DVD
        dDDdt = _r_dth * (DHD + DQD)
        dDTdt = r_d * p_d * I
        return [
            dSdt, dEdt, dIdt, dDHDdt, dDQDdt, dDDdt, dDTdt
        ]
    x_sol_i = solve_ivp(
        fun=model_covid,
        y0=x_0_cases_i,
        t_span=[t_cases[0], t_cases[-1]],
        t_eval=t_cases,
        args=tuple(params),
    ).y
    weights_i = list(range(1, len(fitcasesnd_i) + 1))
    residuals_value_i = sum(
        np.multiply((x_sol_i[6, :] - fitcasesnd_i) ** 2, weights_i)
        + balance_i * balance_i * np.multiply((x_sol_i[5, :] - fitcasesd_i) ** 2, weights_i)
    )    
    return residuals_value_i
    

def residuals_totalcases(list_all_params):
    """
    Wanted to start with solve_ivp because figures will be faster to debug
    params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
    """
    sublist_params_total = list()
    list_all_params_without_common = list(list_all_params[:-7])  # Doesn't include b_1,...,b_9 but includes b_0s for each region
    list_all_params_common = list(list_all_params[-7:])
    for k, (continent_k, country_k, province_k) in enumerate(list_tuples_with_data):
        # Parameters retrieval for this tuple (continent, country, province)
        sublist_params = list()
        sublist_params.extend(list_all_params_without_common[6*k: (6*k + 6)])
        sublist_params.extend(list_all_params_common)
        dict_necessary_data_i = dict_necessary_data_per_tuple[(continent_k, country_k, province_k)]
        sublist_params.append(dict_necessary_data_i)
        sublist_params.append(dict_fixed_parameters)
        sublist_params.append(dict_df_measures_oxford[(continent_k, country_k, province_k)])
        sublist_params.extend([continent_k, country_k, province_k])
        sublist_params_total.append(sublist_params)
    residuals_value_total = sum(pool.map(residuals_inner,sublist_params_total))
    return residuals_value_total





def solve_best_params_and_predict(list_all_optimal_params):
    start_idx_params = 0
    dict_all_solutions = {}
    list_all_params_without_common = list(list_all_optimal_params[:-7])  # Doesn't include b_1,...,b_9 but includes b_0s for each region
    list_all_params_common = list(list_all_optimal_params[-7:])
    for k, (continent_k, country_k, province_k) in enumerate(list_tuples_with_data):
        # Parameters retrieval for this tuple (continent, country, province)
        sublist_params = list_all_params_without_common[start_idx_params: start_idx_params + 6]
        sublist_params.extend(list_all_params_common)
        # sublist_params = np.array(sublist_params)
        alpha, r_dth, p_dth, k1, k2, b0, b1, b2, b3, b4, b5, b6, b7 = sublist_params
        start_idx_params += 6
        dict_necessary_data_i = dict_necessary_data_per_tuple[(continent_k, country_k, province_k)]
        optimal_params_i = (
            max(alpha, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0),
            b0, b1, b2, b3, b4, b5, b6, b7
        )
        GLOBAL_PARAMS_FIXED_i = (
            dict_necessary_data_i["N"], dict_necessary_data_i["PopulationCI"],
            dict_necessary_data_i["PopulationR"], dict_necessary_data_i["PopulationD"],
            dict_necessary_data_i["PopulationI"], dict_fixed_parameters["p_d"],
            dict_fixed_parameters["p_h"], dict_fixed_parameters["p_v"],
        )
        # Variables Initialization for the ODE system
        x_0_cases_i = get_initial_conditions_v3(
            params_fitted=optimal_params_i,
            global_params_fixed=GLOBAL_PARAMS_FIXED_i
        )
        t_predictions_i = [i for i in range(dict_necessary_data_i["maxT"])]
        measures_oxford_i = dict_df_measures_oxford[(continent_k, country_k, province_k)]
        #if len(measures_oxford_i) == 0:
        #    continue

        def model_covid(
                t, x, _alpha, _r_dth, _p_dth, _k1, _k2, _b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7
        ):
            """
            SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
            _alpha: Infection rate
            _p_dth: Mortality rate
            _k1: Internal parameter 1
            _k2: Internal parameter 2
            y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
            7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
            """
            r_i = dict_fixed_parameters["r_i"]  # Rate of infection leaving incubation phase
            r_d = dict_fixed_parameters["r_d"]  # Rate of detection
            r_ri = dict_fixed_parameters["r_ri"]  # Rate of recovery not under infection
            r_rh = dict_fixed_parameters["r_rh"]  # Rate of recovery under hospitalization
            r_rv = dict_fixed_parameters["r_rv"]  # Rate of recovery under ventilation
            p_d = dict_fixed_parameters["p_d"]
            p_h = dict_fixed_parameters["p_h"]
            p_v = dict_fixed_parameters["p_v"]
            gamma_t = 1 / (1 + np.exp(-(
                    _b0 + _b1 * measures_oxford_i["policy_1"][int(t)] + _b2 * measures_oxford_i["policy_2"][int(t)] +
                    _b3 * measures_oxford_i["policy_3"][int(t)] + _b4 * measures_oxford_i["policy_4"][int(t)] +
                    _b5 * measures_oxford_i["policy_5"][int(t)] + _b6 * measures_oxford_i["policy_6"][int(t)] +
                    _b7 * measures_oxford_i["policy_7"][int(t)])
            ))
            assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
            S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
            # Equations on main variables
            dSdt = -_alpha * gamma_t * S * I / N
            dEdt = _alpha * gamma_t * S * I / N - r_i * E
            dIdt = r_i * E - r_d * I
            dARdt = r_d * (1 - _p_dth) * (1 - p_d) * I - r_ri * AR
            dDHRdt = r_d * (1 - _p_dth) * p_d * p_h * I - r_rh * DHR
            dDQRdt = r_d * (1 - _p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
            dADdt = r_d * _p_dth * (1 - p_d) * I - _r_dth * AD
            dDHDdt = r_d * _p_dth * p_d * p_h * I - _r_dth * DHD
            dDQDdt = r_d * _p_dth * p_d * (1 - p_h) * I - _r_dth * DQD
            dRdt = r_ri * (AR + DQR) + r_rh * DHR
            dDdt = _r_dth * (AD + DQD + DHD)
            # Helper states (usually important for some kind of output)
            dTHdt = r_d * p_d * p_h * I
            dDVRdt = r_d * (1 - _p_dth) * p_d * p_h * p_v * I - r_rv * DVR
            dDVDdt = r_d * _p_dth * p_d * p_h * p_v * I - _r_dth * DVD
            dDDdt = _r_dth * (DHD + DQD)
            dDTdt = r_d * p_d * I
            return [
                dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
            ]
        x_sol_best_i = solve_ivp(
            fun=model_covid,
            y0=x_0_cases_i,
            t_span=[t_predictions_i[0], t_predictions_i[-1]],
            t_eval=t_predictions_i,
            args=tuple(optimal_params_i),
        ).y
        dict_all_solutions[(continent_k, country_k, province_k)] = x_sol_best_i

    return dict_all_solutions


if __name__ == '__main__':
    yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
    # TODO: Find a way to make these paths automatic, whoever the user is...
    PATH_TO_FOLDER_DANGER_MAP = (
        "E:/Github/covid19orc/danger_map"
        # "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
        # "4. COVID19_Global/covid19orc/danger_map"
    )
    PATH_TO_WEBSITE_PREDICTED = (
        "E:/Github/website/data"
    )
    os.chdir(PATH_TO_FOLDER_DANGER_MAP)
    popcountries = pd.read_csv(
        f"processed/Global/Population_Global.csv"
    )
    measures_oxford = read_measures_oxford_data()
    dict_df_measures_oxford = {}
    # TODO: Uncomment these and delete the line with pastparameters=None once 1st run in Python is done!
    # try:
    #     pastparameters = pd.read_csv(
    #         f"predicted/Parameters_Global_{yesterday}.csv"
    #     )
    # except:
    pastparameters = None
    # Initalizing lists of the different dataframes that will be concatenated in the end
    list_df_global_predictions_since_today = []
    list_df_global_predictions_since_100_cases = []
    list_df_global_parameters = []
    list_tuples_with_data = []  # Tuples (continent, country, province) that have data AND more than 100 cases
    dict_necessary_data_per_tuple = {}  # key is (continent, country, province), value is another dict with necessary data
    list_all_params_fitted = []
    list_all_bounds_fitted = []  # Will have to be fed as: ((min_bound_1, max_bound_1), ..., (min_bound_K, max_bound_K))
    """
    Global Fixed Parameters based on meta-analysis:
    RecoverHD: Average Days till Recovery
    VentilationD: Number of Days on Ventilation for Ventilated Patients
    IncubeD: Number of Incubation Days
    RecoverID: Number of Recovery days after Incubation
    p_v: Percentage of Hospitalized Patients Ventilated
    p_d: Percentage of True Cases Detected
    p_h: Hospitalization Percentage
    balance: Ratio of Fitting between cases and deaths
    """
    RecoverHD = 15
    VentilatedD = 10
    IncubeD = 5
    RecoverID = 10
    DetectD = 2
    dict_fixed_parameters = {
        "r_i": np.log(2) / IncubeD,  # Rate of infection leaving incubation phase
        "r_d": np.log(2) / DetectD,  # Rate of detection
        "r_ri": np.log(2) / RecoverID,  # Rate of recovery not under infection
        "r_rh": np.log(2) / RecoverHD,  # Rate of recovery under hospitalization
        "r_rv": np.log(2) / VentilatedD,  # Rate of recovery under ventilation
        "p_v": 0.25,
        "p_d": 0.2,
        "p_h": 0.15
    }
    COUNTRIES_KEPT_CLUSTERING = [
        'France', 'Turkey', 'India', 'Israel', 'Singapore', 'Indonesia', 'Iran', 'Brazil', 'Switzerland',
        'Peru', 'Ireland', 'Austria', 'Netherlands', 'Sweden', 'Chile', 'Morocco', 'Moldova', 'Greece',
        'Italy', 'Japan', 'Korea, South', 'Vietnam', 'Mongolia', 'Russia', 'Romania', 'United Arab Emirates',
        'Serbia', 'South Africa', 'Spain', 'Germany', 'United Kingdom', 'Belgium', 'Portugal', 'Ecuador'
    ]
    for continent, country, province in zip(
            popcountries.Continent.tolist(),
            popcountries.Country.tolist(),
            popcountries.Province.tolist(),
    ):
        if country not in COUNTRIES_KEPT_CLUSTERING:
            continue  # TODO: Maybe not the right place to do this, but at least we're sure to only predict on those on which we have data
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")
        if os.path.exists(f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
            totalcases = pd.read_csv(
                f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
            )
            maxT, date_day_since100, validcases, balance, fitcasesnd, fitcasesd, parameter_list, bounds_params = (
                preprocess_past_parameters_and_historical_data_v3(
                    continent=continent, country=country, province=province,
                    totalcases=totalcases, pastparameters=pastparameters
                )
            )
            # Only returns (None, None, None, None,...) if there are not enough cases in that (continent, country, province)
            if (
                    (maxT, date_day_since100, validcases, balance, fitcasesnd, fitcasesd, parameter_list, bounds_params)
                    != (None, None, None, None, None, None, None, None)
            ):
                if len(validcases) > 7:
                    list_tuples_with_data.append((continent, country, province))
                    list_all_params_fitted.extend(parameter_list)
                    list_all_bounds_fitted.extend(list(bounds_params))
                    PopulationT = popcountries[
                        (popcountries.Country == country) & (popcountries.Province == province)
                        ].pop2016.item()
                    # We do not scale
                    N = PopulationT
                    PopulationI = validcases.loc[0, "case_cnt"]
                    PopulationR = validcases.loc[0, "death_cnt"] * 5
                    PopulationD = validcases.loc[0, "death_cnt"]
                    PopulationCI = PopulationI - PopulationD - PopulationR
                    dict_necessary_data_per_tuple[(continent, country, province)] = {
                        "maxT": maxT,
                        "date_day_since100": date_day_since100,
                        "validcases": validcases,
                        "balance": balance,
                        "fitcasesnd": fitcasesnd,
                        "fitcasesd": fitcasesd,
                        "parameter_list": parameter_list,
                        "bounds_params": bounds_params,
                        "N": N,
                        "PopulationI": PopulationI,
                        "PopulationR": PopulationR,
                        "PopulationD": PopulationD,
                        "PopulationCI": PopulationCI,
                    }
                    measures_oxford_i = measures_oxford[
                        (measures_oxford.CountryName == country) & (measures_oxford.Date >= date_day_since100)
                    ].drop(["CountryName", "Date"], axis=1).reset_index(drop=True)
                    measures_oxford_i.columns = [f"policy_{i+1}" for i in range(len(measures_oxford_i.columns))]
                    dict_df_measures_oxford[(continent, country, province)] = measures_oxford_i.to_dict()
                else:
                    print(f"Not enough historical data (less than a week)" +
                          f"for Continent={continent}, Country={country} and Province={province}")
                    continue
        else:
            continue
    
    # And now we add b_1, b_2, ..., b_9 since they are common to all regions in the world, so only need to appear once
    list_all_params_fitted.extend([1, 1, 1, 1, 1, 1, 1, ])
    list_all_bounds_fitted.extend([(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), ])
    print("Finished preprocessing all files, starting modeling V3")
    # Modeling V3
    time_before = datetime.now()
    print(f"Starting Minimization at {time_before}")
    # Number of threads to use
    pool = Pool(8)
    list_best_params = minimize(
        residuals_totalcases,
        np.array(list_all_params_fitted),
        method='trust-constr',  # Can't use Nelder-Mead if I want to put bounds on the params
        bounds=tuple(list_all_bounds_fitted),
        options={'maxiter':100,'verbose':3}
    ).x
    pool.close()
    # list_best_params = differential_evolution(
    #     residuals_totalcases,
    #     bounds=tuple(list_all_bounds_fitted),
    #     workers = -1
    # ).x
    print(list_best_params)
    print(f"Finished Minimizing; Time to minimize {datetime.now() - time_before}")

# dict_all_best_solutions = solve_best_params_and_predict(list_best_params)
# start_idx_best_params = 0
# for j, (continent, country, province) in enumerate(list_tuples_with_data):
#     x_sol_final_j = dict_all_best_solutions[(continent, country, province)]
#     best_params_j = list_best_params[start_idx_best_params: start_idx_best_params + 7]
#     dict_necessary_data_j_opt = dict_necessary_data_per_tuple[(continent, country, province)]
#     date_day_since100_j = dict_necessary_data_j_opt["date_day_since100"]
#     start_idx_best_params += 7
#     data_creator = DELPHIDataCreator(
#         x_sol_final=x_sol_final_j, date_day_since100=date_day_since100_j, best_params=best_params_j,
#         continent=continent, country=country, province=province,
#     )
#     # Creating the parameters dataset for this (Continent, Country, Province)
#     fitcasesnd_j = dict_necessary_data_j_opt["fitcasesnd"]
#     fitcasesd_j = dict_necessary_data_j_opt["fitcasesd"]
#     mape_data_j = (
#             mape(fitcasesnd_j, x_sol_final_j[15, :len(fitcasesnd_j)]) +
#             mape(fitcasesd_j, x_sol_final_j[14, :len(fitcasesd_j)])
#     ) / 2
#     df_parameters_cont_country_prov = data_creator.create_dataset_parameters(mape_data_j)
#     list_df_global_parameters.append(df_parameters_cont_country_prov)
#     # Creating the datasets for predictions of this (Continent, Country, Province)
#     df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
#         data_creator.create_datasets_predictions()
#     )
#     list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
#     list_df_global_predictions_since_100_cases.append(df_predictions_since_100_cont_country_prov)
#     print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")

# today_date_str = "".join(str(datetime.now().date()).split("-"))
# df_global_parameters = pd.concat(list_df_global_parameters)
# df_global_predictions_since_today = pd.concat(list_df_global_predictions_since_today)
# df_global_predictions_since_today = DELPHIAggregations.append_all_aggregations(
#     df_global_predictions_since_today
# )
# # TODO: Discuss with website team how to save this file to visualize it and compare with historical data
# df_global_predictions_since_100_cases = pd.concat(list_df_global_predictions_since_100_cases)
# df_global_predictions_since_100_cases = DELPHIAggregations.append_all_aggregations(
#     df_global_predictions_since_100_cases
# )
# delphi_data_saver = DELPHIDataSaver(
#     path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
#     path_to_website_predicted=PATH_TO_WEBSITE_PREDICTED,
#     df_global_parameters=df_global_parameters,
#     df_global_predictions_since_today=df_global_predictions_since_today,
#     df_global_predictions_since_100_cases=df_global_predictions_since_100_cases,
# )
# # TODO: Uncomment when finished
# # delphi_data_saver.save_all_datasets(save_since_100_cases=False)
# print("Exported all 3 datasets to website & danger_map repositories")
