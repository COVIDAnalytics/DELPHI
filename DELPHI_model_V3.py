# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from DELPHI_utils import (
    DELPHIDataCreator, DELPHIAggregations, DELPHIDataSaver,
    get_initial_conditions, mape, preprocess_past_parameters_and_historical_data
)
import os

yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
# TODO: Find a way to make these paths automatic, whoever the user is...
PATH_TO_FOLDER_DANGER_MAP = (
    #"E:/Github/covid19orc/danger_map"
    "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
    "4. COVID19_Global/covid19orc/danger_map"
)
PATH_TO_WEBSITE_PREDICTED = (
    "E:/Github/website/data"
)
os.chdir(PATH_TO_FOLDER_DANGER_MAP)
popcountries = pd.read_csv(
    f"processed/Global/Population_Global.csv"
)
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
list_all_params = []
list_all_bounds = []  # Will have to be fed as: ((min_bound_1, max_bound_1), ..., (min_bound_K, max_bound_K))
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
for continent, country, province in zip(
        popcountries.Continent.tolist()[:30],
        popcountries.Country.tolist()[:30],
        popcountries.Province.tolist()[:30],
):
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if os.path.exists(f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        maxT, date_day_since100, validcases, parameter_list, bounds_params = (
            preprocess_past_parameters_and_historical_data(
                continent=continent, country=country, province=province,
                totalcases=totalcases, pastparameters=pastparameters
            )
        )
        # Only returns (None, None, None, None) if there are not enough cases in that (continent, country, province)
        if (maxT, date_day_since100, validcases, parameter_list, bounds_params) != (None, None, None, None, None):
            if len(validcases) > 7:
                list_tuples_with_data.append((continent, country, province))
                list_all_params.extend(parameter_list)
                list_all_bounds.extend(list(bounds_params))
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
                    "parameter_list": parameter_list,
                    "bounds_params": bounds_params,
                    "N": N,
                    "PopulationI": PopulationI,
                    "PopulationR": PopulationR,
                    "PopulationD": PopulationD,
                    "PopulationCI": PopulationCI,
                }
            else:
                print(f"Not enough historical data (less than a week)" +
                      f"for Continent={continent}, Country={country} and Province={province}")
                continue
    else:
        continue
print("Finished preprocessing all files, starting modeling V3")
# Modeling V3


def model_covid(
        t, x, _alpha, _days, _r_s, _r_dth, _p_dth, _k1, _k2
):
    """
    SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
    _alpha: Infection rate
    _days: Median day of action
    _r_s: Median rate of action
    _p_dth: Mortality rate
    _k1: Internal parameter 1
    _k2: Internal parameter 2
    y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
    7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
    """
    r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
    r_d = np.log(2) / DetectD  # Rate of detection
    r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
    r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
    r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
    p_d = dict_fixed_parameters["p_d"]
    p_h = dict_fixed_parameters["p_h"]
    p_v = dict_fixed_parameters["p_v"]
    gamma_t = (2 / np.pi) * np.arctan(-(t - _days) / 20 * _r_s) + 1
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


def residuals_totalcases(list_all_params):
    """
    Wanted to start with solve_ivp because figures will be faster to debug
    params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
    """
    residuals_value_total = 0
    start_idx_params = 0
    for i, (_continent, _country, _province) in enumerate(list_tuples_with_data):
        # Parameters retrieval for this tuple (continent, country, province)
        alpha, days, r_s, r_dth, p_dth, k1, k2 = list_all_params[start_idx_params: start_idx_params + 7]
        start_idx_params += 7
        dict_necessary_data_i = dict_necessary_data_per_tuple[(_continent, _country, _province)]
        params = max(alpha, 0), days, max(r_s, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0)
        GLOBAL_PARAMS_FIXED = (
            dict_necessary_data_i["N"], dict_necessary_data_i["PopulationCI"],
            dict_necessary_data_i["PopulationR"], dict_necessary_data_i["PopulationD"],
            dict_necessary_data_i["PopulationI"], dict_fixed_parameters["p_d"],
            dict_fixed_parameters["p_h"], dict_fixed_parameters["p_v"],
        )
        # Variables Initialization for the ODE system
        x_0_cases_i = get_initial_conditions(
            params_fitted=params,
            global_params_fixed=GLOBAL_PARAMS_FIXED
        )
        # Fitting Data and Fitting Timespan
        t_cases = (
                dict_necessary_data_i["validcases"]["day_since100"].tolist() -
                dict_necessary_data_i["validcases"].loc[0, "day_since100"]
        )
        validcases_nondeath = dict_necessary_data_i["validcases"]["case_cnt"].tolist()
        validcases_death = dict_necessary_data_i["validcases"]["death_cnt"].tolist()
        balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
        fitcasesnd = validcases_nondeath
        fitcasesd = validcases_death
        x_sol_i = solve_ivp(
            fun=model_covid,
            y0=x_0_cases_i,
            t_span=[t_cases[0], t_cases[-1]],
            t_eval=t_cases,
            args=tuple(params),
        ).y
        weights = list(range(1, len(fitcasesnd) + 1))
        residuals_value = sum(
            np.multiply((x_sol_i[15, :] - fitcasesnd) ** 2, weights)
            + balance * balance * np.multiply((x_sol_i[14, :] - fitcasesd) ** 2, weights)
        )
        residuals_value_total += residuals_value

    return residuals_value_total


time_before = datetime.now()
print("Gonna start minimizing at {time_before}")
best_params = minimize(
    residuals_totalcases,
    np.array(list_all_params),
    method='trust-constr',  # Can't use Nelder-Mead if I want to put bounds on the params
    bounds=tuple(list_all_bounds)
).x
print(best_params)
print(f"Time to minimize: {datetime.now() - time_before}")
print("Finished Minimizing")
#t_predictions = [i for i in range(maxT)]
