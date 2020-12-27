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

## Initializing Global Variables ##########################################################################
today = "".join(str(datetime.now().date()).split("-"))
OPTIMIZER = "annealing"
#############################################################################################################
pop_connecticut_areas = {"Hartford": 396603, "HOCC": 150066, "Midstate": 
  121482, "Backus": 142920, "Windham": 46449, "CHH": 
  71460, "St.V's": 128628}
pop_connecticut = 3565000
IncubeD = 5
DetectD = 2
default_bounds_params = (
    (0.1, 2), (-200, 100), (1, 15), (0.02, 0.5), (0.01, 0.25),  (0.0001, 5), (0.0001, 5), (0, 3), (0, 250), (0.1, 100), (3,15), (0.01, 0.5)
)  
default_parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3, 0.1, 30, 30, 10, 0.2] # Default parameters for the solver

start_date = pd.to_datetime("2020-07-01")
default_maxT = pd.to_datetime("2021-03-01")
start_state_file = "E://Github//DELPHI//data_sandbox//predicted//raw_predictions//Predicted_model_state_V3_2020-07-01.csv"
ct_prediction_file = "E://Github//covid19orc//danger_map//predicted//Parameters_Global_CT_annealing_20201130.csv"
past_parameters_file = "HHC_Prediction_Parameters_20201202.csv"
past_parameters = pd.read_csv(past_parameters_file)
raw_data = pd.read_excel("MIT Census.xlsx",engine='openpyxl')
locations = raw_data.LOC_NAME.unique()
list_ct_output = []
list_ct_parameters_output = []
for location in locations:
    hospital_data = raw_data[(raw_data.LOC_NAME == location) & (raw_data.CENSUSDATE >= start_date)].sort_values("CENSUSDATE",axis = 0)
    hospitalized_data_fit = hospital_data.COVIDPOSITIVE.values
    recovered_data_fit =  np.cumsum(hospital_data.COVIDANDRECOVERED.values)
    deaths_data_fit = np.cumsum(hospital_data.COVIDANDDECEASED.values)
    t_cases = list(range(len(recovered_data_fit)))
    if "Hartford" in location:
        location_abbr = "Hartford"
    elif "Central" in location:
        location_abbr = "HOCC"
    elif "Midstate" in location:
        location_abbr = "Midstate"
    elif "Backus" in location:
        location_abbr = "Backus"
    elif "Windham" in location:
        location_abbr = "Windham"
    elif "Charlotte" in location:
        location_abbr = "CHH"
    elif "St. Vincent's" in location:
        location_abbr = "St.V's"  
    else:
        continue
    print(f"Predicting for {location_abbr}")
    try:
        past_parameters_line = list(past_parameters[past_parameters.Hospital == location_abbr].iloc[0][3:].values)
    except:
        past_parameters_line = None
    if past_parameters_line is None:
        parameter_list = default_parameter_list
        bounds_params = default_bounds_params
    else:
        parameter_list = past_parameters_line
        alpha, days, r_s, r_dth, p_dth, k1, k2, jump, t_jump, std_normal, t_rh, p_d = parameter_list
        parameter_list = [
            max(alpha, 0),
            days,
            max(r_s, 0),
            max(min(r_dth, 1), 0.01),
            max(min(p_dth, 1), 0),
            max(k1, 0),
            max(k2, 0),
            max(jump, 0),
            max(t_jump, 0),
            max(std_normal, 1),
            max(t_rh, 0),
            min(max(p_d, 0.01),1)
        ]
        param_list_lower = [
            x - max(0.8 * abs(x), 0.8) for x in
            parameter_list
        ]
        (
            alpha_lower, days_lower, r_s_lower, r_dth_lower, p_dth_lower,
            k1_lower, k2_lower, jump_lower, t_jump_lower, std_normal_lower, t_rh_lower, p_d_lower
        ) = param_list_lower
        param_list_lower = [
            max(alpha_lower, 0),
            days_lower,
            max(r_s_lower, 0),
            max(min(r_dth_lower, 1), 0.01),
            max(min(p_dth_lower, 1), 0),
            max(k1_lower, 0),
            max(k2_lower, 0),
            max(jump_lower, 0),
            max(t_jump_lower, 0),
            max(std_normal_lower, 1),
            max(t_rh_lower, 0),
            min(max(p_d_lower, 0.01),1)
        ]
        param_list_upper = [
            x + max(0.8 * abs(x), 0.8) for x in
            parameter_list
        ]
        (
            alpha_upper, days_upper, r_s_upper, r_dth_upper, p_dth_upper, 
            k1_upper, k2_upper, jump_upper, t_jump_upper, std_normal_upper, t_rh_upper, p_d_upper
        ) = param_list_upper
        param_list_upper = [
            max(alpha_upper, 0),
            days_upper,
            max(r_s_upper, 0),
            max(min(r_dth_upper, 1), 0.01),
            max(min(p_dth_upper, 1), 0),
            max(k1_upper, 0),
            max(k2_upper, 0),
            max(jump_upper, 0),
            max(t_jump_upper, 0),
            max(std_normal_upper, 1),
            max(t_rh_upper, 0),
            min(max(p_d_upper, 0.01),1)
        ]
        bounds_params = [(lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)]
    N = pop_connecticut_areas[location_abbr]
    starting_state = pd.read_csv(start_state_file)
    PopulationR = starting_state[(starting_state.country == "US") & (starting_state.province == "Connecticut")].R.values.item() * N/ pop_connecticut
    PopulationD = starting_state[(starting_state.country == "US") & (starting_state.province == "Connecticut")].D.values.item() * N/ pop_connecticut
    PopulationCI = max(hospitalized_data_fit[0],1)
    p_h = pd.read_csv(ct_prediction_file)["Percentage Hospitalized"].values.item()
            
    maxT = (default_maxT - start_date).days + 1
    balance_deaths = sum(hospitalized_data_fit) / sum(deaths_data_fit)
    balance_recovered = sum(hospitalized_data_fit) / sum(recovered_data_fit)
    def model_covid(
        t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, jump, t_jump, std_normal, t_rh, p_d
    ) -> list:
        r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
        r_d = np.log(2) / DetectD  # Rate of detection
        r_rh = np.log(2) / t_rh  # Rate of recovery under hospitalization
        gamma_t = (
            (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
            + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
        )
        assert (
            len(x) == 8
        ), f"Too many input variables, got {len(x)}, expected 16"
        S, E, I,  DHR, DHD, DH, DDH, DRH = x
        # Equations on main variables
        dSdt = -alpha * gamma_t * S * I / N
        dEdt = alpha * gamma_t * S * I / N - r_i * E
        dIdt = r_i * E - r_d * I
        dDHRdt = r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
        dDHDdt = r_d * p_dth * p_d * p_h * I - r_dth * DHD
        dDHdt = dDHDdt + dDHRdt
        dDDHdt = r_dth * DHD
        dDRHdt = r_rh * DHR
        return [
            dSdt, dEdt, dIdt, dDHRdt,  dDHDdt,
            dDHdt, dDDHdt, dDRHdt
        ]

    def residuals_totalcases(params) -> float:
        """
        Function that makes sure the parameters are in the right range during the fitting process and computes
        the loss function depending on the optimizer that has been chosen for this run as a global variable
        :param params: currently fitted values of the parameters during the fitting process
        :return: the value of the loss function as a float that is optimized against (in our case, minimized)
        """
        # Variables Initialization for the ODE system
        alpha, days, r_s, r_dth, p_dth, k1, k2, jump, t_jump, std_normal, t_rh, p_d = params
        # Force params values to stay in a certain range during the optimization process with re-initializations
        params = (
            max(alpha, 0),
            days,
            max(r_s, 0),
            max(min(r_dth, 1), 0),
            max(min(p_dth, 1),0),
            max(k1, 0),
            max(k2, 0),
            max(jump, 0),
            max(t_jump, 0),
            max(std_normal, 1),
            max(t_rh, 0),
            min(max(p_d, 0.01),1)
        )
        alpha, days, r_s, r_dth, p_dth, k1, k2, jump, t_jump, std_normal, t_rh, p_d = params
    
        S_0 = (N
            - (PopulationCI / p_h) * (k1 + k2 + 1 / p_d)
            - (PopulationR / p_d)
            - (PopulationD / p_d)
        )
        E_0 = PopulationCI / p_h * k1
        I_0 = PopulationCI / p_h * k2
        DHR_0 = PopulationCI  * (1 - p_dth)
        DHD_0 = PopulationCI * p_dth
        DH_0 = PopulationCI
        DDH_0 = deaths_data_fit[0]
        DRH_0 = recovered_data_fit[0]
        x_0_cases = [
            S_0, E_0, I_0, DHR_0, DHD_0, DH_0, DDH_0,
            DRH_0
        ]
        x_sol_total = solve_ivp(
            fun=model_covid,
            y0=x_0_cases,
            t_span=[t_cases[0], t_cases[-1]],
            t_eval=t_cases,
            args=tuple(params),
        )
        x_sol = x_sol_total.y
        weights = list(range(1, len(hospitalized_data_fit) + 1))
        # weights = [(x/len(cases_data_fit))**2 for x in weights]
        if x_sol_total.status == 0:
            residuals_value = sum(
            np.multiply((x_sol[5, :] - hospitalized_data_fit) ** 2, weights)
            + balance_deaths
            * balance_deaths
            * np.multiply((x_sol[6, :] - deaths_data_fit) ** 2, weights)
            + balance_recovered
            * balance_recovered
            * np.multiply((x_sol[7,:] - recovered_data_fit) ** 2, weights)
            )
        else:
            residuals_value = 1e16
        return residuals_value
    output = dual_annealing(
        residuals_totalcases, x0=parameter_list, bounds=bounds_params
    )
    best_params = output.x
    t_predictions = [i for i in range(maxT)]

    def solve_best_params_and_predict(optimal_params):
        # Variables Initialization for the ODE system
        alpha, days, r_s, r_dth, p_dth, k1, k2, jump, t_jump, std_normal, t_rh, p_d = optimal_params 
        optimal_params = (
        max(alpha, 0),
        days,
        max(r_s, 0),
        max(min(r_dth, 1), 0),
        max(min(p_dth, 1),0),
        max(k1, 0),
        max(k2, 0),
        max(jump, 0),
        max(t_jump, 0),
        max(std_normal, 1),
        max(t_rh, 0),
        min(max(p_d, 0.01),1)
    )
        S_0 = (N
            - (PopulationCI / p_h) * (k1 + k2 + 1 / p_d)
            - (PopulationR / p_d)
            - (PopulationD / p_d)
        )
        E_0 = PopulationCI / p_h * k1
        I_0 = PopulationCI / p_h * k2
        DHR_0 = PopulationCI  * (1 - p_dth)
        DHD_0 = PopulationCI * p_dth
        DH_0 = PopulationCI
        DDH_0 = 0
        DRH_0 = 0
        x_0_cases = [
            S_0, E_0, I_0, DHR_0, DHD_0, DH_0, DDH_0,
            DRH_0
        ]
        x_sol_best = solve_ivp(
            fun=model_covid,
            y0=x_0_cases,
            t_span=[t_predictions[0], t_predictions[-1]],
            t_eval=t_predictions,
            args=tuple(optimal_params),
        ).y
        return x_sol_best

    x_sol_final = solve_best_params_and_predict(best_params)
    n_days_btw_today_since_100 = (datetime.now() - start_date).days
    n_days_since_today = x_sol_final.shape[1] - n_days_btw_today_since_100

    all_dates_since_today = [
        str((datetime.now() + timedelta(days=i)).date())
        for i in range(n_days_since_today)
    ]
    # Predictions
    active_hospitalized = x_sol_final[5, :]  # DHR + DHD
    active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
    hospitalization_deaths = x_sol_final[6, :]  # DDH
    hospitalization_deaths = [int(round(x, 0)) for x in hospitalization_deaths]
    hospitalization_recoveries = x_sol_final[7, :]  # DRH
    hospitalization_recoveries = [int(round(x, 0)) for x in hospitalization_recoveries]
    # Generation of the dataframe since today
    df_predictions_since_today_area = pd.DataFrame(
        {
            "Province": ["Connecticut" for _ in range(n_days_since_today)],
            "Hospital": [location_abbr for _ in range(n_days_since_today)],
            "Day": all_dates_since_today,
            "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
            "Hospitalization Deaths": hospitalization_deaths[
                                       n_days_btw_today_since_100:
                                       ],
            "Hospitalization Recoveries": hospitalization_recoveries[
                                     n_days_btw_today_since_100:
                                     ],
        }
    )
    list_ct_output.append(df_predictions_since_today_area)
    df_parameters_area = pd.DataFrame(
        {
            "Province": ["Connecticut"],
            "Hospital": [location_abbr],
            "Infection Rate": [best_params[0]],
            "Median Day of Action": [best_params[1]],
            "Rate of Action": [best_params[2]],
            "Rate of Death": [best_params[3]],
            "Mortality Rate": [best_params[4]],
            "Internal Parameter 1": [best_params[5]],
            "Internal Parameter 2": [best_params[6]],
            "Jump Magnitude": [best_params[7]],
            "Jump Time": [best_params[8]],
            "Jump Decay": [best_params[9]],
            "Duration of Hospitalized Stay": [best_params[10]],
            "Probability of Detection": [best_params[11]]
        }
    )
    list_ct_parameters_output.append(df_parameters_area)

output_df = pd.concat(list_ct_output)
output_df_parameters = pd.concat(list_ct_parameters_output)
output_df.to_csv("HHC_Prediction_"+today + ".csv")
output_df_parameters.to_csv("HHC_Prediction_Parameters_"+today + ".csv")