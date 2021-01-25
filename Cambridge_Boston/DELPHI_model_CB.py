#%%

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

#%%

## Initializing Global Variables ##########################################################################
today = "".join(str(datetime.now().date()).split("-"))
OPTIMIZER = "annealing"
bounds_params = (
    (0.1, 1.5), (-200, 100), (1, 15), (0.001, 5), (0.001, 5), (0, 5), (-100, 300), (0.1, 100), (0.1999, 0.2001)
)
# alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d
default_parameter_list = [1, 0, 2, 3, 3, 0.1, 3, 1, 0.2] # default parameter initialization
dict_default_reinit_parameters = {
    "alpha": 0, "days": None, "r_s": 0, "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "p_d": 0.01
}  # Allows for reinitialization of parameters in case they reach a value that is too low/high
dict_default_reinit_lower_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "p_d": 0.01
}  # Allows for reinitialization of lower bounds in case they reach a value that is too low
dict_default_reinit_upper_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "p_d": 0.01
} # Allows for reinitialization of upper bounds in case they reach a value that is too low
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated
PopulationT = 4309000+119703
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.03  # Percentage of detected cases hospitalized
max_iter = 500 # for tnc
############################################################################################################

# %%

def get_residuals_value(
        optimizer: str, balance: float, x_sol: list, 
        cases_data_fit: list, 
        # deaths_data_fit: list, 
        weights: list
) -> float:
    """
    Obtain the value of the loss function depending on the optimizer (as it is different for global optimization using
    simulated annealing)
    :param optimizer: String, for now either tnc, trust-constr or annealing
    :param balance: Regularization coefficient between cases and deaths
    :param x_sol: Solution previously fitted by the optimizer containing fitted values for all 16 states
    :param fitcasend: cases data to be fitted on
    :param deaths_data_fit: deaths data to be fitted on
    :param weights: time-related weights to give more importance to recent data points in the fit (in the loss function)
    :return: float, corresponding to the value of the loss function
    """
    if optimizer in ["tnc", "trust-constr"]:
        residuals_value = sum(
            np.multiply((x_sol[7, :] - cases_data_fit) ** 2, weights)
            # + balance
            # * balance
            # * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        )
    elif optimizer == "annealing":
        residuals_value = sum(
            np.multiply((x_sol[7, :] - cases_data_fit) ** 2, weights)
            # + balance
            # * balance
            # * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        ) + sum(
            np.multiply(
                (x_sol[7, 7:] - x_sol[7, :-7] - cases_data_fit[7:] + cases_data_fit[:-7]) ** 2,
                weights[7:],
            )
            # + balance * balance * np.multiply(
            #     (x_sol[14, 7:] - x_sol[14, :-7] - deaths_data_fit[7:] + deaths_data_fit[:-7]) ** 2,
            #     weights[7:],
            # )
        )
    else:
        raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

    return residuals_value


def get_initial_conditions(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 9 states of the DELPHI model
    """
    alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d = params_fitted 
    N, R_upperbound, R_0, PopulationI, p_h = global_params_fixed

    PopulationR = min(int(R_upperbound*p_d) - 1, int(R_0*p_d))
    PopulationCI = (PopulationI - PopulationR)

    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d)
        # - (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    UR_0 = (PopulationCI / p_d - PopulationCI)
    DHR_0 = (PopulationCI * p_h)
    DQR_0 = PopulationCI * (1 - p_h)
    DT_0 = PopulationI
    DA_0 = PopulationCI
    x_0_cases = [
        S_0, E_0, I_0, UR_0, DHR_0, DQR_0, R_0, DT_0, DA_0
    ]
    return x_0_cases

def solve_and_predict_area(
        past_parameters_: pd.DataFrame,
        validcases: pd.DataFrame,
        start_date: pd.datetime,
        end_date: pd.datetime,
        R_0: int,
        R_upperbound: int
):
    time_entering = time.time()

    # Now we start the modeling part:
    N = PopulationT
    PopulationI = validcases.loc[0, "case_cnt"]

    # R_0 = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
    # R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]

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
    maxT = (default_maxT - start_date).days + 1
    t_cases = list(range(len(validcases)))
    cases_data_fit = validcases.case_cnt.to_list()
    GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_0, PopulationI, p_h)
    parameter_list = default_parameter_list

    def model_covid(
        t, x, alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d
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
        :param k1: Internal parameter 1 (used for initial conditions)
        :param k2: Internal parameter 2 (used for initial conditions)
        :param jump: Amplitude of the Gaussian jump modeling the resurgence in cases
        :param t_jump: Time where the Gaussian jump will reach its maximum value
        :param std_normal: Standard Deviation of the Gaussian jump (~ time span of the resurgence in cases)
        :return: predictions for all 8 states, which are the following
        [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 R, 7 DT, 8 DA]
        """
        r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
        r_d = np.log(2) / DetectD  # Rate of detection
        r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
        r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
        gamma_t = (
            (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
            + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
        )
        # p_dth_mod = (2 / np.pi) * (p_dth - 0.001) * (np.arctan(-t / 20 * r_dthdecay) + np.pi / 2) + 0.001
        assert (
            len(x) == 9
        ), f"Too many input variables, got {len(x)}, expected 9"
        S, E, I, AR, DHR, DQR, R, DT, DA = x
        # Equations on main variables
        dSdt = -alpha * gamma_t * S * I / N
        dEdt = alpha * gamma_t * S * I / N - r_i * E
        dIdt = r_i * E - r_d * I
        dARdt = r_d * (1 - p_d) * I - r_ri * AR
        dDHRdt = r_d * p_d * p_h * I - r_rh * DHR
        dDQRdt = r_d * p_d * (1 - p_h) * I - r_ri * DQR
        dRdt = r_ri * (AR + DQR) + r_rh * DHR

        # Helper states (usually important for some kind of output)
        dDTdt = r_d * p_d * I
        dDAdt = dDHRdt + dDQRdt

        return [
            dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dRdt, dDTdt, dDAdt
        ]

    def residuals_totalcases(params) -> float:
        """
        Function that makes sure the parameters are in the right range during the fitting process and computes
        the loss function depending on the optimizer that has been chosen for this run as a global variable
        :param params: currently fitted values of the parameters during the fitting process
        :return: the value of the loss function as a float that is optimized against (in our case, minimized)
        """
        # Variables Initialization for the ODE system
        alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d = params
        # Force params values to stay in a certain range during the optimization process with re-initializations
        params = (
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
            max(p_d, dict_default_reinit_parameters["p_d"])
        )

        x_0_cases = get_initial_conditions(
            params_fitted=params, global_params_fixed=GLOBAL_PARAMS_FIXED
        )
        x_sol_total = solve_ivp(
            fun=model_covid,
            y0=x_0_cases,
            t_span=[t_cases[0], t_cases[-1]],
            t_eval=t_cases,
            args=tuple(params),
        )
        x_sol = x_sol_total.y
        weights = list(range(1, len(cases_data_fit) + 1))
        weights = [(x/len(cases_data_fit))**2 for x in weights]
        if x_sol_total.status == 0:
            residuals_value = get_residuals_value(
                optimizer=OPTIMIZER,
                balance=1,
                x_sol=x_sol,
                cases_data_fit=cases_data_fit,
                # deaths_data_fit=deaths_data_fit,
                weights=weights
            )
        else:
            residuals_value = 1e16
        return residuals_value

    if OPTIMIZER in ["tnc", "trust-constr"]:
        output = minimize(
            residuals_totalcases,
            parameter_list,
            method=OPTIMIZER,
            bounds=bounds_params,
            options={"maxiter": max_iter},
        )
    elif OPTIMIZER == "annealing":
        output = dual_annealing(
            residuals_totalcases, x0=parameter_list, bounds=bounds_params
        )
    else:
        raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

    if (OPTIMIZER in ["tnc", "trust-constr"]) or (OPTIMIZER == "annealing" and output.success):
        best_params = output.x
        t_predictions = [i for i in range(maxT)]

        def solve_best_params_and_predict(optimal_params):
            # Variables Initialization for the ODE system
            alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d = optimal_params
            optimal_params = [
                max(alpha, dict_default_reinit_parameters["alpha"]),
                days,
                max(r_s, dict_default_reinit_parameters["r_s"]),
                max(k1, dict_default_reinit_parameters["k1"]),
                max(k2, dict_default_reinit_parameters["k2"]),
                max(jump, dict_default_reinit_parameters["jump"]),
                max(t_jump, dict_default_reinit_parameters["t_jump"]),
                max(std_normal, dict_default_reinit_parameters["std_normal"]),
                max(p_d, dict_default_reinit_parameters["p_d"])
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

        x_sol_final = solve_best_params_and_predict(best_params) # = S, E, I, AR, DHR, DQR, R, DT, DA
        n_days_btw_today_since_start = (end_date - start_date).days 
        # here today means the end date of data selected, can be different from actual current date
        n_days_since_today = x_sol_final.shape[1] - n_days_btw_today_since_start

        all_dates_since_today = [
            str((end_date + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        all_dates = [
            str((start_date + timedelta(days=i)).date())
            for i in range(len(x_sol_final[0]))
        ]
        # Predictions
        active_cases = x_sol_final[8,:]
        total_cases = x_sol_final[7,:]
        # Generation of the dataframe since today
        df_predictions = pd.DataFrame(
            {
                "Day": all_dates,
                "Active Cases": active_cases,
                "Total Cases": total_cases
            }
        )
        
        # alpha, days, r_s, k1, k2, jump, t_jump, std_normal, p_d
        df_parameters = pd.DataFrame(
            {
                "Area": ["Cambridge and Boston"],
                "Infection Rate": [best_params[0]],
                "Median Day of Action": [best_params[1]],
                "Rate of Action": [best_params[2]],
                "Internal Parameter 1": [best_params[3]],
                "Internal Parameter 2": [best_params[4]],
                "Jump Magnitude": [best_params[5]],
                "Jump Time": [best_params[6]],
                "Jump Decay": [best_params[7]],
                "Probability of Detection": [best_params[8]]
            }
        )
        time_leaving = time.time()
        print(time_leaving - time_entering)
        return df_predictions, df_parameters
    else:
        return None

#%%

start_date = pd.to_datetime("2020-12-16")
default_maxT = pd.to_datetime("2021-03-15")
end_date = pd.to_datetime("2021-01-13")


### Initial Analysis ####################################

# %%

boston_cases = pd.read_excel("data/processed/boston_cases.xlsx")
boston_cases.Timestamp = pd.to_datetime(boston_cases.Timestamp, errors='coerce')
boston_cases = boston_cases[(boston_cases.Timestamp >= start_date) & (boston_cases.Timestamp <= end_date)]
R_boston = boston_cases['Cases: Total Recovered'].values[0] + boston_cases['Total Boston Resident Deaths'].values[0]

#%%

cambridge_cases = pd.read_csv("data/processed/COVID-19_Case_Count_by_Date.csv")
cambridge_cases.Date = pd.to_datetime(cambridge_cases.Date)
R_cambridge = cambridge_cases[cambridge_cases.Date >= str(start_date- pd.Timedelta(14, 'D'))]['Cumulative Confirmed Cases'].values[0]
R2_cambridge = cambridge_cases[cambridge_cases.Date >= str(start_date- pd.Timedelta(9, 'D'))]['Cumulative Confirmed Cases'].values[0]
cambridge_cases = cambridge_cases.query("Date >= @start_date and Date <= @end_date")

# %%
validcases = pd.DataFrame({
    'date': np.array(cambridge_cases.Date),
    'case_cnt': np.array(cambridge_cases['Cumulative Confirmed Cases']) + np.array(boston_cases['Cases: Total Positive'])
})
# validcases.head()

# %%
R_0 = (R_boston + R_cambridge)
R_upperbound = (R_boston + R2_cambridge)

# %%

df_predictions, df_parameters = solve_and_predict_area(
        past_parameters_= None,
        validcases=validcases,
        start_date=start_date,
        end_date=end_date,
        R_0=R_0,
        R_upperbound=R_upperbound
)
# %%
df_predictions.to_csv('predictions/predicted_cases_v2.csv', index=False)



###################################################################


# %% 
### data imputation experiment #############################
PopulationT = 4309000

start_date = pd.to_datetime("2020-07-01")
default_maxT = pd.to_datetime("2021-01-15")
end_date = pd.to_datetime("2020-12-11")

boston_cases = pd.read_csv('data/processed/boston_cases_till_dec11.csv')
boston_cases.Date = pd.to_datetime(boston_cases.Date)

R_0 = boston_cases[boston_cases.Date >= str(start_date- pd.Timedelta(14, 'D'))]['total_cases'].values[0] / p_d
R_upperbound = boston_cases[boston_cases.Date >= str(start_date- pd.Timedelta(9, 'D'))]['total_cases'].values[0] / p_d
boston_cases = boston_cases.query("Date >= @start_date and Date <= @end_date")

validcases = pd.DataFrame({
    'date': np.array(boston_cases.Date),
    'case_cnt': np.array(boston_cases.total_cases)
})

# %%
df_predictions, df_parameters = solve_and_predict_area(
        past_parameters_= None,
        validcases=validcases,
        start_date=start_date,
        end_date=end_date,
        R_0=R_0,
        R_upperbound=R_upperbound
)
df_predictions.to_csv('predictions/boston_predicted_cases_Dec2020.csv', index=False)
###################################################################

# %%

### final model #####################################################

start_date = pd.to_datetime("2020-07-01")
default_maxT = pd.to_datetime("2021-03-15")
end_date = pd.to_datetime("2021-01-13")

validcases = pd.read_csv('data/processed/cambridge_boston_combined.csv')
validcases.date = pd.to_datetime(validcases.date)

R_0 = validcases[validcases.date >= str(start_date- pd.Timedelta(14, 'D'))]['case_cnt'].values[0] / p_d
R_upperbound = validcases[validcases.date >= str(start_date- pd.Timedelta(9, 'D'))]['case_cnt'].values[0] / p_d
validcases = validcases.query("date >= @start_date and date <= @end_date")
validcases.reset_index(inplace=True)


# %%
df_predictions, df_parameters = solve_and_predict_area(
        past_parameters_= None,
        validcases=validcases,
        start_date=start_date,
        end_date=end_date,
        R_0=R_0,
        R_upperbound=R_upperbound
)
df_predictions.to_csv('predictions/predicted_cases_v3.csv', index=False)

# %%
