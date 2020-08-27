# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from DELPHI_utils import (
    get_initial_conditions, )
import dateutil.parser as dtparser
import os
import yaml
import copy


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "michael"
# TODO: Find a way to make these paths automatic, whoever the user is...
reference_date = "20200517"
further_days = 0
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["website"][USER_RUNNING]
popcountries = pd.read_csv(
    PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
)
try:
    pastparameters = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{reference_date}.csv"
    )
except:
    raise ValueError(
        f"Need latest parameters to do proper backtesting; seems like they're not available for {reference_date}"
    )
if pd.to_datetime(reference_date) < pd.to_datetime("2020-05-07"):
    param_MATHEMATICA = True
else:
    param_MATHEMATICA = False
# Initalizing lists of the different dataframes that will be concatenated in the end
list_df_global_predictions_since_today = []
obj_value = 0
for continent, country, province in zip(
        popcountries.Continent.tolist(),
        popcountries.Country.tolist(),
        popcountries.Province.tolist(),
):
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
            continue
        print(country + ", " + province)
        if pastparameters is not None:
            parameter_list_total = pastparameters[
                (pastparameters.Country == country) &
                (pastparameters.Province == province)
            ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                columns_of_interest = [
                    "Data Start Date", "Infection Rate", "Median Day of Action", "Rate of Action",
                    "Rate of Death", "Mortality Rate", "Internal Parameter 1", "Internal Parameter 2"
                ]
                parameter_list_line = parameter_list_total.loc[
                    len(parameter_list_total)-1, columns_of_interest
                ].values.tolist()
                if param_MATHEMATICA:
                    parameter_list = parameter_list_line[1:]
                    parameter_list[3] = np.log(2) / parameter_list[3]
                else:
                    parameter_list = parameter_list_line[1:]
                # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                param_list_lower = [x - 0.05 * abs(x) for x in parameter_list]
                param_list_upper = [x + 0.05 * abs(x) for x in parameter_list]
                bounds_params = tuple(
                    [(lower, upper)
                     for lower, upper in zip(param_list_lower, param_list_upper)]
                )
                date_day_since100 = pd.to_datetime(parameter_list_line[0])
                validcases = totalcases[[
                    (dtparser.parse(x) >= dtparser.parse(parameter_list_line[0])) & (dtparser.parse(x) <= dtparser.parse(reference_date))
                    for x in totalcases.date
                ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            else:
                continue
        else:
            raise ValueError("Past parameters must exist")
        old_shift_day = parameter_list[1]
        print(f"The old shift day is {old_shift_day}")
        # Now we start the modeling part:
        if len(validcases) > 7:  # Cuz otherwise less train than testing data...
            for shift_day in [210]:
                parameter_list_temp = copy.copy(parameter_list)
                if shift_day < 210:
                    parameter_list_temp[1] = parameter_list_temp[1] + shift_day
                new_shift_day = parameter_list_temp[1] 
                print(f"The new shift day is {new_shift_day}")
                IncubeD = 5
                RecoverID = 10
                DetectD = 2
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
                RecoverHD = 15  # Recovery Time when Hospitalized
                VentilatedD = 10  # Recovery Time when Ventilated
                # Maximum timespan of prediction, defaulted to go to 15/06/2020
                # maxT = (datetime(2020, 6, 15) - date_day_since100).days + 1
                p_v = 0.25  # Percentage of ventilated
                p_d = 0.2  # Percentage of infection cases detected.
                p_h = 0.15  # Percentage of detected cases hospitalized
                """ Fit on Total Cases """
                t_cases_all = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
                t_cases_all = np.concatenate((t_cases_all,np.array(list(range(t_cases_all[-1]+1,t_cases_all[-1]+1+further_days)))))
                validcases_nondeath = validcases["case_cnt"].tolist()
                validcases_death = validcases["death_cnt"].tolist()
                balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
                GLOBAL_PARAMS_FIXED = (
                    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v
                )
    
                def model_covid(
                        t, x, alpha, days, r_s, r_dth, p_dth, k1, k2
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
                    if shift_day < 210:
                        gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                    else:
                        gamma_t = (2 / np.pi) * np.arctan(days / 20 * r_s) + 1
                    assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
                    S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
                    # Equations on main variables
                    dSdt = -alpha * gamma_t * S * I / N
                    dEdt = alpha * gamma_t * S * I / N - r_i * E
                    dIdt = r_i * E - r_d * I
                    dARdt = r_d * (1 - p_dth) * (1 - p_d) * I - r_ri * AR
                    dDHRdt = r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
                    dDQRdt = r_d * (1 - p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
                    dADdt = r_d * p_dth * (1 - p_d) * I - r_dth * AD
                    dDHDdt = r_d * p_dth * p_d * p_h * I - r_dth * DHD
                    dDQDdt = r_d * p_dth * p_d * (1 - p_h) * I - r_dth * DQD
                    dRdt = r_ri * (AR + DQR) + r_rh * DHR
                    dDdt = r_dth * (AD + DQD + DHD)
                    # Helper states (usually important for some kind of output)
                    dTHdt = r_d * p_d * p_h * I
                    dDVRdt = r_d * (1 - p_dth) * p_d * p_h * p_v * I - r_rv * DVR
                    dDVDdt = r_d * p_dth * p_d * p_h * p_v * I - r_dth * DVD
                    dDDdt = r_dth * (DHD + DQD)
                    dDTdt = r_d * p_d * I
                    return [
                        dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                        dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
                    ]
    
                best_params = parameter_list_temp
                def solve_best_params_and_predict(optimal_params):
                    # Variables Initialization for the ODE system
                    x_0_cases = get_initial_conditions(
                        params_fitted=optimal_params,
                        global_params_fixed=GLOBAL_PARAMS_FIXED
                    )
                    x_sol_best = solve_ivp(
                        fun=model_covid,
                        y0=x_0_cases,
                        t_span=[t_cases_all[0], t_cases_all[-1]],
                        t_eval=t_cases_all,
                        args=tuple(optimal_params),
                    ).y
                    return x_sol_best
    
                x_sol_final = solve_best_params_and_predict(best_params)
                all_dates = [
                str((date_day_since100 + timedelta(days=i)).date())
                for i in range(len(x_sol_final[15,:]))
                ]
                # import pdb
                # pdb.set_trace()
                df_predictions_since_today_cont_country_prov = pd.DataFrame({
                    "Continent": [continent for _ in range(len(x_sol_final[15,:]))],
                    "Country": [country for _ in range(len(x_sol_final[15,:]))],
                    "Province": [province for _ in range(len(x_sol_final[15,:]))],
                    "Day": all_dates,
                    "Shift": [shift_day for _ in range(len(x_sol_final[15,:]))],
                    "Total Detected": x_sol_final[15,:],
                    "Total Detected Deaths": x_sol_final[14,:]
                })
                list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
                print(f"Finished backtesting for Continent={continent}, Country={country}, Province={province}")
        else:  # len(validcases) <= 7
            print(f"Not enough historical data (less than 1 month)" +
                  f"for Continent={continent}, Country={country}, Province={province}")
            continue
    else:  # file for that tuple (country, province) doesn't exist in processed files
        continue

# Appending parameters, aggregations per country, per continent, and for the world
# for predictions today & since 100
today_date_str = "".join(str(datetime.now().date()).split("-"))
df_backtest_scenarios = pd.concat(list_df_global_predictions_since_today).reset_index(drop=True)
df_backtest_scenarios_provinces = df_backtest_scenarios[df_backtest_scenarios["Province"] != "None"]
df_backtest_scenarios_provinces_agg = df_backtest_scenarios_provinces.groupby(["Continent", "Country", "Day","Shift"]).sum().reset_index()
df_backtest_scenarios_provinces_agg["Province"] = "None"
df_backtest_scenarios_provinces_agg = df_backtest_scenarios_provinces_agg[[
    'Continent', 'Country', 'Province', 'Day', "Shift", 'Total Detected', 'Total Detected Deaths'
]]
df_backtest_scenarios_total = pd.concat([
    df_backtest_scenarios, df_backtest_scenarios_provinces_agg
])
df_backtest_scenarios_total.to_csv(
    PATH_TO_FOLDER_DANGER_MAP + f"predicted/{reference_date}_backtest_scenarios_python.csv",
    index=False
)
print("Exported backtest results to danger_map repository")
