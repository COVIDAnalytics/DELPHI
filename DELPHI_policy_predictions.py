# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from DELPHI_utils import (
    DELPHIDataCreator, DELPHIDataSaver,
    get_initial_conditions, mape, read_policy_data_us_only,
    get_normalized_policy_shifts_and_current_policy
)
import dateutil.parser as dtparser
import os
import matplotlib.pyplot as plt

yesterday = "".join(str(datetime.now().date() - timedelta(days=4)).split("-"))
# TODO: Find a way to make these paths automatic, whoever the user is...
PATH_TO_FOLDER_DANGER_MAP = (
    "E:/Github/covid19orc/danger_map/"
    # "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
    # "4. COVID19_Global/covid19orc/danger_map/"
)
PATH_TO_WEBSITE_PREDICTED = (
    "E:/Github/website/data/"
)
policy_data_us_only = read_policy_data_us_only()
#os.chdir(PATH_TO_FOLDER_DANGER_MAP)
popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
pastparameters = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{yesterday}.csv")
param_MATHEMATICA = False
# True if we use the Mathematica run parameters, False if we use those from Python runs
# This is because the pastparameters dataframe's columns are not in the same order in both cases

# Get the policies shifts from the CART tree to compute different values of gamma(t)
# Depending on the policy in place in the future to affect predictions
dict_policies_shift, dict_last_policy = get_normalized_policy_shifts_and_current_policy(
    policy_data_us_only=policy_data_us_only,
    pastparameters=pastparameters,
)
# Initalizing lists of the different dataframes that will be concatenated in the end
list_df_global_predictions_since_today_scenarios = []
list_df_global_predictions_since_100_cases_scenarios = []
obj_value = 0
for continent, country, province in zip(
        popcountries.Continent.tolist(),
        popcountries.Country.tolist(),
        popcountries.Province.tolist(),
):
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if (
            (os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"))
            and (country == "US")
    ):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
            continue
        print(country + " " + province)
        if pastparameters is not None:
            parameter_list_total = pastparameters[
                (pastparameters.Country == country) &
                (pastparameters.Province == province)
            ]
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                if param_MATHEMATICA:
                    parameter_list = parameter_list_line[4:]
                    parameter_list[3] = np.log(2) / parameter_list[3]
                else:
                    parameter_list = parameter_list_line[5:]
                date_day_since100 = pd.to_datetime(parameter_list_line[3])
                # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                validcases = totalcases[[
                    dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                    for x in totalcases.date
                ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            else:
                raise ValueError(f"Must have past parameters for {country} and {province}")
        else:
            raise ValueError("Must have past parameters")

        # Now we start the modeling part:
        if len(validcases) > 7:
            IncubeD = 5
            RecoverID = 10
            DetectD = 2
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
            ].pop2016.item()
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
            maxT = (datetime(2020, 9, 15) - date_day_since100).days + 1
            p_v = 0.25  # Percentage of ventilated
            p_d = 0.2  # Percentage of infection cases detected.
            p_h = 0.15  # Percentage of detected cases hospitalized
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
            best_params = parameter_list
            t_predictions = [i for i in range(maxT)]
            future_policies = [
                'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
                'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
                'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
            ]
            future_times = [0, 7, 14, 28, 42]
            plt.figure(figsize=(14, 6))
            for future_policy in future_policies:
                for future_time in future_times:
                    def model_covid_predictions(
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
                        gamma_t = (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                        gamma_t_future = (2 / np.pi) * np.arctan(-(t_cases[-1] + future_time - days) / 20 * r_s) + 1
                        gamma_0 = (2 / np.pi) * np.arctan(days / 20 * r_s) + 1
                        if t > t_cases[-1] + future_time:
                            gamma_t = (
                                gamma_t + min(
                                    (2-gamma_t_future) / (1 - dict_policies_shift[future_policy]),
                                    (gamma_t_future / dict_policies_shift[dict_last_policy[province]]) *
                                    (dict_policies_shift[future_policy] - dict_policies_shift[dict_last_policy[province]])
                                )
                            )
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
        
                    def solve_best_params_and_predict(optimal_params):
                        # Variables Initialization for the ODE system
                        x_0_cases = get_initial_conditions(
                            params_fitted=optimal_params,
                            global_params_fixed=GLOBAL_PARAMS_FIXED
                        )
                        x_sol_best = solve_ivp(
                            fun=model_covid_predictions,
                            y0=x_0_cases,
                            t_span=[t_predictions[0], t_predictions[-1]],
                            t_eval=t_predictions,
                            args=tuple(optimal_params),
                        ).y
                        return x_sol_best

                    x_sol_final = solve_best_params_and_predict(best_params)
                    data_creator = DELPHIDataCreator(
                        x_sol_final=x_sol_final, date_day_since100=date_day_since100, best_params=best_params,
                        continent=continent, country=country, province=province,
                    )
                    # Creating the parameters dataset for this (Continent, Country, Province)
                    mape_data = (
                                        mape(fitcasesnd, x_sol_final[15, :len(fitcasesnd)]) +
                                        mape(fitcasesd, x_sol_final[14, :len(fitcasesd)])
                                ) / 2
                    mape_data_2 = (
                            mape(fitcasesnd[-15:], x_sol_final[15, len(fitcasesnd)-15:len(fitcasesnd)]) +
                            mape(fitcasesd[-15:], x_sol_final[14, len(fitcasesnd)-15:len(fitcasesd)])
                    ) / 2
                    print(
                        "Policy: ", future_policy, "\t Enacting Time: ", future_time, "\t Total MAPE=", mape_data,
                        "\t MAPE on last 15 days=", mape_data_2
                    )
                    # print(best_params)
                    # print(country + ", " + province)
                    # plt.semilogy(x_sol_final[15, :], label=f"Future Policy: {future_policy} in {future_time} days")
                    # Creating the datasets for predictions of this (Continent, Country, Province)
                    df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                        data_creator.create_datasets_predictions_scenario(
                            policy=future_policy,
                            time=future_time,
                            totalcases=totalcases,
                        )
                    )
                    list_df_global_predictions_since_today_scenarios.append(df_predictions_since_today_cont_country_prov)
                    list_df_global_predictions_since_100_cases_scenarios.append(df_predictions_since_100_cont_country_prov)
            print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")
            # plt.semilogy(fitcasesnd, label="Historical Data")
            # plt.legend()
            # plt.title(f"{province} Predictions & Historical for # Cases")
            # plt.savefig(province + "_prediction.png")
            print("--------------------------------------------------------------------------")
        else:  # len(validcases) <= 7
            print(f"Not enough historical data (less than a week)" +
                  f"for Continent={continent}, Country={country} and Province={province}")
            continue
    else:  # file for that tuple (country, province) doesn't exist in processed files
        continue

# Appending parameters, aggregations per country, per continent, and for the world
# for predictions today & since 100
today_date_str = "".join(str(datetime.now().date()).split("-"))
df_global_predictions_since_today_scenarios = pd.concat(
    list_df_global_predictions_since_today_scenarios
).reset_index(drop=True)
df_global_predictions_since_100_cases_scenarios = pd.concat(
    list_df_global_predictions_since_100_cases_scenarios
).reset_index(drop=True)
delphi_data_saver = DELPHIDataSaver(
    path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
    path_to_website_predicted=PATH_TO_WEBSITE_PREDICTED,
    df_global_parameters=None,
    df_global_predictions_since_today=df_global_predictions_since_today_scenarios,
    df_global_predictions_since_100_cases=df_global_predictions_since_100_cases_scenarios,
)
delphi_data_saver.save_policy_predictions_to_dict_pickle(website=False)
print("Exported all policy-dependent predictions for US states to website & danger_map repositories")
