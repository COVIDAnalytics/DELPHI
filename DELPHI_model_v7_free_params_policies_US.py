# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from DELPHI_utils_v7_free_params_policies_US import (
    DELPHIDataCreator, DELPHIAggregations, DELPHIDataSaver,
    get_initial_conditions_v7_free_params, mape,
    read_mobility_data, query_mobility_data_tuple,
    read_policy_data_us_only, query_us_policy_data_tuple, gamma
)
import dateutil.parser as dtparser
import os
import matplotlib.pyplot as plt
from copy import deepcopy

yesterday = "".join(str(datetime.now().date() - timedelta(days=4)).split("-"))
# TODO: Find a way to make these paths automatic, whoever the user is...
PATH_TO_FOLDER_DANGER_MAP = (
    #"E:/Github/covid19orc/danger_map/"
    "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
    "4. COVID19_Global/covid19orc/danger_map/"
)
PATH_TO_WEBSITE_PREDICTED = (
    "E:/Github/website/data/"
)
policy_data_us_only = read_policy_data_us_only()
#os.chdir(PATH_TO_FOLDER_DANGER_MAP)
popcountries = pd.read_csv(
    PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv"
)
try:
    pastparameters = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_Python_{yesterday}.csv"
    )
except:
    pastparameters = None

policy_data_us_only['province_cl'] = policy_data_us_only['province'].apply(lambda x: x.replace(',','').strip().lower())
states_set = set(policy_data_us_only['province_cl'])
pastparameters_copy = deepcopy(pastparameters)
pastparameters_copy['Province'] = pastparameters_copy['Province'].apply(lambda x: str(x).replace(',','').strip().lower())
params_dic = {}
for state in states_set:
    params_dic[state] = pastparameters_copy.query('Province == @state')[['Data Start Date', 'Median Day of Action', 'Rate of Action']].iloc[0]

policy_data_us_only['Gamma'] = [gamma(day, state, params_dic) for day, state in zip(policy_data_us_only['date'], policy_data_us_only['province_cl'])]
n_measures = policy_data_us_only.iloc[:, 3:-2].shape[1]
policies_dic_shift = {policy_data_us_only.columns[3 + i]: policy_data_us_only[policy_data_us_only.iloc[:, 3 + i] == 1].iloc[:, -1].mean() for i in range(n_measures)}

# Initalizing lists of the different dataframes that will be concatenated in the end
list_df_global_predictions_since_today = []
list_df_global_predictions_since_100_cases = []
list_df_global_parameters = []
# mobility_data = read_mobility_data()
for continent, country, province in zip(
        popcountries.Continent.tolist(),
        popcountries.Country.tolist(),
        popcountries.Province.tolist(),
):
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if (
            os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv")
            and (country == "US") and (province in ["California", "New York", "Massachusetts"])
    ):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if totalcases.day_since100.max() < 0:
            print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
            continue

        if pastparameters is not None:
            parameter_list_total = pastparameters[
                (pastparameters.Country == country) &
                (pastparameters.Province == province)
                ].reset_index(drop=True)
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.loc[
                    len(parameter_list_total) - 1,
                    ["Data Start Date", "Infection Rate", "Rate of Death",
                     "Mortality Rate", "Internal Parameter 1", "Internal Parameter 2"]
                ].tolist()
                date_day_since100 = pd.to_datetime(parameter_list_line[0])
                alpha_ = parameter_list_line[1]
                r_dth_ = parameter_list_line[2]
                p_dth_ = parameter_list_line[3]
                k1_ = parameter_list_line[4]
                k2_ = parameter_list_line[5]
                n_params_in_gamma_t = 8  # Intercept + 7 variables (b1, ..., b6) + days_param
                parameter_list = [alpha_, r_dth_, p_dth_, k1_, k2_] + [0 for _ in range(n_params_in_gamma_t)]
                parameter_list = np.array(parameter_list)
                # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                param_list_lower = [x - 0.5 * abs(x) for x in parameter_list[:5]]
                param_list_upper = [x + 0.5 * abs(x) for x in parameter_list[:5]]
                bounds_params = tuple(
                    [
                        (lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)
                    ] + [(-5, 0) for _ in range(n_params_in_gamma_t - 1)] + [(0, 10)]  # That's for days_param
                )
                validcases = totalcases[[
                    dtparser.parse(x) >= dtparser.parse(parameter_list_line[0])
                    for x in totalcases.date
                ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            else:
                raise ValueError("Couldn't run model, need past parameters")
                # Otherwise use established lower/upper bounds
                # parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
                # bounds_params = (
                #    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
                # )
                # date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
                # validcases = totalcases[totalcases.day_since100 >= 0][
                #    ["day_since100", "case_cnt", "death_cnt"]
                # ].reset_index(drop=True)
        else:
            raise ValueError("Couldn't run model, need past parameters")
            # Otherwise use established lower/upper bounds
            # parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
            # bounds_params = (
            #    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
            # )
            # date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
            # validcases = totalcases[totalcases.day_since100 >= 0][
            #    ["day_since100", "case_cnt", "death_cnt"]
            # ].reset_index(drop=True)

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
            maxT = (datetime(2020, 7, 15) - date_day_since100).days + 1
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
            policy_data_us_only_tuple = query_us_policy_data_tuple(
                policy_data_us_only=policy_data_us_only, country=country, province=province,
                date_day_since100=date_day_since100, maxT=maxT, future_policy="minimization only"
            )

            def model_covid(
                    t, x, alpha, r_dth, p_dth, k1, k2, b1, b2, b3, b4, b5, b6, b7, days_param
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
                # gamma_t = 1 / (1 + np.exp(-(
                #         b0 + b1 * policy_data_us_only_tuple["policy_1"][int(t)] +
                #         b2 * policy_data_us_only_tuple["policy_2"][int(t)] +
                #         b3 * policy_data_us_only_tuple["policy_3"][int(t)] +
                #         b4 * policy_data_us_only_tuple["policy_4"][int(t)] +
                #         b5 * policy_data_us_only_tuple["policy_5"][int(t)] +
                #         b6 * policy_data_us_only_tuple["policy_6"][int(t)] +
                #         b7 * policy_data_us_only_tuple["policy_7"][int(t)]
                # ) * t))
                inside_arctan = (
                        b1 * policy_data_us_only_tuple["policy_1"][int(t)] +
                        b2 * policy_data_us_only_tuple["policy_2"][int(t)] +
                        b3 * policy_data_us_only_tuple["policy_3"][int(t)] +
                        b4 * policy_data_us_only_tuple["policy_4"][int(t)] +
                        b5 * policy_data_us_only_tuple["policy_5"][int(t)] +
                        b6 * policy_data_us_only_tuple["policy_6"][int(t)] +
                        b7 * policy_data_us_only_tuple["policy_7"][int(t)]
                )
                gamma_t = (2/np.pi) * np.arctan(inside_arctan * (t - days_param) / 20) + 1
                #gamma_t = np.tanh(inside_arctan * (t - days_param) / 20) + 1
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


            def residuals_totalcases(params):
                """
                Wanted to start with solve_ivp because figures will be faster to debug
                params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
                """
                # Variables Initialization for the ODE system
                alpha, r_dth, p_dth, k1, k2, b1, b2, b3, b4, b5, b6, b7, days_param = params
                params = (
                    max(alpha, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0),
                    b1, b2, b3, b4, b5, b6, b7, days_param
                )
                x_0_cases = get_initial_conditions_v7_free_params(
                    params_fitted=params,
                    global_params_fixed=GLOBAL_PARAMS_FIXED
                )
                x_sol = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_cases[0], t_cases[-1]],
                    t_eval=t_cases,
                    args=tuple(params),
                ).y
                weights = list(range(1, len(fitcasesnd) + 1))
                residuals_value = sum(
                    np.multiply((x_sol[15, :] - fitcasesnd) ** 2, weights)
                    + balance * balance * np.multiply((x_sol[14, :] - fitcasesd) ** 2, weights)
                )
                return residuals_value


            best_params = minimize(
                residuals_totalcases,
                parameter_list,
                method='trust-constr',  # Can't use Nelder-Mead if I want to put bounds on the params
                bounds=bounds_params
            ).x
            t_predictions = [i for i in range(maxT)]
            plt.figure(figsize=(14, 6))
            list_to_plot = []
            for j, future_policy_j in enumerate(["current", "no policy", "current 4weeks then open schools"]):
                policy_data_us_only_tuple_policy_j = query_us_policy_data_tuple(
                    policy_data_us_only=policy_data_us_only, country=country, province=province,
                    date_day_since100=date_day_since100, maxT=maxT, future_policy=future_policy_j
                )

                def model_covid_predictions(
                        t, x, alpha, r_dth, p_dth, k1, k2, b1, b2, b3, b4, b5, b6, b7, days_param
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
                    inside_arctan = (
                            b1 * policy_data_us_only_tuple_policy_j["policy_1"][int(t)] +
                            b2 * policy_data_us_only_tuple_policy_j["policy_2"][int(t)] +
                            b3 * policy_data_us_only_tuple_policy_j["policy_3"][int(t)] +
                            b4 * policy_data_us_only_tuple_policy_j["policy_4"][int(t)] +
                            b5 * policy_data_us_only_tuple_policy_j["policy_5"][int(t)] +
                            b6 * policy_data_us_only_tuple_policy_j["policy_6"][int(t)] +
                            b7 * policy_data_us_only_tuple_policy_j["policy_7"][int(t)]
                    )
                    gamma_t = (2/np.pi) * np.arctan(inside_arctan * (t - days_param) / 20) + 1
                    #gamma_t = np.tanh(inside_arctan * (t - days_param) / 20) + 1
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
                    x_0_cases = get_initial_conditions_v7_free_params(
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


                vars()[f"x_sol_final_{j}"] = solve_best_params_and_predict(best_params)
                data_creator = DELPHIDataCreator(
                    x_sol_final=vars()[f"x_sol_final_{j}"], date_day_since100=date_day_since100, best_params=best_params,
                    continent=continent, country=country, province=province,
                )
                # Creating the parameters dataset for this (Continent, Country, Province)
                mape_data = (
                                    mape(fitcasesnd, vars()[f"x_sol_final_{j}"][15, :len(fitcasesnd)]) +
                                    mape(fitcasesd, vars()[f"x_sol_final_{j}"][14, :len(fitcasesd)])
                            ) / 2
                mape_data_last_15 = (
                                            mape(fitcasesnd[-15:], vars()[f"x_sol_final_{j}"][15, len(fitcasesnd) - 15: len(fitcasesnd)]) +
                                            mape(fitcasesd[-15:], vars()[f"x_sol_final_{j}"][14, len(fitcasesd) - 15: len(fitcasesd)])
                                    ) / 2
                print("Total MAPE=", mape_data, "\t MAPE on last 15 days=", mape_data_last_15)
                print(f"State {province}, Policy {future_policy_j}\n" +
                      f"Cases last 15 predictions: {vars()[f'x_sol_final_{j}'][15, -20:]}" +
                      f"Deaths last 15 predictions: {vars()[f'x_sol_final_{j}'][14, -20:]}")
                print(best_params)
                print(country + "," + province)
                #plt.plot(fitcasesnd)
                plt.semilogy(vars()[f"x_sol_final_{j}"][15, :], label=f"Future Policy: {future_policy_j}")
                #plt.savefig(province+"_prediction.png")
                df_parameters_cont_country_prov = data_creator.create_dataset_parameters(mape_data)
                list_df_global_parameters.append(df_parameters_cont_country_prov)
                # Creating the datasets for predictions of this (Continent, Country, Province)
                df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                    data_creator.create_datasets_predictions()
                )
                # TODO: Need to deal with the saving and concatenation of files, maybe add a column for "future_policy"
                list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
                list_df_global_predictions_since_100_cases.append(df_predictions_since_100_cont_country_prov)
            print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")
            plt.semilogy(fitcasesnd, label="Historical Data")
            plt.legend()
            plt.title(f"{province} Predictions & Historical for # Cases")
            plt.savefig(province + "_prediction.png")
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
df_global_parameters = pd.concat(list_df_global_parameters)
df_global_predictions_since_today = pd.concat(list_df_global_predictions_since_today)
df_global_predictions_since_today = DELPHIAggregations.append_all_aggregations(
    df_global_predictions_since_today
)
# TODO: Discuss with website team how to save this file to visualize it and compare with historical data
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
# delphi_data_saver.save_all_datasets(save_since_100_cases=False)
print("Exported all 3 datasets to website & danger_map repositories")
