# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from DELPHI_utils import (
    DELPHIDataCreator, DELPHIDataSaver,
    get_initial_conditions, mape,
    read_measures_oxford_data, get_normalized_policy_shifts_and_current_policy_all_countries,
    get_normalized_policy_shifts_and_current_policy_us_only, read_policy_data_us_only
)
from DELPHI_policies_utils import (
    update_tracking_when_policy_changed, update_tracking_without_policy_change,
    get_list_and_bounds_params, get_params_constant_policies, add_policy_tracking_row_country
)
from DELPHI_params import (
    date_MATHEMATICA, validcases_threshold, n_params_without_policy_params,
    IncubeD, RecoverID, RecoverHD, DetectD, VentilatedD,
    default_maxT, p_v, p_d, p_h, future_policies, max_iter
)
import yaml
import os
import matplotlib.pyplot as plt


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "hamza"
yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
# TODO: Find a way to make these paths automatic, whoever the user is...
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["danger_map"]["michael"]
policy_data_countries = read_measures_oxford_data()
policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING])
popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
pastparameters = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{yesterday}.csv")
if pd.to_datetime(yesterday) < pd.to_datetime(date_MATHEMATICA):
    param_MATHEMATICA = True
else:
    param_MATHEMATICA = False
# True if we use the Mathematica run parameters, False if we use those from Python runs
# This is because the pastparameters dataframe's columns are not in the same order in both cases

# Get the policies shifts from the CART tree to compute different values of gamma(t)
# Depending on the policy in place in the future to affect predictions
dict_normalized_policy_gamma_countries, dict_current_policy_countries = (
    get_normalized_policy_shifts_and_current_policy_all_countries(
        policy_data_countries=policy_data_countries[policy_data_countries.date <= yesterday],
        pastparameters=pastparameters,
    )
)
# Setting same value for these 2 policies because of the inherent structure of the tree
dict_normalized_policy_gamma_countries[future_policies[3]] = dict_normalized_policy_gamma_countries[future_policies[5]]

## US Only Policies
dict_normalized_policy_gamma_us_only, dict_current_policy_us_only = (
    get_normalized_policy_shifts_and_current_policy_us_only(
        policy_data_us_only=policy_data_us_only[policy_data_us_only.date <= yesterday],
        pastparameters=pastparameters,
    )
)
dict_normalized_policy_gamma_international = dict_normalized_policy_gamma_countries.copy()
dict_normalized_policy_gamma_international.update(dict_normalized_policy_gamma_us_only)
dict_current_policy_international = dict_current_policy_countries.copy()
dict_current_policy_international.update(dict_current_policy_us_only)

# Dataset with history of previous policy changes
POLICY_TRACKING_ALREADY_EXISTS = False
if os.path.exists(CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] + f"policy_change_tracking_world_updated.csv"):
    POLICY_TRACKING_ALREADY_EXISTS = True
    df_policy_change_tracking_world = pd.read_csv(
        CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] + f"policy_change_tracking_world_updated.csv"
    )
else:
    dict_policy_change_tracking_world = {
        "continent": [np.nan], "country": [np.nan], "province": [np.nan], "date": [np.nan],
        "policy_shift_initial_param_list": [np.nan], "n_policies_enacted": [np.nan], "n_policy_changes": [np.nan],
        "last_policy": [np.nan], "start_date_last_policy": [np.nan],
        "end_date_last_policy": [np.nan], "n_days_enacted_last_policy": [np.nan], "current_policy": [np.nan],
        "start_date_current_policy": [np.nan], "end_date_current_policy": [np.nan],
        "n_days_enacted_current_policy": [np.nan], "n_params_fitted": [np.nan], "n_params_constant_tree": [np.nan],
        "init_values_params": [np.nan],
    }
    df_policy_change_tracking_world = pd.DataFrame(dict_policy_change_tracking_world)


# Initalizing lists of the different dataframes that will be concatenated in the end
list_df_policy_tracking_concat = []
list_df_global_predictions_since_today_scenarios = []
list_df_global_predictions_since_100_cases_scenarios = []
obj_value = 0
for continent, country, province in zip(
        popcountries.Continent.tolist(),
        popcountries.Country.tolist(),
        popcountries.Province.tolist(),
):
    if province not in ["Georgia", "Tennessee"]:
        continue
    #if country not in ["New Zealand"]:
    #    continue
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if (
            (os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"))
            and ((country, province) in dict_current_policy_international.keys())
    ):
        df_temp_policy_change_tracking_tuple_previous = df_policy_change_tracking_world[
            (df_policy_change_tracking_world.country == country)
            & (df_policy_change_tracking_world.province == province)
        ].sort_values(by="date").reset_index(drop=True)
        if len(df_temp_policy_change_tracking_tuple_previous) == 0:
            # Means that this (country, province) doesn't exist yet in the tracking, so create it
            df_temp_policy_change_tracking_tuple_previous = add_policy_tracking_row_country(
                continent=continent, country=country, province=province, yesterday=yesterday,
                dict_current_policy_international=dict_current_policy_international
            )
            list_df_policy_tracking_concat.append(df_temp_policy_change_tracking_tuple_previous)

        # In any case we update the tracking below
        if (
                POLICY_TRACKING_ALREADY_EXISTS and
                (df_temp_policy_change_tracking_tuple_previous.date.max() != str(pd.to_datetime(yesterday).date()))
        ):
            df_temp_policy_change_tracking_tuple_previous = df_temp_policy_change_tracking_tuple_previous.iloc[
                    len(df_temp_policy_change_tracking_tuple_previous) - 1:, :
            ].reset_index(drop=True)
            current_policy_in_tracking = df_temp_policy_change_tracking_tuple_previous.loc[0, "current_policy"]
            current_policy_in_new_data = dict_current_policy_international[(country, province)]
            df_temp_policy_change_tracking_tuple_updated = df_temp_policy_change_tracking_tuple_previous.copy()
            df_temp_policy_change_tracking_tuple_updated.loc[0, "date"] = str(pd.to_datetime(yesterday).date())
            if current_policy_in_tracking != current_policy_in_new_data:
                df_temp_policy_change_tracking_tuple_updated = update_tracking_when_policy_changed(
                    df_updated=df_temp_policy_change_tracking_tuple_updated,
                    df_previous=df_temp_policy_change_tracking_tuple_previous,
                    yesterday=yesterday,
                    current_policy_in_tracking=current_policy_in_tracking,
                    current_policy_in_new_data=current_policy_in_new_data,
                    dict_normalized_policy_gamma_international=dict_normalized_policy_gamma_international
                )
                list_df_policy_tracking_concat.append(df_temp_policy_change_tracking_tuple_updated)
            else:
                df_temp_policy_change_tracking_tuple_updated = update_tracking_without_policy_change(
                    df_updated=df_temp_policy_change_tracking_tuple_updated,
                    yesterday=yesterday,
                )
                list_df_policy_tracking_concat.append(df_temp_policy_change_tracking_tuple_updated)
        elif (
                POLICY_TRACKING_ALREADY_EXISTS and
                (df_temp_policy_change_tracking_tuple_previous.date.max() == str(pd.to_datetime(yesterday).date()))
        ):    # Already contains the latest yesterday date so has already been updated
            print(country, province)
            df_temp_policy_change_tracking_tuple_updated = df_temp_policy_change_tracking_tuple_previous.copy()
            df_temp_policy_change_tracking_tuple_updated = df_temp_policy_change_tracking_tuple_updated.iloc[
                    len(df_temp_policy_change_tracking_tuple_updated) - 1:, :
            ].reset_index(drop=True)
        else:
            raise NotImplementedError(
                f"Either Policy Tracking file doesn't exist or there is a problem with max date for {country, province}"
            )
        ### The whole if/else above is meant to retrieve & update the policy change tracking: now starts the fitting
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
                date_day_since100 = pd.to_datetime(parameter_list_line[3])
                parameter_list_fitted, start_dates_fitting_policies, bounds_params_fitted = get_list_and_bounds_params(
                    df_updated=df_temp_policy_change_tracking_tuple_updated,
                    parameter_list_line=parameter_list_line,
                    param_MATHEMATICA=param_MATHEMATICA,
                )
                params_policies_constant, start_end_dates_constant = get_params_constant_policies(
                    df_updated=df_temp_policy_change_tracking_tuple_updated, yesterday=yesterday
                )
                validcases = totalcases[[
                    dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                    for x in totalcases.date
                ]][["date", "day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            else:
                print(f"Must have past parameters for {country}, {province}, at least to create policy shifts...")
                continue
        else:
            print(f"Must have past parameters for {country}, {province}, at least to create policy shifts...")
            continue

        # Now we start the modeling part:
        if len(validcases) > validcases_threshold:
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
            # Maximum timespan of prediction, defaulted to go to 15/06/2020
            maxT = (default_maxT - date_day_since100).days + 1
            """ Fit on Total Cases """
            print(parameter_list_fitted, "\n", start_dates_fitting_policies, "\n", bounds_params_fitted,
                  "\n", params_policies_constant, "\n", start_end_dates_constant)
            t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
            t_start_fitting_policies = [
                validcases[validcases.date == date_start_fitting].day_since100.iloc[0]
                for date_start_fitting in start_dates_fitting_policies
            ]
            t_start_end_constant_policy = [
                [validcases[validcases.date == dates_constant[0]].day_since100.iloc[0],
                 validcases[validcases.date == dates_constant[1]].day_since100.iloc[0]]
                for dates_constant in start_end_dates_constant
            ]
            print("t_cases, start fitting policies & start/end constant policies")
            print(t_cases, "\n", t_start_fitting_policies, "\n", t_start_end_constant_policy)
            validcases_nondeath = validcases["case_cnt"].tolist()
            validcases_death = validcases["death_cnt"].tolist()
            balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
            fitcasesnd = validcases_nondeath
            fitcasesd = validcases_death
            GLOBAL_PARAMS_FIXED = (
                N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v
            )
            N_POLICIES_FITTED_TUPLE = len(t_start_fitting_policies)
            N_POLICIES_CONSTANT_TUPLE = len(t_start_end_constant_policy)
            print(N_POLICIES_FITTED_TUPLE, N_POLICIES_CONSTANT_TUPLE)
            raise ValueError("Got Here")
            # TODO: Need to use parameter_list_fitted, start_dates_fitting_policies, bounds_params_fitted
            #       and params_policies_constant, start_end_dates_constant

            def model_covid(
                    t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, *args
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
                if N_POLICIES_FITTED_TUPLE > 0:  # Means that we'll have some args in the function!
                    params_policies_fitted = args
                    for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                        if t >= t_start_fitting_policies[param_policy_i]:
                            gamma_t += min(0,params_policies_fitted[param_policy_i]) # TODO: adjust formula with gamma_t_future etc.

                for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                    if (
                            (t >= t_start_end_constant_policy[param_policy_constant_i][0])
                            and (t >= t_start_end_constant_policy[param_policy_constant_i][1])
                    ):
                        gamma_t += min(0, params_policies_constant[param_policy_constant_i])  # TODO: adjust formula

                # TODO: Ask Michael: what will be the new future_time when we fit to the data ? 0 (since we take t_cases[-1]) ?
                #gamma_t_future = (2 / np.pi) * np.arctan(-(t_cases[-1] + future_time - days) / 20 * r_s) + 1
                # gamma_0 = (2 / np.pi) * np.arctan(days / 20 * r_s) + 1
                # TODO: Need to modify this! Should use the lists for fitted params and constant params
                # if t > t_cases[-1] + future_time:
                #     normalized_gamma_future_policy = dict_normalized_policy_gamma_countries[future_policy]
                #     normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                #         dict_current_policy_international[(country, province)]
                #     ]
                # TODO: Ask Michael: is the future policy the new current policy, and current_policy the previous one ?
                #       e.g. if we moved from Lockdown to Restrict MG, then future_policy is RMG (although it's the 'current' policy)
                #       and current_policy is 'lockdown' ?

                #     gamma_t = gamma_t + min(
                #         (2 - gamma_t_future) / (1 - normalized_gamma_future_policy),
                #         (gamma_t_future / normalized_gamma_current_policy) *
                #         (normalized_gamma_future_policy - normalized_gamma_current_policy)
                #     )
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
                alpha, days, r_s, r_dth, p_dth, k1, k2 = params
                params = max(alpha, 0), days, max(r_s, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0)
                x_0_cases = get_initial_conditions(
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
            output = minimize(
                residuals_totalcases,
                parameter_list_fitted,
                method='trust-constr',  # Can't use Nelder-Mead if I want to put bounds on the params
                bounds=bounds_params_fitted,
                options={'maxiter': max_iter, 'verbose': 0}
            )
            best_params = output.x
            obj_value = obj_value + output.fun
            print(obj_value)
            t_predictions = [i for i in range(maxT)]

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
            # TODO: Need to modify the data creator at least for the best_params saving!
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
                      mape(fitcasesnd[-15:],
                           x_sol_final[15, len(fitcasesnd) - 15:len(fitcasesnd)]) +
                      mape(fitcasesd[-15:], x_sol_final[14, len(fitcasesnd) - 15:len(fitcasesd)])
              ) / 2
            print(
                "Policy: ", future_policy, "\t Enacting Time: ", future_time, "\t Total MAPE=", mape_data,
                "\t MAPE on last 15 days=", mape_data_2
            )
            # print(best_params)
            # print(country + ", " + province)
            # if future_policy in [
            #     'No_Measure', 'Restrict_Mass_Gatherings', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others', 'Lockdown'
            # ]:
            #     plt.plot(x_sol_final[15, :], label=f"Future Policy: {future_policy} in {future_time} days")
            # Creating the datasets for predictions of this (Continent, Country, Province)
            # TODO Check this
            df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                data_creator.create_datasets_predictions_scenario(
                    policy=future_policy,
                    time=future_time,
                    totalcases=totalcases,
                )
            )
            list_df_global_predictions_since_today_scenarios.append(
                df_predictions_since_today_cont_country_prov)
            list_df_global_predictions_since_100_cases_scenarios.append(
                df_predictions_since_100_cont_country_prov)
            print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")
            # plt.plot(fitcasesnd, label="Historical Data")
            # plt.legend()
            # plt.title(f"{country}, {province} Predictions & Historical for # Cases")
            # plt.savefig(country + "_" + province + "_prediction_cases.png")
            print("--------------------------------------------------------------------------")
        else:  # len(validcases) <= 7
            print(f"Not enough historical data (less than a week)" +
                  f"for Continent={continent}, Country={country} and Province={province}")
            continue
    else:  # file for that tuple (country, province) doesn't exist in processed files
        continue

# Appending parameters, aggregations per country, per continent, and for the world
# for predictions today & since 100
df_tracking = pd.concat([df_policy_change_tracking_world] + list_df_policy_tracking_concat).sort_values(
    by=["continent", "country", "province", "date"]
).reset_index(drop=True)
#df_tracking.to_csv(CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] + f"policy_change_tracking_world_updated.csv", index=False)
print("Saved the dataset of initial tracking")
#except:
#    print(e, f"No params file for {yesterday}")

# today_date_str = "".join(str(datetime.now().date()).split("-"))
# df_global_predictions_since_today_scenarios = pd.concat(
#     list_df_global_predictions_since_today_scenarios
# ).reset_index(drop=True)
# df_global_predictions_since_100_cases_scenarios = pd.concat(
#     list_df_global_predictions_since_100_cases_scenarios
# ).reset_index(drop=True)
# delphi_data_saver = DELPHIDataSaver(
#     path_to_folder_danger_map=PATH_TO_FOLDER_DANGER_MAP,
#     path_to_website_predicted=PATH_TO_WEBSITE_PREDICTED,
#     df_global_parameters=None,
#     df_global_predictions_since_today=df_global_predictions_since_today_scenarios,
#     df_global_predictions_since_100_cases=df_global_predictions_since_100_cases_scenarios,
# )
# # df_global_predictions_since_100_cases_scenarios.to_csv('df_global_predictions_since_100_cases_scenarios_world.csv', index=False)
# delphi_data_saver.save_policy_predictions_to_dict_pickle(website=False)
#print("Exported all policy-dependent predictions for all countries to website & danger_map repositories")
