# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
# This is DELPHI with Continuous Retraining on Policies implemented
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
from DELPHI_utils import (
    DELPHIDataCreator, DELPHIDataSaver, DELPHIAggregations,
    get_initial_conditions, mape,
    read_measures_oxford_data, get_normalized_policy_shifts_and_current_policy_all_countries,
    get_normalized_policy_shifts_and_current_policy_us_only, read_policy_data_us_only
)
from DELPHI_policies_utils import (
    update_tracking_when_policy_changed, update_tracking_without_policy_change,
    get_list_and_bounds_params, get_params_constant_policies, add_policy_tracking_row_country,
    get_policy_names_before_after_fitted, get_policy_names_before_after_constant,
    update_tracking_fitted_params
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
training_start_date = "2020-06-13"
training_end_date = "2020-06-14"
max_days_before = (pd.to_datetime(training_end_date) - pd.to_datetime(training_start_date)).days
time_start = datetime.now()
for n_days_before in range(max_days_before, 0, -1):
    yesterday = "".join(str(pd.to_datetime(training_end_date).date() - timedelta(days=n_days_before)).split("-"))
    print(yesterday)
    print(f"Runtime for {yesterday}: {datetime.now() - time_start}")
    PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
    PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
    PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["danger_map"]["michael"]
    policy_data_countries = read_measures_oxford_data(yesterday=yesterday)
    policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=PATH_TO_DATA_SANDBOX)
    popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
    # if pd.to_datetime(yesterday) <= pd.to_datetime(training_start_date):
    #     pastparameters = pd.read_csv(
    #         PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{yesterday}.csv"
    #     )
    #     param_MATHEMATICA = True
    # else:
    #     pastparameters = pd.read_csv(
    #         PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_CR_{yesterday}_newtry.csv"
    #     )
    #     param_MATHEMATICA = False
    pastparameters = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"predicted/parameters_global_CR_all/Parameters_Global_CR_{yesterday}.csv"
    )
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
    dict_current_policy_international = dict_current_policy_countries.copy()
    dict_current_policy_international.update(dict_current_policy_us_only)

    # Dataset with history of previous policy changes
    POLICY_TRACKING_ALREADY_EXISTS = False
    if os.path.exists(PATH_TO_DATA_SANDBOX + f"policy_change_tracking_world_updated_final.csv"):
        POLICY_TRACKING_ALREADY_EXISTS = True
        df_policy_change_tracking_world = pd.read_csv(
            PATH_TO_DATA_SANDBOX + f"policy_change_tracking_world_updated_final.csv"
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
    list_df_global_predictions_since_today = []
    list_df_global_predictions_since_100_cases = []
    list_df_global_parameters = []
    list_df_policy_tracking_concat = []
    obj_value = 0
    for continent, country, province in zip(
            popcountries.Continent.tolist(),
            popcountries.Country.tolist(),
            popcountries.Province.tolist(),
    ):
        if country == "US":  # This line is necessary because the keys are the same in both cases
            dict_normalized_policy_gamma_international = dict_normalized_policy_gamma_us_only.copy()
        else:
            dict_normalized_policy_gamma_international = dict_normalized_policy_gamma_countries.copy()

        CREATED_COUNTRY_IN_TRACKING = False
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
                CREATED_COUNTRY_IN_TRACKING = True

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
                    print(f"There's a Policy Change on {yesterday}")
                    df_temp_policy_change_tracking_tuple_updated = update_tracking_when_policy_changed(
                        df_updated=df_temp_policy_change_tracking_tuple_updated,
                        df_previous=df_temp_policy_change_tracking_tuple_previous,
                        yesterday=yesterday,
                        current_policy_in_tracking=current_policy_in_tracking,
                        current_policy_in_new_data=current_policy_in_new_data,
                        dict_normalized_policy_gamma_international=dict_normalized_policy_gamma_international
                    )
                else:
                    print(f"There's no Policy Change on {yesterday}")
                    df_temp_policy_change_tracking_tuple_updated = update_tracking_without_policy_change(
                        df_updated=df_temp_policy_change_tracking_tuple_updated,
                        yesterday=yesterday,
                    )
            elif (
                    POLICY_TRACKING_ALREADY_EXISTS and
                    (df_temp_policy_change_tracking_tuple_previous.date.max() == str(pd.to_datetime(yesterday).date()))
                    and (CREATED_COUNTRY_IN_TRACKING == True)
            ):    # Already contains the latest yesterday date so has already been updated
                print(country, province)
                df_temp_policy_change_tracking_tuple_updated = df_temp_policy_change_tracking_tuple_previous.copy()
                df_temp_policy_change_tracking_tuple_updated = df_temp_policy_change_tracking_tuple_updated.iloc[
                        len(df_temp_policy_change_tracking_tuple_updated) - 1:, :
                ].reset_index(drop=True)
            elif(
                    POLICY_TRACKING_ALREADY_EXISTS and
                    (df_temp_policy_change_tracking_tuple_previous.date.max() == str(pd.to_datetime(yesterday).date()))
                    and (CREATED_COUNTRY_IN_TRACKING == False)
            ):
                raise ValueError(
                    f"Policy Tracking line for {country}, {province} already exists on {yesterday}; " +
                    "make sure you want to re-predict on that day and if so modify the code to get rid of the " +
                    "current policy tracking line (otherwise you'll get a duplicate)"
                )
            else:
                raise NotImplementedError(
                    f"Either Policy Tracking file doesn't exist or there is a problem "
                    f"with max date for {country, province}"
                )
            ### The whole if/else above is meant to retrieve & update the policy change tracking: now starts the fitting
            totalcases = pd.read_csv(
                PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
            )
            if totalcases.day_since100.max() < 0:
                print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
                continue

            print(f"Current Runtime for {yesterday}: {datetime.now() - time_start}")
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
                    policy_shift_names_fitted = get_policy_names_before_after_fitted(
                        df_updated=df_temp_policy_change_tracking_tuple_updated
                    )
                    params_policies_constant, start_end_dates_constant = get_params_constant_policies(
                        df_updated=df_temp_policy_change_tracking_tuple_updated, yesterday=yesterday
                    )
                    policy_shift_names_constant = get_policy_names_before_after_constant(
                        df_updated=df_temp_policy_change_tracking_tuple_updated
                    )
                    validcases = totalcases[[
                        dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                        for x in totalcases.date
                    ]][["date", "day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                    validcases = validcases[
                        validcases.date <= str((pd.to_datetime(yesterday) + timedelta(days=1)).date())
                    ]
                    print("Historical Data until:", validcases.date.max(), f"\t Parameters from: {yesterday}")
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
                #print(parameter_list_fitted, "\n", start_dates_fitting_policies, "\n", bounds_params_fitted,
                #      "\n", params_policies_constant, "\n", start_end_dates_constant)
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
                date_yesterday_int = validcases[
                    validcases.date == str(pd.to_datetime(yesterday).date())
                ].day_since100.iloc[0]
                #print("t_cases, start fitting policies & start/end constant policies")
                #print(t_cases, "\n", t_start_fitting_policies, "\n", t_start_end_constant_policy)
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
                print("Policy Shifts Fitted:", N_POLICIES_FITTED_TUPLE, " ##### "
                      "Policy Shifts Constant:", N_POLICIES_CONSTANT_TUPLE)
                if N_POLICIES_FITTED_TUPLE == 0:  # No fitted params but at least constant ones are still taken into account
                    def model_covid(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2,
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
                        if N_POLICIES_CONSTANT_TUPLE >= 1:
                            for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                                start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                                end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                                if (
                                        (t >= start_date_constant_i)
                                        and (t <= end_date_constant_i)
                                ):
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 1:  # 1 fitted + potential constant ones
                    def model_covid(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1,
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
                        params_policies_fitted = [k_policy_shift_1]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if (
                                    (t >= start_date_constant_i)
                                    and (t <= end_date_constant_i)
                            ):
                                past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy_cst]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                        param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 2:  # 2 fitted + potential constant ones
                    def model_covid(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1, k_policy_shift_2
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
                        params_policies_fitted = [k_policy_shift_1, k_policy_shift_2]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if (
                                    (t >= start_date_constant_i)
                                    and (t <= end_date_constant_i)
                            ):
                                past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy_cst]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                        param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 3:  # 3 fitted + potential constant ones
                    def model_covid(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2,
                            k_policy_shift_1, k_policy_shift_2, k_policy_shift_3
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
                        params_policies_fitted = [k_policy_shift_1, k_policy_shift_2, k_policy_shift_3]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if (
                                    (t >= start_date_constant_i)
                                    and (t <= end_date_constant_i)
                            ):
                                past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy_cst]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                        param_policy_constant_i]
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
                else:
                    raise NotImplementedError(f"Only Implemented up to 3 fitted parameters, got {N_POLICIES_FITTED_TUPLE}")

                def residuals_totalcases(params):
                    """
                    Wanted to start with solve_ivp because figures will be faster to debug
                    params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
                    """
                    # Variables Initialization for the ODE system
                    if len(params) == n_params_without_policy_params:
                        alpha, days, r_s, r_dth, p_dth, k1, k2 = params
                        params = max(alpha, 0), days, max(r_s, 0), max(r_dth, 0), max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0)
                    elif len(params) == 8:
                        alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1 = params
                        params = (
                            max(alpha, 0), days, max(r_s, 0), max(r_dth, 0),
                            max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0), k_policy_shift_1
                        )
                    elif len(params) == 9:
                        alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1, k_policy_shift_2 = params
                        params = (
                            max(alpha, 0), days, max(r_s, 0), max(r_dth, 0),
                            max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0), k_policy_shift_1,
                            k_policy_shift_2
                        )
                    elif len(params) == 10:
                        alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1, k_policy_shift_2, k_policy_shift_3 = params
                        params = (
                            max(alpha, 0), days, max(r_s, 0), max(r_dth, 0),
                            max(min(p_dth, 1), 0), max(k1, 0), max(k2, 0), k_policy_shift_1,
                            k_policy_shift_2, k_policy_shift_3
                        )
                    else:
                        raise NotImplementedError(
                            f"Only Implemented up to 3 fitted parameters, got {N_POLICIES_FITTED_TUPLE}"
                        )
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
                print(f"# Params Fitted: {len(parameter_list_fitted)}, Params Fitted: {parameter_list_fitted}")
                output = minimize(
                    residuals_totalcases,
                    parameter_list_fitted,
                    method='trust-constr',  # Can't use Nelder-Mead if I want to put bounds on the params
                    bounds=bounds_params_fitted,
                    options={'maxiter': max_iter, 'verbose': 0}
                )
                print("FINISHED MINIMIZING")
                best_params = output.x
                obj_value = obj_value + output.fun
                print(output.fun)
                t_predictions = [i for i in range(maxT)]
                if N_POLICIES_FITTED_TUPLE == 0:  # No fitted params but at least constant ones are still taken into account
                    def model_covid_predictions(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2,
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
                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if end_date_constant_i != date_yesterday_int:
                                assert end_date_constant_i < date_yesterday_int
                                if (
                                        (t >= start_date_constant_i)
                                        and (t <= end_date_constant_i)
                                ):
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[param_policy_constant_i]
                                    )
                            else:
                                # if end date of a constant param is set to yesterday,
                                # it means that it continues even in the future
                                if t >= start_date_constant_i:
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 1:  # 1 fitted + potential constant ones
                    def model_covid_predictions(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1,
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
                        params_policies_fitted = [k_policy_shift_1]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if end_date_constant_i != date_yesterday_int:
                                assert end_date_constant_i < date_yesterday_int
                                if (
                                        (t >= start_date_constant_i)
                                        and (t <= end_date_constant_i)
                                ):
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
                                    )
                            else:
                                # if end date of a constant param is set to yesterday,
                                # it means that it continues even in the future
                                if t >= start_date_constant_i:
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 2:  # 2 fitted + potential constant ones
                    def model_covid_predictions(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2, k_policy_shift_1, k_policy_shift_2
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
                        params_policies_fitted = [k_policy_shift_1, k_policy_shift_2]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if end_date_constant_i != date_yesterday_int:
                                assert end_date_constant_i < date_yesterday_int
                                if (
                                        (t >= start_date_constant_i)
                                        and (t <= end_date_constant_i)
                                ):
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
                                    )
                            else:
                                # if end date of a constant param is set to yesterday,
                                # it means that it continues even in the future
                                if t >= start_date_constant_i:
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
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
                elif N_POLICIES_FITTED_TUPLE == 3:  # 3 fitted + potential constant ones
                    def model_covid_predictions(
                            t, x, alpha, days, r_s, r_dth, p_dth, k1, k2,
                            k_policy_shift_1, k_policy_shift_2, k_policy_shift_3
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
                        params_policies_fitted = [k_policy_shift_1, k_policy_shift_2, k_policy_shift_3]
                        for param_policy_i in range(N_POLICIES_FITTED_TUPLE):
                            if t >= t_start_fitting_policies[param_policy_i]:
                                past_policy = policy_shift_names_fitted[param_policy_i][0]
                                current_new_policy = policy_shift_names_fitted[param_policy_i][1]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    current_new_policy]
                                normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy]
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                    (gamma_t / normalized_gamma_past_policy) * params_policies_fitted[param_policy_i]
                                )

                        for param_policy_constant_i in range(N_POLICIES_CONSTANT_TUPLE):
                            start_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][0]
                            end_date_constant_i = t_start_end_constant_policy[param_policy_constant_i][1]
                            if end_date_constant_i != date_yesterday_int:
                                assert end_date_constant_i < date_yesterday_int
                                if (
                                        (t >= start_date_constant_i)
                                        and (t <= end_date_constant_i)
                                ):
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
                                    )
                            else:
                                # if end date of a constant param is set to yesterday,
                                # it means that it continues even in the future
                                if t >= start_date_constant_i:
                                    past_policy_cst = policy_shift_names_constant[param_policy_constant_i][0]
                                    current_new_policy_cst = policy_shift_names_constant[param_policy_constant_i][1]
                                    normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                        current_new_policy_cst]
                                    normalized_gamma_past_policy = dict_normalized_policy_gamma_countries[past_policy_cst]
                                    gamma_t = gamma_t + min(
                                        (2 - gamma_t) / (1 - normalized_gamma_current_policy),
                                        (gamma_t / normalized_gamma_past_policy) * params_policies_constant[
                                            param_policy_constant_i]
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
                else:
                    raise NotImplementedError(f"Only Implemented up to 3 fitted parameters, got {N_POLICIES_FITTED_TUPLE}")

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

                print(f"Optimized Fitted Params: {best_params}")
                x_sol_final = solve_best_params_and_predict(best_params)
                if len(best_params) > n_params_without_policy_params:
                    best_params_base = best_params[:n_params_without_policy_params]
                    best_params_policies = best_params[n_params_without_policy_params:]
                else:
                    best_params_base = best_params
                    best_params_policies = []
                # Creating the parameters dataset for this (Continent, Country, Province)
                #print(fitcasesnd)
                #print("Historical Cases:", x_sol_final[15, :])
                #print("Historical Deaths:", x_sol_final[14, :])
                mape_data = round((
                        mape(fitcasesnd, x_sol_final[15, :len(fitcasesnd)]) +
                        mape(fitcasesd, x_sol_final[14, :len(fitcasesd)])
                ) / 2, 3)
                try:
                    mape_data_last_15 = round((
                              mape(fitcasesnd[-15:],
                                   x_sol_final[15, len(fitcasesnd) - 15:len(fitcasesnd)]) +
                              mape(fitcasesd[-15:], x_sol_final[14, len(fitcasesnd) - 15:len(fitcasesd)])
                      ) / 2, 3)
                except:
                    mape_data_last_15 = "Less than 15 historical points"
                print(
                    "\t Total MAPE=", mape_data,
                    "\t MAPE on last 15 days=", mape_data_last_15
                )
                df_temp_policy_change_tracking_tuple_updated = update_tracking_fitted_params(
                    df_updated=df_temp_policy_change_tracking_tuple_updated,
                    n_policy_shifts_fitted=N_POLICIES_FITTED_TUPLE,
                    new_best_params_fitted_policies=best_params_policies,
                )
                # Then we add it to the list of df to be concatenated to update the tracking df
                list_df_policy_tracking_concat.append(df_temp_policy_change_tracking_tuple_updated)
                # Creating the datasets for predictions of this (Continent, Country, Province)
                data_creator = DELPHIDataCreator(
                    x_sol_final=x_sol_final, date_day_since100=date_day_since100, best_params=best_params_base,
                    continent=continent, country=country, province=province,
                )
                df_parameters_cont_country_prov = data_creator.create_dataset_parameters(mape_data_last_15)
                list_df_global_parameters.append(df_parameters_cont_country_prov)
                # Creating the datasets for predictions of this (Continent, Country, Province)
                df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                    data_creator.create_datasets_predictions()
                )
                list_df_global_predictions_since_today.append(df_predictions_since_today_cont_country_prov)
                list_df_global_predictions_since_100_cases.append(df_predictions_since_100_cont_country_prov)
                print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")
                print("###########################################################################################")
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
    df_tracking.to_csv(
        PATH_TO_DATA_SANDBOX + f"policy_change_tracking_world_updated_final.csv",
        index=False
    )
    today_date_str = "".join(str(datetime.now().date()).split("-"))
    day_after_yesterday_date_str = "".join(str((pd.to_datetime(yesterday) + timedelta(days=1)).date()).split("-"))
    df_global_parameters_continuous_retraining = pd.concat(list_df_global_parameters).reset_index(drop=True)
    # The code below is commented out but can be used if adding a country from scratch (ie April 28) in DELPHI 3.0+CR
    # df_global_parameters_continuous_retraining = pd.concat([
    #     pd.read_csv(
    #         PATH_TO_FOLDER_DANGER_MAP +
    #         f"predicted/parameters_global_CR_all/Parameters_Global_CR_{day_after_yesterday_date_str}_newtry.csv"
    #     ),
    #     df_global_parameters_continuous_retraining
    # ]).sort_values(["Continent", "Country", "Province"]).reset_index(drop=True)
    df_global_parameters_continuous_retraining.to_csv(
        PATH_TO_FOLDER_DANGER_MAP +
        f"predicted/parameters_global_CR_all/Parameters_Global_CR_{day_after_yesterday_date_str}.csv",
        index=False
    )
    df_global_predictions_since_today_scenarios = pd.concat(
        list_df_global_predictions_since_today
    ).reset_index(drop=True)
    df_global_predictions_since_100_cases_scenarios = pd.concat(
        list_df_global_predictions_since_100_cases
    ).reset_index(drop=True)
    df_global_predictions_since_100_cases_scenarios = DELPHIAggregations.append_all_aggregations(
        df_global_predictions_since_100_cases_scenarios
    ).reset_index(drop=True)
    df_global_predictions_since_100_cases_scenarios.to_csv(
        PATH_TO_DATA_SANDBOX +
        f"./predictions_DELPHI_3_continuous_retraining_{day_after_yesterday_date_str}_final.csv",
        index=False
    )
    print("Saved the dataset of updated tracking & predictions in data_sandbox, Parameters_CR_Global in danger_map")
    print("#############################################################")
# TODO: Modify data saver in order to save separately the new parameters dataframe
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
