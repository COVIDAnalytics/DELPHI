# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from DELPHI_utils import (
    read_measures_oxford_data, get_normalized_policy_shifts_and_current_policy_all_countries,
    get_normalized_policy_shifts_and_current_policy_us_only, read_policy_data_us_only
)
from DELPHI_policies_utils import (
    update_tracking_when_policy_changed, update_tracking_without_policy_change,
    add_policy_tracking_row_country
)
from DELPHI_params import (
    date_MATHEMATICA, future_policies
)
import yaml
import os


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "hamza"
max_days_before = (datetime.now().date() - datetime(2020, 4, 27).date()).days - 1
for n_days_before in range(max_days_before, max_days_before - 1, -1):
    yesterday = "".join(str(datetime.now().date() - timedelta(days=n_days_before)).split("-"))
    print(yesterday)
    PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
    PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["danger_map"]["michael"]
    policy_data_countries = read_measures_oxford_data(yesterday=yesterday)
    policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING])
    popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
    try:
        pastparameters = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{yesterday}.csv")
    except:
        print(f"No pastparameters for {yesterday}")
        continue
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
    if os.path.exists(CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] + f"policy_change_tracking_world_init_2704.csv"):
        POLICY_TRACKING_ALREADY_EXISTS = True
        df_policy_change_tracking_world = pd.read_csv(
            CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] + f"policy_change_tracking_world_init_2704.csv"
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
        #if province not in ["Georgia", "Tennessee"]:
        #    continue
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
                raise ValueError()
            continue

        else:  # file for that tuple (country, province) doesn't exist in processed files
            continue

    # Appending parameters, aggregations per country, per continent, and for the world
    # for predictions today & since 100
    df_tracking = pd.concat([df_policy_change_tracking_world] + list_df_policy_tracking_concat).sort_values(
        by=["continent", "country", "province", "date"]
    ).reset_index(drop=True)
    df_tracking.to_csv(CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING] +
                       f"policy_change_tracking_world_updated_newtry.csv", index=False)
    print("Saved the dataset of tracking")
