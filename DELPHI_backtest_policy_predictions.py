import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import json
import yaml
import os
from pandas.core.common import SettingWithCopyWarning
from DELPHI_params import future_policies, future_times
from DELPHI_utils import (
    read_measures_oxford_data, read_policy_data_us_only, mae_and_mape,
    get_normalized_policy_shifts_and_current_policy_us_only,
    get_normalized_policy_shifts_and_current_policy_all_countries
)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
future_policies_filtered = [
    'No_Measure',
    'Restrict_Mass_Gatherings_and_Schools',
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Lockdown'
]


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "hamza"
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
prediction_date = "2020-06-01"
prediction_date_file = "".join(prediction_date.split("-"))
testing_date = "2020-06-15"
testing_date_file = "".join(testing_date.split("-"))
n_days_testing_data = (pd.to_datetime(testing_date) - pd.to_datetime(prediction_date)).days
print(f"Prediction date: {prediction_date} and Testing Date: {testing_date} so " +
      f"{n_days_testing_data} days of testing data in this run")
with open(
        PATH_TO_FOLDER_DANGER_MAP + f'predicted/world_Python_{prediction_date_file}_Scenarios_since_100_cases.json',
        'rb'  # Need to read it as binary
) as json_file:
    dict_scenarios_world = json.load(json_file)

###################################################################
##### Retrieving current policies for all countries/provinces #####
###################################################################
policy_data_countries = read_measures_oxford_data(yesterday=testing_date)
policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=PATH_TO_DATA_SANDBOX)
popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
pastparameters = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_Global_{testing_date_file}.csv")
# True if we use the Mathematica run parameters, False if we use those from Python runs
# This is because the pastparameters dataframe's columns are not in the same order in both cases

# Get the policies shifts from the CART tree to compute different values of gamma(t)
# Depending on the policy in place in the future to affect predictions
dict_normalized_policy_gamma_countries, dict_current_policy_countries = (
    get_normalized_policy_shifts_and_current_policy_all_countries(
        policy_data_countries=policy_data_countries[policy_data_countries.date <= testing_date],
        pastparameters=pastparameters,
    )
)
# Setting same value for these 2 policies because of the inherent structure of the tree
dict_normalized_policy_gamma_countries[future_policies[3]] = dict_normalized_policy_gamma_countries[future_policies[5]]

## US Only Policies
dict_normalized_policy_gamma_us_only, dict_current_policy_us_only = (
    get_normalized_policy_shifts_and_current_policy_us_only(
        policy_data_us_only=policy_data_us_only[policy_data_us_only.date <= testing_date],
        pastparameters=pastparameters,
    )
)
dict_current_policy_international = dict_current_policy_countries.copy()
dict_current_policy_international.update(dict_current_policy_us_only)
df_current_policies = pd.DataFrame({
    "country": [x[0] for x in dict_current_policy_international.keys()],
    "province": [x[1] for x in dict_current_policy_international.keys()],
    "date": [testing_date for _ in range(len(dict_current_policy_international))],
    f"current_policy_{testing_date_file[4:]}": [
        dict_current_policy_international[x]
        for x in dict_current_policy_international.keys()
    ]
})

########################################################################
##### Retrieving previous international policy predictions json file ###
########################################################################
list_df_concat = []
for continent in dict_scenarios_world.keys():
    dict_continent = dict_scenarios_world[continent]
    for country in dict_continent.keys():
        dict_country = dict_continent[country]
        for province in dict_country.keys():
            dict_province = dict_country[province]
            for policy in future_policies:
                dict_country_prov_policy = dict_province[policy]
                for time in ["Now", "One Week", "Two Weeks", "Four Weeks", "Six Weeks"]:
                    dict_country_prov_policy_time = dict_country_prov_policy[time]
                    n = len(dict_country_prov_policy_time["Total Detected"])
                    list_df_concat.append(pd.DataFrame({
                        "continent": [continent for _ in range(n)],
                        "country": [country for _ in range(n)],
                        "province": [province for _ in range(n)],
                        "policy": [policy for _ in range(n)],
                        "timing": [time for _ in range(n)],
                        "date": dict_province["Day"],
                        "Total Detected": dict_country_prov_policy_time["Total Detected"],
                        "Total Detected Deaths": dict_country_prov_policy_time["Total Detected Deaths"],
                    }))

print("Finished parsing the International Policy Predictions JSON file into a dataframe")
df_pred_international_policies = pd.concat(list_df_concat).reset_index(drop=True)
df_pred_international_policies = df_pred_international_policies[
    (df_pred_international_policies.date >= "2020-05-23") &
    (df_pred_international_policies.date <= "2020-06-08")
].reset_index(drop=True)

###########################################################################
##### Retrieving latest historical cases & deaths from processed data #####
###########################################################################
files_cases_historical = [
    x for x in sorted(os.listdir(PATH_TO_FOLDER_DANGER_MAP + "processed/Global/"))
    if "Cases" in x
]
df_historical_world = []
for filename in files_cases_historical:
    df_historical_world.append(pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + "processed/Global/" + filename
    ))
df_historical_world = pd.concat(df_historical_world).reset_index(drop=True)[[
    "country", "province", "date", "day_since100", "case_cnt", "death_cnt"
]]
df_historical_world.province.fillna("None", inplace=True)
df_historical_world = df_historical_world[
    df_historical_world.date <= testing_date
].reset_index(drop=True)

#########################################################################
##### Merging both historical ground truth and previous predictions #####
#########################################################################
df_pred_international_policies = df_pred_international_policies.merge(
    df_historical_world,
    on=["country", "province", "date"]
)
df_pred_international_policies["tuple"] = list(zip(
    df_pred_international_policies.country,
    df_pred_international_policies.province
))

##### Adding MAPE #####
df_mape = []
for tuple_value in df_pred_international_policies.tuple.unique():
    country, province = tuple_value
    for policy in future_policies:
        for time in ["Now", "One Week", "Two Weeks", "Four Weeks", "Six Weeks"]:
            df_temp = df_pred_international_policies[
                (df_pred_international_policies.policy == policy) &
                (df_pred_international_policies.timing == time) &
                (df_pred_international_policies.country == country) &
                (df_pred_international_policies.province == province)
            ]
            mae_cases, mape_cases = mae_and_mape(
                y_true=np.array(df_temp["case_cnt"]),
                y_pred=np.array(df_temp["Total Detected"])
            )
            mae_deaths, mape_deaths = mae_and_mape(
                y_true=np.array(df_temp["death_cnt"]),
                y_pred=np.array(df_temp["Total Detected Deaths"])
            )

            df_mape.append(pd.DataFrame({
                "country": [country],
                "province": [province],
                "tuple": [(country, province)],
                "policy": [policy],
                "timing": [time],
                "mape_cases": [mape_cases],
                "mape_deaths": [mape_deaths],
            }))

print("Finished Creating MAPE Columns")
df_mape = pd.concat(df_mape).reset_index(drop=True)
df_mape = df_mape[
    df_mape.timing.isin(["Now", "One Week", "Two Weeks"])
].reset_index(drop=True)
df_mape_filtered = df_mape[
    (df_mape.policy.isin(future_policies_filtered)) &
    (df_mape.timing == "Now")
]
df_mape_filtered_full = df_mape_filtered.merge(
    df_current_policies.drop("date", axis=1),
    on=["country", "province"],
    how="left"
)
# Obtaining the minimal MAPE on Cases
df_min_mape_cases = []
for tuple_cp in df_mape_filtered_full.tuple.unique():
    df_temp = df_mape_filtered_full[
        df_mape_filtered_full.tuple == tuple_cp
    ]
    df_temp = df_temp[df_temp.mape_cases == df_temp.mape_cases.min()]
    df_temp["best_cases_policy_same_as_current_policy"] = (
        df_temp.policy == df_temp[f"current_policy_{testing_date_file[4:]}"]
    )
    df_min_mape_cases.append(df_temp)

df_min_mape_cases = pd.concat(df_min_mape_cases).reset_index(drop=True).rename(columns={
    "policy": "best_policy_cases",
    "mape_cases": "best_mape_cases_c",
    "mape_deaths": "best_mape_deaths_c"
})[[
    "country", "province", "best_policy_cases", "best_mape_cases_c", "best_mape_deaths_c",
    f"current_policy_{testing_date_file[4:]}", "best_cases_policy_same_as_current_policy"
]]
df_min_mape_cases.drop_duplicates(subset=["country", "province"], inplace=True)

# Obtaining the minimal MAPE on Cases
df_min_mape_deaths = []
for tuple_cp in df_mape_filtered_full.tuple.unique():
    df_temp = df_mape_filtered_full[
        df_mape_filtered_full.tuple == tuple_cp
    ]
    df_temp = df_temp[df_temp.mape_deaths == df_temp.mape_deaths.min()]
    df_temp["best_deaths_policy_same_as_current_policy"] = (
        df_temp.policy == df_temp[f"current_policy_{testing_date_file[4:]}"]
    )
    df_min_mape_deaths.append(df_temp)

df_min_mape_deaths = pd.concat(df_min_mape_deaths).reset_index(drop=True).rename(columns={
    "policy": "best_policy_deaths",
    "mape_cases": "best_mape_cases_d",
    "mape_deaths": "best_mape_deaths_d"
})[[
    "country", "province", "best_policy_deaths", "best_mape_cases_d", "best_mape_deaths_d",
    "best_deaths_policy_same_as_current_policy"
]]
df_min_mape_deaths.drop_duplicates(subset=["country", "province"], inplace=True)

# Merging the df of min for cases & for deaths into one (columns are distinguishable)
df_min_all = df_min_mape_cases.merge(df_min_mape_deaths, on=["country", "province"])
df_min_all = df_min_all[[
    "country", "province", f"current_policy_{testing_date_file[4:]}", "best_policy_cases", "best_policy_deaths",
    "best_cases_policy_same_as_current_policy", "best_deaths_policy_same_as_current_policy",
    "best_mape_cases_c", "best_mape_deaths_c", "best_mape_cases_d", "best_mape_deaths_d",
]]
for col in ["best_mape_cases_c", "best_mape_deaths_c", "best_mape_cases_d", "best_mape_deaths_d",]:
    df_min_all[col] = df_min_all[col].round(3)

df_min_all.to_csv(
    f"./backtesting/{testing_date_file}_backtest" +
    f"_policy_predictions_pred_from_{prediction_date_file}.csv",
    index=False
)
print(f"Finished performing backtesting from {prediction_date} to {testing_date}")
