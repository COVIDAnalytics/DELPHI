# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union
from copy import deepcopy
from itertools import compress
import json

time_dict = {0: "Now", 7: "One Week", 14: "Two Weeks", 28: "Four Weeks", 42: "Six Weeks"}


class DELPHIDataSaver:
    def __init__(
            self, path_to_folder_danger_map: str,
            path_to_website_predicted: str,
            df_global_parameters: Union[pd.DataFrame, None],
            df_global_predictions_since_today: pd.DataFrame,
            df_global_predictions_since_100_cases: pd.DataFrame,
    ):
        self.PATH_TO_FOLDER_DANGER_MAP = path_to_folder_danger_map
        self.PATH_TO_WEBSITE_PREDICTED = path_to_website_predicted
        self.df_global_parameters = df_global_parameters
        self.df_global_predictions_since_today = df_global_predictions_since_today
        self.df_global_predictions_since_100_cases = df_global_predictions_since_100_cases

    def save_all_datasets(self, save_since_100_cases=False, website=False):
        today_date_str = "".join(str(datetime.now().date()).split("-"))
        # Save parameters
        self.df_global_parameters.to_csv(
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_{today_date_str}.csv", index=False
        )
        self.df_global_parameters.to_csv(
            self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Parameters_Global_{today_date_str}.csv", index=False
        )
        # Save predictions since today
        self.df_global_predictions_since_today.to_csv(
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Global_{today_date_str}.csv", index=False
        )
        self.df_global_predictions_since_today.to_csv(
            self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_{today_date_str}.csv", index=False
        )
        if website:
            self.df_global_parameters.to_csv(
                self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Parameters_Global_Python_{today_date_str}.csv",
                index=False
            )
            self.df_global_predictions_since_today.to_csv(
                self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_Python_{today_date_str}_Scenarios.csv",
                index=False
            )
            self.df_global_predictions_since_today.to_csv(
                self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_Python_Scenarios.csv", index=False
            )
        if save_since_100_cases:
            # Save predictions since 100 cases
            self.df_global_predictions_since_100_cases.to_csv(
                self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Global_since100_{today_date_str}.csv", index=False
            )
            self.df_global_predictions_since_100_cases.to_csv(
                self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_since100_{today_date_str}.csv", index=False
            )
            if website:
                self.df_global_predictions_since_100_cases.to_csv(
                    self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_since100_{today_date_str}_Scenarios.csv",
                    index=False
                )
                self.df_global_predictions_since_100_cases.to_csv(
                    self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_since100_Scenarios.csv", index=False
                )

    @staticmethod
    def create_nested_dict_from_final_dataframe(df_predictions: pd.DataFrame) -> dict:
        dict_all_results = {}
        default_policy = "Lockdown"
        default_policy_enaction_time = "Now"
        for province in df_predictions.Province.unique():
            df_predictions_province = df_predictions[df_predictions.Province == province].reset_index(drop=True)
            dict_all_results[province] = {
                "Day": sorted(list(df_predictions_province.Day.unique())),
                "Total Detected True": df_predictions_province[
                    (df_predictions_province.Policy == default_policy)
                    & (df_predictions_province.Time == default_policy_enaction_time)
                    ].sort_values("Day")["Total Detected True"].tolist(),
                "Total Detected Deaths True": df_predictions_province[
                    (df_predictions_province.Policy == default_policy)
                    & (df_predictions_province.Time == default_policy_enaction_time)
                    ].sort_values("Day")["Total Detected Deaths True"].tolist(),
            }
            dict_all_results[province].update({
                policy: {
                    policy_enaction_time: {
                        "Total Detected": df_predictions_province[
                            (df_predictions_province.Policy == policy)
                            & (df_predictions_province.Time == policy_enaction_time)
                        ].sort_values("Day")["Total Detected"].tolist(),
                        "Total Detected Deaths": df_predictions_province[
                            (df_predictions_province.Policy == policy)
                            & (df_predictions_province.Time == policy_enaction_time)
                        ].sort_values("Day")["Total Detected Deaths"].tolist(),
                    }
                    for policy_enaction_time in df_predictions_province.Time.unique()
                }
                for policy in df_predictions_province.Policy.unique()
            })

        return dict_all_results

    def save_policy_predictions_to_dict_pickle(self, website=False):
        today_date_str = "".join(str(datetime.now().date()).split("-"))
        dict_predictions_policies_US_since_100_cases = DELPHIDataSaver.create_nested_dict_from_final_dataframe(
            self.df_global_predictions_since_100_cases
        )
        with open(
                self.PATH_TO_FOLDER_DANGER_MAP +
                f'/predicted/US_Python_{today_date_str}_Scenarios_since_100_cases.json', 'w'
        ) as handle:
            json.dump(dict_predictions_policies_US_since_100_cases, handle)

        with open(
                self.PATH_TO_FOLDER_DANGER_MAP + f'/predicted/US_Python_Scenarios_since_100_cases.json', 'w'
        ) as handle:
            json.dump(dict_predictions_policies_US_since_100_cases, handle)

        if website:
            with open(
                self.PATH_TO_WEBSITE_PREDICTED +
                f'/predicted/US_Python_{today_date_str}_Scenarios_since_100_cases.json', 'w'
            ) as handle:
                json.dump(dict_predictions_policies_US_since_100_cases, handle)

            with open(
                    self.PATH_TO_WEBSITE_PREDICTED + f'/predicted/US_Python_Scenarios_since_100_cases.json', 'w'
            ) as handle:
                json.dump(dict_predictions_policies_US_since_100_cases, handle)


class DELPHIDataCreator:
    def __init__(
            self, x_sol_final: np.array, date_day_since100: datetime,
            best_params: np.array, continent: str, country: str, province: str,
    ):
        assert len(best_params) == 7, f"Expected 7 best parameters, got {len(best_params)}"
        self.x_sol_final = x_sol_final
        self.date_day_since100 = date_day_since100
        self.best_params = best_params
        self.continent = continent
        self.country = country
        self.province = province

    def create_dataset_parameters(self, mape) -> pd.DataFrame:
        df_parameters = pd.DataFrame({
            "Continent": [self.continent], "Country": [self.country], "Province": [self.province],
            "Data Start Date": [self.date_day_since100], "MAPE": [mape], "Infection Rate": [self.best_params[0]],
            "Median Day of Action": [self.best_params[1]], "Rate of Action": [self.best_params[2]],
            "Rate of Death": [self.best_params[3]], "Mortality Rate": [self.best_params[4]],
            "Internal Parameter 1": [self.best_params[5]], "Internal Parameter 2": [self.best_params[6]],
        })
        return df_parameters

    def create_datasets_predictions(self) -> (pd.DataFrame, pd.DataFrame):
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        # Predictions
        total_detected = self.x_sol_final[15, :]  # DT
        total_detected = [round(x, 0) for x in total_detected]
        active_cases = (
                self.x_sol_final[4, :] + self.x_sol_final[5, :] + self.x_sol_final[7, :] + self.x_sol_final[8, :]
        )  # DHR + DQR + DHD + DQD
        active_cases = [round(x, 0) for x in active_cases]
        active_hospitalized = self.x_sol_final[4, :] + self.x_sol_final[7, :]  # DHR + DHD
        active_hospitalized = [round(x, 0) for x in active_hospitalized]
        cumulative_hospitalized = self.x_sol_final[11, :]  # TH
        cumulative_hospitalized = [round(x, 0) for x in cumulative_hospitalized]
        total_detected_deaths = self.x_sol_final[14, :]  # DD
        total_detected_deaths = [round(x, 0) for x in total_detected_deaths]
        active_ventilated = self.x_sol_final[12, :] + self.x_sol_final[13, :]  # DVR + DVD
        active_ventilated = [round(x, 0) for x in active_ventilated]
        # Generation of the dataframe since today
        df_predictions_since_today_cont_country_prov = pd.DataFrame({
            "Continent": [self.continent for _ in range(n_days_since_today)],
            "Country": [self.country for _ in range(n_days_since_today)],
            "Province": [self.province for _ in range(n_days_since_today)],
            "Day": all_dates_since_today,
            "Total Detected": total_detected[n_days_btw_today_since_100:],
            "Active": active_cases[n_days_btw_today_since_100:],
            "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
            "Cumulative Hospitalized": cumulative_hospitalized[n_days_btw_today_since_100:],
            "Total Detected Deaths": total_detected_deaths[n_days_btw_today_since_100:],
            "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
        })

        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((self.date_day_since100 + timedelta(days=i)).date())
            for i in range(self.x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame({
            "Continent": [self.continent for _ in range(len(all_dates_since_100))],
            "Country": [self.country for _ in range(len(all_dates_since_100))],
            "Province": [self.province for _ in range(len(all_dates_since_100))],
            "Day": all_dates_since_100,
            "Total Detected": total_detected,
            "Active": active_cases,
            "Active Hospitalized": active_hospitalized,
            "Cumulative Hospitalized": cumulative_hospitalized,
            "Total Detected Deaths": total_detected_deaths,
            "Active Ventilated": active_ventilated,
        })
        return df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov

    def create_datasets_predictions_scenario(
            self, policy="Lockdown", time=0, totalcases=None,
    ) -> (pd.DataFrame, pd.DataFrame):
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        # Predictions
        total_detected = self.x_sol_final[15, :]  # DT
        total_detected = [round(x, 0) for x in total_detected]
        active_cases = (
                self.x_sol_final[4, :] + self.x_sol_final[5, :] + self.x_sol_final[7, :] + self.x_sol_final[8, :]
        )  # DHR + DQR + DHD + DQD
        active_cases = [round(x, 0) for x in active_cases]
        active_hospitalized = self.x_sol_final[4, :] + self.x_sol_final[7, :]  # DHR + DHD
        active_hospitalized = [round(x, 0) for x in active_hospitalized]
        cumulative_hospitalized = self.x_sol_final[11, :]  # TH
        cumulative_hospitalized = [round(x, 0) for x in cumulative_hospitalized]
        total_detected_deaths = self.x_sol_final[14, :]  # DD
        total_detected_deaths = [round(x, 0) for x in total_detected_deaths]
        active_ventilated = self.x_sol_final[12, :] + self.x_sol_final[13, :]  # DVR + DVD
        active_ventilated = [round(x, 0) for x in active_ventilated]
        # Generation of the dataframe since today
        df_predictions_since_today_cont_country_prov = pd.DataFrame({
            "Policy": [policy for _ in range(n_days_since_today)],
            "Time": [time_dict[time] for _ in range(n_days_since_today)],
            "Continent": [self.continent for _ in range(n_days_since_today)],
            "Country": [self.country for _ in range(n_days_since_today)],
            "Province": [self.province for _ in range(n_days_since_today)],
            "Day": all_dates_since_today,
            "Total Detected": total_detected[n_days_btw_today_since_100:],
            "Active": active_cases[n_days_btw_today_since_100:],
            "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
            "Cumulative Hospitalized": cumulative_hospitalized[n_days_btw_today_since_100:],
            "Total Detected Deaths": total_detected_deaths[n_days_btw_today_since_100:],
            "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
        })

        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((self.date_day_since100 + timedelta(days=i)).date())
            for i in range(self.x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame({
            "Policy": [policy for _ in range(len(all_dates_since_100))],
            "Time": [time_dict[time] for _ in range(len(all_dates_since_100))],
            "Continent": [self.continent for _ in range(len(all_dates_since_100))],
            "Country": [self.country for _ in range(len(all_dates_since_100))],
            "Province": [self.province for _ in range(len(all_dates_since_100))],
            "Day": all_dates_since_100,
            "Total Detected": total_detected,
            "Active": active_cases,
            "Active Hospitalized": active_hospitalized,
            "Cumulative Hospitalized": cumulative_hospitalized,
            "Total Detected Deaths": total_detected_deaths,
            "Active Ventilated": active_ventilated,
        })
        if totalcases is not None:  # Merging the historical values to both dataframes when available
            df_predictions_since_today_cont_country_prov = df_predictions_since_today_cont_country_prov.merge(
                totalcases[["country", "province", "date", "case_cnt", "death_cnt"]],
                left_on=["Country", "Province", "Day"],
                right_on=["country", "province", "date"],
                how="left",
            )
            df_predictions_since_today_cont_country_prov.rename(
                columns={"case_cnt": "Total Detected True", "death_cnt": "Total Detected Deaths True"},
                inplace=True,
            )
            df_predictions_since_today_cont_country_prov.drop(
                ["country", "province", "date"], axis=1, inplace=True
            )
            df_predictions_since_100_cont_country_prov = df_predictions_since_100_cont_country_prov.merge(
                totalcases[["country", "province", "date", "case_cnt", "death_cnt"]],
                left_on=["Country", "Province", "Day"],
                right_on=["country", "province", "date"],
                how="left",
            )
            df_predictions_since_100_cont_country_prov.rename(
                columns={"case_cnt": "Total Detected True", "death_cnt": "Total Detected Deaths True"},
                inplace=True,
            )
            df_predictions_since_100_cont_country_prov.drop(
                ["country", "province", "date"], axis=1, inplace=True
            )
        return df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov


class DELPHIAggregations:
    @staticmethod
    def get_aggregation_per_country(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Province"] != "None"]
        df_agg_country = df.groupby(["Continent", "Country", "Day"]).sum().reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_country

    @staticmethod
    def get_aggregation_per_continent(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_continent = df.groupby(["Continent", "Day"]).sum().reset_index()
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_continent

    @staticmethod
    def get_aggregation_world(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_world = df.groupby("Day").sum().reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_world

    @staticmethod
    def append_all_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(df)
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent(df)
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world(df)
        df = pd.concat([
            df, df_agg_since_today_per_country,
            df_agg_since_today_per_continent, df_agg_since_today_world
        ])
        df.sort_values(["Continent", "Country", "Province", "Day"], inplace=True)
        return df


class DELPHIAggregations:
    @staticmethod
    def get_aggregation_per_country(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Province"] != "None"]
        df_agg_country = df.groupby(["Continent", "Country", "Day"]).sum().reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_country

    @staticmethod
    def get_aggregation_per_continent(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_continent = df.groupby(["Continent", "Day"]).sum().reset_index()
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_continent

    @staticmethod
    def get_aggregation_world(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_world = df.groupby("Day").sum().reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[[
            'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_world

    @staticmethod
    def append_all_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(df)
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent(df)
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world(df)
        df = pd.concat([
            df, df_agg_since_today_per_country,
            df_agg_since_today_per_continent, df_agg_since_today_world
        ])
        df.sort_values(["Continent", "Country", "Province", "Day"], inplace=True)
        return df


class DELPHIAggregationsPolicies:
    @staticmethod
    def get_aggregation_per_country(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Province"] != "None"]
        df_agg_country = df.groupby(["Policy", "Time", "Continent", "Country", "Day"]).sum().reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[[
        'Policy', 'Time', 'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_country

    @staticmethod
    def get_aggregation_per_continent(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_continent = df.groupby(["Policy", "Time", "Continent", "Day"]).sum().reset_index()
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[[
            'Policy', 'Time', 'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_continent

    @staticmethod
    def get_aggregation_world(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_world = df.groupby(["Policy", "Time", "Day"]).sum().reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[[
            'Policy', 'Time', 'Continent', 'Country', 'Province', 'Day', 'Total Detected', 'Active',
            'Active Hospitalized', 'Cumulative Hospitalized', 'Total Detected Deaths', 'Active Ventilated'
        ]]
        return df_agg_world

    @staticmethod
    def append_all_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(df)
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent(df)
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world(df)
        df = pd.concat([
            df, df_agg_since_today_per_country,
            df_agg_since_today_per_continent, df_agg_since_today_world
        ])
        df.sort_values(["Policy", "Time", "Continent", "Country", "Province", "Day"], inplace=True)
        return df


def get_initial_conditions(params_fitted, global_params_fixed):
    alpha, days, r_s, r_dth, p_dth, k1, k2 = params_fitted
    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed
    S_0 = (
            (N - PopulationCI / p_d) -
            (PopulationCI / p_d * (k1 + k2)) -
            (PopulationR / p_d) -
            (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    AR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    AD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
    DHD_0 = PopulationCI * p_h * p_dth
    DQD_0 = PopulationCI * (1 - p_h) * p_dth
    R_0 = PopulationR / p_d
    D_0 = PopulationD / p_d
    TH_0 = PopulationCI * p_h
    DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth)
    DVD_0 = (PopulationCI * p_h * p_v) * p_dth
    DD_0 = PopulationD
    DT_0 = PopulationI
    x_0_cases = [
        S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0,
        R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0
    ]
    return x_0_cases


def create_fitting_data_from_validcases(validcases):
    validcases_nondeath = validcases["case_cnt"].tolist()
    validcases_death = validcases["death_cnt"].tolist()
    balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
    fitcasesnd = validcases_nondeath
    fitcasesd = validcases_death
    return balance, fitcasesnd, fitcasesd


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100


def convert_dates_us_policies(x):
    if x == "Not implemented":
        return np.nan
    else:
        x_long = x + "20"
        return pd.to_datetime(x_long, format="%d-%b-%Y")


def read_policy_data_us_only():
    data_path = (
        "E:/Github/DELPHI/data_sandbox"
        # "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
        # "4. COVID19_Global/DELPHI/data_sandbox"
    )
    df = pd.read_csv(data_path + "/25042020_raw_policy_data_US_only.csv")
    df.State = df.State.apply(lambda x: x[0].upper() + x[1:])
    concat_data = []
    for i, measure in enumerate(df.Measure.unique()):
        df_temp = df[df.Measure == measure].reset_index(drop=True)
        df_concat = pd.DataFrame({
            f"province_{i}": df_temp.State,
            f"{measure}": df_temp.Date
        })
        concat_data.append(df_concat)

    df_format = pd.concat(concat_data, axis=1)
    df_format.drop(
        [f"province_{i}" for i in range(1, len(df.Measure.unique()))],
        axis=1, inplace=True
    )
    df_format.columns = ["province"] + list(df_format.columns)[1:]
    for col in list(df_format.columns)[1:]:
        df_format[col] = df_format[col].apply(
            lambda x: convert_dates_us_policies(x)
        )

    n_dates = (datetime.now() - datetime(2020, 3, 1)).days + 1
    list_all_dates = [
        datetime(2020, 3, 1) + timedelta(days=i)
        for i in range(n_dates)
    ]
    df_format["province"] = df_format.province.replace({
        "District-of-columbia": "District of Columbia",
        "New-york": "New York", "North-carolina": "North Carolina",
        "North-dakota": "North Dakota", "Rhode-island": "Rhode Island",
        "South-carolina": "South Carolina", "South-dakota": "South Dakota",
        "West-virginia": "West Virginia", "New-jersey": "New Jersey",
        "New-hampshire": "New Hampshire", "New-mexico": "New Mexico",
    })
    df_to_concat_final = []
    for i, province in enumerate(df_format.province.unique()):
        df_temp = df_format[
            df_format.province == province
            ].reset_index(drop=True)
        columns_to_iter = [
            "Mass_Gathering_Restrictions", "Initial_Business_Closure",
            "Educational_Facilities_Closed", "Non_Essential_Services_Closed",
            "Stay_at_home_order"
        ]
        df_i = pd.DataFrame({
            "province": [province for _ in range(n_dates)],
            "date": list_all_dates,
            "Mass_Gathering_Restrictions": [0 for _ in range(n_dates)],
            "Initial_Business_Closure": [0 for _ in range(n_dates)],
            "Educational_Facilities_Closed": [0 for _ in range(n_dates)],
            "Non_Essential_Services_Closed": [0 for _ in range(n_dates)],
            "Stay_at_home_order": [0 for _ in range(n_dates)],
            "Travel_severely_limited": [0 for _ in range(n_dates)],
        })
        date_mgr = df_temp.iloc[0, 1]
        date_ibc = df_temp.iloc[0, 2]
        date_efc = df_temp.iloc[0, 3]
        date_nesc = df_temp.iloc[0, 4]
        date_saho = df_temp.iloc[0, 5]
        # No date_tsl as no state actually implemented it
        for col, date_col in zip(
                columns_to_iter,
                [date_mgr, date_ibc, date_efc, date_nesc, date_saho]
        ):
            df_i.loc[df_i["date"] >= date_col, col] = 1
        df_to_concat_final.append(df_i)

    df_final = pd.concat(df_to_concat_final)
    df_final.reset_index(drop=True, inplace=True)
    output = deepcopy(df_final)
    msr = ['No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
           'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
           'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown']
    output['No_Measure'] = (df_final.sum(axis=1) == 0).apply(lambda x: int(x))
    output['Restrict_Mass_Gatherings'] = [int(a and b) for a, b in
                                      zip(df_final.sum(axis=1) == 1, df_final['Mass_Gathering_Restrictions'] == 1)]
    output['Mass_Gatherings_Authorized_But_Others_Restricted'] = [
        int(a and b and c) for a, b, c in zip(
            df_final.sum(axis=1) > 0,
            df_final['Mass_Gathering_Restrictions'] == 0,
            df_final['Stay_at_home_order'] == 0,
        )
    ]
    output['Restrict_Mass_Gatherings_and_Schools'] = [
        int(a and b and c)
        for a, b, c in zip(
            df_final.sum(axis=1) == 2,
            df_final['Educational_Facilities_Closed'] == 1,
            df_final['Mass_Gathering_Restrictions'] == 1,
        )
    ]
    output['Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others'] = [
        int(a and b and c and d) for a, b, c, d in zip(
            df_final.sum(axis=1) > 1,
            df_final['Educational_Facilities_Closed'] == 0,
            df_final['Mass_Gathering_Restrictions'] == 1,
            df_final['Stay_at_home_order'] == 0,
        )
    ]
    output['Restrict_Mass_Gatherings_and_Schools_and_Others'] = [
        int(a and b and c and d) for a, b, c, d in zip(
            df_final.sum(axis=1) > 2,
            df_final['Educational_Facilities_Closed'] == 1,
            df_final['Mass_Gathering_Restrictions'] == 1,
            df_final['Stay_at_home_order'] == 0,
        )
    ]
    output['Lockdown'] = (df_final['Stay_at_home_order'] == 1).apply(lambda x: int(x))
    output['country'] = "US"
    output = output.loc[:, ['country', 'province', 'date'] + msr]
    return output


def read_measures_oxford_data():
    measures = pd.read_csv('https://ocgptweb.azurewebsites.net/CSVDownload')
    filtr = ['CountryName', 'CountryCode', 'Date']
    target = ['ConfirmedCases', 'ConfirmedDeaths']
    msr = ['S1_School closing',
           'S2_Workplace closing', 'S3_Cancel public events',
           'S4_Close public transport',
           'S5_Public information campaigns',
           'S6_Restrictions on internal movement',
           'S7_International travel controls', 'S8_Fiscal measures',
           'S9_Monetary measures',
           'S10_Emergency investment in health care',
           'S11_Investment in Vaccines']
    measures = measures.loc[:, filtr + msr + target]
    measures['Date'] = measures['Date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    for col in target:
        measures[col] = measures[col].fillna(0)
    measures = measures.loc[:, measures.isnull().mean() < 0.1]
    msr = set(measures.columns).intersection(set(msr))
    measures = measures.fillna(0)
    for col in msr:
        measures[col] = measures[col].apply(lambda x: int(x > 0))
    measures = measures[[
        'CountryName', 'Date', 'S1_School closing', 'S2_Workplace closing', 'S3_Cancel public events',
        'S4_Close public transport', 'S5_Public information campaigns',
        'S6_Restrictions on internal movement', 'S7_International travel controls'
    ]]
    measures["CountryName"] = measures.CountryName.replace({
        "United States": "US", "South Korea": "Korea, South", "Democratic Republic of Congo": "Congo (Kinshasa)",
        "Czech Republic": "Czechia", "Slovak Republic": "Slovakia",
    })
    return measures


def gamma_t(day, state, params_dic):
    dsd, median_day_of_action, rate_of_action = params_dic[state]
    t = (day - pd.to_datetime(dsd)).days
    gamma = (2 / np.pi) * np.arctan(-(t - median_day_of_action) / 20 * rate_of_action) + 1
    return gamma


def get_normalized_policy_shifts_and_current_policy(
        policy_data_us_only: pd.DataFrame,
        pastparameters: pd.DataFrame,
):
    dict_last_policy = {}
    policy_list = ['No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
                   'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
                   'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown']
    policy_data_us_only['province_cl'] = policy_data_us_only['province'].apply(
        lambda x: x.replace(',', '').strip().lower()
    )
    states_upper_set = set(policy_data_us_only['province'])
    for state in states_upper_set:
        dict_last_policy[state] = list(compress(policy_list, (policy_data_us_only.query('province == @state')[
                                                                 policy_data_us_only.query('province == @state')[
                                                                     "date"] == policy_data_us_only.date.max()][
                                                                 policy_list] == 1).values.flatten().tolist()))[0]
    states_set = set(policy_data_us_only['province_cl'])
    pastparameters_copy = deepcopy(pastparameters)
    pastparameters_copy['Province'] = pastparameters_copy['Province'].apply(
        lambda x: str(x).replace(',', '').strip().lower())
    params_dic = {}
    for state in states_set:
        params_dic[state] = pastparameters_copy.query('Province == @state')[
            ['Data Start Date', 'Median Day of Action', 'Rate of Action']].iloc[0]

    policy_data_us_only['Gamma'] = [gamma_t(day, state, params_dic) for day, state in
                                    zip(policy_data_us_only['date'], policy_data_us_only['province_cl'])]
    n_measures = policy_data_us_only.iloc[:, 3:-2].shape[1]
    dict_policies_shift = {
        policy_data_us_only.columns[3 + i]: policy_data_us_only[
                                                policy_data_us_only.iloc[:, 3 + i] == 1
                                            ].iloc[:, -1].mean()
        for i in range(n_measures)
    }
    normalize_val = dict_policies_shift["No_Measure"]
    for policy in dict_policies_shift.keys():
        dict_policies_shift[policy] = dict_policies_shift[policy] / normalize_val

    return dict_policies_shift, dict_last_policy


def add_aggregations_backtest(df_backtest_performance: pd.DataFrame) -> pd.DataFrame:
    df_temp = df_backtest_performance.copy()
    df_temp_continent = df_temp.groupby("continent")[
        "train_mape_cases", "train_mape_deaths", "test_mape_cases", "test_mape_deaths"
    ].mean().reset_index()
    df_temp_country = df_temp.groupby(["continent", "country"])[
        "train_mape_cases", "train_mape_deaths", "test_mape_cases", "test_mape_deaths"
    ].mean().reset_index()
    columns_nan = ["country", "province", "train_start_date", "train_end_date", "test_start_date", "test_end_date"]
    for col in columns_nan:
        df_temp_continent[col] = "None"
    for col in columns_nan[1:]:
        df_temp_country[col] = "None"

    all_columns = [
        "continent", "country", "province", "train_start_date", "train_end_date",
        "train_mape_cases", "train_mape_deaths", "test_start_date", "test_end_date",
        "test_mape_cases", "test_mape_deaths"
    ]
    df_temp_continent = df_temp_continent[all_columns]
    df_temp_country = df_temp_country[all_columns]
    df_backtest_perf_final = pd.concat([df_backtest_performance, df_temp_continent, df_temp_country]).sort_values(
        ["continent", "country", "province"]
    ).reset_index(drop=True)
    for col in ["train_mape_cases", "train_mape_deaths", "test_mape_cases", "test_mape_deaths"]:
        df_backtest_perf_final[col] = df_backtest_perf_final[col].round(3)

    df_backtest_perf_final.drop_duplicates(subset=["continent", "country", "province"], inplace=True)
    return df_backtest_perf_final
