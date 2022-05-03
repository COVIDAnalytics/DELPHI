# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import os
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime, timedelta
from typing import Union
import json
from logging import Logger
from DELPHI_params_V4 import (
    TIME_DICT,
    default_policy,
    default_policy_enaction_time,
)
from DELPHI_utils_V4_dynamic import make_increasing


class DELPHIDataSaver:
    def __init__(
            self,
            path_to_folder_danger_map: str,
            path_to_website_predicted: str,
            df_global_parameters: Union[pd.DataFrame, None],
            df_global_predictions_since_today: pd.DataFrame,
            df_global_predictions_since_100_cases: pd.DataFrame,
            logger: Logger = None
    ):
        self.PATH_TO_FOLDER_DANGER_MAP = path_to_folder_danger_map
        self.PATH_TO_WEBSITE_PREDICTED = path_to_website_predicted
        self.df_global_parameters = df_global_parameters
        self.df_global_predictions_since_today = df_global_predictions_since_today
        self.df_global_predictions_since_100_cases = (
            df_global_predictions_since_100_cases
        )
        self.logger = logger

    @staticmethod
    def save_dataframe(df, path, logger):
        attempt = 0
        success = False
        filename = path
        while attempt <= 5 and not success:
            success = True
            attempt+=1
            try:
                df.to_csv(filename, index=False)
            except OSError:
                success = False
                filename = path.replace(".csv", f"_try_{attempt}.csv")

        if not success:
            logger.error(
                f"Unable to save file {path}, skipping after {attempt} attempts"
            )
            return attempt
        return 0

    def save_all_datasets(
            self, optimizer: str, save_since_100_cases: bool = False, website: bool = False
    ):
        """
        Saves the parameters and predictions datasets (since 100 cases and since the day of running)
        based on the different flags and the inputs to the DELPHIDataSaver initializer
        :param optimizer: needs to be in (tnc, trust-constr, annealing) and will save files differently accordingly; the
        default name corresponds to tnc where we don't specify the optimizer because that's the default one
        :param save_since_100_cases: boolean, whether or not we also want to save the predictions since 100 cases
        for all the areas (instead of since the day we actually ran the optimization)
        :param website: boolean, whether or not we want to save the files in the website repository as well
        :return:
        """
        today_date_str = "".join(str(datetime.now().date()).split("-"))
        if optimizer == "tnc":
            subname_file = "Global_V4"
        elif optimizer == "annealing":
            subname_file = "Global_V4_annealing"
        elif optimizer == "trust-constr":
            subname_file = "Global_V4_trust"
        else:
            raise ValueError("Optimizer not supported in this implementation")
        # Save parameters

        DELPHIDataSaver.save_dataframe(
            self.df_global_parameters,
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_{subname_file}_{today_date_str}.csv",
            self.logger
            )
        # Save predictions since today
        DELPHIDataSaver.save_dataframe(
            self.df_global_predictions_since_today,
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/{subname_file}_{today_date_str}.csv",
            self.logger
            )
        if website:
            DELPHIDataSaver.save_dataframe(
                self.df_global_parameters,
                self.PATH_TO_WEBSITE_PREDICTED + f"data/predicted/Parameters_{subname_file}_{today_date_str}.csv",
                self.logger
                )
            DELPHIDataSaver.save_dataframe(
                self.df_global_predictions_since_today,
                self.PATH_TO_WEBSITE_PREDICTED
                + f"data/predicted/{subname_file}_{today_date_str}.csv",
                self.logger
                )
            DELPHIDataSaver.save_dataframe(
                self.df_global_predictions_since_today,
                self.PATH_TO_WEBSITE_PREDICTED + f"data/predicted/Global.csv",
                self.logger
                )
        if save_since_100_cases:
            # Save predictions since 100 cases
            DELPHIDataSaver.save_dataframe(
                self.df_global_predictions_since_100_cases,
                self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/{subname_file}_since100_{today_date_str}.csv",
                self.logger
                )
            if website:
                DELPHIDataSaver.save_dataframe(
                    self.df_global_predictions_since_100_cases,
                    self.PATH_TO_WEBSITE_PREDICTED + f"data/predicted/{subname_file}_since100_{today_date_str}.csv",
                    self.logger
                    )
                DELPHIDataSaver.save_dataframe(
                    self.df_global_predictions_since_100_cases,
                    self.PATH_TO_WEBSITE_PREDICTED + f"data/predicted/{subname_file}_since100.csv",
                    self.logger
                    )

    def save_policy_predictions_to_json(self, website: bool = False, local_delphi: bool = False):
        """
        Saves the policy predictions as a JSON file based on the different flags
        :param website: boolean, whether or not we want to save the JSON file in the website repository as well
        :param local_delphi: boolean, whether or not we want to save the JSON file in the DELPHI repository as well
        :return:
        """
        today_date_str = "".join(str(datetime.now().date()).split("-"))
        dict_predictions_policies_world_since_100_cases = DELPHIDataSaver.create_nested_dict_from_final_dataframe(
            self.df_global_predictions_since_100_cases
        )
        with open(
                self.PATH_TO_FOLDER_DANGER_MAP
                + f"/predicted/world_Python_{today_date_str}_Scenarios_since_100_cases.json",
                "w",
        ) as handle:
            json.dump(dict_predictions_policies_world_since_100_cases, handle)

        with open(
                self.PATH_TO_FOLDER_DANGER_MAP
                + f"/predicted/world_Python_Scenarios_since_100_cases.json",
                "w",
        ) as handle:
            json.dump(dict_predictions_policies_world_since_100_cases, handle)

        if local_delphi:
            with open(
                    f"./world_Python_{today_date_str}_Scenarios_since_100_cases.json", "w"
            ) as handle:
                json.dump(dict_predictions_policies_world_since_100_cases, handle)

        if website:
            with open(
                    self.PATH_TO_WEBSITE_PREDICTED
                    + f"assets/policies/World_Scenarios.json",
                    "w",
            ) as handle:
                json.dump(dict_predictions_policies_world_since_100_cases, handle)

    @staticmethod
    def create_nested_dict_from_final_dataframe(df_predictions: pd.DataFrame) -> dict:
        """
        Generates the nested dictionary with all the policy predictions which will then be saved as a JSON file
        to be used on the website
        :param df_predictions: dataframe with all policy predictions
        :return: dictionary with nested keys and policy predictions to be saved as a JSON file
        """
        dict_all_results = {
            continent: {} for continent in df_predictions.Continent.unique()
        }
        for continent in dict_all_results.keys():
            countries_in_continent = list(
                df_predictions[df_predictions.Continent == continent].Country.unique()
            )
            dict_all_results[continent] = {
                country: {} for country in countries_in_continent
            }

        keys_country_province = list(
            set(
                [
                    (continent, country, province)
                    for continent, country, province in zip(
                    df_predictions.Continent.tolist(),
                    df_predictions.Country.tolist(),
                    df_predictions.Province.tolist(),
                )
                ]
            )
        )
        for continent, country, province in keys_country_province:
            df_predictions_province = df_predictions[
                (df_predictions.Country == country)
                & (df_predictions.Province == province)
                ].reset_index(drop=True)
            # The first part contains only ground truth value, so it doesn't matter which
            # policy/enaction time we choose to report these values
            dict_all_results[continent][country][province] = {
                "Day": sorted(list(df_predictions_province.Day.unique())),
                "Total Detected True": df_predictions_province[
                    (df_predictions_province.Policy == default_policy)
                    & (df_predictions_province.Time == default_policy_enaction_time)
                    ]
                    .sort_values("Day")["Total Detected True"]
                    .tolist(),
                "Total Detected Deaths True": df_predictions_province[
                    (df_predictions_province.Policy == default_policy)
                    & (df_predictions_province.Time == default_policy_enaction_time)
                    ]
                    .sort_values("Day")["Total Detected Deaths True"]
                    .tolist(),
            }
            dict_all_results[continent][country][province].update(
                {
                    policy: {
                        policy_enaction_time: {
                            "Total Detected": df_predictions_province[
                                (df_predictions_province.Policy == policy)
                                & (df_predictions_province.Time == policy_enaction_time)
                                ]
                                .sort_values("Day")["Total Detected"]
                                .tolist(),
                            "Total Detected Deaths": df_predictions_province[
                                (df_predictions_province.Policy == policy)
                                & (df_predictions_province.Time == policy_enaction_time)
                                ]
                                .sort_values("Day")["Total Detected Deaths"]
                                .tolist(),
                        }
                        for policy_enaction_time in df_predictions_province.Time.unique()
                    }
                    for policy in df_predictions_province.Policy.unique()
                }
            )

        return dict_all_results


class DELPHIDataCreator:
    def __init__(
            self,
            x_sol_final: np.array,
            date_day_since100: datetime,
            best_params: np.array,
            continent: str,
            country: str,
            province: str,
            testing_data_included: bool = False,
    ):
        if testing_data_included:
            assert (
                    len(best_params) == 15
            ), f"Expected 9 best parameters, got {len(best_params)}"
        else:
            assert (
                    len(best_params) == 12
            ), f"Expected 7 best parameters, got {len(best_params)}"
        self.x_sol_final = x_sol_final
        self.date_day_since100 = date_day_since100
        self.best_params = best_params
        self.continent = continent
        self.country = country
        self.province = province
        self.testing_data_included = testing_data_included

    def create_dataset_parameters(self, mape: float) -> pd.DataFrame:
        """
        Creates the parameters dataset with the results from the optimization and the pre-computed MAPE
        :param mape: MAPE on the last 15 days (or less if less historical days available) for that particular area
        :return: dataframe with parameters and MAPE
        """
        if self.testing_data_included:
            print(
                f"Parameters dataset created without the testing data parameters"
                + " beta_0, beta_1: code will have to be modified"
            )
        df_parameters = pd.DataFrame(
            {
                "Continent": [self.continent],
                "Country": [self.country],
                "Province": [self.province],
                "Data Start Date": [self.date_day_since100],
                "MAPE": [mape],
                "Infection Rate": [self.best_params[0]],
                "Median Day of Action": [self.best_params[1]],
                "Rate of Action": [self.best_params[2]],
                "Rate of Death": [self.best_params[3]],
                "Mortality Rate": [self.best_params[4]],
                "Rate of Mortality Rate Decay": [self.best_params[5]],
                "Internal Parameter 1": [self.best_params[6]],
                "Internal Parameter 2": [self.best_params[7]],
                "Jump Magnitude": [self.best_params[8]],
                "Jump Time": [self.best_params[9]],
                "Jump Decay": [self.best_params[10]],
                "Internal Parameter 3": [self.best_params[11]],
            }
        )
        return df_parameters

    def create_datasets_predictions(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Creates two dataframes with the predictions of the DELPHI model, the first one since the day of the prediction,
        the second since the day the area had 100 cases
        :return: tuple of dataframes with predictions from DELPHI model
        """
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        # Predictions
        total_detected = self.x_sol_final[15, :]  # DT
        total_detected = [int(round(x, 0)) for x in total_detected]
        active_cases = (
                self.x_sol_final[4, :]
                + self.x_sol_final[5, :]
                + self.x_sol_final[7, :]
                + self.x_sol_final[8, :]
        )  # DHR + DQR + DHD + DQD
        active_cases = [int(round(x, 0)) for x in active_cases]
        active_hospitalized = (
                self.x_sol_final[4, :] + self.x_sol_final[7, :]
        )  # DHR + DHD
        active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
        cumulative_hospitalized = self.x_sol_final[11, :]  # TH
        cumulative_hospitalized = [int(round(x, 0)) for x in cumulative_hospitalized]
        total_detected_deaths = self.x_sol_final[14, :]  # DD
        total_detected_deaths = [int(round(x, 0)) for x in total_detected_deaths]
        active_ventilated = (
                self.x_sol_final[12, :] + self.x_sol_final[13, :]
        )  # DVR + DVD
        active_ventilated = [int(round(x, 0)) for x in active_ventilated]
        # Generation of the dataframe since today
        df_predictions_since_today_cont_country_prov = pd.DataFrame(
            {
                "Continent": [self.continent for _ in range(n_days_since_today)],
                "Country": [self.country for _ in range(n_days_since_today)],
                "Province": [self.province for _ in range(n_days_since_today)],
                "Day": all_dates_since_today,
                "Total Detected": total_detected[n_days_btw_today_since_100:],
                "Active": active_cases[n_days_btw_today_since_100:],
                "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
                "Cumulative Hospitalized": cumulative_hospitalized[
                                           n_days_btw_today_since_100:
                                           ],
                "Total Detected Deaths": total_detected_deaths[
                                         n_days_btw_today_since_100:
                                         ],
                "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
            }
        )

        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((self.date_day_since100 + timedelta(days=i)).date())
            for i in range(self.x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame(
            {
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
            }
        )
        return (
            df_predictions_since_today_cont_country_prov,
            df_predictions_since_100_cont_country_prov,
        )

    def create_datasets_raw(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Creates a dataset in the right format (with values for all 16 states of the DELPHI model)
        for the Optimal Vaccine Allocation team
        """
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]

        df_predictions_since_today_cont_country_prov = pd.DataFrame(
            {
                "Continent": [self.continent for _ in range(n_days_since_today)],
                "Country": [self.country for _ in range(n_days_since_today)],
                "Province": [self.province for _ in range(n_days_since_today)],
                "Day": all_dates_since_today,
            }
        )

        intr_since_today = pd.DataFrame(
            self.x_sol_final[:, n_days_btw_today_since_100:].transpose()
        )
        intr_since_today.columns = [
            "S",
            "E",
            "I",
            "AR",
            "DHR",
            "DQR",
            "AD",
            "DHD",
            "DQD",
            "R",
            "D",
            "TH",
            "DVR",
            "DVD",
            "DD",
            "DT",
        ]
        df_predictions_since_today_cont_country_prov = pd.concat(
            [df_predictions_since_today_cont_country_prov, intr_since_today], axis=1
        )
        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((self.date_day_since100 + timedelta(days=i)).date())
            for i in range(self.x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame(
            {
                "Continent": [self.continent for _ in range(len(all_dates_since_100))],
                "Country": [self.country for _ in range(len(all_dates_since_100))],
                "Province": [self.province for _ in range(len(all_dates_since_100))],
                "Day": all_dates_since_100,
            }
        )

        intr_since_100 = pd.DataFrame(self.x_sol_final.transpose())

        intr_since_100.columns = [
            "S",
            "E",
            "I",
            "AR",
            "DHR",
            "DQR",
            "AD",
            "DHD",
            "DQD",
            "R",
            "D",
            "TH",
            "DVR",
            "DVD",
            "DD",
            "DT",
        ]

        df_predictions_since_100_cont_country_prov = pd.concat(
            [df_predictions_since_100_cont_country_prov, intr_since_100], axis=1
        )

        return (
            df_predictions_since_today_cont_country_prov,
            df_predictions_since_100_cont_country_prov,
        )

    def create_datasets_with_confidence_intervals(
            self,
            cases_data_fit: list,
            deaths_data_fit: list,
            past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
            past_prediction_date: str = "2020-07-04",
            q: float = 0.5,
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Generates the prediction datasets from the date with 100 cases and from the day of running, including columns
        containing Confidence Intervals used in the website for cases and deaths
        :param cases_data_fit: list, contains data used to fit on number of cases
        :param deaths_data_fit: list, contains data used to fit on number of deaths
        :param past_prediction_file: past prediction file's path for CI generation
        :param past_prediction_date: past prediction's date for CI generation
        :param q: quantile used for the CIs
        :return: tuple of dataframes (since day of optimization & since 100 cases in the area) with predictions and
        confidence intervals
        """
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        # Predictions
        total_detected = self.x_sol_final[15, :]  # DT
        total_detected = [int(round(x, 0)) for x in total_detected]
        active_cases = (
                self.x_sol_final[4, :]
                + self.x_sol_final[5, :]
                + self.x_sol_final[7, :]
                + self.x_sol_final[8, :]
        )  # DHR + DQR + DHD + DQD
        active_cases = [int(round(x, 0)) for x in active_cases]
        active_hospitalized = (
                self.x_sol_final[4, :] + self.x_sol_final[7, :]
        )  # DHR + DHD
        active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
        cumulative_hospitalized = self.x_sol_final[11, :]  # TH
        cumulative_hospitalized = [int(round(x, 0)) for x in cumulative_hospitalized]
        total_detected_deaths = self.x_sol_final[14, :]  # DD
        total_detected_deaths = [int(round(x, 0)) for x in total_detected_deaths]
        active_ventilated = (
                self.x_sol_final[12, :] + self.x_sol_final[13, :]
        )  # DVR + DVD
        active_ventilated = [int(round(x, 0)) for x in active_ventilated]

        past_predictions = pd.read_csv(past_prediction_file)
        past_predictions = (
            past_predictions[
                (past_predictions["Day"] > past_prediction_date)
                & (past_predictions["Country"] == self.country)
                & (past_predictions["Province"] == self.province)
                ]
        ).sort_values("Day")
        if len(past_predictions) > 0:
            known_dates_since_100 = [
                str((self.date_day_since100 + timedelta(days=i)).date())
                for i in range(len(cases_data_fit))
            ]
            cases_data_fit_past = [
                y
                for x, y in zip(known_dates_since_100, cases_data_fit)
                if x > past_prediction_date
            ]
            deaths_data_fit_past = [
                y
                for x, y in zip(known_dates_since_100, deaths_data_fit)
                if x > past_prediction_date
            ]
            total_detected_past = past_predictions["Total Detected"].values[
                                  : len(cases_data_fit_past)
                                  ]
            total_detected_deaths_past = past_predictions[
                                             "Total Detected Deaths"
                                         ].values[: len(deaths_data_fit_past)]
            residual_cases_lb = np.sqrt(
                np.mean(
                    [(x - y) ** 2 for x, y in zip(cases_data_fit_past, total_detected_past)]
                )
            ) * scipy.stats.norm.ppf(0.5 - q / 2)
            residual_cases_ub = np.sqrt(
                np.mean(
                    [(x - y) ** 2 for x, y in zip(cases_data_fit_past, total_detected_past)]
                )
            ) * scipy.stats.norm.ppf(0.5 + q / 2)
            residual_deaths_lb = np.sqrt(
                np.mean(
                    [
                        (x - y) ** 2
                        for x, y in zip(deaths_data_fit_past, total_detected_deaths_past)
                    ]
                )
            ) * scipy.stats.norm.ppf(0.5 - q / 2)
            residual_deaths_ub = np.sqrt(
                np.mean(
                    [
                        (x - y) ** 2
                        for x, y in zip(deaths_data_fit_past, total_detected_deaths_past)
                    ]
                )
            ) * scipy.stats.norm.ppf(0.5 + q / 2)

            # Generation of the dataframe since today
            df_predictions_since_today_cont_country_prov = pd.DataFrame(
                {
                    "Continent": [self.continent for _ in range(n_days_since_today)],
                    "Country": [self.country for _ in range(n_days_since_today)],
                    "Province": [self.province for _ in range(n_days_since_today)],
                    "Day": all_dates_since_today,
                    "Total Detected": total_detected[n_days_btw_today_since_100:],
                    "Active": active_cases[n_days_btw_today_since_100:],
                    "Active Hospitalized": active_hospitalized[
                                           n_days_btw_today_since_100:
                                           ],
                    "Cumulative Hospitalized": cumulative_hospitalized[
                                               n_days_btw_today_since_100:
                                               ],
                    "Total Detected Deaths": total_detected_deaths[
                                             n_days_btw_today_since_100:
                                             ],
                    "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
                    "Total Detected True": [np.nan for _ in range(n_days_since_today)],
                    "Total Detected Deaths True": [
                        np.nan for _ in range(n_days_since_today)
                    ],
                    "Total Detected LB": make_increasing([
                        max(int(round(v + residual_cases_lb * np.sqrt(c), 0)), 0)
                        for c, v in enumerate(
                            total_detected[n_days_btw_today_since_100:]
                        )
                    ]),
#                    "Active LB": [
#                        max(
#                            int(round(v + residual_cases_lb * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_cases[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
#                    "Active Hospitalized LB": [
#                        max(
#                            int(round(v + residual_cases_lb * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_hospitalized[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
#                    "Cumulative Hospitalized LB": make_increasing([
#                        max(
#                            int(round(v + residual_cases_lb * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                cumulative_hospitalized[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ]),
                    "Total Detected Deaths LB": make_increasing([
                        max(int(round(v + residual_deaths_lb * np.sqrt(c), 0)), 0)
                        for c, v in enumerate(
                            total_detected_deaths[n_days_btw_today_since_100:]
                        )
                    ]),
#                    "Active Ventilated LB": [
#                        max(
#                            int(round(v + residual_cases_lb * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_ventilated[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
                    "Total Detected UB": [
                        max(int(round(v + residual_cases_ub * np.sqrt(c), 0)), 0)
                        for c, v in enumerate(
                            total_detected[n_days_btw_today_since_100:]
                        )
                    ],
#                    "Active UB": [
#                        max(
#                            int(round(v + residual_cases_ub * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_cases[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
#                    "Active Hospitalized UB": [
#                        max(
#                            int(round(v + residual_cases_ub * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_hospitalized[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
#                    "Cumulative Hospitalized UB": [
#                        max(
#                            int(round(v + residual_cases_ub * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                cumulative_hospitalized[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
                    "Total Detected Deaths UB": [
                        max(int(round(v + residual_deaths_ub * np.sqrt(c), 0)), 0)
                        for c, v in enumerate(
                            total_detected_deaths[n_days_btw_today_since_100:]
                        )
                    ],
#                    "Active Ventilated UB": [
#                        max(
#                            int(round(v + residual_cases_ub * np.sqrt(c) * v / u, 0)), 0
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(
#                                active_ventilated[n_days_btw_today_since_100:],
#                                total_detected[n_days_btw_today_since_100:],
#                            )
#                        )
#                    ],
                }
            )
            # Generation of the dataframe from the day since 100th case
            all_dates_since_100 = [
                str((self.date_day_since100 + timedelta(days=i)).date())
                for i in range(self.x_sol_final.shape[1])
            ]
            df_predictions_since_100_cont_country_prov = pd.DataFrame(
                {
                    "Continent": [
                        self.continent for _ in range(len(all_dates_since_100))
                    ],
                    "Country": [self.country for _ in range(len(all_dates_since_100))],
                    "Province": [
                        self.province for _ in range(len(all_dates_since_100))
                    ],
                    "Day": all_dates_since_100,
                    "Total Detected": total_detected,
                    "Active": active_cases,
                    "Active Hospitalized": active_hospitalized,
                    "Cumulative Hospitalized": cumulative_hospitalized,
                    "Total Detected Deaths": total_detected_deaths,
                    "Active Ventilated": active_ventilated,
                    "Total Detected True": cases_data_fit
                                           + [
                                               np.nan
                                               for _ in range(len(all_dates_since_100) - len(cases_data_fit))
                                           ],
                    "Total Detected Deaths True": deaths_data_fit
                                                  + [
                                                      np.nan for _ in range(len(all_dates_since_100) - len(deaths_data_fit))
                                                  ],
                    "Total Detected LB": make_increasing([
                        max(
                            int(
                                round(
                                    v
                                    + residual_cases_lb
                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                    0,
                                    )
                            ),
                            0,
                        )
                        for c, v in enumerate(total_detected)
                    ]),
#                    "Active LB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_lb
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(zip(active_cases, total_detected))
#                    ],
#                    "Active Hospitalized LB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_lb
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(active_hospitalized, total_detected)
#                        )
#                    ],
#                    "Cumulative Hospitalized LB": make_increasing([
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_lb
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(cumulative_hospitalized, total_detected)
#                        )
#                    ]),
                    "Total Detected Deaths LB": make_increasing([
                        max(
                            int(
                                round(
                                    v
                                    + residual_deaths_lb
                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                    0,
                                    )
                            ),
                            0,
                        )
                        for c, v in enumerate(total_detected_deaths)
                    ]),
#                    "Active Ventilated LB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_lb
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(active_ventilated, total_detected)
#                        )
#                    ],
                    "Total Detected UB": [
                        max(
                            int(
                                round(
                                    v
                                    + residual_cases_ub
                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                    0,
                                    )
                            ),
                            0,
                        )
                        for c, v in enumerate(total_detected)
                    ],
#                    "Active UB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_ub
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(zip(active_cases, total_detected))
#                    ],
#                    "Active Hospitalized UB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_ub
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(active_hospitalized, total_detected)
#                        )
#                    ],
#                    "Cumulative Hospitalized UB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_ub
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(cumulative_hospitalized, total_detected)
#                        )
#                    ],
                    "Total Detected Deaths UB": [
                        max(
                            int(
                                round(
                                    v
                                    + residual_deaths_ub
                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                    0,
                                    )
                            ),
                            0,
                        )
                        for c, v in enumerate(total_detected_deaths)
                    ],
#                    "Active Ventilated UB": [
#                        max(
#                            int(
#                                round(
#                                    v
#                                    + residual_cases_ub
#                                    * np.sqrt(max(c - n_days_btw_today_since_100, 0))
#                                    * v
#                                    / u,
#                                    0,
#                                    )
#                            ),
#                            0,
#                        )
#                        for c, (v, u) in enumerate(
#                            zip(active_ventilated, total_detected)
#                        )
#                    ],
                }
            )
        else:
            df_predictions_since_today_cont_country_prov = pd.DataFrame(
                {
                    "Continent": [self.continent for _ in range(n_days_since_today)],
                    "Country": [self.country for _ in range(n_days_since_today)],
                    "Province": [self.province for _ in range(n_days_since_today)],
                    "Day": all_dates_since_today,
                    "Total Detected": total_detected[n_days_btw_today_since_100:],
                    "Active": active_cases[n_days_btw_today_since_100:],
                    "Active Hospitalized": active_hospitalized[
                                           n_days_btw_today_since_100:
                                           ],
                    "Cumulative Hospitalized": cumulative_hospitalized[
                                               n_days_btw_today_since_100:
                                               ],
                    "Total Detected Deaths": total_detected_deaths[
                                             n_days_btw_today_since_100:
                                             ],
                    "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
                    "Total Detected True": [np.nan for _ in range(n_days_since_today)],
                    "Total Detected Deaths True": [
                        np.nan for _ in range(n_days_since_today)
                    ],
                    "Total Detected LB": [np.nan for _ in range(n_days_since_today)],
#                    "Active LB": [np.nan for _ in range(n_days_since_today)],
#                    "Active Hospitalized LB": [
#                        np.nan for _ in range(n_days_since_today)
#                    ],
#                    "Cumulative Hospitalized LB": [
#                        np.nan for _ in range(n_days_since_today)
#                    ],
                    "Total Detected Deaths LB": [
                        np.nan for _ in range(n_days_since_today)
                    ],
#                    "Active Ventilated LB": [np.nan for _ in range(n_days_since_today)],
                    "Total Detected UB": [np.nan for _ in range(n_days_since_today)],
#                    "Active UB": [np.nan for _ in range(n_days_since_today)],
#                    "Active Hospitalized UB": [
#                        np.nan for _ in range(n_days_since_today)
#                    ],
#                    "Cumulative Hospitalized UB": [
#                        np.nan for _ in range(n_days_since_today)
#                    ],
                    "Total Detected Deaths UB": [
                        np.nan for _ in range(n_days_since_today)
                    ]
#                    "Active Ventilated UB": [np.nan for _ in range(n_days_since_today)],
                }
            )
            # Generation of the dataframe from the day since 100th case
            all_dates_since_100 = [
                str((self.date_day_since100 + timedelta(days=i)).date())
                for i in range(self.x_sol_final.shape[1])
            ]
            df_predictions_since_100_cont_country_prov = pd.DataFrame(
                {
                    "Continent": [
                        self.continent for _ in range(len(all_dates_since_100))
                    ],
                    "Country": [self.country for _ in range(len(all_dates_since_100))],
                    "Province": [
                        self.province for _ in range(len(all_dates_since_100))
                    ],
                    "Day": all_dates_since_100,
                    "Total Detected": total_detected,
                    "Active": active_cases,
                    "Active Hospitalized": active_hospitalized,
                    "Cumulative Hospitalized": cumulative_hospitalized,
                    "Total Detected Deaths": total_detected_deaths,
                    "Active Ventilated": active_ventilated,
                    "Total Detected True": cases_data_fit
                                           + [
                                               np.nan
                                               for _ in range(len(all_dates_since_100) - len(cases_data_fit))
                                           ],
                    "Total Detected Deaths True": deaths_data_fit
                                                  + [
                                                      np.nan for _ in range(len(all_dates_since_100) - len(deaths_data_fit))
                                                  ],
                    "Total Detected LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
#                    "Active LB": [np.nan for _ in range(len(all_dates_since_100))],
#                    "Active Hospitalized LB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
#                    "Cumulative Hospitalized LB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
                    "Total Detected Deaths LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
#                    "Active Ventilated LB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
                    "Total Detected UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
#                    "Active UB": [np.nan for _ in range(len(all_dates_since_100))],
#                    "Active Hospitalized UB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
#                    "Cumulative Hospitalized UB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
                    "Total Detected Deaths UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
#                    "Active Ventilated UB": [
#                        np.nan for _ in range(len(all_dates_since_100))
#                    ],
                }
            )
        return (
            df_predictions_since_today_cont_country_prov,
            df_predictions_since_100_cont_country_prov,
        )

    def create_datasets_predictions_scenario(
            self, policy: str = "Lockdown", time: int = 0, totalcases=None
    ) -> (pd.DataFrame, pd.DataFrame):
        n_days_btw_today_since_100 = (datetime.now() - self.date_day_since100).days
        n_days_since_today = self.x_sol_final.shape[1] - n_days_btw_today_since_100
        all_dates_since_today = [
            str((datetime.now() + timedelta(days=i)).date())
            for i in range(n_days_since_today)
        ]
        # Predictions
        total_detected = self.x_sol_final[15, :]  # DT
        total_detected = [int(round(x, 0)) for x in total_detected]
        active_cases = (
                self.x_sol_final[4, :]
                + self.x_sol_final[5, :]
                + self.x_sol_final[7, :]
                + self.x_sol_final[8, :]
        )  # DHR + DQR + DHD + DQD
        active_cases = [int(round(x, 0)) for x in active_cases]
        active_hospitalized = (
                self.x_sol_final[4, :] + self.x_sol_final[7, :]
        )  # DHR + DHD
        active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
        cumulative_hospitalized = self.x_sol_final[11, :]  # TH
        cumulative_hospitalized = [int(round(x, 0)) for x in cumulative_hospitalized]
        total_detected_deaths = self.x_sol_final[14, :]  # DD
        total_detected_deaths = [int(round(x, 0)) for x in total_detected_deaths]
        active_ventilated = (
                self.x_sol_final[12, :] + self.x_sol_final[13, :]
        )  # DVR + DVD
        active_ventilated = [int(round(x, 0)) for x in active_ventilated]
        # Generation of the dataframe since today
        df_predictions_since_today_cont_country_prov = pd.DataFrame(
            {
                "Policy": [policy for _ in range(n_days_since_today)],
                "Time": [TIME_DICT[time] for _ in range(n_days_since_today)],
                "Continent": [self.continent for _ in range(n_days_since_today)],
                "Country": [self.country for _ in range(n_days_since_today)],
                "Province": [self.province for _ in range(n_days_since_today)],
                "Day": all_dates_since_today,
                "Total Detected": total_detected[n_days_btw_today_since_100:],
                "Active": active_cases[n_days_btw_today_since_100:],
                "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
                "Cumulative Hospitalized": cumulative_hospitalized[
                                           n_days_btw_today_since_100:
                                           ],
                "Total Detected Deaths": total_detected_deaths[
                                         n_days_btw_today_since_100:
                                         ],
                "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
            }
        )

        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((self.date_day_since100 + timedelta(days=i)).date())
            for i in range(self.x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame(
            {
                "Policy": [policy for _ in range(len(all_dates_since_100))],
                "Time": [TIME_DICT[time] for _ in range(len(all_dates_since_100))],
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
            }
        )
        if (
                totalcases is not None
        ):  # Merging the historical values to both dataframes when available
            df_predictions_since_today_cont_country_prov = df_predictions_since_today_cont_country_prov.merge(
                totalcases[
                    ["country", "province", "date", "case_cnt", "death_cnt"]
                ].fillna("None"),
                left_on=["Country", "Province", "Day"],
                right_on=["country", "province", "date"],
                how="left",
            )
            df_predictions_since_today_cont_country_prov.rename(
                columns={
                    "case_cnt": "Total Detected True",
                    "death_cnt": "Total Detected Deaths True",
                },
                inplace=True,
            )
            df_predictions_since_today_cont_country_prov.drop(
                ["country", "province", "date"], axis=1, inplace=True
            )
            df_predictions_since_100_cont_country_prov = df_predictions_since_100_cont_country_prov.merge(
                totalcases[
                    ["country", "province", "date", "case_cnt", "death_cnt"]
                ].fillna("None"),
                left_on=["Country", "Province", "Day"],
                right_on=["country", "province", "date"],
                how="left",
            )
            df_predictions_since_100_cont_country_prov.rename(
                columns={
                    "case_cnt": "Total Detected True",
                    "death_cnt": "Total Detected Deaths True",
                },
                inplace=True,
            )
            df_predictions_since_100_cont_country_prov.drop(
                ["country", "province", "date"], axis=1, inplace=True
            )
        return (
            df_predictions_since_today_cont_country_prov,
            df_predictions_since_100_cont_country_prov,
        )


class DELPHIAggregations:
    @staticmethod
    def get_aggregation_per_country(df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates predictions at the country level from the predictions dataframe
        :param df_predictions: DELPHI predictions dataframe
        :return: DELPHI predictions dataframe aggregated at the country level
        """
        df_predictions = df_predictions[df_predictions["Province"] != "None"]
        df_agg_country = df_predictions.groupby(["Continent", "Country", "Day"]).sum().reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[df_predictions.columns]
        return df_agg_country

    @staticmethod
    def get_aggregation_per_continent(df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates predictions at the continent level from the predictions dataframe
        :param df_predictions: DELPHI predictions dataframe
        :return: DELPHI predictions dataframe aggregated at the continent level
        """
        df_agg_continent = df_predictions.groupby(["Continent", "Day"]).sum().reset_index()
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[df_predictions.columns]
        return df_agg_continent

    @staticmethod
    def get_aggregation_world(df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates predictions at the world level from the predictions dataframe
        :param df_predictions: DELPHI predictions dataframe
        :return: DELPHI predictions dataframe aggregated at the world level (only one row in this dataframe)
        """
        df_agg_world = df_predictions.groupby("Day").sum().reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[df_predictions.columns]
        return df_agg_world

    @staticmethod
    def append_all_aggregations(df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and appends all the predictions' aggregations at the country, continent and world levels
        :param df_predictions: dataframe with the raw predictions from DELPHI
        :return: dataframe with raw predictions from DELPHI and aggregated ones at the country, continent & world levels
        """
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(df_predictions)
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent(
            df_predictions
        )
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world(df_predictions)
        df_predictions = pd.concat(
            [
                df_predictions,
                df_agg_since_today_per_country,
                df_agg_since_today_per_continent,
                df_agg_since_today_world,
            ]
        )
        df_predictions.sort_values(["Continent", "Country", "Province", "Day"], inplace=True)
        return df_predictions

    @staticmethod
    def get_aggregation_per_country_with_cf(
            df_predictions: pd.DataFrame,
            past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
            past_prediction_date: str = "2020-07-04",
            q: float = 0.5
    ) -> pd.DataFrame:
        """
        Creates aggregations at the country level as well as associated confidence intervals
        :param df_predictions: dataframe containing the raw predictions from the DELPHI model
        :param past_prediction_file: past prediction file's path for CI generation
        :param past_prediction_date: past prediction's date for CI generation
        :param q: quantile used for the CIs
        :return: dataframe with country level aggregated predictions & associated confidence intervals
        """
        df_predictions = df_predictions[df_predictions["Province"] != "None"]
        columns_without_bounds = [x for x in df_predictions.columns if ("LB" not in x) and ("UB" not in x)]
        df_agg_country = df_predictions[columns_without_bounds].groupby(["Continent", "Country", "Day"]).sum(min_count = 1).reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[columns_without_bounds]
        aggregated_countries = set(zip(df_agg_country["Country"],df_agg_country["Province"]))
        past_predictions = pd.read_csv(past_prediction_file)
        if len(aggregated_countries)>0:
            list_df_aggregated_countries = []
            for country, province in aggregated_countries:
                past_predictions_temp = (past_predictions[(past_predictions['Day'] > past_prediction_date) & (past_predictions['Country'] == country) & (past_predictions['Province'] == province)]).sort_values("Day")
                df_agg_country_temp = (df_agg_country[(df_agg_country['Country'] == country) & (df_agg_country['Province'] == province)]).sort_values("Day").reset_index(drop=True)
                total_detected = df_agg_country_temp['Total Detected'] 
                total_detected_deaths = df_agg_country_temp['Total Detected Deaths'] 
    #            active_cases = df_agg_country_temp['Active'] 
    #            active_hospitalized = df_agg_country_temp['Active Hospitalized'] 
    #            cumulative_hospitalized = df_agg_country_temp['Cumulative Hospitalized'] 
    #            active_ventilated = df_agg_country_temp['Active Ventilated'] 
                cases_fit_data = df_agg_country_temp['Total Detected True'] 
                deaths_fit_data = df_agg_country_temp['Total Detected Deaths True'] 
                since_100_dates = df_agg_country_temp['Day'] 
                n_days_btw_today_since_100 = (datetime.now() - pd.to_datetime(min(since_100_dates))).days
                if len(past_predictions_temp) > 0:
                    cases_fit_data_past = [y for x, y in zip(since_100_dates,cases_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
                    deaths_fit_data_past = [y for x, y in zip(since_100_dates,deaths_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
                    total_detected_past = past_predictions_temp["Total Detected"].values[:len(cases_fit_data_past)]
                    total_detected_deaths_past = past_predictions_temp["Total Detected Deaths"].values[:len(deaths_fit_data_past)]
                    residual_cases_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
                    residual_cases_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 + q /2)
                    residual_deaths_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
                    residual_deaths_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) *  scipy.stats.norm.ppf(0.5 + q /2)
            
                    # Generation of the dataframe from the day since 100th case
                    df_predictions_since_100_cont_country_prov = pd.DataFrame({
                        "Total Detected LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)]),
    #                    "Active LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
    #                    "Active Hospitalized LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
    #                    "Cumulative Hospitalized LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))]),
                        "Total Detected Deaths LB": make_increasing([max(int(round(v + residual_deaths_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)]),
    #                    "Active Ventilated LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
                        "Total Detected UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)],
    #                    "Active UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
    #                    "Active Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
    #                    "Cumulative Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))],
                        "Total Detected Deaths UB": [max(int(round(v + residual_deaths_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)],
    #                    "Active Ventilated UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
                    })
                    df_agg_country_temp = pd.concat([df_agg_country_temp, df_predictions_since_100_cont_country_prov], axis = 1)
                else:
                    df_predictions_since_100_cont_country_prov = pd.DataFrame({
                        "Total Detected LB": [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active LB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active Hospitalized LB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Cumulative Hospitalized LB":  [np.nan for _ in range(len(df_agg_country_temp))],  
                        "Total Detected Deaths LB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active Ventilated LB":  [np.nan for _ in range(len(df_agg_country_temp))],  
                        "Total Detected UB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active UB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active Hospitalized UB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Cumulative Hospitalized UB":  [np.nan for _ in range(len(df_agg_country_temp))],  
                        "Total Detected Deaths UB":  [np.nan for _ in range(len(df_agg_country_temp))],  
    #                    "Active Ventilated UB": [np.nan for _ in range(len(df_agg_country_temp))]
                    })
                    df_agg_country_temp = pd.concat([df_agg_country_temp, df_predictions_since_100_cont_country_prov], axis = 1)
    
                list_df_aggregated_countries.append(df_agg_country_temp)
            df_agg_country_final = pd.concat(list_df_aggregated_countries)
            return df_agg_country_final 
        else:
            return None

    @staticmethod
    def get_aggregation_per_continent_with_cf(
            df_predictions: pd.DataFrame,
            past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
            past_prediction_date: str = "2020-07-04",
            q: float = 0.5
    ) -> pd.DataFrame:
        """
        Creates aggregations at the continent level as well as associated confidence intervals
        :param df_predictions: dataframe containing the raw predictions from the DELPHI model
        :param past_prediction_file: past prediction file's path for CI generation
        :param past_prediction_date: past prediction's date for CI generation
        :param q: quantile used for the CIs
        :return: dataframe with continent level aggregated predictions & associated confidence intervals
        """
        columns_without_bounds = [x for x in df_predictions.columns if ("LB" not in x) and ("UB" not in x)]
        df_agg_continent = df_predictions[columns_without_bounds].groupby(["Continent", "Day"]).sum(min_count = 1).reset_index()
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[columns_without_bounds]
        aggregated_continents = set(zip(df_agg_continent["Continent"], df_agg_continent["Country"],df_agg_continent["Province"]))
        past_predictions = pd.read_csv(past_prediction_file)
        list_df_aggregated_continents = []
        for continent, country, province in aggregated_continents:
            past_predictions_temp = (past_predictions[(past_predictions['Day'] > past_prediction_date) & (past_predictions['Continent'] == continent) & (past_predictions['Country'] == country) & (past_predictions['Country'] == province)]).sort_values("Day")
            df_agg_continent_temp = (df_agg_continent[(df_agg_continent['Continent'] == continent)]).sort_values("Day").reset_index(drop=True)
            total_detected = df_agg_continent_temp['Total Detected'] 
            total_detected_deaths = df_agg_continent_temp['Total Detected Deaths'] 
#            active_cases = df_agg_continent_temp['Active'] 
#            active_hospitalized = df_agg_continent_temp['Active Hospitalized'] 
#            cumulative_hospitalized = df_agg_continent_temp['Cumulative Hospitalized'] 
#            active_ventilated = df_agg_continent_temp['Active Ventilated'] 
            cases_fit_data = df_agg_continent_temp['Total Detected True'] 
            deaths_fit_data = df_agg_continent_temp['Total Detected Deaths True'] 
            since_100_dates = df_agg_continent_temp['Day']   
            n_days_btw_today_since_100 = (datetime.now() - pd.to_datetime(min(since_100_dates))).days
            if len(past_predictions_temp) > 0:
                cases_fit_data_past = [y for x, y in zip(since_100_dates,cases_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
                deaths_fit_data_past = [y for x, y in zip(since_100_dates,deaths_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
                total_detected_past = past_predictions_temp["Total Detected"].values[:len(cases_fit_data_past)]
                total_detected_deaths_past = past_predictions_temp["Total Detected Deaths"].values[:len(deaths_fit_data_past)]
                residual_cases_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
                residual_cases_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 + q /2)
                residual_deaths_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
                residual_deaths_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) *  scipy.stats.norm.ppf(0.5 + q /2)
                # Generation of the dataframe from the day since 100th case
                df_predictions_since_100_cont_country_prov = pd.DataFrame({
                    "Total Detected LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)]),
#                    "Active LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
#                    "Active Hospitalized LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
#                    "Cumulative Hospitalized LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))]),
                    "Total Detected Deaths LB": make_increasing([max(int(round(v + residual_deaths_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)]),
#                    "Active Ventilated LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
                    "Total Detected UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)],
#                    "Active UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
#                    "Active Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
#                    "Cumulative Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))],
                    "Total Detected Deaths UB": [max(int(round(v + residual_deaths_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)],
#                    "Active Ventilated UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
                })
                df_agg_continent_temp = pd.concat([df_agg_continent_temp, df_predictions_since_100_cont_country_prov], axis = 1)
            else:
                df_predictions_since_100_cont_country_prov = pd.DataFrame({
                    "Total Detected LB": [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active LB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active Hospitalized LB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Cumulative Hospitalized LB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
                    "Total Detected Deaths LB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active Ventilated LB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
                    "Total Detected UB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active UB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active Hospitalized UB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Cumulative Hospitalized UB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
                    "Total Detected Deaths UB":  [np.nan for _ in range(len(df_agg_continent_temp))],  
#                    "Active Ventilated UB": [np.nan for _ in range(len(df_agg_continent_temp))]
                })
                df_agg_continent_temp = pd.concat([df_agg_continent_temp, df_predictions_since_100_cont_country_prov], axis = 1)

            list_df_aggregated_continents.append(df_agg_continent_temp)
        df_agg_continent_final = pd.concat(list_df_aggregated_continents)
        return df_agg_continent_final 

    @staticmethod
    def get_aggregation_world_with_cf(
            df_predictions: pd.DataFrame,
            past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
            past_prediction_date: str = "2020-07-04",
            q: float = 0.5
    ) -> pd.DataFrame:
        """
        Creates aggregations at the world level as well as associated confidence intervals
        :param df_predictions: dataframe containing the raw predictions from the DELPHI model
        :param past_prediction_file: past prediction file's path for CI generation
        :param past_prediction_date: past prediction's date for CI generation
        :param q: quantile used for the CIs
        :return: dataframe with continent world aggregated predictions & associated confidence intervals
        """
        columns_without_bounds = [x for x in df_predictions.columns if ("LB" not in x) and ("UB" not in x)]
        df_agg_world = df_predictions[columns_without_bounds].groupby(["Day"]).sum(min_count = 1).reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[columns_without_bounds]
        past_predictions = pd.read_csv(past_prediction_file)
        past_predictions_temp = (past_predictions[(past_predictions['Day'] > past_prediction_date) & (past_predictions['Continent'] == "None") & (past_predictions['Country'] == "None") & (past_predictions['Province'] == "None")]).sort_values("Day")
        total_detected = df_agg_world['Total Detected'] 
        total_detected_deaths = df_agg_world['Total Detected Deaths'] 
#        active_cases = df_agg_world['Active'] 
#        active_hospitalized = df_agg_world['Active Hospitalized'] 
#        cumulative_hospitalized = df_agg_world['Cumulative Hospitalized'] 
#        active_ventilated = df_agg_world['Active Ventilated'] 
        cases_fit_data = df_agg_world['Total Detected True'] 
        deaths_fit_data = df_agg_world['Total Detected Deaths True'] 
        since_100_dates = df_agg_world['Day']   
        n_days_btw_today_since_100 = (datetime.now() - pd.to_datetime(min(since_100_dates))).days
        if len(past_predictions_temp) > 0:
            cases_fit_data_past = [y for x, y in zip(since_100_dates,cases_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
            deaths_fit_data_past = [y for x, y in zip(since_100_dates,deaths_fit_data) if ((x > past_prediction_date) and (not np.isnan(y)))]
            total_detected_past = past_predictions_temp["Total Detected"].values[:len(cases_fit_data_past)]
            total_detected_deaths_past = past_predictions_temp["Total Detected Deaths"].values[:len(deaths_fit_data_past)]
            residual_cases_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
            residual_cases_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(cases_fit_data_past,total_detected_past)])) * scipy.stats.norm.ppf(0.5 + q /2)
            residual_deaths_lb = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) * scipy.stats.norm.ppf(0.5 - q /2)
            residual_deaths_ub = np.sqrt(np.mean([(x- y) ** 2 for x,y in zip(deaths_fit_data_past,total_detected_deaths_past)])) *  scipy.stats.norm.ppf(0.5 + q /2)
    
            # Generation of the dataframe from the day since 100th case
            df_predictions_since_100_cont_country_prov = pd.DataFrame({
                "Total Detected LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)]),
#                "Active LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
#                "Active Hospitalized LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
#                "Cumulative Hospitalized LB": make_increasing([max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))]),
                "Total Detected Deaths LB": make_increasing([max(int(round(v + residual_deaths_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)]),
#                "Active Ventilated LB": [max(int(round(v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
                "Total Detected UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected)],
#                "Active UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_cases, total_detected))],
#                "Active Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_hospitalized, total_detected))],
#                "Cumulative Hospitalized UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(cumulative_hospitalized, total_detected))],
                "Total Detected Deaths UB": [max(int(round(v + residual_deaths_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),0)),0) for c, v in enumerate(total_detected_deaths)],
#                "Active Ventilated UB": [max(int(round(v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)) * v / u,0)),0) for c, (v, u) in enumerate(zip(active_ventilated, total_detected))],
            })
            df_agg_world_final = pd.concat([df_agg_world, df_predictions_since_100_cont_country_prov], axis = 1)
        else:
            df_predictions_since_100_cont_country_prov = pd.DataFrame({
                "Total Detected LB": [np.nan for _ in range(len(df_agg_world))],  
#                "Active LB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Active Hospitalized LB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Cumulative Hospitalized LB":  [np.nan for _ in range(len(df_agg_world))],  
                "Total Detected Deaths LB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Active Ventilated LB":  [np.nan for _ in range(len(df_agg_world))],  
                "Total Detected UB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Active UB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Active Hospitalized UB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Cumulative Hospitalized UB":  [np.nan for _ in range(len(df_agg_world))],  
                "Total Detected Deaths UB":  [np.nan for _ in range(len(df_agg_world))],  
#                "Active Ventilated UB": [np.nan for _ in range(len(df_agg_world))]
            })
            df_agg_world_final = pd.concat([df_agg_world, df_predictions_since_100_cont_country_prov], axis = 1)
        return df_agg_world_final 

    @staticmethod
    def append_all_aggregations_cf(
            df_predictions: pd.DataFrame,
            past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
            past_prediction_date: str = "2020-07-04",
            q: float = 0.5
    ) -> pd.DataFrame:
        """
        Creates and appends all the predictions' aggregations & Confidnece Intervals at the country, continent and
        world levels
        :param df_predictions: dataframe with the raw predictions from DELPHI
        :param past_prediction_file: past prediction file's path for CI generation
        :param past_prediction_date: past prediction's date for CI generation
        :param q: quantile used for the CIs
        :return: dataframe with predictions raw from DELPHI and aggregated ones, as well as associated confidence
        intervals at the country, continent & world levels
        """
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country_with_cf(
            df_predictions=df_predictions,
            past_prediction_file=past_prediction_file,
            past_prediction_date=past_prediction_date,
            q=q
        )
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent_with_cf(
            df_predictions=df_predictions,
            past_prediction_file=past_prediction_file,
            past_prediction_date=past_prediction_date,
            q=q
        )
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world_with_cf(
            df_predictions=df_predictions,
            past_prediction_file=past_prediction_file,
            past_prediction_date=past_prediction_date,
            q=q
        )
        df_predictions = pd.concat([
            df_predictions, df_agg_since_today_per_country,
            df_agg_since_today_per_continent, df_agg_since_today_world
            ],sort=False)
        df_predictions.sort_values(["Continent", "Country", "Province", "Day"], inplace=True)
        df_predictions_from_today = df_predictions[df_predictions.Day >= str((pd.to_datetime(datetime.now())).date())]
        return df_predictions_from_today, df_predictions


class DELPHIAggregationsPolicies:
    @staticmethod
    def get_aggregation_per_country(df_policy_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates policy predictions at the country level from the predictions dataframe
        :param df_policy_predictions: DELPHI policy predictions dataframe
        :return: DELPHI policy predictions dataframe aggregated at the country level
        """
        df_policy_predictions = df_policy_predictions[df_policy_predictions["Province"] != "None"]
        df_agg_country = df_policy_predictions.groupby(
            ["Policy", "Time", "Continent", "Country", "Day"]
        ).sum().reset_index()
        df_agg_country["Province"] = "None"
        df_agg_country = df_agg_country[
            [
                "Policy", "Time", "Continent", "Country", "Province", "Day", "Total Detected", "Active",
                "Active Hospitalized", "Cumulative Hospitalized", "Total Detected Deaths", "Active Ventilated",
            ]
        ]
        return df_agg_country

    @staticmethod
    def get_aggregation_per_continent(df_policy_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates policy predictions at the continent level from the predictions dataframe
        :param df_policy_predictions: DELPHI policy predictions dataframe
        :return: DELPHI policy predictions dataframe aggregated at the continent level
        """
        df_agg_continent = (
            df_policy_predictions.groupby(["Policy", "Time", "Continent", "Day"]).sum().reset_index()
        )
        df_agg_continent["Country"] = "None"
        df_agg_continent["Province"] = "None"
        df_agg_continent = df_agg_continent[
            [
                "Policy", "Time", "Continent", "Country", "Province", "Day", "Total Detected", "Active",
                "Active Hospitalized", "Cumulative Hospitalized", "Total Detected Deaths", "Active Ventilated",
            ]
        ]
        return df_agg_continent

    @staticmethod
    def get_aggregation_world(df_policy_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates policy predictions at the world level from the predictions dataframe
        :param df_policy_predictions: DELPHI policy predictions dataframe
        :return: DELPHI policy predictions dataframe aggregated at the world level
        """
        df_agg_world = df_policy_predictions.groupby(["Policy", "Time", "Day"]).sum().reset_index()
        df_agg_world["Continent"] = "None"
        df_agg_world["Country"] = "None"
        df_agg_world["Province"] = "None"
        df_agg_world = df_agg_world[
            [
                "Policy", "Time", "Continent", "Country", "Province", "Day", "Total Detected", "Active",
                "Active Hospitalized", "Cumulative Hospitalized", "Total Detected Deaths", "Active Ventilated",
            ]
        ]
        return df_agg_world

    @staticmethod
    def append_all_aggregations(df_policy_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and appends all the policy predictions' aggregations at the country, continent and world levels
        :param df_predictions: dataframe with the raw policy predictions from DELPHI
        :return: dataframe with raw policy predictions from DELPHI and aggregated ones at the country,
        continent & world levels
        """
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(df_policy_predictions)
        df_agg_since_today_per_continent = DELPHIAggregations.get_aggregation_per_continent(df_policy_predictions)
        df_agg_since_today_world = DELPHIAggregations.get_aggregation_world(df_policy_predictions)
        df_policy_predictions = pd.concat(
            [
                df_policy_predictions,
                df_agg_since_today_per_country,
                df_agg_since_today_per_continent,
                df_agg_since_today_world,
            ]
        )
        df_policy_predictions.sort_values(
            ["Policy", "Time", "Continent", "Country", "Province", "Day"], inplace=True
        )
        return df_policy_predictions


class DELPHIBacktest:
    def __init__(
            self, path_to_folder_danger_map: str, prediction_date: str, n_days_backtest: int,
            get_mae: bool, get_mse: bool, logger: Logger,
    ):
        self.historical_data_path = path_to_folder_danger_map + "processed/Global/"
        self.prediction_data_path = path_to_folder_danger_map + "predicted/"
        self.prediction_date = prediction_date
        self.n_days_backtest = n_days_backtest
        self.get_mae = get_mae
        self.get_mse = get_mse
        self.logger = logger

    def get_historical_data_df(self) -> pd.DataFrame:
        """
        Generates a concatenation of all historical data available in the danger_map folder, all areas
        starting from the prediction date given by the user, and keeping only relevant columns
        :return: a dataframe with all relevant historical data
        """
        list_historical_data_filepaths = [
            self.historical_data_path + filename
            for filename in os.listdir(self.historical_data_path)
            if "Cases_" in filename
        ]
        df_historical = []
        for filepath_historical in list_historical_data_filepaths:
            df_historical.append(pd.read_csv(filepath_historical))

        df_historical = pd.concat(df_historical).sort_values(
            ["country", "province", "date"]
        ).reset_index(drop=True)[
            ["country", "province", "date", "day_since100", "case_cnt", "death_cnt"]
        ]
        df_historical["province"].fillna("None", inplace=True)
        df_historical.rename(
            columns={"country": "Country", "province": "Province", "date": "Day"}, inplace=True
        )
        df_historical["tuple"] = list(zip(df_historical.Country, df_historical.Province))
        df_historical = df_historical[
            (df_historical.Day >= self.prediction_date)
        ].reset_index(drop=True)
        df_historical = df_historical[df_historical.tuple != ("US", "None")].reset_index(drop=True)
        return df_historical

    def get_prediction_data(self) -> pd.DataFrame:
        """
        Retrieve the predicted data on the prediction_date given as an input by the user running
        :param prediction_date: prediction date to be used to look for the file in the danger_map folder, format
        has to be YYYY-MM-DD (it is asserted outside of this function)
        :return: a dataframe that contains the relevant predictions on the relevant prediction date
        """
        prediction_date_filename = "".join(self.prediction_date.split("-"))
        if os.path.exists(self.prediction_data_path + f"Global_V2_{prediction_date_filename}.csv"):
            self.logger.info("Backtesting on DELPHI V3.0 predictions because filename contains _V2")
            df_prediction = pd.read_csv(self.prediction_data_path + f"Global_V2_{prediction_date_filename}.csv")
        elif os.path.exists(self.prediction_data_path + f"Global_V2_{prediction_date_filename}.csv"):
            self.logger.info("Backtesting on DELPHI V1.0 or V2.0 predictions because filename doesn't contain _V2")
            df_prediction = pd.read_csv(self.prediction_data_path + f"Global_V2_{prediction_date_filename}.csv")
        else:
            raise ValueError(f"The file on prediction date {self.prediction_date} has never been generated")

        return df_prediction[["Continent", "Country", "Province", "Day", "Total Detected", "Total Detected Deaths"]]

    def get_feasibility_flag(self, df_historical: pd.DataFrame, df_prediction: pd.DataFrame) -> bool:
        """
        Checks that there is enough historical and prediction data to perform the backtest based on the user input
        :param df_historical: a dataframe with all relevant historical data
        :param df_prediction: a dataframe that contains the relevant predictions on the relevant prediction date
        :return: a True boolean, otherwise will raise a ValueError with more details as to why backtest is infeasible
        """
        max_date_historical = df_historical.Day.max()
        max_date_prediction = df_prediction.Day.max()
        max_date_backtest = str((pd.to_datetime(self.prediction_date) + timedelta(days=self.n_days_backtest)).date())
        days_missing_historical = (pd.to_datetime(max_date_backtest) - pd.to_datetime(max_date_historical)).days
        days_missing_prediction = (pd.to_datetime(max_date_backtest) - pd.to_datetime(max_date_prediction)).days
        if (days_missing_historical > 0) or (days_missing_prediction > 0):
            feasibility_flag = False
        else:
            feasibility_flag = True

        if not feasibility_flag:
            error_message = (
                    "Backtest date and number of days of backtest incompatible with available data: missing " +
                    f"{days_missing_historical} days of historical data and {days_missing_prediction} days of prediction data " +
                    "(if negative value for number of days: not missing)"
            )
            self.logger.warning(error_message)
            raise ValueError(error_message)

        self.logger.info(f"Backtest is feasible based on input prediction date and number of backtesting days")
        return True

    def generate_empty_metrics_dict(self) -> dict:
        """
        Generates the format of the dictionary that will compose the dataframe with all backtest metrics
        based on the get_mae and get_mse flags given by the user
        :return: a dictionary with either 5, 7 or 9 keys depending on the MAE and MSE flags
        """
        dict_df_backtest_metrics = {
            "prediction_date": [],
            "n_days_backtest": [],
            "tuple_area": [],
            "mape_cases": [],
            "mape_deaths": [],
        }
        if self.get_mae:
            dict_df_backtest_metrics["mae_cases"] = []
            dict_df_backtest_metrics["mae_deaths"] = []

        if self.get_mse:
            dict_df_backtest_metrics["mse_cases"] = []
            dict_df_backtest_metrics["mse_deaths"] = []

        return dict_df_backtest_metrics

    def get_backtest_metrics_area(
            self, df_backtest: pd.DataFrame, tuple_area: tuple, dict_df_backtest_metrics: dict,
    ) -> dict:
        """
        Updates the backtest metrics dictionary with metrics values for that particular area tuple
        :param df_backtest: pre-processed dataframe containing historical and prediction data
        :param tuple_area: tuple of the format (Continent, Country, Province)
        :param dict_df_backtest_metrics: dictionary containing all the metrics and will be used to create final
        backtest metrics dataframe
        :return: updated dictionary with backtest metrics for that particular area tuple
        """
        df_temp = df_backtest[df_backtest.tuple_complete == tuple_area]
        max_date_backtest = str((pd.to_datetime(self.prediction_date) + timedelta(days=self.n_days_backtest)).date())
        df_temp = df_temp[(df_temp.Day >= self.prediction_date) & (df_temp.Day <= max_date_backtest)].sort_values(
            ["Continent", "Country", "Province", "Day"]
        ).reset_index(drop=True)
        mae_cases, mape_cases = compute_mae_and_mape(
            y_true=df_temp.case_cnt.tolist(), y_pred=df_temp["Total Detected"].tolist()
        )
        mae_deaths, mape_deaths = compute_mae_and_mape(
            y_true=df_temp.death_cnt.tolist(), y_pred=df_temp["Total Detected Deaths"].tolist()
        )
        mse_cases = compute_mse(y_true=df_temp.case_cnt.tolist(), y_pred=df_temp["Total Detected"].tolist())
        mse_deaths = compute_mse(y_true=df_temp.death_cnt.tolist(), y_pred=df_temp["Total Detected Deaths"].tolist())
        dict_df_backtest_metrics["prediction_date"].append(self.prediction_date)
        dict_df_backtest_metrics["n_days_backtest"].append(self.n_days_backtest)
        dict_df_backtest_metrics["tuple_area"].append(tuple_area)
        dict_df_backtest_metrics["mape_cases"].append(mape_cases)
        dict_df_backtest_metrics["mape_deaths"].append(mape_deaths)
        if self.get_mae:
            dict_df_backtest_metrics["mae_cases"].append(mae_cases)
            dict_df_backtest_metrics["mae_deaths"].append(mae_deaths)

        if self.get_mse:
            dict_df_backtest_metrics["mse_cases"].append(mse_cases)
            dict_df_backtest_metrics["mse_deaths"].append(mse_deaths)

        return dict_df_backtest_metrics

def get_initial_conditions(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params_fitted 
    N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed

    PopulationR = min(R_upperbound - 1, min(int(R_0*p_d), R_heuristic))
    PopulationCI = (PopulationI - PopulationD - PopulationR)*k3

    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d)
        - (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    UR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    UD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
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
        S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0, DQD_0, R_0,
        D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0,
    ]
    return x_0_cases


def get_initial_conditions_with_testing(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2) when including testing in the modeling
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, k1, k2, beta_0, beta_1 = params_fitted
    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v = (
        global_params_fixed
    )
    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d)
        - (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    UR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    UD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
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
        S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0,
        DQD_0, R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0,
    ]
    return x_0_cases


def create_fitting_data_from_validcases(validcases: pd.DataFrame) -> (float, float, list, list, list):
    """
    Creates the balancing coefficient (regularization coefficient between cases & deaths in cost function) as well as
    the cases and deaths data on which to be fitted
    :param validcases: Dataframe containing cases and deaths data on the relevant time period for our optimization
    :return: the balancing coefficient and two lists containing cases and deaths over the right time span for fitting
    """
    validcases_nondeath = validcases["case_cnt"].tolist()
    validcases_death = validcases["death_cnt"].tolist()
    # balance = validcases_nondeath[-1] / max(validcases_death[-1], 10)
    cases_data_fit = validcases_nondeath
    deaths_data_fit = validcases_death
    weights = list(range(1, len(cases_data_fit) + 1))
    # weights = [(x/len(cases_data_fit))**2 for x in weights]
    balance = np.average(cases_data_fit, weights = weights) / max(np.average(deaths_data_fit, weights = weights), 10)
    balance_total_difference = np.average(np.abs(np.array(cases_data_fit[7:])-np.array(cases_data_fit[:-7])), weights = weights[7:]) / np.average(np.abs(cases_data_fit), weights = weights)
    return balance, balance_total_difference, cases_data_fit, deaths_data_fit, weights


def get_residuals_value(
        optimizer: str, balance: float, x_sol: list, cases_data_fit: list, deaths_data_fit: list, weights: list, balance_total_difference: float
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
    if optimizer in ["trust-constr"]:
        residuals_value = sum(
            np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            + balance
            * balance
            * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        )
    elif optimizer in ["tnc", "annealing"]:
        residuals_value =  sum(      
            np.multiply(
                (x_sol[15, 7:] - x_sol[15, :-7] - cases_data_fit[7:] + cases_data_fit[:-7]) ** 2,
                weights[7:],
            )
            + balance * balance * np.multiply(
                (x_sol[14, 7:] - x_sol[14, :-7] - deaths_data_fit[7:] + deaths_data_fit[:-7]) ** 2,
                weights[7:],
            )
        ) + sum(
            np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            + balance
            * balance
            * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        ) * balance_total_difference * balance_total_difference
    else:
        raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

    return residuals_value


def get_mape_data_fitting(cases_data_fit: list, deaths_data_fit: list, x_sol_final: np.array) -> float:
    """
    Computes MAPE on cases & deaths (averaged) either on last 15 days of historical data (if there are more than 15)
    or exactly the number of days in the historical data (if less than 15)
    :param cases_data_fit: list, contains data used to fit on number of cases
    :param deaths_data_fit: list, contains data used to fit on number of deaths
    :param x_sol_final: numpy array, contains the predicted solution by the DELPHI model for all 16 states
    :return: a float corresponding to the average MAPE on cases and deaths on a given period of time (15 days is default
    but otherwise the number of days available in the historical data)
    """
    if len(cases_data_fit) > 15:  # In which case we can compute MAPE on last 15 days
        mape_data = (
                compute_mape(
                    cases_data_fit[-15:],
                    x_sol_final[15, len(cases_data_fit) - 15: len(cases_data_fit)],
                ) + compute_mape(
                    deaths_data_fit[-15:],
                    x_sol_final[14, len(deaths_data_fit) - 15: len(deaths_data_fit)],
                )
        ) / 2
    else:  # We take MAPE on all available previous days (less than 15)
        mape_data = (
                compute_mape(cases_data_fit, x_sol_final[15, : len(cases_data_fit)])
                + compute_mape(deaths_data_fit, x_sol_final[14, : len(deaths_data_fit)])
        ) / 2

    return mape_data


def compute_sign_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the sign of the Mean Percentage Error, mainly to know if we're constantly over or undershooting
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float, +1 or -1 for the sign of the MPE
    """
    # Mean Percentage Error, without the absolute value
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mpe = np.mean((y_true - y_pred)[y_true > 0] / y_true[y_true > 0]) * 100
    sign = np.sign(mpe)
    return sign


def compute_mape_daily_delta_since_last_train(
    true_last_train: list, pred_last_train: list, y_true: list, y_pred: list
) -> float:
    """
    Computed the Mean Absolute Percentage Error between the daily differences of prediction between a previous train
    and a current train true/predicted values
    :param true_last_train: list of true historical values for the previous train considered
    :param pred_last_train: list of predicted values for the previous train considered
    :param y_true: list of true historical values for the current train considered
    :param y_pred: list of predicted values for the current train considered
    :return: a float corresponding between the MAPE between the daily differences of prediction/truth between 2
    different training processes
    """
    delta_true = np.array([y_true_i - true_last_train for y_true_i in y_true])
    delta_pred = np.array([y_pred_i - pred_last_train for y_pred_i in y_pred])
    mape_daily_delta = (
        np.mean(
            np.abs(delta_true - delta_pred)[delta_true > 0] / delta_true[delta_true > 0]
        )
        * 100
    )
    return mape_daily_delta


def compute_mse(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Squared Error between two lists
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float, corresponding to the MSE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)


def compute_mae_and_mape(y_true: list, y_pred: list) -> (float, float):
    """
    Compute the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a tuple of floats, corresponding to (MAE, MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs((y_true - y_pred)))
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    return mae, mape


def compute_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float corresponding to the MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    return mape