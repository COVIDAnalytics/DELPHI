import os
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime, timedelta
from DELPHI_utils_V3_dynamic import make_increasing
from DELPHI_params_V3 import (
    TIME_DICT,
    default_policy,
    default_policy_enaction_time,
)

def set_initial_states():
    return

def get_initial_conditions_new_wave(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params_fitted 
    N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed

    PopulationR = min(R_upperbound - 1, k3*min(int(R_0*p_d), R_heuristic))
    PopulationCI = PopulationI - PopulationD - PopulationR

    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d * k3)
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
    R_0 = PopulationR / p_d * k3
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
                "Internal Parameter 3": [self.best_params[11]]
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
                    "Active LB": [np.nan for _ in range(len(all_dates_since_100))],
                    "Active Hospitalized LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Cumulative Hospitalized LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Total Detected Deaths LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Active Ventilated LB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Total Detected UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Active UB": [np.nan for _ in range(len(all_dates_since_100))],
                    "Active Hospitalized UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Cumulative Hospitalized UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Total Detected Deaths UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
                    "Active Ventilated UB": [
                        np.nan for _ in range(len(all_dates_since_100))
                    ],
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
        df_agg_since_today_per_country = DELPHIAggregations.get_aggregation_per_country(
            df_predictions
        )
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