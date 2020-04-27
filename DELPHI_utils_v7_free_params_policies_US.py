# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import pandas as pd
import numpy as np
import dateutil.parser as dtparser
from datetime import datetime, timedelta
from typing import Union
from copy import deepcopy


class DELPHIDataSaver:
    def __init__(
            self, path_to_folder_danger_map: str,
            path_to_website_predicted: str,
            df_global_parameters: pd.DataFrame,
            df_global_predictions_since_today: pd.DataFrame,
            df_global_predictions_since_100_cases: pd.DataFrame,
    ):
        self.PATH_TO_FOLDER_DANGER_MAP = path_to_folder_danger_map
        self.PATH_TO_WEBSITE_PREDICTED = path_to_website_predicted
        self.df_global_parameters = df_global_parameters
        self.df_global_predictions_since_today = df_global_predictions_since_today
        self.df_global_predictions_since_100_cases = df_global_predictions_since_100_cases

    def save_all_datasets(self, save_since_100_cases=False):
        today_date_str = "".join(str(datetime.now().date()).split("-"))
        # Save parameters
        self.df_global_parameters.to_csv(
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Parameters_Global_Python_{today_date_str}.csv"
        )
        self.df_global_parameters.to_csv(
            self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Parameters_Global_Python_{today_date_str}.csv"
        )
        # Save predictions since today
        self.df_global_predictions_since_today.to_csv(
            self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Global_Python_{today_date_str}.csv"
        )
        self.df_global_predictions_since_today.to_csv(
            self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_Python_{today_date_str}.csv"
        )
        if save_since_100_cases:
            # Save predictions since 100 cases
            self.df_global_predictions_since_100_cases.to_csv(
                self.PATH_TO_FOLDER_DANGER_MAP + f"/predicted/Global_since100_{today_date_str}.csv"
            )
            self.df_global_predictions_since_100_cases.to_csv(
                self.PATH_TO_WEBSITE_PREDICTED + f"/predicted/Global_since100_{today_date_str}.csv"
            )


class DELPHIDataCreator:
    def __init__(
            self, x_sol_final: np.array, date_day_since100: datetime,
            best_params: np.array, continent: str, country: str, province: str,
    ):
        assert len(best_params) == 13, f"Expected 13 best parameters, got {len(best_params)}"
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


def get_initial_conditions_v5_final_prediction(params_used_init, global_params_fixed):
    r_dth, p_dth, k1, k2 = params_used_init
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


def get_initial_conditions_v4_final_prediction(params_fitted, global_params_fixed):
    alpha, r_dth, p_dth, k1, k2, b0, b1, b2, b3, b4, b5, b6, b7 = params_fitted
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


def get_initial_conditions_v3(params_fitted, global_params_fixed):
    alpha, r_dth, p_dth, k1, k2, b0, b1, b2, b3, b4, b5, b6, b7 = params_fitted
    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed
    S_0 = (
            (N - PopulationCI / p_d) -
            (PopulationCI / p_d * (k1 + k2)) -
            (PopulationR / p_d) -
            (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    # AR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    # DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    # DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    # AD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
    DHD_0 = PopulationCI * p_h * p_dth
    DQD_0 = PopulationCI * (1 - p_h) * p_dth
    # R_0 = PopulationR / p_d
    # D_0 = PopulationD / p_d
    # TH_0 = PopulationCI * p_h
    # DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth)
    # DVD_0 = (PopulationCI * p_h * p_v) * p_dth
    DD_0 = PopulationD
    DT_0 = PopulationI
    x_0_cases = [
        S_0, E_0, I_0, DHD_0, DQD_0, DD_0, DT_0
    ]
    return x_0_cases


def get_initial_conditions_v5(params_used_init, global_params_fixed):
    r_dth, p_dth, k1, k2 = params_used_init
    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed
    S_0 = (
            (N - PopulationCI / p_d) -
            (PopulationCI / p_d * (k1 + k2)) -
            (PopulationR / p_d) -
            (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    # AR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    # DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    # DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    # AD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
    DHD_0 = PopulationCI * p_h * p_dth
    DQD_0 = PopulationCI * (1 - p_h) * p_dth
    # R_0 = PopulationR / p_d
    # D_0 = PopulationD / p_d
    # TH_0 = PopulationCI * p_h
    # DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth)
    # DVD_0 = (PopulationCI * p_h * p_v) * p_dth
    DD_0 = PopulationD
    DT_0 = PopulationI
    x_0_cases = [
        S_0, E_0, I_0, DHD_0, DQD_0, DD_0, DT_0
    ]
    return x_0_cases


def get_initial_conditions_v7_free_params(params_fitted, global_params_fixed):
    alpha, r_dth, p_dth, k1, k2, b0, b1, b2, b3, b4, b5, b6, b7 = params_fitted
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


def preprocess_past_parameters_and_historical_data_v5(
        continent: str, country: str, province: str,
        totalcases: pd.DataFrame, pastparameters: Union[None, pd.DataFrame]
):
    if totalcases.day_since100.max() < 0:
        print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
        return None, None, None, None, None, None, None, None, None

    if pastparameters is None:
        raise ValueError(f"No past parameters for {country}, {province}, can't run full model")
        # parameter_list = [1, 0.2, 0.05, 3, 3, 0]
        # bounds_params = (
        #    (0.75, 1.25), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10), (-2, 2)
        # )
        # date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
        # validcases = totalcases[totalcases.day_since100 >= 0][
        #    ["day_since100", "case_cnt", "death_cnt"]
        # ].reset_index(drop=True)
        # balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)
    else:
        parameter_list_total = pastparameters[
            (pastparameters.Country == country) &
            (pastparameters.Province == province)
        ].reset_index(drop=True)
        if len(parameter_list_total) > 0:
            # TODO: Add the day since 100 parameter because needed later + deal with initial value of other parameters
            parameter_list_line = parameter_list_total.loc[
                len(parameter_list_total) - 1,
                ["Data Start Date", "Infection Rate", "Rate of Death", "Mortality Rate",
                 "Internal Parameter 1", "Internal Parameter 2"]
            ].tolist()
            date_day_since100 = pd.to_datetime(parameter_list_line[0])
            #alpha_past_params = parameter_list_line[1]
            #alpha_bounds = (0, 2)
            #dict_nonfitted_params = {
            #    "r_dth": parameter_list_line[2],
            #    "p_dth": parameter_list_line[3],
            #    "k1": parameter_list_line[4],
            #    "k2": parameter_list_line[5],
            #}

            parameter_list = parameter_list_line[1:6]  # b_0, b_1, b_2, b_3, b_4, b_5, b_6, alpha, r_dth, p_dth, k1, k2
            # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
            param_list_lower = [x - 0.5 * abs(x) for x in parameter_list]
            param_list_upper = [x + 0.5 * abs(x) for x in parameter_list]
            bounds_params = tuple(
                [(lower, upper)
                 for lower, upper in zip(param_list_lower, param_list_upper)]
            )

            validcases = totalcases[[
                dtparser.parse(x) >= dtparser.parse(parameter_list_line[0])
                for x in totalcases.date
            ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)
        else:
            raise ValueError(f"No past parameters for {country}, {province}, can't run full model")
            # Otherwise use established lower/upper bounds
            #parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
            #bounds_params = (
            #    (0.75, 1.25), (-30, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
            #)
            #date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
            #validcases = totalcases[totalcases.day_since100 >= 0][
            #    ["day_since100", "case_cnt", "death_cnt"]
            #].reset_index(drop=True)
            #balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)

    # Maximum timespan of prediction, defaulted to go to 15/06/2020
    maxT = (datetime(2020, 6, 15) - date_day_since100).days + 1
    return (
        maxT, date_day_since100, validcases, balance,
        fitcasesnd, fitcasesd, parameter_list, bounds_params
    )


def preprocess_past_parameters_and_historical_data_v3(
    continent: str, country: str, province: str,
    totalcases: pd.DataFrame, pastparameters: Union[None, pd.DataFrame]
):
    if totalcases.day_since100.max() < 0:
        print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
        return None, None, None, None, None, None, None, None

    if pastparameters is None:
        raise ValueError("No pas parameters for this country")
        # TODO Initialization of param list was modified for V3, deleted days & r_s and added b_0 at the end
        parameter_list = [1, 0.2, 0.05, 3, 3, 0]
        bounds_params = (
            (0.75, 1.25), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10), (-2, 2)
        )
        date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
        validcases = totalcases[totalcases.day_since100 >= 0][
            ["day_since100", "case_cnt", "death_cnt"]
        ].reset_index(drop=True)
        balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)
    else:
        parameter_list_total = pastparameters[
            (pastparameters.Country == country) &
            (pastparameters.Province == province)
        ]
        if len(parameter_list_total) > 0:
            parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
            parameter_list = parameter_list_line[5:]
            # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
            param_list_lower = [x - 0.05 * abs(x) for x in parameter_list]
            param_list_upper = [x + 0.05 * abs(x) for x in parameter_list]
            bounds_params = tuple(
                [(lower, upper)
                 for lower, upper in zip(param_list_lower, param_list_upper)]
            )
            date_day_since100 = pd.to_datetime(parameter_list_line[3])
            validcases = totalcases[[
                dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                for x in totalcases.date
            ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
            balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)
        else:
            # Otherwise use established lower/upper bounds
            parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
            bounds_params = (
                (0.75, 1.25), (-30, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
            )
            date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].item())
            validcases = totalcases[totalcases.day_since100 >= 0][
                ["day_since100", "case_cnt", "death_cnt"]
            ].reset_index(drop=True)
            balance, fitcasesnd, fitcasesd = create_fitting_data_from_validcases(validcases)

    # Maximum timespan of prediction, defaulted to go to 15/06/2020
    maxT = (datetime(2020, 6, 15) - date_day_since100).days + 1
    return (
        maxT, date_day_since100, validcases, balance,
        fitcasesnd, fitcasesd, parameter_list, bounds_params
    )


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


def read_mobility_data():
    mobility = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv')
    mobility_states = mobility[mobility['sub_region_2'].apply(lambda x: str(x) == 'nan')][
        mobility['sub_region_1'].apply(lambda x: str(x) != 'nan')
    ].query("country_region_code == 'US'").dropna(axis=1)
    mobility_states.drop("country_region", axis=1, inplace=True)
    mobility_states.columns = [
        "country", "province", "date",
        "mobility_retail_recreation", "mobility_grocery_pharmacy",
        "mobility_parks", "mobility_transit", "mobility_workplaces", "mobility_residential"
    ]
    for col in ["mobility_retail_recreation", "mobility_grocery_pharmacy",
        "mobility_parks", "mobility_transit", "mobility_workplaces", "mobility_residential"]:
        mobility_states[col] = mobility_states[col] / 100
    mobility_states["date"] = pd.to_datetime(mobility_states.date)
    mobility_states.reset_index(drop=True, inplace=True)
    return mobility_states


def query_mobility_data_tuple(
        mobility_data: pd.DataFrame,
        country: str,
        province: str,
        date_day_since100: datetime,
        maxT: int,
):
    mobility_data_i = mobility_data[
        (mobility_data.country == country) &
        (mobility_data.province == province) &
        (mobility_data.date >= date_day_since100)
        ].drop(["country", "province", "date"], axis=1).reset_index(drop=True)
    mobility_data_i.columns = [f"mobility_{i + 1}" for i in range(len(mobility_data_i.columns))]
    length_to_complete_for_prediction = maxT - len(mobility_data_i)
    df_to_append_measures_i = pd.DataFrame({
        f"mobility_{i + 1}": [
            mobility_data_i.loc[len(mobility_data_i) - 1, f"mobility_{i + 1}"].item()
            for _ in range(length_to_complete_for_prediction)
        ]
        for i in range(len(mobility_data_i.columns))
    })
    mobility_data_i = pd.concat([mobility_data_i, df_to_append_measures_i]).reset_index(drop=True)
    return mobility_data_i


def read_pop_density_data():
    # Population density is by square mile
    pop_density = pd.read_csv(
        "https://raw.githubusercontent.com/camillol/cs424p3/master/data/Population-Density%20By%20State.csv"
    )
    pop_density.columns = ["col1", "col2", "province", "pop_density"]
    pop_density.drop(["col1", "col2"], axis=1, inplace=True)
    pop_density["country"] = "US"
    pop_density = pop_density[["country", "province", "pop_density"]]
    pop_density = pop_density[pop_density.province != "District of Columbia"]
    pop_density.reset_index(drop=True, inplace=True)
    pop_density["pop_density"] = pop_density["pop_density"] / pop_density["pop_density"].max()
    return pop_density


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
    msr = ['NO_MEASURE', 'MASS_GATHERINGS_ONLY', 'MASS_GATHERINGS_PERMITTED_BUT_OTHERS',
           'MASS_GATHERINGS_AND_SCHOOLS_ONLY', 'MASS_GATHERINGS_AND_OTHERS_NO_SCHOOLS',
           'MASS_GATHERINGS_AND_SCHOOLS_AND_OTHERS', 'LOCKDOWN']
    output['NO_MEASURE'] = (df_final.sum(axis=1) == 0).apply(lambda x: int(x))
    output['MASS_GATHERINGS_ONLY'] = [int(a and b) for a, b in
                                      zip(df_final.sum(axis=1) == 1, df_final['Mass_Gathering_Restrictions'] == 1)]
    output['MASS_GATHERINGS_PERMITTED_BUT_OTHERS'] = [int(a and b and c) for a, b, c in
                                                      zip(df_final.sum(axis=1) > 0,
                                                          df_final['Mass_Gathering_Restrictions'] == 0,
                                                          df_final['Stay_at_home_order'] == 0)]
    output['MASS_GATHERINGS_AND_SCHOOLS_ONLY'] = [
        int(a and b and c)
        for a, b, c in zip(
            df_final.sum(axis=1) == 2,
            df_final['Educational_Facilities_Closed'] == 1,
            df_final['Mass_Gathering_Restrictions'] == 1)
    ]
    output['MASS_GATHERINGS_AND_OTHERS_NO_SCHOOLS'] = [
        int(a and b and c and d) for a, b, c, d in
        zip(df_final.sum(axis=1) > 1,
            df_final['Educational_Facilities_Closed'] == 0,
            df_final['Mass_Gathering_Restrictions'] == 1,
            df_final['Stay_at_home_order'] == 0)
    ]
    output['MASS_GATHERINGS_AND_SCHOOLS_AND_OTHERS'] = [
        int(a and b and c and d) for a, b, c, d in
        zip(df_final.sum(axis=1) > 2,
            df_final['Educational_Facilities_Closed'] == 1,
            df_final['Mass_Gathering_Restrictions'] == 1,
            df_final['Stay_at_home_order'] == 0)
    ]
    output['LOCKDOWN'] = (df_final['Stay_at_home_order'] == 1).apply(lambda x: int(x))
    output['country'] = "US"
    output = output.loc[:, ['country', 'province', 'date'] + msr]
    return output


def query_us_policy_data_tuple(
        policy_data_us_only: pd.DataFrame,
        country: str,
        province: str,
        date_day_since100: datetime,
        maxT: int,
):
    policy_data_us_only_i = policy_data_us_only[
        (policy_data_us_only.country == country) &
        (policy_data_us_only.province == province) &
        (policy_data_us_only.date >= date_day_since100)
        ].drop(["country", "province", "date"], axis=1).reset_index(drop=True)
    policy_data_us_only_i.columns = [f"policy_{i + 1}" for i in range(len(policy_data_us_only_i.columns))]
    length_to_complete_for_prediction = maxT - len(policy_data_us_only_i)
    if length_to_complete_for_prediction > 0:
        df_to_append_measures_i = pd.DataFrame({
            f"policy_{i + 1}": [
                policy_data_us_only_i.loc[len(policy_data_us_only_i) - 1, f"policy_{i + 1}"].item()
                for _ in range(length_to_complete_for_prediction)
            ]
            for i in range(len(policy_data_us_only_i.columns))
        })
        policy_data_us_only_i = pd.concat([policy_data_us_only_i, df_to_append_measures_i]).reset_index(drop=True)
    n_policies = len(policy_data_us_only_i.columns)
    assert n_policies == 7, f"Expected 7 possible measures/policies, got {n_policies}"
    return policy_data_us_only_i
