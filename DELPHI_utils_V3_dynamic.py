# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
from itertools import compress
from DELPHI_params_V3 import MAPPING_STATE_CODE_TO_STATE_NAME, future_policies


def get_bounds_params_from_pastparams(
        optimizer: str, parameter_list: list, dict_default_reinit_parameters: dict, percentage_drift_lower_bound: float,
        default_lower_bound: float, dict_default_reinit_lower_bounds: dict, percentage_drift_upper_bound: float,
        default_upper_bound: float, dict_default_reinit_upper_bounds: dict,
        percentage_drift_lower_bound_annealing: float, default_lower_bound_annealing: float,
        percentage_drift_upper_bound_annealing: float, default_upper_bound_annealing: float,
        default_lower_bound_jump: float, default_upper_bound_jump: float, default_lower_bound_std_normal: float,
        default_upper_bound_std_normal: float,
) -> list:
    """

    :param optimizer:
    :param parameter_list:
    :param dict_default_reinit_parameters:
    :param percentage_drift_lower_bound:
    :param default_lower_bound:
    :param dict_default_reinit_lower_bounds:
    :param percentage_drift_upper_bound:
    :param default_upper_bound:
    :param dict_default_reinit_upper_bounds:
    :param percentage_drift_lower_bound_annealing:
    :param default_lower_bound_annealing:
    :param percentage_drift_upper_bound_annealing:
    :param default_upper_bound_annealing:
    :param default_lower_bound_jump:
    :param default_upper_bound_jump:
    :param default_lower_bound_std_normal:
    :param default_upper_bound_std_normal:
    :return:
    """
    if optimizer in ["tnc", "trust-constr"]:
        # Allowing a drift for parameters
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = parameter_list
        parameter_list = [
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
            max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
            max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
        ]
        param_list_lower = [x - max(percentage_drift_lower_bound * abs(x), default_lower_bound) for x in parameter_list]
        (
            alpha_lower, days_lower, r_s_lower, r_dth_lower, p_dth_lower, r_dthdecay_lower,
            k1_lower, k2_lower, jump_lower, t_jump_lower, std_normal_lower
        ) = param_list_lower
        param_list_lower = [
            max(alpha_lower, dict_default_reinit_lower_bounds["alpha"]),
            days_lower,
            max(r_s_lower, dict_default_reinit_lower_bounds["r_s"]),
            max(min(r_dth_lower, 1), dict_default_reinit_lower_bounds["r_dth"]),
            max(min(p_dth_lower, 1), dict_default_reinit_lower_bounds["p_dth"]),
            max(r_dthdecay_lower, dict_default_reinit_lower_bounds["r_dthdecay"]),
            max(k1_lower, dict_default_reinit_lower_bounds["k1"]),
            max(k2_lower, dict_default_reinit_lower_bounds["k2"]),
            max(jump_lower, dict_default_reinit_lower_bounds["jump"]),
            max(t_jump_lower, dict_default_reinit_lower_bounds["t_jump"]),
            max(std_normal_lower, dict_default_reinit_lower_bounds["std_normal"]),
        ]
        param_list_upper = [
            x + max(percentage_drift_upper_bound * abs(x), default_upper_bound) for x in parameter_list
        ]
        (
            alpha_upper, days_upper, r_s_upper, r_dth_upper, p_dth_upper, r_dthdecay_upper,
            k1_upper, k2_upper, jump_upper, t_jump_upper, std_normal_upper
        ) = param_list_upper
        param_list_upper = [
            max(alpha_upper, dict_default_reinit_upper_bounds["alpha"]),
            days_upper,
            max(r_s_upper, dict_default_reinit_upper_bounds["r_s"]),
            max(min(r_dth_upper, 1), dict_default_reinit_upper_bounds["r_dth"]),
            max(min(p_dth_upper, 1), dict_default_reinit_upper_bounds["p_dth"]),
            max(r_dthdecay_upper, dict_default_reinit_upper_bounds["r_dthdecay"]),
            max(k1_upper, dict_default_reinit_upper_bounds["k1"]),
            max(k2_upper, dict_default_reinit_upper_bounds["k2"]),
            max(jump_upper, dict_default_reinit_upper_bounds["jump"]),
            max(t_jump_upper, dict_default_reinit_upper_bounds["t_jump"]),
            max(std_normal_upper, dict_default_reinit_upper_bounds["std_normal"]),
        ]
    elif optimizer == "annealing":  # Annealing procedure for global optimization
        param_list_lower = [
            x - max(percentage_drift_lower_bound_annealing * abs(x), default_lower_bound_annealing) for x in
            parameter_list
        ]
        param_list_upper = [
            x + max(percentage_drift_upper_bound_annealing * abs(x), default_upper_bound_annealing) for x in
            parameter_list
        ]
        param_list_lower[8] = default_lower_bound_jump  # jump lower bound
        param_list_upper[8] = default_upper_bound_jump  # jump upper bound
        param_list_lower[10] = default_lower_bound_std_normal  # std_normal lower bound
        param_list_upper[10] = default_upper_bound_std_normal  # std_normal upper bound
    else:
        raise ValueError(f"Optimizer {optimizer} not supported in this implementation so can't generate bounds")

    bounds_params = [(lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)]
    return bounds_params


def convert_dates_us_policies(x):
    if x == "Not implemented":
        return np.nan
    else:
        x_long = x + "20"
        return pd.to_datetime(x_long, format="%d-%b-%Y")


def check_us_policy_data_consistency(policies: list, df_policy_raw_us: pd.DataFrame):
    for policy in policies:
        assert (
            len(
                df_policy_raw_us.loc[
                    (df_policy_raw_us[f"{policy}_start_date"].isnull())
                    & (~df_policy_raw_us[f"{policy}_end_date"].isnull()),
                    :,
                ]
            )
            == 0
        ), f"Problem in data, policy {policy} has no start date but has an end date"


def create_features_from_ihme_dates(
    df_policy_raw_us: pd.DataFrame, dict_state_to_policy_dates: dict, policies: list
) -> pd.DataFrame:
    list_df_concat = []
    n_dates = (datetime.now() - datetime(2020, 3, 1)).days + 1
    date_range = [datetime(2020, 3, 1) + timedelta(days=i) for i in range(n_dates)]
    for location in df_policy_raw_us.location_name.unique():
        df_temp = pd.DataFrame(
            {
                "continent": ["North America" for _ in range(len(date_range))],
                "country": ["US" for _ in range(len(date_range))],
                "province": [location for _ in range(len(date_range))],
                "date": date_range,
            }
        )
        for policy in policies:
            start_date_policy_location = dict_state_to_policy_dates[location][policy][0]
            start_date_policy_location = (
                start_date_policy_location
                if start_date_policy_location is not np.nan
                else "2030-01-02"
            )
            end_date_policy_location = dict_state_to_policy_dates[location][policy][1]
            end_date_policy_location = (
                end_date_policy_location
                if end_date_policy_location is not np.nan
                else "2030-01-01"
            )
            df_temp[policy] = 0
            df_temp.loc[
                (
                    (df_temp.date >= start_date_policy_location)
                    & (df_temp.date <= end_date_policy_location)
                ),
                policy,
            ] = 1

        list_df_concat.append(df_temp)

    df_policies_US = pd.concat(list_df_concat).reset_index(drop=True)
    df_policies_US.rename(
        columns={
            "travel_limit": "Travel_severely_limited",
            "stay_home": "Stay_at_home_order",
            "educational_fac": "Educational_Facilities_Closed",
            "any_gathering_restrict": "Mass_Gathering_Restrictions",
            "any_business": "Initial_Business_Closure",
            "all_non-ess_business": "Non_Essential_Services_Closed",
        },
        inplace=True,
    )
    return df_policies_US


def create_final_policy_features_us(df_policies_US: pd.DataFrame) -> pd.DataFrame:
    df_policies_US_final = deepcopy(df_policies_US)
    msr = future_policies
    df_policies_US_final[msr[0]] = (df_policies_US.sum(axis=1) == 0).apply(
        lambda x: int(x)
    )
    df_policies_US_final[msr[1]] = [
        int(a and b)
        for a, b in zip(
            df_policies_US.sum(axis=1) == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
        )
    ]
    df_policies_US_final[msr[2]] = [
        int(a and b and c)
        for a, b, c in zip(
            df_policies_US.sum(axis=1) > 0,
            df_policies_US["Mass_Gathering_Restrictions"] == 0,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[3]] = [
        int(a and b and c)
        for a, b, c in zip(
            df_policies_US.sum(axis=1) == 2,
            df_policies_US["Educational_Facilities_Closed"] == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
        )
    ]
    df_policies_US_final[msr[4]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            df_policies_US.sum(axis=1) > 1,
            df_policies_US["Educational_Facilities_Closed"] == 0,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[5]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            df_policies_US.sum(axis=1) > 2,
            df_policies_US["Educational_Facilities_Closed"] == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[6]] = (df_policies_US["Stay_at_home_order"] == 1).apply(
        lambda x: int(x)
    )
    df_policies_US_final["country"] = "US"
    df_policies_US_final = df_policies_US_final.loc[
        :, ["country", "province", "date"] + msr
    ]
    return df_policies_US_final


def read_policy_data_us_only(filepath_data_sandbox: str):
    policies = [
        "travel_limit", "stay_home", "educational_fac", "any_gathering_restrict",
        "any_business", "all_non-ess_business",
    ]
    list_US_states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
        "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
        "Washington", "West Virginia", "Wisconsin", "Wyoming",
    ]
    df = pd.read_csv(filepath_data_sandbox + "12062020_raw_policy_data_us_only.csv")
    df = df[df.location_name.isin(list_US_states)][
        [
            "location_name", "travel_limit_start_date", "travel_limit_end_date", "stay_home_start_date",
            "stay_home_end_date", "educational_fac_start_date", "educational_fac_end_date",
            "any_gathering_restrict_start_date", "any_gathering_restrict_end_date", "any_business_start_date",
            "any_business_end_date", "all_non-ess_business_start_date", "all_non-ess_business_end_date",
        ]
    ]
    dict_state_to_policy_dates = {}
    for location in df.location_name.unique():
        df_temp = df[df.location_name == location].reset_index(drop=True)
        dict_state_to_policy_dates[location] = {
            policy: [
                df_temp.loc[0, f"{policy}_start_date"],
                df_temp.loc[0, f"{policy}_end_date"],
            ]
            for policy in policies
        }
    check_us_policy_data_consistency(policies=policies, df_policy_raw_us=df)
    df_policies_US = create_features_from_ihme_dates(
        df_policy_raw_us=df,
        dict_state_to_policy_dates=dict_state_to_policy_dates,
        policies=policies,
    )
    df_policies_US_final = create_final_policy_features_us(
        df_policies_US=df_policies_US
    )
    return df_policies_US_final


def read_measures_oxford_data(yesterday: str):
    measures = pd.read_csv(
        "https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv"
    )
    filtr = ["CountryName", "CountryCode", "Date"]
    target = ["ConfirmedCases", "ConfirmedDeaths"]
    msr = [
        "C1_School closing",
        "C2_Workplace closing",
        "C3_Cancel public events",
        "C4_Restrictions on gatherings",
        "C5_Close public transport",
        "C6_Stay at home requirements",
        "C7_Restrictions on internal movement",
        "C8_International travel controls",
        "H1_Public information campaigns",
    ]

    flags = ["C" + str(i) + "_Flag" for i in range(1, 8)] + ["H1_Flag"]
    measures = measures.loc[:, filtr + msr + flags + target]
    measures["Date"] = measures["Date"].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d")
    )
    for col in target:
        # measures[col] = measures[col].fillna(0)
        measures[col] = measures.groupby("CountryName")[col].ffill()

    measures["C1_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C1_School closing"], measures["C1_Flag"])
    ]
    measures["C2_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C2_Workplace closing"], measures["C2_Flag"])
    ]
    measures["C3_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C3_Cancel public events"], measures["C3_Flag"])
    ]
    measures["C4_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["C4_Restrictions on gatherings"], measures["C4_Flag"]
        )
    ]
    measures["C5_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C5_Close public transport"], measures["C5_Flag"])
    ]
    measures["C6_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C6_Stay at home requirements"], measures["C6_Flag"])
    ]
    measures["C7_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["C7_Restrictions on internal movement"], measures["C7_Flag"]
        )
    ]
    measures["H1_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["H1_Public information campaigns"], measures["H1_Flag"]
        )
    ]

    measures["C1_School closing"] = [
        int(a and b)
        for a, b in zip(measures["C1_School closing"] >= 2, measures["C1_Flag"] == 1)
    ]

    measures["C2_Workplace closing"] = [
        int(a and b)
        for a, b in zip(measures["C2_Workplace closing"] >= 2, measures["C2_Flag"] == 1)
    ]

    measures["C3_Cancel public events"] = [
        int(a and b)
        for a, b in zip(
            measures["C3_Cancel public events"] >= 2, measures["C3_Flag"] == 1
        )
    ]

    measures["C4_Restrictions on gatherings"] = [
        int(a and b)
        for a, b in zip(
            measures["C4_Restrictions on gatherings"] >= 1, measures["C4_Flag"] == 1
        )
    ]

    measures["C5_Close public transport"] = [
        int(a and b)
        for a, b in zip(
            measures["C5_Close public transport"] >= 2, measures["C5_Flag"] == 1
        )
    ]

    measures["C6_Stay at home requirements"] = [
        int(a and b)
        for a, b in zip(
            measures["C6_Stay at home requirements"] >= 2, measures["C6_Flag"] == 1
        )
    ]

    measures["C7_Restrictions on internal movement"] = [
        int(a and b)
        for a, b in zip(
            measures["C7_Restrictions on internal movement"] >= 2,
            measures["C7_Flag"] == 1,
        )
    ]

    measures["C8_International travel controls"] = [
        int(a) for a in (measures["C8_International travel controls"] >= 3)
    ]

    measures["H1_Public information campaigns"] = [
        int(a and b)
        for a, b in zip(
            measures["H1_Public information campaigns"] >= 1, measures["H1_Flag"] == 1
        )
    ]

    # measures = measures.loc[:, measures.isnull().mean() < 0.1]
    msr = set(measures.columns).intersection(set(msr))

    # measures = measures.fillna(0)
    measures = measures.dropna()
    for col in msr:
        measures[col] = measures[col].apply(lambda x: int(x > 0))
    measures = measures[["CountryName", "Date"] + list(sorted(msr))]
    measures["CountryName"] = measures.CountryName.replace(
        {
            "United States": "US",
            "South Korea": "Korea, South",
            "Democratic Republic of Congo": "Congo (Kinshasa)",
            "Czech Republic": "Czechia",
            "Slovak Republic": "Slovakia",
        }
    )

    measures = measures.fillna(0)
    msr = future_policies

    measures["Restrict_Mass_Gatherings"] = [
        int(a or b or c)
        for a, b, c in zip(
            measures["C3_Cancel public events"],
            measures["C4_Restrictions on gatherings"],
            measures["C5_Close public transport"],
        )
    ]
    measures["Others"] = [
        int(a or b or c)
        for a, b, c in zip(
            measures["C2_Workplace closing"],
            measures["C7_Restrictions on internal movement"],
            measures["C8_International travel controls"],
        )
    ]

    del measures["C2_Workplace closing"]
    del measures["C3_Cancel public events"]
    del measures["C4_Restrictions on gatherings"]
    del measures["C5_Close public transport"]
    del measures["C7_Restrictions on internal movement"]
    del measures["C8_International travel controls"]

    output = deepcopy(measures)
    output[msr[0]] = (measures.iloc[:, 2:].sum(axis=1) == 0).apply(lambda x: int(x))
    output[msr[1]] = [
        int(a and b)
        for a, b in zip(
            measures.iloc[:, 2:].sum(axis=1) == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
        )
    ]
    output[msr[2]] = [
        int(a and b and c)
        for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) > 0,
            measures["Restrict_Mass_Gatherings"] == 0,
            measures["C6_Stay at home requirements"] == 0,
        )
    ]
    output[msr[3]] = [
        int(a and b and c)
        for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) == 2,
            measures["C1_School closing"] == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
        )
    ]
    output[msr[4]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 1,
            measures["C1_School closing"] == 0,
            measures["Restrict_Mass_Gatherings"] == 1,
            measures["C6_Stay at home requirements"] == 0,
        )
    ]
    output[msr[5]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 2,
            measures["C1_School closing"] == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
            measures["C6_Stay at home requirements"] == 0,
        )
    ]
    output[msr[6]] = (measures["C6_Stay at home requirements"] == 1).apply(
        lambda x: int(x)
    )
    output.rename(columns={"CountryName": "country", "Date": "date"}, inplace=True)
    output["province"] = "None"
    output = output.loc[:, ["country", "province", "date"] + msr]
    output = output[output.date <= yesterday].reset_index(drop=True)
    return output


def gamma_t(day, state, params_dic):
    dsd, median_day_of_action, rate_of_action = params_dic[state]
    t = (day - pd.to_datetime(dsd)).days
    gamma = (2 / np.pi) * np.arctan(
        -(t - median_day_of_action) / 20 * rate_of_action
    ) + 1
    return gamma


def make_increasing(sequence):
    for i in range(len(sequence)):
        sequence[i] = max(sequence[i],sequence[max(i-1,0)])
    return sequence


def get_normalized_policy_shifts_and_current_policy_us_only(
    policy_data_us_only: pd.DataFrame, pastparameters: pd.DataFrame
):
    dict_current_policy = {}
    policy_list = future_policies
    policy_data_us_only["province_cl"] = policy_data_us_only["province"].apply(
        lambda x: x.replace(",", "").strip().lower()
    )
    states_upper_set = set(policy_data_us_only["province"])
    for state in states_upper_set:
        dict_current_policy[("US", state)] = list(
            compress(
                policy_list,
                (
                    policy_data_us_only.query("province == @state")[
                        policy_data_us_only.query("province == @state")["date"]
                        == policy_data_us_only.date.max()
                    ][policy_list]
                    == 1
                )
                .values.flatten()
                .tolist(),
            )
        )[0]
    states_set = set(policy_data_us_only["province_cl"])
    pastparameters_copy = deepcopy(pastparameters)
    pastparameters_copy["Province"] = pastparameters_copy["Province"].apply(
        lambda x: str(x).replace(",", "").strip().lower()
    )
    params_dic = {}
    for state in states_set:
        params_dic[state] = pastparameters_copy.query("Province == @state")[
            ["Data Start Date", "Median Day of Action", "Rate of Action"]
        ].iloc[0]

    policy_data_us_only["Gamma"] = [
        gamma_t(day, state, params_dic)
        for day, state in zip(
            policy_data_us_only["date"], policy_data_us_only["province_cl"]
        )
    ]
    n_measures = policy_data_us_only.iloc[:, 3:-2].shape[1]
    dict_normalized_policy_gamma = {
        policy_data_us_only.columns[3 + i]: policy_data_us_only[
            policy_data_us_only.iloc[:, 3 + i] == 1
        ]
        .iloc[:, -1]
        .mean()
        for i in range(n_measures)
    }
    normalize_val = dict_normalized_policy_gamma[policy_list[0]]
    for policy in dict_normalized_policy_gamma.keys():
        dict_normalized_policy_gamma[policy] = (
            dict_normalized_policy_gamma[policy] / normalize_val
        )

    return dict_normalized_policy_gamma, dict_current_policy


def get_normalized_policy_shifts_and_current_policy_all_countries(
    policy_data_countries: pd.DataFrame, pastparameters: pd.DataFrame
):
    dict_current_policy = {}
    policy_list = future_policies
    policy_data_countries["country_cl"] = policy_data_countries["country"].apply(
        lambda x: x.replace(",", "").strip().lower()
    )
    pastparameters_copy = deepcopy(pastparameters)
    pastparameters_copy["Country"] = pastparameters_copy["Country"].apply(
        lambda x: str(x).replace(",", "").strip().lower()
    )
    params_countries = pastparameters_copy["Country"]
    params_countries = set(params_countries)
    policy_data_countries_bis = policy_data_countries.query(
        "country_cl in @params_countries"
    )
    countries_upper_set = set(
        policy_data_countries[policy_data_countries.country != "US"]["country"]
    )
    # countries_in_oxford_and_params = params_countries.intersection(countries_upper_set)
    for country in countries_upper_set:
        dict_current_policy[(country, "None")] = list(
            compress(
                policy_list,
                (
                    policy_data_countries.query("country == @country")[
                        policy_data_countries.query("country == @country")["date"]
                        == policy_data_countries.query("country == @country").date.max()
                    ][policy_list]
                    == 1
                )
                .values.flatten()
                .tolist(),
            )
        )[0]
    countries_common = sorted([x.lower() for x in countries_upper_set])
    pastparam_tuples_in_oxford = pastparameters_copy[
        (pastparameters_copy.Country.isin(countries_common))
        & (pastparameters_copy.Province != "None")
    ].reset_index(drop=True)
    pastparam_tuples_in_oxford["tuple_name"] = list(
        zip(pastparam_tuples_in_oxford.Country, pastparam_tuples_in_oxford.Province)
    )
    for tuple in pastparam_tuples_in_oxford.tuple_name.unique():
        country, province = tuple
        country = country[0].upper() + country[1:]
        dict_current_policy[(country, province)] = dict_current_policy[
            (country, "None")
        ]

    countries_set = set(policy_data_countries["country_cl"])

    params_dic = {}
    countries_set = countries_set.intersection(params_countries)
    for country in countries_set:
        params_dic[country] = pastparameters_copy.query("Country == @country")[
            ["Data Start Date", "Median Day of Action", "Rate of Action"]
        ].iloc[0]

    policy_data_countries_bis["Gamma"] = [
        gamma_t(day, country, params_dic)
        for day, country in zip(
            policy_data_countries_bis["date"], policy_data_countries_bis["country_cl"]
        )
    ]
    n_measures = policy_data_countries_bis.iloc[:, 3:-2].shape[1]
    dict_normalized_policy_gamma = {
        policy_data_countries_bis.columns[3 + i]: policy_data_countries_bis[
            policy_data_countries_bis.iloc[:, 3 + i] == 1
        ]
        .iloc[:, -1]
        .mean()
        for i in range(n_measures)
    }
    normalize_val = dict_normalized_policy_gamma[policy_list[0]]
    for policy in dict_normalized_policy_gamma.keys():
        dict_normalized_policy_gamma[policy] = (
            dict_normalized_policy_gamma[policy] / normalize_val
        )

    return dict_normalized_policy_gamma, dict_current_policy


def get_testing_data_us() -> pd.DataFrame:
    """
    :return: a DataFrame where the column of interest is 'testing_cnt_daily'
    which gives the numbers of new daily tests per state
    """
    df_test = pd.read_csv("https://covidtracking.com/api/v1/states/daily.csv")
    df_test["country"] = "US"
    df_test["continent"] = "North America"
    df_test["province"] = df_test.state.map(MAPPING_STATE_CODE_TO_STATE_NAME)
    df_test.drop("state", axis=1, inplace=True)
    df_test["date"] = df_test.date.apply(
        lambda x: str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:]
    )
    df_test["date"] = pd.to_datetime(df_test.date)
    df_test = df_test.sort_values(["province", "date"]).reset_index(drop=True)
    df_test = df_test[["continent", "country", "province", "date", "totalTestResults"]]
    df_test.rename(columns={"totalTestResults": "testing_cnt"}, inplace=True)
    list_df_concat = []
    for state in df_test.province.unique():
        df_temp = df_test[df_test.province == state].reset_index(drop=True)
        df_temp["testing_cnt_shift"] = df_temp.testing_cnt.shift(1)
        df_temp["testing_cnt_daily"] = df_temp.testing_cnt - df_temp.testing_cnt_shift
        df_temp.loc[0, "testing_cnt_daily"] = df_temp.loc[0, "testing_cnt"]
        list_df_concat.append(df_temp)

    df_test_final = pd.concat(list_df_concat).reset_index(drop=True)
    df_test_final.drop(["testing_cnt", "testing_cnt_shift"], axis=1, inplace=True)
    return df_test_final
