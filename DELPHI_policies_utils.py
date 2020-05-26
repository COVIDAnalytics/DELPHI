# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
import json
from DELPHI_params import n_params_without_policy_params


def update_tracking_without_policy_change(
        df_updated: pd.DataFrame, yesterday: str,
) -> pd.DataFrame:
    df_updated.loc[0, "n_days_enacted_current_policy"] = (
        (pd.to_datetime(yesterday) - pd.to_datetime(df_updated.loc[0, "start_date_current_policy"])).days + 1
    )
    df_updated = update_n_params_fitted_without_policy_change(df_updated=df_updated, yesterday=yesterday)
    df_updated.loc[0, "n_params_constant_from_tree"] = (
            df_updated.loc[0, "n_policy_changes"] - df_updated.loc[0, "n_params_fitted"]
    )
    return df_updated


def update_tracking_when_policy_changed(
        df_updated: pd.DataFrame, df_previous: pd.DataFrame, yesterday: str,
        current_policy_in_tracking: str, current_policy_in_new_data: str,
        dict_normalized_policy_gamma_international: dict,
) -> pd.DataFrame:
    df_updated.loc[0, "n_policies_enacted"] = df_previous.loc[
        0, "n_policies_enacted"
    ] + 1
    df_updated.loc[0, "n_policy_changes"] = df_updated.loc[
        0, "n_policies_enacted"
    ] - 1
    # Modifying track of policy changes constant (adding the new one)
    policy_change_id_str = str(int(df_updated.loc[0, "n_policy_changes"] - 1))
    policy_changes_ids_constant = set(df_updated.loc[0, "policy_changes_ids_constant"].strip())
    policy_changes_ids_constant.add(policy_change_id_str)
    new_policy_changes_ids_constant = "".join(policy_changes_ids_constant)
    df_updated.loc[0, "policy_changes_ids_constant"] = new_policy_changes_ids_constant

    df_updated.loc[0, "last_policy"] = current_policy_in_tracking
    df_updated.loc[0, "current_policy"] = current_policy_in_new_data
    df_updated.loc[0, "last_policy_fitted"] = df_previous.loc[0, "current_policy_fitted"]
    df_updated.loc[0, "current_policy_fitted"] = False  # Necessarily as it's 1 day old
    df_updated.loc[0, "start_date_last_policy"] = df_previous.loc[
        0, "start_date_current_policy"
    ]
    df_updated.loc[0, "end_date_last_policy"] = str(pd.to_datetime(yesterday).date())
    df_updated.loc[0, "start_date_current_policy"] = str(pd.to_datetime(yesterday).date())
    df_updated.loc[0, "n_days_enacted_last_policy"] = df_previous.loc[
        0, "n_days_enacted_current_policy"
    ]
    df_updated.loc[0, "n_days_enacted_current_policy"] = 1
    df_updated.loc[0, "policy_shift_names"] = (
            df_updated.loc[0, "policy_shift_names"] + f"{current_policy_in_tracking}->{current_policy_in_new_data};"
    )
    df_updated.loc[0, "policy_shift_dates"] = (
            df_updated.loc[0, "policy_shift_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
    )
    df_updated.loc[0, "policy_shift_initial_param_values"] = add_policy_shift_initial_param(
        df_previous=df_previous, current_policy_in_tracking=current_policy_in_tracking,
        current_policy_in_new_data=current_policy_in_new_data,
        dict_normalized_policy_gamma_international=dict_normalized_policy_gamma_international
    )
    df_updated.loc[0, "n_params_constant_from_tree"] = (
            df_updated.loc[0, "n_policy_changes"] - df_updated.loc[0, "n_params_fitted"]
    )
    return df_updated


def add_policy_shift_initial_param(
        df_previous: pd.DataFrame,
        current_policy_in_tracking: str,
        current_policy_in_new_data: str,
        dict_normalized_policy_gamma_international: dict,
) -> str:
    """
    This corresponds to the new value of k_0' (normalized difference of values in the tree for going from
    one policy to another)
    """
    previous_policy_shift_initial_params = df_previous.loc[0, "policy_shift_initial_param_values"]
    value_new_data = dict_normalized_policy_gamma_international[current_policy_in_new_data]
    value_tracking = dict_normalized_policy_gamma_international[current_policy_in_tracking]
    value_param_init = value_new_data - value_tracking
    updated_policy_shift_names = (
            previous_policy_shift_initial_params + f"{value_param_init};"
    )
    return updated_policy_shift_names


def update_n_params_fitted_without_policy_change(
        df_updated: pd.DataFrame, yesterday: str,
) -> pd.DataFrame:
    """
    When we do not update the policy (i.e. there hasn't been any change in policy) then both current_policy
    and last_policy are susceptible to change so we need to check if we have enough data to fit the parameter
    instead of using a constant value, and if so, update the flags for param fitting 'last/current_policy_fitted'
    """
    n_days_data_before_fitting = 10
    flag_new_param_fitted_current_policy = len(df_updated[
        (df_updated.n_days_enacted_current_policy >= n_days_data_before_fitting)
        & (df_updated.n_policy_changes >= 1)
        & (df_updated.current_policy_fitted == False)
    ]) > 0

    if flag_new_param_fitted_current_policy:
        # The policy change id is the n_policy_changes - 1 (so when there's the first policy change, its id will be '0')
        policy_change_id_str = str(int(df_updated.loc[0, "n_policy_changes"] - 1))
        df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
        df_updated.loc[0, "current_policy_fitted"] = True
        df_updated.loc[0, "params_fitting_starting_dates"] = (
                df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
        )
        # Modifying track of policy changes fitted (adding this one as it's now fitted)
        policy_changes_ids_fitted = set(df_updated.loc[0, "policy_changes_ids_fitted"].strip())
        policy_changes_ids_fitted.add(policy_change_id_str)
        new_policy_changes_ids_fitted = "".join(policy_changes_ids_fitted)
        df_updated.loc[0, "policy_changes_ids_fitted"] = new_policy_changes_ids_fitted

        # Modifying track of policy changes constant (popping it from constant as it's now fitted)
        policy_changes_ids_constant = set(df_updated.loc[0, "policy_changes_ids_constant"].strip())
        policy_changes_ids_constant.remove(policy_change_id_str)
        policy_changes_ids_constant.add(" ")
        new_policy_changes_ids_constant = "".join(policy_changes_ids_constant)
        df_updated.loc[0, "policy_changes_ids_constant"] = new_policy_changes_ids_constant

        # Adding the data to the end dates for this policy
        dict_policy_changes_constant_end_dates = json.loads(df_updated.loc[0, "policy_changes_constant_end_dates"])
        dict_policy_changes_constant_end_dates[policy_change_id_str] = str(pd.to_datetime(yesterday).date())
        df_updated.loc[0, "policy_changes_constant_end_dates"] = json.dumps(dict_policy_changes_constant_end_dates)

    return df_updated


def get_params_fitted_policies(
        df_updated: pd.DataFrame
) -> (list, list):
    params_fitted_present = len(df_updated[df_updated.n_params_fitted >= 1]) > 0
    if not params_fitted_present:
        return [], []
    else:
        policy_shifts_fitted_ids = list(set(df_updated.loc[0, "policy_changes_ids_fitted"].strip()))
        policy_shifts_fitted_ids = [int(x) for x in policy_shifts_fitted_ids]
        n_params_fitted = df_updated.n_params_fitted.iloc[0].astype(int)
        assert n_params_fitted == len(policy_shifts_fitted_ids), "Discrepancy between # fitted ids and n_params_fitted"
        params_fitted = df_updated.policy_shift_initial_param_values.iloc[0].strip().split(";")[:-1]
        # print(type(df_updated.policy_shift_initial_param_values.iloc[0]))
        # print(df_updated.policy_shift_initial_param_values)
        # print(params_fitted)
        params_fitted = [float(params_fitted[policy_id]) for policy_id in policy_shifts_fitted_ids]
        # Actually corresponds to the start date of the policy!
        start_date_fitting = df_updated.policy_shift_dates.iloc[0].strip().split(";")[:-1]
        start_date_fitting = [start_date_fitting[policy_id] for policy_id in policy_shifts_fitted_ids]
        return params_fitted, start_date_fitting


def get_constant_params_end_date(dict_end_dates_constant: dict, policy_shift_id: str, yesterday: str):
    try:
        end_date_policy_id = dict_end_dates_constant[policy_shift_id]
    except KeyError:
        end_date_policy_id = str(pd.to_datetime(yesterday).date())
    return end_date_policy_id


def get_params_constant_policies(
        df_updated: pd.DataFrame, yesterday: str
):
    yesterday_date = str(pd.to_datetime(yesterday).date())
    params_constant_present = len(df_updated[df_updated.n_params_constant_from_tree >= 1]) > 0
    if not params_constant_present:
        return [], []
    else:
        n_params_constant = df_updated.n_params_constant_from_tree.iloc[0].astype(int)
        policy_shifts_constant_ids = list(set(df_updated.loc[0, "policy_changes_ids_constant"].strip()))
        assert n_params_constant == len(policy_shifts_constant_ids), \
            "Discrepancy between # constant ids and n_params_constant"
        params_constant = df_updated.policy_shift_initial_param_values.iloc[0].strip().split(";")[:-1]
        params_constant = [float(params_constant[int(policy_id)]) for policy_id in policy_shifts_constant_ids]
        start_dates_constant = df_updated.policy_shift_dates.iloc[0].strip().split(";")[:-1]
        start_dates_constant = [start_dates_constant[int(policy_id)] for policy_id in policy_shifts_constant_ids]
        dict_policy_changes_constant_end_dates = json.loads(df_updated.loc[0, "policy_changes_constant_end_dates"])
        end_dates_constant = [
            get_constant_params_end_date(
                dict_end_dates_constant=dict_policy_changes_constant_end_dates,
                policy_shift_id=policy_shift_id,
                yesterday=yesterday
            )
            for policy_shift_id in policy_shifts_constant_ids
        ]
        assert len(end_dates_constant) == len(start_dates_constant), "# constant start dates != # constant end dates"
        start_end_dates_constant = [
            (start_date, end_date) for start_date, end_date
            in zip(start_dates_constant[-n_params_constant:], end_dates_constant[-n_params_constant:])
        ]
        return params_constant, start_end_dates_constant


def get_policy_names_before_after_fitted(df_updated: pd.DataFrame):
    """
    :return: a list of tuples like [(policy_str_before_shift, policy_str_after_shift)],
    here for the fitted params only
    """
    policy_shifts_fitted_ids = list(set(df_updated.loc[0, "policy_changes_ids_fitted"].strip()))
    policy_shifts_fitted_ids = [int(shift_id_str) for shift_id_str in policy_shifts_fitted_ids]
    all_policy_shifts_tuples = get_policy_shift_names_tuples(df_updated=df_updated)
    all_policy_shifts_tuples_fitted = [
        all_policy_shifts_tuples[shift_id_fitted] for shift_id_fitted in policy_shifts_fitted_ids
    ]
    return all_policy_shifts_tuples_fitted


def get_policy_names_before_after_constant(df_updated: pd.DataFrame):
    """
    :return: a list of tuples like [(policy_str_before_shift, policy_str_after_shift)],
    here for the constant params only
    """
    policy_shifts_constant_ids = list(set(df_updated.loc[0, "policy_changes_ids_constant"].strip()))
    policy_shifts_constant_ids = [int(shift_id_str) for shift_id_str in policy_shifts_constant_ids]
    all_policy_shifts_tuples = get_policy_shift_names_tuples(df_updated=df_updated)
    all_policy_shifts_tuples_constant = [
        all_policy_shifts_tuples[shift_id_constant] for shift_id_constant in policy_shifts_constant_ids
    ]
    return all_policy_shifts_tuples_constant


def get_policy_shift_names_tuples(df_updated: pd.DataFrame):
    policy_shift_names_raw = df_updated.loc[0, "policy_shift_names"].strip().split(";")[:-1]
    policy_shift_names_tuples = [
        (policy_shift.split("->")[0], policy_shift.split("->")[1])
        for policy_shift in policy_shift_names_raw
    ]
    return policy_shift_names_tuples


def get_list_and_bounds_params(
        df_updated: pd.DataFrame, parameter_list_line: list, param_MATHEMATICA: bool
):
    # if param_MATHEMATICA:
    #     parameter_list_fitted = parameter_list_line[4:]
    #     parameter_list_fitted[3] = np.log(2) / parameter_list_fitted[3]
    # else:
    parameter_list_fitted = parameter_list_line[5:]

    params_policies_fitted, start_dates_fitting_policies = get_params_fitted_policies(df_updated)
    parameter_list_fitted = parameter_list_fitted + params_policies_fitted
    param_bounds_list_lower = [x - 0.1 * abs(x) for x in parameter_list_fitted]
    param_bounds_list_upper = [x + 0.1 * abs(x) for x in parameter_list_fitted]
    bounds_params_fitted = tuple(
        [(lower, upper) for lower, upper in zip(param_bounds_list_lower, param_bounds_list_upper)]
    )
    assert len(start_dates_fitting_policies) == len(parameter_list_fitted) - n_params_without_policy_params, \
        "Should have as many extra fitted policy params as start fitting dates for policy params"
    return parameter_list_fitted, start_dates_fitting_policies, bounds_params_fitted


def add_policy_tracking_row_country(
        continent: str, country: str, province: str, yesterday: str, dict_current_policy_international:dict
):
    "continent, country, province, date, n_policies_enacted, n_policy_changes, last_policy, \
        start_date_last_policy, end_date_last_policy, n_days_enacted_last_policy, current_policy, \
        start_date_current_policy, end_date_current_policy, n_days_enacted_current_policy, policy_shift_names, \
        policy_shift_dates, policy_shift_initial_param_values, n_params_fitted, current_policy_fitted, \
        last_policy_fitted, params_fitting_starting_dates, n_params_constant_from_tree"
    yesterday_date = str(pd.to_datetime(yesterday).date())
    current_policy = dict_current_policy_international[(country, province)]
    dict_policy_change_tracking_area = {
        "continent": [continent], "country": [country], "province": [province], "date": [yesterday_date],
        "n_policies_enacted": [1], "n_policy_changes": [0], "last_policy": [current_policy],
        "start_date_last_policy": [yesterday_date], "end_date_last_policy": [" "], "n_days_enacted_last_policy": [1],
        "current_policy": [current_policy], "start_date_current_policy": [yesterday_date],
        "end_date_current_policy": [" "], "n_days_enacted_current_policy": [1], "policy_shift_names": [" "],
        "policy_shift_dates": [" "], "policy_shift_initial_param_values": [" "],"n_params_fitted": [0],
        "current_policy_fitted": [" "], "last_policy_fitted": [" "], "params_fitting_starting_dates": [" "],
        "n_params_constant_from_tree": [0], "policy_changes_ids_fitted": [" "], "policy_changes_ids_constant": [" "],
        "policy_changes_constant_end_dates": [json.dumps({})]  # This is going to be a json dictionary
    }
    df_policy_change_tracking_area = pd.DataFrame(dict_policy_change_tracking_area)
    return df_policy_change_tracking_area


def update_tracking_fitted_params(
        df_updated: pd.DataFrame,
        n_policy_shifts_fitted: int,
        new_best_params_fitted_policies: list,
):
    if len(new_best_params_fitted_policies) == 0:
        return df_updated
    else:
        policy_shifts_fitted_ids = list(set(df_updated.loc[0, "policy_changes_ids_fitted"].strip()))
        policy_shifts_fitted_ids = [int(x) for x in policy_shifts_fitted_ids]
        assert len(new_best_params_fitted_policies) == len(policy_shifts_fitted_ids) == n_policy_shifts_fitted, \
            "Supposed to have as many policy shift ids as new best params for fitted policies"
        params_all_saved = np.array(df_updated.policy_shift_initial_param_values.iloc[0].strip().split(";")[:-1])
        params_all_saved = np.array([float(x) for x in params_all_saved])
        for i, j in zip(policy_shifts_fitted_ids, range(len(new_best_params_fitted_policies))):
            params_all_saved[i] = new_best_params_fitted_policies[j]

        params_all_updated = [
            str(new_best_params_fitted_policies[i]) if i in policy_shifts_fitted_ids else str(params_all_saved[i])
            for i in range(len(params_all_saved))
        ]
        params_all_updated_str = ";".join(params_all_updated) + ";"
        df_updated.loc[0, "policy_shift_initial_param_values"] = str(params_all_updated_str)
    return df_updated
