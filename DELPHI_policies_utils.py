# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union
from copy import deepcopy
from itertools import compress
import json
from DELPHI_params import (n_params_without_policy_params, TIME_DICT, MAPPING_STATE_CODE_TO_STATE_NAME, default_policy,
                           default_policy_enaction_time, future_policies)


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
    #df_updated = update_n_params_fitted_with_policy_change(
    #    df_updated=df_updated,
    #    yesterday=yesterday,
    #)
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


# def update_n_params_fitted_with_policy_change(
#         df_updated: pd.DataFrame, yesterday: str
# ) -> pd.DataFrame:
#     """
#     When we update the policy (i.e. when there's been a change in policy) the former current_policy becomes last_policy
#     and the new current_policy is for sure not fitted as it is 1 day old. If it is ready to be fitted, we need to update
#     the flags for param fitting 'last/current_policy_fitted'
#     """
#     n_days_data_before_fitting = 10
#     flag_new_param_fitted = len(df_updated[
#         (df_updated.n_days_enacted_last_policy >= n_days_data_before_fitting)
#         & (df_updated.n_policy_changes >= 1)
#         & (df_updated.last_policy_fitted == False)
#     ]) > 0
#     if flag_new_param_fitted:
#         df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
#         df_updated.loc[0, "last_policy_fitted"] = True
#         df_updated.loc[0, "params_fitting_starting_dates"] = (
#                 df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
#         )
#
#     return df_updated


def update_n_params_fitted_without_policy_change(
        df_updated: pd.DataFrame, yesterday: str,
) -> pd.DataFrame:
    """
    When we do not update the policy (i.e. there hasn't been any change in policy) then both current_policy
    and last_policy are susceptible to change so we need to check if we have enough data to fit the parameter
    instead of using a constant value, and if so, update the flags for param fitting 'last/current_policy_fitted'
    """
    n_days_data_before_fitting = 10
    # flag_new_param_fitted_last_policy = len(df_updated[
    #     (df_updated.n_days_enacted_last_policy >= n_days_data_before_fitting)
    #     & (df_updated.n_policy_changes >= 1)
    #     & (df_updated.last_policy_fitted == False)
    # ]) > 0
    flag_new_param_fitted_current_policy = len(df_updated[
        (df_updated.n_days_enacted_current_policy >= n_days_data_before_fitting)
        & (df_updated.n_policy_changes >= 1)
        & (df_updated.current_policy_fitted == False)
    ]) > 0
    # if flag_new_param_fitted_last_policy:
    #     df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
    #     df_updated.loc[0, "last_policy_fitted"] = True
    #     df_updated.loc[0, "params_fitting_starting_dates"] = (
    #             df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
    #     )

    if flag_new_param_fitted_current_policy:
        df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
        df_updated.loc[0, "current_policy_fitted"] = True
        df_updated.loc[0, "params_fitting_starting_dates"] = (
                df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
        )

    return df_updated


def get_params_fitted_policies(
        df_updated: pd.DataFrame
) -> (list, list):
    params_fitted_present = len(df_updated[df_updated.n_params_fitted >= 1]) > 0
    if not params_fitted_present:
        return [], []
    else:
        n_params_fitted = df_updated.n_params_fitted.iloc[0].astype(int)
        params_fitted = df_updated.policy_shift_initial_param_values.iloc[0].strip().split(";")[:-1]
        params_fitted = [float(x) for x in params_fitted]
        # Actually corresponds to the start date of the policy!
        start_date_fitting = df_updated.policy_shift_dates.iloc[0].strip().split(";")[:-1]
        start_date_fitting = start_date_fitting[:n_params_fitted]
        return params_fitted, start_date_fitting


def get_params_constant_policies(
        df_updated: pd.DataFrame, yesterday: str
):
    yesterday_date = str(pd.to_datetime(yesterday).date())
    params_constant_present = len(df_updated[df_updated.n_params_constant_from_tree >= 1]) > 0
    if not params_constant_present:
        return [], []
    else:
        n_params_constant = df_updated.n_params_constant_from_tree.iloc[0].astype(int)
        params_constant = df_updated.policy_shift_initial_param_values.iloc[0].strip().split(";")[:-1]
        # Since all the first ones are going to be fitted, we need to only keep the last n_params_constant
        params_constant = [float(x) for x in params_constant][-n_params_constant:]
        start_dates_constant = df_updated.policy_shift_dates.iloc[0].strip().split(";")[:-1]
        end_dates_constant = df_updated.params_fitting_starting_dates.iloc[0].strip().split(";")[:-1]
        if len(end_dates_constant) < len(start_dates_constant):
            end_dates_constant = end_dates_constant + [yesterday_date]
            # Basically means that this policy is still constant for the time being
            assert len(end_dates_constant) == len(start_dates_constant)

        start_end_dates_constant = [
            (start_date, end_date) for start_date, end_date
            in zip(start_dates_constant[-n_params_constant:], end_dates_constant[-n_params_constant:])
        ]
        return params_constant, start_end_dates_constant


def get_list_and_bounds_params(
        df_updated: pd.DataFrame, parameter_list_line: list, param_MATHEMATICA: bool
):
    if param_MATHEMATICA:
        parameter_list_fitted = parameter_list_line[4:]
        parameter_list_fitted[3] = np.log(2) / parameter_list_fitted[3]
    else:
        parameter_list_fitted = parameter_list_line[5:]

    # We might be storing NaN values for the policy parameters so we want to make sure to only have non-NaN values
    parameter_list_fitted = [param for param in parameter_list_fitted if param is not np.nan]
    if len(parameter_list_fitted) > n_params_without_policy_params:  # Saved 7 basic parameters + some for policy change
        params_base_fitted_saved = parameter_list_fitted[:n_params_without_policy_params]
        param_base_list_lower = [x - 0.1 * abs(x) for x in params_base_fitted_saved]
        param_base_list_upper = [x + 0.1 * abs(x) for x in params_base_fitted_saved]
        params_policy_fitted_saved = parameter_list_fitted[n_params_without_policy_params:]
        params_policies_fitted, start_dates_fitting_policies = get_params_fitted_policies(df_updated)
        assert params_policies_fitted is not None, "Supposed to have at least one fitted parameter for policies"
        if len(params_policy_fitted_saved) == len(params_policies_fitted):  # No new fitted parameter
            # Allowing a 30% drift for the extra policy parameters fitted and saved
            # (as they're != from initial value a priori because of the 30% drift allowed at each fit)
            params_policies_lower_bounds = [x - 0.3 * abs(x) for x in params_policy_fitted_saved]
            params_policies_upper_bounds = [x + 0.3 * abs(x) for x in params_policy_fitted_saved]
            bounds_params_fitted = tuple(
                [(lower, upper) for lower, upper in zip(param_base_list_lower, param_base_list_upper)] +
                [
                    (lower_pol, upper_pol) for lower_pol, upper_pol
                    in zip(params_policies_lower_bounds, params_policies_upper_bounds)
                ]
            )
            parameter_list_fitted = params_base_fitted_saved + params_policies_fitted
            assert len(parameter_list_fitted) == len(bounds_params_fitted), \
                f"Should have n_params=n_bounds, got {len(parameter_list_fitted)}!={len(bounds_params_fitted)}"
        elif len(params_policy_fitted_saved) < len(params_policies_fitted):  # 1 new fitted parameter
            # The initial values are thus all the ones saved in the parameters dataset (with possible 30% drift)
            # + extra parameters that are new and have initial values from latest CART
            new_params_policy_fitted = params_policy_fitted_saved + params_policies_fitted[len(params_policy_fitted_saved):]
            params_policies_lower_bounds = [x - 0.3 * abs(x) for x in new_params_policy_fitted]
            params_policies_upper_bounds = [x + 0.3 * abs(x) for x in new_params_policy_fitted]
            bounds_params_fitted = tuple(
                [(lower, upper) for lower, upper in zip(param_base_list_lower, param_base_list_upper)] +
                [
                    (lower_pol, upper_pol) for lower_pol, upper_pol
                    in zip(params_policies_lower_bounds, params_policies_upper_bounds)
                ]
            )
            parameter_list_fitted = params_base_fitted_saved + new_params_policy_fitted
            assert len(parameter_list_fitted) == len(bounds_params_fitted), \
                f"Should have n_params=n_bounds, got {len(parameter_list_fitted)}!={len(bounds_params_fitted)}"

        else:
            raise ValueError(
                "Number of parameters for policy fitted and saved in Params file should be <= than in tracking file " +
                f"but got {len(params_policy_fitted_saved)} > {len(params_policies_fitted)}"
            )

    elif len(parameter_list_fitted) == n_params_without_policy_params:  # Only saved the 7 basic parameters so far
        # Allowing a 10% drift for the 7 initial parameters
        param_list_lower = [x - 0.1 * abs(x) for x in parameter_list_fitted]
        param_list_upper = [x + 0.1 * abs(x) for x in parameter_list_fitted]
        params_policies_fitted, start_dates_fitting_policies = get_params_fitted_policies(df_updated)
        if (params_policies_fitted, start_dates_fitting_policies) != ([], []):
            # Allowing a 30% drift for the extra policy parameters fitted
            params_policies_lower_bounds = [x - 0.3 * abs(x) for x in params_policies_fitted]
            params_policies_upper_bounds = [x + 0.3 * abs(x) for x in params_policies_fitted]
            bounds_params_fitted = tuple(
                [(lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)] +
                [
                    (lower_pol, upper_pol) for lower_pol, upper_pol
                    in zip(params_policies_lower_bounds, params_policies_upper_bounds)
                ]
            )
            parameter_list_fitted.extend(params_policies_fitted)
        else:
            bounds_params_fitted = tuple(
                [(lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)]
            )
    else:
        raise ValueError(f"Expected at least {n_params_without_policy_params}" +
                         f"parameters saved, got {len(parameter_list_fitted)}")

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
        "policy_shift_initial_param_list": [" "], "n_policies_enacted": [1], "n_policy_changes": [0],
        "last_policy": [current_policy], "start_date_last_policy": [yesterday_date],
        "end_date_last_policy": [" "], "n_days_enacted_last_policy": [1], "current_policy": [current_policy],
        "start_date_current_policy": [yesterday_date], "end_date_current_policy": [" "],
        "n_days_enacted_current_policy": [1], "n_params_fitted": [0], "n_params_constant_tree": [0],
        "init_values_params": [" "],
    }
    df_policy_change_tracking_area = pd.DataFrame(dict_policy_change_tracking_area)
    return df_policy_change_tracking_area

