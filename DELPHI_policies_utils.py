# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union
from copy import deepcopy
from itertools import compress
import json
from DELPHI_params import (TIME_DICT, MAPPING_STATE_CODE_TO_STATE_NAME, default_policy,
                           default_policy_enaction_time, future_policies)


def update_tracking_without_policy_change(
        df_updated: pd.DataFrame, yesterday: str,
) -> pd.DataFrame:
    df_updated.loc[0, "n_days_enacted_current_policy"] = (
        (pd.to_datetime(yesterday) - pd.to_datetime(df_updated.loc[0, "start_date_current_policy"])).days
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
    df_updated = update_n_params_fitted_with_policy_change(
        df_updated=df_updated,
        yesterday=yesterday,
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


def update_n_params_fitted_with_policy_change(
        df_updated: pd.DataFrame, yesterday: str
) -> pd.DataFrame:
    """
    When we update the policy (i.e. when there's been a change in policy) the former current_policy becomes last_policy
    and the new current_policy is for sure not fitted as it is 1 day old. If it is ready to be fitted, we need to update
    the flags for param fitting 'last/current_policy_fitted'
    """
    n_days_data_before_fitting = 10
    flag_new_param_fitted = len(df_updated[
        (df_updated.n_days_enacted_last_policy >= n_days_data_before_fitting)
        & (df_updated.n_policy_changes >= 1)
        & (df_updated.last_policy_fitted == False)
    ]) > 0
    if flag_new_param_fitted:
        df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
        df_updated.loc[0, "last_policy_fitted"] = True
        df_updated.loc[0, "params_fitting_starting_dates"] = (
                df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
        )

    return df_updated


def update_n_params_fitted_without_policy_change(
        df_updated: pd.DataFrame, yesterday: str,
) -> pd.DataFrame:
    """
    When we do not update the policy (i.e. there hasn't been any change in policy) then both current_policy
    and last_policy are susceptible to change so we need to check if we have enough data to fit the parameter
    instead of using a constant value, and if so, update the flags for param fitting 'last/current_policy_fitted'
    """
    n_days_data_before_fitting = 10
    flag_new_param_fitted_last_policy = len(df_updated[
        (df_updated.n_days_enacted_last_policy >= n_days_data_before_fitting)
        & (df_updated.n_policy_changes >= 1)
        & (df_updated.last_policy_fitted == False)
    ]) > 0
    flag_new_param_fitted_current_policy = len(df_updated[
        (df_updated.n_days_enacted_current_policy >= n_days_data_before_fitting)
        & (df_updated.n_policy_changes >= 1)
        & (df_updated.current_policy_fitted == False)
    ]) > 0
    if flag_new_param_fitted_last_policy:
        df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
        df_updated.loc[0, "last_policy_fitted"] = True
        df_updated.loc[0, "params_fitting_starting_dates"] = (
                df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
        )

    if flag_new_param_fitted_current_policy:
        df_updated.loc[0, "n_params_fitted"] = df_updated.loc[0, "n_params_fitted"] + 1
        df_updated.loc[0, "current_policy_fitted"] = True
        df_updated.loc[0, "params_fitting_starting_dates"] = (
                df_updated.loc[0, "params_fitting_starting_dates"] + str(pd.to_datetime(yesterday).date()) + ";"
        )

    return df_updated

