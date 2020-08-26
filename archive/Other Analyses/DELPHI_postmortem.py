# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:47:43 2020

@author: omars
"""
#%% Libs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from DELPHI_utils import (
    read_measures_oxford_data, get_normalized_policy_shifts_and_current_policy_all_countries,
    get_normalized_policy_shifts_and_current_policy_us_only, read_policy_data_us_only
)
from DELPHI_params import (future_policies)
import yaml

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from statsmodels.api import OLS
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
#%% Env
with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
USER_RUNNING = "omar"
training_start_date = "2020-06-14"  # First date that will be considered for Parameters_Global_<date>.csv
training_end_date = "2020-06-15"  # First date that excluded for Parameters_Global_<date>.csv
max_days_before = (pd.to_datetime(training_end_date) - pd.to_datetime(training_start_date)).days
time_start = datetime.now()
prediction_date = "2020-06-01"
prediction_date_file = "".join(prediction_date.split("-"))
testing_date = "2020-06-15"
testing_date_file = "".join(testing_date.split("-"))
n_days_testing_data = (pd.to_datetime(testing_date) - pd.to_datetime(prediction_date)).days

#%% Backtesting output
df_min_all = pd.read_csv(
    f"./backtesting/{testing_date_file}_backtest" +
    f"_policy_predictions_pred_from_{prediction_date_file}.csv"
)

#%% Gamma Dicts
yesterday = "".join(str(pd.to_datetime(training_end_date).date() - timedelta(days=1)).split("-"))
print(yesterday)
print(f"Runtime for {yesterday}: {datetime.now() - time_start}")
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
policy_data_countries = read_measures_oxford_data(yesterday=yesterday)
policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=PATH_TO_DATA_SANDBOX)
popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")

pastparameters = pd.read_csv(
PATH_TO_FOLDER_DANGER_MAP + f"predicted/parameters_global_CR_all/Parameters_Global_CR_{yesterday}.csv"
)
param_MATHEMATICA = False
dict_normalized_policy_gamma_countries, dict_current_policy_countries = (
get_normalized_policy_shifts_and_current_policy_all_countries(
    policy_data_countries=policy_data_countries[policy_data_countries.date <= yesterday],
    pastparameters=pastparameters,
)
)
dict_normalized_policy_gamma_countries[future_policies[3]] = dict_normalized_policy_gamma_countries[future_policies[5]]

dict_normalized_policy_gamma_us_only, dict_current_policy_us_only = (
get_normalized_policy_shifts_and_current_policy_us_only(
    policy_data_us_only=policy_data_us_only[policy_data_us_only.date <= yesterday],
    pastparameters=pastparameters,
)
)
dict_current_policy_international = dict_current_policy_countries.copy()
dict_current_policy_international.update(dict_current_policy_us_only)

#%% Mobility
mobility = pd.read_csv('C:/Users/omars/Downloads/Global_Mobility_Report.csv')
mobility = mobility.query('country_region_code == "US"')[[not(a) and b for a, b in zip(mobility.query('country_region_code == "US"').sub_region_1.isnull(), mobility.query('country_region_code == "US"').sub_region_2.isnull())]]

mobility['date'] = mobility['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
mob_columns = list(mobility.columns[7:])

mobility_agg = mobility[[a and b for a, b in zip(mobility['date'] >= prediction_date, mobility['date'] <= testing_date)]].groupby('sub_region_1')[mob_columns].mean().dropna(axis=1).reset_index()
#%% Construct Agg Dset for US

target = 'cases'
df = df_min_all.query('country == "US"').query('province != "None"')
df = df.merge(mobility_agg, how='left', left_on='province', right_on='sub_region_1').drop('sub_region_1', axis=1)
curr_policy = np.array(list(df.columns))[[a.find('current_policy') == 0 for a in df.columns]][0]
best_policy = 'best_policy_' + target
df['Shift'] = df[curr_policy].apply(lambda x: dict_normalized_policy_gamma_us_only[x]) - df[best_policy].apply(lambda x: dict_normalized_policy_gamma_us_only[x])

#%% Predict Shift based on Mobility

def visualize_tree(sktree):
    dot_data = tree.export_graphviz(sktree, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=False,
                                    feature_names=(mob_columns))
    return graphviz.Source(dot_data)

X = df.loc[:, mob_columns]
y = df.loc[:, 'Shift']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

OLS(y_train,X_train).fit().summary()

cart = DecisionTreeRegressor(max_depth=2)
cart.fit(X_train, y_train)
print(r2_score(y_train, cart.predict(X_train)))
print(r2_score(y_test, cart.predict(X_test)))

lasso = LassoCV()
lasso.fit(X_train, y_train)
print(r2_score(y_train, lasso.predict(X_train)))
print(r2_score(y_test, lasso.predict(X_test)))

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
print(r2_score(y_train, xgb.predict(X_train)))
print(r2_score(y_test, xgb.predict(X_test)))


#%%

dumdic = {'No_Measure': 0,
          'Mass_Gatherings_Authorized_But_Others_Restricted': 1,
          'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 2,
          'Restrict_Mass_Gatherings_and_Schools': 3,
          'Restrict_Mass_Gatherings_and_Schools_and_Others': 4,
          'Lockdown': 5}

df['Current Policy'] = df[curr_policy].apply(lambda x: dumdic[x])
df['Inferred Policy'] = df[best_policy].apply(lambda x: dumdic[x])

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

target1, target2 = 'Current Policy', 'Inferred Policy'
X = df.loc[:, mob_columns]
y1 = df.loc[:, target1]
y2 = df.loc[:, target2]
X_train, X_test, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42, shuffle=True)
X_train, X_test, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42, shuffle=True)

cart1 = DecisionTreeClassifier()
cart2 = DecisionTreeClassifier()
log1 = LogisticRegressionCV()
log2 = LogisticRegressionCV()
xgb1 = XGBClassifier()
xgb2 = XGBClassifier()

cart1.fit(X_train, y_train1)
cart2.fit(X_train, y_train2)
# log1.fit(X_train, y_train1)
# log2.fit(X_train, y_train2)
xgb1.fit(X_train, y_train1)
xgb2.fit(X_train, y_train2)


print('CART Ouf-of-Sample Accuracy for True Policy: ', accuracy_score(y_test1, cart1.predict(X_test)))
# print('LogReg Ouf-of-Sample Accuracy for True Policy: ',accuracy_score(y_test1, log1.predict(X_test)))
print('XGB Ouf-of-Sample Accuracy for True Policy: ',accuracy_score(y_test1, xgb1.predict(X_test)))
print('CART Ouf-of-Sample Accuracy for Inferred Policy: ',accuracy_score(y_test2, cart2.predict(X_test)))
# print('LogReg Ouf-of-Sample Accuracy for Inferred Policy: ',accuracy_score(y_test2, log2.predict(X_test)))
print('XGB Ouf-of-Sample Accuracy for Inferred Policy: ',accuracy_score(y_test2, xgb2.predict(X_test)))


#%%
import seaborn as sns
%matplotlib inline


# calculate the correlation matrix
corr = df.loc[:, mob_columns + ['Shift']].corr()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=[x[:-29] if x != 'Shift' else x for x in corr.columns],
        yticklabels=[x[:-29] if x != 'Shift' else x for x in corr.columns],
        annot=True)


