# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
from itertools import compress
from DELPHI_params import (future_policies, provinces_Brazil,
                           provinces_Peru, provinces_South_Africa)
from DELPHI_utils import (check_us_policy_data_consistency, create_features_from_ihme_dates,
                          create_final_policy_features_us)


def read_policy_data_us_only_jj_version(filepath_data_sandbox: str):
    policies = [
        "travel_limit", "stay_home", "educational_fac", "any_gathering_restrict",
        "any_business", "all_non-ess_business"
    ]
    list_US_states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas',
        'California', 'Colorado', 'Connecticut', 'Delaware',
        'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming'
    ]
    df = pd.read_csv(filepath_data_sandbox + "10052020_raw_policy_data_US_only.csv")
    df = df[df.location_name.isin(list_US_states)][[
        "location_name", "travel_limit_start_date", "travel_limit_end_date",
        "stay_home_start_date", "stay_home_end_date", "educational_fac_start_date",
        "educational_fac_end_date", "any_gathering_restrict_start_date",
        "any_gathering_restrict_end_date", "any_business_start_date", "any_business_end_date",
        "all_non-ess_business_start_date", "all_non-ess_business_end_date"
    ]]
    dict_state_to_policy_dates = {}
    for location in df.location_name.unique():
        df_temp = df[df.location_name == location].reset_index(drop=True)
        dict_state_to_policy_dates[location] = {
            policy: [df_temp.loc[0, f"{policy}_start_date"], df_temp.loc[0, f"{policy}_end_date"]]
            for policy in policies
        }
    check_us_policy_data_consistency(policies=policies, df_policy_raw_us=df)
    df_policies_US = create_features_from_ihme_dates(
        df_policy_raw_us=df,
        dict_state_to_policy_dates=dict_state_to_policy_dates,
        policies=policies,
    )
    df_policies_US_final = create_final_policy_features_us(df_policies_US=df_policies_US)
    # Adding cities in there
    df_policies_US_final_Chicago = df_policies_US_final[
        df_policies_US_final.province == "Illinois"
    ]
    df_policies_US_final_Chicago.province.replace({"Illinois": "Chicago Metropolitan"}, inplace=True)
    df_policies_US_final_Detroit = df_policies_US_final[
        df_policies_US_final.province == "Michigan"
        ]
    df_policies_US_final_Detroit.province.replace({"Michigan": "Detroit Metropolitan"}, inplace=True)
    df_policies_US_final_NYC = df_policies_US_final[
        df_policies_US_final.province == "New York"
        ]
    df_policies_US_final_NYC.province.replace({"New York": "NY-NJ Metropolitan"}, inplace=True)

    df_policies_US_final_NHM = df_policies_US_final[
        df_policies_US_final.province == "Connecticut"
        ]
    df_policies_US_final_NHM.province.replace({"Connecticut": "New-Haven Metropolitan"}, inplace=True)

    df_policies_US_final_MMinM = df_policies_US_final[
        df_policies_US_final.province == "Minnesota"
        ]
    df_policies_US_final_MMinM.province.replace({"Minnesota": "Minneapolis Metropolitan"}, inplace=True)
    df_policies_US_final_DCArl = df_policies_US_final[
        df_policies_US_final.province == "Virginia"
        ]
    df_policies_US_final_DCArl.province.replace({"Virginia": "Washington-Arlington-Alexandria Metropolitan"}, inplace=True)

    df_policies_US_final_Tucson = df_policies_US_final[
        df_policies_US_final.province == "Arizona"
        ]
    df_policies_US_final_Tucson.province.replace({"Arizona": "Tucson Metropolitan"}, inplace=True)

    df_policies_US_final_Phoenix = df_policies_US_final[
        df_policies_US_final.province == "Arizona"
        ]
    df_policies_US_final_Phoenix.province.replace({"Arizona": "Phoenix Metropolitan"}, inplace=True)

    df_policies_US_final_LAOC = df_policies_US_final[
        df_policies_US_final.province == "California"
        ]
    df_policies_US_final_LAOC.province.replace({"California": "LA-LB-OC Metropolitan"}, inplace=True)

    df_policies_US_final_Houston = df_policies_US_final[
        df_policies_US_final.province == "Texas"
        ]
    df_policies_US_final_Houston.province.replace({"Texas": "Houston Metropolitan"}, inplace=True)

    df_policies_US_final_Dallas = df_policies_US_final[
        df_policies_US_final.province == "Texas"
        ]
    df_policies_US_final_Dallas.province.replace({"Texas": "DALLAS-FW-ARLINGTON Metropolitan"}, inplace=True)

    df_policies_US_final_Baltimore = df_policies_US_final[
        df_policies_US_final.province == "Maryland"
        ]
    df_policies_US_final_Baltimore.province.replace({"Maryland": "Baltimore-Columbia-Towson Metropolitan"}, inplace=True)

# Concatenating
    df_policies_US_final = pd.concat([
        df_policies_US_final, df_policies_US_final_Chicago, df_policies_US_final_Detroit, df_policies_US_final_NYC,
        df_policies_US_final_NHM, df_policies_US_final_MMinM, df_policies_US_final_Baltimore,df_policies_US_final_Dallas,
        df_policies_US_final_Houston,df_policies_US_final_LAOC,df_policies_US_final_Phoenix,df_policies_US_final_Tucson,
        df_policies_US_final_DCArl
    ]).reset_index(drop=True)

    return df_policies_US_final


def get_normalized_policy_shifts_and_current_policy_all_countries_jj_version(
        policy_data_countries: pd.DataFrame,
        pastparameters: pd.DataFrame,
):
    dict_current_policy = {}
    policy_list = future_policies
    policy_data_countries['country_cl'] = policy_data_countries['country'].apply(
        lambda x: x.replace(',', '').strip().lower()
    )
    policy_data_countries['province_cl'] = policy_data_countries['province'].apply(
        lambda x: x.replace(',', '').strip().lower()
    )
    pastparameters_copy = deepcopy(pastparameters)
    pastparameters_copy['Country'] = pastparameters_copy['Country'].apply(
        lambda x: str(x).replace(',', '').strip().lower()
    )
    pastparameters_copy['Province'] = pastparameters_copy['Province'].apply(
        lambda x: str(x).replace(',', '').strip().lower()
    )
    # params_countries = pastparameters_copy['Country']
    # params_countries = set(params_countries)
    # params_provinces = pastparameters_copy['Province']
    # params_provinces = set(params_provinces)
    pastparameters_copy["tuple_country_province"] = list(
        zip(pastparameters_copy.Country, pastparameters_copy.Province)
    )
    params_countries_provinces = set(pastparameters_copy["tuple_country_province"])
    policy_data_countries["tuple_country_province_cl"] = list(
        zip(policy_data_countries.country_cl, policy_data_countries.province_cl)
    )
    policy_data_countries_provinces_bis = policy_data_countries.query(
        "tuple_country_province_cl in @params_countries_provinces"
    )
    countries_upper_set = set(zip(policy_data_countries['country'], policy_data_countries['province']))

    for country_province in countries_upper_set:
        country = country_province[0]
        province = country_province[1]
        dict_current_policy[(country, province)] = list(compress(
            policy_list,
            (policy_data_countries.query('country == @country and province == @province')[
                 policy_data_countries.query('country == @country and province == @province')["date"]
                 == policy_data_countries.query('country == @country and province == @province').date.max()
                 ][policy_list] == 1).values.flatten().tolist()
        ))[0]

    countries_provinces_set = set(policy_data_countries['tuple_country_province_cl'])
    # policy_data_countries_provinces_bis.drop("tuple_country_province_cl", axis=1, inplace=True)
    params_dic = {}
    countries_provinces_set = countries_provinces_set.intersection(params_countries_provinces)
    for country_province in countries_provinces_set:
        country = country_province[0]
        province = country_province[1]
        params_dic[(country, province)] = pastparameters_copy.query('Country == @country and Province == @province')[
            ['Data Start Date', 'Median Day of Action', 'Rate of Action']
        ].iloc[0]

    policy_data_countries_provinces_bis['Gamma'] = [
        gamma_t_jj_version(day, country, province, params_dic) for day, country, province in
        zip(policy_data_countries_provinces_bis['date'], policy_data_countries_provinces_bis['country_cl'],
            policy_data_countries_provinces_bis["province_cl"])
    ]
    n_measures = policy_data_countries_provinces_bis.iloc[:, 3:-4].shape[1]
    dict_normalized_policy_gamma = {
        policy_data_countries_provinces_bis.columns[3 + i]: policy_data_countries_provinces_bis[
                                                      policy_data_countries_provinces_bis.iloc[:, 3 + i] == 1
                                                      ].iloc[:, -1].mean()
        for i in range(n_measures)
    }
    normalize_val = dict_normalized_policy_gamma[policy_list[0]]
    for policy in dict_normalized_policy_gamma.keys():
        dict_normalized_policy_gamma[policy] = dict_normalized_policy_gamma[policy] / normalize_val
    return dict_normalized_policy_gamma, dict_current_policy


def read_measures_oxford_data_jj_version():
    measures = pd.read_csv('https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv')
    filtr = ['CountryName', 'CountryCode', 'Date']
    target = ['ConfirmedCases', 'ConfirmedDeaths']
    msr = [
        'C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings',
        'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
        'C8_International travel controls', 'H1_Public information campaigns'
    ]
    measures = measures.loc[:, filtr + msr + target]
    measures['Date'] = measures['Date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    for col in target:
        # measures[col] = measures[col].fillna(0)
        measures[col] = measures.groupby('CountryName')[col].ffill()

    # measures = measures.loc[:, measures.isnull().mean() < 0.1]
    msr = set(measures.columns).intersection(set(msr))

    # measures = measures.fillna(0)
    measures = measures.dropna()
    for col in msr:
        measures[col] = measures[col].apply(lambda x: int(x > 0))
    measures = measures[
        ['CountryName', 'Date'] + list(sorted(msr))
        ]
    measures["CountryName"] = measures.CountryName.replace({
        "United States": "US", "South Korea": "Korea, South", "Democratic Republic of Congo": "Congo (Kinshasa)",
        "Czech Republic": "Czechia", "Slovak Republic": "Slovakia",
    })

    measures = measures.fillna(0)
    msr = future_policies

    measures['Restrict_Mass_Gatherings'] = [
        int(a or b or c) for a, b, c in zip(
            measures['C3_Cancel public events'],
            measures['C4_Restrictions on gatherings'],
            measures['C5_Close public transport']
        )
    ]
    measures['Others'] = [
        int(a or b or c) for a, b, c in zip(
            measures['C2_Workplace closing'],
            measures['C7_Restrictions on internal movement'],
            measures['C8_International travel controls']
        )
    ]

    del measures['C2_Workplace closing']
    del measures['C3_Cancel public events']
    del measures['C4_Restrictions on gatherings']
    del measures['C5_Close public transport']
    del measures['C7_Restrictions on internal movement']
    del measures['C8_International travel controls']

    output = deepcopy(measures)
    output[msr[0]] = (measures.iloc[:, 2:].sum(axis=1) == 0).apply(lambda x: int(x))
    output[msr[1]] = [
        int(a and b) for a, b in zip(
            measures.iloc[:, 2:].sum(axis=1) == 1,
            measures['Restrict_Mass_Gatherings'] == 1
        )
    ]
    output[msr[2]] = [
        int(a and b and c) for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) > 0,
            measures['Restrict_Mass_Gatherings'] == 0,
            measures['C6_Stay at home requirements'] == 0,
        )
    ]
    output[msr[3]] = [
        int(a and b and c)
        for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) == 2,
            measures['C1_School closing'] == 1,
            measures['Restrict_Mass_Gatherings'] == 1,
        )
    ]
    output[msr[4]] = [
        int(a and b and c and d) for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 1,
            measures['C1_School closing'] == 0,
            measures['Restrict_Mass_Gatherings'] == 1,
            measures['C6_Stay at home requirements'] == 0,
        )
    ]
    output[msr[5]] = [
        int(a and b and c and d) for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 2,
            measures['C1_School closing'] == 1,
            measures['Restrict_Mass_Gatherings'] == 1,
            measures['C6_Stay at home requirements'] == 0,
        )
    ]
    output[msr[6]] = (measures['C6_Stay at home requirements'] == 1).apply(lambda x: int(x))
    output.rename(columns={'CountryName': 'country', 'Date': 'date'}, inplace=True)
    output['province'] = "None"
    output = output.loc[:, ['country', 'province', 'date'] + msr]

    # Adding provinces for South Africa, Brazil, Peru
    # Brazil
    outputs_provinces_Brazil = []
    for province in provinces_Brazil:
        output_Brazil_temp = output[output.country == "Brazil"]
        output_Brazil_temp.loc[:, "province"] = province
        outputs_provinces_Brazil.append(output_Brazil_temp)
    outputs_provinces_Brazil = pd.concat(outputs_provinces_Brazil).reset_index(drop=True)
    # South Africa
    outputs_provinces_South_Africa = []
    for province in provinces_South_Africa:
        output_South_Africa_temp = output[output.country == "South Africa"]
        output_South_Africa_temp.loc[:, "province"] = province
        outputs_provinces_South_Africa.append(output_South_Africa_temp)
    outputs_provinces_South_Africa = pd.concat(outputs_provinces_South_Africa).reset_index(drop=True)
    # Peru
    outputs_provinces_Peru = []
    for province in provinces_Peru:
        output_Peru_temp = output[output.country == "Peru"]
        output_Peru_temp.loc[:, "province"] = province
        outputs_provinces_Peru.append(output_Peru_temp)
    outputs_provinces_Peru = pd.concat(outputs_provinces_Peru).reset_index(drop=True)

    output = pd.concat([
        output, outputs_provinces_Brazil, outputs_provinces_South_Africa, outputs_provinces_Peru
    ]).sort_values(["country", "province", "date"]).reset_index(drop=True)
    return output


def gamma_t_jj_version(day, country, province, params_dic):
    dsd, median_day_of_action, rate_of_action = params_dic[(country, province)]
    t = (day - pd.to_datetime(dsd)).days
    gamma = (2 / np.pi) * np.arctan(-(t - median_day_of_action) / 20 * rate_of_action) + 1
    return gamma
