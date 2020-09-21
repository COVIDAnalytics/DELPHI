# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
from itertools import compress
from DELPHI_params import (future_policies,
                           city_policies, list_US_states,MAPPING_STATE_CODE_TO_STATE_NAME)
from DELPHI_utils import (check_us_policy_data_consistency, create_features_from_ihme_dates,
                          create_final_policy_features_us)


def read_policy_data_us_only_jj_version(filepath_data_sandbox: str):
    policies = [
        "travel_limit", "stay_home", "educational_fac", "any_gathering_restrict",
        "any_business", "all_non-ess_business"
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
    df_policies_US_final = create_additional_cities_for_policies_data(df_policies_US_final=df_policies_US_final,
                                                                      filepath_data_sandbox= filepath_data_sandbox)
    return df_policies_US_final


def create_additional_cities_for_policies_data(df_policies_US_final: pd.DataFrame, filepath_data_sandbox: str) -> pd.DataFrame:
    df_policies_US_final_Chicago = df_policies_US_final[
        df_policies_US_final.province == "Illinois"
    ]
    df_policies_US_final_Chicago.province.replace({"Illinois": "Chicago_Naperville_Elgin_IL_IN_WI"}, inplace=True)
    #df_policies_US_final_Detroit = df_policies_US_final[
    #    df_policies_US_final.province == "Michigan"
    #]
    #df_policies_US_final_Detroit.province.replace({"Michigan": "Detroit Metropolitan"}, inplace=True)
    df_policies_US_final_NYC = df_policies_US_final[
        df_policies_US_final.province == "New York"
    ]
    df_policies_US_final_NYC.province.replace({"New York": "New_York_Newark_Jersey_City_New_Brunswick_Lakewood_NY_NJ"}, inplace=True)
    df_policies_US_final_Birmingham_Hoover = df_policies_US_final[
        df_policies_US_final.province == "Alabama"
    ]
    df_policies_US_final_Birmingham_Hoover.province.replace({"Alabama": "Birmingham_Hoover_AL"}, inplace=True)

    df_policies_US_final_Detroit_Warren_Dearborn = df_policies_US_final[
        df_policies_US_final.province == "Michigan"
        ]
    df_policies_US_final_Detroit_Warren_Dearborn.province.replace({"Michigan": "Detroit_Warren_Dearborn_Ann_Arbor_MI"}, inplace=True)

    df_policies_US_final_New_Haven_Milford = df_policies_US_final[
        df_policies_US_final.province == "Connecticut"
        ]
    df_policies_US_final_New_Haven_Milford.province.replace({"Connecticut": "New_Haven_Milford_CT"}, inplace=True)

    df_policies_US_final_Los_Angeles_Long_Beach_Orange_County = df_policies_US_final[
        df_policies_US_final.province == "California"
        ]
    df_policies_US_final_Los_Angeles_Long_Beach_Orange_County.province.replace(
        {"California": "Los_Angeles_Long_Beach_Orange_County_CA"}, inplace=True
    )

    df_policies_US_final_Nashville_Davidson_Murfreesboro_Franklin = df_policies_US_final[
        df_policies_US_final.province == "Tennessee"
        ]
    df_policies_US_final_Nashville_Davidson_Murfreesboro_Franklin.province.replace(
        {"Tennessee": "Nashville_Davidson_Murfreesboro_Franklin_TN"}, inplace=True
    )

    df_policies_US_final_Washington_Arlington_Alexandria = df_policies_US_final[
        df_policies_US_final.province == "Maryland"
    ]
    df_policies_US_final_Washington_Arlington_Alexandria.province.replace(
        {"Maryland": "Washington_Arlington_Alexandria_DC_VA_MD_WV"}, inplace=True
    )

    df_policies_US_final_Columbus = df_policies_US_final[
        df_policies_US_final.province == "Ohio"
    ]
    df_policies_US_final_Columbus.province.replace(
        {"Ohio": "Columbus_OH"}, inplace=True
    )

    df_policies_US_final_Cincinnati = df_policies_US_final[
        df_policies_US_final.province == "Ohio"
    ]
    df_policies_US_final_Cincinnati.province.replace(
        {"Ohio": "Cincinnati_OH_KY_IN"}, inplace=True
    )

    # df_policies_US_final_Las_Vegas_Henderson_Paradise = df_policies_US_final[
    #     df_policies_US_final.province == "California"
    # ]
    # df_policies_US_final_Las_Vegas_Henderson_Paradise.province.replace(
    #     {"California": "Las Vegas_Henderson_Paradise_NV"}, inplace=True
    # )

    df_policies_US_final_Cleveland_Elyria = df_policies_US_final[
        df_policies_US_final.province == "Ohio"
        ]
    df_policies_US_final_Cleveland_Elyria.province.replace(
        {"Ohio": "Cleveland_Elyria_OH"}, inplace=True
    )

    df_policies_US_final_San_Jose_Sunnyvale_Santa_Clara = df_policies_US_final[
        df_policies_US_final.province == "California"
        ]
    df_policies_US_final_San_Jose_Sunnyvale_Santa_Clara.province.replace(
        {"California": "San_Jose_Sunnyvale_Santa_Clara_San_Francisco_Oakland_CA"}, inplace=True
    )

    df_policies_US_final_Baltimore_Columbia_Towson = df_policies_US_final[
        df_policies_US_final.province == "Maryland"
        ]
    df_policies_US_final_Baltimore_Columbia_Towson.province.replace(
        {"Maryland": "Baltimore_Columbia_Towson_MD"}, inplace=True
    )

    df_policies_US_final_Tampa = df_policies_US_final[
        df_policies_US_final.province == "Florida"
        ]
    df_policies_US_final_Tampa.province.replace(
        {"Florida": "Tampa_St_Petersburg_Clearwater_FL"}, inplace=True
    )

    df_policies_US_final_Philadelphia_Camden_Wilmington = df_policies_US_final[
        df_policies_US_final.province == "Delaware"
        ]
    df_policies_US_final_Philadelphia_Camden_Wilmington.province.replace(
        {"Delaware": "Philadelphia_Camden_Wilmington_PA_NJ_DE_MD"}, inplace=True
    )

    df_policies_US_final_Pittsburgh = df_policies_US_final[
        df_policies_US_final.province == "Pennsylvania"
        ]
    df_policies_US_final_Pittsburgh.province.replace(
        {"Pennsylvania": "Pittsburgh_PA"}, inplace=True
    )

    df_policies_US_final_San_Diego_Chula_Vista_Carlsbad = df_policies_US_final[
        df_policies_US_final.province == "California"
        ]
    df_policies_US_final_San_Diego_Chula_Vista_Carlsbad.province.replace(
        {"California": "San_Diego_Chula_Vista_Carlsbad_CA"}, inplace=True
    )

    df_policies_US_final_Sioux_Falls = df_policies_US_final[
        df_policies_US_final.province == "South Dakota"
        ]
    df_policies_US_final_Sioux_Falls.province.replace(
        {"South Dakota": "Sioux_Falls_SD"}, inplace=True
    )

    df_policies_US_final_Minneapolis = df_policies_US_final[
        df_policies_US_final.province == "Minnesota"
        ]
    df_policies_US_final_Minneapolis.province.replace(
        {"Minnesota": "Minneapolis_MN_WI"}, inplace=True
    )

    df_policies_US_final_New_Orleans_Metairie = df_policies_US_final[
        df_policies_US_final.province == "Louisiana"
        ]
    df_policies_US_final_New_Orleans_Metairie.province.replace(
        {"Louisiana": "New_Orleans_Metairie_LA"}, inplace=True
    )

    df_policies_US_final_St_Louis = df_policies_US_final[
        df_policies_US_final.province == "Missouri"
        ]
    df_policies_US_final_St_Louis.province.replace(
        {"Missouri": "St._Louis_MO_IL"}, inplace=True
    )

    df_policies_US_final_Mobile = df_policies_US_final[
        df_policies_US_final.province == "Alabama"
        ]
    df_policies_US_final_Mobile.province.replace(
        {"Alabama": "Mobile_AL"}, inplace=True
    )

    df_policies_US_final_Seattle_Tacoma_Bellevue = df_policies_US_final[
        df_policies_US_final.province == "Washington"
        ]
    df_policies_US_final_Seattle_Tacoma_Bellevue.province.replace(
        {"Washington": "Seattle_Tacoma_Bellevue_WA"}, inplace=True
    )

    df_policies_US_final_Atlanta_Sandy_Springs_Alpharetta = df_policies_US_final[
        df_policies_US_final.province == "Georgia"
        ]
    df_policies_US_final_Atlanta_Sandy_Springs_Alpharetta.province.replace(
        {"Georgia": "Atlanta_Sandy_Springs_Alpharetta_GA"}, inplace=True
    )

    df_policies_US_final_Omaha_Council_Bluffs = df_policies_US_final[
        df_policies_US_final.province == "Iowa"
        ]
    df_policies_US_final_Omaha_Council_Bluffs.province.replace(
        {"Iowa": "Omaha_Council_Bluffs_NE_IA"}, inplace=True
    )

    df_policies_US_final_Boston_Cambridge_Newton = df_policies_US_final[
        df_policies_US_final.province == "Massachusetts"
        ]
    df_policies_US_final_Boston_Cambridge_Newton.province.replace(
        {"Massachusetts": "Boston_Cambridge_Newton_MA_NH"}, inplace=True
    )

    df_policies_US_final_Rochester = df_policies_US_final[
        df_policies_US_final.province == "New York"
        ]
    df_policies_US_final_Rochester.province.replace(
        {"New York": "Rochester_NY"}, inplace=True
    )

    df_policies_US_final_Durham_Chapel_Hill = df_policies_US_final[
        df_policies_US_final.province == "North Carolina"
        ]
    df_policies_US_final_Durham_Chapel_Hill.province.replace(
        {"North Carolina": "Durham_Chapel_Hill_Raleigh_Cary_NC"}, inplace=True
    )

    df_policies_US_final_Orlando_Kissimmee_Sanford = df_policies_US_final[
        df_policies_US_final.province == "Florida"
        ]
    df_policies_US_final_Orlando_Kissimmee_Sanford.province.replace(
        {"Florida": "Orlando_Kissimmee_Sanford_FL"}, inplace=True
    )

    df_policies_US_final_Phoenix = df_policies_US_final[
        df_policies_US_final.province == "Arizona"
        ]
    df_policies_US_final_Phoenix.province.replace(
        {"Arizona": "Phoenix_Mesa_AZ"}, inplace=True
    )

    df_policies_US_final_Dallas_Fort_Worth_Arlington = df_policies_US_final[
        df_policies_US_final.province == "Texas"
        ]
    df_policies_US_final_Dallas_Fort_Worth_Arlington.province.replace(
        {"Texas": "Dallas_Fort_Worth_Arlington_TX"}, inplace=True
    )

    df_policies_US_final_Houston_The_Woodlands_Sugar_Land = df_policies_US_final[
        df_policies_US_final.province == "Texas"
        ]
    df_policies_US_final_Houston_The_Woodlands_Sugar_Land.province.replace(
        {"Texas": "Houston_The_Woodlands_Sugar_Land_TX"}, inplace=True
    )

    df_policies_US_final_Miami_Fort_Lauderdale_Pompano_Beach = df_policies_US_final[
        df_policies_US_final.province == "Florida"
        ]
    df_policies_US_final_Miami_Fort_Lauderdale_Pompano_Beach.province.replace(
        {"Florida": "Miami_Fort_Lauderdale_Pompano_Beach_FL"}, inplace=True
    )

    df_policies_US_final_Tucson = df_policies_US_final[
        df_policies_US_final.province == "Arizona"
        ]
    df_policies_US_final_Tucson.province.replace(
        {"Arizona": "Tucson_AZ"}, inplace=True
    )

    df_policies_US_final_Austin_Round_Rock_Georgetown = df_policies_US_final[
        df_policies_US_final.province == "Texas"
        ]
    df_policies_US_final_Austin_Round_Rock_Georgetown.province.replace(
        {"Texas": "Austin_Round_Rock_Georgetown_TX"}, inplace=True
    )
    df_policies_US_final_concat_v2 = df_policies_US_final
    count = 0
    for k in city_policies.keys():
        df_policies_US_final_Temp = df_policies_US_final[
            df_policies_US_final.province == city_policies[k]
            ]
        df_policies_US_final_Temp.province.replace({ city_policies[k]: k}, inplace=True)
        if count == 0:
            df_policies_US_final_concat_v2 = pd.concat(
                [df_policies_US_final] + [df_policies_US_final_Temp]
            )
        else:
            df_policies_US_final_concat_v2 = pd.concat(
                [df_policies_US_final_concat_v2] + [df_policies_US_final_Temp]
            )
        count += 1
    us_county_names = pd.read_csv(
        filepath_data_sandbox + f"processed/US_counties.csv"
    )
    ## USE a more smart way to get the date of parameters
    parameters_20200811 = pd.read_csv(
        filepath_data_sandbox + f"predicted/parameters/Parameters_J&J_20200811.csv"
    )

    for ind, row in us_county_names.iterrows():
        if row['Province'] in parameters_20200811.Province.values: # this is needed since if some cities/counties does not exist
            # in parameters. it will gives out error. not sure why...
            state = MAPPING_STATE_CODE_TO_STATE_NAME[row['Province'].split('_')[0]]
            df_policies_US_final_Temp = df_policies_US_final[
                df_policies_US_final.province == state
                ]
            df_policies_US_final_Temp.province.replace({ state: row['Province']}, inplace=True)
            df_policies_US_final_concat_v2 = pd.concat(
                [df_policies_US_final_concat_v2] + [df_policies_US_final_Temp]
            )

    df_policies_US_final_concat = pd.concat(
        [df_policies_US_final_concat_v2] + [
            df_policies_US_final_Chicago, df_policies_US_final_NYC,
            df_policies_US_final_Austin_Round_Rock_Georgetown, df_policies_US_final_Tucson,
            df_policies_US_final_Miami_Fort_Lauderdale_Pompano_Beach, df_policies_US_final_Phoenix,
            df_policies_US_final_Houston_The_Woodlands_Sugar_Land, df_policies_US_final_Dallas_Fort_Worth_Arlington,
            df_policies_US_final_Orlando_Kissimmee_Sanford, df_policies_US_final_Durham_Chapel_Hill,
            df_policies_US_final_Rochester, df_policies_US_final_Boston_Cambridge_Newton,
            df_policies_US_final_Omaha_Council_Bluffs, df_policies_US_final_Atlanta_Sandy_Springs_Alpharetta,
            df_policies_US_final_Seattle_Tacoma_Bellevue, df_policies_US_final_Mobile, df_policies_US_final_St_Louis,
            df_policies_US_final_New_Orleans_Metairie, df_policies_US_final_Minneapolis, df_policies_US_final_Tampa,
            df_policies_US_final_Sioux_Falls, df_policies_US_final_San_Diego_Chula_Vista_Carlsbad,
            df_policies_US_final_Pittsburgh, df_policies_US_final_Philadelphia_Camden_Wilmington,
            df_policies_US_final_Baltimore_Columbia_Towson, df_policies_US_final_San_Jose_Sunnyvale_Santa_Clara,
            df_policies_US_final_Cleveland_Elyria,
            df_policies_US_final_Cincinnati, df_policies_US_final_Columbus, df_policies_US_final_New_Haven_Milford,
            df_policies_US_final_Washington_Arlington_Alexandria, df_policies_US_final_Detroit_Warren_Dearborn,
            df_policies_US_final_Nashville_Davidson_Murfreesboro_Franklin, df_policies_US_final_Birmingham_Hoover,
            df_policies_US_final_Los_Angeles_Long_Beach_Orange_County,
        ]
    ).reset_index(drop=True)

    df_policies_US_final_concat =  df_policies_US_final_concat[
        (df_policies_US_final_concat['province'].isin(list_US_states)) |
        (df_policies_US_final_concat['province'].isin(us_county_names.Province))
    ]
    return df_policies_US_final_concat



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

    flags = ['C' + str(i) + '_Flag' for i in range(1, 8)] + ["H1_Flag"]
    measures = measures.loc[:, filtr + msr + flags + target]
    measures['Date'] = measures['Date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    for col in target:
        #measures[col] = measures[col].fillna(0)
        measures[col] = measures.groupby('CountryName')[col].ffill()

    measures['C1_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C1_School closing'],measures['C1_Flag'])]
    measures['C2_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C2_Workplace closing'],measures['C2_Flag'])]
    measures['C3_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C3_Cancel public events'],measures['C3_Flag'])]
    measures['C4_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C4_Restrictions on gatherings'],measures['C4_Flag'])]
    measures['C5_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C5_Close public transport'],measures['C5_Flag'])]
    measures['C6_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C6_Stay at home requirements'],measures['C6_Flag'])]
    measures['C7_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['C7_Restrictions on internal movement'],measures['C7_Flag'])]
    measures['H1_Flag'] = [0 if x <= 0  else y for (x,y) in zip(measures['H1_Public information campaigns'],measures['H1_Flag'])]

    measures['C1_School closing'] = [int(a and b) for a, b in zip(measures['C1_School closing'] >= 2, measures['C1_Flag'] == 1)]

    measures['C2_Workplace closing'] = [int(a and b) for a, b in zip(measures['C2_Workplace closing'] >= 2, measures['C2_Flag'] == 1)]

    measures['C3_Cancel public events'] = [int(a and b) for a, b in zip(measures['C3_Cancel public events'] >= 2, measures['C3_Flag'] == 1)]

    measures['C4_Restrictions on gatherings'] = [int(a and b) for a, b in zip(measures['C4_Restrictions on gatherings'] >= 1, measures['C4_Flag'] == 1)]

    measures['C5_Close public transport'] = [int(a and b) for a, b in zip(measures['C5_Close public transport'] >= 2, measures['C5_Flag'] == 1)]

    measures['C6_Stay at home requirements'] = [int(a and b) for a, b in zip(measures['C6_Stay at home requirements'] >= 2, measures['C6_Flag'] == 1)]

    measures['C7_Restrictions on internal movement'] = [int(a and b) for a, b in zip(measures['C7_Restrictions on internal movement'] >= 2, measures['C7_Flag'] == 1)]

    measures['C8_International travel controls'] = [int(a) for a in (measures['C8_International travel controls'] >= 3)]

    measures['H1_Public information campaigns'] = [int(a and b) for a, b in zip(measures['H1_Public information campaigns'] >= 1, measures['H1_Flag'] == 1)]

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

    # counties:
    ex_us_county_names = pd.read_csv(
         "data_sandbox/processed/Ex_US_counties.csv"
    )

    ex_us_regions_names = pd.read_csv(
        "data_sandbox/processed/Ex_US_regions.csv"
    )

    countries = ex_us_regions_names.Country.unique()
    for country in countries:
        provinces = ex_us_regions_names[ex_us_regions_names.Country == country].Province.values
        provinces_keyNames = ex_us_county_names[ex_us_county_names.Country == country].Province.values
        outputs_provinces = []
        for province in np.concatenate((provinces_keyNames,provinces)):
            temp = output[output.country == country]
            temp.loc[:, "province"] = province
            outputs_provinces.append(temp)
        outputs_provinces = pd.concat(outputs_provinces).reset_index(drop=True)
        output = pd.concat([
            output, outputs_provinces]
        )
    output = output.sort_values(["country", "province", "date"]).reset_index(drop=True)
    return output


def gamma_t_jj_version(day, country, province, params_dic):
    dsd, median_day_of_action, rate_of_action = params_dic[(country, province)]
    t = (day - pd.to_datetime(dsd)).days
    gamma = (2 / np.pi) * np.arctan(-(t - median_day_of_action) / 20 * rate_of_action) + 1
    return gamma
