# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
import pandas as pd
import numpy as np
import zipfile
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from DELPHI_utils_V3 import (
    DELPHIDataCreator,
    get_initial_conditions, mape,
    get_normalized_policy_shifts_and_current_policy_us_only,
    upload_s3_file
)
from DELPHI_utils_additional_states import (
    read_measures_oxford_data_jj_version, get_normalized_policy_shifts_and_current_policy_all_countries_jj_version,
    read_policy_data_us_only_jj_version
)
from DELPHI_params_V3 import (
    validcases_threshold_policy,
    IncubeD, RecoverID, RecoverHD, DetectD, VentilatedD,
    default_maxT_policies, p_v, p_d, p_h, future_policies, future_times
)
from DELPHI_params import ( future_times_JJ, future_policies_JJ)

import os, sys, yaml
import matplotlib.pyplot as plt


with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]

def run_policy_prediction_additional_state(PATH_TO_DATA_SANDBOX,PATH_TO_FOLDER_DANGER_MAP,
                                           trainging_date, country_lists,provinces_lists,popcountries,replace_deathcounts,
                                           upload_to_s3,today_time):
    training_end_date = trainging_date

    yesterday = "".join(str(training_end_date.date() - timedelta(days=1)).split("-"))
    day_after_yesterday = "".join(str(pd.to_datetime(yesterday).date() + timedelta(days=1)).split("-"))
    print(f" using the parameters in day {yesterday} for policy prediction, day_after_yesterday {day_after_yesterday}")
    policy_data_countries = read_measures_oxford_data_jj_version()


    def createParameters_JJ_Global(path_to, PATH_TO_DATA_SANDBOX, yesterday ):
        Parameters_Global = pd.read_csv(path_to+f'Parameters_Global_V2_20200925.csv')
        Parameters_J = pd.read_csv(PATH_TO_DATA_SANDBOX + f'predicted/parameters/Parameters_J&J_{yesterday}.csv')

        parameter_Global_J = pd.concat([Parameters_J,Parameters_Global]).reset_index(drop=True)
        already_exist = os.path.exists(PATH_TO_DATA_SANDBOX +f'predicted/parameters/Parameters_J&J_Global_{yesterday}.csv')
        if already_exist:
            os.remove(PATH_TO_DATA_SANDBOX + f'predicted/parameters/Parameters_J&J_Global_{yesterday}.csv')

        parameter_Global_J.to_csv(PATH_TO_DATA_SANDBOX + f'predicted/parameters/Parameters_J&J_Global_{yesterday}.csv',index = False)
        return os.path.exists(PATH_TO_DATA_SANDBOX +f'predicted/parameters/Parameters_J&J_Global_{yesterday}.csv')


    PATH_TO_PARAM_GLOBAL = PATH_TO_FOLDER_DANGER_MAP + 'predicted/'
    param_global_JJ_created = createParameters_JJ_Global(PATH_TO_PARAM_GLOBAL, PATH_TO_DATA_SANDBOX, yesterday)
    if not param_global_JJ_created:
        print( f"predicted/parameters/Parameters_J&J_Global_{yesterday}.csv is not created")
        sys.exit()

    pastparameters = pd.read_csv(PATH_TO_DATA_SANDBOX + f"predicted/parameters/Parameters_J&J_Global_{yesterday}.csv")
    # TODO/ files Parameters_J&J_Global_{yesterday}.csv must be a concatenation of (1) the existing regular parameter file
    #  (Parameters_Global_{yesterday}.csv) used by DELPHI on the website, and (2) the Parameters_J&J_{yesterday}.csv
    #  which is the output of DELPHI_model_additional_states.py because (2) contains extra regions/provinces (e.g states of
    #  Brazil, provinces of Peru, US Cities...) that (1) doesn't contain, and we need all of them to generate shifting
    #  coefficients for the policies! So you guys might need to get a daily or bi-daily update of our parameters (1) so that
    #  you can concatenate them with (2) that you can generate on your own. In the worst case, it's okay to use a more
    #  a more anterior date for (1) compared to (2) (which should be the actual yesterday of the day you're running)

    # Get the policies shifts from the CART tree to compute different values of gamma(t)
    # Depending on the policy in place in the future to affect predictions
    dict_normalized_policy_gamma_countries, dict_current_policy_countries = (
        get_normalized_policy_shifts_and_current_policy_all_countries_jj_version(
            policy_data_countries=policy_data_countries,
            pastparameters=pastparameters,
        )
    )
    # Setting same value for these 2 policies because of the inherent structure of the tree
    dict_normalized_policy_gamma_countries[future_policies[3]] = dict_normalized_policy_gamma_countries[future_policies[5]]
    # Parameters_JJ = pd.read_csv(PATH_TO_DATA_SANDBOX + f'predicted/parameters/Parameters_J&J_{yesterday}.csv')
    policy_data_us_only = read_policy_data_us_only_jj_version(filepath_data_sandbox=PATH_TO_DATA_SANDBOX,
                                                              parameters_us_state = pastparameters)

    ## US Only Policies
    dict_normalized_policy_gamma_us_only, dict_current_policy_us_only = (
        get_normalized_policy_shifts_and_current_policy_us_only(
            policy_data_us_only=policy_data_us_only,
            pastparameters=pastparameters,
        )
    )
    dict_current_policy_international = dict_current_policy_countries.copy()
    dict_current_policy_international.update(dict_current_policy_us_only)
    str_date = "".join(str(today_time.date()).split("-"))
    path_to_results = 'data_sandbox/predicted/policy_scenario_predictions/'
    dic_file_name = f'policy_{str_date}_provinces.csv'
    with open(path_to_results + dic_file_name, 'w') as f:
        for key in dict_current_policy_international.keys():
            c_name , p_name = key
            f.write("%s,%s\n"%((c_name.replace(',',' '),p_name),dict_current_policy_international[key]))
    if upload_to_s3 and 'US' in country_lists:
        upload_s3_file(path_to_results + dic_file_name,dic_file_name)
    dict_normalized_policy_gamma_us_only = {
        'No_Measure': 1.0,
        'Restrict_Mass_Gatherings': 0.873,
        'Mass_Gatherings_Authorized_But_Others_Restricted': 0.668,
        'Restrict_Mass_Gatherings_and_Schools': 0.479,
        'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.794,
        'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.423,
        'Lockdown': 0.239,
    }
    dict_normalized_policy_gamma_countries = {
        'No_Measure': 1.0,
        'Restrict_Mass_Gatherings': 0.873,
        'Mass_Gatherings_Authorized_But_Others_Restricted': 0.668,
        'Restrict_Mass_Gatherings_and_Schools': 0.479,
        'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.794,
        'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.423,
        'Lockdown': 0.239,
    }

    # Initalizing lists of the different dataframes that will be concatenated in the end
    list_df_global_predictions_since_today_scenarios = []
    list_df_global_predictions_since_100_cases_scenarios = []
    obj_value = 0
    us_county_names = pd.read_csv(
        PATH_TO_DATA_SANDBOX + f"processed/US_counties.csv"
    )

    ex_us_regions = pd.read_csv(
        PATH_TO_DATA_SANDBOX + f"processed/Ex_US_regions.csv"
    )


    for continent, country, province in zip(
            popcountries.Continent.tolist(),
            popcountries.Country.tolist(),
            popcountries.Province.tolist(),
    ):
        if country == "US":  # This line is necessary because the keys are the same in both cases
            dict_normalized_policy_gamma_international = dict_normalized_policy_gamma_us_only.copy()
        else:
            dict_normalized_policy_gamma_international = dict_normalized_policy_gamma_countries.copy()

        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")


        if country_sub not in country_lists:
            continue
        regions_name_values = ex_us_regions[ex_us_regions.Country == country].Province.values
        regions_name = [x.replace(" ", "_") for x in regions_name_values]
        all_us_non_us_prov = np.concatenate((us_county_names.Province.values,regions_name))
        if province_sub == "None" or province_sub not in all_us_non_us_prov or \
                (len(provinces_lists) > 0 and province_sub not in provinces_lists):
            continue

        if os.path.exists(
                PATH_TO_DATA_SANDBOX + f"processed/{country_sub}_J&J/Cases_{country_sub}_{province_sub}.csv"):
            print(country + ", " + province)
            totalcases = pd.read_csv(
                PATH_TO_DATA_SANDBOX + f"processed/{country_sub}_J&J/Cases_{country_sub}_{province_sub}.csv"
            )
            if province_sub in replace_deathcounts:
                totalcases.loc[totalcases['day_since100'] <= 2, ['death_cnt']] = 1
            if totalcases.day_since100.max() < 0:
                print(f"Not enough cases for Continent={continent}, Country={country} and Province={province}")
                continue
            print(country + " " + province)
            if pastparameters is not None:
                parameter_list_total = pastparameters[
                    (pastparameters.Country == country) &
                    (pastparameters.Province == province)
                    ]
                if len(parameter_list_total) > 0:
                    parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                    parameter_list = parameter_list_line[5:]
                    date_day_since100 = pd.to_datetime(parameter_list_line[3])
                    # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                    validcases = totalcases[
                        (totalcases.day_since100 >= 0) &
                        (totalcases.date <= str(pd.to_datetime(day_after_yesterday).date()))
                    ][["date", "day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                else:
                    print(f"Must have past parameters for {country} and {province}")
                    continue
            else:
                print("Must have past parameters")
                continue

            # Now we start the modeling part:
            if len(validcases) > validcases_threshold_policy:
                PopulationT = popcountries[
                    (popcountries.Country == country) & (popcountries.Province == province)
                    ].iloc[0].pop2016.item()
                # We do not scale
                N = PopulationT
                PopulationI = validcases.loc[0, "case_cnt"]
                PopulationR = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
                PopulationD = validcases.loc[0, "death_cnt"]
                PopulationCI = PopulationI - PopulationD - PopulationR
                """
                Fixed Parameters based on meta-analysis:
                p_h: Hospitalization Percentage
                RecoverHD: Average Days till Recovery
                VentilationD: Number of Days on Ventilation for Ventilated Patients
                maxT: Maximum # of Days Modeled
                p_d: Percentage of True Cases Detected
                p_v: Percentage of Hospitalized Patients Ventilated,
                balance: Ratio of Fitting between cases and deaths
                """
                # Currently fit on alpha, a and b, r_dth,
                # Maximum timespan of prediction, defaulted to go to 15/06/2020
                maxT = (default_maxT_policies - date_day_since100).days + 1
                """ Fit on Total Cases """
                t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
                validcases_nondeath = validcases["case_cnt"].tolist()
                validcases_death = validcases["death_cnt"].tolist()
                balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
                fitcasesnd = validcases_nondeath
                fitcasesd = validcases_death
                GLOBAL_PARAMS_FIXED = (
                    N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v
                )
                best_params = parameter_list
                t_predictions = [i for i in range(maxT)]
                # plt.figure(figsize=(16, 10))
                for future_policy in future_policies_JJ:  # This is the list of policies generated, the possibilities are in DELPHI_params.py
                    for future_time in future_times_JJ:
                        # Only generate the policies with timing "Now", the possibilities are in DELPHI_params.py
                        def model_covid_predictions(
                                t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal
                        ):
                            """
                            SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
                            alpha: Infection rate
                            days: Median day of action
                            r_s: Median rate of action
                            p_dth: Mortality rate
                            k1: Internal parameter 1
                            k2: Internal parameter 2
                            y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
                            7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
                            """
                            r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
                            r_d = np.log(2) / DetectD  # Rate of detection
                            r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
                            r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
                            r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
                            gamma_t = (
                                    (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1 +
                                    jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
                            )
                            gamma_t_future = (
                                    (2 / np.pi) * np.arctan(-(t_cases[-1] + future_time - days) / 20 * r_s) + 1 +
                                    jump * np.exp(-(t_cases[-1] + future_time - t_jump) ** 2 / (2 * std_normal ** 2))
                            )
                            p_dth_mod = (2 / np.pi) * (p_dth - 0.01) * (np.arctan(- t / 20 * r_dthdecay) + np.pi / 2) + 0.01
                            if t > t_cases[-1] + future_time:
                                normalized_gamma_future_policy = dict_normalized_policy_gamma_countries[future_policy]
                                normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                                    dict_current_policy_international[(country, province)]
                                ]
                                epsilon = 1e-4
                                gamma_t = gamma_t + min(
                                    (2 - gamma_t_future) / (1 - normalized_gamma_future_policy + epsilon),
                                    (gamma_t_future / normalized_gamma_current_policy) *
                                    (normalized_gamma_future_policy - normalized_gamma_current_policy)
                                )

                            assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
                            S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
                            # Equations on main variables
                            dSdt = -alpha * gamma_t * S * I / N
                            dEdt = alpha * gamma_t * S * I / N - r_i * E
                            dIdt = r_i * E - r_d * I
                            dARdt = r_d * (1 - p_dth_mod) * (1 - p_d) * I - r_ri * AR
                            dDHRdt = r_d * (1 - p_dth_mod) * p_d * p_h * I - r_rh * DHR
                            dDQRdt = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * I - r_ri * DQR
                            dADdt = r_d * p_dth_mod * (1 - p_d) * I - r_dth * AD
                            dDHDdt = r_d * p_dth_mod * p_d * p_h * I - r_dth * DHD
                            dDQDdt = r_d * p_dth_mod * p_d * (1 - p_h) * I - r_dth * DQD
                            dRdt = r_ri * (AR + DQR) + r_rh * DHR
                            dDdt = r_dth * (AD + DQD + DHD)
                            # Helper states (usually important for some kind of output)
                            dTHdt = r_d * p_d * p_h * I
                            dDVRdt = r_d * (1 - p_dth_mod) * p_d * p_h * p_v * I - r_rv * DVR
                            dDVDdt = r_d * p_dth_mod * p_d * p_h * p_v * I - r_dth * DVD
                            dDDdt = r_dth * (DHD + DQD)
                            dDTdt = r_d * p_d * I
                            return [
                                dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                                dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
                            ]


                        def solve_best_params_and_predict(optimal_params):
                            # Variables Initialization for the ODE system
                            x_0_cases = get_initial_conditions(
                                params_fitted=optimal_params,
                                global_params_fixed=GLOBAL_PARAMS_FIXED
                            )
                            x_sol_best = solve_ivp(
                                fun=model_covid_predictions,
                                y0=x_0_cases,
                                t_span=[t_predictions[0], t_predictions[-1]],
                                t_eval=t_predictions,
                                args=tuple(optimal_params),
                            ).y
                            return x_sol_best


                        x_sol_final = solve_best_params_and_predict(best_params)
                        data_creator = DELPHIDataCreator(
                            x_sol_final=x_sol_final, date_day_since100=date_day_since100, best_params=best_params,
                            continent=continent, country=country, province=province,
                        )
                        # Creating the parameters dataset for this (Continent, Country, Province)
                        mape_data = (
                                            mape(fitcasesnd, x_sol_final[15, :len(fitcasesnd)]) +
                                            mape(fitcasesd, x_sol_final[14, :len(fitcasesd)])
                                    ) / 2
                        try:
                            mape_data_2 = (
                                                  mape(fitcasesnd[-15:],
                                                       x_sol_final[15, len(fitcasesnd) - 15:len(fitcasesnd)]) +
                                                  mape(fitcasesd[-15:],
                                                       x_sol_final[14, len(fitcasesnd) - 15:len(fitcasesd)])
                                          ) / 2
                        except IndexError:
                            mape_data_2 = mape_data
                        print(
                            "Policy: ", future_policy, "\t Enacting Time: ", future_time, "\t Total MAPE=", mape_data,
                            "\t MAPE on last 15 days=", mape_data_2
                        )
                        # print(best_params)
                        # print(country + ", " + province)
                        # if future_policy in [
                        #     'No_Measure', 'Restrict_Mass_Gatherings',
                        #     'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others', 'Lockdown'
                        # ]:
                        #     plt.plot(x_sol_final[15, :], label=f"Future Policy: {future_policy} in {future_time} days")
                        # Creating the datasets for predictions of this (Continent, Country, Province)
                        df_predictions_since_today_cont_country_prov, df_predictions_since_100_cont_country_prov = (
                            data_creator.create_datasets_predictions_scenario(
                                policy=future_policy,
                                time=future_time,
                                totalcases=totalcases,
                            )
                        )
                        list_df_global_predictions_since_today_scenarios.append(
                            df_predictions_since_today_cont_country_prov)
                        list_df_global_predictions_since_100_cases_scenarios.append(
                            df_predictions_since_100_cont_country_prov)
                print(f"Finished predicting for Continent={continent}, Country={country} and Province={province}")
                # plt.plot(fitcasesnd, label="Historical Data")
                # plt.legend()
                # plt.title(f"{country}, {province} Predictions & Historical for # Cases")
                # plt.savefig(country + "_" + province + "_prediction_cases.png")
                print("--------------------------------------------------------------------------")
            else:  # len(validcases) <= 7
                print(f"Not enough historical data (less than a week)" +
                      f"for Continent={continent}, Country={country} and Province={province}")
                continue
        else:  # file for that tuple (country, province) doesn't exist in processed files
            continue

    df_global_predictions_since_today_scenarios = pd.concat(
        list_df_global_predictions_since_today_scenarios
    ).reset_index(drop=True)
    df_global_predictions_since_100_cases_scenarios = pd.concat(
        list_df_global_predictions_since_100_cases_scenarios
    ).reset_index(drop=True)
    file_name =  f'df_scenarios_provinces_j&j_{str_date}'+'_US' if 'US' in country_lists else  \
        f'df_scenarios_provinces_j&j_{str_date}'+'_Ex_US'
    path_to_output = PATH_TO_DATA_SANDBOX + f'predicted/policy_scenario_predictions/' + file_name + '.csv'
    path_to_output_zip = PATH_TO_DATA_SANDBOX + f'predicted/policy_scenario_predictions/' + file_name + '.zip'
    df_global_predictions_since_100_cases_scenarios.to_csv(
        path_to_output,
        index=False
    )
    zipfile.ZipFile(path_to_output_zip, 'w', zipfile.ZIP_DEFLATED).write(path_to_output,file_name + '.csv')
    print("Exported all policy-dependent predictions for all countries to data_sandbox")
    if upload_to_s3:
        upload_s3_file(path_to_output,file_name + '.csv')
    os.remove(path_to_output)
