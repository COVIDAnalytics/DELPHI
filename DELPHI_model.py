# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import datetime, timedelta
import dateutil.parser as dtparser
import os

yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
# TODO: Find a way to make this path automatic, whoever the user is...

PATH_TO_FOLDER = (
    # "E:/Github/covid19orc/danger_map"
        "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
        "4. COVID19_Global/covid19orc/danger_map"
)
os.chdir(PATH_TO_FOLDER)

popcountries = pd.read_csv(
    f"processed/Global/Population_Global.csv"
)
try:
    pastparameters = pd.read_csv(
        f"predicted/Parameters_Global_{yesterday}.csv"
    )
except:
    pastparameters = None
for continent, country, province in zip(
        popcountries.Continent.tolist()[:10],
        popcountries.Country.tolist()[:10],
        popcountries.Province.tolist()[:10]
):
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    if os.path.exists(f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if pastparameters is not None:
            parameter_list_total = pastparameters[
                (pastparameters.Country == country) &
                (pastparameters.Province == province)
                ]
            if len(parameter_list_total) > 0:
                parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
                parameter_list = parameter_list_line[4:]
                # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
                param_list_lower = [x - 0.05 * abs(x) for x in parameter_list]
                param_list_upper = [x + 0.05 * abs(x) for x in parameter_list]
                bounds_params = tuple(
                    [(lower, upper)
                     for lower, upper in zip(param_list_lower, param_list_upper)]
                )
                # TODO 17/04/2020: Check that format is right for validcases
                if country == "US":
                    validcases = totalcases[[
                        dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                        for x in totalcases.date
                    ]][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
                else:
                    validcases = totalcases[[
                        dtparser.parse(x) >= dtparser.parse(parameter_list_line[3])
                        for x in totalcases.date
                    ]][["date_count", "case_cnt", "death_cnt"]].reset_index(drop=True)
            else:
                # Otherwise use established lower/upper bounds
                parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
                bounds_params = (
                    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
                )
                if country == "US":
                    validcases = totalcases[totalcases.day_since100 >= 0][
                        ["day_since100", "case_cnt", "death_cnt"]
                    ].reset_index(drop=True)
                else:
                    validcases = totalcases[totalcases.case_cnt >= 100][
                        ["date_count", "case_cnt", "death_cnt"]
                    ].reset_index(drop=True)
        else:
            # Otherwise use established lower/upper bounds
            parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3]
            bounds_params = (
                (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
            )
            if country == "US":
                validcases = totalcases[totalcases.day_since100 >= 0][
                    ["day_since100", "case_cnt", "death_cnt"]
                ].reset_index(drop=True)
            else:
                validcases = totalcases[totalcases.case_cnt >= 100][
                    ["date_count", "case_cnt", "death_cnt"]
                ].reset_index(drop=True)
                # Now we start the modeling part:
        if len(validcases) > 7:
            IncubeD = 5
            RecoverID = 10
            DetectD = 2
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
                ].pop2016.item()
            PopulationI = validcases.loc[0, "case_cnt"]
            PopulationR = validcases.loc[0, "death_cnt"] * 10  # TODO: 10x more recovered than dead ? Is that from studies or guts ?
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
            # & initial condition of exposed state and infected state
            RecoverHD = 15  # TODO 17/04/2020: Add comment: What is this?
            VentilatedD = 10  # TODO 17/04/2020: Add comment: What is this ?
            maxT = 100  # Maximum timespan of prediction
            p_v = 0.25  # Percentage of ventilated
            p_d = 0.2  # Percentage of infection cases detected.
            p_h = 0.15  # Percentage of detected cases hospitalized
            """ Fit on Total Cases """
            if country == "US":
                t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
            else:
                t_cases = validcases["date_count"].tolist() - validcases.loc[0, "date_count"]
            validcases_nondeath = validcases["case_cnt"].tolist()
            validcases_death = validcases["death_cnt"].tolist()
            balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
            fitcasesnd = [x / PopulationT for x in validcases_nondeath]
            fitcasesd = [x / PopulationT for x in validcases_death]


            def model_covid(
                    t, x, alpha, days, r_s, r_dth, p_dth, k1, k2
            ):
                """
                SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
                alpha: Infection rate
                days:
                r_s:
                p_dth: Mortality rate
                k1: Internal parameter 1
                k2: Internal parameter 2
                y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD,
                7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
                """
                r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
                r_d = np.log(2) / DetectD  # Rate of death leaving incubation phase
                r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
                r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
                r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
                gamma_t = (2 / np.pi) * np.arctan(-(t - days) / r_s) + 1
                assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
                S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
                # Equations on main variables
                dSdt = -alpha * gamma_t * S * I
                dEdt = -alpha * gamma_t * S * I - r_i * E
                dIdt = r_i * E - r_d * I
                dARdt = r_d * (1 - p_dth) * (1 - p_d) * I - r_ri * AR
                dDQRdt = r_d * (1 - p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
                dDHRdt = r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
                dADdt = r_d * p_dth * (1 - p_d) * I - r_dth * AD
                dDQDdt = r_d * p_dth * p_d * (1 - p_h) * I - r_dth * DQD
                dDHDdt = r_d * p_dth * p_d * p_h * I - r_dth * DHD
                dRdt = r_ri * (AR + DQR) + r_rh * DHR
                dDdt = r_dth * (AD + DQD + DHD)

                # Helper states (not in the ODEs but important for fitting)
                dTHdt = r_d * p_d * p_h * I
                dDVRdt = r_d * (1 - p_dth) * p_d * p_h * p_v * I - r_rv * DVR
                dDVDdt = r_d * p_dth * p_d * p_h * p_v * I - r_dth * DVD
                dDDdt = r_dth * (DHD + DQD)
                dDTdt = r_d * p_d * I
                return [
                    dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                    dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
                ]


            def residuals_totalcases(params):
                """
                Wanted to start with solve_ivp because figures will be faster to debug
                """
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, k1, k2 = params
                S_0 = (
                        (1 - PopulationCI / (p_d * PopulationT)) -
                        (PopulationCI / (p_d * PopulationT) * (k1 + k2)) -
                        (PopulationR / (p_d * PopulationT)) -
                        (PopulationD / (p_d * PopulationT))
                )
                E_0 = PopulationCI / (p_d * PopulationT) * k1
                I_0 = PopulationCI / (p_d * PopulationT) * k2
                AR_0 = (
                               (PopulationCI / (p_d * PopulationT))
                               - (PopulationCI / PopulationT)
                       ) * (1 - p_dth)
                DHR_0 = (PopulationCI * p_h / PopulationT) * (1 - p_dth)
                DQR_0 = (PopulationCI * (1 - p_h) / PopulationT) * (1 - p_dth)
                AD_0 = (
                               (PopulationCI / (p_d * PopulationT))
                               - (PopulationCI / PopulationT)
                       ) * p_dth
                DHD_0 = PopulationCI * p_h / PopulationT * p_dth
                DQD_0 = PopulationCI * (1 - p_h) / PopulationT * p_dth
                R_0 = PopulationR / (PopulationT * p_d)
                D_0 = PopulationD / (PopulationT * p_d)
                TH_0 = PopulationCI * p_h / PopulationT
                DVR_0 = (PopulationCI * p_h * p_v / PopulationT) * (1 - p_dth)
                DVD_0 = (PopulationCI * p_h * p_v / PopulationT) * p_dth
                DD_0 = PopulationD / PopulationT
                DT_0 = PopulationI / PopulationT
                x_0_cases = [
                    S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0,
                    R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0
                ]
                x_sol = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_cases[0], t_cases[-1]],
                    t_eval=t_cases,
                    args=tuple(params),
                ).y
                weights = list(range(1, len(fitcasesnd) + 1))
                residuals_value = sum(
                    np.multiply((x_sol[14, :] - fitcasesnd) ** 2, weights)
                    + balance * np.multiply((x_sol[13, :] - fitcasesd) ** 2, weights)
                )
                return residuals_value


            print(country + ", " + province)
            best_params = minimize(
                residuals_totalcases,
                parameter_list,
                method='L-BFGS-B',  # Can't use Nelder-Mead if I want to put bounds on the params
                bounds=bounds_params
            ).x
            t_predictions = [i for i in range(maxT)]

            def solve_best_params_and_predict(optimal_params):
                # Variables Initialization for the ODE system
                alpha, days, r_s, r_dth, p_dth, k1, k2 = optimal_params
                S_0 = (
                        (1 - PopulationCI / (p_d * PopulationT)) -
                        (PopulationCI / (p_d * PopulationT) * (k1 + k2)) -
                        (PopulationR / (p_d * PopulationT)) -
                        (PopulationD / (p_d * PopulationT))
                )
                E_0 = PopulationCI / (p_d * PopulationT) * k1
                I_0 = PopulationCI / (p_d * PopulationT) * k2
                AR_0 = ((PopulationCI / (p_d * PopulationT)) - (PopulationCI / PopulationT)) * (1 - p_dth)
                DHR_0 = (PopulationCI * p_h / PopulationT) * (1 - p_dth)
                DQR_0 = (PopulationCI * (1 - p_h) / PopulationT) * (1 - p_dth)
                AD_0 = (
                               (PopulationCI / (p_d * PopulationT))
                               - (PopulationCI / PopulationT)
                       ) * p_dth
                DHD_0 = PopulationCI * p_h / PopulationT * p_dth
                DQD_0 = PopulationCI * (1 - p_h) / PopulationT * p_dth
                R_0 = PopulationR / (PopulationT * p_d)
                D_0 = PopulationD / (PopulationT * p_d)
                TH_0 = PopulationCI * p_h / PopulationT
                DVR_0 = (PopulationCI * p_h * p_v / PopulationT) * (1 - p_dth)
                DVD_0 = (PopulationCI * p_h * p_v / PopulationT) * p_dth
                DD_0 = PopulationD / PopulationT
                DT_0 = PopulationI / PopulationT
                x_0_cases = [
                    S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0,
                    R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0
                ]
                x_sol_best = solve_ivp(
                    fun=model_covid,
                    y0=x_0_cases,
                    t_span=[t_predictions[0], t_predictions[-1]],
                    t_eval=t_predictions,
                    args=tuple(optimal_params),
                ).y
                return x_sol_best

            x_sol_final = solve_best_params_and_predict(best_params)
