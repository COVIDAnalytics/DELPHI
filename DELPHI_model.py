# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from datetime import datetime, timedelta
import dateutil.parser as dtparser
import os
import time

yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
# TODO: Find a way to make this path automatic, whoever the user is...

PATH_TO_FOLDER = (
    "E:/Github/covid19orc/danger_map"
    # "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
    # "4. COVID19_Global/covid19orc/danger_map"
)
os.chdir(PATH_TO_FOLDER)

popcountries = pd.read_csv(
    f"processed/Global/Population_Global.csv"
)
try:
    pastparameters = pd.read_csv(
    f"predicted/Parameters_Global_{yesterday}.csv")
except:
    pastparameters = None
for continent, country, province in zip(popcountries.Continent.tolist(),popcountries.Country.tolist(),popcountries.Province.tolist()):
    country_sub = country.replace(" ","_")
    province_sub = province.replace(" ","_")
    if os.path.exists(f"processed/Global/Cases_{country_sub}_{province_sub}.csv"):
        totalcases = pd.read_csv(
            f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
        )
        if pastparameters is not None:
            parameter_list_total = pastparameters[
                                 (pastparameters.Country == country) &  (pastparameters.Province == province)
                              ]
            if len(parameter_list_total)>0:
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
                    validcases = totalcases[[dtparser.parse(x)>= dtparser.parse(parameter_list_line[3]) for x in totalcases.date]
                        ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)                   
                else:
                    validcases = totalcases[[dtparser.parse(x)>= dtparser.parse(parameter_list_line[3]) for x in totalcases.date]
                        ][["date_count", "case_cnt", "death_cnt"]].reset_index(drop=True)  
            else:
                # Otherwise use established lower/upper bounds
                parameter_list = [1,0,2,0.2,0.05,3,3]
                
                bounds_params = (
                    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
                )
                if country == "US":
                    validcases = totalcases[totalcases.day_since100 >= 0][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)                    
                else:
                    validcases = totalcases[totalcases.case_cnt >= 100][["date_count", "case_cnt", "death_cnt"]].reset_index(drop=True)
        # Now we start the modeling part:
        if len(validcases) > 7:
            IncubeD = 5
            RecoverID = 10
            DetectD = 2
            PopulationT = popcountries[
                (popcountries.Country == country) & (popcountries.Province == province)
            ].pop2016.item()
            PopulationI = validcases.loc[0, "case_cnt"]
            PopulationR = validcases.loc[0, "death_cnt"]*10  # TODO: 10x more recovered than dead ? Is that from studies or guts ?
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
            # TODO 17/04/2020: Verifity that this balance is right
    
            """ Fit on Total Cases """
            if country == "US":
                t_cases = validcases["day_since100"].tolist() -  validcases.loc[0,"day_since100"]
            else:
                t_cases = validcases["date_count"].tolist() -  validcases.loc[0,"date_count"]
            validcasesnd = validcases["case_cnt"].tolist()
            validcasesd = validcases["death_cnt"].tolist()
            balance = validcasesnd[-1] / max(validcasesd[-1],10) / 3        
            fitcasesnd = [x / PopulationT for x in validcasesnd]
            fitcasesd =  [x / PopulationT for x in validcasesd]   
            
            
            # TODO 17/04/2020 : Translate this code from Mathematica (don't really understand)
            #validcasesnd = validcases[[All, 1;; 2]];  TODO 17/04/2020
            #fitcasesnd = validcasesnd[[2;;, All]];  TODO 17/04/2020
            #fitcasesnd[[All, 2]] /= PopulationT;  TODO 17/04/2020
            #fitcasesnd[[All, 1]] -= fitcasesnd[[1, 1]] - 1;  TODO 17/04/2020
    
            """ Fit on Deaths """
    
    
            # TODO 17/04/2020 : Translate this code from Mathematica (don't really understand)
            #validcasesd = validcases[[All, {1, 3}]];  TODO 17/04/2020
            #fitcasesd = validcasesd[[2;;, All]];  TODO 17/04/2020
            #fitcasesd[[All, 2]] /= (PopulationT / balance);  TODO 17/04/2020
            #fitcasesd[[All, 1]] -= fitcasesd[[1, 1]] - 1;  TODO 17/04/2020
    
            def model_covid(
                    x, t,
                    alpha, days, r_s, r_dth, p_dth, k1, k2):
                """
                SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
                alpha: Infection rate
                days:
                r_s:
                p_dth: Mortality rate
                k1: Internal parameter 1
                k2: Internal parameter 2
                y = [0 S, 1 E,  2 I, 3 AR,   4 DHR,  5 DQR, 6 AD, 
              7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DV, 13 DD, 14 DT]
                """
                r_i = np.log(2)/IncubeD  # Rate of infection leaving incubation phase
                r_d = np.log(2)/DetectD  # Rate of death leaving incubation phase TODO 17/04/2020: Verify ?
                r_ri = np.log(2)/RecoverID  # Rate of recovery not under infection
                r_rh = np.log(2)/RecoverHD  # Rate of recovery under hospitalization
                r_rv = np.log(2)/VentilatedD  # Rate of recovery under ventilation
                gamma_t = (2/np.pi) * np.arctan(-(t - days) / r_s) + 1
                # TODO: Don't know how to deal with this gamma to take t as a parameter
                #       Or maybe use it as a parameter in odeint, no idea really
                assert len(x) == 15, f"Too many input variables, got {len(x)}, expected 12"
                S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DV, DD, DT = x
                # Equations on main variables
                dSdt = -alpha * gamma_t * S * I
                dEdt = -alpha * gamma_t * S * I - r_i * E
                dIdt = r_i * E - r_d * I
                dARdt = r_d * (1-p_dth) * I - r_ri * AR
                dDHRdt = r_d * (1-p_dth) * p_d * p_h * I - r_rh * DHR
                dDQRdt = r_d * (1-p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
                dADdt = r_d * p_dth * (1-p_d) * I -  r_dth * AD
                dDHDdt = r_d * p_dth * p_d * p_h * I -  r_dth * DHD
                dDQDdt = r_d * p_dth * p_d * (1 - p_h) * I -  r_dth * DQD
                dRdt = r_ri * (AR + DQR) + r_rh * DHR
                dDdt = r_dth * (AD + DQD + DHD)
    
                # Helper states (not in the ODEs but important for fitting)
                dTHdt = r_d * p_d * p_h * I
                dDVdt = r_d * p_d * p_h * p_v * I - (r_rv + r_dth) * DV
                dDDdt = r_dth * (DHD + DQD)
                dDTdt = r_d * p_d * I
                return [
                    dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
                    dRdt, dDdt, dTHdt, dDVdt, dDDdt, dDTdt
                ]
    
    
            # Initialization of parameters to be fitted (from previous fits)
            alpha, days, r_s, r_dth, p_dth, k1, k2 = parameter_list
            # Variables Initialization for the ODE system
    
    
    
            def residuals_totalcases(params):
                """
                Wanted to start with odeint because figures will be faster to debug
                """
                alpha, days, r_s, r_dth, p_dth, k1, k2 = params
                S_0 = (
                    (1 - PopulationCI/(p_d * PopulationT)) -
                    (PopulationCI / (p_d * PopulationT) * (k1+k2)) -
                    (PopulationR / (p_d * PopulationT)) -
                    (PopulationD / (p_d * PopulationT))
                )
                E_0 = PopulationCI / (p_d * PopulationT) * k1
                I_0 = PopulationCI / (p_d * PopulationT) * k2
                AR_0 = ((PopulationCI / (p_d * PopulationT)) - (PopulationCI / PopulationT)) * (1 - p_dth)
                DHR_0 = PopulationCI * p_h / PopulationT * (1 - p_dth)
                DQR_0 = PopulationCI * (1 - p_h) / PopulationT * (1 - p_dth)
                AD_0 = (PopulationCI / (p_d * PopulationT)) - (PopulationCI / PopulationT) * p_dth
                DHD_0 = PopulationCI * p_h / PopulationT * p_dth
                DQD_0 = PopulationCI * (1 - p_h) / PopulationT * p_dth
                R_0 = PopulationR / (PopulationT * p_d)
                D_0 = PopulationD / (PopulationT * p_d)
                TH_0 = PopulationCI * p_h / PopulationT
                DV_0 = PopulationCI * p_h * p_v / PopulationT
                DD_0 = PopulationD / PopulationT
                DT_0 = PopulationI / PopulationT
                x_0_cases = [
                    S_0, E_0, I_0, AR_0, DHR_0, DQR_0,AD_0, DHD_0, DQD_0,
                    R_0, D_0, TH_0, DV_0, DD_0, DT_0
                ]
                x_sol = odeint(
                    func=model_covid,
                    y0=x_0_cases,
                    t=t_cases,
                    args=tuple(params),
                )
                weights = list(range(1,len(fitcasesnd)+1))
                residuals_value = sum(
                    np.multiply((x_sol[:, 14] - fitcasesnd)**2, weights) + balance * np.multiply((x_sol[:, 13] - fitcasesd)**2, weights)
                )
                return residuals_value
    
            print(country +"," + province)
            best_params = minimize(
                residuals_totalcases,
                parameter_list,
                method='L-BFGS-B',  # Can't use Nelder-Mead if I want to put bounds on the params
                bounds=bounds_params
            )
            # TODO 17/04/2020: Not sure what to do once this is done: do we re-solve
            #      Using the best parameters that we found ? I think so but if yes
            #      that's the easiest part of the code :-)