# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu)
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from datetime import datetime, timedelta
yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
# TODO: Find a way to make this path automatic, whoever the user is...
PATH_TO_FOLDER = (
    "/Users/hamzatazi/Desktop/MIT/999.1 Research Assistantship/" +
    "4. COVID19_Global/covid19orc/danger_map"
)
popstates = pd.read_csv(
    f"{PATH_TO_FOLDER}/processed/Population2019_AllStates.csv"
)
pastparameters = pd.read_csv(
    f"{PATH_TO_FOLDER}/predicted/Parameters_Allstates_{yesterday}.csv"
)
for state in popstates.state.unique():
    totalcases = pd.read_csv(
        f"{PATH_TO_FOLDER}/processed_Cases_{state}.csv"
    )
    # We take 2: because we don't need "State" nor "date"
    # Since we have the column "day_since100" in totalcases
    parameter_list = pastparameters[
                         pastparameters.State == state
                      ].iloc[-1, 2:].values.tolist()
    if len(parameter_list) > 0:
        # Allowing a 5% drift for states with past predictions
        param_list_lower = [x - 0.05 * x for x in parameter_list]
        param_list_upper = [x + 0.05 * x for x in parameter_list]
        bounds_params = tuple(
            [(lower, upper)
             for lower, upper in zip(param_list_lower, param_list_upper)]
        )
        # TODO 17/04/2020: Check that format is right for validcases
        validcases = totalcases[
            totalcases.day_since100 >= 0
        ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
    else:
        # Otherwise use established lower/upper bounds
        bounds_params = (
            (0.75, 1.25), (-10, 10), (1, 3), (0.01, 0.1), (0.1, 10), (0.1, 10)
        )
        validcases = totalcases[
            totalcases.day_since100 >= 0
        ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)

    # Now we start the modeling part:
    if len(validcases) > 7:
        IncubeD = 5
        RecoverID = 10
        DetectD = 2
        PopulationT = popstates[
            popstates.state == state
        ].pop2019.item()
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
        balance = validcases.loc[-1, "case_cnt"] / validcases.loc[-1, "death_cnt"] / 5

        """ Fit on Total Cases """
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
                x: list, t: list, parameter_list:list,
                # Will have to put the parameters that are fitted here
        ):
            """
            SEIR + Undetected, Deaths, Hospitalized, corrected with ArcTan response curve
            alpha: Infection rate
            days:
            r_s:
            p_dth: Mortality rate
            k1: Internal parameter 1
            k2: Internal parameter 2
            """
            alpha, days, r_s, p_dth, k1, k2 = parameter_list
            r_i = np.log(2)/IncubeD  # Rate of infection leaving incubation phase
            r_d = np.log(2)/DetectD  # Rate of death leaving incubation phase TODO 17/04/2020: Verify ?
            r_ri = np.log(2)/RecoverID  # Rate of recovery not under infection
            r_rh = np.log(2)/RecoverHD  # Rate of recovery under hospitalization
            r_rv = np.log(2)/VentilatedD  # Rate of recovery under ventilation
            r_dth = -np.log(1 - DeathP)/RecoverHD  # Rate of death
            gamma_t = (2/np.pi) * np.arctan(-(t - days) / r_s) + 1
            # TODO: Don't know how to deal with this gamma to take t as a parameter
            #       Or maybe use it as a parameter in odeint, no idea really
            assert len(x) == 12, f"Too many input variables, got {len(x)}, expected 12"
            S, E, I, A, DH, DQ, R, D, TH, DV, DD, DT = x
            # Equations on main variables
            dSdt = -alpha * gamma_t * S * I
            dEdt = -alpha * gamma_t * S * I - r_i * E
            dIdt = r_i * E - r_d * I
            dAdt = r_d * (1-p_d) * I - (r_ri + r_dth) * A
            dDHdt = r_d * p_d * p_h * I - (r_rh + r_dth) * DH
            dDQdt = r_d * p_d * (1 - p_h) * I - (r_ri + r_dth) * DQ
            dRdt = r_ri * (A + DQ) + r_rh * DH
            dDdt = r_dth * (A + DQ + DH)

            # Helper states (not in the ODEs but important for fitting)
            dTHdt = r_d * p_d * p_h * I
            dDVdt = r_d * p_d * p_h * p_v * I - (r_rv + r_dth) * DV
            dDDdt = r_dth * (DH + DQ)
            dDTdt = r_d * p_d * I
            return [
                dSdt, dEdt, dIdt, dAdt, dDHdt, dDQdt,
                dRdt, dDdt, dTHdt, dDVdt, dDDdt, dDTdt
            ]


        # Initialization of parameters to be fitted (from previous fits)
        alpha, days, r_s, DeathP, k1, k2 = parameter_list
        # Variables Initialization for the ODE system
        S_0 = (
                (1 - PopulationCI/(p_d * PopulationT)) -
                (PopulationCI / (p_d * PopulationT) * (k1+k2)) -
                (PopulationR / (p_d * PopulationT)) -
                (PopulationD / (p_d * PopulationT))
        )
        E_0 = PopulationCI / (p_d * PopulationT) * k1
        I_0 = PopulationCI / (p_d * PopulationT) * k2
        A_0 = (PopulationCI / (p_d * PopulationT)) - (PopulationCI / PopulationT)
        DH_0 = PopulationCI * p_h / PopulationT
        DQ_0 = PopulationCI * (1 - p_h) / PopulationT
        R_0 = PopulationR / (PopulationT * p_d)
        D_0 = PopulationD / (PopulationT * p_d)
        TH_0 = PopulationCI * p_h / PopulationT
        DV_0 = PopulationCI * p_h * p_v / PopulationT
        DD_0 = PopulationD / PopulationT
        DT_0 = PopulationI / PopulationT
        x_0_cases = [
            S_0, E_0, I_0, A_0, DH_0, DQ_0,
            R_0, D_0, TH_0, DV_0, DD_0, DT_0,
        ]
        # TODO 17/04/2020: Define the time mesh (is it the number of examples on which we'll fit later ?)
        # TODO 17/04/2020: What is the dataset/format we'll be using (fitcasesnd ?) ? Technically, we only need the list of case_cnt
        #                  to solve with minimize (check residuals_totalcases() )
        n_points = len(fitcasesnd)
        t_cases = np.linspace(1, n_points, n_points)


        def residuals_totalcases(params):
            """
            Wanted to start with odeint because figures will be faster to debug
            """
            x_sol = odeint(
                func=model_covid,
                y0=x_0_cases,
                t=t_cases,
                args=params,
            )
            residuals_value = sum(
                (x_sol[:, -1] - fitcasesnd)**2  # TODO 17/04/2020: Figure out which data to use, only need a list
            )
            return residuals_value


        best_params = minimize(
            residuals_totalcases,
            parameter_list,
            method='L-BFGS-B',  # Can't use Nelder-Mead if I want to put bounds on the params
            bounds=bounds_params
        )
        # TODO 17/04/2020: Not sure what to do once this is done: do we re-solve
        #      Using the best parameters that we found ? I think so but if yes
        #      that's the easiest part of the code :-)

        # TODO 17/04/2020: Also saw that you weighted the recent data more,
        #      don't know how to do that!