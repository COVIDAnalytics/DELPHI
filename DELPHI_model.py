import numpy as np
from scipy.integrate import odeint

def model_covid(x, t):
    alpha = 0  # Infection rate
    r_i = 0  # Rate of infection leaving incubation phase
    r_d = 0
    r_ri = 0  # Rate of recovery not under hospitalization
    r_rih = 0  # Rate of recovery under hospitalization
    r_dth = 0  # Rate of death
    p_d = 0  # Percentage of infection cases detected.
    p_h = 0  # Percentage of detected cases hospitalized
    p_dth = 0  # Mortality rate
    a = 0  # parameter for arctan
    b = 0  # parameter for arctan
    def gamma(a, b, t):
        value = (2/np.pi) * np.arctan(-(t-a)/b) + 1
        return value

    S, E, I, A, DH, DQ, R, D = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    dSdt = -alpha * gamma(a, b, t) * S * I
    dEdt = -alpha * gamma(a, b, t) * S * I - r_i * E
    dIdt = r_i * E - r_d * I
    dAdt = r_d * (1-p_d) * I - (r_ri + r_dth) * A
    dDHdt = r_d * p_d * p_h * I - (r_rih + r_dth) * DH
    dDQdt = r_d * p_d * (1 - p_h) * I - (r_ri + r_dth) * DQ
    dRdt = r_ri * (A + DQ) + r_rih * DH
    dDdt = r_dth * (A + DQ + DH)

    return [dSdt, dEdt, dIdt, dAdt, dDHdt, dDQdt, dRdt, dDdt]

# Initialization
x_0_cases = []
t_cases = np.linspace(1, 100, 100000)
x_cases = odeint(
    func=model_covid,
    y0=x_0_cases,
    t=t_cases,
)