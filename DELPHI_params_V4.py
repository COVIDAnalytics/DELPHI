# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
from datetime import datetime
from datetime import date

# Default parameters - TNC & Trust Region
date_MATHEMATICA = "2020-05-07"  # Transition date from Mathematica to Python
fitting_start_date = "2021-01-01" # date to start model fitting from. predictions of all model states on this date are needed
default_parameter_list = [1, 0, 2, 0.2, 0.05, 0.2, 3, 3, 0.1, 3, 1, 1] # Default parameters for the solver
dict_default_reinit_parameters = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0.001, "r_dthdecay": -0.2,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "k3": 0,
}  # Allows for reinitialization of parameters in case they reach a value that is too low/high
dict_default_reinit_lower_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0.001, "r_dthdecay": -0.2,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "k3": 0,
}  # Allows for reinitialization of lower bounds in case they reach a value that is too low
dict_default_reinit_upper_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0.001, "r_dthdecay": -0.2,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": -100, "std_normal": 1, "k3": 0,
}  # Allows for reinitialization of upper bounds in case they reach a value that is too high
default_upper_bound = 0.2
percentage_drift_upper_bound = 0.2
default_lower_bound = 0.2
percentage_drift_lower_bound = 0.2
default_bounds_params = (
    (0.1, 1.5), (-200, 100), (1, 15), (0.02, 0.5), (0.01, 0.25), (-0.2, 5.0), (0.001, 5), (0.001, 5), (0, 5), (0, 300), (0.1, 100), (0.2,2.0)
)   # Bounds for the solver
validcases_threshold = 7  # Minimum number of cases to fit the base-DELPHI
validcases_threshold_policy = 15  # Minimum number of cases to train the country-level policy predictions
max_iter = 500  # Maximum number of iterations for the algorithm

# Default parameters - Annealing
percentage_drift_upper_bound_annealing = 1
default_upper_bound_annealing = 1
percentage_drift_lower_bound_annealing = 1
default_lower_bound_annealing = 1
default_lower_bound_jump = 0
default_upper_bound_jump = 5
default_lower_bound_t_jump = 0
default_upper_bound_t_jump = (date.today() - datetime.strptime(fitting_start_date,"%Y-%m-%d").date()).days + 10
default_lower_bound_std_normal = 1
default_upper_bound_std_normal = 200

# Initial condition of exposed state and infected state
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated
default_maxT = datetime(2021, 7, 15)  # Maximum timespan of prediction
n_params_without_policy_params = 7  # alpha, r_dth, p_dth, a, b, k1, k2
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.03  # Percentage of detected cases hospitalized

# Policies and future times for counterfactual predictions
future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]
default_maxT_policies = datetime(2021, 3, 15) # Maximum timespan of prediction under different policy scenarios
future_times = [0, 7, 14, 28, 42]

# Default normalized gamma shifts from runs in May 2020
default_dict_normalized_policy_gamma = {
    'No_Measure': 1.0,
    'Restrict_Mass_Gatherings': 0.873,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.794,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 0.668,
    'Restrict_Mass_Gatherings_and_Schools': 0.479,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.423,
    'Lockdown': 0.239
}


# Additional utils inputs
TIME_DICT = {0: "Now", 7: "One Week", 14: "Two Weeks", 28: "Four Weeks", 42: "Six Weeks"}
MAPPING_STATE_CODE_TO_STATE_NAME ={
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan',
    'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
    'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', "AS": "American Samoa",
    "GU": "Guam", "MP": "Northern Marianas", "PR": "Puerto Rico", "VI": "Virgin Islands"
}
default_policy = "Lockdown"  # Eventually change to future_policies[-1]
default_policy_enaction_time = 'Now'  # Eventually change to TIME_DICT[0]
