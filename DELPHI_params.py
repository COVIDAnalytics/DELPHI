# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu)
from datetime import datetime

# Default parameters
date_MATHEMATICA = "2020-05-07"  # Transition date from Mathematica to Python
default_parameter_list = [1, 0, 2, 0.2, 0.05, 3, 3] # Default parameters for the solver
default_bounds_params = (
                    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0.1, 10), (0.1, 10)
                ) # Bounds for the solver
validcases_threshold = 7  # Minimum number of cases to fit the base-DELPHI
validcases_threshold_policy = 15  # Minimum number of cases to train the country-level policy predictions
max_iter = 1000  # Maximum number of iterations for the algorithm

# Initial condition of exposed state and infected state
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated
default_maxT = datetime(2021, 4, 1)  # Maximum timespan of prediction
n_params_without_policy_params = 7  # alpha, r_dth, p_dth, a, b, k1, k2
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.15  # Percentage of detected cases hospitalized

# Policies and future times for counterfactual predictions
future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]
default_maxT_policies = datetime(2021, 1, 15)  # Maximum timespan of prediction under different policy scenarios
future_policies_JJ = [
    'No_Measure', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools', 'Lockdown', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools_and_Others','Restrict_Mass_Gatherings'
]

future_times = [0, 7, 14, 28, 42]
future_times_JJ = [0, 7, 14, 28, 42]

# Additional utils inputs
TIME_DICT = {0: "Now", 7: "One Week", 14: "Two Weeks", 28: "Four Weeks", 42: "Six Weeks"}
DAYS_IN_WEEK = 7

# Function to find year, week, days
def translate_days(number_of_days):

    # Assume that years is
    # of 365 days
    year = int(number_of_days / 365)
    week = int((number_of_days % 365) /
                DAYS_IN_WEEK)
    days = (number_of_days % 365) % DAYS_IN_WEEK

    if year + week + days > 0:
        return((str(year) + ' Year' + ('s' if year != 1 else '') + (' and ' if week !=0 or days != 0 else '') if year != 0 else '') + (str(week) + ' Week' + ('s' if week != 1 else '') + (' and ' if days != 0 else '') if week != 0 else '') + (str(days) + ' Day' + ('s' if days != 1 else '') if days != 0 else ''))
    else:
        return('Now')

# TIME_DICT = {i: translate_days(i) for i in range(400)}

TIME_DICT = {0: "Now", 7: "One Week", 14: "Two Weeks", 28: "Four Weeks", 42: "Six Weeks"}

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


city_policies = {'Savannah_Hinesville_Statesboro_GA':'Georgia',
                 'Reno_Carson_City_Fernley_NV': 'Nevada',
                 'Columbia_Orangeburg_Newberry_SC': 'South Carolina',
                 'Indianapolis_Carmel_Muncie_IN':'Indiana',
                 'Charleston_North_Charleston_SC': 'South Carolina',
                 'Louisville_Jefferson_County_Elizabethtown_Bardstown_KY_IN':'Kentucky',
                 'Greenville_Spartanburg_Anderson_SC':'South Carolina',
                 'Fayetteville_Sanford_Lumberton_NC':'North Carolina',
                 'Kansas_City_Overland_Park_Kansas_City_MO_KS':'Missouri',
                 'Lexington_Fayette_Richmond_Frankfort_KY':'Kentucky',
                 'Medford_Grants_Pass_OR':'Oregon',
                 'Johnson_City_Kingsport_Bristol_TN_VA':'Tennessee',
                 'Gainesville_Lake_City_FL':'Florida',
                 'Greensboro_Winston_Salem_High_Point_NC':'North Carolina',
                 'Charlotte_Concord_NC_SC':'North Carolina',
                 'Little_Rock_North_Little_Rock_AR':'Arkansas',
                 'Jonesboro_Paragould_AR':'Arkansas',
                 'Lynchburg_VA':'Virginia',
                 'Memphis_Forrest_City_TN_MS_AR':'Tennessee',
                 'Richmond_VA':'Virginia',
                 'Jackson_Vicksburg_Brookhaven_MS':'Mississippi',
                 'Virginia_Beach_Norfolk_VA_NC':'Virginia',
                 'Salt_Lake_City_Provo_Orem_UT':'Utah',
                 'Denver_Aurora_CO':'Colorado',
                 'Hartford_East_Hartford_CT':'Connecticut',
                 'Albuquerque_Santa_Fe_Las_Vegas_NM':'New Hampshire',
                 'Fargo_Wahpeton_ND_MN':'North Dakota',
                 'Knoxville_TN':'Tennessee',
                 'Portland_Vancouver_Salem_OR_WA':'Oregon'
                 }
