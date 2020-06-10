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
default_maxT = datetime(2020, 7, 15)  # Maximum timespan of prediction
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.15  # Percentage of detected cases hospitalized

# Policies and future times for counterfactual predictions
future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]

future_policies_JJ = [
    'No_Measure', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools', 'Lockdown'
]
default_maxT_policies = datetime(2020, 9, 15)  # Maximum timespan of prediction under different policy scenarios
future_times = [0, 7, 14, 28, 42]
future_times_JJ = [0]

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
provinces_Brazil = [
    'Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara', 'Distrito Federal', 'Espirito Santo', 'Goias',
    'Maranhao', 'MatoGrosso', 'MatoGrosso do Sul', 'Minas Gerais', 'Para', 'Paraiba', 'Parana', 'Pernambuco',
    'Piaui', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondonia', 'Roraima', 'Santa Catarina',
    'Sao Paulo', 'Sergipe', 'Tocantins', 'Espiritu Santo', 'Mato Grosso', 'Mato Grosso do Sul'
]
provinces_Peru = [
    'Amazonas', 'Ancash', 'Apurimac', 'Arequipa', 'Ayacucho', 'Cajamarca', 'Cusco', 'Callao', 'Huancavelica',
    'Huanuco', 'Ica', 'Junin', 'La Libertad', 'Lambayeque', 'Lima', 'Loreto', 'Madre de dios', 'Moquegua',
    'Pasco', 'Piura', 'Puno', 'San Martin', 'Tacna', 'Tumbes', 'Ucayali'
]
provinces_South_Africa = [
    'Eastern Cape', 'Free State', 'Gauteng', 'KwaZulu Natal', 'Limpopo',
    'Mpumalanga', 'Northern Cape', 'North West', 'Western Cape'
]

provinces_Russia = [
    'Altayskiy Kray', 'Amursk Oblast', 'Arkhangelsk Oblast',
    'Astrahan Oblast', 'Belgorod Oblast', 'Briansk Oblast',
    'Chechen Republic', 'Cheliabinsk Oblast',
    'Chukotskiy Autonomous Oblast', 'Habarovskiy Kray',
    'Hanty-Mansiyskiy AO', 'Ingushetia Republic', 'Irkutsk Oblast',
    'Ivanovo Oblast', 'Jewish Autonomous Oblast', 'Kaliningrad Oblast',
    'Kaluga Oblast', 'Kamchatskiy Kray', 'Kemerovo Oblast',
    'Kirov Oblast', 'Komi Republic', 'Kostroma Oblast',
    'Krasnodarskiy Kray', 'Krasnoyarskiy Kray', 'Kurgan Oblast',
    'Kursk Oblast', 'Leningradskaya Oblast', 'Lipetsk Oblast',
    'Magadan Oblast', 'Moscow', 'Moscow Oblast', 'Murmansk Oblast',
    'Nenetskiy Autonomous Oblast', 'Nizhegorodskaya Oblast',
    'Novgorod Oblast', 'Novosibirsk Oblast', 'Omsk Oblast',
    'Orel Oblast', 'Orenburg Oblast', 'Pensa Oblast', 'Perm Oblast',
    'Primorskiy Kray', 'Pskov Oblast', 'Republic of Adygeia',
    'Republic of Bashkortostan', 'Republic of Buriatia',
    'Republic of Chuvashia', 'Republic of Dagestan',
    'Republic of Hakassia', 'Republic of Kabardino-Balkaria',
    'Republic of Kalmykia', 'Republic of Karachaevo-Cherkessia',
    'Republic of Karelia', 'Republic of Mariy El',
    'Republic of Mordovia', 'Republic of North Osetia-Alania',
    'Republic of Tatarstan', 'Republic of Tyva',
    'Republic of Udmurtia', 'Rostov Oblast', 'Ryazan Oblast',
    'Saha Republic', 'Saint Petersburg', 'Sakhalin Oblast',
    'Samara Oblast', 'Saratov Oblast', 'Smolensk Oblast',
    'Stavropolskiy Kray', 'Sverdlov Oblast', 'Tambov Oblast',
    'Tomsk Oblast', 'Tula Oblast', 'Tumen Oblast', 'Tver Oblast',
    'Ulianovsk Oblast', 'Vladimir Oblast', 'Volgograd Oblast',
    'Vologda Oblast', 'Voronezh Oblast', 'Yamalo-Nenetskiy AO',
    'Yaroslavl Oblast', 'Zabaykalskiy Kray'
]

provinces_Chile = [
    'Antofagasta', 'Araucania', 'Arica y Parinacota', 'Atacama',
     'Aysen', 'Biobio', 'Coquimbo', 'Los Lagos', 'Los Rios',
     'Magallanes', 'Maule', 'Nuble', "O'Higgins", 'Santiago',
     'Tarapaca', 'Valparaiso'
]

provinces_Mexico = [
    'Aguascalientes', 'Baja California', 'Baja California Sur',
 'Campeche', 'Chiapas', 'Chihuahua', 'Ciudad de Mexico', 'Coahuila',
 'Colima', 'Durango', 'Guanajuato', 'Guerrero', 'Hidalgo',
 'Jalisco', 'Mexico', 'Michoacan', 'Morelos', 'Nayarit',
 'Nuevo Leon', 'Oaxaca', 'Puebla', 'Queretaro', 'Quintana Roo',
 'San Luis Potosi', 'Sinaloa', 'Sonora', 'Tabasco', 'Tamaulipas',
 'Tlaxcala', 'Veracruz', 'Yucatan', 'Zacatecas'
]