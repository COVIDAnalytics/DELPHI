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
default_maxT_policies = datetime(2020, 11, 15)  # Maximum timespan of prediction under different policy scenarios
future_policies_JJ = [
    'No_Measure', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools', 'Lockdown'
]

future_times = [0, 7, 14, 28, 42]
future_times_JJ = [0]

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

provinces_Russia = ['Altai Republic',
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

provinces_Colombia =[
    'Amazonas', 'Antioquia', 'Arauca', 'Atlantico', 'Bogota',
                     'Bolivar', 'Boyaca', 'Caldas', 'Caqueta', 'Casanare', 'Cauca',
                     'Cesar', 'Choco', 'Cordoba', 'Cundinamarca', 'Guainia', 'Guaviare',
                     'Huila', 'La Guajira', 'Magdalena', 'Meta', 'Narino',
                     'Norte de Santander', 'Putumayo', 'Quindio', 'Risaralda',
                     'San Andres y Providencia', 'Santander', 'Sucre', 'Tolima',
                     'Valle del Cauca', 'Vaupes', 'Vichada'
]

provinces_Spain = ['Andalucía', 'Aragón', 'Asturias', 'Cantabria', 'Ceuta',
     'Castilla y León', 'Castilla-La Mancha', 'Islas Canarias',
     'Cataluña', 'Extremadura', 'Galicia', 'Islas Baleares',
     'Region de Murcia', 'Comunidad de Madrid', 'Melilla', 'Navarra',
     'País Vasco', 'La Rioja', 'Comunidad Valenciana']

provinces_Argentina =['Salta', 'Buenos Aires Province', 'City of Buenos Aires',
    'San Luis', 'Entre Ríos', 'La Rioja', 'Santiago del Estero',
    'Chaco', 'San Juan', 'Catamarca', 'La Pampa', 'Mendoza',
    'Misiones', 'Formosa', 'Neuquén', 'Río Negro', 'Santa Fe',
    'Tucumán', 'Chubut', 'Tierra del Fuego', 'Corrientes', 'Córdoba',
    'Jujuy', 'Santa Cruz']

provinces_Italy =[
    'Piedmont', 'Aosta Valley', 'Lombardy', 'Veneto',
    'Friuli Venezia Giulia', 'Liguria', 'Emilia-Romagna', 'Tuscany',
    'Umbria', 'Marche', 'Lazio', 'Abruzzo', 'Molise', 'Campania',
    'Apulia', 'Basilicata', 'Calabria', 'Sicily', 'Sardinia',
    'South Tyrol', 'Trentino-Alto Adige']
