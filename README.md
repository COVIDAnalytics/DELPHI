# Epidemic Model for COVIDAnalytics Research Effort

The repository contains code for a epidemiological model utilized in the research effort of our group called COVIDAnalytics, with projections on the website:

http://www.covidanalytics.io/

This repository contains two separate models, internally coded V2.0 and V3.0:
1. DELPHI V2.0 - This is the current model on the website. It assumes the interventions would continue to slow down the epidemic. 
2. DELPHI V3.0 - This is an experimental model that is currently under evaluation. It takes into account interventions being lifted, causing a resurgence in the number of cases.

We provide two implementations for V2.0: The (deprecated) Mathematica version `COVID-19_MIT_ORC_training_script_global_V2.nb` in folder `old_mathematica_scripts`, and the python3 version `DELPHI_model.py`. The Mathematica notebook was written with Mathematica 12.1 but should work with any version greater than 10.0. The Python3 version is tested with Python 3.7. 

The only implementation for DELPHI V3.0 is in python3 named `DELPHI_model_cr_with_policies.py` and is currently experimental. Utilize this model at your own risk.

Documentation is contained in the pdf document: DELPHI_Explainer.

Code created by Michael Lingzhi Li, Hamza Tazi Bouardi, and Omar Skali Lami.

Please Cite the following when you are utilizing our results:

ML Li, H Tazi Bouardi, O Skali Lami, N Trichakis, T Trikalinos, D Bertsimas. Forecasting COVID-19 and Analyzing the Effect of Government Interventions. (2020) submitted for publication.


### Path File for Python

To run the model successfully for python, please first add a new user in the `config.yml` file and record the appropriate absolute paths:

- `delphi_repo`: This is the local location for this repo. 
- `data_sandbox`: This is the location for policy data used in DELPHI V3.0 (only necessary for DELPHI V3.0).
- `danger_map`: This is the location for saving the final predictions and loading the case files. 
- `website`: This is only utilized internally for publishing on the website, and could be ignored.
