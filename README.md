# Epidemic Model for COVIDAnalytics Research Effort

The repository contains code for a epidemiological model utilized in the research effort of our group called COVIDAnalytics, with projections on the website:

http://www.covidanalytics.io/

**Note: The model was significantly updated on July 4th on the website. On July 18th, we have updated both the code and the naming convention for our models on the master branch.**

The root directory contains the main files for our newest model (Updated 0704). The latest model takes into account a resurgence in cases in different places using an exponential type correction. The model is named V3 and the main model file is in `DELPHI_model_V3.py`. The supporting files are `DELPHI_utils_V3.py` and  `DELPHI_params_V3.py`. The code is tested with Python 3.7. 

The original model (originally codenamed "DELPHI V2.0") is now archived and renamed V1 under the folder `archive/V1 - No Jump`. Both the Mathematica version `COVID-19_MIT_ORC_training_script_global_V2.nb` and the python3 version  `DELPHI_model.py` are contained in this subfolder.

The documentation for the newest model is at "documentation/DELPHI_Explainer_V3.pdf". The documentation for the original model on the website is at "documentation/DELPHI_Explainer.pdf".

Code created by Michael Lingzhi Li, Hamza Tazi Bouardi, and Omar Skali Lami.

Please Cite the following when you are utilizing our results:

ML Li, H Tazi Bouardi, O Skali Lami, N Trichakis, T Trikalinos, D Bertsimas. Forecasting COVID-19 and Analyzing the Effect of Government Interventions. (2020) submitted for publication.

## FAQ

#### 1. The results that I produced don't match the results shown on the COVIDAnalytics website.

This is usually due to the difference in historical parameter files. If there are no historical parameter files, the model would run with default bounds which are unsuitable for a lot of countries. Sample parameters for a recent run of the model are included in the "predicted" subfolder and should be used in order to replicate the results on our website. 


#### 2. The projections produced for a certain country are lower than the latest true values known. Why is this the case, and how can I fix it?

This is due to the fact that **our model fits on cumulative cases/deaths rather than daily cases/deaths**. Although we use many tactics to ensure that it has a good fit on the recent data, it cannot not have a *perfect* fit on the latest true values. That means that if we know that on July 1st a country had 100 deaths, the projections might say it had 98 deaths. This is usually not a problem for most countries as the error is usually small (<<1%), but for some countries it could be appreciable.

To fix this (and what we utilize in our submission to CDC), you could additively shift our projections so that the projections match exactly the historical data on the latest known date. So for example if our model outputs 98 deaths on the date with latest known figures, and the true value is 100, one could additively shift all our projections upward by 2 so that it would match the last day perfectly. This would NOT change our predictions for the number of daily deaths, and thus retains our core predictive power while resolving this issue. 

## V3 Instructions (Current Model)

To run the V3 model successfully, you would require the following files for each region (in addition to configuring the path file correctly in the bottom section):
1. Historical Case Files - This should be provided in the same format as the examples given in folder `data/processed`.
2. Population File - This file should record the population at each location that needs to be predicted. An example of such is in `data/processed/Population_Global.csv`.
3. Historical Parameter Files (optional) - This file record previously trained parameters and the optimization bounds would be within 10% of the original trained parameters. This should be provided in the format given in the example file `predicted/Parameters_Global_V3_20200718.csv`.

## V1 (Old "DELPHI V2.0") Instructions (Deprecated)

To run the model successfully, you would require the following files for each region (in addition to configuring the path file correctly in the bottom section):
1. Historical Case Files - This should be provided in the same format as the examples given in folder `data/processed`.
2. Population File - This file should record the population at each location that needs to be predicted. An example of such is in `data/processed/Population_Global.csv`.
3. Historical Parameter Files (optional) - This file record previously trained parameters and the optimization bounds would be within 10% of the original trained parameters. This should be provided in the format given in the example file `predicted/Parameters_Global_20200621.csv`.

## Path File for Python

To run the model successfully for python, please first add a new user in the `config.yml` file and record the appropriate absolute paths:

- `delphi_repo`: This is the local location for this repo. 
- `data_sandbox`: This is the location for policy data used in DELPHI V3.0 (only necessary for DELPHI V3.0).
- `danger_map`: This is the location for saving the final predictions and loading the case files. 
- `website`: This is only utilized internally for publishing on the website, and could be ignored.

## Archived Models

The  `archive` subfolder contains various other models that we attempted but did not find its performance satisfying. In total the archive includes:
1. "Adaptive Policy Model" - This is a model that utilizes the current implemented policies to derive the resurgence continuously over time. Although the methodology is sound, the results were unfortunately not as appealing as the current model utilized on the website. 
2. "V1 - No Jump" - This is the original model used for the website before 07/04. We provide two implementations: The Mathematica version `COVID-19_MIT_ORC_training_script_global_V2.nb` , and the python3 version `DELPHI_model.py`. The Mathematica notebook was written with Mathematica 12.1 but should work with any version greater than 10.0. The Python3 version is tested with Python 3.7. 
3. "V2 - Discrete Jump" - This is the original model augmented with a resurgence modeled as a discrete step function.
4. "V3 - Normal Jump + Trust Solver" - This is the same as the current model on the website, but uses a different solver that produces good results sometimes, but fails spectacularly on a lot of countries.
5. "V4 - ArcTan Jump" - This is the original model augmented with a resurgence modeled as a second ArcTan function. The results were acceptable but not as good as the exponential correction. 
