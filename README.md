# Epidemic Model for COVIDAnalytics Research Effort

The repository contains code for a epidemiological model utilized in the research effort of our group called COVIDAnalytics, with projections on the website:

http://www.covidanalytics.io/

This repository contains multiple models, most of them being archived in the `/archive` folder, but also the main and current DELPHI model (as of September 5th 2020) which we call V3.0. Below are the characteristics of each models:
1. DELPHI V3.0 - This is the current model on the website. It is an improvement upon the previous version of the model, as it takes into account interventions being lifted for a little while, causing a resurgence in the number of cases/deaths. This resurgence is modeled using a 3 parameters normal distribution added to our gamma(t) function (cf. `documentation/DELPHI_Explainer_V3.pdf`)
2. Adaptive Policy Model / Continuous Retraining - This is an experimental model we tried implementing that would continuously retrain the parameters associated to the lifting or implementation of a policy in a given area and was supposed to help model a resurgence in cases. It unfortunately didn't yield good enough results and we decided to discard it.
3. V1 - No Jump: Initial DELPHI Model used until late May 2020.
4. V2 - Discrete Jump: This is an attempt at modeling a resurgence in cases using a simple discrete jump (similar to what is done with policy evaluations). This implementation was unsuccessful.
5. V3 - Normal Jump + Trust Solver: Very similar implementation to the final V3.0 version we are currently using (as of September 5th 2020) where we use a trust region constrained solver instead of a TNC solver. The newer implementation actually allows for either so this can be discarded.
6. V4 - Arctan Jump: This is an attempt at modeling a resurgence in cases using an arctan jump. This implementation was unsuccessful.
7. Other analyses: contains a bunch of other analyses and modeling experiments we have conducted, among which other second wave modeling (as opposed to resurgence). They were unsuccessful.

We provide two implementations for the previous V2.0: The (deprecated) Mathematica version
`archive/V1 - No Jump/COVID-19_MIT_ORC_training_script_global_V2.nb`,
and the python3 version `archive/V1 - No Jump/DELPHI_model.py`. The Mathematica notebook was written with Mathematica
12.1 but should work with any version greater than 10.0. The Python3 version is tested with Python 3.7.

The currently used implementation on the website and all other extra analyses for DELPHI V3.0 is in python3 under file
`DELPHI_model_V3.py` and is currently the final version. The policy-based predictions utilize the file
`DELPHI_model_V3_with_policies.py` which uses the outputs from `DELPHI_model_V3.py`.

The latest documentation for the model is contained in the pdf document: `documentation/DELPHI_Explainer_V3.pdf`.

Code created by Michael Lingzhi Li (mlli@mit.edu), Hamza Tazi Bouardi (htazi@mit.edu),
and Omar Skali Lami (oskali@mit.edu).

Please Cite the following when you are utilizing our results:

ML Li, H Tazi Bouardi, O Skali Lami, N Trichakis, T Trikalinos, D Bertsimas. Forecasting COVID-19 and Analyzing the Effect of Government Interventions. (2020) submitted for publication.

## V3.0 How To Run Instructions
### Files Needed
To run the V3.0 model successfully, you would require the following files for each region:
1. Historical Case Files - This should be provided in the same format as the examples given in folder `data/processed`.
2. Population File - This file should record the population at each location that needs to be predicted.
An example of such is in `data/processed/Population_Global.csv`.
3. Historical Parameter Files (optional) - This file record previously trained parameters and the optimization bounds would be within 10% of the original trained parameters. This should be provided in the format given in the example file `predicted/Parameters_Global_20200621.csv`.

### File Paths for Python

To run the model successfully for python, please first add a new user in the `config.yml` file and record the appropriate absolute paths:
- `delphi_repo`: This is the local location for this repo.
- `data_sandbox`: This is the location for policy data used in DELPHI V3.0 (only necessary for DELPHI V3.0).
- `danger_map`: This is the location for saving the final predictions and loading the case files.
- `website`: This is only utilized internally for publishing on the website, and could be ignored.
- `logs`: This is the local file path to the logs folder inside the DELPHI repository.

### Command Line Interface
In order to run the model, run the following command on your terminal: 
`python3 DELPHI_model_V3.py --user <USER> --optimizer <OPTIMIZER> --confidence_intervals <0 or 1> --since100case <0 or 1> --website <0 or 1>` or a shorter
version of it: `python3 DELPHI_model_V3.py -u <USER> -o <OPTIMIZER> -ci <0 or 1> -s100 <0 or 1> -w <0 or 1>`. 

If one wants to run the policy model, the following command should be run on the terminal: 
`python3 DELPHI_model_V3_with_policies.py --user <USER> --optimizer <OPTIMIZER> --website <0 or 1>` or a shorter
version of it: `python3 DELPHI_model_V3_with_policies.py -u <USER> -o <OPTIMIZER> -w <0 or 1>`.

The `USER` must have its file paths referenced in the `config.yml` file, otherwise the script will throw an error.
Similarly, the `OPTIMIZER` must be one of the three currently supported in our implementation (`tnc`, `trust-constr`
or `annealing`), otherwise it will throw an error. It is also important for the policy predictions in order to know
from which optimizer the parameters that will be used will come from. Finally, the `confidence_intervals` parameter must 
be a 0 (for False) or 1 (for True), depending on whether or not the user wants a final output containing confidence 
intervals on the number of cases and deaths (like the ones generated for the website). 
We advise users of this codebase to use 0 as default. Parameter `since100case` allows to save (or not) a prediction file
starting from the date at which each area had its 100th case (varies from one area to another) on top of the file for 
which predictions start on the day of running the script. This is especially useful when one wants to evaluate model 
fitting on historical data. Finally, the `website` parameter allows to choose whether or not to save the prediction and 
parameters files on the `DELPHI/website` repository (default should be 0).

## Backtest How To Run Instructions
Very similarly, to perform a backtest of the model (computing certain metrics on number of cases and number of deaths) one should just use the Command Line Interface running the following command:
`python3 DELPHI_backtest.py --user <USER_RUNNING> --prediction_date <YYYY-MM-DD> --n_days <INTEGER> --mse <0 or 1> --mae <0 or 1>`  or
equivalently `python3 DELPHI_backtest.py -u <USER_RUNNING> -pd <YYYY-MM-DD> -n_days <INTEGER> -mse <0 or 1> -mae <0 or 1>`

The `USER` must have its file paths referenced in the `config.yml` file, otherwise the script will throw an error.
Similarly, the `prediction_date` must have the correct format, otherwise the script will throw an error.
The `n_days` needs to be an integer. The script will automatically check that the backtest is feasible given the available historical
and prediction data in the `danger_map` folder and the two latter inputs from the user running the script.
The flags `mse` and `mae` must be 0 or 1, depending on whether or not the user wants to compute MSE and MAE as well. The default
metric is MAPE and is always computed for both cases and deaths.

## Compare Performance of Annealing and TNC
DELPHI_compare.py can be run to compare the performance of Annealing and TNC at province level using the Command Line Interface in the following way:
`python3 DELPHI_compare.py --user <USER_RUNNING> --run_model <RUN_MODEL> --plots <PLOT_OPTION>`
alternately,
`python3 DELPHI_compare.py -u <USER_RUNNING> -r <RUN_MODEL> -p <PLOT_OPTION>`

As mentioned for other use cases, `USER` should have file paths in the `config.yml` file. If the `run_model` option is 1, the script will run the model with TNC and Annealing consecutively and then compare the metrics, otherwise predictions till current day with annealing and tnc both should be present in the `covid19orc/danger_map/predicted` and the script will automatically read those. If `plot_option` is 1, it will save the plot comparing predictions for every region in the `data_sandbox` folder. The metrics used for default are KL divergence and Maximum Absolute Percentage Error.
