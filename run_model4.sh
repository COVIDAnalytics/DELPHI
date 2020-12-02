python archive/V3\ Model/DELPHI_model_V3_predict_additional_state.py -d "2020-07-01" -t 'USExUS'
#python archive/V3\ Model/DELPHI_model_V3_predict_additional_state.py -d "2020-07-01" -t 'global'

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'ExUS'
#python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'US'

git pull
git add data_sandbox/predicted/parameters/*
git add data_sandbox/predicted/policy_scenario_predictions/*
git commit -m "Ex US results" --author "Server <ali@dynideas.com"
git push https://$GIT_USER:$GIT_PASS@github.com/COVIDAnalytics/DELPHI.git --all

