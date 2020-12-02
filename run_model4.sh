python archive/V3\ Model/DELPHI_model_V3_predict_additional_state.py -d "2020-07-01" -t 'USExUS'
#python archive/V3\ Model/DELPHI_model_V3_predict_additional_state.py -d "2020-07-01" -t 'global'

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'ExUS'
#python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'US'

git config --global user.email "ali@dynideas.com"
git config --global user.name "Server"
git pull
git add data_sandbox/predicted/parameters/*
git add data_sandbox/predicted/policy_scenario_predictions/*
git add danger_map/predicted/*
git -c user.name='Server' -c user.email=ali@dynideas.com commit -m "Ex US results"
git push https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/DELPHI.git --all

