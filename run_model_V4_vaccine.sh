git config --global user.email "ali@dynideas.com"; git config --global user.name "Server" ;
cd ..;cd ..;cd /repos/DELPHI
#python DELPHI_model_V3_predict_additional_state.py -d "2020-07-01" -t 'global'
python DELPHI_model_V4.py -rc run_configs/annealing-run-config.yml

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'global_vaccine'

git pull
git add data_sandbox/predicted/policy_scenario_predictions/*
git add logs/*
git -c user.name='Server' -c user.email=ali@dynideas.com commit -m "With vaccies US State results"
git push https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/DELPHI.git --all

