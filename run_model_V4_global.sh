git config --global user.email "ali@dynideas.com"
git config --global user.name "Server"
cd ..;cd ..;cd mnt
git clone https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/covid19orc.git
cd covid19orc/
git checkout danger_map
cd ..
'cp' -rf covid19orc/danger_map/processed/* /repos/DELPHI/danger_map/processed/
'cp' -rf covid19orc/danger_map/predicted/* /repos/DELPHI/danger_map/predicted/
rm -rf covid19orc;cd ..;cd /repos/DELPHI
#python DELPHI_model_V3_predict_additional_state.py -d "2020-10-01" -t 'global'

python DELPHI_model_V4_additional_states.py -rc run_configs/annealing-run-config.yml -t 'global'

git pull
git add danger_map/processed/*
git add danger_map/predicted/*
git add data_sandbox/predicted/policy_scenario_predictions/*
git add logs/*
git -c user.name='Server' -c user.email=ali@dynideas.com commit -m "Global results"
git push https://"$GIT_USER":"$GIT_PASS"@github.com/COVIDAnalytics/DELPHI.git --all

