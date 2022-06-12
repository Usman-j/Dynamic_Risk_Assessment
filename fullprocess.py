#%%
import os
import json
from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from apicalls import api_call
import diagnostics
from reporting import score_model_matrix
from ingestion import merge_multiple_dataframe
#%%
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path'])
dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(),config['test_data_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path'])
##################Check and read new data
#first, read ingestedfiles.txt
f_ingested = open(prod_deployment_path+'/ingestedfiles.txt', 'r')
ingested_record = f_ingested.readlines()
f_ingested.close()
ingested_csv_list = [l[:-1] for l in ingested_record[2:]]
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = os.listdir(os.path.join(os.getcwd(),input_folder_path))
source_files = [s for s in source_files if s[-3:]=='csv']
new_data_files = [f for f in source_files if f not in ingested_csv_list]

if len(new_data_files) > 0:
    merge_multiple_dataframe()

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
f_ingested_new = open(dataset_csv_path+'/ingestedfiles.txt', 'r')
new_ingested_record = f_ingested_new.readlines()
f_ingested_new.close()
new_ingested_csv_list = [l[:-1] for l in new_ingested_record[2:]]
new_ingested_files = [f for f in new_ingested_csv_list if f not in ingested_csv_list]

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
model_drifted = False
if len(new_ingested_files) > 0:
    f_score = open(prod_deployment_path+'/latestscore.txt', 'r')
    latest_score = float(f_score.readlines()[0])
    f_score.close()
    curr_score = score_model(model_name=prod_deployment_path+'/trainedmodel.pkl',
                                test_file=dataset_csv_path+'/finaldata.csv')
    if curr_score < latest_score:
        model_drifted = True
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if model_drifted:
    train_model()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if model_drifted:
    store_model_into_pickle()
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
score_model_matrix()
api_call()






