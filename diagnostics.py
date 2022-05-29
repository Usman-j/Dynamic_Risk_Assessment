
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
import subprocess
from training import train_model
from ingestion import merge_multiple_dataframe
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path'])
dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(),config['test_data_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path'])
feat_list = config['feat_list']

##################Function to get model predictions
def model_predictions(df_test, model_name='trainedmodel.pkl'):
    '''
    Performs model predictions for the input dataframe 
    Inputs:
    df_test: pd.DataFrame
        Test dataset
    model_name: str
        Name of model for predicting
    Outputs:
    predicted: list
        List containing model predictions for each row/sample in input dataframe
    '''
    
    with open(prod_deployment_path+'/'+model_name, 'rb') as file:
        model = pickle.load(file)
    X_test = df_test.loc[:,feat_list].values.reshape(-1, len(feat_list))
    # y_test = df_test[target_var].values.reshape(-1, 1).ravel()
    predicted = model.predict(X_test)
    return list(predicted)

##################Function to get summary statistics
def dataframe_summary(df_data):
    '''
    Calculates mean, median and std for each numeric column of input dataframe
    Inputs:
    df_data: pd.DataFrame
        Input dataset
    Outputs:
    stats_list: list
        List containing mean, median and std for each numeric column of input dataframe
    '''
    #calculate summary statistics here
    stats_list = []
    stats_list.extend(df_data[feat_list].mean().values)
    stats_list.extend(df_data[feat_list].median().values)
    stats_list.extend(df_data[feat_list].std().values)
    return stats_list

##################Function to get missing data proportions
def check_missing_data(df_data):
    '''
    Calculates percentage of missing values in each column of input dataframe
    Inputs:
    df_data: pd.DataFrame
        Input dataset
    Outputs:
    napercents: list
        List containing percentage of missing values in each column of input dataframe
    '''
    
    nas=list(df_data.isna().sum())
    napercents=[(nas[i]/len(df_data))*100 for i in range(len(nas))]
    return napercents

##################Function to get timings
def execution_time():
    '''
    Calculates timing of functions defined in training.py and ingestion.py
    Outputs:
    exec_times: list
        List containing time values (in seconds) for ingestion and training
    '''
    
    exec_times = []
    starttime = timeit.default_timer()
    merge_multiple_dataframe()
    exec_times.append(timeit.default_timer() - starttime)
    starttime = timeit.default_timer()
    train_model()
    exec_times.append(timeit.default_timer() - starttime)
    return exec_times

##################Function to check dependencies
def outdated_packages_list():
    '''
    Gets a list of installed and latest versions of all packages. 
    Outputs:
    df_output: pd.DataFrame
        Dataframe containing package names with their respective installed and latest versions.
    '''
    
    installed = subprocess.check_output(['pip', 'list','--outdated'], encoding='utf-8')
    str_output = installed.splitlines()
    pkg_name, curr_v, latest_v = [], [], []
    for line in str_output[2:]: #skippong header line
        fields = line.split()
        pkg_name.append(fields[0])
        curr_v.append(fields[1])
        latest_v.append(fields[2])

    df_output = pd.DataFrame({'Package Name':pkg_name, 'Installed Version':curr_v, 'Latest Version':latest_v})
    return df_output

if __name__ == '__main__':
    df_test = pd.read_csv(test_data_path+'/testdata.csv')
    y_pred = model_predictions(df_test)
    print(len(df_test) == len(y_pred))
    df_data = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    stats_list = dataframe_summary(df_data)
    print(stats_list)
    napercents = check_missing_data(df_data)
    print(napercents)
    exec_times = execution_time()
    print(exec_times)
    df_versions = outdated_packages_list()
    print(df_versions.head())





    
