from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path'])
dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(),config['test_data_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path'])
feat_list = config['feat_list']

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():    
    '''
    Calls the model prediction function

    Output:
        List of model prediction values on the given data path
    '''    
    
    data_file_name = request.get_json()['filepath']
    df_test = pd.read_csv(data_file_name)
    y_pred = model_predictions(df_test)
    y_pred = [int(i) for i in y_pred]
    return jsonify(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    '''
    Scores the deployed model for the test data

    Output:
        str of f1score
    '''
    f1_score = score_model()
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    '''
    Checks means, medians, and modes for each numeric column

    Output:
        list of all calculated summary statistics
    '''
    df_data = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    stats_list = dataframe_summary(df_data)
    return jsonify(stats_list) 

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    '''
    Checks missing data, execution times, and outdated packages

    Output:
        dict consisting of list of missing data percentage in numeric columns, 
        list of execution times, and dataframe of outdated packages
    '''
    df_data = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    napercents = check_missing_data(df_data)
    exec_times = execution_time()
    df_versions = outdated_packages_list()
    out = {'missing_data': napercents, 
            'execution_times': exec_times,
            'outdated_packages': df_versions.to_dict(orient='records')}
    return out

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
