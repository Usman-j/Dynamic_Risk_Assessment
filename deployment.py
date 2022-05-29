import pandas as pd
import numpy as np
import pickle
import os
import shutil
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(),config['output_folder_path'])
model_path = os.path.join(os.getcwd(),config['output_model_path'])
prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle(model_name='trainedmodel.pkl',
                            score_file='latestscore.txt',
                            ingestion_record='ingestedfiles.txt'):
    '''
    Copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory.
    Inputs: 
    model_name: str
        Name of pickled model file
    score_file: str
        Name of the score file for the given latest model
    ingestion_record: str
        Ingestion record of the given latest model
    '''                        
    
    shutil.copy(output_folder_path+'/'+ingestion_record, prod_deployment_path+'/')
    shutil.copy(model_path+'/'+model_name, prod_deployment_path+'/')
    shutil.copy(model_path+'/'+score_file, prod_deployment_path+'/')

if __name__ == '__main__':
    store_model_into_pickle()
        

