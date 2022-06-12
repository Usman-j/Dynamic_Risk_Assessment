import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path']) 
feat_list = config['feat_list']

#################Function for model scoring
def score_model(model_name=model_path+'/trainedmodel.pkl',
                test_file=test_data_path+'/testdata.csv',
                target_var='exited'):
    '''
    This function takes a trained model, loads test data, and calculates an F1 score for the model relative to the test data.
    It also writes the result to the latestscore.txt file.
    Inputs: 
    model_name: str
        Name of pickled model file
    test_file: str
        Name of test data file
    target_var: str
        Name of the target variable
    Outputs:
    f1score: float
        f1score of the model on the input date
    '''
    
    score_file = open(model_path+'/latestscore.txt', 'w')
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    df_test = pd.read_csv(test_file)
    X_test = df_test.loc[:,feat_list].values.reshape(-1, len(feat_list))
    y_test = df_test[target_var].values.reshape(-1, 1).ravel()
    predicted = model.predict(X_test)
    f1score = metrics.f1_score(predicted,y_test)
    score_file.write(str(f1score)+'\n')
    score_file.close()
    return f1score

if __name__ == '__main__':
    score_model()