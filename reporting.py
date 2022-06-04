import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path'])
dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(),config['test_data_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path'])
feat_list = config['feat_list']

##############Function for reporting
def score_model():
    '''
    Calculates a confusion matrix using the test data and the deployed model and saves the figure.
    '''
    df_test = pd.read_csv(test_data_path+'/testdata.csv')
    y_pred = model_predictions(df_test)
    y_test = df_test['exited'].values
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cf_matrix)
    ax = sns.heatmap(cf_matrix, annot=True)
    ax.set_xlabel('Predicted Exited Labels')
    ax.set_ylabel('Actual Exited Labels')
    plt.savefig(model_path+'/confusionmatrix.png')
    plt.close()

if __name__ == '__main__':
    score_model()
