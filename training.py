import pandas as pd
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path']) 
model_path = os.path.join(os.getcwd(),config['output_model_path']) 
feat_list = config['feat_list']

#################Function for training the model
def train_model(data_file='finaldata.csv',
                 target_var='exited'):
    '''
    Trains a logistic regression model on the given data file and features.
    Inputs:
    data_file: str
        Name of the training data file
    target_var: str
        Name of the target variable
    '''
    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    df_train = pd.read_csv(dataset_csv_path+'/'+data_file)
    X = df_train.loc[:,feat_list].values.reshape(-1, len(feat_list))
    y = df_train[target_var].values.reshape(-1, 1).ravel()
    # print(y)
    # print(X[:3])
    model = logit.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path+'/trainedmodel.pkl', 'wb'))

if __name__ == '__main__':
    train_model()