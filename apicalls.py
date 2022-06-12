import requests
import subprocess
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"
with open('config.json','r') as f:
    config = json.load(f)
model_path = os.path.join(os.getcwd(),config['output_model_path'])
test_data_path = os.path.join(os.getcwd(),config['test_data_path'])

def api_call():

    api_responses_file = open(model_path+'/apireturns.txt', 'w')
    # print(URL+'/prediction')
    #Call each API endpoint and store the responses
    response1 = requests.post(URL+'/prediction', 
                    json={'filepath': os.path.join(test_data_path,'testdata.csv')}).text
    response2 = requests.get(URL+'/scoring').text
    response3 = requests.get(URL+'/summarystats').text
    response4 = requests.get(URL+'/diagnostics').text
    #combine all API responses
    responses = 'Test Data Predictions: \n' + response1 + '\n'
    responses += 'Test Data f1score: \n' + response2 + '\n'
    responses += 'Summary statistics of training data: \n' + response3 + '\n'
    responses += 'Diagnsotics: \n' + response4 + '\n'
    #write the responses to your workspace
    api_responses_file.write(responses)
    api_responses_file.close()

if __name__ == '__main__':
    api_call()
